# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import threading
import time
from typing import Any

from pydantic import BaseModel

from .utils.config import (
    setup_calibration_files,
    setup_follower_calibration_file,
    setup_leader_calibration_file,
)
from .utils.devices import follower_lock, safe_disconnect_device
from .utils.robot_devices import (
    SO101_ROBOT_TYPE,
    SO101_TELEOPERATOR_TYPE,
    UR_ROBOT_TYPE,
    make_robot,
    make_robot_config,
    make_teleoperator,
    make_teleoperator_config,
    robot_needs_calibration,
    teleoperator_needs_calibration,
)

logger = logging.getLogger(__name__)

# Global variables for teleoperation state
teleoperation_active = False
teleoperation_thread: threading.Thread | None = None
current_robot = None
current_teleop = None
current_teleop_cameras: dict[str, Any] = {}
current_robot_type = SO101_ROBOT_TYPE


class TeleoperateRequest(BaseModel):
    leader_port: str = ""
    follower_port: str = ""
    leader_config: str = ""
    follower_config: str = ""
    robot_type: str = SO101_ROBOT_TYPE
    teleoperator_type: str = SO101_TELEOPERATOR_TYPE
    robot_name: str = ""
    cameras: dict[str, dict[str, Any]] = {}


def get_joint_positions_from_robot(robot, robot_type: str = SO101_ROBOT_TYPE) -> dict[str, float]:
    """
    Extract current joint positions from the robot and convert to URDF joint format.

    lerobot drives the SO-101 with ``use_degrees=True`` by default, so each
    ``observation["<motor>.pos"]`` is already the joint angle in degrees relative
    to the calibration center — which is also the URDF's zero pose. The URDF
    joint value is therefore just that angle converted to radians, for every
    joint. (The gripper reports 0–100 rather than degrees, but that range lands
    inside the Jaw limit, matching the open/closed sweep.)

    UR observations use ROS joint names and are already in radians.

    Args:
        robot: The follower robot instance.
        robot_type: The configured follower type.

    Returns:
        Dictionary mapping URDF joint names to radian values
    """
    motor_to_urdf_mapping = {
        "shoulder_pan": "Rotation",
        "shoulder_lift": "Pitch",
        "elbow_flex": "Elbow",
        "wrist_flex": "Wrist_Pitch",
        "wrist_roll": "Wrist_Roll",
        "gripper": "Jaw",
    }

    try:
        observation = robot.get_observation()

        if robot_type == UR_ROBOT_TYPE:
            ur_joint_names = (
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
                "robotiq_85_left_knuckle_joint",
            )
            positions = {
                name: observation[f"{name}.pos"]
                for name in ur_joint_names
                if f"{name}.pos" in observation
            }
            missing = [
                f"{name}.pos" for name in ur_joint_names[:6] if f"{name}.pos" not in observation
            ]
            now = time.monotonic()
            if missing and now - getattr(get_joint_positions_from_robot, "_last_ur_warning", -math.inf) >= 5:
                logger.warning("UR joint positions not available: %s", ", ".join(missing))
                get_joint_positions_from_robot._last_ur_warning = now
            return positions

        joint_positions: dict[str, float] = {}
        debug_rows = []
        for motor_name, urdf_joint_name in motor_to_urdf_mapping.items():
            motor_key = f"{motor_name}.pos"
            if motor_key not in observation:
                logger.warning(f"Motor {motor_key} not found in observation")
                joint_positions[urdf_joint_name] = 0.0
                continue

            angle_degrees = observation[motor_key]
            joint_positions[urdf_joint_name] = angle_degrees * math.pi / 180.0
            debug_rows.append(f"{motor_name:14s} {angle_degrees:+8.2f}° → {urdf_joint_name:11s}")

        # Throttled debug print (~once per second at 20 Hz broadcast).
        now = time.time()
        if now - getattr(get_joint_positions_from_robot, "_last_log", 0) > 1.0:
            get_joint_positions_from_robot._last_log = now
            logger.info("[joint-debug]\n  " + "\n  ".join(debug_rows))

        return joint_positions

    except Exception as e:
        logger.error(f"Error getting joint positions: {e}")
        if robot_type == UR_ROBOT_TYPE:
            return {}
        return dict.fromkeys(motor_to_urdf_mapping.values(), 0.0)


def _safe_disconnect(device) -> None:
    """Disconnect a robot/teleop device, swallowing (but logging) any error.

    Used on the SO-101 connection-failure cleanup path so one device's failure
    can't leave the other holding its serial port open.
    """
    safe_disconnect_device(device, logger)


def _make_managed_cameras(camera_data: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Create cameras managed by LeLab rather than by a persistent robot."""
    if not camera_data:
        return {}

    from lerobot.cameras.opencv import OpenCVCamera

    from . import record

    configs = record._build_camera_configs(camera_data, record._platform_backend())
    return {name: OpenCVCamera(config) for name, config in configs.items()}


def _disconnect_cameras(cameras: dict[str, Any]) -> None:
    for name, camera in cameras.items():
        try:
            camera.disconnect()
        except Exception as e:
            logger.warning("Failed to disconnect teleoperation camera %s: %s", name, e)


def teleoperation_camera_feed_frames(cam_key: str, fps: float = 15.0):
    """Yield MJPEG frames from a camera connected for the active teleoperation."""
    import cv2

    interval = 1.0 / fps
    while teleoperation_active:
        camera = current_teleop_cameras.get(cam_key)
        if camera is None:
            break
        try:
            frame = camera.read_latest()
        except (TimeoutError, RuntimeError):
            time.sleep(interval)
            continue
        except Exception as e:
            logger.warning("teleoperation camera feed %s: unexpected read error: %s", cam_key, e)
            break

        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr)
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        time.sleep(interval)


def handle_start_teleoperation(request: TeleoperateRequest, websocket_manager=None) -> dict[str, Any]:
    """Handle start teleoperation request.

    Connects to both arms *synchronously* so that a connection failure (arm
    unplugged, port busy, power off) is reported back to the caller, rather than
    dying silently in the worker thread while the API has already claimed
    success. Only the teleoperation loop runs in the background thread.
    """
    global teleoperation_active, teleoperation_thread, current_robot, current_teleop
    global current_teleop_cameras, current_robot_type

    from . import record as _record, replay as _replay, rollout as _rollout

    with follower_lock:
        if teleoperation_active:
            return {"success": False, "message": "Teleoperation is already active"}
        if _record.recording_active:
            return {"success": False, "message": "Recording is currently active. Stop it first."}
        if _rollout.inference_active:
            return {"success": False, "message": "Inference is currently active. Stop it first."}
        if _replay.replay_active:
            return {"success": False, "message": "Replay is currently active. Stop it first."}
        teleoperation_active = True

    robot = None
    teleop_device = None
    managed_cameras: dict[str, Any] = {}
    try:
        logger.info(
            f"Starting teleoperation with leader port: {request.leader_port}, follower port: {request.follower_port}"
        )

        leader_config_name = request.leader_config.removesuffix(".json") or (
            request.robot_name or "teachbot"
        )
        follower_config_name = request.follower_config.removesuffix(".json") or (
            request.robot_name or "ur"
        )
        if (
            robot_needs_calibration(request.robot_type)
            and teleoperator_needs_calibration(request.teleoperator_type)
        ):
            leader_config_name, follower_config_name = setup_calibration_files(
                request.leader_config, request.follower_config
            )
        else:
            if robot_needs_calibration(request.robot_type):
                follower_config_name = setup_follower_calibration_file(request.follower_config)
            if teleoperator_needs_calibration(request.teleoperator_type):
                leader_config_name = setup_leader_calibration_file(request.leader_config)

        robot_config = make_robot_config(
            request.robot_type,
            port=request.follower_port,
            device_id=follower_config_name,
            cameras=request.cameras if request.robot_type == SO101_ROBOT_TYPE else None,
        )
        teleop_config = make_teleoperator_config(
            request.teleoperator_type,
            port=request.leader_port,
            device_id=leader_config_name,
        )
        # Connect synchronously. If either device fails to connect, clean up the
        # other (so its serial port is released) and report the error — do NOT
        # leave the caller thinking teleoperation started.
        logger.info("Initializing robot and teleop device...")
        robot = make_robot(robot_config)
        teleop_device = make_teleoperator(teleop_config)

        # Connect each arm separately so the error names which one failed and
        # tells the user what to do, instead of a generic "failed to start".
        logger.info("Connecting to follower arm...")
        try:
            if robot_needs_calibration(request.robot_type):
                robot.bus.connect()
        except Exception as e:
            raise RuntimeError(
                f"Could not connect to the follower arm on {request.follower_port}. "
                "Make sure it's plugged in and powered on, then try again."
            ) from e

        logger.info("Connecting to leader arm...")
        try:
            if teleoperator_needs_calibration(request.teleoperator_type):
                teleop_device.bus.connect()
        except Exception as e:
            raise RuntimeError(
                f"Could not connect to the leader arm on {request.leader_port}. "
                "Make sure it's plugged in and powered on, then try again."
            ) from e

        # Feetech motors only store calibration registers with the torque off,
        # which also opens their EEPROM lock.
        if robot_needs_calibration(request.robot_type):
            logger.info("Writing follower calibration to motors...")
            robot.bus.disable_torque()
            robot.bus.write_calibration(robot.calibration)
        if teleoperator_needs_calibration(request.teleoperator_type):
            logger.info("Writing leader calibration to motors...")
            teleop_device.bus.disable_torque()
            teleop_device.bus.write_calibration(teleop_device.calibration)

        # Connect cameras and configure motors
        logger.info("Connecting cameras and configuring motors...")
        if robot_needs_calibration(request.robot_type):
            for cam in robot.cameras.values():
                cam.connect()
            robot.configure()
            managed_cameras = robot.cameras
        elif request.robot_type == UR_ROBOT_TYPE:
            managed_cameras = _make_managed_cameras(request.cameras)
            for cam in managed_cameras.values():
                cam.connect()
        if teleoperator_needs_calibration(request.teleoperator_type):
            teleop_device.configure()
        logger.info("Successfully connected to both devices")

        current_robot = robot
        current_teleop = teleop_device
        current_teleop_cameras = managed_cameras
        current_robot_type = request.robot_type

        # Stream the arms in the background; the worker disconnects SO-101
        # devices while ROS-backed devices remain cached between sessions.
        def teleoperation_worker():
            global teleoperation_active, current_robot, current_teleop, current_teleop_cameras

            logger.info("Starting teleoperation loop...")
            try:
                last_broadcast_time = 0
                broadcast_interval = 1 / 30  # 30 FPS

                while teleoperation_active:
                    action = teleop_device.get_action()
                    robot.send_action(action)

                    current_time = time.time()
                    if current_time - last_broadcast_time >= broadcast_interval:
                        try:
                            joint_positions = get_joint_positions_from_robot(robot, request.robot_type)
                            joint_data = {
                                "type": "joint_update",
                                "joints": joint_positions,
                                "timestamp": current_time,
                            }
                            if websocket_manager and websocket_manager.active_connections:
                                websocket_manager.broadcast_joint_data_sync(joint_data)
                            last_broadcast_time = current_time
                        except Exception as e:
                            logger.error(f"Error broadcasting joint data: {e}")

                    time.sleep(0.001)
            except Exception as e:
                logger.error(f"Error during teleoperation loop: {e}")
            finally:
                teleoperation_active = False
                current_teleop_cameras = {}
                if robot_needs_calibration(request.robot_type):
                    _safe_disconnect(robot)
                else:
                    _disconnect_cameras(managed_cameras)
                if teleoperator_needs_calibration(request.teleoperator_type):
                    _safe_disconnect(teleop_device)
                logger.info("Teleoperation stopped")
                current_robot = None
                current_teleop = None

        teleoperation_thread = threading.Thread(
            target=teleoperation_worker, name="teleoperation-worker", daemon=True
        )
        teleoperation_thread.start()

        return {
            "success": True,
            "message": "Teleoperation started successfully",
            "leader_port": request.leader_port,
            "follower_port": request.follower_port,
        }

    except Exception as e:
        # Connection (or setup) failed before the loop started: release any
        # device that did open, reset state, and surface the error.
        if robot is not None and robot_needs_calibration(request.robot_type):
            _safe_disconnect(robot)
        elif managed_cameras:
            _disconnect_cameras(managed_cameras)
        if teleop_device is not None and teleoperator_needs_calibration(request.teleoperator_type):
            _safe_disconnect(teleop_device)
        teleoperation_active = False
        current_robot = None
        current_teleop = None
        current_teleop_cameras = {}
        logger.error(f"Failed to start teleoperation: {e}")
        # str(e) is already a user-facing message for the connection failures
        # raised above; the toast title supplies the "error starting" context.
        return {"success": False, "message": str(e)}


def handle_stop_teleoperation() -> dict[str, Any]:
    """Handle stop teleoperation request.

    Signals the worker via `teleoperation_active = False` and waits for it to
    exit. The worker owns SO-101 disconnects, avoiding a race on the serial bus.
    """
    global teleoperation_active, teleoperation_thread

    if not teleoperation_active:
        return {"success": False, "message": "No teleoperation session is active"}

    logger.info("Stop teleoperation triggered from web interface")
    teleoperation_active = False

    worker = teleoperation_thread
    if worker is not None and worker.is_alive():
        worker.join(timeout=5.0)
        if worker.is_alive():
            logger.warning("Teleoperation worker did not exit within 5s")
    teleoperation_thread = None

    return {"success": True, "message": "Teleoperation stopped successfully"}


def handle_teleoperation_status() -> dict[str, Any]:
    """Handle teleoperation status request"""
    return {
        "teleoperation_active": teleoperation_active,
        "available_controls": {
            "stop_teleoperation": teleoperation_active,
        },
        "message": "Teleoperation status retrieved successfully",
    }


def handle_get_joint_positions() -> dict[str, Any]:
    """Handle get current robot joint positions request"""
    global current_robot

    if not teleoperation_active or current_robot is None:
        return {"success": False, "message": "No active teleoperation session"}

    try:
        joint_positions = get_joint_positions_from_robot(current_robot, current_robot_type)
        return {"success": True, "joint_positions": joint_positions, "timestamp": time.time()}
    except Exception as e:
        logger.error(f"Error getting joint positions: {e}")
        return {"success": False, "message": f"Failed to get joint positions: {str(e)}"}
