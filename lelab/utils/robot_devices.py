import atexit
from importlib import import_module
import logging
import threading
from typing import Any, Callable

from lerobot.robots import make_robot_from_config
from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.teleoperators.so_leader import SO101LeaderConfig

from .config import DEFAULT_UR_ACTION_TYPE, UR_ACTION_TYPES

SO101_ROBOT_TYPE = "so101_follower"
UR_ROBOT_TYPE = "lerobot_robot_ur"
SO101_TELEOPERATOR_TYPE = "so101_leader"
TEACHBOT_TYPE = "lerobot_teleoperator_teachbot"
KEYBOARD_JOINT_JOG_TYPE = "keyboard_joint_jog"
GAMEPAD_JOINT_JOG_TYPE = "gamepad_joint_jog"
JOINT_JOG_TELEOPERATOR_TYPES = {KEYBOARD_JOINT_JOG_TYPE, GAMEPAD_JOINT_JOG_TYPE}

_PERSISTENT_DEVICE_TYPES = {UR_ROBOT_TYPE, TEACHBOT_TYPE}
_PERSISTENT_TELEOPERATOR_TYPES = {TEACHBOT_TYPE}
_persistent_devices: dict[tuple[str, str], tuple[str, Any]] = {}
_persistent_devices_lock = threading.RLock()
logger = logging.getLogger(__name__)


def robot_needs_calibration(robot_type: str) -> bool:
    return robot_type == SO101_ROBOT_TYPE


def teleoperator_needs_calibration(teleoperator_type: str) -> bool:
    return teleoperator_type == SO101_TELEOPERATOR_TYPE


def teleoperator_is_persistent(teleoperator_type: str) -> bool:
    return teleoperator_type in _PERSISTENT_TELEOPERATOR_TYPES


def make_robot_config(
    robot_type: str,
    *,
    port: str,
    device_id: str,
    cameras: dict | None = None,
    ur_action_type: str = DEFAULT_UR_ACTION_TYPE,
):
    if robot_type == SO101_ROBOT_TYPE:
        return SO101FollowerConfig(port=port, id=device_id, cameras=cameras or {})
    if robot_type == UR_ROBOT_TYPE:
        if not isinstance(ur_action_type, str) or ur_action_type not in UR_ACTION_TYPES:
            raise ValueError(f"Unsupported UR action type: {ur_action_type}")
        config_module = import_module("lerobot_robot_ur.config_ur")
        config = import_module("lerobot_robot_ur").UrConfig(id=device_id, cameras=cameras or {})
        config.ros2_interface.action_type = config_module.ActionType(ur_action_type)
        return config
    raise ValueError(f"Unsupported robot type: {robot_type}")


def make_teleoperator_config(
    teleoperator_type: str,
    *,
    port: str,
    device_id: str,
    robot_type: str = SO101_ROBOT_TYPE,
):
    if teleoperator_type == SO101_TELEOPERATOR_TYPE:
        return SO101LeaderConfig(port=port, id=device_id)
    if teleoperator_type == TEACHBOT_TYPE:
        config_module = import_module("lerobot_teleoperator_teachbot.config_teachbot")
        return config_module.TeachbotConfig(id=device_id)
    if teleoperator_type in {KEYBOARD_JOINT_JOG_TYPE, GAMEPAD_JOINT_JOG_TYPE}:
        from ..joint_jog import GamepadJointJogConfig, KeyboardJointJogConfig

        config_class = (
            KeyboardJointJogConfig
            if teleoperator_type == KEYBOARD_JOINT_JOG_TYPE
            else GamepadJointJogConfig
        )
        return config_class(id=device_id, robot_type=robot_type)
    raise ValueError(f"Unsupported teleoperator type: {teleoperator_type}")


def _is_persistent_device(config: Any) -> bool:
    return getattr(config, "type", None) in _PERSISTENT_DEVICE_TYPES


def _config_type(config: Any) -> str:
    device_type = getattr(config, "type", None)
    if not isinstance(device_type, str):
        raise TypeError(f"Device config has no string type: {type(config).__name__}")
    return device_type


def _connect_persistent_device(device: Any) -> None:
    if not getattr(device, "is_connected", False):
        device.connect(calibrate=False)
    device.configure()


def _disconnect_persistent_device(device: Any) -> None:
    try:
        device.disconnect()
    except Exception:
        logger.exception("Failed to disconnect persistent ROS device during shutdown")


def _make_robot(config: Any) -> Any:
    if _config_type(config) == UR_ROBOT_TYPE:
        return import_module("lerobot_robot_ur.ur").Ur(config)
    return make_robot_from_config(config)


def _make_teleoperator(config: Any) -> Any:
    if _config_type(config) == TEACHBOT_TYPE:
        return import_module("lerobot_teleoperator_teachbot.teachbot").Teachbot(config)
    if _config_type(config) in {KEYBOARD_JOINT_JOG_TYPE, GAMEPAD_JOINT_JOG_TYPE}:
        from ..joint_jog import GamepadJointJogTeleoperator, KeyboardJointJogTeleoperator

        teleoperator_class = (
            KeyboardJointJogTeleoperator
            if _config_type(config) == KEYBOARD_JOINT_JOG_TYPE
            else GamepadJointJogTeleoperator
        )
        return teleoperator_class(config)
    return make_teleoperator_from_config(config)


def _get_or_create_persistent_device(
    kind: str,
    config: Any,
    factory: Callable[[Any], Any],
) -> Any:
    device_type = _config_type(config)
    key = (kind, device_type)
    signature = repr(config)

    with _persistent_devices_lock:
        cached = _persistent_devices.pop(key, None)
        if cached is not None and cached[0] == signature:
            try:
                _connect_persistent_device(cached[1])
            except Exception:
                _disconnect_persistent_device(cached[1])
                raise
            _persistent_devices[key] = cached
            return cached[1]
        if cached is not None:
            _disconnect_persistent_device(cached[1])

        device = factory(config)
        try:
            _connect_persistent_device(device)
        except Exception:
            _disconnect_persistent_device(device)
            raise

        _persistent_devices[key] = (signature, device)
        return device


def make_robot(config: Any) -> Any:
    if _is_persistent_device(config):
        return _get_or_create_persistent_device("robot", config, _make_robot)
    return _make_robot(config)


def make_teleoperator(config: Any) -> Any:
    if _is_persistent_device(config):
        return _get_or_create_persistent_device(
            "teleoperator", config, _make_teleoperator
        )
    return _make_teleoperator(config)


def disconnect_persistent_devices() -> None:
    with _persistent_devices_lock:
        for _, device in _persistent_devices.values():
            _disconnect_persistent_device(device)
        _persistent_devices.clear()


atexit.register(disconnect_persistent_devices)
