import math
import threading
import time
from dataclasses import dataclass
from typing import Any

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.gamepad.configuration_gamepad import GamepadTeleopConfig
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardTeleopConfig

from .utils.robot_devices import SO101_ROBOT_TYPE, UR_ROBOT_TYPE

KEYBOARD_JOINT_JOG_TYPE = "keyboard_joint_jog"
GAMEPAD_JOINT_JOG_TYPE = "gamepad_joint_jog"

_SO101_ARM_ACTIONS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
)
_UR_ARM_ACTIONS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
_KEY_PAIRS = (("q", "a"), ("w", "s"), ("e", "d"), ("r", "f"), ("t", "g"), ("y", "h"))
_GRIPPER_KEY_PAIR = ("u", "j")
_MAX_JOG_INTERVAL_S = 0.05
_SO101_JOINT_SPEED_DEG_S = 25.0
_UR_JOINT_SPEED_RAD_S = 0.25
_SO101_GRIPPER_SPEED_S = 100.0
_UR_GRIPPER_SPEED_S = 1.0
_GAMEPAD_DEADZONE = 0.15


@TeleoperatorConfig.register_subclass(KEYBOARD_JOINT_JOG_TYPE)
@dataclass
class KeyboardJointJogConfig(KeyboardTeleopConfig):
    robot_type: str = SO101_ROBOT_TYPE


@TeleoperatorConfig.register_subclass(GAMEPAD_JOINT_JOG_TYPE)
@dataclass
class GamepadJointJogConfig(GamepadTeleopConfig):
    robot_type: str = SO101_ROBOT_TYPE


def _action_keys(robot_type: str) -> tuple[tuple[str, ...], str]:
    if robot_type == SO101_ROBOT_TYPE:
        return _SO101_ARM_ACTIONS, "gripper.pos"
    if robot_type == UR_ROBOT_TYPE:
        return _UR_ARM_ACTIONS, "gripper_joint"
    raise ValueError(f"Joint-jog teleoperation does not support robot type: {robot_type}")


def _current_position(observation: dict[str, Any], action_key: str) -> float:
    observation_key = action_key if action_key in observation else f"{action_key}.pos"
    value = observation.get(observation_key)
    if value is None:
        raise RuntimeError(f"Robot observation is missing joint position {observation_key!r}")
    try:
        position = float(value)
    except (TypeError, ValueError) as e:
        raise RuntimeError(f"Robot joint position {observation_key!r} is not numeric") from e
    if not math.isfinite(position):
        raise RuntimeError(f"Robot joint position {observation_key!r} is not finite")
    return position


def _joint_limits(robot: Any, robot_type: str, action_key: str, index: int) -> tuple[float, float]:
    if robot_type == UR_ROBOT_TYPE:
        interface = robot.config.ros2_interface
        if index >= len(interface.min_joint_positions) or index >= len(interface.max_joint_positions):
            raise RuntimeError(f"UR joint limits are missing for {action_key!r}")
        lower = interface.min_joint_positions[index]
        upper = interface.max_joint_positions[index]
    else:
        motor_name = action_key.removesuffix(".pos")
        calibration = robot.calibration.get(motor_name)
        motor = robot.bus.motors.get(motor_name)
        if calibration is None or motor is None:
            raise RuntimeError(f"SO-101 calibration is missing for joint {motor_name!r}")
        resolution = robot.bus.model_resolution_table[motor.model] - 1
        midpoint = (calibration.range_min + calibration.range_max) / 2
        lower = (calibration.range_min - midpoint) * 360 / resolution
        upper = (calibration.range_max - midpoint) * 360 / resolution

    lower = float(lower)
    upper = float(upper)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise RuntimeError(f"Invalid joint limits for {action_key!r}: {lower}, {upper}")
    return lower, upper


def make_joint_jog_action(
    robot: Any,
    robot_type: str,
    directions: dict[str, float],
    interval_s: float,
) -> dict[str, float]:
    """Convert normalized joint jog inputs into bounded absolute robot targets."""
    arm_actions, gripper_action = _action_keys(robot_type)
    observation = robot.get_observation()
    interval_s = min(max(float(interval_s), 0.0), _MAX_JOG_INTERVAL_S)
    action: dict[str, float] = {}

    for index, action_key in enumerate(arm_actions):
        current = _current_position(observation, action_key)
        direction = min(max(float(directions.get(action_key, 0.0)), -1.0), 1.0)
        speed = _SO101_JOINT_SPEED_DEG_S if robot_type == SO101_ROBOT_TYPE else _UR_JOINT_SPEED_RAD_S
        target = current + direction * speed * interval_s
        lower, upper = _joint_limits(robot, robot_type, action_key, index)
        action[action_key] = min(max(target, lower), upper)

    current_gripper = _current_position(observation, gripper_action)
    gripper_direction = min(max(float(directions.get(gripper_action, 0.0)), -1.0), 1.0)
    if robot_type == SO101_ROBOT_TYPE:
        gripper_limits = (0.0, 100.0)
        gripper_speed = _SO101_GRIPPER_SPEED_S
    else:
        interface = robot.config.ros2_interface
        gripper_limits = tuple(
            sorted((float(interface.gripper_open_position), float(interface.gripper_close_position)))
        )
        gripper_speed = _UR_GRIPPER_SPEED_S
        gripper_direction = -gripper_direction
    gripper_target = current_gripper + gripper_direction * gripper_speed * interval_s
    action[gripper_action] = min(max(gripper_target, gripper_limits[0]), gripper_limits[1])
    return action


class JointJogTeleoperator(Teleoperator):
    def __init__(self, config: KeyboardJointJogConfig | GamepadJointJogConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.robot_type
        self.robot: Any | None = None
        self._last_action_time = 0.0

    @property
    def action_features(self) -> dict:
        arm_actions, gripper_action = _action_keys(self.robot_type)
        return dict.fromkeys((*arm_actions, gripper_action), float)

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_calibrated(self) -> bool:
        return True

    def bind_robot(self, robot: Any) -> None:
        self.robot = robot

    def configure(self) -> None:
        pass

    def calibrate(self) -> None:
        pass

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def _target_action(self, directions: dict[str, float]) -> dict[str, float]:
        if self.robot is None:
            raise RuntimeError("Joint-jog teleoperator is not bound to a robot")
        now = time.monotonic()
        interval = now - self._last_action_time if self._last_action_time else 0.0
        self._last_action_time = now
        return make_joint_jog_action(self.robot, self.robot_type, directions, interval)


class KeyboardJointJogTeleoperator(JointJogTeleoperator):
    config_class = KeyboardJointJogConfig
    name = KEYBOARD_JOINT_JOG_TYPE

    def __init__(self, config: KeyboardJointJogConfig):
        super().__init__(config)
        self.listener = None
        self._pressed: set[str] = set()
        self._pressed_lock = threading.Lock()

    @property
    def is_connected(self) -> bool:
        return self.listener is not None and self.listener.is_alive()

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise RuntimeError("Keyboard teleoperator is already connected")

        from pynput import keyboard

        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()
        try:
            self.listener.wait()
        except Exception:
            self.disconnect()
            raise
        if not self.is_connected:
            self.listener = None
            raise RuntimeError("Keyboard listener did not start; check that a desktop input session is available")

    def _on_press(self, key: Any) -> None:
        char = getattr(key, "char", None)
        if isinstance(char, str):
            with self._pressed_lock:
                self._pressed.add(char.lower())

    def _on_release(self, key: Any) -> None:
        char = getattr(key, "char", None)
        if isinstance(char, str):
            with self._pressed_lock:
                self._pressed.discard(char.lower())

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise RuntimeError("Keyboard teleoperator is not connected")
        arm_actions, gripper_action = _action_keys(self.robot_type)
        with self._pressed_lock:
            pressed = self._pressed.copy()

        directions: dict[str, float] = {}
        for index, action_key in enumerate(arm_actions):
            negative, positive = _KEY_PAIRS[index]
            directions[action_key] = float(positive in pressed) - float(negative in pressed)

        negative, positive = _GRIPPER_KEY_PAIR
        directions[gripper_action] = float(positive in pressed) - float(negative in pressed)
        return self._target_action(directions)

    def disconnect(self) -> None:
        listener = self.listener
        self.listener = None
        if listener is not None:
            listener.stop()
            if listener is not threading.current_thread():
                listener.join(timeout=1.0)


class GamepadJointJogTeleoperator(JointJogTeleoperator):
    config_class = GamepadJointJogConfig
    name = GAMEPAD_JOINT_JOG_TYPE

    def __init__(self, config: GamepadJointJogConfig):
        super().__init__(config)
        self._pygame = None
        self.joystick = None

    @property
    def is_connected(self) -> bool:
        return self.joystick is not None

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise RuntimeError("Gamepad teleoperator is already connected")

        import pygame

        self._pygame = pygame
        try:
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() == 0:
                raise RuntimeError("No gamepad detected; connect a gamepad to the computer running LeLab")
            joystick = pygame.joystick.Joystick(0)
            joystick.init()
            self.joystick = joystick
        except Exception:
            self.disconnect()
            raise

    def get_action(self) -> dict[str, float]:
        if not self.is_connected or self._pygame is None:
            raise RuntimeError("Gamepad teleoperator is not connected")
        self._pygame.event.pump()
        arm_actions, gripper_action = _action_keys(self.robot_type)
        directions: dict[str, float] = {}

        axis_joints = min(4, len(arm_actions))
        for index in range(axis_joints):
            value = self._axis(index)
            if abs(value) < _GAMEPAD_DEADZONE:
                value = 0.0
            else:
                value = (abs(value) - _GAMEPAD_DEADZONE) / (1.0 - _GAMEPAD_DEADZONE)
                value = math.copysign(value, self._axis(index))
            if index in (1, 3):
                value = -value
            directions[arm_actions[index]] = value

        if len(arm_actions) > 4:
            directions[arm_actions[4]] = float(self._button(5)) - float(self._button(4))
        if len(arm_actions) > 5:
            hat = self.joystick.get_hat(0) if self.joystick.get_numhats() else (0, 0)
            directions[arm_actions[5]] = float(hat[1])

        close = self._button(6) or self._trigger_pressed(4)
        open_ = self._button(7) or self._trigger_pressed(5)
        directions[gripper_action] = float(open_) - float(close)
        return self._target_action(directions)

    def _axis(self, index: int) -> float:
        return float(self.joystick.get_axis(index)) if index < self.joystick.get_numaxes() else 0.0

    def _button(self, index: int) -> bool:
        return bool(self.joystick.get_button(index)) if index < self.joystick.get_numbuttons() else False

    def _trigger_pressed(self, axis: int) -> bool:
        return axis < self.joystick.get_numaxes() and self._axis(axis) > 0.2

    def disconnect(self) -> None:
        if self.joystick is not None:
            self.joystick.quit()
            self.joystick = None
        if self._pygame is not None:
            self._pygame.quit()
            self._pygame = None
