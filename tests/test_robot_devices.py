from __future__ import annotations

from enum import Enum
from types import SimpleNamespace

import pytest


def test_make_ur_robot_config_sets_action_type(monkeypatch: pytest.MonkeyPatch) -> None:
    from lelab.utils import robot_devices

    class ActionType(Enum):
        MOVEGROUP_SERVO_JOG = "movegroup_servo_jog"
        MOVEGROUP_SERVO_TWIST = "movegroup_servo_twist"
        MOVEGROUP_FOLLOW_JOINT_TRAJECTORY = "movegroup_follow_joint_trajectory"
        JOINT_POSITION = "joint_position"

    class UrConfig:
        def __init__(self, *, id: str, cameras: dict) -> None:
            self.id = id
            self.cameras = cameras
            self.ros2_interface = SimpleNamespace(action_type=ActionType.MOVEGROUP_SERVO_JOG)

    modules = {
        "lerobot_robot_ur": SimpleNamespace(UrConfig=UrConfig),
        "lerobot_robot_ur.config_ur": SimpleNamespace(ActionType=ActionType),
    }
    monkeypatch.setattr(robot_devices, "import_module", modules.__getitem__)

    config = robot_devices.make_robot_config(
        "lerobot_robot_ur",
        port="",
        device_id="ur",
        ur_action_type="joint_position",
    )

    assert config.ros2_interface.action_type is ActionType.JOINT_POSITION


def test_make_ur_robot_config_rejects_unsupported_action_type() -> None:
    from lelab.utils import robot_devices

    with pytest.raises(ValueError, match="Unsupported UR action type"):
        robot_devices.make_robot_config("lerobot_robot_ur", port="", device_id="ur", ur_action_type="none")
