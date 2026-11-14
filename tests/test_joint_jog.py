from __future__ import annotations

from types import SimpleNamespace

import pytest


class _FakeSO101:
    def __init__(self, shoulder_pan: float = 0.0) -> None:
        self.calibration = {
            name: SimpleNamespace(range_min=0, range_max=4095)
            for name in (
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            )
        }
        self.bus = SimpleNamespace(
            motors={
                name: SimpleNamespace(model="sts3215")
                for name in self.calibration
            },
            model_resolution_table={"sts3215": 4096},
        )
        self.positions = {
            "shoulder_pan.pos": shoulder_pan,
            "shoulder_lift.pos": 0.0,
            "elbow_flex.pos": 0.0,
            "wrist_flex.pos": 0.0,
            "wrist_roll.pos": 0.0,
            "gripper.pos": 50.0,
        }

    def get_observation(self) -> dict[str, float]:
        return self.positions


class _FakeUR:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            ros2_interface=SimpleNamespace(
                min_joint_positions=[-1.0] * 6,
                max_joint_positions=[1.0] * 6,
                gripper_open_position=0.0,
                gripper_close_position=1.0,
            )
        )
        self.positions = {
            "shoulder_pan_joint.pos": 0.99,
            "shoulder_lift_joint.pos": 0.0,
            "elbow_joint.pos": 0.0,
            "wrist_1_joint.pos": 0.0,
            "wrist_2_joint.pos": 0.0,
            "wrist_3_joint.pos": 0.0,
            "gripper_joint.pos": 0.9,
        }

    def get_observation(self) -> dict[str, float]:
        return self.positions


def test_joint_jog_creates_calibrated_so101_targets_and_clamps_to_limits() -> None:
    from lelab.joint_jog import make_joint_jog_action

    robot = _FakeSO101(shoulder_pan=180.0)
    action = make_joint_jog_action(
        robot,
        "so101_follower",
        {"shoulder_pan.pos": 1.0, "gripper.pos": 1.0},
        interval_s=0.05,
    )

    assert action["shoulder_pan.pos"] == pytest.approx(180.0)
    assert action["gripper.pos"] == pytest.approx(55.0)
    assert set(action) == set(robot.positions)


def test_joint_jog_creates_full_ur_target_and_clamps_gripper_direction() -> None:
    from lelab.joint_jog import make_joint_jog_action

    robot = _FakeUR()
    action = make_joint_jog_action(
        robot,
        "lerobot_robot_ur",
        {"shoulder_pan_joint": 1.0, "gripper_joint": 1.0},
        interval_s=0.05,
    )

    assert action["shoulder_pan_joint"] == pytest.approx(1.0)
    assert action["gripper_joint"] == pytest.approx(0.85)
    assert set(action) == {
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
        "gripper_joint",
    }


def test_joint_jog_rejects_missing_joint_observation() -> None:
    from lelab.joint_jog import make_joint_jog_action

    robot = _FakeUR()
    robot.positions.pop("wrist_3_joint.pos")

    with pytest.raises(RuntimeError, match="wrist_3_joint.pos"):
        make_joint_jog_action(robot, "lerobot_robot_ur", {}, interval_s=0.05)


@pytest.mark.parametrize(
    ("teleoperator_type", "robot_type"),
    [
        ("keyboard_joint_jog", "so101_follower"),
        ("gamepad_joint_jog", "lerobot_robot_ur"),
    ],
)
def test_joint_jog_teleoperator_configs_are_robot_specific(
    teleoperator_type: str, robot_type: str
) -> None:
    from lelab.utils.robot_devices import make_teleoperator_config

    config = make_teleoperator_config(
        teleoperator_type,
        port="",
        device_id="test-robot",
        robot_type=robot_type,
    )

    assert config.type == teleoperator_type
    assert config.robot_type == robot_type


@pytest.mark.parametrize("teleoperator_type", ["keyboard_joint_jog", "gamepad_joint_jog"])
def test_joint_jog_robot_record_needs_no_leader_port_or_calibration(
    teleoperator_type: str,
) -> None:
    from lelab.utils.config import is_robot_record_clean

    assert is_robot_record_clean(
        {"robot_type": "lerobot_robot_ur", "teleoperator_type": teleoperator_type}
    )
