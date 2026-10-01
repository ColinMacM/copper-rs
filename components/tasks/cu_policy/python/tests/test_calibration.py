"""The conversion must equal LeRobot's own normalisation, so a checkpoint trained on a LeRobot
dataset receives and produces the units it was trained in."""
import json

import pytest

from copper_policy import calibration

lerobot = pytest.importorskip("lerobot")

from lerobot.motors import Motor, MotorCalibration, MotorNormMode  # noqa: E402
from lerobot.motors.feetech import FeetechMotorsBus  # noqa: E402

CAL = {
    "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": 12, "range_min": 690, "range_max": 3410},
    "elbow_flex": {"id": 3, "drive_mode": 0, "homing_offset": -30, "range_min": 880, "range_max": 3100},
    "gripper": {"id": 6, "drive_mode": 0, "homing_offset": 5, "range_min": 2040, "range_max": 3250},
}


def bus(mode):
    norm = {
        "degrees": MotorNormMode.DEGREES,
        "range_m100_100": MotorNormMode.RANGE_M100_100,
    }
    motors = {
        "shoulder_pan": Motor(1, "sts3215", norm[mode]),
        "elbow_flex": Motor(3, "sts3215", norm[mode]),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }
    calibration_ = {k: MotorCalibration(**v) for k, v in CAL.items()}
    return FeetechMotorsBus(port="/dev/null", motors=motors, calibration=calibration_)


@pytest.fixture
def cal_file(tmp_path):
    p = tmp_path / "cal.json"
    p.write_text(json.dumps(CAL))
    return p


@pytest.mark.parametrize("mode", ["degrees", "range_m100_100"])
def test_raw_to_units_equals_lerobot_normalize(mode, cal_file):
    b = bus(mode)
    motors = calibration.load(cal_file, {"shoulder_pan": mode, "elbow_flex": mode, "gripper": "range_0_100"})
    ids = {1: "shoulder_pan", 3: "elbow_flex", 6: "gripper"}
    for raw in (0, 500, 690, 1500, 2048, 2900, 3410, 3700, 4095):
        expected = b._normalize({i: raw for i in ids})
        for m, (i, name) in zip(motors, ids.items()):
            assert calibration.raw_to_units(m, raw) == pytest.approx(expected[i], abs=1e-9), (mode, name, raw)


@pytest.mark.parametrize("mode", ["degrees", "range_m100_100"])
def test_units_to_raw_equals_lerobot_unnormalize(mode, cal_file):
    b = bus(mode)
    motors = calibration.load(cal_file, {"shoulder_pan": mode, "elbow_flex": mode, "gripper": "range_0_100"})
    ids = {1: "shoulder_pan", 3: "elbow_flex", 6: "gripper"}
    values = [-120.0, -100.0, -45.5, 0.0, 12.25, 77.0, 100.0, 130.0]
    for v in values:
        body = {i: v for i in (1, 3)}
        grip = {6: min(100.0, max(0.0, v))}
        expected = b._unnormalize({**body, **grip})
        for m, (i, name) in zip(motors, ids.items()):
            got = calibration.units_to_raw(m, v if name != "gripper" else grip[6])
            assert got == expected[i], (mode, name, v, got, expected[i])


def test_roundtrip_is_within_one_tick(cal_file):
    motors = calibration.load(cal_file, {"shoulder_pan": "degrees", "elbow_flex": "degrees", "gripper": "range_0_100"})
    for m in motors:
        for raw in range(m.range_min, m.range_max, 37):
            back = calibration.units_to_raw(m, calibration.raw_to_units(m, raw))
            assert abs(back - raw) <= 1


def test_missing_joint_and_inverted_range_are_errors(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"a": {"id": 1, "drive_mode": 0, "homing_offset": 0, "range_min": 10, "range_max": 10}}))
    with pytest.raises(ValueError):
        calibration.load(p, {"a": "degrees"})
    with pytest.raises(KeyError):
        calibration.load(p, {"b": "degrees"})
