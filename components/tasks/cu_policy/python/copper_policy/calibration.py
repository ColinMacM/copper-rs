"""Raw servo ticks <-> LeRobot units, using LeRobot's own calibration file as the single source.

The Copper side runs `cu_feetech` with `units: "raw"`, so observations and goals on the wire are
raw ticks. A LeRobot policy works in the units of its dataset: degrees or the -100..100 range for
body joints and 0..100 for the gripper. The formulas are those of
`lerobot/motors/motors_bus.py` (`_normalize` / `_unnormalize`); `tests/test_calibration.py`
checks them against the installed LeRobot.
"""
import json
from dataclasses import dataclass

RESOLUTION = 4096  # STS3215
MAX_RES = RESOLUTION - 1


@dataclass(frozen=True)
class Motor:
    name: str
    range_min: int
    range_max: int
    mode: str  # "degrees" | "range_m100_100" | "range_0_100"

    @property
    def mid(self):
        return (self.range_min + self.range_max) / 2


def load(path, modes):
    """`modes`: joint name -> mode, in policy joint order (SO-100/101: five body joints, then
    the gripper). Returns the motors in that order."""
    with open(path) as f:
        raw = json.load(f)
    motors = []
    for name, mode in modes.items():
        if name not in raw:
            raise KeyError(f"calibration file has no joint {name!r}")
        c = raw[name]
        if c["range_min"] >= c["range_max"]:
            raise ValueError(f"{name}: range_min must be below range_max")
        motors.append(Motor(name, int(c["range_min"]), int(c["range_max"]), mode))
    return motors


def raw_to_units(motor, raw):
    lo, hi = motor.range_min, motor.range_max
    bounded = min(hi, max(lo, raw))
    if motor.mode == "range_m100_100":
        return (bounded - lo) / (hi - lo) * 200 - 100
    if motor.mode == "range_0_100":
        return (bounded - lo) / (hi - lo) * 100
    if motor.mode == "degrees":
        return (raw - motor.mid) * 360 / MAX_RES
    raise ValueError(f"unknown mode {motor.mode!r}")


def units_to_raw(motor, value):
    lo, hi = motor.range_min, motor.range_max
    if motor.mode == "range_m100_100":
        bounded = min(100.0, max(-100.0, value))
        return int((bounded + 100) / 200 * (hi - lo) + lo)
    if motor.mode == "range_0_100":
        bounded = min(100.0, max(0.0, value))
        return int(bounded / 100 * (hi - lo) + lo)
    if motor.mode == "degrees":
        return int(value * MAX_RES / 360 + motor.mid)
    raise ValueError(f"unknown mode {motor.mode!r}")


def state_to_units(motors, raw_state):
    return [raw_to_units(m, r) for m, r in zip(motors, raw_state)]


def actions_to_raw(motors, steps):
    """steps: iterable of per-step unit vectors -> flat list of raw ticks, step-major."""
    out = []
    for step in steps:
        out.extend(float(units_to_raw(m, v)) for m, v in zip(motors, step))
    return out
