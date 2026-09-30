# cu_vla_loop

A policy-driven arm loop: a Python policy receives joint state over Zenoh and returns action
chunks, the action governor gates them, and an arm follows. The arm here is a simulation that
reproduces the `cu_feetech` behaviors that matter for safety, so the whole loop runs without
hardware.

```text
arm/positions -> obs -> link/obs ~~Zenoh~~> vla_runner (Python policy)
                                               |
arm/positions -> gov <- link/action <~~Zenoh~~-+
                  |
                  +-> arm/goals

camera -> link/img ~~Zenoh~~> vla_runner          link/status -> status_probe (logged)
```

## Pieces

| Part | Crate / path | Role |
| --- | --- | --- |
| Governor | `components/tasks/cu_action_governor` | Rejects non-finite, malformed, stale, out-of-order and unknown-observation chunks; clamps to joint limits; limits per-cycle step and lead over the measurement; holds when no fresh chunk or no measurement arrives. |
| Link | `components/bridges/cu_policy_link` | Zenoh bridge whose cycle side only copies into fixed slots; a worker thread owns the session. Camera frames cross as a pooled-buffer handle, copied by the worker. Drops are counted, never blocking, and the counters are a `LinkStatus` message in the log. |
| Runner | `python/vla_runner` | Serves a policy: blocks until the newest observation lands, then sends a chunk. Scripted and LeRobot ACT policies; LeRobot calibration conversion. |
| Arm | `cu_feetech` | Hardened: goal set to the present position before torque, non-finite goals refused, goals clamped to the calibrated range, failed reads yield no measurement, protective servo errors cut torque, optional goal timeout, torque off on drop. |

## Wire format

Little-endian, fixed-width integers, payload only:

- observation: `seq: u64`, `tov_ns: u64`, `len: u32`, `len` x `f32`
- image: `seq: u64`, `tov_ns: u64`, `width`, `height`, `stride: u32`, `pixel_format: [u8; 4]`, `len: u32`, `len` x `u8`
- action chunk: `obs_seq: u64`, `len: u32`, `len` x `f32` (row-major, 6 values per step, at most 300)

The chunk names the observation it was computed from. The governor ages a chunk by the time it
first saw that observation on its own clock, never by the policy's clock, and starts a late chunk
at the step that matches the elapsed time, so a chunk answered `d` cycles late continues the
trajectory it was computed for. The governor counts chunk switches and the jump each makes in the
policy's own targets (`switch_jump_max`), so a scheduling change can be compared by number.

## Latency

`tests/latency.rs` measures the round trip from building an observation to the answering chunk
reaching the graph, through a real Python process over loopback Zenoh. Release build, pinned,
scripted policy (inference is almost free):

| Cycle rate | p50 | p99 | answered |
| --- | --- | --- | --- |
| 30 Hz | 33.34 ms | 33.46 ms | 448 / 450 |
| 200 Hz | 5.00 ms | 5.12 ms | 1494 / 1500 |
| 1000 Hz | 1.00 ms | 1.04 ms | 3936 / 4000 |

Each figure is one cycle: the chunk is first seen by the next cycle, so the measurement cannot
resolve less than a period, and the link plus Python takes less than that. A real policy adds its
inference time, which is not measured here.

## Units

The arm bridge runs with `units: "raw"`, so the wire carries servo ticks. The runner converts to
and from LeRobot units with the LeRobot calibration file (`vla_runner/calibration.py`), using
the formulas of `lerobot/motors/motors_bus.py`; `python/tests/test_calibration.py` compares them
with the installed LeRobot. Calibrate the arm with LeRobot first.

## Running

```bash
just vla-loop-check
```

The end-to-end tests start `python3 -m vla_runner` and need `import zenoh` to work (`pip install
eclipse-zenoh`); the ACT test also needs `lerobot`. Tests that cannot find them print `skipped`.

## Limits

- The camera is synthetic, and the ACT policy in the tests has random weights and takes joint state only. No trained checkpoint or flow-matching policy is wired in, so Real-Time Chunking is not implemented.
- A killed process leaves servo torque on: `Drop` covers panics and clean exits, not `SIGKILL`.
  Keep a power cutoff within reach.
- The governor's observation history holds 64 entries; `max_age_ms` must fit inside it (checked at startup).
