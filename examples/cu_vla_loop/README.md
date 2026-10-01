# cu_vla_loop

A policy-driven arm loop built on `cu-policy`: a Python policy receives joint state and camera
frames over Zenoh and returns action chunks, the governor gates them, and a simulated arm
follows. The governor and the link come from the `cu-policy-loop` plugin; the application adds
the arm, the camera and the probes.

```text
arm/positions -> obs -> vla_link/obs ~~Zenoh~~> copper_policy (Python policy)
                                                   |
arm/positions -> vla_gov <- vla_link/action <~~Zenoh~~-+
                   |
                   +-> arm/goals

camera -> vla_link/img ~~Zenoh~~> copper_policy      vla_link/status -> status_probe (logged)
```

## Application parts

| Part | Role |
| --- | --- |
| `bridges::MockArm` | A simulated follower with the saturating-cast behavior of `cu_feetech` (NaN becomes raw 0) and a slew limit, so the loop runs without hardware. |
| `tasks::ObsBuilder` | Numbers each measured position and sends it to the link and the governor. |
| `tasks::SyntheticCamera` | Hands out pooled frames painted with a pattern the Python side verifies pixel by pixel. |
| `tasks::StatusProbe`, `tasks::ChunkProbe` | Keep the newest link status and the round trip of each chunk for the tests. |

`run_configured` replaces governor settings at run time, for example to turn the scheduler on.

## Tests

`just vla-loop-check` runs them. They start a real Python policy over loopback Zenoh and check
that the arm moves while every goal stays inside the limits; that a chunk with NaN, a wild
target, a stale chunk or a dead policy never reaches the arm as an unsafe goal; that frames and
link counters arrive; that a recorded run, including the scheduler's requests, replays
identically with no Zenoh session; that the cycle thread allocates nothing with the link live;
the observation-to-chunk round trip; and real-time chunking end to end with a trained flow
policy.

The arm bridge `cu_feetech` is the hardware counterpart of `MockArm`: it sets the goal to the
present position before torque, refuses non-finite goals, clamps to the calibrated range and
cuts torque on protective servo errors.
