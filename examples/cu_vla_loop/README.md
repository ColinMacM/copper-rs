# cu_vla_loop

A policy-driven arm loop: a Python policy receives joint state over Zenoh and returns action
chunks, the action governor gates them, and an arm follows. The arm here is a simulation that
reproduces the `cu_feetech` behaviors that matter for safety, so the whole loop runs without
hardware.

```text
arm/positions -> obs -> link/obs ~~Zenoh~~> copper_policy (Python policy)
                                               |
arm/positions -> gov <- link/action <~~Zenoh~~-+
                  |
                  +-> arm/goals

camera -> link/img ~~Zenoh~~> copper_policy          link/status -> status_probe (logged)
```

## Pieces

| Part | Crate / path | Role |
| --- | --- | --- |
| Governor | `components/tasks/cu_policy` | Rejects non-finite, malformed, stale, out-of-order and unknown-observation chunks; clamps to joint limits; limits per-cycle step and lead over the measurement; holds when no fresh chunk or no measurement arrives. |
| Link | `components/bridges/cu_policy_link` | Zenoh bridge whose cycle side only copies into fixed slots; a worker thread owns the session. Camera frames cross as a pooled-buffer handle, copied by the worker. Drops are counted, never blocking, and the counters are a `LinkStatus` message in the log. |
| Runner | `python/copper_policy` | Serves a policy: blocks until the newest observation lands, then sends a chunk. Scripted and LeRobot ACT policies; LeRobot calibration conversion. |
| Arm | `cu_feetech` | Hardened: goal set to the present position before torque, non-finite goals refused, goals clamped to the calibrated range, failed reads yield no measurement, protective servo errors cut torque, optional goal timeout, torque off on drop. |

## Wire format

Little-endian, fixed-width integers, payload only:

- observation: `seq: u64`, `tov_ns: u64`, `len: u32`, `len` x `f32`
- exec state (`vla/exec`, every cycle): `stamp_seq: u64`, `chunk_seq: u64`, `next_index: u32`, `flags: u32`, `accept_skip: u32`, `reject: u32`, `tracking_err: f32`
- inference request (`vla/infer`, when the scheduler fires): `obs_seq: u64`, `delay: u32`, `executed: u32`, `reason: u32`, `state_len: u32`, `state_len` x `f32`, `previous_len: u32`, `previous_len` x `f32`
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

## Real-time chunking

`python/copper_policy/rtc.py` implements Real-Time Chunking (Black, Galliker, Levine,
[arXiv 2506.07339](https://arxiv.org/abs/2506.07339)) from the paper: pseudoinverse-guided flow
matching (Eq. 2-4) with the soft mask of Eq. 5 and the clipped guidance weight, and the
bookkeeping of Algorithm 1 (`Chunker`: delay buffer, execution horizon, when to start the next
inference). While a chunk executes, the next one is generated with its first `d` actions frozen
to the executing chunk's unplayed actions and the rest inpainted to agree with them.

The scheduler of Algorithm 1 runs in the governor (`SchedParams`), which already holds the
active chunk, the step it is on and the observation history. On the cycles where the next
inference should start it emits an `InferenceRequest` on `vla/infer`: the observation, the delay
estimate `d`, the number of steps `s` already played, and the unplayed remainder of the active
chunk (up to 50 x 6 floats, about 1.2 KB). The policy server therefore keeps no state: it
answers requests. Because the request is a CopperList output, a replay re-fires it at the same
cycles with the same contents (`tests/replay.rs` compares all of them bit for bit).

Governor keys, all optional; `sched_s_min` of 0 or absent turns the scheduler off:

| Key | Meaning |
| --- | --- |
| `sched_s_min` | Steps of a chunk to play before the next inference starts (`s_min`). |
| `sched_margin` | The horizon is at least the delay estimate plus this, so an answer is not due before the steps it replaces have played. |
| `sched_d_init` | Delay assumed until one has been measured, in cycles. |
| `sched_horizon` | Prediction horizon `H`; the horizon never exceeds `H - d`. |
| `replan_threshold` | Tracking error, in goal units, above which the plan is replaced at once, whatever `s_min` says; 0 disables. |
| `sched_pending_timeout` | Cycles after which an unanswered request is dropped and asked again. |

The delay estimate is the largest of the last ten delays the governor measured: when it accepts a
chunk it knows the step the chunk starts at (`ExecState.accept_skip`), which is the delay that
chunk really had. `ExecState` also says why a chunk was refused (`reject`) and how far the
measurement is from the target played last cycle (`tracking_err`), the signal behind
`replan_threshold`.

```bash
python -m copper_policy --policy flow --checkpoint flow.pt --connect-port P --rtc on   # RTC
python -m copper_policy --policy flow --checkpoint flow.pt --connect-port P --rtc off  # naive async
python -m copper_policy.flow_policy --out flow.pt                                      # train the demo policy
```

`--rtc off` keeps the same asynchronous schedule and samples each chunk freely, the paper's
naive baseline. The governor's `hold_deadline_ms` must exceed the time a chunk plays before the
next is computed (`s_min` steps); the example's default of 500 ms is for one chunk per
observation, and `run_configured` overrides it. `tests/rtc.rs` turns the scheduler on through
the same override.

The demo policy (`flow_policy.py`) is a small flow-matching network trained on synthetic
demonstrations that fork: the same path, then a detour to the left or right. It learns the first
12 DCT coefficients of each joint; the exact velocity of the remaining high frequencies is added
analytically. It stands in for a VLA, which this repository does not contain.

### Options added on top of the paper

Each is measured in `rtc_sim.ablate` (40 seeded episodes, 200 hand-overs per row, delay of 8
cycles, jumps and accelerations in ticks, policy units times 600) and tested. They can be combined.

| Option | Where | What it does | Mean jump / max jump / accel |
| --- | --- | --- | --- |
| (naive async) | | sample freely, replace the old chunk | 78 / 349 / 150 |
| RTC | `rtc.guided_inference` | the paper | 45 / 189 / 89 |
| positional noise | `--positional-noise` | the noise of step `t` is a fixed function of `(seed, t)`, so chunks that overlap start their overlapping steps from the same noise; nothing to remember | RTC + it: 36 / 108 / 75; naive + it: 57 / 237 / 115 |
| roll the observation forward | `--roll-obs` | plan from the state expected after the frozen steps, then put those steps in front | 37 / 83 / 73 |
| best of K | `--best-of K` | draw K guided samples, keep the one with the smallest weighted prefix residual | K = 4: 37 / 110 / 75 |
| exact projection | `--project` | set the frozen steps to the previous chunk's values; a guarantee, not a speed-up: the steps it fixes are the ones the governor skips when the delay estimate covers the true delay, so what is played is unchanged | 45 / 189 / 89 |
| governor blend | `blend_steps` | the played target moves from the old chunk's step to the new chunk's over that many steps | 4 steps: 17 / 46 / 66 |

The blend lowers the jump by construction, since it smooths the executed target, so it is also
checked by acceleration (lower) and by the distance from the target at the end of the episode
(not worse). Combining everything was not better than the blend alone on these metrics.

Health check: after guidance the frozen steps differ from the previous chunk's by some amount.
`PlanInfo.residual` reports it and `healthy` compares it with `--health-tol` (0.05 policy units,
30 ticks). Plain guidance at 5 denoising steps is unhealthy for 83 to 98% of chunks by that
tolerance; projection makes it zero. The runner counts them (`unhealthy`, `max_residual`).

What the tests establish:

- `tests/test_rtc.py`: the mask is Eq. 5, the guidance weight is Eq. 2 (4.25 at tau = 0.2, as in
  the paper's Fig. 7), the autodiff term equals a finite-difference Jacobian product, and on a
  Gaussian flow with a closed-form solution the guided sample matches the frozen prefix and
  follows the exact conditional mean.
- `tests/test_rtc_extras.py`: positional noise is shared by overlapping chunks, the roll-forward
  is exact on a ramp, projection leaves the frozen steps bit-identical, best-of-K returns the
  smallest residual of its draws, the health check flags and projection cures; and the ablation
  rows above are asserted with margin.
- `tests/test_rtc_sim.py`: over 40 simulated episodes per setting the hand-over jump is smaller
  with RTC than with the naive baseline, and the gap grows with the delay; with the guidance
  weight at zero RTC equals the baseline exactly.
- `tests/rtc.rs`: the Copper loop, a Python process and the trained policy run RTC end to end:
  guided chunks flow, the governor measures the delay, goals stay inside the governor's limits.
- the governor's unit tests (scheduler horizon, delay window, pending and timeout, event trigger,
  keyframe round trip), its allocation test (no heap use with requests being emitted) and
  `tests/replay.rs` (requests reproduced bit for bit in a replay without a network session).

In a first 12-episode measurement the mean hand-over jump was 44 ticks with RTC against 74
naive at a delay of 8 cycles, and 50 against 93 at 12 (`rtc_sim.compare` reproduces it). In the
single end-to-end run the two were about equal, since one run holds one fork: the end-to-end test
checks the plumbing, not the gain.

## Units

The arm bridge runs with `units: "raw"`, so the wire carries servo ticks. The runner converts to
and from LeRobot units with the LeRobot calibration file (`copper_policy/calibration.py`), using
the formulas of `lerobot/motors/motors_bus.py`; `python/tests/test_calibration.py` compares them
with the installed LeRobot. Calibrate the arm with LeRobot first.

## Running

```bash
just vla-loop-check
```

The end-to-end tests start `python3 -m copper_policy` and need `import zenoh` to work (`pip install
eclipse-zenoh`); the ACT test also needs `lerobot`. Tests that cannot find them print `skipped`.

## Limits

- The camera is synthetic, and the ACT policy in the tests has random weights and takes joint state only. No trained checkpoint or flow-matching policy is wired in, so Real-Time Chunking is not implemented.
- A killed process leaves servo torque on: `Drop` covers panics and clean exits, not `SIGKILL`.
  Keep a power cutoff within reach.
- The governor's observation history holds 64 entries; `max_age_ms` must fit inside it (checked at startup).
