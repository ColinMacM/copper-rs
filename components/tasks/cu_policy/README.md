# cu-policy

A policy loop for Copper. A policy process, such as a vision-language-action model, proposes
chunks of future actions. A governor inside the Copper graph checks, limits and plays them, and
a scheduler decides when the policy is asked again, so the next chunk is computed while the
current one executes.

| Part | Where | What it is |
| --- | --- | --- |
| Wire format | `src/wire.rs` | The bytes of every message between the graph and the policy process. No dependencies; builds without `std`. |
| Governor | `src/governor.rs` | `ActionGovernor`, a `CuTask`, with the chunk scheduler. |
| Link | `src/link/` | `PolicyLink`, the Zenoh bridge to the policy process. |
| Policy server, Rust | `src/server.rs` | Serves a policy written in Rust over Zenoh. |
| Python package `copper` | `python/copper_policy/` | Serves a policy written in Python over Zenoh; installs the module `copper_policy`. |
| Plugin | `plugin.ron`, `fragments/` | The governor and the link as a static plugin, `cu-policy-loop`. |

## Features

| Feature | Provides |
| --- | --- |
| `std` (default) | The governor, the scheduler and the payloads. Without it the crate is `wire` alone and builds as `no_std`. |
| `link` | The Zenoh bridge `link::PolicyLink`. |
| `server` | The Rust policy server, `server::serve`. |
| `testkit` | Deterministic synthetic sources and a sink for tests and benchmarks. |

## Wire format

`src/wire.rs` defines each message once: `Obs`, `Chunk`, `Exec`, `Request` and `ImageHeader`, as
little-endian, fixed-width fields. `tests/golden/vectors.json` holds byte-exact vectors, written
by an independent encoder, that the Rust tests (`tests/golden.rs`) and the Python tests
(`python/tests/test_golden.py`) both check. The payload types in `payloads.rs` encode through
the same functions, so the bytes on the link and the fields in the Copper log follow one layout.

Decoding fills buffers the caller provides, never allocates, and refuses a length above the
capacity of the message before reading any value.

## Governor

`ActionGovernor` is the last software gate between a remote policy and an arm. Everything
`process()` touches has a fixed capacity, so the cycle never allocates.

Ports, in the order the task binds them:

| Direction | Message | Meaning |
| --- | --- | --- |
| input | `JointPositions` | The measured joint positions. |
| input | `ObsStamp` | The sequence number of the observation sent to the policy this cycle. |
| input | `ActionChunk` | A chunk from the policy; the plugin connects it from the link. |
| output | `JointPositions` | The goal for the arm this cycle. |
| output | `ExecState` | The active chunk and the step played, the delay the last accepted chunk had, why a chunk was refused, and the tracking error. |
| output | `InferenceRequest` | Present on the cycles where the scheduler asks the policy again. |

A chunk is refused when it is malformed, holds a non-finite value, is older than `max_age_ms`,
names an observation the governor never stamped, or is older than the chunk already accepted.
An accepted chunk starts at the step that matches the time that has passed since its
observation, so a chunk that arrives `d` cycles late continues the trajectory it was computed
for. Each goal is clamped to the joint limits, to `max_step` away from the previous goal and to
`max_lead` away from the measurement. The governor holds its last goal when no chunk was
accepted within `hold_deadline_ms` or no valid measurement arrived; its status per cycle is
`nogoal`, `play`, `hold`, `exhausted`, `expired` or `nofeedback`.

Configuration keys (goal units are the units of the arm's commands):

| Key | Meaning |
| --- | --- |
| `min_0` ... `min_5`, `max_0` ... `max_5` | Joint limits. |
| `max_step`, `max_lead` | Largest change per cycle, and largest distance from the measurement. |
| `cycle_ms` | The control period; sets how many steps a late chunk skips. |
| `max_age_ms` | Oldest accepted chunk, measured from its observation. At most 63 cycles. |
| `hold_deadline_ms` | How long a chunk keeps playing without a newer one. |
| `time_from_feedback` | Take "now" from the measurement's time (default `true`), which makes a replay exact. |
| `blend_steps` | Cross-fade between the old and the new chunk over this many steps (default 0). |

### Chunk scheduler

The scheduler implements the asynchronous schedule of real-time chunking
([arXiv 2506.07339](https://arxiv.org/abs/2506.07339), Algorithm 1) inside the governor, which
holds the active chunk, the step it is on and the observation history. On the cycles where the
next inference should start it emits an `InferenceRequest`: the observation, the delay estimate
`d`, the number of steps `s` already played, and the unplayed remainder of the active chunk. The
policy process keeps no state. The request is a CopperList output, so a replay asks at the same
cycles with the same contents.

| Key | Meaning |
| --- | --- |
| `sched_s_min` | Steps of a chunk to play before the next request. 0 turns the scheduler off. |
| `sched_margin` | The horizon is at least the delay estimate plus this (default 4). |
| `sched_d_init` | Delay assumed until one has been measured (default 3). |
| `sched_horizon` | The policy's prediction horizon; the horizon never exceeds it minus `d` (default 50). |
| `replan_threshold` | Tracking error above which the plan is replaced at once; 0 turns it off. |
| `sched_pending_timeout` | Cycles after which an unanswered request is asked again (default 25). |

The delay estimate is the largest of the last ten delays the governor measured: it knows the step
an accepted chunk starts at (`ExecState.accept_skip`), which is the delay that chunk had. A chunk
plays for `s_min` steps, so `hold_deadline_ms` has to exceed that time.

## Link

`link::PolicyLink` is a bridge whose cycle side only copies into fixed slots; a worker thread
owns the Zenoh session. A full ring drops the new message and a mailbox keeps the newest
sample; both are counted. The worker opens the session in the background and reconnects, so an
absent peer or router does not stall the graph.

| Channel | Direction | Message |
| --- | --- | --- |
| `obs` | out | `ObsPacket`: measured state with its sequence number and time |
| `img` | out | `CuImage<Vec<u8>>`: a camera frame, handed to the worker as a pooled-buffer handle |
| `exec` | out | `ExecState` |
| `infer` | out | `InferenceRequest` |
| `action` | in | `ActionChunk` |
| `status` | in | `LinkStatus`: the link's counters, emitted on change and at least every 30 cycles |

A channel's `route` is its Zenoh key; the Python package derives all of them from one
`--key-prefix`. The session is configured with the bridge's `zenoh_config_json` or
`zenoh_config_file` keys.

## Writing a policy

A policy answers the governor's `InferenceRequest`s: the observation, the delay estimate `d`, the
steps `s` of the active chunk already played, and that chunk's unplayed remainder. It returns a
chunk, steps of 6 values, whose first step belongs to the control cycle of the request's
observation. The scheduler asks only when `s_min` is above 0. In both languages the policy server
subscribes to `<prefix>/infer` and publishes on `<prefix>/action`, where `<prefix>` is the
plugin's instance name.

In Python (`copper_policy.server`):

```python
import math
from copper_policy import server

def policy(request):  # request: wire.Request
    return [2048 + 400 * math.sin(0.08 * (request.obs_seq + i)) for i in range(20) for _ in range(6)]

server.serve(policy, connect_port=7447, seconds=60, prefix="vla")
```

In Rust (`cu_policy::server`, feature `server`):

```rust
use std::sync::atomic::AtomicBool;
use cu_policy::server::{ServerConfig, serve};
use cu_policy::wire::{CHUNK_LEN, JOINTS, Request};

let policy = |request: &Request, out: &mut [f32; CHUNK_LEN]| {
    for (i, step) in out.as_chunks_mut::<JOINTS>().0.iter_mut().take(20).enumerate() {
        step.fill(2048.0 + 400.0 * ((request.obs_seq as f32 + i as f32) * 0.08).sin());
    }
    20 * JOINTS
};
let stats = serve(policy, ServerConfig::new("vla"), &AtomicBool::new(false))?;
```

`server::answer` in both languages is the request-to-reply step on its own: decode a request, ask
the policy, encode the chunk. A request that does not decode, and a plan that is not whole steps
within 50 steps, are refused and counted. `ServerConfig::from_json5` takes the Zenoh session
settings in the form of the link's `zenoh_config_json`. `tests/rust_policy.rs` and
`tests/python_policy.rs` of `examples/cu_vla_loop` run the loop against each.

## Python package

```bash
pip install ./components/tasks/cu_policy            # copper: zenoh, numpy
pip install "./components/tasks/cu_policy[flow]"    # adds torch: real-time chunking, flow policies
pip install "./components/tasks/cu_policy[act]"     # adds torch and lerobot: the ACT policy
python -m copper_policy --connect-port 7447 --key-prefix vla
```

`copper_policy.wire` is the codec; `runner` is the Zenoh transport and the command line. The
policies are `--policy scripted` (a sine for testing), `--policy act` (a LeRobot ACT policy with
`--calibration`, which converts between LeRobot's units and servo ticks) and `--policy flow`
(a flow-matching policy from `--checkpoint`, trained with `python -m copper_policy.flow_policy`).

### Real-time chunking

`copper_policy.rtc` implements Real-Time Chunking from the paper: pseudoinverse-guided flow
matching (Eq. 2-4) with the soft mask of Eq. 5 and the clipped guidance weight. The first `d`
actions of the new chunk are frozen to the executing chunk's unplayed actions and the rest is
inpainted to agree with them. `--rtc on` (default) does this; `--rtc off` samples each chunk
freely.

Options of `--policy flow`, measured in `copper_policy.rtc_sim.ablate` (40 seeded episodes, 200
hand-overs per row, a delay of 8 cycles; jump and acceleration in ticks):

| Option | What it does | Mean jump / max jump / acceleration |
| --- | --- | --- |
| `--rtc off` | sample freely, replace the old chunk | 78 / 349 / 150 |
| `--rtc on` | the paper | 45 / 189 / 89 |
| `--positional-noise` | the noise of step `t` is a fixed function of `(seed, t)`, so chunks that overlap start their overlapping steps from the same noise | with RTC 36 / 108 / 75 |
| `--roll-obs` | plan from the state expected after the frozen steps, then put those steps in front | 37 / 83 / 73 |
| `--best-of K` | draw K guided samples and keep the one with the smallest weighted prefix residual | K = 4: 37 / 110 / 75 |
| `--project` | set the frozen steps to the previous chunk's values exactly | 45 / 189 / 89; the guarantee is the exact prefix |
| governor `blend_steps` | the played target moves from the old chunk's step to the new one's | 4 steps: 17 / 46 / 66 |

`--health-tol` sets the prefix residual (in policy units) above which a chunk counts as
unhealthy; the runner reports `unhealthy` and `max_residual`.

## Plugin

`plugin.ron` and `fragments/loop.ron` make the governor and the link a static plugin,
`cu-policy-loop` (see `doc/static-plugins.md`). The plugin directory is the crate directory: the
Python modules and the wire vectors are its assets, and its version is the crate's version (a
test checks that the crate, the plugin and `pyproject.toml` agree).

```ron
plugins: [
    (
        path: "../../components/tasks/cu_policy",
        fragment: "loop",
        instance: "vla",
        params: {
            "min_0": 200.0, "max_0": 3900.0,   // ... through joint 5
            "max_step": 30.0, "max_lead": 300.0, "cycle_ms": 33.333,
        },
        pin: "blake3:<just plugin-pin components/tasks/cu_policy>",
    ),
],
```

The instance creates `<instance>_gov` and `<instance>_link`, both public. The application
connects the measured positions and the stamp into `<instance>_gov`, takes the goals out of it,
and connects its observation and camera into `<instance>_link/obs` and `<instance>_link/img`.
The fragment connects the chunk from the link into the governor and the governor's `exec` and
`infer` outputs into the link. The routes are `<instance>/obs`, `/img`, `/exec`, `/infer` and
`/action`, so `--key-prefix <instance>` matches.

The plugin passes every parameter to the governor. The governor's own defaults (scheduler off,
`blend_steps` 0) apply to a graph that configures the governor directly.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `min_0` ... `min_5`, `max_0` ... `max_5` | required | Joint limits. |
| `max_step`, `max_lead`, `cycle_ms` | required | See the governor keys. |
| `max_age_ms` | 600 | See the governor keys. |
| `hold_deadline_ms` | 2500 | See the governor keys. |
| `horizon` | 50 | `sched_horizon`. |
| `s_min` | 25 | `sched_s_min`; 0 turns the scheduler off. |
| `d_init` | 3 | `sched_d_init`. |
| `blend_steps` | 3 | `blend_steps`. |

## Tests

```bash
just vla-loop-check
```

runs clippy and the tests of this crate, the plugin validation, the end-to-end tests of
`examples/cu_vla_loop` and the Python tests. The end-to-end tests start `python -m copper_policy`
and need `zenoh` importable; the ACT test also needs `lerobot` and the flow tests `torch`. A
test whose Python dependencies are missing prints `skipped`.
