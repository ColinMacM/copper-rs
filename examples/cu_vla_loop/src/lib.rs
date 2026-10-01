//! A policy-driven arm loop.
//!
//! `arm/positions` -> `obs` -> `link/obs` -> (Zenoh) -> Python policy -> `link/action` -> `gov`
//! -> `arm/goals`. The arm here is a simulation with the behaviors of a `cu_feetech` follower
//! that matter to safety: a goal converts to a raw position with a saturating cast (NaN becomes
//! 0), and the servo moves toward its last goal at a limited slew rate.

use cu_sensor_payloads::{CuImage, CuImageBufferFormat};
use cu29::prelude::*;
use std::sync::{Arc, Mutex};

pub use cu_action_governor::JointPositions;
use cu_action_governor::{ExecState, InferenceRequest, ObsPacket, ObsStamp};
pub use cu_policy_link::LinkStatus;

/// Every goal the mock arm received and every position it reported, one entry per cycle.
/// Reserved up front so recording does not allocate on the cycle.
#[derive(Default)]
pub struct ArmTrace {
    pub goals: Vec<Option<[f32; 8]>>,
    pub positions: Vec<[f32; 8]>,
}

pub static TRACE: Mutex<ArmTrace> = Mutex::new(ArmTrace {
    goals: Vec::new(),
    positions: Vec::new(),
});

/// The newest link status the graph saw. Tests read it; the same messages are in the log.
pub static LAST_STATUS: Mutex<Option<LinkStatus>> = Mutex::new(None);

/// Called with `true` before and `false` after the camera takes a buffer from its pool. Taking
/// one allocates the `Arc` of the handle inside Copper's pool, a cost of any Copper camera that
/// the link does not add; the allocation test excludes exactly that call with this hook.
pub static POOL_ACQUIRE_PROBE: std::sync::OnceLock<fn(bool)> = std::sync::OnceLock::new();

/// Round-trip measurement. `ObsBuilder` stamps the Copper-clock time of each observation by
/// sequence number; `ChunkProbe` subtracts it when the chunk answering that observation comes
/// back. Both run on the cycle thread and only touch preallocated atomics and a reserved Vec.
const SENT_SLOTS: usize = 256;
static SENT_NS: [std::sync::atomic::AtomicU64; SENT_SLOTS] =
    [const { std::sync::atomic::AtomicU64::new(0) }; SENT_SLOTS];

/// `(obs_seq, round trip in ns)` of the first chunk that answered each observation.
pub static ROUND_TRIPS: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

/// Clears the round-trip record and reserves room for `n` entries.
pub fn reset_round_trips(n: usize) {
    let mut r = ROUND_TRIPS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *r = Vec::with_capacity(n);
}

const TRACE_CAPACITY: usize = 65_536;

/// Clears the trace and reserves capacity for a run.
pub fn reset_trace() {
    let mut t = TRACE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    t.goals = Vec::with_capacity(TRACE_CAPACITY);
    t.positions = Vec::with_capacity(TRACE_CAPACITY);
}

/// Frame size of the synthetic camera, and the pattern it paints: byte `i` of the frame with
/// sequence number `seq` is `(i + seq * 31) % 256`. A peer recomputes it to prove that a frame
/// arrived whole and is the frame it claims to be.
pub const FRAME_WIDTH: u32 = 160;
pub const FRAME_HEIGHT: u32 = 120;
pub const PIXEL_FORMAT: [u8; 4] = *b"RGB3";

#[must_use]
pub fn pattern_offset(seq: u64) -> usize {
    ((seq % 256) * 31 % 256) as usize
}

pub mod bridges {
    use super::*;
    use cu_action_governor::ActionChunk;

    tx_channels! { pub struct ArmTx : ArmTxId { goals => JointPositions = "goals" } }
    rx_channels! { pub struct ArmRx : ArmRxId { positions => JointPositions = "positions" } }

    tx_channels! {
        pub struct LinkTx : LinkTxId {
            obs => ObsPacket = "vla/obs",
            img => CuImage<Vec<u8>> = "vla/img",
            exec => ExecState = "vla/exec",
            infer => InferenceRequest = "vla/infer",
        }
    }
    rx_channels! {
        pub struct LinkRx : LinkRxId {
            action => ActionChunk = "vla/action",
            status => LinkStatus = "vla/link_status",
        }
    }

    pub type PolicyLink = cu_policy_link::PolicyLinkBridge<LinkTx, LinkRx>;

    #[derive(Reflect)]
    #[reflect(from_reflect = false)]
    pub struct MockArm {
        position: [f32; 8],
        goal: Option<[f32; 8]>,
        slew: f32,
    }

    impl Freezable for MockArm {}

    impl CuBridge for MockArm {
        type Tx = ArmTx;
        type Rx = ArmRx;
        type Resources<'r> = ();

        fn new(
            config: Option<&ComponentConfig>,
            _tx: &[BridgeChannelConfig<ArmTxId>],
            _rx: &[BridgeChannelConfig<ArmRxId>],
            _resources: Self::Resources<'_>,
        ) -> CuResult<Self> {
            let get = |k: &str, default: f64| -> CuResult<f32> {
                Ok(match config {
                    Some(c) => c.get::<f64>(k)?.unwrap_or(default) as f32,
                    None => default as f32,
                })
            };
            Ok(Self {
                position: [get("start", 2048.0)?; 8],
                goal: None,
                slew: get("slew", 40.0)?,
            })
        }

        fn send<'a, Payload>(
            &mut self,
            _ctx: &CuContext,
            _channel: &'static BridgeChannel<ArmTxId, Payload>,
            msg: &CuMsg<Payload>,
        ) -> CuResult<()>
        where
            Payload: CuMsgPayload + 'a,
        {
            // The only Tx channel carries `JointPositions`.
            let Some(msg) = (msg as &dyn std::any::Any).downcast_ref::<CuMsg<JointPositions>>()
            else {
                return Err(CuError::from("MockArm: goals must be JointPositions"));
            };
            let Some(payload) = msg.payload() else {
                return Ok(());
            };
            let mut goal = [0f32; 8];
            for (slot, v) in goal.iter_mut().zip(payload.as_slice()) {
                *slot = *v;
            }
            if let Ok(mut t) = TRACE.try_lock()
                && t.goals.len() < t.goals.capacity()
            {
                t.goals.push(Some(goal));
            }
            // A servo converts the goal with a saturating cast: NaN becomes raw 0.
            let mut raw = [0f32; 8];
            for (r, g) in raw.iter_mut().zip(goal) {
                *r = f32::from(g.round().clamp(0.0, 65535.0) as u16);
            }
            self.goal = Some(raw);
            Ok(())
        }

        fn receive<'a, Payload>(
            &mut self,
            ctx: &CuContext,
            _channel: &'static BridgeChannel<ArmRxId, Payload>,
            msg: &mut CuMsg<Payload>,
        ) -> CuResult<()>
        where
            Payload: CuMsgPayload + 'a,
        {
            if let Some(goal) = self.goal {
                for (p, g) in self.position.iter_mut().zip(goal) {
                    *p += (g - *p).clamp(-self.slew, self.slew);
                }
            }
            let Some(out) = (msg as &mut dyn std::any::Any).downcast_mut::<CuMsg<JointPositions>>()
            else {
                return Err(CuError::from("MockArm: positions must be JointPositions"));
            };
            let mut arr = JointPositions::new();
            arr.fill_from_iter(self.position);
            out.set_payload(arr);
            out.tov = Tov::Time(ctx.now());
            if let Ok(mut t) = TRACE.try_lock()
                && t.positions.len() < t.positions.capacity()
            {
                t.positions.push(self.position);
            }
            Ok(())
        }
    }
}

pub mod tasks {
    use super::*;

    /// Stand-in for a camera driver: hands out pooled buffers that it paints with the frame
    /// pattern, like DMA would fill them. Nothing on the cycle allocates.
    #[derive(Reflect)]
    #[reflect(from_reflect = false)]
    pub struct SyntheticCamera {
        #[reflect(ignore)]
        pool: Arc<CuHostMemoryPool<Vec<u8>>>,
        #[reflect(ignore)]
        pattern: Vec<u8>,
        #[reflect(ignore)]
        format: CuImageBufferFormat,
        seq: u64,
    }

    impl Freezable for SyntheticCamera {
        fn freeze<E: cu29::bincode::enc::Encoder>(
            &self,
            e: &mut E,
        ) -> Result<(), cu29::bincode::error::EncodeError> {
            cu29::bincode::Encode::encode(&self.seq, e)
        }

        fn thaw<D: cu29::bincode::de::Decoder>(
            &mut self,
            d: &mut D,
        ) -> Result<(), cu29::bincode::error::DecodeError> {
            self.seq = cu29::bincode::Decode::decode(d)?;
            Ok(())
        }
    }

    impl CuSrcTask for SyntheticCamera {
        type Resources<'r> = ();
        type Output<'m> = output_msg!(CuImage<Vec<u8>>);

        fn new(_c: Option<&ComponentConfig>, _r: Self::Resources<'_>) -> CuResult<Self> {
            let format = CuImageBufferFormat {
                width: FRAME_WIDTH,
                height: FRAME_HEIGHT,
                stride: FRAME_WIDTH * 3,
                pixel_format: PIXEL_FORMAT,
            };
            let len = format.byte_size();
            // Enough slots for the frame being produced, the ones in the link's ring, the one
            // the worker is copying and the log's hold on a frame.
            let pool = CuHostMemoryPool::new("vla_camera", 8, || vec![0u8; len])?;
            let pattern = (0..len + 256).map(|i| (i % 256) as u8).collect();
            Ok(Self {
                pool,
                pattern,
                format,
                seq: 0,
            })
        }

        fn process(&mut self, ctx: &CuContext, out: &mut Self::Output<'_>) -> CuResult<()> {
            self.seq += 1;
            let probe = POOL_ACQUIRE_PROBE.get();
            if let Some(p) = probe {
                p(true);
            }
            let acquired = self.pool.acquire();
            if let Some(p) = probe {
                p(false);
            }
            let handle = acquired.ok_or_else(|| CuError::from("camera pool exhausted"))?;
            let off = pattern_offset(self.seq);
            handle.with_inner_mut(|buf| {
                let len = buf.len();
                buf.copy_from_slice(&self.pattern[off..off + len]);
            });
            let mut image = CuImage::new(self.format, handle);
            image.seq = self.seq;
            out.tov = Tov::Time(ctx.now());
            out.set_payload(image);
            Ok(())
        }
    }

    /// Consumes the link's status messages; the log keeps them, this keeps the newest for tests.
    #[derive(Reflect)]
    pub struct StatusProbe;

    impl Freezable for StatusProbe {}

    impl CuSinkTask for StatusProbe {
        type Resources<'r> = ();
        type Input<'m> = input_msg!(LinkStatus);

        fn new(_c: Option<&ComponentConfig>, _r: Self::Resources<'_>) -> CuResult<Self> {
            Ok(Self)
        }

        fn process(&mut self, _ctx: &CuContext, input: &Self::Input<'_>) -> CuResult<()> {
            if let Some(status) = input.payload()
                && let Ok(mut last) = LAST_STATUS.try_lock()
            {
                *last = Some(*status);
            }
            Ok(())
        }
    }

    /// Records how long a policy takes to answer an observation, for latency measurements.
    #[derive(Reflect)]
    pub struct ChunkProbe {
        last_answered: Option<u64>,
    }

    impl Freezable for ChunkProbe {}

    impl CuSinkTask for ChunkProbe {
        type Resources<'r> = ();
        type Input<'m> = input_msg!(cu_action_governor::ActionChunk);

        fn new(_c: Option<&ComponentConfig>, _r: Self::Resources<'_>) -> CuResult<Self> {
            Ok(Self {
                last_answered: None,
            })
        }

        fn process(&mut self, ctx: &CuContext, input: &Self::Input<'_>) -> CuResult<()> {
            let Some(chunk) = input.payload() else {
                return Ok(());
            };
            if self.last_answered == Some(chunk.obs_seq) {
                return Ok(());
            }
            self.last_answered = Some(chunk.obs_seq);
            let sent = SENT_NS[chunk.obs_seq as usize % SENT_SLOTS]
                .load(std::sync::atomic::Ordering::Relaxed);
            let now = ctx.now().as_nanos();
            if sent != 0
                && now >= sent
                && let Ok(mut r) = ROUND_TRIPS.try_lock()
                && r.len() < r.capacity()
            {
                r.push((chunk.obs_seq, now - sent));
            }
            Ok(())
        }
    }

    /// Numbers each measured position so the policy can name the observation it used.
    #[derive(Reflect)]
    pub struct ObsBuilder {
        seq: u64,
    }

    impl Freezable for ObsBuilder {
        fn freeze<E: cu29::bincode::enc::Encoder>(
            &self,
            e: &mut E,
        ) -> Result<(), cu29::bincode::error::EncodeError> {
            cu29::bincode::Encode::encode(&self.seq, e)
        }

        fn thaw<D: cu29::bincode::de::Decoder>(
            &mut self,
            d: &mut D,
        ) -> Result<(), cu29::bincode::error::DecodeError> {
            self.seq = cu29::bincode::Decode::decode(d)?;
            Ok(())
        }
    }

    impl CuTask for ObsBuilder {
        type Resources<'r> = ();
        type Input<'m> = input_msg!(JointPositions);
        type Output<'m> = output_msg!(ObsPacket, ObsStamp);

        fn new(_c: Option<&ComponentConfig>, _r: Self::Resources<'_>) -> CuResult<Self> {
            Ok(Self { seq: 0 })
        }

        fn process(
            &mut self,
            _ctx: &CuContext,
            input: &Self::Input<'_>,
            output: &mut Self::Output<'_>,
        ) -> CuResult<()> {
            let (packet, stamp) = output;
            match input.payload() {
                Some(state) => {
                    SENT_NS[self.seq as usize % SENT_SLOTS]
                        .store(_ctx.now().as_nanos(), std::sync::atomic::Ordering::Relaxed);
                    packet.set_payload(ObsPacket {
                        seq: self.seq,
                        tov_ns: match input.tov {
                            Tov::Time(t) => t.as_nanos(),
                            _ => 0,
                        },
                        state: state.clone(),
                    });
                    packet.tov = input.tov;
                    stamp.set_payload(ObsStamp { seq: self.seq });
                    stamp.tov = input.tov;
                    self.seq += 1;
                }
                None => {
                    packet.clear_payload();
                    stamp.clear_payload();
                }
            }
            Ok(())
        }
    }
}

#[copper_runtime(config = "copperconfig.ron")]
struct VlaLoopApp {}

/// Runs the loop for `cycles` cycles at `hz` (0 = as fast as possible). Tests read [`TRACE`].
pub fn run(cycles: usize, hz: f64, log_path: &std::path::Path, zenoh_json: &str) -> CuResult<()> {
    run_hooked(cycles, hz, log_path, zenoh_json, |_| {}, |_| {})
}

/// Like [`run`], calling `before(i)` and `after(i)` around cycle `i`.
pub fn run_hooked(
    cycles: usize,
    hz: f64,
    log_path: &std::path::Path,
    zenoh_json: &str,
    before: impl FnMut(usize),
    after: impl FnMut(usize),
) -> CuResult<()> {
    run_configured(cycles, hz, log_path, zenoh_json, &[], before, after)
}

/// Like [`run_hooked`], with governor settings replaced by `governor` (key, value) pairs.
/// Real-time chunking needs a longer `hold_deadline_ms` than the default: the governor stops
/// playing a chunk that was not replaced within that time, and RTC lets a chunk play for its
/// minimum execution horizon before the next one is computed.
pub fn run_configured(
    cycles: usize,
    hz: f64,
    log_path: &std::path::Path,
    zenoh_json: &str,
    governor: &[(&str, f64)],
    mut before: impl FnMut(usize),
    mut after: impl FnMut(usize),
) -> CuResult<()> {
    let mut config = CuConfig::deserialize_ron(&<VlaLoopApp as CuApplication<
        memmap::MmapSectionStorage,
        UnifiedLoggerWrite,
    >>::get_original_config())?;
    let link = config
        .bridges
        .iter_mut()
        .find(|b| b.id == "link")
        .ok_or_else(|| CuError::from("link bridge missing from the configuration"))?;
    link.config
        .get_or_insert_with(ComponentConfig::default)
        .set("zenoh_config_json", zenoh_json.to_string());
    if !governor.is_empty() {
        let graph = config.get_graph_mut(None)?;
        let id = graph
            .get_node_id_by_name("gov")
            .ok_or_else(|| CuError::from("gov task missing from the configuration"))?;
        let node = graph
            .get_node_mut(id)
            .ok_or_else(|| CuError::from("gov node missing from the graph"))?;
        for (key, value) in governor {
            node.set_param(key, *value);
        }
    }
    let app = VlaLoopApp::builder()
        .with_log_path(log_path, Some(64 * 1024 * 1024))?
        .with_config(config)
        .build()?;
    reset_trace();
    let mut running = app.start()?;
    let period = if hz > 0.0 {
        std::time::Duration::from_secs_f64(1.0 / hz)
    } else {
        std::time::Duration::ZERO
    };
    let start = std::time::Instant::now();
    for i in 0..cycles {
        before(i);
        running.run_one_iteration()?;
        after(i);
        if !period.is_zero() {
            let due = start + period * (i as u32 + 1);
            if let Some(rest) = due.checked_duration_since(std::time::Instant::now()) {
                std::thread::sleep(rest);
            }
        }
    }
    running.stop()?;
    Ok(())
}

/// Zenoh settings for the app side: listen on loopback, no multicast scouting.
#[must_use]
pub fn listen_config(port: u16) -> String {
    format!(
        r#"{{mode:"peer",scouting:{{multicast:{{enabled:false}}}},listen:{{endpoints:["tcp/127.0.0.1:{port}"]}},connect:{{endpoints:[]}}}}"#
    )
}
