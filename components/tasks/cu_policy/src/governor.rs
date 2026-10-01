use crate::payloads::{
    ActionChunk, CHUNK_LEN, ExecState, FLAG_POSITIONAL_NOISE, FLAG_PROJECT, FLAG_ROLL_OBS,
    InferenceRequest, JOINTS, MAX_STEPS, MODE_NAIVE, MODE_RTC, ObsStamp,
};
use cu29::bincode::de::Decoder;
use cu29::bincode::enc::Encoder;
use cu29::bincode::error::{DecodeError, EncodeError};
use cu29::bincode::{Decode, Encode};
use cu29::prelude::*;

/// Observations remembered so a chunk can be aged by the observation it names. At one
/// observation per cycle this covers `OBS_RING` cycles; `max_age_ms` must fit inside it.
const OBS_RING: usize = 64;

/// Delays the scheduler remembers for its estimate (the `b` of Algorithm 1).
const DELAY_RING: usize = 10;

pub type JointPositions = CuArray<f32, 8>; // same type cu_feetech publishes / consumes

/// The chunk scheduler of real-time chunking (Algorithm 1 of arXiv:2506.07339): decides when the
/// next inference starts and hands the policy what it needs. Disabled when `s_min` is 0.
#[derive(Debug, Clone, Copy, Reflect)]
pub struct SchedParams {
    /// Minimum execution horizon: steps of a chunk to play before the next inference starts.
    pub s_min: u32,
    /// The horizon grows to the delay estimate plus this, so that the answer is not due before
    /// the steps it replaces have played.
    pub margin: u32,
    /// Delay assumed until a real one has been measured, in cycles.
    pub d_init: u32,
    /// Prediction horizon `H` in steps; the execution horizon never exceeds `H - d`.
    pub horizon: u32,
    /// Tracking error (goal units) above which the plan is replaced at once, regardless of
    /// `s_min`; 0 disables.
    pub replan_threshold: f32,
    /// Cycles after which a request that was never answered is dropped.
    pub pending_timeout: u32,
    /// How the policy plans, sent with every request: `MODE_NAIVE` or `MODE_RTC`.
    pub mode: u32,
    /// Denoising steps of a flow policy.
    pub denoise_steps: u32,
    /// Guided samples drawn per chunk.
    pub best_of: u32,
    /// `FLAG_*` bits.
    pub flags: u32,
    /// Clip of the guidance weight.
    pub beta: f32,
    /// Cross-chunk handover: after a new chunk is accepted, the played target moves from the
    /// old chunk's step to the new chunk's over this many steps (weight `(n + 1) / (blend + 1)`
    /// for the n-th), so a late or inconsistent chunk cannot make the target jump. 0 disables.
    pub blend_steps: u32,
}

impl Default for SchedParams {
    fn default() -> Self {
        Self {
            s_min: 0,
            margin: 4,
            d_init: 3,
            horizon: MAX_STEPS as u32,
            replan_threshold: 0.0,
            pending_timeout: 25,
            mode: MODE_NAIVE,
            denoise_steps: 5,
            best_of: 1,
            flags: 0,
            beta: 5.0,
            blend_steps: 0,
        }
    }
}

/// What happened to a chunk offered to the governor.
enum Offer {
    Ignored,
    Accepted(u32),
    Rejected(u32),
}

/// Static limits, read once from the RON config in `new()`.
#[derive(Debug, Clone, Reflect)]
pub struct GovernorParams {
    pub min: [f32; JOINTS],
    pub max: [f32; JOINTS],
    /// Max change of the commanded goal per cycle (goal units).
    pub max_step: f32,
    /// Max distance of a commanded goal from the measured position.
    pub max_lead: f32,
    pub max_age_ns: u64,
    pub hold_deadline_ns: u64,
    /// 0 disables skipping the steps that elapsed while the chunk was in flight.
    pub cycle_ns: u64,
    /// true: take "now" from the feedback message's Tov (recorded => exact replay).
    pub time_from_feedback: bool,
    pub sched: SchedParams,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    NoGoal,
    Play,
    Hold,
    HoldExhausted,
    HoldExpired,
    /// No valid measurement this cycle: the goal is held and the chunk does not advance.
    HoldNoFeedback,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Status::NoGoal => "nogoal",
            Status::Play => "play",
            Status::Hold => "hold",
            Status::HoldExhausted => "exhausted",
            Status::HoldExpired => "expired",
            Status::HoldNoFeedback => "nofeedback",
        }
    }
}

/// Everything that evolves; frozen into keyframes. Pure logic, no Copper types in `step`.
#[derive(Debug, Clone, Reflect)]
#[reflect(from_reflect = false)]
pub struct GovernorCore {
    pub params: GovernorParams,
    goal: [f32; JOINTS],
    have_goal: bool,
    chunk: CuArray<f32, CHUNK_LEN>,
    cursor: u32,
    active: bool,
    last_seq: u64,
    have_seq: bool,
    last_accept_ns: u64,
    ring_seq: [u64; OBS_RING],
    ring_ns: [u64; OBS_RING],
    ring_len: u32,
    ring_head: u32,
    pub accepted: u32,
    pub rej_shape: u32,
    pub rej_nonfinite: u32,
    pub rej_order: u32,
    pub rej_unknown_obs: u32,
    pub rej_stale: u32,
    pub held_cycles: u32,
    pub bad_feedback: u32,
    /// Raw target (before the governor's limits) played last cycle, to measure chunk switches.
    last_target: [f32; JOINTS],
    have_target: bool,
    switched: bool,
    /// Chunk switches that happened while playing, with the jump each made in the policy's own
    /// targets: the largest change of any joint between the last step of the old chunk and the
    /// first step played from the new one. Smooth hand-over keeps this near one step of motion.
    pub switches: u32,
    pub switch_jump_max: f32,
    pub switch_jump_sum: f32,
    exec: ExecState,
    #[reflect(ignore)]
    request: Option<InferenceRequest>,
    sched_delays: [u32; DELAY_RING],
    sched_len: u32,
    sched_head: u32,
    sched_pending: bool,
    sched_pending_seq: u64,
    sched_pending_since: u64,
    prev_chunk: CuArray<f32, CHUNK_LEN>,
    prev_cursor: u32,
    blend_n: u32,
    blend_active: bool,
}

#[inline]
fn clamp(v: f32, lo: f32, hi: f32) -> f32 {
    // not f32::clamp: that panics when lo > hi, and v is finite here
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

impl GovernorCore {
    pub fn new(params: GovernorParams) -> Self {
        Self {
            params,
            goal: [0.0; JOINTS],
            have_goal: false,
            chunk: CuArray::new(),
            cursor: 0,
            active: false,
            last_seq: 0,
            have_seq: false,
            last_accept_ns: 0,
            ring_seq: [0; OBS_RING],
            ring_ns: [0; OBS_RING],
            ring_len: 0,
            ring_head: 0,
            accepted: 0,
            rej_shape: 0,
            rej_nonfinite: 0,
            rej_order: 0,
            rej_unknown_obs: 0,
            rej_stale: 0,
            held_cycles: 0,
            bad_feedback: 0,
            last_target: [0.0; JOINTS],
            have_target: false,
            switched: false,
            switches: 0,
            switch_jump_max: 0.0,
            switch_jump_sum: 0.0,
            exec: ExecState::default(),
            request: None,
            sched_delays: [0; DELAY_RING],
            sched_len: 0,
            sched_head: 0,
            sched_pending: false,
            sched_pending_seq: 0,
            sched_pending_since: 0,
            prev_chunk: CuArray::new(),
            prev_cursor: 0,
            blend_n: 0,
            blend_active: false,
        }
    }

    fn obs_time(&self, seq: u64) -> Option<u64> {
        (0..self.ring_len as usize)
            .find(|&i| self.ring_seq[i] == seq)
            .map(|i| self.ring_ns[i])
    }

    fn push_obs(&mut self, seq: u64, ns: u64) {
        let h = self.ring_head as usize;
        self.ring_seq[h] = seq;
        self.ring_ns[h] = ns;
        self.ring_head = ((h + 1) % OBS_RING) as u32;
        self.ring_len = (self.ring_len + 1).min(OBS_RING as u32);
    }

    fn offer(&mut self, now: u64, chunk: &ActionChunk) -> Offer {
        let v = chunk.values.as_slice();
        if self.have_seq && chunk.obs_seq == self.last_seq {
            return Offer::Ignored; // bridge re-emitting the same chunk: not a new one, not an error
        }
        if v.is_empty() || !v.len().is_multiple_of(JOINTS) {
            self.rej_shape += 1;
            return Offer::Rejected(ExecState::REJECT_SHAPE);
        }
        if v.iter().any(|x| !x.is_finite()) {
            self.rej_nonfinite += 1;
            return Offer::Rejected(ExecState::REJECT_NONFINITE);
        }
        if self.have_seq && chunk.obs_seq < self.last_seq {
            self.rej_order += 1;
            return Offer::Rejected(ExecState::REJECT_ORDER);
        }
        let Some(obs_ns) = self.obs_time(chunk.obs_seq) else {
            self.rej_unknown_obs += 1;
            return Offer::Rejected(ExecState::REJECT_UNKNOWN_OBS);
        };
        let age = now.saturating_sub(obs_ns);
        if age > self.params.max_age_ns {
            self.rej_stale += 1;
            return Offer::Rejected(ExecState::REJECT_STALE);
        }
        let steps_before = (self.chunk.as_slice().len() / JOINTS) as u32;
        if self.params.sched.blend_steps > 0 && self.active && self.cursor < steps_before {
            // The step the old chunk would play this cycle is where the crossfade starts from.
            self.prev_chunk.clone_from(&self.chunk);
            self.prev_cursor = self.cursor;
            self.blend_n = 0;
            self.blend_active = true;
        } else {
            self.blend_active = false;
        }
        self.chunk.clone_from(&chunk.values);
        // `cycle_ns == 0` disables skipping the steps that elapsed while the chunk was in flight.
        self.cursor = age
            .checked_div(self.params.cycle_ns)
            .unwrap_or(0)
            .min(u64::from(u32::MAX)) as u32;
        self.active = true;
        self.last_seq = chunk.obs_seq;
        self.have_seq = true;
        self.last_accept_ns = now;
        self.accepted += 1;
        self.switched = true;
        Offer::Accepted(self.cursor)
    }

    /// One control cycle. Returns the goal to command (None until a measurement was seen).
    pub fn step(
        &mut self,
        now: u64,
        stamp: Option<u64>,
        chunk: Option<&ActionChunk>,
        feedback: Option<&[f32]>,
    ) -> (Option<[f32; JOINTS]>, Status) {
        self.request = None;
        if let Some(seq) = stamp {
            self.push_obs(seq, now);
        }
        self.note_exec(stamp, false);
        let p = &self.params;
        let meas = match feedback {
            Some(f) if f.len() >= JOINTS && f[..JOINTS].iter().all(|x| x.is_finite()) => {
                let mut m = [0.0; JOINTS];
                m.copy_from_slice(&f[..JOINTS]);
                Some(m)
            }
            Some(_) => {
                self.bad_feedback += 1;
                None
            }
            None => None,
        };
        if !self.have_goal {
            let Some(m) = meas else {
                return (None, Status::NoGoal);
            };
            for (((goal, measured), lo), hi) in self.goal.iter_mut().zip(m).zip(p.min).zip(p.max) {
                *goal = clamp(measured, lo, hi);
            }
            self.have_goal = true;
        }
        if let Some(c) = chunk {
            let offer = self.offer(now, c);
            self.note_exec(stamp, false);
            match offer {
                Offer::Accepted(skip) => {
                    self.exec.flags |= ExecState::ACCEPTED;
                    self.exec.accept_skip = skip;
                    self.push_delay(skip);
                    if self.sched_pending && c.obs_seq >= self.sched_pending_seq {
                        self.sched_pending = false;
                    }
                }
                Offer::Rejected(code) => self.exec.reject = code,
                Offer::Ignored => {}
            }
        }
        if meas.is_none() {
            // Without a measurement the lead window cannot be applied and a failed read may mean
            // the arm is not where the chunk assumes, so nothing is played this cycle.
            self.held_cycles += 1;
            return (Some(self.goal), Status::HoldNoFeedback);
        }

        let p = &self.params;
        let mut status = Status::Hold;
        let mut target = self.goal;
        // Where the measurement is against the raw target played last cycle, before this
        // cycle's step moves it on.
        let tracking_err = match (self.have_target, meas) {
            (true, Some(m)) => m
                .iter()
                .zip(self.last_target)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max),
            _ => 0.0,
        };
        self.exec.tracking_err = tracking_err;
        let played_index = self.cursor;
        if self.active && now.saturating_sub(self.last_accept_ns) > p.hold_deadline_ns {
            self.active = false;
            self.exec.flags &= !ExecState::CHUNK_ACTIVE;
            status = Status::HoldExpired;
        } else if self.active {
            let start = self.cursor as usize * JOINTS;
            let v = self.chunk.as_slice();
            if start + JOINTS <= v.len() {
                target.copy_from_slice(&v[start..start + JOINTS]);
                self.exec.flags |= ExecState::PLAYED;
                if self.blend_active {
                    let old_at = self.prev_cursor as usize * JOINTS;
                    let old = self.prev_chunk.as_slice();
                    let length = p.sched.blend_steps;
                    if self.blend_n < length && old_at + JOINTS <= old.len() {
                        let w = (self.blend_n + 1) as f32 / (length + 1) as f32;
                        for (t, o) in target.iter_mut().zip(&old[old_at..old_at + JOINTS]) {
                            *t = (1.0 - w) * *o + w * *t;
                        }
                        self.prev_cursor += 1;
                        self.blend_n += 1;
                    } else {
                        self.blend_active = false;
                    }
                }
                if self.switched && self.have_target {
                    let jump = target
                        .iter()
                        .zip(self.last_target)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    self.switches += 1;
                    self.switch_jump_sum += jump;
                    self.switch_jump_max = self.switch_jump_max.max(jump);
                }
                self.switched = false;
                self.last_target = target;
                self.have_target = true;
                self.cursor += 1;
                status = Status::Play;
            } else {
                status = Status::HoldExhausted;
            }
        }
        if status != Status::Play {
            self.held_cycles += 1;
        }

        for j in 0..JOINTS {
            let mut t = clamp(target[j], p.min[j], p.max[j]);
            if let Some(m) = meas {
                t = clamp(t, m[j] - p.max_lead, m[j] + p.max_lead);
                t = clamp(t, p.min[j], p.max[j]);
            }
            let d = clamp(t - self.goal[j], -p.max_step, p.max_step);
            self.goal[j] = clamp(self.goal[j] + d, p.min[j], p.max[j]);
        }
        if let Some(m) = meas {
            self.schedule(stamp, m, played_index, tracking_err);
        }
        (Some(self.goal), status)
    }

    /// The scheduler's request for the policy, if this cycle's `step` decided to ask.
    pub fn request(&self) -> Option<&InferenceRequest> {
        self.request.as_ref()
    }

    /// Conservative estimate of the next delay: the largest of the recent ones.
    pub fn delay_estimate(&self) -> u32 {
        if self.sched_len == 0 {
            return self.params.sched.d_init;
        }
        self.sched_delays[..self.sched_len as usize]
            .iter()
            .copied()
            .max()
            .unwrap_or(self.params.sched.d_init)
    }

    fn push_delay(&mut self, delay: u32) {
        let h = self.sched_head as usize;
        self.sched_delays[h] = delay;
        self.sched_head = ((h + 1) % DELAY_RING) as u32;
        self.sched_len = (self.sched_len + 1).min(DELAY_RING as u32);
    }

    /// Algorithm 1's decision, once per observation: start the next inference when the active
    /// chunk has played its execution horizon, or at once when the arm has strayed from it.
    /// `played_index` is the step played this cycle, whose number is also the offset of the
    /// active chunk from the observation, so the remaining steps start at this cycle.
    fn schedule(&mut self, stamp: Option<u64>, meas: [f32; JOINTS], played_index: u32, err: f32) {
        let sp = self.params.sched;
        let Some(seq) = stamp else { return };
        if sp.s_min == 0 {
            return;
        }
        if self.sched_pending
            && seq.saturating_sub(self.sched_pending_since) > u64::from(sp.pending_timeout)
        {
            self.sched_pending = false; // never answered, or refused: stop waiting
        }
        if self.sched_pending {
            return;
        }
        let d = self.delay_estimate();
        let steps = (self.chunk.as_slice().len() / JOINTS) as u32;
        let have_chunk = self.active && played_index < steps;
        let reason = if !have_chunk {
            InferenceRequest::REASON_FIRST
        } else {
            let cap = sp.horizon.saturating_sub(d).max(1);
            let horizon = sp.s_min.max(d + sp.margin).min(cap);
            if sp.replan_threshold > 0.0 && err > sp.replan_threshold {
                InferenceRequest::REASON_EVENT
            } else if played_index >= horizon {
                InferenceRequest::REASON_SCHEDULED
            } else {
                return;
            }
        };
        let mut request = InferenceRequest {
            obs_seq: seq,
            delay: d,
            executed: if have_chunk { played_index } else { 0 },
            reason,
            horizon: sp.horizon,
            mode: sp.mode,
            denoise_steps: sp.denoise_steps,
            best_of: sp.best_of,
            flags: sp.flags,
            beta: sp.beta,
            ..InferenceRequest::default()
        };
        request.state.fill_from_iter(meas.iter().copied());
        if have_chunk {
            let from = played_index as usize * JOINTS;
            request
                .previous
                .fill_from_iter(self.chunk.as_slice()[from..].iter().copied());
        }
        self.request = Some(request);
        self.sched_pending = true;
        self.sched_pending_seq = seq;
        self.sched_pending_since = seq;
    }

    /// Which chunk is executing and where, as of the last `step`.
    pub fn exec_state(&self) -> ExecState {
        self.exec
    }

    fn note_exec(&mut self, stamp: Option<u64>, played: bool) {
        let mut flags = 0;
        if stamp.is_some() {
            flags |= ExecState::HAS_STAMP;
        }
        if self.active {
            flags |= ExecState::CHUNK_ACTIVE;
        }
        if played {
            flags |= ExecState::PLAYED;
        }
        self.exec = ExecState {
            stamp_seq: stamp.unwrap_or(0),
            chunk_seq: self.last_seq,
            next_index: self.cursor,
            flags,
            accept_skip: 0,
            reject: 0,
            tracking_err: 0.0,
        };
    }

    pub fn goal(&self) -> Option<[f32; JOINTS]> {
        self.have_goal.then_some(self.goal)
    }

    fn freeze<E: Encoder>(&self, e: &mut E) -> Result<(), EncodeError> {
        Encode::encode(&self.goal, e)?;
        Encode::encode(&self.have_goal, e)?;
        Encode::encode(&self.chunk, e)?;
        Encode::encode(&self.cursor, e)?;
        Encode::encode(&self.active, e)?;
        Encode::encode(&self.last_seq, e)?;
        Encode::encode(&self.have_seq, e)?;
        Encode::encode(&self.last_accept_ns, e)?;
        Encode::encode(&self.ring_seq, e)?;
        Encode::encode(&self.ring_ns, e)?;
        Encode::encode(&self.ring_len, e)?;
        Encode::encode(&self.ring_head, e)?;
        Encode::encode(
            &[
                self.accepted,
                self.rej_shape,
                self.rej_nonfinite,
                self.rej_order,
                self.rej_unknown_obs,
                self.rej_stale,
                self.held_cycles,
                self.bad_feedback,
            ],
            e,
        )?;
        Encode::encode(&self.last_target, e)?;
        Encode::encode(&self.have_target, e)?;
        Encode::encode(&self.switched, e)?;
        Encode::encode(&self.switches, e)?;
        Encode::encode(&self.switch_jump_max, e)?;
        Encode::encode(&self.switch_jump_sum, e)?;
        Encode::encode(&self.sched_delays, e)?;
        Encode::encode(&self.sched_len, e)?;
        Encode::encode(&self.sched_head, e)?;
        Encode::encode(&self.sched_pending, e)?;
        Encode::encode(&self.sched_pending_seq, e)?;
        Encode::encode(&self.sched_pending_since, e)?;
        Encode::encode(&self.prev_chunk, e)?;
        Encode::encode(&self.prev_cursor, e)?;
        Encode::encode(&self.blend_n, e)?;
        Encode::encode(&self.blend_active, e)
    }

    fn thaw<D: Decoder>(&mut self, d: &mut D) -> Result<(), DecodeError> {
        self.goal = Decode::decode(d)?;
        self.have_goal = Decode::decode(d)?;
        // CuArray only implements Decode<()>, and Freezable::thaw is generic over the
        // decoder context, so read its wire format (u32 len + f32s) by hand.
        let len: u32 = Decode::decode(d)?;
        let len = len as usize;
        if len > CHUNK_LEN {
            return Err(DecodeError::ArrayLengthMismatch {
                required: CHUNK_LEN,
                found: len,
            });
        }
        let mut tmp = [0f32; CHUNK_LEN];
        for slot in tmp[..len].iter_mut() {
            *slot = Decode::decode(d)?;
        }
        self.chunk.fill_from_iter(tmp[..len].iter().copied());
        self.cursor = Decode::decode(d)?;
        self.active = Decode::decode(d)?;
        self.last_seq = Decode::decode(d)?;
        self.have_seq = Decode::decode(d)?;
        self.last_accept_ns = Decode::decode(d)?;
        self.ring_seq = Decode::decode(d)?;
        self.ring_ns = Decode::decode(d)?;
        self.ring_len = Decode::decode(d)?;
        self.ring_head = Decode::decode(d)?;
        let c: [u32; 8] = Decode::decode(d)?;
        [
            self.accepted,
            self.rej_shape,
            self.rej_nonfinite,
            self.rej_order,
            self.rej_unknown_obs,
            self.rej_stale,
            self.held_cycles,
            self.bad_feedback,
        ] = c;
        self.last_target = Decode::decode(d)?;
        self.have_target = Decode::decode(d)?;
        self.switched = Decode::decode(d)?;
        self.switches = Decode::decode(d)?;
        self.switch_jump_max = Decode::decode(d)?;
        self.switch_jump_sum = Decode::decode(d)?;
        self.sched_delays = Decode::decode(d)?;
        self.sched_len = Decode::decode(d)?;
        self.sched_head = Decode::decode(d)?;
        self.sched_pending = Decode::decode(d)?;
        self.sched_pending_seq = Decode::decode(d)?;
        self.sched_pending_since = Decode::decode(d)?;
        // CuArray again has only `Decode<()>`: read its format (u32 length, f32s) by hand.
        let len: u32 = Decode::decode(d)?;
        let len = len as usize;
        if len > CHUNK_LEN {
            return Err(DecodeError::ArrayLengthMismatch {
                required: CHUNK_LEN,
                found: len,
            });
        }
        let mut tmp = [0f32; CHUNK_LEN];
        for slot in tmp[..len].iter_mut() {
            *slot = Decode::decode(d)?;
        }
        self.prev_chunk.fill_from_iter(tmp[..len].iter().copied());
        self.prev_cursor = Decode::decode(d)?;
        self.blend_n = Decode::decode(d)?;
        self.blend_active = Decode::decode(d)?;
        Ok(())
    }
}

#[derive(Reflect)]
#[reflect(from_reflect = false)]
pub struct ActionGovernor {
    core: GovernorCore,
}

impl ActionGovernor {
    pub fn from_params(params: GovernorParams) -> Self {
        Self {
            core: GovernorCore::new(params),
        }
    }

    pub fn core(&self) -> &GovernorCore {
        &self.core
    }
}

fn param(c: &ComponentConfig, key: &str) -> CuResult<f64> {
    let v = c
        .get::<f64>(key)?
        .ok_or_else(|| CuError::from(format!("governor: missing config key {key}")))?;
    if v.is_finite() {
        Ok(v)
    } else {
        Err(CuError::from(format!("governor: {key} is not finite")))
    }
}

impl SchedParams {
    /// Optional keys `sched_s_min` (0 or absent turns the scheduler off), `sched_margin`,
    /// `sched_d_init`, `sched_horizon`, `replan_threshold`, `sched_pending_timeout`,
    /// `blend_steps`, and the options sent to the policy with every request: `rtc_mode`
    /// (`"rtc"` or `"naive"`), `rtc_beta`, `rtc_denoise_steps`, `rtc_best_of`, `rtc_project`,
    /// `rtc_roll_obs` and `rtc_positional_noise`.
    pub fn from_config(c: &ComponentConfig) -> CuResult<Self> {
        let d = Self::default();
        let int = |key: &str, default: u32| -> CuResult<u32> {
            match c.get::<f64>(key)? {
                None => Ok(default),
                Some(v) if v.is_finite() && v >= 0.0 && v <= f64::from(u32::MAX) => Ok(v as u32),
                Some(_) => Err(CuError::from(format!(
                    "governor: {key} must be a non-negative number"
                ))),
            }
        };
        let mode = match c.get::<String>("rtc_mode")?.as_deref() {
            None | Some("naive") => MODE_NAIVE,
            Some("rtc") => MODE_RTC,
            Some(other) => {
                return Err(CuError::from(format!(
                    "governor: rtc_mode is \"rtc\" or \"naive\", got {other:?}"
                )));
            }
        };
        let flag = |key: &str, bit: u32| -> CuResult<u32> {
            Ok(if c.get::<bool>(key)?.unwrap_or(false) {
                bit
            } else {
                0
            })
        };
        let flags = flag("rtc_project", FLAG_PROJECT)?
            | flag("rtc_roll_obs", FLAG_ROLL_OBS)?
            | flag("rtc_positional_noise", FLAG_POSITIONAL_NOISE)?;
        let sched = Self {
            mode,
            denoise_steps: int("rtc_denoise_steps", d.denoise_steps)?,
            best_of: int("rtc_best_of", d.best_of)?,
            flags,
            beta: c.get::<f64>("rtc_beta")?.unwrap_or(f64::from(d.beta)) as f32,
            s_min: int("sched_s_min", d.s_min)?,
            margin: int("sched_margin", d.margin)?,
            d_init: int("sched_d_init", d.d_init)?,
            horizon: int("sched_horizon", d.horizon)?,
            replan_threshold: c.get::<f64>("replan_threshold")?.unwrap_or(0.0) as f32,
            pending_timeout: int("sched_pending_timeout", d.pending_timeout)?,
            blend_steps: int("blend_steps", d.blend_steps)?,
        };
        if !(1..=64).contains(&sched.denoise_steps) || !(1..=16).contains(&sched.best_of) {
            return Err(CuError::from(
                "governor: rtc_denoise_steps is 1 to 64 and rtc_best_of is 1 to 16",
            ));
        }
        if !sched.beta.is_finite() || sched.beta < 0.0 {
            return Err(CuError::from(
                "governor: rtc_beta must be a non-negative number",
            ));
        }
        if !sched.replan_threshold.is_finite() || sched.replan_threshold < 0.0 {
            return Err(CuError::from(
                "governor: replan_threshold must be a non-negative number",
            ));
        }
        if sched.s_min > 0 {
            if sched.horizon as usize > MAX_STEPS {
                return Err(CuError::from(format!(
                    "governor: sched_horizon exceeds the chunk capacity {MAX_STEPS}"
                )));
            }
            if sched.d_init + sched.s_min > sched.horizon {
                return Err(CuError::from(
                    "governor: sched_d_init + sched_s_min must not exceed sched_horizon",
                ));
            }
        }
        Ok(sched)
    }
}

impl GovernorParams {
    pub fn from_config(c: &ComponentConfig) -> CuResult<Self> {
        let mut min = [0.0; JOINTS];
        let mut max = [0.0; JOINTS];
        for j in 0..JOINTS {
            min[j] = param(c, &format!("min_{j}"))? as f32;
            max[j] = param(c, &format!("max_{j}"))? as f32;
            if min[j] > max[j] {
                return Err(CuError::from(format!("governor: min_{j} > max_{j}")));
            }
        }
        let ms = |k: &str| -> CuResult<u64> { Ok((param(c, k)?.max(0.0) * 1e6) as u64) };
        let (max_age_ns, cycle_ns) = (ms("max_age_ms")?, ms("cycle_ms")?);
        if cycle_ns > 0 && max_age_ns / cycle_ns >= OBS_RING as u64 {
            return Err(CuError::from(format!(
                "governor: max_age_ms covers {} cycles but only the last {OBS_RING} observations are remembered; \
                 a chunk older than that could never be matched to its observation",
                max_age_ns / cycle_ns
            )));
        }
        Ok(Self {
            min,
            max,
            max_step: param(c, "max_step")? as f32,
            max_lead: param(c, "max_lead")? as f32,
            max_age_ns,
            hold_deadline_ns: ms("hold_deadline_ms")?,
            cycle_ns,
            time_from_feedback: c.get::<bool>("time_from_feedback")?.unwrap_or(true),
            sched: SchedParams::from_config(c)?,
        })
    }
}

impl Freezable for ActionGovernor {
    fn freeze<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        self.core.freeze(encoder)
    }

    fn thaw<D: Decoder>(&mut self, decoder: &mut D) -> Result<(), DecodeError> {
        self.core.thaw(decoder)
    }
}

impl CuTask for ActionGovernor {
    type Resources<'r> = ();
    type Input<'m> = input_msg!('m, JointPositions, ObsStamp, ActionChunk);
    type Output<'m> = output_msg!(JointPositions, ExecState, InferenceRequest);

    fn new(config: Option<&ComponentConfig>, _resources: Self::Resources<'_>) -> CuResult<Self> {
        let config = config.ok_or_else(|| CuError::from("governor: config required"))?;
        Ok(Self {
            core: GovernorCore::new(GovernorParams::from_config(config)?),
        })
    }

    fn process<'i, 'o>(
        &mut self,
        ctx: &CuContext,
        input: &Self::Input<'i>,
        output: &mut Self::Output<'o>,
    ) -> CuResult<()> {
        let (feedback, stamp, chunk) = *input;
        let now = match feedback.tov {
            Tov::Time(t) if self.core.params.time_from_feedback => t.as_nanos(),
            _ => ctx.now().as_nanos(),
        };
        let (goal_out, exec_out, request_out) = output;
        let (goal, status) = self.core.step(
            now,
            stamp.payload().map(|s| s.seq),
            chunk.payload(),
            feedback.payload().map(|f| f.as_slice()),
        );
        match goal {
            Some(g) => {
                let mut arr = JointPositions::new();
                arr.fill_from_iter(g);
                goal_out.set_payload(arr);
                goal_out.tov = Tov::Time(CuTime(now));
            }
            None => goal_out.clear_payload(),
        }
        goal_out.metadata.set_status(status.as_str());
        exec_out.set_payload(self.core.exec_state());
        exec_out.tov = Tov::Time(CuTime(now));
        match self.core.request() {
            Some(r) => {
                request_out.set_payload(r.clone());
                request_out.tov = Tov::Time(CuTime(now));
            }
            None => request_out.clear_payload(),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: u64 = 1_000_000;

    fn core() -> GovernorCore {
        GovernorCore::new(GovernorParams {
            min: [-1.0; JOINTS],
            max: [1.0; JOINTS],
            max_step: 0.05,
            max_lead: 0.5,
            max_age_ns: 100 * MS,
            hold_deadline_ns: 200 * MS,
            cycle_ns: 0,
            time_from_feedback: true,
            sched: SchedParams::default(),
        })
    }

    fn chunk(seq: u64, rows: &[[f32; JOINTS]]) -> ActionChunk {
        let mut c = ActionChunk {
            obs_seq: seq,
            values: CuArray::new(),
        };
        c.values.fill_from_iter(rows.iter().flatten().copied());
        c
    }

    const Z: [f32; JOINTS] = [0.0; JOINTS];

    #[test]
    fn no_goal_until_a_finite_measurement_and_goal_starts_at_measurement() {
        let mut g = core();
        assert_eq!(g.step(0, None, None, None), (None, Status::NoGoal));
        let bad = [f32::NAN; JOINTS];
        assert_eq!(g.step(MS, None, None, Some(&bad)).0, None);
        let m = [0.2; JOINTS];
        let (goal, st) = g.step(2 * MS, None, None, Some(&m));
        assert_eq!((goal, st), (Some(m), Status::Hold));
    }

    #[test]
    fn nonfinite_and_malformed_chunks_are_rejected_whole() {
        let mut g = core();
        g.step(0, Some(1), None, Some(&Z));
        let mut nan = [[0.1; JOINTS]; 3];
        nan[2][4] = f32::NAN;
        g.step(MS, None, Some(&chunk(1, &nan)), Some(&Z));
        let mut ragged = chunk(1, &[Z]);
        ragged.values.fill_from_iter([0.1f32; 7]);
        g.step(2 * MS, None, Some(&ragged), Some(&Z));
        g.step(3 * MS, None, Some(&chunk(1, &[])), Some(&Z));
        assert_eq!((g.accepted, g.rej_nonfinite, g.rej_shape), (0, 1, 2));
        assert_eq!(g.goal(), Some(Z));
    }

    #[test]
    fn plays_one_step_per_cycle_with_joint_clamp_and_step_limit() {
        let mut g = core();
        g.step(0, Some(7), None, Some(&Z));
        let rows = [[0.03; JOINTS], [9.0; JOINTS], [9.0; JOINTS]];
        let (a, s) = g.step(MS, None, Some(&chunk(7, &rows)), Some(&Z));
        assert_eq!((a.unwrap()[0], s), (0.03, Status::Play));
        // 9.0 -> clamp 1.0 -> lead window of measured (0.0 +- 0.5) -> step limit 0.05 from 0.03
        let (b, _) = g.step(2 * MS, None, None, Some(&Z));
        assert!((b.unwrap()[0] - 0.08).abs() < 1e-6);
        // measurement follows the goal: lead stays satisfied, goal keeps ramping by max_step
        let m = [0.08; JOINTS];
        let (c, _) = g.step(3 * MS, None, None, Some(&m));
        assert!((c.unwrap()[0] - 0.13).abs() < 1e-6);
        assert_eq!(
            g.step(4 * MS, None, None, Some(&m)).1,
            Status::HoldExhausted
        );
    }

    #[test]
    fn stale_unknown_and_out_of_order_chunks_are_dropped() {
        let mut g = core();
        g.step(0, Some(1), None, Some(&Z));
        g.step(150 * MS, Some(2), Some(&chunk(1, &[Z])), Some(&Z)); // obs 1 is 150 ms old
        g.step(151 * MS, None, Some(&chunk(99, &[Z])), Some(&Z)); // never seen
        g.step(152 * MS, None, Some(&chunk(2, &[Z])), Some(&Z)); // fresh: accepted
        g.step(153 * MS, None, Some(&chunk(1, &[Z])), Some(&Z)); // older than accepted
        g.step(154 * MS, None, Some(&chunk(2, &[Z])), Some(&Z)); // duplicate: ignored silently
        assert_eq!(
            (g.rej_stale, g.rej_unknown_obs, g.accepted, g.rej_order),
            (1, 1, 1, 1)
        );
    }

    #[test]
    fn hold_deadline_and_replacement() {
        let mut g = core();
        g.step(0, Some(1), None, Some(&Z));
        let ten = [[0.04; JOINTS]; 10];
        g.step(MS, None, Some(&chunk(1, &ten)), Some(&Z));
        g.step(2 * MS, Some(2), Some(&chunk(1, &ten)), Some(&Z)); // duplicate of the active one
        let m = [0.04; JOINTS];
        // a newer chunk replaces the remaining steps
        let (r, _) = g.step(
            3 * MS,
            None,
            Some(&chunk(2, &[[-0.04; JOINTS]; 10])),
            Some(&m),
        );
        assert!(r.unwrap()[0] < 0.04);
        // no fresh chunk within the deadline: stop playing, keep commanding the last goal
        let (h, st) = g.step(300 * MS, None, None, Some(&m));
        assert_eq!(st, Status::HoldExpired);
        let (h2, st2) = g.step(301 * MS, None, None, Some(&m));
        assert_eq!((h, st2), (h2, Status::Hold));
    }

    #[test]
    fn an_age_limit_the_observation_history_cannot_cover_is_a_config_error() {
        let mut c = ComponentConfig::default();
        for j in 0..JOINTS {
            c.set(&format!("min_{j}"), -1.0f64);
            c.set(&format!("max_{j}"), 1.0f64);
        }
        for (k, v) in [
            ("max_step", 0.05),
            ("max_lead", 0.5),
            ("hold_deadline_ms", 200.0),
            ("cycle_ms", 10.0),
        ] {
            c.set(k, v);
        }
        c.set("max_age_ms", 100.0f64);
        assert!(GovernorParams::from_config(&c).is_ok());
        c.set("max_age_ms", 10.0 * OBS_RING as f64);
        let e = GovernorParams::from_config(&c).unwrap_err().to_string();
        assert!(e.contains("observations are remembered"), "{e}");
    }

    #[test]
    fn a_cycle_without_a_measurement_holds_and_does_not_advance_the_chunk() {
        let mut g = core();
        g.step(0, Some(1), None, Some(&Z));
        let rows = [[0.01; JOINTS], [0.02; JOINTS], [0.03; JOINTS]];
        let (a, _) = g.step(MS, None, Some(&chunk(1, &rows)), Some(&Z));
        assert_eq!(a.unwrap()[0], 0.01);
        for t in 2..5 {
            let (held, st) = g.step(t * MS, None, None, None);
            assert_eq!((held.unwrap()[0], st), (0.01, Status::HoldNoFeedback));
        }
        let bad = [f32::NAN; JOINTS];
        assert_eq!(
            g.step(5 * MS, None, None, Some(&bad)).1,
            Status::HoldNoFeedback
        );
        // The measurement returns: the chunk resumes at step 2, not step 5.
        let (b, st) = g.step(6 * MS, None, None, Some(&Z));
        assert_eq!((b.unwrap()[0], st), (0.02, Status::Play));
    }

    /// A policy answering observation `k` with the ramp `x(t) = RATE * t` from `t = k` on. A
    /// chunk that arrives `delay` cycles late only continues the trajectory if the governor
    /// starts it `delay` steps in.
    const RATE: f32 = 0.01;

    fn ramp_chunk(obs_k: u64, len: usize) -> ActionChunk {
        let rows: Vec<[f32; JOINTS]> = (0..len)
            .map(|i| [RATE * (obs_k as f32 + i as f32); JOINTS])
            .collect();
        chunk(obs_k, &rows)
    }

    /// Plays ramp chunks of 20 steps answered every 10 cycles, each `delays[n % len]` cycles late.
    /// Returns the governor and the largest distance between a played target and the ideal
    /// trajectory `x(t)`.
    fn run_ramp(cycle_ns: u64, delays: &[u64]) -> (GovernorCore, f32) {
        let mut p = core().params;
        p.cycle_ns = cycle_ns;
        p.max_lead = 10.0;
        p.max_step = 1.0;
        let mut g = GovernorCore::new(p);
        let mut answers: Vec<(u64, ActionChunk)> = Vec::new();
        let mut worst = 0.0f32;
        for t in 0..80u64 {
            let m = [RATE * t as f32; JOINTS];
            if t % 10 == 0 {
                let delay = delays[(t / 10) as usize % delays.len()];
                answers.push((t + delay, ramp_chunk(t, 20)));
            }
            let due = answers.iter().position(|(at, _)| *at == t);
            let c = due.map(|i| answers.remove(i).1);
            let (goal, status) = g.step(t * MS, Some(t), c.as_ref(), Some(&m));
            if status == Status::Play {
                worst = worst.max((goal.unwrap()[0] - m[0]).abs());
            }
        }
        (g, worst)
    }

    #[test]
    fn a_late_chunk_continues_the_trajectory_when_the_elapsed_steps_are_skipped() {
        let delays = [1, 4, 2, 6];
        let (aware, aware_err) = run_ramp(MS, &delays);
        let (naive, naive_err) = run_ramp(0, &delays);
        assert!(aware.switches >= 5 && naive.switches >= 5);
        assert!(
            aware_err < RATE * 0.01 && aware.switch_jump_max <= RATE * 1.01,
            "compensated: error {aware_err}, jump {}",
            aware.switch_jump_max
        );
        // Starting every chunk at step 0 plays the trajectory `delay` steps late, and the
        // lateness changes whenever the delay does.
        assert!(naive_err >= RATE * 5.99, "naive tracking error {naive_err}");
        assert!(
            naive.switch_jump_max >= RATE * 3.99,
            "naive jump {}",
            naive.switch_jump_max
        );
    }

    #[test]
    fn compensation_keeps_up_with_any_delay_inside_the_age_limit() {
        for delay in [0u64, 1, 5, 9] {
            let (g, err) = run_ramp(MS, &[delay]);
            assert!(
                err < RATE * 0.01 && g.switch_jump_max <= RATE * 1.01,
                "delay {delay}"
            );
        }
    }

    #[test]
    fn the_exec_state_names_the_active_chunk_and_the_step_played() {
        let mut g = core();
        let rows = [[0.01; JOINTS], [0.02; JOINTS], [0.03; JOINTS]];
        g.step(0, Some(4), None, Some(&Z));
        let e = g.exec_state();
        assert!(e.has(ExecState::HAS_STAMP) && !e.has(ExecState::CHUNK_ACTIVE));
        assert_eq!(e.stamp_seq, 4);
        // accepted and played in the same cycle: step 0 of the chunk answering observation 4
        g.step(MS, None, Some(&chunk(4, &rows)), Some(&Z));
        let e = g.exec_state();
        assert_eq!((e.chunk_seq, e.next_index), (4, 0));
        assert!(e.has(ExecState::CHUNK_ACTIVE) && e.has(ExecState::PLAYED));
        assert!(!e.has(ExecState::HAS_STAMP));
        g.step(2 * MS, None, None, Some(&Z));
        assert_eq!(g.exec_state().next_index, 1);
        // a cycle without a measurement holds: the index stays where the next step will play
        g.step(3 * MS, None, None, None);
        let e = g.exec_state();
        assert_eq!(e.next_index, 2);
        assert!(!e.has(ExecState::PLAYED));
        g.step(4 * MS, None, None, Some(&Z));
        assert_eq!(g.exec_state().next_index, 2);
        // exhausted: still active, no step played
        g.step(5 * MS, None, None, Some(&Z));
        let e = g.exec_state();
        assert!(e.has(ExecState::CHUNK_ACTIVE) && !e.has(ExecState::PLAYED));
        // expired: no longer active
        g.step(400 * MS, None, None, Some(&Z));
        assert!(!g.exec_state().has(ExecState::CHUNK_ACTIVE));
    }

    #[test]
    fn a_late_chunk_reports_the_steps_that_elapsed_as_its_first_index() {
        let mut p = core().params;
        p.cycle_ns = MS;
        let mut g = GovernorCore::new(p);
        g.step(0, Some(1), None, Some(&Z));
        g.step(MS, None, None, Some(&Z));
        g.step(2 * MS, None, None, Some(&Z));
        g.step(3 * MS, None, Some(&ramp_chunk(1, 10)), Some(&Z));
        // 3 cycles after observation 1, step 3 of its chunk plays: the observed delay
        assert_eq!(g.exec_state().next_index, 3);
    }

    fn sched_core(s_min: u32, threshold: f32) -> GovernorCore {
        let mut p = core().params;
        p.cycle_ns = MS;
        p.max_age_ns = 60 * MS;
        p.hold_deadline_ns = 10_000 * MS;
        p.max_lead = 10.0;
        p.max_step = 1.0;
        p.sched = SchedParams {
            s_min,
            margin: 2,
            d_init: 3,
            horizon: 50,
            replan_threshold: threshold,
            pending_timeout: 25,
            blend_steps: 0,
            ..SchedParams::default()
        };
        GovernorCore::new(p)
    }

    fn long_chunk(seq: u64) -> ActionChunk {
        let rows: Vec<[f32; JOINTS]> = (0..50).map(|i| [0.001 * i as f32; JOINTS]).collect();
        chunk(seq, &rows)
    }

    #[test]
    fn the_scheduler_asks_for_a_free_sample_when_nothing_is_executing() {
        let mut g = sched_core(10, 0.0);
        g.step(0, Some(0), None, Some(&Z));
        let r = g.request().expect("no request without a chunk");
        assert_eq!((r.obs_seq, r.delay, r.executed), (0, 3, 0));
        assert_eq!(r.reason, InferenceRequest::REASON_FIRST);
        assert!(r.previous.as_slice().is_empty());
        assert_eq!(r.state.as_slice().len(), JOINTS);
        // asking again while that request is in flight would be a duplicate
        g.step(MS, Some(1), None, Some(&Z));
        assert!(g.request().is_none());
    }

    #[test]
    fn the_scheduler_waits_for_the_horizon_and_sends_the_unplayed_remainder() {
        let mut g = sched_core(10, 0.0);
        g.step(0, Some(0), None, Some(&Z));
        g.step(MS, Some(1), None, Some(&Z));
        // The chunk for observation 0 arrives 4 cycles late: it starts at step 4.
        for t in 2..4u64 {
            g.step(t * MS, Some(t), None, Some(&Z));
        }
        g.step(4 * MS, Some(4), Some(&long_chunk(0)), Some(&Z));
        assert_eq!(g.exec_state().accept_skip, 4);
        assert!(g.exec_state().has(ExecState::ACCEPTED));
        assert_eq!(g.delay_estimate(), 4);
        // horizon = max(s_min 10, d 4 + margin 2) = 10: steps 4..9 play without a request
        for t in 5..10u64 {
            g.step(t * MS, Some(t), None, Some(&Z));
            assert!(g.request().is_none(), "cycle {t}");
        }
        g.step(10 * MS, Some(10), None, Some(&Z));
        assert_eq!(g.exec_state().next_index, 10);
        let r = g.request().expect("the horizon was reached");
        assert_eq!((r.obs_seq, r.delay, r.executed), (10, 4, 10));
        assert_eq!(r.reason, InferenceRequest::REASON_SCHEDULED);
        // the remainder starts at the step played this cycle
        assert_eq!(r.previous.as_slice().len(), 40 * JOINTS);
        assert_eq!(r.previous.as_slice()[0], 0.001 * 10.0);
    }

    #[test]
    fn a_pending_request_blocks_the_next_one_until_answered_or_timed_out() {
        let mut g = sched_core(5, 0.0);
        g.step(0, Some(0), None, Some(&Z));
        g.step(MS, Some(1), Some(&long_chunk(0)), Some(&Z));
        // The request goes out at step 5 (cycle 5) and is never answered: nothing follows it
        // until the 25-cycle timeout has passed.
        let mut asked = Vec::new();
        for t in 2..31u64 {
            g.step(t * MS, Some(t), None, Some(&Z));
            if g.request().is_some() {
                asked.push(t);
            }
        }
        assert_eq!(asked, vec![5], "one request at the horizon, then silence");
        let mut again = Vec::new();
        for t in 31..45u64 {
            g.step(t * MS, Some(t), None, Some(&Z));
            if g.request().is_some() {
                again.push(t);
            }
        }
        assert_eq!(
            again.first(),
            Some(&31),
            "after 25 unanswered cycles it asks again"
        );
    }

    #[test]
    fn the_horizon_grows_with_the_measured_delay() {
        let mut g = sched_core(5, 0.0);
        g.step(0, Some(0), None, Some(&Z));
        for t in 1..9u64 {
            g.step(t * MS, Some(t), None, Some(&Z));
        }
        // observation 0 is answered 8 cycles late
        g.step(9 * MS, Some(9), Some(&long_chunk(0)), Some(&Z));
        assert_eq!(g.exec_state().accept_skip, 9);
        assert_eq!(g.delay_estimate(), 9);
        // horizon = max(5, 9 + 2) = 11, so no request at step 10 although s_min is 5
        g.step(10 * MS, Some(10), None, Some(&Z));
        assert!(g.request().is_none());
        g.step(11 * MS, Some(11), None, Some(&Z));
        assert_eq!(g.exec_state().next_index, 11);
        assert_eq!(g.request().unwrap().executed, 11);
    }

    #[test]
    fn the_delay_estimate_is_the_largest_of_the_last_ten_measured() {
        let mut g = sched_core(5, 0.0);
        assert_eq!(
            g.delay_estimate(),
            3,
            "the initial estimate until one is measured"
        );
        // Chunks answering observation `k` accepted `skip` cycles later: delays 8, then 2, 2, ...
        let mut t = 0u64;
        let mut answer = |g: &mut GovernorCore, skip: u64| {
            let k = t;
            g.step(t * MS, Some(k), None, Some(&Z));
            t += skip;
            for _ in 1..skip {
                g.step(t * MS - (skip - 1) * MS, None, None, Some(&Z));
            }
            g.step(t * MS, None, Some(&long_chunk(k)), Some(&Z));
            assert_eq!(g.exec_state().accept_skip as u64, skip);
            t += 1;
        };
        answer(&mut g, 8);
        assert_eq!(g.delay_estimate(), 8);
        answer(&mut g, 2);
        assert_eq!(g.delay_estimate(), 8, "the maximum, not the latest");
        for _ in 0..8 {
            answer(&mut g, 2);
        }
        assert_eq!(g.delay_estimate(), 8, "the 8 is still among the last ten");
        answer(&mut g, 2);
        assert_eq!(g.delay_estimate(), 2, "the 8 has left the window of ten");
    }

    #[test]
    fn a_large_tracking_error_replans_at_once() {
        let mut g = sched_core(25, 0.2);
        g.step(0, Some(0), None, Some(&Z));
        g.step(MS, Some(1), Some(&long_chunk(0)), Some(&Z));
        let mut on_track = [0.0f32; JOINTS];
        for t in 2..8u64 {
            on_track = [0.001 * (t - 1) as f32; JOINTS];
            g.step(t * MS, Some(t), None, Some(&on_track));
            assert!(g.request().is_none(), "no error, no request before s_min");
        }
        assert!(g.exec_state().tracking_err < 0.01);
        // the arm is pushed: the measurement is far from where the chunk put it
        let pushed = [on_track[0] + 0.5; JOINTS];
        g.step(8 * MS, Some(8), None, Some(&pushed));
        assert!((g.exec_state().tracking_err - 0.5).abs() < 0.01);
        let r = g.request().expect("replan");
        assert_eq!(r.reason, InferenceRequest::REASON_EVENT);
        assert!(r.executed < 25, "well before the minimum horizon");
        assert!(
            !r.previous.as_slice().is_empty(),
            "still stays consistent with the plan"
        );
    }

    #[test]
    fn rejections_are_reported_by_reason() {
        let mut g = sched_core(10, 0.0);
        g.step(0, Some(0), None, Some(&Z));
        let mut nan = long_chunk(0);
        nan.values
            .fill_from_iter(core::iter::repeat_n(f32::NAN, 12));
        g.step(MS, Some(1), Some(&nan), Some(&Z));
        assert_eq!(g.exec_state().reject, ExecState::REJECT_NONFINITE);
        g.step(2 * MS, Some(2), Some(&long_chunk(99)), Some(&Z));
        assert_eq!(g.exec_state().reject, ExecState::REJECT_UNKNOWN_OBS);
        g.step(3 * MS, Some(3), Some(&long_chunk(0)), Some(&Z));
        assert_eq!(g.exec_state().reject, 0);
        assert!(g.exec_state().has(ExecState::ACCEPTED));
        g.step(4 * MS, Some(4), None, Some(&Z));
        assert_eq!(
            (
                g.exec_state().reject,
                g.exec_state().flags & ExecState::ACCEPTED
            ),
            (0, 0)
        );
    }

    fn constant_chunk(seq: u64, value: f32, steps: usize) -> ActionChunk {
        chunk(seq, &vec![[value; JOINTS]; steps])
    }

    /// Plays an old chunk of 0.0, then accepts one of 1.0 mid-way, and returns the targets the
    /// governor played from the cycle the new chunk is accepted.
    fn handover(blend_steps: u32, old_steps: usize) -> Vec<f32> {
        let mut g = sched_core(0, 0.0);
        g.params.sched.blend_steps = blend_steps;
        g.step(0, Some(0), None, Some(&Z));
        g.step(
            MS,
            Some(1),
            Some(&constant_chunk(0, 0.0, old_steps)),
            Some(&Z),
        );
        for t in 2..6u64 {
            g.step(t * MS, Some(t), None, Some(&Z));
        }
        g.step(6 * MS, Some(6), Some(&constant_chunk(6, 1.0, 40)), Some(&Z));
        let mut played = vec![g.last_target[0]];
        for t in 7..14u64 {
            g.step(t * MS, Some(t), None, Some(&Z));
            played.push(g.last_target[0]);
        }
        played
    }

    #[test]
    fn a_blend_moves_the_target_from_the_old_chunk_to_the_new_one_in_equal_steps() {
        let plain = handover(0, 40);
        assert_eq!(plain[0], 1.0, "without a blend the target jumps at once");
        let blended = handover(4, 40);
        let want = [0.2, 0.4, 0.6, 0.8, 1.0, 1.0];
        for (got, want) in blended.iter().zip(want) {
            assert!((got - want).abs() < 1e-6, "{blended:?}");
        }
        assert!(
            blended.windows(2).all(|w| w[1] - w[0] <= 0.2 + 1e-6),
            "{blended:?}"
        );
    }

    #[test]
    fn a_blend_has_nothing_to_fade_from_when_the_old_chunk_is_used_up() {
        // The old chunk has only 4 steps and ended before the new one arrives.
        let played = handover(4, 4);
        assert_eq!(played[0], 1.0, "{played:?}");
    }

    #[test]
    fn the_blend_state_survives_a_keyframe() {
        let mut g = sched_core(0, 0.0);
        g.params.sched.blend_steps = 6;
        g.step(0, Some(0), None, Some(&Z));
        g.step(MS, Some(1), Some(&constant_chunk(0, 0.0, 40)), Some(&Z));
        g.step(4 * MS, Some(4), Some(&constant_chunk(4, 1.0, 40)), Some(&Z));
        g.step(5 * MS, Some(5), None, Some(&Z)); // mid-blend
        let cfg = cu29::bincode::config::standard();
        let mut buf = [0u8; 8192];
        let n = cu29::bincode::encode_into_slice(
            cu29::prelude::BincodeAdapter(&ActionGovernor { core: g.clone() }),
            &mut buf,
            cfg,
        )
        .unwrap();
        let mut h = ActionGovernor::from_params(g.params.clone());
        let mut dec = cu29::bincode::de::DecoderImpl::new(
            cu29::bincode::de::read::SliceReader::new(&buf[..n]),
            cfg,
            (),
        );
        h.thaw(&mut dec).unwrap();
        let mut a = g;
        let mut b = h.core;
        for t in 6..16u64 {
            assert_eq!(
                a.step(t * MS, Some(t), None, Some(&Z)),
                b.step(t * MS, Some(t), None, Some(&Z)),
                "cycle {t}"
            );
        }
    }

    #[test]
    fn the_request_carries_the_options_the_governor_is_configured_with() {
        let mut g = sched_core(10, 0.0);
        g.params.sched.mode = MODE_RTC;
        g.params.sched.denoise_steps = 8;
        g.params.sched.best_of = 3;
        g.params.sched.flags = FLAG_PROJECT | FLAG_POSITIONAL_NOISE;
        g.params.sched.beta = 2.5;
        g.step(0, Some(0), None, Some(&Z));
        let r = g.request().expect("the first cycle asks");
        assert_eq!(
            (
                r.horizon,
                r.mode,
                r.denoise_steps,
                r.best_of,
                r.flags,
                r.beta
            ),
            (
                50,
                MODE_RTC,
                8,
                3,
                FLAG_PROJECT | FLAG_POSITIONAL_NOISE,
                2.5
            )
        );
    }

    fn config(json: &str) -> ComponentConfig {
        serde_json::from_str(json).expect("a configuration of scalars")
    }

    #[test]
    fn the_options_are_read_from_the_configuration() {
        let p = SchedParams::from_config(&config(r#"{"sched_s_min": 25}"#)).unwrap();
        assert_eq!(
            (p.mode, p.denoise_steps, p.best_of, p.flags, p.beta),
            (MODE_NAIVE, 5, 1, 0, 5.0),
            "the defaults"
        );
        let p = SchedParams::from_config(&config(
            r#"{"sched_s_min": 25, "rtc_mode": "rtc", "rtc_beta": 2.0, "rtc_denoise_steps": 10,
                "rtc_best_of": 4, "rtc_project": true, "rtc_roll_obs": true,
                "rtc_positional_noise": false}"#,
        ))
        .unwrap();
        assert_eq!(
            (p.mode, p.denoise_steps, p.best_of, p.flags, p.beta),
            (MODE_RTC, 10, 4, FLAG_PROJECT | FLAG_ROLL_OBS, 2.0)
        );
    }

    #[test]
    fn an_option_outside_its_documented_values_is_a_configuration_error() {
        for bad in [
            r#"{"rtc_mode": "guided"}"#,
            r#"{"rtc_beta": -1.0}"#,
            r#"{"rtc_denoise_steps": 0}"#,
            r#"{"rtc_best_of": 17}"#,
        ] {
            let e = SchedParams::from_config(&config(bad))
                .unwrap_err()
                .to_string();
            assert!(e.contains("rtc_"), "{bad}: {e}");
        }
    }

    #[test]
    fn the_scheduler_is_off_unless_configured() {
        let mut g = core();
        for t in 0..40u64 {
            g.step(t * MS, Some(t), None, Some(&Z));
            assert!(g.request().is_none());
        }
    }

    #[test]
    fn the_scheduler_state_survives_a_keyframe() {
        let mut g = sched_core(5, 0.0);
        g.step(0, Some(0), None, Some(&Z));
        g.step(MS, Some(1), Some(&long_chunk(0)), Some(&Z));
        for t in 2..5u64 {
            g.step(t * MS, Some(t), None, Some(&Z));
        }
        let cfg = cu29::bincode::config::standard();
        let mut buf = [0u8; 4096];
        let n = cu29::bincode::encode_into_slice(
            cu29::prelude::BincodeAdapter(&ActionGovernor { core: g.clone() }),
            &mut buf,
            cfg,
        )
        .unwrap();
        let mut h = ActionGovernor::from_params(g.params.clone());
        let mut dec = cu29::bincode::de::DecoderImpl::new(
            cu29::bincode::de::read::SliceReader::new(&buf[..n]),
            cfg,
            (),
        );
        h.thaw(&mut dec).unwrap();
        let mut a = g;
        let mut b = h.core;
        for t in 5..40u64 {
            a.step(t * MS, Some(t), None, Some(&Z));
            b.step(t * MS, Some(t), None, Some(&Z));
            assert_eq!(a.exec_state(), b.exec_state(), "cycle {t}");
            assert_eq!(
                a.request().map(|r| (r.obs_seq, r.reason, r.executed)),
                b.request().map(|r| (r.obs_seq, r.reason, r.executed)),
                "cycle {t}"
            );
        }
    }

    #[test]
    fn frozen_state_roundtrips() {
        let mut g = core();
        g.step(0, Some(1), None, Some(&Z));
        g.step(MS, None, Some(&chunk(1, &[[0.03; JOINTS]; 5])), Some(&Z));
        let cfg = cu29::bincode::config::standard();
        let mut buf = [0u8; 4096];
        let n = cu29::bincode::encode_into_slice(
            cu29::prelude::BincodeAdapter(&ActionGovernor { core: g.clone() }),
            &mut buf,
            cfg,
        )
        .unwrap();
        let mut h = ActionGovernor::from_params(g.params.clone());
        let mut dec = cu29::bincode::de::DecoderImpl::new(
            cu29::bincode::de::read::SliceReader::new(&buf[..n]),
            cfg,
            (),
        );
        h.thaw(&mut dec).unwrap();
        let mut a = g;
        let mut b = h.core;
        for t in 2..8u64 {
            assert_eq!(
                a.step(t * MS, None, None, Some(&Z)),
                b.step(t * MS, None, None, Some(&Z))
            );
        }
    }
}
