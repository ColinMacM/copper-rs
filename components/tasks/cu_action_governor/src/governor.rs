use crate::payloads::{ActionChunk, CHUNK_LEN, ExecState, JOINTS, ObsStamp};
use cu29::bincode::de::Decoder;
use cu29::bincode::enc::Encoder;
use cu29::bincode::error::{DecodeError, EncodeError};
use cu29::bincode::{Decode, Encode};
use cu29::prelude::*;

/// Observations remembered so a chunk can be aged by the observation it names. At one
/// observation per cycle this covers `OBS_RING` cycles; `max_age_ms` must fit inside it.
const OBS_RING: usize = 64;

pub type JointPositions = CuArray<f32, 8>; // same type cu_feetech publishes / consumes

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

    fn offer(&mut self, now: u64, chunk: &ActionChunk) {
        let v = chunk.values.as_slice();
        if self.have_seq && chunk.obs_seq == self.last_seq {
            return; // bridge re-emitting the same chunk: not a new one, not an error
        }
        if v.is_empty() || !v.len().is_multiple_of(JOINTS) {
            self.rej_shape += 1;
            return;
        }
        if v.iter().any(|x| !x.is_finite()) {
            self.rej_nonfinite += 1;
            return;
        }
        if self.have_seq && chunk.obs_seq < self.last_seq {
            self.rej_order += 1;
            return;
        }
        let Some(obs_ns) = self.obs_time(chunk.obs_seq) else {
            self.rej_unknown_obs += 1;
            return;
        };
        let age = now.saturating_sub(obs_ns);
        if age > self.params.max_age_ns {
            self.rej_stale += 1;
            return;
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
    }

    /// One control cycle. Returns the goal to command (None until a measurement was seen).
    pub fn step(
        &mut self,
        now: u64,
        stamp: Option<u64>,
        chunk: Option<&ActionChunk>,
        feedback: Option<&[f32]>,
    ) -> (Option<[f32; JOINTS]>, Status) {
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
            self.offer(now, c);
            self.note_exec(stamp, false);
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
        (Some(self.goal), status)
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
        Encode::encode(&self.switch_jump_sum, e)
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
    type Input<'m> = input_msg!('m, ActionChunk, JointPositions, ObsStamp);
    type Output<'m> = output_msg!(JointPositions, ExecState);

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
        let (chunk, feedback, stamp) = *input;
        let now = match feedback.tov {
            Tov::Time(t) if self.core.params.time_from_feedback => t.as_nanos(),
            _ => ctx.now().as_nanos(),
        };
        let (goal_out, exec_out) = output;
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
