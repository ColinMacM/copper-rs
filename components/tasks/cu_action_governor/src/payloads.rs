use bincode::{Decode, Encode};
use cu29::prelude::*;
use serde::{Deserialize, Serialize};

pub const JOINTS: usize = 6;
pub const MAX_STEPS: usize = 50;
pub const CHUNK_LEN: usize = JOINTS * MAX_STEPS;

/// One policy output: `values.len() / JOINTS` steps, row-major. `CuArray` is a
/// fixed-capacity ArrayVec: `Default` works past 32 elements (plain arrays do not) and
/// only the used prefix is encoded into the log.
#[derive(Default, Debug, Clone, Serialize, Deserialize, Reflect)]
#[reflect(from_reflect = false)]
pub struct ActionChunk {
    /// Sequence number of the observation the model used (echoed by the policy).
    pub obs_seq: u64,
    pub values: CuArray<f32, CHUNK_LEN>,
}

/// Emitted with every observation handed to the policy bridge. The governor maps
/// `seq` to its own clock, so the age of a chunk never depends on the policy's clock.
#[derive(
    Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Encode, Decode, Reflect,
)]
pub struct ObsStamp {
    pub seq: u64,
}

/// What the governor is executing, emitted every cycle after the observation stamp of that
/// cycle. A policy that overlaps inference with execution (real-time chunking) needs the
/// actions of its previous chunk that have not been played yet; this says which chunk is
/// active and which of its steps is played next.
///
/// Wire format: `stamp_seq: u64`, `chunk_seq: u64`, `next_index: u32`, `flags: u32`,
/// `accept_skip: u32`, `reject: u32`, `tracking_err: f32`.
#[derive(
    Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Encode, Decode, Reflect,
)]
pub struct ExecState {
    /// `seq` of the observation stamped this cycle; valid when [`ExecState::HAS_STAMP`] is set.
    pub stamp_seq: u64,
    /// `obs_seq` of the active chunk; valid when [`ExecState::CHUNK_ACTIVE`] is set.
    pub chunk_seq: u64,
    /// Index, within the active chunk, of the step played this cycle or, if the cycle held,
    /// of the step that plays next. Step `i` of the chunk answering observation `k` belongs to
    /// the cycle of `k` plus `i`, so this is also the chunk's offset from that observation.
    pub next_index: u32,
    pub flags: u32,
    /// With [`ExecState::ACCEPTED`]: the step the accepted chunk starts at, which is the delay
    /// that chunk really had, in cycles.
    pub accept_skip: u32,
    /// Why a chunk offered this cycle was refused (`REJECT_*`), or 0.
    pub reject: u32,
    /// Largest distance, over the arm joints, between the measurement and the raw target played
    /// in the previous cycle. Zero until a target has been played.
    pub tracking_err: f32,
}

impl ExecState {
    /// A chunk was accepted this cycle.
    pub const ACCEPTED: u32 = 8;
    pub const REJECT_SHAPE: u32 = 1;
    pub const REJECT_NONFINITE: u32 = 2;
    pub const REJECT_ORDER: u32 = 3;
    pub const REJECT_UNKNOWN_OBS: u32 = 4;
    pub const REJECT_STALE: u32 = 5;

    pub const HAS_STAMP: u32 = 1;
    /// A chunk was accepted and has not expired; it may still be exhausted.
    pub const CHUNK_ACTIVE: u32 = 2;
    /// A step of the chunk was played this cycle.
    pub const PLAYED: u32 = 4;

    #[must_use]
    pub fn has(&self, flag: u32) -> bool {
        self.flags & flag != 0
    }
}

/// A request to the policy, emitted by the governor's chunk scheduler on the cycles where the
/// next inference should start (real-time chunking, Algorithm 1 of arXiv:2506.07339). It carries
/// everything the policy needs, so the policy process keeps no state: the observation, the
/// delay estimate `delay` (`d`), the number of steps `executed` of the active chunk already
/// played (`s`), and `previous`, the unplayed remainder of that chunk, whose first step belongs
/// to the control cycle of `obs_seq`. `previous` is empty when nothing is executing yet.
///
/// Wire format: `obs_seq: u64`, `delay: u32`, `executed: u32`, `reason: u32`, `state_len: u32`,
/// `state_len` x `f32`, `previous_len: u32`, `previous_len` x `f32`.
#[derive(Default, Debug, Clone, Serialize, Deserialize, Reflect)]
#[reflect(from_reflect = false)]
pub struct InferenceRequest {
    pub obs_seq: u64,
    pub delay: u32,
    pub executed: u32,
    /// `REASON_*` bits.
    pub reason: u32,
    pub state: CuArray<f32, OBS_JOINTS>,
    pub previous: CuArray<f32, CHUNK_LEN>,
}

impl InferenceRequest {
    /// No chunk was executing: sample freely.
    pub const REASON_FIRST: u32 = 1;
    /// The execution horizon was reached.
    pub const REASON_SCHEDULED: u32 = 2;
    /// The measurement strayed from the chunk, so the plan is replaced early.
    pub const REASON_EVENT: u32 = 4;
}

impl Encode for InferenceRequest {
    fn encode<E: cu29::bincode::enc::Encoder>(
        &self,
        e: &mut E,
    ) -> Result<(), cu29::bincode::error::EncodeError> {
        Encode::encode(&self.obs_seq, e)?;
        Encode::encode(&self.delay, e)?;
        Encode::encode(&self.executed, e)?;
        Encode::encode(&self.reason, e)?;
        encode_f32s(self.state.as_slice(), e)?;
        encode_f32s(self.previous.as_slice(), e)
    }
}

impl Decode<()> for InferenceRequest {
    fn decode<D: cu29::bincode::de::Decoder<Context = ()>>(
        d: &mut D,
    ) -> Result<Self, cu29::bincode::error::DecodeError> {
        let obs_seq = Decode::decode(d)?;
        let delay = Decode::decode(d)?;
        let executed = Decode::decode(d)?;
        let reason = Decode::decode(d)?;
        let mut state_buf = [0f32; OBS_JOINTS];
        let n = decode_f32s(d, &mut state_buf)?;
        let mut state = CuArray::new();
        state.fill_from_iter(state_buf[..n].iter().copied());
        let mut prev_buf = [0f32; CHUNK_LEN];
        let m = decode_f32s(d, &mut prev_buf)?;
        let mut previous = CuArray::new();
        previous.fill_from_iter(prev_buf[..m].iter().copied());
        Ok(Self {
            obs_seq,
            delay,
            executed,
            reason,
            state,
            previous,
        })
    }
}

/// Wire format (little-endian, fixed-width integers), shared with the Python runner:
///
/// `ActionChunk`: `obs_seq: u64`, `len: u32`, `len` x `f32`.
/// `ObsPacket`:   `seq: u64`, `len: u32`, `len` x `f32`.
///
/// Decoding reads into a stack buffer and never touches the heap, so it is safe on the cycle.
impl Encode for ActionChunk {
    fn encode<E: cu29::bincode::enc::Encoder>(
        &self,
        e: &mut E,
    ) -> Result<(), cu29::bincode::error::EncodeError> {
        Encode::encode(&self.obs_seq, e)?;
        encode_f32s(self.values.as_slice(), e)
    }
}

impl Decode<()> for ActionChunk {
    fn decode<D: cu29::bincode::de::Decoder<Context = ()>>(
        d: &mut D,
    ) -> Result<Self, cu29::bincode::error::DecodeError> {
        let obs_seq = Decode::decode(d)?;
        let mut tmp = [0f32; CHUNK_LEN];
        let len = decode_f32s(d, &mut tmp)?;
        let mut values = CuArray::new();
        values.fill_from_iter(tmp[..len].iter().copied());
        Ok(Self { obs_seq, values })
    }
}

/// Joint state handed to the policy. `seq` identifies the observation; the policy echoes it
/// in the chunk it computes from it.
pub const OBS_JOINTS: usize = 8;

#[derive(Default, Debug, Clone, Serialize, Deserialize, Reflect)]
#[reflect(from_reflect = false)]
pub struct ObsPacket {
    pub seq: u64,
    /// Time of validity of the measurement, nanoseconds of the Copper clock. Camera frames carry
    /// the same clock, so a policy pairs a frame with an observation by time.
    pub tov_ns: u64,
    pub state: CuArray<f32, OBS_JOINTS>,
}

impl Encode for ObsPacket {
    fn encode<E: cu29::bincode::enc::Encoder>(
        &self,
        e: &mut E,
    ) -> Result<(), cu29::bincode::error::EncodeError> {
        Encode::encode(&self.seq, e)?;
        Encode::encode(&self.tov_ns, e)?;
        encode_f32s(self.state.as_slice(), e)
    }
}

impl Decode<()> for ObsPacket {
    fn decode<D: cu29::bincode::de::Decoder<Context = ()>>(
        d: &mut D,
    ) -> Result<Self, cu29::bincode::error::DecodeError> {
        let seq = Decode::decode(d)?;
        let tov_ns = Decode::decode(d)?;
        let mut tmp = [0f32; OBS_JOINTS];
        let len = decode_f32s(d, &mut tmp)?;
        let mut state = CuArray::new();
        state.fill_from_iter(tmp[..len].iter().copied());
        Ok(Self { seq, tov_ns, state })
    }
}

fn encode_f32s<E: cu29::bincode::enc::Encoder>(
    values: &[f32],
    e: &mut E,
) -> Result<(), cu29::bincode::error::EncodeError> {
    Encode::encode(&(values.len() as u32), e)?;
    for v in values {
        Encode::encode(v, e)?;
    }
    Ok(())
}

fn decode_f32s<D: cu29::bincode::de::Decoder<Context = ()>>(
    d: &mut D,
    out: &mut [f32],
) -> Result<usize, cu29::bincode::error::DecodeError> {
    let len: u32 = Decode::decode(d)?;
    let len = len as usize;
    if len > out.len() {
        return Err(cu29::bincode::error::DecodeError::ArrayLengthMismatch {
            required: out.len(),
            found: len,
        });
    }
    for slot in out[..len].iter_mut() {
        *slot = Decode::decode(d)?;
    }
    Ok(len)
}
