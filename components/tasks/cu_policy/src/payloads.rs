use bincode::{Decode, Encode};
use cu29::prelude::*;
use serde::{Deserialize, Serialize};

pub use crate::wire::{
    CHUNK_LEN, FLAG_POSITIONAL_NOISE, FLAG_PROJECT, FLAG_ROLL_OBS, JOINTS, MAX_STEPS, MODE_NAIVE,
    MODE_RTC, OBS_JOINTS,
};

use crate::wire::{self, Sink, Source};
use cu29::bincode::de::Decoder;
use cu29::bincode::enc::Encoder;
use cu29::bincode::error::{DecodeError, EncodeError};

/// Writes a wire message into a bincode encoder, field by field, without an intermediate buffer.
/// With the fixed-width little-endian configuration of the policy link the bytes are those of
/// [`crate::wire`]'s slice writer; the Copper log uses its own configuration for the same fields.
struct EncodeSink<'a, E: Encoder>(&'a mut E);

impl<E: Encoder> Sink for EncodeSink<'_, E> {
    type Error = EncodeError;

    fn too_long(_max: usize, _found: usize) -> EncodeError {
        EncodeError::Other("a list is longer than the capacity of its message")
    }

    fn put_u32(&mut self, v: u32) -> Result<(), EncodeError> {
        Encode::encode(&v, self.0)
    }

    fn put_u64(&mut self, v: u64) -> Result<(), EncodeError> {
        Encode::encode(&v, self.0)
    }

    fn put_f32(&mut self, v: f32) -> Result<(), EncodeError> {
        Encode::encode(&v, self.0)
    }
}

/// Reads a wire message from a bincode decoder, field by field.
struct DecodeSource<'a, D: Decoder<Context = ()>>(&'a mut D);

impl<D: Decoder<Context = ()>> Source for DecodeSource<'_, D> {
    type Error = DecodeError;

    fn too_long(max: usize, found: usize) -> DecodeError {
        DecodeError::ArrayLengthMismatch {
            required: max,
            found,
        }
    }

    fn get_u32(&mut self) -> Result<u32, DecodeError> {
        Decode::decode(self.0)
    }

    fn get_u64(&mut self) -> Result<u64, DecodeError> {
        Decode::decode(self.0)
    }

    fn get_f32(&mut self) -> Result<f32, DecodeError> {
        Decode::decode(self.0)
    }
}

/// One policy output: `values.len() / JOINTS` steps, row-major. `CuArray` is a
/// fixed-capacity ArrayVec: `Default` holds for any capacity and
/// only the used prefix is encoded into the log.
#[derive(Default, Debug, Clone, Serialize, Deserialize, Reflect)]
#[reflect(from_reflect = false)]
pub struct ActionChunk {
    /// Sequence number of the observation the model used (echoed by the policy).
    pub obs_seq: u64,
    pub values: CuArray<f32, CHUNK_LEN>,
}

/// Emitted with every observation handed to the policy bridge. The governor maps
/// `seq` to its own clock, so the age of a chunk is measured on the governor's clock.
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
/// Layout on the wire: [`wire::Exec`].
#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Reflect)]
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
/// Layout on the wire: [`wire::Request`].
#[derive(Default, Debug, Clone, Serialize, Deserialize, Reflect)]
#[reflect(from_reflect = false)]
pub struct InferenceRequest {
    pub obs_seq: u64,
    pub delay: u32,
    pub executed: u32,
    /// `REASON_*` bits.
    pub reason: u32,
    /// Prediction horizon of the policy, in steps.
    pub horizon: u32,
    /// `MODE_NAIVE` or `MODE_RTC`.
    pub mode: u32,
    /// Denoising steps of a flow policy.
    pub denoise_steps: u32,
    /// Guided samples drawn per chunk.
    pub best_of: u32,
    /// `FLAG_*` bits.
    pub flags: u32,
    /// Clip of the guidance weight.
    pub beta: f32,
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
    fn encode<E: Encoder>(&self, e: &mut E) -> Result<(), EncodeError> {
        wire::write_request(
            &mut EncodeSink(e),
            self.obs_seq,
            self.delay,
            self.executed,
            self.reason,
            &wire::PolicyOptions {
                horizon: self.horizon,
                mode: self.mode,
                denoise_steps: self.denoise_steps,
                best_of: self.best_of,
                flags: self.flags,
                beta: self.beta,
            },
            self.state.as_slice(),
            self.previous.as_slice(),
        )
    }
}

impl Decode<()> for InferenceRequest {
    fn decode<D: Decoder<Context = ()>>(d: &mut D) -> Result<Self, DecodeError> {
        let mut state_buf = [0f32; OBS_JOINTS];
        let mut prev_buf = [0f32; CHUNK_LEN];
        let h = wire::read_request(&mut DecodeSource(d), &mut state_buf, &mut prev_buf)?;
        let mut state = CuArray::new();
        state.fill_from_iter(state_buf[..h.state_len].iter().copied());
        let mut previous = CuArray::new();
        previous.fill_from_iter(prev_buf[..h.previous_len].iter().copied());
        Ok(Self {
            obs_seq: h.obs_seq,
            delay: h.delay,
            executed: h.executed,
            reason: h.reason,
            horizon: h.options.horizon,
            mode: h.options.mode,
            denoise_steps: h.options.denoise_steps,
            best_of: h.options.best_of,
            flags: h.options.flags,
            beta: h.options.beta,
            state,
            previous,
        })
    }
}

impl Encode for ExecState {
    fn encode<E: Encoder>(&self, e: &mut E) -> Result<(), EncodeError> {
        wire::write_exec(
            &mut EncodeSink(e),
            &wire::Exec {
                stamp_seq: self.stamp_seq,
                chunk_seq: self.chunk_seq,
                next_index: self.next_index,
                flags: self.flags,
                accept_skip: self.accept_skip,
                reject: self.reject,
                tracking_err: self.tracking_err,
            },
        )
    }
}

impl Decode<()> for ExecState {
    fn decode<D: Decoder<Context = ()>>(d: &mut D) -> Result<Self, DecodeError> {
        let e = wire::read_exec(&mut DecodeSource(d))?;
        Ok(Self {
            stamp_seq: e.stamp_seq,
            chunk_seq: e.chunk_seq,
            next_index: e.next_index,
            flags: e.flags,
            accept_skip: e.accept_skip,
            reject: e.reject,
            tracking_err: e.tracking_err,
        })
    }
}

/// The bytes of these messages are defined once, in [`crate::wire`]. Decoding reads into a stack
/// buffer and stays off the heap, so it is safe on the cycle.
impl Encode for ActionChunk {
    fn encode<E: Encoder>(&self, e: &mut E) -> Result<(), EncodeError> {
        wire::write_chunk(&mut EncodeSink(e), self.obs_seq, self.values.as_slice())
    }
}

impl Decode<()> for ActionChunk {
    fn decode<D: Decoder<Context = ()>>(d: &mut D) -> Result<Self, DecodeError> {
        let mut tmp = [0f32; CHUNK_LEN];
        let h = wire::read_chunk(&mut DecodeSource(d), &mut tmp)?;
        let mut values = CuArray::new();
        values.fill_from_iter(tmp[..h.len].iter().copied());
        Ok(Self {
            obs_seq: h.obs_seq,
            values,
        })
    }
}

/// Joint state handed to the policy. `seq` identifies the observation; the policy echoes it
/// in the chunk it computes from it.
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
    fn encode<E: Encoder>(&self, e: &mut E) -> Result<(), EncodeError> {
        wire::write_obs(
            &mut EncodeSink(e),
            self.seq,
            self.tov_ns,
            self.state.as_slice(),
        )
    }
}

impl Decode<()> for ObsPacket {
    fn decode<D: Decoder<Context = ()>>(d: &mut D) -> Result<Self, DecodeError> {
        let mut tmp = [0f32; OBS_JOINTS];
        let h = wire::read_obs(&mut DecodeSource(d), &mut tmp)?;
        let mut state = CuArray::new();
        state.fill_from_iter(tmp[..h.len].iter().copied());
        Ok(Self {
            seq: h.seq,
            tov_ns: h.tov_ns,
            state,
        })
    }
}
