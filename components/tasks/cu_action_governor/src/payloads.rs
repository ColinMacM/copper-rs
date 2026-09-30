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
