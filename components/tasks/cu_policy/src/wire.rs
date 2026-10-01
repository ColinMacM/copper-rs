//! Wire format of the policy link: little-endian, fixed-width integers, payload only.
//!
//! | Message | Layout |
//! | --- | --- |
//! | [`Obs`] | `seq: u64`, `tov_ns: u64`, `len: u32`, `len` x `f32` |
//! | [`Chunk`] | `obs_seq: u64`, `len: u32`, `len` x `f32` |
//! | [`Exec`] | `stamp_seq: u64`, `chunk_seq: u64`, `next_index: u32`, `flags: u32`, `accept_skip: u32`, `reject: u32`, `tracking_err: f32` |
//! | [`Request`] | `obs_seq: u64`, `delay: u32`, `executed: u32`, `reason: u32`, the [`PolicyOptions`] (`horizon: u32`, `mode: u32`, `denoise_steps: u32`, `best_of: u32`, `flags: u32`, `beta: f32`), `state_len: u32`, `state_len` x `f32`, `previous_len: u32`, `previous_len` x `f32` |
//! | [`ImageHeader`] | `seq: u64`, `tov_ns: u64`, `width: u32`, `height: u32`, `stride: u32`, `pixel_format: [u8; 4]`, `len: u32`, followed by `len` bytes of pixels |
//!
//! Each message has one definition, a pair of functions generic over [`Sink`] and [`Source`].
//! [`SliceWriter`] and [`SliceReader`] put it on a byte slice; an adapter for another encoder
//! (the Copper log's) implements the two traits and gets the same layout without a copy.
//! The `read_*` functions fill buffers the caller provides, so decoding is stack-only and
//! copies no values; the owned [`Obs`], [`Chunk`] and [`Request`] and the `decode_*`
//! functions serve tools and tests. A length field above the capacity is an error, and the
//! capacity is checked before any value is read.

use core::fmt;

/// Arm joints in a chunk step.
pub const JOINTS: usize = 6;
/// Steps in a chunk.
pub const MAX_STEPS: usize = 50;
/// Values in a chunk.
pub const CHUNK_LEN: usize = JOINTS * MAX_STEPS;
/// Values in an observation.
pub const OBS_JOINTS: usize = 8;
/// Bytes of an [`ImageHeader`].
pub const IMAGE_HEADER_BYTES: usize = 8 + 8 + 4 + 4 + 4 + 4 + 4;

/// Why a message could not be encoded or decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireError {
    /// The input ended before the message did.
    Truncated,
    /// A length field exceeds the capacity of the message.
    TooLong { max: usize, found: usize },
    /// The message exceeds the output slice.
    BufferTooSmall,
}

impl fmt::Display for WireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WireError::Truncated => f.write_str("the message ends early"),
            WireError::TooLong { max, found } => {
                write!(f, "a length field says {found}, the capacity is {max}")
            }
            WireError::BufferTooSmall => f.write_str("the output buffer is too small"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for WireError {}

/// Where a message is written: a byte slice, or another encoder.
pub trait Sink {
    type Error;
    /// The error for a list of `found` values above the capacity `max`.
    fn too_long(max: usize, found: usize) -> Self::Error;
    fn put_u32(&mut self, v: u32) -> Result<(), Self::Error>;
    fn put_u64(&mut self, v: u64) -> Result<(), Self::Error>;
    fn put_f32(&mut self, v: f32) -> Result<(), Self::Error>;
}

/// Where a message is read from: a byte slice, or another decoder.
pub trait Source {
    type Error;
    /// The error for a length field `found` above the capacity `max`.
    fn too_long(max: usize, found: usize) -> Self::Error;
    fn get_u32(&mut self) -> Result<u32, Self::Error>;
    fn get_u64(&mut self) -> Result<u64, Self::Error>;
    fn get_f32(&mut self) -> Result<f32, Self::Error>;
}

/// Writes into a byte slice.
pub struct SliceWriter<'a> {
    buf: &'a mut [u8],
    pos: usize,
}

impl<'a> SliceWriter<'a> {
    #[must_use]
    pub fn new(buf: &'a mut [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    /// Bytes written so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pos
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pos == 0
    }

    fn put(&mut self, bytes: &[u8]) -> Result<(), WireError> {
        let end = self.pos + bytes.len();
        let slot = self
            .buf
            .get_mut(self.pos..end)
            .ok_or(WireError::BufferTooSmall)?;
        slot.copy_from_slice(bytes);
        self.pos = end;
        Ok(())
    }
}

impl Sink for SliceWriter<'_> {
    type Error = WireError;

    fn too_long(max: usize, found: usize) -> WireError {
        WireError::TooLong { max, found }
    }

    fn put_u32(&mut self, v: u32) -> Result<(), WireError> {
        self.put(&v.to_le_bytes())
    }

    fn put_u64(&mut self, v: u64) -> Result<(), WireError> {
        self.put(&v.to_le_bytes())
    }

    fn put_f32(&mut self, v: f32) -> Result<(), WireError> {
        self.put(&v.to_le_bytes())
    }
}

/// Reads from a byte slice.
pub struct SliceReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> SliceReader<'a> {
    #[must_use]
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    /// Bytes consumed so far.
    #[must_use]
    pub fn consumed(&self) -> usize {
        self.pos
    }

    fn take<const N: usize>(&mut self) -> Result<[u8; N], WireError> {
        let end = self.pos + N;
        let bytes = self.buf.get(self.pos..end).ok_or(WireError::Truncated)?;
        let mut out = [0u8; N];
        out.copy_from_slice(bytes);
        self.pos = end;
        Ok(out)
    }
}

impl Source for SliceReader<'_> {
    type Error = WireError;

    fn too_long(max: usize, found: usize) -> WireError {
        WireError::TooLong { max, found }
    }

    fn get_u32(&mut self) -> Result<u32, WireError> {
        self.take::<4>().map(u32::from_le_bytes)
    }

    fn get_u64(&mut self) -> Result<u64, WireError> {
        self.take::<8>().map(u64::from_le_bytes)
    }

    fn get_f32(&mut self) -> Result<f32, WireError> {
        self.take::<4>().map(f32::from_le_bytes)
    }
}

fn write_f32s<S: Sink>(s: &mut S, values: &[f32], max: usize) -> Result<(), S::Error> {
    if values.len() > max {
        return Err(S::too_long(max, values.len()));
    }
    s.put_u32(values.len() as u32)?;
    for v in values {
        s.put_f32(*v)?;
    }
    Ok(())
}

/// Reads a length-prefixed list of `f32` into `out` and returns how many were read.
fn read_f32s<S: Source>(s: &mut S, out: &mut [f32]) -> Result<usize, S::Error> {
    let len = s.get_u32()? as usize;
    if len > out.len() {
        return Err(S::too_long(out.len(), len));
    }
    for slot in &mut out[..len] {
        *slot = s.get_f32()?;
    }
    Ok(len)
}

/// An observation: the measured joint state with its sequence number and time of validity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Obs {
    pub seq: u64,
    pub tov_ns: u64,
    pub len: usize,
    pub state: [f32; OBS_JOINTS],
}

impl Obs {
    #[must_use]
    pub fn state(&self) -> &[f32] {
        &self.state[..self.len]
    }
}

/// Writes an observation. `state` has at most [`OBS_JOINTS`] values.
pub fn write_obs<S: Sink>(s: &mut S, seq: u64, tov_ns: u64, state: &[f32]) -> Result<(), S::Error> {
    s.put_u64(seq)?;
    s.put_u64(tov_ns)?;
    write_f32s(s, state, OBS_JOINTS)
}

/// The scalar fields of an observation read by [`read_obs`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObsHeader {
    pub seq: u64,
    pub tov_ns: u64,
    /// Values written into the buffer.
    pub len: usize,
}

/// Reads an observation; its values go to the front of `state`.
pub fn read_obs<S: Source>(
    s: &mut S,
    state: &mut [f32; OBS_JOINTS],
) -> Result<ObsHeader, S::Error> {
    let seq = s.get_u64()?;
    let tov_ns = s.get_u64()?;
    let len = read_f32s(s, state)?;
    Ok(ObsHeader { seq, tov_ns, len })
}

/// An action chunk: steps of [`JOINTS`] values, row-major, answering the observation `obs_seq`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Chunk {
    pub obs_seq: u64,
    pub len: usize,
    pub values: [f32; CHUNK_LEN],
}

impl Chunk {
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values[..self.len]
    }
}

/// Writes a chunk. `values` has at most [`CHUNK_LEN`] values.
pub fn write_chunk<S: Sink>(s: &mut S, obs_seq: u64, values: &[f32]) -> Result<(), S::Error> {
    s.put_u64(obs_seq)?;
    write_f32s(s, values, CHUNK_LEN)
}

/// The scalar fields of a chunk read by [`read_chunk`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkHeader {
    pub obs_seq: u64,
    /// Values written into the buffer.
    pub len: usize,
}

/// Reads a chunk; its values go to the front of `values`.
pub fn read_chunk<S: Source>(
    s: &mut S,
    values: &mut [f32; CHUNK_LEN],
) -> Result<ChunkHeader, S::Error> {
    let obs_seq = s.get_u64()?;
    let len = read_f32s(s, values)?;
    Ok(ChunkHeader { obs_seq, len })
}

/// What the governor executes, sent every cycle.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Exec {
    pub stamp_seq: u64,
    pub chunk_seq: u64,
    pub next_index: u32,
    pub flags: u32,
    pub accept_skip: u32,
    pub reject: u32,
    pub tracking_err: f32,
}

pub fn write_exec<S: Sink>(s: &mut S, e: &Exec) -> Result<(), S::Error> {
    s.put_u64(e.stamp_seq)?;
    s.put_u64(e.chunk_seq)?;
    s.put_u32(e.next_index)?;
    s.put_u32(e.flags)?;
    s.put_u32(e.accept_skip)?;
    s.put_u32(e.reject)?;
    s.put_f32(e.tracking_err)
}

pub fn read_exec<S: Source>(s: &mut S) -> Result<Exec, S::Error> {
    Ok(Exec {
        stamp_seq: s.get_u64()?,
        chunk_seq: s.get_u64()?,
        next_index: s.get_u32()?,
        flags: s.get_u32()?,
        accept_skip: s.get_u32()?,
        reject: s.get_u32()?,
        tracking_err: s.get_f32()?,
    })
}

/// No guidance: each chunk is sampled freely and replaces the executing one.
pub const MODE_NAIVE: u32 = 0;
/// Real-time chunking: the new chunk is inpainted against the executing chunk's unplayed steps.
pub const MODE_RTC: u32 = 1;

/// [`PolicyOptions::flags`]: set the frozen steps to the executing chunk's values exactly.
pub const FLAG_PROJECT: u32 = 1;
/// [`PolicyOptions::flags`]: plan from the state expected after the frozen steps.
pub const FLAG_ROLL_OBS: u32 = 2;
/// [`PolicyOptions::flags`]: draw the noise of each step from a function of its control cycle.
pub const FLAG_POSITIONAL_NOISE: u32 = 4;

/// How the policy is to plan, sent with every request so that the graph's configuration is the
/// only place these are set.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolicyOptions {
    /// Prediction horizon of the policy, in steps.
    pub horizon: u32,
    /// [`MODE_NAIVE`] or [`MODE_RTC`].
    pub mode: u32,
    /// Denoising steps of a flow policy.
    pub denoise_steps: u32,
    /// Guided samples drawn per chunk; the one with the smallest prefix residual is kept.
    pub best_of: u32,
    /// `FLAG_*` bits.
    pub flags: u32,
    /// Clip of the guidance weight.
    pub beta: f32,
}

impl Default for PolicyOptions {
    fn default() -> Self {
        Self {
            horizon: MAX_STEPS as u32,
            mode: MODE_NAIVE,
            denoise_steps: 5,
            best_of: 1,
            flags: 0,
            beta: 5.0,
        }
    }
}

/// A request to the policy: the observation, the delay estimate, the steps of the active chunk
/// already played, and that chunk's unplayed remainder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Request {
    pub obs_seq: u64,
    pub delay: u32,
    pub executed: u32,
    pub reason: u32,
    pub options: PolicyOptions,
    pub state_len: usize,
    pub state: [f32; OBS_JOINTS],
    pub previous_len: usize,
    pub previous: [f32; CHUNK_LEN],
}

impl Request {
    #[must_use]
    pub fn state(&self) -> &[f32] {
        &self.state[..self.state_len]
    }

    #[must_use]
    pub fn previous(&self) -> &[f32] {
        &self.previous[..self.previous_len]
    }
}

/// Writes a request. `state` has at most [`OBS_JOINTS`] values and `previous` at most
/// [`CHUNK_LEN`].
#[allow(clippy::too_many_arguments)]
pub fn write_request<S: Sink>(
    s: &mut S,
    obs_seq: u64,
    delay: u32,
    executed: u32,
    reason: u32,
    options: &PolicyOptions,
    state: &[f32],
    previous: &[f32],
) -> Result<(), S::Error> {
    s.put_u64(obs_seq)?;
    s.put_u32(delay)?;
    s.put_u32(executed)?;
    s.put_u32(reason)?;
    s.put_u32(options.horizon)?;
    s.put_u32(options.mode)?;
    s.put_u32(options.denoise_steps)?;
    s.put_u32(options.best_of)?;
    s.put_u32(options.flags)?;
    s.put_f32(options.beta)?;
    write_f32s(s, state, OBS_JOINTS)?;
    write_f32s(s, previous, CHUNK_LEN)
}

/// The scalar fields of a request read by [`read_request`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RequestHeader {
    pub obs_seq: u64,
    pub delay: u32,
    pub executed: u32,
    pub reason: u32,
    pub options: PolicyOptions,
    /// Values written into the state buffer.
    pub state_len: usize,
    /// Values written into the previous buffer.
    pub previous_len: usize,
}

/// Reads a request; the state goes to the front of `state` and the unplayed remainder to the
/// front of `previous`.
pub fn read_request<S: Source>(
    s: &mut S,
    state: &mut [f32; OBS_JOINTS],
    previous: &mut [f32; CHUNK_LEN],
) -> Result<RequestHeader, S::Error> {
    let obs_seq = s.get_u64()?;
    let delay = s.get_u32()?;
    let executed = s.get_u32()?;
    let reason = s.get_u32()?;
    let options = PolicyOptions {
        horizon: s.get_u32()?,
        mode: s.get_u32()?,
        denoise_steps: s.get_u32()?,
        best_of: s.get_u32()?,
        flags: s.get_u32()?,
        beta: s.get_f32()?,
    };
    let state_len = read_f32s(s, state)?;
    let previous_len = read_f32s(s, previous)?;
    Ok(RequestHeader {
        obs_seq,
        delay,
        executed,
        reason,
        options,
        state_len,
        previous_len,
    })
}

/// The header of a camera frame; `len` bytes of pixels follow it on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageHeader {
    pub seq: u64,
    pub tov_ns: u64,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub pixel_format: [u8; 4],
    pub len: u32,
}

impl ImageHeader {
    /// The header as bytes.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; IMAGE_HEADER_BYTES] {
        let mut out = [0u8; IMAGE_HEADER_BYTES];
        let mut w = SliceWriter::new(&mut out);
        // The array is exactly the header's size, so none of these can fail.
        let _ = w.put_u64(self.seq);
        let _ = w.put_u64(self.tov_ns);
        let _ = w.put_u32(self.width);
        let _ = w.put_u32(self.height);
        let _ = w.put_u32(self.stride);
        let _ = w.put(&self.pixel_format);
        let _ = w.put_u32(self.len);
        out
    }

    /// Reads a header from the start of `bytes`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, WireError> {
        let mut r = SliceReader::new(bytes);
        Ok(Self {
            seq: r.get_u64()?,
            tov_ns: r.get_u64()?,
            width: r.get_u32()?,
            height: r.get_u32()?,
            stride: r.get_u32()?,
            pixel_format: r.take::<4>()?,
            len: r.get_u32()?,
        })
    }
}

/// Encodes an observation into `out` and returns the bytes written.
pub fn encode_obs(
    out: &mut [u8],
    seq: u64,
    tov_ns: u64,
    state: &[f32],
) -> Result<usize, WireError> {
    let mut w = SliceWriter::new(out);
    write_obs(&mut w, seq, tov_ns, state)?;
    Ok(w.len())
}

/// Decodes an observation from the start of `bytes`; returns it and the bytes consumed.
pub fn decode_obs(bytes: &[u8]) -> Result<(Obs, usize), WireError> {
    let mut r = SliceReader::new(bytes);
    let mut state = [0f32; OBS_JOINTS];
    let h = read_obs(&mut r, &mut state)?;
    let obs = Obs {
        seq: h.seq,
        tov_ns: h.tov_ns,
        len: h.len,
        state,
    };
    Ok((obs, r.consumed()))
}

/// Encodes a chunk into `out` and returns the bytes written.
pub fn encode_chunk(out: &mut [u8], obs_seq: u64, values: &[f32]) -> Result<usize, WireError> {
    let mut w = SliceWriter::new(out);
    write_chunk(&mut w, obs_seq, values)?;
    Ok(w.len())
}

/// Decodes a chunk from the start of `bytes`; returns it and the bytes consumed.
pub fn decode_chunk(bytes: &[u8]) -> Result<(Chunk, usize), WireError> {
    let mut r = SliceReader::new(bytes);
    let mut values = [0f32; CHUNK_LEN];
    let h = read_chunk(&mut r, &mut values)?;
    let chunk = Chunk {
        obs_seq: h.obs_seq,
        len: h.len,
        values,
    };
    Ok((chunk, r.consumed()))
}

/// Encodes an exec state into `out` and returns the bytes written.
pub fn encode_exec(out: &mut [u8], e: &Exec) -> Result<usize, WireError> {
    let mut w = SliceWriter::new(out);
    write_exec(&mut w, e)?;
    Ok(w.len())
}

/// Decodes an exec state from the start of `bytes`; returns it and the bytes consumed.
pub fn decode_exec(bytes: &[u8]) -> Result<(Exec, usize), WireError> {
    let mut r = SliceReader::new(bytes);
    let exec = read_exec(&mut r)?;
    Ok((exec, r.consumed()))
}

/// Encodes a request into `out` and returns the bytes written.
#[allow(clippy::too_many_arguments)]
pub fn encode_request(
    out: &mut [u8],
    obs_seq: u64,
    delay: u32,
    executed: u32,
    reason: u32,
    options: &PolicyOptions,
    state: &[f32],
    previous: &[f32],
) -> Result<usize, WireError> {
    let mut w = SliceWriter::new(out);
    write_request(
        &mut w, obs_seq, delay, executed, reason, options, state, previous,
    )?;
    Ok(w.len())
}

/// Decodes a request from the start of `bytes`; returns it and the bytes consumed.
pub fn decode_request(bytes: &[u8]) -> Result<(Request, usize), WireError> {
    let mut r = SliceReader::new(bytes);
    let mut state = [0f32; OBS_JOINTS];
    let mut previous = [0f32; CHUNK_LEN];
    let h = read_request(&mut r, &mut state, &mut previous)?;
    let request = Request {
        obs_seq: h.obs_seq,
        delay: h.delay,
        executed: h.executed,
        reason: h.reason,
        options: h.options,
        state_len: h.state_len,
        state,
        previous_len: h.previous_len,
        previous,
    };
    Ok((request, r.consumed()))
}
