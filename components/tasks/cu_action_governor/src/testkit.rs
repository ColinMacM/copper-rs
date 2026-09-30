//! Deterministic synthetic sources/sink used by the tests and benchmarks. Everything is
//! derived from a per-task cycle counter, so a replayed keyframe restores it exactly.
use crate::governor::JointPositions;
use crate::payloads::{ActionChunk, JOINTS, MAX_STEPS, ObsStamp};
use cu29::bincode::de::Decoder;
use cu29::bincode::enc::Encoder;
use cu29::bincode::error::{DecodeError, EncodeError};
use cu29::bincode::{Decode, Encode};
use cu29::prelude::*;

pub const OBS_EVERY: u64 = 5;
pub const CHUNK_EVERY: u64 = 20;

macro_rules! counter_freeze {
    () => {
        fn freeze<E: Encoder>(&self, e: &mut E) -> Result<(), EncodeError> {
            Encode::encode(&self.n, e)
        }
        fn thaw<D: Decoder>(&mut self, d: &mut D) -> Result<(), DecodeError> {
            self.n = Decode::decode(d)?;
            Ok(())
        }
    };
}

/// Observation stamp every OBS_EVERY cycles (seq = cycle / OBS_EVERY).
#[derive(Reflect)]
pub struct StampSrc {
    n: u64,
}
impl Freezable for StampSrc {
    counter_freeze!();
}
impl CuSrcTask for StampSrc {
    type Resources<'r> = ();
    type Output<'m> = output_msg!(ObsStamp);
    fn new(_: Option<&ComponentConfig>, _: Self::Resources<'_>) -> CuResult<Self> {
        Ok(Self { n: 0 })
    }
    fn process(&mut self, ctx: &CuContext, out: &mut Self::Output<'_>) -> CuResult<()> {
        if self.n.is_multiple_of(OBS_EVERY) {
            out.set_payload(ObsStamp {
                seq: self.n / OBS_EVERY,
            });
            out.tov = Tov::Time(ctx.now());
        } else {
            out.clear_payload();
        }
        self.n += 1;
        Ok(())
    }
}

/// Follower feedback: triangle wave, small, always inside limits. Every 97th cycle
/// it reports a NaN joint (a failed read republished as garbage).
#[derive(Reflect)]
pub struct FeedbackSrc {
    n: u64,
}
impl Freezable for FeedbackSrc {
    counter_freeze!();
}
impl CuSrcTask for FeedbackSrc {
    type Resources<'r> = ();
    type Output<'m> = output_msg!(JointPositions);
    fn new(_: Option<&ComponentConfig>, _: Self::Resources<'_>) -> CuResult<Self> {
        Ok(Self { n: 0 })
    }
    fn process(&mut self, ctx: &CuContext, out: &mut Self::Output<'_>) -> CuResult<()> {
        let tri = ((self.n % 200) as f32 - 100.0).abs() / 100.0 - 0.5; // [-0.5,0.5]
        let mut a = JointPositions::new();
        a.fill_from_iter((0..JOINTS).map(|j| {
            if self.n % 97 == 96 && j == 2 {
                f32::NAN
            } else {
                tri * 0.1 * (j as f32 + 1.0)
            }
        }));
        out.set_payload(a);
        out.tov = Tov::Time(ctx.now());
        self.n += 1;
        Ok(())
    }
}

/// Policy stand-in. Every CHUNK_EVERY cycles emits a MAX_STEPS chunk referring to an
/// observation ~3 cycles old. Every 7th chunk holds a NaN, every 11th holds values far
/// outside the joint limits, every 13th refers to an observation 40 cycles old (stale),
/// every 17th is a replay of an older sequence number (out of order).
#[derive(Reflect)]
pub struct ChunkSrc {
    n: u64,
    every: u64,
}
impl Freezable for ChunkSrc {
    counter_freeze!();
}
impl CuSrcTask for ChunkSrc {
    type Resources<'r> = ();
    type Output<'m> = output_msg!(ActionChunk);
    fn new(c: Option<&ComponentConfig>, _: Self::Resources<'_>) -> CuResult<Self> {
        let every = match c {
            Some(c) => c.get::<u32>("chunk_every")?.unwrap_or(CHUNK_EVERY as u32) as u64,
            None => CHUNK_EVERY,
        };
        Ok(Self { n: 0, every })
    }
    fn process(&mut self, _ctx: &CuContext, out: &mut Self::Output<'_>) -> CuResult<()> {
        if self.n.is_multiple_of(self.every) {
            let k = self.n / self.every;
            let lag = if k % 13 == 12 { 40 } else { 3 };
            let mut obs_seq = self.n.saturating_sub(lag) / OBS_EVERY;
            if k % 17 == 16 {
                obs_seq = obs_seq.saturating_sub(4);
            }
            let (nan, wild) = (k % 7 == 6, k % 11 == 10);
            let base = (k % 5) as f32 * 0.05;
            let mut c = ActionChunk {
                obs_seq,
                values: CuArray::new(),
            };
            c.values.fill_from_iter((0..MAX_STEPS * JOINTS).map(|i| {
                let (s, j) = (i / JOINTS, i % JOINTS);
                match (nan && s == 20 && j == 1, wild) {
                    (true, _) => f32::NAN,
                    (_, true) => 50.0 + s as f32,
                    _ => base + 0.004 * s as f32 * (j as f32 + 1.0),
                }
            }));
            out.set_payload(c);
        } else {
            out.clear_payload();
        }
        self.n += 1;
        Ok(())
    }
}

/// Test sink: remembers the last goal (frozen, so it participates in replay state).
#[derive(Reflect)]
pub struct GoalSink {
    n: u64,
}
impl Freezable for GoalSink {
    counter_freeze!();
}
impl CuSinkTask for GoalSink {
    type Resources<'r> = ();
    type Input<'m> = input_msg!(JointPositions);
    fn new(_: Option<&ComponentConfig>, _: Self::Resources<'_>) -> CuResult<Self> {
        Ok(Self { n: 0 })
    }
    fn process(&mut self, _ctx: &CuContext, input: &Self::Input<'_>) -> CuResult<()> {
        if let Some(g) = input.payload() {
            std::hint::black_box(g.as_slice());
        }
        std::hint::black_box(input.metadata.status_txt.0.as_str());
        self.n += 1;
        Ok(())
    }
}
