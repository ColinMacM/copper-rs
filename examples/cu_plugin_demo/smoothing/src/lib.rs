//! Tasks used by the `smoothing` plugin in `plugin/`.

use bincode::de::Decoder;
use bincode::enc::Encoder;
use bincode::error::{DecodeError, EncodeError};
use bincode::{Decode, Encode};
use cu29::prelude::*;
use serde::{Deserialize, Serialize};

/// One measurement.
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize, Encode, Decode, Reflect)]
pub struct Sample {
    pub value: f64,
}

/// A filtered measurement and the number of samples that contributed to it.
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize, Encode, Decode, Reflect)]
pub struct Smoothed {
    pub value: f64,
    pub filled: u32,
}

/// Averages the last `window` samples and multiplies the result by `gain`.
#[derive(Reflect)]
pub struct MovingAverage {
    samples: Vec<f64>,
    next: usize,
    filled: usize,
    gain: f64,
}

impl Freezable for MovingAverage {
    fn freeze<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        Encode::encode(&self.samples, encoder)?;
        Encode::encode(&(self.next as u64), encoder)?;
        Encode::encode(&(self.filled as u64), encoder)
    }

    fn thaw<D: Decoder>(&mut self, decoder: &mut D) -> Result<(), DecodeError> {
        self.samples = Decode::decode(decoder)?;
        let next: u64 = Decode::decode(decoder)?;
        let filled: u64 = Decode::decode(decoder)?;
        self.next = next as usize;
        self.filled = filled as usize;
        Ok(())
    }
}

impl CuTask for MovingAverage {
    type Input<'m> = input_msg!(Sample);
    type Output<'m> = output_msg!(Smoothed);
    type Resources<'r> = ();

    fn new(config: Option<&ComponentConfig>, _resources: Self::Resources<'_>) -> CuResult<Self> {
        let config = config.ok_or("MovingAverage needs a config with 'window'")?;
        let window = config
            .get::<u32>("window")?
            .ok_or("'window' not found in config")?;
        if window == 0 {
            return Err("'window' must be at least 1".into());
        }
        Ok(Self {
            samples: vec![0.0; window as usize],
            next: 0,
            filled: 0,
            gain: config.get::<f64>("gain")?.unwrap_or(1.0),
        })
    }

    fn process(
        &mut self,
        _ctx: &CuContext,
        input: &Self::Input<'_>,
        output: &mut Self::Output<'_>,
    ) -> CuResult<()> {
        let Some(sample) = input.payload() else {
            output.clear_payload();
            return Ok(());
        };
        self.samples[self.next] = sample.value;
        self.next = (self.next + 1) % self.samples.len();
        self.filled = (self.filled + 1).min(self.samples.len());
        let sum: f64 = self.samples[..self.filled].iter().sum();
        output.set_payload(Smoothed {
            value: sum / self.filled as f64 * self.gain,
            filled: self.filled as u32,
        });
        Ok(())
    }
}

/// Clamps a smoothed value to `[-max_value, max_value]`.
#[derive(Reflect)]
pub struct Limit {
    max_value: f64,
}

impl Freezable for Limit {}

impl CuTask for Limit {
    type Input<'m> = input_msg!(Smoothed);
    type Output<'m> = output_msg!(Smoothed);
    type Resources<'r> = ();

    fn new(config: Option<&ComponentConfig>, _resources: Self::Resources<'_>) -> CuResult<Self> {
        let max_value = match config {
            Some(config) => config.get::<f64>("max_value")?.unwrap_or(f64::MAX),
            None => f64::MAX,
        };
        Ok(Self { max_value })
    }

    fn process(
        &mut self,
        _ctx: &CuContext,
        input: &Self::Input<'_>,
        output: &mut Self::Output<'_>,
    ) -> CuResult<()> {
        let Some(smoothed) = input.payload() else {
            output.clear_payload();
            return Ok(());
        };
        output.set_payload(Smoothed {
            value: smoothed.value.clamp(-self.max_value, self.max_value),
            filled: smoothed.filled,
        });
        Ok(())
    }
}
