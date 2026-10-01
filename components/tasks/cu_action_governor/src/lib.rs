//! Action governor: the last software gate between a remote, untrusted policy and an arm.
//!
//! [`ActionGovernor`] is a `CuTask`. Its inputs are an [`ActionChunk`], the measured joint
//! positions and an [`ObsStamp`]; its output is one goal per cycle. Everything `process()`
//! touches has a fixed capacity, so the cycle never allocates.

pub mod governor;
pub mod payloads;
#[cfg(feature = "testkit")]
pub mod testkit;

pub use governor::{ActionGovernor, GovernorCore, GovernorParams, JointPositions, Status};
pub use payloads::{
    ActionChunk, CHUNK_LEN, ExecState, JOINTS, MAX_STEPS, OBS_JOINTS, ObsPacket, ObsStamp,
};
