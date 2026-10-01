//! Policy loop for Copper: a policy process, such as a vision-language-action model, proposes
//! chunks of future actions; a governor in the Copper graph gates them before they reach an arm.
//!
//! - [`wire`] is the byte format of every message between the graph and the policy process. It
//!   has no dependencies and works without `std`.
//! - [`governor::ActionGovernor`] is a `CuTask`. Its inputs are an [`ActionChunk`], the measured
//!   joint positions and an [`ObsStamp`]; it outputs one goal per cycle, the state of what it
//!   executes ([`ExecState`]) and, when its scheduler decides the policy should be asked again,
//!   an [`InferenceRequest`]. Everything `process()` touches has a fixed capacity, so the cycle
//!   is allocation-free.
//! - [`link`] (feature `link`) is the Zenoh bridge that carries these messages to the policy
//!   process.
//! - [`server`] (feature `server`) serves a policy written in Rust to the loop.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
pub mod governor;
#[cfg(feature = "link")]
pub mod link;
#[cfg(feature = "std")]
pub mod payloads;
#[cfg(feature = "server")]
pub mod server;
#[cfg(feature = "testkit")]
pub mod testkit;
pub mod wire;

#[cfg(feature = "std")]
pub use governor::{
    ActionGovernor, GovernorCore, GovernorParams, JointPositions, SchedParams, Status,
};
#[cfg(feature = "std")]
pub use payloads::{
    ActionChunk, CHUNK_LEN, ExecState, InferenceRequest, JOINTS, MAX_STEPS, OBS_JOINTS, ObsPacket,
    ObsStamp,
};
