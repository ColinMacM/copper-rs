//! Serve a policy written in Rust to the policy loop.
//!
//! The governor's scheduler sends an [`InferenceRequest`](crate::InferenceRequest) whenever the
//! policy should be asked again: the observation, the delay estimate, the steps of the active
//! chunk already played and the unplayed remainder of that chunk. A [`Policy`] answers each
//! request with a chunk; [`serve`] carries the requests and the answers over Zenoh. The server
//! keeps no state beyond the policy itself.
//!
//! ```no_run
//! use std::sync::atomic::AtomicBool;
//! use cu_policy::server::{ServerConfig, serve};
//! use cu_policy::wire::{CHUNK_LEN, JOINTS, Request};
//!
//! // A policy is anything that turns a request into steps of `JOINTS` values.
//! let hold_still = |request: &Request, out: &mut [f32; CHUNK_LEN]| {
//!     for step in out.as_chunks_mut::<JOINTS>().0.iter_mut().take(20) {
//!         step.copy_from_slice(&request.state()[..JOINTS]);
//!     }
//!     20 * JOINTS
//! };
//!
//! let stop = AtomicBool::new(false);
//! let stats = serve(hold_still, ServerConfig::new("vla"), &stop).unwrap();
//! println!("answered {} requests", stats.chunks);
//! ```
//!
//! `prefix` is the instance name of the plugin, or the prefix of the routes configured on the
//! link's channels; the server subscribes to `<prefix>/infer` and publishes on
//! `<prefix>/action`.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::wire::{self, CHUNK_LEN, JOINTS, Request, WireError};

/// Bytes of the largest chunk message.
pub const CHUNK_BYTES: usize = 8 + 4 + 4 * CHUNK_LEN;

/// How long the server waits for a request before it checks whether to stop.
const POLL: Duration = Duration::from_millis(50);

/// Plans the chunk that answers a request.
pub trait Policy {
    /// Writes the chunk to the front of `out`, steps of [`JOINTS`] values, and returns the
    /// number of values. The first step belongs to the control cycle of `request.obs_seq`.
    /// `request.previous()` holds the unplayed remainder of the chunk being executed, for a
    /// policy that stays consistent with it, and is empty when nothing is executing.
    fn plan(&mut self, request: &Request, out: &mut [f32; CHUNK_LEN]) -> usize;
}

impl<F> Policy for F
where
    F: FnMut(&Request, &mut [f32; CHUNK_LEN]) -> usize,
{
    fn plan(&mut self, request: &Request, out: &mut [f32; CHUNK_LEN]) -> usize {
        self(request, out)
    }
}

/// Why a request could not be answered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnswerError {
    /// The request does not decode.
    Request(WireError),
    /// The policy returned a length that is not a whole number of steps, or above the capacity.
    Plan(usize),
    /// The chunk does not fit the reply buffer.
    Reply(WireError),
}

impl fmt::Display for AnswerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AnswerError::Request(e) => write!(f, "the request does not decode: {e}"),
            AnswerError::Plan(n) => write!(
                f,
                "the policy returned {n} values, not a whole number of {JOINTS}-value steps within {CHUNK_LEN}"
            ),
            AnswerError::Reply(e) => write!(f, "the chunk does not fit the reply: {e}"),
        }
    }
}

impl std::error::Error for AnswerError {}

/// Answers one request: decodes `request_bytes`, asks the policy and encodes the chunk into
/// `reply`. Returns the length of the reply. The chunk names the observation of the request.
pub fn answer<P: Policy>(
    policy: &mut P,
    request_bytes: &[u8],
    reply: &mut [u8],
) -> Result<usize, AnswerError> {
    let (request, _) = wire::decode_request(request_bytes).map_err(AnswerError::Request)?;
    let mut values = [0f32; CHUNK_LEN];
    let len = policy.plan(&request, &mut values);
    if len > CHUNK_LEN || !len.is_multiple_of(JOINTS) {
        return Err(AnswerError::Plan(len));
    }
    wire::encode_chunk(reply, request.obs_seq, &values[..len]).map_err(AnswerError::Reply)
}

/// The Zenoh routes of one policy loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Keys {
    pub infer: String,
    pub action: String,
}

impl Keys {
    #[must_use]
    pub fn new(prefix: &str) -> Self {
        Self {
            infer: format!("{prefix}/infer"),
            action: format!("{prefix}/action"),
        }
    }
}

/// Where and how the server connects.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Prefix of the routes, as configured on the link's channels.
    pub key_prefix: String,
    /// The Zenoh session settings; the default opens a peer session that discovers its routers.
    pub zenoh: zenoh::Config,
}

impl ServerConfig {
    /// The default session settings with the given route prefix.
    #[must_use]
    pub fn new(key_prefix: &str) -> Self {
        Self {
            key_prefix: key_prefix.to_owned(),
            zenoh: zenoh::Config::default(),
        }
    }

    /// Session settings from a JSON5 string, the form of the link's `zenoh_config_json`.
    pub fn from_json5(key_prefix: &str, json5: &str) -> Result<Self, ServerError> {
        let zenoh = zenoh::Config::from_json5(json5)
            .map_err(|e| ServerError(format!("cannot parse the Zenoh settings: {e}")))?;
        Ok(Self {
            key_prefix: key_prefix.to_owned(),
            zenoh,
        })
    }
}

/// Counters of one [`serve`] call.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ServerStats {
    /// Requests received.
    pub requests: u64,
    /// Chunks published.
    pub chunks: u64,
    /// Requests that did not decode, or that the policy answered with an unusable length.
    pub refused: u64,
}

/// A failure to open the session or to publish.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerError(pub String);

impl fmt::Display for ServerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ServerError {}

/// Answers requests until `stop` is set. Of the requests that arrive while the policy is busy,
/// only the newest is answered, so a slow policy works on current observations.
pub fn serve<P: Policy>(
    mut policy: P,
    config: ServerConfig,
    stop: &AtomicBool,
) -> Result<ServerStats, ServerError> {
    let keys = Keys::new(&config.key_prefix);
    let fail = |what: &str, e: &dyn fmt::Display| ServerError(format!("{what}: {e}"));
    let session = zenoh::Wait::wait(zenoh::open(config.zenoh))
        .map_err(|e| fail("cannot open the Zenoh session", &e))?;
    let publisher = zenoh::Wait::wait(session.declare_publisher(keys.action.clone()))
        .map_err(|e| fail("cannot declare the publisher", &e))?;
    let requests = zenoh::Wait::wait(
        session
            .declare_subscriber(keys.infer.clone())
            .with(zenoh::handlers::RingChannel::new(1)),
    )
    .map_err(|e| fail("cannot declare the subscriber", &e))?;

    let mut stats = ServerStats::default();
    let mut reply = [0u8; CHUNK_BYTES];
    while !stop.load(Ordering::Acquire) {
        let sample = match requests.recv_timeout(POLL) {
            Ok(Some(sample)) => sample,
            Ok(None) => continue,
            Err(_) => break, // the subscriber is closed
        };
        stats.requests += 1;
        let payload = sample.payload().to_bytes();
        match answer(&mut policy, payload.as_ref(), &mut reply) {
            Ok(len) => {
                zenoh::Wait::wait(publisher.put(&reply[..len]))
                    .map_err(|e| fail("cannot publish a chunk", &e))?;
                stats.chunks += 1;
            }
            Err(_) => stats.refused += 1,
        }
    }
    zenoh::Wait::wait(session.close()).map_err(|e| fail("cannot close the session", &e))?;
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request_bytes(obs_seq: u64, previous: &[f32]) -> Vec<u8> {
        request_with(obs_seq, &wire::PolicyOptions::default(), previous)
    }

    fn request_with(obs_seq: u64, options: &wire::PolicyOptions, previous: &[f32]) -> Vec<u8> {
        let mut buf = vec![0u8; 2048];
        let n =
            wire::encode_request(&mut buf, obs_seq, 3, 7, 2, options, &[1.0; 6], previous).unwrap();
        buf.truncate(n);
        buf
    }

    #[test]
    fn a_reply_names_the_observation_and_carries_the_plan() {
        let mut policy = |r: &Request, out: &mut [f32; CHUNK_LEN]| {
            for (i, v) in out.iter_mut().take(12).enumerate() {
                *v = r.state()[0] + i as f32;
            }
            12
        };
        let mut reply = [0u8; CHUNK_BYTES];
        let n = answer(&mut policy, &request_bytes(41, &[]), &mut reply).unwrap();
        let (chunk, used) = wire::decode_chunk(&reply[..n]).unwrap();
        assert_eq!((used, chunk.obs_seq, chunk.len), (n, 41, 12));
        assert_eq!(chunk.values()[11], 12.0);
    }

    #[test]
    fn the_policy_sees_the_unplayed_remainder_and_the_scheduler_fields() {
        let mut seen = None;
        let mut policy = |r: &Request, _: &mut [f32; CHUNK_LEN]| {
            seen = Some((r.delay, r.executed, r.reason, r.previous().to_vec()));
            JOINTS
        };
        let mut reply = [0u8; CHUNK_BYTES];
        answer(&mut policy, &request_bytes(1, &[0.5; 12]), &mut reply).unwrap();
        assert_eq!(seen, Some((3, 7, 2, vec![0.5; 12])));
    }

    #[test]
    fn the_policy_sees_how_it_is_to_plan() {
        let options = wire::PolicyOptions {
            horizon: 40,
            mode: wire::MODE_RTC,
            denoise_steps: 8,
            best_of: 4,
            flags: wire::FLAG_PROJECT | wire::FLAG_POSITIONAL_NOISE,
            beta: 2.5,
        };
        let mut seen = None;
        let mut policy = |r: &Request, _: &mut [f32; CHUNK_LEN]| {
            seen = Some(r.options);
            JOINTS
        };
        let mut reply = [0u8; CHUNK_BYTES];
        answer(&mut policy, &request_with(1, &options, &[]), &mut reply).unwrap();
        assert_eq!(seen, Some(options));
    }

    #[test]
    fn a_plan_that_is_not_whole_steps_or_too_long_is_refused() {
        let mut reply = [0u8; CHUNK_BYTES];
        for len in [JOINTS + 1, CHUNK_LEN + JOINTS] {
            let mut policy = |_: &Request, _: &mut [f32; CHUNK_LEN]| len;
            assert_eq!(
                answer(&mut policy, &request_bytes(1, &[]), &mut reply),
                Err(AnswerError::Plan(len))
            );
        }
    }

    #[test]
    fn a_request_that_does_not_decode_is_refused_without_asking_the_policy() {
        let mut asked = false;
        let mut policy = |_: &Request, _: &mut [f32; CHUNK_LEN]| {
            asked = true;
            0
        };
        let mut reply = [0u8; CHUNK_BYTES];
        let bytes = request_bytes(1, &[]);
        let err = answer(&mut policy, &bytes[..bytes.len() - 1], &mut reply).unwrap_err();
        assert_eq!(err, AnswerError::Request(WireError::Truncated));
        assert!(!asked);
    }

    #[test]
    fn a_reply_buffer_that_is_too_small_is_an_error() {
        let mut policy = |_: &Request, _: &mut [f32; CHUNK_LEN]| JOINTS;
        let mut small = [0u8; 8];
        assert_eq!(
            answer(&mut policy, &request_bytes(1, &[]), &mut small),
            Err(AnswerError::Reply(WireError::BufferTooSmall))
        );
    }

    #[test]
    fn the_routes_follow_the_prefix() {
        let keys = Keys::new("arm_left");
        assert_eq!(
            (keys.infer.as_str(), keys.action.as_str()),
            ("arm_left/infer", "arm_left/action")
        );
    }
}
