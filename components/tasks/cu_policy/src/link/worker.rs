use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use cu_sensor_payloads::CuImage;
use cu29::prelude::*;
use rtrb::Consumer;
use zenoh::Config;

use super::{RX_SLOT_BYTES, TX_SLOT_BYTES};
use crate::wire::{IMAGE_HEADER_BYTES, ImageHeader};

/// Counters, all monotonic.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LinkStats {
    pub session_up: bool,
    pub session_opens: u64,
    pub tx_published: u64,
    pub tx_dropped_full: u64,
    pub tx_too_large: u64,
    pub tx_publish_errors: u64,
    pub rx_received: u64,
    /// A sample replaced another that was still waiting in the mailbox.
    pub rx_overwritten: u64,
    pub rx_too_large: u64,
    pub rx_decode_errors: u64,
    pub img_published: u64,
    /// Frames dropped because the image ring was full.
    pub img_dropped_full: u64,
}

#[derive(Default)]
pub(crate) struct Shared {
    session_up: AtomicBool,
    session_opens: AtomicU64,
    tx_published: AtomicU64,
    pub(crate) tx_dropped_full: AtomicU64,
    pub(crate) tx_too_large: AtomicU64,
    tx_publish_errors: AtomicU64,
    rx_received: AtomicU64,
    rx_overwritten: AtomicU64,
    rx_too_large: AtomicU64,
    pub(crate) rx_decode_errors: AtomicU64,
    img_published: AtomicU64,
    pub(crate) img_dropped_full: AtomicU64,
}

impl Shared {
    pub(crate) fn snapshot(&self) -> LinkStats {
        let g = |a: &AtomicU64| a.load(Ordering::Relaxed);
        LinkStats {
            session_up: self.session_up.load(Ordering::Relaxed),
            session_opens: g(&self.session_opens),
            tx_published: g(&self.tx_published),
            tx_dropped_full: g(&self.tx_dropped_full),
            tx_too_large: g(&self.tx_too_large),
            tx_publish_errors: g(&self.tx_publish_errors),
            rx_received: g(&self.rx_received),
            rx_overwritten: g(&self.rx_overwritten),
            rx_too_large: g(&self.rx_too_large),
            rx_decode_errors: g(&self.rx_decode_errors),
            img_published: g(&self.img_published),
            img_dropped_full: g(&self.img_dropped_full),
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct TxSlot {
    pub(crate) channel: u8,
    pub(crate) len: u32,
    pub(crate) bytes: [u8; TX_SLOT_BYTES],
}

impl TxSlot {
    pub(crate) fn empty(channel: u8) -> Self {
        Self {
            channel,
            len: 0,
            bytes: [0; TX_SLOT_BYTES],
        }
    }
}

/// A camera frame on its way to the worker. The handle keeps the pool buffer alive until the
/// worker has copied it, so the frame moves between threads as a handle.
pub(crate) struct ImgSlot {
    pub(crate) channel: u8,
    pub(crate) seq: u64,
    pub(crate) tov_ns: u64,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) stride: u32,
    pub(crate) pixel_format: [u8; 4],
    pub(crate) handle: CuHandle<Vec<u8>>,
}

impl ImgSlot {
    pub(crate) fn from_image(channel: u8, tov_ns: u64, image: &CuImage<Vec<u8>>) -> Self {
        Self {
            channel,
            seq: image.seq,
            tov_ns,
            width: image.format.width,
            height: image.format.height,
            stride: image.format.stride,
            pixel_format: image.format.pixel_format,
            handle: image.buffer_handle.clone(),
        }
    }

    /// The header ([`ImageHeader`]) followed by the pixels. `None` if the buffer is shorter than
    /// the format says.
    fn encode(&self) -> Option<Vec<u8>> {
        let len = (self.stride as usize).checked_mul(self.height as usize)?;
        self.handle.with_inner(|inner| {
            let data: &[u8] = inner;
            if data.len() < len {
                return None;
            }
            let header = ImageHeader {
                seq: self.seq,
                tov_ns: self.tov_ns,
                width: self.width,
                height: self.height,
                stride: self.stride,
                pixel_format: self.pixel_format,
                len: u32::try_from(len).ok()?,
            };
            let mut wire = Vec::with_capacity(IMAGE_HEADER_BYTES + len);
            wire.extend_from_slice(&header.to_bytes());
            wire.extend_from_slice(&data[..len]);
            Some(wire)
        })
    }
}

/// Newest received sample for one Rx channel.
pub(crate) struct RxMailbox {
    fresh: bool,
    len: usize,
    bytes: [u8; RX_SLOT_BYTES],
}

impl RxMailbox {
    pub(crate) fn new() -> Self {
        Self {
            fresh: false,
            len: 0,
            bytes: [0; RX_SLOT_BYTES],
        }
    }

    /// Returns true if an unread sample was replaced.
    fn put(&mut self, sample: &[u8]) -> bool {
        let replaced = self.fresh;
        self.bytes[..sample.len()].copy_from_slice(sample);
        self.len = sample.len();
        self.fresh = true;
        replaced
    }

    pub(crate) fn take_into(&mut self, out: &mut [u8; RX_SLOT_BYTES]) -> Option<usize> {
        if !self.fresh {
            return None;
        }
        out[..self.len].copy_from_slice(&self.bytes[..self.len]);
        self.fresh = false;
        Some(self.len)
    }
}

pub(crate) struct WorkerChannels {
    pub(crate) tx_routes: Vec<String>,
    pub(crate) rx_routes: Vec<String>,
    pub(crate) rx_mailboxes: Vec<Arc<Mutex<RxMailbox>>>,
}

#[derive(Clone)]
pub(crate) struct SessionSettings {
    config: Config,
}

impl SessionSettings {
    pub(crate) fn from_config(config: Option<&ComponentConfig>) -> CuResult<Self> {
        let zenoh_config = match config {
            Some(c) => {
                if let Some(path) = c.get::<String>("zenoh_config_file")? {
                    Config::from_file(&path).map_err(|e| {
                        CuError::from(format!("PolicyLink: cannot read zenoh config: {e}"))
                    })?
                } else if let Some(json) = c.get::<String>("zenoh_config_json")? {
                    Config::from_json5(&json).map_err(|e| {
                        CuError::from(format!("PolicyLink: cannot parse zenoh config: {e}"))
                    })?
                } else {
                    Config::default()
                }
            }
            None => Config::default(),
        };
        Ok(Self {
            config: zenoh_config,
        })
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

const IDLE: Duration = Duration::from_micros(100);
const RETRY: Duration = Duration::from_secs(1);

struct Session {
    session: zenoh::Session,
    publishers: Vec<zenoh::pubsub::Publisher<'static>>,
    subscribers:
        Vec<zenoh::pubsub::Subscriber<zenoh::handlers::RingChannelHandler<zenoh::sample::Sample>>>,
}

fn open(settings: &SessionSettings, channels: &WorkerChannels) -> Result<Session, String> {
    let session =
        zenoh::Wait::wait(zenoh::open(settings.config.clone())).map_err(|e| e.to_string())?;
    let mut publishers = Vec::new();
    for route in &channels.tx_routes {
        let key =
            zenoh::key_expr::KeyExpr::<'static>::new(route.clone()).map_err(|e| e.to_string())?;
        publishers
            .push(zenoh::Wait::wait(session.declare_publisher(key)).map_err(|e| e.to_string())?);
    }
    let mut subscribers = Vec::new();
    for route in &channels.rx_routes {
        let key =
            zenoh::key_expr::KeyExpr::<'static>::new(route.clone()).map_err(|e| e.to_string())?;
        subscribers.push(
            zenoh::Wait::wait(
                session
                    .declare_subscriber(key)
                    .with(zenoh::handlers::RingChannel::new(1)),
            )
            .map_err(|e| e.to_string())?,
        );
    }
    Ok(Session {
        session,
        publishers,
        subscribers,
    })
}

fn run(
    settings: &SessionSettings,
    channels: &WorkerChannels,
    mut ring: Consumer<TxSlot>,
    mut images: Consumer<ImgSlot>,
    stop: &AtomicBool,
    shared: &Shared,
) {
    let mut current: Option<Session> = None;
    let mut next_attempt = Instant::now();
    while !stop.load(Ordering::Relaxed) {
        if current.is_none() && Instant::now() >= next_attempt {
            match open(settings, channels) {
                Ok(s) => {
                    shared.session_opens.fetch_add(1, Ordering::Relaxed);
                    shared.session_up.store(true, Ordering::Relaxed);
                    current = Some(s);
                }
                Err(_) => next_attempt = Instant::now() + RETRY,
            }
        }
        let mut idle = true;
        // Messages produced while the session is down are discarded, not queued: a stale
        // observation is worse than none.
        while let Ok(slot) = ring.pop() {
            idle = false;
            let Some(s) = current.as_ref() else { continue };
            let bytes = slot.bytes[..slot.len as usize].to_vec();
            match s.publishers.get(slot.channel as usize) {
                Some(p) => match zenoh::Wait::wait(p.put(bytes)) {
                    Ok(()) => {
                        shared.tx_published.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(_) => {
                        shared.tx_publish_errors.fetch_add(1, Ordering::Relaxed);
                    }
                },
                None => {
                    shared.tx_publish_errors.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        while let Ok(slot) = images.pop() {
            idle = false;
            let Some(s) = current.as_ref() else { continue };
            let wire = slot.encode();
            let channel = slot.channel as usize;
            drop(slot); // the pool buffer is free again as soon as it has been copied
            match (wire, s.publishers.get(channel)) {
                (Some(wire), Some(p)) => match zenoh::Wait::wait(p.put(wire)) {
                    Ok(()) => {
                        shared.img_published.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(_) => {
                        shared.tx_publish_errors.fetch_add(1, Ordering::Relaxed);
                    }
                },
                _ => {
                    shared.tx_publish_errors.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        if let Some(s) = current.as_ref() {
            for (sub, mailbox) in s.subscribers.iter().zip(&channels.rx_mailboxes) {
                match sub.try_recv() {
                    Ok(Some(sample)) => {
                        idle = false;
                        shared.rx_received.fetch_add(1, Ordering::Relaxed);
                        let payload = sample.payload().to_bytes();
                        if payload.len() > RX_SLOT_BYTES {
                            shared.rx_too_large.fetch_add(1, Ordering::Relaxed);
                        } else if lock(mailbox).put(payload.as_ref()) {
                            shared.rx_overwritten.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    Ok(None) => {}
                    Err(_) => {
                        // The subscriber's channel is gone: drop the session and reopen.
                        shared.session_up.store(false, Ordering::Relaxed);
                        current = None;
                        next_attempt = Instant::now() + RETRY;
                        break;
                    }
                }
            }
        }
        if idle {
            std::thread::sleep(IDLE);
        }
    }
    if let Some(s) = current.take() {
        for p in s.publishers {
            let _ = zenoh::Wait::wait(p.undeclare());
        }
        for sub in s.subscribers {
            let _ = zenoh::Wait::wait(sub.undeclare());
        }
        let _ = zenoh::Wait::wait(s.session.close());
    }
    shared.session_up.store(false, Ordering::Relaxed);
}

pub(crate) fn spawn(
    settings: SessionSettings,
    channels: WorkerChannels,
    ring: Consumer<TxSlot>,
    images: Consumer<ImgSlot>,
    stop: Arc<AtomicBool>,
    shared: Arc<Shared>,
) -> CuResult<JoinHandle<()>> {
    std::thread::Builder::new()
        .name("cu-policy-link".into())
        .spawn(move || run(&settings, &channels, ring, images, &stop, &shared))
        .map_err(|e| CuError::from(format!("PolicyLink: cannot start the worker thread: {e}")))
}

/// Waits up to two seconds for the worker; a worker stuck inside Zenoh is left to finish on its own.
pub(crate) fn join(handle: JoinHandle<()>) {
    let deadline = Instant::now() + Duration::from_secs(2);
    while !handle.is_finished() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    if handle.is_finished() {
        let _ = handle.join();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_mailbox_hands_out_each_sample_once_and_keeps_only_the_newest() {
        let mut m = RxMailbox::new();
        let mut out = [0u8; RX_SLOT_BYTES];
        assert_eq!(m.take_into(&mut out), None, "nothing received yet");
        assert!(!m.put(&[1, 2, 3]), "first sample replaces nothing");
        assert!(
            m.put(&[9, 8]),
            "a second sample before a read replaces the first"
        );
        assert_eq!(m.take_into(&mut out), Some(2));
        assert_eq!(&out[..2], &[9, 8], "the newest sample, not the first");
        assert_eq!(m.take_into(&mut out), None, "a sample is delivered once");
    }

    #[test]
    fn a_full_size_sample_fits_the_mailbox() {
        let mut m = RxMailbox::new();
        let sample = vec![7u8; RX_SLOT_BYTES];
        m.put(&sample);
        let mut out = [0u8; RX_SLOT_BYTES];
        assert_eq!(m.take_into(&mut out), Some(RX_SLOT_BYTES));
        assert!(out.iter().all(|b| *b == 7));
    }

    #[test]
    fn the_stats_snapshot_reports_every_counter() {
        let s = Shared::default();
        s.tx_dropped_full.fetch_add(3, Ordering::Relaxed);
        s.rx_decode_errors.fetch_add(2, Ordering::Relaxed);
        let snap = s.snapshot();
        assert_eq!((snap.tx_dropped_full, snap.rx_decode_errors), (3, 2));
        assert!(!snap.session_up);
    }

    #[test]
    fn a_tx_slot_holds_what_an_observation_needs() {
        // 8 (seq) + 4 (len) + 4 * 8 joints.
        const { assert!(TX_SLOT_BYTES >= 8 + 4 + 4 * 8) };
        // 8 (obs_seq) + 4 (len) + 4 * 300 values.
        const { assert!(RX_SLOT_BYTES >= 8 + 4 + 4 * 300) };
    }
}
