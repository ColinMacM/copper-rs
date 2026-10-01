//! Bridge between a Copper graph and an out-of-process policy, over Zenoh.
//!
//! The cycle thread never performs I/O, never allocates and never waits:
//!
//! - `send` encodes the payload into a fixed slot and pushes it into a lock-free ring;
//! - `receive` copies the newest received sample out of a fixed mailbox and decodes it;
//! - a worker thread owns the Zenoh session, publishes from the ring and fills the mailboxes.
//!
//! A full ring drops the new message, and a mailbox keeps only the newest sample. Both are
//! counted in [`LinkStats`]. The worker opens the session in the background and reconnects on
//! failure, so an absent peer or router never stalls the graph.
//!
//! Wire format: bincode with fixed-width little-endian integers, payload only (no `CuMsg`
//! envelope), so a non-Rust peer can decode it with `struct`.

mod status;
mod worker;

use std::any::{Any, TypeId};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use cu_sensor_payloads::CuImage;
use cu29::bincode::config::{Configuration, Fixint, LittleEndian, NoLimit};
use cu29::prelude::*;
use rtrb::{Producer, RingBuffer};

pub use crate::wire::IMAGE_HEADER_BYTES;
pub use status::LinkStatus;
use status::StatusEmitter;
pub use worker::LinkStats;
use worker::{ImgSlot, RxMailbox, SessionSettings, Shared, TxSlot, WorkerChannels};

/// Largest encoded message on a Tx channel.
pub const TX_SLOT_BYTES: usize = 1536;
/// Largest encoded sample accepted on an Rx channel.
pub const RX_SLOT_BYTES: usize = 2048;
/// Messages the Tx ring holds before it starts dropping.
pub const TX_RING_SLOTS: usize = 16;
/// Camera frames in flight between the cycle and the worker. Each holds a pool buffer, so this
/// also bounds how many buffers the link can pin.
pub const IMAGE_RING_SLOTS: usize = 2;
/// A status channel reports at least this often (in cycles) even when nothing changed.
pub const STATUS_HEARTBEAT_CYCLES: u32 = 30;

pub(crate) type WireConfig = Configuration<LittleEndian, Fixint, NoLimit>;

pub(crate) fn wire_config() -> WireConfig {
    cu29::bincode::config::standard().with_fixed_int_encoding()
}

struct TxChannel<Id: Copy> {
    id: Id,
    index: u8,
}

struct RxChannel<Id: Copy> {
    id: Id,
    status: Mutex<StatusEmitter>,
    mailbox: Arc<Mutex<RxMailbox>>,
}

struct Running {
    // Only the cycle thread locks this; the mutex exists because bridges must be `Sync`.
    tx: Mutex<Producer<TxSlot>>,
    images: Mutex<Producer<ImgSlot>>,
    stop: Arc<AtomicBool>,
    worker: JoinHandle<()>,
}

#[derive(Reflect)]
#[reflect(from_reflect = false, no_field_bounds, type_path = false)]
pub struct PolicyLinkBridge<Tx, Rx>
where
    Tx: BridgeChannelSet + 'static,
    Rx: BridgeChannelSet + 'static,
    Tx::Id: Send + Sync + 'static,
    Rx::Id: Send + Sync + 'static,
{
    #[reflect(ignore)]
    settings: SessionSettings,
    #[reflect(ignore)]
    tx_routes: Vec<String>,
    #[reflect(ignore)]
    rx_routes: Vec<String>,
    #[reflect(ignore)]
    tx_channels: Vec<TxChannel<Tx::Id>>,
    #[reflect(ignore)]
    rx_channels: Vec<RxChannel<Rx::Id>>,
    #[reflect(ignore)]
    shared: Arc<Shared>,
    #[reflect(ignore)]
    running: Option<Running>,
}

impl<Tx, Rx> Freezable for PolicyLinkBridge<Tx, Rx>
where
    Tx: BridgeChannelSet + 'static,
    Rx: BridgeChannelSet + 'static,
    Tx::Id: Send + Sync + 'static,
    Rx::Id: Send + Sync + 'static,
{
}

impl<Tx, Rx> cu29::reflect::TypePath for PolicyLinkBridge<Tx, Rx>
where
    Tx: BridgeChannelSet + 'static,
    Rx: BridgeChannelSet + 'static,
    Tx::Id: Send + Sync + 'static,
    Rx::Id: Send + Sync + 'static,
{
    fn type_path() -> &'static str {
        "cu_policy::link::PolicyLinkBridge"
    }
    fn short_type_path() -> &'static str {
        "PolicyLinkBridge"
    }
    fn type_ident() -> Option<&'static str> {
        Some("PolicyLinkBridge")
    }
    fn crate_name() -> Option<&'static str> {
        Some("cu_policy")
    }
    fn module_path() -> Option<&'static str> {
        Some("cu_policy::link")
    }
}

impl<Tx, Rx> PolicyLinkBridge<Tx, Rx>
where
    Tx: BridgeChannelSet + 'static,
    Rx: BridgeChannelSet + 'static,
    Tx::Id: Send + Sync + 'static,
    Rx::Id: Send + Sync + 'static,
{
    /// Counters maintained by the worker and the cycle side.
    #[must_use]
    pub fn stats(&self) -> LinkStats {
        self.shared.snapshot()
    }

    /// Hands a camera frame to the worker without copying it: the slot clones the pooled
    /// buffer's handle. A full ring drops the frame, because a stale frame is of no use.
    fn send_image<Payload: CuMsgPayload>(
        &self,
        running: &Running,
        index: u8,
        ctx: &CuContext,
        msg: &CuMsg<Payload>,
    ) -> CuResult<()> {
        let Some(out) = (msg as &dyn Any).downcast_ref::<CuMsg<CuImage<Vec<u8>>>>() else {
            return Ok(());
        };
        let Some(image) = out.payload() else {
            return Ok(());
        };
        let tov = match out.tov {
            Tov::Time(t) => t,
            _ => ctx.now(),
        };
        let slot = ImgSlot::from_image(index, tov.as_nanos(), image);
        let pushed = running
            .images
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(slot);
        if pushed.is_err() {
            self.shared.img_dropped_full.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }
}

fn route_of<Id: Copy + core::fmt::Debug>(channel: &BridgeChannelConfig<Id>) -> CuResult<String> {
    channel
        .effective_route()
        .map(|r| r.into_owned())
        .ok_or_else(|| {
            CuError::from(format!(
                "PolicyLink: missing route for channel {:?}",
                channel.channel.id
            ))
        })
}

impl<Tx, Rx> CuBridge for PolicyLinkBridge<Tx, Rx>
where
    Tx: BridgeChannelSet + 'static,
    Rx: BridgeChannelSet + 'static,
    Tx::Id: core::fmt::Debug + Send + Sync + 'static,
    Rx::Id: core::fmt::Debug + Send + Sync + 'static,
{
    type Tx = Tx;
    type Rx = Rx;
    type Resources<'r> = ();

    fn new(
        config: Option<&ComponentConfig>,
        tx_channels: &[BridgeChannelConfig<<Self::Tx as BridgeChannelSet>::Id>],
        rx_channels: &[BridgeChannelConfig<<Self::Rx as BridgeChannelSet>::Id>],
        _resources: Self::Resources<'_>,
    ) -> CuResult<Self>
    where
        Self: Sized,
    {
        let settings = SessionSettings::from_config(config)?;
        let mut tx_routes = Vec::new();
        let mut tx = Vec::new();
        for (i, channel) in tx_channels.iter().enumerate() {
            tx_routes.push(route_of(channel)?);
            tx.push(TxChannel {
                id: channel.channel.id,
                index: u8::try_from(i)
                    .map_err(|_| CuError::from("PolicyLink: too many Tx channels"))?,
            });
        }
        let mut rx_routes = Vec::new();
        let mut rx = Vec::new();
        for channel in rx_channels {
            rx_routes.push(route_of(channel)?);
            rx.push(RxChannel {
                id: channel.channel.id,
                status: Mutex::new(StatusEmitter::new(STATUS_HEARTBEAT_CYCLES)),
                mailbox: Arc::new(Mutex::new(RxMailbox::new())),
            });
        }
        Ok(Self {
            settings,
            tx_routes,
            rx_routes,
            tx_channels: tx,
            rx_channels: rx,
            shared: Arc::new(Shared::default()),
            running: None,
        })
    }

    fn start(&mut self, _ctx: &CuContext) -> CuResult<()> {
        let (producer, consumer) = RingBuffer::<TxSlot>::new(TX_RING_SLOTS);
        let (image_producer, image_consumer) = RingBuffer::<ImgSlot>::new(IMAGE_RING_SLOTS);
        let stop = Arc::new(AtomicBool::new(false));
        let channels = WorkerChannels {
            tx_routes: self.tx_routes.clone(),
            rx_routes: self.rx_routes.clone(),
            rx_mailboxes: self
                .rx_channels
                .iter()
                .map(|c| Arc::clone(&c.mailbox))
                .collect(),
        };
        let worker = worker::spawn(
            self.settings.clone(),
            channels,
            consumer,
            image_consumer,
            Arc::clone(&stop),
            Arc::clone(&self.shared),
        )?;
        self.running = Some(Running {
            tx: Mutex::new(producer),
            images: Mutex::new(image_producer),
            stop,
            worker,
        });
        Ok(())
    }

    fn send<'a, Payload>(
        &mut self,
        _ctx: &CuContext,
        channel: &'static BridgeChannel<<Self::Tx as BridgeChannelSet>::Id, Payload>,
        msg: &CuMsg<Payload>,
    ) -> CuResult<()>
    where
        Payload: CuMsgPayload + 'a,
    {
        if msg.payload().is_none() {
            return Ok(());
        }
        let running = self
            .running
            .as_ref()
            .ok_or_else(|| CuError::from("PolicyLink: bridge not started"))?;
        let index = self
            .tx_channels
            .iter()
            .find(|c| c.id == channel.id())
            .ok_or_else(|| {
                CuError::from(format!("PolicyLink: unknown Tx channel {:?}", channel.id()))
            })?
            .index;
        if TypeId::of::<Payload>() == TypeId::of::<CuImage<Vec<u8>>>() {
            return self.send_image(running, index, _ctx, msg);
        }
        let Some(payload) = msg.payload() else {
            return Ok(());
        };
        let mut slot = TxSlot::empty(index);
        match cu29::bincode::encode_into_slice(payload, &mut slot.bytes, wire_config()) {
            Ok(len) => slot.len = len as u32,
            Err(_) => {
                self.shared.tx_too_large.fetch_add(1, Ordering::Relaxed);
                return Ok(());
            }
        }
        let pushed = running
            .tx
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(slot);
        if pushed.is_err() {
            self.shared.tx_dropped_full.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    fn receive<'a, Payload>(
        &mut self,
        ctx: &CuContext,
        channel: &'static BridgeChannel<<Self::Rx as BridgeChannelSet>::Id, Payload>,
        msg: &mut CuMsg<Payload>,
    ) -> CuResult<()>
    where
        Payload: CuMsgPayload + 'a,
    {
        msg.tov = Tov::Time(ctx.now());
        let rx = self
            .rx_channels
            .iter()
            .find(|c| c.id == channel.id())
            .ok_or_else(|| {
                CuError::from(format!("PolicyLink: unknown Rx channel {:?}", channel.id()))
            })?;
        if TypeId::of::<Payload>() == TypeId::of::<LinkStatus>() {
            let status = LinkStatus::from(self.shared.snapshot());
            let due = rx
                .status
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .next(status);
            let out = (msg as &mut dyn Any).downcast_mut::<CuMsg<LinkStatus>>();
            if let Some(out) = out {
                match due {
                    Some(s) => out.set_payload(s),
                    None => out.clear_payload(),
                }
            }
            msg.metadata.clear_origin();
            return Ok(());
        }
        // The worker holds this lock only for a memcpy; if it is mid-write, the sample is
        // taken on the next cycle instead of waiting.
        let mut taken = [0u8; RX_SLOT_BYTES];
        let len = match rx.mailbox.try_lock() {
            Ok(mut mailbox) => mailbox.take_into(&mut taken),
            Err(std::sync::TryLockError::Poisoned(p)) => p.into_inner().take_into(&mut taken),
            Err(std::sync::TryLockError::WouldBlock) => None,
        };
        match len {
            Some(len) => {
                match cu29::bincode::decode_from_slice::<Payload, _>(&taken[..len], wire_config()) {
                    Ok((payload, _)) => msg.set_payload(payload),
                    Err(_) => {
                        self.shared.rx_decode_errors.fetch_add(1, Ordering::Relaxed);
                        msg.clear_payload();
                    }
                }
            }
            None => msg.clear_payload(),
        }
        msg.metadata.clear_origin();
        Ok(())
    }

    fn stop(&mut self, _ctx: &CuContext) -> CuResult<()> {
        if let Some(running) = self.running.take() {
            running.stop.store(true, Ordering::SeqCst);
            // The worker wakes every few hundred microseconds; a wedged session is abandoned
            // after the join timeout in `worker::join`.
            worker::join(running.worker);
        }
        Ok(())
    }
}
