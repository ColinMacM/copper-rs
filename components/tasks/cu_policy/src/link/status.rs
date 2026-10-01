//! The link's counters as a message, so they are logged like everything else.

use bincode::{Decode, Encode};
use cu29::prelude::*;
use serde::{Deserialize, Serialize};

use super::worker::LinkStats;

/// Counters of one link, all monotonic except `session_up`. A Rx channel configured with
/// `kind: "status"` carries this message.
#[derive(
    Default, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Encode, Decode, Reflect,
)]
pub struct LinkStatus {
    pub session_up: bool,
    pub session_opens: u32,
    pub tx_published: u64,
    /// Messages dropped because the Tx ring was full.
    pub tx_dropped_full: u64,
    /// Messages that did not fit a Tx slot.
    pub tx_too_large: u64,
    pub tx_publish_errors: u64,
    pub rx_received: u64,
    /// A received sample replaced another that the cycle never took.
    pub rx_overwritten: u64,
    pub rx_too_large: u64,
    /// A received sample that did not decode. A schema mismatch shows up here and nowhere else:
    /// the receive channel just stays empty.
    pub rx_decode_errors: u64,
    pub img_published: u64,
    /// Frames dropped because the image ring was full.
    pub img_dropped_full: u64,
}

impl From<LinkStats> for LinkStatus {
    fn from(s: LinkStats) -> Self {
        Self {
            session_up: s.session_up,
            session_opens: u32::try_from(s.session_opens).unwrap_or(u32::MAX),
            tx_published: s.tx_published,
            tx_dropped_full: s.tx_dropped_full,
            tx_too_large: s.tx_too_large,
            tx_publish_errors: s.tx_publish_errors,
            rx_received: s.rx_received,
            rx_overwritten: s.rx_overwritten,
            rx_too_large: s.rx_too_large,
            rx_decode_errors: s.rx_decode_errors,
            img_published: s.img_published,
            img_dropped_full: s.img_dropped_full,
        }
    }
}

/// Decides when a status message is emitted: whenever a counter changed, and at least every
/// `every` cycles so a quiet link still shows it is alive.
#[derive(Debug, Clone)]
pub(crate) struct StatusEmitter {
    last: Option<LinkStatus>,
    since: u32,
    every: u32,
}

impl StatusEmitter {
    pub(crate) fn new(every: u32) -> Self {
        Self {
            last: None,
            since: 0,
            every: every.max(1),
        }
    }

    pub(crate) fn next(&mut self, now: LinkStatus) -> Option<LinkStatus> {
        self.since += 1;
        if self.last != Some(now) || self.since >= self.every {
            self.last = Some(now);
            self.since = 0;
            Some(now)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_cycle_always_reports() {
        let mut e = StatusEmitter::new(10);
        assert_eq!(e.next(LinkStatus::default()), Some(LinkStatus::default()));
    }

    #[test]
    fn an_unchanged_link_is_quiet_until_the_heartbeat() {
        let mut e = StatusEmitter::new(4);
        let s = LinkStatus::default();
        assert!(e.next(s).is_some());
        assert!(e.next(s).is_none() && e.next(s).is_none() && e.next(s).is_none());
        assert!(
            e.next(s).is_some(),
            "the fourth quiet cycle is the heartbeat"
        );
        assert!(e.next(s).is_none(), "and the count starts over");
    }

    #[test]
    fn a_changed_counter_reports_at_once() {
        let mut e = StatusEmitter::new(1000);
        let mut s = LinkStatus::default();
        e.next(s);
        assert!(e.next(s).is_none());
        s.rx_decode_errors = 1;
        assert_eq!(e.next(s).unwrap().rx_decode_errors, 1);
        assert!(e.next(s).is_none());
    }

    #[test]
    fn every_counter_is_carried_over() {
        let stats = LinkStats {
            session_up: true,
            session_opens: 2,
            tx_published: 3,
            tx_dropped_full: 4,
            tx_too_large: 5,
            tx_publish_errors: 6,
            rx_received: 7,
            rx_overwritten: 8,
            rx_too_large: 9,
            rx_decode_errors: 10,
            img_published: 11,
            img_dropped_full: 12,
        };
        let s = LinkStatus::from(stats);
        assert_eq!(
            [
                s.tx_published,
                s.tx_dropped_full,
                s.tx_too_large,
                s.tx_publish_errors,
                s.rx_received,
                s.rx_overwritten,
                s.rx_too_large,
                s.rx_decode_errors,
                s.img_published,
                s.img_dropped_full
            ],
            [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        );
        assert!(s.session_up && s.session_opens == 2);
    }
}
