//! Policy loop for Copper.
//!
//! [`wire`] is the byte format of every message that crosses between a Copper graph and a
//! policy process. It has no dependencies and works without `std`.

#![cfg_attr(not(feature = "std"), no_std)]

pub mod wire;
