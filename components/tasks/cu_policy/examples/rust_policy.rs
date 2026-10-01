//! A policy server written in Rust: it moves every joint along a sine around the start position.
//!
//! ```text
//! cargo run -p cu-policy --features server --example rust_policy -- \
//!     --connect-port 7447 --key-prefix vla --seconds 60
//! ```
//!
//! `--key-prefix` is the instance name of the `cu-policy-loop` plugin. Without `--connect-port` the
//! server discovers its peers through Zenoh scouting.

use std::sync::atomic::AtomicBool;

use cu_policy::server::{ServerConfig, serve};
use cu_policy::wire::{CHUNK_LEN, JOINTS, Request};

const STEPS: usize = 20;

/// Step `i` of the chunk answering observation `k` belongs to cycle `k + i`, so the sine is a
/// function of that cycle and consecutive chunks agree where they overlap.
fn sine(request: &Request, out: &mut [f32; CHUNK_LEN]) -> usize {
    for (i, step) in out
        .as_chunks_mut::<JOINTS>()
        .0
        .iter_mut()
        .take(STEPS)
        .enumerate()
    {
        let cycle = request.obs_seq as f32 + i as f32;
        step.fill(2048.0 + 400.0 * (cycle * 0.08).sin());
    }
    STEPS * JOINTS
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut prefix = "vla".to_owned();
    let mut port: Option<u16> = None;
    let mut seconds = 10.0f64;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = |name: &str| args.next().ok_or(format!("{name} needs a value"));
        match arg.as_str() {
            "--key-prefix" => prefix = value("--key-prefix")?,
            "--connect-port" => port = Some(value("--connect-port")?.parse()?),
            "--seconds" => seconds = value("--seconds")?.parse()?,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let config = match port {
        Some(port) => ServerConfig::from_json5(
            &prefix,
            &format!(
                r#"{{mode:"peer",scouting:{{multicast:{{enabled:false}}}},connect:{{endpoints:["tcp/127.0.0.1:{port}"]}}}}"#
            ),
        )?,
        None => ServerConfig::new(&prefix),
    };
    let stop = AtomicBool::new(false);
    let stats = std::thread::scope(|scope| {
        scope.spawn(|| {
            std::thread::sleep(std::time::Duration::from_secs_f64(seconds));
            stop.store(true, std::sync::atomic::Ordering::Release);
        });
        serve(sine, config, &stop)
    })?;
    println!(
        "requests {} chunks {} refused {}",
        stats.requests, stats.chunks, stats.refused
    );
    Ok(())
}
