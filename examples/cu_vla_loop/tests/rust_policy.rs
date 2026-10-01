//! The loop against a policy written in Rust: `cu_policy::server` answers the governor's
//! inference requests over loopback Zenoh, no Python involved.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use cu_policy::server::{ServerConfig, ServerStats, serve};
use cu_policy::wire::{CHUNK_LEN, JOINTS, Request};
use cu_vla_loop::{Setting, TRACE, listen_config, run_configured};

static SERIAL: Mutex<()> = Mutex::new(());

const HZ: f64 = 30.0;
const STEPS: usize = 20;

/// Moves every joint along a sine around the start position, `STEPS` steps per chunk. Step `i`
/// of the chunk answering observation `k` belongs to cycle `k + i`, so the sine is a function
/// of that cycle and consecutive chunks agree where they overlap.
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

fn free_port() -> u16 {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    l.local_addr().unwrap().port()
}

#[test]
fn a_policy_written_in_rust_drives_the_arm_inside_the_limits() {
    let _g = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let port = free_port();
    let dir = tempfile::tempdir().unwrap();
    let stop = AtomicBool::new(false);
    let mut stats = ServerStats::default();

    std::thread::scope(|scope| {
        let server = scope.spawn(|| {
            let config = ServerConfig::from_json5(
                "vla",
                &format!(
                    r#"{{mode:"peer",scouting:{{multicast:{{enabled:false}}}},connect:{{endpoints:["tcp/127.0.0.1:{port}"]}}}}"#
                ),
            )
            .unwrap();
            serve(sine, config, &stop).expect("the server runs until it is stopped")
        });
        run_configured(
            300,
            HZ,
            &dir.path().join("rust_policy.copper"),
            &listen_config(port),
            // The scheduler asks for a new chunk after 8 of its 20 steps, and a chunk keeps
            // playing for that long.
            &[
                ("sched_s_min", Setting::Num(8.0)),
                ("sched_horizon", Setting::Num(STEPS as f64)),
                ("hold_deadline_ms", Setting::Num(2500.0)),
            ],
            |_| {},
            |_| {},
        )
        .expect("the loop runs to completion");
        stop.store(true, Ordering::Release);
        stats = server.join().unwrap();
    });

    println!("server: {stats:?}");
    assert!(
        stats.requests >= 8,
        "the scheduler asked only {} times",
        stats.requests
    );
    assert_eq!(stats.chunks, stats.requests, "every request was answered");
    assert_eq!(stats.refused, 0);

    let t = TRACE.lock().unwrap();
    let mut prev: Option<[f32; 8]> = None;
    for (i, g) in t.goals.iter().enumerate() {
        let Some(g) = g else { continue };
        for j in 0..JOINTS {
            assert!(
                g[j].is_finite() && (200.0..=3900.0).contains(&g[j]),
                "goal {i} joint {j} = {}",
                g[j]
            );
            if let Some(p) = prev {
                assert!(
                    (g[j] - p[j]).abs() <= 30.0 + 1e-3,
                    "goal {i} joint {j} jumped"
                );
            }
        }
        prev = Some(*g);
    }
    let lo = t.positions.iter().map(|p| p[0]).fold(f32::MAX, f32::min);
    let hi = t.positions.iter().map(|p| p[0]).fold(f32::MIN, f32::max);
    assert!(hi - lo > 200.0, "the arm barely moved: {}", hi - lo);
}
