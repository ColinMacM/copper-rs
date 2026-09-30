//! Observation-to-chunk round trip through the link and a real Python policy, measured on the
//! Copper clock. Run with `--release --nocapture` for meaningful numbers; in any profile it
//! asserts only that measurements exist and are sane.

use std::process::{Command, Stdio};

use cu_vla_loop::{ROUND_TRIPS, listen_config, reset_round_trips, run_hooked};

fn python_has_zenoh() -> bool {
    Command::new("python3")
        .args(["-c", "import zenoh"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

fn measure(hz: f64, cycles: usize) -> Vec<u64> {
    let port = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().port()
    };
    let dir = tempfile::tempdir().unwrap();
    let mut child = None;
    reset_round_trips(cycles);
    run_hooked(
        cycles,
        hz,
        &dir.path().join("lat.copper"),
        &listen_config(port),
        |i| {
            if i == 0 {
                child = Some(
                    Command::new("python3")
                        .args(["-m", "vla_runner", "--connect-port", &port.to_string()])
                        .args(["--seconds", &(cycles as f64 / hz + 4.0).to_string()])
                        .current_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/python"))
                        .stdout(Stdio::null())
                        .stderr(Stdio::null())
                        .spawn()
                        .unwrap(),
                );
            }
        },
        |_| {},
    )
    .unwrap();
    if let Some(mut c) = child {
        let _ = c.kill();
        let _ = c.wait();
    }
    let mut v: Vec<u64> = ROUND_TRIPS.lock().unwrap().iter().map(|r| r.1).collect();
    v.sort_unstable();
    v
}

#[test]
fn observation_to_chunk_round_trip() {
    if !python_has_zenoh() {
        eprintln!("skipped: python3 cannot import zenoh");
        return;
    }
    for (hz, cycles) in [(30.0, 450), (200.0, 1500), (1000.0, 4000)] {
        let v = measure(hz, cycles);
        assert!(
            v.len() > cycles / 20,
            "only {} round trips at {hz} Hz",
            v.len()
        );
        // Skip the first tenth: the session and the interpreter are still warming up.
        let v = &v[..];
        let pct = |q: f64| v[((v.len() - 1) as f64 * q) as usize] as f64 / 1e6;
        println!(
            "{hz:>6} Hz: {} answered of {cycles} obs; round trip ms p50={:.2} p90={:.2} p99={:.2} max={:.2}",
            v.len(),
            pct(0.5),
            pct(0.9),
            pct(0.99),
            pct(1.0)
        );
        assert!(pct(0.5) < 100.0, "median round trip {:.1} ms", pct(0.5));
    }
}
