//! The loop against a policy written in Python with `copper_policy.server`: the same requests
//! and answers as `rust_policy.rs`, from a Python process. Skipped when `python3` cannot import
//! `zenoh`.

use std::process::{Command, Stdio};
use std::sync::Mutex;

use cu_vla_loop::{TRACE, listen_config, run_configured};

static SERIAL: Mutex<()> = Mutex::new(());

const HZ: f64 = 30.0;

const POLICY: &str = r#"
import math, sys, json
from copper_policy import server

def policy(request):
    # Step i of the chunk answering observation k belongs to cycle k + i.
    return [2048 + 400 * math.sin(0.08 * (request.obs_seq + i)) for i in range(20) for _ in range(6)]

print(json.dumps(server.serve(policy, int(sys.argv[1]), float(sys.argv[2]), prefix="vla")))
"#;

fn python_has_zenoh() -> bool {
    Command::new("python3")
        .args(["-c", "import zenoh"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

fn count(out: &str, key: &str) -> u64 {
    let needle = format!("\"{key}\": ");
    out.split(&needle)
        .nth(1)
        .and_then(|r| {
            r.chars()
                .take_while(char::is_ascii_digit)
                .collect::<String>()
                .parse()
                .ok()
        })
        .unwrap_or(0)
}

#[test]
fn a_policy_written_in_python_drives_the_arm_inside_the_limits() {
    if !python_has_zenoh() {
        eprintln!("skipped: python3 cannot import zenoh");
        return;
    }
    let _g = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let port = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().port()
    };
    let dir = tempfile::tempdir().unwrap();
    let cycles = 300;
    let mut child = None;
    run_configured(
        cycles,
        HZ,
        &dir.path().join("python_policy.copper"),
        &listen_config(port),
        &[
            ("sched_s_min", 8.0),
            ("sched_horizon", 20.0),
            ("hold_deadline_ms", 2500.0),
        ],
        |i| {
            if i == 0 {
                child = Some(
                    Command::new("python3")
                        .args(["-c", POLICY, &port.to_string()])
                        .arg((cycles as f64 / HZ + 2.0).to_string())
                        .current_dir(concat!(
                            env!("CARGO_MANIFEST_DIR"),
                            "/../../components/tasks/cu_policy/python"
                        ))
                        .stdout(Stdio::piped())
                        .stderr(Stdio::piped())
                        .spawn()
                        .expect("python3"),
                );
            }
        },
        |_| {},
    )
    .expect("the loop runs to completion");
    let out = child.unwrap().wait_with_output().unwrap();
    let text =
        String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr);
    println!("python server: {text}");
    assert!(count(&text, "requests") >= 8, "{text}");
    assert_eq!(count(&text, "chunks"), count(&text, "requests"), "{text}");
    assert_eq!(count(&text, "refused"), 0, "{text}");

    let t = TRACE.lock().unwrap();
    let mut prev: Option<[f32; 8]> = None;
    for (i, g) in t.goals.iter().enumerate() {
        let Some(g) = g else { continue };
        for j in 0..6 {
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
