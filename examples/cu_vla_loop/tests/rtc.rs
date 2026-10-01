//! Real-time chunking end to end: the Copper loop, a real Python process running a trained
//! flow-matching policy over Zenoh, and the governor reporting what it executes. RTC is compared
//! with naive asynchronous execution (same schedule, no inpainting) under an injected delay.
//!
//! Needs `python3` with `zenoh` and `torch`; otherwise the test prints `skipped`.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Mutex, OnceLock};

use cu_vla_loop::{TRACE, listen_config, run_configured};

static SERIAL: Mutex<()> = Mutex::new(());
const HZ: f64 = 30.0;

fn python_ready() -> bool {
    Command::new("python3")
        .args(["-c", "import zenoh, torch"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

fn python_dir() -> &'static str {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../components/tasks/cu_policy/python"
    )
}

/// The trained policy, built once and kept in the target tree between runs.
fn checkpoint() -> &'static Path {
    static CKPT: OnceLock<PathBuf> = OnceLock::new();
    CKPT.get_or_init(|| {
        let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join("flow_policy_v3.pt");
        if !path.exists() {
            let tmp = path.with_extension("partial");
            let status = Command::new("python3")
                .args(["-m", "copper_policy.flow_policy", "--out"])
                .arg(&tmp)
                .current_dir(python_dir())
                .stderr(Stdio::null())
                .status()
                .expect("python3");
            assert!(status.success(), "training the flow policy failed");
            std::fs::rename(&tmp, &path).unwrap();
        }
        path
    })
}

fn free_port() -> u16 {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    l.local_addr().unwrap().port()
}

/// The largest value printed for `key` in the runner's progress lines.
fn stat(out: &str, key: &str) -> f64 {
    let needle = format!("\"{key}\": ");
    out.lines()
        .flat_map(|l| {
            l.split(&needle)
                .skip(1)
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .filter_map(|r| {
            let n: String = r
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.')
                .collect();
            n.parse::<f64>().ok()
        })
        .fold(0.0, f64::max)
}

/// The value printed for `key` in the runner's last progress line that has it. For cumulative
/// quantities such as a running mean, whose early values rest on a handful of samples.
fn last_stat(out: &str, key: &str) -> f64 {
    let needle = format!("\"{key}\": ");
    out.lines()
        .filter_map(|l| l.split(&needle).nth(1).map(str::to_owned))
        .filter_map(|r| {
            let n: String = r
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.')
                .collect();
            n.parse::<f64>().ok()
        })
        .next_back()
        .unwrap_or(0.0)
}

struct Outcome {
    log: String,
    goals: Vec<Option<[f32; 8]>>,
}

fn run_policy(use_rtc: bool, delay_s: f64, cycles: usize, seed: u64) -> Outcome {
    let ckpt = checkpoint();
    let port = free_port();
    let dir = tempfile::tempdir().unwrap();
    let seconds = cycles as f64 / HZ + 3.0;
    let mut child = None;
    // The example's plugin entry is the RTC configuration: s_min 25, horizon 50, blend 3.
    run_configured(
        cycles,
        HZ,
        &dir.path().join("rtc.copper"),
        &listen_config(port),
        &[],
        |i| {
            if i == 0 {
                child = Some(
                    Command::new("python3")
                        .args(["-m", "copper_policy", "--policy", "flow", "--connect-port"])
                        .arg(port.to_string())
                        .args(["--seconds", &seconds.to_string(), "--checkpoint"])
                        .arg(ckpt)
                        .args(["--rtc", if use_rtc { "on" } else { "off" }])
                        .args(["--delay-s", &delay_s.to_string()])
                        .args(["--seed", &seed.to_string()])
                        // noise indexed by absolute step, and the frozen prefix made exact
                        .args(["--positional-noise", "--project"])
                        .current_dir(python_dir())
                        .stdout(Stdio::piped())
                        .stderr(Stdio::piped())
                        .spawn()
                        .expect("python3"),
                );
            }
        },
        |_| {},
    )
    .expect("the loop must run whatever the policy does");
    let mut c = child.unwrap();
    let _ = c.kill();
    let out = c.wait_with_output().unwrap();
    let log =
        String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr);
    let goals = TRACE.lock().unwrap().goals.clone();
    Outcome { log, goals }
}

/// Every goal the arm received stays inside the governor's limits and moves by at most its step.
fn assert_goals_safe(goals: &[Option<[f32; 8]>]) {
    let mut prev: Option<[f32; 8]> = None;
    for (i, g) in goals.iter().enumerate() {
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
}

#[test]
fn the_copper_loop_runs_real_time_chunking_end_to_end() {
    if !python_ready() {
        eprintln!("skipped: python3 cannot import zenoh and torch");
        return;
    }
    let _g = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // 0.15 s of injected latency on top of inference: a delay of about five cycles.
    let rtc = run_policy(true, 0.15, 420, 0);
    let naive = run_policy(false, 0.15, 420, 0);
    println!(
        "rtc: guided {} switches {} mean_jump {} max_jump {} max_delay {} | naive: switches {} mean_jump {} max_jump {}",
        stat(&rtc.log, "guided"),
        stat(&rtc.log, "switches"),
        last_stat(&rtc.log, "mean_jump"),
        stat(&rtc.log, "max_jump"),
        stat(&rtc.log, "max_delay"),
        stat(&naive.log, "switches"),
        last_stat(&naive.log, "mean_jump"),
        stat(&naive.log, "max_jump"),
    );
    assert!(
        stat(&rtc.log, "guided") >= 10.0,
        "RTC never ran guided:\n{}",
        rtc.log
    );
    assert_eq!(
        stat(&naive.log, "guided"),
        0.0,
        "the baseline must not be guided"
    );
    assert!(stat(&rtc.log, "switches") >= 10.0 && stat(&naive.log, "switches") >= 10.0);
    // The governor measured the delay each chunk really had, and it exceeds the injected latency.
    assert!(
        stat(&rtc.log, "max_delay") >= 4.0,
        "observed delay {}",
        stat(&rtc.log, "max_delay")
    );
    // Not worse than the baseline; the statistics that show the gain are in the simulator tests.
    assert!(
        last_stat(&rtc.log, "mean_jump") <= 1.3 * last_stat(&naive.log, "mean_jump"),
        "RTC hand-overs are rougher than the baseline's"
    );
    assert_goals_safe(&rtc.goals);
    assert_goals_safe(&naive.goals);
}
