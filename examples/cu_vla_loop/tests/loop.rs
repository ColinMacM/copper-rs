//! The loop against a real Python policy process over real Zenoh (loopback), driving the
//! governor and a mock arm. Skipped when `import zenoh` fails in `python3`.

use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

use cu_vla_loop::{LAST_STATUS, TRACE, listen_config, run_hooked};

static SERIAL: Mutex<()> = Mutex::new(());

/// Goals, positions and the policy's output of one scenario.
type Run = (Vec<Option<[f32; 8]>>, Vec<[f32; 8]>, Option<String>);

const HZ: f64 = 30.0;
const MIN: f32 = 200.0;
const MAX: f32 = 3900.0;
const MAX_STEP: f32 = 30.0;
const START: f32 = 2048.0;

fn python_has_zenoh() -> bool {
    Command::new("python3")
        .args(["-c", "import zenoh"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

fn free_port() -> u16 {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    l.local_addr().unwrap().port()
}

fn spawn_policy(port: u16, seconds: f64, extra: &[&str]) -> Child {
    Command::new("python3")
        .args([
            "-m",
            "copper_policy",
            "--connect-port",
            &port.to_string(),
            "--seconds",
            &seconds.to_string(),
        ])
        .args(extra)
        .current_dir(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../components/tasks/cu_policy/python"
        ))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("python3")
}

/// Runs `cycles` cycles; `policy_args` of None runs with no policy process at all.
fn scenario(cycles: usize, policy_args: Option<&[&str]>) -> Run {
    let port = free_port();
    let dir = tempfile::tempdir().unwrap();
    let seconds = cycles as f64 / HZ + 3.0;
    let mut child = None;
    let args: Option<Vec<String>> =
        policy_args.map(|a| a.iter().map(|s| (*s).to_owned()).collect());
    run_hooked(
        cycles,
        HZ,
        &dir.path().join("loop.copper"),
        &listen_config(port),
        |i| {
            if i == 0
                && let Some(a) = &args
            {
                let refs: Vec<&str> = a.iter().map(String::as_str).collect();
                child = Some(spawn_policy(port, seconds, &refs));
            }
        },
        |_| {},
    )
    .expect("the loop must run to completion whatever the policy does");
    let output = child.map(|mut c| {
        let _ = c.kill();
        let out = c.wait_with_output().unwrap();
        String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr)
    });
    let t = TRACE.lock().unwrap();
    (t.goals.clone(), t.positions.clone(), output)
}

/// The largest value the policy runner printed for `key` in its progress lines and summary.
fn stat(out: &str, key: &str) -> u64 {
    let needle = format!("\"{key}\": ");
    out.lines()
        .flat_map(|l| {
            l.split(&needle)
                .skip(1)
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .filter_map(|r| {
            let n: String = r.chars().take_while(char::is_ascii_digit).collect();
            n.parse::<u64>().ok()
        })
        .max()
        .unwrap_or(0)
}

fn joints(v: &[f32; 8]) -> &[f32] {
    &v[..6]
}

fn assert_goals_safe(goals: &[Option<[f32; 8]>]) {
    let mut prev: Option<[f32; 8]> = None;
    for (i, g) in goals.iter().enumerate() {
        let Some(g) = g else { continue };
        for (j, v) in joints(g).iter().enumerate() {
            assert!(v.is_finite(), "goal {i} joint {j} is {v}");
            assert!(
                (MIN..=MAX).contains(v),
                "goal {i} joint {j} = {v} outside [{MIN}, {MAX}]"
            );
        }
        if let Some(p) = prev {
            for j in 0..6 {
                assert!(
                    (g[j] - p[j]).abs() <= MAX_STEP + 1e-3,
                    "goal {i} joint {j} jumped {} -> {}",
                    p[j],
                    g[j]
                );
            }
        }
        prev = Some(*g);
    }
}

fn excursion(positions: &[[f32; 8]]) -> f32 {
    let lo = positions.iter().map(|p| p[0]).fold(f32::MAX, f32::min);
    let hi = positions.iter().map(|p| p[0]).fold(f32::MIN, f32::max);
    hi - lo
}

macro_rules! need_python {
    () => {
        if !python_has_zenoh() {
            eprintln!("skipped: python3 cannot import zenoh");
            return;
        }
    };
}

#[test]
fn a_live_policy_moves_the_arm_and_every_goal_stays_inside_the_limits() {
    need_python!();
    let _g = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (goals, positions, out) = scenario(300, Some(&[]));
    let out = out.unwrap();
    println!("policy: {out}");
    let chunks = stat(&out, "chunks");
    assert!(
        chunks > 40,
        "the ACT policy produced only {chunks} chunks; it may not have run at all:\n{out}"
    );
    assert_goals_safe(&goals);
    assert!(goals.iter().filter(|g| g.is_some()).count() > 250);
    assert!(
        excursion(&positions) > 150.0,
        "the arm barely moved: {}",
        excursion(&positions)
    );
}

#[test]
fn a_target_beyond_the_joint_limits_is_clamped() {
    need_python!();
    let _g = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (goals, positions, _) = scenario(240, Some(&["--target", "9000"]));
    assert_goals_safe(&goals);
    let top = goals
        .iter()
        .flatten()
        .map(|g| g[0])
        .fold(f32::MIN, f32::max);
    assert!(
        top > 2200.0,
        "the arm should have been driven toward the limit, got {top}"
    );
    assert!(
        positions.iter().all(|p| p[..6].iter().all(|v| *v <= MAX)),
        "the arm passed the limit"
    );
}

#[test]
fn nan_in_a_chunk_is_rejected_and_never_reaches_the_servo_as_a_zero_goal() {
    need_python!();
    let _g = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (goals, positions, _) = scenario(300, Some(&["--inject", "nan"]));
    assert_goals_safe(&goals);
    let low = positions
        .iter()
        .flat_map(|p| p[..6].iter().copied())
        .fold(f32::MAX, f32::min);
    assert!(
        low > 1000.0,
        "a NaN goal drove the arm toward raw 0 (min position {low})"
    );
}

#[test]
fn when_the_policy_process_dies_the_arm_holds_its_last_goal() {
    need_python!();
    let _g = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (goals, positions, _) = scenario(300, Some(&["--kill-after", "3.5"]));
    assert_goals_safe(&goals);
    let tail: Vec<[f32; 8]> = goals.iter().rev().take(90).flatten().copied().collect();
    assert!(tail.len() >= 80);
    assert!(
        tail.iter().all(|g| g[..6] == tail[0][..6]),
        "goals kept changing after the policy died"
    );
    let last: Vec<&[f32; 8]> = positions.iter().rev().take(30).collect();
    let drift = last
        .iter()
        .map(|p| (p[0] - last[0][0]).abs())
        .fold(0.0, f32::max);
    assert!(
        drift < 1.0,
        "the arm kept moving after the policy died ({drift})"
    );
    assert!(
        excursion(&positions) > 100.0,
        "the policy should have moved the arm before it died"
    );
}

#[test]
fn chunks_older_than_the_age_limit_are_not_executed() {
    // 50-step chunks, 1 s in flight: about 30 steps would be skipped and 20 played if the chunk
    // were accepted, so any motion means the age limit did not hold.
    need_python!();
    let _g = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (goals, positions, _) = scenario(240, Some(&["--delay-s", "1.0", "--steps", "50"]));
    assert_goals_safe(&goals);
    assert!(
        excursion(&positions) < 60.0,
        "stale chunks moved the arm by {}",
        excursion(&positions)
    );
}

#[test]
fn camera_frames_reach_the_policy_whole_and_stamped() {
    need_python!();
    let _g = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (_, _, out) = scenario(240, Some(&[]));
    let out = out.unwrap();
    let frames = stat(&out, "frames");
    assert!(frames > 50, "only {frames} frames arrived:\n{out}");
    assert_eq!(
        stat(&out, "bad_frames"),
        0,
        "a frame arrived truncated or with wrong pixels:\n{out}"
    );
    assert!(
        stat(&out, "last_frame_tov") > 0,
        "frames carry no time of validity"
    );
}

#[test]
fn link_counters_reach_the_graph_and_show_undecodable_messages() {
    need_python!();
    let _g = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *LAST_STATUS.lock().unwrap() = None;
    let (goals, _, out) = scenario(240, Some(&["--inject", "garbage"]));
    let status = LAST_STATUS
        .lock()
        .unwrap()
        .expect("no status message reached the graph");
    println!("status: {status:?}\n{}", out.unwrap());
    assert!(status.session_up && status.tx_published > 50 && status.img_published > 50);
    assert!(status.rx_received > 50);
    assert!(
        status.rx_decode_errors > 0,
        "garbage on the action route was not counted: {status:?}"
    );
    assert_goals_safe(&goals);
}

#[test]
fn with_no_policy_at_all_the_loop_runs_and_the_arm_stays_put() {
    let _g = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (goals, positions, _) = scenario(120, None);
    assert_goals_safe(&goals);
    assert!(
        goals
            .iter()
            .flatten()
            .all(|g| joints(g).iter().all(|v| (*v - START).abs() < 1.0))
    );
    assert!(excursion(&positions) < 1.0);
}

/// A real LeRobot ACT policy (random weights) through the same loop. Its output is meaningless
/// but it must stay safe and keep the loop alive; needs `lerobot` importable.
#[test]
fn a_lerobot_act_policy_drives_the_loop_inside_the_limits() {
    let ok = Command::new("python3")
        .args(["-c", "import zenoh, lerobot"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success());
    if !ok {
        eprintln!("skipped: python3 cannot import zenoh and lerobot");
        return;
    }
    let _g = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let dir = tempfile::tempdir().unwrap();
    let cal = dir.path().join("cal.json");
    let joint = |id: u32| {
        format!(
            r#"{{"id": {id}, "drive_mode": 0, "homing_offset": 0, "range_min": 200, "range_max": 3900}}"#
        )
    };
    let names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ];
    let body: Vec<String> = names
        .iter()
        .enumerate()
        .map(|(i, n)| format!(r#""{n}": {}"#, joint(i as u32 + 1)))
        .collect();
    std::fs::write(&cal, format!("{{{}}}", body.join(", "))).unwrap();
    let (goals, positions, out) = scenario(
        300,
        Some(&["--policy", "act", "--calibration", cal.to_str().unwrap()]),
    );
    println!("policy: {}", out.unwrap());
    assert_goals_safe(&goals);
    assert!(goals.iter().filter(|g| g.is_some()).count() > 250);
    assert!(
        positions
            .iter()
            .all(|p| p[..6].iter().all(|v| (MIN..=MAX).contains(v)))
    );
}
