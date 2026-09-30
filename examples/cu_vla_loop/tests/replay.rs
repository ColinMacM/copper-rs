//! Record a live run (real Python policy over Zenoh), then replay the log with no Zenoh session:
//! the recorded bridge outputs are injected, the governor re-executes, and its output must equal
//! the recorded one cycle for cycle.
#![allow(deprecated)]

use std::path::Path;
use std::process::{Command, Stdio};

use cu_vla_loop::{bridges, listen_config, run_hooked, tasks};
use cu29::prelude::*;
use cu29::simulation::CuBridgeLifecycleState;
use cu29_export::copperlists_reader;
use cu29_unifiedlog::{UnifiedLogger, UnifiedLoggerBuilder, UnifiedLoggerIOReader};

#[copper_runtime(config = "copperconfig.ron", sim_mode = true)]
struct Replay {}

/// Governor output of one CopperList: goal bits and status.
type Out = (Option<Vec<u32>>, String);

fn read_outputs(base: &Path) -> Vec<Out> {
    let UnifiedLogger::Read(r) = UnifiedLoggerBuilder::new()
        .file_base_name(base)
        .build()
        .unwrap()
    else {
        panic!("not a readable log");
    };
    let mut reader = UnifiedLoggerIOReader::new(r, UnifiedLogType::CopperList);
    copperlists_reader::<default::CuStampedDataSet>(&mut reader)
        .map(|cl| {
            let m = cl.msgs.get_gov_output();
            (
                m.payload()
                    .map(|p| p.as_slice().iter().map(|f| f.to_bits()).collect()),
                m.metadata.status_txt.0.to_string(),
            )
        })
        .collect()
}

fn record(dir: &Path) -> usize {
    let port = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().port()
    };
    let mut child = None;
    run_hooked(
        240,
        30.0,
        &dir.join("rec.copper"),
        &listen_config(port),
        |i| {
            if i == 0 {
                child = Some(
                    Command::new("python3")
                        .args([
                            "-m",
                            "vla_runner",
                            "--connect-port",
                            &port.to_string(),
                            "--seconds",
                            "12",
                        ])
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
    240
}

#[test]
fn a_recorded_policy_run_replays_identically_without_a_zenoh_session() {
    let ok = Command::new("python3")
        .args(["-c", "import zenoh"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success());
    if !ok {
        eprintln!("skipped: python3 cannot import zenoh");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    record(dir.path());
    let recorded = read_outputs(&dir.path().join("rec.copper"));
    assert!(
        recorded.len() >= 200,
        "recorded only {} cycles",
        recorded.len()
    );
    let moving = recorded
        .iter()
        .filter(|(g, s)| g.is_some() && s == "play")
        .count();
    assert!(
        moving > 50,
        "the policy drove only {moving} cycles, so the replay would prove little"
    );

    // Replay with a valid listen endpoint that we then probe: if the bridge were started, a
    // session would be listening on it.
    let probe_port = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().port()
    };
    let (clock, mock) = RobotClock::mock();
    let mut config = CuConfig::deserialize_ron(&Replay::original_config()).unwrap();
    let link = config.bridges.iter_mut().find(|b| b.id == "link").unwrap();
    link.config
        .get_or_insert_with(ComponentConfig::default)
        .set("zenoh_config_json", listen_config(probe_port));
    // Replay never touches the network: the link bridge's lifecycle (construction, start, stop)
    // is handled by the simulation, everything else by the runtime.
    let mut default_cb = |s: default::SimStep<'_>| match s {
        default::SimStep::LinkBridge(
            CuBridgeLifecycleState::Start | CuBridgeLifecycleState::Stop,
        ) => SimOverride::ExecutedBySim,
        _ => SimOverride::ExecuteByRuntime,
    };
    let mut app = Replay::builder()
        .with_clock(clock)
        .with_config(config)
        .with_log_path(dir.path().join("replay.copper"), Some(64 * 1024 * 1024))
        .unwrap()
        .with_sim_callback(&mut default_cb)
        .build()
        .unwrap()
        .into_inner();
    app.start_all_tasks(&mut default_cb).unwrap();
    let UnifiedLogger::Read(r) = UnifiedLoggerBuilder::new()
        .file_base_name(&dir.path().join("rec.copper"))
        .build()
        .unwrap()
    else {
        panic!("not a readable log");
    };
    let mut reader = UnifiedLoggerIOReader::new(r, UnifiedLogType::CopperList);
    for (n, entry) in copperlists_reader::<default::CuStampedDataSet>(&mut reader).enumerate() {
        if n == 100 {
            std::thread::sleep(std::time::Duration::from_millis(500)); // time for a worker to bind, if one existed
            assert!(
                std::net::TcpStream::connect(("127.0.0.1", probe_port)).is_err(),
                "a Zenoh session is listening during replay"
            );
        }
        if let Tov::Time(t) = entry.msgs.get_arm_rx_positions().tov {
            mock.set_value(t.as_nanos());
        }
        let mut cb = |s: default::SimStep<'_>| match s {
            default::SimStep::LinkBridge(CuBridgeLifecycleState::Start) => {
                SimOverride::ExecutedBySim
            }
            other => default::recorded_replay_step(other, &entry),
        };
        if let Err(e) = app.run_one_iteration(&mut cb) {
            panic!("replay failed at cycle {n} (cl {}): {e}", entry.id);
        }
    }
    app.stop_all_tasks(&mut default_cb).unwrap();
    drop(app); // flushes the replay's own log
    let replayed = read_outputs(&dir.path().join("replay.copper"));

    assert_eq!(replayed.len(), recorded.len());
    let mismatches = recorded
        .iter()
        .zip(&replayed)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        mismatches,
        0,
        "{mismatches} of {} cycles differ on replay",
        recorded.len()
    );
    // Control: the comparison can fail. Shifted by one cycle it must not match everywhere.
    let shifted = recorded
        .iter()
        .skip(1)
        .zip(&replayed)
        .filter(|(a, b)| a == b)
        .count();
    assert!(
        shifted < recorded.len() - 1,
        "the comparison cannot tell cycles apart"
    );
    println!(
        "replayed {} cycles identically; {moving} were policy-driven; shifted matches: {shifted}",
        recorded.len()
    );
}
