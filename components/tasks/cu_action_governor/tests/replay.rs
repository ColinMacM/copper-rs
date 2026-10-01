// Record -> resim with task-state restore from keyframes. Drives the raw (deprecated)
// lifecycle API on purpose, exactly like examples/cu_caterpillar/src/resim.rs.
#![allow(deprecated)]

use cu29::prelude::*;
use std::sync::Mutex;

static SERIAL: Mutex<()> = Mutex::new(());

const CYCLES: u64 = 400;
const PERIOD_NS: u64 = 10_000_000;
/// Clock advance per sim-callback invocation while recording: tasks therefore run at
/// different instants inside one cycle, like on a real (non-mock) clock.
const JITTER_NS: u64 = 40_000;

/// (payload bits, status) of the governor output for one copperlist.
type Out = (Option<Vec<u32>>, String);

fn out_of(msg: &CuMsg<cu_action_governor::governor::JointPositions>) -> Out {
    (
        msg.payload()
            .map(|p| p.as_slice().iter().map(|f| f.to_bits()).collect()),
        msg.metadata.status_txt.0.to_string(),
    )
}

macro_rules! suite {
    ($m:ident, $cfg:literal) => {
        mod $m {
            use super::*;
            use cu29::prelude::app::CuSimApplication;
            use cu29_export::{copperlists_reader, keyframes_reader};
            use cu29_unifiedlog::memmap::{MmapSectionStorage, MmapUnifiedLoggerWrite};
            use default::SimStep::*;
            use std::path::Path;

            #[copper_runtime(config = $cfg, sim_mode = true)]
            struct App {}

            fn reader(base: &Path, ty: UnifiedLogType) -> UnifiedLoggerIOReader {
                let UnifiedLogger::Read(r) = UnifiedLoggerBuilder::new()
                    .file_base_name(base)
                    .build()
                    .unwrap()
                else {
                    panic!("no reader")
                };
                UnifiedLoggerIOReader::new(r, ty)
            }

            fn build(base: &Path) -> (App, RobotClockMock) {
                let (clock, mock) = RobotClock::mock();
                let mut noop = |_: default::SimStep| SimOverride::ExecuteByRuntime;
                let app = App::builder()
                    .with_clock(clock)
                    .with_log_path(base, Some(16 * 1024 * 1024))
                    .unwrap()
                    .with_sim_callback(&mut noop)
                    .build()
                    .unwrap()
                    .into_inner();
                (app, mock)
            }

            fn record(base: &Path) -> Vec<Out> {
                let (mut app, mock) = build(base);
                let m2 = mock.clone();
                let mut cb = move |_: default::SimStep| {
                    m2.increment(CuDuration(JITTER_NS));
                    SimOverride::ExecuteByRuntime
                };
                app.start_all_tasks(&mut cb).unwrap();
                for i in 0..CYCLES {
                    mock.set_value(1_000_000_000 + i * PERIOD_NS);
                    app.run_one_iteration(&mut cb).unwrap();
                }
                app.stop_all_tasks(&mut cb).unwrap();
                drop(app);
                let mut r = reader(base, UnifiedLogType::CopperList);
                copperlists_reader::<default::CuStampedDataSet>(&mut r)
                    .map(|cl| out_of(cl.msgs.get_gov_output_0()))
                    .collect()
            }

            /// Replays recorded copperlists; sources are injected, the governor re-executes.
            fn replay(
                rec: &Path,
                dst: &Path,
                from_keyframe: Option<u64>,
                restore: bool,
            ) -> Vec<(u64, Out)> {
                let (mut app, mock) = build(dst);
                let got = std::cell::RefCell::new(Vec::new());
                let mut noop = |_: default::SimStep| SimOverride::ExecuteByRuntime;
                app.start_all_tasks(&mut noop).unwrap();
                let mut kr = reader(rec, UnifiedLogType::FrozenTasks);
                let kfs: Vec<KeyFrame> = keyframes_reader(&mut kr).collect();
                let start = from_keyframe.map(|i| &kfs[i as usize]);
                let mut r = reader(rec, UnifiedLogType::CopperList);
                for cl in copperlists_reader::<default::CuStampedDataSet>(&mut r) {
                    if let Some(kf) = start {
                        if cl.id < kf.culistid {
                            continue;
                        }
                        if cl.id == kf.culistid {
                            if restore {
                                <App as CuSimApplication<
                                    MmapSectionStorage,
                                    MmapUnifiedLoggerWrite,
                                >>::restore_keyframe(&mut app, kf)
                                .unwrap();
                            }
                        }
                    }
                    let t = cl
                        .msgs
                        .get_stamp_output()
                        .metadata
                        .process_time
                        .start
                        .unwrap()
                        .as_nanos();
                    mock.set_value(t);
                    let id = cl.id;
                    let msgs = cl.msgs;
                    let mut cb = |step: default::SimStep| -> SimOverride {
                        use CuTaskCallbackState::*;
                        match step {
                            Stamp(Process(_, o)) => {
                                *o = msgs.get_stamp_output().clone();
                                SimOverride::ExecutedBySim
                            }
                            Fb(Process(_, o)) => {
                                *o = msgs.get_fb_output().clone();
                                SimOverride::ExecutedBySim
                            }
                            Chunk(Process(_, o)) => {
                                *o = msgs.get_chunk_output().clone();
                                SimOverride::ExecutedBySim
                            }
                            Sink(Process(input, _)) => {
                                got.borrow_mut().push((id, out_of(input)));
                                SimOverride::ExecutedBySim
                            }
                            _ => SimOverride::ExecuteByRuntime,
                        }
                    };
                    app.run_one_iteration(&mut cb).unwrap();
                }
                app.stop_all_tasks(&mut noop).unwrap();
                got.into_inner()
            }

            pub fn run() -> (usize, usize, usize, usize) {
                let _g = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
                let dir = tempfile::tempdir().unwrap();
                let rec = dir.path().join("rec.copper");
                let expected = record(&rec);
                assert_eq!(expected.len() as u64, CYCLES);
                let hold = expected.iter().filter(|o| o.1 != "play").count();

                let mut kr = reader(&rec, UnifiedLogType::FrozenTasks);
                let nkf = keyframes_reader(&mut kr).count();
                assert!(nkf >= 10, "keyframes: {nkf}");

                // (a) from the start: every output identical.
                let full = replay(&rec, &dir.path().join("r0.copper"), None, false);
                let mism_full = full.iter().filter(|(id, o)| expected[*id as usize] != *o).count();
                // (b) from each keyframe with its frozen state restored.
                let mut mism_kf = 0;
                let mut mism_norestore = 0;
                for k in [1u64, 5, 17, (nkf - 1) as u64] {
                    let d = dir.path().join(format!("r{k}.copper"));
                    let got = replay(&rec, &d, Some(k), true);
                    assert!(!got.is_empty());
                    mism_kf += got.iter().filter(|(id, o)| expected[*id as usize] != *o).count();
                    let d = dir.path().join(format!("n{k}.copper"));
                    let got = replay(&rec, &d, Some(k), false);
                    mism_norestore += got.iter().filter(|(id, o)| expected[*id as usize] != *o).count();
                }
                println!(
                    "[{}] cycles={CYCLES} non-play={hold} keyframes={nkf} mismatches: full={mism_full} from-keyframe(restore)={mism_kf} from-keyframe(NO restore)={mism_norestore}",
                    stringify!($m)
                );
                (mism_full, mism_kf, mism_norestore, hold)
            }
        }
    };
}

suite!(feedback_clock, "tests/configs/replay.ron");
suite!(ctx_clock, "tests/configs/replay_ctx.ron");

#[test]
fn governor_replay_is_exact_with_feedback_time_base() {
    let (full, kf, norestore, hold) = feedback_clock::run();
    assert_eq!(full, 0);
    assert_eq!(kf, 0);
    assert!(
        norestore > 0,
        "test has no teeth: state restore is not needed?"
    );
    assert!(hold > 0);
}

#[test]
fn governor_replay_with_ctx_clock_reports_divergence() {
    // Documents the pitfall: ctx.now() differs between a live run and replay.
    let (full, kf, _norestore, _hold) = ctx_clock::run();
    println!("ctx clock: full={full} kf={kf}");
}
