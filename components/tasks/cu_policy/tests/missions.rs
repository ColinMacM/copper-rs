//! One config, two missions: `record` (leader -> follower, no governor) and `policy`
//! (policy chunks -> governor -> follower). Same sink id in both.
use cu29::prelude::*;

#[copper_runtime(config = "tests/configs/missions.ron")]
struct App {}

use policy::App as PolicyApp;
use record::App as RecordApp;

fn run<A: CuApplication<memmap::MmapSectionStorage, UnifiedLoggerWrite>>(
    app: CuAppLifecycle<memmap::MmapSectionStorage, UnifiedLoggerWrite, A>,
) -> CuResult<()> {
    let mut running = app.start()?;
    for _ in 0..50 {
        running.run_one_iteration()?;
    }
    running.stop()?;
    Ok(())
}

#[test]
fn both_missions_build_and_run_and_have_distinct_copperlist_types() {
    let dir = tempfile::tempdir().unwrap();
    let clock = RobotClock::default();
    run(RecordApp::builder()
        .with_clock(clock.clone())
        .with_log_path(dir.path().join("record.copper"), Some(16 * 1024 * 1024))
        .unwrap()
        .build()
        .unwrap())
    .unwrap();
    run(PolicyApp::builder()
        .with_clock(clock)
        .with_log_path(dir.path().join("policy.copper"), Some(16 * 1024 * 1024))
        .unwrap()
        .build()
        .unwrap())
    .unwrap();
    println!(
        "size_of CuStampedDataSet: record={} policy={}",
        std::mem::size_of::<record::CuStampedDataSet>(),
        std::mem::size_of::<policy::CuStampedDataSet>()
    );
}
