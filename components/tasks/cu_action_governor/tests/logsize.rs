//! Log volume per CopperList: chunk every 20 cycles vs every cycle (a latest-wins bridge
//! re-emitting the held chunk) vs no governor/chunk wiring at all.
#![allow(deprecated)]
use cu29::prelude::*;
use cu29_export::copperlists_reader;
use cu29_unifiedlog::memmap::{MmapSectionStorage, MmapUnifiedLoggerWrite};

macro_rules! variant {
    ($m:ident, $cfg:literal) => {
        mod $m {
            use super::*;
            #[copper_runtime(config = $cfg)]
            struct App {}
            pub fn run() -> (usize, usize, usize) {
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("l.copper");
                let app = App::builder()
                    .with_log_path(&path, Some(64 * 1024 * 1024))
                    .unwrap()
                    .build()
                    .unwrap();
                let mut running = app.start().unwrap();
                for _ in 0..1000 {
                    running.run_one_iteration().unwrap();
                }
                drop(running.stop().unwrap());
                let UnifiedLogger::Read(r) = UnifiedLoggerBuilder::new()
                    .file_base_name(&path)
                    .build()
                    .unwrap()
                else {
                    panic!()
                };
                let mut rd = UnifiedLoggerIOReader::new(r, UnifiedLogType::CopperList);
                let (mut n, mut bytes) = (0, 0);
                for cl in copperlists_reader::<default::CuStampedDataSet>(&mut rd) {
                    bytes += bincode::encode_to_vec(&cl, bincode::config::standard())
                        .unwrap()
                        .len();
                    n += 1;
                }
                (n, bytes, std::mem::size_of::<default::CuStampedDataSet>())
            }
        }
    };
}
variant!(gov, "tests/configs/gov.ron");
variant!(hot, "tests/configs/gov_hot.ron");
variant!(base, "tests/configs/base.ron");
variant!(base_hot, "tests/configs/base_hot.ron");

#[test]
fn report_log_bytes_per_copperlist() {
    let _ = std::any::type_name::<(MmapSectionStorage, MmapUnifiedLoggerWrite)>();
    for (name, (n, bytes, sz)) in [
        ("base (no governor, chunk not consumed)", base::run()),
        ("gov, chunk every 20 cycles", gov::run()),
        ("gov, chunk every cycle", hot::run()),
        ("base, chunk every cycle (no governor)", base_hot::run()),
    ] {
        println!(
            "logsize {name}: {n} CLs, {:.1} B/CL encoded, in-memory CuStampedDataSet {sz} B",
            bytes as f64 / n as f64
        );
    }
}
