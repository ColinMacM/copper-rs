//! A Copper application whose graph contains an instance of the `smoothing` plugin.
//!
//! `copperconfig.ron` declares the plugin; Copper expands it while the configuration is read,
//! so the generated runtime contains the plugin's tasks like any others.

use cu_demo_smoothing::{Sample, Smoothed};
use cu29::prelude::*;
use std::path::Path;
use std::sync::{LazyLock, Mutex};

static COLLECTED: LazyLock<Mutex<Vec<Smoothed>>> = LazyLock::new(|| Mutex::new(Vec::new()));

/// Smoothed values received by the sink since the last [`reset`].
pub fn collected() -> Vec<Smoothed> {
    COLLECTED.lock().unwrap().clone()
}

/// Forget the values received so far.
pub fn reset() {
    COLLECTED.lock().unwrap().clear();
}

pub mod tasks {
    use super::*;

    /// Emits 1.0, 2.0, 3.0, ...
    #[derive(Reflect)]
    pub struct Ticker {
        next: f64,
    }

    impl Freezable for Ticker {
        fn freeze<E: bincode::enc::Encoder>(
            &self,
            encoder: &mut E,
        ) -> Result<(), bincode::error::EncodeError> {
            bincode::Encode::encode(&self.next, encoder)
        }

        fn thaw<D: bincode::de::Decoder>(
            &mut self,
            decoder: &mut D,
        ) -> Result<(), bincode::error::DecodeError> {
            self.next = bincode::Decode::decode(decoder)?;
            Ok(())
        }
    }

    impl CuSrcTask for Ticker {
        type Resources<'r> = ();
        type Output<'m> = output_msg!(Sample);

        fn new(
            _config: Option<&ComponentConfig>,
            _resources: Self::Resources<'_>,
        ) -> CuResult<Self> {
            Ok(Self { next: 1.0 })
        }

        fn process(&mut self, _ctx: &CuContext, output: &mut Self::Output<'_>) -> CuResult<()> {
            output.set_payload(Sample { value: self.next });
            self.next += 1.0;
            Ok(())
        }
    }

    /// Stores what it receives so a test or the demo binary can print it.
    #[derive(Default, Reflect)]
    pub struct Collector;

    impl Freezable for Collector {}

    impl CuSinkTask for Collector {
        type Resources<'r> = ();
        type Input<'m> = input_msg!(Smoothed);

        fn new(
            _config: Option<&ComponentConfig>,
            _resources: Self::Resources<'_>,
        ) -> CuResult<Self> {
            Ok(Self)
        }

        fn process(&mut self, _ctx: &CuContext, input: &Self::Input<'_>) -> CuResult<()> {
            if let Some(value) = input.payload() {
                COLLECTED.lock().unwrap().push(value.clone());
            }
            Ok(())
        }
    }
}

#[copper_runtime(config = "copperconfig.ron")]
struct PluginDemoApp {}

/// The configuration the runtime was generated from, with the plugin expanded.
pub fn resolved_config() -> String {
    <PluginDemoApp as CuApplication<memmap::MmapSectionStorage, UnifiedLoggerWrite>>::get_original_config()
}

/// Run the application for `iterations` cycles, logging to `log_path`.
pub fn run(iterations: usize, log_path: &Path) -> CuResult<()> {
    let app = PluginDemoApp::builder()
        .with_log_path(log_path, Some(16 * 1024 * 1024))?
        .build()?;
    reset();
    let mut running = app.start()?;
    for _ in 0..iterations {
        running.run_one_iteration()?;
    }
    running.stop()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{LazyLock, Mutex};

    // The sink stores into a process-wide list, so the tests take turns.
    static SERIAL: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

    fn run_once(dir: &Path, name: &str) -> Vec<Smoothed> {
        run(6, &dir.join(name)).expect("run");
        collected()
    }

    #[test]
    fn the_plugin_nodes_compute_inside_the_generated_runtime() {
        let _guard = SERIAL.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let values = run_once(dir.path(), "a.copper");
        // Ticker emits 1..6; window 3, gain 2.0: moving average of the last three samples, doubled.
        let expected = [(2.0, 1), (3.0, 2), (4.0, 3), (6.0, 3), (8.0, 3), (10.0, 3)];
        let got: Vec<(f64, u32)> = values.iter().map(|s| (s.value, s.filled)).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn the_resolved_configuration_names_the_plugin_instance() {
        let resolved = resolved_config();
        for needle in [
            "resolved_plugins",
            "cu-demo-smoothing",
            "instance: \"smooth\"",
            "smooth_average",
            "smooth_limit",
            "blake3:4de452b52939ab224907809c519dec97d0c58e935c7e66ac950c95d5dd5d8b59",
        ] {
            assert!(resolved.contains(needle), "{needle:?} not in:\n{resolved}");
        }
    }

    #[test]
    fn the_unified_log_records_which_plugin_content_ran() {
        let _guard = SERIAL.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        run(3, &dir.path().join("b.copper")).expect("run");
        let logged = std::fs::read(dir.path().join("b_0.copper"))
            .or_else(|_| std::fs::read(dir.path().join("b.copper")))
            .expect("log file");
        let pin = b"blake3:4de452b52939ab224907809c519dec97d0c58e935c7e66ac950c95d5dd5d8b59";
        assert!(
            logged.windows(pin.len()).any(|w| w == pin),
            "the plugin pin is not in the unified log"
        );
    }

    #[test]
    fn two_runs_produce_identical_output() {
        let _guard = SERIAL.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let first = run_once(dir.path(), "c.copper");
        let second = run_once(dir.path(), "d.copper");
        assert_eq!(first, second);
    }
}
