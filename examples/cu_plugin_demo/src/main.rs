use cu_plugin_demo::{collected, run};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let logs = std::path::Path::new("logs");
    std::fs::create_dir_all(logs)?;
    run(6, &logs.join("plugin_demo.copper"))?;
    for (cycle, value) in collected().iter().enumerate() {
        println!(
            "cycle {cycle}: {:.3} (window filled: {})",
            value.value, value.filled
        );
    }
    Ok(())
}
