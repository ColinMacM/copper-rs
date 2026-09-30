use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use cu_plugin::{check, default_parent, describe, expand_config, new_plugin, parse_override, pin};

#[derive(Parser)]
#[command(
    name = "cu-plugin",
    about = "Create, inspect, check and pin Copper static plugins"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Create a plugin directory with a manifest and a fragment to start from
    New {
        name: String,
        /// Directory that will contain the new plugin directory
        #[arg(long)]
        dir: Option<PathBuf>,
    },
    /// Show a plugin's parameters, fragments, public nodes, assets and pin
    Describe { path: PathBuf },
    /// Validate a plugin by rendering every fragment with default or sample values
    Check {
        path: PathBuf,
        /// Parameter value as key=value (repeatable)
        #[arg(long = "param")]
        params: Vec<String>,
        /// Application Cargo.toml: confirm it depends on the crates the manifest lists
        #[arg(long)]
        app: Option<PathBuf>,
    },
    /// Print the content pin for an application's `plugins` entry
    Pin { path: PathBuf },
    /// Print the resolved configuration of an application
    Expand {
        config: PathBuf,
        /// Cargo features used to evaluate `when` predicates (comma separated)
        #[arg(long, value_delimiter = ',')]
        features: Vec<String>,
        /// Print nodes, connections and plugin instances instead of the full RON
        #[arg(long)]
        summary: bool,
    },
}

fn run(cli: Cli) -> Result<String, String> {
    match cli.command {
        Command::New { name, dir } => new_plugin(&dir.unwrap_or_else(default_parent), &name),
        Command::Describe { path } => describe(&path),
        Command::Check { path, params, app } => {
            let overrides = params
                .iter()
                .map(|p| parse_override(p))
                .collect::<Result<Vec<_>, _>>()?;
            check(&path, &overrides, app.as_deref())
        }
        Command::Pin { path } => pin(&path),
        Command::Expand {
            config,
            features,
            summary,
        } => {
            let features: Vec<&str> = features.iter().map(String::as_str).collect();
            expand_config(&config, &features, summary)
        }
    }
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(text) => {
            println!("{}", text.trim_end());
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}
