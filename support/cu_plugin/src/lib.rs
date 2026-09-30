//! Commands of the `cu-plugin` tool. Each returns the text to print.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use cu29_plugin::{
    Host, Manifest, ParamKind, ParamSpec, ParamValue, PluginUse, content_pin, expand,
};

/// Version of the Copper this tool was built with; plugins are checked against it.
pub const COPPER_VERSION: &str = env!("CARGO_PKG_VERSION");

pub type Result<T> = std::result::Result<T, String>;

fn host() -> Host<'static> {
    Host {
        copper_version: COPPER_VERSION,
    }
}

fn load(dir: &Path) -> Result<Manifest> {
    Manifest::load(dir).map_err(|e| e.to_string())
}

/// Text that is a valid plugin id (`[a-z][a-z0-9-]*`) once underscores become dashes.
fn plugin_id(name: &str) -> Result<String> {
    let id = name.replace('_', "-");
    let mut chars = id.chars();
    if chars.next().is_some_and(|c| c.is_ascii_lowercase())
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
    {
        Ok(id)
    } else {
        Err(format!("plugin name {name:?} must match [a-z][a-z0-9_-]*"))
    }
}

/// Creates `<parent>/<name>/` with a manifest and one fragment to start from.
pub fn new_plugin(parent: &Path, name: &str) -> Result<String> {
    let id = plugin_id(name)?;
    let dir = parent.join(name);
    if dir.exists() {
        return Err(format!("{} already exists", dir.display()));
    }
    std::fs::create_dir_all(dir.join("fragments"))
        .map_err(|e| format!("{}: {e}", dir.display()))?;
    let manifest = format!(
        r#"(
    format: 1,
    id: "{id}",
    version: "0.1.0",
    description: "Describe what this plugin provides",
    copper: ">={COPPER_VERSION}",
    rust: [],
    params: {{
        "label": (kind: Str, default: "{name}"),
    }},
    fragments: {{
        "main": (path: "fragments/main.ron", public: ["node"]),
    }},
    assets: [],
)
"#
    );
    let fragment = r#"(
    tasks: [
        (
            id: "{{instance}}_node",
            type: "replace_with::YourTask",
            config: {"label": "{{label}}"},
        ),
    ],
    cnx: [],
)
"#;
    std::fs::write(dir.join("plugin.ron"), manifest).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("fragments/main.ron"), fragment).map_err(|e| e.to_string())?;
    Ok(format!(
        "created {}\nnext: edit fragments/main.ron, then `cu-plugin check {}` and `cu-plugin pin {}`",
        dir.display(),
        dir.display(),
        dir.display()
    ))
}

fn kind_name(kind: &ParamKind) -> String {
    match kind {
        ParamKind::Bool => "Bool".into(),
        ParamKind::Int => "Int".into(),
        ParamKind::Float => "Float".into(),
        ParamKind::Str => "Str".into(),
        ParamKind::Enum(choices) => format!("Enum({})", choices.join(" | ")),
    }
}

fn range_text(spec: &ParamSpec) -> String {
    match (spec.min, spec.max) {
        (None, None) => String::new(),
        (min, max) => format!(
            " [{}..{}]",
            min.map_or(String::new(), |v| v.to_string()),
            max.map_or(String::new(), |v| v.to_string())
        ),
    }
}

/// Human-readable summary of a plugin.
pub fn describe(dir: &Path) -> Result<String> {
    let m = load(dir)?;
    let pin = content_pin(dir, &m).map_err(|e| e.to_string())?;
    let mut out = String::new();
    let _ = writeln!(out, "{} {} - {}", m.id, m.version, m.description);
    let _ = writeln!(out, "copper: {}", m.copper);
    for dep in &m.rust {
        let _ = writeln!(out, "rust: {} {}", dep.package, dep.version);
    }
    let _ = writeln!(out, "pin: {pin}");
    let _ = writeln!(out, "parameters:");
    if m.params.is_empty() {
        let _ = writeln!(out, "  (none)");
    }
    for (name, spec) in &m.params {
        let default = spec
            .default
            .as_ref()
            .map_or("required".to_owned(), |d| format!("default {d}"));
        let _ = writeln!(
            out,
            "  {name}: {}{} - {default}",
            kind_name(&spec.kind),
            range_text(spec)
        );
    }
    let _ = writeln!(out, "fragments:");
    for (name, f) in &m.fragments {
        let public = if f.public.is_empty() {
            "no public nodes".to_owned()
        } else {
            format!(
                "public nodes: {}",
                f.public
                    .iter()
                    .map(|p| format!("<instance>_{p}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        let _ = writeln!(out, "  {name} ({}) - {public}", f.path);
    }
    if !m.assets.is_empty() {
        let _ = writeln!(out, "assets: {}", m.assets.join(", "));
    }
    Ok(out)
}

/// The content pin to put in an application's `plugins` entry.
pub fn pin(dir: &Path) -> Result<String> {
    let m = load(dir)?;
    content_pin(dir, &m).map_err(|e| e.to_string())
}

/// `key=value` into a parameter value: a bool, integer or float when it reads as one, else text.
pub fn parse_override(text: &str) -> Result<(String, ParamValue)> {
    let (key, value) = text
        .split_once('=')
        .ok_or_else(|| format!("parameter override {text:?} must be key=value"))?;
    let parsed = if let Ok(b) = value.parse::<bool>() {
        ParamValue::Bool(b)
    } else if let Ok(i) = value.parse::<i64>() {
        ParamValue::Int(i)
    } else if let Ok(f) = value.parse::<f64>() {
        ParamValue::Float(f)
    } else {
        ParamValue::Str(value.to_owned())
    };
    Ok((key.to_owned(), parsed))
}

/// A number inside the spec's bounds: the minimum if there is one, else the maximum when it is
/// negative, else zero.
fn in_range(spec: &ParamSpec) -> f64 {
    spec.min
        .or(spec.max.filter(|max| *max < 0.0))
        .unwrap_or(0.0)
}

/// A value that satisfies `spec`, used to render fragments during `check`.
fn sample(spec: &ParamSpec) -> ParamValue {
    if let Some(default) = &spec.default {
        return default.clone();
    }
    match &spec.kind {
        ParamKind::Bool => ParamValue::Bool(false),
        ParamKind::Int =>
        {
            #[allow(clippy::cast_possible_truncation)]
            ParamValue::Int(in_range(spec) as i64)
        }
        ParamKind::Float => ParamValue::Float(in_range(spec)),
        ParamKind::Str => ParamValue::Str("sample".into()),
        ParamKind::Enum(choices) => ParamValue::Str(choices[0].clone()),
    }
}

/// Validates a plugin by rendering every fragment with default or sample parameter values.
///
/// Returns the report; an `Err` carries every problem found.
pub fn check(
    dir: &Path,
    overrides: &[(String, ParamValue)],
    app_manifest: Option<&Path>,
) -> Result<String> {
    let manifest = load(dir)?;
    let absolute = std::fs::canonicalize(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut params: BTreeMap<String, ParamValue> = manifest
        .params
        .iter()
        .filter(|(_, spec)| spec.default.is_none())
        .map(|(name, spec)| (name.clone(), sample(spec)))
        .collect();
    params.extend(overrides.iter().cloned());

    let mut report = String::new();
    let mut problems = Vec::new();
    let mut used: Vec<String> = Vec::new();
    for (name, fragment) in &manifest.fragments {
        let text = std::fs::read_to_string(absolute.join(&fragment.path)).unwrap_or_default();
        used.extend(
            manifest
                .params
                .keys()
                .filter(|p| {
                    text.contains(&format!("{{{{{p}}}}}"))
                        || text.contains(&format!("{{{{ {p} }}}}"))
                })
                .cloned(),
        );
        let entry = PluginUse {
            path: absolute.to_string_lossy().into_owned(),
            fragment: name.clone(),
            instance: "check".into(),
            params: params.clone(),
            pin: None,
            dev: true,
        };
        match expand(&entry, Path::new("/"), &host()) {
            Ok(x) => {
                let _ = writeln!(
                    report,
                    "fragment '{name}': ok ({} nodes, {} connections, public: {})",
                    x.provenance.nodes.len(),
                    x.provenance.connections.len(),
                    if x.provenance.public.is_empty() {
                        "none".to_owned()
                    } else {
                        x.provenance.public.join(", ")
                    }
                );
            }
            Err(e) => problems.push(format!("fragment '{name}': {e}")),
        }
    }
    for param in manifest.params.keys().filter(|p| !used.contains(p)) {
        let _ = writeln!(
            report,
            "warning: parameter '{param}' is not used by any fragment"
        );
    }
    if let Some(app) = app_manifest {
        match check_rust_dependencies(&manifest, app) {
            Ok(text) => report.push_str(&text),
            Err(e) => problems.push(e),
        }
    }
    if problems.is_empty() {
        let _ = writeln!(report, "{} {}: ok", manifest.id, manifest.version);
        Ok(report)
    } else {
        Err(problems.join("\n"))
    }
}

/// Confirms an application's resolved dependencies satisfy the manifest's `rust` entries.
fn check_rust_dependencies(manifest: &Manifest, app_manifest: &Path) -> Result<String> {
    if manifest.rust.is_empty() {
        return Ok(String::new());
    }
    let output = Command::new("cargo")
        .args(["metadata", "--format-version", "1", "--manifest-path"])
        .arg(app_manifest)
        .output()
        .map_err(|e| format!("cannot run cargo metadata: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let meta: serde_json::Value =
        serde_json::from_slice(&output.stdout).map_err(|e| e.to_string())?;
    let root = meta["resolve"]["root"]
        .as_str()
        .ok_or_else(|| {
            format!(
                "{} is a workspace manifest; point --app at an application package's Cargo.toml",
                app_manifest.display()
            )
        })?
        .to_owned();
    let nodes = meta["resolve"]["nodes"]
        .as_array()
        .ok_or("cargo metadata has no resolve graph")?;
    let direct: Vec<&str> = nodes
        .iter()
        .find(|n| n["id"] == root.as_str())
        .and_then(|n| n["dependencies"].as_array())
        .map(|d| d.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();
    let packages = meta["packages"]
        .as_array()
        .ok_or("cargo metadata has no packages")?;
    let mut out = String::new();
    let mut problems = Vec::new();
    for dep in &manifest.rust {
        let req = semver::VersionReq::parse(&dep.version).map_err(|e| e.to_string())?;
        let found = packages.iter().find(|p| {
            p["name"] == dep.package.as_str() && direct.contains(&p["id"].as_str().unwrap_or(""))
        });
        match found {
            None => problems.push(format!(
                "the application does not depend on '{}'; add it to its Cargo.toml (plugin requires {})",
                dep.package, dep.version
            )),
            Some(p) => {
                let version = p["version"].as_str().unwrap_or("?");
                let ok = semver::Version::parse(version).is_ok_and(|v| req.matches(&v));
                if ok {
                    let _ = writeln!(out, "rust: {} {version} satisfies {}", dep.package, dep.version);
                } else {
                    problems.push(format!("'{}' resolves to {version}, which does not satisfy {}", dep.package, dep.version));
                }
            }
        }
    }
    if problems.is_empty() {
        Ok(out)
    } else {
        Err(problems.join("\n"))
    }
}

/// The resolved configuration of an application, or a summary of it.
pub fn expand_config(config: &Path, features: &[&str], summary: bool) -> Result<String> {
    let path = config.to_str().ok_or("configuration path is not UTF-8")?;
    let (cfg, resolved) =
        cu29_runtime::config::read_configuration_with_resolved_ron_and_features(path, features)
            .map_err(|e| e.to_string())?;
    if !summary {
        return Ok(resolved);
    }
    let graph = cfg.get_graph(None).map_err(|e| e.to_string())?;
    let mut out = String::new();
    let mut nodes: Vec<String> = graph
        .get_all_nodes()
        .iter()
        .map(|(_, n)| format!("{} ({})", n.get_id(), n.get_type()))
        .collect();
    nodes.sort();
    let _ = writeln!(out, "nodes:");
    for n in nodes {
        let _ = writeln!(out, "  {n}");
    }
    let _ = writeln!(out, "connections:");
    for c in graph.edges() {
        let _ = writeln!(out, "  {} -> {} ({})", c.src, c.dst, c.msg);
    }
    let _ = writeln!(out, "plugins:");
    let marker = resolved.find("resolved_plugins");
    if marker.is_none() {
        let _ = writeln!(out, "  (none)");
    }
    if let Some(at) = marker {
        for line in resolved[at..].lines().filter(|l| {
            l.trim_start().starts_with("instance:")
                || l.trim_start().starts_with("pin:")
                || l.trim_start().starts_with("id:")
                || l.trim_start().starts_with("version:")
        }) {
            let _ = writeln!(out, "  {}", line.trim());
        }
    }
    Ok(out)
}

/// Default plugin parent directory for `new`.
#[must_use]
pub fn default_parent() -> PathBuf {
    PathBuf::from("plugins")
}
