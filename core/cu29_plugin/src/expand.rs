use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use crate::error::{PluginError, Result};
use crate::fragment::inspect;
use crate::load::LoadedPlugin;
use crate::manifest::is_identifier;
use crate::params::{check_string_safe, render_value, resolve_params};
use crate::pin::PIN_PREFIX;
use crate::record::{ParamValue, Provenance};
use crate::render::render;

/// What an application's `plugins` entry asks for.
#[derive(Debug, Clone, PartialEq)]
pub struct PluginUse {
    /// Plugin directory, as written in the configuration.
    pub path: String,
    pub fragment: String,
    pub instance: String,
    pub params: BTreeMap<String, ParamValue>,
    pub pin: Option<String>,
    pub dev: bool,
}

/// The running Copper.
#[derive(Debug, Clone, Copy)]
pub struct Host<'a> {
    pub copper_version: &'a str,
}

/// A rendered fragment, its record and the files it was built from.
#[derive(Debug, Clone)]
pub struct Expanded {
    pub ron: String,
    pub provenance: Provenance,
    /// Every pinned file of the plugin, so a build system can rebuild when one changes.
    pub files: Vec<PathBuf>,
}

/// Directory of a plugin, relative paths being relative to the configuration file's directory.
#[must_use]
pub fn plugin_dir(config_dir: &Path, path: &str) -> PathBuf {
    if path.starts_with('/') {
        PathBuf::from(path)
    } else {
        config_dir.join(path)
    }
}

/// Loads the plugin named by `use_`, checks it and renders the requested fragment.
///
/// The plugin's files are read once; the pin is computed from, and the fragment rendered from, the
/// same bytes.
pub fn expand(use_: &PluginUse, config_dir: &Path, host: &Host<'_>) -> Result<Expanded> {
    let loaded = LoadedPlugin::load(&plugin_dir(config_dir, &use_.path))?;
    let id = loaded.manifest.id.clone();
    expand_loaded(use_, &loaded, host).map_err(|e| {
        if e.message().starts_with("plugin '") {
            e
        } else {
            e.in_instance(&id, &use_.instance)
        }
    })
}

fn check_pin(use_: &PluginUse, pin: &str) -> Result<()> {
    match (&use_.pin, use_.dev) {
        (Some(_), true) => Err(PluginError::new("sets both `pin` and `dev: true`; use one")),
        (None, false) => Err(PluginError::new(format!(
            "has no pin; add `pin: \"{pin}\"` to the entry (or `dev: true` while editing the plugin)"
        ))),
        (Some(expected), false) if !expected.eq_ignore_ascii_case(pin) => {
            if expected.starts_with(PIN_PREFIX) {
                Err(PluginError::new(format!(
                    "pin mismatch: the configuration pins \"{expected}\" but the plugin content hashes to \"{pin}\"; \
                     review the plugin changes and update the pin"
                )))
            } else {
                Err(PluginError::new(format!(
                    "pin {expected:?} must start with \"{PIN_PREFIX}\"; the current content hashes to \"{pin}\""
                )))
            }
        }
        _ => Ok(()),
    }
}

fn expand_loaded(use_: &PluginUse, loaded: &LoadedPlugin, host: &Host<'_>) -> Result<Expanded> {
    let manifest = &loaded.manifest;
    let host_version = semver::Version::parse(host.copper_version).map_err(|e| {
        PluginError::new(format!(
            "the running Copper version {:?} is not semver: {e}",
            host.copper_version
        ))
    })?;
    let requirement = manifest.copper_requirement()?;
    if !requirement.matches(&host_version) {
        return Err(PluginError::new(format!(
            "requires Copper {requirement}, but this is Copper {host_version}"
        )));
    }
    if !is_identifier(&use_.instance, false) {
        return Err(PluginError::new(format!(
            "instance name {:?} must match [a-z][a-z0-9_]*",
            use_.instance
        )));
    }
    let fragment = manifest.fragments.get(&use_.fragment).ok_or_else(|| {
        let known: Vec<&str> = manifest.fragments.keys().map(String::as_str).collect();
        PluginError::new(format!(
            "has no fragment '{}' (available: {})",
            use_.fragment,
            known.join(", ")
        ))
    })?;

    let pin = loaded.pin();
    check_pin(use_, &pin)?;

    let params = resolve_params(&manifest.params, &use_.params)?;
    let mut vars: BTreeMap<String, String> = params
        .iter()
        .map(|(k, v)| (k.clone(), render_value(v)))
        .collect();
    check_string_safe(&use_.path)
        .map_err(|why| PluginError::new(format!("plugin path {:?} {why}", use_.path)))?;
    vars.insert("instance".into(), use_.instance.clone());
    vars.insert("plugin_dir".into(), use_.path.clone());

    let template = loaded
        .file(&fragment.path)
        .ok_or_else(|| PluginError::new(format!("fragment '{}' was not loaded", use_.fragment)))?;
    let template = std::str::from_utf8(template).map_err(|_| {
        PluginError::new(format!("fragment '{}' is not valid UTF-8", use_.fragment))
    })?;
    let ron = render(template, &vars)
        .map_err(|e| PluginError::new(format!("fragment '{}': {}", use_.fragment, e.message())))?;
    let info = inspect(&ron)
        .map_err(|e| PluginError::new(format!("fragment '{}': {}", use_.fragment, e.message())))?;

    let prefix = format!("{}_", use_.instance);
    let nodes = info.nodes();
    if let Some(bad) = nodes.iter().find(|n| !n.starts_with(&prefix)) {
        return Err(PluginError::new(format!(
            "fragment '{}' declares id {bad:?}, which must start with \"{{{{instance}}}}_\" (here \"{prefix}\")",
            use_.fragment
        )));
    }
    if let Some(dup) = nodes.windows(2).find(|w| w[0] == w[1]) {
        return Err(PluginError::new(format!(
            "fragment '{}' declares id {:?} twice",
            use_.fragment, dup[0]
        )));
    }
    let mut public = Vec::new();
    for name in &fragment.public {
        let full = format!("{prefix}{name}");
        if !nodes.contains(&full) {
            return Err(PluginError::new(format!(
                "fragment '{}' lists public node '{name}' but declares no node \"{full}\"",
                use_.fragment
            )));
        }
        public.push(full);
    }
    public.sort();

    let mut connections = info.connections;
    connections.sort();
    let mut resources = info.resources;
    resources.sort();
    Ok(Expanded {
        ron,
        provenance: Provenance {
            id: manifest.id.clone(),
            version: manifest.version.clone(),
            fragment: use_.fragment.clone(),
            instance: use_.instance.clone(),
            pin,
            dev: use_.dev,
            params: params
                .iter()
                .map(|(k, v)| (k.clone(), render_value(v)))
                .collect(),
            nodes,
            resources,
            public,
            connections,
        },
        files: loaded.paths(),
    })
}

/// Ids already present in the application when a plugin is merged.
#[derive(Debug, Clone, Default)]
pub struct ExistingIds {
    pub tasks: BTreeSet<String>,
    pub bridges: BTreeSet<String>,
    pub resources: BTreeSet<String>,
}

/// Fails if an instance name is used twice.
pub fn check_instance_names(names: &[&str]) -> Result<()> {
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for name in names {
        if !seen.insert(name) {
            return Err(PluginError::new(format!(
                "instance name '{name}' is used by more than one plugin entry"
            )));
        }
    }
    Ok(())
}

/// Fails if any node of `record` already exists in the application or in another plugin instance.
pub fn check_collisions(
    record: &Provenance,
    existing: &ExistingIds,
    others: &[Provenance],
) -> Result<()> {
    let taken: BTreeSet<&String> = existing
        .tasks
        .iter()
        .chain(&existing.bridges)
        .chain(&existing.resources)
        .collect();
    for node in &record.nodes {
        if taken.contains(node) {
            return Err(PluginError::new(format!(
                "node id \"{node}\" is already declared by the application"
            ))
            .in_instance(&record.id, &record.instance));
        }
        if let Some(other) = others
            .iter()
            .find(|o| o.instance != record.instance && o.nodes.contains(node))
        {
            return Err(PluginError::new(format!(
                "node id \"{node}\" is also declared by instance '{}' of plugin '{}'",
                other.instance, other.id
            ))
            .in_instance(&record.id, &record.instance));
        }
    }
    Ok(())
}

/// The node part of a connection endpoint (`bridge/channel` names the bridge).
fn endpoint_node(endpoint: &str) -> &str {
    endpoint.split('/').next().unwrap_or(endpoint)
}

/// Fails if a connection that is not part of an instance's own fragment reaches one of its private
/// nodes. `connections` are the `(src, dst, msg)` triples of the final, merged configuration; a
/// connection is the instance's own only if all three match.
pub fn check_encapsulation(
    records: &[Provenance],
    connections: &[(String, String, String)],
) -> Result<()> {
    for record in records {
        let private: BTreeSet<&String> = record
            .nodes
            .iter()
            .filter(|n| !record.public.contains(n))
            .collect();
        for (src, dst, msg) in connections {
            let own = record
                .connections
                .iter()
                .any(|c| &c.src == src && &c.dst == dst && &c.msg == msg);
            if own {
                continue;
            }
            for endpoint in [src, dst] {
                let node = endpoint_node(endpoint).to_owned();
                if private.contains(&node) {
                    return Err(PluginError::new(format!(
                        "the connection {src:?} -> {dst:?} ({msg}) reaches private node \"{node}\"; \
                         only its public nodes ({}) accept outside connections",
                        if record.public.is_empty() { "none".to_owned() } else { record.public.join(", ") }
                    ))
                    .in_instance(&record.id, &record.instance));
                }
            }
        }
    }
    Ok(())
}

/// Fails if a node that is not part of an instance binds one of the instance's private resource
/// bundles. `bindings` are `(node id, resource bundle id)` pairs from the merged configuration.
pub fn check_resource_access(records: &[Provenance], bindings: &[(String, String)]) -> Result<()> {
    for record in records {
        for (node, bundle) in bindings {
            let private = record.resources.contains(bundle) && !record.public.contains(bundle);
            if private && !record.nodes.contains(node) {
                return Err(PluginError::new(format!(
                    "node \"{node}\" binds resource bundle \"{bundle}\", which is private to the plugin; \
                     only its public nodes ({}) can be used from outside",
                    if record.public.is_empty() { "none".to_owned() } else { record.public.join(", ") }
                ))
                .in_instance(&record.id, &record.instance));
            }
        }
    }
    Ok(())
}
