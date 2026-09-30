use std::collections::BTreeMap;
use std::path::{Component, Path};

use ron::Options;
use ron::extensions::Extensions;
use semver::{Version, VersionReq};
use serde::{Deserialize, Serialize};

use crate::error::{PluginError, Result};
use crate::params::ParamSpec;

/// File name of the manifest inside a plugin directory.
pub const MANIFEST_FILE: &str = "plugin.ron";

/// The manifest format understood by this crate.
pub const MANIFEST_FORMAT: u32 = 1;

/// Names that fragments receive from Copper and that parameters therefore cannot take.
pub const RESERVED_PLACEHOLDERS: [&str; 2] = ["instance", "plugin_dir"];

/// A crate whose items the fragments reference through `type:` paths.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RustDependency {
    pub package: String,
    pub version: String,
}

/// One named configuration template of a plugin.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FragmentSpec {
    /// Path of the template, relative to the plugin directory.
    pub path: String,
    /// Node ids, without the instance prefix, that the application may connect to.
    #[serde(default)]
    pub public: Vec<String>,
}

/// The parsed `plugin.ron`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub format: u32,
    pub id: String,
    pub version: String,
    pub description: String,
    /// Semantic-version requirement on the Copper runtime version.
    pub copper: String,
    #[serde(default)]
    pub rust: Vec<RustDependency>,
    #[serde(default)]
    pub params: BTreeMap<String, ParamSpec>,
    pub fragments: BTreeMap<String, FragmentSpec>,
    #[serde(default)]
    pub assets: Vec<String>,
}

pub(crate) fn ron_options() -> Options {
    Options::default()
        .with_default_extension(Extensions::IMPLICIT_SOME)
        .with_default_extension(Extensions::UNWRAP_NEWTYPES)
        .with_default_extension(Extensions::UNWRAP_VARIANT_NEWTYPES)
}

/// `[a-z][a-z0-9_]*`, plus `-` when `dash` is set.
pub(crate) fn is_identifier(value: &str, dash: bool) -> bool {
    let mut chars = value.chars();
    chars.next().is_some_and(|c| c.is_ascii_lowercase())
        && chars
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || (dash && c == '-'))
}

/// A relative path that stays inside the plugin directory.
fn check_relative_path(path: &str, what: &str) -> Result<()> {
    let bad = |why: &str| Err(PluginError::new(format!("{what} path {path:?} {why}")));
    if path.is_empty() {
        return bad("is empty");
    }
    if path.contains('\\') {
        return bad("uses a backslash; write paths with '/'");
    }
    let p = Path::new(path);
    if p.is_absolute() {
        return bad("must be relative to the plugin directory");
    }
    if p.components().any(|c| !matches!(c, Component::Normal(_))) {
        return bad("must not contain '.' or '..' components");
    }
    Ok(())
}

impl Manifest {
    /// Parses and validates manifest text.
    pub fn parse(text: &str) -> Result<Self> {
        let manifest: Manifest = ron_options().from_str(text).map_err(|e| {
            PluginError::new(format!(
                "cannot parse {MANIFEST_FILE}: {} at position {}",
                e.code, e.span
            ))
        })?;
        manifest.validate().map_err(|e| e.in_plugin(&manifest.id))?;
        Ok(manifest)
    }

    /// Reads `plugin.ron` from a plugin directory.
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join(MANIFEST_FILE);
        let text = std::fs::read_to_string(&path)
            .map_err(|e| PluginError::new(format!("cannot read {}: {e}", path.display())))?;
        Self::parse(&text).map_err(|e| {
            if e.message().starts_with("plugin '") {
                e
            } else {
                PluginError::new(format!("{}: {}", path.display(), e.message()))
            }
        })
    }

    #[must_use]
    pub fn version_parsed(&self) -> Option<Version> {
        Version::parse(&self.version).ok()
    }

    /// The requirement on the Copper runtime version.
    pub fn copper_requirement(&self) -> Result<VersionReq> {
        VersionReq::parse(&self.copper).map_err(|e| {
            PluginError::new(format!(
                "copper requirement {:?} is not valid: {e}",
                self.copper
            ))
        })
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.format != MANIFEST_FORMAT {
            return Err(PluginError::new(format!(
                "manifest format {} is not supported (this Copper reads format {MANIFEST_FORMAT})",
                self.format
            )));
        }
        if !is_identifier(&self.id, true) {
            return Err(PluginError::new(format!(
                "id {:?} must match [a-z][a-z0-9-]*",
                self.id
            )));
        }
        Version::parse(&self.version).map_err(|e| {
            PluginError::new(format!("version {:?} is not semver: {e}", self.version))
        })?;
        if self.description.trim().is_empty() {
            return Err(PluginError::new("description is empty"));
        }
        self.copper_requirement()?;
        for dep in &self.rust {
            if dep.package.is_empty() {
                return Err(PluginError::new(
                    "rust dependency has an empty package name",
                ));
            }
            VersionReq::parse(&dep.version).map_err(|e| {
                PluginError::new(format!(
                    "rust dependency '{}' version {:?} is not valid: {e}",
                    dep.package, dep.version
                ))
            })?;
        }
        for (name, spec) in &self.params {
            if !is_identifier(name, false) {
                return Err(PluginError::new(format!(
                    "parameter name {name:?} must match [a-z][a-z0-9_]*"
                )));
            }
            if RESERVED_PLACEHOLDERS.contains(&name.as_str()) {
                return Err(PluginError::new(format!(
                    "parameter name '{name}' is reserved"
                )));
            }
            spec.validate(name)?;
        }
        if self.fragments.is_empty() {
            return Err(PluginError::new("declares no fragments"));
        }
        for (name, fragment) in &self.fragments {
            if !is_identifier(name, false) {
                return Err(PluginError::new(format!(
                    "fragment name {name:?} must match [a-z][a-z0-9_]*"
                )));
            }
            check_relative_path(&fragment.path, &format!("fragment '{name}'"))?;
            for public in &fragment.public {
                if !is_identifier(public, false) {
                    return Err(PluginError::new(format!(
                        "fragment '{name}': public node {public:?} must match [a-z][a-z0-9_]*"
                    )));
                }
            }
        }
        for asset in &self.assets {
            check_relative_path(asset, "asset")?;
        }
        if self
            .assets
            .iter()
            .enumerate()
            .any(|(i, a)| self.assets[..i].contains(a))
        {
            return Err(PluginError::new("an asset is listed twice"));
        }
        Ok(())
    }
}
