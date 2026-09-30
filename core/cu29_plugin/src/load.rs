use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::error::{PluginError, Result};
use crate::manifest::{MANIFEST_FILE, Manifest};
use crate::pin::{pin_of, pinned_files};

/// Largest single file a plugin may ship (manifest, fragment or asset).
pub const MAX_FILE_BYTES: usize = 64 * 1024 * 1024;

/// A plugin directory read into memory once.
///
/// The manifest, every fragment and every asset are read exactly one time. The pin is computed
/// from those bytes and the fragment is rendered from those same bytes, so the content that is
/// pinned is the content that is used, even if a file changes on disk afterwards.
#[derive(Debug)]
pub struct LoadedPlugin {
    /// Canonical plugin directory.
    pub root: PathBuf,
    pub manifest: Manifest,
    files: BTreeMap<String, Vec<u8>>,
}

fn read_bounded(root: &Path, relative: &str, limit: usize, role: &str) -> Result<Vec<u8>> {
    let path = root.join(relative);
    let unreadable = |e: std::io::Error| {
        PluginError::new(format!(
            "{role} {relative:?} cannot be read ({}): {e}",
            path.display()
        ))
    };
    // A symlink that leaves the plugin directory would make the plugin depend on files the pin
    // and the review never covered, and device files would never end.
    let canonical = std::fs::canonicalize(&path).map_err(unreadable)?;
    if !canonical.starts_with(root) {
        return Err(PluginError::new(format!(
            "{role} {relative:?} resolves to {}, outside the plugin directory",
            canonical.display()
        )));
    }
    let file = std::fs::File::open(&canonical).map_err(unreadable)?;
    if !file.metadata().map_err(unreadable)?.is_file() {
        return Err(PluginError::new(format!(
            "{role} {relative:?} is not a regular file"
        )));
    }
    let mut bytes = Vec::new();
    file.take(limit as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(unreadable)?;
    if bytes.len() > limit {
        return Err(PluginError::new(format!(
            "{role} {relative:?} is larger than {limit} bytes"
        )));
    }
    Ok(bytes)
}

fn role_of(relative: &str, manifest: &Manifest) -> &'static str {
    if relative == MANIFEST_FILE {
        "manifest"
    } else if manifest.assets.iter().any(|a| a == relative) {
        "asset"
    } else {
        "fragment"
    }
}

/// Reads every pinned file of `manifest` once, in pin order.
pub(crate) fn read_files(
    dir: &Path,
    manifest: &Manifest,
    limit: usize,
) -> Result<BTreeMap<String, Vec<u8>>> {
    let root = std::fs::canonicalize(dir)
        .map_err(|e| PluginError::new(format!("cannot read {}: {e}", dir.display())))?;
    let mut files = BTreeMap::new();
    for relative in pinned_files(manifest) {
        let role = role_of(&relative, manifest);
        files.insert(
            relative.clone(),
            read_bounded(&root, &relative, limit, role)?,
        );
    }
    Ok(files)
}

impl LoadedPlugin {
    pub fn load(dir: &Path) -> Result<Self> {
        Self::load_with_limit(dir, MAX_FILE_BYTES)
    }

    pub fn load_with_limit(dir: &Path, limit: usize) -> Result<Self> {
        let root = std::fs::canonicalize(dir).map_err(|e| {
            PluginError::new(format!(
                "cannot read {}: {e}",
                dir.join(MANIFEST_FILE).display()
            ))
        })?;
        let manifest_bytes = read_bounded(&root, MANIFEST_FILE, limit, "manifest")?;
        let text = String::from_utf8(manifest_bytes.clone()).map_err(|_| {
            PluginError::new(format!(
                "{} is not valid UTF-8",
                root.join(MANIFEST_FILE).display()
            ))
        })?;
        let manifest = Manifest::parse(&text).map_err(|e| {
            if e.message().starts_with("plugin '") {
                e
            } else {
                PluginError::new(format!(
                    "{}: {}",
                    root.join(MANIFEST_FILE).display(),
                    e.message()
                ))
            }
        })?;
        let mut files = BTreeMap::new();
        files.insert(MANIFEST_FILE.to_owned(), manifest_bytes);
        for relative in pinned_files(&manifest) {
            if relative == MANIFEST_FILE {
                continue;
            }
            let role = role_of(&relative, &manifest);
            files.insert(
                relative.clone(),
                read_bounded(&root, &relative, limit, role)
                    .map_err(|e| e.in_plugin(&manifest.id))?,
            );
        }
        Ok(Self {
            root,
            manifest,
            files,
        })
    }

    /// The pin of the bytes held in memory.
    #[must_use]
    pub fn pin(&self) -> String {
        pin_of(&self.files)
    }

    /// The bytes of a pinned file, by its manifest-relative path.
    #[must_use]
    pub fn file(&self, relative: &str) -> Option<&[u8]> {
        self.files.get(relative).map(Vec::as_slice)
    }

    /// Absolute paths of every pinned file, for build systems that must notice changes.
    #[must_use]
    pub fn paths(&self) -> Vec<PathBuf> {
        self.files
            .keys()
            .map(|relative| self.root.join(relative))
            .collect()
    }
}
