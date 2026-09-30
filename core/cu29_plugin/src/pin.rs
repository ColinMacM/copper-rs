use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::error::Result;
use crate::load::read_files;
use crate::manifest::{MANIFEST_FILE, Manifest};

/// Prefix of a pin string.
pub const PIN_PREFIX: &str = "blake3:";

/// The files that define a plugin, ordered by path: the manifest, every fragment and every asset.
#[must_use]
pub fn pinned_files(manifest: &Manifest) -> Vec<String> {
    let mut files: BTreeSet<String> = BTreeSet::new();
    files.insert(MANIFEST_FILE.to_owned());
    files.extend(manifest.fragments.values().map(|f| f.path.clone()));
    files.extend(manifest.assets.iter().cloned());
    files.into_iter().collect()
}

/// The pin of files that are already in memory: `blake3:<hex>` over every path and its bytes,
/// each preceded by its length, so moving bytes between files or renaming a file changes the pin.
#[must_use]
pub fn pin_of(files: &BTreeMap<String, Vec<u8>>) -> String {
    let mut hasher = blake3::Hasher::new();
    for (relative, bytes) in files {
        hasher.update(&(relative.len() as u64).to_le_bytes());
        hasher.update(relative.as_bytes());
        hasher.update(&(bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    }
    format!("{PIN_PREFIX}{}", hasher.finalize().to_hex())
}

/// Content pin of a plugin directory, for tools that print it. Expansion does not use this: it
/// hashes the same bytes it renders from (see `LoadedPlugin`).
pub fn content_pin(dir: &Path, manifest: &Manifest) -> Result<String> {
    Ok(pin_of(&read_files(
        dir,
        manifest,
        crate::load::MAX_FILE_BYTES,
    )?))
}
