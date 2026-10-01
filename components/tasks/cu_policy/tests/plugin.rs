//! The plugin that ships in this crate (`plugin.ron`, `fragments/`) and the Python package
//! (`pyproject.toml`, `python/`) are versioned and packaged with the crate.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use cu29_plugin::{LoadedPlugin, Manifest};

fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The Python (PEP 440) spelling of a crate version: `1.3.0-dev` is `1.3.0.dev0`.
fn python_version(crate_version: &str) -> String {
    match crate_version.split_once("-dev") {
        Some((release, "")) => format!("{release}.dev0"),
        _ => {
            assert!(
                !crate_version.contains('-'),
                "no Python spelling is defined for the pre-release {crate_version}"
            );
            crate_version.to_owned()
        }
    }
}

fn pyproject_version() -> String {
    let text = std::fs::read_to_string(crate_dir().join("pyproject.toml")).unwrap();
    let line = text
        .lines()
        .find(|l| l.starts_with("version = "))
        .expect("pyproject.toml has a version");
    line.trim_start_matches("version = ")
        .trim_matches('"')
        .to_owned()
}

#[test]
fn the_crate_the_plugin_and_the_python_package_have_one_version() {
    let manifest = Manifest::load(&crate_dir()).unwrap();
    let version = env!("CARGO_PKG_VERSION");
    assert_eq!(manifest.version, version, "plugin.ron");
    assert_eq!(
        pyproject_version(),
        python_version(version),
        "pyproject.toml"
    );
    assert!(
        manifest.rust.iter().any(|r| r.package == "cu-policy"),
        "the plugin names the crate that provides its types"
    );
}

#[test]
fn the_plugin_ships_every_python_module_and_the_wire_vectors() {
    let dir = crate_dir();
    let plugin = LoadedPlugin::load(&dir).expect("every listed file exists and the plugin loads");
    let listed: BTreeSet<&str> = plugin.manifest.assets.iter().map(String::as_str).collect();
    let on_disk: BTreeSet<String> = std::fs::read_dir(dir.join("python/copper_policy"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p: &PathBuf| p.extension().is_some_and(|x| x == "py"))
        .map(|p| format!("python/copper_policy/{}", file_name(&p)))
        .collect();
    for module in &on_disk {
        assert!(
            listed.contains(module.as_str()),
            "{module} is not an asset of the plugin"
        );
    }
    assert!(listed.contains("tests/golden/vectors.json"));
    assert_eq!(
        listed.len(),
        on_disk.len() + 1,
        "an asset that is not a module or the vectors"
    );
}

fn file_name(p: &Path) -> String {
    p.file_name().unwrap().to_string_lossy().into_owned()
}
