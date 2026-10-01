//! The plugin that ships in this crate (`plugin.ron`, `fragments/`) and the Python package
//! (`pyproject.toml`, `python/`) are versioned and packaged with the crate.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use cu29_plugin::{
    ExistingIds, Expanded, Host, LoadedPlugin, Manifest, ParamValue, PluginUse, check_collisions,
    expand,
};

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

const HOST: Host<'static> = Host {
    copper_version: env!("CARGO_PKG_VERSION"),
};

/// Values for the parameters that have no default.
fn required_params() -> BTreeMap<String, ParamValue> {
    let mut params = BTreeMap::new();
    for joint in 0..6 {
        params.insert(format!("min_{joint}"), ParamValue::Float(200.0));
        params.insert(format!("max_{joint}"), ParamValue::Float(3900.0));
    }
    params.insert("max_step".into(), ParamValue::Float(30.0));
    params.insert("max_lead".into(), ParamValue::Float(300.0));
    params.insert("cycle_ms".into(), ParamValue::Float(33.333));
    params
}

fn instance(fragment: &str, name: &str, extra: &[(&str, ParamValue)]) -> Expanded {
    let mut params = required_params();
    params.extend(extra.iter().map(|(k, v)| ((*k).to_owned(), v.clone())));
    let use_ = PluginUse {
        path: crate_dir().to_string_lossy().into_owned(),
        fragment: fragment.to_owned(),
        instance: name.to_owned(),
        params,
        pin: None,
        dev: true,
    };
    expand(&use_, &crate_dir(), &HOST).unwrap_or_else(|e| panic!("{fragment}: {}", e.message()))
}

/// What a fragment of the plugin declares, with `{i}` standing for the instance name.
struct Expected {
    fragment: &'static str,
    nodes: &'static [&'static str],
    public: &'static [&'static str],
    routes: &'static [&'static str],
    connections: &'static [(&'static str, &'static str, &'static str)],
}

const FRAGMENTS: &[Expected] = &[Expected {
    fragment: "loop",
    nodes: &["{i}_gov", "{i}_link"],
    public: &["{i}_gov", "{i}_link"],
    routes: &[
        "{i}/obs",
        "{i}/img",
        "{i}/exec",
        "{i}/infer",
        "{i}/action",
        "{i}/link_status",
    ],
    connections: &[
        ("{i}_gov", "{i}_link/exec", "cu_policy::ExecState"),
        ("{i}_gov", "{i}_link/infer", "cu_policy::InferenceRequest"),
        ("{i}_link/action", "{i}_gov", "cu_policy::ActionChunk"),
    ],
}];

fn named(items: &[&str], i: &str) -> Vec<String> {
    items.iter().map(|s| s.replace("{i}", i)).collect()
}

#[test]
fn every_fragment_declares_its_ids_routes_and_connections() {
    let manifest = Manifest::load(&crate_dir()).unwrap();
    let listed: BTreeSet<&str> = manifest.fragments.keys().map(String::as_str).collect();
    let tested: BTreeSet<&str> = FRAGMENTS.iter().map(|f| f.fragment).collect();
    assert_eq!(
        listed, tested,
        "a fragment without an expectation, or the reverse"
    );
    for want in FRAGMENTS {
        let got = instance(want.fragment, "vla", &[]);
        let p = &got.provenance;
        assert_eq!(p.fragment, want.fragment);
        assert_eq!(
            p.nodes,
            named(want.nodes, "vla"),
            "{}: nodes",
            want.fragment
        );
        assert_eq!(
            p.public,
            named(want.public, "vla"),
            "{}: public",
            want.fragment
        );
        let mut connections: Vec<(String, String, String)> = want
            .connections
            .iter()
            .map(|(s, d, m)| {
                (
                    s.replace("{i}", "vla"),
                    d.replace("{i}", "vla"),
                    (*m).to_owned(),
                )
            })
            .collect();
        connections.sort();
        let have: Vec<(String, String, String)> = p
            .connections
            .iter()
            .map(|c| (c.src.clone(), c.dst.clone(), c.msg.clone()))
            .collect();
        assert_eq!(have, connections, "{}: connections", want.fragment);
        for route in want.routes {
            let route = route.replace("{i}", "vla");
            assert!(
                got.ron.contains(&format!("route: \"{route}\"")),
                "{}: no channel routed on {route}",
                want.fragment
            );
        }
        assert_eq!(
            got.ron.matches("route: ").count(),
            want.routes.len(),
            "{}: a channel without an expected route",
            want.fragment
        );
    }
}

#[test]
fn every_parameter_is_used_by_a_fragment() {
    let dir = crate_dir();
    let manifest = Manifest::load(&dir).unwrap();
    let text: String = manifest
        .fragments
        .values()
        .map(|f| std::fs::read_to_string(dir.join(&f.path)).unwrap())
        .collect();
    let unused: Vec<&String> = manifest
        .params
        .keys()
        .filter(|name| !text.contains(&format!("{{{{{name}}}}}")))
        .collect();
    assert!(unused.is_empty(), "parameters no fragment uses: {unused:?}");
}

#[test]
fn two_instances_in_one_graph_share_no_id_and_no_route() {
    for want in FRAGMENTS {
        let left = instance(want.fragment, "arm_left", &[]);
        let right = instance(want.fragment, "arm_right", &[]);
        check_collisions(
            &left.provenance,
            &ExistingIds::default(),
            std::slice::from_ref(&right.provenance),
        )
        .unwrap();
        for route in want.routes {
            let (l, r) = (
                route.replace("{i}", "arm_left"),
                route.replace("{i}", "arm_right"),
            );
            assert!(left.ron.contains(&format!("route: \"{l}\"")));
            assert!(
                !left.ron.contains(&format!("route: \"{r}\"")),
                "{r} appears in the left instance"
            );
            assert!(right.ron.contains(&format!("route: \"{r}\"")));
        }
    }
}

#[test]
fn the_scheduler_parameters_reach_the_governor_as_integers() {
    let got = instance(
        "loop",
        "vla",
        &[
            ("s_min", ParamValue::Int(12)),
            ("horizon", ParamValue::Int(40)),
            ("d_init", ParamValue::Int(2)),
            ("blend_steps", ParamValue::Int(5)),
        ],
    );
    for line in [
        "\"sched_s_min\": 12,",
        "\"sched_horizon\": 40,",
        "\"sched_d_init\": 2,",
        "\"blend_steps\": 5,",
    ] {
        assert!(
            got.ron.contains(line),
            "{line} is missing from\n{}",
            got.ron
        );
    }
}
