#![cfg(feature = "expand")]

use std::collections::BTreeMap;
use std::path::Path;

use cu29_plugin::{
    Connection, ExistingIds, Host, LoadedPlugin, Manifest, ParamKind, ParamSpec, ParamValue,
    PluginUse, Provenance, check_collisions, check_encapsulation, check_instance_names,
    content_pin, expand, inspect, render, resolve_params,
};

const HOST: Host<'static> = Host {
    copper_version: "1.3.0-dev",
};

const MANIFEST: &str = r#"(
    format: 1,
    id: "cu-pid-loop",
    version: "0.1.0",
    description: "Single-axis PID loop",
    copper: ">=1.3.0-dev",
    rust: [(package: "cu-pid", version: "^1.3")],
    params: {
        "rate_hz": (kind: Int, min: 1, max: 10000, default: 100),
        "gain_p": (kind: Float, default: 1.0),
        "label": (kind: Str),
        "mode": (kind: Enum(["position", "velocity"]), default: "position"),
        "enabled": (kind: Bool, default: true),
    },
    fragments: {
        "loop": (path: "fragments/loop.ron", public: ["pid"]),
    },
    assets: ["python/policy.py"],
)"#;

const FRAGMENT: &str = r#"(
    tasks: [
        (
            id: "{{instance}}_pid",
            type: "cu_pid::PIDTask",
            config: {"kp": {{gain_p}}, "rate_hz": {{rate_hz}}, "label": "{{label}}", "mode": "{{mode}}", "on": {{enabled}}, "script": "{{plugin_dir}}/python/policy.py"},
        ),
        (id: "{{instance}}_out", type: "cu_pid::Out"),
    ],
    cnx: [
        (src: "{{instance}}_pid", dst: "{{instance}}_out", msg: "cu_pid::Cmd"),
    ],
)"#;

fn write(dir: &Path, relative: &str, text: &str) {
    let path = dir.join(relative);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

fn plugin_with(manifest: &str, fragment: &str) -> tempfile::TempDir {
    let t = tempfile::tempdir().unwrap();
    write(t.path(), "plugin.ron", manifest);
    write(t.path(), "fragments/loop.ron", fragment);
    write(t.path(), "python/policy.py", "print('policy')\n");
    t
}

fn plugin() -> tempfile::TempDir {
    plugin_with(MANIFEST, FRAGMENT)
}

fn params(pairs: &[(&str, ParamValue)]) -> BTreeMap<String, ParamValue> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_owned(), v.clone()))
        .collect()
}

fn use_of(dir: &Path, instance: &str, pin: Option<String>) -> PluginUse {
    PluginUse {
        path: dir.to_string_lossy().into_owned(),
        fragment: "loop".into(),
        instance: instance.into(),
        params: params(&[("label", ParamValue::Str("elbow".into()))]),
        dev: pin.is_none(),
        pin,
    }
}

fn pinned(dir: &Path, instance: &str) -> PluginUse {
    let pin = content_pin(dir, &Manifest::load(dir).unwrap()).unwrap();
    use_of(dir, instance, Some(pin))
}

fn err_of<T: std::fmt::Debug>(r: Result<T, cu29_plugin::PluginError>) -> String {
    r.expect_err("expected an error").to_string()
}

fn contains(text: &str, needle: &str) {
    assert!(text.contains(needle), "{needle:?} not found in {text:?}");
}

// ---- manifest ----

#[test]
fn the_documented_manifest_parses() {
    let m = Manifest::parse(MANIFEST).unwrap();
    assert_eq!(
        (m.id.as_str(), m.version.as_str()),
        ("cu-pid-loop", "0.1.0")
    );
    assert_eq!(m.params.len(), 5);
    assert_eq!(m.params["rate_hz"].min, Some(1.0));
    assert_eq!(m.params["rate_hz"].default, Some(ParamValue::Int(100)));
    assert_eq!(m.params["gain_p"].default, Some(ParamValue::Float(1.0)));
    assert_eq!(
        m.params["mode"].kind,
        ParamKind::Enum(vec!["position".into(), "velocity".into()])
    );
    assert_eq!(m.fragments["loop"].public, ["pid"]);
    assert_eq!(m.assets, ["python/policy.py"]);
}

fn mutated(from: &str, to: &str) -> String {
    assert!(
        MANIFEST.contains(from),
        "test bug: {from:?} not in the manifest"
    );
    MANIFEST.replacen(from, to, 1)
}

#[test]
fn manifest_rejections_name_the_problem_and_the_plugin() {
    let cases: Vec<(String, &str)> = vec![
        (
            mutated("format: 1", "format: 2"),
            "format 2 is not supported",
        ),
        (
            mutated("id: \"cu-pid-loop\"", "id: \"Bad_Id\""),
            "must match [a-z][a-z0-9-]*",
        ),
        (
            mutated("version: \"0.1.0\"", "version: \"one\""),
            "is not semver",
        ),
        (
            mutated("copper: \">=1.3.0-dev\"", "copper: \"soon\""),
            "copper requirement",
        ),
        (
            mutated(
                "description: \"Single-axis PID loop\"",
                "description: \"  \"",
            ),
            "description is empty",
        ),
        (
            mutated("\"label\": (kind: Str)", "\"instance\": (kind: Str)"),
            "reserved",
        ),
        (
            mutated("\"label\": (kind: Str)", "\"Label\": (kind: Str)"),
            "must match [a-z][a-z0-9_]*",
        ),
        (mutated("default: 100", "default: 0"), "below the minimum"),
        (
            mutated("min: 1, max: 10000", "min: 10, max: 1"),
            "min 10 is greater than max 1",
        ),
        (
            mutated("min: 1, max: 10000", "min: 1.5, max: 10000"),
            "not an exactly representable integer",
        ),
        (
            mutated("(kind: Str)", "(kind: Str, min: 1)"),
            "min/max apply to Int and Float",
        ),
        (
            mutated("Enum([\"position\", \"velocity\"])", "Enum([])"),
            "at least one choice",
        ),
        (
            mutated("Enum([\"position\", \"velocity\"])", "Enum([\"a\\\"b\"])"),
            "cannot be placed inside a RON string",
        ),
        (
            mutated("default: \"position\"", "default: \"sideways\""),
            "is not one of",
        ),
        (
            mutated("path: \"fragments/loop.ron\"", "path: \"/etc/passwd\""),
            "must be relative",
        ),
        (
            mutated("path: \"fragments/loop.ron\"", "path: \"../loop.ron\""),
            "'.' or '..'",
        ),
        (
            mutated("path: \"fragments/loop.ron\"", "path: \"a\\\\b.ron\""),
            "backslash",
        ),
        (
            mutated("path: \"fragments/loop.ron\"", "path: \"\""),
            "is empty",
        ),
        (
            mutated("public: [\"pid\"]", "public: [\"Pid\"]"),
            "public node",
        ),
        (
            mutated(
                "assets: [\"python/policy.py\"]",
                "assets: [\"a.py\", \"a.py\"]",
            ),
            "listed twice",
        ),
        (
            mutated("assets: [\"python/policy.py\"]", "assets: [\"../escape\"]"),
            "'.' or '..'",
        ),
        (
            mutated(
                "rust: [(package: \"cu-pid\", version: \"^1.3\")]",
                "rust: [(package: \"cu-pid\", version: \"nope\")]",
            ),
            "rust dependency 'cu-pid'",
        ),
    ];
    for (text, needle) in cases {
        let e = err_of(Manifest::parse(&text));
        contains(&e, needle);
    }
    let e = err_of(Manifest::parse(&mutated(
        "format: 1,",
        "format: 1, colour: \"red\",",
    )));
    contains(&e, "cannot parse plugin.ron");
    let e = err_of(Manifest::parse(&mutated(
        "id: \"cu-pid-loop\"",
        "id: \"Bad\"",
    )));
    assert!(e.starts_with("plugin 'Bad':"), "{e}");
}

#[test]
fn a_manifest_with_no_fragments_is_rejected() {
    let text = MANIFEST.replace(
        "\"loop\": (path: \"fragments/loop.ron\", public: [\"pid\"]),",
        "",
    );
    contains(&err_of(Manifest::parse(&text)), "declares no fragments");
}

// ---- parameters ----

fn spec(kind: ParamKind) -> ParamSpec {
    ParamSpec {
        kind,
        default: None,
        min: None,
        max: None,
    }
}

#[test]
fn params_resolve_defaults_overrides_and_int_to_float() {
    let specs = Manifest::parse(MANIFEST).unwrap().params;
    let r = resolve_params(
        &specs,
        &params(&[
            ("label", ParamValue::Str("x".into())),
            ("gain_p", ParamValue::Int(2)),
        ]),
    )
    .unwrap();
    assert_eq!(r["rate_hz"], ParamValue::Int(100));
    assert_eq!(
        r["gain_p"],
        ParamValue::Float(2.0),
        "an integer literal is accepted for a Float"
    );
    assert_eq!(r["mode"], ParamValue::Str("position".into()));
    assert_eq!(r["enabled"], ParamValue::Bool(true));
}

#[test]
fn param_errors_name_the_parameter() {
    let specs = Manifest::parse(MANIFEST).unwrap().params;
    let label = ("label", ParamValue::Str("x".into()));
    let with = |extra: (&str, ParamValue)| resolve_params(&specs, &params(&[label.clone(), extra]));
    contains(
        &err_of(resolve_params(&specs, &BTreeMap::new())),
        "required parameter 'label'",
    );
    let e = err_of(with(("typo", ParamValue::Int(1))));
    contains(&e, "unknown parameter 'typo'");
    contains(&e, "rate_hz");
    contains(
        &err_of(with(("rate_hz", ParamValue::Float(10.0)))),
        "expected Int",
    );
    contains(
        &err_of(with(("rate_hz", ParamValue::Int(0)))),
        "below the minimum",
    );
    contains(
        &err_of(with(("rate_hz", ParamValue::Int(10_001)))),
        "above the maximum",
    );
    contains(
        &err_of(with(("enabled", ParamValue::Int(1)))),
        "expected Bool",
    );
    contains(
        &err_of(with(("mode", ParamValue::Str("sideways".into())))),
        "is not one of",
    );
    contains(
        &err_of(with(("gain_p", ParamValue::Float(f64::NAN)))),
        "finite",
    );
    contains(
        &err_of(with(("gain_p", ParamValue::Float(f64::INFINITY)))),
        "finite",
    );
}

#[test]
fn strings_cannot_break_out_of_the_ron_string_they_are_placed_in() {
    let mut specs = BTreeMap::new();
    specs.insert("s".to_owned(), spec(ParamKind::Str));
    for evil in ["a\"b", "a\\b", "line\nbreak", "tab\t", "x\"}, (id: \"pwn"] {
        let e = err_of(resolve_params(
            &specs,
            &params(&[("s", ParamValue::Str(evil.into()))]),
        ));
        contains(&e, "cannot be placed inside a RON string");
    }
    assert!(
        resolve_params(
            &specs,
            &params(&[("s", ParamValue::Str("fine text {braces} é".into()))])
        )
        .is_ok()
    );
}

#[test]
fn param_values_deserialize_from_ron_by_their_literal_form() {
    let v: BTreeMap<String, ParamValue> =
        ron::from_str(r#"{"a": 100, "b": 1.5, "c": true, "d": "x", "e": -3, "f": 2.0}"#).unwrap();
    assert_eq!(v["a"], ParamValue::Int(100));
    assert_eq!(v["b"], ParamValue::Float(1.5));
    assert_eq!(v["c"], ParamValue::Bool(true));
    assert_eq!(v["d"], ParamValue::Str("x".into()));
    assert_eq!(v["e"], ParamValue::Int(-3));
    assert_eq!(v["f"], ParamValue::Float(2.0));
}

// ---- rendering ----

fn vars(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect()
}

#[test]
fn placeholders_are_replaced_with_optional_inner_spaces() {
    let out = render("{{a}}-{{ b }}-{{a}}", &vars(&[("a", "1"), ("b", "two")])).unwrap();
    assert_eq!(out, "1-two-1");
}

#[test]
fn undefined_placeholders_are_all_reported_once() {
    let e = err_of(render("{{x}} {{y}} {{x}} {{a}}", &vars(&[("a", "1")])));
    contains(&e, "{{x}}, {{y}}");
    assert_eq!(e.matches("{{x}}").count(), 1, "{e}");
    contains(&e, "defined: a");
}

#[test]
fn text_that_only_resembles_a_placeholder_is_left_alone() {
    let t = "{{ not a name }} {{1x}} {{}} {{a";
    assert_eq!(render(t, &vars(&[])).unwrap(), t);
    assert_eq!(
        render("{ {a} } {{a}}}", &vars(&[("a", "Q")])).unwrap(),
        "{ {a} } Q}"
    );
}

#[test]
fn rendered_numbers_keep_their_type_through_expansion() {
    let t = plugin();
    let render_with = |gain: ParamValue| {
        let mut u = pinned(t.path(), "elbow");
        u.params.insert("gain_p".into(), gain);
        expand(&u, Path::new("/"), &HOST).unwrap().ron
    };
    // An integer supplied for a Float stays a float literal; config readers treat 2 and 2.0 differently.
    contains(&render_with(ParamValue::Int(2)), "\"kp\": 2.0,");
    contains(&render_with(ParamValue::Float(2.5)), "\"kp\": 2.5,");
    for exponent in [1e-7, 1e21, -3.5e-300] {
        let ron = render_with(ParamValue::Float(exponent));
        let info =
            inspect(&ron).unwrap_or_else(|e| panic!("{exponent:e} broke the fragment: {e}\n{ron}"));
        assert_eq!(info.tasks.len(), 2);
        let parsed: ron::Value = ron::from_str(&ron).unwrap();
        let text = format!("{parsed:?}");
        assert!(
            text.contains("Number(F64") || text.contains("F64"),
            "{exponent:e} did not stay a float: {text}"
        );
    }
    // Ints render without a decimal point.
    let mut u = pinned(t.path(), "elbow");
    u.params.insert("rate_hz".into(), ParamValue::Int(250));
    contains(
        &expand(&u, Path::new("/"), &HOST).unwrap().ron,
        "\"rate_hz\": 250,",
    );
}

// ---- pin ----

#[test]
fn the_pin_is_stable_and_covers_manifest_fragment_and_assets() {
    let t = plugin();
    let m = Manifest::load(t.path()).unwrap();
    let base = content_pin(t.path(), &m).unwrap();
    assert!(base.starts_with("blake3:") && base.len() == "blake3:".len() + 64);
    assert_eq!(content_pin(t.path(), &m).unwrap(), base, "deterministic");
    for (file, change) in [
        ("fragments/loop.ron", format!("{FRAGMENT}\n")),
        ("python/policy.py", "print('changed')\n".to_owned()),
    ] {
        let original = std::fs::read_to_string(t.path().join(file)).unwrap();
        write(t.path(), file, &change);
        assert_ne!(
            content_pin(t.path(), &m).unwrap(),
            base,
            "{file} is covered"
        );
        write(t.path(), file, &original);
    }
    write(t.path(), "plugin.ron", &format!("{MANIFEST}\n"));
    let m2 = Manifest::load(t.path()).unwrap();
    assert_ne!(
        content_pin(t.path(), &m2).unwrap(),
        base,
        "the manifest is covered"
    );
    write(t.path(), "README.md", "not part of the pin");
    write(t.path(), "plugin.ron", MANIFEST);
    assert_eq!(
        content_pin(t.path(), &m).unwrap(),
        base,
        "unlisted files are ignored"
    );
}

#[test]
fn moving_bytes_between_two_files_changes_the_pin() {
    let a = plugin_with(MANIFEST, FRAGMENT);
    let b = plugin_with(MANIFEST, FRAGMENT);
    write(a.path(), "python/policy.py", "AB");
    write(b.path(), "python/policy.py", "A");
    write(b.path(), "fragments/loop.ron", &format!("{FRAGMENT}B"));
    write(a.path(), "fragments/loop.ron", FRAGMENT);
    let m = Manifest::load(a.path()).unwrap();
    assert_ne!(
        content_pin(a.path(), &m).unwrap(),
        content_pin(b.path(), &m).unwrap()
    );
}

#[test]
fn a_missing_asset_is_reported_as_an_asset() {
    let t = plugin();
    std::fs::remove_file(t.path().join("python/policy.py")).unwrap();
    let m = Manifest::load(t.path()).unwrap();
    let e = err_of(content_pin(t.path(), &m));
    contains(&e, "asset \"python/policy.py\"");
}

// ---- fragments ----

#[test]
fn inspect_reads_ids_and_connections() {
    let info = inspect(
        r#"(
        tasks: [(id: "a_x", type: "t::X"), (id: "a_y", type: "t::Y")],
        bridges: [(id: "a_b", type: "t::B", channels: [Rx(id: "c")])],
        resources: [(id: "a_hw", provider: "p::P")],
        cnx: [(src: "a_x", dst: "a_y", msg: "m::M"), (src: "a_b/c", dst: "a_x", msg: "m::M")],
    )"#,
    )
    .unwrap();
    assert_eq!(info.tasks, ["a_x", "a_y"]);
    assert_eq!(info.bridges, ["a_b"]);
    assert_eq!(info.resources, ["a_hw"]);
    assert_eq!(info.nodes(), ["a_b", "a_hw", "a_x", "a_y"]);
    assert_eq!(
        info.connections[1],
        Connection {
            src: "a_b/c".into(),
            dst: "a_x".into(),
            msg: "m::M".into()
        }
    );
}

#[test]
fn application_level_sections_are_not_allowed_in_a_fragment() {
    for section in [
        "logging: (file: \"x\")",
        "runtime: (rate_target_hz: 10)",
        "monitors: []",
        "missions: []",
        "includes: []",
        "plugins: []",
        "log_streaming: ()",
        "constants: []",
    ] {
        let e = err_of(inspect(&format!("(tasks: [], {section})")));
        contains(&e, "is not allowed in a fragment");
        contains(&e, "allowed: tasks, bridges, resources, cnx");
    }
}

#[test]
fn malformed_fragments_are_reported() {
    contains(
        &err_of(inspect("(tasks: [(type: \"x\")])")),
        "tasks[0] has no string 'id'",
    );
    contains(&err_of(inspect("(tasks: 5)")), "must be a list");
    contains(&err_of(inspect("(cnx: [(src: \"a\")])")), "no string 'dst'");
    contains(&err_of(inspect("[1, 2]")), "must be a struct");
    contains(&err_of(inspect("(tasks: [")), "does not parse");
}

// ---- expand ----

#[test]
fn a_pinned_instance_expands_with_full_provenance() {
    let t = plugin();
    let use_ = pinned(t.path(), "elbow");
    let x = expand(&use_, Path::new("/unused"), &HOST).unwrap();
    let p = &x.provenance;
    assert_eq!(
        (
            p.id.as_str(),
            p.version.as_str(),
            p.fragment.as_str(),
            p.instance.as_str()
        ),
        ("cu-pid-loop", "0.1.0", "loop", "elbow")
    );
    assert_eq!(p.pin, use_.pin.clone().unwrap());
    assert!(!p.dev);
    assert_eq!(p.nodes, ["elbow_out", "elbow_pid"]);
    assert_eq!(p.public, ["elbow_pid"]);
    assert_eq!(
        p.connections,
        [Connection {
            src: "elbow_pid".into(),
            dst: "elbow_out".into(),
            msg: "cu_pid::Cmd".into()
        }]
    );
    assert_eq!(p.params["rate_hz"], "100", "defaults are recorded");
    assert_eq!(p.params["label"], "elbow");
    contains(&x.ron, "\"kp\": 1.0");
    contains(&x.ron, "\"rate_hz\": 100");
    contains(&x.ron, "\"label\": \"elbow\"");
    contains(&x.ron, "\"on\": true");
    contains(
        &x.ron,
        &format!("\"script\": \"{}/python/policy.py\"", t.path().display()),
    );
    assert!(!x.ron.contains("{{"), "no placeholder survives");
}

#[test]
fn relative_plugin_paths_resolve_against_the_config_directory() {
    let t = plugin();
    let parent = t.path().parent().unwrap();
    let name = t.path().file_name().unwrap().to_string_lossy().into_owned();
    let mut use_ = pinned(t.path(), "elbow");
    use_.path = name.clone();
    let x = expand(&use_, parent, &HOST).unwrap();
    contains(&x.ron, &format!("\"script\": \"{name}/python/policy.py\""));
}

#[test]
fn a_pin_is_required_unless_dev_is_set_and_the_error_prints_the_hash() {
    let t = plugin();
    let mut use_ = use_of(t.path(), "elbow", None);
    use_.dev = false;
    let e = err_of(expand(&use_, Path::new("/"), &HOST));
    contains(&e, "has no pin");
    contains(&e, "blake3:");
    contains(&e, "plugin 'cu-pid-loop' instance 'elbow':");
    use_.dev = true;
    let x = expand(&use_, Path::new("/"), &HOST).unwrap();
    assert!(x.provenance.dev);
    use_.pin = Some("blake3:00".into());
    contains(
        &err_of(expand(&use_, Path::new("/"), &HOST)),
        "sets both `pin` and `dev: true`",
    );
}

#[test]
fn a_changed_plugin_fails_its_pin_and_reports_both_hashes() {
    let t = plugin();
    let use_ = pinned(t.path(), "elbow");
    let old = use_.pin.clone().unwrap();
    write(t.path(), "python/policy.py", "print('tampered')\n");
    let e = err_of(expand(&use_, Path::new("/"), &HOST));
    contains(&e, "pin mismatch");
    contains(&e, &old);
    contains(&e, "review the plugin changes");
    let mut bad = use_.clone();
    bad.pin = Some("abc".into());
    contains(
        &err_of(expand(&bad, Path::new("/"), &HOST)),
        "must start with \"blake3:\"",
    );
}

#[test]
fn an_uppercase_pin_is_accepted() {
    let t = plugin();
    let mut use_ = pinned(t.path(), "elbow");
    use_.pin = use_
        .pin
        .map(|p| p.to_uppercase().replace("BLAKE3:", "blake3:"));
    assert!(expand(&use_, Path::new("/"), &HOST).is_ok());
}

#[test]
fn the_copper_requirement_is_checked_against_the_running_version() {
    let t = plugin_with(&MANIFEST.replace(">=1.3.0-dev", ">=2.0.0"), FRAGMENT);
    let e = err_of(expand(&pinned(t.path(), "elbow"), Path::new("/"), &HOST));
    contains(&e, "requires Copper >=2.0.0, but this is Copper 1.3.0-dev");
    assert!(
        expand(
            &pinned(t.path(), "elbow"),
            Path::new("/"),
            &Host {
                copper_version: "2.1.0"
            }
        )
        .is_ok()
    );
    contains(
        &err_of(expand(
            &pinned(t.path(), "elbow"),
            Path::new("/"),
            &Host {
                copper_version: "garbage",
            },
        )),
        "is not semver",
    );
}

#[test]
fn unknown_fragments_and_bad_instances_are_reported() {
    let t = plugin();
    let mut use_ = pinned(t.path(), "elbow");
    use_.fragment = "nope".into();
    let e = err_of(expand(&use_, Path::new("/"), &HOST));
    contains(&e, "has no fragment 'nope' (available: loop)");
    for bad in ["Elbow", "2arm", "a-b", "", "a b"] {
        let mut u = pinned(t.path(), "elbow");
        u.instance = bad.into();
        contains(&err_of(expand(&u, Path::new("/"), &HOST)), "instance name");
    }
}

#[test]
fn every_declared_id_must_carry_the_instance_prefix() {
    let frag = FRAGMENT.replace("id: \"{{instance}}_out\"", "id: \"out\"");
    let t = plugin_with(MANIFEST, &frag);
    let e = err_of(expand(&pinned(t.path(), "elbow"), Path::new("/"), &HOST));
    contains(&e, "declares id \"out\"");
    contains(&e, "must start with");
    contains(&e, "\"elbow_\"");
}

#[test]
fn duplicate_ids_and_undeclared_public_nodes_are_rejected() {
    let dup = FRAGMENT.replace("id: \"{{instance}}_out\"", "id: \"{{instance}}_pid\"");
    let t = plugin_with(MANIFEST, &dup);
    contains(
        &err_of(expand(&pinned(t.path(), "elbow"), Path::new("/"), &HOST)),
        "declares id \"elbow_pid\" twice",
    );
    let t = plugin_with(
        &MANIFEST.replace("public: [\"pid\"]", "public: [\"ghost\"]"),
        FRAGMENT,
    );
    contains(
        &err_of(expand(&pinned(t.path(), "elbow"), Path::new("/"), &HOST)),
        "public node 'ghost' but declares no node \"elbow_ghost\"",
    );
}

#[test]
fn undefined_placeholders_and_param_problems_surface_with_plugin_context() {
    let t = plugin_with(MANIFEST, &FRAGMENT.replace("{{gain_p}}", "{{gain_q}}"));
    let e = err_of(expand(&pinned(t.path(), "elbow"), Path::new("/"), &HOST));
    contains(&e, "fragment 'loop'");
    contains(&e, "{{gain_q}}");
    let t = plugin();
    let mut u = pinned(t.path(), "elbow");
    u.params.insert("rate_hz".into(), ParamValue::Int(0));
    let e = err_of(expand(&u, Path::new("/"), &HOST));
    contains(
        &e,
        "plugin 'cu-pid-loop' instance 'elbow': parameter 'rate_hz'",
    );
    let mut u = pinned(t.path(), "elbow");
    u.params
        .insert("label".into(), ParamValue::Str("x\"}, (id: \"pwn".into()));
    contains(
        &err_of(expand(&u, Path::new("/"), &HOST)),
        "cannot be placed inside a RON string",
    );
}

#[test]
fn a_fragment_with_application_sections_is_rejected_at_expansion() {
    let t = plugin_with(
        MANIFEST,
        &FRAGMENT.replace("cnx: [", "logging: (file: \"x\"),\n    cnx: ["),
    );
    contains(
        &err_of(expand(&pinned(t.path(), "elbow"), Path::new("/"), &HOST)),
        "section 'logging' is not allowed",
    );
}

#[test]
fn a_plugin_directory_that_does_not_exist_is_an_error() {
    let use_ = PluginUse {
        path: "nowhere".into(),
        fragment: "loop".into(),
        instance: "a".into(),
        params: BTreeMap::new(),
        pin: None,
        dev: true,
    };
    contains(
        &err_of(expand(&use_, Path::new("/tmp"), &HOST)),
        "cannot read",
    );
}

// ---- checks across instances ----

fn record(
    instance: &str,
    nodes: &[&str],
    public: &[&str],
    connections: &[(&str, &str, &str)],
) -> Provenance {
    record_with_resources(instance, nodes, &[], public, connections)
}

fn record_with_resources(
    instance: &str,
    nodes: &[&str],
    resources: &[&str],
    public: &[&str],
    connections: &[(&str, &str, &str)],
) -> Provenance {
    Provenance {
        id: "p".into(),
        version: "0.1.0".into(),
        fragment: "f".into(),
        instance: instance.into(),
        pin: "blake3:0".into(),
        dev: false,
        params: BTreeMap::new(),
        nodes: nodes.iter().map(|s| (*s).to_owned()).collect(),
        resources: resources.iter().map(|s| (*s).to_owned()).collect(),
        public: public.iter().map(|s| (*s).to_owned()).collect(),
        connections: connections
            .iter()
            .map(|(a, b, m)| Connection {
                src: (*a).into(),
                dst: (*b).into(),
                msg: (*m).into(),
            })
            .collect(),
    }
}

fn triple(a: &str, b: &str, m: &str) -> (String, String, String) {
    (a.to_owned(), b.to_owned(), m.to_owned())
}

#[test]
fn instance_names_must_be_unique() {
    assert!(check_instance_names(&["a", "b"]).is_ok());
    contains(
        &err_of(check_instance_names(&["a", "b", "a"])),
        "instance name 'a'",
    );
}

#[test]
fn collisions_with_the_application_and_other_instances_are_errors() {
    let mine = record("arm", &["arm_left_x"], &[], &[]);
    let mut existing = ExistingIds::default();
    assert!(check_collisions(&mine, &existing, &[]).is_ok());
    existing.tasks.insert("arm_left_x".into());
    contains(
        &err_of(check_collisions(&mine, &existing, &[])),
        "already declared by the application",
    );
    existing.tasks.clear();
    existing.bridges.insert("arm_left_x".into());
    assert!(
        check_collisions(&mine, &existing, &[]).is_err(),
        "bridge ids count"
    );
    existing.bridges.clear();
    existing.resources.insert("arm_left_x".into());
    assert!(
        check_collisions(&mine, &existing, &[]).is_err(),
        "resource ids count"
    );
    // Prefix ambiguity: instance "arm" with local id "left_x" versus instance "arm_left" with local id "x".
    let other = record("arm_left", &["arm_left_x"], &[], &[]);
    let e = err_of(check_collisions(&mine, &ExistingIds::default(), &[other]));
    contains(&e, "also declared by instance 'arm_left'");
}

#[test]
fn private_nodes_accept_only_their_own_fragments_connections() {
    let r = record(
        "a",
        &["a_in", "a_priv"],
        &["a_in"],
        &[("a_in", "a_priv", "m::M")],
    );
    let one = std::slice::from_ref(&r);
    assert!(check_encapsulation(one, &[triple("a_in", "a_priv", "m::M")]).is_ok());
    assert!(check_encapsulation(one, &[triple("sensor", "a_in", "m::M")]).is_ok());
    let e = err_of(check_encapsulation(
        one,
        &[triple("sensor", "a_priv", "m::M")],
    ));
    contains(&e, "reaches private node \"a_priv\"");
    contains(&e, "public nodes (a_in)");
    contains(&e, "plugin 'p' instance 'a'");
    assert!(
        check_encapsulation(one, &[triple("a_priv", "sink", "m::M")]).is_err(),
        "outputs of private nodes too"
    );
    // The fragment's own (src, dst) with another message type is not the fragment's connection.
    let e = err_of(check_encapsulation(
        one,
        &[triple("a_in", "a_priv", "other::Msg")],
    ));
    contains(&e, "reaches private node \"a_priv\"");
    let with_bridge = record("a", &["a_bridge"], &[], &[]);
    let e = err_of(check_encapsulation(
        std::slice::from_ref(&with_bridge),
        &[triple("a_bridge/chan", "sink", "m::M")],
    ));
    contains(&e, "private node \"a_bridge\"");
    assert!(
        check_encapsulation(
            &[record("a", &["a_p"], &[], &[])],
            &[triple("x", "__nc__", "m::M")]
        )
        .is_ok(),
        "unrelated connections pass"
    );
}

#[test]
fn private_resource_bundles_cannot_be_bound_from_outside() {
    let r = record_with_resources("x", &["x_bus", "x_drv"], &["x_bus"], &["x_drv"], &[]);
    let one = std::slice::from_ref(&r);
    let own = vec![("x_drv".to_owned(), "x_bus".to_owned())];
    assert!(
        cu29_plugin::check_resource_access(one, &own).is_ok(),
        "the instance's own node"
    );
    let outside = vec![("t2".to_owned(), "x_bus".to_owned())];
    let e = err_of(cu29_plugin::check_resource_access(one, &outside));
    contains(&e, "node \"t2\" binds resource bundle \"x_bus\"");
    contains(&e, "plugin 'p' instance 'x'");
    let public = record_with_resources("x", &["x_bus"], &["x_bus"], &["x_bus"], &[]);
    assert!(
        cu29_plugin::check_resource_access(&[public], &outside).is_ok(),
        "public bundles can be bound"
    );
}

#[test]
fn the_pin_is_computed_from_the_bytes_that_are_rendered() {
    let t = plugin();
    let loaded = LoadedPlugin::load(t.path()).unwrap();
    let pin = loaded.pin();
    // Change a file after loading: the loaded plugin keeps the bytes it hashed.
    write(t.path(), "fragments/loop.ron", "(tasks: [], cnx: [])");
    assert_eq!(loaded.pin(), pin);
    assert_eq!(
        loaded.file("fragments/loop.ron").unwrap(),
        FRAGMENT.as_bytes()
    );
    assert_ne!(LoadedPlugin::load(t.path()).unwrap().pin(), pin);
}

#[test]
fn a_symlink_leaving_the_plugin_directory_is_refused() {
    let t = plugin();
    let outside = tempfile::tempdir().unwrap();
    std::fs::write(outside.path().join("evil.ron"), "(tasks: [], cnx: [])").unwrap();
    std::fs::remove_file(t.path().join("fragments/loop.ron")).unwrap();
    std::os::unix::fs::symlink(
        outside.path().join("evil.ron"),
        t.path().join("fragments/loop.ron"),
    )
    .unwrap();
    let e = err_of(LoadedPlugin::load(t.path()));
    contains(&e, "outside the plugin directory");
    // A directory or device in place of a file is not a regular file either.
    std::fs::remove_file(t.path().join("fragments/loop.ron")).unwrap();
    std::os::unix::fs::symlink("/dev/zero", t.path().join("fragments/loop.ron")).unwrap();
    contains(
        &err_of(LoadedPlugin::load(t.path())),
        "outside the plugin directory",
    );
}

#[test]
fn oversized_files_are_refused() {
    let t = plugin();
    let e = err_of(LoadedPlugin::load_with_limit(t.path(), 40));
    contains(&e, "is larger than 40 bytes");
}

#[test]
fn an_int_bound_is_compared_exactly() {
    let text = MANIFEST.replace("min: 1, max: 10000", "min: 1, max: 9007199254740992");
    let specs = Manifest::parse(&text).unwrap().params;
    let label = ("label", ParamValue::Str("x".into()));
    assert!(
        resolve_params(
            &specs,
            &params(&[
                label.clone(),
                ("rate_hz", ParamValue::Int(9_007_199_254_740_992))
            ])
        )
        .is_ok()
    );
    contains(
        &err_of(resolve_params(
            &specs,
            &params(&[label, ("rate_hz", ParamValue::Int(9_007_199_254_740_993))]),
        )),
        "above the maximum",
    );
    // An Int that a Float cannot hold exactly is refused rather than rounded.
    let floats = Manifest::parse(MANIFEST).unwrap().params;
    contains(
        &err_of(resolve_params(
            &floats,
            &params(&[
                ("label", ParamValue::Str("x".into())),
                ("gain_p", ParamValue::Int(9_007_199_254_740_993)),
            ]),
        )),
        "without rounding",
    );
}
