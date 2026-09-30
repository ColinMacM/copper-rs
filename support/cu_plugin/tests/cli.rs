use std::path::Path;
use std::process::Command;

use cu_plugin::{check, describe, expand_config, new_plugin, parse_override, pin};
use cu29_plugin::ParamValue;

fn write(dir: &Path, relative: &str, text: &str) {
    let path = dir.join(relative);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

fn contains(text: &str, needle: &str) {
    assert!(text.contains(needle), "{needle:?} not found in:\n{text}");
}

const MANIFEST: &str = r#"(
    format: 1,
    id: "cu-demo",
    version: "0.1.0",
    description: "Demo",
    copper: ">=1.3.0-dev",
    rust: [(package: "cu-pid", version: "^1.3")],
    params: {
        "rate": (kind: Int, min: 5, max: 50),
        "gain": (kind: Float, default: 1.5),
        "name": (kind: Str),
        "mode": (kind: Enum(["a", "b"])),
        "on": (kind: Bool),
        "unused": (kind: Int, default: 1),
    },
    fragments: {"main": (path: "fragments/main.ron", public: ["node"])},
    assets: [],
)"#;

const FRAGMENT: &str = r#"(
    tasks: [(id: "{{instance}}_node", type: "t::T", config: {"rate": {{rate}}, "gain": {{gain}}, "name": "{{name}}", "mode": "{{mode}}", "on": {{on}}})],
    cnx: [],
)"#;

fn demo(dir: &Path) {
    write(dir, "plugin.ron", MANIFEST);
    write(dir, "fragments/main.ron", FRAGMENT);
}

#[test]
fn new_creates_a_plugin_that_checks_describes_and_pins() {
    let t = tempfile::tempdir().unwrap();
    let out = new_plugin(t.path(), "pid_loop").unwrap();
    contains(&out, "created");
    let dir = t.path().join("pid_loop");
    let report = check(&dir, &[], None).unwrap();
    contains(
        &report,
        "fragment 'main': ok (1 nodes, 0 connections, public: check_node)",
    );
    contains(&report, "pid-loop 0.1.0: ok");
    let described = describe(&dir).unwrap();
    contains(&described, "pid-loop 0.1.0");
    contains(&described, "label: Str - default");
    contains(&described, "public nodes: <instance>_node");
    assert!(pin(&dir).unwrap().starts_with("blake3:"));
    contains(
        &new_plugin(t.path(), "pid_loop").unwrap_err(),
        "already exists",
    );
    contains(&new_plugin(t.path(), "Bad Name").unwrap_err(), "must match");
}

#[test]
fn check_renders_with_samples_for_required_parameters_and_warns_about_unused_ones() {
    let t = tempfile::tempdir().unwrap();
    demo(t.path());
    let report = check(t.path(), &[], None).unwrap();
    contains(&report, "fragment 'main': ok");
    contains(
        &report,
        "warning: parameter 'unused' is not used by any fragment",
    );
    assert!(!report.contains("parameter 'gain' is not used"), "{report}");
}

#[test]
fn check_honors_overrides_and_reports_bad_values() {
    let t = tempfile::tempdir().unwrap();
    demo(t.path());
    assert!(check(t.path(), &[("rate".into(), ParamValue::Int(10))], None).is_ok());
    contains(
        &check(t.path(), &[("rate".into(), ParamValue::Int(1))], None).unwrap_err(),
        "below the minimum 5",
    );
    contains(
        &check(
            t.path(),
            &[("mode".into(), ParamValue::Str("z".into()))],
            None,
        )
        .unwrap_err(),
        "is not one of",
    );
    contains(
        &check(t.path(), &[("ghost".into(), ParamValue::Int(1))], None).unwrap_err(),
        "unknown parameter 'ghost'",
    );
}

#[test]
fn check_reports_every_broken_fragment() {
    let t = tempfile::tempdir().unwrap();
    let manifest = MANIFEST.replace(
        "fragments: {\"main\": (path: \"fragments/main.ron\", public: [\"node\"])},",
        "fragments: {\"main\": (path: \"fragments/main.ron\", public: [\"node\"]), \"bad\": (path: \"fragments/bad.ron\")},",
    );
    write(t.path(), "plugin.ron", &manifest);
    write(t.path(), "fragments/main.ron", FRAGMENT);
    write(
        t.path(),
        "fragments/bad.ron",
        "(tasks: [(id: \"unprefixed\", type: \"t::T\")], cnx: [])",
    );
    let e = check(t.path(), &[], None).unwrap_err();
    contains(&e, "fragment 'bad'");
    contains(&e, "must start with");
    assert!(
        !e.contains("fragment 'main'"),
        "the good fragment is not an error: {e}"
    );
}

#[test]
fn parse_override_reads_the_natural_type() {
    assert_eq!(parse_override("a=3").unwrap().1, ParamValue::Int(3));
    assert_eq!(parse_override("a=3.5").unwrap().1, ParamValue::Float(3.5));
    assert_eq!(parse_override("a=true").unwrap().1, ParamValue::Bool(true));
    assert_eq!(
        parse_override("a=hello").unwrap().1,
        ParamValue::Str("hello".into())
    );
    assert_eq!(
        parse_override("a=x=y").unwrap().1,
        ParamValue::Str("x=y".into())
    );
    contains(&parse_override("nokey").unwrap_err(), "key=value");
}

#[test]
fn the_pin_follows_the_content() {
    let t = tempfile::tempdir().unwrap();
    demo(t.path());
    let before = pin(t.path()).unwrap();
    assert_eq!(pin(t.path()).unwrap(), before);
    write(t.path(), "fragments/main.ron", &format!("{FRAGMENT}\n"));
    assert_ne!(pin(t.path()).unwrap(), before);
}

fn application(dir: &Path, cu_pid_version: Option<&str>) -> std::path::PathBuf {
    if let Some(version) = cu_pid_version {
        write(
            dir,
            "cu-pid/Cargo.toml",
            &format!("[package]\nname = \"cu-pid\"\nversion = \"{version}\"\nedition = \"2024\"\n"),
        );
        write(dir, "cu-pid/src/lib.rs", "");
    }
    let dependency = if cu_pid_version.is_some() {
        "cu-pid = { path = \"../cu-pid\" }\n"
    } else {
        ""
    };
    write(
        dir,
        "app/Cargo.toml",
        &format!(
            "[package]\nname = \"app\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n[dependencies]\n{dependency}"
        ),
    );
    write(dir, "app/src/lib.rs", "");
    dir.join("app/Cargo.toml")
}

#[test]
fn check_app_confirms_the_applications_dependencies() {
    let plugin = tempfile::tempdir().unwrap();
    demo(plugin.path());

    let ok = tempfile::tempdir().unwrap();
    let report = check(
        plugin.path(),
        &[],
        Some(&application(ok.path(), Some("1.3.2"))),
    )
    .unwrap();
    contains(&report, "rust: cu-pid 1.3.2 satisfies ^1.3");

    let missing = tempfile::tempdir().unwrap();
    contains(
        &check(plugin.path(), &[], Some(&application(missing.path(), None))).unwrap_err(),
        "does not depend on 'cu-pid'",
    );

    let wrong = tempfile::tempdir().unwrap();
    let e = check(
        plugin.path(),
        &[],
        Some(&application(wrong.path(), Some("2.0.0"))),
    )
    .unwrap_err();
    contains(
        &e,
        "'cu-pid' resolves to 2.0.0, which does not satisfy ^1.3",
    );
}

#[test]
fn expand_summarizes_an_application_with_plugins() {
    let t = tempfile::tempdir().unwrap();
    demo(&t.path().join("plugins/demo"));
    let pin = pin(&t.path().join("plugins/demo")).unwrap();
    write(
        t.path(),
        "app.ron",
        &format!(
            r#"(
    tasks: [(id: "camera", type: "t::Camera")],
    cnx: [(src: "camera", dst: "left_node", msg: "m::M")],
    plugins: [
        (path: "plugins/demo", fragment: "main", instance: "left", params: {{"rate": 10, "name": "l", "mode": "a", "on": true}}, pin: "{pin}"),
        (path: "plugins/demo", fragment: "main", instance: "right", params: {{"rate": 10, "name": "r", "mode": "b", "on": false}}, pin: "{pin}", when: Some(Feature("right"))),
    ],
)"#
        ),
    );
    let off = expand_config(&t.path().join("app.ron"), &[], true).unwrap();
    contains(&off, "left_node (t::T)");
    contains(&off, "camera -> left_node (m::M)");
    contains(&off, "instance: \"left\"");
    assert!(!off.contains("right_node"), "{off}");
    let on = expand_config(&t.path().join("app.ron"), &["right"], true).unwrap();
    contains(&on, "right_node (t::T)");
    let full = expand_config(&t.path().join("app.ron"), &[], false).unwrap();
    contains(&full, "resolved_plugins");
    contains(
        &expand_config(&t.path().join("none.ron"), &[], true).unwrap_err(),
        "Failed to read configuration file",
    );
}

#[test]
fn the_binary_reports_success_and_failure_through_its_exit_code() {
    let t = tempfile::tempdir().unwrap();
    demo(t.path());
    let bin = env!("CARGO_BIN_EXE_cu-plugin");
    let ok = Command::new(bin)
        .args(["pin"])
        .arg(t.path())
        .output()
        .unwrap();
    assert!(ok.status.success());
    assert!(String::from_utf8_lossy(&ok.stdout).starts_with("blake3:"));
    let missing = Command::new(bin)
        .args(["pin", "/nonexistent/plugin"])
        .output()
        .unwrap();
    assert!(!missing.status.success());
    contains(&String::from_utf8_lossy(&missing.stderr), "error:");
    let bad_param = Command::new(bin)
        .args(["check"])
        .arg(t.path())
        .args(["--param", "oops"])
        .output()
        .unwrap();
    assert!(!bad_param.status.success());
    contains(&String::from_utf8_lossy(&bad_param.stderr), "key=value");
}
