//! Static plugins: expansion while a configuration is read. See `doc/static-plugins.md`.
//! Needs the `plugins` feature: `cargo test -p cu29-runtime --features plugins --test plugins`.
#![cfg(feature = "plugins")]
#[cfg(all(test, feature = "std"))]
mod tests {
    use cu29_runtime::config::{
        CuConfig, read_configuration, read_configuration_str, read_configuration_with_features,
        read_configuration_with_resolved_ron, read_configuration_with_resolved_ron_and_features,
    };
    use std::fs::{create_dir_all, write};
    use std::path::{Path, PathBuf};
    use tempfile::{TempDir, tempdir};

    const MANIFEST: &str = r#"(
    format: 1,
    id: "cu-pid-loop",
    version: "0.2.0",
    description: "Single-axis PID loop",
    copper: ">=1.3.0-dev",
    params: {
        "rate_hz": (kind: Int, min: 1, max: 10000, default: 100),
        "gain_p": (kind: Float, default: 1.0),
        "label": (kind: Str),
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
            type: "tasks::Pid",
            config: {"kp": {{gain_p}}, "rate_hz": {{rate_hz}}, "label": "{{label}}"},
        ),
        (id: "{{instance}}_out", type: "tasks::Out"),
    ],
    cnx: [
        (src: "{{instance}}_pid", dst: "{{instance}}_out", msg: "messages::Cmd"),
    ],
)"#;

    struct App {
        dir: TempDir,
    }

    impl App {
        fn new() -> Self {
            let app = Self {
                dir: tempdir().unwrap(),
            };
            app.write("plugins/pid_loop/plugin.ron", MANIFEST);
            app.write("plugins/pid_loop/fragments/loop.ron", FRAGMENT);
            app.write("plugins/pid_loop/python/policy.py", "print('x')\n");
            app
        }

        fn root(&self) -> &Path {
            self.dir.path()
        }

        fn write(&self, relative: &str, text: &str) -> PathBuf {
            let path = self.dir.path().join(relative);
            create_dir_all(path.parent().unwrap()).unwrap();
            write(&path, text).unwrap();
            path
        }

        fn pin(&self) -> String {
            let dir = self.root().join("plugins/pid_loop");
            let manifest = cu29_plugin::Manifest::load(&dir).unwrap();
            cu29_plugin::content_pin(&dir, &manifest).unwrap()
        }

        fn config(&self, body: &str) -> PathBuf {
            self.write("copperconfig.ron", body)
        }
    }

    fn ids(config: &CuConfig) -> Vec<String> {
        let graph = config.get_graph(None).unwrap();
        let mut ids: Vec<String> = graph
            .get_all_nodes()
            .iter()
            .map(|(_, n)| n.get_id())
            .collect();
        ids.sort();
        ids
    }

    fn app_with_entry(app: &App, entry: &str, extra: &str) -> PathBuf {
        app.config(&format!(
            r#"(
    tasks: [
        (id: "camera", type: "tasks::Camera"),
        (id: "logger", type: "tasks::Logger"),
    ],
    cnx: [
        (src: "camera", dst: "elbow_pid", msg: "messages::Feedback"),
        {extra}
    ],
    plugins: [{entry}],
)"#
        ))
    }

    fn entry(app: &App, instance: &str, extra_fields: &str) -> String {
        format!(
            r#"(path: "plugins/pid_loop", fragment: "loop", instance: "{instance}", params: {{"label": "{instance}"}}, pin: "{}"{extra_fields})"#,
            app.pin()
        )
    }

    fn error_of(path: &Path) -> String {
        read_configuration(path.to_str().unwrap())
            .expect_err("expected an error")
            .to_string()
    }

    fn contains(text: &str, needle: &str) {
        assert!(text.contains(needle), "{needle:?} not found in:\n{text}");
    }

    #[test]
    fn a_plugin_expands_into_the_application_graph() {
        let app = App::new();
        let path = app_with_entry(&app, &entry(&app, "elbow", ""), "");
        let config = read_configuration(path.to_str().unwrap()).unwrap();
        assert_eq!(ids(&config), ["camera", "elbow_out", "elbow_pid", "logger"]);

        let graph = config.get_graph(None).unwrap();
        let edges: Vec<(String, String)> = graph
            .edges()
            .map(|c| (c.src.clone(), c.dst.clone()))
            .collect();
        assert!(
            edges.contains(&("camera".into(), "elbow_pid".into())),
            "application connects to the public node: {edges:?}"
        );
        assert!(
            edges.contains(&("elbow_pid".into(), "elbow_out".into())),
            "the fragment's own connection: {edges:?}"
        );

        let pid = graph
            .get_all_nodes()
            .into_iter()
            .find(|(_, n)| n.get_id() == "elbow_pid")
            .unwrap()
            .1
            .clone();
        assert_eq!(pid.get_type(), "tasks::Pid");
        assert_eq!(
            pid.get_param::<f64>("kp").unwrap(),
            Some(1.0),
            "defaults render as floats"
        );
        assert_eq!(pid.get_param::<u32>("rate_hz").unwrap(), Some(100));
        assert_eq!(
            pid.get_param::<String>("label").unwrap().as_deref(),
            Some("elbow")
        );
    }

    #[test]
    fn the_resolved_configuration_records_the_plugin_and_round_trips() {
        let app = App::new();
        let path = app_with_entry(&app, &entry(&app, "elbow", ""), "");
        let (_, resolved) = read_configuration_with_resolved_ron(path.to_str().unwrap()).unwrap();
        contains(&resolved, "resolved_plugins");
        contains(&resolved, "\"cu-pid-loop\"");
        contains(&resolved, "\"0.2.0\"");
        contains(&resolved, &app.pin());
        contains(&resolved, "elbow_pid");
        assert!(
            !resolved
                .lines()
                .any(|l| l.trim_start().starts_with("plugins:")),
            "the input entries are consumed, only the record remains:\n{resolved}"
        );

        // What the log stores must read back without the plugin directory.
        let replayed = read_configuration_str(resolved, None).unwrap();
        assert_eq!(
            ids(&replayed),
            ["camera", "elbow_out", "elbow_pid", "logger"]
        );
    }

    #[test]
    fn the_effective_configuration_carries_the_plugin_records() {
        let app = App::new();
        let path = app_with_entry(&app, &entry(&app, "elbow", ""), "");
        let config = read_configuration(path.to_str().unwrap()).unwrap();
        assert_eq!(config.plugins.len(), 1);
        assert_eq!(config.plugins[0].instance, "elbow");
        assert_eq!(config.plugins[0].pin, app.pin());

        // The runtime writes `serialize_ron()` into the unified log as the effective configuration.
        let effective = config.serialize_ron().unwrap();
        contains(&effective, "resolved_plugins");
        contains(&effective, &app.pin());
        let replayed = read_configuration_str(effective, None).unwrap();
        assert_eq!(
            replayed.plugins, config.plugins,
            "the records survive a read-back"
        );
        assert_eq!(ids(&replayed), ids(&config));
    }

    #[test]
    fn a_configuration_without_plugins_serializes_without_plugin_records() {
        let app = App::new();
        let path = app.config(r#"(tasks: [(id: "a", type: "t::A")], cnx: [])"#);
        let config = read_configuration(path.to_str().unwrap()).unwrap();
        assert!(config.plugins.is_empty());
        let effective = config.serialize_ron().unwrap();
        assert!(!effective.contains("plugins"), "{effective}");
    }

    #[test]
    fn a_configuration_without_plugins_resolves_exactly_as_before() {
        let app = App::new();
        let path = app.config(r#"(tasks: [(id: "a", type: "t::A")], cnx: [])"#);
        let (_, resolved) = read_configuration_with_resolved_ron(path.to_str().unwrap()).unwrap();
        assert!(!resolved.contains("plugins"), "{resolved}");
    }

    #[test]
    fn one_plugin_can_be_instantiated_twice() {
        let app = App::new();
        let both = format!("{}, {}", entry(&app, "elbow", ""), entry(&app, "wrist", ""));
        let path = app_with_entry(&app, &both, "");
        let config = read_configuration(path.to_str().unwrap()).unwrap();
        assert_eq!(
            ids(&config),
            [
                "camera",
                "elbow_out",
                "elbow_pid",
                "logger",
                "wrist_out",
                "wrist_pid"
            ]
        );
    }

    #[test]
    fn typed_parameters_reach_the_task_config() {
        let app = App::new();
        let e = format!(
            r#"(path: "plugins/pid_loop", fragment: "loop", instance: "elbow", params: {{"label": "arm", "gain_p": 2, "rate_hz": 250}}, pin: "{}")"#,
            app.pin()
        );
        let config = read_configuration(app_with_entry(&app, &e, "").to_str().unwrap()).unwrap();
        let graph = config.get_graph(None).unwrap();
        let pid = graph
            .get_all_nodes()
            .into_iter()
            .find(|(_, n)| n.get_id() == "elbow_pid")
            .unwrap()
            .1
            .clone();
        assert_eq!(
            pid.get_param::<f64>("kp").unwrap(),
            Some(2.0),
            "an integer for a Float parameter is still a float"
        );
        assert_eq!(pid.get_param::<u32>("rate_hz").unwrap(), Some(250));
    }

    #[test]
    fn an_id_the_application_already_declares_is_an_error_not_a_silent_skip() {
        let app = App::new();
        let path = app.config(&format!(
            r#"(tasks: [(id: "elbow_pid", type: "tasks::Mine")], cnx: [], plugins: [{}])"#,
            entry(&app, "elbow", "")
        ));
        let e = error_of(&path);
        contains(&e, "plugin 'cu-pid-loop' instance 'elbow'");
        contains(
            &e,
            "node id \"elbow_pid\" is already declared by the application",
        );
    }

    #[test]
    fn a_plugin_id_that_an_include_declares_is_an_error() {
        let app = App::new();
        app.write(
            "more.ron",
            r#"(tasks: [(id: "elbow_out", type: "tasks::Other")], cnx: [])"#,
        );
        let path = app.config(&format!(
            r#"(tasks: [], cnx: [], includes: [(path: "more.ron", params: {{}})], plugins: [{}])"#,
            entry(&app, "elbow", "")
        ));
        contains(
            &error_of(&path),
            "\"elbow_out\" is already declared by the application",
        );
    }

    #[test]
    fn the_same_instance_name_twice_names_the_real_cause() {
        let app = App::new();
        let twice = format!("{}, {}", entry(&app, "elbow", ""), entry(&app, "elbow", ""));
        let e = error_of(&app_with_entry(&app, &twice, ""));
        contains(
            &e,
            "instance name 'elbow' is used by more than one plugin entry",
        );
    }

    #[test]
    fn connecting_to_a_private_node_from_the_application_is_an_error() {
        let app = App::new();
        let path = app_with_entry(
            &app,
            &entry(&app, "elbow", ""),
            r#"(src: "camera", dst: "elbow_out", msg: "messages::Cmd"),"#,
        );
        let e = error_of(&path);
        contains(&e, "reaches private node \"elbow_out\"");
        contains(&e, "public nodes (elbow_pid)");
        contains(&e, "plugin 'cu-pid-loop' instance 'elbow'");
    }

    #[test]
    fn reading_from_a_private_node_is_an_error_too() {
        let app = App::new();
        let path = app_with_entry(
            &app,
            &entry(&app, "elbow", ""),
            r#"(src: "elbow_out", dst: "logger", msg: "messages::Cmd"),"#,
        );
        contains(&error_of(&path), "reaches private node \"elbow_out\"");
    }

    #[test]
    fn the_pin_is_required_and_checked() {
        let app = App::new();
        let no_pin = r#"(path: "plugins/pid_loop", fragment: "loop", instance: "elbow", params: {"label": "x"})"#;
        let e = error_of(&app_with_entry(&app, no_pin, ""));
        contains(&e, "has no pin");
        contains(&e, &app.pin());

        let wrong = r#"(path: "plugins/pid_loop", fragment: "loop", instance: "elbow", params: {"label": "x"}, pin: "blake3:0000000000000000000000000000000000000000000000000000000000000000")"#;
        let e = error_of(&app_with_entry(&app, wrong, ""));
        contains(&e, "pin mismatch");
        contains(&e, &app.pin());

        let dev = r#"(path: "plugins/pid_loop", fragment: "loop", instance: "elbow", params: {"label": "x"}, dev: true)"#;
        let path = app_with_entry(&app, dev, "");
        let (config, resolved) =
            read_configuration_with_resolved_ron(path.to_str().unwrap()).unwrap();
        assert_eq!(ids(&config).len(), 4);
        contains(&resolved, "dev: true");
    }

    #[test]
    fn editing_a_pinned_plugin_stops_the_build_until_the_pin_is_updated() {
        let app = App::new();
        let path = app_with_entry(&app, &entry(&app, "elbow", ""), "");
        assert!(read_configuration(path.to_str().unwrap()).is_ok());
        app.write("plugins/pid_loop/python/policy.py", "print('edited')\n");
        contains(&error_of(&path), "pin mismatch");
        let repinned = app_with_entry(&app, &entry(&app, "elbow", ""), "");
        assert!(read_configuration(repinned.to_str().unwrap()).is_ok());
    }

    #[test]
    fn when_selects_a_plugin_by_feature() {
        let app = App::new();
        let gated = format!(
            r#"{}, (path: "plugins/pid_loop", fragment: "loop", instance: "wrist", params: {{"label": "w"}}, pin: "{}", when: Some(Feature("wrist")))"#,
            entry(&app, "elbow", ""),
            app.pin()
        );
        let path = app_with_entry(&app, &gated, "");
        let off = read_configuration_with_features(path.to_str().unwrap(), &[]).unwrap();
        assert!(!ids(&off).contains(&"wrist_pid".to_owned()));
        let on = read_configuration_with_features(path.to_str().unwrap(), &["wrist"]).unwrap();
        assert!(ids(&on).contains(&"wrist_pid".to_owned()));
        let (_, resolved) =
            read_configuration_with_resolved_ron_and_features(path.to_str().unwrap(), &["wrist"])
                .unwrap();
        contains(&resolved, "wrist");
    }

    #[test]
    fn a_plugin_named_in_an_included_file_resolves_relative_to_that_file() {
        let app = App::new();
        app.write("sub/robot.ron", &format!(
            r#"(tasks: [(id: "camera", type: "tasks::Camera")], cnx: [(src: "camera", dst: "elbow_pid", msg: "m::F")], plugins: [(path: "../plugins/pid_loop", fragment: "loop", instance: "elbow", params: {{"label": "x"}}, pin: "{}")])"#,
            app.pin()
        ));
        let path = app.config(r#"(tasks: [(id: "logger", type: "tasks::Logger")], cnx: [], includes: [(path: "sub/robot.ron", params: {})])"#);
        let (config, resolved) =
            read_configuration_with_resolved_ron(path.to_str().unwrap()).unwrap();
        assert_eq!(ids(&config), ["camera", "elbow_out", "elbow_pid", "logger"]);
        contains(&resolved, "resolved_plugins");
        contains(&resolved, &app.pin());
    }

    #[test]
    fn mistyped_entry_fields_and_bad_parameters_are_reported_at_the_entry() {
        let app = App::new();
        let typo = r#"(path: "plugins/pid_loop", fragment: "loop", instanse: "elbow", dev: true)"#;
        contains(
            &error_of(&app_with_entry(&app, typo, "")),
            "Unexpected field named `instanse` in `PluginConfig`",
        );
        let bad_param = format!(
            r#"(path: "plugins/pid_loop", fragment: "loop", instance: "elbow", params: {{"label": "x", "rate_hz": 0}}, pin: "{}")"#,
            app.pin()
        );
        contains(
            &error_of(&app_with_entry(&app, &bad_param, "")),
            "parameter 'rate_hz': 0 is below the minimum 1",
        );
        let unknown = format!(
            r#"(path: "plugins/pid_loop", fragment: "loop", instance: "elbow", params: {{"label": "x", "gain": 1.0}}, pin: "{}")"#,
            app.pin()
        );
        contains(
            &error_of(&app_with_entry(&app, &unknown, "")),
            "unknown parameter 'gain'",
        );
    }

    #[test]
    fn a_plugin_that_needs_a_newer_copper_is_refused() {
        let app = App::new();
        app.write(
            "plugins/pid_loop/plugin.ron",
            &MANIFEST.replace(">=1.3.0-dev", ">=99.0.0"),
        );
        let e = error_of(&app_with_entry(&app, &entry(&app, "elbow", ""), ""));
        contains(&e, "requires Copper >=99.0.0");
    }

    #[test]
    fn a_fragment_cannot_override_application_settings() {
        let app = App::new();
        app.write(
            "plugins/pid_loop/fragments/loop.ron",
            &FRAGMENT.replace(
                "cnx: [",
                "logging: (enable_task_logging: false),\n    cnx: [",
            ),
        );
        let e = error_of(&app_with_entry(&app, &entry(&app, "elbow", ""), ""));
        contains(&e, "section 'logging' is not allowed in a fragment");
    }

    #[test]
    fn a_missing_plugin_directory_is_reported() {
        let app = App::new();
        let e = error_of(&app_with_entry(
            &app,
            r#"(path: "plugins/missing", fragment: "loop", instance: "elbow", dev: true)"#,
            "",
        ));
        contains(&e, "cannot read");
        contains(&e, "plugin.ron");
    }

    /// The provenance block, which plugins control. Key order inside a task's `config` map is
    /// not part of it.
    fn provenance_of(resolved: &str) -> &str {
        &resolved[resolved
            .find("resolved_plugins")
            .expect("resolved_plugins present")..]
    }

    #[test]
    fn plugin_expansion_is_deterministic() {
        let app = App::new();
        let both = format!("{}, {}", entry(&app, "elbow", ""), entry(&app, "wrist", ""));
        let path = app_with_entry(&app, &both, "");
        let (first_config, first) =
            read_configuration_with_resolved_ron(path.to_str().unwrap()).unwrap();
        for _ in 0..10 {
            let (config, again) =
                read_configuration_with_resolved_ron(path.to_str().unwrap()).unwrap();
            assert_eq!(provenance_of(&first), provenance_of(&again));
            assert_eq!(ids(&first_config), ids(&config));
            let order = |c: &CuConfig| -> Vec<String> {
                c.get_graph(None)
                    .unwrap()
                    .get_all_nodes()
                    .iter()
                    .map(|(_, n)| n.get_id())
                    .collect()
            };
            assert_eq!(order(&first_config), order(&config), "node order is stable");
        }
    }

    #[test]
    fn an_outside_node_cannot_bind_a_private_resource_bundle() {
        let app = App::new();
        app.write(
            "plugins/pid_loop/fragments/loop.ron",
            &FRAGMENT.replace(
                "tasks: [",
                "resources: [(id: \"{{instance}}_bus\", provider: \"p::Bus\")],\n    tasks: [",
            ),
        );
        let path = app_with_entry(&app, &entry(&app, "elbow", ""), "");
        assert!(
            read_configuration(path.to_str().unwrap()).is_ok(),
            "the plugin alone is fine"
        );
        let text = std::fs::read_to_string(&path).unwrap().replace(
            "tasks: [",
            "tasks: [\n        (id: \"t2\", type: \"tasks::T\", resources: {\"bus\": \"elbow_bus.r\"}),",
        );
        let stolen = app.config(&text);
        let e = error_of(&stolen);
        contains(&e, "node \"t2\" binds resource bundle \"elbow_bus\"");
        contains(&e, "plugin 'cu-pid-loop' instance 'elbow'");
    }

    #[test]
    fn a_connection_with_another_message_type_does_not_borrow_the_fragments_exemption() {
        let app = App::new();
        let path = app_with_entry(
            &app,
            &entry(&app, "elbow", ""),
            r#"(src: "elbow_pid", dst: "elbow_out", msg: "other::Msg"),"#,
        );
        contains(&error_of(&path), "reaches private node \"elbow_out\"");
    }

    #[test]
    fn plugins_in_a_string_configuration_are_an_error_not_silently_dropped() {
        let app = App::new();
        let text = format!(
            r#"(tasks: [(id: "a", type: "t::A")], cnx: [], plugins: [{}])"#,
            entry(&app, "elbow", "")
        );
        let e = read_configuration_str(text, None)
            .expect_err("must not drop the plugin")
            .to_string();
        contains(&e, "no file path");
    }

    #[test]
    fn a_user_written_resolved_plugins_block_is_rejected() {
        let app = App::new();
        let forged = app.config(
            r#"(tasks: [], cnx: [], resolved_plugins: [(id: "trusted", version: "1.0.0", fragment: "f", instance: "x", pin: "blake3:00", params: {}, nodes: [], public: [], connections: [])])"#,
        );
        contains(
            &error_of(&forged),
            "`resolved_plugins` is written by Copper",
        );
    }

    #[test]
    fn the_files_that_were_read_are_reported_for_rebuild_tracking() {
        let app = App::new();
        app.write("more.ron", r#"(tasks: [], cnx: [])"#);
        let path = app.config(&format!(
            r#"(tasks: [], cnx: [], includes: [(path: "more.ron", params: {{}})], plugins: [{}])"#,
            entry(&app, "elbow", "")
        ));
        let (_, _, files) =
            cu29_runtime::config::read_configuration_with_resolved_ron_files_and_features(
                path.to_str().unwrap(),
                &[],
            )
            .unwrap();
        let has = |suffix: &str| files.iter().any(|f| f.ends_with(suffix));
        assert!(has("copperconfig.ron"), "{files:?}");
        assert!(has("more.ron"), "{files:?}");
        assert!(has("plugin.ron"), "{files:?}");
        assert!(has("fragments/loop.ron"), "{files:?}");
        assert!(has("python/policy.py"), "{files:?}");
    }
}
