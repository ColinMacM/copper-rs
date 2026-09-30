//! Without the `plugins` feature a configuration that names plugins is refused with a message,
//! and a configuration that only carries the record of expanded plugins still reads.
#![cfg(all(feature = "std", not(feature = "plugins")))]

use cu29_runtime::config::read_configuration_str;

#[test]
fn naming_plugins_without_the_feature_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.ron");
    std::fs::write(
        &path,
        r#"(tasks: [], cnx: [], plugins: [(path: "p", fragment: "f", instance: "a", dev: true)])"#,
    )
    .unwrap();
    let e = cu29_runtime::config::read_configuration(path.to_str().unwrap())
        .expect_err("plugins need the feature")
        .to_string();
    assert!(e.contains("without its `plugins` feature"), "{e}");
}

#[test]
fn an_effective_configuration_with_plugin_records_reads_without_the_feature() {
    let effective = r#"(
        tasks: [(id: "a_node", type: "t::T")],
        cnx: [],
        resolved_plugins: [(
            id: "p", version: "0.1.0", fragment: "f", instance: "a", pin: "blake3:00",
            params: {"x": "1"}, nodes: ["a_node"], public: ["a_node"], connections: [],
        )],
    )"#;
    let config = read_configuration_str(effective.to_owned(), None).unwrap();
    assert_eq!(config.plugins.len(), 1);
    assert_eq!(config.plugins[0].params["x"], "1");
}
