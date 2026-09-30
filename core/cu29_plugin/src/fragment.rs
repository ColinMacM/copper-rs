use ron::Value;

use crate::record::Connection;

use crate::error::{PluginError, Result};
use crate::manifest::ron_options;

/// Sections a fragment may contain. Everything else describes the application as a whole.
const ALLOWED_SECTIONS: [&str; 4] = ["tasks", "bridges", "resources", "cnx"];

/// What a rendered fragment declares.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FragmentInfo {
    pub tasks: Vec<String>,
    pub bridges: Vec<String>,
    pub resources: Vec<String>,
    pub connections: Vec<Connection>,
}

impl FragmentInfo {
    /// Every node id the fragment declares, sorted.
    #[must_use]
    pub fn nodes(&self) -> Vec<String> {
        let mut nodes: Vec<String> = self
            .tasks
            .iter()
            .chain(&self.bridges)
            .chain(&self.resources)
            .cloned()
            .collect();
        nodes.sort();
        nodes
    }
}

fn field<'a>(map: &'a ron::Map, name: &str) -> Option<&'a Value> {
    map.iter().find_map(|(k, v)| match k {
        Value::String(key) if key == name => Some(v),
        _ => None,
    })
}

fn text(value: &Value) -> Option<&str> {
    match value {
        Value::String(s) => Some(s),
        Value::Option(Some(inner)) => text(inner),
        _ => None,
    }
}

fn entries<'a>(section: &str, value: &'a Value) -> Result<&'a [Value]> {
    match value {
        Value::Seq(items) => Ok(items),
        Value::Option(Some(inner)) => entries(section, inner),
        Value::Option(None) => Ok(&[]),
        _ => Err(PluginError::new(format!(
            "section '{section}' must be a list"
        ))),
    }
}

fn ids(section: &str, value: &Value) -> Result<Vec<String>> {
    entries(section, value)?
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let Value::Map(map) = entry else {
                return Err(PluginError::new(format!("{section}[{i}] must be a struct")));
            };
            field(map, "id")
                .and_then(text)
                .map(str::to_owned)
                .ok_or_else(|| PluginError::new(format!("{section}[{i}] has no string 'id'")))
        })
        .collect()
}

/// Reads the ids and connections out of a rendered fragment and rejects sections that belong to the application.
pub fn inspect(rendered: &str) -> Result<FragmentInfo> {
    let root: Value = ron_options().from_str(rendered).map_err(|e| {
        PluginError::new(format!(
            "fragment does not parse: {} at position {}",
            e.code, e.span
        ))
    })?;
    let Value::Map(root) = root else {
        return Err(PluginError::new(
            "fragment must be a struct such as `(tasks: [...], cnx: [...])`",
        ));
    };
    let mut info = FragmentInfo::default();
    for (key, value) in root.iter() {
        let Value::String(name) = key else {
            return Err(PluginError::new("fragment has a non-field key"));
        };
        match name.as_str() {
            "tasks" => info.tasks = ids("tasks", value)?,
            "bridges" => info.bridges = ids("bridges", value)?,
            "resources" => info.resources = ids("resources", value)?,
            "cnx" => {
                for (i, entry) in entries("cnx", value)?.iter().enumerate() {
                    let Value::Map(map) = entry else {
                        return Err(PluginError::new(format!("cnx[{i}] must be a struct")));
                    };
                    let end = |which: &str| {
                        field(map, which)
                            .and_then(text)
                            .map(str::to_owned)
                            .ok_or_else(|| {
                                PluginError::new(format!("cnx[{i}] has no string '{which}'"))
                            })
                    };
                    info.connections.push(Connection {
                        src: end("src")?,
                        dst: end("dst")?,
                        msg: end("msg")?,
                    });
                }
            }
            other => {
                return Err(PluginError::new(format!(
                    "section '{other}' is not allowed in a fragment; allowed: {}",
                    ALLOWED_SECTIONS.join(", ")
                )));
            }
        }
    }
    Ok(info)
}
