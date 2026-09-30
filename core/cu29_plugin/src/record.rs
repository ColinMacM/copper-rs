use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

/// One connection a plugin fragment declares, with the endpoints and message type as written.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Connection {
    pub src: String,
    pub dst: String,
    pub msg: String,
}

/// The record of one expanded plugin instance.
///
/// It is written into the resolved configuration and from there into the effective configuration
/// that the unified log stores, so a recorded run identifies the plugin content that produced it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    pub id: String,
    pub version: String,
    pub fragment: String,
    pub instance: String,
    pub pin: String,
    /// True when the instance was accepted without a pin.
    #[serde(default)]
    pub dev: bool,
    /// Every parameter's final value, as the text that was placed into the fragment.
    pub params: BTreeMap<String, String>,
    /// Every node id the instance declares (tasks, bridges and resource bundles), sorted.
    pub nodes: Vec<String>,
    /// The resource bundle ids among `nodes`.
    #[serde(default)]
    pub resources: Vec<String>,
    /// The node ids the application may connect to or bind.
    pub public: Vec<String>,
    /// The connections the fragment itself declares, sorted.
    pub connections: Vec<Connection>,
}

/// A value supplied for, or defaulted into, a plugin parameter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ParamValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
}

impl fmt::Display for ParamValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bool(v) => write!(f, "{v}"),
            Self::Int(v) => write!(f, "{v}"),
            Self::Float(v) => write!(f, "{v:?}"),
            Self::Str(v) => write!(f, "{v:?}"),
        }
    }
}
