use std::fmt;

/// A problem found while reading, validating or expanding a plugin.
///
/// The message names the plugin (and instance, when known) so it can be shown as-is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginError {
    message: String,
}

impl PluginError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// Prefix the message with `plugin '<id>'`.
    pub(crate) fn in_plugin(self, id: &str) -> Self {
        Self::new(format!("plugin '{id}': {}", self.message))
    }

    /// Prefix the message with `plugin '<id>' instance '<instance>'`.
    pub(crate) fn in_instance(self, id: &str, instance: &str) -> Self {
        Self::new(format!(
            "plugin '{id}' instance '{instance}': {}",
            self.message
        ))
    }

    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for PluginError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for PluginError {}

pub type Result<T> = std::result::Result<T, PluginError>;
