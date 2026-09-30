use std::collections::BTreeMap;

use crate::error::{PluginError, Result};

/// Replaces `{{name}}` placeholders with the text in `vars`.
///
/// Every placeholder must be defined. Text that only resembles a placeholder (for example a RON map
/// that happens to contain `{{`) is left alone.
pub fn render(template: &str, vars: &BTreeMap<String, String>) -> Result<String> {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    let mut undefined: Vec<String> = Vec::new();
    while let Some(start) = rest.find("{{") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        match parse_placeholder(after) {
            Some((name, consumed)) => {
                if let Some(value) = vars.get(name) {
                    out.push_str(value);
                } else if !undefined.iter().any(|u| u == name) {
                    undefined.push(name.to_owned());
                }
                rest = &after[consumed..];
            }
            None => {
                out.push_str("{{");
                rest = after;
            }
        }
    }
    out.push_str(rest);
    if undefined.is_empty() {
        Ok(out)
    } else {
        let defined: Vec<&str> = vars.keys().map(String::as_str).collect();
        Err(PluginError::new(format!(
            "fragment uses undefined placeholder(s) {} (defined: {})",
            undefined
                .iter()
                .map(|u| format!("{{{{{u}}}}}"))
                .collect::<Vec<_>>()
                .join(", "),
            defined.join(", ")
        )))
    }
}

/// Parses `  name  }}` and returns the name and the number of bytes consumed.
fn parse_placeholder(text: &str) -> Option<(&str, usize)> {
    let trimmed = text.trim_start_matches(' ');
    let lead = text.len() - trimmed.len();
    let end = trimmed
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .unwrap_or(trimmed.len());
    if end == 0 || !trimmed.as_bytes()[0].is_ascii_alphabetic() && trimmed.as_bytes()[0] != b'_' {
        return None;
    }
    let name = &trimmed[..end];
    let tail = trimmed[end..].trim_start_matches(' ');
    let spaces_after = trimmed[end..].len() - tail.len();
    tail.strip_prefix("}}")?;
    Some((name, lead + end + spaces_after + 2))
}
