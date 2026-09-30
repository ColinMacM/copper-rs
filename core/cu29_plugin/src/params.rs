use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::error::{PluginError, Result};
pub use crate::record::ParamValue;

/// The type of a parameter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParamKind {
    Bool,
    Int,
    Float,
    Str,
    Enum(Vec<String>),
}

impl ParamKind {
    fn name(&self) -> &'static str {
        match self {
            Self::Bool => "Bool",
            Self::Int => "Int",
            Self::Float => "Float",
            Self::Str => "Str",
            Self::Enum(_) => "Enum",
        }
    }
}

/// Declaration of one parameter in a manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParamSpec {
    pub kind: ParamKind,
    #[serde(default)]
    pub default: Option<ParamValue>,
    #[serde(default)]
    pub min: Option<f64>,
    #[serde(default)]
    pub max: Option<f64>,
}

/// Largest magnitude an `Int` bound may have so it converts to `f64` without rounding.
const EXACT_INT_LIMIT: f64 = 9_007_199_254_740_992.0;

/// Text that can sit between the quotes of a RON string without changing the document structure.
pub(crate) fn check_string_safe(value: &str) -> std::result::Result<(), String> {
    if let Some(c) = value
        .chars()
        .find(|c| *c == '"' || *c == '\\' || c.is_control())
    {
        return Err(format!(
            "contains {c:?}, which cannot be placed inside a RON string"
        ));
    }
    Ok(())
}

impl ParamSpec {
    /// Checks the declaration itself: bounds belong to the kind, and the default satisfies the spec.
    pub(crate) fn validate(&self, name: &str) -> Result<()> {
        let bad = |m: String| Err(PluginError::new(format!("parameter '{name}': {m}")));
        let numeric = matches!(self.kind, ParamKind::Int | ParamKind::Float);
        if !numeric && (self.min.is_some() || self.max.is_some()) {
            return bad(format!(
                "min/max apply to Int and Float, not {}",
                self.kind.name()
            ));
        }
        for bound in [self.min, self.max].into_iter().flatten() {
            if !bound.is_finite() {
                return bad("min/max must be finite".into());
            }
            if matches!(self.kind, ParamKind::Int)
                && (bound.fract() != 0.0 || bound.abs() > EXACT_INT_LIMIT)
            {
                return bad(format!(
                    "Int bound {bound} is not an exactly representable integer"
                ));
            }
        }
        if let (Some(min), Some(max)) = (self.min, self.max)
            && min > max
        {
            return bad(format!("min {min} is greater than max {max}"));
        }
        if let ParamKind::Enum(choices) = &self.kind {
            if choices.is_empty() {
                return bad("Enum needs at least one choice".into());
            }
            for choice in choices {
                if let Err(why) = check_string_safe(choice) {
                    return bad(format!("Enum choice {choice:?} {why}"));
                }
            }
        }
        if let Some(default) = &self.default {
            self.resolve(default).map_err(|e| {
                PluginError::new(format!("parameter '{name}': default is invalid: {e}"))
            })?;
        }
        Ok(())
    }

    /// Converts a supplied value to the canonical value for this kind, or explains why it does not fit.
    fn resolve(&self, value: &ParamValue) -> std::result::Result<ParamValue, String> {
        let check_range = |v: f64| -> std::result::Result<(), String> {
            if let Some(min) = self.min
                && v < min
            {
                return Err(format!("{v} is below the minimum {min}"));
            }
            if let Some(max) = self.max
                && v > max
            {
                return Err(format!("{v} is above the maximum {max}"));
            }
            Ok(())
        };
        match (&self.kind, value) {
            (ParamKind::Bool, ParamValue::Bool(v)) => Ok(ParamValue::Bool(*v)),
            (ParamKind::Int, ParamValue::Int(v)) => {
                // Bounds were validated to be exact integers, so compare in integers: going
                // through f64 would round a value beyond 2^53 onto its bound.
                #[allow(clippy::cast_possible_truncation)]
                if let Some(min) = self.min
                    && *v < min as i64
                {
                    return Err(format!("{v} is below the minimum {min}"));
                }
                #[allow(clippy::cast_possible_truncation)]
                if let Some(max) = self.max
                    && *v > max as i64
                {
                    return Err(format!("{v} is above the maximum {max}"));
                }
                Ok(ParamValue::Int(*v))
            }
            (ParamKind::Float, ParamValue::Float(v)) => {
                if !v.is_finite() {
                    return Err("must be a finite number".into());
                }
                check_range(*v)?;
                Ok(ParamValue::Float(*v))
            }
            (ParamKind::Float, ParamValue::Int(v)) => {
                if v.unsigned_abs() > EXACT_INT_LIMIT as u64 {
                    return Err(format!(
                        "{v} cannot be converted to a float without rounding"
                    ));
                }
                #[allow(clippy::cast_precision_loss)]
                let f = *v as f64;
                check_range(f)?;
                Ok(ParamValue::Float(f))
            }
            (ParamKind::Str, ParamValue::Str(v)) => {
                check_string_safe(v)?;
                Ok(ParamValue::Str(v.clone()))
            }
            (ParamKind::Enum(choices), ParamValue::Str(v)) => {
                if choices.contains(v) {
                    Ok(ParamValue::Str(v.clone()))
                } else {
                    Err(format!("{v:?} is not one of {choices:?}"))
                }
            }
            (kind, other) => Err(format!("expected {}, got {other}", kind.name())),
        }
    }
}

/// Validates supplied parameters against the declarations and returns every parameter's final value.
///
/// Unknown names, missing required parameters, wrong kinds and out-of-range values are errors.
pub fn resolve_params(
    specs: &BTreeMap<String, ParamSpec>,
    given: &BTreeMap<String, ParamValue>,
) -> Result<BTreeMap<String, ParamValue>> {
    if let Some(unknown) = given.keys().find(|k| !specs.contains_key(*k)) {
        let known: Vec<&str> = specs.keys().map(String::as_str).collect();
        return Err(PluginError::new(format!(
            "unknown parameter '{unknown}' (declared: {})",
            if known.is_empty() {
                "none".to_owned()
            } else {
                known.join(", ")
            }
        )));
    }
    let mut out = BTreeMap::new();
    for (name, spec) in specs {
        let value = match (given.get(name), &spec.default) {
            (Some(v), _) => v,
            (None, Some(d)) => d,
            (None, None) => {
                return Err(PluginError::new(format!(
                    "required parameter '{name}' ({}) has no value",
                    spec.kind.name()
                )));
            }
        };
        let resolved = spec
            .resolve(value)
            .map_err(|why| PluginError::new(format!("parameter '{name}': {why}")))?;
        out.insert(name.clone(), resolved);
    }
    Ok(out)
}

/// The RON text a resolved value contributes to a fragment.
///
/// Strings are inserted without quotes, like `includes` parameters, so a template writes the
/// quotes itself; `check_string_safe` guarantees the text cannot close them early.
pub(crate) fn render_value(value: &ParamValue) -> String {
    match value {
        ParamValue::Bool(v) => v.to_string(),
        ParamValue::Int(v) => v.to_string(),
        ParamValue::Float(v) => format!("{v:?}"),
        ParamValue::Str(v) => v.clone(),
    }
}
