//! Static plugins for Copper.
//!
//! A plugin is a directory with a `plugin.ron` manifest, configuration fragments and assets.
//! An application selects plugins in its `copperconfig.ron`; Copper expands them while reading
//! the configuration. See `doc/static-plugins.md`.
//!
//! The `expand` feature (on by default) provides the manifest model, parameter validation,
//! placeholder rendering, content pins and the checks applied to an expanded instance. Without it
//! the crate provides only [`Provenance`], the record a configuration carries.

mod record;

pub use record::{Connection, ParamValue, Provenance};

#[cfg(feature = "expand")]
mod error;
#[cfg(feature = "expand")]
mod expand;
#[cfg(feature = "expand")]
mod fragment;
#[cfg(feature = "expand")]
mod load;
#[cfg(feature = "expand")]
mod manifest;
#[cfg(feature = "expand")]
mod params;
#[cfg(feature = "expand")]
mod pin;
#[cfg(feature = "expand")]
mod render;

#[cfg(feature = "expand")]
pub use error::{PluginError, Result};
#[cfg(feature = "expand")]
pub use expand::{
    ExistingIds, Expanded, Host, PluginUse, check_collisions, check_encapsulation,
    check_instance_names, check_resource_access, expand, plugin_dir,
};
#[cfg(feature = "expand")]
pub use fragment::{FragmentInfo, inspect};
#[cfg(feature = "expand")]
pub use load::{LoadedPlugin, MAX_FILE_BYTES};
#[cfg(feature = "expand")]
pub use manifest::{
    FragmentSpec, MANIFEST_FILE, MANIFEST_FORMAT, Manifest, RESERVED_PLACEHOLDERS, RustDependency,
};
#[cfg(feature = "expand")]
pub use params::{ParamKind, ParamSpec, resolve_params};
#[cfg(feature = "expand")]
pub use pin::{PIN_PREFIX, content_pin, pin_of, pinned_files};
#[cfg(feature = "expand")]
pub use render::render;
