//! The engine's environment variables (HYG-E1).
//!
//! Every `IZWI_*` variable izwi-core reads is declared once in [`CORE_ENV`],
//! which also renders `docs/dev/env/core.md`; a test checks that every
//! `IZWI_*` literal in the crate's production source is registered.
//!
//! Booleans share one vocabulary, [`izwi_env::parse_bool`]: `true`, `1`,
//! `yes`, `on` and `false`, `0`, `no`, `off` (any case). A few switches also
//! take a registered word of their own (`legacy` for the fused-path kill
//! switches, for example). An empty value counts as unset. The runtime
//! constructors call [`validate_process_env`], so a boolean that does not
//! parse stops the engine at startup with an error naming the variable
//! instead of silently meaning true, false or the default.
//!
//! Hot paths never read the process environment: per-step and per-request
//! switches are read once through [`ReadOnce`] and cached for the life of
//! the process. Unit tests of this crate read them live, so a test can still
//! scope a variable around the code it exercises.

mod registry;

use std::sync::OnceLock;

pub use registry::CORE_ENV;

/// What kind of value a variable takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvKind {
    /// A boolean in the shared vocabulary, plus the variable's own aliases.
    Flag,
    /// `auto` or a boolean (true means `auto`, false means off).
    Mode,
    /// A non-negative integer.
    Int,
    /// A decimal number.
    Float,
    /// A duration in the unit the name gives.
    Duration,
    /// A size in bytes, or in the unit the name gives.
    Bytes,
    /// One of the words the description lists.
    Choice,
    /// A filesystem path.
    Path,
    /// Free text.
    Text,
    /// A comma-separated list.
    List,
}

impl EnvKind {
    fn label(self) -> &'static str {
        match self {
            Self::Flag => "flag",
            Self::Mode => "auto/flag",
            Self::Int => "integer",
            Self::Float => "number",
            Self::Duration => "duration",
            Self::Bytes => "bytes",
            Self::Choice => "choice",
            Self::Path => "path",
            Self::Text => "text",
            Self::List => "list",
        }
    }
}

/// When izwi-core reads a variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvRead {
    /// Once, when the engine (or the server's runtime config) is built.
    Startup,
    /// At each model load.
    Load,
    /// Once, at first use, then cached for the life of the process.
    Once,
    /// Declared here for a binary that reads it; izwi-core does not.
    Binary,
    /// Read only by izwi-core's own ignored or opt-in test harnesses.
    TestHarness,
}

impl EnvRead {
    fn label(self) -> &'static str {
        match self {
            Self::Startup => "startup",
            Self::Load => "model load",
            Self::Once => "first use",
            Self::Binary => "binary",
            Self::TestHarness => "test harness",
        }
    }
}

/// One documented izwi-core environment variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoreEnvVar {
    pub name: &'static str,
    pub kind: EnvKind,
    /// The behaviour when unset, as an operator would read it.
    pub default: &'static str,
    pub read: EnvRead,
    pub description: &'static str,
    /// Extra words a [`EnvKind::Flag`] or [`EnvKind::Mode`] accepts, with the
    /// boolean each means.
    pub aliases: &'static [(&'static str, bool)],
}

impl CoreEnvVar {
    pub(crate) const fn new(
        name: &'static str,
        kind: EnvKind,
        default: &'static str,
        read: EnvRead,
        description: &'static str,
    ) -> Self {
        Self {
            name,
            kind,
            default,
            read,
            description,
            aliases: &[],
        }
    }

    pub(crate) const fn aliases(mut self, aliases: &'static [(&'static str, bool)]) -> Self {
        self.aliases = aliases;
        self
    }

    /// Parses a set value of this variable as a boolean.
    fn parse_flag(&self, raw: &str) -> Result<bool, izwi_env::EnvError> {
        let value = raw.trim();
        if let Some((_, meaning)) = self
            .aliases
            .iter()
            .find(|(alias, _)| value.eq_ignore_ascii_case(alias))
        {
            return Ok(*meaning);
        }
        izwi_env::parse_bool(self.name, value)
    }

    fn validate(&self, raw: &str) -> Result<(), izwi_env::EnvError> {
        match self.kind {
            EnvKind::Flag => self.parse_flag(raw).map(drop),
            EnvKind::Mode if raw.trim().eq_ignore_ascii_case("auto") => Ok(()),
            EnvKind::Mode => self.parse_flag(raw).map(drop),
            _ => Ok(()),
        }
    }
}

/// The registry entry for `name`. Reading an unregistered variable is a bug
/// the registry test cannot see (the name may be built at run time), so it
/// panics in debug builds.
fn registered(name: &str) -> Option<&'static CoreEnvVar> {
    let var = CORE_ENV.iter().find(|var| var.name == name);
    debug_assert!(
        var.is_some(),
        "{name} is read but not registered in izwi_core::env::CORE_ENV"
    );
    var
}

/// The value of `name`, trimmed; `None` when unset, empty or not UTF-8.
pub fn value(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

/// A registered boolean: `None` when unset or empty. A value outside the
/// vocabulary also yields `None` (the caller's default): the runtime
/// constructors already refused to start with it.
pub fn flag(name: &str) -> Option<bool> {
    let raw = value(name)?;
    match registered(name) {
        Some(var) => var.parse_flag(&raw).ok(),
        None => izwi_env::parse_bool(name, &raw).ok(),
    }
}

/// A registered boolean, `default` when unset.
pub fn flag_or(name: &str, default: bool) -> bool {
    flag(name).unwrap_or(default)
}

/// Checks every registered boolean and `auto`/boolean variable that is set.
/// Called by the runtime constructors, so a value that does not parse stops
/// the engine at startup with an error naming the variable.
pub fn validate_process_env() -> crate::Result<()> {
    validate_with(value)
}

fn validate_with(lookup: impl Fn(&str) -> Option<String>) -> crate::Result<()> {
    let errors: Vec<String> = CORE_ENV
        .iter()
        .filter_map(|var| {
            let raw = lookup(var.name)?;
            var.validate(&raw).err().map(|error| match var.aliases {
                [] => error.to_string(),
                aliases => format!(
                    "{error} (or {})",
                    aliases
                        .iter()
                        .map(|(alias, _)| *alias)
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            })
        })
        .collect();
    if errors.is_empty() {
        Ok(())
    } else {
        Err(crate::Error::ConfigError(errors.join("; ")))
    }
}

/// A value read from the environment once, at first use, and cached for the
/// life of the process: what a per-step or per-request path consults instead
/// of the environment. This crate's own unit tests re-read on every call so
/// a test can scope a variable around the code it exercises.
pub(crate) struct ReadOnce<T> {
    cell: OnceLock<T>,
    read: fn() -> T,
}

impl<T: Clone> ReadOnce<T> {
    pub(crate) const fn new(read: fn() -> T) -> Self {
        Self {
            cell: OnceLock::new(),
            read,
        }
    }

    pub(crate) fn get(&self) -> T {
        if cfg!(test) {
            return (self.read)();
        }
        self.cell.get_or_init(self.read).clone()
    }
}

/// Renders `docs/dev/env/core.md` from [`CORE_ENV`].
pub fn render_reference() -> String {
    let mut out = String::from(
        "# izwi-core environment\n\n<!-- Generated from izwi_core::env::CORE_ENV; the \
         crate's env::tests::reference_doc_is_current test keeps it current. -->\n\n\
         The variables the engine library (`izwi-core`) reads, in every binary that \
         embeds it (`izwi-server`'s local role and `izwi-serving-worker`). The \
         binaries' own variables are in the sibling files.\n\n\
         Booleans (`flag`) accept `true`, `1`, `yes`, `on` or `false`, `0`, `no`, \
         `off`, in any case; a few also take the extra words listed against them. \
         `auto/flag` variables take `auto` or a boolean. An empty value counts as \
         unset. A boolean that does not parse stops the engine at startup with an \
         error naming the variable.\n\n\
         *Read* says when the engine reads the variable: at `startup`, at each \
         `model load`, at `first use` (then cached for the life of the process), by \
         a `binary` that embeds the engine, or only by an opt-in `test harness`. No \
         request or per-step path reads the environment.\n\n\
         | Variable | Kind | Default | Read | Meaning |\n|---|---|---|---|---|\n",
    );
    for var in CORE_ENV {
        let default = if var.default.is_empty() {
            "unset".to_string()
        } else if var.default.contains(' ') {
            var.default.to_string()
        } else {
            format!("`{}`", var.default)
        };
        let mut description = var.description.to_string();
        if !var.aliases.is_empty() {
            let aliases: Vec<String> = var
                .aliases
                .iter()
                .map(|(alias, meaning)| format!("`{alias}` means {meaning}"))
                .collect();
            description.push_str(&format!(" Also: {}.", aliases.join(", ")));
        }
        out.push_str(&format!(
            "| `{}` | {} | {} | {} | {} |\n",
            var.name,
            var.kind.label(),
            default,
            var.read.label(),
            description.replace('|', "\\|")
        ));
    }
    out
}

#[cfg(test)]
mod tests;
