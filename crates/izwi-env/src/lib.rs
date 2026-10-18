//! One environment-variable policy for Izwi (HYG-E1).
//!
//! Each binary, and the engine library, declares every `IZWI_*` variable it
//! reads in one registry of [`EnvVar`]s. A binary captures their values once
//! at startup into an [`EnvSnapshot`] and builds its typed configuration from
//! that snapshot, so no request path reads the process environment. Every
//! crate parses booleans the same way and fails closed: a boolean accepts
//! `true`, `false`, `1`, `0`, `yes`, `no`, `on` or `off` (any case) and
//! nothing else, and a value that does not parse is an error naming the
//! variable, never a silent default. The registry also renders each crate's
//! environment reference (`docs/dev/env/`), and a test per crate checks that
//! every `IZWI_*` name in its source is registered, so a new variable cannot
//! go undocumented.
//!
//! This crate has no dependencies beyond `thiserror`, so the engine
//! (`izwi-core`) and the accelerator-free serving crates share it without
//! either depending on the other.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{OsStr, OsString};
use std::fmt::Display;
use std::path::Path;
use std::str::FromStr;

/// One documented environment variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnvVar {
    pub name: &'static str,
    /// The value used when the variable is unset, as an operator would read it.
    pub default: &'static str,
    pub description: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EnvError {
    #[error("{name} is not valid UTF-8")]
    NotUnicode { name: String },
    #[error("{name} must be one of true, false, 1, 0, yes, no, on or off, not {value:?}")]
    InvalidBool { name: String, value: String },
    #[error("invalid {name}={value:?}: {reason}")]
    Invalid {
        name: String,
        value: String,
        reason: String,
    },
    #[error("{name} is required")]
    Missing { name: String },
}

/// Parses a boolean variable: `true`/`1`/`yes`/`on` or `false`/`0`/`no`/`off`
/// (any case, surrounding whitespace ignored). Anything else is an error.
pub fn parse_bool(name: &str, value: &str) -> Result<bool, EnvError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" => Ok(false),
        _ => Err(EnvError::InvalidBool {
            name: name.to_string(),
            value: value.to_string(),
        }),
    }
}

/// The registered variables' values, read from the environment once.
#[derive(Clone)]
pub struct EnvSnapshot {
    registry: &'static [EnvVar],
    values: BTreeMap<&'static str, OsString>,
}

impl std::fmt::Debug for EnvSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Values can be secrets; show only which variables are set.
        f.debug_set().entries(self.values.keys()).finish()
    }
}

impl EnvSnapshot {
    /// Reads every variable in `registry` from the process environment.
    pub fn capture(registry: &'static [EnvVar]) -> Self {
        let values = registry
            .iter()
            .filter_map(|var| std::env::var_os(var.name).map(|value| (var.name, value)))
            .collect();
        Self { registry, values }
    }

    /// A snapshot of explicit values, for tests. Every name must be registered.
    pub fn from_pairs<'a>(
        registry: &'static [EnvVar],
        pairs: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> Self {
        let values = pairs
            .into_iter()
            .map(|(name, value)| (registered(registry, name), OsString::from(value)))
            .collect();
        Self { registry, values }
    }

    /// The raw value of a registered variable, if set (paths may be any bytes).
    pub fn get_os(&self, name: &str) -> Option<&OsStr> {
        self.values
            .get(registered(self.registry, name))
            .map(OsString::as_os_str)
    }

    /// The value of a registered variable, if set; non-UTF-8 is an error.
    pub fn get(&self, name: &str) -> Result<Option<&str>, EnvError> {
        self.get_os(name)
            .map(|value| {
                value.to_str().ok_or_else(|| EnvError::NotUnicode {
                    name: name.to_string(),
                })
            })
            .transpose()
    }

    /// The value, or `fallback` when unset.
    pub fn get_or<'a>(&'a self, name: &str, fallback: &'a str) -> Result<&'a str, EnvError> {
        Ok(self.get(name)?.unwrap_or(fallback))
    }

    /// A set, non-blank value; an error otherwise.
    pub fn required(&self, name: &str) -> Result<&str, EnvError> {
        self.get(name)?
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| EnvError::Missing {
                name: name.to_string(),
            })
    }

    /// A boolean, `None` when unset.
    pub fn flag(&self, name: &str) -> Result<Option<bool>, EnvError> {
        self.get(name)?
            .map(|value| parse_bool(name, value))
            .transpose()
    }

    /// A boolean, `fallback` when unset.
    pub fn flag_or(&self, name: &str, fallback: bool) -> Result<bool, EnvError> {
        Ok(self.flag(name)?.unwrap_or(fallback))
    }

    /// A parsed value (surrounding whitespace ignored), `None` when unset.
    pub fn parse<T>(&self, name: &str) -> Result<Option<T>, EnvError>
    where
        T: FromStr,
        T::Err: Display,
    {
        self.get(name)?
            .map(|value| {
                value
                    .trim()
                    .parse()
                    .map_err(|error: T::Err| EnvError::Invalid {
                        name: name.to_string(),
                        value: value.to_string(),
                        reason: error.to_string(),
                    })
            })
            .transpose()
    }

    /// A parsed value, `fallback` when unset.
    pub fn parse_or<T>(&self, name: &str, fallback: T) -> Result<T, EnvError>
    where
        T: FromStr,
        T::Err: Display,
    {
        Ok(self.parse(name)?.unwrap_or(fallback))
    }
}

fn registered(registry: &'static [EnvVar], name: &str) -> &'static str {
    match registry.iter().find(|var| var.name == name) {
        Some(var) => var.name,
        None => panic!("{name} is read but not registered in this binary's environment reference"),
    }
}

/// Renders a binary's environment reference as Markdown.
pub fn render_reference(title: &str, intro: &str, vars: &[EnvVar]) -> String {
    let mut out = format!(
        "# {title}\n\n<!-- Generated from the binary's environment registry; \
         the crate's env_reference test keeps it current. -->\n\n{intro}\n\n\
         | Variable | Default | Meaning |\n|---|---|---|\n"
    );
    for var in vars {
        // A literal default is code; a described one ("the engine default")
        // is prose.
        let default = if var.default.is_empty() {
            "unset".to_string()
        } else if var.default.contains(' ') {
            var.default.to_string()
        } else {
            format!("`{}`", var.default)
        };
        out.push_str(&format!(
            "| `{}` | {} | {} |\n",
            var.name,
            default,
            var.description.replace('|', "\\|")
        ));
    }
    out
}

/// The `IZWI_*` string literals in Rust source text: what a binary's
/// environment reference must cover.
pub fn izwi_literals(source: &str) -> BTreeSet<&str> {
    let mut names = BTreeSet::new();
    let mut rest = source;
    while let Some(start) = rest.find("\"IZWI_") {
        let candidate = &rest[start + 1..];
        let end = candidate
            .find(|c: char| !(c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_'))
            .unwrap_or(candidate.len());
        if end > "IZWI_".len() && candidate[end..].starts_with('"') {
            names.insert(&candidate[..end]);
        }
        rest = &candidate[end..];
    }
    names
}

/// The `IZWI_*` names in a crate's production source under `source_dir`:
/// every `.rs` file except test files (`tests.rs`, `*_tests.rs`), each read
/// up to its top-level `#[cfg(test)] mod` block.
pub fn production_env_names(source_dir: &Path) -> BTreeSet<String> {
    production_env_names_excluding(source_dir, &[])
}

/// [`production_env_names`], skipping the `excluded` files and directories
/// (such as the registry module that lists every name).
pub fn production_env_names_excluding(source_dir: &Path, excluded: &[&Path]) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut pending = vec![source_dir.to_path_buf()];
    while let Some(dir) = pending.pop() {
        let entries = std::fs::read_dir(&dir)
            .unwrap_or_else(|error| panic!("read {}: {error}", dir.display()));
        for entry in entries {
            let path = entry.expect("directory entry").path();
            if excluded.contains(&path.as_path()) {
                continue;
            }
            let file_name = path.file_name().and_then(OsStr::to_str).unwrap_or_default();
            if path.is_dir() {
                pending.push(path);
            } else if file_name.ends_with(".rs")
                && file_name != "tests.rs"
                && !file_name.ends_with("_tests.rs")
            {
                let source = std::fs::read_to_string(&path)
                    .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
                names.extend(
                    izwi_literals(without_test_modules(&source))
                        .into_iter()
                        .map(str::to_string),
                );
            }
        }
    }
    names
}

/// The source before its first top-level inline `#[cfg(test)] mod name {`
/// block. A `#[cfg(test)] mod name;` declaration (whose file the walk reads
/// on its own) does not end the production source: code after it still
/// counts.
fn without_test_modules(source: &str) -> &str {
    let mut offset = 0;
    let mut lines = source.split_inclusive('\n').peekable();
    while let Some(line) = lines.next() {
        if line.trim_end() == "#[cfg(test)]"
            && lines
                .peek()
                .is_some_and(|next| next.starts_with("mod ") && next.trim_end().ends_with('{'))
        {
            return &source[..offset];
        }
        offset += line.len();
    }
    source
}

#[cfg(test)]
mod tests {
    use super::*;

    static REGISTRY: &[EnvVar] = &[
        EnvVar {
            name: "IZWI_TEST_FLAG",
            default: "false",
            description: "A flag.",
        },
        EnvVar {
            name: "IZWI_TEST_COUNT",
            default: "4",
            description: "A count | with a pipe.",
        },
        EnvVar {
            name: "IZWI_TEST_NAME",
            default: "",
            description: "A name.",
        },
        EnvVar {
            name: "IZWI_TEST_BUDGET",
            default: "sized at load",
            description: "A budget.",
        },
    ];

    #[test]
    fn booleans_accept_one_vocabulary_and_fail_closed() {
        for value in ["true", "TRUE", " 1 ", "yes", "On"] {
            assert_eq!(parse_bool("IZWI_X", value), Ok(true), "{value}");
        }
        for value in ["false", "0", "No", "OFF"] {
            assert_eq!(parse_bool("IZWI_X", value), Ok(false), "{value}");
        }
        for value in ["", "2", "enabled", "y", "truthy"] {
            let error = parse_bool("IZWI_X", value).unwrap_err();
            assert!(
                error.to_string().starts_with("IZWI_X must be one of"),
                "{error}"
            );
        }
    }

    #[test]
    fn snapshot_values_parse_or_fail_with_the_variable_name() {
        let env = EnvSnapshot::from_pairs(
            REGISTRY,
            [("IZWI_TEST_FLAG", "yes"), ("IZWI_TEST_COUNT", " 8 ")],
        );
        assert_eq!(env.flag("IZWI_TEST_FLAG"), Ok(Some(true)));
        assert_eq!(env.parse_or("IZWI_TEST_COUNT", 4u32), Ok(8));
        assert_eq!(env.get("IZWI_TEST_NAME"), Ok(None));
        assert_eq!(env.get_or("IZWI_TEST_NAME", "fallback"), Ok("fallback"));
        assert_eq!(
            env.required("IZWI_TEST_NAME").unwrap_err().to_string(),
            "IZWI_TEST_NAME is required"
        );

        let bad = EnvSnapshot::from_pairs(
            REGISTRY,
            [("IZWI_TEST_FLAG", "maybe"), ("IZWI_TEST_COUNT", "four")],
        );
        assert!(bad.flag("IZWI_TEST_FLAG").is_err());
        let error = bad.parse::<u32>("IZWI_TEST_COUNT").unwrap_err();
        assert!(error
            .to_string()
            .starts_with("invalid IZWI_TEST_COUNT=\"four\""));
    }

    #[test]
    #[should_panic(expected = "IZWI_TEST_UNKNOWN is read but not registered")]
    fn reading_an_unregistered_variable_is_a_bug() {
        let _ = EnvSnapshot::from_pairs(REGISTRY, []).get("IZWI_TEST_UNKNOWN");
    }

    #[test]
    fn debug_output_hides_values() {
        let env = EnvSnapshot::from_pairs(REGISTRY, [("IZWI_TEST_NAME", "secret-value")]);
        let shown = format!("{env:?}");
        assert!(shown.contains("IZWI_TEST_NAME") && !shown.contains("secret-value"));
    }

    #[test]
    fn reference_renders_a_table() {
        let doc = render_reference("Test", "Intro.", REGISTRY);
        assert!(doc.contains("| `IZWI_TEST_FLAG` | `false` | A flag. |"));
        assert!(doc.contains("| `IZWI_TEST_COUNT` | `4` | A count \\| with a pipe. |"));
        assert!(doc.contains("| `IZWI_TEST_NAME` | unset | A name. |"));
        assert!(doc.contains("| `IZWI_TEST_BUDGET` | sized at load | A budget. |"));
    }

    #[test]
    fn test_modules_are_left_out() {
        let source = "#[cfg(test)]\nmod attach_tests;\nconst A: &str = \"IZWI_A\";\n#[cfg(test)]\nfn helper() {}\nconst B: &str = \"IZWI_B\";\n#[cfg(test)]\nmod tests {\n    const C: &str = \"IZWI_C\";\n}\n";
        assert_eq!(
            izwi_literals(without_test_modules(source))
                .into_iter()
                .collect::<Vec<_>>(),
            vec!["IZWI_A", "IZWI_B"]
        );
    }

    #[test]
    fn literals_are_whole_quoted_izwi_names() {
        let source = r#"
            const A: &str = "IZWI_ALPHA";
            let b = env.get("IZWI_BETA_2");
            // "IZWI_" alone, or a prefix like "IZWI_GAMMA_{}" is not a name.
            let c = format!("IZWI_GAMMA_{}", x);
        "#;
        assert_eq!(
            izwi_literals(source).into_iter().collect::<Vec<_>>(),
            vec!["IZWI_ALPHA", "IZWI_BETA_2"]
        );
    }
}
