//! Every environment variable the supervisor process reads (HYG-E1).
//!
//! `run_from_cli` captures these once at startup; the supervisor reads no
//! other `IZWI_*` variable. Worker bearer tokens and the gateway admin key
//! are read from the variables the node config names. The variables the
//! supervisor sets for each worker are the worker's own (see
//! `docs/dev/env/serving-worker.md`); `docs/dev/env/serving-supervisor.md`
//! is rendered from this registry and the supervisor-owned worker list.

use izwi_serving_protocol::env::{EnvSnapshot, EnvVar};

pub(crate) const LOG_FORMAT_ENV: &str = "IZWI_LOG_FORMAT";
/// A path the supervisor keeps a Prometheus textfile at (for a node
/// exporter's textfile collector); unset writes none.
pub(crate) const METRICS_TEXTFILE_ENV: &str = "IZWI_SUPERVISOR_METRICS_TEXTFILE";

pub(crate) static SUPERVISOR_ENV: &[EnvVar] = &[
    EnvVar {
        name: LOG_FORMAT_ENV,
        default: "text",
        description: "Log line format on stderr: `text` or `json`.",
    },
    EnvVar {
        name: METRICS_TEXTFILE_ENV,
        default: "",
        description: "Path of a Prometheus textfile the supervisor keeps current (for a node \
                      exporter's textfile collector); unset writes none.",
    },
];

/// The environment `run_from_cli` reads once at startup.
pub(crate) fn capture() -> EnvSnapshot {
    EnvSnapshot::capture(SUPERVISOR_ENV)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::{INHERITED_ENV_ALLOWLIST, SUPERVISOR_OWNED_ENVIRONMENT};
    use izwi_serving_protocol::env::{production_env_names, render_reference};
    use std::path::Path;

    const REFERENCE: &str = "docs/dev/env/serving-supervisor.md";

    fn reference() -> String {
        let mut doc = render_reference(
            "izwi-serving-supervisor environment",
            "The variables `izwi-serving-supervisor` reads, once, at startup. Worker \
             bearer tokens and the gateway admin key come from the variables the node \
             config names (`bearer_token_env`, `gateway_ack.admin_key_env`).",
            SUPERVISOR_ENV,
        );
        doc.push_str(
            "\n## Set for each worker\n\nA worker starts with a cleared environment: the \
             inherited allowlist below, the node config's `[workers.environment]` table, \
             then these variables, which the supervisor owns and a table may not set. \
             Their meaning is in [serving-worker.md](serving-worker.md).\n\n",
        );
        for name in SUPERVISOR_OWNED_ENVIRONMENT {
            doc.push_str(&format!("- `{name}`\n"));
        }
        doc.push_str("\nInherited from the supervisor when set:\n\n");
        for name in INHERITED_ENV_ALLOWLIST {
            doc.push_str(&format!("- `{name}`\n"));
        }
        doc
    }

    /// Every `IZWI_*` name in this crate's production source is either read
    /// (registered above) or owned for workers.
    #[test]
    fn every_variable_in_the_source_is_documented() {
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let undocumented: Vec<String> = production_env_names(&source)
            .into_iter()
            .filter(|name| {
                SUPERVISOR_ENV.iter().all(|var| var.name != name)
                    && !SUPERVISOR_OWNED_ENVIRONMENT.contains(&name.as_str())
            })
            .collect();
        assert!(
            undocumented.is_empty(),
            "register these in daemon/env_config.rs or env.rs: {undocumented:?}"
        );
    }

    /// `docs/dev/env/serving-supervisor.md` is rendered from the registry.
    /// Run with `UPDATE_ENV_REFERENCE=1` to rewrite it.
    #[test]
    fn reference_doc_is_current() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .unwrap();
        let path = root.join(REFERENCE);
        let expected = reference();
        if std::env::var_os("UPDATE_ENV_REFERENCE").is_some() {
            std::fs::write(&path, &expected).unwrap();
        }
        let actual = std::fs::read_to_string(&path).unwrap_or_default();
        assert!(
            actual == expected,
            "{REFERENCE} is stale; rerun with UPDATE_ENV_REFERENCE=1"
        );
    }
}
