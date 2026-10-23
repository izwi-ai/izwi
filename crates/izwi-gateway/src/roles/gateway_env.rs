//! The gateway role's environment reference (HYG-E1).
//!
//! The gateway reads two kinds of variables, all once at startup:
//! command-line flags that clap also takes from an environment variable
//! (`env = "IZWI_..."` on `GatewayCli`; `izwi-server --role gateway` takes
//! the same `GatewayArgs`), and the variables below, which the listener,
//! perimeter, fleet, quota, approvals, TLS and principal-key configurations
//! read through their `from_env` constructors.
//! Every value is validated and fails closed; booleans use the shared
//! `izwi_serving_protocol::env::parse_bool` vocabulary.
//! `docs/dev/env/gateway.md` is rendered from both, and the tests keep it in
//! step with the gateway sources. The module is test-only: the registry is
//! the reference's source, not a reader.

use izwi_serving_protocol::env::EnvVar;

const fn var(name: &'static str, default: &'static str, description: &'static str) -> EnvVar {
    EnvVar {
        name,
        default,
        description,
    }
}

/// The gateway variables read outside clap.
pub static GATEWAY_ENV: &[EnvVar] = &[
    // Listener (config.rs; izwi-server's gateway role reads them through
    // its runtime configuration)
    var(
        "IZWI_TIMEOUT",
        "300",
        "Deadline of one request in seconds (legacy REQUEST_TIMEOUT_SECS).",
    ),
    var(
        "IZWI_CORS",
        "false",
        "Allow cross-origin browser requests from IZWI_CORS_ORIGINS.",
    ),
    var(
        "IZWI_CORS_ORIGINS",
        "*",
        "Comma-separated allowed origins; the gateway refuses a wildcard while CORS is on.",
    ),
    var(
        "IZWI_OPENAI_COMPAT_PROFILE",
        "strict",
        "OpenAI compatibility profile: `strict` or `relaxed`.",
    ),
    // Public perimeter (gateway_security.rs)
    var(
        "IZWI_GATEWAY_API_KEY_REF",
        "env:IZWI_GATEWAY_API_KEY",
        "`env:VARIABLE` or `file:/absolute/path` naming the public inference API key (16-4096 printable bytes).",
    ),
    var(
        "IZWI_GATEWAY_METRICS_API_KEY_REF",
        "",
        "`env:VARIABLE` or `file:/absolute/path` naming the /metrics key; unset turns /metrics off. Must differ from the inference key.",
    ),
    var(
        "IZWI_GATEWAY_ADMIN_API_KEY_REF",
        "",
        "`env:VARIABLE` or `file:/absolute/path` naming the admin API key; unset turns the admin API off. Must differ from the inference key.",
    ),
    var(
        "IZWI_GATEWAY_API_PRINCIPAL_ID",
        "gateway-api-key",
        "Principal id of the perimeter key.",
    ),
    var(
        "IZWI_GATEWAY_TENANT_ID",
        "default",
        "Tenant of the perimeter key.",
    ),
    var(
        "IZWI_GATEWAY_MAX_CHAT_BODY_BYTES",
        "1048576",
        "Largest chat request body (1 KiB to 32 MiB).",
    ),
    var(
        "IZWI_GATEWAY_TRUSTED_INGRESS_TLS",
        "false",
        "TLS terminates at a trusted ingress, allowing a non-loopback plaintext bind.",
    ),
    var(
        "IZWI_GATEWAY_PRINCIPAL_KEYS_MANIFEST",
        "",
        "Path of the scoped principal-key manifest; unset leaves the durable key store untouched.",
    ),
    var(
        "IZWI_DB_PATH",
        "the platform data directory's izwi.sqlite3",
        "SQLite store for scoped principal keys (read only when a manifest is set).",
    ),
    var(
        "IZWI_DATABASE_URL",
        "",
        "Database URL for scoped principal keys instead of IZWI_DB_PATH.",
    ),
    // Fleet coordination (gateway_fleet)
    var(
        "IZWI_GATEWAY_FLEET_SIZE",
        "",
        "Gateways in a partitioned fleet; each takes 1/N of every per-tenant budget.",
    ),
    var(
        "IZWI_GATEWAY_FLEET_PARTITION",
        "",
        "This gateway's partition index (0..N); required with IZWI_GATEWAY_FLEET_SIZE.",
    ),
    var(
        "IZWI_GATEWAY_FLEET_DB_PATH",
        "",
        "Shared coordination store (SQLite path or postgres:// URL with db-postgres) for fleet capacity claims.",
    ),
    var(
        "IZWI_GATEWAY_ID",
        "a random id per boot",
        "This gateway's identity for fleet claim ownership.",
    ),
    var(
        "IZWI_GATEWAY_FLEET_CLAIM_TTL_MS",
        "30000",
        "Lifetime of a fleet capacity claim.",
    ),
    var(
        "IZWI_GATEWAY_FLEET_CLAIM_TIMEOUT_MS",
        "100",
        "How long a claim may wait on the coordination store.",
    ),
    // Tenant quotas (izwi-gateway)
    var(
        "IZWI_GATEWAY_TENANT_REQUESTS_PER_MINUTE",
        "600",
        "Per-tenant request rate.",
    ),
    var(
        "IZWI_GATEWAY_TENANT_BURST_REQUESTS",
        "64",
        "Per-tenant request burst.",
    ),
    var(
        "IZWI_GATEWAY_TENANT_TOKENS_PER_MINUTE",
        "",
        "Per-tenant token rate; unset leaves tokens unlimited.",
    ),
    var(
        "IZWI_GATEWAY_MAX_TRACKED_TENANTS",
        "16384",
        "Tenants the rate limiter tracks at once.",
    ),
    var(
        "IZWI_GATEWAY_TENANT_MAX_CONCURRENT",
        "8 (at most the gateway's max in flight)",
        "Concurrent requests per tenant.",
    ),
    // Shared approvals and worker transport (izwi-gateway)
    var(
        "IZWI_GATEWAY_SHARED_APPROVALS_PATH",
        "",
        "Absolute path of a fleet-wide worker approvals file every gateway reads.",
    ),
    var(
        "IZWI_GATEWAY_SHARED_APPROVALS_TTL_MS",
        "30000",
        "How often the shared approvals file is re-read (1 s to 1 h).",
    ),
    var(
        "IZWI_GATEWAY_WORKER_TLS_CA_REFS",
        "",
        "Comma-separated `file:/absolute/path` private CA roots for worker TLS.",
    ),
    var(
        "IZWI_GATEWAY_WORKER_TLS_CLIENT_CERT_REF",
        "",
        "`file:/absolute/path` client certificate for worker mTLS (with the key).",
    ),
    var(
        "IZWI_GATEWAY_WORKER_TLS_CLIENT_KEY_REF",
        "",
        "`file:/absolute/path` client private key for worker mTLS (warned about when other users can read it).",
    ),
    // Shutdown (izwi-gateway shutdown.rs)
    var(
        "IZWI_DESKTOP_OWNER_PIPE",
        "",
        "`1` when a parent process owns the gateway: stdin closing shuts it down.",
    ),
    var(
        "IZWI_OTLP_TRACES_ENDPOINT",
        "",
        "OTLP/HTTP collector for request spans; unset exports none.",
    ),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::GatewayCli;
    use clap::CommandFactory;
    use izwi_serving_protocol::env::{production_env_names, render_reference};
    use std::collections::BTreeSet;
    use std::path::{Path, PathBuf};

    const REFERENCE: &str = "docs/dev/env/gateway.md";

    /// Names that only test modules read; the source scan sees their files.
    const TEST_ONLY: &[&str] = &["IZWI_TEST_FLEET_PG_URL", "IZWI_UPDATE_SSE_GOLDENS"];

    /// Flags clap also reads from an environment variable: (variable,
    /// flag, default, help).
    fn gateway_flags() -> Vec<(String, String, String, String)> {
        GatewayCli::command()
            .get_arguments()
            .filter_map(|arg| {
                let env = arg.get_env()?.to_str()?.to_string();
                let defaults: Vec<_> = arg
                    .get_default_values()
                    .iter()
                    .filter_map(|value| value.to_str())
                    .collect();
                let default = if defaults.is_empty() {
                    "unset".to_string()
                } else {
                    format!("`{}`", defaults.join(","))
                };
                let help = arg
                    .get_help()
                    .map(|help| help.to_string().replace('\n', " ").replace('|', "\\|"))
                    .unwrap_or_default();
                let flag = format!("`--{}`", arg.get_long().unwrap_or_default());
                Some((env, flag, default, help))
            })
            .collect()
    }

    fn reference() -> String {
        let mut doc = render_reference(
            "izwi-gateway environment",
            "The gateway (`izwi-gateway`, or `izwi-server --role gateway`) reads these \
             variables once, at startup. Every value is validated and a bad one stops the \
             gateway with an error. Booleans accept `true`, `false`, `1`, `0`, `yes`, `no`, \
             `on` or `off`. `izwi-server --role gateway` takes the listener variables \
             (`IZWI_HOST`, `IZWI_PORT`, `IZWI_TIMEOUT`, `IZWI_CORS*`) through its runtime \
             configuration, and the role from `--role` or `IZWI_SERVER_ROLE`.",
            GATEWAY_ENV,
        );
        doc.push_str(
            "\n## Flags with an environment variable\n\nEach of these flags can also be \
             set through its variable; the flag wins when both are given.\n\n\
             | Variable | Flag | Default | Meaning |\n|---|---|---|---|\n",
        );
        for (env, flag, default, help) in gateway_flags() {
            doc.push_str(&format!("| `{env}` | {flag} | {default} | {help} |\n"));
        }
        doc
    }

    fn crates_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf()
    }

    fn documented() -> BTreeSet<String> {
        GATEWAY_ENV
            .iter()
            .map(|var| var.name.to_string())
            .chain(gateway_flags().into_iter().map(|(env, ..)| env))
            .chain(TEST_ONLY.iter().map(|name| name.to_string()))
            .collect()
    }

    /// Every gateway variable in the source is documented: each
    /// `IZWI_GATEWAY_*` name in izwi-server and each `IZWI_*` name in the
    /// izwi-gateway crate.
    #[test]
    fn every_gateway_variable_in_the_source_is_documented() {
        let crates = crates_dir();
        let names = production_env_names(&crates.join("izwi-server/src"))
            .into_iter()
            .filter(|name| name.starts_with("IZWI_GATEWAY_"))
            .chain(production_env_names(&crates.join("izwi-gateway/src")));
        let documented = documented();
        let undocumented: Vec<String> = names.filter(|name| !documented.contains(name)).collect();
        assert!(
            undocumented.is_empty(),
            "document these in roles/gateway_env.rs: {undocumented:?}"
        );
    }

    /// Every documented variable is still read somewhere.
    #[test]
    fn every_documented_variable_exists() {
        let crates = crates_dir();
        let mut names = production_env_names(&crates.join("izwi-server/src"));
        names.extend(production_env_names(&crates.join("izwi-gateway/src")));
        names.extend(production_env_names(
            &crates.join("izwi-serving-client/src"),
        ));
        let stale: Vec<&EnvVar> = GATEWAY_ENV
            .iter()
            .filter(|var| !names.contains(var.name))
            .collect();
        assert!(stale.is_empty(), "no source reads {stale:?}");
    }

    /// `docs/dev/env/gateway.md` is rendered from the registry and clap.
    /// Run with `UPDATE_ENV_REFERENCE=1` to rewrite it.
    #[test]
    fn reference_doc_is_current() {
        let path = crates_dir().parent().unwrap().join(REFERENCE);
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
