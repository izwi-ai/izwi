use super::*;
use crate::cli::ServerArgs;
use clap::CommandFactory;
use izwi_serving_protocol::env::{production_env_names, render_reference};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

const REFERENCE: &str = "docs/dev/env/server.md";

/// Names that only test modules read; the source scan sees their files.
const TEST_ONLY: &[&str] = &["IZWI_TEST_STORE_PG_URL", "IZWI_UPDATE_GATEWAY_OPENAPI"];

/// Compile-time variables: baked into the binary by the build, never read
/// from the running process's environment.
static BUILD_ENV: &[EnvVar] = &[var(
    "IZWI_BUILD_GIT_SHA",
    "unknown",
    "Git commit reported by /internal/health and the start-up log.",
)];

/// Flags clap also reads from an environment variable:
/// (variable, flag, default, help).
fn env_flags() -> Vec<(String, String, String, String)> {
    ServerArgs::command()
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
        "izwi-server local role environment",
        "The local role (`izwi-server`, the default `--role local`) reads these variables \
         once, at start-up; no request reads the environment. Request-path settings are \
         validated when the server starts, and a bad value stops it with an error naming \
         the variable. Booleans accept `true`, `false`, `1`, `0`, `yes`, `no`, `on` or \
         `off`. The engine's own variables (`IZWI_HOST`, `IZWI_PORT`, `IZWI_MAX_*` engine \
         limits and so on) are read by izwi-core and listed in core.md; the gateway \
         role's are in gateway.md.",
        SERVER_ENV,
    );
    doc.push_str(
        "\n## Flags with an environment variable\n\nEach of these flags can also be set \
         through its variable; the flag wins when both are given.\n\n\
         | Variable | Flag | Default | Meaning |\n|---|---|---|---|\n",
    );
    for (env, flag, default, help) in env_flags() {
        if !env.starts_with("IZWI_GATEWAY_") {
            doc.push_str(&format!("| `{env}` | {flag} | {default} | {help} |\n"));
        }
    }
    doc.push_str(
        "\n## Build-time variables\n\nRead by the compiler when the binary is built, not \
         from the running process.\n\n| Variable | Default | Meaning |\n|---|---|---|\n",
    );
    for var in BUILD_ENV {
        doc.push_str(&format!(
            "| `{}` | `{}` | {} |\n",
            var.name, var.default, var.description
        ));
    }
    doc
}

fn crate_src() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

/// Every `IZWI_*` name in this crate's source is documented: in this
/// registry or as a clap flag (izwi-gateway documents the gateway's).
#[test]
fn every_variable_in_the_source_is_documented() {
    let documented: BTreeSet<String> = SERVER_ENV
        .iter()
        .chain(BUILD_ENV)
        .map(|var| var.name.to_string())
        .chain(env_flags().into_iter().map(|(env, ..)| env))
        .chain(TEST_ONLY.iter().map(|name| name.to_string()))
        .collect();
    let undocumented: Vec<String> = production_env_names(&crate_src())
        .into_iter()
        .filter(|name| !documented.contains(name))
        .collect();
    assert!(
        undocumented.is_empty(),
        "document these in env_config.rs: {undocumented:?}"
    );
}

/// No name is registered twice or also documented as a flag.
#[test]
fn every_registered_variable_is_listed_once() {
    let flags: BTreeSet<String> = env_flags().into_iter().map(|(env, ..)| env).collect();
    let mut seen = BTreeSet::new();
    for var in SERVER_ENV.iter().chain(BUILD_ENV) {
        assert!(seen.insert(var.name), "{} is registered twice", var.name);
        assert!(!flags.contains(var.name), "{} is also a flag", var.name);
    }
}

/// `docs/dev/env/server.md` is rendered from the registry and clap. Run
/// with `UPDATE_ENV_REFERENCE=1` to rewrite it.
#[test]
fn reference_doc_is_current() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(REFERENCE);
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

fn tuning_with(pairs: &[(&str, &str)]) -> anyhow::Result<ServerTuning> {
    ServerTuning::from_env(&EnvSnapshot::from_pairs(SERVER_ENV, pairs.iter().copied()))
}

#[test]
fn unset_variables_keep_the_previous_defaults() {
    let tuning = tuning_with(&[]).unwrap();
    assert!(!tuning.allow_unauthenticated_network);
    assert_eq!(
        tuning.openai_compat_profile,
        OpenAiCompatibilityProfile::Strict
    );
    assert!(tuning.fish_long_form_enabled);
    assert!(!tuning.tts_opaque_final_wav_enabled);
    assert_eq!(tuning.tts_max_text_bytes, 1024 * 1024);
    assert_eq!(tuning.audio_stream_stall_timeout, Duration::from_secs(30));
    assert_eq!(tuning.tts_stream_max_pcm_bytes, None);
    assert_eq!(tuning.tts_max_active_jobs, 256);
    assert_eq!(tuning.tts_pcm_publish_interval, Duration::from_millis(250));
    assert_eq!(
        tuning.idempotency_limits,
        QuotaLimits {
            global: 1_048_576,
            per_tenant: 65_536,
        }
    );
    assert_eq!(tuning.retention.terminal_job_ms, 30 * 24 * 60 * 60 * 1_000);
    assert_eq!(tuning.retention.ephemeral_artifact_ms, 60 * 60 * 1_000);
}

/// Booleans take the shared vocabulary and fail closed. The unauthenticated
/// bind came from clap's falsey parser before, which read any value it did
/// not know ("fasle") as true.
#[test]
fn booleans_use_the_shared_vocabulary_and_fail_closed() {
    let tuning = tuning_with(&[
        ("IZWI_FISH_LONG_FORM_ENABLED", "no"),
        ("IZWI_TTS_OPAQUE_FINAL_WAV_ENABLED", "ON"),
        ("IZWI_ALLOW_UNAUTHENTICATED_NETWORK", "yes"),
    ])
    .unwrap();
    assert!(!tuning.fish_long_form_enabled);
    assert!(tuning.tts_opaque_final_wav_enabled);
    assert!(tuning.allow_unauthenticated_network);
    for name in [
        "IZWI_ALLOW_UNAUTHENTICATED_NETWORK",
        "IZWI_FISH_LONG_FORM_ENABLED",
        "IZWI_TTS_OPAQUE_FINAL_WAV_ENABLED",
    ] {
        let error = tuning_with(&[(name, "fasle")]).unwrap_err().to_string();
        assert!(error.contains(name), "{error}");
    }
}

/// A value that does not parse, or is below its minimum, stops the server
/// instead of silently taking the default.
#[test]
fn bad_values_are_errors_naming_the_variable() {
    for (name, value) in [
        ("IZWI_TTS_MAX_TEXT_BYTES", "0"),
        ("IZWI_TTS_MAX_TEXT_BYTES", "lots"),
        ("IZWI_TTS_STREAM_MAX_PCM_BYTES", "1"),
        ("IZWI_TTS_TOTAL_SPOOL_BYTES", "43"),
        ("IZWI_MAX_SPEECH_REQUESTS_PER_TENANT", "0"),
        ("IZWI_OPENAI_COMPAT_PROFILE", "lenient"),
        ("IZWI_JOB_RETENTION_SECONDS", "-1"),
    ] {
        let error = tuning_with(&[(name, value)]).unwrap_err().to_string();
        assert!(error.contains(name), "{name}={value}: {error}");
    }
    let tuning = tuning_with(&[
        ("IZWI_OPENAI_COMPAT_PROFILE", " Relaxed "),
        ("IZWI_TTS_PCM_PUBLISH_INTERVAL_MS", "0"),
        ("IZWI_MAX_SPEECH_REQUESTS_PER_TENANT", "4"),
    ])
    .unwrap();
    assert_eq!(
        tuning.openai_compat_profile,
        OpenAiCompatibilityProfile::Relaxed
    );
    assert_eq!(tuning.tts_pcm_publish_interval, Duration::ZERO);
    assert_eq!(tuning.max_speech_requests_per_tenant, Some(4));
}
