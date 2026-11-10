//! The local role's environment (HYG-E1).
//!
//! [`SERVER_ENV`] registers every `IZWI_*` variable this crate reads outside
//! the gateway role (whose variables `roles/gateway_env.rs` documents):
//! start-up settings (storage, admission limits, preload) that their own
//! constructors read once, and the request-path and maintenance settings
//! that [`ServerTuning`] captures once at start-up, so no request reads the
//! process environment. Every value in [`ServerTuning`] is validated when
//! the server starts and a bad one stops it with an error naming the
//! variable; booleans use the shared `izwi_serving_protocol::env` vocabulary.
//! `docs/dev/env/server.md` is rendered from the registry, and the tests keep
//! it in step with the crate's sources.

use std::sync::OnceLock;
use std::time::Duration;

use anyhow::bail;
use izwi_serving_protocol::env::{EnvSnapshot, EnvVar};

use crate::api::openai::batches::webhook::WebhookPolicy;
use crate::api::openai::compat::OpenAiCompatibilityProfile;
use crate::batch_runtime::store::{QuotaLimits, RetentionPolicy};
use crate::local_security::LocalExposure;

const fn var(name: &'static str, default: &'static str, description: &'static str) -> EnvVar {
    EnvVar {
        name,
        default,
        description,
    }
}

/// Every local-role variable this crate reads.
pub(crate) static SERVER_ENV: &[EnvVar] = &[
    // Storage (storage_layout.rs, db/sqlite.rs), read at start-up.
    var(
        "IZWI_DB_PATH",
        "the platform data directory's izwi.sqlite3",
        "SQLite store database path.",
    ),
    var(
        "IZWI_DATABASE_URL",
        "",
        "Store database URL (`postgres://`, `sqlite://`); wins over IZWI_DB_PATH.",
    ),
    var(
        "IZWI_MEDIA_DIR",
        "the platform data directory's media",
        "Root of stored media (uploads, generated audio, artifacts).",
    ),
    var(
        "IZWI_SPEECH_SPOOL_DIR",
        "the temp directory's izwi/speech-spool",
        "Scratch root for local speech requests.",
    ),
    var(
        "IZWI_DB_MAX_CONNECTIONS",
        "5",
        "Database pool size (1-1024).",
    ),
    var(
        "IZWI_DB_ACQUIRE_TIMEOUT_MS",
        "3000",
        "How long a request waits for a database connection.",
    ),
    // Node identity (node_identity.rs), read at start-up.
    var(
        "IZWI_NODE_ID",
        "local",
        "This node's id on a shared database (1-128 ASCII letters, digits, `-`, `_`); startup fails only the in-process jobs it owns.",
    ),
    // Security and start-up (local_security.rs, roles/local.rs, shutdown.rs).
    var(
        "IZWI_API_KEY",
        "",
        "API key required on /v1 and /internal routes (16-4096 printable bytes). Required for a non-loopback bind.",
    ),
    var(
        "IZWI_API_KEY_FILE",
        "",
        "Absolute path of a file holding the API key instead (a Docker or systemd secret; one trailing newline is ignored). Setting both is an error.",
    ),
    var(
        "IZWI_ALLOW_UNAUTHENTICATED_NETWORK",
        "false",
        "Serve a non-loopback bind without IZWI_API_KEY (same as --allow-unauthenticated-network).",
    ),
    var(
        "IZWI_LOOPBACK_PUBLISHED",
        "false",
        "The port is published on the host's loopback only (a container's `-p 127.0.0.1:…`): treat the bind as loopback, so no key is required and only loopback or IZWI_ALLOWED_HOSTS Host names are served.",
    ),
    var(
        "IZWI_ALLOWED_HOSTS",
        "",
        "Comma-separated extra Host names to accept (port optional). On a non-loopback bind, setting it turns the Host check on.",
    ),
    var(
        "IZWI_PRELOAD_MODELS",
        "",
        "Comma-separated model ids loaded at start-up; readiness waits for them.",
    ),
    var(
        "IZWI_WARMUP_PRELOADED_MODELS",
        "true",
        "Run a warm-up request on each preloaded model.",
    ),
    var(
        "IZWI_ASR_WARMUP_DURATION_MS",
        "800",
        "Length of the ASR warm-up clip (100-5000).",
    ),
    var(
        "IZWI_GRANITE_DECODE_PROFILE",
        "",
        "Set to 1 for the engine by --granite-decode-profile; not read here.",
    ),
    var(
        "IZWI_GRANITE_SPEECH_DTYPE",
        "",
        "Set for the engine by --granite-speech-dtype; not read here.",
    ),
    var(
        "IZWI_DESKTOP_OWNER_PIPE",
        "",
        "`1` when the desktop app owns this process: stdin closing shuts it down.",
    ),
    var(
        "IZWI_HTTP_SHUTDOWN_PRESTOP_SECS",
        "0",
        "Pre-stop window at shutdown (0-60): readiness is unready while requests are still served.",
    ),
    var(
        "IZWI_HTTP_SHUTDOWN_GRACE_SECS",
        "20",
        "How long in-flight HTTP requests may run after the pre-stop window (1-300); open chat streams get a terminal shutting-down event shortly before it ends.",
    ),
    var(
        "IZWI_BATCH_WORKER_CONCURRENCY",
        "the runtime's capacity",
        "Durable jobs the local batch worker runs at once, and `/v1/batches` lines in flight at once (at most the runtime's capacity).",
    ),
    var(
        "IZWI_BATCH_STAGE_TIMEOUT_SECS",
        "0 (none)",
        "Wall-clock limit of one durable stage (30-86400; 0 disables).",
    ),
    var(
        "IZWI_BATCH_WORKER_DRAIN_TIMEOUT_SECS",
        "20",
        "How long the batch worker drains at shutdown (1-300).",
    ),
    // Admission (state.rs), read at start-up.
    var(
        "IZWI_MAX_REALTIME_REQUESTS",
        "the global request limit",
        "Concurrent realtime-class requests.",
    ),
    var(
        "IZWI_MAX_INTERACTIVE_REQUESTS",
        "the global request limit",
        "Concurrent interactive-class requests.",
    ),
    var(
        "IZWI_MAX_STREAMING_REQUESTS",
        "the global request limit",
        "Concurrent streaming-class requests.",
    ),
    var(
        "IZWI_MAX_ONLINE_REQUESTS",
        "the global request limit",
        "Concurrent online-class requests.",
    ),
    var(
        "IZWI_MAX_BATCH_REQUESTS",
        "the global request limit minus one",
        "Concurrent batch-class requests.",
    ),
    var(
        "IZWI_MAX_BACKGROUND_REQUESTS",
        "the global request limit minus one",
        "Concurrent background-class requests.",
    ),
    var(
        "IZWI_MAX_REALTIME_SESSIONS",
        "32",
        "Open realtime websocket sessions.",
    ),
    var(
        "IZWI_ADMISSION_QUEUE_TIMEOUT_SECS",
        "30",
        "How long a chat, agent or /v1/audio request waits for an admission slot before it is refused with 429 (0 waits without limit). Realtime sessions, record and durable-job routes and background work always wait.",
    ),
    // The speech API (api/openai/audio), captured in ServerTuning.
    var(
        "IZWI_OPENAI_MODEL_ALIASES",
        "",
        "OpenAI model names mapped to Izwi models for /v1/audio/*: comma-separated `name=Model`, e.g. `tts-1=Kokoro-82M,whisper-1=Parakeet-TDT-0.6B-v3`. Unmapped OpenAI names are refused.",
    ),
    var(
        "IZWI_SPEECH_DEFAULT_FORMAT",
        "wav",
        "The /v1/audio/speech `response_format` when a request names none (`mp3` gives OpenAI's default).",
    ),
    var(
        "IZWI_TTS_BATCH_MAX_ITEMS",
        "32",
        "Items one /v1/audio/speech/batch request may hold.",
    ),
    var(
        "IZWI_TTS_BATCH_CONCURRENCY",
        "8",
        "Items of one speech batch synthesized at once.",
    ),
    var(
        "IZWI_TTS_BATCH_MAX_OUTPUT_MB",
        "256",
        "Encoded audio one speech batch may return, in MiB.",
    ),
    var(
        "IZWI_SPEECH_OUTCOME_CAPACITY",
        "1024",
        "Speech request outcomes GET /v1/audio/speech/{id} remembers (process-local).",
    ),
    var(
        "IZWI_SPEECH_OUTCOME_TTL_SECS",
        "3600",
        "How long a speech request outcome is remembered.",
    ),
    var(
        "IZWI_SPEECH_STREAM_MAX_PENDING_CHARS",
        "65536",
        "Text a /v1/audio/speech/stream session may hold unspoken.",
    ),
    var(
        "IZWI_SPEECH_STREAM_IDLE_TIMEOUT_SECS",
        "30",
        "How long an idle /v1/audio/speech/stream session stays open.",
    ),
    var(
        "IZWI_VOICE_CONSENT_MODE",
        "`attestation` (local role), `recording` (server role)",
        "What creating a voice on /v1/audio/voices requires: `attestation` (a non-empty `consent`), `recording` (a stored consent recording's id) or `verified` (a consent recording whose transcript matches a published consent phrase).",
    ),
    var(
        "IZWI_ALLOWED_MEDIA_DOMAINS",
        "unset (no URLs)",
        "Hosts reference audio may be fetched from by URL, comma-separated; `.example.com` also allows subdomains. Unset, reference audio URLs are refused.",
    ),
    var(
        "IZWI_REALTIME_PROFILES",
        "unset",
        "Extra OpenAI Realtime profiles `?model=` can name: a JSON array of `{id, asr_model_id, text_model_id, tts_model_id}` (modular) or `{id, s2s_model_id}` (speech-to-speech), each with an optional default `voice`; an id equal to a built-in (`izwi-voice`, `izwi-voice-compact`) replaces it.",
    ),
    var(
        "IZWI_REALTIME_DEFAULT_PROFILE",
        "`izwi-voice`",
        "The OpenAI Realtime profile a session gets when `model` is absent, empty or an OpenAI model name (`gpt-realtime*`).",
    ),
    var(
        "IZWI_REALTIME_PERSIST",
        "off",
        "`1` records OpenAI Realtime sessions and their turns in the voice history; off, they leave nothing behind.",
    ),
    var(
        "IZWI_MAX_RESPONSE_STORE_ENTRIES",
        "512",
        "Stored Responses API responses kept in memory.",
    ),
    var(
        "IZWI_MAX_AGENT_SESSION_STORE_ENTRIES",
        "512",
        "Agent sessions kept in memory.",
    ),
    // Request paths (ServerTuning), captured once at start-up.
    var(
        "IZWI_OPENAI_COMPAT_PROFILE",
        "strict",
        "OpenAI compatibility profile: `strict` or `relaxed`.",
    ),
    var(
        "IZWI_OPENAI_AUDIO_UPLOAD_LIMIT_MB",
        "64 (relaxed) or the strict profile's limit",
        "Largest audio upload on the OpenAI audio and media routes, in MiB.",
    ),
    var(
        "IZWI_MEDIA_DECODE_LANES",
        "2",
        "Audio uploads decoded at once.",
    ),
    var(
        "IZWI_MAX_SPEECH_REQUESTS_PER_TENANT",
        "the global request limit",
        "Concurrent speech requests per tenant (at most the global limit): speech and batch requests until their response ends, and open /v1/audio/speech/stream and /v1/realtime sockets.",
    ),
    var(
        "IZWI_TTS_MAX_TEXT_BYTES",
        "1048576",
        "Longest speech input one request may carry, in UTF-8 bytes.",
    ),
    var(
        "IZWI_TTS_LONG_FORM_CHUNK_MAX_CHARS",
        "the model's segment size",
        "Characters per long-form speech segment (clamped to 80-4000).",
    ),
    var(
        "IZWI_FISH_LONG_FORM_ENABLED",
        "true",
        "Allow Fish S2 requests that need more than one segment.",
    ),
    var(
        "IZWI_FISH_DRAMA3_LONG_FORM_ENABLED",
        "true",
        "Allow Fish Audio Drama 3 requests that need more than one chunk (its chunk session).",
    ),
    var(
        "IZWI_TTS_SEGMENT_TIMEOUT_SECS",
        "1800",
        "How long a long-form segment may publish nothing before it fails.",
    ),
    var(
        "IZWI_TTS_STREAM_MAX_PCM_BYTES",
        "the model's allowance",
        "Most PCM bytes one streamed or durable speech output may hold (at least 2).",
    ),
    var(
        "IZWI_AUDIO_STREAM_EVENT_QUEUE_CAPACITY",
        "32",
        "Events a speech stream buffers for its reader.",
    ),
    var(
        "IZWI_AUDIO_STREAM_STALL_TIMEOUT_SECS",
        "30",
        "How long a stalled speech stream reader is waited for.",
    ),
    var(
        "IZWI_AUDIO_STREAM_MAX_EVENT_BYTES",
        "4194304",
        "Buffered event bytes per speech stream.",
    ),
    var(
        "IZWI_AUDIO_STREAM_TOTAL_EVENT_BYTES",
        "67108864",
        "Buffered speech stream event bytes across all streams (at least 44).",
    ),
    var(
        "IZWI_TTS_TOTAL_SPOOL_BYTES",
        "1073741824",
        "Disk spooled by speech outputs at once (at least 44).",
    ),
    var(
        "IZWI_TTS_TOTAL_UPLOAD_BYTES",
        "268435456",
        "Speech reference upload bytes held at once (at least 44).",
    ),
    var(
        "IZWI_TTS_OPAQUE_FINAL_WAV_ENABLED",
        "false",
        "Store durable speech outputs as opaque artifacts.",
    ),
    var(
        "IZWI_TTS_MAX_JOURNAL_BYTES",
        "8589934592",
        "Replay journal bytes one durable speech job may write.",
    ),
    var(
        "IZWI_TTS_MAX_JOB_SECONDS",
        "86400",
        "Age after which a durable speech job is not resumed.",
    ),
    var(
        "IZWI_TTS_REPLAY_RETENTION_SECONDS",
        "86400",
        "How long the replay PCM of a terminal speech job is kept.",
    ),
    var(
        "IZWI_TTS_PCM_PUBLISH_INTERVAL_MS",
        "250",
        "How often durable speech publishes PCM; 0 publishes every chunk.",
    ),
    var(
        "IZWI_TTS_PCM_PUBLISH_MAX_CHUNKS",
        "8",
        "Chunks a durable speech publication may hold.",
    ),
    var(
        "IZWI_TTS_MAX_ACTIVE_JOBS",
        "256",
        "Queued and running durable speech jobs in total.",
    ),
    var(
        "IZWI_TTS_MAX_ACTIVE_JOBS_PER_TENANT",
        "32",
        "Queued and running durable speech jobs per tenant.",
    ),
    var(
        "IZWI_VOICE_WS_OUTBOUND_ITEMS",
        "256",
        "Messages a voice websocket queues for its client.",
    ),
    var(
        "IZWI_VOICE_WS_OUTBOUND_AUDIO_BYTES",
        "4194304",
        "Assistant audio bytes a voice websocket queues.",
    ),
    var(
        "IZWI_VOICE_WS_OUTBOUND_AUDIO_MS",
        "5000",
        "Assistant audio a voice websocket queues, in milliseconds.",
    ),
    var(
        "IZWI_FILES_MAX_BYTES",
        "209715200",
        "Largest file `/v1/files` accepts, in bytes (at most 1 GiB).",
    ),
    var(
        "IZWI_BATCH_MAX_REQUESTS",
        "50000",
        "Requests one `/v1/batches` input file may hold.",
    ),
    var(
        "IZWI_BATCH_WEBHOOK_SECRET",
        "unset (webhooks off)",
        "Turns on `/v1/batches` completion webhooks (`metadata.webhook_url`) and signs each delivery (Standard Webhooks, HMAC-SHA256). A `whsec_` value is base64 of the key; any other value is the key. Unset, a batch that names a webhook is refused.",
    ),
    var(
        "IZWI_BATCH_WEBHOOK_ALLOW_HTTP",
        "false",
        "Allows `http://` batch webhook URLs; `https://` only otherwise.",
    ),
    var(
        "IZWI_BATCH_WEBHOOK_ALLOW_PRIVATE",
        "false",
        "Allows batch webhook URLs whose host resolves to a loopback, private, link-local or other non-public address.",
    ),
    var(
        "IZWI_BATCH_WORKER_HEARTBEAT_STALE_SECS",
        "5",
        "Heartbeat age after which readiness reports a batch worker stale.",
    ),
    var(
        "IZWI_IDEMPOTENCY_MAX_RECORDS",
        "1048576",
        "Live idempotency records in total.",
    ),
    var(
        "IZWI_IDEMPOTENCY_MAX_RECORDS_PER_TENANT",
        "65536",
        "Live idempotency records per tenant.",
    ),
    var(
        "IZWI_PROVIDER_WRITE_MAX_RESERVATIONS",
        "65536",
        "Outstanding provider write reservations in total.",
    ),
    var(
        "IZWI_PROVIDER_WRITE_MAX_RESERVATIONS_PER_TENANT",
        "16384",
        "Outstanding provider write reservations per tenant.",
    ),
    var(
        "IZWI_JOB_RETENTION_SECONDS",
        "2592000",
        "How long terminal durable jobs are kept.",
    ),
    var(
        "IZWI_ARTIFACT_EPHEMERAL_RETENTION_SECONDS",
        "3600",
        "How long ephemeral artifacts, and unreferenced input media at least, are kept.",
    ),
    var(
        "IZWI_WORKER_HEARTBEAT_RETENTION_SECONDS",
        "86400",
        "How long heartbeat rows of stopped workers are kept.",
    ),
    var(
        "IZWI_MEDIA_TOMBSTONE_RETENTION_SECONDS",
        "86400",
        "How long media tombstones outlive their objects.",
    ),
];

/// Request-path, maintenance and security settings, captured once at
/// start-up.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ServerTuning {
    pub(crate) allow_unauthenticated_network: bool,
    pub(crate) local_exposure: LocalExposure,
    /// Unauthenticated probes carry uptime, capacity, queues and startup
    /// warnings only while no API key protects the server (loopback, the
    /// desktop app, development); a keyed server keeps them on its keyed
    /// health routes.
    pub(crate) public_probe_details: bool,
    pub(crate) openai_compat_profile: OpenAiCompatibilityProfile,
    /// `None` takes the profile's default.
    pub(crate) openai_audio_upload_limit_mb: Option<usize>,
    pub(crate) media_decode_lanes: usize,
    /// `None` allows every tenant the global speech limit.
    pub(crate) max_speech_requests_per_tenant: Option<usize>,
    pub(crate) tts_max_text_bytes: usize,
    /// Clamped to 80-4000; `None` takes the model's segment size.
    pub(crate) tts_long_form_chunk_max_chars: Option<usize>,
    pub(crate) fish_long_form_enabled: bool,
    pub(crate) fish_drama3_long_form_enabled: bool,
    pub(crate) tts_segment_timeout: Duration,
    /// `None` takes each model's allowance.
    pub(crate) tts_stream_max_pcm_bytes: Option<usize>,
    pub(crate) audio_stream_event_queue_capacity: usize,
    pub(crate) audio_stream_stall_timeout: Duration,
    pub(crate) audio_stream_max_event_bytes: usize,
    pub(crate) audio_stream_total_event_bytes: usize,
    pub(crate) tts_total_spool_bytes: usize,
    pub(crate) tts_total_upload_bytes: usize,
    pub(crate) tts_opaque_final_wav_enabled: bool,
    pub(crate) tts_max_journal_bytes: u64,
    pub(crate) tts_max_job_seconds: u64,
    pub(crate) tts_replay_retention: Duration,
    pub(crate) tts_pcm_publish_interval: Duration,
    pub(crate) tts_pcm_publish_max_chunks: usize,
    pub(crate) tts_max_active_jobs: usize,
    pub(crate) tts_max_active_jobs_per_tenant: usize,
    pub(crate) voice_ws_outbound_items: usize,
    pub(crate) voice_ws_outbound_audio_bytes: usize,
    pub(crate) voice_ws_outbound_audio_ms: u64,
    pub(crate) batch_worker_heartbeat_stale_ms: u64,
    pub(crate) files_max_bytes: u64,
    pub(crate) batch_max_requests: usize,
    /// `None` leaves `/v1/batches` completion webhooks off.
    pub(crate) batch_webhook: Option<WebhookPolicy>,
    pub(crate) idempotency_limits: QuotaLimits,
    pub(crate) provider_write_limits: QuotaLimits,
    pub(crate) retention: RetentionPolicy,
    pub(crate) speech_api: SpeechApiTuning,
}

/// How a voice upload proves its speaker agreed (`IZWI_VOICE_CONSENT_MODE`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VoiceConsentMode {
    /// Any non-empty `consent`, kept as provenance.
    Attestation,
    /// `consent` names a stored consent recording.
    Recording,
    /// `consent` names a consent recording whose transcript matched a
    /// published consent phrase.
    Verified,
}

/// Settings of the speech API (`/v1/audio/*`).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SpeechApiTuning {
    /// Lower-cased OpenAI model name to Izwi model id.
    pub(crate) model_aliases: Vec<(String, String)>,
    pub(crate) default_format: String,
    pub(crate) batch_max_items: usize,
    pub(crate) batch_concurrency: usize,
    pub(crate) batch_max_output_bytes: usize,
    pub(crate) outcome_capacity: usize,
    pub(crate) outcome_ttl: Duration,
    pub(crate) stream_max_pending_chars: usize,
    pub(crate) stream_idle_timeout: Duration,
    /// `None` takes the role's default.
    pub(crate) voice_consent_mode: Option<VoiceConsentMode>,
    /// Lower-cased hosts; a leading `.` allows subdomains. Empty refuses URLs.
    pub(crate) allowed_media_domains: Vec<String>,
}

impl SpeechApiTuning {
    fn from_env(env: &EnvSnapshot) -> anyhow::Result<Self> {
        Ok(Self {
            model_aliases: model_aliases(env.get("IZWI_OPENAI_MODEL_ALIASES")?.unwrap_or(""))?,
            default_format: env
                .get("IZWI_SPEECH_DEFAULT_FORMAT")?
                .map(|value| value.trim().to_ascii_lowercase())
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| "wav".to_string()),
            batch_max_items: at_least(env, "IZWI_TTS_BATCH_MAX_ITEMS", 1, 32)?,
            batch_concurrency: at_least(env, "IZWI_TTS_BATCH_CONCURRENCY", 1, 8)?,
            batch_max_output_bytes: at_least(env, "IZWI_TTS_BATCH_MAX_OUTPUT_MB", 1usize, 256)?
                .saturating_mul(1024 * 1024),
            outcome_capacity: at_least(env, "IZWI_SPEECH_OUTCOME_CAPACITY", 1, 1024)?,
            outcome_ttl: Duration::from_secs(at_least(
                env,
                "IZWI_SPEECH_OUTCOME_TTL_SECS",
                1,
                3600,
            )?),
            stream_max_pending_chars: at_least(
                env,
                "IZWI_SPEECH_STREAM_MAX_PENDING_CHARS",
                1,
                65_536,
            )?,
            stream_idle_timeout: Duration::from_secs(at_least(
                env,
                "IZWI_SPEECH_STREAM_IDLE_TIMEOUT_SECS",
                1,
                30,
            )?),
            voice_consent_mode: match env
                .get("IZWI_VOICE_CONSENT_MODE")?
                .map(|value| value.trim().to_ascii_lowercase())
                .as_deref()
            {
                None | Some("") => None,
                Some("attestation") => Some(VoiceConsentMode::Attestation),
                Some("recording") => Some(VoiceConsentMode::Recording),
                Some("verified") => Some(VoiceConsentMode::Verified),
                Some(other) => bail!(
                    "IZWI_VOICE_CONSENT_MODE must be attestation, recording or verified, not {other:?}"
                ),
            },
            allowed_media_domains: media_domains(
                env.get("IZWI_ALLOWED_MEDIA_DOMAINS")?.unwrap_or(""),
            )?,
        })
    }
}

/// `IZWI_OPENAI_MODEL_ALIASES`: `name=Model` pairs, comma-separated.
fn model_aliases(value: &str) -> anyhow::Result<Vec<(String, String)>> {
    value
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(|entry| {
            let Some((name, model)) = entry.split_once('=') else {
                bail!("IZWI_OPENAI_MODEL_ALIASES entries are `name=Model`, not {entry:?}");
            };
            let (name, model) = (name.trim(), model.trim());
            if name.is_empty() || model.is_empty() {
                bail!("IZWI_OPENAI_MODEL_ALIASES entries are `name=Model`, not {entry:?}");
            }
            Ok((name.to_ascii_lowercase(), model.to_string()))
        })
        .collect()
}

/// `IZWI_ALLOWED_MEDIA_DOMAINS`: lower-cased host names, `.suffix` allowed.
fn media_domains(value: &str) -> anyhow::Result<Vec<String>> {
    value
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(|entry| {
            let host = entry.trim_end_matches('.').to_ascii_lowercase();
            let bare = host.strip_prefix('.').unwrap_or(&host);
            if bare.is_empty()
                || bare.contains(['/', '*', '@', ':'])
                || bare.chars().any(char::is_whitespace)
            {
                bail!("IZWI_ALLOWED_MEDIA_DOMAINS entries must be host names, not {entry:?}");
            }
            Ok(host)
        })
        .collect()
}

impl ServerTuning {
    pub(crate) fn from_env(env: &EnvSnapshot) -> anyhow::Result<Self> {
        let secs =
            |name: &str, default: u64| -> anyhow::Result<u64> { at_least(env, name, 1, default) };
        let retention_ms = |name: &str, default: u64| -> anyhow::Result<u64> {
            Ok(secs(name, default)?.saturating_mul(1_000))
        };
        Ok(Self {
            allow_unauthenticated_network: env
                .flag_or("IZWI_ALLOW_UNAUTHENTICATED_NETWORK", false)?,
            local_exposure: LocalExposure {
                loopback_published: env.flag_or("IZWI_LOOPBACK_PUBLISHED", false)?,
                allowed_hosts: allowed_hosts(env.get("IZWI_ALLOWED_HOSTS")?.unwrap_or(""))?,
            },
            public_probe_details: ["IZWI_API_KEY", "IZWI_API_KEY_FILE"]
                .iter()
                .all(|name| env.get_os(name).is_none_or(|value| value.is_empty())),
            openai_compat_profile: match env
                .get("IZWI_OPENAI_COMPAT_PROFILE")?
                .map(|value| value.trim().to_ascii_lowercase())
                .as_deref()
            {
                None | Some("strict") => OpenAiCompatibilityProfile::Strict,
                Some("relaxed") => OpenAiCompatibilityProfile::Relaxed,
                Some(other) => {
                    bail!("IZWI_OPENAI_COMPAT_PROFILE must be strict or relaxed, not {other:?}")
                }
            },
            openai_audio_upload_limit_mb: optional_at_least(
                env,
                "IZWI_OPENAI_AUDIO_UPLOAD_LIMIT_MB",
                1,
            )?,
            media_decode_lanes: at_least(env, "IZWI_MEDIA_DECODE_LANES", 1, 2)?,
            max_speech_requests_per_tenant: optional_at_least(
                env,
                "IZWI_MAX_SPEECH_REQUESTS_PER_TENANT",
                1,
            )?,
            tts_max_text_bytes: at_least(env, "IZWI_TTS_MAX_TEXT_BYTES", 1, 1024 * 1024)?,
            tts_long_form_chunk_max_chars: env.parse("IZWI_TTS_LONG_FORM_CHUNK_MAX_CHARS")?,
            fish_long_form_enabled: env.flag_or("IZWI_FISH_LONG_FORM_ENABLED", true)?,
            fish_drama3_long_form_enabled: env
                .flag_or("IZWI_FISH_DRAMA3_LONG_FORM_ENABLED", true)?,
            tts_segment_timeout: Duration::from_secs(secs("IZWI_TTS_SEGMENT_TIMEOUT_SECS", 1800)?),
            tts_stream_max_pcm_bytes: optional_at_least(env, "IZWI_TTS_STREAM_MAX_PCM_BYTES", 2)?,
            audio_stream_event_queue_capacity: at_least(
                env,
                "IZWI_AUDIO_STREAM_EVENT_QUEUE_CAPACITY",
                1,
                32,
            )?,
            audio_stream_stall_timeout: Duration::from_secs(secs(
                "IZWI_AUDIO_STREAM_STALL_TIMEOUT_SECS",
                30,
            )?),
            audio_stream_max_event_bytes: at_least(
                env,
                "IZWI_AUDIO_STREAM_MAX_EVENT_BYTES",
                1,
                4 * 1024 * 1024,
            )?,
            audio_stream_total_event_bytes: at_least(
                env,
                "IZWI_AUDIO_STREAM_TOTAL_EVENT_BYTES",
                44,
                64 * 1024 * 1024,
            )?,
            tts_total_spool_bytes: at_least(
                env,
                "IZWI_TTS_TOTAL_SPOOL_BYTES",
                44,
                1024 * 1024 * 1024,
            )?,
            tts_total_upload_bytes: at_least(
                env,
                "IZWI_TTS_TOTAL_UPLOAD_BYTES",
                44,
                256 * 1024 * 1024,
            )?,
            tts_opaque_final_wav_enabled: env
                .flag_or("IZWI_TTS_OPAQUE_FINAL_WAV_ENABLED", false)?,
            tts_max_journal_bytes: at_least(
                env,
                "IZWI_TTS_MAX_JOURNAL_BYTES",
                1,
                8 * 1024 * 1024 * 1024,
            )?,
            tts_max_job_seconds: secs("IZWI_TTS_MAX_JOB_SECONDS", 24 * 3600)?,
            tts_replay_retention: Duration::from_secs(secs(
                "IZWI_TTS_REPLAY_RETENTION_SECONDS",
                86_400,
            )?),
            tts_pcm_publish_interval: Duration::from_millis(
                env.parse_or("IZWI_TTS_PCM_PUBLISH_INTERVAL_MS", 250)?,
            ),
            tts_pcm_publish_max_chunks: at_least(env, "IZWI_TTS_PCM_PUBLISH_MAX_CHUNKS", 1, 8)?,
            tts_max_active_jobs: at_least(env, "IZWI_TTS_MAX_ACTIVE_JOBS", 1, 256)?,
            tts_max_active_jobs_per_tenant: at_least(
                env,
                "IZWI_TTS_MAX_ACTIVE_JOBS_PER_TENANT",
                1,
                32,
            )?,
            voice_ws_outbound_items: at_least(env, "IZWI_VOICE_WS_OUTBOUND_ITEMS", 1, 256)?,
            voice_ws_outbound_audio_bytes: at_least(
                env,
                "IZWI_VOICE_WS_OUTBOUND_AUDIO_BYTES",
                1,
                4 * 1024 * 1024,
            )?,
            voice_ws_outbound_audio_ms: at_least(env, "IZWI_VOICE_WS_OUTBOUND_AUDIO_MS", 1, 5_000)?,
            batch_worker_heartbeat_stale_ms: secs("IZWI_BATCH_WORKER_HEARTBEAT_STALE_SECS", 5)?
                .saturating_mul(1_000),
            files_max_bytes: at_least(env, "IZWI_FILES_MAX_BYTES", 1, 200 * 1024 * 1024)?,
            batch_max_requests: at_least(env, "IZWI_BATCH_MAX_REQUESTS", 1, 50_000)?,
            batch_webhook: WebhookPolicy::from_env(env)?,
            idempotency_limits: QuotaLimits {
                global: at_least(env, "IZWI_IDEMPOTENCY_MAX_RECORDS", 1, 1_048_576)?,
                per_tenant: at_least(env, "IZWI_IDEMPOTENCY_MAX_RECORDS_PER_TENANT", 1, 65_536)?,
            },
            provider_write_limits: QuotaLimits {
                global: at_least(env, "IZWI_PROVIDER_WRITE_MAX_RESERVATIONS", 1, 65_536)?,
                per_tenant: at_least(
                    env,
                    "IZWI_PROVIDER_WRITE_MAX_RESERVATIONS_PER_TENANT",
                    1,
                    16_384,
                )?,
            },
            retention: RetentionPolicy {
                terminal_job_ms: retention_ms("IZWI_JOB_RETENTION_SECONDS", 30 * 24 * 60 * 60)?,
                ephemeral_artifact_ms: retention_ms(
                    "IZWI_ARTIFACT_EPHEMERAL_RETENTION_SECONDS",
                    60 * 60,
                )?,
                worker_heartbeat_ms: retention_ms(
                    "IZWI_WORKER_HEARTBEAT_RETENTION_SECONDS",
                    24 * 60 * 60,
                )?,
                media_tombstone_ms: retention_ms(
                    "IZWI_MEDIA_TOMBSTONE_RETENTION_SECONDS",
                    24 * 60 * 60,
                )?,
                legacy_idempotency_ms: RetentionPolicy::LEGACY_IDEMPOTENCY_MS,
            },
            speech_api: SpeechApiTuning::from_env(env)?,
        })
    }
}

/// `name` parsed and at least `min`, `default` when unset.
fn at_least<T>(env: &EnvSnapshot, name: &str, min: T, default: T) -> anyhow::Result<T>
where
    T: std::str::FromStr + PartialOrd + Copy + std::fmt::Display,
    T::Err: std::fmt::Display,
{
    Ok(optional_at_least(env, name, min)?.unwrap_or(default))
}

/// `IZWI_ALLOWED_HOSTS`: lower-cased host names, ports and blanks dropped.
fn allowed_hosts(value: &str) -> anyhow::Result<Vec<String>> {
    value
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(|entry| {
            let name = crate::local_security::host_name(entry);
            if name.is_empty()
                || name.contains(['/', '*', '@'])
                || name.chars().any(char::is_whitespace)
            {
                bail!("IZWI_ALLOWED_HOSTS entries must be host names, not {entry:?}");
            }
            Ok(name.to_ascii_lowercase())
        })
        .collect()
}

fn optional_at_least<T>(env: &EnvSnapshot, name: &str, min: T) -> anyhow::Result<Option<T>>
where
    T: std::str::FromStr + PartialOrd + Copy + std::fmt::Display,
    T::Err: std::fmt::Display,
{
    match env.parse::<T>(name)? {
        Some(value) if value < min => bail!("{name} must be at least {min}, not {value}"),
        value => Ok(value),
    }
}

static TUNING: OnceLock<ServerTuning> = OnceLock::new();

/// Capture the request-path settings; a bad value is an error naming its
/// variable. Both roles call this before serving.
pub(crate) fn init() -> anyhow::Result<()> {
    let tuning = ServerTuning::from_env(&EnvSnapshot::capture(SERVER_ENV))?;
    let _ = TUNING.set(tuning);
    Ok(())
}

/// The settings [`init`] captured. Code that runs without it (tests, the
/// in-process routers other crates' tests build) captures them on first use.
pub(crate) fn tuning() -> &'static ServerTuning {
    TUNING.get_or_init(|| {
        ServerTuning::from_env(&EnvSnapshot::capture(SERVER_ENV))
            .unwrap_or_else(|error| panic!("invalid server environment: {error:#}"))
    })
}

#[cfg(test)]
#[path = "env_config_tests.rs"]
mod tests;
