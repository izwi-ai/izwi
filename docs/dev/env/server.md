# izwi-server local role environment

<!-- Generated from the binary's environment registry; the crate's env_reference test keeps it current. -->

The local role (`izwi-server`, the default `--role local`) reads these variables once, at start-up; no request reads the environment. Request-path settings are validated when the server starts, and a bad value stops it with an error naming the variable. Booleans accept `true`, `false`, `1`, `0`, `yes`, `no`, `on` or `off`. The engine's own variables (`IZWI_HOST`, `IZWI_PORT`, `IZWI_MAX_*` engine limits and so on) are read by izwi-core and listed in core.md; the gateway role's are in gateway.md.

| Variable | Default | Meaning |
|---|---|---|
| `IZWI_DB_PATH` | the platform data directory's izwi.sqlite3 | SQLite store database path. |
| `IZWI_DATABASE_URL` | unset | Store database URL (`postgres://`, `sqlite://`); wins over IZWI_DB_PATH. |
| `IZWI_MEDIA_DIR` | the platform data directory's media | Root of stored media (uploads, generated audio, artifacts). |
| `IZWI_SPEECH_SPOOL_DIR` | the temp directory's izwi/speech-spool | Scratch root for local speech requests. |
| `IZWI_DB_MAX_CONNECTIONS` | `5` | Database pool size (1-1024). |
| `IZWI_DB_ACQUIRE_TIMEOUT_MS` | `3000` | How long a request waits for a database connection. |
| `IZWI_NODE_ID` | `local` | This node's id on a shared database (1-128 ASCII letters, digits, `-`, `_`); startup fails only the in-process jobs it owns. |
| `IZWI_API_KEY` | unset | API key required on /v1 and /internal routes (16-4096 printable bytes). Required for a non-loopback bind. |
| `IZWI_ALLOW_UNAUTHENTICATED_NETWORK` | `false` | Serve a non-loopback bind without IZWI_API_KEY (same as --allow-unauthenticated-network). |
| `IZWI_PRELOAD_MODELS` | unset | Comma-separated model ids loaded at start-up; readiness waits for them. |
| `IZWI_WARMUP_PRELOADED_MODELS` | `true` | Run a warm-up request on each preloaded model. |
| `IZWI_ASR_WARMUP_DURATION_MS` | `800` | Length of the ASR warm-up clip (100-5000). |
| `IZWI_GRANITE_DECODE_PROFILE` | unset | Set to 1 for the engine by --granite-decode-profile; not read here. |
| `IZWI_GRANITE_SPEECH_DTYPE` | unset | Set for the engine by --granite-speech-dtype; not read here. |
| `IZWI_DESKTOP_OWNER_PIPE` | unset | `1` when the desktop app owns this process: stdin closing shuts it down. |
| `IZWI_HTTP_SHUTDOWN_PRESTOP_SECS` | `0` | Pre-stop window at shutdown (0-60): readiness is unready while requests are still served. |
| `IZWI_HTTP_SHUTDOWN_GRACE_SECS` | `20` | How long in-flight HTTP requests may run after the pre-stop window (1-300); open chat streams get a terminal shutting-down event shortly before it ends. |
| `IZWI_BATCH_WORKER_CONCURRENCY` | the runtime's capacity | Durable jobs the local batch worker runs at once (at most the runtime's capacity). |
| `IZWI_BATCH_STAGE_TIMEOUT_SECS` | 0 (none) | Wall-clock limit of one durable stage (30-86400; 0 disables). |
| `IZWI_BATCH_WORKER_DRAIN_TIMEOUT_SECS` | `20` | How long the batch worker drains at shutdown (1-300). |
| `IZWI_MAX_REALTIME_REQUESTS` | the global request limit | Concurrent realtime-class requests. |
| `IZWI_MAX_INTERACTIVE_REQUESTS` | the global request limit | Concurrent interactive-class requests. |
| `IZWI_MAX_STREAMING_REQUESTS` | the global request limit | Concurrent streaming-class requests. |
| `IZWI_MAX_ONLINE_REQUESTS` | the global request limit | Concurrent online-class requests. |
| `IZWI_MAX_BATCH_REQUESTS` | the global request limit minus one | Concurrent batch-class requests. |
| `IZWI_MAX_BACKGROUND_REQUESTS` | the global request limit minus one | Concurrent background-class requests. |
| `IZWI_MAX_REALTIME_SESSIONS` | `32` | Open realtime websocket sessions. |
| `IZWI_MAX_RESPONSE_STORE_ENTRIES` | `512` | Stored Responses API responses kept in memory. |
| `IZWI_MAX_AGENT_SESSION_STORE_ENTRIES` | `512` | Agent sessions kept in memory. |
| `IZWI_OPENAI_COMPAT_PROFILE` | `strict` | OpenAI compatibility profile: `strict` or `relaxed`. |
| `IZWI_OPENAI_AUDIO_UPLOAD_LIMIT_MB` | 64 (relaxed) or the strict profile's limit | Largest audio upload on the OpenAI audio and media routes, in MiB. |
| `IZWI_MEDIA_DECODE_LANES` | `2` | Audio uploads decoded at once. |
| `IZWI_MAX_SPEECH_REQUESTS_PER_TENANT` | the global request limit | Concurrent speech requests per tenant (at most the global limit). |
| `IZWI_TTS_MAX_TEXT_BYTES` | `1048576` | Longest speech input one request may carry, in UTF-8 bytes. |
| `IZWI_TTS_LONG_FORM_CHUNK_MAX_CHARS` | the model's segment size | Characters per long-form speech segment (clamped to 80-4000). |
| `IZWI_FISH_LONG_FORM_ENABLED` | `true` | Allow Fish S2 requests that need more than one segment. |
| `IZWI_TTS_SEGMENT_TIMEOUT_SECS` | `1800` | How long a long-form segment may publish nothing before it fails. |
| `IZWI_TTS_STREAM_MAX_PCM_BYTES` | the model's allowance | Most PCM bytes one streamed or durable speech output may hold (at least 2). |
| `IZWI_AUDIO_STREAM_EVENT_QUEUE_CAPACITY` | `32` | Events a speech stream buffers for its reader. |
| `IZWI_AUDIO_STREAM_STALL_TIMEOUT_SECS` | `30` | How long a stalled speech stream reader is waited for. |
| `IZWI_AUDIO_STREAM_MAX_EVENT_BYTES` | `4194304` | Buffered event bytes per speech stream. |
| `IZWI_AUDIO_STREAM_TOTAL_EVENT_BYTES` | `67108864` | Buffered speech stream event bytes across all streams (at least 44). |
| `IZWI_TTS_TOTAL_SPOOL_BYTES` | `1073741824` | Disk spooled by speech outputs at once (at least 44). |
| `IZWI_TTS_TOTAL_UPLOAD_BYTES` | `268435456` | Speech reference upload bytes held at once (at least 44). |
| `IZWI_TTS_OPAQUE_FINAL_WAV_ENABLED` | `false` | Store durable speech outputs as opaque artifacts. |
| `IZWI_TTS_MAX_JOURNAL_BYTES` | `8589934592` | Replay journal bytes one durable speech job may write. |
| `IZWI_TTS_MAX_JOB_SECONDS` | `86400` | Age after which a durable speech job is not resumed. |
| `IZWI_TTS_REPLAY_RETENTION_SECONDS` | `86400` | How long the replay PCM of a terminal speech job is kept. |
| `IZWI_TTS_PCM_PUBLISH_INTERVAL_MS` | `250` | How often durable speech publishes PCM; 0 publishes every chunk. |
| `IZWI_TTS_PCM_PUBLISH_MAX_CHUNKS` | `8` | Chunks a durable speech publication may hold. |
| `IZWI_TTS_MAX_ACTIVE_JOBS` | `256` | Queued and running durable speech jobs in total. |
| `IZWI_TTS_MAX_ACTIVE_JOBS_PER_TENANT` | `32` | Queued and running durable speech jobs per tenant. |
| `IZWI_VOICE_WS_OUTBOUND_ITEMS` | `256` | Messages a voice websocket queues for its client. |
| `IZWI_VOICE_WS_OUTBOUND_AUDIO_BYTES` | `4194304` | Assistant audio bytes a voice websocket queues. |
| `IZWI_VOICE_WS_OUTBOUND_AUDIO_MS` | `5000` | Assistant audio a voice websocket queues, in milliseconds. |
| `IZWI_BATCH_WORKER_HEARTBEAT_STALE_SECS` | `5` | Heartbeat age after which readiness reports a batch worker stale. |
| `IZWI_IDEMPOTENCY_MAX_RECORDS` | `1048576` | Live idempotency records in total. |
| `IZWI_IDEMPOTENCY_MAX_RECORDS_PER_TENANT` | `65536` | Live idempotency records per tenant. |
| `IZWI_PROVIDER_WRITE_MAX_RESERVATIONS` | `65536` | Outstanding provider write reservations in total. |
| `IZWI_PROVIDER_WRITE_MAX_RESERVATIONS_PER_TENANT` | `16384` | Outstanding provider write reservations per tenant. |
| `IZWI_JOB_RETENTION_SECONDS` | `2592000` | How long terminal durable jobs are kept. |
| `IZWI_ARTIFACT_EPHEMERAL_RETENTION_SECONDS` | `3600` | How long ephemeral artifacts, and unreferenced input media at least, are kept. |
| `IZWI_WORKER_HEARTBEAT_RETENTION_SECONDS` | `86400` | How long heartbeat rows of stopped workers are kept. |
| `IZWI_MEDIA_TOMBSTONE_RETENTION_SECONDS` | `86400` | How long media tombstones outlive their objects. |

## Flags with an environment variable

Each of these flags can also be set through its variable; the flag wins when both are given.

| Variable | Flag | Default | Meaning |
|---|---|---|---|
| `IZWI_SERVER_ROLE` | `--role` | `local` | Process role: local inference server or hardware-independent gateway |
| `IZWI_BACKEND` | `--backend` | unset | Backend preference (`auto`, `cpu`, `metal`, `cuda`) |
| `IZWI_LOG_FORMAT` | `--log-format` | `text` | Log output format (`text`, `json`) |

## Build-time variables

Read by the compiler when the binary is built, not from the running process.

| Variable | Default | Meaning |
|---|---|---|
| `IZWI_BUILD_GIT_SHA` | `unknown` | Git commit reported by /internal/health and the start-up log. |
