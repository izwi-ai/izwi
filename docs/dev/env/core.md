# izwi-core environment

<!-- Generated from izwi_core::env::CORE_ENV; the crate's env::tests::reference_doc_is_current test keeps it current. -->

The variables the engine library (`izwi-core`) reads, in every binary that embeds it (`izwi-server`'s local role and `izwi-serving-worker`). The binaries' own variables are in the sibling files.

Booleans (`flag`) accept `true`, `1`, `yes`, `on` or `false`, `0`, `no`, `off`, in any case; a few also take the extra words listed against them. `auto/flag` variables take `auto` or a boolean. An empty value counts as unset. A boolean that does not parse stops the engine at startup with an error naming the variable.

*Read* says when the engine reads the variable: at `startup`, at each `model load`, at `first use` (then cached for the life of the process), by a `binary` that embeds the engine, or only by an opt-in `test harness`. No request or per-step path reads the environment.

| Variable | Kind | Default | Read | Meaning |
|---|---|---|---|---|
| `IZWI_ALLOW_SYNTHETIC_QWEN36_MOE_GEOMETRY` | flag | `false` | model load | Accepts non-published (synthetic) Qwen3.6-MoE checkpoint geometry for benchmark/CI fixtures; structural checks still apply. |
| `IZWI_ALLOW_SYNTHETIC_QWEN38_GEOMETRY` | flag | `false` | model load | Accepts non-27B (synthetic) Qwen3.8 checkpoint geometry for benchmark/CI fixtures; also switches load admission to measured estimates. |
| `IZWI_ASR_CHUNK_MAX_SECS` | number | `30` | first use | Sets the hard max long-form ASR chunk length in seconds; also used by diarization and LFM2.5 chunking. Non-positive ignored. |
| `IZWI_ASR_CHUNK_OVERLAP_SECS` | number | `3` | first use | Sets the overlap in seconds between consecutive long-form ASR chunks; also used by diarization and LFM2.5 chunking. |
| `IZWI_ASR_CHUNK_TARGET_SECS` | number | `24` | first use | Sets the preferred long-form ASR chunk length in seconds before silence-split search; also used by diarization and LFM2.5. |
| `IZWI_ASR_REALTIME_IDLE_TIMEOUT_SECS` | duration | engine per-backend default | startup | Seconds a realtime ASR session may sit idle before it is closed (u64; invalid ignored). |
| `IZWI_ASR_REALTIME_MAX_LIFETIME_SECS` | duration | engine per-backend default | startup | Maximum lifetime in seconds of one realtime ASR session (u64; invalid ignored). |
| `IZWI_ASR_REALTIME_MAX_SESSIONS` | integer | engine per-backend default | startup | Maximum concurrent realtime ASR sessions (positive integer; 0 or invalid ignored). |
| `IZWI_ASR_REALTIME_NOISE_GATE` | flag | `true` | first use | Silences non-speech before it reaches a realtime ASR model that hallucinates on it (Voxtral), at a 500 ms transcript delay; false lets noise reach the model. |
| `IZWI_ASR_STREAM_CHUNK_MAX_SECS` | number | 3.2 (audio <=12s) else min(long-form max, 8) | first use | Overrides the hard max of low-latency streaming ASR chunks; setting any STREAM_CHUNK var disables the single-chunk shortcut. |
| `IZWI_ASR_STREAM_CHUNK_MIN_SECS` | number | 1.0 (audio <=12s) else 2.0 | first use | Overrides the minimum chunk length in seconds for low-latency streaming ASR chunk planning. |
| `IZWI_ASR_STREAM_CHUNK_OVERLAP_SECS` | number | 0.45 (audio <=12s) else min(long-form overlap, 1.0) | first use | Overrides the overlap in seconds between low-latency streaming ASR chunks (capped at 45% of target). |
| `IZWI_ASR_STREAM_CHUNK_SILENCE_SEARCH_SECS` | number | 0.75 (audio <=12s) else 1.5 | first use | Overrides the silence-search window in seconds around streaming ASR chunk boundaries (capped at half the target). |
| `IZWI_ASR_STREAM_CHUNK_TARGET_SECS` | number | 2.4 (audio <=12s) else min(long-form target, 6) | first use | Overrides the preferred chunk length in seconds for low-latency streaming ASR chunk planning. |
| `IZWI_BACKEND` | choice | `auto` | startup | Selects the execution backend: auto, cpu, metal (alias mps) or cuda (alias gpu); unknown values ignored. |
| `IZWI_BUFFER_POOL` | flag | `true` | model load | Enables the global pre-allocated inference buffer pool (initialized by Qwen3 ASR load). |
| `IZWI_BUFFER_POOL_LARGE` | list | `4,262144` | model load | Sets the large buffer pool as "count,elements"; malformed values ignored. |
| `IZWI_BUFFER_POOL_MEDIUM` | list | `8,32768` | model load | Sets the medium buffer pool as "count,elements"; malformed values ignored. |
| `IZWI_BUFFER_POOL_SMALL` | list | `16,4096` | model load | Sets the small buffer pool as "count,elements"; malformed values ignored. |
| `IZWI_CAPACITY_REPLAY` | flag | `true` | startup | Enables recompute preemption: under managed KV pressure, suspend a low-priority session and replay its tokens instead of failing. |
| `IZWI_CHAT_DTYPE` | choice | checkpoint/backend default | model load | Overrides the Qwen3 dense chat model dtype (e.g. f32, f16, bf16); takes precedence over IZWI_QWEN_DTYPE. |
| `IZWI_CHUNKED_PREFILL_THRESHOLD` | integer | 0 (backend automatic chunk) | startup | Sets the chunked-prefill chunk size in tokens; 0 or invalid ignored, leaving the backend's automatic chunk. |
| `IZWI_CORS` | flag | `false` | startup | Enables CORS on the HTTP server. |
| `IZWI_CORS_ORIGINS` | list | `*` | startup | Comma-separated list of allowed CORS origins (entries trimmed, empties dropped). |
| `IZWI_CPU_MEMORY_BUDGET_BYTES` | bytes | derived from probed host memory | startup | Caps the CPU backend memory budget in bytes for admission; must be a positive integer or runtime construction fails. |
| `IZWI_CPU_PROJECTION_BACKEND` | choice | `auto` | startup | Chooses CPU block-FP8 projection residency: auto, q8 (requantize to Q8_0) or expanded (F32); invalid = startup error. |
| `IZWI_CUDA_CONTEXT_RESERVE_BYTES` | bytes | unset (max of 1 GiB reserve and 3% of device memory) | startup | Overrides CUDA memory kept free by the resident-context fit, in bytes; non-positive or unparsable values warn and are ignored. |
| `IZWI_CUDA_DECODE_GRAPHS` | auto/flag | `auto` | startup | Enables CUDA graph capture for decode (auto/off mode). |
| `IZWI_CUDA_DEVICE_ORDINAL` | integer | `0` | startup | Selects which CUDA device ordinal to open; invalid values warn and fall back to 0. |
| `IZWI_CUDA_DEVICE_SAMPLING` | auto/flag | `auto` | startup | Enables on-device token sampling on CUDA (auto/off mode). |
| `IZWI_CUDA_FUSED_DECODE` | auto/flag | `auto` | startup | Enables CUDA fused decode kernels (auto/off mode). |
| `IZWI_CUDA_HOST_MEMORY_BUDGET_BYTES` | bytes | unset (no explicit host cap) | startup | Caps the host memory budget in bytes on the CUDA backend; must be a positive integer or runtime construction fails. |
| `IZWI_CUDA_INCREMENTAL_CHAT` | flag | `true` | startup | Enables default-on CUDA incremental admission for replay-capable Qwen3.8 chat requests. |
| `IZWI_CUDA_MEMORY_BUDGET_BYTES` | bytes | unset (device-reported capacity) | startup | Caps the CUDA device memory budget in bytes for admission; must be a positive integer or runtime construction fails. |
| `IZWI_CUDA_MODE` | auto/flag | `auto` | startup | Master switch for CUDA performance features; off forces every CUDA feature mode off. |
| `IZWI_CUDA_MTP` | auto/flag | `auto` | startup | Enables the multi-token-prediction (MTP) head; an explicit true value also asks Qwen3.8 for MTP over graph decode. |
| `IZWI_CUDA_MTP_ADAPTIVE` | flag | `true` | startup | Enables adaptive MTP speculation (backs off drafting when acceptance is poor). |
| `IZWI_CUDA_MTP_DRAFT_TOKENS` | integer | `1` | startup | Number of MTP draft tokens per step (1 to 3; other values are a startup error). |
| `IZWI_CUDA_MTP_IN_CONTINUOUS` | auto/flag | `auto` | startup | Allows shared speculative MTP envelopes inside continuous batches, not just solo rows (auto/off mode). |
| `IZWI_CUDA_MTP_QUANTUM` | auto/flag | `auto` | startup | Enables the quantum (multi-step) MTP decode path (auto/off mode). |
| `IZWI_CUDA_PACKED_PROJECTIONS` | auto/flag | `auto` | startup | Enables packed (fused) projection weights on CUDA (auto/off mode). |
| `IZWI_CUDA_PROJECTION_BACKEND` | choice | `auto` | startup | Chooses CUDA block-FP8 projection backend: auto, q8 or native_fp8; invalid = startup error. |
| `IZWI_CUDA_RUNTIME_ACTIVE` | flag | `false` | startup | Marks that the process is running inside the packaged private CUDA runtime. |
| `IZWI_CUDA_RUNTIME_BINARY` | path | unset (search next to the executable and lib/izwi/runtime/cuda) | startup | Path to the private CUDA runtime binary, tried first before the built-in search locations. |
| `IZWI_CUDA_SPLIT_DECODE` | flag | `true` | model load | Enables the CUDA grouped-query split-decode attention kernel; read once per KV arena. |
| `IZWI_CUDA_STATE_BUDGET_FRACTION` | number | `0.25` | startup | Sets the share (0,1] of the post-weight CUDA budget retained recurrent state may take; the rest backs KV pages. |
| `IZWI_DIARIZATION_CAPACITY_WAIT_SECS` | duration | `120` | first use | Sets how long a diarization job waits for memory other admitted work holds (at admission and per stage) before failing; 0 fails at once. |
| `IZWI_ENABLE_CHUNKED_PREFILL` | flag | `false` | startup | Enables chunked prefill in the scheduler. |
| `IZWI_ENABLE_PREFIX_CACHING` | flag | false (catalog auto prefix reuse applies) | startup | Explicitly enables or kills prefix caching; any valid value also turns off catalog-auto prefix reuse. |
| `IZWI_FISH_NATIVE_BATCHING` | flag | `false` | model load | Enables native batched decoding for Fish S2 TTS when the model loads. |
| `IZWI_FISH_S2_BACKEND` | choice | auto (smoke.rs); required in streaming_smoke.rs | test harness | Backend for Fish S2 smoke tests: auto, cpu, metal, cuda (streaming smoke rejects auto). |
| `IZWI_FISH_S2_DTYPE` | choice | Fish S2 per-backend dtype policy | model load | Overrides the Fish Audio S2 Pro weight dtype (e.g. f32, f16, bf16); also used by the load memory estimate. |
| `IZWI_FISH_S2_MODEL_DIR` | path | unset (test fails) | test harness | Fish S2 Pro model directory; must end in the variant directory name. |
| `IZWI_FISH_S2_REFERENCE_TEXT` | text | contents of data/fox.md | test harness | Transcript of the reference WAV; required when IZWI_FISH_S2_REFERENCE_WAV is set. |
| `IZWI_FISH_S2_REFERENCE_WAV` | path | `data/fox.wav` | test harness | Reference voice WAV that conditions Fish S2 smoke synthesis. |
| `IZWI_FISH_S2_SMOKE_CONTEXT` | integer | `4096` | test harness | Max sequence length for the Fish S2 smoke runtime; an unparsable value fails the test. |
| `IZWI_FISH_S2_SMOKE_MAX_FRAMES` | integer | `512` | test harness | Max generated frames (max_tokens) for the Fish S2 smoke request. |
| `IZWI_FISH_S2_SMOKE_OUTPUT_WAV` | path | unset (no file written) | test harness | When set, writes the generated Fish S2 smoke audio to this WAV path. |
| `IZWI_FISH_S2_SMOKE_TEMPERATURE` | number | `0.0` | test harness | Sampling temperature for the Fish S2 smoke request. |
| `IZWI_FISH_S2_SMOKE_TOP_P` | number | `1.0` | test harness | Top-p for the Fish S2 smoke request. |
| `IZWI_FISH_S2_TARGET_TEXT` | text | the reference transcript | test harness | Text the Fish S2 smoke test synthesizes. |
| `IZWI_FUSED_KERNELS` | flag | `true` | first use | Enables fused Metal/CUDA (and some CPU) kernels. |
| `IZWI_GEMMA3_CUDA_RMS_NORM` | flag | true on CUDA (no effect elsewhere) | model load | Toggles the Candle CUDA RMSNorm kernel for Gemma3 norms on CUDA devices. |
| `IZWI_GGUF_MMAP` | choice | auto (mmap on all backends) | model load | Chooses GGUF file loading: auto, on (1\|true\|yes) for mmap, or off (0\|false\|no) for buffered I/O; unknown ignored. |
| `IZWI_GRANITE_ASR_PREFIX_REPLAY_WORDS` | integer | 0 (off) | first use | Sets how many trailing words of prior transcript Granite ASR replays as each chunk's prompt prefix; capped at 240. |
| `IZWI_GRANITE_CACHED_LINEAR_T` | flag | on for Metal, off elsewhere | model load | Caches a transposed copy of each Granite LM linear weight at load. |
| `IZWI_GRANITE_CHUNKED_STOP_CHECK` | flag | on for Metal (when no deltas/stop sequences) | first use | Checks for the Granite stop token once per interval instead of every decode step. |
| `IZWI_GRANITE_CHUNKED_STOP_INTERVAL` | integer | `8` | first use | Sets decode steps between Granite stop-token checks when chunked stop checking is on; clamped to 1..max_new_tokens. |
| `IZWI_GRANITE_DECODE_PROFILE` | flag | `false` | startup | Enables Granite per-step and per-layer decode profiling (or the server's --granite-decode-profile); IZWI_GRANITE_DECODE_PROFILING is an alias. |
| `IZWI_GRANITE_DECODE_PROFILING` | flag | `false` | startup | Alias of IZWI_GRANITE_DECODE_PROFILE, consulted only when that variable is unset. |
| `IZWI_GRANITE_DEFER_STOP_CHECK` | flag | `false` | first use | Defers Granite stop-token checks to the end of decode, running the whole token budget (opt-in). |
| `IZWI_GRANITE_F16_ATTENTION_CORE` | flag | on for Metal with F32 weights unless F32 pinned; needs F16 QKV | model load | Runs the Granite attention core in F16; IZWI_GRANITE_F16_ATTN_CORE is an alias. |
| `IZWI_GRANITE_F16_ATTENTION_OUTPUT` | flag | on for Metal with F32 weights unless F32 pinned | model load | Loads the Granite attention output projection as F16; IZWI_GRANITE_F16_O_PROJ is an alias. |
| `IZWI_GRANITE_F16_ATTN_CORE` | flag | on for Metal with F32 weights unless F32 pinned | model load | Alias of IZWI_GRANITE_F16_ATTENTION_CORE, consulted only when that variable is unset. |
| `IZWI_GRANITE_F16_LM_HEAD` | flag | on for Metal with F32 weights unless F32 pinned | model load | Uses an F16 Granite LM head (F16 copy of a tied embedding). |
| `IZWI_GRANITE_F16_MLP` | flag | on for Metal with F32 weights unless F32 pinned | model load | Loads Granite LM gate/up and down projections as F16. |
| `IZWI_GRANITE_F16_O_PROJ` | flag | on for Metal with F32 weights unless F32 pinned | model load | Alias of IZWI_GRANITE_F16_ATTENTION_OUTPUT, consulted only when that variable is unset. |
| `IZWI_GRANITE_F16_QKV` | flag | on for Metal with F32 weights unless F32 pinned | model load | Loads Granite LM Q/K/V projections as F16. |
| `IZWI_GRANITE_FUSED_RMS_NORM` | flag | on for Metal, off elsewhere | first use | Tries the fused RMSNorm kernel for Granite LM norms before the candle fallback. |
| `IZWI_GRANITE_FUSED_SILU_MUL` | flag | `true` | first use | Tries the fused SiLU-multiply kernel in the Granite LM MLP. |
| `IZWI_GRANITE_GATE_UP_PROJECTION_FUSION` | flag | on (falls back to IZWI_GRANITE_PROJECTION_FUSION) | model load | Concatenates Granite MLP gate and up projections into one matmul at load. |
| `IZWI_GRANITE_NATIVE_GREEDY_LOGITS` | flag | `true` | first use | Skips dividing Granite logits by logits_scaling (argmax unchanged); off restores scaled logits. |
| `IZWI_GRANITE_PROJECTION_FUSION` | flag | on (Metal, CUDA and CPU) | model load | Fuses Granite Q/K/V projections into one matmul at load; also the fallback for gate/up fusion. |
| `IZWI_GRANITE_RESIDUAL_PRESCALE` | flag | on for Metal, off elsewhere | model load | Pre-scales Granite o_proj and down_proj weights by the residual multiplier at load. |
| `IZWI_GRANITE_ROPE_BHTD_KERNEL` | flag | on for Metal, off elsewhere | first use | Allows the fused BHTD-layout RoPE kernel in Granite attention (even head_dim, F16/BF16/F32). |
| `IZWI_GRANITE_ROPE_CACHE_ATTENTION_DTYPE` | flag | on for Metal, off elsewhere | first use | Builds the Granite RoPE cache in F16 when the attention core runs in F16. |
| `IZWI_GRANITE_ROPE_KERNEL` | flag | on for CUDA, off elsewhere | first use | Allows the THD-layout fused RoPE kernel in Granite attention when the BSHD/BHTD kernels are not used. |
| `IZWI_GRANITE_ROPE_PAIR_BSHD_KERNEL` | flag | on for Metal, off elsewhere | first use | Allows the fused BSHD-layout RoPE pair kernel in Granite attention (even head_dim, F16/F32). |
| `IZWI_GRANITE_SAA_CHUNK_MAX_SECS` | number | 510 (and <=95% of the model context limit) | first use | Sets the max chunk length in seconds for Granite speaker-attributed ASR long-form planning. |
| `IZWI_GRANITE_SAA_CHUNK_MIN_SECS` | number | `30` | first use | Sets the minimum chunk length in seconds for Granite speaker-attributed ASR long-form planning. |
| `IZWI_GRANITE_SAA_CHUNK_OVERLAP_SECS` | number | `12` | first use | Sets the overlap in seconds between Granite speaker-attributed ASR chunks (capped at 45% of target). |
| `IZWI_GRANITE_SAA_CHUNK_SILENCE_SEARCH_SECS` | number | `12` | first use | Sets the silence-search window in seconds around Granite speaker-attributed ASR chunk boundaries. |
| `IZWI_GRANITE_SAA_CHUNK_TARGET_SECS` | number | `240` | first use | Sets the preferred chunk length in seconds for Granite speaker-attributed ASR long-form planning. |
| `IZWI_GRANITE_SAA_MAX_NEW_TOKENS` | integer | 256 + 8 per audio second, clamped 800..2500 | first use | Overrides the max new tokens per Granite speaker-attributed ASR chunk (positive integer). |
| `IZWI_GRANITE_SAA_PREFIX_MODE` | choice | `none` | first use | Selects the prefix fed to each Granite SAA chunk: full, full_transcript or legacy = full transcript; anything else = none. |
| `IZWI_GRANITE_SPEECH_DTYPE` | choice | auto (family/backend default) | startup | Forces the Granite Speech dtype (f32, f16, bf16 and aliases; --granite-speech-dtype wins); f32 also disables the Metal F16 encoder and LM paths. |
| `IZWI_GRANITE_TOKEN_DEBUG` | flag | `false` | first use | Adds per-token debug detail to Granite diagnostics; IZWI_GRANITE_TOKEN_DIAGNOSTICS is an alias. |
| `IZWI_GRANITE_TOKEN_DIAGNOSTICS` | flag | `false` | first use | Alias of IZWI_GRANITE_TOKEN_DEBUG, consulted only when that variable is unset. |
| `IZWI_HOST` | text | `127.0.0.1` | startup | Sets the HTTP server bind address. |
| `IZWI_INFERENCE_BROKER` | choice | `off` | startup | Sets the rollout inference broker mode: shadow or audit = shadow; enabled or a true boolean = on; anything else = off. |
| `IZWI_KEEP_NEMO_ARCHIVES` | flag | `false` | model load | Keeps the Parakeet and Nemotron ASR .nemo archives after a load from their extracted checkpoints; by default they are deleted (2.4-2.5 GB each). |
| `IZWI_KOKORO_CPU_PREDECODER` | flag | `true` | first use | Runs the Kokoro CPU pre-decoder stages in parallel; off/serial runs them sequentially. Also: `serial` means false, `sequential` means false. |
| `IZWI_KOKORO_CPU_RESBLOCKS` | flag | `true` | first use | Runs Kokoro decoder residual-block kernels in parallel on CPU; off/serial runs them sequentially. Also: `serial` means false, `sequential` means false. |
| `IZWI_KOKORO_CPU_STAGE_BRANCHES` | flag | `true` | first use | Runs the two branches of each Kokoro generator upsample stage in parallel on CPU; off/serial runs them sequentially. Also: `serial` means false, `sequential` means false. |
| `IZWI_KOKORO_METAL_HARMONIC_OVERLAP` | flag | `true` | first use | Overlaps Kokoro harmonic-source generation with decoder blocks on Metal; off/serial disables the overlap. Also: `serial` means false, `sequential` means false. |
| `IZWI_KOKORO_PROFILE` | flag | `false` | first use | Logs per-stage Kokoro TTS timings to stderr. |
| `IZWI_KOKORO_PROFILE_SYNC` | flag | `false` | first use | Synchronizes the device before each Kokoro generator-stage profile timing so stage times are accurate. |
| `IZWI_KV_DISABLE_OPTIMIZED_PROVIDER` | flag | `false` | model load | Demotes certified optimized KV/state providers to their certified portable provider. |
| `IZWI_KV_HOST_OFFLOAD` | flag | on when a host pool budget is set | binary | Read by izwi-serving-worker: false turns hierarchical KV offload off. |
| `IZWI_KV_HOST_POOL_BUDGET_BYTES` | integer | 0 (off) | binary | Read by izwi-serving-worker: Host KV pool byte budget that enables offload. |
| `IZWI_KV_METAL_EAGER_FENCES` | flag | `false` | model load | Drains the Metal device after every KV mutation instead of using epoch fences (kill switch). |
| `IZWI_KV_OFFLOAD_HIGH_WATERMARK` | number | `0.85` | binary | Read by izwi-serving-worker: Device KV occupancy fraction that starts offload. |
| `IZWI_KV_OFFLOAD_LOW_WATERMARK` | number | `0.70` | binary | Read by izwi-serving-worker: Device KV occupancy fraction that stops offload. |
| `IZWI_KV_OFFLOAD_MAX_IN_FLIGHT_PAGES` | integer | `8` | binary | Read by izwi-serving-worker: Maximum KV offload pages in flight at once. |
| `IZWI_KV_OFFLOAD_MAX_PROMOTION_PAGES` | integer | `64` | binary | Read by izwi-serving-worker: Maximum KV pages promoted back per step. |
| `IZWI_KV_PAGE_SIZE` | integer | `64` | startup | Sets the KV page size in tokens until the runtime publishes a configured size; invalid values warn once and use 64. |
| `IZWI_KV_SOAK_DURATION_SECONDS` | integer | `0` | test harness | Duration in seconds of the ignored managed-KV soak test. |
| `IZWI_KV_SOAK_ITERATIONS` | integer | `2` | test harness | Iteration count of the ignored managed-KV soak test. |
| `IZWI_LANE_STEP_DRIVERS` | flag | `false` | startup | Runs one step driver per execution lane (loaded model) instead of one for the whole engine. |
| `IZWI_LFM25_ASR_CHUNK_MAX_NEW_TOKENS` | integer | 8 per chunk second, clamped 128..256 | first use | Overrides max new tokens per LFM2.5-Audio ASR chunk when the request sets none (positive integer). |
| `IZWI_LFM25_ASR_CHUNK_MAX_SECS` | number | IZWI_ASR_CHUNK_MAX_SECS, else 30 | first use | Overrides the LFM2.5-Audio ASR hard max chunk length in seconds, taking precedence over IZWI_ASR_CHUNK_MAX_SECS. |
| `IZWI_LFM25_ASR_CHUNK_OVERLAP_SECS` | number | IZWI_ASR_CHUNK_OVERLAP_SECS, else 3 | first use | Overrides the LFM2.5-Audio ASR chunk overlap in seconds, taking precedence over IZWI_ASR_CHUNK_OVERLAP_SECS. |
| `IZWI_LFM25_ASR_CHUNK_TARGET_SECS` | number | IZWI_ASR_CHUNK_TARGET_SECS, else 24 | first use | Overrides the LFM2.5-Audio ASR target chunk length in seconds, taking precedence over IZWI_ASR_CHUNK_TARGET_SECS. |
| `IZWI_LFM25_ASR_MAX_NEW_TOKENS` | integer | `1024` | first use | Overrides single-pass max new tokens for LFM2.5-Audio ASR when the request sets none (positive integer). |
| `IZWI_LFM25_ASR_STOP_CHECK_INTERVAL` | integer | `92` | first use | Sets LFM2.5-Audio ASR decode tokens between host stop checks (clamped 1-128); >1 enables deferred device decode on GPU. |
| `IZWI_LFM25_METAL_PREFILL_SDPA` | flag | `true` | first use | Enables the Metal fused SDPA path for LFM2.5 full-attention prefill. |
| `IZWI_LFM25_TTS_AUDIO_STOP_CHECK_INTERVAL` | integer | `20` | first use | Sets LFM2.5-Audio TTS audio frames between stop checks under greedy audio sampling (clamped 1-32). |
| `IZWI_LFM2_DIAGNOSTICS` | flag | `false` | first use | Enables opt-in LFM2 activation finiteness checks and token diagnostics during forward passes and sampling. |
| `IZWI_LFM_CUDA_DEPTHWISE_CONV` | flag | `false` | first use | Enables the experimental chunked CUDA depthwise Conv1d path in the LFM2.5-Audio conformer. |
| `IZWI_LOADING_CACHE_DIR` | path | unset (loader default location) | startup | Directory for the derived-weight loading cache. |
| `IZWI_LOADING_CACHE_MAX_BYTES` | bytes | 68719476736 (64 GiB) | startup | Size cap in bytes for the derived-weight loading cache (u64; invalid = startup error). |
| `IZWI_LOADING_DERIVED_WEIGHT_CACHE` | auto/flag | `auto` | startup | Enables caching of derived (converted) weights on disk across loads (auto/off mode). |
| `IZWI_LOADING_IO_STRATEGY` | choice | `auto` | startup | Chooses weight-file I/O during load: auto, mmap or sequential; invalid = startup error. |
| `IZWI_LOADING_MAX_STAGING_BYTES` | bytes | 268435456 (256 MiB) | startup | Caps host staging memory in bytes used while loading weights (must be > 0; invalid = startup error). |
| `IZWI_LOADING_MODE` | auto/flag | `auto` | startup | Master switch for loading optimizations; off forces derived cache, parallel conversion and pinned uploads off. |
| `IZWI_LOADING_PARALLEL_CONVERSION` | auto/flag | `auto` | startup | Enables parallel weight conversion during model load (auto/off mode). |
| `IZWI_LOADING_PINNED_UPLOADS` | auto/flag | `auto` | startup | Enables pinned host memory for device weight uploads (auto/off mode). |
| `IZWI_LOADING_WORKERS` | integer | 0 (automatic) | startup | Number of loader worker threads; 0 picks automatically (invalid = startup error). |
| `IZWI_MANAGED_CHAT_CLAIM_QUANTUM_TOKENS` | integer | `1024` | startup | Sets the token quantum for managed chat KV claims; 0 restores full up-front claims; unparsable keeps 1024. |
| `IZWI_MANAGED_PREFIX_CACHE_SALT` | text | `unset` | startup | Salt mixed into managed prefix-cache keys to isolate cache entries. |
| `IZWI_MAX_BATCH_SIZE` | choice | `auto` | startup | Sets the maximum batch rows: auto or a positive integer; invalid ignored. |
| `IZWI_MAX_CONCURRENT` | integer | `100` | startup | Maximum concurrent HTTP requests (legacy alias MAX_CONCURRENT_REQUESTS; 0 or invalid ignored). |
| `IZWI_MAX_LOADED_MODELS` | integer | `4` | startup | Maximum models resident at once (residency guardrail; 0 or invalid ignored, floor 1). |
| `IZWI_MAX_PHYSICAL_IN_FLIGHT` | integer | `1` | startup | Engine-wide ceiling on overlapping physical model launches (1 to 1024; invalid ignored). |
| `IZWI_MAX_PREFIX_CACHE_PAGES` | integer | unset (automatic) | startup | Prefix-cache page budget; 0 or invalid ignored, leaving the automatic budget. |
| `IZWI_MAX_PREFIX_SNAPSHOTS` | integer | `8` | startup | Caps committed hybrid recurrent-state prefix snapshots per model (0..=1024; 0 keeps hybrid state private). |
| `IZWI_MAX_QUEUED_REQUESTS` | integer | `128` | startup | Maximum requests waiting in the engine queue (0 or invalid ignored). |
| `IZWI_MAX_RETAINED_SEQUENCES` | integer | 0 (automatic device ceiling) | startup | Maximum retained sequences; 0 or invalid ignored, leaving the automatic ceiling. |
| `IZWI_MAX_SCHEDULER_BATCH_SIZE` | integer | 0 (automatic device ceiling) | startup | Maximum scheduler batch size; 0 or invalid ignored, leaving the automatic ceiling. |
| `IZWI_MAX_SEQUENCE_LENGTH` | choice | `auto` | startup | Context-length cap: auto (model's context) or a positive token count; invalid ignored. |
| `IZWI_MAX_STAGED_TRANSACTIONS` | integer | 0 (automatic device ceiling) | startup | Maximum staged transactions; 0 or invalid ignored, leaving the automatic ceiling. |
| `IZWI_MAX_TOKENS_PER_STEP` | integer | 0 (backend automatic budget) | startup | Scheduler token budget per step; 0 or invalid ignored, leaving the backend's automatic budget. |
| `IZWI_MEDIA_LOCAL_ROOT` | path | unset (local file media disabled) | first use | Sets the absolute directory Qwen3.5 vision may read file:// media from; unset disables local media. |
| `IZWI_METAL_DTYPE` | choice | `auto` | model load | Overrides the Metal compute dtype for all families: auto, f32, f16 or bf16 (aliases accepted); invalid warns, uses auto. |
| `IZWI_METAL_GQA_DECODE` | flag | `true` | first use | Enables the Metal simdgroup GQA decode attention kernels; off restores scalar kernels. |
| `IZWI_METAL_LOAD_ADMISSION` | choice | `advisory` | startup | How host memory gates Metal loads and requests: advisory (only the GPU working set gates; a load larger than macOS reports available is logged and reported), strict (host memory gates too) or off (no live check before a load); invalid = startup error. |
| `IZWI_METAL_MEMORY_BUDGET_BYTES` | bytes | derived from the Metal working set | startup | Caps the Metal memory budget in bytes for admission; must be a positive integer or runtime construction fails. |
| `IZWI_METAL_PROJECTION_BACKEND` | choice | `auto` | startup | Chooses Metal block-FP8 projection residency: auto, native_fp8 (keep FP8) or expanded (F16); invalid = startup error. |
| `IZWI_METAL_SDPA_F32_PREFILL_F16` | flag | `true` | first use | Casts F32 Metal prefill attention (q_seq > 8) to F16 for the SDPA kernel. |
| `IZWI_METAL_SDPA_MASKED` | flag | `false` | first use | Allows Metal SDPA for masked attention when the mask shape is supported. |
| `IZWI_MIXED_FORCE_SPLIT` | flag | `false` | first use | Runs mixed prefill+decode batches through the split prefill and decode handlers (parity oracle and kill switch). |
| `IZWI_MIXED_STEP` | flag | certified table (currently off for all) | model load | Overrides whether a loaded chat model publishes mixed prefill+decode steps. |
| `IZWI_MODELS_DIR` | path | `<data_local_dir>/izwi/models` | startup | Directory where model weights are stored and looked up. |
| `IZWI_MODEL_KEEP_ALIVE_SECS` | duration | `600` | startup | Idle keep-alive in seconds before auto-loaded models are unloaded; 0 disables the reaper. |
| `IZWI_NEMOTRON_ASR_DTYPE` | choice | family/backend default (IZWI_METAL_DTYPE on Metal) | model load | Overrides the Nemotron ASR dtype (e.g. f32, f16, bf16); on Metal f32 also keeps encoder projections F32. |
| `IZWI_NEMOTRON_METAL_DEPTHWISE_CONV1D` | flag | true (Metal only) | first use | Enables the custom Metal depthwise Conv1d kernel for Nemotron ASR. |
| `IZWI_NEMOTRON_METAL_DEPTHWISE_CONV2D` | flag | true (Metal only) | first use | Enables the custom Metal depthwise Conv2d kernel for Nemotron ASR. |
| `IZWI_NEMOTRON_REALTIME_MAX_SECONDS` | integer | `3600` | first use | Caps a Nemotron realtime ASR session length in seconds and sizes its reservation; a non-integer value is a config error. |
| `IZWI_NO_UI` | flag | false (UI enabled) | startup | Disables serving the web UI when true (inverted into ui_enabled). |
| `IZWI_NUM_THREADS` | integer | min(available cores, 8) (4 if unknown) | startup | Number of compute threads (0 or invalid ignored; floor 1). |
| `IZWI_OFFLINE` | flag | `false` | startup | Never contacts Hugging Face: downloads are refused and download sizes come from the built-in estimates. |
| `IZWI_PARAKEET_ASR_DTYPE` | choice | F16 encoder projections on Metal, else F32 | model load | Parakeet dtype override: on Metal f32 keeps every weight F32 and f16 stores encoder projections in F16. |
| `IZWI_PARAKEET_METAL_CONV1D_BN_SWISH` | flag | true (Metal only) | first use | Enables the fused Metal depthwise Conv1d + bias + Swish kernel for Parakeet ASR conformer convs. |
| `IZWI_PARAKEET_METAL_DEPTHWISE_CONV1D` | flag | true (Metal only) | first use | Enables the custom Metal depthwise Conv1d kernel for Parakeet ASR. |
| `IZWI_PARAKEET_METAL_DEPTHWISE_CONV2D` | flag | true (Metal only) | first use | Enables the custom Metal depthwise Conv2d kernel for Parakeet ASR. |
| `IZWI_PERFORMANCE_CPU_PROJECTION_BACKEND` | choice | `auto` | startup | Deprecated alias of IZWI_CPU_PROJECTION_BACKEND, used only when the canonical name is unset. |
| `IZWI_PERFORMANCE_CUDA_DECODE_GRAPHS` | auto/flag | `auto` | startup | Deprecated alias of IZWI_CUDA_DECODE_GRAPHS, used only when the canonical name is unset. |
| `IZWI_PERFORMANCE_CUDA_DEVICE_SAMPLING` | auto/flag | `auto` | startup | Deprecated alias of IZWI_CUDA_DEVICE_SAMPLING, used only when the canonical name is unset. |
| `IZWI_PERFORMANCE_CUDA_FUSED_DECODE` | auto/flag | `auto` | startup | Deprecated alias of IZWI_CUDA_FUSED_DECODE, used only when the canonical name is unset. |
| `IZWI_PERFORMANCE_CUDA_MODE` | auto/flag | `auto` | startup | Deprecated alias of IZWI_CUDA_MODE, used only when the canonical name is unset. |
| `IZWI_PERFORMANCE_CUDA_MTP` | auto/flag | `auto` | startup | Deprecated alias of IZWI_CUDA_MTP (canonical unset only); a true value also requests the Qwen3.8 MTP head explicitly. |
| `IZWI_PERFORMANCE_CUDA_MTP_ADAPTIVE` | flag | `true` | startup | Deprecated alias of IZWI_CUDA_MTP_ADAPTIVE; first non-empty alias is used when the canonical name is unset. |
| `IZWI_PERFORMANCE_CUDA_MTP_DRAFT_TOKENS` | integer | `1` | startup | Deprecated alias of IZWI_CUDA_MTP_DRAFT_TOKENS (1 to 3), used only when the canonical name is unset. |
| `IZWI_PERFORMANCE_CUDA_MTP_IN_CONTINUOUS` | auto/flag | `auto` | startup | Deprecated alias of IZWI_CUDA_MTP_IN_CONTINUOUS, used only when the canonical name is unset. |
| `IZWI_PERFORMANCE_CUDA_MTP_QUANTUM` | auto/flag | `auto` | startup | Deprecated alias of IZWI_CUDA_MTP_QUANTUM, used only when the canonical name is unset. |
| `IZWI_PERFORMANCE_CUDA_PACKED_PROJECTIONS` | auto/flag | `auto` | startup | Deprecated alias of IZWI_CUDA_PACKED_PROJECTIONS, used only when the canonical name is unset. |
| `IZWI_PERFORMANCE_CUDA_PROJECTION_BACKEND` | choice | `auto` | startup | Deprecated alias of IZWI_CUDA_PROJECTION_BACKEND (auto, q8, native_fp8), used only when the canonical name is unset. |
| `IZWI_PERFORMANCE_LOADING_CACHE_DIR` | path | unset (loader default location) | startup | Deprecated alias of IZWI_LOADING_CACHE_DIR, used only when the canonical name is unset. |
| `IZWI_PERFORMANCE_LOADING_CACHE_MAX_BYTES` | bytes | 68719476736 (64 GiB) | startup | Deprecated alias of IZWI_LOADING_CACHE_MAX_BYTES, used only when the canonical name is unset. |
| `IZWI_PERFORMANCE_LOADING_DERIVED_WEIGHT_CACHE` | auto/flag | `auto` | startup | Deprecated alias of IZWI_LOADING_DERIVED_WEIGHT_CACHE, used only when the canonical name is unset. |
| `IZWI_PERFORMANCE_LOADING_IO_STRATEGY` | choice | `auto` | startup | Deprecated alias of IZWI_LOADING_IO_STRATEGY (auto, mmap, sequential), used only when the canonical name is unset. |
| `IZWI_PERFORMANCE_LOADING_MAX_STAGING_BYTES` | bytes | 268435456 (256 MiB) | startup | Deprecated alias of IZWI_LOADING_MAX_STAGING_BYTES, used only when the canonical name is unset. |
| `IZWI_PERFORMANCE_LOADING_MODE` | auto/flag | `auto` | startup | Deprecated alias of IZWI_LOADING_MODE, used only when the canonical name is unset. |
| `IZWI_PERFORMANCE_LOADING_PARALLEL_CONVERSION` | auto/flag | `auto` | startup | Deprecated alias of IZWI_LOADING_PARALLEL_CONVERSION, used only when the canonical name is unset. |
| `IZWI_PERFORMANCE_LOADING_PINNED_UPLOADS` | auto/flag | `auto` | startup | Deprecated alias of IZWI_LOADING_PINNED_UPLOADS, used only when the canonical name is unset. |
| `IZWI_PERFORMANCE_LOADING_WORKERS` | integer | 0 (automatic) | startup | Deprecated alias of IZWI_LOADING_WORKERS, used only when the canonical name is unset. |
| `IZWI_PERFORMANCE_METAL_PROJECTION_BACKEND` | choice | `auto` | startup | Deprecated alias of IZWI_METAL_PROJECTION_BACKEND (auto, native_fp8, expanded), used only when the canonical name is unset. |
| `IZWI_PHYSICAL_EXECUTION_MODE` | choice | `serial` | startup | Physical execution mode for model launches: serial, shadow or concurrent; invalid ignored. |
| `IZWI_PORT` | integer | `8080` | startup | Sets the HTTP server port (u16; invalid ignored). |
| `IZWI_POWER_SAVE` | flag | `false` | startup | Forces power-save scheduling mode. |
| `IZWI_PREFIX_REUSE_AUTO` | flag | `true` | startup | Enables catalog-driven automatic prefix reuse per loaded model where evidence exists. |
| `IZWI_QWEN35_CUDA_GRAPH` | flag | `false` | model load | Opts Qwen3.5 decode into CUDA graph capture (off until validated on GPU). |
| `IZWI_QWEN35_DECODE_QUANTUM` | integer | 8 on CUDA/Metal, 1 on CPU | first use | Sets Qwen3.5 decode tokens per engine step for a request decoding alone (capped at 16; 0 or invalid ignored). |
| `IZWI_QWEN35_FINITE_DIAGNOSTICS` | flag | `false` | model load | Enables finite-value (NaN/Inf) checks on Qwen3.5 and Qwen3.6-MoE activations for debugging. |
| `IZWI_QWEN35_FUSED_DECODE` | flag | true (fused) | model load | Kill switch: legacy/off forces the Candle op chain instead of the fused Qwen3.5 DeltaNet decode. Also: `legacy` means false. |
| `IZWI_QWEN35_KQUANT` | flag | true (k-quant kernels) | model load | Kill switch: legacy/off keeps every Qwen3.5 projection on QMatMul instead of the CUDA k-quant kernels. Also: `legacy` means false. |
| `IZWI_QWEN35_PREFILL_CHUNK_SIZE` | integer | 256 (Qwen3.6-MoE may prefer up to 1024) | first use | Sets the Qwen3.5/Qwen3.6-MoE prefill forward chunk size in tokens (max 2048; 0 or invalid ignored). |
| `IZWI_QWEN35_ROPE_KERNEL` | flag | true on Metal/CUDA, false on CPU | model load | Toggles the custom RoPE kernel for Qwen3.5 and Qwen3.6-MoE attention. |
| `IZWI_QWEN35_TILED_RECURRENCE` | flag | `true` | model load | Enables the tiled DeltaNet recurrence for Qwen3.5 and Qwen3.6-MoE prefill. |
| `IZWI_QWEN35_TILED_RECURRENCE_TILE_SIZE` | integer | auto by sequence length (16-64) | model load | Overrides the Qwen3.5/Qwen3.6-MoE tiled DeltaNet recurrence tile size (min 1, capped at sequence length). |
| `IZWI_QWEN36_CHUNKED_GDN` | flag | `true` | model load | Kill switch: legacy/off disables the chunked DeltaNet prefill kernels for Qwen3.6-MoE. Also: `legacy` means false. |
| `IZWI_QWEN36_CUDA_BF16_KV` | flag | true (CUDA SM 8.0+ only) | model load | Toggles BF16 persistent KV cache storage for Qwen3.6-MoE on CUDA; other backends keep F32. |
| `IZWI_QWEN36_CUDA_GRAPHS` | flag | `true` | model load | Kill switch: legacy/off disables CUDA graph decode for Qwen3.6-MoE. Also: `legacy` means false. |
| `IZWI_QWEN36_DECODE_QUANTUM` | integer | `8` | first use | Sets Qwen3.6-MoE tokens per engine step for a stream decoding alone on graph decode (1-64, else default). |
| `IZWI_QWEN36_FP8_LM_HEAD` | flag | true (CUDA native-FP8 residency only) | model load | Kill switch: legacy/off keeps the Qwen3.6-MoE LM head in BF16 instead of block FP8. Also: `legacy` means false. |
| `IZWI_QWEN36_FUSED_DECODE` | flag | `true` | model load | Kill switch: legacy/off keeps Qwen3.6-MoE decode on the Candle op chain (fused DeltaNet, norms, q/k, gate, FP8 GEMV). Also: `legacy` means false. |
| `IZWI_QWEN36_GROUPED_MMA` | flag | `true` | model load | Kill switch: legacy/off disables the tensor-core grouped MoE prefill kernels for Qwen3.6-MoE. Also: `legacy` means false. |
| `IZWI_QWEN36_MOE_BACKEND` | choice | `auto` | model load | Selects the Qwen3.6-MoE sparse-expert path: auto (fused kernels when allowed), or legacy or a false boolean (host-routed loop). |
| `IZWI_QWEN36_MTP_HANDOFF` | flag | `false` | model load | Explicit opt-in that, with the MTP performance knob, lets Qwen3.6-MoE load its multi-token-prediction draft head. |
| `IZWI_QWEN36_REAL_CHECKPOINT_E2E` | path | unset (probe skipped) | test harness | Real Qwen3.6 checkpoint directory for the ASCII-coherence end-to-end probe. |
| `IZWI_QWEN36_STEP_GRAPH` | flag | `true` | model load | Kill switch: legacy/off disables the Qwen3.6-MoE whole-step CUDA graph; piecewise graph segments remain. Also: `legacy` means false. |
| `IZWI_QWEN38_CAUSAL_CONV_DECODE` | auto/flag | `auto` | startup | Legacy Qwen3.8 alias folded into IZWI_CUDA_FUSED_DECODE, used only when the canonical name is unset. |
| `IZWI_QWEN38_CPU_LM_HEAD` | flag | true (Q8_0 head on CPU) | model load | Uses the Q8_0 Qwen3.8 CPU LM head while CPU projections are Q8_0; false keeps the head dense F32. Also: `dense` means false, `legacy` means false. |
| `IZWI_QWEN38_CPU_PROJECTION_BACKEND` | choice | `auto` | startup | Deprecated alias of IZWI_CPU_PROJECTION_BACKEND (auto, expanded, q8), used only when the canonical name is unset. |
| `IZWI_QWEN38_CUDA_BF16_KV` | flag | true (CUDA SM 8.0+ only) | model load | Toggles BF16 persistent KV cache storage for Qwen3.8 on CUDA; false falls back to F16. |
| `IZWI_QWEN38_CUDA_GRAPHS` | flag | `true` | model load | Kill switch: legacy/off disables Qwen3.8 CUDA graph decode (eager fused path); global switch is IZWI_CUDA_DECODE_GRAPHS. Also: `legacy` means false. |
| `IZWI_QWEN38_DECODE_EPILOGUES` | auto/flag | `auto` | startup | Legacy Qwen3.8 alias folded into IZWI_CUDA_FUSED_DECODE, used only when the canonical name is unset. |
| `IZWI_QWEN38_DECODE_GRAPHS` | auto/flag | `auto` | startup | Deprecated alias of IZWI_CUDA_DECODE_GRAPHS, used only when the canonical name is unset. |
| `IZWI_QWEN38_DECODE_PIPELINE` | flag | `true` | first use | Kill switch: false runs each Qwen3.8 graph-decode quantum step by step instead of pipelined. Also: `legacy` means false. |
| `IZWI_QWEN38_DECODE_QUANTUM` | integer | `8` | first use | Sets Qwen3.8 tokens per engine step under CUDA graph decode without MTP (1-64, else the default). |
| `IZWI_QWEN38_DELTANET_DECODE` | auto/flag | `auto` | startup | Legacy Qwen3.8 alias folded into IZWI_CUDA_FUSED_DECODE, used only when the canonical name is unset. |
| `IZWI_QWEN38_DEVICE_SAMPLING` | auto/flag | `auto` | startup | Deprecated alias of IZWI_CUDA_DEVICE_SAMPLING, used only when the canonical name is unset. |
| `IZWI_QWEN38_FINITE_DIAGNOSTICS` | flag | `false` | model load | Enables finite-value (NaN/Inf) checks on Qwen3.8 activations for debugging. |
| `IZWI_QWEN38_FP8_GEMV` | flag | `true` | model load | Kill switch: legacy/off runs the reference FP8 decode GEMV instead of the vectorized one for Qwen3.8. Also: `legacy` means false. |
| `IZWI_QWEN38_FP8_LM_HEAD` | flag | `true` | model load | Kill switch: legacy/off keeps the Qwen3.8 LM head in the activation dtype instead of block FP8. Also: `legacy` means false. |
| `IZWI_QWEN38_FP8_PREFILL_GEMM` | flag | true (auto) | model load | Kill switch: legacy/off disables the compact dequantized FP8 prefill GEMM for long Qwen3.8 CUDA spans. Also: `auto` means true, `legacy` means false. |
| `IZWI_QWEN38_FP8_PROJECTIONS` | flag | `true` | model load | Kill switch: legacy/off keeps Qwen3.8 CUDA projections in Q8_0 instead of native block FP8. Also: `legacy` means false. |
| `IZWI_QWEN38_FUSED_AB_GEMV` | flag | `true` | model load | Kill switch: legacy/off sends Qwen3.8 DeltaNet alpha/beta projections through cuBLAS instead of the router GEMV. Also: `legacy` means false. |
| `IZWI_QWEN38_FUSED_ATTN_GATE` | flag | `true` | model load | Kill switch: legacy/off disables the one-launch Qwen3.8 attention output gate. Also: `legacy` means false. |
| `IZWI_QWEN38_FUSED_DECODE` | auto/flag | `auto` | startup | Deprecated alias of IZWI_CUDA_FUSED_DECODE, used only when the canonical name is unset. |
| `IZWI_QWEN38_FUSED_GDN` | flag | `true` | model load | Kill switch: legacy/off disables the fused Qwen3.8 DeltaNet (GDN) decode. Also: `legacy` means false. |
| `IZWI_QWEN38_FUSED_KV_WRITE` | flag | `true` | model load | Kill switch: legacy/off disables writing K/V from the fused Qwen3.8 q/k launch. Also: `legacy` means false. |
| `IZWI_QWEN38_FUSED_NORM` | flag | `true` | model load | Kill switch: legacy/off disables the fused Qwen3.8 residual add + RMSNorm. Also: `legacy` means false. |
| `IZWI_QWEN38_FUSED_QK_ROPE` | flag | `true` | model load | Kill switch: legacy/off disables the fused Qwen3.8 q/k norm + partial M-RoPE. Also: `legacy` means false. |
| `IZWI_QWEN38_GDN_PREFILL` | flag | true (auto) | model load | Kill switch: legacy/off disables the grouped Qwen3.8 DeltaNet prefill kernel (q/k expanded instead). Also: `auto` means true, `legacy` means false. |
| `IZWI_QWEN38_LOADING_CACHE_DIR` | path | unset (loader default location) | startup | Deprecated alias of IZWI_LOADING_CACHE_DIR, used only when the canonical name is unset. |
| `IZWI_QWEN38_LOADING_CACHE_MAX_BYTES` | bytes | 68719476736 (64 GiB) | startup | Deprecated alias of IZWI_LOADING_CACHE_MAX_BYTES, used only when the canonical name is unset. |
| `IZWI_QWEN38_LOADING_DERIVED_WEIGHT_CACHE` | auto/flag | `auto` | startup | Deprecated alias of IZWI_LOADING_DERIVED_WEIGHT_CACHE, used only when the canonical name is unset. |
| `IZWI_QWEN38_LOADING_IO_STRATEGY` | choice | `auto` | startup | Deprecated alias of IZWI_LOADING_IO_STRATEGY (auto, mmap, sequential), used only when the canonical name is unset. |
| `IZWI_QWEN38_LOADING_MAX_STAGING_BYTES` | bytes | 268435456 (256 MiB) | startup | Deprecated alias of IZWI_LOADING_MAX_STAGING_BYTES, used only when the canonical name is unset. |
| `IZWI_QWEN38_LOADING_MODE` | auto/flag | `auto` | startup | Deprecated alias of IZWI_LOADING_MODE, used only when the canonical name is unset. |
| `IZWI_QWEN38_LOADING_PARALLEL_CONVERSION` | auto/flag | `auto` | startup | Deprecated alias of IZWI_LOADING_PARALLEL_CONVERSION, used only when the canonical name is unset. |
| `IZWI_QWEN38_LOADING_PINNED_UPLOADS` | auto/flag | `auto` | startup | Deprecated alias of IZWI_LOADING_PINNED_UPLOADS, used only when the canonical name is unset. |
| `IZWI_QWEN38_LOADING_WORKERS` | integer | 0 (automatic) | startup | Deprecated alias of IZWI_LOADING_WORKERS, used only when the canonical name is unset. |
| `IZWI_QWEN38_LOAD_TIMINGS` | flag | true (CUDA only) | model load | Kill switch: legacy/off skips the Qwen3.8 load-time CUDA kernel timing measurements reported in diagnostics. Also: `legacy` means false. |
| `IZWI_QWEN38_METAL_PROJECTION_BACKEND` | choice | `auto` | startup | Deprecated alias of IZWI_METAL_PROJECTION_BACKEND (auto, native_fp8, expanded), used only when the canonical name is unset. |
| `IZWI_QWEN38_MODE` | auto/flag | `auto` | startup | Deprecated alias of IZWI_CUDA_MODE, used only when the canonical name is unset. |
| `IZWI_QWEN38_MTP` | auto/flag | `auto` | startup | Deprecated alias of IZWI_CUDA_MTP (canonical unset only); a true value also requests the Qwen3.8 MTP head explicitly. |
| `IZWI_QWEN38_MTP_ADAPTIVE` | flag | `true` | startup | Deprecated alias of IZWI_CUDA_MTP_ADAPTIVE; first non-empty alias is used when the canonical name is unset. |
| `IZWI_QWEN38_MTP_DRAFT_TOKENS` | integer | `1` | startup | Deprecated alias of IZWI_CUDA_MTP_DRAFT_TOKENS (1 to 3), used only when the canonical name is unset. |
| `IZWI_QWEN38_MTP_IN_CONTINUOUS` | auto/flag | `auto` | startup | Deprecated alias of IZWI_CUDA_MTP_IN_CONTINUOUS, used only when the canonical name is unset. |
| `IZWI_QWEN38_MTP_QUANTUM` | auto/flag | `auto` | startup | Deprecated alias of IZWI_CUDA_MTP_QUANTUM, used only when the canonical name is unset. |
| `IZWI_QWEN38_PACKED_PROJECTIONS` | auto/flag | `auto` | startup | Deprecated alias of IZWI_CUDA_PACKED_PROJECTIONS, used only when the canonical name is unset. |
| `IZWI_QWEN38_PREFILL_CHUNK_SIZE` | integer | 1024 on CUDA with the FP8 prefill GEMM, else 256 | model load | Sets the Qwen3.8 prefill forward chunk size in tokens (max 2048; 0 or invalid ignored). |
| `IZWI_QWEN38_PREFIX_REUSE` | flag | `true` | model load | Qwen3.8-only kill switch for cross-turn prefix reuse; an invalid value fails the load. |
| `IZWI_QWEN38_PROJECTION_BACKEND` | choice | `auto` | startup | Deprecated alias of IZWI_CUDA_PROJECTION_BACKEND (auto, q8, native_fp8), used only when the canonical name is unset. |
| `IZWI_QWEN38_ROPE_KERNEL` | auto/flag | true on Metal/CUDA (CPU never) | model load | Toggles the Qwen3.8 fused RoPE kernel at load; also a legacy alias folded into IZWI_CUDA_FUSED_DECODE. |
| `IZWI_QWEN38_STEP_GRAPH` | flag | `true` | model load | Kill switch: legacy/off disables the Qwen3.8 whole-step CUDA graph; piecewise graph segments remain. Also: `legacy` means false. |
| `IZWI_QWEN38_TILED_RECURRENCE` | auto/flag | `true` | model load | Toggles Qwen3.8 tiled DeltaNet recurrence at load; also a legacy alias folded into IZWI_CUDA_FUSED_DECODE. |
| `IZWI_QWEN38_TILED_RECURRENCE_TILE_SIZE` | integer | auto by sequence length (16-64) | model load | Overrides the Qwen3.8 tiled DeltaNet recurrence tile size (min 1, capped at sequence length). |
| `IZWI_QWEN3_ASR_GGUF_QMATMUL_TEXT` | flag | `true` | model load | Toggles quantized QMatMul execution for the Qwen3-ASR GGUF text decoder; also used by load memory planning. |
| `IZWI_QWEN3_DENSE_PROJECTION_FUSION` | flag | true on Metal, false elsewhere | model load | Toggles fusing Qwen3 dense projections (Q/K/V, gate/up) into packed matmuls. |
| `IZWI_QWEN3_METAL_QMATMUL_F32_INPUT` | flag | true (Metal only) | first use | Casts non-F32 activations to F32 before Qwen3 quantized matmuls on Metal. |
| `IZWI_QWEN3_PROFILE` | flag | `false` | first use | Enables Qwen3 runtime profiling counters and timers (qmatmul casts, fused SiLU-mul, etc.). |
| `IZWI_QWEN3_QK_RMS_NORM_FUSION` | flag | true on Metal/CUDA, false on CPU | first use | Toggles the fused Q/K RMSNorm kernel in Qwen3 single-token decode attention. |
| `IZWI_QWEN3_QUANTIZED_PROJECTION_GROUPING` | flag | Metal: true when hidden_size<=1024; always false off Metal | model load | Packs quantized Qwen3 Q/K/V and gate/up projections into one QMatMul each on Metal. |
| `IZWI_QWEN3_ROPE_KERNEL` | flag | true on Metal/CUDA, false on CPU | model load | Toggles the custom RoPE kernel for Qwen3 attention. |
| `IZWI_QWEN_ASR_AUDIO_DTYPE` | choice | backend default (GGUF: text dtype) | model load | Overrides the Qwen3-ASR audio tower dtype (e.g. f32, f16, bf16). |
| `IZWI_QWEN_ASR_STREAM_CHARS_PER_TOKEN` | integer | 4 (clamped 1..12) | first use | Sets chars per rollback token used to derive streaming ASR text holdback when HOLDBACK_CHARS is unset (all families). |
| `IZWI_QWEN_ASR_STREAM_HOLDBACK_CHARS` | integer | ROLLBACK_TOKENS x CHARS_PER_TOKEN = 20 | first use | Sets characters of streamed ASR chunk text held back as unstable before emission (applies to all families). |
| `IZWI_QWEN_ASR_STREAM_ROLLBACK_TOKENS` | integer | 5 (max 32) | first use | Sets rollback tokens used to derive streaming ASR text holdback when HOLDBACK_CHARS is unset (all families). |
| `IZWI_QWEN_ASR_TEXT_DTYPE` | choice | IZWI_QWEN_DTYPE, else backend default | model load | Overrides the Qwen3-ASR text decoder dtype (e.g. f32, f16, bf16); falls back to IZWI_QWEN_DTYPE. |
| `IZWI_QWEN_DTYPE` | choice | backend/checkpoint default | model load | Shared dtype fallback for Qwen3 chat and Qwen3-ASR text when their specific overrides are unset; Qwen3-TTS ignores it. |
| `IZWI_QWEN_TTS_DTYPE` | choice | backend default | model load | Overrides the Qwen3-TTS model dtype (e.g. f32, f16, bf16); does not inherit IZWI_QWEN_DTYPE. |
| `IZWI_REQUEST_PARALLELISM` | integer | derived from backend, CPU capacity and tensor batch cap | startup | Overrides how many requests the executor runs in parallel (positive integer). |
| `IZWI_REQUIRE_CUDA_TEST_DEVICE` | flag | `false` | test harness | Makes CUDA-feature tests fail instead of skipping when no usable CUDA device exists. |
| `IZWI_REQUIRE_METAL_PARITY` | flag | false (skip when no Metal device) | test harness | Makes Metal-feature tests fail instead of skipping when no Metal device exists. |
| `IZWI_SORTFORMER_PP_FILTER_SPEECH_FIRST` | flag | true (preset value) | first use | Overrides Sortformer diarization post-processing filter_speech_first. |
| `IZWI_SORTFORMER_PP_MIN_DURATION_OFF` | number | request min_duration_off_ms, else 0 (preset value) | first use | Overrides the Sortformer post-processing minimum non-speech gap in seconds (floored at 0). |
| `IZWI_SORTFORMER_PP_MIN_DURATION_ON` | number | request min_duration_on_ms, else 0 (preset value) | first use | Overrides the Sortformer post-processing minimum speech segment duration in seconds (floored at 0). |
| `IZWI_SORTFORMER_PP_OFFSET` | number | 0.5 (model preset) | first use | Overrides the Sortformer speech-offset probability threshold (clamped 0-1). |
| `IZWI_SORTFORMER_PP_ONSET` | number | 0.5 (model preset) | first use | Overrides the Sortformer speech-onset probability threshold (clamped 0-1). |
| `IZWI_SORTFORMER_PP_PAD_OFFSET` | number | 0.0 (model preset) | first use | Overrides seconds of padding added after each Sortformer speech segment (floored at 0). |
| `IZWI_SORTFORMER_PP_PAD_ONSET` | number | 0.0 (model preset) | first use | Overrides seconds of padding added before each Sortformer speech segment (floored at 0). |
| `IZWI_SORTFORMER_PP_PRESET` | choice | `model` | first use | Selects Sortformer post-processing preset: model, callhome/callhome_v2, dihard3/dihard3_v2, legacy/legacy_model (unknown=model). |
| `IZWI_SORTFORMER_STREAMING_PROFILE` | choice | `model` | model load | Selects the Sortformer streaming chunk profile: model, low_latency (low), high_latency (high); unknown values ignored. |
| `IZWI_SORTFORMER_VAD_GATE` | flag | `false` | first use | Gates Sortformer diarization output by a VAD speech mask when the request does not set vad_gate. |
| `IZWI_STREAM_AUDIO_QUEUE_CAPACITY` | integer | 8 (TTS) / 64 (speech-to-speech) | first use | Sets the engine streaming queue depth for TTS and speech-to-speech; overrides IZWI_STREAM_QUEUE_CAPACITY. |
| `IZWI_STREAM_QUEUE_CAPACITY` | integer | per task: TTS 8, S2S 64, ASR 4096, chat 64 | first use | Sets the fallback engine streaming queue depth for every task without a task-specific override (positive integer). |
| `IZWI_STREAM_SLOW_CONSUMER_TIMEOUT_MS` | duration | 30000 (30 s) | first use | Sets how long streamed text output may wait on a stalled consumer before the invocation fails, in milliseconds. |
| `IZWI_STREAM_TEXT_QUEUE_CAPACITY` | integer | 4096 (ASR) / 64 (chat) | first use | Sets the engine streaming queue depth for ASR and chat text; overrides IZWI_STREAM_QUEUE_CAPACITY. |
| `IZWI_TEST_LFM25_DETOKENIZER_GGUF` | path | ~/Library/Application Support/izwi/models/LFM2.5-Audio-1.5B-GGUF tokenizer Q4_0 GGUF | test harness | LFM2.5-Audio tokenizer GGUF used by detokenizer parity tests. |
| `IZWI_TEST_QWEN_TTS_BASE_DIR` | path | ~/Library/Application Support/izwi/models/Qwen3-TTS-12Hz-0.6B-Base | test harness | Qwen3-TTS base checkpoint for the speaker-encoder PyTorch parity test. |
| `IZWI_THERMAL_PRESSURE` | number | `0.0` | startup | Provides an external thermal pressure hint, clamped to [0,1], for power-adaptive scheduling. |
| `IZWI_TIMEOUT` | duration | `300` | startup | Request timeout in seconds (legacy alias REQUEST_TIMEOUT_SECS; invalid ignored). |
| `IZWI_UI_DIR` | path | `ui/dist` | startup | Directory the server serves the web UI from. |
| `IZWI_USE_FLASH_ATTENTION` | choice | `auto` | first use | Selects flash attention: force; auto (or empty) or a true boolean = auto; a false boolean or any other word = off. |
| `IZWI_VIBEVOICE_ASR_CUDA_MAX_AUDIO_SECS` | number | `3600` | first use | Lowers the VibeVoice ASR max audio-seconds hint on CUDA (positive, capped at 3600); ignored off CUDA. |
| `IZWI_VIBEVOICE_ASR_DTYPE` | choice | auto (Metal BF16-checkpoint tier, else backend default) | model load | Forces the VibeVoice-ASR LM dtype (f32, f16, bf16); f32 also keeps Metal tokenizer FFN weights F32. |
| `IZWI_VIBEVOICE_BACKEND` | choice | required (no default) | test harness | Backend for the VibeVoice clone smoke test: cpu, metal or cuda. |
| `IZWI_VIBEVOICE_CFG_BATCHING` | auto/flag | auto (true on Metal/CUDA, false on CPU) | first use | Toggles batching the CFG positive and negative diffusion predictions in VibeVoice TTS. |
| `IZWI_VIBEVOICE_CUDA_DDPM_STEPS` | integer | requested diffusion steps (default 10) | first use | Overrides VibeVoice TTS diffusion (DDPM) steps on CUDA only; invalid values warn and are ignored. |
| `IZWI_VIBEVOICE_MAX_WER` | number | `0.34` | test harness | Maximum word error rate the VibeVoice clone smoke test accepts. |
| `IZWI_VIBEVOICE_OUT_DIR` | path | required (no default) | test harness | Directory where the VibeVoice clone smoke test writes outputs. |
| `IZWI_VIBEVOICE_REFERENCE_TEXT` | text | required (no default) | test harness | Transcript of the VibeVoice smoke reference WAV. |
| `IZWI_VIBEVOICE_REFERENCE_WAV` | path | required (no default) | test harness | Reference voice WAV for the VibeVoice clone smoke test. |
| `IZWI_VIBEVOICE_TEXT` | text | built-in TARGET sentence | test harness | Text the VibeVoice clone smoke test synthesizes. |
| `IZWI_VIBEVOICE_TTS_DTYPE` | choice | VibeVoice family/backend default | model load | Overrides the VibeVoice TTS weight dtype (e.g. f32, f16, bf16); also used by load memory planning. |
| `IZWI_VOXTRAL_DTYPE` | choice | backend default | model load | Fallback Voxtral TTS dtype override used when IZWI_VOXTRAL_TTS_DTYPE is unset. |
| `IZWI_VOXTRAL_TTS_DTYPE` | choice | IZWI_VOXTRAL_DTYPE, else backend default | model load | Overrides the Voxtral TTS model dtype (e.g. f32, f16, bf16). |
| `IZWI_WHISPER_ADAPTIVE_MAX_NEW_TOKENS` | flag | `true` | model load | Sizes the Whisper decode budget from audio duration instead of a fixed cap. |
| `IZWI_WHISPER_DEFAULT_LANGUAGE` | text | unset (auto-detect) | model load | Sets the language Whisper uses when a request names none (non-empty string). |
| `IZWI_WHISPER_DEVICE_GREEDY` | flag | `true` | model load | Runs Whisper greedy token selection on the device instead of on host logits. |
| `IZWI_WHISPER_DTYPE` | choice | family default (F16 on Metal; backend default elsewhere) | model load | Forces the Whisper weight and activation dtype (f32, f16, bf16; BF16 runs F16 on Metal). |
| `IZWI_WHISPER_INITIAL_PROMPT_MAX_TOKENS` | integer | `224` | model load | Caps the initial-prompt tokens fed to the Whisper decoder. |
| `IZWI_WHISPER_MAX_FALLBACK_RETRIES` | integer | `1` | model load | Sets the number of temperature-fallback retries in Whisper decoding. |
| `IZWI_WHISPER_MAX_NEW_TOKENS_BUFFER` | integer | `8` | model load | Adds buffer tokens to the Whisper adaptive decode budget. |
| `IZWI_WHISPER_MAX_NEW_TOKENS_CAP` | integer | `448` | model load | Caps Whisper max new tokens per decode window. |
| `IZWI_WHISPER_MAX_NEW_TOKENS_PER_SECOND` | number | `12.0` | model load | Sets tokens per audio second for the Whisper adaptive decode budget. |
| `IZWI_WHISPER_METAL_SDPA` | flag | `false` | first use | Allows the Metal SDPA kernel for Whisper dense attention. |
| `IZWI_WHISPER_MIN_NEW_TOKENS` | integer | `32` | model load | Sets the floor of the Whisper adaptive decode budget. |
| `IZWI_WHISPER_NO_FALLBACK` | flag | `false` | model load | Disables Whisper temperature fallback retries. |
| `IZWI_WHISPER_PROFILE_SYNC` | flag | `false` | model load | Synchronizes the device around Whisper decode profiling for accurate stage timings. |
| `IZWI_WHISPER_SILENCE_TRIM_MARGIN_MS` | integer | `120` | model load | Sets the margin in ms kept around speech when Whisper trims silence. |
| `IZWI_WHISPER_SILENCE_TRIM_MIN_ABS` | number | `0.0015` | model load | Sets the absolute amplitude floor for Whisper silence-trim detection. |
| `IZWI_WHISPER_SILENCE_TRIM_MIN_CLIP_SECS` | number | `0.8` | model load | Sets the minimum clip length in seconds before Whisper silence trimming applies. |
| `IZWI_WHISPER_SILENCE_TRIM_MIN_LEADING_MS` | integer | `500` | model load | Sets the minimum leading silence in ms that Whisper trims. |
| `IZWI_WHISPER_SILENCE_TRIM_MIN_TRAILING_MS` | integer | `160` | model load | Sets the minimum trailing silence in ms that Whisper trims. |
| `IZWI_WHISPER_SILENCE_TRIM_THRESHOLD_SCALE` | number | `0.02` | model load | Scales the peak-relative threshold for Whisper silence-trim detection. |
| `IZWI_WHISPER_SUPPRESS_BLANK` | flag | `true` | model load | Suppresses blank tokens at the start of Whisper decoding. |
| `IZWI_WHISPER_SUPPRESS_NUMERALS` | flag | `false` | model load | Suppresses numeral and symbol tokens in Whisper decoding. |
| `IZWI_WHISPER_TRIM_SILENCE` | flag | `true` | model load | Trims leading and trailing silence from audio before Whisper decoding. |
