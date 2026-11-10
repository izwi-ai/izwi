---
title: "Real-time and batch workloads"
description: "Run live calls, voice agents and streaming speech beside bulk transcription and synthesis on one Izwi server: which route to use, how live traffic is protected, and what to tune and watch."
sidebarTitle: "Real-time and batch"
icon: "gauge"
---
One Izwi server can answer live traffic (phone calls, voice agents, streaming playback) and work through bulk jobs (recordings, long scripts) on the same device. The route a client calls tells the server which kind of work a request is, and the server keeps capacity free for the live kind.

This page is for the operator who sizes and watches the server, and for the developer who chooses routes.

---

## Choose a route for the workload

| Workload | Route | Runs as |
|----------|-------|---------|
| Live calls, voice agents, live captions | `GET /v1/realtime`, or `GET /v1/realtime/media-stream` for a carrier's media stream ([Phone calls](/features/telephony)) | `realtime` |
| Speech played while it is generated | `POST /v1/audio/speech` with `stream_format: "sse"` or `"audio"`, or the `/v1/audio/speech/stream` WebSocket | `streaming` |
| One request, answered whole, someone waiting | `POST /v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/audio/translations` | `interactive` |
| Bulk work on a synchronous route | The same routes with `service_tier: "flex"` ([Audio Service Tier](/api#audio-service-tier)) | `batch` |
| A small set of texts, answered in one response | `POST /v1/audio/speech/batch` (up to `IZWI_TTS_BATCH_MAX_ITEMS`, 32) | `batch` unless the request names another tier |
| Thousands of recordings or texts, results within 24 hours | `POST /v1/files` then `POST /v1/batches` ([Batch API](/api#batch-api)) | `batch` |

The class a request ran as is the `workload_class` label on the server's metrics.

Pick by who is waiting:

- **A person is listening or talking now.** Use `/v1/realtime` or a streaming speech route. Do not send live traffic as `flex`.
- **A program wants one answer soon.** Use the synchronous routes without a tier.
- **Nobody is waiting.** Use `flex` for a handful of requests your code already sends one by one, and `/v1/batches` for anything large: a batch survives a restart, retries lines the server had no room for, and does not hold a connection open. A batch can call a [webhook](/api#batch-api) when it ends.

The first-party job routes (`/v1/speech-to-text/jobs`, `/v1/text-to-speech`) take no `service_tier`. Their durable jobs always run as batch work.

---

## What the server does to protect live traffic

All of this is on by default.

- **Classes.** Every request is admitted and scheduled under its class. Between models that share a device, device time is shared by class weight, and batch and background work have the smallest shares.
- **Admission.** Batch and background requests can never take the server's last request slot (`IZWI_MAX_BATCH_REQUESTS` and `IZWI_MAX_BACKGROUND_REQUESTS` default to the global limit less one).
- **The real-time reserve.** While a model has live demand (a real-time or streaming request, or a real-time session operation, in the last 30 seconds), batch and background requests of that model may start only up to the model's rows less a reserve. A live request then starts in a free row instead of waiting for batch work to finish. With no live demand batch work uses every row. A model that keeps more sequences than it decodes in one step (`IZWI_MAX_NUM_SEQS` above its rows) still leaves the reserve free among the extra sequences, which add no batch throughput, so the first live request after a quiet spell does not wait for a batch request to finish.
- **Class-aware encoding.** When more transcription requests wait for a model's audio encoder than one batch holds, the batch takes the requests of the higher classes first, so a live request does not wait behind a backlog of batch rows.
- **Bounded batch steps.** A long recording is transcribed in steps, and live requests are served between them. Parakeet ends its encode after each window group while its model has live demand.
- **A separate real-time queue.** Real-time requests and open sessions wait in a runtime queue of their own, so a flood of batch requests that fills the general queue does not refuse a call.
- **Pacing.** A speech stream that has produced more than `IZWI_STREAM_LEAD_SECS` of audio beyond what has had time to play is scheduled from the batch class's share until its lead falls to half of that. Nothing is throttled: on an idle device the stream still runs at full speed.
- **Bounded waiting.** A chat or `/v1/audio/*` request that finds the server at capacity waits up to `IZWI_ADMISSION_QUEUE_TIMEOUT_SECS` and is then refused with `429` and a `Retry-After` header, before any stream starts. A `flex` request's refusal has code `resource_unavailable` and a message starting `Resource Unavailable`, so a client can retry later or fall back to the default tier. A `/v1/batches` line refused this way is run again by the server rather than failed.

A running device job is never interrupted. Protection comes from keeping rows free and keeping batch jobs short, not from stopping work that has started.

---

## Settings

Set these in the server's environment. The defaults suit a server that does both kinds of work.

| Variable | Default | What it does |
|----------|---------|--------------|
| `IZWI_REALTIME_ROW_RESERVE` | A quarter of the model's rows, at least 1 | Rows each model keeps free of batch and background requests while it has live demand, and sequences beyond its rows it keeps free of them at all times. `0` turns the reserve off. Raise it if live requests queue behind batch work; lower it if batch throughput matters more. |
| `IZWI_STREAM_LEAD_SECS` | `3` | Seconds of audio a speech stream may run ahead of real time before it is scheduled from the batch share. `0` turns pacing off. |
| `IZWI_ADMISSION_QUEUE_TIMEOUT_SECS` | `30` | How long a request waits for an admission slot before `429`. `0` waits without limit. Real-time sessions, the first-party job routes and background work always wait. |
| `IZWI_MAX_REALTIME_REQUESTS`, `IZWI_MAX_STREAMING_REQUESTS`, `IZWI_MAX_INTERACTIVE_REQUESTS`, `IZWI_MAX_ONLINE_REQUESTS` | The global request limit (`--max-concurrent`, 100) | Concurrent requests of each class. |
| `IZWI_MAX_BATCH_REQUESTS`, `IZWI_MAX_BACKGROUND_REQUESTS` | The global limit less one | Concurrent batch and background requests. Lower `IZWI_MAX_BATCH_REQUESTS` to cap how much of the server bulk work may use. |
| `IZWI_MAX_REALTIME_SESSIONS` | `32` | Open real-time WebSocket sessions. Each call or voice session takes one; a full server answers with an error and close code `1013`. |
| `IZWI_ASR_REALTIME_MAX_SESSIONS` | `IZWI_MAX_NUM_SEQS` if set, else 16 on CUDA and 4 on CPU and Metal | Concurrent native live-transcription streams (Nemotron 3.5 ASR Streaming, Voxtral Realtime). A session over the cap is still served, by re-transcribing a rolling window, and is told so. |
| `IZWI_MAX_QUEUED_REQUESTS` | `128` | Requests the general runtime queue holds; a request that finds it full gets `503`. The real-time queue is the same size, or the real-time session cap if that is larger. |
| `IZWI_FILES_MAX_BYTES` | `209715200` (200 MB) | Largest file `/v1/files` accepts. |
| `IZWI_BATCH_MAX_REQUESTS` | `50000` | Requests one `/v1/batches` input file may hold. |
| `IZWI_BATCH_WORKER_CONCURRENCY` | The runtime's capacity | `/v1/batches` lines in flight at once, and durable jobs run at once. |
| `IZWI_BATCH_STAGE_TIMEOUT_SECS` | `21600` (six hours) | Wall-clock limit of one durable job stage. `0` turns it off. |
| `IZWI_BATCH_WEBHOOK_SECRET` | Unset | Turns on batch completion webhooks and signs them. `IZWI_BATCH_WEBHOOK_ALLOW_HTTP` and `IZWI_BATCH_WEBHOOK_ALLOW_PRIVATE` relax which URLs a batch may name. |

---

## What to watch

Scrape `GET /v1/metrics/prometheus`. The [metrics reference](/api#metrics) describes every series.

| Question | Metric |
|----------|--------|
| Is live latency holding? | `izwi_request_ttft_seconds` and `izwi_request_queue_wait_seconds` with `workload_class="realtime"` or `"streaming"`; `izwi_tts_first_audio_seconds` for streamed speech; the `izwi_voice_*` histograms for voice sessions. |
| Is batch work getting through? | `izwi_request_duration_seconds{workload_class="batch"}`; the rate of `izwi_request_input_audio_seconds_sum` (audio transcribed per second) and of `izwi_tts_output_audio_seconds_total` (audio spoken per second); `izwi_batch_runtime_api_batches` and `izwi_batch_runtime_api_batch_requests` for `/v1/batches`. |
| Is the reserve doing work? | `izwi_engine_scheduler_reserve_deferrals_total` counts batch requests whose start waited for it. A steady rise with good live latency is the reserve working. Zero under mixed load means batch work never reached the reserve. |
| Are long recordings yielding? | `izwi_engine_asr_encoder_live_yields_total` counts Parakeet encodes that ended a job early for a live request. |
| Are streams running ahead? | `izwi_engine_scheduler_stream_pacing_transitions_total` counts the times a stream was moved to the batch share. |
| Is a queue filling? | `izwi_inference_coordinator_queue_depth` against `izwi_inference_coordinator_queue_capacity`, by `pool` (`general`, `realtime`). |
| Are clients being refused? | `izwi_request_duration_seconds_count` by `outcome`, and `429` and `503` responses at your proxy. |

---

## Known limits

- **VibeVoice-ASR and Nemotron 3.5 Transcribe** read a whole recording in one pass. A batch request of either holds the device for that pass, and a live request of the same model waits for it. For strict live latency, run these two for recordings on a separate server or device.
- **Kokoro streams are not paced.** A streamed Kokoro request runs outside the scheduler that paces other speech streams. It still counts as its model's live demand, so the reserve applies to batch Kokoro work beside it.
- **First-party durable jobs take no tier.** They always run as batch work.
- **Running work is not preempted.** A live request can wait for the batch job in front of it on the device to finish; bounded batch steps keep that wait short for the models that support them.
- **One server holds a batch.** Files and batches live in the server's own store. Behind the gateway they are served only when its upstreams share that store ([Batch API](/api#batch-api)).

---

## Measure your capacity

No capacity figures are published yet: how many live sessions a device carries, alone and beside a batch load, has not been measured across models and hardware. Measure it on yours before you promise a latency.

`scripts/bench/run-mixed-load.py` holds a fixed live load while it raises a batch load on the same server, and reports what each batch level does to live latency:

```bash
# Live transcription sessions with batch transcriptions of the same model on top.
python3 scripts/bench/run-mixed-load.py --out target/mixed-load/parakeet \
    --realtime-kind asr --realtime-model Parakeet-TDT-0.6B-v3 \
    --batch-levels 0,4,16,64 --batch-audio data/diarization.wav \
    --batch-service-tier flex

# Live speech streams with buffered speech of the same model on top.
python3 scripts/bench/run-mixed-load.py --out target/mixed-load/kokoro \
    --realtime-kind tts --realtime-model Kokoro-82M \
    --batch-levels 0,4,16,64 --batch-service-tier flex

# Live transcription beside a batch speech load on another model.
python3 scripts/bench/run-mixed-load.py --out target/mixed-load/parakeet-kokoro \
    --realtime-kind asr --realtime-model Parakeet-TDT-0.6B-v3 \
    --batch-kind tts --batch-model Kokoro-82M \
    --batch-levels 0,4,16,64 --batch-service-tier flex
```

Level 0 is the live load alone and is the baseline. Each step records live latency (first partial and final transcript, or first audio), batch throughput, refusals on both sides, and the change in the server's metrics. The run checks each step against latency targets you can set (`--slo-asr-first-partial-s`, `--slo-asr-final-s`, `--slo-tts-first-audio-s`) and against the rise over the baseline (`--slo-degradation`). Raise `--realtime-streams` until the baseline no longer meets your targets to find how many live sessions the device carries, then read off the batch level it tolerates. `scripts/bench/README.md` has every option.

---

## See also

- [Phone calls and call centres](/features/telephony)
- [Audio Service Tier](/api#audio-service-tier)
- [Batch API](/api#batch-api)
- [Metrics](/api#metrics)
