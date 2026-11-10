#!/usr/bin/env python3
"""Mixed real-time and batch speech load for izwi: a fixed real-time load with a batch ramp on top.

Each step holds the same real-time load for --step-secs while a batch load
runs beside it, at one concurrency level of --batch-levels per step. Level 0
is the real-time-only baseline and runs first.

Real-time side (--realtime-kind):
- `asr`: --realtime-streams `/v1/realtime?intent=transcription` sessions. Each
  speaks --audio (24 kHz PCM16 WAV) in real time, 20 ms a frame, then silence,
  lets the server's VAD end the utterance and waits for the transcript; then
  it speaks again, until the step ends.
- `tts`: --realtime-streams streamed `/v1/audio/speech` requests
  (--tts-stream-format audio or sse), each replaced by the next when it ends,
  so that many streams are always live.

Batch side (--batch-kind), closed loop at the step's level:
- `asr`: buffered `/v1/audio/transcriptions` of clips cut from --batch-audio;
- `tts`: buffered `/v1/audio/speech`.
--batch-model may differ from --realtime-model (cross-model interference).
--batch-service-tier is sent as the `service_tier` field and the
`X-Izwi-Service-Tier` header; without it neither is sent.

Reported per step:
- real-time ASR: first partial (from the first audio frame, and from the
  server's `speech_stopped`) and final transcript (from `speech_stopped`, and
  from the end of the spoken audio);
- real-time TTS: first byte, first audio (the WAV header plus 100 ms) and the
  gaps between audio chunks;
- batch: audio seconds per second and request latency;
- each side: completed, failed and refused (HTTP 429/503, WebSocket close
  1013) counts;
- the server's own view, as deltas of its Prometheus metrics over the step:
  request TTFT, queue wait and duration by `workload_class`, TTS output audio
  seconds, decode widths and KV defers. A series the server does not have, or
  a server without the endpoint, is recorded as missing and fails nothing.

Each step is checked against the SLOs (--slo-*): ASR first partial p95, ASR
final p95, TTS first audio p95, and how far each moved from the level-0 step.
--enforce-slo exits 4 when a step fails one.

Writes OUT/step-<NN>-batch-c<level>.json per step (schema
"izwi.mixed_load.step") and OUT/summary.json (schema "izwi.mixed_load").
Standard library only: runs with plain python3.
"""
import argparse
import asyncio
import base64
import datetime
import importlib.util
import itertools
import json
import os
import pathlib
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

SCHEMA = "izwi.mixed_load"
STEP_SCHEMA = "izwi.mixed_load.step"
SCHEMA_VERSION = 1
SLO_FAILURE_EXIT = 4
REPO = pathlib.Path(__file__).resolve().parents[2]


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, pathlib.Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audio_load = _load("audio_load", "run-audio-load.py")
tts = _load("tts_concurrency", "run-tts-concurrency.py")
voice = _load("voice_realtime_load", "run-voice-realtime-load.py")
serving = audio_load.serving

# Header plus 100 ms of 24 kHz mono s16, as run-tts-concurrency.py counts it.
FIRST_AUDIO_BYTES = 44 + 4800
SSE_MAX_EVENT_BYTES = 16 << 20
REFUSAL_KINDS = ("http_429", "http_503", "ws_1013")
ERROR_GRACE_S = 2.0

TRANSCRIPT_DELTA = "conversation.item.input_audio_transcription.delta"
TRANSCRIPT_COMPLETED = "conversation.item.input_audio_transcription.completed"
TRANSCRIPT_FAILED = "conversation.item.input_audio_transcription.failed"
SPEECH_STOPPED = "input_audio_buffer.speech_stopped"

# Request histograms, split by their `workload_class` label once the server
# has it.
REQUEST_HISTOGRAMS = (
    "izwi_request_ttft_seconds",
    "izwi_request_queue_wait_seconds",
    "izwi_request_duration_seconds",
)
# Rows per engine step (the decode width) and per ASR preparation batch.
WIDTH_HISTOGRAMS = ("izwi_engine_batch_rows", "izwi_asr_preparation_batch_rows")
COUNTERS = (
    "izwi_tts_output_audio_seconds_total",
    "izwi_engine_executor_model_tensor_batch_width_calls_total",
    "izwi_engine_executor_physical_dispatch_width_dispatches_total",
    "izwi_engine_executor_physical_defers_total",
)
# Which real-time distribution each latency SLO reads.
PARTIAL_REFERENCES = {"start": "first_partial_s", "vad": "first_partial_after_vad_s"}
FINAL_REFERENCES = {"vad": "final_after_vad_s", "audio-end": "final_after_audio_s"}


# ---------------------------------------------------------------- statistics

def stats(values):
    """count, mean, p50, p95 and max of the values that are not None."""
    values = [value for value in values if value is not None]
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p50": serving.percentile(values, 50),
        "p95": serving.percentile(values, 95),
        "max": max(values),
    }


def arrival_summary(arrivals, first_audio_bytes=FIRST_AUDIO_BYTES):
    """Timing of one audio stream from its [(seconds, bytes)] chunk arrivals:
    the first byte, the first audio (when `first_audio_bytes` have arrived)
    and the gaps between consecutive chunks."""
    total, first_audio = 0, None
    for now, size in arrivals:
        total += size
        if first_audio is None and total >= first_audio_bytes:
            first_audio = now
    times = [now for now, _ in arrivals]
    return {
        "first_byte_s": times[0] if times else None,
        "first_audio_s": first_audio,
        "gaps_s": [later - earlier for earlier, later in zip(times, times[1:])],
        "chunks": len(times),
        "bytes": total,
    }


# ------------------------------------------------------------------ outcomes

def refusal_kind(http_status=None, close_code=None):
    """The refusal a failure is, or None: the server shedding load."""
    if http_status in (429, 503):
        return f"http_{http_status}"
    if close_code == 1013:
        return "ws_1013"
    return None


def upgrade_status(message):
    """The HTTP status in a refused WebSocket upgrade's status line."""
    match = re.search(r"HTTP/\d(?:\.\d)? (\d{3})", message)
    return int(match.group(1)) if match else None


def failure_record(error, close_code=None):
    """The record of a failed request or session: refused or failed."""
    if isinstance(error, urllib.error.HTTPError):
        status = error.code
        detail = f"HTTP {status}: {error.read(2048).decode('utf-8', 'replace')}"
    else:
        status = upgrade_status(str(error)) if isinstance(error, ConnectionError) else None
        detail = f"{type(error).__name__}: {error}"
    refusal = refusal_kind(status, close_code)
    record = {"status": "refused" if refusal else "failed", "error": detail[:500]}
    if refusal:
        record["refusal"] = refusal
    return record


def outcomes(records):
    """Completed, failed and refused counts of one side's records."""
    refused = {kind: 0 for kind in REFUSAL_KINDS}
    for record in records:
        if record["status"] == "refused":
            refused[record["refusal"]] += 1
    return {
        "requests": len(records),
        "completed": sum(record["status"] == "ok" for record in records),
        "errors": sum(record["status"] == "failed" for record in records),
        "refused": dict(refused, total=sum(refused.values())),
        "error_samples": sorted({record["error"] for record in records if "error" in record})[:5],
    }


# ----------------------------------------------------------------- real-time

def asr_turn_record(events, started, audio_end):
    """One transcription turn from its [(time, event)] log. `started` is when
    the first audio frame was sent and `audio_end` when the speech ended."""
    first = {}
    for now, event in events:
        first.setdefault(event.get("type"), now)

    def since(mark, origin):
        return None if mark not in first or origin is None else first[mark] - origin

    stopped = first.get(SPEECH_STOPPED)
    record = {
        "status": "ok" if TRANSCRIPT_COMPLETED in first else "failed",
        "first_partial_s": since(TRANSCRIPT_DELTA, started),
        "first_partial_after_vad_s": since(TRANSCRIPT_DELTA, stopped),
        "final_after_vad_s": since(TRANSCRIPT_COMPLETED, stopped),
        "final_after_audio_s": since(TRANSCRIPT_COMPLETED, audio_end),
        "audio_s": audio_end - started,
    }
    for _, event in events:
        if event.get("type") == TRANSCRIPT_COMPLETED:
            record["transcript"] = event.get("transcript", "")
        elif event.get("type") in (TRANSCRIPT_FAILED, "error"):
            record["error"] = str((event.get("error") or {}).get("message"))[:500]
    return record


async def next_event(ws, deadline, waiting_for):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(f"no {waiting_for} before the deadline")
    message = await asyncio.wait_for(ws.recv(), remaining)
    if message is None:
        raise ConnectionError(f"socket closed (code {ws.close_code}) while waiting for {waiting_for}")
    return json.loads(message) if isinstance(message, str) else {}


async def expect(ws, wanted, deadline):
    """Receive until `wanted`. After an `error` event the server gets
    ERROR_GRACE_S more to send it or to close (a refusal closes with 1013);
    then the error fails the setup."""
    failure = None
    while True:
        try:
            event = await next_event(ws, deadline, wanted)
        except TimeoutError:
            if failure is None:
                raise
            raise RuntimeError(failure) from None
        except ConnectionError as error:
            raise ConnectionError(f"{error}: {failure}" if failure else str(error)) from None
        if event.get("type") == "error":
            failure = str((event.get("error") or {}).get("message") or "error event")
            deadline = min(deadline, time.monotonic() + ERROR_GRACE_S)
        if event.get("type") == wanted:
            return event


def transcription_session_update(config):
    transcription = {"model": config["realtime_model"]}
    if config.get("language"):
        transcription["language"] = config["language"]
    return {"type": "session.update", "session": {
        "type": "transcription",
        "audio": {"input": {
            "format": {"type": "audio/pcm", "rate": voice.OPENAI_SAMPLE_RATE},
            "transcription": transcription,
            "turn_detection": {"type": "server_vad", "silence_duration_ms": config["silence_ms"]},
        }},
    }}


async def asr_turn(ws, config, frames):
    events = []
    started = time.monotonic()
    audio_end = started + len(frames) * voice.FRAME_MS / 1000
    silence_frames = (config["silence_ms"] + config["trailing_silence_ms"]) // voice.FRAME_MS
    sender = asyncio.ensure_future(voice.stream_openai_audio(ws, frames, silence_frames))
    try:
        while True:
            event = await next_event(ws, started + config["turn_timeout"], "the transcript")
            events.append((time.monotonic(), event))
            if event.get("type") in (TRANSCRIPT_COMPLETED, TRANSCRIPT_FAILED):
                break
    except BaseException:
        sender.cancel()
        await asyncio.gather(sender, return_exceptions=True)
        raise
    await sender
    return asr_turn_record(events, started, audio_end)


async def asr_session(index, config, frames, stop_at, records, max_turns=None):
    """One live transcription session speaking turn after turn until
    `stop_at`. A refused or failed session is recorded and reopened."""
    url = (config["base_url"].replace("http", "ws", 1) + "/v1/realtime?"
           + urllib.parse.urlencode({"intent": "transcription"}))
    turn = 0

    def more():
        return time.monotonic() < stop_at and (max_turns is None or turn < max_turns)

    while more():
        ws = None
        try:
            ws = await voice.WebSocket.connect(url, headers=config["headers"])
            deadline = time.monotonic() + config["turn_timeout"]
            await expect(ws, "session.created", deadline)
            await ws.send_json(transcription_session_update(config))
            await expect(ws, "session.updated", deadline)
            while more():
                records.append({"session": index, "turn": turn, **await asr_turn(ws, config, frames)})
                turn += 1
        except Exception as error:  # noqa: BLE001 - a failed session is a result
            records.append({"session": index, "turn": turn,
                            **failure_record(error, ws.close_code if ws else None)})
            turn += 1
            await asyncio.sleep(config["retry_delay"])
        finally:
            if ws:
                await ws.close()


async def run_asr_sessions(config, frames, sessions, stop_at, max_turns=None):
    records = []
    await asyncio.gather(*(asr_session(index, config, frames, stop_at, records, max_turns)
                           for index in range(sessions)))
    return records


def sse_audio(events):
    """The audio bytes of each `speech.audio.delta` among SSE `data` values."""
    pieces = []
    for data in events:
        try:
            event = json.loads(data)
        except ValueError:
            continue  # `[DONE]`
        if not isinstance(event, dict):
            continue
        if event.get("type") == "error" or event.get("event") == "audio.failed":
            error = event.get("error")
            raise RuntimeError(str((error.get("message") if isinstance(error, dict) else error)
                                   or event.get("message") or "speech stream failed"))
        if event.get("type") == "speech.audio.delta":
            pieces.append(base64.b64decode(event.get("audio") or ""))
    return pieces


def read_audio_stream(response, stream_format, started):
    """([(seconds since `started`, bytes)], audio bytes) of a streamed speech
    response: raw chunks (`audio`) or base64 deltas (`sse`)."""
    parser = serving.SseParser(SSE_MAX_EVENT_BYTES) if stream_format == "sse" else None
    arrivals, parts = [], []
    while True:
        chunk = response.read1(65536)
        if not chunk:
            break
        now = time.monotonic() - started
        for piece in ([chunk] if parser is None else sse_audio(parser.feed(chunk))):
            if piece:
                arrivals.append((now, len(piece)))
                parts.append(piece)
    return arrivals, b"".join(parts)


def wav_seconds(payload):
    rate, channels, bits, pcm = tts.wav_pcm(payload)
    frame = max(1, channels * bits // 8)
    return len(pcm) / frame / rate if rate else 0.0


def speech_body(model, index, stream_format, reference_b64, extra=None, service_tier=None):
    """The speech request for input `index`: run-tts-concurrency.py's text,
    seed and per-model fields (a model it does not know gets text and seed)."""
    if model in tts.MODELS:
        body = tts.request_body(model, index, False, reference_b64)
    else:
        body = {"model": model, "input": tts.POOL[index % len(tts.POOL)],
                "response_format": "wav", "seed": index + 1}
    body.update(extra or {})
    if stream_format:
        body["stream_format"] = stream_format
    if service_tier:
        body["service_tier"] = service_tier
    return body


def speech_request(config, body, headers=None):
    return urllib.request.Request(
        config["base_url"] + "/v1/audio/speech", data=json.dumps(body).encode(), method="POST",
        headers=dict(config["headers"], **{"Content-Type": "application/json"}, **(headers or {})))


def stream_speech(config, index):
    """One streamed speech request, read as it arrives."""
    body = speech_body(config["realtime_model"], index, config["tts_stream_format"],
                       config["reference_b64"], config["tts_extra"])
    started = time.monotonic()
    with urllib.request.urlopen(speech_request(config, body), timeout=config["timeout"]) as response:
        arrivals, payload = read_audio_stream(response, config["tts_stream_format"], started)
    return dict(arrival_summary(arrivals), audio_s=wav_seconds(payload))


# --------------------------------------------------------------------- batch

def service_tier_headers(service_tier):
    return {"X-Izwi-Service-Tier": service_tier} if service_tier else {}


def transcription_fields(config):
    fields = {"model": config["batch_model"], "response_format": "json"}
    if config.get("language"):
        fields["language"] = config["language"]
    if config.get("batch_service_tier"):
        fields["service_tier"] = config["batch_service_tier"]
    return fields


def batch_transcribe(config, clips, index):
    clip = clips[index % len(clips)]
    body, content_type = audio_load.multipart(transcription_fields(config), "file", "clip.wav", clip["wav"])
    headers = dict(config["headers"], **{"Content-Type": content_type},
                   **service_tier_headers(config.get("batch_service_tier")))
    audio_load.post(config["base_url"] + "/v1/audio/transcriptions", body, headers, config["timeout"])
    return {"audio_s": clip["audio_secs"]}


def batch_speech(config, index):
    tier = config.get("batch_service_tier")
    body = speech_body(config["batch_model"], index, None, config["reference_b64"],
                       config["tts_extra"], tier)
    request = speech_request(config, body, service_tier_headers(tier))
    with urllib.request.urlopen(request, timeout=config["timeout"]) as response:
        return {"audio_s": wav_seconds(response.read())}


def batch_clips(paths, count, min_secs, max_secs, rng):
    """`count` clips cut by run-audio-load.py, each encoded once."""
    sources = [(str(path), *audio_load.read_wav(str(path))) for path in paths]
    clips = audio_load.cut_clips(sources, count, min_secs, max_secs, rng)
    return [{"audio_secs": clip["audio_secs"], "wav": audio_load.wav_bytes(clip["rate"], clip["samples"])}
            for clip in clips]


def attempt(send, index):
    """`send(index)` as a record: its fields and "ok", or the failure."""
    started = time.monotonic()
    try:
        record = dict(send(index), status="ok")
    except Exception as error:  # noqa: BLE001 - a failed request is a result
        record = failure_record(error)
    record.update({"index": index, "latency_s": time.monotonic() - started})
    return record


class ClosedLoop:
    """`workers` threads, each sending `send(index)` again and again until
    `stop()`; `finish()` waits for the requests in flight."""

    def __init__(self, send, workers, retry_delay=0.5, first_index=0):
        self.records = []
        self._send = send
        self._retry_delay = retry_delay
        self._indexes = itertools.count(first_index)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._threads = [threading.Thread(target=self._work, daemon=True) for _ in range(workers)]
        self._epoch = None

    def _work(self):
        while not self._stop.is_set():
            with self._lock:
                index = next(self._indexes)
            record = attempt(self._send, index)
            record["finished_s"] = time.monotonic() - self._epoch
            with self._lock:
                self.records.append(record)
            if record["status"] != "ok":
                self._stop.wait(self._retry_delay)

    def start(self):
        self._epoch = time.monotonic()
        for thread in self._threads:
            thread.start()
        return self

    def stop(self):
        self._stop.set()

    def finish(self):
        """(records, seconds from the start to the last response)."""
        self.stop()
        for thread in self._threads:
            thread.join()
        return self.records, max((record["finished_s"] for record in self.records), default=0.0)


# ----------------------------------------------------------------- summaries

def realtime_summary(kind, records, duration):
    summary = dict(outcomes(records), duration_s=duration)
    ok = [record for record in records if record["status"] == "ok"]
    if kind == "asr":
        for field in (*PARTIAL_REFERENCES.values(), *FINAL_REFERENCES.values()):
            summary[field] = stats([record.get(field) for record in ok])
        summary["empty_transcripts"] = sum(not record.get("transcript", "").strip() for record in ok)
    else:
        summary["first_byte_s"] = stats([record.get("first_byte_s") for record in ok])
        summary["first_audio_s"] = stats([record.get("first_audio_s") for record in ok])
        summary["inter_chunk_gap_s"] = stats([gap for record in ok for gap in record.get("gaps_s", [])])
        audio = sum(record.get("audio_s", 0.0) for record in ok)
        summary["audio_seconds"] = audio
        summary["audio_seconds_per_second"] = audio / duration if duration > 0 else None
    return summary


def batch_summary(records, duration):
    ok = [record for record in records if record["status"] == "ok"]
    audio = sum(record["audio_s"] for record in ok)
    return dict(
        outcomes(records),
        duration_s=duration,
        audio_seconds=audio,
        audio_seconds_per_second=audio / duration if duration > 0 else None,
        latency_s=stats([record["latency_s"] for record in ok]),
    )


# ---------------------------------------------------------------- Prometheus

def _series_summary(series):
    count = series["count"]
    return {
        "count": count,
        "mean": (series["sum"] or 0.0) / count,
        "p50": serving.histogram_quantile(0.50, series["buckets"]),
        "p95": serving.histogram_quantile(0.95, series["buckets"]),
    }


def histogram_by_label(by_labels, label, unlabelled):
    """One histogram family's series merged by the value of `label`."""
    groups = {}
    for labels, series in by_labels.items():
        if series["count"]:
            groups.setdefault(dict(labels).get(label, unlabelled), []).append(series)
    return {value: _series_summary(serving._merge_series(group)) for value, group in sorted(groups.items())}


def metrics_delta(before_text, after_text):
    """What the watched server metrics did between two Prometheus scrapes.
    Histograms are summarized from their bucket deltas, the request ones by
    `workload_class` and the width ones by `model`; counters are per label
    set. Families the server did not expose are listed in `missing`."""
    after = serving.parse_prometheus(after_text)
    delta = serving.sample_delta(serving.parse_prometheus(before_text), after)
    families = serving.histograms(delta)
    exposed = {name for name, _ in after}
    report = {"histograms": {}, "counters": {}, "missing": []}
    for names, label, unlabelled in ((REQUEST_HISTOGRAMS, "workload_class", "unlabelled"),
                                     (WIDTH_HISTOGRAMS, "model", "all")):
        for name in names:
            if name + "_count" in exposed:
                report["histograms"][name] = histogram_by_label(families.get(name, {}), label, unlabelled)
            else:
                report["missing"].append(name)
    for name in COUNTERS:
        if name not in exposed:
            report["missing"].append(name)
            continue
        series = {",".join(f"{key}={value}" for key, value in labels) or "total": value
                  for (sample, labels), value in sorted(delta.items()) if sample == name and value}
        report["counters"][name] = {"total": sum(series.values()), "series": series}
    return report


def scrape(config):
    """The metrics text, or the error that kept it from us."""
    if not config["metrics_url"]:
        return None, "disabled"
    try:
        return serving.fetch_text(config["metrics_url"], config["headers"]), None
    except (OSError, ValueError) as error:
        return None, f"{type(error).__name__}: {error}"


def server_metrics(config, before, after):
    (before_text, before_error), (after_text, after_error) = before, after
    report = {"url": config["metrics_url"], "available": before_text is not None and after_text is not None,
              "error": before_error or after_error}
    if report["available"]:
        report.update(metrics_delta(before_text, after_text))
    return report


# ----------------------------------------------------------------------- SLO

def slo_values(kind, realtime, limits):
    """The real-time p95s the SLOs read, by check name."""
    if kind == "asr":
        return {"asr_first_partial_p95": realtime[limits["partial_field"]]["p95"],
                "asr_final_p95": realtime[limits["final_field"]]["p95"]}
    return {"tts_first_audio_p95": realtime["first_audio_s"]["p95"]}


def evaluate_slos(kind, realtime, baseline, limits):
    """Each SLO for one step. A latency check passes when its p95 exists and
    is within its limit. A degradation check compares the p95 with the
    level-0 step's (`baseline`, a real-time summary) and is None (not
    evaluated) on the baseline step itself or without a baseline value."""
    ceilings = {"asr_first_partial_p95": limits["asr_first_partial_s"],
                "asr_final_p95": limits["asr_final_s"],
                "tts_first_audio_p95": limits["tts_first_audio_s"]}
    values = slo_values(kind, realtime, limits)
    reference = slo_values(kind, baseline, limits) if baseline else {}
    checks = {}
    for name, value in values.items():
        checks[name] = {"value": value, "limit": ceilings[name],
                        "pass": value is not None and value <= ceilings[name]}
        base = reference.get(name)
        change = (value - base) / base if value is not None and base else None
        checks[f"degradation_{name}"] = {
            "value": change, "baseline": base, "limit": limits["degradation"],
            "pass": None if not base else change is not None and change <= limits["degradation"],
        }
    return {"checks": checks, "pass": all(check["pass"] is not False for check in checks.values())}


def summarize_slos(steps):
    """Each SLO over every step: False when a step failed it, None when no
    step evaluated it, else True."""
    verdicts = {}
    for step in steps:
        for name, check in step["slo"]["checks"].items():
            verdicts.setdefault(name, []).append(check["pass"])
    checks = {name: (False if False in passes else True if True in passes else None)
              for name, passes in verdicts.items()}
    return {"checks": checks,
            "pass": all(step["slo"]["pass"] for step in steps),
            "failed_steps": [step["step"]["name"] for step in steps if not step["slo"]["pass"]]}


# ---------------------------------------------------------------------- ramp

def parse_levels(text):
    """Batch concurrency levels; 0 (the real-time-only baseline) goes first."""
    try:
        levels = [int(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError("levels are comma-separated integers") from None
    if not levels or any(level < 0 for level in levels) or len(set(levels)) != len(levels):
        raise argparse.ArgumentTypeError("levels are distinct and not negative")
    if 0 in levels[1:]:
        raise argparse.ArgumentTypeError("level 0 (the baseline) must be the first level")
    return levels


def plan_steps(levels):
    return [{"index": index, "name": f"step-{index:02d}-batch-c{level}",
             "batch_concurrency": level, "baseline": level == 0}
            for index, level in enumerate(levels)]


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def public_config(config):
    """The run's settings, without secrets or audio."""
    hidden = ("headers", "reference_b64", "limits")
    return {key: value for key, value in config.items() if key not in hidden}


def run_realtime(config, inputs):
    """The real-time load for --step-secs; returns (records, seconds it ran)."""
    started = time.monotonic()
    if config["realtime_kind"] == "asr":
        records = asyncio.run(run_asr_sessions(
            config, inputs["frames"], config["realtime_streams"], started + config["step_secs"]))
    else:
        loop = ClosedLoop(lambda index: stream_speech(config, index), config["realtime_streams"],
                          config["retry_delay"]).start()
        time.sleep(config["step_secs"])
        records, _ = loop.finish()
    return records, time.monotonic() - started


def batch_sender(config, inputs):
    if config["batch_kind"] == "asr":
        return lambda index: batch_transcribe(config, inputs["clips"], index)
    return lambda index: batch_speech(config, index)


def run_step(config, step, inputs, baseline):
    """One ramp step: the batch level starts, the real-time load runs beside
    it for --step-secs, and the batch requests in flight finish."""
    started_at = utc_now()
    before = scrape(config)
    # Batch speech inputs start past the real-time side's, so the two sides
    # never ask for the same (text, seed).
    batch = ClosedLoop(batch_sender(config, inputs), step["batch_concurrency"],
                       config["retry_delay"], first_index=len(tts.POOL) // 2).start()
    if step["batch_concurrency"]:
        time.sleep(config["batch_lead_s"])
    realtime_records, realtime_duration = run_realtime(config, inputs)
    batch_records, batch_duration = batch.finish()
    after = scrape(config)

    realtime = dict(kind=config["realtime_kind"], model=config["realtime_model"],
                    streams=config["realtime_streams"],
                    **realtime_summary(config["realtime_kind"], realtime_records, realtime_duration))
    result = {
        "schema": STEP_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "started_at": started_at,
        "ended_at": utc_now(),
        "step": step,
        "config": public_config(config),
        "realtime": realtime,
        "batch": dict(kind=config["batch_kind"], model=config["batch_model"],
                      concurrency=step["batch_concurrency"],
                      service_tier=config["batch_service_tier"],
                      **batch_summary(batch_records, batch_duration)),
        "server_metrics": server_metrics(config, before, after),
        "slo": evaluate_slos(config["realtime_kind"], realtime, baseline, config["limits"]),
    }
    if config["save_records"]:
        result["records"] = {"realtime": realtime_records, "batch": batch_records}
    return result


def step_digest(result, path):
    """A step's headline numbers, for the summary."""
    realtime, batch = result["realtime"], result["batch"]
    keep = ("requests", "completed", "errors", "refused")
    digest = {
        "step": result["step"],
        "file": path,
        "realtime": {key: realtime[key] for key in keep},
        "batch": {key: batch[key] for key in (*keep, "audio_seconds_per_second")},
        "slo": result["slo"],
    }
    digest["batch"]["latency_s"] = {key: batch["latency_s"][key] for key in ("p50", "p95")}
    fields = ((*PARTIAL_REFERENCES.values(), *FINAL_REFERENCES.values())
              if realtime["kind"] == "asr" else ("first_audio_s", "inter_chunk_gap_s"))
    for field in fields:
        digest["realtime"][field] = {key: realtime[field][key] for key in ("p50", "p95", "max")}
    return digest


def fmt(value, digits=2):
    return "-" if value is None else f"{value:.{digits}f}"


def print_step(result, out=None):
    out = out or sys.stdout
    realtime, batch = result["realtime"], result["batch"]
    limits = result["config"]
    if realtime["kind"] == "asr":
        partial = realtime[PARTIAL_REFERENCES[limits["partial_reference"]]]
        final = realtime[FINAL_REFERENCES[limits["final_reference"]]]
        latency = (f"first partial p50/p95 {fmt(partial['p50'])}/{fmt(partial['p95'])} s, "
                   f"final p50/p95 {fmt(final['p50'])}/{fmt(final['p95'])} s")
    else:
        first, gap = realtime["first_audio_s"], realtime["inter_chunk_gap_s"]
        latency = (f"first audio p50/p95 {fmt(first['p50'])}/{fmt(first['p95'])} s, "
                   f"gap p50/p95/max {fmt(gap['p50'], 3)}/{fmt(gap['p95'], 3)}/{fmt(gap['max'], 3)} s")
    failed = [name for name, check in result["slo"]["checks"].items() if check["pass"] is False]
    print(f"{result['step']['name']}: real-time {realtime['completed']}/{realtime['requests']} ok "
          f"({realtime['refused']['total']} refused), {latency}; "
          f"batch {batch['completed']}/{batch['requests']} ok ({batch['refused']['total']} refused), "
          f"{fmt(batch['audio_seconds_per_second'])} audio s/s, "
          f"latency p50/p95 {fmt(batch['latency_s']['p50'])}/{fmt(batch['latency_s']['p95'])} s; "
          f"SLO {'pass' if result['slo']['pass'] else 'FAIL ' + ', '.join(failed)}", file=out, flush=True)
    for side in (realtime, batch):
        for error in side["error_samples"]:
            print(f"  {side['kind']} error: {error}", file=out)
    if not result["server_metrics"]["available"]:
        print(f"  server metrics not scraped: {result['server_metrics']['error']}", file=out)


def warm_up(config, inputs, with_batch):
    """One real-time unit and one batch request, discarded, so model loads
    are not measured. Returns the first failure, or None."""
    if config["realtime_kind"] == "asr":
        records = asyncio.run(run_asr_sessions(
            config, inputs["frames"], 1, time.monotonic() + config["turn_timeout"], max_turns=1))
    else:
        records = [attempt(lambda index: stream_speech(config, index), 0)]
    if with_batch:
        records.append(attempt(batch_sender(config, inputs), 0))
    return next((f"{record['status']}: {record.get('error')}" for record in records
                 if record["status"] != "ok"), None)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--api-key", default=os.environ.get("IZWI_BENCH_API_KEY"),
                        help="bearer token (default $IZWI_BENCH_API_KEY)")
    parser.add_argument("--out", required=True, type=pathlib.Path, help="directory for the step and summary JSON")
    parser.add_argument("--realtime-kind", choices=("asr", "tts"), required=True)
    parser.add_argument("--realtime-model", required=True)
    parser.add_argument("--realtime-streams", type=int, default=8,
                        help="live sessions or streams held through every step (default 8)")
    parser.add_argument("--batch-kind", choices=("asr", "tts"), help="default --realtime-kind")
    parser.add_argument("--batch-model", help="default --realtime-model")
    parser.add_argument("--batch-levels", type=parse_levels, default=parse_levels("0,4,16,64"),
                        help="batch concurrency per step; 0 is the real-time-only baseline and "
                             "goes first (default 0,4,16,64)")
    parser.add_argument("--batch-service-tier",
                        help="sent as `service_tier` and X-Izwi-Service-Tier on batch requests")
    parser.add_argument("--step-secs", type=float, default=30.0,
                        help="how long each step holds the real-time load (default 30)")
    parser.add_argument("--batch-lead-s", type=float, default=2.0,
                        help="batch head start before the real-time load of a step (default 2)")
    parser.add_argument("--audio", type=pathlib.Path, default=REPO / "data" / "fox.wav",
                        help="24 kHz PCM16 WAV each real-time ASR turn speaks (default data/fox.wav)")
    parser.add_argument("--batch-audio", type=pathlib.Path, action="append", default=[],
                        help="batch ASR source WAV (repeatable; default --audio)")
    parser.add_argument("--batch-min-secs", type=float, default=2.0, help="shortest batch ASR clip (default 2)")
    parser.add_argument("--batch-max-secs", type=float, default=30.0, help="longest batch ASR clip (default 30)")
    parser.add_argument("--batch-clips", type=int, default=64, help="distinct batch ASR clips (default 64)")
    parser.add_argument("--language", help="ASR language hint, both sides")
    parser.add_argument("--silence-ms", type=int, default=600, help="VAD silence that ends an utterance")
    parser.add_argument("--trailing-silence-ms", type=int, default=400,
                        help="extra silence sent past the VAD window")
    parser.add_argument("--tts-stream-format", choices=("audio", "sse"), default="audio",
                        help="real-time TTS delivery (default audio: a chunked WAV body)")
    parser.add_argument("--tts-extra", type=json.loads, default={},
                        help="JSON object of extra speech fields for every TTS request, e.g. '{\"voice\": \"Ryan\"}'")
    parser.add_argument("--reference-wav", type=pathlib.Path, default=REPO / "data" / "fox.wav",
                        help=f'voice-clone reference; its text must be "{tts.REFERENCE_TEXT}"')
    parser.add_argument("--seed", type=int, default=0, help="seeds the batch ASR clip cuts")
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds per HTTP request (default 600)")
    parser.add_argument("--turn-timeout", type=float, default=120.0,
                        help="seconds a real-time ASR turn may take (default 120)")
    parser.add_argument("--retry-delay", type=float, default=0.5,
                        help="pause after a failed or refused request (default 0.5)")
    parser.add_argument("--no-warmup", action="store_true", help="skip the discarded warm-up requests")
    parser.add_argument("--metrics-url", help="Prometheus endpoint (default <base-url>/v1/metrics/prometheus)")
    parser.add_argument("--no-metrics", action="store_true", help="do not scrape the server's metrics")
    parser.add_argument("--slo-asr-first-partial-s", type=float, default=0.5,
                        help="ASR first partial p95 ceiling (default 0.5)")
    parser.add_argument("--slo-asr-final-s", type=float, default=0.8,
                        help="ASR final transcript p95 ceiling after the end of speech (default 0.8)")
    parser.add_argument("--slo-tts-first-audio-s", type=float, default=0.5,
                        help="TTS first audio p95 ceiling (default 0.5)")
    parser.add_argument("--slo-degradation", type=float, default=0.25,
                        help="largest relative rise of those p95s over the level-0 step (default 0.25)")
    parser.add_argument("--partial-reference", choices=tuple(PARTIAL_REFERENCES), default="start",
                        help="the first-partial SLO counts from the first audio frame (start) or "
                             "the server's speech_stopped (vad) (default start)")
    parser.add_argument("--final-reference", choices=tuple(FINAL_REFERENCES), default="vad",
                        help="the final-transcript SLO counts from the server's speech_stopped (vad) "
                             "or the end of the spoken audio, VAD silence included (audio-end) (default vad)")
    parser.add_argument("--enforce-slo", action="store_true",
                        help=f"exit {SLO_FAILURE_EXIT} when a step fails an SLO")
    parser.add_argument("--save-records", action="store_true", help="keep every request record in the step files")
    args = parser.parse_args(argv)
    if args.realtime_streams < 1:
        parser.error("--realtime-streams must be at least 1")
    if args.step_secs <= 0:
        parser.error("--step-secs must be positive")
    if not isinstance(args.tts_extra, dict):
        parser.error("--tts-extra must be a JSON object")
    return args


def build_config(args):
    base_url = args.base_url.rstrip("/")
    return {
        "base_url": base_url,
        "headers": {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {},
        "realtime_kind": args.realtime_kind,
        "realtime_model": args.realtime_model,
        "realtime_streams": args.realtime_streams,
        "batch_kind": args.batch_kind or args.realtime_kind,
        "batch_model": args.batch_model or args.realtime_model,
        "batch_levels": args.batch_levels,
        "batch_service_tier": args.batch_service_tier,
        "step_secs": args.step_secs,
        "batch_lead_s": args.batch_lead_s,
        "audio": str(args.audio),
        "batch_audio": [str(path) for path in args.batch_audio or [args.audio]],
        "batch_min_secs": args.batch_min_secs,
        "batch_max_secs": args.batch_max_secs,
        "batch_clips": args.batch_clips,
        "language": args.language,
        "silence_ms": args.silence_ms,
        "trailing_silence_ms": args.trailing_silence_ms,
        "tts_stream_format": args.tts_stream_format,
        "tts_extra": args.tts_extra,
        "reference_b64": None,
        "seed": args.seed,
        "timeout": args.timeout,
        "turn_timeout": args.turn_timeout,
        "retry_delay": args.retry_delay,
        "metrics_url": None if args.no_metrics else args.metrics_url or base_url + "/v1/metrics/prometheus",
        "partial_reference": args.partial_reference,
        "final_reference": args.final_reference,
        "save_records": args.save_records,
        "limits": {
            "asr_first_partial_s": args.slo_asr_first_partial_s,
            "asr_final_s": args.slo_asr_final_s,
            "tts_first_audio_s": args.slo_tts_first_audio_s,
            "degradation": args.slo_degradation,
            "partial_field": PARTIAL_REFERENCES[args.partial_reference],
            "final_field": FINAL_REFERENCES[args.final_reference],
        },
    }


def needs_reference(config):
    """Whether a TTS side's model clones the reference clip."""
    sides = ((config["realtime_kind"], config["realtime_model"]), (config["batch_kind"], config["batch_model"]))
    return any(kind == "tts" and tts.CLONE in tts.MODELS.get(model, {}) for kind, model in sides)


def build_inputs(config):
    """The real-time ASR frames and the batch ASR clips the chosen kinds need."""
    inputs = {}
    if config["realtime_kind"] == "asr":
        pcm, rate = voice.read_wav(config["audio"])
        if rate != voice.OPENAI_SAMPLE_RATE:
            raise ValueError(f"/v1/realtime needs 24 kHz audio; {config['audio']} is {rate} Hz")
        inputs["frames"] = voice.pcm_frames(pcm, rate)
    if config["batch_kind"] == "asr":
        inputs["clips"] = batch_clips(config["batch_audio"], config["batch_clips"], config["batch_min_secs"],
                                      config["batch_max_secs"], random.Random(config["seed"]))
    return inputs


def main(argv=None):
    args = parse_args(argv)
    config = build_config(args)
    try:
        inputs = build_inputs(config)
        if needs_reference(config):
            config["reference_b64"] = base64.b64encode(args.reference_wav.read_bytes()).decode()
    except (OSError, ValueError) as error:
        print(f"inputs: {error}", file=sys.stderr)
        return 2
    steps = plan_steps(config["batch_levels"])
    args.out.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()

    if not args.no_warmup:
        failure = warm_up(config, inputs, any(step["batch_concurrency"] for step in steps))
        if failure:
            print(f"warm-up failed: {failure}", file=sys.stderr)
            return 1

    results, digests, baseline = [], [], None
    for step in steps:
        result = run_step(config, step, inputs, baseline)
        if step["baseline"]:
            baseline = result["realtime"]
        path = args.out / f"{step['name']}.json"
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print_step(result)
        results.append(result)
        digests.append(step_digest(result, path.name))

    slo = summarize_slos(results)
    summary = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "started_at": started_at,
        "ended_at": utc_now(),
        "metadata": {
            "git": serving.git_info(),
            "host": serving.host_info(),
            "server": serving.server_info(config["base_url"], config["headers"]),
        },
        "config": public_config(config),
        "slo_limits": config["limits"],
        "steps": digests,
        "slo": slo,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"SLO {'pass' if slo['pass'] else 'FAIL in ' + ', '.join(slo['failed_steps'])}; "
          f"results in {args.out}")
    return SLO_FAILURE_EXIT if args.enforce_slo and not slo["pass"] else 0


if __name__ == "__main__":
    sys.exit(main())
