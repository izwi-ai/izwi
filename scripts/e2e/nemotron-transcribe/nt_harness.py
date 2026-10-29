#!/usr/bin/env python3
"""Nemotron 3.5 Transcribe end-to-end and load harness (Python stdlib only).

Subcommands (run-e2e.sh, run-load.sh, run-parity.sh and collect.sh call them):

  run       start izwi-server, load the model and run the E2E matrix of plan
            §10.4; every case passes, fails, errors or is skipped into the
            `e2e` section of report.json (written after every case)
  load      start izwi-server and run scripts/bench/run-audio-load.py
            --require-parity at each concurrency, then chat inter-token
            latency beside a one-hour transcription (with --chat-model)
  merge     put a JSON file into one section of report.json
  snapshot  save /v1/metrics, /v1/health and the admin model record
  audio     convert the repo's clips into the kit's 16 kHz inputs (ffmpeg)

The server is started with RUST_LOG=info and JSON logs, and the run stops
unless its logged `build_git_sha` (and /v1/health's) equals the binary's
recorded sha (build-info.json) — lessons 2026-10-27: a stale binary cost
whole runs.
"""
from __future__ import annotations

import argparse
import datetime
import http.client
import json
import os
import random
import re
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
VARIANT = "Nemotron-3.5-Transcribe-30B-A4B-FP8"
REPORT_SCHEMA = "izwi.nemotron_transcribe.report"
MIB = 1 << 20
GIB = 1 << 30
# izwi prompt.rs VERBATIM_INSTRUCTION (sent as a `prompt` override).
VERBATIM = ("Audio transcription. Language: as spoken. Wording: verbatim, including fillers and "
            "repetitions. Formatting: punctuation and capitalization. Output: transcript only.")
BUDGET_ENV = {"cpu": "IZWI_CPU_MEMORY_BUDGET_BYTES", "metal": "IZWI_METAL_MEMORY_BUDGET_BYTES",
              "cuda": "IZWI_CUDA_MEMORY_BUDGET_BYTES"}
REQUEST_TIMEOUT = {"cpu": 7200, "metal": 3600, "cuda": 1800}
LONG_TIMEOUT = {"cpu": 6 * 3600, "metal": 3 * 3600, "cuda": 3600}
GERMAN_MARKERS = {"der", "die", "das", "und", "nicht", "ist", "wir", "mit", "ein", "eine", "zu",
                  "sich", "es", "von", "fuchs", "über"}
ENGLISH_MARKERS = {"the", "and", "is", "of", "to", "a", "in", "that", "it", "we", "for", "with"}


def log(message):
    print(f"[nt-harness] {message}", file=sys.stderr, flush=True)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def words(text):
    return re.findall(r"[\w']+", (text or "").lower())


def overlap(a, b):
    """Fraction of b's words found in a (bag of words)."""
    wa, wb = words(a), words(b)
    if not wb:
        return 1.0 if not wa else 0.0
    pool = {}
    for w in wa:
        pool[w] = pool.get(w, 0) + 1
    hit = 0
    for w in wb:
        if pool.get(w, 0) > 0:
            pool[w] -= 1
            hit += 1
    return hit / len(wb)


# ------------------------------------------------------------------- HTTP

def http(method, url, body=None, headers=None, timeout=600):
    request = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read(), time.time() - started
    except urllib.error.HTTPError as error:
        return error.code, error.read(), time.time() - started


def parse(data):
    try:
        return json.loads(data)
    except (ValueError, TypeError):
        return data.decode(errors="replace") if isinstance(data, bytes) else data


def error_message(payload):
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("message"))
        return str(error or payload)
    return str(payload)[:500]


def multipart(fields, files):
    boundary = "----izwi-nt-" + uuid.uuid4().hex
    parts = []
    for name, value in fields:
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
                     f"{value}\r\n".encode())
    for name, filename, data, content_type in files:
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
                     f'filename="{filename}"\r\nContent-Type: {content_type}\r\n\r\n'.encode()
                     + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def mime(path):
    return {".wav": "audio/wav", ".mp3": "audio/mpeg", ".ogg": "audio/ogg", ".opus": "audio/ogg",
            ".flac": "audio/flac"}.get(Path(path).suffix.lower(), "application/octet-stream")


# ----------------------------------------------------------------- server

def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Server:
    def __init__(self, binary, backend, models_dir, out_dir, label, expected_sha=None,
                 port=None, env_extra=None):
        self.binary, self.backend = Path(binary), backend
        self.models_dir, self.out_dir, self.label = models_dir, Path(out_dir), label
        self.expected_sha = expected_sha
        self.port = port or free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.env_extra = env_extra or {}
        self.process = None
        self.log_path = self.out_dir / f"server-{label}.log"
        self.build_sha = None

    def start(self, timeout=600):
        state = self.out_dir / f"state-{self.label}"
        for sub in ("media", "spool"):
            (state / sub).mkdir(parents=True, exist_ok=True)
        env = {k: v for k, v in os.environ.items() if k != "IZWI_API_KEY"}
        env.update({
            "IZWI_BACKEND": self.backend, "IZWI_HOST": "127.0.0.1", "IZWI_PORT": str(self.port),
            "IZWI_MODELS_DIR": str(self.models_dir), "IZWI_DB_PATH": str(state / "izwi.sqlite3"),
            "IZWI_MEDIA_DIR": str(state / "media"), "IZWI_SPEECH_SPOOL_DIR": str(state / "spool"),
            "IZWI_LOG_FORMAT": "json", "NO_COLOR": "1",
            "RUST_LOG": os.environ.get("RUST_LOG", "info"),
            "IZWI_OPENAI_AUDIO_UPLOAD_LIMIT_MB": os.environ.get(
                "IZWI_OPENAI_AUDIO_UPLOAD_LIMIT_MB", "64"),
        })
        env.update(self.env_extra)
        self.env_used = {k: env[k] for k in sorted(env) if k.startswith(("IZWI_", "RUST_LOG"))}
        log(f"starting {self.binary} ({self.label}) on {self.base}; log {self.log_path}")
        handle = open(self.log_path, "w")
        self.process = subprocess.Popen([str(self.binary)], env=env, stdout=handle,
                                        stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.time() + timeout
        while True:
            if self.process.poll() is not None:
                raise RuntimeError(f"izwi-server exited with {self.process.returncode}:\n"
                                   + self.tail(40))
            try:
                status, _, _ = http("GET", self.base + "/readyz", timeout=5)
                if status == 200:
                    break
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                pass
            if time.time() > deadline:
                raise RuntimeError(f"izwi-server not ready after {timeout} s:\n" + self.tail(40))
            time.sleep(1)
        self.verify_sha()
        return self

    def tail(self, lines):
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-lines:])
        except OSError:
            return ""

    def verify_sha(self):
        logged = None
        for line in self.log_path.read_text(errors="replace").splitlines():
            match = re.search(r'"build_git_sha"\s*:\s*"([^"]*)"', line) or \
                re.search(r'build_git_sha="?([0-9a-f]{7,40}|unknown)', line)
            if match:
                logged = match.group(1)
        _, body, _ = http("GET", self.base + "/v1/health", timeout=30)
        health = parse(body)
        reported = ((health or {}).get("runtime") or {}).get("build_git_sha") \
            if isinstance(health, dict) else None
        self.build_sha = logged or reported
        self.health = health
        if self.expected_sha:
            if logged != self.expected_sha:
                raise RuntimeError(f"logged build_git_sha {logged!r} is not the built "
                                   f"{self.expected_sha!r} (stale binary?)")
            if reported not in (None, self.expected_sha):
                raise RuntimeError(f"/v1/health build_git_sha {reported!r} is not "
                                   f"{self.expected_sha!r}")
        log(f"server build_git_sha {self.build_sha}")

    def stop(self):
        if self.process is None or self.process.poll() is not None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
            self.process.wait(timeout=90)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass

    # API helpers
    def get(self, path, timeout=60):
        status, body, elapsed = http("GET", self.base + path, timeout=timeout)
        return status, parse(body), elapsed

    def post(self, path, body=None, timeout=600, content_type="application/json"):
        data = json.dumps(body).encode() if isinstance(body, (dict, list)) else body
        headers = {"Content-Type": content_type} if data is not None else {}
        status, raw, elapsed = http("POST", self.base + path, data, headers, timeout)
        return status, parse(raw), elapsed

    def transcribe(self, audio, fields=(), timeout=600, model=VARIANT):
        audio = Path(audio)
        body, content_type = multipart([("model", model)] + list(fields),
                                       [("file", audio.name, audio.read_bytes(), mime(audio))])
        return self.post("/v1/audio/transcriptions", body, timeout, content_type)

    def submit_job(self, audio, fields=(), job_kind="transcription", model=VARIANT):
        audio = Path(audio)
        body, content_type = multipart([("model", model)] + list(fields),
                                       [("file", audio.name, audio.read_bytes(), mime(audio))])
        return self.post(f"/v1/speech-to-text/jobs?job_kind={job_kind}", body, 600, content_type)

    def job(self, record_id):
        return self.get(f"/v1/speech-to-text/jobs/{urllib.parse.quote(str(record_id))}")

    def wait_job(self, record_id, timeout):
        deadline = time.time() + timeout
        while True:
            status, record, _ = self.job(record_id)
            state = (record or {}).get("processing_status") if isinstance(record, dict) else None
            if state in ("ready", "failed"):
                return record
            if time.time() > deadline:
                raise TimeoutError(f"job {record_id} still {state} after {timeout} s")
            time.sleep(2)

    def metrics(self):
        status, body, _ = self.get("/v1/metrics", timeout=30)
        return body if status == 200 and isinstance(body, dict) else {}

    def model_diagnostics(self, metrics=None):
        metrics = metrics if metrics is not None else self.metrics()
        for model in metrics.get("models", []) or []:
            if VARIANT.lower() in json.dumps(model).lower():
                return model.get("family_diagnostics") or {}
        return {}

    def idle(self, metrics):
        coordinator = metrics.get("coordinator") or {}
        return (metrics.get("requests_active", 0) == 0 and coordinator.get("active_jobs", 0) == 0
                and coordinator.get("active_executions", 0) == 0)

    def wait_idle(self, timeout=600, poll=0.05):
        started = time.time()
        while time.time() - started < timeout:
            if self.idle(self.metrics()):
                return time.time() - started
            time.sleep(poll)
        raise TimeoutError(f"engine not idle after {timeout} s")

    def snapshot(self, out_dir, label):
        snapshot(self.base, out_dir, label)


def snapshot(base, out_dir, label):
    folder = Path(out_dir) / "snapshots"
    folder.mkdir(parents=True, exist_ok=True)
    for name, path in (("metrics", "/v1/metrics"), ("health", "/v1/health"),
                       ("admin", f"/v1/admin/models/{VARIANT}"), ("readyz", "/readyz")):
        try:
            status, body, _ = http("GET", base + path, timeout=30)
            payload = {"status": status, "captured_at": now(), "body": parse(body)}
        except Exception as error:  # noqa: BLE001
            payload = {"error": repr(error), "captured_at": now()}
        (folder / f"{label}-{name}.json").write_text(json.dumps(payload, indent=1) + "\n")


# ----------------------------------------------------------------- report

def merge_section(report_path, section, payload, meta=None):
    path = Path(report_path)
    report = json.loads(path.read_text()) if path.is_file() else {}
    report.setdefault("schema", REPORT_SCHEMA)
    report.setdefault("schema_version", 1)
    report.setdefault("sections", {})
    if meta:
        report.update(meta)
    report["sections"][section] = payload
    report["updated_at"] = now()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=1) + "\n")
    tmp.replace(path)


class Cases:
    """Runs cases and writes the e2e section after each one."""

    def __init__(self, report_path, meta, only=None, skip=()):
        self.report_path, self.meta = report_path, meta
        self.only, self.skip = only, set(skip)
        self.section = {"started_at": now(), "cases": [], "summary": {}}

    def run(self, name, fn, *args):
        if (self.only and name not in self.only) or name in self.skip:
            return None
        log(f"case {name} ...")
        started = time.time()
        try:
            status, details = fn(*args)
        except Exception as error:  # noqa: BLE001
            status, details = "error", {"exception": repr(error),
                                        "traceback": traceback.format_exc()[-4000:]}
        record = {"name": name, "status": status, "duration_s": round(time.time() - started, 3),
                  "details": details}
        self.section["cases"].append(record)
        log(f"case {name}: {status.upper()} ({record['duration_s']} s)")
        self.save()
        return record

    def save(self):
        summary = {}
        for case in self.section["cases"]:
            summary[case["status"]] = summary.get(case["status"], 0) + 1
        self.section["summary"] = summary
        self.section["updated_at"] = now()
        merge_section(self.report_path, "e2e", self.section, self.meta)

    def ok(self):
        return not any(c["status"] in ("fail", "error") for c in self.section["cases"])


# ------------------------------------------------------------------ audio

def ffmpeg_convert(ffmpeg, source, target, seconds=None, extra=None, loops=0):
    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    if loops:
        cmd += ["-stream_loop", str(loops)]
    cmd += ["-i", str(source)]
    if seconds:
        cmd += ["-t", f"{seconds:g}"]
    cmd += ["-ac", "1", "-ar", "16000"] + (extra or ["-c:a", "pcm_s16le"]) + [str(target)]
    subprocess.run(cmd, check=True)


def prepare_audio(out_dir, data_dir, ffmpeg="ffmpeg", de_clip=None, long_file=True):
    """16 kHz mono 16-bit WAVs of the repo's clips, plus the 60-minute file."""
    folder = Path(out_dir) / "audio"
    folder.mkdir(parents=True, exist_ok=True)
    data = Path(data_dir)
    plan = [("fox", "fox.wav", None), ("test", "test.wav", None),
            ("two_speakers", "diarization.wav", None), ("clip_95s", "diarization-2.mp3", 95),
            ("ceo_600s", "ceo.mp3", 600)]
    paths = {}
    for name, source, seconds in plan:
        target = folder / f"{name}_16k.wav"
        if not target.is_file():
            ffmpeg_convert(ffmpeg, data / source, target, seconds)
        paths[name] = target
    if de_clip:
        target = folder / "german_16k.wav"
        if not target.is_file():
            ffmpeg_convert(ffmpeg, de_clip, target)
        paths["german"] = target
    if long_file:
        target = folder / "long_60min.mp3"
        if not target.is_file():
            try:
                ffmpeg_convert(ffmpeg, data / "ceo.mp3", target, 3599, ["-c:a", "libmp3lame",
                                                                       "-b:a", "32k"], loops=2)
            except subprocess.CalledProcessError:
                target = folder / "long_60min.ogg"
                ffmpeg_convert(ffmpeg, data / "ceo.mp3", target, 3599,
                               ["-c:a", "libopus", "-b:a", "24k"], loops=2)
        paths["long"] = target
    return paths


# ------------------------------------------------------------------ cases

class Context:
    def __init__(self, args, server, audio):
        self.args, self.server, self.audio = args, server, audio
        self.timeout = args.request_timeout or REQUEST_TIMEOUT[args.backend]
        self.texts = {}

    def text(self, key, fields=()):
        """The plain json transcript of a clip (cached)."""
        if key not in self.texts:
            status, body, elapsed = self.server.transcribe(self.audio[key], fields, self.timeout)
            if status != 200:
                raise RuntimeError(f"{key}: {status} {error_message(body)}")
            self.texts[key] = body.get("text", "")
        return self.texts[key]


def case_load(ctx):
    s = ctx.server
    status, info, _ = s.get(f"/v1/admin/models/{VARIANT}")
    model_status = info.get("status") if isinstance(info, dict) else None
    if model_status in ("not_downloaded", "downloading"):
        return "fail", {"status": model_status, "note": "run fetch.sh first (exact pinned sizes)"}
    status, body, elapsed = s.post(f"/v1/admin/models/{VARIANT}/load", None,
                                   ctx.args.load_timeout)
    s.snapshot(ctx.args.out, "after-load")
    details = {"http": status, "load_s": round(elapsed, 2), "response": body,
               "diagnostics": s.model_diagnostics()}
    return ("pass" if status == 200 else "fail"), details


def case_json(ctx):
    status, body, elapsed = ctx.server.transcribe(ctx.audio["fox"], [("response_format", "json")],
                                                  ctx.timeout)
    text = body.get("text", "") if isinstance(body, dict) else ""
    ctx.texts["fox"] = text
    good = status == 200 and len(words(text)) >= 3
    return ("pass" if good else "fail"), {"http": status, "text": text, "elapsed_s": elapsed,
                                          "fox_overlap": overlap(text, "The quick brown fox "
                                                                 "jumps over the lazy dog")}


def case_verbose_json(ctx):
    status, body, elapsed = ctx.server.transcribe(
        ctx.audio["fox"], [("response_format", "verbose_json")], ctx.timeout)
    if status != 200 or not isinstance(body, dict):
        return "fail", {"http": status, "body": body}
    good = body.get("text") == ctx.text("fox") and (body.get("duration") or 0) > 0
    return ("pass" if good else "fail"), {
        "http": status, "keys": sorted(body), "duration": body.get("duration"),
        "language": body.get("language"), "rtf": body.get("rtf"),
        "text_matches_json": body.get("text") == ctx.text("fox"),
        "diagnostics": body.get("izwi_asr_diagnostics")}


def case_srt(ctx):
    status, body, _ = ctx.server.transcribe(ctx.audio["test"], [("response_format", "srt")],
                                            ctx.timeout)
    text = body if isinstance(body, str) else json.dumps(body)
    good = status == 200 and "-->" in text and re.search(r"^1\s*$", text, re.M) is not None
    return ("pass" if good else "fail"), {"http": status, "head": text[:400]}


def case_vtt(ctx):
    status, body, _ = ctx.server.transcribe(ctx.audio["test"], [("response_format", "vtt")],
                                            ctx.timeout)
    text = body if isinstance(body, str) else json.dumps(body)
    good = status == 200 and text.lstrip().startswith("WEBVTT") and "-->" in text
    return ("pass" if good else "fail"), {"http": status, "head": text[:300]}


def stream_transcription(server, audio, fields, timeout, stop_after_first_delta=False):
    """POST stream=true and read the SSE body; optionally hang up after the
    first delta (a client disconnect cancels generation)."""
    audio = Path(audio)
    body, content_type = multipart([("model", VARIANT), ("stream", "true")] + list(fields),
                                   [("file", audio.name, audio.read_bytes(), mime(audio))])
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=timeout)
    started = time.time()
    connection.request("POST", "/v1/audio/transcriptions", body, {"Content-Type": content_type})
    response = connection.getresponse()
    events, first_delta = [], None
    if response.status != 200:
        payload = response.read()
        connection.close()
        return response.status, [parse(payload)], None, time.time() - started
    while True:
        # HTTPResponse.readline decodes chunked transfer encoding.
        chunk = response.readline()
        if not chunk:
            break
        line = chunk.strip()
        if not line.startswith(b"data:"):
            continue
        payload = parse(line[5:].strip())
        events.append(payload)
        if isinstance(payload, dict) and payload.get("type") == "transcript.text.delta":
            if first_delta is None:
                first_delta = time.time() - started
            if stop_after_first_delta:
                break
        if isinstance(payload, dict) and payload.get("type") in ("transcript.text.done", "error"):
            break
    connection.close()
    return 200, events, first_delta, time.time() - started


def case_sse(ctx):
    status, events, first, elapsed = stream_transcription(ctx.server, ctx.audio["clip_95s"], (),
                                                          ctx.timeout)
    if status != 200:
        return "fail", {"http": status, "events": events[:3]}
    deltas = "".join(e.get("delta", "") for e in events
                     if isinstance(e, dict) and e.get("type") == "transcript.text.delta")
    done = [e for e in events if isinstance(e, dict) and e.get("type") == "transcript.text.done"]
    final = done[-1].get("text", "") if done else None
    reference = ctx.text("clip_95s")
    details = {"events": len(events), "first_delta_s": first, "elapsed_s": elapsed,
               "done_text_equals_json": final == reference,
               "deltas_equal_done": deltas.strip() == (final or "").strip(),
               "done_overlap_with_json": overlap(final or "", reference)}
    good = bool(done) and final == reference
    return ("pass" if good else "fail"), details


def case_language(ctx):
    status, body, _ = ctx.server.transcribe(ctx.audio["test"], [("language", "en")], ctx.timeout)
    text = body.get("text", "") if isinstance(body, dict) else ""
    share = overlap(text, ctx.text("test"))
    good = status == 200 and share >= 0.8
    return ("pass" if good else "fail"), {"http": status, "overlap_with_default": share,
                                          "text": text[:300]}


def case_language_refused(ctx):
    status, body, _ = ctx.server.transcribe(ctx.audio["fox"], [("language", "nb")], ctx.timeout)
    return ("pass" if status == 400 else "fail"), {
        "http": status, "message": error_message(body),
        "note": "Norwegian has no ASR in the card's matrix; izwi must refuse it"}


def case_prompt(ctx):
    status, body, _ = ctx.server.transcribe(ctx.audio["test"], [("prompt", VERBATIM)],
                                            ctx.timeout)
    text = body.get("text", "") if isinstance(body, dict) else ""
    good = status == 200 and overlap(text, ctx.text("test")) >= 0.6
    return ("pass" if good else "fail"), {"http": status, "text": text[:300]}


def case_keywords(ctx):
    terms = ["Agentic AI", "sovereign data networks"]
    status, body, _ = ctx.server.transcribe(
        ctx.audio["test"], [("keywords", ", ".join(terms))], ctx.timeout)
    text = body.get("text", "") if isinstance(body, dict) else ""
    flat = " ".join(words(text))
    found = {t: " ".join(words(t)) in flat for t in terms}
    good = status == 200 and all(found.values())
    return ("pass" if good else "fail"), {"http": status, "found": found, "text": text[:400]}


def case_translate_en_de(ctx):
    status, body, _ = ctx.server.transcribe(ctx.audio["test"], [("target_language", "de")],
                                            ctx.timeout)
    text = body.get("text", "") if isinstance(body, dict) else ""
    german = len(set(words(text)) & GERMAN_MARKERS)
    good = status == 200 and german >= 2 and overlap(text, ctx.text("test")) < 0.5
    return ("pass" if good else "fail"), {"http": status, "german_markers": german,
                                          "text": text[:400]}


def case_translate_de_en(ctx):
    if "german" not in ctx.audio:
        return "skip", {"reason": "no German clip; pass --de-clip (any 10-60 s German speech)"}
    status, body, _ = ctx.server.transcribe(ctx.audio["german"], [("target_language", "en")],
                                            ctx.timeout)
    text = body.get("text", "") if isinstance(body, dict) else ""
    english = len(set(words(text)) & ENGLISH_MARKERS)
    good = status == 200 and english >= 2
    return ("pass" if good else "fail"), {"http": status, "english_markers": english,
                                          "text": text[:400]}


def case_translate_refused(ctx):
    status, body, _ = ctx.server.transcribe(ctx.audio["fox"], [("target_language", "tr")],
                                            ctx.timeout)
    return ("pass" if status == 400 else "fail"), {
        "http": status, "message": error_message(body),
        "note": "the card lists no translation for Turkish"}


def case_translate_with_timestamps_refused(ctx):
    status, body, _ = ctx.server.transcribe(
        ctx.audio["fox"], [("target_language", "de"), ("response_format", "verbose_json"),
                           ("timestamp_granularities[]", "word")], ctx.timeout)
    return ("pass" if status == 400 else "fail"), {"http": status,
                                                   "message": error_message(body)}


def case_word_timestamps(ctx):
    status, body, _ = ctx.server.transcribe(
        ctx.audio["two_speakers"], [("response_format", "verbose_json"),
                                    ("timestamp_granularities[]", "word")], ctx.timeout)
    if status != 200 or not isinstance(body, dict):
        return "fail", {"http": status, "message": error_message(body)}
    items = body.get("words") or []
    duration = float(body.get("duration") or 0)
    starts = [float(w.get("start", -1)) for w in items]
    problems = []
    if not items:
        problems.append("no words")
    if any(float(w.get("end", -1)) < float(w.get("start", 0)) for w in items):
        problems.append("a word ends before it starts")
    if any(b < a - 1e-6 for a, b in zip(starts, starts[1:])):
        problems.append("starts are not monotonic")
    if items and duration and float(items[-1].get("end", 0)) > duration + 0.5:
        problems.append("a word ends after the audio")
    share = overlap(" ".join(str(w.get("word", "")) for w in items), body.get("text", ""))
    if share < 0.8:
        problems.append(f"words cover only {share:.2f} of the text")
    return ("fail" if problems else "pass"), {
        "words": len(items), "duration": duration, "problems": problems,
        "first_words": items[:8], "diagnostics_keys": sorted(
            (body.get("izwi_asr_diagnostics") or {}).keys())
        if isinstance(body.get("izwi_asr_diagnostics"), dict) else None}


def case_speaker_attributed(ctx):
    status, record, _ = ctx.server.submit_job(ctx.audio["two_speakers"], (),
                                              job_kind="speaker_attributed_asr")
    if status not in (200, 201, 202) or not isinstance(record, dict):
        return "fail", {"http": status, "message": error_message(record),
                        "note": "SAA is hard-wired to Granite 4.1 Plus until Phase 4b lands"}
    record = ctx.server.wait_job(record["id"], ctx.timeout)
    turns = record.get("speaker_turns") or []
    speakers = {str(t.get("speaker") or t.get("speaker_label") or t.get("speaker_id"))
                for t in turns if isinstance(t, dict)}
    good = record.get("processing_status") == "ready" and len(speakers) >= 2
    return ("pass" if good else "fail"), {
        "processing_status": record.get("processing_status"), "saa_status":
        record.get("saa_status"), "speakers": sorted(speakers), "turns": turns[:6],
        "error": record.get("processing_error")}


def case_one_model_diarization(ctx):
    return "skip", {"reason": "Phase 4c (one-model diarization) has no request shape yet; "
                              "extend this case when it lands"}


def case_long(ctx):
    if "long" not in ctx.audio:
        return "skip", {"reason": "--skip-long"}
    started = time.time()
    status, record, _ = ctx.server.submit_job(ctx.audio["long"])
    if status not in (200, 201, 202) or not isinstance(record, dict):
        return "fail", {"http": status, "message": error_message(record)}
    record = ctx.server.wait_job(record["id"], ctx.args.long_timeout or LONG_TIMEOUT[ctx.args.backend])
    elapsed = time.time() - started
    text = record.get("transcription") or record.get("text") or record.get("transcript") or ""
    count = len(words(text))
    good = record.get("processing_status") == "ready" and count >= 3000
    return ("pass" if good else "fail"), {
        "processing_status": record.get("processing_status"), "words": count,
        "elapsed_s": round(elapsed, 1), "rtf": round(elapsed / 3599.0, 4),
        "error": record.get("processing_error"),
        "tail": text[-300:], "diagnostics": ctx.server.model_diagnostics()}


def _cancel_job_at(ctx, audio, stage):
    """Submit, wait until the model reaches `stage` ("prefill": prefill spans
    advanced while decode steps did not; "decode": 5 decode steps), cancel
    and time cancel -> idle."""
    s = ctx.server
    s.wait_idle(ctx.timeout)
    before = s.model_diagnostics()
    spans0, steps0 = before.get("prefill_spans", 0), before.get("decode_steps", 0)
    status, record, _ = s.submit_job(audio)
    if status not in (200, 201, 202) or not isinstance(record, dict):
        return None, {"http": status, "message": error_message(record)}
    record_id = record["id"]
    deadline = time.time() + ctx.timeout
    reached = None
    while time.time() < deadline:
        diag = s.model_diagnostics()
        spans, steps = diag.get("prefill_spans", 0), diag.get("decode_steps", 0)
        if stage == "prefill" and spans > spans0:
            reached = "prefill" if steps == steps0 else "decode"
            break
        if stage == "decode" and steps >= steps0 + 5:
            reached = "decode"
            break
        _, current, _ = s.job(record_id)
        if isinstance(current, dict) and current.get("processing_status") in ("ready", "failed"):
            reached = "finished"
            break
        time.sleep(0.02)
    cancel_at = time.time()
    status, body, cancel_s = s.post(f"/v1/speech-to-text/jobs/{record_id}/cancel")
    s.wait_idle(600)
    idle_s = round(time.time() - cancel_at, 3)
    final = s.wait_job(record_id, 600)
    return reached, {"cancel_http": status, "cancel_request_s": round(cancel_s, 3),
                     "cancel_to_idle_s": idle_s, "reached": reached,
                     "final_status": final.get("processing_status"),
                     "diagnostics": s.model_diagnostics()}


def case_cancel_prefill(ctx):
    details = {}
    for key in ("ceo_600s", "long"):
        if key not in ctx.audio:
            continue
        reached, details = _cancel_job_at(ctx, ctx.audio[key], "prefill")
        details["audio"] = key
        if reached == "prefill":
            good = details["cancel_http"] in (200, 202) and \
                details["cancel_to_idle_s"] <= ctx.args.cancel_idle_max_s
            return ("pass" if good else "fail"), details
    return "fail", dict(details, note="prefill finished before the harness saw it; rerun on "
                                      "a slower device or with the 60-minute file")


def case_cancel_decode(ctx):
    reached, details = _cancel_job_at(ctx, ctx.audio["ceo_600s"], "decode")
    good = reached == "decode" and details["cancel_http"] in (200, 202) and \
        details["cancel_to_idle_s"] <= ctx.args.cancel_idle_max_s
    return ("pass" if good else "fail"), details


def case_cancel_disconnect(ctx):
    s = ctx.server
    s.wait_idle(ctx.timeout)
    status, events, first, _ = stream_transcription(s, ctx.audio["ceo_600s"], (), ctx.timeout,
                                                    stop_after_first_delta=True)
    hung_up = time.time()
    if status != 200 or first is None:
        return "fail", {"http": status, "events": events[:3]}
    s.wait_idle(600)
    idle_s = round(time.time() - hung_up, 3)
    metrics = s.metrics()
    return ("pass" if idle_s <= ctx.args.cancel_idle_max_s else "fail"), {
        "first_delta_s": first, "disconnect_to_idle_s": idle_s,
        "requests_cancelled": metrics.get("requests_cancelled")}


def case_soak(ctx):
    s = ctx.server
    s.wait_idle(ctx.timeout)
    keys = ["fox", "two_speakers", "test", "clip_95s"]
    before = s.metrics()
    s.snapshot(ctx.args.out, "soak-before")
    untracked0 = (before.get("coordinator") or {}).get("untracked_device_memory_bytes", 0)
    failures, samples, lock = [], [], threading.Lock()
    jobs = [keys[i % len(keys)] for i in range(ctx.args.soak_requests)]

    def worker(queue):
        while True:
            with lock:
                if not queue:
                    return
                key = queue.pop()
            status, body, elapsed = s.transcribe(ctx.audio[key], (), ctx.timeout)
            metrics = s.metrics()
            with lock:
                samples.append({"clip": key, "http": status, "elapsed_s": round(elapsed, 2),
                                "untracked": (metrics.get("coordinator") or {}).get(
                                    "untracked_device_memory_bytes")})
                if status != 200:
                    failures.append({"clip": key, "http": status, "message": error_message(body)})

    threads = [threading.Thread(target=worker, args=(jobs,)) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    s.wait_idle(ctx.timeout)
    after = s.metrics()
    s.snapshot(ctx.args.out, "soak-after")
    untracked1 = (after.get("coordinator") or {}).get("untracked_device_memory_bytes", 0)
    growth = untracked1 - untracked0
    good = not failures and growth <= ctx.args.untracked_growth_mib * MIB
    return ("pass" if good else "fail"), {
        "requests": len(samples), "failures": failures[:5],
        "untracked_before": untracked0, "untracked_after": untracked1, "growth_bytes": growth,
        "max_untracked": max([x["untracked"] or 0 for x in samples] or [0]),
        "coordinator_after": after.get("coordinator"),
        "note": "untracked device memory is 0 on CPU by construction"}


def case_unload_reload(ctx):
    s = ctx.server
    reference = ctx.text("fox")
    status, body, unload_s = s.post(f"/v1/admin/models/{VARIANT}/unload")
    _, info, _ = s.get(f"/v1/admin/models/{VARIANT}")
    unloaded_status = info.get("status") if isinstance(info, dict) else None
    status2, body2, load_s = s.post(f"/v1/admin/models/{VARIANT}/load", None,
                                    ctx.args.load_timeout)
    status3, again, _ = s.transcribe(ctx.audio["fox"], (), ctx.timeout)
    text = again.get("text") if isinstance(again, dict) else None
    good = status == 200 and unloaded_status != "ready" and status2 == 200 and text == reference
    return ("pass" if good else "fail"), {
        "unload_http": status, "unload_s": round(unload_s, 2), "status_after_unload":
        unloaded_status, "reload_http": status2, "reload_s": round(load_s, 2),
        "text_equal_after_reload": text == reference, "text": text}


def case_oom_refusal(args, audio, build_sha):
    budget = args.undersized_budget_gib * GIB
    env = {BUDGET_ENV[args.backend]: str(budget)}
    server = Server(args.binary, args.backend, args.models_dir, args.out, "undersized",
                    build_sha, env_extra=env)
    server.start(args.ready_timeout)
    try:
        started = time.time()
        status, body, elapsed = server.post(f"/v1/admin/models/{VARIANT}/load", None,
                                            args.load_timeout)
        message = error_message(body)
        ready, _, _ = server.get("/readyz")
        server.snapshot(args.out, "undersized")
        memory_words = any(w in message.lower() for w in ("memory", "gib", "gb", "fit", "budget"))
        good = status >= 400 and elapsed <= 120 and ready == 200 and memory_words
        return ("pass" if good else "fail"), {
            "budget_env": env, "http": status, "refused_in_s": round(elapsed, 2),
            "message": message, "ready_after": ready, "mentions_memory": memory_words,
            "wall_s": round(time.time() - started, 2)}
    finally:
        server.stop()


# ------------------------------------------------------------ subcommands

def read_build_info(out_dir, explicit_sha=None):
    path = Path(out_dir) / "build-info.json"
    info = json.loads(path.read_text()) if path.is_file() else {}
    sha = explicit_sha or info.get("git_sha")
    return info, sha


def cmd_run(args):
    args.out = Path(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    build, sha = read_build_info(args.out, args.expect_sha)
    if not sha and not args.no_sha_check:
        raise SystemExit("no expected build sha: run build.sh first or pass --expect-sha")
    audio = prepare_audio(args.out, args.data_dir, args.ffmpeg, args.de_clip,
                          long_file=not args.skip_long)
    meta = {"backend": args.backend, "variant": VARIANT, "git_sha": sha,
            "fp8_mode": os.environ.get("IZWI_NEMOTRON_TRANSCRIBE_FP8_MODE", "w8a16")}
    merge_section(args.out / "report.json", "build", build, meta)
    cases = Cases(args.out / "report.json", meta, only=args.only and set(args.only.split(",")),
                  skip=args.skip.split(",") if args.skip else ())
    server = Server(args.binary, args.backend, args.models_dir, args.out, "main",
                    None if args.no_sha_check else sha)
    server.start(args.ready_timeout)
    cases.section["server"] = {"base": server.base, "build_git_sha": server.build_sha,
                               "env": server.env_used,
                               "health_runtime": (server.health or {}).get("runtime")
                               if isinstance(server.health, dict) else None}
    server.snapshot(args.out, "start")
    ctx = Context(args, server, audio)
    try:
        loaded = cases.run("load", case_load, ctx)
        if loaded and loaded["status"] != "pass":
            log("the model did not load; skipping the cases that need it")
        else:
            for name, fn in [
                ("format_json", case_json), ("format_verbose_json", case_verbose_json),
                ("format_srt", case_srt), ("format_vtt", case_vtt), ("stream_sse", case_sse),
                ("language", case_language), ("language_refused", case_language_refused),
                ("prompt_override", case_prompt), ("keywords", case_keywords),
                ("translate_en_de", case_translate_en_de),
                ("translate_de_en", case_translate_de_en),
                ("translate_refused", case_translate_refused),
                ("translate_with_timestamps_refused", case_translate_with_timestamps_refused),
                ("word_timestamps", case_word_timestamps),
                ("speaker_attributed", case_speaker_attributed),
                ("one_model_diarization", case_one_model_diarization),
                ("cancel_mid_prefill", case_cancel_prefill),
                ("cancel_mid_decode", case_cancel_decode),
                ("cancel_by_disconnect", case_cancel_disconnect),
                ("long_60min", case_long),
                ("soak_untracked_memory", case_soak),
                ("unload_reload", case_unload_reload),
            ]:
                cases.run(name, fn, ctx)
    finally:
        server.snapshot(args.out, "end")
        server.stop()
    cases.run("oom_refusal_undersized_budget", case_oom_refusal, args, audio,
              None if args.no_sha_check else sha)
    cases.save()
    log(f"e2e summary: {cases.section['summary']}; report {args.out / 'report.json'}")
    return 0 if cases.ok() else 1


def chat_stream(base, model, prompt, max_tokens, timeout=600):
    """Inter-token gaps (s) of one streamed chat completion."""
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0, "stream": True}).encode()
    parsed = urllib.parse.urlparse(base)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    connection.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    response = connection.getresponse()
    if response.status != 200:
        raise RuntimeError(f"chat {response.status}: {response.read()[:300]!r}")
    stamps = []
    while True:
        line = response.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        payload = parse(data)
        if isinstance(payload, dict):
            for choice in payload.get("choices") or []:
                if (choice.get("delta") or {}).get("content"):
                    stamps.append(time.time())
    connection.close()
    return [b - a for a, b in zip(stamps, stamps[1:])]


def summarize(values):
    if not values:
        return None
    ordered = sorted(values)

    def pick(q):
        return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]

    return {"count": len(values), "mean": statistics.fmean(values), "p50": pick(0.5),
            "p95": pick(0.95), "max": ordered[-1]}


def cmd_load(args):
    args.out = Path(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    build, sha = read_build_info(args.out, args.expect_sha)
    audio = prepare_audio(args.out, args.data_dir, args.ffmpeg, None,
                          long_file=bool(args.chat_model))
    meta = {"backend": args.backend, "variant": VARIANT, "git_sha": sha,
            "fp8_mode": os.environ.get("IZWI_NEMOTRON_TRANSCRIBE_FP8_MODE", "w8a16")}
    section = {"started_at": now(), "points": [], "chat_itl": None}
    concurrency = [int(c) for c in args.concurrency.split(",")]
    server = Server(args.binary, args.backend, args.models_dir, args.out, "load",
                    None if args.no_sha_check else sha)
    server.start(args.ready_timeout)
    status = 0
    try:
        code, body, load_s = server.post(f"/v1/admin/models/{VARIANT}/load", None,
                                         args.load_timeout)
        if code != 200:
            raise RuntimeError(f"load failed: {code} {error_message(body)}")
        section["load_s"] = round(load_s, 2)
        server.snapshot(args.out, "load-start")
        sources = [audio[k] for k in ("fox", "test", "two_speakers", "clip_95s")]
        for c in concurrency:
            output = args.out / f"load-c{c}.json"
            cmd = [sys.executable, str(REPO / "scripts/bench/run-audio-load.py"), "--task", "asr",
                   "--model", VARIANT, "--base-url", server.base,
                   "--num-requests", str(max(args.min_requests, 2 * c)),
                   "--request-rate", args.request_rate, "--burstiness", "1",
                   "--max-concurrency", str(c), "--seed", str(args.seed),
                   "--timeout", str(args.request_timeout or REQUEST_TIMEOUT[args.backend]),
                   "--min-secs", str(args.min_secs), "--max-secs", str(args.max_secs),
                   "--require-parity", "--output", str(output)]
            for source in sources:
                cmd += ["--audio", str(source)]
            log(f"load point c={c}: {' '.join(cmd[2:])}")
            started = time.time()
            result = subprocess.run(cmd)
            point = {"concurrency": c, "exit_code": result.returncode,
                     "parity": "pass" if result.returncode == 0 else
                     ("fail" if result.returncode == 3 else "error"),
                     "wall_s": round(time.time() - started, 1), "output": output.name}
            if output.is_file():
                data = json.loads(output.read_text())
                point["solo_repeatable"] = (data.get("solo") or {}).get("repeatable")
                point["reports"] = [{"name": r.get("name"),
                                     "summary": (r.get("report") or {}).get("summary")}
                                    for r in data.get("reports", [])]
            section["points"].append(point)
            server.snapshot(args.out, f"load-c{c}")
            merge_section(args.out / "report.json", "load", section, meta)
            if result.returncode != 0:
                status = 1
        if args.chat_model:
            section["chat_itl"] = chat_itl(args, server, audio)
    finally:
        server.snapshot(args.out, "load-end")
        server.stop()
    section["ended_at"] = now()
    merge_section(args.out / "report.json", "load", section, meta)
    return status


def chat_itl(args, server, audio):
    code, body, _ = server.post(f"/v1/admin/models/{urllib.parse.quote(args.chat_model)}/load",
                                None, args.load_timeout)
    if code != 200:
        return {"status": "error", "message": f"chat model load: {code} {error_message(body)}"}
    prompt = "Write a detailed, plain-prose story about a lighthouse keeper. Do not stop early."
    baseline = []
    for _ in range(3):
        baseline += chat_stream(server.base, args.chat_model, prompt, 128)
    code, record, _ = server.submit_job(audio["long"])
    if code not in (200, 201, 202) or not isinstance(record, dict):
        return {"status": "error", "message": f"1-hour job: {code} {error_message(record)}",
                "baseline": summarize(baseline)}
    time.sleep(args.itl_warmup_s)
    during, phases = [], []
    try:
        for _ in range(3):
            diag = server.model_diagnostics()
            phases.append({"prefill_spans": diag.get("prefill_spans"),
                           "decode_steps": diag.get("decode_steps")})
            during += chat_stream(server.base, args.chat_model, prompt, 128)
    finally:
        server.post(f"/v1/speech-to-text/jobs/{record['id']}/cancel")
    base, busy = summarize(baseline), summarize(during)
    ratio = busy["p50"] / base["p50"] if base and busy and base["p50"] else None
    return {"status": "measured", "chat_model": args.chat_model, "baseline_itl_s": base,
            "beside_1h_job_itl_s": busy, "p50_ratio": ratio, "transcription_phases": phases}


def cmd_merge(args):
    payload = json.loads(Path(args.file).read_text())
    merge_section(args.report, args.section, payload)
    return 0


def cmd_snapshot(args):
    snapshot(args.base_url, args.out, args.label)
    return 0


def cmd_audio(args):
    paths = prepare_audio(args.out, args.data_dir, args.ffmpeg, args.de_clip,
                          long_file=not args.skip_long)
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=1))
    return 0


def add_server_args(parser):
    parser.add_argument("--backend", choices=("cpu", "metal", "cuda"), required=True)
    parser.add_argument("--binary", required=True, help="izwi-server built by build.sh")
    parser.add_argument("--models-dir", required=True, help="IZWI_MODELS_DIR")
    parser.add_argument("--out", required=True, help="the backend's report directory")
    parser.add_argument("--data-dir", default=str(REPO / "data"))
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--expect-sha", help="default: build-info.json's git_sha")
    parser.add_argument("--no-sha-check", action="store_true")
    parser.add_argument("--ready-timeout", type=int, default=600)
    parser.add_argument("--load-timeout", type=int, default=3600)
    parser.add_argument("--request-timeout", type=int, default=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="the E2E matrix")
    add_server_args(run)
    run.add_argument("--de-clip", help="German speech for the de -> en translation case")
    run.add_argument("--skip-long", action="store_true", help="skip the 60-minute file")
    run.add_argument("--long-timeout", type=int, default=0)
    run.add_argument("--only", help="comma-separated case names")
    run.add_argument("--skip", help="comma-separated case names")
    run.add_argument("--soak-requests", type=int, default=24)
    run.add_argument("--untracked-growth-mib", type=int, default=256)
    run.add_argument("--cancel-idle-max-s", type=float, default=10.0)
    run.add_argument("--undersized-budget-gib", type=int, default=8)
    load = sub.add_parser("load", help="run-audio-load.py sweep and chat ITL")
    add_server_args(load)
    load.add_argument("--concurrency", required=True, help="e.g. 1,2,4")
    load.add_argument("--request-rate", default="0.5",
                      help="Poisson arrivals per second (staggered); 'inf' for a burst")
    load.add_argument("--min-requests", type=int, default=16)
    load.add_argument("--min-secs", type=float, default=2.0)
    load.add_argument("--max-secs", type=float, default=95.0)
    load.add_argument("--seed", type=int, default=0)
    load.add_argument("--chat-model", help="a small chat model id for ITL beside a 1-hour job")
    load.add_argument("--itl-warmup-s", type=float, default=20.0)
    merge = sub.add_parser("merge", help="put a JSON file into a report section")
    merge.add_argument("--report", required=True)
    merge.add_argument("--section", required=True)
    merge.add_argument("--file", required=True)
    snap = sub.add_parser("snapshot", help="save metrics, health and admin snapshots")
    snap.add_argument("--base-url", required=True)
    snap.add_argument("--out", required=True)
    snap.add_argument("--label", default="manual")
    audio = sub.add_parser("audio", help="prepare the kit's audio inputs")
    audio.add_argument("--out", required=True)
    audio.add_argument("--data-dir", default=str(REPO / "data"))
    audio.add_argument("--ffmpeg", default="ffmpeg")
    audio.add_argument("--de-clip")
    audio.add_argument("--skip-long", action="store_true")
    args = parser.parse_args(argv)
    return {"run": cmd_run, "load": cmd_load, "merge": cmd_merge, "snapshot": cmd_snapshot,
            "audio": cmd_audio}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
