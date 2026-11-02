#!/usr/bin/env python3
"""Fish Audio Drama 3 end-to-end and load harness (Python stdlib only).

Subcommands (run-e2e.sh, run-load.sh, run-parity.sh and collect.sh call them):

  run       start izwi-server, load the model and run the end-to-end matrix;
            every case passes, fails, errors, is inconclusive or is skipped
            into the `e2e` section of report.json (written after every case),
            and every audio it gets back is kept under audio/
  load      start izwi-server and run scripts/bench/run-audio-load.py
            --require-parity at each concurrency, a streaming probe of time to
            first audio, frames/s and RTF at each, then chat inter-token
            latency beside a long Drama 3 generation (with --chat-model)
  plan      print the cases `run` would run (a dry run)
  merge     put a JSON file into one section of report.json
  snapshot  save /v1/metrics, /v1/health, the admin model record and /readyz

The server is started with RUST_LOG=info and JSON logs, and the run stops
unless its logged `build_git_sha` (and /v1/health's) equals the binary's
recorded sha (build-info.json): a stale binary has cost whole runs.

With --synthetic the server is a `build.sh --synthetic` binary serving the
synthetic checkpoint (random tiny weights, a stub codec): requests then
sample at temperature 2 within 160 frames, and a seed whose take does not
end within its budget is retried with the next one. It checks the kit and
the routes, not the model's quality.
"""
from __future__ import annotations

import argparse
import array
import base64
import datetime
import http.client
import importlib.util
import json
import math
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
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
VARIANT = "FishAudio-Drama-3"
REPORT_SCHEMA = "izwi.fish_drama3.report"
SAMPLE_RATE = 44100
SAMPLES_PER_FRAME = 2048
MIB = 1 << 20
GIB = 1 << 30
BUDGET_ENV = {"cpu": "IZWI_CPU_MEMORY_BUDGET_BYTES", "metal": "IZWI_METAL_MEMORY_BUDGET_BYTES",
              "cuda": "IZWI_CUDA_MEMORY_BUDGET_BYTES"}
REQUEST_TIMEOUT = {"cpu": 3600, "metal": 1800, "cuda": 900}
# The synthetic checkpoint's sampling: its random weights draw eos within
# 160 frames at temperature 2 for most seeds (the server tests' setting).
SYNTHETIC_SAMPLING = {"temperature": 2.0, "max_tokens": 160}
# The card's streaming table (one H200, sglang-omni, Seed-TTS English
# prompts, temperature 1.0 / top-p 0.9 / top-k 30): context for the load
# probe, never a gate. concurrency -> (frames/s, TTFA p50 ms, TTFA p95 ms, RTF mean).
CARD_H200 = {1: (166.7, 88.9, 90.8, 0.131), 2: (287.0, 96.2, 130.1, 0.152),
             4: (467.6, 108.2, 156.4, 0.186), 8: (678.1, 153.2, 210.9, 0.256),
             16: (927.6, 210.1, 287.2, 0.373), 32: (1244.7, 304.6, 396.1, 0.551),
             64: (1687.3, 450.9, 1020.1, 0.801)}

TEXTS = {
    "plain": "The quick brown fox jumps over the lazy dog, then naps in the warm afternoon sun.",
    # Two lines that open with a tag: the recipe's leading-silence quirk.
    "tags": ("[sighs] I told you this would happen.\n"
             "[laughs] Fine, have it your way. [pause] We start again tomorrow, early."),
    "take_1": "Ready when you are. The lights are on and the room is quiet.",
    "take_2": "Then let's begin with the first scene, slowly this time.",
    # About 1,400 UTF-8 bytes: five chunks of about 300.
    "long": (
        "The old lighthouse stood at the edge of the harbor for more than a century. Every "
        "evening, the keeper climbed one hundred and twelve steps to light the lamp, and every "
        "morning he climbed them again to put it out. Ships from distant ports used its beam to "
        "find the narrow channel between the rocks. In winter, storms pushed waves high enough "
        "to wash over the lower windows, and the keeper would sit by the stove, listening to "
        "the wind and writing in his logbook. He recorded the weather, the ships that passed, "
        "and the small events of each day. One entry describes a fishing boat that lost its "
        "engine in a gale and drifted toward the cliffs until the crew saw the light and rowed "
        "to safety. Another mentions a family of seals that slept on the breakwater every "
        "spring. When the lighthouse was finally automated, the logbooks were sent to the town "
        "library, where students still read them today. They show how a single person, doing "
        "the same careful work every night, kept hundreds of strangers safe. The building is "
        "now a small museum, and on clear nights the automated lamp still turns."
    ),
}
# A three-line script for two speakers (speaker 0 cloned from data/fox.wav,
# speaker 1 from data/test.wav).
DIALOGUE = [
    (0, "I checked the harbor charts twice this morning, and the channel between the rocks "
        "is narrower than it was last spring, so we should wait for the tide."),
    (1, "Waiting costs us the whole afternoon, though. If we leave now and keep to the "
        "lighthouse beam, the boat will clear the rocks before the light fades."),
    (0, "Fine, but you steer, and if the wind turns against us we come straight back to the "
        "breakwater without arguing about it."),
]


def log(message):
    print(f"[fd3-harness] {message}", file=sys.stderr, flush=True)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


class CaseError(Exception):
    """A case cannot go on; its message becomes the case's details."""


# ------------------------------------------------------------------- audio

def wav_pcm(data):
    """(sample_rate, channels, 16-bit PCM bytes) of WAV bytes; tolerates the
    open sizes of a streamed WAV."""
    if data[:4] != b"RIFF":
        raise ValueError("not a WAV")
    pos, rate, channels = 12, None, None
    while pos + 8 <= len(data):
        chunk, size = data[pos:pos + 4], int.from_bytes(data[pos + 4:pos + 8], "little")
        if chunk == b"fmt ":
            channels = int.from_bytes(data[pos + 10:pos + 12], "little")
            rate = int.from_bytes(data[pos + 12:pos + 16], "little")
        if chunk == b"data":
            end = len(data) if size in (0, 0xFFFFFFFF) or pos + 8 + size > len(data) \
                else pos + 8 + size
            return rate, channels, data[pos + 8:end]
        pos += 8 + size + (size & 1)
    raise ValueError("WAV without a data chunk")


def samples_i16(pcm):
    values = array.array("h")
    values.frombytes(pcm[:len(pcm) - len(pcm) % 2])
    if sys.byteorder != "little":
        values.byteswap()
    return values


def write_wav(path, pcm, rate=SAMPLE_RATE, channels=1):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm)
    return path


def audio_stats(pcm, rate=SAMPLE_RATE):
    """Duration, level, voiced share and leading/trailing silence (20 ms
    windows below -50 dBFS) of 16-bit mono PCM."""
    values = samples_i16(pcm)
    if not values:
        return {"duration_s": 0.0, "frames": 0}
    window = max(1, rate // 50)
    loud = []
    for start in range(0, len(values) - window + 1, window):
        energy = sum((v / 32768.0) ** 2 for v in values[start:start + window]) / window
        loud.append(20 * math.log10(math.sqrt(energy) + 1e-9) >= -50)
    lead = loud.index(True) if True in loud else len(loud)
    trail = loud[::-1].index(True) if True in loud else len(loud)
    mean_square = sum((v / 32768.0) ** 2 for v in values) / len(values)
    return {"duration_s": round(len(values) / rate, 3),
            "frames": round(len(values) / SAMPLES_PER_FRAME, 2),
            "rms_dbfs": round(20 * math.log10(math.sqrt(mean_square) + 1e-9), 1),
            "peak": round(max(abs(v) for v in values) / 32768.0, 4),
            "voiced_share": round(sum(loud) / max(1, len(loud)), 3),
            "lead_silence_s": round(lead * 0.02, 2), "trail_silence_s": round(trail * 0.02, 2)}


def audible(stats, synthetic):
    """Some sound, not mostly silence (the stub codec makes a quiet sine)."""
    if synthetic:
        return stats.get("duration_s", 0) > 0
    return stats.get("duration_s", 0) >= 0.5 and stats.get("rms_dbfs", -100) > -45 \
        and stats.get("voiced_share", 0) >= 0.3


def max_abs_diff(a, b):
    x, y = samples_i16(a), samples_i16(b)
    if len(x) != len(y):
        return None
    return max((abs(p - q) for p, q in zip(x, y)), default=0)


def b64_file(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def unique_reference(source, seconds, path, nonce):
    """`source` repeated to `seconds` at its own rate, its first samples set
    from `nonce`, so the reference cache cannot have seen it."""
    rate, channels, pcm = wav_pcm(Path(source).read_bytes())
    values = samples_i16(pcm)
    if channels and channels > 1:
        values = array.array("h", values[::channels])
    want = int(seconds * rate)
    out = array.array("h")
    while len(out) < want:
        out.extend(values[:want - len(out)])
    rng = random.Random(nonce)
    for i in range(16):
        out[i] = rng.randint(-64, 64)
    return write_wav(path, out.tobytes(), rate)


# ------------------------------------------------------------------- HTTP

def http_call(method, url, body=None, headers=None, timeout=600):
    request = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read(), dict(response.headers), \
                time.time() - started
    except urllib.error.HTTPError as error:
        return error.code, error.read(), dict(error.headers or {}), time.time() - started


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
        return str(error or payload.get("processing_error") or payload)
    if isinstance(payload, bytes):
        payload = payload.decode(errors="replace")
    return str(payload)[:600]


def is_frame_limit(message):
    return "frame_limit" in (message or "") or "frame budget" in (message or "")


def mode_env(synthetic):
    """The synthetic checkpoint's server is a debug build, which polls the
    durable speech path deeper than tokio's 2 MiB worker stacks (the server
    tests run it on 32 MiB ones); a release build needs nothing."""
    return {"RUST_MIN_STACK": str(32 << 20)} if synthetic else {}


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Server:
    def __init__(self, binary, backend, models_dir, out_dir, label, expected_sha=None,
                 env_extra=None):
        self.binary, self.backend = Path(binary), backend
        self.models_dir, self.out_dir, self.label = models_dir, Path(out_dir), label
        self.expected_sha = expected_sha
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.env_extra = env_extra or {}
        self.process = None
        self.log_path = self.out_dir / f"server-{label}.log"
        self.build_sha = None
        self.health = None
        self.env_used = {}

    def start(self, timeout=600):
        state = self.out_dir / f"state-{self.label}"
        # The speech spool must not grant group or other permissions.
        for folder in (state, state / "media", state / "spool"):
            folder.mkdir(parents=True, exist_ok=True)
            folder.chmod(0o700)
        env = {k: v for k, v in os.environ.items() if k != "IZWI_API_KEY"}
        env.update({
            "IZWI_BACKEND": self.backend, "IZWI_HOST": "127.0.0.1", "IZWI_PORT": str(self.port),
            "IZWI_MODELS_DIR": str(self.models_dir), "IZWI_DB_PATH": str(state / "izwi.sqlite3"),
            "IZWI_MEDIA_DIR": str(state / "media"), "IZWI_SPEECH_SPOOL_DIR": str(state / "spool"),
            "IZWI_LOG_FORMAT": "json", "NO_COLOR": "1",
            "RUST_LOG": os.environ.get("RUST_LOG", "info"),
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
                status, _, _, _ = http_call("GET", self.base + "/readyz", timeout=5)
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

    def log_contains(self, needle):
        try:
            return needle in self.log_path.read_text(errors="replace")
        except OSError:
            return False

    def verify_sha(self):
        logged = None
        for line in self.log_path.read_text(errors="replace").splitlines():
            match = re.search(r'"build_git_sha"\s*:\s*"([^"]*)"', line) or \
                re.search(r'build_git_sha="?([0-9a-f]{7,40}|unknown)', line)
            if match:
                logged = match.group(1)
        _, body, _, _ = http_call("GET", self.base + "/v1/health", timeout=30)
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

    def get(self, path, timeout=60):
        status, body, _, elapsed = http_call("GET", self.base + path, timeout=timeout)
        return status, parse(body), elapsed

    def post(self, path, body=None, timeout=600):
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"} if data is not None else {}
        status, raw, _, elapsed = http_call("POST", self.base + path, data, headers, timeout)
        return status, parse(raw), elapsed

    def speech(self, body, timeout):
        """POST /v1/audio/speech: (status, raw bytes, headers, seconds)."""
        return http_call("POST", self.base + "/v1/audio/speech", json.dumps(body).encode(),
                         {"Content-Type": "application/json"}, timeout)

    def load(self, timeout):
        return self.post(f"/v1/admin/models/{VARIANT}/load", None, timeout)

    def metrics(self):
        status, body, _ = self.get("/v1/metrics", timeout=30)
        return body if status == 200 and isinstance(body, dict) else {}

    def diagnostics(self, metrics=None):
        """The loaded model's family diagnostics (with its fast-path counters)."""
        metrics = metrics if metrics is not None else self.metrics()
        for model in metrics.get("models", []) or []:
            if model.get("variant_id") == VARIANT or model.get("variant") == VARIANT:
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
                       ("batch-runtime", "/v1/metrics/batch-runtime"),
                       ("admin", f"/v1/admin/models/{VARIANT}"), ("readyz", "/readyz")):
        try:
            status, body, _, _ = http_call("GET", base + path, timeout=30)
            payload = {"status": status, "captured_at": now(), "body": parse(body)}
        except Exception as error:  # noqa: BLE001
            payload = {"error": repr(error), "captured_at": now()}
        (folder / f"{label}-{name}.json").write_text(json.dumps(payload, indent=1) + "\n")


def counters(diagnostics):
    """The trunk's work counters and the fast-path counters, flat."""
    trunk = diagnostics.get("trunk") or {}
    flat = {f"trunk_{k}": trunk.get(k, 0) for k in ("spans", "span_rows", "steps", "step_rows")}
    for key, value in diagnostics.items():
        if key.endswith("_total") and isinstance(value, (int, float)):
            flat[key] = value
    return flat


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

    def wanted(self, name):
        return not ((self.only and name not in self.only) or name in self.skip)

    def run(self, name, fn, *args, force=False):
        if not (force or self.wanted(name)):
            return None
        log(f"case {name} ...")
        started = time.time()
        try:
            status, details = fn(*args)
        except CaseError as error:
            status, details = "fail", {"reason": str(error)}
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


# ------------------------------------------------------------------ cases

class Context:
    def __init__(self, args, server):
        self.args, self.server = args, server
        self.synthetic = args.synthetic
        self.timeout = args.request_timeout or REQUEST_TIMEOUT[args.backend]
        self.audio_dir = Path(args.out) / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.data = Path(args.data_dir)
        self.fox = self.data / "fox.wav"
        self.fox_text = (self.data / "fox.md").read_text().strip()
        self.second = self.data / "test.wav"
        self.second_text = (self.data / "test.md").read_text().strip()
        self.seed_tries = args.seed_tries or (40 if self.synthetic else 3)
        self.plain = None       # (seed, pcm) of the plain take
        self.reference_take = None  # (seed, pcm) of the inline reference take
        self.dialogue_lines = []    # (speaker, wav) of each line spoken alone
        self.saved = {}         # case -> [(wav path, spoken text)]
        self.counters_start = {}
        self.nonce = random.Random().randrange(1 << 30)

    def body(self, text=None, **fields):
        body = {"model": VARIANT, "response_format": "wav"}
        if text is not None:
            body["input"] = text
        if self.synthetic:
            body.update(SYNTHETIC_SAMPLING)
        body.update(fields)
        return body

    def keep(self, case, name, pcm, text=None, rate=SAMPLE_RATE):
        path = write_wav(self.audio_dir / f"{name}.wav", pcm, rate)
        if text is not None:
            self.saved.setdefault(case, []).append((path, text))
        return path

    def speak(self, base_seed, make_body):
        """The first seed from `base_seed` whose take ends within its budget:
        (seed, pcm, headers, seconds, seeds that ran out)."""
        ran_out = []
        for seed in range(base_seed, base_seed + self.seed_tries):
            status, raw, headers, elapsed = self.server.speech(make_body(seed), self.timeout)
            if status == 200:
                rate, _, pcm = wav_pcm(raw)
                if rate != SAMPLE_RATE:
                    raise CaseError(f"sample rate {rate}, expected {SAMPLE_RATE}")
                return seed, pcm, headers, elapsed, ran_out
            message = error_message(parse(raw))
            if not is_frame_limit(message):
                raise CaseError(f"seed {seed}: HTTP {status}: {message}")
            ran_out.append(seed)
        raise CaseError(f"no seed in {base_seed}..{base_seed + self.seed_tries - 1} ended "
                        "within its frame budget")


def header(headers, name):
    for key, value in headers.items():
        if key.lower() == name.lower():
            return value
    return None


def stream_speech(server, body, timeout, hang_up_after_first_audio=False):
    """POST /v1/audio/speech with an SSE stream: every event (arrival time,
    samples, offset into the PCM), the PCM, the time to first audio and the
    final event."""
    payload = dict(body, stream=True, stream_format="sse", response_format="pcm_i16")
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=timeout)
    started = time.time()
    connection.request("POST", "/v1/audio/speech", json.dumps(payload).encode(),
                       {"Content-Type": "application/json"})
    response = connection.getresponse()
    result = {"http": response.status, "events": [], "pcm": b"", "ttfa_s": None,
              "done": None, "failed": None, "hung_up": False}
    if response.status != 200:
        result["error"] = error_message(parse(response.read()))
        connection.close()
        return result
    pcm = bytearray()
    while True:
        line = response.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        event = parse(line[5:].strip())
        if not isinstance(event, dict):
            continue
        kind = event.get("event")
        stamp = round(time.time() - started, 4)
        record = {"t": stamp, "event": kind}
        if kind == "audio.chunk":
            chunk = base64.b64decode(event.get("audio_base64") or "")
            record.update(n=len(chunk) // 2, offset=len(pcm) // 2)
            pcm.extend(chunk)
            result["events"].append(record)
            if chunk and result["ttfa_s"] is None:
                result["ttfa_s"] = stamp
                if hang_up_after_first_audio:
                    result["hung_up"] = True
                    break
            continue
        result["events"].append(record)
        if kind == "audio.done":
            result["done"] = event
            break
        if kind in ("audio.failed", "error", "done"):
            result["failed"] = event
            break
    connection.close()
    result["elapsed_s"] = round(time.time() - started, 3)
    result["pcm"] = bytes(pcm)
    return result


def stream_error(result):
    return error_message((result.get("failed") or {}).get("error") or result.get("error"))


def stream_until_done(ctx, base_seed, make_body):
    """The first seed from `base_seed` whose streamed take ends every chunk
    within its budget: (seed, result)."""
    for seed in range(base_seed, base_seed + ctx.seed_tries):
        result = stream_speech(ctx.server, make_body(seed), ctx.timeout)
        if result.get("done"):
            return seed, result
        if not is_frame_limit(stream_error(result)):
            raise CaseError(f"seed {seed}: HTTP {result['http']}: {stream_error(result)}")
    raise CaseError(f"no seed in {base_seed}..{base_seed + ctx.seed_tries - 1} ended every "
                    "chunk within its frame budget")


def estimated_boundaries(result, segments):
    """Where a chunk session's chunks meet, in samples: the stream does not
    mark them, but every chunk starts with its own prefill, so they are taken
    as the `segments - 1` longest waits between audio events."""
    audio = [e for e in result["events"] if e["event"] == "audio.chunk" and e.get("n")]
    gaps = sorted(((b["t"] - a["t"], b["offset"]) for a, b in zip(audio, audio[1:])),
                  reverse=True)[:max(0, segments - 1)]
    return sorted(offset for _, offset in gaps)


def case_load(ctx):
    s = ctx.server
    status, info, _ = s.get(f"/v1/admin/models/{VARIANT}")
    model_status = info.get("status") if isinstance(info, dict) else None
    if model_status in ("not_downloaded", "downloading"):
        return "fail", {"status": model_status, "note": "run fetch.sh first"}
    status, body, elapsed = s.load(ctx.args.load_timeout)
    s.snapshot(ctx.args.out, "after-load")
    diagnostics = s.diagnostics()
    ctx.counters_start = counters(diagnostics)
    licence_logged = s.log_contains("Fish Audio Community License")
    details = {"http": status, "load_s": round(elapsed, 2), "response": body,
               "license_logged": licence_logged, "diagnostics": diagnostics}
    good = status == 200 and (licence_logged or ctx.synthetic)
    return ("pass" if good else "fail"), details


def case_plain(ctx):
    """A voice the seed sets; the same seed again is the same take."""
    seed, pcm, headers, elapsed, ran_out = ctx.speak(
        ctx.args.seed, lambda seed: ctx.body(TEXTS["plain"], seed=seed))
    _, again, _, _, _ = ctx.speak(seed, lambda s: ctx.body(TEXTS["plain"], seed=s))
    ctx.plain = (seed, pcm)
    ctx.keep("plain", "plain", pcm, TEXTS["plain"])
    stats = audio_stats(pcm)
    details = {"seed": seed, "seeds_out_of_budget": ran_out, "elapsed_s": round(elapsed, 3),
               "audio": stats, "repeatable": again == pcm,
               "x_tokens_generated": header(headers, "X-Tokens-Generated"),
               "x_stop_reason": header(headers, "X-Stop-Reason"),
               "x_rtf": header(headers, "X-RTF")}
    good = again == pcm and audible(stats, ctx.synthetic) and len(pcm) % (2 * SAMPLES_PER_FRAME) == 0
    return ("pass" if good else "fail"), details


def case_reference_inline(ctx):
    reference = b64_file(ctx.fox)
    seed, pcm, _, elapsed, ran_out = ctx.speak(ctx.args.seed + 100, lambda seed: ctx.body(
        TEXTS["plain"], seed=seed, reference_audio=reference, reference_text=ctx.fox_text))
    ctx.reference_take = (seed, pcm)
    ctx.keep("reference_inline", "reference_inline", pcm, TEXTS["plain"])
    # Without its transcript the reference is refused.
    status, raw, _, _ = ctx.server.speech(ctx.body(TEXTS["plain"], seed=seed,
                                                   reference_audio=reference), ctx.timeout)
    stats = audio_stats(pcm)
    details = {"seed": seed, "seeds_out_of_budget": ran_out, "elapsed_s": round(elapsed, 3),
               "audio": stats, "without_transcript_http": status,
               "without_transcript_message": error_message(parse(raw))}
    good = audible(stats, ctx.synthetic) and status == 400
    return ("pass" if good else "fail"), details


def case_reference_saved(ctx):
    status, voice, _ = ctx.server.post("/v1/voices", {
        "name": f"fd3-e2e-fox-{ctx.nonce}", "reference_text": ctx.fox_text,
        "audio_base64": b64_file(ctx.fox), "audio_mime_type": "audio/wav",
        "audio_filename": "fox.wav", "consent": True})
    if status not in (200, 201) or not isinstance(voice, dict) or not voice.get("id"):
        raise CaseError(f"saving the voice: HTTP {status}: {error_message(voice)}")
    seed, pcm, _, elapsed, ran_out = ctx.speak(ctx.args.seed + 100, lambda seed: ctx.body(
        TEXTS["plain"], seed=seed, saved_voice_id=voice["id"]))
    ctx.keep("reference_saved_voice", "reference_saved_voice", pcm, TEXTS["plain"])
    inline = ctx.reference_take
    stats = audio_stats(pcm)
    details = {"voice_id": voice["id"], "seed": seed, "seeds_out_of_budget": ran_out,
               "elapsed_s": round(elapsed, 3), "audio": stats,
               # The saved copy is canonicalised at upload; the same take is
               # expected but not required.
               "equals_inline_take": bool(inline and inline[0] == seed and inline[1] == pcm)}
    return ("pass" if audible(stats, ctx.synthetic) else "fail"), details


def case_tags(ctx):
    seed, pcm, _, elapsed, ran_out = ctx.speak(
        ctx.args.seed + 200, lambda seed: ctx.body(TEXTS["tags"], seed=seed))
    ctx.keep("tags", "tags", pcm, TEXTS["tags"])
    stats = audio_stats(pcm)
    details = {"seed": seed, "seeds_out_of_budget": ran_out, "elapsed_s": round(elapsed, 3),
               "audio": stats,
               "note": "lines open with [sighs] and [laughs]; lead_silence_s shows the "
                       "recipe's leading-silence quirk; listen to audio/tags.wav"}
    return ("pass" if audible(stats, ctx.synthetic) else "fail"), details


def dialogue_body(ctx, seed, lines):
    return ctx.body(None, seed=seed, script=[{"speaker": s, "text": t} for s, t in lines],
                    references=[
                        {"speaker": 0, "audio": b64_file(ctx.fox), "text": ctx.fox_text},
                        {"speaker": 1, "audio": b64_file(ctx.second), "text": ctx.second_text}])


def case_dialogue(ctx):
    """The script for two speakers with a reference each, streamed; then each
    line alone in the same request shape, so every speaker's audio can be
    heard, transcribed and compared with its reference (the stream does not
    mark where one speaker's chunk ends)."""
    seed, result = stream_until_done(ctx, ctx.args.seed + 300,
                                     lambda seed: dialogue_body(ctx, seed, DIALOGUE))
    timing = result["done"].get("timing") or {}
    ctx.keep("dialogue_two_speakers", "dialogue", result["pcm"],
             " ".join(text for _, text in DIALOGUE))
    stats = audio_stats(result["pcm"])
    lines = []
    for index, (speaker, text) in enumerate(DIALOGUE):
        line_seed, pcm, _, elapsed, _ = ctx.speak(
            seed, lambda s, line=(speaker, text): dialogue_body(ctx, s, [line]))
        path = ctx.keep("dialogue_two_speakers", f"dialogue_line{index}_speaker{speaker}", pcm,
                        text)
        ctx.dialogue_lines.append((speaker, path))
        lines.append({"line": index, "speaker": speaker, "seed": line_seed, "wav": path.name,
                      "elapsed_s": round(elapsed, 3), "audio": audio_stats(pcm)})
    details = {"seed": seed, "planned_segment_count": timing.get("planned_segment_count"),
               "audio": stats, "ttfa_s": result["ttfa_s"], "elapsed_s": result["elapsed_s"],
               "stop_reason": result["done"].get("stop_reason"), "lines": lines}
    good = audible(stats, ctx.synthetic) and all(audible(line["audio"], ctx.synthetic)
                                                  for line in lines)
    return ("pass" if good else "fail"), details


def case_long_text(ctx):
    """About five chunks, each carrying the history; listening samples
    around where the chunks meet; the WAV of the same seed equals the
    stream."""
    seed, result = stream_until_done(ctx, ctx.args.seed + 400,
                                     lambda seed: ctx.body(TEXTS["long"], seed=seed))
    ctx.keep("long_text_history", "long", result["pcm"], TEXTS["long"])
    timing = result["done"].get("timing") or {}
    planned = timing.get("planned_segment_count") or 1
    boundaries, half = [], int(1.5 * SAMPLE_RATE)
    for index, offset in enumerate(estimated_boundaries(result, planned)):
        name = f"long_boundary_{index + 1}"
        ctx.keep("long_text_boundaries", name,
                 result["pcm"][2 * max(0, offset - half):2 * (offset + half)])
        boundaries.append({"at_s": round(offset / SAMPLE_RATE, 3), "wav": f"{name}.wav"})
    status, raw, headers, _ = ctx.server.speech(ctx.body(TEXTS["long"], seed=seed), ctx.timeout)
    whole = wav_pcm(raw)[2] if status == 200 else b""
    stats = audio_stats(result["pcm"])
    details = {"seed": seed, "planned_segment_count": timing.get("planned_segment_count"),
               "token_unit": timing.get("token_unit"), "audio": stats,
               "estimated_boundaries": boundaries, "ttfa_s": result["ttfa_s"],
               "stream_elapsed_s": result["elapsed_s"], "non_stream_http": status,
               "non_stream_equals_stream": whole == result["pcm"],
               "x_tokens_generated": header(headers, "X-Tokens-Generated")}
    good = planned >= 4 and whole == result["pcm"] and audible(stats, ctx.synthetic)
    return ("pass" if good else "fail"), details


def case_stream_sse(ctx):
    """Time to first audio, and the streamed PCM equals the WAV of the same
    seed."""
    seed, wav = ctx.plain
    result = stream_speech(ctx.server, ctx.body(TEXTS["plain"], seed=seed), ctx.timeout)
    if result["http"] != 200 or not result.get("done"):
        raise CaseError(f"HTTP {result['http']}: {result.get('error') or result.get('failed')}")
    chunks = sum(1 for e in result["events"] if e["event"] == "audio.chunk")
    timing = result["done"].get("timing") or {}
    diff = max_abs_diff(result["pcm"], wav)
    details = {"seed": seed, "ttfa_s": result["ttfa_s"], "elapsed_s": result["elapsed_s"],
               "first_pcm_ms": timing.get("first_pcm_ms"), "audio_chunks": chunks,
               "stream_equals_wav": result["pcm"] == wav, "max_abs_diff_i16": diff,
               "stop_reason": result["done"].get("stop_reason"),
               "token_unit": timing.get("token_unit")}
    good = result["pcm"] == wav and chunks > 1 and result["done"].get("stop_reason") == "stop"
    return ("pass" if good else "fail"), details


def case_formats(ctx):
    seed, wav = ctx.plain
    results, problems = {}, []
    status, raw, _, _ = ctx.server.speech(ctx.body(TEXTS["plain"], seed=seed,
                                                   response_format="pcm_i16"), ctx.timeout)
    results["pcm_i16"] = {"http": status, "equals_wav": raw == wav}
    if raw != wav:
        problems.append("pcm_i16 is not the WAV's samples")
    status, raw, _, _ = ctx.server.speech(ctx.body(TEXTS["plain"], seed=seed,
                                                   response_format="pcm_f32"), ctx.timeout)
    floats = array.array("f")
    floats.frombytes(raw[:len(raw) - len(raw) % 4])
    if sys.byteorder != "little":
        floats.byteswap()
    ints = samples_i16(wav)
    worst = max((abs(max(-32768, min(32767, round(f * 32767))) - i) for f, i in zip(floats, ints)),
                default=0) if len(floats) == len(ints) else None
    results["pcm_f32"] = {"http": status, "samples": len(floats), "max_i16_delta": worst}
    if worst is None or worst > 1:
        problems.append("pcm_f32 does not round to the WAV's samples")
    status, raw, _, _ = ctx.server.speech(ctx.body(TEXTS["plain"], seed=seed,
                                                   response_format="mp3"), ctx.timeout)
    results["mp3"] = {"http": status, "message": error_message(parse(raw))}
    if status != 400:
        problems.append("mp3 without allow_format_fallback was not refused")
    status, raw, headers, _ = ctx.server.speech(ctx.body(
        TEXTS["plain"], seed=seed, response_format="mp3", allow_format_fallback=True), ctx.timeout)
    actual = header(headers, "X-Actual-Response-Format")
    fallback_pcm = wav_pcm(raw)[2] if status == 200 else b""
    results["mp3_fallback"] = {"http": status, "actual_format": actual,
                               "equals_wav": fallback_pcm == wav}
    if status != 200 or actual != "wav" or fallback_pcm != wav:
        problems.append("mp3 with allow_format_fallback did not return the WAV")
    return ("fail" if problems else "pass"), {"seed": seed, "formats": results,
                                              "problems": problems}


def create_take(ctx, base_seed, text, previous=()):
    """A durable /v1/text-to-speech take of `text` from the first seed that
    ends within its budget: (seed, ready record, seeds that ran out)."""
    ran_out = []
    for seed in range(base_seed, base_seed + ctx.seed_tries):
        body = {"model_id": VARIANT, "text": text, "seed": seed}
        if ctx.synthetic:
            body.update(SYNTHETIC_SAMPLING)
        if previous:
            body["previous_take_ids"] = list(previous)
        status, record, _ = ctx.server.post("/v1/text-to-speech", body)
        if status not in (200, 201, 202) or not isinstance(record, dict):
            raise CaseError(f"creating a take: HTTP {status}: {error_message(record)}")
        record = wait_record(ctx.server, record["id"], ctx.timeout)
        if record.get("processing_status") == "ready":
            return seed, record, ran_out
        message = record.get("processing_error") or ""
        if not is_frame_limit(message):
            raise CaseError(f"take {record['id']} {record.get('processing_status')}: {message}")
        ran_out.append(seed)
    raise CaseError("no seed ended the take within its frame budget")


def wait_record(server, record_id, timeout):
    deadline = time.time() + timeout
    while True:
        status, record, _ = server.get(f"/v1/text-to-speech/{urllib.parse.quote(record_id)}")
        state = record.get("processing_status") if isinstance(record, dict) else None
        if state in ("ready", "failed", "cancelled"):
            return record
        if time.time() > deadline:
            raise CaseError(f"record {record_id} still {state} after {timeout} s")
        time.sleep(0.25)


def record_pcm(server, record_id):
    status, raw, _, _ = http_call(
        "GET", f"{server.base}/v1/text-to-speech/{urllib.parse.quote(record_id)}/audio",
        timeout=120)
    if status != 200:
        raise CaseError(f"record {record_id} audio: HTTP {status}")
    return wav_pcm(raw)[2]


def case_take_edit(ctx):
    """A take, then an edit continuing it through previous_take_ids: the
    edit records its parent, and the earlier take reached its prompt (the
    same text and seed alone speak otherwise)."""
    seed1, first, ran1 = create_take(ctx, ctx.args.seed + 500, TEXTS["take_1"])
    first_pcm = record_pcm(ctx.server, first["id"])
    ctx.keep("take_edit", "take_1", first_pcm, TEXTS["take_1"])
    seed2, edit, ran2 = create_take(ctx, ctx.args.seed + 600, TEXTS["take_2"], [first["id"]])
    edit_pcm = record_pcm(ctx.server, edit["id"])
    ctx.keep("take_edit", "take_2_edit", edit_pcm, TEXTS["take_2"])
    status, raw, _, _ = ctx.server.speech(ctx.body(TEXTS["take_2"], seed=seed2), ctx.timeout)
    alone = wav_pcm(raw)[2] if status == 200 else None
    problems = []
    if first.get("editable") is not True:
        problems.append("the first take is not editable")
    if first.get("seed") != seed1 or first.get("references") != []:
        problems.append("the first take does not report its seed and seeded voice")
    if edit.get("parent_take_id") != first["id"]:
        problems.append("the edit does not name its parent")
    if alone is not None and alone == edit_pcm:
        problems.append("the edit is the same audio as the text alone: the take did not "
                        "reach its prompt")
    details = {"first": {"id": first["id"], "seed": seed1, "seeds_out_of_budget": ran1,
                         "editable": first.get("editable"), "audio": audio_stats(first_pcm)},
               "edit": {"id": edit["id"], "seed": seed2, "seeds_out_of_budget": ran2,
                        "parent_take_id": edit.get("parent_take_id"),
                        "editable": edit.get("editable"), "audio": audio_stats(edit_pcm)},
               "alone_http": status, "edit_differs_from_text_alone":
                   None if alone is None else alone != edit_pcm, "problems": problems}
    return ("fail" if problems else "pass"), details


def after_cancel_check(ctx):
    """The plain take again: cancellation left nothing behind."""
    seed, pcm = ctx.plain
    status, raw, _, _ = ctx.server.speech(ctx.body(TEXTS["plain"], seed=seed), ctx.timeout)
    return status == 200 and wav_pcm(raw)[2] == pcm


CANCEL_STAGES = ("preparation", "prefill", "decode")


def cancel_record_at(ctx, stage):
    """A durable take with a 58 s reference the cache has not seen (and
    the long text), cancelled when the model reaches `stage`:
      preparation  the engine holds the request and the trunk has run no
                   span yet (the reference is being encoded)
      prefill      the trunk ran a span and no decode step
      decode       five decode steps ran
    Returns the stage reached at cancel time and the details."""
    s = ctx.server
    s.wait_idle(ctx.timeout)
    clip = unique_reference(ctx.second, 58, ctx.audio_dir / f"cancel_{stage}_reference.wav",
                            ctx.nonce + CANCEL_STAGES.index(stage))
    before = counters(s.diagnostics())
    body = {"model_id": VARIANT, "text": TEXTS["long"], "seed": ctx.args.seed + 700,
            "reference_audio": b64_file(clip),
            "reference_text": " ".join([ctx.second_text] * 2)}
    if ctx.synthetic:
        body.update(SYNTHETIC_SAMPLING)
    status, record, _ = s.post("/v1/text-to-speech", body)
    if status not in (200, 201, 202) or not isinstance(record, dict):
        raise CaseError(f"HTTP {status}: {error_message(record)}")
    record_id = record["id"]
    deadline, reached = time.time() + ctx.timeout, None
    while time.time() < deadline:
        metrics = s.metrics()
        now_counters = counters(s.diagnostics(metrics))
        spans = now_counters["trunk_spans"] - before["trunk_spans"]
        steps = now_counters["trunk_steps"] - before["trunk_steps"]
        busy = not s.idle(metrics)
        if stage == "preparation" and busy and spans == 0:
            reached = "preparation"
            break
        if stage == "prefill" and spans > 0:
            reached = "prefill" if steps == 0 else "decode"
            break
        if stage == "decode" and steps >= 5:
            reached = "decode"
            break
        if spans > 0 and stage == "preparation":
            reached = "prefill" if steps == 0 else "decode"
            break
        _, current, _ = s.get(f"/v1/text-to-speech/{record_id}")
        if isinstance(current, dict) and current.get("processing_status") in ("ready", "failed"):
            reached = "finished"
            break
        time.sleep(0.005 if stage == "preparation" else 0.02)
    cancel_at = time.time()
    status, body, cancel_s = s.post(f"/v1/text-to-speech/{record_id}/cancel")
    s.wait_idle(600)
    idle_s = round(time.time() - cancel_at, 3)
    final = wait_record(s, record_id, 600)
    after = counters(s.diagnostics())
    clean = after_cancel_check(ctx)
    details = {"target": stage, "reached": reached, "cancel_http": status,
               "cancel_message": None if status == 200 else error_message(body),
               "cancel_request_s": round(cancel_s, 3), "cancel_to_idle_s": idle_s,
               "final_status": final.get("processing_status"),
               "final_error": final.get("processing_error"),
               "plain_take_unchanged_after": clean,
               "counters_delta": {k: after.get(k, 0) - before.get(k, 0)
                                  for k in ("trunk_spans", "trunk_steps")}}
    if reached != stage:
        return "inconclusive", dict(details, note=f"the request was at {reached} when "
                                    f"cancelled, not {stage}; a faster device can pass a stage "
                                    "between two polls")
    good = status == 200 and idle_s <= ctx.args.cancel_idle_max_s and clean \
        and final.get("processing_status") in ("failed", "cancelled")
    return ("pass" if good else "fail"), details


def case_cancel_after_first_audio(ctx):
    """A client that hangs up once audio has streamed: the codec has run, and
    decode and codec quanta are interleaving when the stop lands."""
    s = ctx.server
    s.wait_idle(ctx.timeout)
    result = stream_speech(s, ctx.body(TEXTS["long"], seed=ctx.args.seed + 800), ctx.timeout,
                           hang_up_after_first_audio=True)
    hung_up = time.time()
    if not result["hung_up"]:
        raise CaseError(f"no audio before the stream ended: {result.get('failed') or result.get('error')}")
    s.wait_idle(600)
    idle_s = round(time.time() - hung_up, 3)
    clean = after_cancel_check(ctx)
    metrics = s.metrics()
    details = {"ttfa_s": result["ttfa_s"], "disconnect_to_idle_s": idle_s,
               "requests_cancelled": metrics.get("requests_cancelled"),
               "plain_take_unchanged_after": clean}
    good = idle_s <= ctx.args.cancel_idle_max_s and clean
    return ("pass" if good else "fail"), details


def case_soak(ctx):
    """Two clients for --soak-minutes, mixing a seeded voice, a cloned one
    and streams; untracked device memory must stay flat."""
    s = ctx.server
    s.wait_idle(ctx.timeout)
    before = s.metrics()
    s.snapshot(ctx.args.out, "soak-before")
    untracked0 = (before.get("coordinator") or {}).get("untracked_device_memory_bytes", 0)
    reference = b64_file(ctx.fox)
    stop_at = time.time() + ctx.args.soak_minutes * 60
    failures, samples, lock = [], [], threading.Lock()
    counter = [0]

    def worker(slot):
        while time.time() < stop_at:
            with lock:
                counter[0] += 1
                index = counter[0]
            seed = ctx.args.seed + 1000 + index
            kind = ("plain", "clone", "stream")[index % 3]
            fields = {"seed": seed}
            if kind == "clone":
                fields.update(reference_audio=reference, reference_text=ctx.fox_text)
            body = ctx.body(TEXTS["plain"] if index % 2 else TEXTS["take_2"], **fields)
            started = time.time()
            if kind == "stream":
                result = stream_speech(s, body, ctx.timeout)
                status = 200 if result.get("done") else result["http"]
                message = None if result.get("done") else stream_error(result)
            else:
                status, raw, _, _ = s.speech(body, ctx.timeout)
                message = None if status == 200 else error_message(parse(raw))
            metrics = s.metrics()
            with lock:
                samples.append({"i": index, "kind": kind, "http": status,
                                "elapsed_s": round(time.time() - started, 2),
                                "untracked": (metrics.get("coordinator") or {}).get(
                                    "untracked_device_memory_bytes")})
                if message and not (ctx.synthetic and is_frame_limit(message)):
                    failures.append({"i": index, "kind": kind, "http": status,
                                     "message": message})

    threads = [threading.Thread(target=worker, args=(slot,)) for slot in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    s.wait_idle(ctx.timeout)
    after = s.metrics()
    s.snapshot(ctx.args.out, "soak-after")
    untracked1 = (after.get("coordinator") or {}).get("untracked_device_memory_bytes", 0)
    growth = untracked1 - untracked0
    good = not failures and growth <= ctx.args.untracked_growth_mib * MIB and samples
    return ("pass" if good else "fail"), {
        "minutes": ctx.args.soak_minutes, "requests": len(samples), "failures": failures[:5],
        "untracked_before": untracked0, "untracked_after": untracked1, "growth_bytes": growth,
        "max_untracked": max([x["untracked"] or 0 for x in samples] or [0]),
        "coordinator_after": after.get("coordinator"),
        "note": "untracked device memory is 0 on the CPU by construction"}


# Counters each backend must have moved by the end of the run.
FAST_PATHS = {
    "metal": ["resident_frames_total", "resident_readbacks_total"],
    "cuda": ["resident_frames_total", "resident_readbacks_total", "step_row_gemv_calls_total",
             "chunked_deltanet_spans_total", "grouped_moe_spans_total"],
}


def case_fast_paths(ctx):
    if ctx.args.backend == "cpu":
        return "skip", {"reason": "the CPU has no device fast paths; its Q8_0 checks are in "
                                  "the load diagnostics"}
    diagnostics = ctx.server.diagnostics()
    end = counters(diagnostics)
    delta = {k: end.get(k, 0) - ctx.counters_start.get(k, 0) for k in end}
    missing = [k for k in FAST_PATHS[ctx.args.backend] if delta.get(k, 0) <= 0]
    problems = [f"{k} did not move" for k in missing]
    if delta.get("resident_host_fallbacks_total", 0) > 0:
        problems.append("resident sampling fell back to the host")
    if ctx.args.backend == "cuda" and delta.get("step_row_loop_calls_total", 0) > 0:
        problems.append("steps ran one GEMV per row (the row-invariant GEMV check failed?)")
    frames = delta.get("resident_frames_total", 0)
    readbacks = delta.get("resident_readbacks_total", 0)
    if frames and readbacks > 2 * frames:
        problems.append(f"{readbacks} readbacks for {frames} frames (at most 2 a frame)")
    return ("fail" if problems else "pass"), {
        "delta": delta, "problems": problems,
        "load_checks": diagnostics.get("metal_checks") or diagnostics.get("cuda_checks")}


def transcribe(server, model, path, timeout=900):
    boundary = "----izwi-fd3-" + os.urandom(8).hex()
    parts = []
    for name, value in (("model", model), ("response_format", "json")):
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
                     f"{value}\r\n".encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
                 f'filename="{Path(path).name}"\r\nContent-Type: audio/wav\r\n\r\n'.encode()
                 + Path(path).read_bytes() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    status, raw, _, _ = http_call("POST", server.base + "/v1/audio/transcriptions",
                                  b"".join(parts),
                                  {"Content-Type": f"multipart/form-data; boundary={boundary}"},
                                  timeout)
    payload = parse(raw)
    if status != 200:
        raise CaseError(f"transcribing {Path(path).name}: HTTP {status}: {error_message(payload)}")
    return payload.get("text", "")


def words(text):
    text = re.sub(r"<\|[^|>]*\|>|\[[^\]]*\]", " ", text or "")
    return re.findall(r"[\w']+", text.lower())


def wer(reference, hypothesis):
    ref, hyp = words(reference), words(hypothesis)
    row = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        previous, row[0] = row[0], i
        for j in range(1, len(hyp) + 1):
            current = min(row[j] + 1, row[j - 1] + 1, previous + (ref[i - 1] != hyp[j - 1]))
            previous, row[j] = row[j], current
    return row[len(hyp)] / max(1, len(ref))


def case_asr_round_trip(ctx):
    """Every kept take transcribed by --asr-model on the same server (the
    dialogue's lines one by one), then each dialogue line's speaker
    similarity to both references."""
    if ctx.synthetic:
        return "skip", {"reason": "the synthetic checkpoint's audio is not speech"}
    if not ctx.args.asr_model:
        return "skip", {"reason": "pass --asr-model (an ASR model izwi has downloaded, e.g. "
                                  "Whisper-Large-v3-Turbo) to transcribe the takes"}
    s = ctx.server
    status, body, _ = s.post(f"/v1/admin/models/{urllib.parse.quote(ctx.args.asr_model)}/load",
                             None, ctx.args.load_timeout)
    if status != 200:
        raise CaseError(f"loading {ctx.args.asr_model}: HTTP {status}: {error_message(body)}")
    rows, worst = [], 0.0
    for case, items in ctx.saved.items():
        for path, text in items:
            heard = transcribe(s, ctx.args.asr_model, path)
            score = round(wer(text, heard), 3)
            worst = max(worst, score)
            rows.append({"case": case, "wav": path.name, "wer": score, "heard": heard[:300]})
    s.post(f"/v1/admin/models/{urllib.parse.quote(ctx.args.asr_model)}/unload")
    similarity = dialogue_similarity(ctx)
    good = rows and worst <= ctx.args.max_wer and similarity.get("own_voice_closer") is not False
    return ("pass" if good else "fail"), {"asr_model": ctx.args.asr_model,
                                          "max_wer": ctx.args.max_wer, "worst_wer": worst,
                                          "takes": rows, "dialogue_similarity": similarity}


def dialogue_similarity(ctx):
    """Each dialogue line's ECAPA similarity to both references (eval-mini's
    scorer, when speechbrain is installed): its own speaker's must be the
    closer."""
    if not ctx.dialogue_lines:
        return {"measured": False, "reason": "no dialogue lines"}
    try:
        sys.dont_write_bytecode = True
        spec = importlib.util.spec_from_file_location("eval_mini", HERE / "eval-mini.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        score = module.EcapaSimilarity()
    except Exception as error:  # noqa: BLE001
        return {"measured": False, "reason": f"no speaker similarity: {error}"}
    references = {0: ctx.fox, 1: ctx.second}
    lines, closer = [], True
    for speaker, path in ctx.dialogue_lines:
        own = score(path, references[speaker])
        other = score(path, references[1 - speaker])
        closer &= own > other
        lines.append({"wav": path.name, "speaker": speaker, "sim_own": round(own, 4),
                      "sim_other": round(other, 4)})
    return {"measured": True, "own_voice_closer": closer, "lines": lines}


def case_unload_reload(ctx):
    s = ctx.server
    seed, reference = ctx.plain
    status, body, unload_s = s.post(f"/v1/admin/models/{VARIANT}/unload")
    _, info, _ = s.get(f"/v1/admin/models/{VARIANT}")
    unloaded_status = info.get("status") if isinstance(info, dict) else None
    status2, body2, load_s = s.load(ctx.args.load_timeout)
    status3, raw, _, _ = s.speech(ctx.body(TEXTS["plain"], seed=seed), ctx.timeout)
    again = wav_pcm(raw)[2] if status3 == 200 else None
    good = status == 200 and unloaded_status != "ready" and status2 == 200 and again == reference
    return ("pass" if good else "fail"), {
        "unload_http": status, "unload_s": round(unload_s, 2),
        "status_after_unload": unloaded_status, "reload_http": status2,
        "reload_s": round(load_s, 2), "plain_take_equal_after_reload": again == reference}


def case_oom_refusal(args, build_sha):
    budget = args.undersized_budget_mib * MIB
    env = {BUDGET_ENV[args.backend]: str(budget)}
    server = Server(args.binary, args.backend, args.models_dir, args.out, "undersized",
                    build_sha, env_extra=dict(mode_env(args.synthetic), **env))
    server.start(args.ready_timeout)
    try:
        status, body, elapsed = server.load(args.load_timeout)
        message = error_message(body)
        ready, _, _ = server.get("/readyz")
        server.snapshot(args.out, "undersized")
        memory_words = any(w in message.lower() for w in ("memory", "gib", "fit", "budget"))
        good = status >= 400 and elapsed <= 120 and ready == 200 and memory_words
        return ("pass" if good else "fail"), {
            "budget_env": env, "http": status, "refused_in_s": round(elapsed, 2),
            "message": message, "ready_after": ready, "mentions_memory": memory_words}
    finally:
        server.stop()


CASES = [
    ("plain", case_plain, "a voice the seed sets; the same seed twice is the same take"),
    ("reference_inline", case_reference_inline,
     "data/fox.wav with its transcript inline; without the transcript, 400"),
    ("reference_saved_voice", case_reference_saved, "the same reference as a saved voice"),
    ("tags", case_tags, "performance tags, two of them leading their lines"),
    ("dialogue_two_speakers", case_dialogue,
     "a three-line script for two speakers with a reference each; then each line alone"),
    ("long_text_history", case_long_text,
     "about five chunks with their history; samples where chunks meet; stream == WAV"),
    ("stream_sse", case_stream_sse, "time to first audio; streamed PCM == the WAV, same seed"),
    ("formats", case_formats, "wav, pcm_i16, pcm_f32; mp3 refused, and with fallback a WAV"),
    ("take_edit", case_take_edit, "a durable take, then an edit through previous_take_ids"),
    ("cancel_mid_preparation", lambda ctx: cancel_record_at(ctx, "preparation"),
     "cancel while the reference is encoded"),
    ("cancel_mid_prefill", lambda ctx: cancel_record_at(ctx, "prefill"),
     "cancel between prefill spans"),
    ("cancel_mid_decode", lambda ctx: cancel_record_at(ctx, "decode"),
     "cancel after five decode steps"),
    ("cancel_after_first_audio", case_cancel_after_first_audio,
     "a streaming client hangs up once audio arrived (decode and codec interleaving)"),
    ("soak_untracked_memory", case_soak, "two clients for --soak-minutes; untracked memory flat"),
    ("fast_paths", case_fast_paths, "the backend's fast-path counters moved; no host fallback"),
    ("asr_round_trip", case_asr_round_trip, "every take through --asr-model, WER per take"),
    ("unload_reload", case_unload_reload, "unload, reload, the plain take unchanged"),
]
OOM_CASE = ("oom_refusal_undersized_budget",
            "a second server with an undersized memory budget refuses the load quickly")


# ------------------------------------------------------------ subcommands

def read_build_info(out_dir, explicit_sha=None):
    path = Path(out_dir) / "build-info.json"
    info = json.loads(path.read_text()) if path.is_file() else {}
    return info, explicit_sha or info.get("git_sha")


def apply_mode_defaults(args):
    if args.soak_minutes is None:
        args.soak_minutes = 1.0 if args.synthetic else 30.0
    if args.undersized_budget_mib is None:
        args.undersized_budget_mib = 1 if args.synthetic else 8192
    if args.cancel_idle_max_s is None:
        args.cancel_idle_max_s = 30.0 if args.synthetic else 10.0


def cmd_plan(args):
    apply_mode_defaults(args)
    only = set(args.only.split(",")) if args.only else None
    skip = set(args.skip.split(",")) if args.skip else set()
    print(f"E2E plan ({'synthetic checkpoint' if args.synthetic else 'published weights'}, "
          f"backend {args.backend}):")
    for name, _, what in [("load", None, "load the model; the licence is logged")] + CASES \
            + [(OOM_CASE[0], None, OOM_CASE[1])]:
        mark = "skip" if (only and name not in only) or name in skip else "run "
        print(f"  {mark} {name:30} {what}")
    print(f"  soak {args.soak_minutes:g} min; undersized budget {args.undersized_budget_mib} MiB; "
          f"cancel to idle within {args.cancel_idle_max_s:g} s")
    return 0


def cmd_run(args):
    apply_mode_defaults(args)
    args.out = Path(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    build, sha = read_build_info(args.out, args.expect_sha)
    if not sha and not args.no_sha_check:
        raise SystemExit("no expected build sha: run build.sh first or pass --expect-sha")
    if args.synthetic and not build.get("synthetic", args.no_sha_check):
        raise SystemExit("--synthetic needs a build.sh --synthetic binary")
    meta = {"backend": args.backend, "variant": VARIANT, "git_sha": sha,
            "synthetic": args.synthetic,
            "env": {k: v for k, v in os.environ.items() if k.startswith("IZWI_FISH_DRAMA3_")}}
    merge_section(args.out / "report.json", "build", build, meta)
    cases = Cases(args.out / "report.json", meta,
                  only=args.only and set(args.only.split(",")),
                  skip=args.skip.split(",") if args.skip else ())
    expected = None if args.no_sha_check else sha
    server = Server(args.binary, args.backend, args.models_dir, args.out, "main", expected,
                    env_extra=mode_env(args.synthetic))
    server.start(args.ready_timeout)
    cases.section["server"] = {"base": server.base, "build_git_sha": server.build_sha,
                               "env": server.env_used,
                               "health_runtime": (server.health or {}).get("runtime")
                               if isinstance(server.health, dict) else None}
    server.snapshot(args.out, "start")
    ctx = Context(args, server)
    try:
        loaded = cases.run("load", case_load, ctx)
        if loaded and loaded["status"] != "pass":
            log("the model did not load; skipping the cases that need it")
        else:
            needs_plain = {"stream_sse", "formats", "cancel_mid_preparation",
                           "cancel_mid_prefill", "cancel_mid_decode", "cancel_after_first_audio",
                           "unload_reload"}
            for name, fn, _ in CASES:
                if name in needs_plain and ctx.plain is None and cases.wanted(name):
                    # Their reference take, even under --only.
                    cases.run("plain", case_plain, ctx, force=True)
                    if ctx.plain is None:
                        break
                if name == "plain" and ctx.plain is not None:
                    continue
                cases.run(name, fn, ctx)
    finally:
        server.snapshot(args.out, "end")
        server.stop()
    cases.run(OOM_CASE[0], case_oom_refusal, args, expected)
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


def streaming_probe(args, server, concurrency, extra):
    """`concurrency` streams at once, `--probe-rounds` times: TTFA, RTF (send
    to last audio over audio length) and generated frames/s, as the card's
    serving table measures them."""
    ttfa, rtf, frames, walls, errors = [], [], 0, 0.0, []
    timeout = args.request_timeout or REQUEST_TIMEOUT[args.backend]
    for round_index in range(args.probe_rounds):
        results = [None] * concurrency

        def one(slot):
            seed = args.seed + 5000 + round_index * 100 + slot
            body = dict(extra, model=VARIANT, input=TEXTS["take_1"] + " " + TEXTS["take_2"],
                        seed=seed)
            started = time.time()
            result = stream_speech(server, body, timeout)
            last = max((e["t"] for e in result["events"] if e["event"] == "audio.chunk"),
                       default=None)
            results[slot] = (result, started, last)

        threads = [threading.Thread(target=one, args=(slot,)) for slot in range(concurrency)]
        began = time.time()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        walls += time.time() - began
        for result, _, last in results:
            if not result.get("done"):
                errors.append(stream_error(result))
                continue
            seconds = len(result["pcm"]) / 2 / SAMPLE_RATE
            frames += len(result["pcm"]) // 2 // SAMPLES_PER_FRAME
            ttfa.append(result["ttfa_s"])
            if seconds and last is not None:
                rtf.append(last / seconds)
    card = CARD_H200.get(concurrency)
    return {"concurrency": concurrency, "rounds": args.probe_rounds,
            "ttfa_s": summarize(ttfa), "rtf": summarize(rtf),
            "frames_per_s": round(frames / walls, 2) if walls else None,
            "errors": errors[:5], "error_count": len(errors),
            "card_h200": None if card is None else {
                "frames_per_s": card[0], "ttfa_p50_ms": card[1], "ttfa_p95_ms": card[2],
                "rtf_mean": card[3]}}


def chat_itl(args, server, extra):
    code, body, _ = server.post(f"/v1/admin/models/{urllib.parse.quote(args.chat_model)}/load",
                                None, args.load_timeout)
    if code != 200:
        return {"status": "error", "message": f"chat model load: {code} {error_message(body)}"}
    prompt = "Write a detailed, plain-prose story about a lighthouse keeper. Do not stop early."
    baseline = []
    for _ in range(3):
        baseline += chat_stream(server.base, args.chat_model, prompt, 128)
    timeout = args.request_timeout or REQUEST_TIMEOUT[args.backend]
    holder = {}

    def speak():
        body = dict(extra, model=VARIANT, input=TEXTS["long"], seed=args.seed + 9000)
        holder["result"] = stream_speech(server, body, timeout)

    thread = threading.Thread(target=speak)
    thread.start()
    time.sleep(args.itl_warmup_s)
    during = []
    for _ in range(3):
        if not thread.is_alive():
            break
        during += chat_stream(server.base, args.chat_model, prompt, 128)
    thread.join()
    base, busy = summarize(baseline), summarize(during)
    ratio = busy["p50"] / base["p50"] if base and busy and base["p50"] else None
    return {"status": "measured" if busy else "too_short",
            "chat_model": args.chat_model, "baseline_itl_s": base,
            "beside_drama3_itl_s": busy, "p50_ratio": ratio,
            "note": None if busy else "the long take ended before chat ran beside it; "
                                      "lower --itl-warmup-s"}


def cmd_load(args):
    args.out = Path(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    build, sha = read_build_info(args.out, args.expect_sha)
    meta = {"backend": args.backend, "variant": VARIANT, "git_sha": sha,
            "synthetic": args.synthetic}
    section = {"started_at": now(), "points": [], "probe": [], "chat_itl": None,
               "voice": args.voice}
    concurrency = [int(c) for c in args.concurrency.split(",")]
    extra = dict(SYNTHETIC_SAMPLING) if args.synthetic else {}
    if args.voice == "clone":
        data = Path(args.data_dir)
        extra.update(reference_audio=b64_file(data / "fox.wav"),
                     reference_text=(data / "fox.md").read_text().strip())
    server = Server(args.binary, args.backend, args.models_dir, args.out, "load",
                    None if args.no_sha_check else sha,
                    env_extra=mode_env(args.synthetic))
    server.start(args.ready_timeout)
    status = 0
    try:
        code, body, load_s = server.load(args.load_timeout)
        if code != 200:
            raise RuntimeError(f"load failed: {code} {error_message(body)}")
        section["load_s"] = round(load_s, 2)
        server.snapshot(args.out, "load-start")
        for c in concurrency:
            output = args.out / f"load-c{c}.json"
            cmd = [sys.executable, str(REPO / "scripts/bench/run-audio-load.py"), "--task", "tts",
                   "--model", VARIANT, "--base-url", server.base,
                   "--num-requests", str(max(args.min_requests, 2 * c)),
                   "--request-rate", args.request_rate, "--burstiness", "1",
                   "--max-concurrency", str(c), "--seed", str(args.seed),
                   "--timeout", str(args.request_timeout or REQUEST_TIMEOUT[args.backend]),
                   "--min-words", str(args.min_words), "--max-words", str(args.max_words),
                   "--tts-tolerance", str(args.tts_tolerance), "--require-parity",
                   "--output", str(output)]
            if extra:
                cmd += ["--tts-extra", json.dumps(extra)]
            log(f"load point c={c}: run-audio-load.py --max-concurrency {c} "
                f"--num-requests {max(args.min_requests, 2 * c)} --seed {args.seed}")
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
            if result.returncode != 0:
                status = 1
            log(f"streaming probe c={c}")
            section["probe"].append(streaming_probe(args, server, c, extra))
            server.snapshot(args.out, f"load-c{c}")
            merge_section(args.out / "report.json", "load", section, meta)
        if args.chat_model:
            section["chat_itl"] = chat_itl(args, server, extra)
    finally:
        server.snapshot(args.out, "load-end")
        server.stop()
    section["ended_at"] = now()
    section["summary"] = {"parity": "pass" if all(p["parity"] == "pass"
                                                  for p in section["points"]) else "fail"}
    merge_section(args.out / "report.json", "load", section, meta)
    return status


def cmd_merge(args):
    merge_section(args.report, args.section, json.loads(Path(args.file).read_text()))
    return 0


def cmd_snapshot(args):
    snapshot(args.base_url, args.out, args.label)
    return 0


def add_server_args(parser, required=True):
    parser.add_argument("--backend", choices=("cpu", "metal", "cuda"), required=True)
    parser.add_argument("--binary", required=required, help="izwi-server built by build.sh")
    parser.add_argument("--models-dir", required=required, help="IZWI_MODELS_DIR")
    parser.add_argument("--out", required=required, help="the backend's report directory")
    parser.add_argument("--data-dir", default=str(REPO / "data"))
    parser.add_argument("--expect-sha", help="default: build-info.json's git_sha")
    parser.add_argument("--no-sha-check", action="store_true")
    parser.add_argument("--synthetic", action="store_true",
                        help="the server serves the synthetic checkpoint (build.sh --synthetic)")
    parser.add_argument("--seed", type=int, default=1, help="the first sampling seed")
    parser.add_argument("--ready-timeout", type=int, default=600)
    parser.add_argument("--load-timeout", type=int, default=3600)
    parser.add_argument("--request-timeout", type=int, default=0)


def add_run_args(parser):
    parser.add_argument("--only", help="comma-separated case names")
    parser.add_argument("--skip", help="comma-separated case names")
    parser.add_argument("--soak-minutes", type=float, default=None,
                        help="the soak's length (default 30; 1 with --synthetic)")
    parser.add_argument("--untracked-growth-mib", type=int, default=256)
    parser.add_argument("--cancel-idle-max-s", type=float, default=None,
                        help="cancel to idle within this (default 10 s; 30 with --synthetic, "
                             "a debug build)")
    parser.add_argument("--undersized-budget-mib", type=int, default=None,
                        help="the OOM case's memory budget (default 8192; 1 with --synthetic)")
    parser.add_argument("--seed-tries", type=int, default=0,
                        help="seeds to try when a take runs out of frames (default 3; 40 "
                             "with --synthetic)")
    parser.add_argument("--asr-model", help="an ASR model to transcribe the takes with")
    parser.add_argument("--max-wer", type=float, default=0.3,
                        help="the worst take's WER asr_round_trip accepts")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="the E2E matrix")
    add_server_args(run)
    add_run_args(run)
    plan = sub.add_parser("plan", help="print the E2E cases (a dry run); takes run's options")
    add_server_args(plan, required=False)
    add_run_args(plan)
    load = sub.add_parser("load", help="run-audio-load.py sweep, streaming probe and chat ITL")
    add_server_args(load)
    load.add_argument("--concurrency", required=True, help="e.g. 1,2")
    load.add_argument("--request-rate", default="0.5",
                      help="Poisson arrivals per second (staggered); 'inf' for a burst")
    load.add_argument("--min-requests", type=int, default=16)
    load.add_argument("--min-words", type=int, default=4)
    load.add_argument("--max-words", type=int, default=60)
    load.add_argument("--tts-tolerance", type=float, default=0.0,
                      help="parity tolerance of full scale (0: bit for bit)")
    load.add_argument("--voice", choices=("seeded", "clone"), default="seeded",
                      help="a voice each request's seed sets, or data/fox.wav cloned")
    load.add_argument("--probe-rounds", type=int, default=2)
    load.add_argument("--chat-model", help="a small chat model id for ITL beside Drama 3")
    load.add_argument("--itl-warmup-s", type=float, default=5.0)
    merge = sub.add_parser("merge", help="put a JSON file into a report section")
    merge.add_argument("--report", required=True)
    merge.add_argument("--section", required=True)
    merge.add_argument("--file", required=True)
    snap = sub.add_parser("snapshot", help="save metrics, health and admin snapshots")
    snap.add_argument("--base-url", required=True)
    snap.add_argument("--out", required=True)
    snap.add_argument("--label", default="manual")
    args = parser.parse_args(argv)
    return {"run": cmd_run, "plan": cmd_plan, "load": cmd_load, "merge": cmd_merge,
            "snapshot": cmd_snapshot}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
