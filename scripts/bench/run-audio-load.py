#!/usr/bin/env python3
"""Open-loop audio serving load for izwi: transcription or speech synthesis.

Sends `/v1/audio/transcriptions` (`--task asr`) or `/v1/audio/speech`
(`--task tts`) requests with Poisson (or gamma) arrivals at one or more
request rates, optionally capped at a maximum concurrency.

Every request has its own input, so batching sees mixed lengths:
- ASR clips are cut from the given WAV files at seeded random offsets, with
  lengths uniform in [--min-secs, --max-secs];
- TTS inputs are seeded random sentences with word counts uniform in
  [--min-words, --max-words].

Before the load, every input runs alone once. Under load each response is
compared with its solo output: ASR text must match exactly (a normalized
match, ignoring case and punctuation, is reported too); TTS audio must have
the same length and stay within --tts-tolerance of the solo samples. A
batching bug that changes one request's output then shows up as a parity
failure, not only as latency. A model that does not repeat its own solo
output (the first input runs twice) is reported as such and not held to
parity. With --require-parity a failure, or a model that does not repeat,
exits 3.

Results are JSON (schema "izwi.audio_load", version 1) with git, host and
server metadata, the solo pass, and one report per sweep point: latency and
real-time-factor distributions, request and audio throughput, errors and
parity. Standard library only: runs with plain python3. See
docs/dev/SERVING_BENCHMARK.md.
"""
import argparse
import array
import datetime
import importlib.util
import io
import json
import math
import os
import pathlib
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave

SCHEMA = "izwi.audio_load"
SCHEMA_VERSION = 1
PARITY_FAILURE_EXIT = 3

_spec = importlib.util.spec_from_file_location(
    "serving_benchmark", pathlib.Path(__file__).with_name("run-serving-benchmark.py"))
serving = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(serving)


# --------------------------------------------------------------------- audio

def read_wav(path):
    """(sample_rate, mono int16 samples) of a PCM16 WAV; channels are averaged."""
    with wave.open(path, "rb") as handle:
        if handle.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM WAV is supported")
        rate = handle.getframerate()
        channels = handle.getnchannels()
        raw = array.array("h", handle.readframes(handle.getnframes()))
    if sys.byteorder == "big":
        raw.byteswap()
    if channels == 1:
        return rate, raw
    mono = array.array("h", (
        int(sum(raw[i:i + channels]) / channels) for i in range(0, len(raw), channels)
    ))
    return rate, mono


def wav_bytes(rate, samples):
    data = array.array("h", samples)
    if sys.byteorder == "big":
        data.byteswap()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(data.tobytes())
    return buffer.getvalue()


def wav_samples(payload):
    """(sample_rate, mono int16 samples) of WAV bytes."""
    return read_wav(io.BytesIO(payload))


# -------------------------------------------------------------------- inputs

def cut_clips(sources, count, min_secs, max_secs, rng):
    """`count` clips, each from a random source at a random offset, with a
    length uniform in [min_secs, max_secs] (a source shorter than that is
    used whole)."""
    if not 0 < min_secs <= max_secs:
        raise ValueError("clip lengths need 0 < --min-secs <= --max-secs")
    clips = []
    for _ in range(count):
        name, rate, samples = rng.choice(sources)
        length = min(len(samples), int(rng.uniform(min_secs, max_secs) * rate))
        offset = rng.randint(0, len(samples) - length)
        clips.append({
            "source": name,
            "offset_secs": offset / rate,
            "audio_secs": length / rate,
            "rate": rate,
            "samples": samples[offset:offset + length],
        })
    return clips


def tts_inputs(count, min_words, max_words, rng):
    """Seeded sentences of random common words, `min_words..max_words` each."""
    if not 0 < min_words <= max_words:
        raise ValueError("input lengths need 0 < --min-words <= --max-words")
    texts = []
    for _ in range(count):
        words = [rng.choice(serving.SYNTHETIC_WORDS) for _ in range(rng.randint(min_words, max_words))]
        sentences = []
        while words:
            size = rng.randint(6, 14)
            sentences.append(words[:size])
            words = words[size:]
        texts.append(" ".join(
            " ".join(sentence).capitalize() + "." for sentence in sentences))
    return texts


# ------------------------------------------------------------------ requests

def multipart(fields, file_field, filename, payload):
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode())
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; "
        f"filename=\"{filename}\"\r\nContent-Type: audio/wav\r\n\r\n".encode())
    parts.append(payload)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def post(url, body, headers, timeout):
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def transcribe(config, clip):
    fields = {"model": config["model"], "response_format": "json"}
    if config.get("language"):
        fields["language"] = config["language"]
    body, content_type = multipart(fields, "file", "clip.wav", wav_bytes(clip["rate"], clip["samples"]))
    headers = dict(config["headers"], **{"Content-Type": content_type})
    payload = json.loads(post(config["base_url"] + "/v1/audio/transcriptions", body, headers,
                              config["timeout"]))
    return {"text": payload.get("text", ""), "audio_secs": clip["audio_secs"]}


def synthesize(config, text, seed):
    request = dict(config.get("tts_extra") or {})
    request.update({"model": config["model"], "input": text, "response_format": "wav", "seed": seed})
    for key in ("voice", "temperature"):
        if config.get(key) is not None:
            request[key] = config[key]
    headers = dict(config["headers"], **{"Content-Type": "application/json"})
    rate, samples = wav_samples(post(config["base_url"] + "/v1/audio/speech",
                                     json.dumps(request).encode(), headers, config["timeout"]))
    return {"samples": samples, "audio_secs": len(samples) / rate if rate else 0.0}


def request_seed(base, index):
    """The TTS seed for input `index`. izwi's samplers draw a clock seed for
    seed 0, so seeds start at 1 to keep every request repeatable."""
    return base + index + 1


def send_one(config, inputs, index):
    """One request for input `index`; the record carries timing and the
    output (`text` or `samples`), or the error."""
    started = time.monotonic()
    record = {"index": index}
    try:
        if config["task"] == "asr":
            output = transcribe(config, inputs[index])
        else:
            output = synthesize(config, inputs[index], request_seed(config["seed"], index))
        record.update(output)
    except (OSError, urllib.error.URLError, ValueError, wave.Error) as error:
        record["error"] = describe_error(error)
    record["latency_s"] = time.monotonic() - started
    return record


def describe_error(error):
    if isinstance(error, urllib.error.HTTPError):
        body = error.read(4096).decode("utf-8", "replace")
        return f"HTTP {error.code}: {body}"
    return f"{type(error).__name__}: {error}"


# -------------------------------------------------------------------- parity

_PUNCTUATION = re.compile(r"[^\w\s]")


def normalize_text(text):
    return " ".join(_PUNCTUATION.sub(" ", text.lower()).split())


def asr_parity(solo, loaded):
    return {
        "exact": solo["text"] == loaded["text"],
        "normalized": normalize_text(solo["text"]) == normalize_text(loaded["text"]),
    }


def tts_parity(solo, loaded, tolerance):
    """Same length and every sample within `tolerance` of full scale."""
    a, b = solo["samples"], loaded["samples"]
    shared = min(len(a), len(b))
    max_diff = max((abs(a[i] - b[i]) for i in range(shared)), default=0) / 32768.0
    same_length = len(a) == len(b)
    return {
        "exact": same_length and max_diff == 0.0,
        "close": same_length and max_diff <= tolerance,
        "length_delta": len(b) - len(a),
        "max_abs_diff": max_diff,
    }


def parity(config, solo, loaded):
    if config["task"] == "asr":
        return asr_parity(solo, loaded)
    return tts_parity(solo, loaded, config["tts_tolerance"])


def parity_holds(config, check):
    return check["exact"] if config["task"] == "asr" else check["close"]


# ---------------------------------------------------------------------- load

def run_point(config, inputs, rate, burstiness, max_concurrency, seed):
    """Sends one request per input at seeded open-loop arrival times."""
    offsets = serving.arrival_offsets(len(inputs), rate, burstiness, random.Random(seed))
    order = list(range(len(inputs)))
    random.Random(seed + 1).shuffle(order)
    limiter = threading.BoundedSemaphore(max_concurrency) if max_concurrency else None
    records = [None] * len(inputs)
    sent = [0.0] * len(inputs)
    threads = []

    def worker(slot, index):
        try:
            record = send_one(config, inputs, index)
        finally:
            if limiter:
                limiter.release()
        record["sent_at_s"] = sent[slot]
        records[slot] = record

    epoch = time.monotonic()
    for slot, (offset, index) in enumerate(zip(offsets, order)):
        delay = epoch + offset - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        if limiter:
            limiter.acquire()
        sent[slot] = time.monotonic() - epoch
        thread = threading.Thread(target=worker, args=(slot, index), daemon=True)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    return records, time.monotonic() - epoch


def summarize(config, records, duration, solo):
    ok = [record for record in records if "error" not in record]
    audio_secs = sum(record["audio_secs"] for record in ok)
    summary = {
        "requests": len(records),
        "completed": len(ok),
        "errors": len(records) - len(ok),
        "duration_s": duration,
        "request_throughput": len(ok) / duration if duration > 0 else None,
        "audio_throughput": audio_secs / duration if duration > 0 else None,
        "latency_s": serving.distribution([record["latency_s"] for record in ok]),
        "rtf": serving.distribution([
            record["latency_s"] / record["audio_secs"] for record in ok if record["audio_secs"] > 0
        ]),
        "error_samples": sorted({record["error"] for record in records if "error" in record})[:5],
    }
    if solo is not None:
        checks = []
        for record in ok:
            reference = solo.get(record["index"])
            if reference is not None:
                checks.append((record["index"], parity(config, reference, record)))
        failed = [index for index, check in checks if not parity_holds(config, check)]
        summary["parity"] = {
            "checked": len(checks),
            "exact": sum(check["exact"] for _, check in checks),
            "failed": len(failed),
            "failed_inputs": sorted(failed)[:20],
        }
        if config["task"] == "asr":
            summary["parity"]["normalized"] = sum(check["normalized"] for _, check in checks)
        else:
            summary["parity"]["close"] = sum(check["close"] for _, check in checks)
            summary["parity"]["max_abs_diff"] = max(
                (check["max_abs_diff"] for _, check in checks), default=0.0)
    return summary


def solo_pass(config, inputs):
    """Every input alone, in order: the parity reference."""
    outputs, records = {}, []
    for index in range(len(inputs)):
        record = send_one(config, inputs, index)
        records.append(record)
        if "error" not in record:
            outputs[index] = record
    return outputs, records


def solo_repeats(config, inputs, solo):
    """Whether the first solo input gives the same output a second time. A
    model that does not repeat itself (an unseeded noise source) cannot be
    held to parity; None when the first input failed."""
    reference = solo.get(0)
    if reference is None:
        return None
    again = send_one(config, inputs, 0)
    return "error" not in again and parity_holds(config, parity(config, reference, again))


def public_record(record):
    """A record without its audio, for --save-samples."""
    return {key: value for key, value in record.items() if key != "samples"}


# --------------------------------------------------------------------- report

def print_summary(name, summary, out=sys.stdout):
    def fmt(value, digits=2):
        return "-" if value is None else f"{value:.{digits}f}"

    latency = summary["latency_s"] or {}
    rtf = summary["rtf"] or {}
    line = (f"{name}: {summary['completed']}/{summary['requests']} ok, "
            f"{fmt(summary['request_throughput'])} req/s, "
            f"{fmt(summary['audio_throughput'])} audio s/s, "
            f"latency p50/p95 {fmt(latency.get('p50'))}/{fmt(latency.get('p95'))} s, "
            f"RTF p50/p95 {fmt(rtf.get('p50'), 3)}/{fmt(rtf.get('p95'), 3)}")
    parity_report = summary.get("parity")
    if parity_report:
        line += f", parity {parity_report['checked'] - parity_report['failed']}/{parity_report['checked']}"
    print(line, file=out)
    for error in summary["error_samples"]:
        print(f"  error: {error}", file=out)
    if parity_report and parity_report["failed"]:
        print(f"  parity failures on inputs {parity_report['failed_inputs']}", file=out)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", choices=("asr", "tts"), required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=os.environ.get("IZWI_BENCH_API_KEY"),
                        help="bearer token (default $IZWI_BENCH_API_KEY)")
    parser.add_argument("--num-requests", type=int, default=32, help="requests per sweep point (default 32)")
    parser.add_argument("--request-rate", default="1",
                        help="comma-separated requests/s per point; 'inf' sends all at once (default 1)")
    parser.add_argument("--burstiness", type=float, default=1.0,
                        help="gamma shape of the gaps: 1 Poisson, <1 burstier (default 1)")
    parser.add_argument("--max-concurrency", type=int, help="cap on requests in flight")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds per request (default 600)")
    parser.add_argument("--audio", action="append", default=[], help="ASR source WAV (repeatable)")
    parser.add_argument("--min-secs", type=float, default=2.0, help="shortest ASR clip (default 2)")
    parser.add_argument("--max-secs", type=float, default=30.0, help="longest ASR clip (default 30)")
    parser.add_argument("--language", help="ASR language hint")
    parser.add_argument("--min-words", type=int, default=4, help="shortest TTS input (default 4)")
    parser.add_argument("--max-words", type=int, default=40, help="longest TTS input (default 40)")
    parser.add_argument("--voice", help="TTS voice")
    parser.add_argument("--temperature", type=float, help="TTS sampling temperature")
    parser.add_argument("--tts-extra", type=json.loads, default={},
                        help="JSON object of extra speech fields sent with every TTS request, "
                             "e.g. '{\"subtalker_do_sample\": false}'")
    parser.add_argument("--tts-tolerance", type=float, default=0.01,
                        help="largest TTS sample difference from solo, as a fraction of full scale (default 0.01)")
    parser.add_argument("--warmup", type=int, default=1, help="discarded requests before the solo pass")
    parser.add_argument("--no-solo", action="store_true", help="skip the solo pass and parity")
    parser.add_argument("--require-parity", action="store_true",
                        help=f"exit {PARITY_FAILURE_EXIT} when any request's output differs from its solo output")
    parser.add_argument("--save-samples", action="store_true", help="keep every request record in the output")
    parser.add_argument("--output", help="write the JSON result here")
    args = parser.parse_args(argv)
    if args.task == "asr" and not args.audio:
        parser.error("--task asr needs at least one --audio WAV")
    if args.num_requests <= 0:
        parser.error("--num-requests must be positive")
    return args


def build_inputs(args, rng):
    if args.task == "asr":
        sources = [(path, *read_wav(path)) for path in args.audio]
        return cut_clips(sources, args.num_requests, args.min_secs, args.max_secs, rng)
    return tts_inputs(args.num_requests, args.min_words, args.max_words, rng)


def main(argv=None):
    args = parse_args(argv)
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    config = {
        "task": args.task,
        "base_url": args.base_url.rstrip("/"),
        "model": args.model,
        "headers": headers,
        "timeout": args.timeout,
        "language": args.language,
        "voice": args.voice,
        "temperature": args.temperature,
        "seed": args.seed,
        "tts_tolerance": args.tts_tolerance,
        "tts_extra": args.tts_extra,
    }
    rates = [serving.parse_rate(raw) for raw in args.request_rate.split(",")]
    inputs = build_inputs(args, random.Random(args.seed))
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    for _ in range(args.warmup):
        warm = send_one(config, inputs, 0)
        if "error" in warm:
            print(f"warm-up failed: {warm['error']}", file=sys.stderr)
            return 1

    solo, solo_records, solo_summary, repeatable = None, [], None, None
    if not args.no_solo:
        solo_started = time.monotonic()
        solo, solo_records = solo_pass(config, inputs)
        solo_summary = summarize(config, solo_records, time.monotonic() - solo_started, None)
        print_summary("solo", solo_summary)
        repeatable = solo_repeats(config, inputs, solo)
        if repeatable is False:
            print("solo output does not repeat for the same input and seed; parity is not checked",
                  file=sys.stderr)
            solo = None

    points = []
    parity_failed = False
    for point, rate in enumerate(rates):
        records, duration = run_point(config, inputs, rate, args.burstiness, args.max_concurrency,
                                      args.seed + 1000 * (point + 1))
        summary = summarize(config, records, duration, solo)
        name = serving.point_name(rate, args.max_concurrency)
        print_summary(name, summary)
        parity_failed |= bool(summary.get("parity", {}).get("failed"))
        report = {"config": {"request_rate": serving.json_rate(rate),
                             "max_concurrency": args.max_concurrency,
                             "burstiness": args.burstiness},
                  "summary": summary}
        if args.save_samples:
            report["samples"] = [public_record(record) for record in records]
        points.append({"name": name, "report": report})

    result = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "started_at": started_at,
        "ended_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "metadata": {
            "git": serving.git_info(),
            "host": serving.host_info(),
            "server": serving.server_info(config["base_url"], headers),
        },
        "config": {
            "task": args.task, "model": args.model, "num_requests": args.num_requests,
            "seed": args.seed, "audio": args.audio, "min_secs": args.min_secs,
            "max_secs": args.max_secs, "min_words": args.min_words, "max_words": args.max_words,
            "voice": args.voice, "temperature": args.temperature, "tts_tolerance": args.tts_tolerance,
            "tts_extra": args.tts_extra,
            "inputs": [
                {key: value for key, value in clip.items() if key != "samples"} if args.task == "asr"
                else {"words": len(clip.split())}
                for clip in inputs
            ],
        },
        "solo": {"summary": solo_summary, "repeatable": repeatable,
                 "samples": [public_record(record) for record in solo_records] if args.save_samples else None},
        "reports": points,
    }
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
    if args.require_parity and (parity_failed or repeatable is False):
        return PARITY_FAILURE_EXIT
    return 0


if __name__ == "__main__":
    sys.exit(main())
