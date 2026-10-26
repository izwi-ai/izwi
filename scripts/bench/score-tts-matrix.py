#!/usr/bin/env python3
"""TTS matrix client and scorer (standard library only).

`request` runs one model family's cases against a running izwi-server and
writes <out>/<case>.wav plus <case>.json (status, timings, response headers,
SSE events). `score` reads those cases, adds audio-health statistics and an
optional ASR round-trip WER, and writes a JSON summary with the stop reason
and the decode_steps/decode_tokens pair (equal means the budget, not end of
speech, stopped the model).

Usage:
  score-tts-matrix.py request --base URL --model M --family F --out DIR --ref WAV
  score-tts-matrix.py score --runs DIR [DIR ...] [--asr-base URL] --out summary.json
"""
import argparse
import array
import base64
import glob
import json
import math
import os
import pathlib
import re
import struct
import sys
import time
import unicodedata
import urllib.error
import urllib.request
import uuid
import wave

ROOT = pathlib.Path(__file__).resolve().parents[2]

TEXTS = {
    "short": "The quick brown fox jumps over the lazy dog.",
    "med": (
        "Good morning. Today we are testing a speech synthesis system on a laptop. "
        "The goal is to measure how fast it runs, how much memory it uses, and whether "
        "every word in this paragraph comes out clearly, without skipping, repeating, "
        "or adding anything at the end."
    ),
    "num": "On March 3rd, 2025, Dr. Smith paid $1,250.75 for 3 items at 4:30 PM.",
    "long": (
        "The old lighthouse stood at the edge of the harbor for more than a century. "
        "Every evening, the keeper climbed one hundred and twelve steps to light the lamp, "
        "and every morning he climbed them again to put it out. Ships from distant ports "
        "used its beam to find the narrow channel between the rocks. In winter, storms "
        "pushed waves high enough to wash over the lower windows, and the keeper would sit "
        "by the stove, listening to the wind and writing in his logbook. He recorded the "
        "weather, the ships that passed, and the small events of each day. One entry "
        "describes a fishing boat that lost its engine in a gale and drifted toward the "
        "cliffs until the crew saw the light and rowed to safety. Another mentions a family "
        "of seals that slept on the breakwater every spring. When the lighthouse was finally "
        "automated, the logbooks were sent to the town library, where students still read "
        "them today. They show how a single person, doing the same careful work every night, "
        "kept hundreds of strangers safe. The building is now a small museum. Visitors can "
        "climb the stairs, look out over the water, and imagine what it was like to watch "
        "the sea alone for so many years. On clear nights, the automated lamp still turns, "
        "sweeping its light across the dark water just as it always has."
    ),
    # Input that begins with "-" must be spoken, never read as an option.
    "dash": "-5 degrees outside.",
}


def family_cases(family, reference):
    """[(case_id, text_key, extra_fields, stream)] for one model family."""
    clone = {"reference_audio": reference[0], "reference_text": reference[1]}
    ryan = {"voice": "ryan", "language": "English"}
    greedy = dict(ryan, temperature=0.0, top_k=1)
    table = {
        "kokoro": [
            ("short", "short", {"voice": "af_heart"}, False),
            ("med", "med", {"voice": "af_heart"}, False),
            ("long", "long", {"voice": "af_heart"}, False),
            ("dash", "dash", {"voice": "af_heart"}, False),
            ("speed025", "short", {"voice": "af_heart", "speed": 0.25}, False),
            ("med_stream", "med", {"voice": "af_heart"}, True),
        ],
        "lfm": [
            ("short", "short", {"voice": "US Female"}, False),
            ("med", "med", {"voice": "US Female"}, False),
            ("med_stream", "med", {"voice": "US Female"}, True),
        ],
        "qwen_cv": [
            ("short", "short", ryan, False),
            ("med", "med", ryan, False),
            ("long", "long", ryan, False),
        ],
        "qwen_base": [
            ("short", "short", dict(clone, language="English"), False),
            ("med", "med", dict(clone, language="English"), False),
        ],
        "qwen_greedy": [
            ("short_t0_a", "short", greedy, False),
            ("short_t0_b", "short", greedy, False),
            ("med_t0_400", "med", dict(greedy, max_tokens=400), False),
        ],
        "vibevoice": [
            ("short", "short", clone, False),
            ("med", "med", clone, False),
            ("med_stream", "med", clone, True),
        ],
        "voxtral": [
            ("short", "short", {"voice": "casual_male"}, False),
            ("med", "med", {"voice": "casual_male"}, False),
        ],
        "fish": [("short", "short", clone, False)],
        # Final-only: no streaming case. "long" exercises automatic long form,
        # "clone" cloning from reference audio alone.
        "chatterbox": [
            ("short", "short", {"voice": "default"}, False),
            ("med", "med", {"voice": "default"}, False),
            ("num", "num", {"voice": "default"}, False),
            ("long", "long", {"voice": "default"}, False),
            ("clone", "med", {"reference_audio": reference[0]}, False),
            ("expressive", "short",
             {"voice": "default", "exaggeration": 0.9, "cfg_weight": 0.3}, False),
        ],
    }
    if family not in table:
        raise SystemExit(f"unknown family {family}; known: {', '.join(sorted(table))}")
    return table[family]


def normalize(text):
    ones = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
            "fourteen fifteen sixteen seventeen eighteen nineteen").split()
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'").replace("-", " ")
    words = re.sub(r"[^\w\s']", " ", text).split()
    return [ones[int(w)] if w.isdigit() and int(w) < 20 else w for w in words]


def wer(reference, hypothesis):
    ref, hyp = normalize(reference), normalize(hypothesis)
    row = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        prev, row[0] = row[0], i
        for j in range(1, len(hyp) + 1):
            cur = min(row[j] + 1, row[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
            prev, row[j] = row[j], cur
    return row[len(hyp)] / max(1, len(ref))


def wav_payload(data):
    """(pcm_bytes, rate, channels) from WAV bytes; tolerates streaming sizes."""
    if data[:4] != b"RIFF":
        return data, None, None
    pos, rate, channels = 12, None, None
    while pos + 8 <= len(data):
        chunk_id = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        if chunk_id == b"fmt ":
            channels, rate = struct.unpack("<HI", data[pos + 10:pos + 16])
        if chunk_id == b"data":
            end = len(data) if size in (0, 0xFFFFFFFF) or pos + 8 + size > len(data) \
                else pos + 8 + size
            return data[pos + 8:end], rate, channels
        pos += 8 + size + (size & 1)
    return b"", rate, channels


def write_wav(path, pcm, rate, channels=1):
    with wave.open(str(path), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm)


def audio_stats(path):
    """Duration, level, peak and leading/trailing silence of a 16-bit WAV."""
    with wave.open(str(path)) as source:
        rate, frames = source.getframerate(), source.getnframes()
        samples = array.array("h", source.readframes(frames))
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return {"duration_s": 0.0}
    scale = 1 / 32768.0
    window = max(1, int(rate * 0.02))
    loud = []
    for start in range(0, len(samples) - window + 1, window):
        energy = sum((s * scale) ** 2 for s in samples[start:start + window]) / window
        loud.append(20 * math.log10(math.sqrt(energy) + 1e-9) >= -50)
    lead = loud.index(True) if True in loud else len(loud)
    trail = loud[::-1].index(True) if True in loud else len(loud)
    mean_square = sum((s * scale) ** 2 for s in samples) / len(samples)
    return {
        "duration_s": round(len(samples) / rate, 3),
        "sample_rate": rate,
        "rms_dbfs": round(20 * math.log10(math.sqrt(mean_square) + 1e-9), 1),
        "peak": round(max(abs(s) for s in samples) * scale, 4),
        "lead_sil_s": round(lead * 0.02, 2),
        "trail_sil_s": round(trail * 0.02, 2),
    }


def post_json(url, body, timeout):
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        return error


def run_case(base, model, out, case_id, key, extra, stream, timeout):
    body = {"model": model, "input": TEXTS[key], "response_format": "pcm" if stream else "wav"}
    body.update(extra)
    if stream:
        body.update(stream=True, stream_format="sse")
    record = {"case": case_id, "model": model, "stream": stream, "text_key": key,
              "fields": sorted(extra)}
    started = time.time()
    try:
        response = post_json(f"{base}/v1/audio/speech", body, timeout)
        record["http"] = response.status
        record["headers"] = {k.lower(): v for k, v in response.headers.items()
                             if k.lower().startswith("x-")}
        content_type = response.headers.get("content-type") or ""
        if not stream or response.status != 200 or "event-stream" not in content_type:
            data = response.read()
            record["total_s"] = round(time.time() - started, 3)
            if response.status != 200:
                record["error_body"] = data[:2000].decode(errors="replace")
            else:
                pcm, rate, channels = wav_payload(data)
                record["sample_rate"] = rate
                if rate:
                    write_wav(out / f"{case_id}.wav", pcm, rate, channels or 1)
        else:
            events, pcm, rate, first = [], bytearray(), None, None
            for raw in response:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                now = round(time.time() - started, 3)
                event = json.loads(line[6:])
                summary = {"t": now, "event": event.get("event")}
                if event.get("event") == "audio.chunk":
                    chunk = base64.b64decode(event.get("audio_base64") or "")
                    rate = rate or event.get("sample_rate")
                    if chunk and first is None:
                        first = now
                    pcm.extend(chunk)
                    summary["n"] = event.get("sample_count")
                for field in ("error", "stop_reason", "tokens_generated"):
                    if event.get(field) is not None:
                        summary[field] = event[field]
                events.append(summary)
            record.update(total_s=round(time.time() - started, 3), ttfa_s=first,
                          sample_rate=rate, events=events,
                          chunks=sum(e["event"] == "audio.chunk" for e in events))
            if rate and pcm:
                write_wav(out / f"{case_id}.wav", bytes(pcm), rate)
    except Exception as error:  # noqa: BLE001 - every failure is evidence
        record.setdefault("http", 0)
        record["exception"] = repr(error)[:500]
        record["total_s"] = round(time.time() - started, 3)
    (out / f"{case_id}.json").write_text(json.dumps(record, indent=1))
    print(json.dumps({k: record.get(k) for k in ("case", "http", "total_s", "ttfa_s", "chunks")}),
          flush=True)


def transcribe(base, model, path, timeout=900):
    """ASR round trip through /v1/audio/transcriptions (multipart, stdlib)."""
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in (("model", model), ("language", "en"), ("response_format", "json")):
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
                     f"{value}\r\n".encode())
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                 f"filename=\"{os.path.basename(path)}\"\r\nContent-Type: audio/wav\r\n\r\n"
                 .encode() + pathlib.Path(path).read_bytes() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    request = urllib.request.Request(
        f"{base}/v1/audio/transcriptions", data=b"".join(parts),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read()).get("text", ""), None
    except urllib.error.HTTPError as error:
        return None, error.read()[:200].decode(errors="replace")


def stop_reason(record):
    """The response's stop reason: header, SSE `audio.done`, or `None`."""
    reason = (record.get("headers") or {}).get("x-stop-reason")
    for event in record.get("events") or []:
        if event.get("event") == "audio.done" and event.get("stop_reason"):
            reason = event["stop_reason"]
    return reason


def score_case(record, wav_path, asr=None):
    headers = record.get("headers") or {}
    try:
        diagnostics = json.loads(headers.get("x-izwi-tts-diagnostics") or "{}")
    except ValueError:
        diagnostics = {}
    breakdown = diagnostics.get("latency_breakdown") or {}
    row = {
        "model": record.get("model"), "case": record.get("case"), "http": record.get("http"),
        "total_s": record.get("total_s"), "ttfa_s": record.get("ttfa_s"),
        "chunks": record.get("chunks"), "x_rtf": headers.get("x-rtf"),
        "tokens_generated": headers.get("x-tokens-generated"),
        "stop_reason": stop_reason(record),
        "decode_steps": breakdown.get("decode_steps"),
        "decode_tokens": breakdown.get("decode_tokens"),
        "error": record.get("error_body") or record.get("exception"),
    }
    if record.get("stream"):
        terminal = [e["event"] for e in record.get("events") or []
                    if e.get("event") in ("audio.done", "audio.failed")]
        row["terminal_event"] = terminal[-1] if terminal else None
    if wav_path and os.path.exists(wav_path):
        row.update(audio_stats(wav_path))
        if asr and row.get("duration_s"):
            hypothesis, error = asr(wav_path)
            row.update(asr_text=hypothesis, asr_error=error)
            if hypothesis is not None and record.get("text_key") in TEXTS:
                row["wer"] = round(wer(TEXTS[record["text_key"]], hypothesis), 3)
    return row


def score(runs, asr=None):
    rows = []
    for run in runs:
        for meta in sorted(glob.glob(os.path.join(run, "*", "cases", "*.json"))):
            record = json.loads(pathlib.Path(meta).read_text())
            row = score_case(record, meta[:-5] + ".wav", asr)
            row["run"] = os.path.basename(run.rstrip("/"))
            rows.append(row)
    for run in runs:
        for path in sorted(glob.glob(os.path.join(run, "*", "peak.json"))):
            rows.append(dict(json.loads(pathlib.Path(path).read_text()), kind="footprint"))
    return rows


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    request = commands.add_parser("request")
    request.add_argument("--base", required=True)
    request.add_argument("--model", required=True)
    request.add_argument("--family", required=True)
    request.add_argument("--out", required=True, type=pathlib.Path)
    request.add_argument("--ref", default=str(ROOT / "data" / "fox.wav"))
    request.add_argument("--ref-text", default=str(ROOT / "data" / "fox.md"))
    request.add_argument("--only", default="")
    request.add_argument("--timeout", type=float, default=3600)
    scoring = commands.add_parser("score")
    scoring.add_argument("--runs", nargs="+", required=True)
    scoring.add_argument("--asr-base")
    scoring.add_argument("--asr-model", default="Nemotron-3.5-ASR-Streaming-0.6B")
    scoring.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)

    if args.command == "request":
        args.out.mkdir(parents=True, exist_ok=True)
        reference = (base64.b64encode(pathlib.Path(args.ref).read_bytes()).decode(),
                     pathlib.Path(args.ref_text).read_text().strip())
        only = set(filter(None, args.only.split(",")))
        for case_id, key, extra, stream in family_cases(args.family, reference):
            if not only or case_id in only:
                run_case(args.base, args.model, args.out, case_id, key, extra, stream,
                         args.timeout)
        return 0

    asr = None
    if args.asr_base:
        def asr(path):
            return transcribe(args.asr_base, args.asr_model, path)
    rows = score(args.runs, asr)
    args.out.write_text(json.dumps(rows, indent=1))
    for row in rows:
        print(json.dumps({k: row.get(k) for k in (
            "model", "case", "http", "duration_s", "wer", "stop_reason", "decode_steps",
            "decode_tokens", "terminal_event", "peak_mib")})[:400])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
