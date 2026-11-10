#!/usr/bin/env python3
"""Transcription accuracy on telephone audio against the original audio.

Takes reference recordings with their transcripts and, for each model,
transcribes every clip through `/v1/audio/transcriptions` under each
condition:

- `original`: the recording as it is;
- `ulaw`: band-limited to 3.4 kHz, resampled to 8 kHz, then G.711 μ-law
  encoded and decoded, which is what a model hears of a phone call;
- `alaw`: the same through G.711 A-law;
- `narrowband` (not run by default): band-limited and resampled only, to
  tell the loss of bandwidth from the loss of companding.

It reports the word error rate (WER) and character error rate (CER) per model
and condition over the whole set, and how far each telephone condition falls
from `original`, absolutely and relative to it. Text is normalized as
score-asr-matrix.py does (case, punctuation).

References are a directory of `name.wav` files, each with a `name.txt` or
`name.md` transcript beside it (`--audio data` scores the repository's
fixtures), or a JSONL manifest of `{"audio": "a.wav", "text": "..."}` lines
(`"text_file"` instead of `"text"` names a transcript file; paths are
relative to the manifest).

Results are JSON (schema "izwi.telephony_accuracy", version 1) with git, host
and server metadata, and a printed table. Standard library only (the filter
and the codec are plain Python): runs with plain python3.
"""
import argparse
import datetime
import importlib.util
import json
import math
import operator
import os
import pathlib
import sys

SCHEMA = "izwi.telephony_accuracy"
SCHEMA_VERSION = 1
CONDITIONS = ("original", "narrowband", "ulaw", "alaw")
DEFAULT_CONDITIONS = "original,ulaw,alaw"
TELEPHONE_RATE = 8_000
# The upper edge of the telephone band.
CUTOFF_HZ = 3_400.0
# Half the low-pass filter's length, in 8 kHz samples.
HALF_WIDTH = 16


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, pathlib.Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


load = _load("audio_load", "run-audio-load.py")
scoring = _load("asr_scoring", "score-asr-matrix.py")
serving = load.serving


# --------------------------------------------------------------------- G.711
# The same companding as crates/izwi-audio-codec/src/g711.rs.

def linear_to_ulaw(sample):
    sign = 0x80 if sample < 0 else 0
    magnitude = min(abs(sample), 32_635) + 0x84
    exponent = magnitude.bit_length() - 8
    return ~(sign | (exponent << 4) | ((magnitude >> (exponent + 3)) & 0x0F)) & 0xFF


def ulaw_to_linear(code):
    code = ~code & 0xFF
    magnitude = ((((code & 0x0F) << 3) + 0x84) << ((code >> 4) & 0x07)) - 0x84
    return -magnitude if code & 0x80 else magnitude


def linear_to_alaw(sample):
    pcm = sample >> 3
    mask = 0xD5
    if pcm < 0:
        pcm, mask = -pcm - 1, 0x55
    segment = max(pcm.bit_length() - 5, 0)
    if segment > 7:
        return 0x7F ^ mask
    return ((segment << 4) | ((pcm >> max(segment, 1)) & 0x0F)) ^ mask


def alaw_to_linear(code):
    code ^= 0x55
    segment = (code & 0x70) >> 4
    value = (code & 0x0F) << 4
    if segment == 0:
        value += 8
    else:
        value = (value + 0x108) << (segment - 1)
    return value if code & 0x80 else -value


def compand(samples, encode, decode):
    """`samples` after one pass through a G.711 law."""
    table = [decode(code) for code in range(256)]
    return [table[encode(sample)] for sample in samples]


# ----------------------------------------------------------------- telephone

def telephone_band(rate, samples):
    """`samples` low-passed at 3.4 kHz and resampled to 8 kHz, with a
    Hann-windowed sinc centred on each output sample."""
    cutoff = min(CUTOFF_HZ, 0.45 * rate) / rate
    half = max(1, HALF_WIDTH * rate // TELEPHONE_RATE)
    padded = [0] * half + list(samples) + [0] * (half + 1)
    kernels = {}
    out = []
    for index in range(len(samples) * TELEPHONE_RATE // rate):
        # The output sample sits at input position whole + phase / 8000.
        whole, phase = divmod(index * rate, TELEPHONE_RATE)
        kernel = kernels.get(phase)
        if kernel is None:
            taps = []
            for tap in range(-half + 1, half + 1):
                x = tap - phase / TELEPHONE_RATE
                sinc = 1.0 if x == 0 else math.sin(2 * math.pi * cutoff * x) / (2 * math.pi * cutoff * x)
                taps.append(sinc * (0.5 + 0.5 * math.cos(math.pi * x / half)))
            gain = sum(taps)
            kernel = kernels[phase] = [tap / gain for tap in taps]
        # padded[whole + 1 + k] is input sample whole - half + 1 + k.
        window = padded[whole + 1:whole + 1 + 2 * half]
        value = round(sum(map(operator.mul, kernel, window)))
        out.append(max(-32_768, min(32_767, value)))
    return out


def condition_audio(condition, rate, samples):
    """(rate, samples) of a clip as `condition` presents it."""
    if condition == "original":
        return rate, samples
    narrow = telephone_band(rate, samples)
    if condition == "ulaw":
        narrow = compand(narrow, linear_to_ulaw, ulaw_to_linear)
    elif condition == "alaw":
        narrow = compand(narrow, linear_to_alaw, alaw_to_linear)
    return TELEPHONE_RATE, narrow


# ---------------------------------------------------------------- references

def read_transcript(path):
    return " ".join(pathlib.Path(path).read_text(encoding="utf-8").split())


def directory_clips(directory):
    clips = []
    for wav in sorted(pathlib.Path(directory).glob("*.wav")):
        transcript = next((path for path in (wav.with_suffix(".txt"), wav.with_suffix(".md"))
                           if path.is_file()), None)
        if transcript is None:
            print(f"skipping {wav}: no transcript beside it", file=sys.stderr)
            continue
        clips.append({"name": wav.stem, "audio": str(wav), "reference": read_transcript(transcript)})
    return clips


def manifest_clips(manifest):
    root = pathlib.Path(manifest).parent
    clips = []
    for number, line in enumerate(pathlib.Path(manifest).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        entry = json.loads(line)
        if "audio" not in entry or not ("text" in entry or "text_file" in entry):
            raise ValueError(f"{manifest}:{number}: an entry needs `audio` and `text` or `text_file`")
        audio = root / entry["audio"]
        reference = (" ".join(entry["text"].split()) if "text" in entry
                     else read_transcript(root / entry["text_file"]))
        clips.append({"name": entry.get("name", audio.stem), "audio": str(audio), "reference": reference})
    return clips


# ------------------------------------------------------------------- scoring

def edit_distance(reference, hypothesis):
    """Substitutions, insertions and deletions that turn one sequence into the other."""
    row = list(range(len(hypothesis) + 1))
    for i, expected in enumerate(reference, 1):
        previous, row[0] = row[0], i
        for j, got in enumerate(hypothesis, 1):
            current = min(row[j] + 1, row[j - 1] + 1, previous + (expected != got))
            previous, row[j] = row[j], current
    return row[len(hypothesis)]


def clip_errors(reference, hypothesis):
    """Word and character error counts of one transcript, on normalized text."""
    ref, hyp = scoring.normalize(reference), scoring.normalize(hypothesis)
    ref_chars, hyp_chars = " ".join(ref), " ".join(hyp)
    return {
        "word_errors": edit_distance(ref, hyp), "reference_words": len(ref),
        "char_errors": edit_distance(ref_chars, hyp_chars), "reference_chars": len(ref_chars),
    }


def summarize(records):
    """WER and CER over the clips of one model and condition that were transcribed."""
    scored = [record for record in records if "error" not in record]
    total = {key: sum(record[key] for record in scored)
             for key in ("word_errors", "reference_words", "char_errors", "reference_chars")}
    return {
        "clips": len(scored),
        "failed": len(records) - len(scored),
        "wer": total["word_errors"] / total["reference_words"] if total["reference_words"] else None,
        "cer": total["char_errors"] / total["reference_chars"] if total["reference_chars"] else None,
        **total,
    }


def degradation(baseline, summary):
    """How far `summary` falls from `baseline`: the rise in each rate, in
    absolute terms and relative to the baseline's (None from a zero baseline)."""
    out = {}
    for rate in ("wer", "cer"):
        base, value = baseline[rate], summary[rate]
        absolute = None if base is None or value is None else value - base
        out[f"{rate}_absolute"] = absolute
        out[f"{rate}_relative"] = absolute / base if absolute is not None and base else None
    return out


def model_report(conditions, records):
    """`records` maps a condition to its clip records."""
    summaries = {condition: summarize(records[condition]) for condition in conditions}
    report = {"conditions": summaries}
    if "original" in summaries:
        report["degradation"] = {condition: degradation(summaries["original"], summary)
                                 for condition, summary in summaries.items() if condition != "original"}
    return report


def print_table(conditions, reports, out=None):
    out = out or sys.stdout

    def rate(value):
        return "n/a" if value is None else f"{100 * value:.2f}%"

    def change(fall, key):
        if not fall or fall[f"{key}_absolute"] is None:
            return "n/a"
        relative = fall[f"{key}_relative"]
        points = f"{100 * fall[f'{key}_absolute']:+.2f}pt"
        return points if relative is None else f"{points} ({100 * relative:+.0f}%)"

    width = max([len("model")] + [len(model) for model in reports])
    print(f"{'model':<{width}}  {'condition':<10}  {'clips':>5}  {'failed':>6}  {'WER':>8}  {'CER':>8}  "
          f"{'WER vs original':>18}  {'CER vs original':>18}", file=out)
    for model, report in reports.items():
        for condition in conditions:
            summary = report["conditions"][condition]
            fall = report.get("degradation", {}).get(condition)
            print(f"{model:<{width}}  {condition:<10}  {summary['clips']:>5}  {summary['failed']:>6}  "
                  f"{rate(summary['wer']):>8}  {rate(summary['cer']):>8}  "
                  f"{change(fall, 'wer'):>18}  {change(fall, 'cer'):>18}", file=out)


# ----------------------------------------------------------------------- run

def transcribe_clip(config, clip, condition):
    rate, samples = clip["conditions"][condition]
    try:
        answer = load.transcribe(config, {"rate": rate, "samples": samples,
                                          "audio_secs": clip["audio_secs"]})
    except Exception as error:  # noqa: BLE001 - a failed clip is reported, not raised
        return {"clip": clip["name"], "error": load.describe_error(error)}
    return {"clip": clip["name"], "text": answer["text"],
            **clip_errors(clip["reference"], answer["text"])}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", help="directory of name.wav files with name.txt or name.md transcripts")
    source.add_argument("--manifest", help="JSONL of {\"audio\": ..., \"text\": ...} references")
    parser.add_argument("--model", action="append", required=True,
                        help="model to score (repeatable, or comma-separated)")
    parser.add_argument("--conditions", default=DEFAULT_CONDITIONS,
                        help=f"comma-separated, of {', '.join(CONDITIONS)} (default {DEFAULT_CONDITIONS})")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--api-key", default=os.environ.get("IZWI_BENCH_API_KEY"),
                        help="bearer token (default $IZWI_BENCH_API_KEY)")
    parser.add_argument("--language", help="ASR language hint")
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds per request (default 600)")
    parser.add_argument("--save-samples", action="store_true", help="keep every transcript in the output")
    parser.add_argument("--output", help="write the JSON result here")
    args = parser.parse_args(argv)
    args.models = [model.strip() for entry in args.model for model in entry.split(",") if model.strip()]
    args.conditions = [condition.strip() for condition in args.conditions.split(",") if condition.strip()]
    unknown = [condition for condition in args.conditions if condition not in CONDITIONS]
    if unknown or not args.conditions:
        parser.error(f"--conditions takes {', '.join(CONDITIONS)}; got {', '.join(unknown) or 'none'}")
    if not args.models:
        parser.error("--model names no model")
    return args


def main(argv=None):
    args = parse_args(argv)
    clips = directory_clips(args.audio) if args.audio else manifest_clips(args.manifest)
    if not clips:
        print("no reference clips with transcripts were found", file=sys.stderr)
        return 1
    for clip in clips:
        rate, samples = load.read_wav(clip["audio"])
        clip["rate"] = rate
        clip["audio_secs"] = len(samples) / rate
        clip["conditions"] = {condition: condition_audio(condition, rate, samples)
                              for condition in args.conditions}

    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    base_url = args.base_url.rstrip("/")
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    reports = {}
    for model in args.models:
        config = {"model": model, "language": args.language, "headers": headers,
                  "base_url": base_url, "timeout": args.timeout}
        records = {condition: [transcribe_clip(config, clip, condition) for clip in clips]
                   for condition in args.conditions}
        reports[model] = model_report(args.conditions, records)
        for condition in args.conditions:
            for record in records[condition]:
                if "error" in record:
                    print(f"{model} {condition} {record['clip']}: {record['error']}", file=sys.stderr)
        if args.save_samples:
            reports[model]["samples"] = records
    print_table(args.conditions, reports)

    result = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "started_at": started_at,
        "ended_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "metadata": {
            "git": serving.git_info(),
            "host": serving.host_info(),
            "server": serving.server_info(base_url, headers),
        },
        "config": {
            "models": args.models, "conditions": args.conditions, "language": args.language,
            "telephone_rate": TELEPHONE_RATE, "cutoff_hz": CUTOFF_HZ,
            "clips": [{"name": clip["name"], "audio": clip["audio"], "rate": clip["rate"],
                       "audio_secs": clip["audio_secs"],
                       "reference_words": len(scoring.normalize(clip["reference"]))}
                      for clip in clips],
        },
        "models": reports,
    }
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
    failed = any(summary["failed"] for report in reports.values()
                 for summary in report["conditions"].values())
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
