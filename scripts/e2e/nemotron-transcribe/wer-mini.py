#!/usr/bin/env python3
"""WER-mini for Nemotron 3.5 Transcribe: 100 LibriSpeech test-clean and 100
FLEURS (20 each of en, de, fr, es, ja) utterances, scored against their
references and reported beside the card's numbers (BF16 model, full sets).

  prepare  download and convert the utterances (stdlib + ffmpeg; the first N
           of each archive in archive order, so every machine scores the same
           ids): LibriSpeech test-clean from openslr.org, FLEURS test from the
           google/fleurs dataset repository on Hugging Face. Writes
           <dir>/audio/*.wav (16 kHz mono 16-bit) and <dir>/manifest.jsonl:
           {"id", "set", "language", "audio", "reference", "synthetic": false}
  run      transcribe every manifest item and score it:
           --engine izwi (default): POST /v1/audio/transcriptions to a running
             server (--base-url) or to one this script starts (--backend
             --binary --models-dir --out, through nt_harness.py);
           --engine vllm: POST /v1/chat/completions to `vllm serve` with the
             NeMo plugin (Leg B), with izwi's exact instruction, so the two
             are comparable on the same machine.
           --compare OTHER.json reports the difference per set and fails when
           this run is worse by more than --max-delta (0.3 absolute points;
           plan §10.5: izwi within +0.3 of Leg B).

Scoring: corpus WER (CER for ja) after normalization: Whisper's English
normalizer when `whisper_normalizer` is installed (else a basic one) for
English; casefold + punctuation/symbol removal elsewhere. The mini sets are
noisy; the card comparison is context, the Leg B comparison is the gate.

Items marked "synthetic": true (TTS or other generated audio) are scored only
after a Whisper intelligibility gate (faster-whisper or openai-whisper when
installed; lessons 2026-10-29): Whisper's WER on the item must be at most
--gate-max-wer, otherwise the item is excluded and reported.
"""
from __future__ import annotations

import argparse
import base64
import importlib.util
import io
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import time
import unicodedata
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
VARIANT = "Nemotron-3.5-Transcribe-30B-A4B-FP8"
LIBRISPEECH_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
FLEURS_BASE = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"
FLEURS = {"en": "en_us", "de": "de_de", "fr": "fr_fr", "es": "es_419", "ja": "ja_jp"}
LANGUAGE_NAMES = {"en": "English", "de": "German", "fr": "French", "es": "Spanish",
                  "ja": "Japanese"}
CARD = {"librispeech_test_clean": 1.09, "fleurs_en": 3.35, "fleurs_de": 2.73,
        "fleurs_fr": 3.16, "fleurs_es": 2.20, "fleurs_ja": 4.42}
# izwi prompt.rs READABLE_INSTRUCTION (Leg B requests use it verbatim).
READABLE = ("Audio transcription. Language: as spoken. Wording: omit fillers and accidental "
            "repetitions; keep the remaining spoken wording unchanged. Formatting: punctuation "
            "and capitalization. Output: transcript only.")


def log(message):
    print(f"[wer-mini] {message}", file=sys.stderr, flush=True)


def load_harness():
    spec = importlib.util.spec_from_file_location("nt_harness", HERE / "nt_harness.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- prepare

def to_wav(ffmpeg, data, suffix, target):
    with tempfile.NamedTemporaryFile(suffix=suffix) as handle:
        handle.write(data)
        handle.flush()
        subprocess.run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i",
                        handle.name, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                        str(target)], check=True)


def open_url(url):
    request = urllib.request.Request(url, headers={"User-Agent": "izwi-wer-mini"})
    return urllib.request.urlopen(request, timeout=600)


def prepare_librispeech(out, count, ffmpeg, local_tar=None):
    items, audio, transcripts = [], {}, {}
    source = open(local_tar, "rb") if local_tar else open_url(LIBRISPEECH_URL)
    with source, tarfile.open(fileobj=source, mode="r|gz") as archive:
        for member in archive:
            name = member.name
            if name.endswith(".trans.txt"):
                for line in archive.extractfile(member).read().decode().splitlines():
                    key, _, text = line.partition(" ")
                    transcripts[key] = text
            elif name.endswith(".flac") and len(audio) < count:
                audio[Path(name).stem] = archive.extractfile(member).read()
            if len(audio) >= count and all(k in transcripts for k in audio):
                break
    for key in sorted(audio)[:count]:
        target = out / "audio" / f"librispeech-{key}.wav"
        if not target.is_file():
            to_wav(ffmpeg, audio[key], ".flac", target)
        items.append({"id": f"librispeech-{key}", "set": "librispeech_test_clean",
                      "language": "en", "audio": str(target.relative_to(out)),
                      "reference": transcripts[key], "synthetic": False})
    return items


def prepare_fleurs(out, language, count, ffmpeg):
    code = FLEURS[language]
    rows = {}
    with open_url(f"{FLEURS_BASE}/{code}/test.tsv") as response:
        for line in response.read().decode().splitlines():
            columns = line.split("\t")
            if len(columns) >= 3:
                rows[columns[1]] = columns[2]   # file name -> raw transcription
    picked = {}
    with open_url(f"{FLEURS_BASE}/{code}/audio/test.tar.gz") as response, \
            tarfile.open(fileobj=response, mode="r|gz") as archive:
        for member in archive:
            name = Path(member.name).name
            if member.isfile() and name in rows and name not in picked:
                picked[name] = archive.extractfile(member).read()
                if len(picked) >= count:
                    break
    items = []
    for name, data in picked.items():
        target = out / "audio" / f"fleurs-{language}-{Path(name).stem}.wav"
        if not target.is_file():
            to_wav(ffmpeg, data, ".wav", target)
        items.append({"id": f"fleurs-{language}-{Path(name).stem}", "set": f"fleurs_{language}",
                      "language": language, "audio": str(target.relative_to(out)),
                      "reference": rows[name], "synthetic": False})
    return items


def cmd_prepare(args):
    out = Path(args.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    items = prepare_librispeech(out, args.librispeech, args.ffmpeg, args.librispeech_tar)
    log(f"LibriSpeech test-clean: {len(items)} utterances")
    for language in args.languages.split(","):
        more = prepare_fleurs(out, language, args.fleurs_per_language, args.ffmpeg)
        log(f"FLEURS {language}: {len(more)} utterances")
        items += more
    with open(out / "manifest.jsonl", "w") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    log(f"wrote {out / 'manifest.jsonl'} ({len(items)} items)")
    return 0


# ---------------------------------------------------------------- scoring

def english_normalizer():
    try:
        from whisper_normalizer.english import EnglishTextNormalizer

        return EnglishTextNormalizer(), "whisper_normalizer.EnglishTextNormalizer"
    except Exception:  # noqa: BLE001
        return None, "basic"


def basic_normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return re.sub(r"\s+", " ", text).strip()


def units(text, language, english):
    if language == "en" and english is not None:
        text = english(text)
    text = basic_normalize(text)
    if language in ("ja", "zh", "th", "ko"):
        return [c for c in text if not c.isspace()]
    return text.split()


def edit_distance(ref, hyp):
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        current = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h))
        previous = current
    return previous[-1]


def whisper_gate(path):
    """Whisper's transcript of `path`, or None when no Whisper is installed."""
    try:
        from faster_whisper import WhisperModel

        model = WhisperModel("small", device="auto")
        segments, _ = model.transcribe(str(path))
        return " ".join(s.text for s in segments)
    except Exception:  # noqa: BLE001
        pass
    try:
        import whisper

        return whisper.load_model("small").transcribe(str(path))["text"]
    except Exception:  # noqa: BLE001
        return None


# -------------------------------------------------------------- engines

def izwi_transcribe(harness, base, item, path, language_hint, timeout):
    fields = [("model", VARIANT), ("response_format", "json")]
    if language_hint:
        fields.append(("language", item["language"]))
    body, content_type = harness.multipart(fields, [("file", path.name, path.read_bytes(),
                                                     "audio/wav")])
    status, raw, elapsed = harness.http("POST", base + "/v1/audio/transcriptions", body,
                                        {"Content-Type": content_type}, timeout)
    payload = harness.parse(raw)
    if status != 200:
        raise RuntimeError(f"{status}: {harness.error_message(payload)}")
    return payload.get("text", ""), elapsed


def vllm_transcribe(harness, base, item, path, language_hint, timeout, model):
    instruction = READABLE
    if language_hint:
        instruction = READABLE.replace("Language: as spoken.",
                                       f"Language: {LANGUAGE_NAMES[item['language']]}.")
    audio = base64.b64encode(path.read_bytes()).decode()
    body = {"model": model, "temperature": 0, "top_p": 1.0, "max_tokens": 4096, "seed": 0,
            "chat_template_kwargs": {"enable_thinking": False}, "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": instruction},
                {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio}"}}]}]}
    status, raw, elapsed = harness.http("POST", base + "/v1/chat/completions",
                                        json.dumps(body).encode(),
                                        {"Content-Type": "application/json"}, timeout)
    payload = harness.parse(raw)
    if status != 200:
        raise RuntimeError(f"{status}: {harness.error_message(payload)}")
    content = payload["choices"][0]["message"].get("content") or ""
    content = re.sub(r"<(ev|sty):[^>]*>", " ", content)
    return content, elapsed


def cmd_run(args):
    harness = load_harness()
    manifest = Path(args.manifest)
    root = manifest.parent
    items = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    english, normalizer = english_normalizer()
    server = None
    base = args.base_url
    if args.engine == "izwi" and not base:
        if not (args.backend and args.binary and args.models_dir and args.out):
            raise SystemExit("give --base-url, or --backend --binary --models-dir --out")
        _, sha = harness.read_build_info(args.out, None)
        server = harness.Server(args.binary, args.backend, args.models_dir, args.out, "wer",
                                sha).start()
        status, body, _ = server.post(f"/v1/admin/models/{VARIANT}/load", None, 3600)
        if status != 200:
            raise SystemExit(f"load failed: {status} {harness.error_message(body)}")
        base = server.base
    if not base:
        raise SystemExit("--engine vllm needs --base-url (the vllm serve address)")
    results, sets = [], {}
    try:
        for index, item in enumerate(items):
            path = root / item["audio"]
            record = {"id": item["id"], "set": item["set"], "language": item["language"],
                      "reference": item["reference"]}
            if item.get("synthetic"):
                heard = whisper_gate(path)
                if heard is None:
                    record.update(excluded="no Whisper installed for the synthetic-input gate")
                    results.append(record)
                    continue
                ref = units(item["reference"], item["language"], english)
                gate = edit_distance(ref, units(heard, item["language"], english)) / max(1, len(ref))
                record["whisper_gate_wer"] = round(gate, 4)
                if gate > args.gate_max_wer:
                    record.update(excluded=f"Whisper WER {gate:.2f} > {args.gate_max_wer}")
                    results.append(record)
                    continue
            try:
                if args.engine == "izwi":
                    text, elapsed = izwi_transcribe(harness, base, item, path,
                                                    not args.no_language_hint, args.timeout)
                else:
                    text, elapsed = vllm_transcribe(harness, base, item, path,
                                                    not args.no_language_hint, args.timeout,
                                                    args.vllm_model)
            except Exception as error:  # noqa: BLE001
                record.update(error=str(error))
                results.append(record)
                continue
            ref = units(item["reference"], item["language"], english)
            hyp = units(text, item["language"], english)
            errors = edit_distance(ref, hyp)
            record.update(hypothesis=text, errors=errors, units=len(ref),
                          elapsed_s=round(elapsed, 3))
            results.append(record)
            bucket = sets.setdefault(item["set"], {"n": 0, "errors": 0, "units": 0,
                                                   "metric": "cer" if item["language"] in
                                                   ("ja", "zh", "th", "ko") else "wer"})
            bucket["n"] += 1
            bucket["errors"] += errors
            bucket["units"] += len(ref)
            if (index + 1) % 20 == 0:
                log(f"{index + 1}/{len(items)} done")
    finally:
        if server is not None:
            server.stop()
    for name, bucket in sets.items():
        bucket["rate"] = round(100.0 * bucket["errors"] / max(1, bucket["units"]), 3)
        bucket["card"] = CARD.get(name)
        bucket["delta_vs_card"] = None if bucket["card"] is None else \
            round(bucket["rate"] - bucket["card"], 3)
    report = {"schema": "izwi.nemotron_transcribe.wer_mini", "schema_version": 1,
              "engine": args.engine, "model": VARIANT if args.engine == "izwi" else args.vllm_model,
              "base_url": base, "normalizer": normalizer,
              "language_hint": not args.no_language_hint, "sets": sets,
              "errors": sum(1 for r in results if "error" in r),
              "excluded": sum(1 for r in results if "excluded" in r), "items": results}
    status = 0
    if args.compare:
        other = json.loads(Path(args.compare).read_text())
        comparison = {}
        for name, bucket in sets.items():
            theirs = (other.get("sets") or {}).get(name)
            if not theirs:
                continue
            delta = round(bucket["rate"] - theirs["rate"], 3)
            comparison[name] = {"this": bucket["rate"], "other": theirs["rate"], "delta": delta,
                                "pass": delta <= args.max_delta}
            if delta > args.max_delta:
                status = 1
        report["compare"] = {"other": str(args.compare), "other_engine": other.get("engine"),
                             "max_delta": args.max_delta, "sets": comparison}
    if report["errors"]:
        status = 1
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    for name, bucket in sorted(sets.items()):
        log(f"{name}: {bucket['metric'].upper()} {bucket['rate']}% over {bucket['n']} "
            f"(card {bucket['card']})")
    if args.report:
        harness.merge_section(args.report, f"wer_{args.engine}", report)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--out", required=True)
    prep.add_argument("--librispeech", type=int, default=100)
    prep.add_argument("--librispeech-tar", help="a local test-clean.tar.gz instead of openslr")
    prep.add_argument("--languages", default="en,de,fr,es,ja")
    prep.add_argument("--fleurs-per-language", type=int, default=20)
    prep.add_argument("--ffmpeg", default="ffmpeg")
    run = sub.add_parser("run")
    run.add_argument("--manifest", required=True)
    run.add_argument("--engine", choices=("izwi", "vllm"), default="izwi")
    run.add_argument("--base-url")
    run.add_argument("--backend", choices=("cpu", "metal", "cuda"))
    run.add_argument("--binary")
    run.add_argument("--models-dir")
    run.add_argument("--out", help="the backend's report directory (server logs, build-info)")
    run.add_argument("--vllm-model", default="nemotron-transcribe",
                     help="the served model name of the vLLM server")
    run.add_argument("--no-language-hint", action="store_true",
                     help="do not send each item's language")
    run.add_argument("--timeout", type=int, default=1800)
    run.add_argument("--gate-max-wer", type=float, default=0.25)
    run.add_argument("--compare", help="another wer-mini JSON (e.g. the vLLM run)")
    run.add_argument("--max-delta", type=float, default=0.3)
    run.add_argument("--output", required=True)
    run.add_argument("--report", help="merge into this report.json as wer_<engine>")
    args = parser.parse_args(argv)
    return cmd_prepare(args) if args.command == "prepare" else cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
