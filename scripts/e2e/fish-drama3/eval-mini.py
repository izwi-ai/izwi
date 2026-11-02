#!/usr/bin/env python3
"""eval-mini for Fish Audio Drama 3: cloned speech scored for intelligibility
(WER, CER for Chinese) and speaker similarity (ECAPA), izwi beside the
official engine (Leg B) on the same machine.

  prepare  the item set: 100 LibriSpeech test-clean pairs (a reference
           utterance and another sentence of the same speaker; with
           --librispeech-pc-list, the punctuated LibriSpeech-PC
           cross-sentence pairs) and 25 + 25 Seed-TTS-eval en/zh items from a
           local seed-tts-eval test set (--seed-tts-dir). Writes
           <dir>/manifest.jsonl ({"id", "set", "language", "text",
           "reference_audio", "reference_text"}) and <dir>/audio/.
  synth    izwi speaks every item, cloned from its reference, through
           /v1/audio/speech of a running server (--base-url) or one it starts
           (--backend --binary --models-dir --out, through fd3_harness.py), at
           the sampling defaults (`defaults`: temperature 1.0, top-p 0.9, no
           main top-k) and at temperature 0.7 / top-p 0.7 / top-k 30
           (`fish`). Writes <wavs>/<profile>/<id>.wav and synth-<profile>.json.
  score    scores a directory of <id>.wav files: izwi's, or the WAVs your own
           Leg B run (sglang-omni or vllm-omni, run however you run them)
           wrote for the same manifest. WER with Whisper large-v3
           (faster-whisper or openai-whisper, whichever is installed) or an
           izwi ASR model (--asr izwi); speaker similarity with SpeechBrain's
           ECAPA (speechbrain/spkrec-ecapa-voxceleb) when installed.
           --compare OTHER.json gates on the acceptance bounds: WER at most
           0.5 points above the other run, similarity at most 0.02 below.

It calls nothing but this repository's server and the ASR and similarity
packages you install; no upstream Fish code runs. The card publishes no WER
or similarity numbers, so Leg B on the same machine is the only comparison.
"""
from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import unicodedata
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
VARIANT = "FishAudio-Drama-3"
LIBRISPEECH_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
# The sampling profiles: the family's defaults, and fish-speech's 0.7 / 0.7 / 30.
PROFILES = {"defaults": {}, "fish": {"temperature": 0.7, "top_p": 0.7, "top_k": 30}}
CJK = ("zh", "ja", "ko")


def log(message):
    print(f"[eval-mini] {message}", file=sys.stderr, flush=True)


def load_harness():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("fd3_harness", HERE / "fd3_harness.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- prepare

def to_wav(ffmpeg, data, suffix, target, rate=None):
    with tempfile.NamedTemporaryFile(suffix=suffix) as handle:
        handle.write(data)
        handle.flush()
        command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i",
                   handle.name, "-ac", "1"]
        if rate:
            command += ["-ar", str(rate)]
        subprocess.run(command + ["-c:a", "pcm_s16le", str(target)], check=True)


def sentence_case(text):
    """LibriSpeech's upper-case transcripts as a sentence a TTS model reads."""
    text = text.strip().lower()
    return (text[:1].upper() + text[1:] + ".") if text else text


def read_librispeech(source, wanted=None, limit_chapters=None, per_chapter=4):
    """{utterance id: flac bytes} and {utterance id: transcript} from the
    test-clean archive, streamed: only `wanted` ids when given, else the
    first `per_chapter` utterances of the first `limit_chapters` chapters."""
    audio, texts = {}, {}
    opener = open(source, "rb") if Path(str(source)).is_file() else urllib.request.urlopen(
        urllib.request.Request(str(source), headers={"User-Agent": "izwi-eval-mini"}),
        timeout=600)
    with opener as stream, tarfile.open(fileobj=stream, mode="r|gz") as archive:
        for member in archive:
            name = member.name
            if name.endswith(".trans.txt"):
                for line in archive.extractfile(member).read().decode().splitlines():
                    key, _, text = line.partition(" ")
                    texts[key] = text
            elif name.endswith(".flac"):
                key = Path(name).stem
                chapter = key.rsplit("-", 1)[0]
                if wanted is not None:
                    keep = key in wanted
                else:
                    keep = sum(1 for k in audio if k.rsplit("-", 1)[0] == chapter) < per_chapter
                if keep:
                    audio[key] = archive.extractfile(member).read()
            if wanted is not None and wanted <= set(audio) and wanted <= set(texts):
                break
            if limit_chapters and len({k.rsplit("-", 1)[0] for k in audio}) > limit_chapters:
                break
    return audio, texts


def prepare_librispeech(args, out):
    items = []
    if args.librispeech_pc_list:
        # F5-TTS's cross-sentence list: ref_utt, ref_dur, ref_txt, gen_utt, gen_dur, gen_txt.
        pairs = []
        for line in Path(args.librispeech_pc_list).read_text().splitlines():
            columns = line.split("\t")
            if len(columns) >= 6:
                pairs.append((columns[0], columns[2], columns[3], columns[5]))
            if len(pairs) >= args.librispeech:
                break
        audio, _ = read_librispeech(args.librispeech_tar, {p[0] for p in pairs})
        source = "librispeech_pc_cross_sentence"
    else:
        # Archive order: in each chapter, utterance 2k is the reference and
        # 2k + 1 the sentence to speak, two pairs a chapter.
        audio, texts = read_librispeech(args.librispeech_tar, None,
                                        limit_chapters=args.librispeech // 2 + 2)
        chapters = {}
        for key in sorted(k for k in audio if k in texts):
            chapters.setdefault(key.rsplit("-", 1)[0], []).append(key)
        pairs = []
        for keys in chapters.values():
            for k in range(0, min(4, len(keys) - 1), 2):
                pairs.append((keys[k], sentence_case(texts[keys[k]]), keys[k + 1],
                              sentence_case(texts[keys[k + 1]])))
        pairs = pairs[:args.librispeech]
        source = "librispeech_test_clean_sentence_cased"
    for ref_id, ref_text, gen_id, gen_text in pairs:
        if ref_id not in audio:
            raise SystemExit(f"{ref_id} is not in {args.librispeech_tar}")
        target = out / "audio" / f"librispeech-{ref_id}.wav"
        if not target.is_file():
            to_wav(args.ffmpeg, audio[ref_id], ".flac", target)
        items.append({"id": f"librispeech-{gen_id}", "set": "librispeech_test_clean",
                      "language": "en", "text": gen_text,
                      "reference_audio": str(target.relative_to(out)),
                      "reference_text": ref_text, "text_source": source})
    return items


def prepare_seed_tts(args, out, language):
    """The first N items of a seed-tts-eval `<language>/meta.lst`
    (utt|prompt_text|prompt_wav|target_text[|target_wav])."""
    root = Path(args.seed_tts_dir) / language
    meta = root / "meta.lst"
    if not meta.is_file():
        raise SystemExit(f"no {meta}: unpack the seed-tts-eval test set and pass --seed-tts-dir")
    items = []
    for line in meta.read_text(encoding="utf-8").splitlines():
        columns = line.strip().split("|")
        if len(columns) < 4:
            continue
        utt, prompt_text, prompt_wav, target_text = columns[:4]
        source = root / prompt_wav
        target = out / "audio" / f"seedtts-{language}-{utt}.wav"
        if not target.is_file():
            to_wav(args.ffmpeg, source.read_bytes(), source.suffix or ".wav", target)
        items.append({"id": f"seedtts-{language}-{utt}", "set": f"seed_tts_{language}",
                      "language": language, "text": target_text,
                      "reference_audio": str(target.relative_to(out)),
                      "reference_text": prompt_text, "text_source": "seed_tts_eval"})
        if len(items) >= args.seed_tts_per_language:
            break
    return items


def cmd_prepare(args):
    out = Path(args.out)
    languages = [lang for lang in args.languages.split(",") if lang]
    if args.dry_run:
        log(f"dry run: {args.librispeech} LibriSpeech pairs from {args.librispeech_tar}"
            + (f" with {args.librispeech_pc_list}" if args.librispeech_pc_list else
               " (sentence-cased transcripts; pass --librispeech-pc-list for LibriSpeech-PC)"))
        log(f"  {args.seed_tts_per_language} Seed-TTS items each of {languages} from "
            f"{args.seed_tts_dir or '(no --seed-tts-dir: skipped)'}; into {out}")
        return 0
    (out / "audio").mkdir(parents=True, exist_ok=True)
    items = prepare_librispeech(args, out) if args.librispeech else []
    log(f"LibriSpeech: {len(items)} items")
    if args.seed_tts_dir:
        for language in languages:
            more = prepare_seed_tts(args, out, language)
            log(f"Seed-TTS {language}: {len(more)} items")
            items += more
    else:
        log("no --seed-tts-dir: the Seed-TTS sets are left out")
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    log(f"wrote {out / 'manifest.jsonl'} ({len(items)} items)")
    return 0


def read_manifest(path):
    path = Path(path)
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    return path.parent, items


# ------------------------------------------------------------------ synth

def cmd_synth(args):
    root, items = read_manifest(args.manifest)
    if args.limit:
        items = items[:args.limit]
    profiles = list(PROFILES) if args.profile == "both" else [args.profile]
    out = Path(args.wavs)
    if args.dry_run:
        for profile in profiles:
            log(f"dry run: {len(items)} items at {profile} {PROFILES[profile] or '(defaults)'} "
                f"into {out / profile}, seeds {args.seed}..{args.seed + len(items) - 1}")
        return 0
    harness = load_harness()
    server, base = None, args.base_url
    if not base:
        if not (args.backend and args.binary and args.models_dir and args.out):
            raise SystemExit("give --base-url, or --backend --binary --models-dir --out")
        _, sha = harness.read_build_info(args.out, None)
        server = harness.Server(args.binary, args.backend, args.models_dir, args.out, "eval",
                                sha).start()
        status, body, _ = server.load(args.load_timeout)
        if status != 200:
            raise SystemExit(f"load failed: {status} {harness.error_message(body)}")
        base = server.base
    status = 0
    try:
        for profile in profiles:
            folder = out / profile
            folder.mkdir(parents=True, exist_ok=True)
            records = []
            for index, item in enumerate(items):
                target = folder / f"{item['id']}.wav"
                record = {"id": item["id"], "seed": args.seed + index}
                if target.is_file() and not args.overwrite:
                    record["skipped"] = "exists"
                    records.append(record)
                    continue
                body = dict(PROFILES[profile], model=VARIANT, input=item["text"],
                            response_format="wav", seed=args.seed + index,
                            reference_audio=base64.b64encode(
                                (root / item["reference_audio"]).read_bytes()).decode(),
                            reference_text=item["reference_text"])
                code, raw, _, elapsed = harness.http_call(
                    "POST", base + "/v1/audio/speech", json.dumps(body).encode(),
                    {"Content-Type": "application/json"}, args.timeout)
                record["elapsed_s"] = round(elapsed, 3)
                if code == 200:
                    target.write_bytes(raw)
                    rate, _, pcm = harness.wav_pcm(raw)
                    record["audio_s"] = round(len(pcm) / 2 / rate, 3)
                else:
                    record["error"] = f"HTTP {code}: {harness.error_message(harness.parse(raw))}"
                records.append(record)
                if (index + 1) % 25 == 0:
                    log(f"{profile}: {index + 1}/{len(items)}")
            errors = sum(1 for r in records if "error" in r)
            report = {"schema": "izwi.fish_drama3.eval_synth", "profile": profile,
                      "sampling": PROFILES[profile], "items": len(records), "errors": errors,
                      "records": records}
            (out / f"synth-{profile}.json").write_text(json.dumps(report, indent=1) + "\n")
            log(f"{profile}: {len(records)} items, {errors} errors")
            status |= 1 if errors else 0
    finally:
        if server is not None:
            server.stop()
    return status


# ------------------------------------------------------------------ score

def basic_normalize(text):
    text = re.sub(r"<\|[^|>]*\|>|\[[^\]]*\]", " ", text or "")
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return re.sub(r"\s+", " ", text).strip()


def english_normalizer():
    try:
        from whisper_normalizer.english import EnglishTextNormalizer

        return EnglishTextNormalizer(), "whisper_normalizer.EnglishTextNormalizer"
    except Exception:  # noqa: BLE001
        return None, "basic"


def units(text, language, english):
    if language == "en" and english is not None:
        text = english(re.sub(r"<\|[^|>]*\|>|\[[^\]]*\]", " ", text or ""))
    text = basic_normalize(text)
    if language in CJK:
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


class WhisperAsr:
    """Whisper large-v3 from faster-whisper or openai-whisper."""

    def __init__(self, size):
        self.name = None
        try:
            from faster_whisper import WhisperModel

            self.model, self.name = WhisperModel(size, device="auto"), f"faster-whisper {size}"
            self.kind = "faster"
            return
        except Exception:  # noqa: BLE001
            pass
        try:
            import whisper

            self.model, self.name = whisper.load_model(size), f"openai-whisper {size}"
            self.kind = "openai"
        except Exception as error:  # noqa: BLE001
            raise SystemExit("install faster-whisper or openai-whisper for --asr whisper, or "
                             f"use --asr izwi ({error})")

    def __call__(self, path, language):
        if self.kind == "faster":
            segments, _ = self.model.transcribe(str(path), language=language, beam_size=5)
            return " ".join(s.text for s in segments)
        return self.model.transcribe(str(path), language=language)["text"]


class IzwiAsr:
    def __init__(self, base, model):
        self.harness, self.base, self.model = load_harness(), base, model
        self.name = f"izwi {model}"

    def __call__(self, path, language):
        server = type("S", (), {"base": self.base})()
        return self.harness.transcribe(server, self.model, path)


class EcapaSimilarity:
    """Cosine similarity of SpeechBrain ECAPA embeddings at 16 kHz."""

    def __init__(self):
        try:
            import torch
            import torchaudio
            try:
                from speechbrain.inference.speaker import EncoderClassifier
            except ImportError:
                from speechbrain.pretrained import EncoderClassifier
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(f"speechbrain is not installed ({error})")
        self.torch, self.torchaudio = torch, torchaudio
        self.model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
        self.name = "speechbrain/spkrec-ecapa-voxceleb"

    def embed(self, path):
        audio, rate = self.torchaudio.load(str(path))
        audio = audio.mean(dim=0, keepdim=True)
        if rate != 16000:
            audio = self.torchaudio.functional.resample(audio, rate, 16000)
        return self.model.encode_batch(audio).squeeze()

    def __call__(self, a, b):
        x, y = self.embed(a), self.embed(b)
        return float(self.torch.nn.functional.cosine_similarity(x, y, dim=0))


def cmd_score(args):
    root, items = read_manifest(args.manifest)
    wavs = Path(args.wavs)
    asr = WhisperAsr(args.whisper_model) if args.asr == "whisper" else \
        IzwiAsr(args.asr_base, args.asr_model)
    similarity, sim_note = None, None
    if not args.no_sim:
        try:
            similarity = EcapaSimilarity()
        except RuntimeError as error:
            sim_note = str(error)
            log(f"no speaker similarity: {error}")
    english, normalizer = english_normalizer()
    results, sets = [], {}
    for index, item in enumerate(items):
        path = wavs / f"{item['id']}.wav"
        record = {"id": item["id"], "set": item["set"]}
        if not path.is_file():
            record["missing"] = True
            results.append(record)
            continue
        heard = asr(path, item["language"])
        ref, hyp = units(item["text"], item["language"], english), \
            units(heard, item["language"], english)
        errors = edit_distance(ref, hyp)
        record.update(heard=heard, errors=errors, units=len(ref))
        if similarity is not None:
            record["sim"] = round(similarity(path, root / item["reference_audio"]), 4)
        results.append(record)
        bucket = sets.setdefault(item["set"], {"n": 0, "errors": 0, "units": 0, "sims": [],
                                               "metric": "cer" if item["language"] in CJK
                                               else "wer"})
        bucket["n"] += 1
        bucket["errors"] += errors
        bucket["units"] += len(ref)
        if "sim" in record:
            bucket["sims"].append(record["sim"])
        if (index + 1) % 25 == 0:
            log(f"{index + 1}/{len(items)} scored")
    for bucket in sets.values():
        bucket["rate"] = round(100.0 * bucket["errors"] / max(1, bucket["units"]), 3)
        sims = bucket.pop("sims")
        bucket["sim"] = round(sum(sims) / len(sims), 4) if sims else None
    report = {"schema": "izwi.fish_drama3.eval_mini", "schema_version": 1,
              "label": args.label, "wavs": str(wavs), "asr": asr.name,
              "similarity": similarity.name if similarity else None,
              "similarity_note": sim_note, "normalizer": normalizer, "sets": sets,
              "missing": sum(1 for r in results if r.get("missing")), "items": results}
    status = 0
    if args.compare:
        other = json.loads(Path(args.compare).read_text())
        comparison = {}
        for name, bucket in sets.items():
            theirs = (other.get("sets") or {}).get(name)
            if not theirs:
                continue
            wer_delta = round(bucket["rate"] - theirs["rate"], 3)
            sim_delta = None if bucket["sim"] is None or theirs.get("sim") is None else \
                round(bucket["sim"] - theirs["sim"], 4)
            passed = wer_delta <= args.max_wer_delta and \
                (sim_delta is None or sim_delta >= args.min_sim_delta)
            comparison[name] = {"this": bucket["rate"], "other": theirs["rate"],
                                "wer_delta": wer_delta, "sim": bucket["sim"],
                                "other_sim": theirs.get("sim"), "sim_delta": sim_delta,
                                "pass": passed}
            status |= 0 if passed else 1
        report["compare"] = {"other": str(args.compare), "other_label": other.get("label"),
                             "max_wer_delta": args.max_wer_delta,
                             "min_sim_delta": args.min_sim_delta, "sets": comparison}
    if report["missing"]:
        status = 1
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    for name, bucket in sorted(sets.items()):
        log(f"{name}: {bucket['metric'].upper()} {bucket['rate']}% SIM {bucket['sim']} over "
            f"{bucket['n']}")
    if args.report:
        load_harness().merge_section(args.report, f"eval_{args.label}", report)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="build the item set")
    prep.add_argument("--out", required=True)
    prep.add_argument("--librispeech", type=int, default=100, help="pairs (0: none)")
    prep.add_argument("--librispeech-tar", default=LIBRISPEECH_URL,
                      help="test-clean.tar.gz, a local path or URL (default: openslr)")
    prep.add_argument("--librispeech-pc-list",
                      help="LibriSpeech-PC cross-sentence pairs (F5-TTS's "
                           "librispeech_pc_test_clean_cross_sentence.lst)")
    prep.add_argument("--seed-tts-dir", help="the seed-tts-eval test set, with en/ and zh/")
    prep.add_argument("--seed-tts-per-language", type=int, default=25)
    prep.add_argument("--languages", default="en,zh")
    prep.add_argument("--ffmpeg", default="ffmpeg")
    prep.add_argument("--dry-run", action="store_true")
    synth = sub.add_parser("synth", help="izwi speaks every item")
    synth.add_argument("--manifest", required=True)
    synth.add_argument("--wavs", required=True, help="writes <wavs>/<profile>/<id>.wav")
    synth.add_argument("--profile", choices=("defaults", "fish", "both"), default="both")
    synth.add_argument("--seed", type=int, default=1, help="item i samples from seed + i")
    synth.add_argument("--limit", type=int, default=0)
    synth.add_argument("--overwrite", action="store_true")
    synth.add_argument("--base-url")
    synth.add_argument("--backend", choices=("cpu", "metal", "cuda"))
    synth.add_argument("--binary")
    synth.add_argument("--models-dir")
    synth.add_argument("--out", help="the backend's report directory (build-info, logs)")
    synth.add_argument("--load-timeout", type=int, default=3600)
    synth.add_argument("--timeout", type=int, default=1800)
    synth.add_argument("--dry-run", action="store_true")
    score = sub.add_parser("score", help="WER and similarity of a directory of WAVs")
    score.add_argument("--manifest", required=True)
    score.add_argument("--wavs", required=True, help="a directory of <id>.wav")
    score.add_argument("--label", required=True, help="e.g. izwi-defaults, legb-defaults")
    score.add_argument("--output", required=True)
    score.add_argument("--asr", choices=("whisper", "izwi"), default="whisper")
    score.add_argument("--whisper-model", default="large-v3")
    score.add_argument("--asr-base", help="an izwi server for --asr izwi")
    score.add_argument("--asr-model", default="Qwen3-ASR-1.7B-GGUF")
    score.add_argument("--no-sim", action="store_true")
    score.add_argument("--compare", help="another score JSON (Leg B's) to gate against")
    score.add_argument("--max-wer-delta", type=float, default=0.5)
    score.add_argument("--min-sim-delta", type=float, default=-0.02)
    score.add_argument("--report", help="merge into this report.json as eval_<label>")
    args = parser.parse_args(argv)
    if args.command == "score" and args.asr == "izwi" and not args.asr_base:
        parser.error("--asr izwi needs --asr-base")
    return {"prepare": cmd_prepare, "synth": cmd_synth, "score": cmd_score}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
