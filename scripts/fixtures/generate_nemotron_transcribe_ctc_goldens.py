#!/usr/bin/env python3
"""Generate the Nemotron 3.5 Transcribe CTC word-timestamp goldens from upstream code.

What it does
------------
1. Loads NeMo's `nemo/collections/speechlm2/parts/ctc_timestamp_utils.py`
   (NeMo Speech @ 743beaf4) straight from its file, with NeMo, OmegaConf and
   the torch CTC head stubbed out: only the aligner's pure host functions run
   (`MultiSpeakerSOTWordTimestampAligner._prepare` / `align_prepared`,
   `_banded_ctc_search`, `_one_pass_speaker_mapping`,
   `_refine_word_timestamps`, `_character_token_ids`, ...). The aligner is
   built as vLLM serves it: `speaker_logprob_weight` 1.0, every other
   hyperparameter at its constructor default, and a CTC head with
   `upsampling_factor` 2 and 23,863 classes (blank 23,862), so the refinement
   cap is 24 CTC frames. NeMo's tokenizer wrapper is replaced by its
   `legacy=False` behaviour: `text_to_ids` = sentencepiece `encode_as_ids`.
   vLLM's `_public_result` / `_public_diarization` / `_seconds` (from
   `vllm/salm/ctc_timestamps.py`, same revision) are copied below verbatim.
   Numba is used when it imports (serving installs have it), else NumPy's
   loop; every search is also re-run with `_ctc_banded_forward_numpy` and
   must agree.
2. Builds deterministic synthetic cases: CTC log-probabilities with planted
   alignments (quantized: value = -code/16, codes u8, so F32 is exact),
   80 ms speaker sigmoids (value = code/256) with overlapping speakers, 10 ms
   diarization labels (run-length coded), transcripts with speaker tags,
   events, styles, words before the first tag, no tags, Chinese characters,
   unmapped and unalignable streams, a zero duration, short (< 1000 frames,
   every state in the band) and long (>= 1000 frames, banded) timelines, and
   transcripts upstream rejects while preparing.
3. Also records: sentencepiece `encode_as_ids` for ~45 strings and
   `_character_token_ids` for a few characters, SciPy
   `linear_sum_assignment` on small tie-heavy matrices, Python
   `round(x, 3)`, `_resample_speaker_probs` on small grids, and the
   `unicodedata.name` prefix ranges `_timestamp_units` detects scripts with.
4. Writes `crates/izwi-core/tests/fixtures/nemotron_transcribe/ctc_goldens.json`
   (about 150 KB; schema below). izwi's `ctc::golden_tests` replays it.

Usage (any macOS/Linux box, CPU only; numpy >= 2, torch, sentencepiece, scipy):

    python3 -I scripts/fixtures/generate_nemotron_transcribe_ctc_goldens.py \\
        --upstream /path/to/NeMo/nemo/collections/speechlm2/parts/ctc_timestamp_utils.py \\
        (--model-dir /path/to/NVIDIA-Nemotron-3.5-Transcribe | --spm /path/to/ctc_spm.model) \\
        [--out crates/izwi-core/tests/fixtures/nemotron_transcribe/ctc_goldens.json]

`--model-dir` reads the CTC SentencePiece model from the U8 tensor
`ctc_timestamp.tokenizer.model_proto` of `model-ctc-bf16.safetensors`;
`--spm` takes that proto already extracted. The model never enters the
fixture: tokenizer goldens are ids only.

Platform note: upstream's float rounding is reproduced as NumPy/torch build
it on the machine that runs this script. On arm64 (the published fixture)
`np.interp` and torch's CPU `linspace` fuse their multiply-adds; izwi
follows that. The fixture records the platform.

ctc_goldens.json schema (`schema` = "izwi.nemotron_transcribe.ctc_goldens",
`schema_version` = 1):

{
  "schema", "schema_version", "generator", "created_at", "seed",
  "versions": {python, platform, machine, numpy, torch, sentencepiece, scipy,
               numba (null when unavailable), upstream_sha256},
  "blank": 23862, "classes": 23863,
  "config": {speaker_activity_threshold, speaker_logprob_weight,
             maximum_token_len, max_word_frames, epsilon},
  "tokenizer": {"pieces": 23862, "unk_id": 0,
                "encode": [{"text", "ids"}],
                "character_ids": [{"character", "ids" | null}]},
  "unicode_name_ranges": {"ko"|"ja"|"zh"|"th": [[first, last], ...]},
  "rounding": [[repr(value), repr(round(value, 3))], ...],
  "assignment": [{"rows", "cols", "cost": [row-major], "pairs": [[r, c]]}],
  "resample": [{"frames", "speakers", "target", "codes": [u8 row-major],
                "output": [f32 row-major]}],
  "cases": [{
    "name", "transcript", "language": null, "ctc_frames", "audio_duration",
    "prepare_error": null | "alignment_failed",
    # present when prepare succeeded:
    "words": [[word, speaker_tag (str) | null, [ids], character (bool)]],
    "columns": [ids], "streams": [{"speaker_tag", "words", "labels",
                                   "state_to_word"}],
    "logp_codes": base64(zlib(u8 ctc_frames x columns)),  # value = -code/16
    "speaker": null | {"frames", "speakers",
                       "codes": base64(zlib(u8 frames x speakers))},  # code/256
    "diarization": null | {"frames", "speakers",
                           "runs": [[[start, end), ...] per speaker]},
    "search": [{"speaker_tag", "column", "retained_frames", "states",
                "half_width", "failed", "score" | null,
                "starts": delta string (first 0, then 0/1/2 per frame) | null,
                "path": [[state, count], ...] | null}],
    "raw_words": [{"word", "speaker", "start", "end"}],  # before refinement
    "expected": {"words", "diarization", "speaker_tag_to_diarization_speaker",
                 "error"}                                 # vLLM's public object
  }]
}
"""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import struct
import sys
import tempfile
import types
import unicodedata
import zlib
from pathlib import Path

import numpy as np
import torch

SCHEMA = "izwi.nemotron_transcribe.ctc_goldens"
BLANK = 23862
CLASSES = 23863
UPSAMPLING = 2
DEFAULT_OUT = "crates/izwi-core/tests/fixtures/nemotron_transcribe/ctc_goldens.json"
PROTO_TENSOR = "ctc_timestamp.tokenizer.model_proto"


# ---------------------------------------------------------------------------
# Upstream loading
# ---------------------------------------------------------------------------


def _stub_module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def load_upstream(path: Path):
    """Import ctc_timestamp_utils.py with NeMo/OmegaConf stubbed out."""
    try:
        import numba  # noqa: F401
        from numba import jit

        @jit(nopython=True)
        def _probe(x):
            return x.sum()

        _probe(np.ones(2))
        numba_ok = True
    except Exception:  # noqa: BLE001 -- numba missing or incompatible with numpy
        numba_ok = False

    class _OmegaConf:
        @staticmethod
        def is_config(_value):
            return False

    _stub_module("omegaconf", OmegaConf=_OmegaConf)
    for name in [
        "nemo",
        "nemo.collections",
        "nemo.collections.asr",
        "nemo.collections.asr.modules",
        "nemo.collections.common",
        "nemo.collections.common.tokenizers",
        "nemo.core",
        "nemo.core.utils",
    ]:
        _stub_module(name)
    _stub_module("nemo.collections.asr.modules.conv_asr", ConvASRDecoder=torch.nn.Module)
    _stub_module("nemo.collections.asr.modules.transformer_encoder", TransformerEncoder=object)
    _stub_module("nemo.collections.common.tokenizers.sentencepiece_tokenizer", SentencePieceTokenizer=object)
    _stub_module("nemo.core.utils.optional_libs", NUMBA_AVAILABLE=numba_ok)

    spec = importlib.util.spec_from_file_location("upstream_ctc_timestamp_utils", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module, numba_ok


# vLLM glue, verbatim from nemo/collections/speechlm2/vllm/salm/ctc_timestamps.py @ 743beaf4.
def _public_result(result: dict) -> dict:
    words = [
        {
            "word": word["word"],
            "start": _seconds(word["start"]),
            "end": _seconds(word["end"]),
            "speaker": str(word["speaker"]),
        }
        for speaker_words in (result.get("speaker_word_timestamps") or {}).values()
        for word in speaker_words
    ]
    words.sort(key=lambda w: (w["start"], w["end"]))
    mapping = {
        str(tag): None if column is None else int(column)
        for tag, column in (result.get("speaker_tag_to_sortformer_column") or {}).items()
    }
    return {
        "words": words,
        "diarization": _public_diarization(result.get("diarization_timestamps") or ()),
        "speaker_tag_to_diarization_speaker": mapping,
        "error": None,
    }


def _public_diarization(segments) -> list:
    return [
        {"speaker": int(segment["speaker"]), "start": _seconds(segment["start"]), "end": _seconds(segment["end"])}
        for segment in segments
    ]


def _seconds(value) -> float:
    return round(float(value), 3)


def _empty_result(error=None, diarization=()) -> dict:
    return {"words": [], "diarization": list(diarization), "speaker_tag_to_diarization_speaker": {}, "error": error}


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def read_proto_from_safetensors(path: Path) -> bytes:
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
        entry = header[PROTO_TENSOR]
        if entry["dtype"] != "U8":
            raise SystemExit(f"{PROTO_TENSOR} is {entry['dtype']}, not U8")
        start, end = entry["data_offsets"]
        handle.seek(8 + header_len + start)
        return handle.read(end - start)


class Tokenizer:
    """NeMo's SentencePieceTokenizer with legacy=False, as the aligner calls it."""

    def __init__(self, proto: bytes):
        import sentencepiece as spm

        self.tokenizer = spm.SentencePieceProcessor(model_proto=proto)

    def text_to_ids(self, text):
        return self.tokenizer.encode_as_ids(text)


TOKENIZER_STRINGS = [
    "",
    " ",
    "   ",
    "hello",
    "hello world",
    "  hello   world  ",
    "Hello, World!",
    "<spk:0> hi there <spk:1> hey",
    "<ev:laughter>",
    "so <ev:laughter> funny",
    "<sty:stt:calm_reflective> okay <sty:end:calm_reflective>",
    "<spk:16> beyond the table",
    "<ev:unknown_event>",
    "<unk> <s> </s>",
    "café naïve résumé",
    "café",
    "Straße İstanbul ǅ",
    "ｈｅｌｌｏ ＷＯＲＬＤ",
    "Ⅻ ① ² ½ ﬁ",
    "12345 3.14 1,000,000",
    "x​y z　w",
    "tab\tnew\nline\r\n",
    "你好，世界",
    "今天天气很好。",
    "こんにちは、世界！",
    "カタカナ ｶﾀｶﾅ",
    "한국어 텍스트",
    "สวัสดีครับ",
    "Привет, мир",
    "مرحبا بالعالم",
    "नमस्ते दुनिया",
    "🦜🦝 emoji 🦜",
    "a<>b <<< >>>",
    "\x01\x02 control",
    "▁ leading lower block",
    "don't won't it's",
    "e-mail co-operate",
    "UPPER lower MiXeD",
    "a b c d e f g",
    "𠀀 龘 鬱",
    "the quick brown fox jumps over the lazy dog",
    "ein Mädchen läuft über die Straße",
    "el niño está aquí",
    "Đặng Thị Ánh",
]

CHARACTERS = ["你", "好", "世", "界", "今", "气", "こ", "ん", "ｶ", "龘", "𠀀", "a", "x", "é", "²", "1"]


# ---------------------------------------------------------------------------
# Synthetic inputs
# ---------------------------------------------------------------------------


def b64z(array: np.ndarray) -> str:
    return base64.b64encode(zlib.compress(np.ascontiguousarray(array, dtype=np.uint8).tobytes(), 9)).decode()


def speaker_codes(rng: random.Random, frames: int, speakers: int, segments: dict) -> np.ndarray:
    """80 ms sigmoid codes: high inside each column's segments (in 80 ms frames), low outside."""
    codes = np.zeros((frames, speakers), np.uint8)
    for column in range(speakers):
        for t in range(frames):
            inside = any(a <= t < b for a, b in segments.get(column, ()))
            edge = any(t in (a - 1, b) for a, b in segments.get(column, ()))
            if inside:
                value = rng.randint(200, 254)
            elif edge:
                value = rng.randint(60, 140)
            elif column in segments:
                value = rng.choice([0, 0, 1, 2, 3, rng.randint(4, 30)])
            else:
                value = 0
            codes[t, column] = value
    return codes


def diarization_runs(rng: random.Random, frames: int, speakers: int, segments_80ms: dict) -> list:
    runs = []
    for column in range(speakers):
        column_runs = []
        for a, b in segments_80ms.get(column, ()):
            start = max(0, a * 8 + rng.randint(-3, 3))
            end = min(frames, b * 8 + rng.randint(-3, 3))
            if end > start:
                column_runs.append([start, end])
        runs.append(column_runs)
    return runs


def plant_log_probs(rng: random.Random, frames: int, columns: int, streams: list, active: dict) -> np.ndarray:
    """Codes (value = -code/16) with each stream's tokens planted in its speaker's active frames.

    `active[i]` lists the CTC frames stream i may emit in (time order).
    """
    codes = np.zeros((frames, columns), np.uint8)
    for t in range(frames):
        # Low-entropy background (keeps the fixture small): blank near 0,
        # every token near one of a few levels.
        codes[t, 0] = rng.randint(0, 3)
        base = rng.choice((96, 128, 160))
        for c in range(1, columns):
            codes[t, c] = base + rng.choice((0, 0, 0, 16))
    used = set()
    for index, stream in enumerate(streams):
        tokens = stream["labels"][1::2]
        frames_ok = [t for t in active[index] if t not in used]
        if not tokens or not frames_ok:
            continue
        stride = max(1, len(frames_ok) // (len(tokens) + 1))
        cursor = rng.randint(0, max(0, stride - 1))
        previous = None
        for token in tokens:
            if cursor >= len(frames_ok):
                break
            if previous == token:
                cursor += 1  # a repeated token needs a blank frame between
            if cursor >= len(frames_ok):
                break
            t = frames_ok[cursor]
            kind = rng.random()
            if kind < 0.08:
                # Weak emission: greedy picks blank here.
                codes[t, token] = rng.randint(20, 40)
                codes[t, 0] = rng.randint(4, 12)
            else:
                codes[t, token] = rng.randint(0, 4)
                codes[t, 0] = rng.randint(30, 90)
                if kind > 0.75 and cursor + 1 < len(frames_ok) and frames_ok[cursor + 1] == t + 1:
                    codes[t + 1, token] = rng.randint(2, 10)
                    codes[t + 1, 0] = rng.randint(30, 60)
                    used.add(t + 1)
            used.add(t)
            previous = token
            cursor += stride
    # Distractors: spurious greedy onsets of random tokens.
    for _ in range(max(1, frames // 60)):
        t = rng.randrange(frames)
        if t in used or columns < 2:
            continue
        c = rng.randrange(1, columns)
        codes[t, c] = rng.randint(0, 6)
        codes[t, 0] = rng.randint(20, 60)
    return codes


ENGLISH = (
    "we will look at the plan for next week and then talk about what the team needs "
    "from you right now so please let me know if that works for everyone here today"
).split()


def long_transcript(rng: random.Random, words: int, speakers: list) -> str:
    out, current = [], None
    for i in range(words):
        if speakers and (current is None or rng.random() < 0.08):
            current = rng.choice([s for s in speakers if s != current] or speakers)
            out.append(f"<spk:{current}>")
        if rng.random() < 0.03:
            out.append("<ev:laughter>")
        out.append(rng.choice(ENGLISH))
    return " ".join(out)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def case_specs(rng: random.Random) -> list:
    """name, transcript, ctc_frames, duration, speakers (None = no sigmoids),
    diarization (bool), segments per column (80 ms frames), tag -> column."""
    specs = [
        dict(
            name="two_speakers_short",
            transcript="<spk:0> hello there how are you doing <spk:1> i am fine thanks <ev:laughter> "
            "<spk:0> great to hear that",
            frames=400,
            speakers=4,
            segments={2: [(5, 95), (150, 195)], 0: [(90, 150)]},
            tags={"0": 2, "1": 0},
        ),
        dict(
            name="untagged_without_speaker_probs",
            transcript="the quick brown fox jumps over the lazy dog the end",
            frames=300,
            speakers=None,
        ),
        dict(
            name="prefix_events_and_styles",
            transcript="well <spk:1> <sty:stt:calm_reflective> so we begin now <sty:end:calm_reflective> "
            "<spk:0> okay <ev:breath> yes indeed <spk:1> right",
            frames=360,
            speakers=4,
            segments={1: [(2, 20)], 3: [(18, 100), (150, 178)], 0: [(95, 150)]},
            tags={"None": 1, "1": 3, "0": 0},
        ),
        dict(
            name="chinese_characters",
            transcript="<spk:0> 你好，世界 <spk:1> 今天天气很好 <ev:laughter> <spk:0> 好",
            frames=300,
            speakers=4,
            segments={1: [(4, 60), (120, 148)], 2: [(55, 125)]},
            tags={"0": 1, "1": 2},
        ),
        dict(
            name="japanese_characters",
            transcript="<spk:0> こんにちは <spk:1> ありがとう ございます",
            frames=240,
            speakers=4,
            segments={0: [(3, 50)], 3: [(52, 118)]},
            tags={"0": 0, "1": 3},
        ),
        dict(
            name="more_streams_than_columns",
            transcript="<spk:0> one two three <spk:1> four five six <spk:0> seven",
            frames=300,
            speakers=1,
            segments={0: [(2, 148)]},
            tags={"0": 0, "1": 0},
        ),
        dict(
            name="unalignable_stream_dropped",
            transcript="<spk:0> alpha beta gamma <spk:1> delta epsilon",
            frames=260,
            speakers=2,
            segments={0: [(2, 128)]},
            tags={"0": 0, "1": 0},
        ),
        dict(
            name="zero_duration",
            transcript="<spk:0> nothing to see here",
            frames=120,
            duration=0.0,
            speakers=2,
            segments={0: [(2, 58)]},
            tags={"0": 0},
        ),
        dict(
            name="only_a_tag",
            transcript="<spk:0>",
            frames=100,
            speakers=2,
            segments={1: [(3, 30)]},
            tags={},
        ),
        dict(
            name="too_long_for_the_timeline",
            transcript="<spk:0> " + " ".join(["internationalization"] * 12),
            frames=40,
            speakers=2,
            segments={0: [(1, 19)]},
            tags={"0": 0},
        ),
        dict(
            name="unrepresentable_character",
            transcript="<spk:0> 你好 龘",
            frames=120,
            speakers=2,
            segments={0: [(1, 59)]},
            tags={"0": 0},
        ),
        dict(
            name="long_two_speakers_overlapping",
            transcript=long_transcript(rng, 90, [0, 1]),
            frames=1240,
            speakers=4,
            segments={3: [(4, 520), (560, 616)], 1: [(240, 320), (450, 540), (580, 616)]},
            tags=None,  # filled from the transcript: tag 0 -> 3, tag 1 -> 1
            tag_columns={"0": 3, "1": 1},
        ),
        dict(
            name="long_untagged_speaker_probs_no_diarization",
            transcript=" ".join(rng.choice(ENGLISH) for _ in range(80)),
            frames=1100,
            speakers=2,
            segments={0: [(3, 540)]},
            tags={"0": 0},
            diarization=False,
        ),
    ]
    return specs


def run_case(up, aligner, tokenizer, spec, rng: random.Random) -> dict:
    frames = spec["frames"]
    duration = spec.get("duration", round(frames * 0.04 - rng.uniform(0.0, 0.06), 4))
    speakers = spec.get("speakers")
    segments = spec.get("segments", {})
    tags = spec.get("tags") or spec.get("tag_columns") or {}
    has_diarization = speakers is not None and spec.get("diarization", True)

    case = {
        "name": spec["name"],
        "transcript": spec["transcript"],
        "language": None,
        "ctc_frames": frames,
        "audio_duration": duration,
        "prepare_error": None,
    }

    sigmoid_frames = frames // UPSAMPLING
    sigmoids = None
    if speakers is not None:
        codes = speaker_codes(rng, sigmoid_frames, speakers, segments)
        case["speaker"] = {"frames": sigmoid_frames, "speakers": speakers, "codes": b64z(codes)}
        sigmoids = torch.from_numpy(codes.astype(np.float32) / 256.0)[None]
    else:
        case["speaker"] = None
    labels = None
    if has_diarization:
        diar_frames = int(duration / 0.01) + 4
        runs = diarization_runs(rng, diar_frames, speakers, segments)
        case["diarization"] = {"frames": diar_frames, "speakers": speakers, "runs": runs}
        dense = np.zeros((speakers, diar_frames), bool)
        for column, column_runs in enumerate(runs):
            for a, b in column_runs:
                dense[column, a:b] = True
        labels = torch.from_numpy(dense)[None]
    else:
        case["diarization"] = None

    def dummy_reader(columns_per_record, lengths):
        return [torch.zeros((length, len(columns))) for columns, length in zip(columns_per_record, lengths)]

    try:
        prepared = aligner._prepare(
            (1, frames, CLASSES),
            dummy_reader,
            sigmoids,
            [spec["transcript"]],
            audio_durations=[duration],
            diarization_labels=labels,
            diarization_frame_seconds=0.01,
        )
    except ValueError as error:
        case["prepare_error"] = "alignment_failed"
        case["prepare_message"] = str(error)
        diarization = []
        if labels is not None:
            diarization = _public_diarization(
                up.MultiSpeakerSOTWordTimestampAligner.diarization_segments(
                    labels[0], frame_seconds=0.01, time_offset=0.0, audio_duration=duration
                )
            )
        case["expected"] = _empty_result("alignment_failed", diarization)
        return case

    record = prepared["records"][0]
    # Upstream's columns: blank, then the sorted ids the words use (the
    # stream labels are already renumbered to positions in this list).
    columns = [BLANK] + sorted({int(i) for w in record["words"] for i in w["token_ids"]} - {BLANK})
    # Recover the original ids of the compact labels (prepared renumbers them).
    case["words"] = [
        [
            w["word"],
            None if w["speaker_tag"] is None else str(w["speaker_tag"]),
            [int(i) for i in w["token_ids"]],
            w.get("alignment_unit") == "character",
        ]
        for w in record["words"]
    ]
    case["columns"] = columns
    case["streams"] = [
        {
            "speaker_tag": None if s["speaker_tag"] is None else str(s["speaker_tag"]),
            "words": [w["word"] for w in s["words"]],
            "labels": [int(x) for x in s["labels"]],
            "state_to_word": s["state_to_word"],
        }
        for s in record["streams"]
    ]

    # Plant every stream in its speaker's active CTC frames.
    width = len(case["columns"])
    active = {}
    for index, stream in enumerate(record["streams"]):
        tag = "None" if stream["speaker_tag"] is None else str(stream["speaker_tag"])
        column = tags.get(tag)
        if column is None or speakers is None:
            active[index] = list(range(frames))
        else:
            active[index] = [t for t in range(frames) if any(a <= t // 2 < b for a, b in segments.get(column, ()))]
    if spec["name"] == "unalignable_stream_dropped":
        # Both streams speak while column 0 is active; column 1 never is.
        active = {i: list(range(4, 250)) for i in active}
    codes = plant_log_probs(rng, frames, width, record["streams"], active)
    case["logp_codes"] = b64z(codes)
    record["ctc_log_probs"] = torch.from_numpy(-(codes.astype(np.float32) / 16.0))

    # Capture each search and the result before refinement.
    searches, before = [], {}
    search_fn = up._banded_ctc_search
    refine_fn = up._refine_word_timestamps

    def search_capture(logp, labels_, positions, prior, greedy=None):
        path, score, starts = search_fn(logp, labels_, positions, prior, greedy)
        # Cross-check the NumPy loop against whichever forward ran.
        if len(positions) and np.isfinite(score):
            half = len(labels_) if len(positions) < 1000 else min(len(labels_), math.ceil(0.10 * len(labels_) + 100))
            width_ = min(len(labels_), 2 * half + 1)
            prev, ptr = up._ctc_banded_forward_numpy(logp, labels_, positions, prior, starts, width_)
            path2, score2 = up._ctc_banded_backtrace(prev, ptr, starts, len(labels_))
            assert np.array_equal(path, path2) and score == score2, "numba and numpy searches disagree"
        searches.append((path, score, starts, len(positions), len(labels_)))
        return path, score, starts

    def refine_capture(result, max_frames=12):
        before["result"] = result
        return refine_fn(result, max_frames)

    up._banded_ctc_search = search_capture
    up._refine_word_timestamps = refine_capture
    try:
        try:
            result = aligner.align_prepared(prepared)[0]
            public = _public_result(result)
        except ValueError as error:
            case["align_message"] = str(error)
            diarization = []
            if record["diarization_labels"] is not None:
                diarization = _public_diarization(
                    aligner.diarization_segments(
                        record["diarization_labels"],
                        frame_seconds=prepared["diarization_frame_seconds"],
                        time_offset=record["time_offset"],
                        audio_duration=record["audio_duration"],
                    )
                )
            public = _empty_result("alignment_failed", diarization)
    finally:
        up._banded_ctc_search = search_fn
        up._refine_word_timestamps = refine_fn

    pre = before.get("result")
    mapping = pre["speaker_tag_to_sortformer_column"] if pre else {}
    search_rows = []
    for stream, (path, score, starts, retained, states) in zip(record["streams"], searches):
        failed = not np.isfinite(score)
        half = states if retained < 1000 else min(states, math.ceil(0.10 * states + 100))
        row = {
            "speaker_tag": None if stream["speaker_tag"] is None else str(stream["speaker_tag"]),
            "column": mapping.get(stream["speaker_tag"]),
            "retained_frames": int(retained),
            "states": int(states),
            "half_width": int(half),
            "failed": bool(failed),
            "score": None if failed else float(score),
            "starts": None,
            "path": None,
        }
        if not failed:
            deltas = np.diff(np.asarray(starts, np.int64))
            assert starts[0] == 0 and deltas.min(initial=0) >= 0 and deltas.max(initial=0) <= 2
            row["starts"] = "0" + "".join(str(int(d)) for d in deltas)
            rle = []
            for state in path.tolist():
                if rle and rle[-1][0] == state:
                    rle[-1][1] += 1
                else:
                    rle.append([int(state), 1])
            row["path"] = rle
        search_rows.append(row)
    case["search"] = search_rows
    case["raw_words"] = (
        [
            {"word": w["word"], "speaker": str(w["speaker"]), "start": float(w["start"]), "end": float(w["end"])}
            for words in (pre["speaker_word_timestamps"] or {}).values()
            for w in words
        ]
        if pre
        else []
    )
    case["expected"] = public
    return case


# ---------------------------------------------------------------------------
# Small goldens
# ---------------------------------------------------------------------------


def name_ranges() -> dict:
    groups = {
        "ko": ("HANGUL",),
        "ja": ("HIRAGANA", "KATAKANA"),
        "zh": ("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH"),
        "th": ("THAI",),
    }
    out = {}
    for key, prefixes in groups.items():
        ranges = []
        for code in range(0x110000):
            if 0xD800 <= code < 0xE000:
                continue
            if unicodedata.name(chr(code), "").startswith(prefixes):
                if ranges and ranges[-1][1] == code - 1:
                    ranges[-1][1] = code
                else:
                    ranges.append([code, code])
        out[key] = ranges
    return out


def rounding_cases(rng: random.Random) -> list:
    values = [0.0625, 0.0635, 1.0005, 2.675, 0.0005, 0.0015, 0.0025, 1e-9, 123.4565, 3599.9995]
    values += [k * 0.0005 for k in range(0, 200, 7)]
    values += [k * 0.04 for k in range(0, 3000, 37)]
    values += [k * 0.01 for k in range(0, 4000, 53)]
    values += [rng.uniform(0, 4000) for _ in range(60)]
    # As strings: serde_json's default float parser can be one ulp off for
    # 17-digit decimals, and these goldens are about the last digit.
    return [[repr(v), repr(round(v, 3))] for v in values]


def assignment_cases(rng: random.Random) -> list:
    from scipy.optimize import linear_sum_assignment

    cases = []
    for trial in range(40):
        rows = rng.randint(1, 5)
        cols = rng.randint(1, 5)
        if trial % 3 == 0:
            cost = [float(rng.randint(-2, 2)) for _ in range(rows * cols)]
        elif trial % 3 == 1:
            cost = [float(rng.choice([0.0, 0.0, -0.5, 1.25])) for _ in range(rows * cols)]
        else:
            cost = [rng.uniform(-1, 1) for _ in range(rows * cols)]
        r, c = linear_sum_assignment(np.asarray(cost).reshape(rows, cols))
        cases.append({"rows": rows, "cols": cols, "cost": cost, "pairs": [[int(a), int(b)] for a, b in zip(r, c)]})
    return cases


def resample_cases(up, rng: random.Random) -> list:
    cases = []
    for frames, speakers, target in [(5, 2, 10), (13, 1, 26), (64, 2, 128), (7, 3, 5), (1, 2, 4), (9, 2, 1)]:
        codes = np.array([[rng.randint(0, 255) for _ in range(speakers)] for _ in range(frames)], np.uint8)
        probs = torch.from_numpy(codes.astype(np.float32) / 256.0)
        out = up.MultiSpeakerSOTWordTimestampAligner._resample_speaker_probs(probs.clamp(0, 1), target)
        cases.append(
            {
                "frames": frames,
                "speakers": speakers,
                "target": target,
                "codes": codes.reshape(-1).tolist(),
                "output": [float(x) for x in out.reshape(-1).tolist()],
            }
        )
    return cases


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--upstream", required=True, type=Path, help="NeMo's ctc_timestamp_utils.py @ 743beaf4")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-dir", type=Path, help="checkpoint dir holding model-ctc-bf16.safetensors")
    source.add_argument("--spm", type=Path, help="the CTC SentencePiece model, already extracted")
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args()

    os.environ.setdefault("NUMBA_CACHE_DIR", tempfile.mkdtemp(prefix="ctc-goldens-numba-"))
    up, numba_ok = load_upstream(args.upstream)
    proto = (
        read_proto_from_safetensors(args.model_dir / "model-ctc-bf16.safetensors")
        if args.model_dir
        else args.spm.read_bytes()
    )
    tokenizer = Tokenizer(proto)
    processor = tokenizer.tokenizer
    if processor.get_piece_size() != BLANK:
        raise SystemExit(f"expected {BLANK} pieces, got {processor.get_piece_size()}")

    aligner = up.MultiSpeakerSOTWordTimestampAligner(tokenizer=tokenizer, blank_id=BLANK)
    # The published head: 80 ms -> 40 ms, 23,863 classes. Only these two attributes are read.
    aligner.ctc_decoder = types.SimpleNamespace(upsampling_factor=UPSAMPLING, num_classes_with_blank=CLASSES)

    rng = random.Random(args.seed)
    character_ids = []
    for character in CHARACTERS:
        try:
            ids = up._character_token_ids(processor, character, BLANK)
        except ValueError:
            ids = None
        character_ids.append({"character": character, "ids": ids})

    cases = [run_case(up, aligner, tokenizer, spec, rng) for spec in case_specs(rng)]

    import scipy
    import sentencepiece

    try:
        import numba

        numba_version = numba.__version__ if numba_ok else None
    except Exception:  # noqa: BLE001
        numba_version = None
    payload = {
        "schema": SCHEMA,
        "schema_version": 1,
        "generator": "scripts/fixtures/generate_nemotron_transcribe_ctc_goldens.py",
        "created_at": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
        "seed": args.seed,
        "versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "sentencepiece": sentencepiece.__version__,
            "scipy": scipy.__version__,
            "numba": numba_version,
            "unicodedata": unicodedata.unidata_version,
            "upstream_sha256": hashlib.sha256(args.upstream.read_bytes()).hexdigest(),
        },
        "blank": BLANK,
        "classes": CLASSES,
        "config": {
            "speaker_activity_threshold": aligner.speaker_activity_threshold,
            "speaker_logprob_weight": aligner.speaker_logprob_weight,
            "maximum_token_len": aligner.maximum_token_len,
            "max_word_frames": aligner._ctc_frames(aligner.max_word_frames),
            "epsilon": aligner.epsilon,
        },
        "tokenizer": {
            "pieces": processor.get_piece_size(),
            "unk_id": processor.unk_id(),
            "encode": [{"text": text, "ids": tokenizer.text_to_ids(text)} for text in TOKENIZER_STRINGS],
            "character_ids": character_ids,
        },
        "unicode_name_ranges": name_ranges(),
        "rounding": rounding_cases(rng),
        "assignment": assignment_cases(rng),
        "resample": resample_cases(up, rng),
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    args.out.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(text) / 1024:.1f} KiB, {len(cases)} cases, numba={numba_ok})")
    for case in cases:
        expected = case["expected"]
        print(
            f"  {case['name']}: words={len(expected['words'])} diar={len(expected['diarization'])} "
            f"error={expected['error']} prepare_error={case['prepare_error']} "
            f"searches={[(s['retained_frames'], s['states'], s['failed']) for s in case.get('search', [])]}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
