#!/usr/bin/env python3
"""Generate the L2 whole-model reference for Fish Audio Drama 3 (plan §10.4, Leg A).

Model: fishaudio/drama-3 at revision 78e5ab79e49aa9a25abe8799eff2c4d73a1c3216
(the whole bundle, codec.pth included), run by upstream fish-speech at
0f8a12e8 in float32. Pattern: `generate_granite_speech5_reference.py`.

Leg B (the official engine, sglang-omni d9c3e708 or vllm-omni e45e43bc, W8A8
on SM89+, for the quality comparison) is NOT run here: it belongs to the
end-to-end kit (`scripts/e2e/fish-drama3/`, plan §10.5).

What it does
------------
1. Checks the bundle: config.json, tokenizer.json and the index hash to the
   pinned sha256, every shard and codec.pth has its pinned size
   (`--verify-shards` also hashes them against the LFS sha256, ~41 GB read).
2. Converts each reference clip ONCE with ffmpeg to 44.1 kHz mono raw f32
   (`<out>/audio/<clip>.f32`, little-endian), so izwi reads the very samples
   the reference encoded. Default clips (this repo's `data/`):
     fox   data/fox.wav  (3.6 s)   transcript data/fox.md
     test  data/test.wav (29.7 s)  transcript data/test.md
3. Encodes them with upstream's codec loader (`inference.load_codec_model`,
   strict load, weight norm folded) in float32 and `DAC.encode`, trimmed to
   ceil(samples / 2048) frames as upstream's `encode_audio` does: the
   reference codes [10, frames] are recorded, so izwi's trunk / tower parity
   does not depend on its codec encoder (that is checked separately against
   the same .f32 samples). The codec is freed before the LM loads.
4. Builds the LM like `FishOmniModel.from_pretrained`, except that the routed
   experts stay FP8-resident (`MoEExperts(fp8=True)`, ~30 GiB) and run
   upstream's dequantize path (W8A16: e4m3 * F32 block scale, F32 math) on
   every device: `FISH_DRAMA3_FP8_EXPERTS=0` would dequantize all 256 x 40
   experts to F32 (~120 GiB), so it is not used. Dense weights are F32
   (~16 GiB). Total resident ~46 GiB. Router weights are rounded to BF16
   (REFERENCE_SEMANTICS in the common module), TF32 and CUDA graphs are off.
5. Runs the cases below. Every upstream `generate_drama3` call ("a
   generation") runs under the recorded-noise sampler (same noise layout and
   order rules as the tiny fixture: candidate order, eos first, RAS draw
   always, heads 1..9, the eos step's head draws cut) with hooks recording
   its prefill, then a self-check: the cached decode's final-norm hidden
   equals one uncached forward over prompt + frames. Generation g of case i
   uses noise seed `--noise-seed + 100 * i + g`.
6. Frees the LM, reloads the codec in float32 and decodes the 10 s
   utterance's codes with upstream `inference.decode_to_audio` (a copy of the
   codes: upstream's RVQ decode clamps its input in place).

Cases (kind):
  plain          (single)    no reference; a sentence long enough for > 64 frames
  one_reference  (single)    fox as reference 1
  two_speakers   (single)    fox as <|speaker:0|>, test as <|speaker:1|>, a
                             three-line tagged dialogue
  long_text      (chunks)    fox as reference 1 and a 410-byte untagged text that
                             upstream's chunker (`split_drama3_text`, 300 bytes)
                             cuts into 3 chunks (65 | 288 | 57 bytes). It runs
                             upstream's own chunk loop `generate_long_drama3`
                             (`inference.py:912-1018`): the reference is always
                             kept, history = the pinned first chunk + the latest
                             chunk with the frames upstream itself generated
                             (`Drama3History`, `build_drama3_chunk_prompt`), each
                             chunk its own generation with `--chunk-frames`
                             (0 = upstream's 600 per chunk), clamped as upstream
                             clamps. Expected histories: [], [0], [0, 1].
  utterance_10s  (utterance) no reference; run to eos or the 10 s cap,
                             ceil(10 * 44100 / 2048) = 216 frames; its full
                             codes and upstream codec PCM are recorded
  take_edit      (take_edit) plan §8.4 (4d): the utterance as the previous take,
                             a completed history turn with its generated codes
                             (`build_drama3_prompt(..., history=[(text, codes)])`,
                             `generate.py:158-169`), then the edited text;
                             `--frames` frames. Needs utterance_10s.

Hardware and cost (all cases, defaults):
  * a >= 80 GB CUDA GPU (`--device cuda`): about 20-25 minutes;
  * a >= 64 GB RAM CPU box (`--device cpu`): about 1-1.5 hours.
  Peak RAM / VRAM stays ~52 GB: ~46 GiB of weights, then the codec (~1.9 GB)
  only once the LM is freed. About 1,000 decode steps in all; the sidecar
  holds ~45,000 noise values per frame, so `reference.safetensors` is about
  200-250 MB (`long_text` ~100 MB, `utterance_10s` up to ~40 MB). Never
  committed.

Environment: a full fish-speech install at 0f8a12e8 (its own `pip install -e .`
environment: torch >= 2.4, torchaudio, hydra-core, omegaconf,
descript-audio-codec, descript-audiotools, loguru, tqdm, click, safetensors,
tokenizers), plus ffmpeg on PATH (`--ffmpeg`). Get the code with
    git clone https://github.com/fishaudio/fish-speech /path/to/fish-speech
    git -C /path/to/fish-speech checkout 0f8a12e8a651062fdee9c33a894bdccd3a8513f1
and the weights at the pin with
    hf download fishaudio/drama-3 --revision 78e5ab79e49aa9a25abe8799eff2c4d73a1c3216 --local-dir DIR
(or the plan's `scripts/e2e/fish-drama3/fetch.sh` once it exists).

Usage:

    python3 -I scripts/fixtures/generate_fish_drama3_reference.py \\
        --model-dir DIR --fish-speech /path/to/fish-speech \\
        [--out target/fish-drama3-reference] [--device cuda|cuda:N|cpu] \\
        [--frames 64] [--chunk-frames 0] \\
        [--cases plain,one_reference,two_speakers,long_text,utterance_10s,take_edit] \\
        [--noise-seed 20261030] [--temperature 1.0 --top-p 0.9 --top-k 30] \\
        [--trunk-layers 0,1,3,39] [--tower-layers 0,4,12] \\
        [--data-dir data] [--ffmpeg ffmpeg] [--verify-shards] \\
        [--router-weights bf16|fp32] [--skip-commit-check] [--no-strict]

`IZWI_FISH_DRAMA3_REFERENCE_DIR=<out>` and `IZWI_FISH_DRAMA3_MODEL_DIR=DIR`
then run izwi's ignored `fish_drama3::reference_tests::reference_*`.

reference.json (`schema` = "izwi.fish_drama3.reference", `schema_version` = 1)
-----------------------------------------------------------------------------
`<t>` names a tensor in `reference.safetensors` (F32). `trace`, `checksum`
and the noise layout as in `fish_drama3_fixture_common.py`; `rollout.steps`
has the tiny fixture's keys. C = 4097, V = 4096, H = 2048.

A generation record (`<generation>`), one per upstream generate_drama3 call:

  {"segments": [str],                  # literals the prompt builder encoded
   "prompt_rows": P, "rows": [[main, c0..c9]], "trace_rows": [r...],
   "trunk": {"input": trace, "final_norm": trace,
             "layers": [{"layer", "kind", "output": trace}],   # residual after
             "routes": [{"layer", ...}],                        # traced layers
             "near_ties_by_layer": {"<layer>": n}},             # all 40 layers
   "tower": {"prefill_rows": P - 1,
             "layers": [{"index", "kind", "injects", "head_predicts",
                         "output": trace}]},
   "main_logits": {"tensor": <t> [C], "top16": [[candidate, token, logit]]},
   "rollout": {"params": {"temperature", "top_p", "top_k"},  # main AND heads
               "eos_mask_first_frame": false, "max_new_tokens",
               "noise_seed", "finish_reason", "frames": [[main, c0..c9]],
               "redraws",
               "noise": {"tensor": <t> [n], "count", "candidates", "codes",
                         "layout"},
               "main_logits": <t> [S, C], "hidden": <t> [S, H], "steps": [...]},
   "self_check": {"rows", "max_abs", "relative_l2"},   # cached vs full
   "seconds": float}

{
  "schema", "schema_version", "generator", "created_at",
  "provenance": {..., "device", "gpu_name"}, "reference_semantics": {...},
  "model": {"repo_id", "revision", "files": {name: {"bytes", "sha256"}},
            "shards_verified": bool},
  "sampling": {"temperature", "top_p", "top_k", "max_new_tokens",
               "chunk_max_new_tokens", "utterance_frame_cap",
               "eos_mask_first_frame": false},
  "clips": [{"name", "source", "source_sha256", "transcript",
             "ffmpeg": {"version", "args"}, "raw_file": "audio/<name>.f32",
             "format": "f32le mono 44100 Hz", "samples", "seconds", "sha256",
             "max_abs", "codes": [10][frames], "codes_frames"}],
  "tensors_file": "reference.safetensors", "tensors_sha256": hex,
  "cases": [
    # kind "single" | "utterance" | "take_edit": the case fields + <generation>
    {"name", "kind", "text", "noise_seed_base",
     "references": [{"clip", "transcript", "codes": [10][frames]}],
     ...<generation>,
     # utterance only:
     "frame_cap": 216,
     "codes": [10][frames],             # frames[:, 1:].T, unclamped
     "pcm": {"tensor": <t> [samples], "sample_rate": 44100, "samples",
             "seconds", "checksum", "decoder"},
     # take_edit only:
     "previous_takes": [{"case", "text", "codes": [10][frames]}],
     "seconds"},
    # kind "chunks":
    {"name", "kind": "chunks", "text", "noise_seed_base", "references": [...],
     "chunker": {"chunk_bytes", "max_speakers", "recent_turns",
                 "requested_frames"},
     "split": [str],                    # upstream split_drama3_text(text)
     "chunks": [{"index", "text",
                 "history": [chunk index],   # turns in prompt order
                 "budget": int,               # max_new_tokens upstream passed
                 ...<generation>}],
     "self_check_chunks": {"count", "histories", "expected_histories",
                           "history_codes_match"},
     "seconds"}],
  "self_checks": {...}
}
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_common():
    spec = importlib.util.spec_from_file_location("fish_drama3_fixture_common", HERE / "fish_drama3_fixture_common.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


common = load_common()

SCHEMA = "izwi.fish_drama3.reference"
GENERATOR = "scripts/fixtures/generate_fish_drama3_reference.py"
DEFAULT_OUT = Path(os.environ.get("IZWI_FISH_DRAMA3_REFERENCE_DIR") or common.REPO_ROOT / "target/fish-drama3-reference")
SAMPLE_RATE = 44_100
FRAME_SAMPLES = 2048
UTTERANCE_SECONDS = 10
UTTERANCE_FRAME_CAP = math.ceil(UTTERANCE_SECONDS * SAMPLE_RATE / FRAME_SAMPLES)  # 216
CLIPS = {"fox": ("fox.wav", "fox.md"), "test": ("test.wav", "test.md")}
# 65 | 288 | 57 UTF-8 bytes: each adjacent pair exceeds the chunker's 300 and
# the middle sentence stays below it, so split_drama3_text gives 3 chunks.
LONG_TEXT = (
    "It was late when the night train finally pulled into the station."
    " Outside, the platform was almost empty, lit only by a row of old lamps that hummed and "
    "flickered in the cold air, while a single porter in a long grey coat pushed a squeaking cart "
    "piled high with suitcases toward the far end, whistling a tune that nobody else seemed to "
    "remember anymore."
    " I stepped down and turned my collar up against the wind."
)
UTTERANCE_TEXT = (
    "It is a quiet morning in the village. The bakery has just opened its doors, and the smell of "
    "fresh bread drifts down the street while the first customers line up outside."
)
TAKE_EDIT_TEXT = (
    "It is a quiet evening in the village. The bakery is about to close its doors, but the smell "
    "of fresh bread still drifts down the street."
)
CASES = [
    {
        "name": "plain",
        "kind": "single",
        "text": "Hello, how are you? I hope the weather is treating you well, because it has "
        "been raining here all week long and the garden has turned into a pond.",
        "references": [],
    },
    {
        "name": "one_reference",
        "kind": "single",
        "text": "The weather is lovely today, so let us take a long walk through the park "
        "before it gets dark and the shops begin to close.",
        "references": [("fox", "")],
    },
    {
        "name": "two_speakers",
        "kind": "single",
        "text": "<|speaker:0|> Did you hear the news this morning? <|speaker:1|> I did, and I "
        "still cannot quite believe it. <|speaker:0|> Neither can I, honestly.",
        "references": [("fox", "<|speaker:0|> "), ("test", "<|speaker:1|> ")],
    },
    {"name": "long_text", "kind": "chunks", "text": LONG_TEXT, "references": [("fox", "")]},
    {"name": "utterance_10s", "kind": "utterance", "text": UTTERANCE_TEXT, "references": []},
    {"name": "take_edit", "kind": "take_edit", "text": TAKE_EDIT_TEXT, "references": [],
     "previous_takes": ["utterance_10s"]},
]
EXPECTED_CHUNK_HISTORIES = [[], [0], [0, 1]]
CACHED_VS_FULL_TOLERANCE = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fish-speech", type=Path, required=True, help="fish-speech checkout at 0f8a12e8")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto", help="cpu, cuda or cuda:N (auto: cuda when available)")
    parser.add_argument("--frames", type=int, default=64, help="max_new_tokens of the single and take_edit cases")
    parser.add_argument("--chunk-frames", type=int, default=0,
                        help="long_text max_new_tokens per chunk, as generate_long_drama3 reads it (0: 600)")
    parser.add_argument("--cases", default=",".join(case["name"] for case in CASES))
    parser.add_argument("--noise-seed", type=int, default=20261030, help="generation g of case i: seed + 100 i + g")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--trunk-layers", default="0,1,3,39")
    parser.add_argument("--tower-layers", default="0,4,12")
    parser.add_argument("--data-dir", type=Path, default=common.REPO_ROOT / "data")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--verify-shards", action="store_true", help="sha256 every shard and codec.pth")
    parser.add_argument("--router-weights", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--route-tie", type=float, default=1e-5)
    parser.add_argument("--skip-commit-check", action="store_true")
    parser.add_argument("--no-strict", action="store_true")
    return parser.parse_args()


def check_model_dir(model_dir: Path, verify: bool) -> dict:
    files = {}
    for name in ("config.json", "tokenizer.json", "model.safetensors.index.json"):
        path = model_dir / name
        if not path.is_file():
            sys.exit(f"{path} is missing")
        files[name] = {"bytes": path.stat().st_size, "sha256": common.check_pinned_file(path, name)}
    pinned = dict(common.SHARDS)
    pinned[common.CODEC_FILE[0]] = common.CODEC_FILE[1:]
    for name, (size, lfs_sha) in sorted(pinned.items()):
        path = model_dir / name
        if not path.is_file() or path.stat().st_size != size:
            sys.exit(f"{path}: missing or not {size} bytes (the pinned size)")
        entry = {"bytes": size, "lfs_sha256": lfs_sha}
        if verify:
            print(f"hashing {name} ...", file=sys.stderr)
            digest = common.sha256_file(path)
            if digest != lfs_sha:
                sys.exit(f"{path}: sha256 {digest} is not the pinned {lfs_sha}")
            entry["sha256"] = digest
        files[name] = entry
    return files


def ffmpeg_version(ffmpeg: str) -> str:
    try:
        return subprocess.run([ffmpeg, "-version"], check=True, capture_output=True, text=True).stdout.splitlines()[0]
    except (OSError, subprocess.CalledProcessError) as error:
        sys.exit(f"ffmpeg is needed to convert the reference clips ({error}); pass --ffmpeg")


def convert_clips(args, names, audio_dir: Path) -> dict:
    import torch

    if not names:
        return {}
    version = ffmpeg_version(args.ffmpeg)
    audio_dir.mkdir(parents=True, exist_ok=True)
    clips = {}
    for name in names:
        wav, transcript = CLIPS[name]
        source = args.data_dir / wav
        raw = audio_dir / f"{name}.f32"
        command = [args.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                   "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", str(raw)]
        subprocess.run(command, check=True)
        samples = torch.frombuffer(bytearray(raw.read_bytes()), dtype=torch.float32)
        clips[name] = {
            "name": name,
            "source": f"{args.data_dir.name}/{wav}",
            "source_sha256": common.sha256_file(source),
            "transcript": (args.data_dir / transcript).read_text(encoding="utf-8").strip(),
            "ffmpeg": {"version": version, "args": command[1:-1] + ["<out>.f32"]},
            "raw_file": f"audio/{name}.f32",
            "format": "f32le mono 44100 Hz",
            "samples": int(samples.numel()),
            "seconds": samples.numel() / SAMPLE_RATE,
            "sha256": common.sha256_file(raw),
            "max_abs": float(samples.abs().max()),
            "_samples": samples,
        }
    return clips


def free_device_memory() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def encode_clips(fish, model_dir: Path, clips: dict, device) -> None:
    import torch

    codec = fish.inference.load_codec_model(model_dir / "codec.pth", device, torch.float32)
    with torch.inference_mode():
        for clip in clips.values():
            samples = clip.pop("_samples")
            audio = samples.to(device)[None, None]
            indices, lengths = codec.encode(audio, torch.tensor([samples.numel()], device=device))
            codes = indices[0, :, : int(lengths[0])].long().cpu()
            if codes.shape[0] != common.FRAME_CODEBOOKS or int(codes[0].max()) >= 4096 or int(codes[1:].max()) >= 1024:
                raise RuntimeError(f"{clip['name']}: unexpected codes {tuple(codes.shape)}")
            clip["codes"] = codes.tolist()
            clip["codes_frames"] = codes.shape[1]
    del codec
    free_device_memory()


def decode_pcm(fish, model_dir: Path, codes, device, sink, name: str) -> dict:
    """Upstream codec PCM of `codes` [10, frames], F32."""
    import torch

    codec = fish.inference.load_codec_model(model_dir / "codec.pth", device, torch.float32)
    with torch.inference_mode():
        # rvq.py:352-359 clamps the indices in place: decode a copy.
        pcm = fish.inference.decode_to_audio(codes.clone().to(device), codec).float().cpu()
    rate = int(codec.sample_rate)
    del codec
    free_device_memory()
    return {
        "tensor": sink.put(f"case/{name}/pcm", pcm),
        "sample_rate": rate,
        "samples": int(pcm.numel()),
        "seconds": pcm.numel() / rate,
        "checksum": common.checksum(pcm),
        "decoder": "fish_speech inference.load_codec_model (float32) + decode_to_audio "
        "(codec.from_indices; semantic codes clamped to 4095, residual to 1023)",
    }


def load_model(fish, model_dir: Path, device):
    """FishOmniModel.from_pretrained (model.py:72-121) with FP8-resident experts on any device."""
    import torch
    from tokenizers import Tokenizer

    config = fish.config.FishOmniConfig.from_dict(json.loads((model_dir / "config.json").read_text()))
    with torch.device("meta"):
        backbone = fish.backbone.FishOmniBackbone(config.text, fp8_experts=True, dtype=torch.float32)
        decoder = fish.tower.FishOmniAudioDecoder(config, dtype=torch.float32)
    backbone.materialize([device])
    fish.backbone.load_backbone_weights(
        backbone, fish.model.iter_checkpoint_tensors(model_dir, skip_prefixes=fish.model.AUDIO_DECODER_PREFIXES)
    )
    decoder.to_empty(device=device)
    fish.tower.load_audio_decoder_weights(
        decoder,
        fish.model.iter_checkpoint_tensors(
            model_dir, skip_prefixes=(fish.backbone.CHECKPOINT_BACKBONE_PREFIX, fish.backbone.CHECKPOINT_LM_HEAD_KEY)
        ),
    )
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    return fish.model.FishOmniModel(config, backbone, decoder, tokenizer).eval()


class RecordingTokenizer:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.parts: list[str] = []

    def token_to_id(self, token: str):
        return self.inner.token_to_id(token)

    def encode(self, text: str, *args, **kwargs):
        self.parts.append(text)
        return self.inner.encode(text, *args, **kwargs)


def trace_rows(rows, config) -> list[int]:
    p = len(rows)
    start, end = config.semantic_start_token_id, config.semantic_end_token_id
    semantic = [r for r in range(p) if start <= int(rows[r][0]) <= end]
    picks = {0, p // 2, p - 2, p - 1}
    runs = []
    for r in semantic:
        if runs and runs[-1][1] == r - 1:
            runs[-1][1] = r
        else:
            runs.append([r, r])
    for first, last in runs:
        picks |= {first - 1, first, first + 1, (first + last) // 2, last, last + 1}
    return sorted(r for r in picks if 0 <= r < p)


class Context:
    """What every generation needs besides its prompt."""

    def __init__(self, fish, model, args, sink, trunk_layers, tower_layers) -> None:
        self.fish, self.model, self.args, self.sink = fish, model, args, sink
        self.trunk_layers, self.tower_layers = trunk_layers, tower_layers

    def sampling(self) -> dict:
        a = self.args
        return {"temperature": a.temperature, "top_k": a.top_k, "top_p": a.top_p}


def record_generation(ctx: Context, prompt, segments, *, seed: int, base: str, max_new_tokens: int,
                      sampling: dict, generate=None) -> tuple[dict, object]:
    """One upstream generate_drama3 call under the recorded-noise sampler with
    the prefill hooks: the generation record (see the docstring) and the frames."""
    import torch

    fish, model, args, sink = ctx.fish, ctx.model, ctx.args, ctx.sink
    config = model.config
    started = time.perf_counter()
    p = len(prompt)
    picks = trace_rows(prompt, config)
    layers = config.text.num_hidden_layers
    tape = common.NoiseTape(seed)
    sampler = common.UpstreamSampler(fish.tower)
    sampler.install("noise", tape=tape)
    recorder = common.GenerationRecorder(
        fish, model, sampler, trunk_layers=ctx.trunk_layers, component_layers=(),
        route_layers=ctx.trunk_layers, tower_layers=ctx.tower_layers, near_tie_layers=range(layers),
    )
    try:
        frames, reason = recorder.run(prompt, generate=generate, max_new_tokens=max_new_tokens, **sampling)
    finally:
        recorder.remove()
        sampler.uninstall()
    top_k = config.text.num_experts_per_tok
    layer_types = config.text.layer_types
    near_ties, routes = {}, []
    for layer in range(layers):
        traced = layer in ctx.trunk_layers
        entry = common.route_entry(recorder.captured(f"router/{layer}"), top_k, picks if traced else [], args.route_tie)
        near_ties[str(layer)] = entry["near_ties"]
        if traced:
            routes.append({"layer": layer} | entry)
    trunk = {
        "input": common.trace_entry(sink, f"{base}/input", recorder.captured("input"), picks),
        "final_norm": common.trace_entry(sink, f"{base}/final_norm", recorder.captured("final_norm"), picks),
        "layers": [
            {"layer": l, "kind": layer_types[l],
             "output": common.trace_entry(sink, f"{base}/trunk/{l}", recorder.captured(f"trunk/{l}"), picks)}
            for l in ctx.trunk_layers
        ],
        "routes": routes,
        "near_ties_by_layer": near_ties,
    }
    tower = {
        "prefill_rows": p - 1,
        "layers": [
            common.tower_kind(i)
            | {"output": common.trace_entry(sink, f"{base}/tower/{i}", recorder.captured(f"tower/{i}"), picks)}
            for i in ctx.tower_layers
        ],
    }
    candidates = recorder.candidates
    first = recorder.decode_logits[0]
    values, order = first.topk(16)
    main_logits = {
        "tensor": sink.put(f"{base}/main_logits", first),
        "top16": [[int(c), int(candidates[c]), float(v)] for v, c in zip(values, order, strict=True)],
    }
    steps, redraws = common.describe_steps(sampler, candidates, config)
    noise = tape.tensor()
    c, v = len(candidates), config.audio_decoder.vocab_size
    if noise.numel() != len(steps) * 2 * c + len(frames) * 9 * v:
        raise RuntimeError(f"{base}: the tape length does not match its layout")
    rollout = {
        "params": {k: sampling[k] for k in ("temperature", "top_p", "top_k")},
        "eos_mask_first_frame": False,
        "max_new_tokens": max_new_tokens,
        "noise_seed": tape.seed,
        "finish_reason": reason,
        "frames": common.frame_rows(frames),
        "redraws": redraws,
        "noise": {
            "tensor": sink.put(f"{base}/noise", noise),
            "count": int(noise.numel()),
            "candidates": c,
            "codes": v,
            "layout": "per decode step: E_first[C], E_retry[C], then E_head1..E_head9[V] "
            "unless the step's main token is eos; candidate / code order",
        },
        "main_logits": sink.put(f"{base}/rollout_main_logits", torch.stack(recorder.decode_logits)),
        "hidden": sink.put(f"{base}/rollout_hidden", torch.stack(recorder.decode_hidden)),
        "steps": steps,
    }
    with torch.inference_mode():
        check = common.cached_vs_full(model, prompt, frames, recorder.decode_hidden)
    del recorder
    gc.collect()
    record = {
        "segments": list(segments),
        "prompt_rows": p,
        "rows": common.frame_rows(prompt),
        "trace_rows": picks,
        "trunk": trunk,
        "tower": tower,
        "main_logits": main_logits,
        "rollout": rollout,
        "self_check": check,
        "seconds": time.perf_counter() - started,
    }
    print(f"  {base}: {p} rows, {len(frames)} frames ({reason}), cached vs full "
          f"{check['relative_l2']:.2e}, {record['seconds']:.0f} s", file=sys.stderr)
    return record, frames


def case_references(case, clips) -> list[tuple[str, str]]:
    """(transcript, clip) per reference; a tag goes in front of the clip's transcript."""
    return [(tag + clips[clip]["transcript"], clip) for clip, tag in case["references"]]


def case_header(case, clips, seed_base: int) -> dict:
    return {
        "name": case["name"],
        "kind": case["kind"],
        "text": case["text"],
        "noise_seed_base": seed_base,
        "references": [{"clip": clip, "transcript": text, "codes": clips[clip]["codes"]}
                       for text, clip in case_references(case, clips)],
    }


def run_prompted(ctx: Context, case, clips, seed_base: int, max_new_tokens: int, history=()) -> tuple[dict, object]:
    """A single, utterance or take_edit case: upstream build_drama3_prompt, one generation."""
    import torch

    model = ctx.model
    references = case_references(case, clips)
    tokenizer = RecordingTokenizer(model.tokenizer)
    prompt = ctx.fish.generate.build_drama3_prompt(
        case["text"], tokenizer, model.config,
        prompt_texts=[text for text, _ in references],
        prompt_tokens=[torch.tensor(clips[clip]["codes"], dtype=torch.long) for _, clip in references],
        history=list(history),
    )
    record, frames = record_generation(
        ctx, prompt, tokenizer.parts, seed=seed_base, base=f"case/{case['name']}",
        max_new_tokens=max_new_tokens, sampling=ctx.sampling(),
    )
    return case_header(case, clips, seed_base) | record, frames


def run_long_text(ctx: Context, case, clips, seed_base: int) -> dict:
    """upstream generate_long_drama3 (inference.py:912-1018) with each of its
    generate_drama3 calls recorded; its prompt builder is watched to learn
    which history turns each chunk kept."""
    import torch

    fish, model, args = ctx.fish, ctx.model, ctx.args
    inference, chunking = fish.inference, fish.chunking
    started = time.perf_counter()
    references = case_references(case, clips)
    prompt_text = [text for text, _ in references]
    prompt_tokens = [torch.tensor(clips[clip]["codes"], dtype=torch.long) for _, clip in references]
    split = chunking.split_drama3_text(case["text"])
    tokenizer = RecordingTokenizer(model.tokenizer)
    built, chunks, chunk_frames = {}, [], []
    codes_match = True
    original_build, original_generate = inference.build_drama3_prompt, inference.generate_drama3
    original_tokenizer = model.tokenizer

    def build(*a, **k):
        tokenizer.parts = []
        rows = original_build(*a, **k)
        turns = k["history"] if "history" in k else (a[5] if len(a) > 5 else ())
        built.update(text=a[0], history=list(turns), segments=list(tokenizer.parts))
        return rows

    def generate(model_arg, prompt, *, max_new_tokens, **sampling):
        nonlocal codes_match
        if model_arg is not model:
            raise RuntimeError("generate_long_drama3 handed over another model")
        index = len(chunks)
        record, frames = record_generation(
            ctx, prompt, built["segments"], seed=seed_base + index,
            base=f"case/{case['name']}/chunk/{index}", max_new_tokens=max_new_tokens,
            sampling=sampling, generate=original_generate,
        )
        texts = [chunk["text"] for chunk in chunks]
        history = []
        for text, codes in built["history"]:
            j = texts.index(text)
            history.append(j)
            # What upstream pushed (generate_long_drama3: frames[:, 1:].T).
            codes_match &= torch.equal(codes.cpu(), chunk_frames[j][:, 1:].T.contiguous())
        chunks.append({"index": index, "text": built["text"], "history": history,
                       "budget": max_new_tokens} | record)
        chunk_frames.append(frames.cpu())
        return frames, record["rollout"]["finish_reason"]

    inference.build_drama3_prompt = build
    inference.generate_drama3 = generate
    model.tokenizer = tokenizer
    try:
        for _ in inference.generate_long_drama3(
            model=model, text=case["text"], max_new_tokens=args.chunk_frames,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            prompt_text=prompt_text, prompt_tokens=prompt_tokens,
        ):
            pass
    finally:
        inference.build_drama3_prompt = original_build
        inference.generate_drama3 = original_generate
        model.tokenizer = original_tokenizer
    histories = [chunk["history"] for chunk in chunks]
    return case_header(case, clips, seed_base) | {
        "chunker": {
            "chunk_bytes": chunking.DRAMA3_CHUNK_BYTES,
            "max_speakers": chunking.DRAMA3_CHUNK_MAX_SPEAKERS,
            "recent_turns": chunking.DRAMA3_HISTORY_RECENT_TURNS,
            "requested_frames": args.chunk_frames if args.chunk_frames > 0 else chunking.DRAMA3_CHUNK_MAX_NEW_TOKENS,
        },
        "split": split,
        "chunks": chunks,
        "self_check_chunks": {
            "count": len(chunks),
            "histories": histories,
            "expected_histories": EXPECTED_CHUNK_HISTORIES,
            "history_codes_match": codes_match,
        },
        "seconds": time.perf_counter() - started,
    }


def main() -> int:
    args = parse_args()
    import torch

    common.set_torch_reference_mode()
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    model_dir = args.model_dir.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    wanted = [name.strip() for name in args.cases.split(",") if name.strip()]
    known = {case["name"] for case in CASES}
    if not set(wanted) <= known:
        sys.exit(f"--cases names unknown cases; known: {[case['name'] for case in CASES]}")
    for case in CASES:
        missing = [need for need in case.get("previous_takes", ()) if case["name"] in wanted and need not in wanted]
        if missing:
            sys.exit(f"--cases: {case['name']} needs {missing}")
    trunk_layers = [int(v) for v in args.trunk_layers.split(",") if v.strip()]
    tower_layers = [int(v) for v in args.tower_layers.split(",") if v.strip()]

    files = check_model_dir(model_dir, args.verify_shards)
    fish = common.import_fish_speech(
        args.fish_speech, args.skip_commit_check,
        extra=("fish_speech.models.text2semantic.inference", "fish_speech.models.text2semantic.fish_omni.chunking"),
    )
    selected = [case for case in CASES if case["name"] in wanted]
    clip_names = sorted({clip for case in selected for clip, _ in case["references"]})
    clips = convert_clips(args, clip_names, out / "audio")
    if clips:
        print(f"encoding {clip_names} with the codec ...", file=sys.stderr)
        encode_clips(fish, model_dir, clips, device)

    print(f"loading the model on {device} ...", file=sys.stderr)
    model = load_model(fish, model_dir, device)
    undo = common.install_expert_patches(fish.moe, bf16_router=args.router_weights == "bf16", force_w8a16=True)
    checks = {"sampler_patch": common.verify_sampler_patch(fish.tower)}
    failures = ["sampler_patch"] if checks["sampler_patch"]["mismatches"] else []
    sink = common.TensorSink()
    ctx = Context(fish, model, args, sink, trunk_layers, tower_layers)
    results, by_name, takes = [], {}, {}
    try:
        for index, case in enumerate(CASES):
            if case["name"] not in wanted:
                continue
            print(f"case {case['name']} ({case['kind']}) ...", file=sys.stderr)
            seed_base = args.noise_seed + 100 * index
            kind = case["kind"]
            if kind == "chunks":
                result = run_long_text(ctx, case, clips, seed_base)
                chunk_check = result["self_check_chunks"]
                if chunk_check["count"] != len(EXPECTED_CHUNK_HISTORIES) or result["split"] != [
                        chunk["text"] for chunk in result["chunks"]]:
                    failures.append(f"{case['name']} chunk count")
                if chunk_check["histories"] != EXPECTED_CHUNK_HISTORIES or not chunk_check["history_codes_match"]:
                    failures.append(f"{case['name']} history")
                generations = result["chunks"]
            else:
                if kind == "take_edit":
                    previous = [(by_name[name]["text"], takes[name]) for name in case["previous_takes"]]
                    result, _ = run_prompted(ctx, case, clips, seed_base, args.frames, history=previous)
                    result["previous_takes"] = [
                        {"case": name, "text": by_name[name]["text"], "codes": takes[name].tolist()}
                        for name in case["previous_takes"]
                    ]
                elif kind == "utterance":
                    result, frames = run_prompted(ctx, case, clips, seed_base, UTTERANCE_FRAME_CAP)
                    takes[case["name"]] = frames[:, 1:].T.contiguous().cpu()
                    result["frame_cap"] = UTTERANCE_FRAME_CAP
                    result["codes"] = takes[case["name"]].tolist()
                else:
                    result, _ = run_prompted(ctx, case, clips, seed_base, args.frames)
                generations = [result]
            for generation in generations:
                if generation["self_check"]["relative_l2"] > CACHED_VS_FULL_TOLERANCE:
                    failures.append(f"{case['name']} cached_vs_full")
            results.append(result)
            by_name[case["name"]] = result
    finally:
        undo()
    del ctx, model
    free_device_memory()

    for name, codes in takes.items():
        if codes.shape[1] == 0:
            failures.append(f"{name} produced no frames")
            continue
        print(f"decoding {name} ({codes.shape[1]} frames) with the codec ...", file=sys.stderr)
        by_name[name]["pcm"] = decode_pcm(fish, model_dir, codes, device, sink, name)
    checks["cached_vs_full_tolerance"] = CACHED_VS_FULL_TOLERANCE
    checks["failures"] = failures

    tensors_sha = sink.save(out / "reference.safetensors", {"schema": SCHEMA, "generator": GENERATOR})
    provenance = common.package_versions(["torch", "torchaudio", "safetensors", "tokenizers", "descript-audio-codec"])
    provenance |= {
        "fish_speech_commit": fish.commit,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_threads": torch.get_num_threads(),
    }
    reference = {
        "schema": SCHEMA,
        "schema_version": 1,
        "generator": GENERATOR,
        "created_at": common.now_utc(),
        "provenance": provenance,
        "reference_semantics": common.REFERENCE_SEMANTICS
        | {"experts_loader": "FP8-resident MoEExperts(fp8=True), upstream's dequantize path per forward "
           "(the W8A8 Triton branch bypassed on CUDA)",
           "router_weights_mode": args.router_weights},
        "model": {"repo_id": common.REPO_ID, "revision": common.CHECKPOINT_REVISION, "files": files,
                  "shards_verified": args.verify_shards},
        "sampling": {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
                     "max_new_tokens": args.frames, "chunk_max_new_tokens": args.chunk_frames,
                     "utterance_frame_cap": UTTERANCE_FRAME_CAP, "eos_mask_first_frame": False},
        "clips": list(clips.values()),
        "tensors_file": "reference.safetensors",
        "tensors_sha256": tensors_sha,
        "cases": results,
        "self_checks": checks,
    }
    common.dump_json(reference, out / "reference.json")
    print(f"wrote {out / 'reference.json'}", file=sys.stderr)
    if failures:
        print(f"self-checks failed: {failures}", file=sys.stderr)
        return 0 if args.no_strict else 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
