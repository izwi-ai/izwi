#!/usr/bin/env python3
"""Generate the L1 tiny upstream fixture for Fish Audio Drama 3 (plan §10.2).

What it does
------------
1. Writes a tiny checkpoint at the L0 synthetic geometry, in the published
   layout izwi's census expects (`checkpoint.rs`): the real tensor names, BF16
   dense tensors, E4M3 routed experts with F32 128x128 `weight_scale_inv`
   (multiplied), two shards plus `model.safetensors.index.json`, and the
   companion files. The values are a port of izwi's
   `synthetic.rs::trained_like_tensors(config, seed)` (xorshift per tensor,
   seeded by FNV-1a of its name; zero-centred norm gains near 0, the DeltaNet
   norm near 1, `A_log` / `dt_bias` near their published means, projections at
   1/sqrt(fan_in)), so `SyntheticCheckpoint::write(seed)` should produce the
   same tensors: `checkpoint.tensor_digest` below is defined so izwi can check.
2. `config.json` is the published config (`tests/fixtures/fish_drama3/
   config.json`, the pinned file) with exactly the edits of
   `synthetic.rs::trunk_geometry_json()`:
     text_config: hidden 128, 8 layers, full attention at 3 and 7 (3:1),
       attention 4 / 2 heads x 32 (rotary 8 = 0.25), DeltaNet 2 key / 4 value
       heads x 32, 8 experts top-2 of width 256 + a shared expert of 64,
       vocab 600, max positions 4096, eos 470, pad 468;
     audio_decoder_config: dim / text_dim 128, 4 / 2 heads x 32,
       intermediate 192, 64 codes (4 shared + 9 residual layers, 10 codebooks);
     top level: eos 470, pad 468, audio pad 480, semantic band 512..575.
   Every other field (switches, eps, theta, quantization_config) stays as
   published, so izwi's `FishDrama3Config` loads it under the Synthetic policy
   and fish-speech's `FishOmniConfig` under its own checks.
3. `tokenizer.json` is a WordLevel tokenizer (WhitespaceSplit, no post
   processor) laid out like the config: `<unk>` 0, the prompt's words 1..,
   `<|endoftext|>` 468, `<|im_start|>` 469, `<|im_end|>` 470, `<|audio_pad|>`
   480, `<|text|>` 493, `<|voice|>` 494, `<|interleave|>` 495,
   `<|speaker:0..15|>` 496..511, `<|semantic:0..63|>` 512..575,
   `<|endofturn|>` 576, `<|audio_separator|>` 577. Its words are collected by
   running upstream's prompt builder once over the cases below, so every
   literal encodes without `<unk>` (checked). izwi and fish-speech read the
   same file with the same `tokenizers` crate.
4. Loads the directory with fish-speech's own `FishOmniModel.from_pretrained`
   (CPU, float32, `FISH_DRAMA3_FP8_EXPERTS=0`: the FP8 experts are
   dequantized by upstream's `dequantize_fp8_blocks` at load, i.e. W8A16 in
   F32), rounds the router weights to BF16 (see REFERENCE_SEMANTICS in the
   common module), and runs upstream `build_drama3_prompt` and
   `generate_drama3` with:
   * a forced sampler (frames FORCED_FRAMES) for 2 decode steps per prompt:
     prefill traces, the main logits at the last prompt row, head logits under
     forced codes, and the main logits after the forced frame;
   * the recorded-noise sampler for the rollouts (32 frames each), scanning
     noise seeds until the rollout meets its goal (32 frames without eos, or
     an eos stop after >= 4 frames).
5. Writes, into `--out` (default crates/izwi-core/tests/fixtures/fish_drama3/
   tiny/): the checkpoint (about 13.3 MB at this geometry: the experts are
   6.3 MB of E4M3, the tower 5.0 MB of BF16), `expected.json` and
   `expected.safetensors` (about 1.5 MB together).

Self-checks (exit 3 when one fails, unless --no-strict): the patched sampler
equals upstream's on random rows; every prompt literal encodes without
`<unk>`; the forced frames come back as forced; each rollout's RAS decisions
replay; each rollout's cached decode equals one uncached forward (relative
L2 < 1e-4); the noise tape length matches its layout.

Environment (CPU only; this Mac works; runtime about 1-2 minutes, < 2 GB RAM)
----------------------------------------------------------------------------
* Python >= 3.10 with torch >= 2.4 (CPU build is enough; float8_e4m3fn
  support), safetensors >= 0.4.3, tokenizers >= 0.19, loguru, tqdm.
* fish-speech at 0f8a12e8a651062fdee9c33a894bdccd3a8513f1:
      git clone https://github.com/fishaudio/fish-speech /path/to/fish-speech
      git -C /path/to/fish-speech checkout 0f8a12e8a651062fdee9c33a894bdccd3a8513f1
  Nothing is pip-installed from it: the checkout goes first on sys.path and
  only `fish_speech.models.text2semantic.fish_omni` is imported. Use the
  Python of an environment that has the packages above (`-I` keeps the
  current directory and user site-packages off the path).

Usage:

    python3 -I scripts/fixtures/generate_fish_drama3_tiny.py \\
        --fish-speech /path/to/fish-speech \\
        [--out crates/izwi-core/tests/fixtures/fish_drama3/tiny] \\
        [--seed 0] [--noise-seed 20261030] [--trace-rows selected|all] \\
        [--router-weights bf16|fp32] [--skip-commit-check] [--no-strict]

expected.json (`schema` = "izwi.fish_drama3.tiny_expected", `schema_version` = 1)
-------------------------------------------------------------------------------
Floats are F32 values written as their shortest round-trip decimal (F64 for
checksums and gaps). `<t>` is a tensor name in `expected.safetensors` (F32
unless noted). `trace` is a trace entry and `checksum` a checksum as defined
in `fish_drama3_fixture_common.py`. Row `r` of a prompt is position `r`.
C = 65 main candidates (eos 470, then 512..575), V = 64 codes, H = 128.

{
  "schema", "schema_version", "generator", "created_at",
  "provenance": {"fish_speech_commit", "python", "platform", "torch",
                 "safetensors", "tokenizers", "torch_threads"},
  "reference_semantics": {...},
  "tensors_file": "expected.safetensors", "tensors_sha256": hex,
  "checkpoint": {"seed": int, "values": str, "tensor_digest": hex,
                 "tensors": int, "tensor_bytes": int,
                 "files": {name: {"bytes", "sha256"}}},
      # tensor_digest: sha256 over the tensors sorted by name (UTF-8 bytes):
      # f"{name}\\0{dtype}\\0{shape joined by ','}\\0{byte length}\\0" + bytes
  "ids": {"unk": 0, "pad": 468, "im_start": 469, "eos": 470, "audio_pad": 480,
          "voice": 494, "speaker_first": 496, "semantic_start": 512,
          "semantic_end": 575, "main_candidates": 65, "codebook_vocab": 64,
          "codebooks": 10},
  "layer_types": [8 x "linear_attention" | "full_attention"],
  "prompts": [{                         # plain, one_reference,
    "name", "text",                     # two_references_two_speakers, history
    "references": [{"transcript", "codes": [10][frames]}],
    "history": [{"text", "codes": [10][frames]}],
    "segments": [str],                  # literals upstream encoded, in order
    "prompt_rows": P, "rows": [[main, c0, ..., c9]] (P rows),
    "trace_rows": [r...],               # rows stored in every trace below
    "trunk": {
      "input": trace,                   # backbone input embedding [P, H]
      "layers": [{"layer", "kind", "output": trace}],   # residual AFTER layer
      "components": [{"layer", "kind", "part": "mixer" | "moe", "output": trace}],
          # layer 0 (DeltaNet) and 3 (attention): the mixer's output and the
          # MoE block's output (routed + gated shared expert), before adding
      "final_norm": trace,              # final-norm hidden [P, H]
      "routes": [{"layer", "rows", "experts": [[top_k]], "weights": [[top_k]],
                  "margins": [..], "near_tie_threshold", "near_ties",
                  "min_margin"}]},      # every layer; weights BF16-rounded;
                                        # margin = p[k-1] - p[k] (F32 softmax)
    "tower": {
      "prefill_rows": P - 1,            # rows 0..P-2, teacher-forced
      "layers": [{"index": 0..12, "kind": "shared" | "residual",
                  "injects": "c0" | "c{j}", "head_predicts": null | "c{j+1}",
                  "output": trace}],    # after final_layernorm, [P - 1, H]
      "teacher_forced_head_logits": {"rows": [r...], "tensor": <t> [R, 9, V]}},
          # head j applied to residual layer j's prefill output at row r
    "forced": {
      "frames": [[main, c0..c9], [main, c0..c9]],   # sampled at steps 1, 2
      "main_logits_full": <t> [600],    # lm_head over the vocab, row P - 1
      "steps": [{"position": P - 1 + s,
                 "hidden": <t> [H],     # final-norm hidden the step reads
                 "main_logits": <t> [C],
                 "head_logits": <t> [9, V]}]}}],   # head j under forced c0..cj
  "rollouts": [{
    "name", "prompt", "goal": "length" | "eos", "goal_met": bool,
    "params": {"temperature", "top_p", "top_k"},   # main AND heads (izwi:
                                       # from_request(t, p, Some(k)))
    "eos_mask_first_frame": false, "max_new_tokens": 32,
    "noise_seed", "seeds_tried", "finish_reason": "length" | "eos",
    "frames": [[main, c0..c9]],        # produced frames, eos excluded
    "redraws": int,
    "noise": {"tensor": <t> [n], "count": n, "candidates": C, "codes": V,
              "layout": str},          # see the common module doc
    "main_logits": <t> [S, C], "hidden": <t> [S, H],   # S = decode steps
    "steps": [{"position", "window": [16 main ids, oldest first],
               "main": {"first", "retry", "redrawn", "token"},
               "frame": null | [main, c0..c9], "noise_offset": int,
               "dropped_head_draws": 0 | 9,
               "draws": [{"kind", "width", "choice", "kept",
                          "cut": "top_p" | "top_k" | "none",
                          "top_p_gap": null | float, "race_gap": float}]}],
        # choice: candidate index (main: 0 = eos, i = semantic i - 1;
        # heads: the code). race_gap = (best - second) / best of p / E;
        # top_p_gap = distance of the cumulative at the cut from top_p
    "self_check": {"rows", "max_abs", "relative_l2"}}],   # cached vs full
  "self_checks": {...}
}
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import re
import sys
from array import array
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMON_FILE = HERE / "fish_drama3_fixture_common.py"


def load_common():
    spec = importlib.util.spec_from_file_location("fish_drama3_fixture_common", COMMON_FILE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


common = load_common()

SCHEMA = "izwi.fish_drama3.tiny_expected"
GENERATOR = "scripts/fixtures/generate_fish_drama3_tiny.py"
DEFAULT_OUT = common.FIXTURE_DIR / "tiny"
FP8_BLOCK = 128

# The synthetic geometry (synthetic.rs::trunk_geometry_json).
TEXT_EDITS = {
    "hidden_size": 128,
    "num_hidden_layers": 8,
    "layer_types": ["full_attention" if layer % 4 == 3 else "linear_attention" for layer in range(8)],
    "mlp_layer_types": ["moe"] * 8,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "linear_num_key_heads": 2,
    "linear_key_head_dim": 32,
    "linear_num_value_heads": 4,
    "linear_value_head_dim": 32,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 256,
    "shared_expert_intermediate_size": 64,
    "vocab_size": 600,
    "max_position_embeddings": 4096,
    "eos_token_id": 470,
    "pad_token_id": 468,
}
TOWER_EDITS = {
    "dim": 128,
    "text_dim": 128,
    "n_head": 4,
    "n_local_heads": 2,
    "head_dim": 32,
    "intermediate_size": 192,
    "vocab_size": 64,
}
TOP_EDITS = {
    "eos_token_id": 470,
    "pad_token_id": 468,
    "audio_pad_token_id": 480,
    "semantic_start_token_id": 512,
    "semantic_end_token_id": 575,
}

UNK = "<unk>"
CONTROL_TOKENS = (
    [("<|endoftext|>", 468), ("<|im_start|>", 469), ("<|im_end|>", 470), ("<|audio_pad|>", 480),
     ("<|text|>", 493), ("<|voice|>", 494), ("<|interleave|>", 495)]
    + [(f"<|speaker:{i}|>", 496 + i) for i in range(16)]
    + [(f"<|semantic:{i}|>", 512 + i) for i in range(64)]
    + [("<|endofturn|>", 576), ("<|audio_separator|>", 577)]
)
IDS = {
    "unk": 0, "pad": 468, "im_start": 469, "eos": 470, "audio_pad": 480, "voice": 494,
    "speaker_first": 496, "semantic_start": 512, "semantic_end": 575,
    "main_candidates": 65, "codebook_vocab": 64, "codebooks": 10,
}
SEMANTIC_START, CODES = 512, 64


def make_codes(frames: int, seed: int) -> list[list[int]]:
    """[10][frames] codes below 64 (the goldens script's formula, mod 64)."""
    return [
        [(seed * 7919 + k * 131 + f * 977 + k * f * 31) % CODES for f in range(frames)]
        for k in range(common.FRAME_CODEBOOKS)
    ]


CASES = [
    {"name": "plain", "text": "Hello, how are you?", "references": [], "history": []},
    {
        "name": "one_reference",
        "text": "Hello, how are you?",
        "references": [("The quick brown fox jumps over the lazy dog.", make_codes(6, 1))],
        "history": [],
    },
    {
        "name": "two_references_two_speakers",
        "text": "<|speaker:0|> Ready? <|speaker:1|> Always.",
        "references": [
            ("<|speaker:0|> First voice here.", make_codes(5, 2)),
            ("<|speaker:1|> Second voice here.", make_codes(4, 3)),
        ],
        "history": [],
    },
    {
        "name": "history",
        "text": "Now the third line.",
        "references": [],
        "history": [("First line.", make_codes(4, 4)), ("<|speaker:1|> Second line.", make_codes(3, 5))],
    },
]
# Decode steps 1 and 2 of every prompt: [main, c0, ..., c9], main = 512 + c0.
FORCED_FRAMES = [
    [549, 37, 5, 63, 0, 17, 42, 8, 31, 60, 1],
    [515, 3, 62, 1, 33, 7, 0, 63, 12, 25, 44],
]
ROLLOUT_PARAMS = {"temperature": 1.0, "top_p": 0.9, "top_k": 30}
ROLLOUT_FRAMES = 32
ROLLOUTS = [
    ("plain_length", "plain", "length"),
    ("one_reference_length", "one_reference", "length"),
    ("two_speakers_length", "two_references_two_speakers", "length"),
    ("history_length", "history", "length"),
    ("plain_eos", "plain", "eos"),
]
EOS_MIN_FRAMES = 4
CACHED_VS_FULL_TOLERANCE = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--fish-speech", type=Path, required=True, help="fish-speech checkout at 0f8a12e8")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0, help="weight seed (SyntheticCheckpoint::write(seed))")
    parser.add_argument("--noise-seed", type=int, default=20261030, help="first rollout noise seed")
    parser.add_argument("--max-seed-scan", type=int, default=64, help="noise seeds tried per rollout")
    parser.add_argument("--trace-rows", choices=("selected", "all"), default="selected")
    parser.add_argument("--router-weights", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--route-tie", type=float, default=1e-5, help="router margin counted as a near tie")
    parser.add_argument("--skip-commit-check", action="store_true")
    parser.add_argument("--no-strict", action="store_true")
    args = parser.parse_args()
    if args.max_seed_scan < 1:
        parser.error("--max-seed-scan must be at least 1")
    return args


# ---------------------------------------------------------------------------
# The checkpoint (port of synthetic.rs)
# ---------------------------------------------------------------------------


def tiny_config() -> dict:
    common.check_pinned_file(common.PUBLISHED_CONFIG, "config.json")
    config = json.loads(common.PUBLISHED_CONFIG.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    config["text_config"].update(copy.deepcopy(TEXT_EDITS))
    config["audio_decoder_config"].update(TOWER_EDITS)
    config.update(TOP_EDITS)
    return config


def expected_census(config: dict) -> list[tuple[str, str, list[int]]]:
    """Port of checkpoint.rs::expected_census, in its order."""
    t, a = config["text_config"], config["audio_decoder_config"]
    h, vocab = t["hidden_size"], t["vocab_size"]
    kh, kd, vh, vd = (t["linear_num_key_heads"], t["linear_key_head_dim"],
                      t["linear_num_value_heads"], t["linear_value_head_dim"])
    heads, kv_heads, head_dim = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
    experts, inter, shared = t["num_experts"], t["moe_intermediate_size"], t["shared_expert_intermediate_size"]
    out = []

    def add(name, dtype, shape):
        out.append((name, dtype, list(shape)))

    def fp8(module, rows, cols):
        add(f"{module}.weight", "F8_E4M3", [rows, cols])
        add(f"{module}.weight_scale_inv", "F32", [-(-rows // FP8_BLOCK), -(-cols // FP8_BLOCK)])

    trunk = "model.language_model"
    add(f"{trunk}.embed_tokens.weight", "BF16", [vocab, h])
    add(f"{trunk}.norm.weight", "BF16", [h])
    add("lm_head.weight", "BF16", [vocab, h])
    for layer, kind in enumerate(t["layer_types"]):
        p = f"{trunk}.layers.{layer}"
        add(f"{p}.input_layernorm.weight", "BF16", [h])
        add(f"{p}.post_attention_layernorm.weight", "BF16", [h])
        if kind == "linear_attention":
            x, conv = f"{p}.linear_attn", 2 * kh * kd + vh * vd
            add(f"{x}.A_log", "BF16", [vh])
            add(f"{x}.dt_bias", "BF16", [vh])
            add(f"{x}.conv1d.weight", "BF16", [conv, 1, t["linear_conv_kernel_dim"]])
            add(f"{x}.in_proj_qkv.weight", "BF16", [conv, h])
            add(f"{x}.in_proj_z.weight", "BF16", [vh * vd, h])
            add(f"{x}.in_proj_a.weight", "BF16", [vh, h])
            add(f"{x}.in_proj_b.weight", "BF16", [vh, h])
            add(f"{x}.norm.weight", "BF16", [vd])
            add(f"{x}.out_proj.weight", "BF16", [h, vh * vd])
        else:
            x = f"{p}.self_attn"
            add(f"{x}.q_proj.weight", "BF16", [2 * heads * head_dim, h])
            add(f"{x}.k_proj.weight", "BF16", [kv_heads * head_dim, h])
            add(f"{x}.v_proj.weight", "BF16", [kv_heads * head_dim, h])
            add(f"{x}.o_proj.weight", "BF16", [h, heads * head_dim])
            add(f"{x}.q_norm.weight", "BF16", [head_dim])
            add(f"{x}.k_norm.weight", "BF16", [head_dim])
        m = f"{p}.mlp"
        add(f"{m}.gate.weight", "BF16", [experts, h])
        add(f"{m}.shared_expert.gate_proj.weight", "BF16", [shared, h])
        add(f"{m}.shared_expert.up_proj.weight", "BF16", [shared, h])
        add(f"{m}.shared_expert.down_proj.weight", "BF16", [h, shared])
        add(f"{m}.shared_expert_gate.weight", "BF16", [1, h])
        for e in range(experts):
            x = f"{m}.experts.{e}"
            fp8(f"{x}.gate_proj", inter, h)
            fp8(f"{x}.up_proj", inter, h)
            fp8(f"{x}.down_proj", h, inter)

    d, codes, codebooks = a["dim"], a["vocab_size"], a["num_codebooks"]
    q_width, kv_width = a["n_head"] * a["head_dim"], a["n_local_heads"] * a["head_dim"]

    def tower_layer(prefix, embedding):
        add(f"{prefix}.{embedding}.weight", "BF16", [codes, d])
        for norm in ("enorm", "hnorm", "final_layernorm"):
            add(f"{prefix}.{norm}.weight", "BF16", [d])
        add(f"{prefix}.eh_proj.weight", "BF16", [d, 2 * d])
        b = f"{prefix}.transformer_layer"
        add(f"{b}.attention.wqkv.weight", "BF16", [2 * q_width + 2 * kv_width, d])
        add(f"{b}.attention.wo.weight", "BF16", [d, q_width])
        add(f"{b}.attention.q_norm.weight", "BF16", [a["head_dim"]])
        add(f"{b}.attention.k_norm.weight", "BF16", [a["head_dim"]])
        add(f"{b}.attention_norm.weight", "BF16", [d])
        add(f"{b}.ffn_norm.weight", "BF16", [d])
        add(f"{b}.feed_forward.w1.weight", "BF16", [a["intermediate_size"], d])
        add(f"{b}.feed_forward.w3.weight", "BF16", [a["intermediate_size"], d])
        add(f"{b}.feed_forward.w2.weight", "BF16", [d, a["intermediate_size"]])

    add("audio_decoder.codebook_embeddings.weight", "BF16", [codebooks * codes, d])
    for index in range(a["shared_enrichment_layers"]):
        tower_layer(f"audio_decoder_mtp_shared.layers.{index}", "semantic_embedding")
    for index in range(codebooks - 1):
        tower_layer(f"audio_decoder_mtp.layers.{index}", "codebook_embedding")
        add(f"audio_decoder_mtp.heads.{index}.weight", "BF16", [codes, d])
    return out


def fan_in_scale(cols: int) -> float:
    """`(3.0 / cols as f32).sqrt()` in F32."""
    return common.f32(math.sqrt(common.f32(3.0 / cols)))


def dense_values(name: str, shape: list[int], seed: int) -> array:
    """Port of synthetic.rs::dense_values."""
    n = math.prod(shape)
    cols = shape[-1]

    def draw(scale):
        return common.xorshift_values(n, common.fnv1a_seed(name, seed), scale)

    def around(center, spread):
        # Rust's `center` is an f32 literal (-0.8f32 is not the double -0.8).
        center = common.f32(center)
        return array("f", [v + center for v in draw(spread)])

    if name.endswith("linear_attn.norm.weight"):
        return around(1.0, 0.2)
    if name.endswith("norm.weight"):
        return draw(0.2)
    if name.endswith("A_log"):
        return around(-1.5, 0.8)
    if name.endswith("dt_bias"):
        return around(-0.8, 0.4)
    if name.endswith("conv1d.weight"):
        return draw(0.5)
    if name.endswith("embed_tokens.weight") or name.endswith("embedding.weight"):
        return draw(1.0)
    if name.endswith("codebook_embeddings.weight"):
        return draw(0.5)
    if name.endswith("mlp.gate.weight"):
        return draw(2.0 * fan_in_scale(cols))
    return draw(fan_in_scale(cols))


def fp8_values(name: str, rows: int, cols: int, seed: int) -> tuple[bytes, array]:
    """Port of synthetic.rs::fp8_values: E4M3 codes and F32 block scales."""
    values = common.xorshift_values(rows * cols, common.fnv1a_seed(name, seed), fan_in_scale(cols))
    row_blocks, col_blocks = -(-rows // FP8_BLOCK), -(-cols // FP8_BLOCK)
    scales = array("f", [0.0] * (row_blocks * col_blocks))
    for b in range(row_blocks * col_blocks):
        rb, cb = divmod(b, col_blocks)
        lo, hi = cb * FP8_BLOCK, min((cb + 1) * FP8_BLOCK, cols)
        amax = 0.0
        for r in range(rb * FP8_BLOCK, min((rb + 1) * FP8_BLOCK, rows)):
            amax = max(amax, max(map(abs, values[r * cols + lo : r * cols + hi])))
        scales[b] = common.f32(common.f32(amax / 448.0) * (1.0 + 0.25 * b))
    codes = bytearray(rows * cols)
    encode = common.encode_e4m3fn
    for r in range(rows):
        base, rb = r * cols, r // FP8_BLOCK
        for cb in range(col_blocks):
            lo, hi = cb * FP8_BLOCK, min((cb + 1) * FP8_BLOCK, cols)
            scale = scales[rb * col_blocks + cb]
            # F32 division (one rounding), then the nearest E4M3 byte.
            quotients = array("f", [v / scale for v in values[base + lo : base + hi]])
            codes[base + lo : base + hi] = bytes(encode(q) for q in quotients)
    return bytes(codes), scales


def trained_like_tensors(config: dict, seed: int) -> dict:
    """{name: (dtype, shape, bytes)}: port of synthetic.rs::trained_like_tensors."""
    tensors, scales = {}, {}
    for name, dtype, shape in expected_census(config):
        if dtype == "BF16":
            data = common.bf16_bytes(dense_values(name, shape, seed))
        elif dtype == "F8_E4M3":
            data, block_scales = fp8_values(name, shape[0], shape[1], seed)
            scales[name.removesuffix(".weight")] = block_scales
        else:
            data = scales.pop(name.removesuffix(".weight_scale_inv")).tobytes()
        tensors[name] = (dtype, shape, data)
    return tensors


def write_checkpoint(out: Path, tensors: dict, config: dict) -> dict:
    """synthetic.rs::write_checkpoint's layout: the trunk in shard 1, the rest in shard 2."""
    shards = {"model-00001-of-00002.safetensors": {}, "model-00002-of-00002.safetensors": {}}
    for name, tensor in tensors.items():
        shard = "model-00001-of-00002.safetensors" if name.startswith("model.language_model") else "model-00002-of-00002.safetensors"
        shards[shard][name] = tensor
    total, weight_map, files = 0, {}, {}
    for shard, members in shards.items():
        total += common.write_safetensors(out / shard, members)
        weight_map.update({name: shard for name in members})
    index = {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "codec.pth").write_bytes(b"synthetic")
    (out / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "eos_token": "<|im_end|>",
                    "pad_token": "<|endoftext|>"}, indent=2) + "\n")
    (out / "special_tokens_map.json").write_text(
        json.dumps({"eos_token": "<|im_end|>", "pad_token": "<|endoftext|>"}, indent=2) + "\n")
    for name in [*shards, "model.safetensors.index.json", "config.json", "codec.pth",
                 "tokenizer_config.json", "special_tokens_map.json"]:
        path = out / name
        files[name] = {"bytes": path.stat().st_size, "sha256": common.sha256_file(path)}
    return {"tensor_bytes": total, "files": files}


# ---------------------------------------------------------------------------
# The tokenizer
# ---------------------------------------------------------------------------


class LiteralCollector:
    """Stands in for the tokenizer while upstream's builder lists its literals."""

    def __init__(self) -> None:
        self.literals: list[str] = []

    def token_to_id(self, token: str):
        return {"<|semantic:0|>": 512, "<|semantic:63|>": 575, "<|im_end|>": 470}.get(token)

    def encode(self, text: str):
        self.literals.append(text)
        return type("Encoding", (), {"ids": [0]})()


class RecordingTokenizer:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.parts: list[str] = []

    def token_to_id(self, token: str):
        return self.inner.token_to_id(token)

    def encode(self, text: str, *args, **kwargs):
        self.parts.append(text)
        return self.inner.encode(text, *args, **kwargs)


def build_prompt(fish, tokenizer, config, case):
    import torch

    references = case["references"]
    return fish.generate.build_drama3_prompt(
        case["text"], tokenizer, config,
        prompt_texts=[text for text, _ in references],
        prompt_tokens=[torch.tensor(codes, dtype=torch.long) for _, codes in references],
        history=[(text, torch.tensor(codes, dtype=torch.long)) for text, codes in case["history"]],
    )


def words_of(literals: list[str]) -> list[str]:
    pattern = re.compile("|".join(re.escape(t) for t, _ in sorted(CONTROL_TOKENS, key=lambda c: -len(c[0]))))
    words = set()
    for literal in literals:
        for piece in pattern.split(literal):
            words.update(piece.split())
    return sorted(words)


def tokenizer_json(words: list[str]) -> dict:
    if len(words) >= 468:
        raise ValueError("too many prompt words for the tiny vocabulary")
    vocab = {UNK: 0}
    vocab.update({word: index + 1 for index, word in enumerate(words)})
    vocab.update(dict(CONTROL_TOKENS))
    added = [
        {"id": token_id, "content": token, "single_word": False, "lstrip": False, "rstrip": False,
         "normalized": False, "special": True}
        for token, token_id in CONTROL_TOKENS
    ]
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": UNK},
    }


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def trace_rows(rows, mode: str) -> list[int]:
    p = len(rows)
    if mode == "all":
        return list(range(p))
    semantic = [r for r in range(p) if SEMANTIC_START <= int(rows[r][0]) <= SEMANTIC_START + CODES - 1]
    picks = {0, p // 2, p - 2, p - 1}
    if semantic:
        picks |= {semantic[0] - 1, semantic[0], semantic[0] + 1, semantic[-1], semantic[-1] + 1}
    return sorted(r for r in picks if 0 <= r < p)


def record_prompt(fish, model, sampler, case, sink, args, layer_types):
    import torch

    tokenizer = RecordingTokenizer(model.tokenizer)
    rows = build_prompt(fish, tokenizer, model.config, case)
    p = len(rows)
    name = case["name"]
    picks = trace_rows(rows, args.trace_rows)
    forced = [[f[0] - SEMANTIC_START + 1] * 2 + f[2:] for f in FORCED_FRAMES]
    layers = len(layer_types)
    sampler.install("forced", forced=forced)
    recorder = common.GenerationRecorder(
        fish, model, sampler, trunk_layers=range(layers), component_layers=(0, 3),
        route_layers=range(layers), tower_layers=range(13),
    )
    try:
        frames, reason = recorder.run(rows, max_new_tokens=len(FORCED_FRAMES), **ROLLOUT_PARAMS)
    finally:
        recorder.remove()
        sampler.uninstall()
    if common.frame_rows(frames) != FORCED_FRAMES or reason != "length":
        raise RuntimeError(f"{name}: forced decode returned {frames.tolist()} ({reason})")

    base = f"prompt/{name}"
    trunk = {
        "input": common.trace_entry(sink, f"{base}/input", recorder.captured("input"), picks),
        "layers": [
            {"layer": l, "kind": layer_types[l],
             "output": common.trace_entry(sink, f"{base}/trunk/{l}", recorder.captured(f"trunk/{l}"), picks)}
            for l in range(layers)
        ],
        "components": [
            {"layer": l, "kind": layer_types[l], "part": part,
             "output": common.trace_entry(sink, f"{base}/{part}/{l}", recorder.captured(f"{part}/{l}"), picks)}
            for l in (0, 3) for part in ("mixer", "moe")
        ],
        "final_norm": common.trace_entry(sink, f"{base}/final_norm", recorder.captured("final_norm"), picks),
        "routes": [
            {"layer": l} | common.route_entry(recorder.captured(f"router/{l}"),
                                             model.config.text.num_experts_per_tok, picks, args.route_tie)
            for l in range(layers)
        ],
    }
    for key, expected in (("input", p), ("final_norm", p), ("trunk/0", p), ("tower/0", p - 1)):
        got = recorder.captured(key).shape[0]
        if got != expected:
            raise RuntimeError(f"{name}: {key} captured {got} rows, expected {expected}")

    tower_picks = [r for r in picks if r <= p - 2]
    tower_outputs = [recorder.captured(f"tower/{i}") for i in range(13)]
    heads = model.audio_decoder.codebook_heads
    with torch.inference_mode():
        teacher = torch.stack(
            [heads[j](tower_outputs[4 + j][tower_picks]) for j in range(9)], dim=1
        ).float()
    tower = {
        "prefill_rows": p - 1,
        "layers": [
            common.tower_kind(i) | {"output": common.trace_entry(sink, f"{base}/tower/{i}", tower_outputs[i], picks)}
            for i in range(13)
        ],
        "teacher_forced_head_logits": {
            "rows": tower_picks, "tensor": sink.put(f"{base}/teacher_forced_head_logits", teacher)
        },
    }
    steps = []
    for s, step in enumerate(sampler.steps):
        head_logits = torch.stack([call["logits"] for call in step["calls"][2:]])
        steps.append({
            "position": p - 1 + s,
            "hidden": sink.put(f"{base}/forced/{s}/hidden", recorder.decode_hidden[s]),
            "main_logits": sink.put(f"{base}/forced/{s}/main_logits", recorder.decode_logits[s]),
            "head_logits": sink.put(f"{base}/forced/{s}/head_logits", head_logits),
        })
    entry = {
        "name": name,
        "text": case["text"],
        "references": [{"transcript": t, "codes": c} for t, c in case["references"]],
        "history": [{"text": t, "codes": c} for t, c in case["history"]],
        "segments": tokenizer.parts,
        "prompt_rows": p,
        "rows": common.frame_rows(rows),
        "trace_rows": picks,
        "trunk": trunk,
        "tower": tower,
        "forced": {
            "frames": FORCED_FRAMES,
            "main_logits_full": sink.put(f"{base}/forced/main_logits_full", recorder.first_full_logits),
            "steps": steps,
        },
    }
    return entry, rows


def run_rollout(fish, model, sampler, prompt, args, index, spec, sink):
    import torch

    name, prompt_name, goal = spec
    candidates = None
    for attempt in range(args.max_seed_scan):
        seed = args.noise_seed + 1000 * index + attempt
        tape = common.NoiseTape(seed)
        sampler.install("noise", tape=tape)
        recorder = common.GenerationRecorder(
            fish, model, sampler, trunk_layers=(), component_layers=(), route_layers=(), tower_layers=()
        )
        try:
            frames, reason = recorder.run(prompt, max_new_tokens=ROLLOUT_FRAMES, **ROLLOUT_PARAMS)
        finally:
            recorder.remove()
            sampler.uninstall()
        candidates = recorder.candidates
        if goal == "length":
            met = reason == "length" and len(frames) == ROLLOUT_FRAMES
        else:
            met = reason == "eos" and len(frames) >= EOS_MIN_FRAMES
        if met:
            break
    steps, redraws = common.describe_steps(sampler, candidates, model.config)
    noise = tape.tensor()
    c, v = IDS["main_candidates"], IDS["codebook_vocab"]
    expected_count = len(steps) * 2 * c + len(frames) * 9 * v
    if noise.numel() != expected_count:
        raise RuntimeError(f"{name}: tape holds {noise.numel()} values, layout says {expected_count}")
    with torch.inference_mode():
        check = common.cached_vs_full(model, prompt, frames, recorder.decode_hidden)
    base = f"rollout/{name}"
    return {
        "name": name,
        "prompt": prompt_name,
        "goal": goal,
        "goal_met": met,
        "params": ROLLOUT_PARAMS,
        "eos_mask_first_frame": False,
        "max_new_tokens": ROLLOUT_FRAMES,
        "noise_seed": tape.seed,
        "seeds_tried": attempt + 1,
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
        "main_logits": sink.put(f"{base}/main_logits", torch.stack(recorder.decode_logits)),
        "hidden": sink.put(f"{base}/hidden", torch.stack(recorder.decode_hidden)),
        "steps": steps,
        "self_check": check,
    }


def main() -> int:
    args = parse_args()
    import torch

    common.set_torch_reference_mode()
    torch.manual_seed(0)
    fish = common.import_fish_speech(args.fish_speech, args.skip_commit_check)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    checks, failures = {}, []

    config = tiny_config()
    print(f"building the tiny checkpoint (seed {args.seed}) ...", file=sys.stderr)
    tensors = trained_like_tensors(config, args.seed)
    written = write_checkpoint(out, tensors, config)
    checkpoint = {
        "seed": args.seed,
        "values": "port of izwi synthetic.rs trained_like_tensors(config, seed)",
        "tensor_digest": common.tensor_digest(tensors),
        "tensors": len(tensors),
        **written,
    }

    fish_config = fish.config.FishOmniConfig.from_dict(config)
    collector = LiteralCollector()
    for case in CASES:
        build_prompt(fish, collector, fish_config, case)
    (out / "tokenizer.json").write_text(json.dumps(tokenizer_json(words_of(collector.literals)), indent=1) + "\n")
    checkpoint["files"]["tokenizer.json"] = {
        "bytes": (out / "tokenizer.json").stat().st_size,
        "sha256": common.sha256_file(out / "tokenizer.json"),
    }

    os.environ["FISH_DRAMA3_FP8_EXPERTS"] = "0"
    model = fish.model.FishOmniModel.from_pretrained(
        out, devices=["cpu"], tower_device="cpu", dtype=torch.float32
    )
    undo = common.install_expert_patches(fish.moe, bf16_router=args.router_weights == "bf16", force_w8a16=False)
    unknown = [lit for lit in collector.literals if 0 in model.tokenizer.encode(lit).ids]
    checks["tokenizer_unknown_literals"] = unknown
    if unknown:
        failures.append("tokenizer")
    checks["sampler_patch"] = common.verify_sampler_patch(fish.tower)
    if checks["sampler_patch"]["mismatches"]:
        failures.append("sampler_patch")

    sink = common.TensorSink()
    sampler = common.UpstreamSampler(fish.tower)
    layer_types = config["text_config"]["layer_types"]
    prompts, prompt_rows = [], {}
    try:
        for case in CASES:
            print(f"prompt {case['name']} ...", file=sys.stderr)
            entry, rows = record_prompt(fish, model, sampler, case, sink, args, layer_types)
            prompts.append(entry)
            prompt_rows[case["name"]] = rows
        rollouts = []
        for index, spec in enumerate(ROLLOUTS):
            print(f"rollout {spec[0]} ...", file=sys.stderr)
            rollout = run_rollout(fish, model, sampler, prompt_rows[spec[1]], args, index, spec, sink)
            rollouts.append(rollout)
            if not rollout["goal_met"]:
                failures.append(f"rollout {spec[0]} goal")
            if rollout["self_check"]["relative_l2"] > CACHED_VS_FULL_TOLERANCE:
                failures.append(f"rollout {spec[0]} cached_vs_full")
    finally:
        undo()
    checks["cached_vs_full_tolerance"] = CACHED_VS_FULL_TOLERANCE
    checks["failures"] = failures

    tensors_sha = sink.save(out / "expected.safetensors", {"schema": SCHEMA, "generator": GENERATOR})
    provenance = common.package_versions(["torch", "safetensors", "tokenizers", "loguru"])
    provenance |= {"fish_speech_commit": fish.commit, "torch_threads": torch.get_num_threads()}
    expected = {
        "schema": SCHEMA,
        "schema_version": 1,
        "generator": GENERATOR,
        "created_at": common.now_utc(),
        "provenance": provenance,
        "reference_semantics": common.REFERENCE_SEMANTICS
        | {"experts_loader": "FishOmniModel.from_pretrained, FISH_DRAMA3_FP8_EXPERTS=0 "
           "(dequantize_fp8_blocks at load, F32)",
           "router_weights_mode": args.router_weights},
        "tensors_file": "expected.safetensors",
        "tensors_sha256": tensors_sha,
        "checkpoint": checkpoint,
        "ids": IDS,
        "layer_types": layer_types,
        "prompts": prompts,
        "rollouts": rollouts,
        "self_checks": checks,
    }
    common.dump_json(expected, out / "expected.json")
    print(f"wrote {out}", file=sys.stderr)
    if failures:
        print(f"self-checks failed: {failures}", file=sys.stderr)
        return 0 if args.no_strict else 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
