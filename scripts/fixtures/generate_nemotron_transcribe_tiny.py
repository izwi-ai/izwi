#!/usr/bin/env python3
"""Generate the L1 tiny-random Nemotron 3.5 Transcribe fixture from upstream code.

What it does
------------
1. Builds a tiny checkpoint in the PUBLISHED 7-file layout (4 backbone shards +
   `perception.safetensors`; no CTC or MTP file) with the tensor names, dtypes
   and FP8 encoding izwi's census expects (`checkpoint.rs`, mirrored in
   `expected_census` below): hidden 64; layers mamba/moe/attention/mamba/moe;
   Mamba 4 heads x 16, 2 groups, state 8, chunk 8; 8 experts top-2 width 96 +
   shared 64; attention 4/2 heads x 16; encoders of width 32 (2 heads, 2
   layers); 8 speakers; vocab 256 with `<|audio|>` = 256 (one past the table);
   the real 128-mel front end with the PUBLISHED featurizer fb/window (read
   from `crates/izwi-core/tests/fixtures/nemotron_transcribe/featurizer_*.bf16`).
   Every FP8 linear is quantized ModelOpt-style: E4M3 codes with one F32
   per-tensor `weight_scale` = amax/448, plus a static `input_scale`.
   `config.json` is the published one with the dimensions shrunk, so izwi's
   `config.rs` and NeMo read the same file.
2. Runs the UPSTREAM forward with the DEQUANTIZED weights in F32 (W8A16
   semantics: weight = fp8_code * weight_scale, no activation quantization):
   * audio side: NeMo Speech's `AudioPerceptionModule` + `ParallelExpertEncoder`
     online (windowed, streaming-Sortformer) path, exactly as the vLLM plugin
     mounts it; if NeMo cannot run here (FlexAttention on CPU), SGLang-Omni's
     pure-PyTorch modules driven through NeMo's window plan
     (`--audio-impl`; the fixture records which one ran in `audio_reference`);
   * decoder: transformers' native `NemotronHBlock` / `NemotronHMoE` /
     `NemotronHMamba2Mixer` classes, patched to vLLM semantics: no dt clamp
     (transformers clamps prefill dt to >= time_step_min = 0.001), F32 dt and
     F32 y (automatic here: the whole model runs in F32), the torch fallback
     kernels (`mamba_ssm` / `causal_conv1d` are blocked from import).
3. Writes `expected.json` (schema below) and the clips (16-bit PCM WAV) into
   the output directory (default `crates/izwi-core/tests/fixtures/
   nemotron_transcribe/tiny/`). The whole directory is about 4-5 MB.

Upstream BF16 rounding points izwi reproduces (and this F32 reference
therefore reproduces too): the log-mel is rounded to BF16; per-feature
normalization runs with torch's BF16 semantics (BF16 mean and squared
deviations, F32 std); the normalized ASR-encoder input is rounded to BF16;
the RoPE cos/sin tables and the speaker kernel are rounded to BF16. Nothing
else is rounded. TF32 is disabled.

Self-checks (the script exits 3 when one fails, unless `--no-strict`):
* every Mamba mixer runs with `time_step_limit == (0, inf)`;
* the first Mamba layer equals an independent F64 recurrence on the prompt;
* the first MoE layer equals an independent F64 restatement;
* cached decode equals a full (uncached) forward for the first steps;
* the window plan NeMo ran equals the plan izwi implements;
* the written safetensors re-read equal the census (names, dtypes, shapes).

Usage (any Linux/macOS box, CPU only; needs torch >= 2.5, numpy, safetensors,
tokenizers, transformers main with native NemotronH (>= bb666bd0), NeMo Speech
main (>= 743beaf4) with omegaconf, or sglang-omni b1097644 for the fallback):

    python3 -I scripts/fixtures/generate_nemotron_transcribe_tiny.py \
        [--out crates/izwi-core/tests/fixtures/nemotron_transcribe/tiny] \
        [--audio-impl auto|nemo|sglang] [--seed 20261029] [--no-edge-clip]

expected.json schema (`schema` = "izwi.nemotron_transcribe.tiny_expected",
`schema_version` = 1). All floats are F32 values written as their shortest
round-trip decimal; lists of floats are row-major.

{
  "schema", "schema_version", "generator", "created_at", "seed",
  "versions": {python, platform, torch, numpy, transformers, transformers_git,
               nemo, nemo_git, sglang_omni, tokenizers, safetensors, omegaconf},
  "audio_reference": "nemo" | "sglang-omni", "audio_fallback_reason": str|null,
  "decoder_reference": "transformers.NemotronH",
  "experts_impl": "transformers.NemotronHExperts" | "restated",
  "reference_semantics": {...},              # free text, see above
  "checkpoint": {"files": {name: {"bytes", "sha256"}}, "tensors": n,
                 "geometry": {...}},
  "tokenizer": {"audio_token": "<|audio|>", "audio_token_id": 256,
                "eos_token_ids": [11]},
  "instruction": str,                        # izwi's READABLE_INSTRUCTION
  "prompt_text": str,                        # the rendered nemotron3p5 prompt
  "max_new_tokens": 32,
  "clips": [{
    "name": str,
    "audio_file": "audio/<name>.wav",        # relative to expected.json
    "audio_format": "wav pcm_s16le mono 16000 Hz; sample = int16 / 32768.0",
    "samples": int, "duration_secs": float, "sha256": hex,
    "mel": {"valid_frames": Tv, "frames": Tv + 1,
            "mean": [n_mels], "std": [n_mels],     # NeMo normalize_batch
            "raw_rows": [{"frame": f, "values": [n_mels]}],   # BF16 log-mel
            "normalized_rows": [{"frame": f, "values": [n_mels]}]},
            # frames f = 0, Tv // 2, Tv - 1, Tv (the 0.0 frame); normalized
            # values are BF16-rounded, frame Tv is all zeros
    "audio_tokens": ceil(Tv / 8),
    "windows": [{"index", "start", "end", "context_start", "context_end",
                 "valid_mel", "left_drop", "core_frames", "first_token",
                 "updates_state": bool,
                 "asr_core_rows": [{"row": r, "values": [d_model]}],
                     # ASR encoder output (final_norm, before asr_norm and
                     # fusion), core-relative rows r = 0, core // 2, core - 1
                 "diar_probs_80ms": [[speakers floats] x core_frames],
                     # Sortformer 80 ms sigmoids aligned to the core
                 "sortformer_cache_frames_before": int,
                 "sortformer_fifo_frames_before": int,
                 "follows_compression": bool}],
                     # the streaming state izwi's update rule predicts before
                     # the window; after a compression, top-k ties may reorder
                     # the speaker cache (compare binarized activity there)
    "diar_near_threshold": int,              # |p - 0.5| < 1e-3 over all cores
    "fused_rows": [{"token": t, "values": [d_model]}],    # after fusion
    "embeddings_rows": [{"token": t, "values": [hidden]}],  # after proj
        # tokens t = 0, each window's first token, audio_tokens // 2, last
    "prompt_ids": [ids...],                  # audio positions = audio_token_id
    "audio_offset": int, "prompt_tokens": int,
    "layer_hidden": [{"layer": l, "kind": "mamba"|"moe"|"attention",
                      "rows": [{"position": p, "values": [hidden]}],
                      "checksum": {"sum", "abs_sum", "rms"}}],
        # residual stream AFTER layer l (h_{l+1} = h_l + mixer(norm(h_l))),
        # layers 0, 1 and the first attention layer; positions 0, the first
        # and last audio positions, the last prompt position; checksum over
        # every prompt position
    "routes": {"layer": l, "positions": n, "near_ties": k,
               "near_ties_by_layer": {"<layer>": k},   # every MoE layer
               "transformers_disagreements": k,
               "rows": [{"position", "experts", "weights", "margin"}]},
        # first MoE layer, sampled prompt positions; experts in izwi order
        # (biased score desc, ties to the lower id); weights = renormalized
        # unbiased sigmoid x 2.5; margin = biased[top_k - 1] - biased[top_k];
        # near_ties counts prompt positions with margin < 1e-5
    "prefill_logits_top8": [{"id", "logit"}], "prefill_margin": float,
    "greedy_ids": [...],                     # izwi's generated(): no stop id
    "greedy_margins": [...],                 # top1 - top2 at each decision,
                                             # the stop decision included
    "generated_tokens": int,
    "finish_reason": "stop"|"length"|"repetition", "stop_token": int|null,
    "decoded": str,                          # tokenizer.decode(ids, False)
    "text": str,                             # card guardrail(decoded.strip())
    "self_checks": {...}
  }],
  "suggested_tolerances": {...},
  "self_checks": {...}
}
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import platform
import re
import subprocess
import sys
import wave
from pathlib import Path

# The torch fallback of transformers' Mamba2 path is the reference: block the
# CUDA kernel packages before transformers is imported anywhere.
if os.environ.get("NEMOTRON_REF_ALLOW_MAMBA_KERNELS") != "1":
    for _blocked in ("mamba_ssm", "causal_conv1d"):
        sys.modules.setdefault(_blocked, None)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = ROOT / "crates/izwi-core/tests/fixtures/nemotron_transcribe"
DEFAULT_OUT = FIXTURE_DIR / "tiny"
SCHEMA = "izwi.nemotron_transcribe.tiny_expected"
SCHEMA_VERSION = 1
SAMPLE_RATE = 16_000

# izwi prompt.rs (keep byte-identical).
SYSTEM_TURN = "<|im_start|>system\n<|im_end|>\n"
USER_OPEN = "<|im_start|>user\n"
ASSISTANT_OPEN = "<|im_end|>\n<|im_start|>assistant\n<think></think>"
READABLE_INSTRUCTION = (
    "Audio transcription. Language: as spoken. Wording: omit fillers and accidental "
    "repetitions; keep the remaining spoken wording unchanged. Formatting: punctuation "
    "and capitalization. Output: transcript only."
)
VERBATIM_INSTRUCTION = (
    "Audio transcription. Language: as spoken. Wording: verbatim, including fillers and "
    "repetitions. Formatting: punctuation and capitalization. Output: transcript only."
)
SPEAKER_INSTRUCTION = (
    "Transcribe this audio with punctuation and capitalization. Include all spoken words "
    "and verbal fillers. Write numbers in their standard written form. Put speaker tags "
    "<spk:0> <spk:1> <spk:2> <spk:3> on the transcription."
)
AUDIO_TOKEN = "<|audio|>"

# NeMo's serving overrides (parallel_expert_encoder.py:395-400, :2167-2170).
ONLINE_WINDOW_TOKENS = 500
ONLINE_CONTEXT_TOKENS = 50
# izwi transcript.rs.
LOOP_REPEATS = 20
LOOP_MAX_BLOCK_TOKENS = 256
NEAR_TIE = 1e-5

LAYER_KIND = {"mamba": "mamba", "moe": "moe", "attention": "attention",
              "linear_attention": "mamba", "full_attention": "attention"}
TRANSFORMERS_LAYER = {"mamba": "linear_attention", "attention": "full_attention",
                      "moe": "moe", "mlp": "mlp"}

torch.set_grad_enabled(False)


# --------------------------------------------------------------- utilities

def log(message):
    print(f"[nemotron-ref] {message}", file=sys.stderr, flush=True)


def disable_tf32():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def f32(value):
    """A float as the shortest decimal that round-trips its F32 value."""
    v = np.float32(value)
    if not np.isfinite(v):
        return None
    return float(str(v))


def f32list(tensor):
    array = tensor.detach().to(torch.float32).cpu().numpy().astype(np.float32).ravel()
    return [f32(v) for v in array]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(obj, path):
    """Indented JSON with every list of numbers kept on one line."""
    text = json.dumps(obj, indent=1, ensure_ascii=False, allow_nan=False)
    text = re.sub(
        r"\[\s+((?:-?[0-9][0-9.eE+-]*|null)(?:,\s+(?:-?[0-9][0-9.eE+-]*|null))*)\s+\]",
        lambda m: "[" + re.sub(r",\s+", ",", m.group(1)) + "]",
        text,
    )
    Path(path).write_text(text + "\n")


def decode_nemo_floats(value):
    """NeMo/HF configs spell non-finite floats as {"__float__": "Infinity"}."""
    if isinstance(value, dict):
        if set(value) == {"__float__"}:
            return float(value["__float__"])
        return {k: decode_nemo_floats(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode_nemo_floats(v) for v in value]
    return value


def module_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        try:
            module = importlib.import_module(name)
            return getattr(module, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            return None


def module_git(name, dist=None):
    """The git commit of an installed package: pip's direct_url.json, else the
    checkout the module was imported from."""
    for candidate in filter(None, [dist, name]):
        try:
            raw = importlib.metadata.distribution(candidate).read_text("direct_url.json")
            if raw:
                info = json.loads(raw)
                commit = info.get("vcs_info", {}).get("commit_id")
                if commit:
                    return commit
                url = info.get("url", "")
                if url.startswith("file://"):
                    path = Path(url[len("file://"):])
                    out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                                         capture_output=True, text=True)
                    if out.returncode == 0:
                        return out.stdout.strip()
        except Exception:  # noqa: BLE001
            pass
    spec = importlib.util.find_spec(name) if name not in sys.modules or sys.modules[name] else None
    if spec and spec.origin:
        out = subprocess.run(["git", "-C", str(Path(spec.origin).parent), "rev-parse", "HEAD"],
                             capture_output=True, text=True)
        if out.returncode == 0:
            return out.stdout.strip()
    return None


def collect_versions(extra=()):
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
    }
    for name, dist in [("transformers", None), ("nemo", "nemo_toolkit"), ("tokenizers", None),
                       ("safetensors", None), ("omegaconf", None), ("sglang_omni", "sglang-omni"),
                       *extra]:
        versions[name] = module_version(dist or name) or module_version(name)
        git = module_git(name, dist)
        if git:
            versions[f"{name}_git"] = git
    return versions


# ---------------------------------------------------------------- geometry

class Geometry:
    """Dimensions read from a (tiny or published) config.json."""

    def __init__(self, cfg):
        pre = cfg["perception"]["preprocessor"]
        pe = cfg["pe_encoder_config"]
        diar = pe["diarization_model_cfg"]
        modules = diar["sortformer_modules"]
        llm = cfg["llm_config"]
        self.sample_rate = int(pre["sample_rate"])
        self.n_mels = int(pre["features"])
        self.n_fft = int(pre["n_fft"])
        self.win = int(round(pre["window_size"] * self.sample_rate))
        self.hop = int(round(pre["window_stride"] * self.sample_rate))
        self.asr = self._encoder(pe["asr_encoder_cfg"])
        self.diar = self._encoder(diar["encoder"])
        self.speakers = int(modules["num_spks"])
        self.fc = int(modules["fc_d_model"])
        self.tf = int(modules["tf_d_model"])
        self.stacking = self.asr["stacking"]
        self.hidden = int(llm["hidden_size"])
        self.vocab = int(llm["vocab_size"])
        self.layers = [LAYER_KIND[k] for k in llm["layers_block_type"]]
        self.m_heads = int(llm["mamba_num_heads"])
        self.m_head_dim = int(llm["mamba_head_dim"])
        self.m_groups = int(llm["n_groups"])
        self.m_state = int(llm["ssm_state_size"])
        self.m_kernel = int(llm["conv_kernel"])
        self.m_chunk = int(llm["chunk_size"])
        self.m_inner = self.m_heads * self.m_head_dim
        self.m_conv = self.m_inner + 2 * self.m_groups * self.m_state
        self.m_in_rows = self.m_inner + self.m_conv + self.m_heads
        self.experts = int(llm["n_routed_experts"])
        self.top_k = int(llm["num_experts_per_tok"])
        self.expert_width = int(llm["moe_intermediate_size"])
        self.shared_width = int(llm["moe_shared_expert_intermediate_size"])
        self.routed_scaling = float(llm["routed_scaling_factor"])
        self.a_heads = int(llm["num_attention_heads"])
        self.a_kv = int(llm["num_key_value_heads"])
        self.a_head_dim = int(llm["head_dim"])
        self.eps = float(llm["layer_norm_epsilon"])

    @staticmethod
    def _encoder(cfg):
        d = int(cfg["d_model"])
        return {
            "d": d,
            "heads": int(cfg["n_heads"]),
            "layers": int(cfg["n_layers"]),
            "ff": int(round(d * float(cfg["ff_expansion"]))),
            "qk_norm": bool(cfg.get("qk_norm", False)),
            "stacking": int(cfg["subsampling_factor"]),
            "feat_in": int(cfg["feat_in"]),
        }

    def first(self, kind):
        return self.layers.index(kind)

    def as_dict(self):
        return {k: v for k, v in vars(self).items()}


def excluded(module, patterns):
    for pattern in patterns:
        if pattern.endswith("*"):
            if module.startswith(pattern[:-1]):
                return True
        elif module == pattern:
            return True
    return False


def expected_census(g, exclude):
    """izwi's `checkpoint.rs::expected_census` without the CTC and MTP heads:
    [(name, dtype, shape)] with dtype in BF16 / F32 / F8_E4M3."""
    out = []

    def entry(name, dtype, shape):
        out.append((name, dtype, tuple(shape)))

    def fp8(module, rows, cols, kv=None):
        entry(f"{module}.weight", "F8_E4M3", (rows, cols))
        entry(f"{module}.weight_scale", "F32", ())
        entry(f"{module}.input_scale", "F32", ())
        if kv:
            entry(f"{module}.{kv}", "F32", ())

    def linear(module, rows, cols, bias):
        if excluded(module, exclude):
            entry(f"{module}.weight", "BF16", (rows, cols))
        else:
            fp8(module, rows, cols)
        if bias:
            entry(f"{module}.bias", "BF16", (rows,))

    def norm(name, width):
        entry(f"{name}.weight", "BF16", (width,))
        entry(f"{name}.bias", "BF16", (width,))

    def transformer(prefix, enc):
        d = enc["d"]
        linear(f"{prefix}.pre_encode.proj", d, enc["feat_in"] * enc["stacking"], False)
        norm(f"{prefix}.embed_norm", d)
        norm(f"{prefix}.final_norm", d)
        for layer in range(enc["layers"]):
            p = f"{prefix}.layers.{layer}"
            norm(f"{p}.norm1", d)
            norm(f"{p}.norm2", d)
            linear(f"{p}.attn.w_qkv", 3 * d, d, False)
            linear(f"{p}.attn.out_proj", d, d, True)
            linear(f"{p}.ffn.net.0", enc["ff"], d, True)
            linear(f"{p}.ffn.net.3", d, enc["ff"], True)
            if enc["qk_norm"]:
                norm(f"{p}.attn.q_norm", d // enc["heads"])
                norm(f"{p}.attn.k_norm", d // enc["heads"])

    for prefix in ("perception.preprocessor.featurizer",
                   "perception.encoder.diarization_model.preprocessor.featurizer"):
        entry(f"{prefix}.fb", "BF16", (1, g.n_mels, g.n_fft // 2 + 1))
        entry(f"{prefix}.window", "BF16", (g.win,))
    transformer("perception.encoder.asr_encoder", g.asr)
    transformer("perception.encoder.diarization_model.encoder", g.diar)
    m = "perception.encoder.diarization_model.sortformer_modules"
    for name, shape, bias in [
        ("encoder_proj", (g.tf, g.fc), g.tf),
        ("first_hidden_to_hidden", (g.tf, g.tf), g.tf),
        ("hidden_to_spks", (g.speakers, 2 * g.tf), g.speakers),
        ("single_hidden_to_spks", (g.speakers, g.tf), g.speakers),
        ("subpixel_upsample", (g.tf * g.stacking, g.tf, 3), g.tf * g.stacking),
    ]:
        entry(f"{m}.{name}.weight", "BF16", shape)
        entry(f"{m}.{name}.bias", "BF16", (bias,))
    entry(f"{m}.learnable_sil_emb", "BF16", (g.fc,))
    norm("perception.encoder.asr_norm", g.asr["d"])
    norm("perception.encoder.diar_norm", g.speakers)
    entry("perception.proj.weight", "BF16", (g.hidden, g.asr["d"]))
    entry("perception.proj.bias", "BF16", (g.hidden,))
    entry("backbone.embeddings.weight", "BF16", (g.vocab, g.hidden))
    entry("backbone.norm_f.weight", "BF16", (g.hidden,))
    entry("lm_head.weight", "BF16", (g.vocab, g.hidden))
    for index, kind in enumerate(g.layers):
        layer = f"backbone.layers.{index}"
        x = f"{layer}.mixer"
        entry(f"{layer}.norm.weight", "BF16", (g.hidden,))
        if kind == "mamba":
            for name in ("A_log", "D", "dt_bias"):
                entry(f"{x}.{name}", "BF16", (g.m_heads,))
            entry(f"{x}.conv1d.weight", "BF16", (g.m_conv, 1, g.m_kernel))
            entry(f"{x}.conv1d.bias", "BF16", (g.m_conv,))
            entry(f"{x}.norm.weight", "BF16", (g.m_inner,))
            linear(f"{x}.in_proj", g.m_in_rows, g.hidden, False)
            linear(f"{x}.out_proj", g.hidden, g.m_inner, False)
        elif kind == "moe":
            entry(f"{x}.gate.weight", "BF16", (g.experts, g.hidden))
            entry(f"{x}.gate.e_score_correction_bias", "F32", (g.experts,))
            for e in range(g.experts):
                linear(f"{x}.experts.{e}.up_proj", g.expert_width, g.hidden, False)
                linear(f"{x}.experts.{e}.down_proj", g.hidden, g.expert_width, False)
            linear(f"{x}.shared_experts.up_proj", g.shared_width, g.hidden, False)
            linear(f"{x}.shared_experts.down_proj", g.hidden, g.shared_width, False)
        else:
            fp8(f"{x}.q_proj", g.a_heads * g.a_head_dim, g.hidden)
            fp8(f"{x}.k_proj", g.a_kv * g.a_head_dim, g.hidden, "k_scale")
            fp8(f"{x}.v_proj", g.a_kv * g.a_head_dim, g.hidden, "v_scale")
            fp8(f"{x}.o_proj", g.hidden, g.a_heads * g.a_head_dim)
    return out


# ---------------------------------------------------------- tensor sources

TORCH_DTYPE_NAME = {torch.bfloat16: "BF16", torch.float32: "F32", torch.uint8: "U8",
                    torch.int64: "I64"}
if hasattr(torch, "float8_e4m3fn"):
    TORCH_DTYPE_NAME[torch.float8_e4m3fn] = "F8_E4M3"


class SafetensorsSource:
    """A checkpoint read through its `model.safetensors.index.json`."""

    def __init__(self, model_dir):
        from safetensors import safe_open

        self._safe_open = safe_open
        self.model_dir = Path(model_dir)
        index = json.loads((self.model_dir / "model.safetensors.index.json").read_text())
        self.where = {name: file for name, file in index["weight_map"].items()
                      if (self.model_dir / file).is_file()}
        self._handles = {}

    def _handle(self, file):
        if file not in self._handles:
            self._handles[file] = self._safe_open(str(self.model_dir / file), framework="pt",
                                                  device="cpu")
        return self._handles[file]

    def names(self):
        return set(self.where)

    def has(self, name):
        return name in self.where

    def dtype(self, name):
        return self._handle(self.where[name]).get_slice(name).get_dtype()

    def shape(self, name):
        return tuple(self._handle(self.where[name]).get_slice(name).get_shape())

    def tensor(self, name):
        return self._handle(self.where[name]).get_tensor(name)


def dequantized(src, module, device="cpu"):
    """`module.weight` as F32: FP8 codes times the per-tensor scale (W8A16)."""
    weight = src.tensor(f"{module}.weight")
    if TORCH_DTYPE_NAME.get(weight.dtype) == "F8_E4M3":
        scale = src.tensor(f"{module}.weight_scale").to(torch.float32)
        return weight.to(device).to(torch.float32) * scale.to(device)
    return weight.to(device=device, dtype=torch.float32)


def module_state(module_keys, src, prefix, device, rename=None):
    """A state dict for `module_keys` from checkpoint names `prefix + key`
    (FP8 weights dequantized). Returns (state, consumed checkpoint names)."""
    state, used, missing = {}, set(), []
    for key in module_keys:
        name = prefix + (rename(key) if rename else key)
        if not src.has(name):
            missing.append(name)
            continue
        if key.endswith(".weight") and src.dtype(name) == "F8_E4M3":
            module = name[: -len(".weight")]
            state[key] = dequantized(src, module, device)
            used.update({name, f"{module}.weight_scale", f"{module}.input_scale"})
        else:
            state[key] = src.tensor(name).to(device=device, dtype=torch.float32)
            used.add(name)
    if missing:
        raise RuntimeError(f"checkpoint lacks {len(missing)} tensors the module needs: {missing[:12]}")
    return state, used


# ------------------------------------------------------------ audio side

def bf16_round(tensor):
    return tensor.to(torch.bfloat16).to(tensor.dtype)


def round_constant_tables(module):
    """Upstream stores RoPE cos/sin (and the speaker kernel) in BF16."""
    rounded = []
    for name, sub in module.named_modules():
        if type(sub).__name__ in ("RotaryPositionalEncoding", "RotaryEmbedding"):
            for attr in ("cos", "sin"):
                table = getattr(sub, attr, None)
                if isinstance(table, torch.Tensor) and table.numel():
                    setattr(sub, attr, bf16_round(table.to(torch.float32)))
                    rounded.append(f"{name}.{attr}")
        kernel = getattr(sub, "diar_kernel", None)
        if isinstance(kernel, torch.Tensor):
            sub.diar_kernel = bf16_round(kernel.to(torch.float32))
            rounded.append(f"{name}.diar_kernel")
    return rounded


def plan_windows(total_frames, valid_frames, stacking):
    """NeMo `_forward_online_impl` windows over T = Tv + 1 frames (izwi
    perception.rs `plan_windows`)."""
    core = ONLINE_WINDOW_TOKENS * stacking
    context = ONLINE_CONTEXT_TOKENS * stacking
    count = max(1, math.ceil(total_frames / core))
    windows, first = [], 0
    for k in range(count):
        start = k * core
        end = min(start + core, total_frames)
        cs = max(start - context, 0)
        ce = min(end + context, total_frames)
        left = (start - cs) // stacking
        core_frames = min(-(-end // stacking) - (-(-start // stacking)),
                          -(-(ce - cs) // stacking) - left)
        windows.append({
            "index": k, "start": start, "end": end, "context_start": cs, "context_end": ce,
            "valid_mel": max(0, min(valid_frames - cs, ce - cs)), "left_drop": left,
            "core_frames": core_frames, "first_token": first, "updates_state": k < count - 1,
        })
        first += core_frames
    return windows


class PerceptionResult:
    """What one clip's perception produced (all on the host, F32)."""

    def __init__(self):
        self.mel = None              # [n_mels, Tv + 1] BF16 values as F32
        self.valid_frames = 0
        self.mean = None
        self.std = None
        self.normalized = None       # [n_mels, Tv + 1] BF16-rounded
        self.windows = []
        self.asr_cores = []          # per window [d_model, core]
        self.diar_80ms = []          # per window [core, speakers] (aligned)
        self.fused = None            # [d_model, T_out]
        self.embeddings = None       # [audio_tokens, hidden]
        self.ctc_inputs = None


def build_nemo_perception(cfg, device):
    from omegaconf import OmegaConf
    from nemo.collections.speechlm2.modules import AudioPerceptionModule
    from nemo.collections.asr.modules.parallel_expert_encoder import ParallelExpertEncoderPT

    # As the vLLM plugin mounts it (nemo/collections/speechlm2/vllm/salm/audio.py).
    perception = AudioPerceptionModule(OmegaConf.create(decode_nemo_floats(cfg["perception"])))
    encoder = ParallelExpertEncoderPT.from_inline_config(
        decode_nemo_floats(cfg["pe_encoder_config"]), map_location="cpu")
    try:
        perception.preprocessor.featurizer.normalize = None
    except AttributeError:
        pass
    perception.encoder = encoder
    encoder._suppress_online_pbar = True
    return perception.to(device=device, dtype=torch.float32).eval()


def build_sglang_perception(cfg, device):
    from sglang_omni.models.nemotron_transcribe.perception import build_audio_perception

    perception = build_audio_perception(cfg["perception"], cfg["pe_encoder_config"])
    return perception.to(device=device, dtype=torch.float32).eval()


def load_perception(perception, src, device):
    keys = list(perception.state_dict().keys())
    state, used = module_state(keys, src, "perception.", device)
    perception.load_state_dict(state, strict=True)
    leftover = sorted(n for n in src.names() if n.startswith("perception.") and n not in used)
    if leftover:
        raise RuntimeError(f"checkpoint perception tensors the module did not take: {leftover[:12]}")
    return round_constant_tables(perception)


def _mel_and_stats(preprocessor, samples, impl):
    x = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32))[None]
    n = torch.tensor([len(samples)], dtype=torch.int64)
    if impl == "nemo":
        mel, mel_len = preprocessor(input_signal=x, length=n)
    else:
        mel, mel_len = preprocessor(x, n)
    mel = mel.to(torch.float32)
    return mel.to(torch.bfloat16), mel_len


def run_nemo_perception(perception, samples, device, ctc=False):
    from nemo.collections.asr.parts.preprocessing.features import normalize_batch

    pee = perception.encoder
    preprocessor = perception.preprocessor
    tensors = list(preprocessor.parameters()) + list(preprocessor.buffers())
    if any(t.device.type != "cpu" for t in tensors):
        # The mel is host work in izwi; compute the reference on the host too.
        preprocessor = copy.deepcopy(preprocessor).cpu()
    mel_bf16, mel_len = _mel_and_stats(preprocessor, samples, "nemo")
    result = PerceptionResult()
    result.valid_frames = int(mel_len[0])
    result.mel = mel_bf16[0].to(torch.float32)
    normalized, mean, std = normalize_batch(mel_bf16, mel_len, normalize_type="per_feature")
    result.mean = mean[0].to(torch.float32)
    result.std = std[0].to(torch.float32)
    result.normalized = bf16_round(normalized[0].to(torch.float32))

    cap = {"windows": None, "cores": None, "diar": [], "fused": None}
    original = {}

    def wrap(name, fn):
        original[name] = getattr(pee, name)
        setattr(pee, name, fn)

    def encode(asr_signal, asr_length, windows):
        cores = original["_encode_asr_windows"](asr_signal, asr_length, windows)
        cap["windows"] = [tuple(int(v) for v in w) for w in windows]
        cap["cores"] = [c.detach().to(torch.float32).cpu() for c in cores]
        return cores

    def window_forward(rows, lengths):
        # Upstream casts the normalized mel to the BF16 encoder dtype here.
        return original["_asr_window_forward"](bf16_round(rows), lengths)

    def diar_step(chunk, chunk_length, state, empty_preds, **kwargs):
        state, preds = original["_streaming_diar_step"](chunk, chunk_length, state, empty_preds,
                                                         **kwargs)
        cap["diar"].append((preds.detach(), bool(kwargs.get("return_native_resolution"))))
        return state, preds

    def fuse(asr_encoded, spk_targets, **kwargs):
        out = original["_fuse_diar_and_asr"](asr_encoded, spk_targets, **kwargs)
        cap["fused"] = out.detach().to(torch.float32).cpu()
        return out

    wrap("_encode_asr_windows", encode)
    wrap("_asr_window_forward", window_forward)
    wrap("_streaming_diar_step", diar_step)
    wrap("_fuse_diar_and_asr", fuse)
    try:
        with pee.online_inference():
            outputs = pee(audio_signal=mel_bf16.to(device), length=mel_len.to(device),
                          return_ctc_timestamp_inputs=ctc)
    finally:
        for name in original:
            delattr(pee, name)
    fused, enc_len = outputs[0], outputs[1]
    if ctc:
        result.ctc_inputs = outputs[2]
    encoded, enc_len = perception.modality_adapter(audio_signal=fused, length=enc_len)
    embeddings = perception.proj(encoded.transpose(1, 2))
    tokens = int(enc_len[0])
    result.embeddings = embeddings[0, :tokens].to(torch.float32).cpu()
    result.fused = cap["fused"][0]

    total = result.mel.shape[1]
    result.windows = plan_windows(total, result.valid_frames, pee.subsampling_factor)
    nemo_plan = [(w["start"], w["end"], w["context_start"], w["context_end"]) for w in result.windows]
    if cap["windows"] != nemo_plan:
        raise RuntimeError(f"NeMo ran windows {cap['windows']}, izwi plans {nemo_plan}")
    modules = pee.diarization_model.sortformer_modules
    factor = int(pee.diarization_model.output_subsampling_factor)
    if len(cap["diar"]) != len(result.windows):
        raise RuntimeError(f"NeMo ran {len(cap['diar'])} Sortformer steps for "
                           f"{len(result.windows)} windows")
    for window, core, (preds, native) in zip(result.windows, cap["cores"], cap["diar"]):
        if native:
            preds = modules.downsample_preds(preds, factor)
        aligned = type(pee)._align_diar_frames(preds, core.shape[-1])
        result.asr_cores.append(core[0])
        result.diar_80ms.append(aligned[0].to(torch.float32).cpu())
        if core.shape[-1] != window["core_frames"]:
            raise RuntimeError(f"window {window['index']}: NeMo core {core.shape[-1]} frames, "
                               f"izwi plans {window['core_frames']}")
    return result


def _sglang_normalize(mel_bf16, mel_len):
    """NeMo's normalize_batch(per_feature) formula, run on the BF16 tensor."""
    x = mel_bf16
    frames = x.shape[2]
    valid = torch.arange(frames)[None] < mel_len[:, None]
    denominator = valid.sum(1)
    reference = x[:, :, 0]
    centred = torch.where(valid[:, None], x - reference[:, :, None], 0.0)
    mean = reference + centred.sum(2) / denominator.clamp_min(1)[:, None]
    std = torch.sqrt(torch.sum(torch.where(valid[:, None], x - mean[:, :, None], 0.0) ** 2, 2)
                     / (denominator[:, None] - 1.0))
    std = std.masked_fill(std.isnan(), 0.0) + 1e-5
    normalized = (x - mean[:, :, None]) / std[:, :, None]
    normalized = normalized.masked_fill(~valid[:, None], 0.0)
    return normalized, mean, std


def run_sglang_perception_impl(perception, samples, device):
    from sglang_omni.models.nemotron_diarization.speaker_cache import SpeakerCache
    from sglang_omni.models.nemotron_transcribe.mel_features import normalize_features

    enc = perception.encoder
    preprocessor = copy.deepcopy(perception.preprocessor).cpu()
    mel_bf16, mel_len = _mel_and_stats(preprocessor, samples, "sglang")
    result = PerceptionResult()
    result.valid_frames = int(mel_len[0])
    result.mel = mel_bf16[0].to(torch.float32)
    normalized, mean, std = _sglang_normalize(mel_bf16, mel_len)
    result.mean = mean[0].to(torch.float32)
    result.std = std[0].to(torch.float32)
    result.normalized = bf16_round(normalized[0].to(torch.float32))

    mel_bf16 = mel_bf16.to(device)
    mel_len = mel_len.to(device)
    asr_mel = bf16_round(normalize_features(mel_bf16, mel_len, enc.asr_normalize).to(torch.float32))
    diar_mel = normalize_features(mel_bf16, mel_len, enc.diar_normalize).to(torch.float32)
    speakers = enc.diarization_model.sortformer_modules.speakers
    state = SpeakerCache(diar_mel, cache_size=enc.cache_limit, fifo_size=enc.recent_limit,
                         update_period=enc.cache_update_frames, batch=1,
                         width=enc.diarization_model.encoder.d_model, speakers=speakers)
    total = torch.zeros((1, 0, speakers), device=device, dtype=torch.float32)
    stacking = enc.subsampling_factor
    result.windows = plan_windows(mel_bf16.shape[-1], result.valid_frames, stacking)
    asr_parts, diar_parts = [], []
    for window in result.windows:
        cs, ce = window["context_start"], window["context_end"]
        lengths = (mel_len - cs).clamp(min=0, max=ce - cs)
        encoded, _ = enc.asr_encoder(asr_mel[:, :, cs:ce], lengths)
        left = window["left_drop"]
        core = encoded[:, :, left:left + window["core_frames"]]
        asr_parts.append(core)
        previous = total.shape[1]
        total = enc.diarization_model.streaming_step(
            state, total, diar_mel[:, :, cs:ce], lengths, window["start"] - cs, ce - window["end"])
        aligned = _align_time(total[:, previous:], core.shape[-1])
        diar_parts.append(aligned)
        result.asr_cores.append(core[0].to(torch.float32).cpu())
        result.diar_80ms.append(aligned[0].to(torch.float32).cpu())
        # SGLang's step always updates the state; after the last window the
        # update is discarded, so outputs match NeMo's skipped update.
    fused = enc.fuse(torch.cat(asr_parts, dim=2), torch.cat(diar_parts, dim=1))
    tokens = -(-result.valid_frames // stacking)
    result.fused = fused[0].to(torch.float32).cpu()
    result.embeddings = perception.proj(fused.transpose(1, 2))[0, :tokens].to(torch.float32).cpu()
    return result


def _align_time(activity, frames):
    if activity.shape[1] < frames:
        last = activity[:, -1:]
        activity = torch.cat([activity, last.repeat(1, frames - activity.shape[1], 1)], dim=1)
    return activity[:, :frames]


def sortformer_schedule(windows, fifo_max=40, update_period=300, cache_max=188):
    """(cache frames, FIFO frames, compressed yet) before each window, by the
    streaming update rule izwi ports (sortformer.rs `update`)."""
    cache = fifo = 0
    compressed = False
    out = []
    for window in windows:
        out.append((cache, fifo, compressed))
        if not window["updates_state"]:
            continue
        chunk = window["core_frames"]
        if fifo + chunk > fifo_max:
            pop = min(max(update_period, fifo + chunk - fifo_max), fifo + chunk)
            cache += pop
            fifo = fifo + chunk - pop
            if cache > cache_max:
                cache, compressed = cache_max, True
        else:
            fifo += chunk
    return out


def perception_record(result, g):
    """The per-clip perception part of the fixture schema."""
    tv = result.valid_frames
    frames = sorted({0, tv // 2, max(tv - 1, 0), tv})
    mel = {
        "valid_frames": tv,
        "frames": int(result.mel.shape[1]),
        "mean": f32list(result.mean),
        "std": f32list(result.std),
        "raw_rows": [{"frame": f, "values": f32list(result.mel[:, f])} for f in frames],
        "normalized_rows": [{"frame": f, "values": f32list(result.normalized[:, f])}
                            for f in frames],
    }
    windows, near = [], 0
    schedule = sortformer_schedule(result.windows)
    for window, core, diar, (cache, fifo, compressed) in zip(
            result.windows, result.asr_cores, result.diar_80ms, schedule):
        n = core.shape[-1]
        rows = sorted({0, n // 2, max(n - 1, 0)}) if n else []
        record = dict(window)
        record["asr_core_rows"] = [{"row": r, "values": f32list(core[:, r])} for r in rows]
        record["diar_probs_80ms"] = [f32list(row) for row in diar]
        record["sortformer_cache_frames_before"] = cache
        record["sortformer_fifo_frames_before"] = fifo
        record["follows_compression"] = compressed
        near += int(((diar - 0.5).abs() < 1e-3).sum())
        windows.append(record)
    tokens = int(result.embeddings.shape[0])
    picks = sorted({0, tokens // 2, tokens - 1,
                    *[w["first_token"] for w in result.windows if w["first_token"] < tokens]})
    return {
        "mel": mel,
        "audio_tokens": tokens,
        "windows": windows,
        "diar_near_threshold": near,
        "fused_rows": [{"token": t, "values": f32list(result.fused[:, t])} for t in picks],
        "embeddings_rows": [{"token": t, "values": f32list(result.embeddings[t])} for t in picks],
    }


# ----------------------------------------------------------------- decoder

def nemotron_h_config(llm, attn="eager"):
    """transformers' NemotronHConfig from the checkpoint's llm_config."""
    from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig

    keys = ["vocab_size", "hidden_size", "tie_word_embeddings", "num_attention_heads",
            "num_key_value_heads", "head_dim", "attention_bias", "mlp_hidden_act", "mlp_bias",
            "ssm_state_size", "mamba_num_heads", "mamba_head_dim", "mamba_hidden_act", "n_groups",
            "conv_kernel", "chunk_size", "use_conv_bias", "mamba_proj_bias",
            "mamba_ssm_cache_dtype", "n_routed_experts", "n_shared_experts",
            "moe_intermediate_size", "moe_shared_expert_intermediate_size", "moe_latent_size",
            "num_experts_per_tok", "routed_scaling_factor", "n_group", "topk_group",
            "norm_topk_prob", "layer_norm_epsilon", "intermediate_size", "pad_token_id",
            "bos_token_id", "eos_token_id", "max_position_embeddings"]
    kwargs = {k: llm[k] for k in keys if k in llm}
    kwargs["layers_block_type"] = [TRANSFORMERS_LAYER[k] for k in llm["layers_block_type"]]
    kwargs.update(use_bias=False, residual_in_fp32=False, rescale_prenorm_residual=False,
                  num_nextn_predict_layers=0)
    config = NemotronHConfig(**kwargs)
    for name, value in (("_attn_implementation", attn), ("_experts_implementation", "eager")):
        try:
            setattr(config, name, value)
        except Exception:  # noqa: BLE001
            try:
                setattr(config, f"{name}_internal", value)
            except Exception:  # noqa: BLE001
                pass
    return config


class RefCache:
    """The cache protocol transformers' NemotronH mixers call, held per layer."""

    METHODS = {"get_conv_state", "get_recurrent_state", "is_recording_past", "update_conv_state",
               "update_recurrent_state", "update", "get_seq_length"}

    def __init__(self, conv_kernel):
        self.kernel = conv_kernel
        self.conv, self.ssm, self.keys, self.values = {}, {}, {}, {}

    def get_conv_state(self, layer_idx, state_idx=0):
        return self.conv.get(layer_idx)

    def get_recurrent_state(self, layer_idx, state_idx=0):
        return self.ssm.get(layer_idx)

    def is_recording_past(self, layer_idx):
        return False

    def update_conv_state(self, hidden_states, layer_idx, conv_kernel_size=None, **_):
        width = (conv_kernel_size or self.kernel) - 1
        previous = self.conv.get(layer_idx)
        if previous is None:
            previous = hidden_states.new_zeros(*hidden_states.shape[:2], width)
        full = torch.cat([previous, hidden_states], dim=-1)
        self.conv[layer_idx] = full[:, :, -width:].clone()
        return full

    def update_recurrent_state(self, state, layer_idx, **_):
        self.ssm[layer_idx] = state.clone()

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        if layer_idx in self.keys:
            self.keys[layer_idx] = torch.cat([self.keys[layer_idx], key_states], dim=2)
            self.values[layer_idx] = torch.cat([self.values[layer_idx], value_states], dim=2)
        else:
            self.keys[layer_idx] = key_states
            self.values[layer_idx] = value_states
        return self.keys[layer_idx], self.values[layer_idx]

    def get_seq_length(self, layer_idx=0):
        for keys in self.keys.values():
            return int(keys.shape[2])
        return 0


def check_transformers_api():
    """Fail closed when transformers' NemotronH internals moved."""
    from transformers.models.nemotron_h import modeling_nemotron_h as mod

    # The whole module's source: decorators may wrap the mixer's forward.
    source = inspect.getsource(mod)
    called = set(re.findall(r"cache_params\.(\w+)", source))
    unknown = called - RefCache.METHODS
    problems = []
    if unknown:
        problems.append(f"Mamba mixer calls cache methods {sorted(unknown)}")
    if "self.time_step_limit" not in source:
        problems.append("Mamba mixer has no time_step_limit")
    for name in ("NemotronHBlock", "NemotronHMoE", "NemotronHExperts", "NemotronHTopkRouter",
                 "NemotronHRMSNorm", "NemotronHAttention"):
        if not hasattr(mod, name):
            problems.append(f"transformers lacks {name}")
    if problems:
        raise RuntimeError("transformers' NemotronH API differs from bb666bd0: " + "; ".join(problems))
    return mod


class RestatedExperts(torch.nn.Module):
    """`NemotronHExperts.forward` restated over per-expert F32 weights fetched
    on demand (`fetch(e) -> (up [I, H], down [H, I])`); used when the stacked
    upstream module would not fit (L2) or its implementation dispatch fails."""

    def __init__(self, experts, fetch, act):
        super().__init__()
        self.num_experts = experts
        self._fetch = fetch
        self.act_fn = act

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states, dtype=top_k_weights.dtype)
        mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero().squeeze(-1)
        for expert in hit.tolist():
            if expert >= self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(mask[expert])
            if token_idx.numel() == 0:
                continue
            up, down = self._fetch(expert)
            h = self.act_fn(F.linear(hidden_states[token_idx], up))
            h = F.linear(h, down) * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, h.to(final.dtype))
        return final.to(hidden_states.dtype)


class RefDecoder:
    """Nemotron-H run block by block with upstream modules: per-layer traces,
    router capture, span prefill and cached decode."""

    def __init__(self, g, config, blocks, embeddings_bf16, norm_f, lm_head, device, eos):
        self.g, self.config, self.blocks = g, config, blocks
        self.embed_table = embeddings_bf16
        self.norm_f, self.lm_head = norm_f, lm_head
        self.device, self.eos = device, list(eos)
        self.gate_hooks = []
        self.trace = None

    def embed(self, ids):
        index = torch.tensor(ids, dtype=torch.long, device=self.embed_table.device)
        return self.embed_table[index].to(device=self.device, dtype=torch.float32)

    @staticmethod
    def causal_mask(tokens, past, device):
        if tokens == 1:
            return None
        mask = torch.full((tokens, past + tokens), float("-inf"), device=device)
        return torch.triu(mask, diagonal=past + 1)[None, None]

    def run(self, x, cache, past, on_layer=None):
        """x [1, T, H] at positions past..past+T-1. Returns the residual
        stream after the last layer [1, T, H]."""
        mask = self.causal_mask(x.shape[1], past, x.device)
        self.position0 = past
        h = x
        for index, block in enumerate(self.blocks):
            if block.block_type == "full_attention":
                h = block(h, past_key_values=cache, attention_mask=mask)
            else:
                h = block(h, past_key_values=cache)
            if on_layer is not None:
                on_layer(index, h, past)
        return h

    def logits(self, h):
        return F.linear(self.norm_f(h), self.lm_head)

    def prefill(self, embeds, span=None, on_layer=None):
        cache = RefCache(self.g.m_kernel)
        total = embeds.shape[1]
        span = span or total
        h_last = None
        for start in range(0, total, span):
            h = self.run(embeds[:, start:start + span], cache, start, on_layer)
            h_last = h[:, -1]
        return cache, self.logits(h_last)[0]

    def step(self, token, cache, past):
        h = self.run(self.embed([token])[None], cache, past)
        return self.logits(h[:, -1])[0]

    def full_logits(self, embeds):
        h = self.run(embeds, None, 0)
        return self.logits(h[:, -1])[0]


def _parameters_on_meta(module):
    return [n for n, p in list(module.named_parameters()) + list(module.named_buffers())
            if p.device.type == "meta"]


def build_decoder(cfg, src, device, expert_mode="stacked", embed_device=None, log_every=0):
    """Upstream NemotronH blocks with dequantized weights. `expert_mode`:
    "stacked" (upstream NemotronHExperts with F32 [E, I, H] parameters) or a
    callable `expert_fetcher(layer) -> fetch(e)` for restated experts."""
    mod = check_transformers_api()
    from transformers.activations import ACT2FN

    g = Geometry(cfg)
    llm = decode_nemo_floats(cfg["llm_config"])
    config = nemotron_h_config(llm)
    blocks = []
    for index in range(len(g.layers)):
        with torch.device("meta"):
            block = mod.NemotronHBlock(config, layer_idx=index)
        prefix = f"backbone.layers.{index}."
        restated = block.block_type == "moe" and callable(expert_mode)
        if restated:
            block.mixer.experts = RestatedExperts(g.experts, expert_mode(index),
                                                  ACT2FN[config.mlp_hidden_act])
        block = block.to_empty(device=device)
        keys = list(block.state_dict().keys())
        stacked = [k for k in keys if k in ("mixer.experts.up_proj", "mixer.experts.down_proj")]
        plain = [k for k in keys if k not in stacked]
        state, _ = module_state(plain, src, prefix, device)
        for key in stacked:
            which = key.rsplit(".", 1)[1]
            state[key] = torch.stack([
                dequantized(src, f"{prefix}mixer.experts.{e}.{which}", device)
                for e in range(g.experts)])
        block.load_state_dict(state, strict=True)
        meta = _parameters_on_meta(block)
        if meta:
            raise RuntimeError(f"layer {index}: {meta} were never materialized")
        if block.block_type == "linear_attention":
            # vLLM / training semantics: dt in [0, inf), no prefill clamp.
            block.mixer.time_step_limit = (0.0, float("inf"))
        blocks.append(block.eval())
        if log_every and (index + 1) % log_every == 0:
            log(f"decoder layer {index + 1}/{len(g.layers)} loaded")
    norm_f = mod.NemotronHRMSNorm(g.hidden, eps=g.eps).to(device)
    norm_f.weight.data.copy_(src.tensor("backbone.norm_f.weight").to(device, torch.float32))
    embeddings = src.tensor("backbone.embeddings.weight").to(embed_device or device)
    lm_head = src.tensor("lm_head.weight").to(device=device, dtype=torch.float32)
    eos = cfg.get("eos_token_id") or [11]
    eos = eos if isinstance(eos, list) else [eos]
    return RefDecoder(g, config, blocks, embeddings, norm_f.eval(), lm_head, device, eos)


def izwi_routes(logits, bias, top_k, scaling):
    """izwi moe.rs `route`: selection by sigmoid + F32 bias (ties to the lower
    id), weights from the unbiased scores renormalized and scaled."""
    scores = torch.sigmoid(logits.to(torch.float32))
    biased = scores + bias.to(torch.float32)
    order = torch.sort(biased, dim=-1, descending=True, stable=True).indices
    experts = order[:, :top_k]
    picked = scores.gather(1, experts)
    weights = picked / picked.sum(-1, keepdim=True) * scaling
    if order.shape[1] > top_k:
        sorted_biased = biased.gather(1, order)
        margin = sorted_biased[:, top_k - 1] - sorted_biased[:, top_k]
    else:
        margin = torch.full((logits.shape[0],), float("inf"), device=logits.device)
    return experts, weights, margin


def ends_in_loop(tokens):
    n = len(tokens)
    for block in range(1, min(LOOP_MAX_BLOCK_TOKENS, n // LOOP_REPEATS) + 1):
        start = n - block * LOOP_REPEATS
        if all(tokens[i] == tokens[i - block] for i in range(start + block, n)):
            return True
    return False


def guardrail(text):
    text = re.sub(r"(.)\1{20,}", r"\1", text, flags=re.DOTALL)
    return re.sub(r"(.{1,4096}?)\1{19,}", r"\1", text, flags=re.DOTALL)


def render_prompt_text(instruction):
    return f"{SYSTEM_TURN}{USER_OPEN}{instruction} {AUDIO_TOKEN}{ASSISTANT_OPEN}"


def prompt_layout(tokenizer, instruction, audio_tokens):
    audio_id = tokenizer.token_to_id(AUDIO_TOKEN)
    ids = tokenizer.encode(render_prompt_text(instruction), add_special_tokens=False).ids
    if ids.count(audio_id) != 1:
        raise RuntimeError(f"the prompt holds {ids.count(audio_id)} audio placeholders")
    at = ids.index(audio_id)
    return ids[:at], ids[at + 1:], audio_id


def run_decoder_case(decoder, tokenizer, instruction, embeddings, record_greedy, max_new_tokens,
                     trace_layers, route_layer, positions_cap=64, span=None,
                     cached_checks=(1, 2, 4), stream_text=True):
    """Prefill, traces, greedy decode and the self-checks of one prompt."""
    g = decoder.g
    before, after, audio_id = prompt_layout(tokenizer, instruction, embeddings.shape[0])
    audio = embeddings.to(decoder.device)
    embeds = torch.cat([decoder.embed(before), audio, decoder.embed(after)], dim=0)[None]
    prompt_ids = before + [audio_id] * audio.shape[0] + after
    total = len(prompt_ids)
    audio_offset = len(before)
    positions = sorted({0, audio_offset, audio_offset + audio.shape[0] - 1, total - 1})
    kinds = {i: g.layers[i] for i in trace_layers}
    hidden_rows = {i: {} for i in trace_layers}
    sums = {i: [0.0, 0.0, 0.0] for i in trace_layers}

    def on_layer(index, h, past):
        if index not in hidden_rows:
            return
        rows = h[0].to(torch.float64)
        sums[index][0] += float(rows.sum())
        sums[index][1] += float(rows.abs().sum())
        sums[index][2] += float((rows * rows).sum())
        for p in positions:
            if past <= p < past + h.shape[1]:
                hidden_rows[index][p] = f32list(h[0, p - past])

    route_rows, near_by_layer, disagreements = [], {}, [0]
    route_positions = sorted(set(range(min(8, total))) | set(range(max(0, total - 8), total))
                             | set(range(0, total, max(1, total // positions_cap))))

    def gate_hook(layer):
        def hook(module, inputs, output):
            logits, _, chosen = output
            experts, weights, margin = izwi_routes(logits, module.e_score_correction_bias,
                                                   g.top_k, g.routed_scaling)
            near_by_layer[layer] = near_by_layer.get(layer, 0) + int((margin < NEAR_TIE).sum())
            if layer == route_layer:
                same = (experts.sort(-1).values == chosen.sort(-1).values).all(-1)
                disagreements[0] += int((~same).sum())
                base = decoder.position0
                for row in range(logits.shape[0]):
                    p = base + row
                    if p in route_positions_set:
                        route_rows.append({"position": p, "experts": experts[row].tolist(),
                                           "weights": f32list(weights[row]),
                                           "margin": f32(margin[row])})
        return hook

    route_positions_set = set(route_positions)
    handles = [block.mixer.gate.register_forward_hook(gate_hook(i))
               for i, block in enumerate(decoder.blocks) if block.block_type == "moe"]
    try:
        cache, logits = decoder.prefill(embeds, span=span, on_layer=on_layer)
    finally:
        for handle in handles:
            handle.remove()

    top = torch.topk(logits, 8)
    prefill_top = [{"id": int(i), "logit": f32(v)} for v, i in zip(top.values, top.indices)]
    generated, margins, finish, stop = [], [], "length", None
    step_logits = {}
    past = total
    while True:
        values = torch.topk(logits, 2).values
        token = int(torch.argmax(logits))
        margins.append(f32(values[0] - values[1]))
        if token in decoder.eos:
            finish, stop = "stop", token
            break
        generated.append(token)
        if len(generated) in cached_checks:
            step_logits[len(generated)] = None
        if len(generated) >= max_new_tokens:
            finish = "length"
            break
        if ends_in_loop(generated):
            finish = "repetition"
            break
        logits = decoder.step(token, cache, past)
        past += 1
        if len(generated) in step_logits:
            step_logits[len(generated)] = logits.detach().clone()
        if stream_text and len(generated) % 256 == 0:
            log(f"  {len(generated)} tokens")

    checks = {}
    for count, cached in sorted(step_logits.items()):
        if cached is None:
            continue
        full = decoder.full_logits(torch.cat([embeds, decoder.embed(generated[:count])[None]], 1))
        scale = float(full.abs().max())
        checks[f"cached_vs_full_step{count}_max_abs"] = float((cached - full).abs().max())
        checks[f"cached_vs_full_step{count}_scale"] = scale
    decoded = tokenizer.decode(generated, skip_special_tokens=False)
    record = {
        "instruction": instruction,
        "prompt_ids": prompt_ids,
        "audio_offset": audio_offset,
        "prompt_tokens": total,
        "layer_hidden": [{
            "layer": i, "kind": kinds[i],
            "rows": [{"position": p, "values": hidden_rows[i][p]} for p in positions],
            "checksum": {"sum": sums[i][0], "abs_sum": sums[i][1],
                         "rms": math.sqrt(sums[i][2] / (total * g.hidden))},
        } for i in trace_layers],
        "routes": {"layer": route_layer, "positions": total,
                   "near_ties": near_by_layer.get(route_layer, 0),
                   "near_ties_by_layer": {str(k): v for k, v in sorted(near_by_layer.items())},
                   "transformers_disagreements": disagreements[0],
                   "rows": sorted(route_rows, key=lambda r: r["position"])},
        "prefill_logits_top8": prefill_top,
        "prefill_margin": f32(top.values[0] - top.values[1]),
        "greedy_ids": generated[:record_greedy],
        "greedy_margins": margins[:record_greedy],
        "generated_tokens": len(generated),
        "finish_reason": finish,
        "stop_token": stop,
        "decoded": decoded,
        "text": guardrail(decoded.strip()),
    }
    return record, checks, (embeds, cache)


def mamba_f64_check(decoder, embeds):
    """The first Mamba layer against an independent F64 recurrence."""
    g = decoder.g
    index = g.first("mamba")
    block = decoder.blocks[index]
    mixer = block.mixer
    x = embeds
    for earlier in decoder.blocks[:index]:
        if earlier.block_type == "full_attention":
            x = earlier(x, attention_mask=decoder.causal_mask(x.shape[1], 0, x.device))
        else:
            x = earlier(x)
    h = block.norm(x)
    got = mixer(h, cache_params=None)[0].to(torch.float64)
    h = h[0].to(torch.float64)
    proj = h @ mixer.in_proj.weight.to(torch.float64).T
    z, xbc, dt = torch.split(proj, [g.m_inner, g.m_conv, g.m_heads], dim=-1)
    w = mixer.conv1d.weight.to(torch.float64)[:, 0, :]
    b = mixer.conv1d.bias.to(torch.float64)
    T = xbc.shape[0]
    padded = torch.cat([xbc.new_zeros(g.m_kernel - 1, g.m_conv), xbc], 0)
    u = b + sum(w[:, k] * padded[k:k + T] for k in range(g.m_kernel))
    u = u * torch.sigmoid(u)
    xs = u[:, :g.m_inner].view(T, g.m_heads, g.m_head_dim)
    B = u[:, g.m_inner:g.m_inner + g.m_groups * g.m_state].view(T, g.m_groups, g.m_state)
    C = u[:, g.m_inner + g.m_groups * g.m_state:].view(T, g.m_groups, g.m_state)
    dt = F.softplus(dt + mixer.dt_bias.to(torch.float64))
    A = -torch.exp(mixer.A_log.to(torch.float64))
    D = mixer.D.to(torch.float64)
    group = torch.arange(g.m_heads, device=h.device) // (g.m_heads // g.m_groups)
    state = h.new_zeros(g.m_heads, g.m_head_dim, g.m_state)
    ys = []
    for t in range(T):
        dA = torch.exp(A * dt[t])
        Bt, Ct = B[t][group], C[t][group]
        state = state * dA[:, None, None] + (dt[t][:, None] * Bt)[:, None, :] * xs[t][:, :, None]
        ys.append(torch.einsum("hpn,hn->hp", state, Ct) + D[:, None] * xs[t])
    y = torch.stack(ys).reshape(T, g.m_inner)
    gated = y * (z * torch.sigmoid(z))
    grouped = gated.view(T, g.m_groups, -1)
    grouped = grouped * torch.rsqrt(grouped.pow(2).mean(-1, keepdim=True) + g.eps)
    out = grouped.reshape(T, g.m_inner) * mixer.norm.weight.to(torch.float64)
    ref = out @ mixer.out_proj.weight.to(torch.float64).T
    return float((got - ref).norm() / ref.norm())


def moe_f64_check(decoder, seed=1):
    """The first MoE layer against an F64 restatement on random rows."""
    g = decoder.g
    block = decoder.blocks[g.first("moe")]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(1, 16, g.hidden, generator=gen).to(decoder.device)
    got = block.mixer(x)[0].to(torch.float64)
    gate = block.mixer.gate
    # Selection from the router's own F32 logits (a near-tie must not flip
    # between the module and this check); the expert math in F64.
    logits = F.linear(x[0].to(torch.float32), gate.weight.to(torch.float32))
    experts, weights, _ = izwi_routes(logits, gate.e_score_correction_bias,
                                      g.top_k, g.routed_scaling)
    fetch = getattr(block.mixer.experts, "_fetch", None)
    ref = torch.zeros(16, g.hidden, dtype=torch.float64, device=x.device)
    for row in range(16):
        for e, w in zip(experts[row].tolist(), weights[row].tolist()):
            if fetch is not None:
                up, down = fetch(e)
            else:
                up, down = block.mixer.experts.up_proj[e], block.mixer.experts.down_proj[e]
            h = torch.relu(x[0, row].to(torch.float64) @ up.to(torch.float64).T) ** 2
            ref[row] += w * (h @ down.to(torch.float64).T)
    shared = block.mixer.shared_experts
    h = torch.relu(x[0].to(torch.float64) @ shared.up_proj.weight.to(torch.float64).T) ** 2
    ref += h @ shared.down_proj.weight.to(torch.float64).T
    return float((got - ref).norm() / ref.norm())


# ------------------------------------------------------ the tiny checkpoint

TINY = {
    "hidden": 64, "vocab": 256, "layers": ["mamba", "moe", "attention", "mamba", "moe"],
    "encoder_width": 32, "encoder_heads": 2, "encoder_layers": 2, "experts": 8, "top_k": 2,
    "expert_width": 96, "shared_width": 64, "speakers": 8, "tf_width": 16,
}
# Per-speaker logit offsets of single_hidden_to_spks: they keep the random
# diarizer's sigmoids away from the 0.5 activity threshold, where a 1e-7
# difference would flip a whole speaker-kernel row.
SPEAKER_BIAS = [1.5, -1.5, 0.8, -0.8, 2.5, -2.5, 0.3, -3.0]


def tiny_config():
    published = json.loads((FIXTURE_DIR / "config.json").read_text())
    cfg = copy.deepcopy(published)
    t = TINY
    d = t["encoder_width"]
    cfg["perception"]["output_dim"] = t["hidden"]
    cfg["perception"]["modality_adapter"]["d_model"] = d
    for enc in (cfg["perception"]["encoder"], cfg["pe_encoder_config"]["asr_encoder_cfg"]):
        enc.update(d_model=d, n_heads=t["encoder_heads"], n_layers=t["encoder_layers"],
                   ff_expansion=4)
    diar = cfg["pe_encoder_config"]["diarization_model_cfg"]
    diar["max_num_of_spks"] = t["speakers"]
    diar["encoder"].update(d_model=d, n_heads=t["encoder_heads"], n_layers=t["encoder_layers"],
                           ff_expansion=2.0)
    diar["sortformer_modules"].update(num_spks=t["speakers"], fc_d_model=d,
                                      tf_d_model=t["tf_width"])
    llm = cfg["llm_config"]
    llm.update(hidden_size=t["hidden"], vocab_size=t["vocab"], layers_block_type=t["layers"],
               mamba_num_heads=4, mamba_head_dim=16, n_groups=2, ssm_state_size=8, chunk_size=8,
               n_routed_experts=t["experts"], num_experts_per_tok=t["top_k"],
               moe_intermediate_size=t["expert_width"], intermediate_size=t["expert_width"],
               moe_shared_expert_intermediate_size=t["shared_width"], num_attention_heads=4,
               num_key_value_heads=2, head_dim=16, num_nextn_predict_layers=0,
               max_position_embeddings=65536)
    exclude = ["backbone.embeddings", "lm_head"]
    for index, kind in enumerate(t["layers"]):
        if kind == "mamba":
            exclude.append(f"backbone.layers.{index}.mixer.conv1d")
        elif kind == "moe":
            exclude.append(f"backbone.layers.{index}.mixer.gate")
    exclude += ["ctc_timestamp*", "perception.encoder.diarization_model*",
                "perception.encoder.asr_encoder.pre_encode*", "perception.proj"]
    cfg["quantization_config"]["ignore"] = exclude
    cfg["quantization_config"]["producer"] = {"name": "modelopt", "version": "izwi-tiny-fixture"}
    cfg["mtp"] = {"enabled": False}
    cfg["compute_mtp"] = False
    cfg.pop("ctc_timestamps", None)
    cfg["eos_token_id"] = [11]
    cfg["pad_token_id"] = 0
    return cfg, exclude


def tiny_tokenizer_json(vocab_size):
    """synthetic.rs's word-level tokenizer, with the instruction words given
    ids so the prompt is not all `<unk>`."""
    names = {0: "<unk>", 1: "<s>", 2: "</s>", 10: "<|im_start|>", 11: "<|im_end|>",
             12: "<think>", 13: "</think>"}
    for i in range(10):
        names[100 + i] = f"<spk:{i}>"
        names[110 + i] = f"<ev:e{i}>"
    for i in range(11):
        names[120 + i] = f"<sty:stt:s{i}>"
        names[131 + i] = f"<sty:end:s{i}>"
    vocab, added = {}, []
    for i in range(142):
        content = names.get(i, f"<reserved_{i}>")
        vocab[content] = i
        added.append({"id": i, "content": content, "single_word": False, "lstrip": False,
                      "rstrip": False, "normalized": False, "special": True})
    added.append({"id": vocab_size, "content": AUDIO_TOKEN, "single_word": False, "lstrip": False,
                  "rstrip": False, "normalized": False, "special": True})
    words = []
    for text in ("system user assistant", READABLE_INSTRUCTION, VERBATIM_INSTRUCTION,
                 SPEAKER_INSTRUCTION, "Please translate the given speech to German English",
                 "Transcribe with punctuation and capitalization. Context terms: , ."):
        for word in re.findall(r"\w+|[^\w\s]+", re.sub(r"<[^>]*>", " ", text)):
            if word not in words:
                words.append(word)
    next_id = 142
    for word in words:
        if next_id >= vocab_size:
            break
        vocab[word] = next_id
        next_id += 1
    for i in range(next_id, vocab_size):
        vocab[f"w{i}"] = i
    return {
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": added,
        "normalizer": None, "pre_tokenizer": {"type": "Whitespace"}, "post_processor": None,
        "decoder": None, "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"},
    }


def read_bf16_file(path, shape):
    raw = np.frombuffer(Path(path).read_bytes(), dtype="<u2").astype(np.uint32) << 16
    return torch.from_numpy(raw.view(np.float32).copy()).reshape(shape).to(torch.bfloat16)


def generate_tensors(g, census, seed):
    """Seeded values per tensor kind (synthetic.rs's distributions)."""
    rng = np.random.default_rng(seed)

    def uniform(shape, scale):
        return torch.from_numpy(rng.uniform(-scale, scale, size=shape).astype(np.float32))

    fb = read_bf16_file(FIXTURE_DIR / "featurizer_fb.bf16", (1, g.n_mels, g.n_fft // 2 + 1))
    window = read_bf16_file(FIXTURE_DIR / "featurizer_window.bf16", (g.win,))
    names = {name for name, _, _ in census}
    tensors = {}
    for name, dtype, shape in census:
        if name in tensors:
            continue
        leaf = name.rsplit(".", 1)[-1]
        module = name.rsplit(".", 1)[0]
        if leaf == "fb":
            value = fb
        elif leaf == "window":
            value = window
        elif dtype == "F8_E4M3":
            cols = shape[1]
            weight = uniform(shape, 1.0 / math.sqrt(cols))
            scale = weight.abs().max() / 448.0
            codes = (weight / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            tensors[name] = codes
            tensors[f"{module}.weight_scale"] = scale.reshape(()).to(torch.float32)
            tensors[f"{module}.input_scale"] = torch.tensor(0.02, dtype=torch.float32)
            if f"{module}.k_scale" in names:
                tensors[f"{module}.k_scale"] = torch.tensor(0.02, dtype=torch.float32)
            if f"{module}.v_scale" in names:
                tensors[f"{module}.v_scale"] = torch.tensor(0.004, dtype=torch.float32)
            continue
        elif leaf in ("weight_scale", "input_scale", "k_scale", "v_scale"):
            continue  # written with their FP8 weight
        elif leaf == "e_score_correction_bias":
            value = 40.69 + uniform(shape, 0.05)
        elif leaf == "A_log":
            value = torch.log(2.0 + uniform(shape, 1.0))
        elif leaf == "D":
            value = 1.0 + uniform(shape, 0.5)
        elif leaf == "dt_bias":
            value = uniform(shape, 1.0)
        elif name.endswith("conv1d.weight"):
            value = uniform(shape, 0.5)
        elif name.endswith("conv1d.bias"):
            value = uniform(shape, 0.1)
        elif name.endswith("gate.weight"):
            value = uniform(shape, 0.3)
        elif name == "backbone.embeddings.weight":
            value = uniform(shape, 1.0)
        elif name == "lm_head.weight":
            value = uniform(shape, 0.5)
        elif leaf == "learnable_sil_emb":
            value = uniform(shape, 0.5)
        elif "subpixel_upsample" in name:
            value = uniform(shape, 0.3)
        elif name.endswith("single_hidden_to_spks.bias"):
            value = torch.tensor(SPEAKER_BIAS[: shape[0]], dtype=torch.float32)
        elif ("norm" in module.rsplit(".", 1)[-1] or module.endswith(("norm1", "norm2")))\
                and leaf == "weight" and len(shape) == 1:
            value = 1.0 + uniform(shape, 0.2)
        elif "norm" in module.rsplit(".", 1)[-1] and leaf == "bias":
            value = uniform(shape, 0.05)
        elif leaf == "weight" and len(shape) == 2:
            value = uniform(shape, 1.0 / math.sqrt(shape[1]))
        elif leaf == "bias":
            value = uniform(shape, 0.02)
        else:
            raise RuntimeError(f"no value rule for {name} {shape}")
        tensors[name] = value.to(torch.bfloat16 if dtype == "BF16" else torch.float32)
    missing = names - set(tensors)
    if missing:
        raise RuntimeError(f"census tensors without values: {sorted(missing)[:8]}")
    return tensors


def shard_of(name, layers):
    if name.startswith("perception."):
        return "perception.safetensors"
    if name == "backbone.embeddings.weight":
        return "model-00001-of-00004.safetensors"
    match = re.match(r"backbone\.layers\.(\d+)\.", name)
    if match:
        half = (layers + 1) // 2
        return ("model-00002-of-00004.safetensors" if int(match.group(1)) < half
                else "model-00003-of-00004.safetensors")
    return "model-00004-of-00004.safetensors"


def write_checkpoint(out, cfg, exclude, tensors, g):
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, value in tensors.items():
        files.setdefault(shard_of(name, len(g.layers)), {})[name] = value.contiguous()
    weight_map, total_bytes, total_params = {}, 0, 0
    for file, group in sorted(files.items()):
        save_file(group, str(out / file))
        for name, value in group.items():
            weight_map[name] = file
            total_params += value.numel()
            total_bytes += value.numel() * value.element_size()
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_parameters": total_params, "total_size": total_bytes},
         "weight_map": dict(sorted(weight_map.items()))}, indent=2) + "\n")
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (out / "hf_quant_config.json").write_text(json.dumps({
        "producer": {"name": "modelopt", "version": "izwi-tiny-fixture"},
        "quantization": {"quant_algo": "FP8", "kv_cache_quant_algo": "FP8",
                         "exclude_modules": exclude}}, indent=2) + "\n")
    (out / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [11], "pad_token_id": 0}, indent=2) + "\n")
    (out / "tokenizer.json").write_text(json.dumps(tiny_tokenizer_json(g.vocab)) + "\n")
    (out / "tokenizer_config.json").write_text(json.dumps(
        {"audio_token": AUDIO_TOKEN, "eos_token": "<|im_end|>"}, indent=2) + "\n")


def verify_written(out, census):
    src = SafetensorsSource(out)
    expected = {name: (dtype, shape) for name, dtype, shape in census}
    found = {name: (src.dtype(name), src.shape(name)) for name in src.names()}
    if found != expected:
        missing = sorted(set(expected) - set(found))
        extra = sorted(set(found) - set(expected))
        wrong = sorted(n for n in set(found) & set(expected) if found[n] != expected[n])
        raise RuntimeError(f"written checkpoint != census: missing {missing[:5]} extra "
                           f"{extra[:5]} wrong {wrong[:5]}")
    return src


# ------------------------------------------------------------------ clips

def write_pcm16(path, samples_int16):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(samples_int16.astype("<i2").tobytes())


def make_clip(kind, seconds, seed):
    """Deterministic speech-like signals, quantized to 16-bit PCM."""
    n = int(round(seconds * SAMPLE_RATE))
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    rng = np.random.default_rng(seed)
    noise = rng.uniform(-1.0, 1.0, n) * 0.01
    if kind == "voices":
        # Two alternating "speakers" (f0 120 Hz and 230 Hz) with short pauses.
        segment = (t // 1.5).astype(np.int64)
        f0 = np.where(segment % 2 == 0, 120.0, 230.0)
        phase = 2 * np.pi * np.cumsum(f0) / SAMPLE_RATE
        voiced = (t % 1.5) < 1.3
        x = voiced * (0.35 * np.sin(phase) + 0.12 * np.sin(2 * phase) + 0.05 * np.sin(3 * phase))
    else:
        f0 = 140.0 + 25.0 * np.sin(2 * np.pi * 0.05 * t) + 20.0 * (seed % 3)
        phase = 2 * np.pi * np.cumsum(f0) / SAMPLE_RATE
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t)
        if kind == "long":
            envelope = envelope * ((t % 10.0) < 8.5)
        x = envelope * (0.3 * np.sin(phase) + 0.1 * np.sin(2 * phase))
    quantized = np.clip(np.round((x + noise) * 32768.0), -32768, 32767).astype(np.int16)
    return quantized


CLIPS = [
    ("tone_3s", "speech", 3.0, 1),
    ("voices_6p5s", "voices", 6.5, 2),
    ("edge_40s", "speech", 40.0, 3),     # Tv = 4000: a last window of one 0.0 frame
    ("long_45s", "long", 45.0, 4),       # Tv = 4500: two online windows
]


# ------------------------------------------------------------- tolerances

def suggested_tolerances(level):
    """Per-stage bounds a Rust test can start from. CPU runs Q8_0 linears (D4),
    Metal F16 activations, CUDA BF16 activations; tighten after the first
    green run, and RED-check each stage (break the mechanism, expect a miss by
    10x or more)."""
    deep = {"cpu": 5e-2, "metal": 3e-2, "cuda": 3e-2} if level == "L2" else \
        {"cpu": 3e-2, "metal": 2e-2, "cuda": 2e-2}
    return {
        "metrics": "rel_l2 = ||izwi - ref|| / ||ref|| per row; max_abs in value units; "
                   "bf16_ulps compares both sides rounded to BF16",
        "mel_raw_rows": {"max_bf16_ulps": 1, "max_fraction_off_by_one_ulp": 0.02},
        "mel_mean": {"max_bf16_ulps": 1},
        "mel_std": {"rel": 2e-3},
        "normalized_rows": {"max_bf16_ulps": 2, "max_fraction_mismatched": 0.02},
        "asr_core_rows": {"rel_l2": {"cpu": 2e-2, "metal": 1e-2, "cuda": 1.5e-2}},
        "diar_probs_80ms": {
            "before_first_compression": {"max_abs": {"cpu": 2e-3, "metal": 2e-3, "cuda": 5e-3}},
            "after_first_compression": {"binary_agreement_min": 0.995},
            "note": "windows with follows_compression run after a speaker-cache compression, "
                    "whose top-k ties may reorder the cache",
        },
        "fused_rows": {"rel_l2": {"cpu": 2e-2, "metal": 1e-2, "cuda": 1.5e-2},
                       "note": "a flipped speaker bit (p within 1e-3 of 0.5) moves a row by a "
                               "whole kernel row; skip rows whose activity disagrees"},
        "embeddings_rows": {"rel_l2": {"cpu": 2e-2, "metal": 1e-2, "cuda": 1.5e-2}},
        "layer_hidden": {"rel_l2": deep, "checksum_rel": deep},
        "routes": {"expert_set_equal_when_margin_ge": NEAR_TIE,
                   "near_tie_rows_may_differ": True, "weights_max_abs": 1e-3},
        "prefill_logits_top8": {"max_abs": {"cpu": 0.1, "metal": 0.05, "cuda": 0.05},
                                "top1_equal_when_margin_ge": 0.05},
        "greedy_ids": {"exact_prefix_until_margin_below": 0.02,
                       "note": "after a step whose greedy_margins entry is below the bound, "
                               "the continuation may legitimately diverge"},
        "text": {"exact": True, "leg_b": "<= 1 word difference per clip (plan §10.5)"},
        "decode_vs_prefill": {"note": "vLLM rounds three SSD dot operands to BF16 in prefill "
                                      "only; compare prefill and decode with separate bounds"},
    }


# -------------------------------------------------------------------- main

def build_audio_reference(impl, cfg, src, device):
    """(name, module, runner, fallback_reason)."""
    reason = None
    if impl in ("auto", "nemo"):
        try:
            perception = build_nemo_perception(cfg, device)
            rounded = load_perception(perception, src, device)
            log(f"NeMo perception built; BF16-rounded tables: {len(rounded)}")
            return "nemo", perception, run_nemo_perception, None
        except Exception as error:  # noqa: BLE001
            if impl == "nemo":
                raise
            reason = f"NeMo perception unavailable: {type(error).__name__}: {error}"
            log(reason)
    perception = build_sglang_perception(cfg, device)
    load_perception(perception, src, device)
    return "sglang-omni", perception, run_sglang_perception_impl, reason


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"fixture directory (default {DEFAULT_OUT.relative_to(ROOT)})")
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--audio-impl", choices=("auto", "nemo", "sglang"), default="auto")
    parser.add_argument("--no-edge-clip", action="store_true",
                        help="skip the 40 s clip whose last window is one 0.0 frame (saves 1.3 MB)")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--no-strict", action="store_true",
                        help="write the fixture even when a self-check fails")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    disable_tf32()
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    from tokenizers import Tokenizer

    cfg, exclude = tiny_config()
    g = Geometry(decode_nemo_floats(cfg))
    census = expected_census(g, exclude)
    tensors = generate_tensors(g, census, args.seed)
    write_checkpoint(args.out, cfg, exclude, tensors, g)
    src = verify_written(args.out, census)
    log(f"wrote {len(census)} tensors in the published layout to {args.out}")

    tokenizer = Tokenizer.from_file(str(args.out / "tokenizer.json"))
    if tokenizer.token_to_id(AUDIO_TOKEN) != g.vocab:
        raise RuntimeError("the tiny tokenizer does not put <|audio|> one past the table")

    audio_reference, perception, run_perception, reason = build_audio_reference(
        args.audio_impl, cfg, src, device)
    decoder = build_decoder(cfg, src, device, "stacked")
    experts_impl = "transformers.NemotronHExperts"
    try:
        moe_check = moe_f64_check(decoder)
    except Exception as error:  # noqa: BLE001
        log(f"upstream NemotronHExperts failed ({error}); using the restatement")
        from transformers.activations import ACT2FN

        for index, block in enumerate(decoder.blocks):
            if block.block_type == "moe":
                stacked = block.mixer.experts

                def fetch(e, stacked=stacked):
                    return stacked.up_proj[e], stacked.down_proj[e]

                block.mixer.experts = RestatedExperts(g.experts, fetch,
                                                      ACT2FN[decoder.config.mlp_hidden_act])
        experts_impl = "restated"
        moe_check = moe_f64_check(decoder)

    trace_layers = [0, 1, g.first("attention")]
    route_layer = g.first("moe")
    clips, failures = [], []
    global_checks = {
        "mamba_time_step_limits": sorted({str(tuple(b.mixer.time_step_limit))
                                          for b in decoder.blocks
                                          if b.block_type == "linear_attention"}),
        "moe_f64_rel_l2": moe_check,
    }
    if global_checks["mamba_time_step_limits"] != [str((0.0, float("inf")))]:
        failures.append("Mamba dt clamp was not patched")
    if not moe_check < 1e-5:
        failures.append(f"MoE restatement mismatch {moe_check:.3e}")

    for name, kind, seconds, seed in CLIPS:
        if args.no_edge_clip and name == "edge_40s":
            continue
        pcm = make_clip(kind, seconds, args.seed + seed)
        wav = args.out / "audio" / f"{name}.wav"
        write_pcm16(wav, pcm)
        samples = pcm.astype(np.float32) / np.float32(32768.0)
        log(f"{name}: {len(samples)} samples")
        try:
            result = run_perception(perception, samples, device)
        except Exception as error:  # noqa: BLE001
            # FlexAttention fails at the first forward, not at build time.
            if audio_reference != "nemo" or args.audio_impl != "auto" or clips:
                raise
            reason = f"NeMo perception forward failed: {type(error).__name__}: {error}"
            log(reason + "; falling back to SGLang-Omni")
            perception = build_sglang_perception(cfg, device)
            load_perception(perception, src, device)
            audio_reference, run_perception = "sglang-omni", run_sglang_perception_impl
            result = run_perception(perception, samples, device)
        record = {
            "name": name,
            "audio_file": f"audio/{name}.wav",
            "audio_format": "wav pcm_s16le mono 16000 Hz; sample = int16 / 32768.0",
            "samples": int(len(samples)),
            "duration_secs": len(samples) / SAMPLE_RATE,
            "sha256": sha256_file(wav),
        }
        record.update(perception_record(result, g))
        case, checks, (embeds, _) = run_decoder_case(
            decoder, tokenizer, READABLE_INSTRUCTION, result.embeddings, args.max_new_tokens,
            args.max_new_tokens, trace_layers, route_layer, positions_cap=24,
            stream_text=False)
        checks["mamba_f64_rel_l2"] = mamba_f64_check(decoder, embeds)
        for key, value in checks.items():
            if key.endswith("_max_abs"):
                scale = checks[key.replace("_max_abs", "_scale")]
                if not value <= 1e-4 * max(scale, 1.0):
                    failures.append(f"{name}: {key} = {value:.3e}")
        if not checks["mamba_f64_rel_l2"] < 1e-5:
            failures.append(f"{name}: Mamba F64 mismatch {checks['mamba_f64_rel_l2']:.3e}")
        case.pop("instruction")
        record.update(case)
        record["self_checks"] = checks
        clips.append(record)
        log(f"{name}: {case['prompt_tokens']} prompt tokens, greedy {case['greedy_ids'][:8]}... "
            f"({case['finish_reason']})")

    files = {}
    for path in sorted(args.out.iterdir()):
        if path.is_file() and path.name != "expected.json":
            files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    fixture = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "generator": "scripts/fixtures/generate_nemotron_transcribe_tiny.py",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "versions": collect_versions(),
        "audio_reference": audio_reference,
        "audio_fallback_reason": reason,
        "decoder_reference": "transformers.NemotronH",
        "experts_impl": experts_impl,
        "reference_semantics": {
            "weights": "W8A16: E4M3 code * F32 per-tensor weight_scale, multiplied in F32; "
                       "BF16 tensors upcast exactly; input/k/v scales unused",
            "compute": "float32 throughout, TF32 off, eager attention",
            "bf16_rounding_points": ["log-mel output", "per-feature normalization (torch BF16 "
                                     "semantics: BF16 mean and squared deviations, F32 std)",
                                     "normalized ASR-encoder input", "RoPE cos/sin tables",
                                     "speaker kernel"],
            "mamba": "transformers torch path; time_step_limit (0, inf) (prefill clamp removed); "
                     "dt, y and the SSM state F32",
            "perception": "NeMo online windows over T = Tv + 1 frames: core 500, context 50 "
                          "encoder frames; streaming Sortformer chunk 500, FIFO 40, update 300, "
                          "cache 188; the last window does not update the state",
            "decode": "greedy (ties to the lowest id); stops on 11, the token cap or izwi's "
                      "online loop detector (20 repeats)",
        },
        "checkpoint": {"files": files, "tensors": len(census), "geometry": g.as_dict()},
        "tokenizer": {"audio_token": AUDIO_TOKEN, "audio_token_id": g.vocab,
                      "eos_token_ids": decoder.eos},
        "instruction": READABLE_INSTRUCTION,
        "prompt_text": render_prompt_text(READABLE_INSTRUCTION),
        "max_new_tokens": args.max_new_tokens,
        "clips": clips,
        "suggested_tolerances": suggested_tolerances("L1"),
        "self_checks": dict(global_checks, failures=failures),
    }
    dump_json(fixture, args.out / "expected.json")
    size = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
    log(f"wrote {args.out / 'expected.json'}; fixture directory is {size / 1e6:.2f} MB")
    if failures:
        log("SELF-CHECK FAILURES:\n  " + "\n  ".join(failures))
        if not args.no_strict:
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
