#!/usr/bin/env python3
"""Generate the Qwen3.8 HF golden-parity fixture.

Builds a tiny random-init `Qwen3_5ForCausalLM` (transformers' reference
implementation of the dense Qwen3.5/3.8 text architecture), quantizes it into
the published `Qwen/Qwen3.8-27B-FP8` checkpoint layout (block-FP8 projections
with BF16 `weight_scale_inv`, BF16 dense tensors, text tensors under
`model.language_model.*`, an untied BF16 `lm_head`, and the 22-tensor `mtp.*`
head the native loader validates), and records the reference logits for every
position of a fixed token sequence. The HF model runs on exactly the values
the checkpoint stores (dequantized FP8, BF16-rounded dense), so izwi's native
loader + Qwen3.8 trunk must reproduce the logits to F32 tolerance.

The geometry keeps the structurally important ratios of Qwen3.8-27B: 3 value
heads per DeltaNet key head (16/48 there, 2/6 here, so the grouped
value-head order matters), 6 query heads per KV head with two KV heads (so the
GQA mapping matters), the output-gated attention, partial rotary (0.25) with
interleaved M-RoPE sections scaled to the rotary width, theta 1e7, a full
attention layer every fourth layer (two of them here), untied embeddings, and
random non-zero zero-centered norm gains, so loader value-convention errors
(the `1 + w` gain, the value-head order, `A_log`, conv orientation, FP8 block
scales) change the logits instead of hiding behind symmetry.

The MTP head is random filler: HF has no Qwen3.5 MTP implementation, and the
Rust test loads the checkpoint with MTP disabled. It exists because the
native loader validates the exact published MTP manifest.

Before writing anything, the script checks the reference against itself: HF's
cached decode (prompt prefill, then one token at a time through the DeltaNet
recurrence and the KV cache) must reproduce the full-sequence forward (the
chunked DeltaNet path) at every decoded position.

Usage (needs torch and transformers >= 5.0; run offline and isolated):

    HF_HUB_OFFLINE=1 python -I scripts/fixtures/generate_qwen38_golden.py \
        crates/izwi-core/tests/fixtures/qwen38_golden

The Rust test is `qwen38::text::golden_tests::native_trunk_matches_the_hf_reference_logits`.
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from safetensors.torch import save_file
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

SEED = 20261011
BLOCK = 16
FP8_MAX = 448.0
VOCAB = 128
PROMPT_TOKENS = 12
TOTAL_TOKENS = 32
# HF's chunked (prefill) and recurrent (decode) DeltaNet paths agree to F32
# rounding; anything larger means the reference itself is inconsistent.
SELF_CHECK_TOLERANCE = 1e-4

LAYERS = 8
FULL_ATTENTION_INTERVAL = 4
LAYER_TYPES = [
    "full_attention" if (index + 1) % FULL_ATTENTION_INTERVAL == 0 else "linear_attention"
    for index in range(LAYERS)
]

TEXT_CONFIG = {
    "vocab_size": VOCAB,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": LAYERS,
    "num_attention_heads": 12,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 6,
    "linear_key_head_dim": 32,
    "linear_value_head_dim": 32,
    "linear_conv_kernel_dim": 4,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
    "partial_rotary_factor": 0.25,
    "layer_types": LAYER_TYPES,
    "rope_parameters": {
        "rope_type": "default",
        "mrope_interleaved": True,
        # [11, 11, 10] over 32 rotary pairs, scaled to 4 pairs (head_dim 32 * 0.25 / 2).
        "mrope_section": [2, 1, 1],
        "rope_theta": 10000000.0,
        "partial_rotary_factor": 0.25,
    },
}

# Projections the published checkpoint stores as 128x128 (here 16x16) block FP8.
FP8_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.out_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def quantize_block_fp8(weight: torch.Tensor):
    """Return (e4m3 weight, BF16 weight_scale_inv, dequantized F32 weight)."""
    rows, cols = weight.shape
    assert rows % BLOCK == 0 and cols % BLOCK == 0, weight.shape
    blocks = weight.reshape(rows // BLOCK, BLOCK, cols // BLOCK, BLOCK)
    amax = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-8)
    scale = (amax / FP8_MAX).to(torch.bfloat16)
    scale_f32 = scale.float()[:, None, :, None]
    quantized = (blocks / scale_f32).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    dequantized = (quantized.float() * scale_f32).reshape(rows, cols)
    return quantized.reshape(rows, cols), scale, dequantized


def random_value(name: str, shape, generator: torch.Generator) -> torch.Tensor:
    """A distinctive non-trivial value for a parameter, by naming convention."""
    if name.endswith("A_log"):
        return torch.rand(shape, generator=generator) * 2.0 - 1.0
    if name.endswith("dt_bias"):
        return torch.randn(shape, generator=generator) * 0.5
    if name.endswith("linear_attn.norm.weight"):
        # Gated DeltaNet norm: plain gain, ones-initialized upstream.
        return 1.0 + torch.randn(shape, generator=generator) * 0.3
    if name.endswith("norm.weight") or name.endswith("layernorm.weight"):
        # Zero-centered gains (applied as 1 + w).
        return torch.randn(shape, generator=generator) * 0.3
    if name.endswith("conv1d.weight"):
        return torch.randn(shape, generator=generator) * 0.4
    if name.endswith("embed_tokens.weight"):
        return torch.randn(shape, generator=generator)
    return torch.randn(shape, generator=generator) / shape[-1] ** 0.5


def randomize_(model: Qwen3_5ForCausalLM, generator: torch.Generator) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(random_value(name, param.shape, generator))


def checkpoint_tensors(model: Qwen3_5ForCausalLM):
    """Map HF parameters onto the published checkpoint layout.

    Returns (tensors to save, F32 values the HF model must run on).
    """
    saved = {}
    effective = {}

    def store(published: str, value: torch.Tensor) -> torch.Tensor:
        if any(published.endswith(suffix) for suffix in FP8_SUFFIXES):
            quantized, scale, dequantized = quantize_block_fp8(value.float())
            saved[published] = quantized.contiguous()
            saved[published.removesuffix(".weight") + ".weight_scale_inv"] = scale.contiguous()
            return dequantized
        stored = value.to(torch.bfloat16).contiguous()
        saved[published] = stored
        return stored.float()

    for hf_name, param in model.state_dict().items():
        published = hf_name
        if published.startswith("model."):
            published = "model.language_model." + published.removeprefix("model.")
        effective[hf_name] = store(published, param)
    return saved, effective, store


def add_mtp_head(store, generator: torch.Generator) -> None:
    """Random tensors for the published one-layer MTP manifest (unused by HF)."""
    hidden = TEXT_CONFIG["hidden_size"]
    intermediate = TEXT_CONFIG["intermediate_size"]
    head_dim = TEXT_CONFIG["head_dim"]
    query = TEXT_CONFIG["num_attention_heads"] * head_dim
    kv = TEXT_CONFIG["num_key_value_heads"] * head_dim
    shapes = {
        "mtp.fc.weight": (hidden, 2 * hidden),
        "mtp.norm.weight": (hidden,),
        "mtp.pre_fc_norm_embedding.weight": (hidden,),
        "mtp.pre_fc_norm_hidden.weight": (hidden,),
        "mtp.layers.0.input_layernorm.weight": (hidden,),
        "mtp.layers.0.post_attention_layernorm.weight": (hidden,),
        "mtp.layers.0.mlp.gate_proj.weight": (intermediate, hidden),
        "mtp.layers.0.mlp.up_proj.weight": (intermediate, hidden),
        "mtp.layers.0.mlp.down_proj.weight": (hidden, intermediate),
        "mtp.layers.0.self_attn.q_norm.weight": (head_dim,),
        "mtp.layers.0.self_attn.k_norm.weight": (head_dim,),
        "mtp.layers.0.self_attn.q_proj.weight": (2 * query, hidden),
        "mtp.layers.0.self_attn.k_proj.weight": (kv, hidden),
        "mtp.layers.0.self_attn.v_proj.weight": (kv, hidden),
        "mtp.layers.0.self_attn.o_proj.weight": (hidden, query),
    }
    for name, shape in shapes.items():
        store(name, random_value(name, shape, generator))


def cached_decode_logits(model, token_ids: torch.Tensor) -> torch.Tensor:
    """Prefill the prompt, then decode one token at a time through the cache.

    Returns one logits row per position from `PROMPT_TOKENS - 1` onward.
    """
    with torch.no_grad():
        output = model(input_ids=token_ids[:, :PROMPT_TOKENS], use_cache=True)
        rows = [output.logits[0, -1].float()]
        cache = output.past_key_values
        for position in range(PROMPT_TOKENS, TOTAL_TOKENS):
            output = model(
                input_ids=token_ids[:, position : position + 1],
                past_key_values=cache,
                use_cache=True,
            )
            cache = output.past_key_values
            rows.append(output.logits[0, -1].float())
    return torch.stack(rows)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator().manual_seed(SEED)
    torch.manual_seed(SEED)
    config = Qwen3_5TextConfig(**TEXT_CONFIG)
    config._attn_implementation = "eager"
    model = Qwen3_5ForCausalLM(config).float().eval()
    randomize_(model, generator)

    saved, effective, store = checkpoint_tensors(model)
    model.load_state_dict(effective, strict=True)
    add_mtp_head(store, generator)

    token_ids = torch.randint(0, VOCAB, (1, TOTAL_TOKENS), generator=generator)
    with torch.no_grad():
        logits = model(input_ids=token_ids, use_cache=False).logits[0].float()

    decoded = cached_decode_logits(model, token_ids)
    self_check = (decoded - logits[PROMPT_TOKENS - 1 :]).abs().max().item()
    if not self_check < SELF_CHECK_TOLERANCE:
        print(f"HF cached decode diverges from the full forward by {self_check}")
        return 1

    save_file(saved, out / "model.safetensors")
    (out / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {}, "weight_map": {name: "model.safetensors" for name in sorted(saved)}},
            indent=1,
        )
        + "\n"
    )
    rope = TEXT_CONFIG["rope_parameters"]
    text_config = {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attn_output_gate": True,
        "bos_token_id": 1,
        "dtype": "bfloat16",
        "eos_token_id": 2,
        "full_attention_interval": FULL_ATTENTION_INTERVAL,
        "head_dim": TEXT_CONFIG["head_dim"],
        "hidden_act": "silu",
        "hidden_size": TEXT_CONFIG["hidden_size"],
        "intermediate_size": TEXT_CONFIG["intermediate_size"],
        "layer_types": LAYER_TYPES,
        "linear_conv_kernel_dim": TEXT_CONFIG["linear_conv_kernel_dim"],
        "linear_key_head_dim": TEXT_CONFIG["linear_key_head_dim"],
        "linear_num_key_heads": TEXT_CONFIG["linear_num_key_heads"],
        "linear_num_value_heads": TEXT_CONFIG["linear_num_value_heads"],
        "linear_value_head_dim": TEXT_CONFIG["linear_value_head_dim"],
        "mamba_ssm_dtype": "float32",
        "max_position_embeddings": TEXT_CONFIG["max_position_embeddings"],
        "model_type": "qwen3_5_text",
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
        "num_attention_heads": TEXT_CONFIG["num_attention_heads"],
        "num_hidden_layers": LAYERS,
        "num_key_value_heads": TEXT_CONFIG["num_key_value_heads"],
        "output_gate_type": "swish",
        "partial_rotary_factor": TEXT_CONFIG["partial_rotary_factor"],
        "rms_norm_eps": TEXT_CONFIG["rms_norm_eps"],
        "rope_parameters": rope,
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": VOCAB,
    }
    (out / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "language_model_only": False,
                "model_type": "qwen3_5",
                "tie_word_embeddings": False,
                "text_config": text_config,
                "quantization_config": {
                    "activation_scheme": "dynamic",
                    "fmt": "e4m3",
                    "quant_method": "fp8",
                    "weight_block_size": [BLOCK, BLOCK],
                },
            },
            indent=1,
        )
        + "\n"
    )
    save_file(
        {
            "token_ids": token_ids[0].to(torch.int64).contiguous(),
            "logits": logits.contiguous(),
            "prompt_tokens": torch.tensor([PROMPT_TOKENS], dtype=torch.int64),
        },
        out / "golden.safetensors",
    )
    print(
        f"wrote {len(saved)} tensors and {tuple(logits.shape)} reference logits to {out}; "
        f"logit scale {logits.abs().max().item():.3f}, "
        f"HF cached decode vs full forward max |diff| {self_check:.2e}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
