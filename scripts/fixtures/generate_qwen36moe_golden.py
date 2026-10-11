#!/usr/bin/env python3
"""Generate the Qwen3.6-MoE HF golden-parity fixture.

Builds a tiny random-init `Qwen3_5MoeForCausalLM` (transformers' reference
implementation of the Qwen3.5/3.6-MoE architecture), quantizes it into the
published `Qwen/Qwen3.6-35B-A3B-FP8` checkpoint layout (block-FP8 projections
with BF16 `weight_scale_inv`, BF16 dense tensors, per-expert tensor names
under `model.language_model.*`), and records the reference logits for every
position of a fixed token sequence. The HF model runs on exactly the values
the checkpoint stores (dequantized FP8, BF16-rounded dense), so izwi's native
loader + shared trunk must reproduce the logits to F32 tolerance.

Unlike the constant-valued native fixture, every tensor is random and every
zero-centered norm gain is non-zero, and the DeltaNet uses fewer key heads
than value heads, so loader value-convention errors (the zero-centered
`1 + w`, the value-head order) change the logits instead of hiding behind
symmetry.

Usage (needs torch and transformers >= 5.0):

    python scripts/fixtures/generate_qwen36moe_golden.py \
        crates/izwi-core/tests/fixtures/qwen36moe_golden

The Rust test is `qwen36moe::native::tests::native_trunk_matches_the_hf_reference_logits`.
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from safetensors.torch import save_file
from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

SEED = 20261010
BLOCK = 16
FP8_MAX = 448.0
VOCAB = 128
PROMPT_TOKENS = 12
TOTAL_TOKENS = 32

TEXT_CONFIG = {
    "vocab_size": VOCAB,
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "linear_key_head_dim": 32,
    "linear_value_head_dim": 32,
    "linear_conv_kernel_dim": 4,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 32,
    "shared_expert_intermediate_size": 32,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
    "layer_types": [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ],
    "rope_parameters": {
        "rope_type": "default",
        "mrope_interleaved": True,
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
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
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


def randomize_(model: Qwen3_5MoeForCausalLM, generator: torch.Generator) -> None:
    """Give every parameter a distinctive non-trivial value."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("A_log"):
                values = torch.rand(param.shape, generator=generator) * 2.0 - 1.0
            elif name.endswith("dt_bias"):
                values = torch.randn(param.shape, generator=generator) * 0.5
            elif name.endswith("linear_attn.norm.weight"):
                # Gated norm: plain gain, ones-initialized upstream.
                values = 1.0 + torch.randn(param.shape, generator=generator) * 0.3
            elif name.endswith("norm.weight"):
                # Zero-centered gains (applied as 1 + w).
                values = torch.randn(param.shape, generator=generator) * 0.3
            elif name.endswith("conv1d.weight"):
                values = torch.randn(param.shape, generator=generator) * 0.4
            elif name.endswith("embed_tokens.weight"):
                values = torch.randn(param.shape, generator=generator)
            else:
                fan_in = param.shape[-1]
                values = torch.randn(param.shape, generator=generator) / fan_in**0.5
            param.copy_(values)


def checkpoint_tensors(model: Qwen3_5MoeForCausalLM):
    """Map HF parameters onto the published checkpoint layout.

    Returns (tensors to save, F32 values the HF model must run on).
    """
    saved = {}
    effective = {}
    intermediate = TEXT_CONFIG["moe_intermediate_size"]

    def store(published: str, hf_name: str, value: torch.Tensor) -> torch.Tensor:
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
        if hf_name.endswith("mlp.experts.gate_up_proj"):
            stem = published.removesuffix("gate_up_proj")
            parts = []
            for expert in range(param.shape[0]):
                gate = store(f"{stem}{expert}.gate_proj.weight", hf_name, param[expert, :intermediate])
                up = store(f"{stem}{expert}.up_proj.weight", hf_name, param[expert, intermediate:])
                parts.append(torch.cat([gate, up], dim=0))
            effective[hf_name] = torch.stack(parts)
        elif hf_name.endswith("mlp.experts.down_proj"):
            stem = published.removesuffix("down_proj")
            effective[hf_name] = torch.stack(
                [
                    store(f"{stem}{expert}.down_proj.weight", hf_name, param[expert])
                    for expert in range(param.shape[0])
                ]
            )
        else:
            effective[hf_name] = store(published, hf_name, param)
    return saved, effective


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator().manual_seed(SEED)
    torch.manual_seed(SEED)
    config = Qwen3_5MoeTextConfig(**TEXT_CONFIG)
    model = Qwen3_5MoeForCausalLM(config).float().eval()
    randomize_(model, generator)

    saved, effective = checkpoint_tensors(model)
    model.load_state_dict(effective, strict=True)

    token_ids = torch.randint(0, VOCAB, (1, TOTAL_TOKENS), generator=generator)
    with torch.no_grad():
        logits = model(input_ids=token_ids, use_cache=False).logits[0].float()

    save_file(saved, out / "model.safetensors")
    (out / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {}, "weight_map": {name: "model.safetensors" for name in sorted(saved)}},
            indent=1,
        )
        + "\n"
    )
    text_config = dict(TEXT_CONFIG)
    text_config["full_attention_interval"] = 4
    text_config["mamba_ssm_dtype"] = "float32"
    (out / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "model_type": "qwen3_5_moe",
                "tie_word_embeddings": False,
                "text_config": text_config,
                "quantization_config": {
                    "quant_method": "fp8",
                    "fmt": "e4m3",
                    "activation_scheme": "dynamic",
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
    print(f"wrote {len(saved)} tensors and {tuple(logits.shape)} reference logits to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
