#!/usr/bin/env python3
"""Generate the Inkling 2 golden-parity fixture (text trunk).

Builds a tiny random Inkling 2 "loop" model, writes it in the **published**
`thinkingmachines/Inkling-2-Mini` checkpoint layout (original tensor names,
BF16 dense tensors, NVFP4 experts on the early layers and MXFP8 experts on
the last ones, padded unembedding, split message queries), and records the
reference activations and logits of a fixed token sequence.

The forward pass below is a plain-PyTorch transcription of transformers'
`models/inkling2_loop/modeling_inkling2_loop.py` (`Inkling2LoopTextModel` and
`Inkling2LoopForCausalLM`), run in F32 on exactly the values the checkpoint
stores (dequantized experts, BF16-rounded dense tensors). It needs torch only,
because the transformers release that carries `inkling2_loop` is not installed
on the machine this was written on. `--check-transformers` loads the written
checkpoint with transformers itself and compares every layer application and
the logits; it passed against transformers 5.24.1 (F64 logits within 1e-12).

The tiny geometry covers every layer class: sliding and full attention, inside
and outside the loop, both expert formats, a window shorter than the sequence,
and thresholds low enough that the L2 clips, the log scale and the softcaps all
act. The loop spans two depth-residual blocks, so a loop pass leaves a block
state behind and the message read has two states to weigh: with one block the
read would return the same message whatever its query.

Usage:

    python3 scripts/fixtures/generate_inkling2_golden.py \
        crates/izwi-core/tests/fixtures/inkling2_golden

The Rust test is `inkling2::tests::composed_trunk_matches_the_reference_logits`.
"""

import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SEED = 20261126
TOTAL_TOKENS = 40
PROMPT_TOKENS = 14

TEXT_CONFIG = {
    "model_type": "inkling2_loop_text",
    "model_max_length": 4096,
    "torch_dtype": "bfloat16",
    "hidden_size": 64,
    "num_hidden_layers": 8,
    "vocab_size": 160,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "d_rel": 4,
    "rel_extent": 16,
    "q_bias": False,
    "o_bias": False,
    "log_scaling_n_floor": 8,
    "log_scaling_alpha": 0.1,
    "rms_norm_eps": 1e-06,
    "use_value_norm_no_gamma": True,
    "qk_l2_clip_threshold": 2.0,
    "rel_l2_clip_threshold": 1.0,
    "local_layer_ids": [0, 1, 3, 4, 6],
    # transformers does not read `local_layer_ids`: it takes `layer_types`, and
    # without one makes every sixth layer a full one, which is the published
    # layout but not this one.
    "layer_types": [
        "hybrid_sliding" if layer in (0, 1, 3, 4, 6) else "hybrid" for layer in range(8)
    ],
    "sconv_kernel_size": 4,
    "unpadded_vocab_size": 150,
    "final_logit_softcapping": 30.0,
    "swa_head_dim": 8,
    "swa_num_attention_heads": 4,
    "swa_num_key_value_heads": 2,
    "sliding_window_size": 8,
    "n_layers_per_attn_res_block": 2,
    "attn_res_normalize_values": True,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "n_shared_experts": 0,
    "intermediate_size": 32,
    "mlp_gate_softcap": 4.0,
    "mlp_up_softcap": 25.0,
    "route_scale": 8.0,
    "use_gate_bias": True,
    "loop_config": {"core_start": 2, "core_end": 6, "num_passes": 3, "gate_rank": 8, "is_varloop": False},
}
# A tiny vision tower: 2-pixel patches merged 2 x 2 (one image token per 4 x 4
# pixels), a 4 x 4 position grid, two layers.
VISION_CONFIG = {
    "model_type": "inkling2_vision",
    "text_hidden_size": 64,
    "temporal_patch_size": 2,
    "native_patch_size": 2,
    "adapter_merge_size": 2,
    "projection_hidden_dim": 48,
    "projector_bias": False,
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "mlp_ratio": 2.0,
    "posemb_grid_size": 4,
    "layer_norm_eps": 1e-05,
}
VISION_ROPE_THETA = 10000.0  # transformers' default; the published config does not set it
IMAGE_TOKEN_ID = 140
# (height, width, channels) of the fixture's images: one with transparency whose
# sides are not multiples of the cell, one opaque single cell.
IMAGES = [(7, 10, 4), (4, 4, 3)]
CONFIG = {
    "dtype": "bfloat16",
    "architectures": ["Inkling2LoopForConditionalGeneration"],
    "model_type": "inkling2_loop",
    "eos_token_id": 149,
    "image_token_id": IMAGE_TOKEN_ID,
    "text_config": TEXT_CONFIG,
    "vision_config": VISION_CONFIG,
    "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"},
}
# Layers whose experts are stored as MXFP8; the rest are NVFP4 (as published:
# the last layers carry no `_amax` tensors).
MXFP8_LAYERS = {6, 7}
NVFP4_DENOMINATOR = 2688.0  # 6 * 448, the value in the published checkpoint

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def e4m3_table() -> torch.Tensor:
    table = torch.zeros(256)
    for byte in range(256):
        sign = -1.0 if byte & 0x80 else 1.0
        exponent, mantissa = (byte >> 3) & 0xF, byte & 0x7
        if exponent == 0xF and mantissa == 0x7:
            value = float("nan")
        elif exponent == 0:
            value = mantissa / 8.0 * 2.0**-6
        else:
            value = (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
        table[byte] = sign * value
    return table


E4M3 = e4m3_table()
_E4M3_POSITIVE = E4M3[:127]  # ascending, 0 .. 448


def e4m3_encode(values: torch.Tensor) -> torch.Tensor:
    """Nearest FP8 E4M3 byte of each value (saturating at +-448)."""
    magnitude = values.abs().clamp(max=448.0)
    upper = torch.searchsorted(_E4M3_POSITIVE, magnitude.contiguous()).clamp(max=126)
    lower = (upper - 1).clamp(min=0)
    pick_lower = (magnitude - _E4M3_POSITIVE[lower]).abs() <= (_E4M3_POSITIVE[upper] - magnitude).abs()
    code = torch.where(pick_lower, lower, upper)
    return (code + 128 * (values < 0)).to(torch.uint8)


def bf16(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(torch.bfloat16).to(torch.float32)


# --- quantizers (the inverse of transformers' Inkling2LoopDequantizeExperts) ---


def nvfp4_quantize(weight: torch.Tensor):
    """`[experts, rows, columns]` F32 -> packed codes, E4M3 block scales, amax, denominator."""
    experts, rows, columns = weight.shape
    amax = weight.abs().amax(dim=(1, 2))
    factor = (amax / NVFP4_DENOMINATOR)[:, None, None, None]
    blocks = weight.view(experts, rows, columns // 16, 16)
    block_scale = blocks.abs().amax(dim=-1, keepdim=True) / 6.0 / factor
    scale_bytes = e4m3_encode(block_scale)
    scale = E4M3[scale_bytes.long()]
    target = blocks / (scale * factor).clamp(min=1e-30)
    magnitudes = E2M1[:8]
    code = (target.abs().unsqueeze(-1) - magnitudes).abs().argmin(dim=-1)
    code = code + 8 * (target < 0)
    packed = (code[..., 0::2] | (code[..., 1::2] << 4)).to(torch.uint8)
    return (
        packed.reshape(experts, rows, columns // 2),
        scale_bytes.reshape(experts, rows, columns // 16),
        amax,
        torch.tensor([NVFP4_DENOMINATOR]),
    )


def nvfp4_dequantize(packed, scale_bytes, amax, denominator) -> torch.Tensor:
    experts, rows, half = packed.shape
    codes = packed.long()
    values = torch.stack((E2M1[codes & 0xF], E2M1[codes >> 4]), dim=-1).view(experts, rows, half // 8, 16)
    values = values * E4M3[scale_bytes.long()][..., None]
    values = values * (amax.float() / denominator.float())[:, None, None, None]
    return values.reshape(experts, rows, half * 2)


def mxfp8_quantize(weight: torch.Tensor):
    """`[experts, rows, columns]` F32 -> E4M3 value bytes and one power-of-two scale byte per 32 values."""
    experts, rows, columns = weight.shape
    blocks = weight.view(experts, rows, columns // 32, 32)
    block_max = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    exponent = torch.ceil(torch.log2(block_max / 448.0)).clamp(-127, 127)
    scale_bytes = (exponent + 127).to(torch.uint8)
    value_bytes = e4m3_encode(blocks / torch.exp2(exponent))
    return value_bytes.reshape(experts, rows, columns), scale_bytes.reshape(experts, rows, columns // 32)


def mxfp8_dequantize(value_bytes, scale_bytes) -> torch.Tensor:
    experts, rows, columns = value_bytes.shape
    values = E4M3[value_bytes.long()].view(experts, rows, columns // 32, 32)
    values = values * torch.exp2(scale_bytes.float() - 127.0)[..., None]
    return values.reshape(experts, rows, columns)


# --- reference forward (transcribed from modeling_inkling2_loop.py) ---


def unweighted_rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def l2_clip(x: torch.Tensor, threshold: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return x * (threshold / (norm + 1e-6)).clamp(max=1.0)


def softcap(x: torch.Tensor, cap: float) -> torch.Tensor:
    return torch.tanh(x / cap) * cap


class Reference:
    def __init__(self, config: dict, weights: dict[str, torch.Tensor]):
        self.c = config
        self.w = weights
        self.hidden = config["hidden_size"]
        self.layers = config["num_hidden_layers"]
        self.local = set(config["local_layer_ids"])
        self.block = config["n_layers_per_attn_res_block"]
        loop = config["loop_config"]
        self.loop_start, self.loop_end, self.passes = loop["core_start"], loop["core_end"], loop["num_passes"]
        self.eps = config["rms_norm_eps"]
        self.trace: dict[str, torch.Tensor] = {}

    def p(self, name: str) -> torch.Tensor:
        return self.w[name]

    def log_scale(self, positions: torch.Tensor) -> torch.Tensor:
        ratio = ((positions + 1).float() / self.c["log_scaling_n_floor"]).clamp(min=1.0)
        return 1.0 + self.c["log_scaling_alpha"] * torch.log(ratio)

    def depth_attention(self, query: torch.Tensor, block_states: list[torch.Tensor], partial: torch.Tensor):
        """`Inkling2LoopBlockAttentionResidual`: learned queries attend over the RMS-normed block states."""
        stacked = unweighted_rms_norm(torch.stack(block_states + [partial], dim=-2))
        query = query.view(-1, self.hidden)
        weights = torch.softmax(torch.einsum("qh,snh->sqn", query, stacked) * self.hidden**-0.5, dim=-1)
        return torch.einsum("sqn,snh->sqh", weights, stacked)

    def short_conv(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """`Inkling2LoopShortConvolution`; `weight` is `[channels, 1, kernel]`, last tap = current token."""
        kernel = weight.shape[-1]
        out = F.conv1d(x.T.unsqueeze(0), weight, padding=kernel - 1, groups=x.shape[-1])[0, :, : x.shape[0]].T
        return (out + x) / math.sqrt(kernel + 1)

    def dynamic_short_conv(self, x, dynamic_input, static_w, weight_proj, heads: int) -> torch.Tensor:
        """`Inkling2LoopDynamicShortConvolution`; tap index is the backward distance."""
        seq, channels = x.shape
        kernel = static_w.shape[0]
        dynamic = (dynamic_input @ weight_proj.T).view(seq, heads, 1, kernel)
        weights = static_w.T.reshape(heads, -1, kernel) + dynamic  # [seq, heads, head_dim, kernel]
        padded = F.pad(x.T, (kernel - 1, 0))  # [channels, seq + kernel - 1]
        windows = padded.unfold(-1, kernel, 1).flip(-1).permute(1, 0, 2)  # [seq, channels, lag]
        windows = windows.reshape(seq, heads, -1, kernel)
        out = (windows * weights).sum(-1).reshape(seq, channels)
        return (out + x) / math.sqrt(2 * kernel + 1)

    def attention(self, layer: int, hq, hk, hv, positions, tag: str) -> torch.Tensor:
        c, prefix = self.c, f"model.llm.layers.{layer}.attn."
        sliding = layer in self.local
        heads = c["swa_num_attention_heads"] if sliding else c["num_attention_heads"]
        kv_heads = c["swa_num_key_value_heads"] if sliding else c["num_key_value_heads"]
        head_dim = c["swa_head_dim"] if sliding else c["head_dim"]
        extent = c["sliding_window_size"] if sliding else c["rel_extent"]
        seq = hq.shape[0]
        scale = self.log_scale(positions) if not sliding else None

        q = (hq @ self.p(prefix + "wq_du.weight").T).view(seq, heads, head_dim)
        q = l2_clip(q * self.p(prefix + "q_l2_clip.scale"), c["qk_l2_clip_threshold"])
        if scale is not None:
            q = q * scale[:, None, None]

        k = hk @ self.p(prefix + "wk_dv.weight").T
        v = hv @ self.p(prefix + "wv_dv.weight").T
        if sliding:
            k = self.short_conv(k, self.p(prefix + "k_sconv.weight"))
            v = self.short_conv(v, self.p(prefix + "v_sconv.weight"))
        else:
            k = self.dynamic_short_conv(
                k, hk, self.p(prefix + "k_sconv.static_w"), self.p(prefix + "k_sconv.weight_proj"), kv_heads
            )
            v = self.dynamic_short_conv(
                v, hv, self.p(prefix + "v_sconv.static_w"), self.p(prefix + "v_sconv.weight_proj"), kv_heads
            )
        k = l2_clip(k.view(seq, kv_heads, head_dim), c["qk_l2_clip_threshold"])
        v = unweighted_rms_norm(v.view(seq, kv_heads, head_dim), self.eps)
        groups = heads // kv_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)

        rel = (hq @ self.p(prefix + "wr_du.weight").T).view(seq, heads, c["d_rel"])
        rel = l2_clip(rel * self.p(prefix + "rel_l2_clip.scale"), c["rel_l2_clip_threshold"])
        if scale is not None:
            rel = rel * scale[:, None, None]
        rel_logits = rel @ self.p(prefix + "rel_logits_proj.proj")  # [seq, heads, extent]
        distance = positions[:, None] - positions[None, :]  # [query, key]
        outside = (distance < 0) | (distance >= extent)
        index = distance.clamp(0, extent - 1)[:, None, :].expand(seq, heads, seq)
        bias = rel_logits.gather(-1, index).masked_fill(outside[:, None, :], 0.0)

        logits = torch.einsum("qhd,khd->qhk", q, k) * head_dim**-0.5 + bias
        visible = (distance >= 0) & ((distance < c["sliding_window_size"]) if sliding else (distance >= 0))
        logits = logits.masked_fill(~visible[:, None, :], float("-inf"))
        sinks = (hq @ self.p(prefix + "attn_sink_proj.weight").T).unsqueeze(-1)  # [seq, heads, 1]
        probs = torch.softmax(torch.cat([logits, sinks], dim=-1), dim=-1)[..., :-1]
        out = torch.einsum("qhk,khd->qhd", probs, v).reshape(seq, heads * head_dim)

        gate = torch.sigmoid(hq @ self.p(prefix + "gate_attn.weight").T)
        return (out * gate) @ self.p(prefix + "wo_ud.weight").T

    def moe(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        c, prefix = self.c, f"model.llm.layers.{layer}.mlp."
        scores = torch.sigmoid(x @ self.p(prefix + "gate.weight").T)
        top = torch.topk(scores + self.p(prefix + "gate.bias"), c["num_experts_per_tok"], dim=-1).indices
        weights = scores.gather(1, top)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20) * c["route_scale"]
        w13, w2 = self.p(prefix + "experts.w13"), self.p(prefix + "experts.w2")
        out = torch.zeros_like(x)
        for token in range(x.shape[0]):
            # ascending expert order, so the sum order is fixed
            order = torch.argsort(top[token])
            for slot in order:
                expert = top[token, slot]
                gate_up = w13[expert] @ x[token]
                gate, up = gate_up[0::2], gate_up[1::2]  # published rows interleave gate and up
                act = softcap(gate, c["mlp_gate_softcap"]) * torch.sigmoid(gate) * softcap(up, c["mlp_up_softcap"])
                out[token] += (w2[expert] @ act) * weights[token, slot]
        return out

    def message_gate(self, layer: int, which: str, hidden: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        prefix = f"model.llm.layers.{layer}.controllers.{which}."
        joined = torch.cat([unweighted_rms_norm(hidden, self.eps), unweighted_rms_norm(message, self.eps)], dim=-1)
        low = joined @ self.p(prefix + "down_weight").T
        forget = torch.tanh(low @ self.p(prefix + "forget_weight").T)
        gain = torch.tanh(low @ self.p(prefix + "gain_weight").T)
        return (1 - forget) * hidden + gain * message

    def layer(self, layer: int, hidden, block_states: list, positions, messages, tag: str) -> torch.Tensor:
        """One layer application. `messages` is `[seq, 4, hidden]` (query, key, value, mlp) or None."""
        sliding = layer in self.local
        lp = f"model.llm.layers.{layer}."
        boundary = layer % self.block == 0
        block_residual = hidden

        if layer == 0:
            inputs = [hidden]
        else:
            query = self.p(f"model.llm.attn_res.attn_queries.{layer - 1}")
            inputs = list(self.depth_attention(query, block_states, hidden).unbind(-2))
        if sliding:
            hq = hk = hv = inputs[0]
            if messages is not None:
                hq = self.message_gate(layer, "attn", inputs[0], messages[:, 0])
                hk = self.message_gate(layer, "k", inputs[0], messages[:, 1])
                hv = self.message_gate(layer, "v", inputs[0], messages[:, 2])
            norm = self.p(lp + "attn_norm.weight")
            hq, hk, hv = (rms_norm(h, norm, self.eps) for h in (hq, hk, hv))
        else:
            hq, hk, hv = inputs if len(inputs) == 3 else (inputs[0],) * 3
            if messages is not None:
                hq = self.message_gate(layer, "attn", hq, messages[:, 0])
                hk = self.message_gate(layer, "k", hk, messages[:, 1])
                hv = self.message_gate(layer, "v", hv, messages[:, 2])
            hq = rms_norm(hq, self.p(lp + "q_input_norm.weight"), self.eps)
            hk = rms_norm(hk, self.p(lp + "k_input_norm.weight"), self.eps)
            hv = rms_norm(hv, self.p(lp + "v_input_norm.weight"), self.eps)

        attn = self.attention(layer, hq, hk, hv, positions, tag)
        attn = self.short_conv(attn, self.p(lp + "attn_sconv.weight"))
        self.trace[f"{tag}.attn"] = attn

        if boundary:
            residual = attn
            states = block_states + [block_residual]
        else:
            residual = block_residual + attn
            states = block_states
        mlp_in = self.depth_attention(self.p(f"model.llm.attn_res.mlp_queries.{layer}"), states, residual)[:, 0]
        if messages is not None:
            mlp_in = self.message_gate(layer, "mlp", mlp_in, messages[:, 3])
        mlp = self.moe(layer, rms_norm(mlp_in, self.p(lp + "mlp_norm.weight"), self.eps))
        mlp = self.short_conv(mlp, self.p(lp + "mlp_sconv.weight"))
        out = residual + mlp
        self.trace[f"{tag}.out"] = out
        return out

    def forward(self, tokens: torch.Tensor, image_features: torch.Tensor | None = None) -> torch.Tensor:
        positions = torch.arange(tokens.shape[0])
        hidden = rms_norm(self.p("model.llm.embed.weight")[tokens], self.p("model.llm.embed_norm.weight"), self.eps)
        if image_features is not None:
            # `Inkling2LoopModel.forward`: image features replace the normed embeddings of the image tokens.
            hidden = hidden.clone()
            hidden[tokens == IMAGE_TOKEN_ID] = image_features.to(hidden.dtype)
        self.trace["embed"] = hidden
        block_states: list[torch.Tensor] = []

        def run(layer, hidden, block_states, messages, tag):
            out = self.layer(layer, hidden, block_states, positions, messages, tag)
            if layer % self.block == 0:
                block_states = block_states + [hidden]
            return out, block_states

        for layer in range(self.loop_start):
            hidden, block_states = run(layer, hidden, block_states, None, f"app.{layer}")
        pre_hidden, pre_states = hidden, block_states

        application = self.loop_start
        for layer in range(self.loop_start, self.loop_end):
            hidden, block_states = run(layer, hidden, block_states, None, f"app.{application}")
            application += 1

        queries = self.p("model.llm.middle_cycle_cross_pass_queries").view(-1, 2, self.hidden)
        kv_queries = self.p("model.llm.middle_cycle_cross_pass_kv_queries").view(-1, 2, self.hidden)
        message_queries = torch.cat([queries[:, :1], kv_queries, queries[:, 1:]], dim=1)  # query, key, value, mlp
        for _ in range(1, self.passes):
            previous_states = block_states[len(pre_states) + 1 :]
            if not previous_states:
                raise SystemExit("the loop spans one block: message queries would not matter")
            previous_hidden = hidden
            hidden, block_states = pre_hidden, pre_states
            for index, layer in enumerate(range(self.loop_start, self.loop_end)):
                messages = self.depth_attention(message_queries[index], previous_states, previous_hidden)
                hidden, block_states = run(layer, hidden, block_states, messages, f"app.{application}")
                application += 1

        for layer in range(self.loop_end, self.layers):
            hidden, block_states = run(layer, hidden, block_states, None, f"app.{application}")
            application += 1

        hidden = self.depth_attention(self.p("model.llm.attn_res.output_query"), block_states, hidden)[:, 0]
        hidden = rms_norm(hidden, self.p("model.llm.norm.weight"), self.eps)
        self.trace["final_hidden"] = hidden
        unembed = self.p("model.llm.unembed.weight")[: self.c["unpadded_vocab_size"]]
        return softcap(hidden @ unembed.T, self.c["final_logit_softcapping"])


# --- vision (transcribed from `Inkling2VisionModel` and `Inkling2ImageProcessor`) ---


def preprocess_image(image: torch.Tensor, cell: int) -> tuple[torch.Tensor, int, int]:
    """`[height, width, 3 or 4]` U8 -> (`[rows * cols, cell, cell, 3]` in [-1, 1], rows, cols).

    Transparency is composited over a checkerboard of 8-pixel squares, the
    image is padded on the bottom and right to whole cells with -1 on the
    0-255 scale, and there is no resize: one image token per cell.
    """
    height, width, channels = image.shape
    pixels = image.to(torch.int32)
    if channels == 4:
        rows = torch.arange(height)[:, None] // 8
        columns = torch.arange(width)[None, :] // 8
        background = torch.where((rows + columns) % 2 == 0, 204, 153)[..., None]
        alpha = pixels[..., 3:4]
        pixels = (pixels[..., :3] * alpha + background * (255 - alpha) + 127) // 255
    rows, columns = math.ceil(height / cell), math.ceil(width / cell)
    padded = torch.full((rows * cell, columns * cell, 3), -1.0)
    padded[:height, :width] = pixels.float()
    padded = (padded * (1.0 / 255.0) - 0.5) / 0.5
    cells = padded.reshape(rows, cell, columns, cell, 3).permute(0, 2, 1, 3, 4)
    return cells.reshape(rows * columns, cell, cell, 3), rows, columns


def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)


def vision_forward(w: dict[str, torch.Tensor], pixels: torch.Tensor, rows: int, columns: int) -> torch.Tensor:
    """One image's `[rows * columns, text_hidden]` features from its `[cells, cell, cell, 3]` pixels."""
    v, prefix = VISION_CONFIG, "model.visual."
    backbone = prefix + "backbone."
    merge, patch, hidden, heads = v["adapter_merge_size"], v["native_patch_size"], v["hidden_size"], v["num_attention_heads"]
    side, eps = v["posemb_grid_size"], v["layer_norm_eps"]
    dtype = w[backbone + "ln_pre.weight"].dtype
    pixels = pixels.to(dtype)

    # Patches in block-major order: a cell's merge x merge patches are adjacent.
    patches = pixels.reshape(-1, merge, patch, merge, patch, 3).transpose(2, 3).reshape(-1, patch * patch * 3)
    states = patches @ w[backbone + "embeddings.patch_embedding.weight"].T
    grid_rows, grid_columns = torch.meshgrid(
        torch.arange(rows * merge), torch.arange(columns * merge), indexing="ij"
    )
    order = (rows, merge, columns, merge)
    row = grid_rows.reshape(order).transpose(1, 2).flatten()
    column = grid_columns.reshape(order).transpose(1, 2).flatten()

    # The learned side x side position grid, resampled bilinearly (half-pixel centres, edges clamped).
    def taps(index: torch.Tensor, size: int):
        source = (index.to(dtype) + 0.5) * side / size - 0.5
        floor = torch.floor(source)
        at = torch.stack([floor.long(), floor.long() + 1], dim=1).clamp(0, side - 1)
        distance = (source[:, None] - floor[:, None] - torch.arange(2)).abs()
        return at, (1 - distance).clamp(min=0)

    row_taps, row_weights = taps(row, rows * merge)
    column_taps, column_weights = taps(column, columns * merge)
    indices = (row_taps[:, :, None] * side + column_taps[:, None, :]).reshape(-1, 4)
    weights = (row_weights[:, :, None] * column_weights[:, None, :]).reshape(-1, 4)
    table = w[backbone + "embeddings.position_embedding"]
    states = states + (table[indices] * weights[:, :, None]).sum(1)

    # A class token leads the image; it sits at (0, 0) and the patches are 1-indexed.
    cls = w[backbone + "embeddings.class_embedding"] + w[backbone + "embeddings.cls_position_embedding"]
    states = torch.cat([cls[None], states])
    states = layer_norm(states, w[backbone + "ln_pre.weight"], w[backbone + "ln_pre.bias"], eps)
    row = torch.cat([torch.zeros(1, dtype=torch.long), row + 1])
    column = torch.cat([torch.zeros(1, dtype=torch.long), column + 1])

    head_dim = hidden // heads
    spatial = head_dim // 2
    # The frequencies are an F32 table whatever the model's dtype (a buffer built at construction).
    inv_freq = torch.tensor(
        [np.float32(1.0) / np.float32(VISION_ROPE_THETA) ** (np.float32(i) / np.float32(spatial)) for i in range(0, spatial, 2)],
        dtype=dtype,
    )
    # Width angles then height angles, each repeated for an interleaved channel pair.
    angles = torch.cat([column[:, None].to(dtype) * inv_freq, row[:, None].to(dtype) * inv_freq], dim=-1)
    cos = angles.cos().repeat_interleave(2, dim=-1)[:, None]
    sin = angles.sin().repeat_interleave(2, dim=-1)[:, None]

    def rotate(x: torch.Tensor) -> torch.Tensor:
        pairs = x.view(*x.shape[:-1], -1, 2)
        turned = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(start_dim=-2)
        return x * cos + turned * sin

    count = states.shape[0]
    for layer in range(v["num_hidden_layers"]):
        lp = f"{backbone}layers.{layer}."

        def linear(x: torch.Tensor, name: str) -> torch.Tensor:
            return x @ w[lp + name + ".weight"].T + w[lp + name + ".bias"]

        x = layer_norm(states, w[lp + "ln_1.weight"], w[lp + "ln_1.bias"], eps)
        q = rotate(linear(x, "self_attn.q_proj").view(count, heads, head_dim)).transpose(0, 1)
        k = rotate(linear(x, "self_attn.k_proj").view(count, heads, head_dim)).transpose(0, 1)
        value = linear(x, "self_attn.v_proj").view(count, heads, head_dim).transpose(0, 1)
        attention = torch.softmax(q @ k.transpose(-1, -2) * head_dim**-0.5, dim=-1) @ value
        x = linear(attention.transpose(0, 1).reshape(count, hidden), "self_attn.out_proj")
        states = states + w[lp + "ls_1.gamma"] * x
        x = layer_norm(states, w[lp + "ln_2.weight"], w[lp + "ln_2.bias"], eps)
        x = linear(F.gelu(linear(x, "mlp.c_fc")), "mlp.c_proj")
        states = states + w[lp + "ls_2.gamma"] * x

    merged = states[1:].reshape(-1, merge * merge * hidden)
    return F.gelu(merged @ w[prefix + "proj_in.weight"].T) @ w[prefix + "proj_out.weight"].T


def build_vision(stored: dict, reference: dict) -> None:
    """Adds a random vision tower under its published names, from its own generator."""
    v, backbone = VISION_CONFIG, "model.visual.backbone."
    generator = torch.Generator().manual_seed(SEED + 1)
    hidden, width = v["hidden_size"], int(v["hidden_size"] * v["mlp_ratio"])
    merged = v["adapter_merge_size"] ** 2 * hidden

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, generator=generator) * scale

    def dense(name: str, tensor: torch.Tensor):
        value = bf16(tensor)
        stored[name] = ("BF16", value)
        reference[name] = value

    dense(backbone + "embeddings.class_embedding", randn(hidden))
    dense(backbone + "embeddings.cls_position_embedding", randn(hidden))
    dense(backbone + "embeddings.patch_embedding.weight", randn(hidden, 3 * v["native_patch_size"] ** 2, scale=0.5))
    dense(backbone + "embeddings.position_embedding", randn(v["posemb_grid_size"] ** 2, hidden))
    dense(backbone + "ln_pre.weight", 1.0 + randn(hidden, scale=0.2))
    dense(backbone + "ln_pre.bias", randn(hidden, scale=0.2))
    for layer in range(v["num_hidden_layers"]):
        lp = f"{backbone}layers.{layer}."
        for norm in ("ln_1", "ln_2"):
            dense(lp + norm + ".weight", 1.0 + randn(hidden, scale=0.2))
            dense(lp + norm + ".bias", randn(hidden, scale=0.2))
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            dense(lp + f"self_attn.{name}.weight", randn(hidden, hidden, scale=hidden**-0.5))
            dense(lp + f"self_attn.{name}.bias", randn(hidden, scale=0.2))
        dense(lp + "mlp.c_fc.weight", randn(width, hidden, scale=hidden**-0.5))
        dense(lp + "mlp.c_fc.bias", randn(width, scale=0.2))
        dense(lp + "mlp.c_proj.weight", randn(hidden, width, scale=width**-0.5))
        dense(lp + "mlp.c_proj.bias", randn(hidden, scale=0.2))
        dense(lp + "ls_1.gamma", 0.5 + randn(hidden, scale=0.2))
        dense(lp + "ls_2.gamma", 0.5 + randn(hidden, scale=0.2))
    dense("model.visual.proj_in.weight", randn(v["projection_hidden_dim"], merged, scale=merged**-0.5))
    dense(
        "model.visual.proj_out.weight",
        randn(v["text_hidden_size"], v["projection_hidden_dim"], scale=v["projection_hidden_dim"] ** -0.5),
    )


def build_images(reference: dict):
    """The fixture's images, their processed pixels and features, and a prompt that carries them."""
    generator = torch.Generator().manual_seed(SEED + 2)
    cell = VISION_CONFIG["native_patch_size"] * VISION_CONFIG["adapter_merge_size"]
    images = []
    for height, width, channels in IMAGES:
        image = torch.randint(0, 256, (height, width, channels), generator=generator, dtype=torch.uint8)
        pixels, rows, columns = preprocess_image(image, cell)
        images.append({"image": image, "pixels": pixels, "grid": (rows, columns)})
    with torch.no_grad():
        for entry in images:
            entry["features"] = vision_forward(reference, entry["pixels"], *entry["grid"])
    # Text, the first image, text, the second image, text.
    text = lambda count: torch.randint(0, IMAGE_TOKEN_ID, (count,), generator=generator)
    slots = [torch.full((entry["features"].shape[0],), IMAGE_TOKEN_ID) for entry in images]
    tokens = torch.cat([text(5), slots[0], text(3), slots[1], text(6)])
    return images, tokens


# --- checkpoint construction ---


def build_checkpoint(generator: torch.Generator):
    """Returns (stored tensors as (dtype, tensor) in the published layout, reference weights in F32)."""
    c = TEXT_CONFIG
    hidden, experts, inter = c["hidden_size"], c["n_routed_experts"], c["intermediate_size"]
    loop = c["loop_config"]
    loop_layers = loop["core_end"] - loop["core_start"]
    stored: dict[str, tuple[str, torch.Tensor]] = {}
    reference: dict[str, torch.Tensor] = {}

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, generator=generator) * scale

    def dense(name: str, tensor: torch.Tensor, dtype: str = "BF16"):
        value = bf16(tensor) if dtype == "BF16" else tensor.float()
        stored[name] = (dtype, value)
        reference[name] = value

    dense("model.llm.embed.weight", randn(c["vocab_size"], hidden))
    dense("model.llm.embed_norm.weight", 1.0 + randn(hidden, scale=0.2))
    dense("model.llm.norm.weight", 1.0 + randn(hidden, scale=0.2))
    unembed = randn(c["vocab_size"], hidden, scale=0.6)
    unembed[c["unpadded_vocab_size"] :] = 0.0
    dense("model.llm.unembed.weight", unembed)
    dense("model.llm.attn_res.output_query", randn(hidden), "F32")
    dense("model.llm.middle_cycle_cross_pass_queries", randn(2 * loop_layers, hidden), "F32")
    dense("model.llm.middle_cycle_cross_pass_kv_queries", randn(2 * loop_layers, hidden), "F32")

    local = set(c["local_layer_ids"])
    for layer in range(c["num_hidden_layers"]):
        sliding = layer in local
        heads = c["swa_num_attention_heads"] if sliding else c["num_attention_heads"]
        kv_heads = c["swa_num_key_value_heads"] if sliding else c["num_key_value_heads"]
        head_dim = c["swa_head_dim"] if sliding else c["head_dim"]
        extent = c["sliding_window_size"] if sliding else c["rel_extent"]
        kernel = c["sconv_kernel_size"]
        lp = f"model.llm.layers.{layer}."

        if layer > 0:
            # `attn_queries.{i}` belongs to layer `i + 1`; full layers carry query, key and value queries.
            shape = (hidden,) if sliding else (3, hidden)
            dense(f"model.llm.attn_res.attn_queries.{layer - 1}", randn(*shape), "F32")
        dense(f"model.llm.attn_res.mlp_queries.{layer}", randn(hidden), "F32")

        dense(lp + "attn.wq_du.weight", randn(heads * head_dim, hidden, scale=0.25))
        dense(lp + "attn.wk_dv.weight", randn(kv_heads * head_dim, hidden, scale=0.25))
        dense(lp + "attn.wv_dv.weight", randn(kv_heads * head_dim, hidden, scale=0.25))
        dense(lp + "attn.wo_ud.weight", randn(hidden, heads * head_dim, scale=0.2))
        dense(lp + "attn.gate_attn.weight", randn(heads * head_dim, hidden, scale=0.2))
        dense(lp + "attn.wr_du.weight", randn(heads * c["d_rel"], hidden, scale=0.2))
        dense(lp + "attn.attn_sink_proj.weight", randn(heads, hidden, scale=0.2))
        dense(lp + "attn.q_l2_clip.scale", 1.0 + randn(heads, head_dim, scale=0.2))
        dense(lp + "attn.rel_l2_clip.scale", 1.0 + randn(heads, c["d_rel"], scale=0.2))
        dense(lp + "attn.rel_logits_proj.proj", randn(c["d_rel"], extent))
        if sliding:
            dense(lp + "attn.k_sconv.weight", randn(kv_heads * head_dim, 1, kernel, scale=0.5))
            dense(lp + "attn.v_sconv.weight", randn(kv_heads * head_dim, 1, kernel, scale=0.5))
            dense(lp + "attn_norm.weight", 1.0 + randn(hidden, scale=0.2))
        else:
            for name in ("k_sconv", "v_sconv"):
                dense(lp + f"attn.{name}.static_w", randn(kernel, kv_heads * head_dim, scale=0.5))
                dense(lp + f"attn.{name}.weight_proj", randn(kv_heads * kernel, hidden, scale=0.1))
            for name in ("q_input_norm", "k_input_norm", "v_input_norm"):
                dense(lp + f"{name}.weight", 1.0 + randn(hidden, scale=0.2))
        dense(lp + "attn_sconv.weight", randn(hidden, 1, kernel, scale=0.5))
        dense(lp + "mlp_sconv.weight", randn(hidden, 1, kernel, scale=0.5))
        dense(lp + "mlp_norm.weight", 1.0 + randn(hidden, scale=0.2))
        dense(lp + "mlp.gate.weight", randn(experts, hidden, scale=0.3))
        dense(lp + "mlp.gate.bias", randn(experts, scale=0.5), "F32")

        if loop["core_start"] <= layer < loop["core_end"]:
            for which in ("attn", "k", "v", "mlp"):
                dense(lp + f"controllers.{which}.down_weight", randn(loop["gate_rank"], 2 * hidden, scale=0.2))
                dense(lp + f"controllers.{which}.forget_weight", randn(hidden, loop["gate_rank"], scale=0.3))
                dense(lp + f"controllers.{which}.gain_weight", randn(hidden, loop["gate_rank"], scale=0.3))

        # Experts. Scales differ per expert so the per-expert NVFP4 factor matters; the gate rows are
        # large enough for the gate softcap to bend.
        expert_gain = 0.5 + torch.rand(experts, 1, 1, generator=generator)
        w13 = randn(experts, 2 * inter, hidden, scale=0.6) * expert_gain
        w2 = randn(experts, hidden, inter, scale=0.3) * expert_gain
        for name, weight in (("w13", w13), ("w2", w2)):
            key = lp + f"mlp.experts.{name}_weight"
            if layer in MXFP8_LAYERS:
                values, scale = mxfp8_quantize(weight)
                stored[key] = ("F8_E4M3", values)
                stored[key + "_scale"] = ("U8", scale)
                reference[lp + f"mlp.experts.{name}"] = mxfp8_dequantize(values, scale)
            else:
                packed, scale, amax, denominator = nvfp4_quantize(weight)
                stored[key] = ("U8", packed)
                stored[key + "_scale"] = ("F8_E4M3", scale)
                stored[key + "_amax"] = ("F32", amax)
                stored[key + "_outer_scale_denominator"] = ("F32", denominator)
                reference[lp + f"mlp.experts.{name}"] = nvfp4_dequantize(packed, scale, amax, denominator)
    return stored, reference


def tensor_bytes(dtype: str, tensor: torch.Tensor) -> bytes:
    if dtype == "BF16":
        return tensor.to(torch.bfloat16).contiguous().view(torch.int16).numpy().tobytes()
    if dtype == "F32":
        return tensor.to(torch.float32).contiguous().numpy().astype("<f4").tobytes()
    if dtype in ("U8", "F8_E4M3"):
        return tensor.to(torch.uint8).contiguous().numpy().tobytes()
    if dtype == "I64":
        return tensor.to(torch.int64).contiguous().numpy().astype("<i8").tobytes()
    raise ValueError(dtype)


def write_safetensors(path: Path, tensors: dict[str, tuple[str, torch.Tensor]]) -> None:
    entries, blobs, offset = {"__metadata__": {"format": "pt"}}, [], 0
    for key in sorted(tensors):
        dtype, tensor = tensors[key]
        blob = tensor_bytes(dtype, tensor)
        entries[key] = {"dtype": dtype, "shape": list(tensor.shape), "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    head = json.dumps(entries, separators=(",", ":")).encode()
    head += b" " * (-len(head) % 8)
    path.write_bytes(struct.pack("<Q", len(head)) + head + b"".join(blobs))


class widened:
    """Runs transformers' explicit F32 steps in F64 while it is entered.

    Its grouped matrix product refuses F64, so the same product is done group
    by group.
    """

    def __enter__(self):
        import transformers.integrations.moe as moe

        def grouped_mm(input, weight, offs):
            rows, start = [], 0
            for group, end in enumerate(offs.tolist()):
                rows.append(input[start:end].to(weight.dtype) @ weight[group])
                start = end
            return torch.cat(rows, 0)

        self.moe = moe
        self.saved = (torch.float32, torch.float, torch.Tensor.float, moe._grouped_mm)
        torch.float32 = torch.float = torch.float64
        torch.Tensor.float, moe._grouped_mm = torch.Tensor.double, grouped_mm

    def __exit__(self, *_):
        torch.float32, torch.float, torch.Tensor.float, self.moe._grouped_mm = self.saved


def double(weights: dict) -> dict:
    return {name: value.double() if value.is_floating_point() else value for name, value in weights.items()}


def check_with_transformers(out: Path, tokens: torch.Tensor, logits: torch.Tensor, reference: dict) -> None:
    """Compare the text transcription with transformers on the written checkpoint.

    In F32 the two differ by rounding alone, and the tiny random model
    amplifies it (activations near 500, sixteen layer applications): about
    2e-3 on the logits. So the deciding comparison runs both in F64, where a
    transcription error cannot hide.
    """
    from transformers import Inkling2LoopForCausalLM  # needs the release that carries inkling2_loop

    model = Inkling2LoopForCausalLM.from_pretrained(out, dtype=torch.float32).eval()
    with torch.no_grad():
        theirs = model(input_ids=tokens[None]).logits[0]
    single = (theirs - logits).abs().max().item()
    print(f"transformers F32 max |logit difference|: {single:.3e}")

    states = []
    hooks = [
        layer.register_forward_hook(lambda _module, _inputs, output: states.append(output))
        for layer in model.model.layers
    ]
    with widened(), torch.no_grad():
        wide = model.double()(input_ids=tokens[None]).logits[0]
        ours = Reference(TEXT_CONFIG, double(reference))
        ours_logits = ours.forward(tokens)
    for hook in hooks:
        hook.remove()
    if wide.dtype != torch.float64 or ours_logits.dtype != torch.float64:
        raise SystemExit("the F64 comparison did not run in F64")
    exact = (wide - ours_logits).abs().max().item()
    layers = max(
        (state[0] - ours.trace[f"app.{index}.out"].reshape(state[0].shape)).abs().max().item()
        for index, state in enumerate(states)
    )
    print(f"transformers F64 max |logit difference|: {exact:.3e}, worst layer application: {layers:.3e}")
    print(f"F32 against the F64 result: transformers {(theirs - wide).abs().max().item():.3e}, "
          f"transcription {(logits - wide).abs().max().item():.3e}")
    if exact > 1e-9 or layers > 1e-8 or single > 1e-2:
        raise SystemExit("the transcription disagrees with transformers")


def check_images_with_transformers(out: Path, images: list, tokens, logits, reference: dict) -> None:
    """The image processor, the vision tower and a prompt with images, against transformers."""
    from PIL import Image
    from transformers import Inkling2LoopForConditionalGeneration
    from transformers.models.inkling2.image_processing_inkling2 import Inkling2ImageProcessor

    cell = VISION_CONFIG["native_patch_size"] * VISION_CONFIG["adapter_merge_size"]
    processor = Inkling2ImageProcessor(size={"height": cell, "width": cell})
    pictures = [Image.fromarray(entry["image"].numpy()) for entry in images]
    processed = processor(images=pictures, return_tensors="pt")
    pixels = torch.cat([entry["pixels"] for entry in images])
    grid = torch.tensor([[1, *entry["grid"]] for entry in images])
    if not torch.equal(processed["image_grid_thw"], grid):
        raise SystemExit("the image grids disagree with transformers")
    pixel_error = (processed["pixel_values"] - pixels).abs().max().item()
    print(f"image processor max |pixel difference|: {pixel_error:.3e}")

    model = Inkling2LoopForConditionalGeneration.from_pretrained(out, dtype=torch.float32).eval()
    features = torch.cat([entry["features"] for entry in images])
    with torch.no_grad():
        theirs = torch.cat(model.model.get_image_features(pixels, grid).pooler_output)
        theirs_logits = model(input_ids=tokens[None], pixel_values=pixels, image_grid_thw=grid).logits[0]
    feature_error = (theirs - features).abs().max().item()
    single = (theirs_logits - logits).abs().max().item()
    print(f"transformers F32 max |image feature difference|: {feature_error:.3e} "
          f"(features up to {features.abs().max().item():.2f}), |logit difference| with images: {single:.3e}")

    with widened(), torch.no_grad():
        model = model.double()
        wide = torch.cat(model.model.get_image_features(pixels.double(), grid).pooler_output)
        wide_logits = model(input_ids=tokens[None], pixel_values=pixels.double(), image_grid_thw=grid).logits[0]
        weights = double(reference)
        ours = torch.cat([vision_forward(weights, e["pixels"].double(), *e["grid"]) for e in images])
        ours_logits = Reference(TEXT_CONFIG, weights).forward(tokens, ours)
    if wide.dtype != torch.float64 or ours.dtype != torch.float64 or ours_logits.dtype != torch.float64:
        raise SystemExit("the F64 image comparison did not run in F64")
    exact = (wide - ours).abs().max().item()
    exact_logits = (wide_logits - ours_logits).abs().max().item()
    print(f"transformers F64 max |image feature difference|: {exact:.3e}, |logit difference| with images: {exact_logits:.3e}")
    if pixel_error > 1e-6 or exact > 1e-9 or exact_logits > 1e-9 or feature_error > 1e-3 or single > 1e-2:
        raise SystemExit("the image transcription disagrees with transformers")


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    stored, reference = build_checkpoint(generator)
    build_vision(stored, reference)

    c = TEXT_CONFIG
    tokens = torch.randint(0, c["unpadded_vocab_size"] - 1, (TOTAL_TOKENS,), generator=generator)
    model = Reference(c, reference)
    with torch.no_grad():
        logits = model.forward(tokens)
    if not torch.isfinite(logits).all():
        raise SystemExit("non-finite reference logits")

    # The fixture must exercise what it claims to.
    stats = {
        "max_abs_logit": logits.abs().max().item(),
        "log_scale_max": model.log_scale(torch.arange(TOTAL_TOKENS)).max().item(),
    }
    print(json.dumps(stats))
    if stats["log_scale_max"] <= 1.0:
        raise SystemExit("the log scale never acts")

    write_safetensors(out / "model.safetensors", stored)
    (out / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    golden = {
        "tokens": ("I64", tokens),
        "prompt_tokens": ("I64", torch.tensor([PROMPT_TOKENS])),
        "logits": ("F32", logits),
    }
    for key, value in model.trace.items():
        golden[f"trace.{key}"] = ("F32", value)
    write_safetensors(out / "golden.safetensors", golden)

    images, image_tokens = build_images(reference)
    with torch.no_grad():
        image_logits = Reference(c, reference).forward(image_tokens, torch.cat([e["features"] for e in images]))
    if not torch.isfinite(image_logits).all():
        raise SystemExit("non-finite reference logits with images")
    image_golden = {"tokens": ("I64", image_tokens), "logits": ("F32", image_logits)}
    for index, entry in enumerate(images):
        image_golden[f"image.{index}.rgba"] = ("U8", entry["image"])
        image_golden[f"image.{index}.grid"] = ("I64", torch.tensor(entry["grid"]))
        image_golden[f"image.{index}.pixels"] = ("F32", entry["pixels"])
        image_golden[f"image.{index}.features"] = ("F32", entry["features"])
    write_safetensors(out / "images.safetensors", image_golden)
    print("wrote", out, "tensors:", len(stored), "trace:", len(model.trace), "images:", len(images))

    if "--check-transformers" in sys.argv:
        check_with_transformers(out, tokens, logits, reference)
        check_images_with_transformers(out, images, image_tokens, image_logits, reference)


if __name__ == "__main__":
    main()
