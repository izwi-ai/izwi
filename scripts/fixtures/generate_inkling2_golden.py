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
CONFIG = {
    "dtype": "bfloat16",
    "architectures": ["Inkling2LoopForConditionalGeneration"],
    "model_type": "inkling2_loop",
    "eos_token_id": 149,
    "text_config": TEXT_CONFIG,
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

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.shape[0])
        hidden = rms_norm(self.p("model.llm.embed.weight")[tokens], self.p("model.llm.embed_norm.weight"), self.eps)
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


def check_with_transformers(out: Path, tokens: torch.Tensor, logits: torch.Tensor, reference: dict) -> None:
    """Compare the transcription with transformers on the written checkpoint.

    In F32 the two differ by rounding alone, and the tiny random model
    amplifies it (activations near 500, sixteen layer applications): about
    2e-3 on the logits. So the deciding comparison runs both in F64, where a
    transcription error cannot hide: transformers' explicit F32 steps are
    widened for the duration of the run, and its grouped matrix product, which
    refuses F64, is replaced by the same product group by group.
    """
    import transformers.integrations.moe as moe
    from transformers import Inkling2LoopForCausalLM  # needs the release that carries inkling2_loop

    model = Inkling2LoopForCausalLM.from_pretrained(out, dtype=torch.float32).eval()
    with torch.no_grad():
        theirs = model(input_ids=tokens[None]).logits[0]
    single = (theirs - logits).abs().max().item()
    print(f"transformers F32 max |logit difference|: {single:.3e}")

    def grouped_mm(input, weight, offs):
        rows, start = [], 0
        for group, end in enumerate(offs.tolist()):
            rows.append(input[start:end].to(weight.dtype) @ weight[group])
            start = end
        return torch.cat(rows, 0)

    states = []
    hooks = [
        layer.register_forward_hook(lambda _module, _inputs, output: states.append(output))
        for layer in model.model.layers
    ]
    saved = (torch.float32, torch.Tensor.float, moe._grouped_mm)
    torch.float32, torch.Tensor.float, moe._grouped_mm = torch.float64, torch.Tensor.double, grouped_mm
    try:
        with torch.no_grad():
            wide = model.double()(input_ids=tokens[None]).logits[0]
            ours = Reference(
                TEXT_CONFIG,
                {name: value.double() if value.is_floating_point() else value for name, value in reference.items()},
            )
            ours_logits = ours.forward(tokens)
    finally:
        torch.float32, torch.Tensor.float, moe._grouped_mm = saved
        for hook in hooks:
            hook.remove()
    if wide.dtype != torch.float64 or ours_logits.dtype != torch.float64:
        raise SystemExit("the F64 comparison did not run in F64")
    double = (wide - ours_logits).abs().max().item()
    layers = max(
        (state[0] - ours.trace[f"app.{index}.out"].reshape(state[0].shape)).abs().max().item()
        for index, state in enumerate(states)
    )
    print(f"transformers F64 max |logit difference|: {double:.3e}, worst layer application: {layers:.3e}")
    print(f"F32 against the F64 result: transformers {(theirs - wide).abs().max().item():.3e}, "
          f"transcription {(logits - wide).abs().max().item():.3e}")
    if double > 1e-9 or layers > 1e-8 or single > 1e-2:
        raise SystemExit("the transcription disagrees with transformers")


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    stored, reference = build_checkpoint(generator)

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
    print("wrote", out, "tensors:", len(stored), "trace:", len(model.trace))

    if "--check-transformers" in sys.argv:
        check_with_transformers(out, tokens, logits, reference)


if __name__ == "__main__":
    main()
