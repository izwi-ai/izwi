#!/usr/bin/env python3
"""Generate the Inkling 2 block drafter golden-parity fixture.

Builds a tiny random drafter in the layout of the published
`thinkingmachines/Inkling-2-Mini` `mtp.safetensors` (original tensor names,
BF16 tensors), a `config.json` that carries its `mtp_config` and the few
main-model fields the drafter reads, and records the reference values of one
draft round: the fused states, every layer's context keys and values, the
block's final hidden states, the base logits, a greedy chain and a chain
driven by a fixed token sequence that is not the argmax.

The forward pass below is plain PyTorch written from the engines' sources
(no published reference implementation of the drafter exists): vLLM at commit
1185d98ea797be7a526d98f4c262cbdf36abefc2, `vllm/models/inkling2/nvidia/dspark.py`
(model), `.../nvidia/ops/dspark.py` (kernels), `vllm/models/inkling2/speculator.py`
(the sequential head) and `tests/models/inkling2/test_dspark.py` (its torch
references), cross-read against SGLang 4c085b66 and llama.cpp ca123ba7. It
runs in F32 on exactly the values the checkpoint stores (BF16-rounded); it
does not round activations to BF16 the way the engines' kernels do.

It is written step by step from that description on purpose, so that it is
not the same code as the line-by-line transcription of vLLM's forward that
was kept with the working notes (`drafter_reference.py`). `--check-transcription
PATH` loads that file and holds this one to it: every chain logit within 1e-4
(of values up to 30; the two differ in the order of their F32 sums) and the
same greedy tokens. The check passed when the fixture was written.

The tiny geometry keeps what the published one has: four taps, one fusion row
per draft layer plus one for the head, fewer key/value heads than query heads,
heads whose total width is not the hidden size, two local layers around a
global one, a window shorter than the committed length (so local layers drop
context), a padded vocabulary, and logits large enough for the soft cap to
act.

Usage:

    python3 scripts/fixtures/generate_inkling2_drafter_golden.py \
        crates/izwi-core/tests/fixtures/inkling2_drafter_golden

The Rust tests are in `inkling2::drafter_tests`.
"""

import importlib.util
import json
import struct
import sys
from pathlib import Path

import torch

# Chosen so the greedy chain picks several different tokens, each well clear
# of the runner-up (checked below).
SEED = 20261128
COMMITTED = 20  # committed positions; the anchor sits at this position
ANCHOR_TOKEN = 37
# The second chain's tokens: fixed, and checked below not to be the argmax.
DRIVEN_TOKENS = [5, 149, 0, 77, 77, 12, 101]

MTP_CONFIG = {
    "hidden_size": 64,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "intermediate_size": 96,
    "max_position_embeddings": 4096,
    "torch_dtype": "bfloat16",
    "q_bias": False,
    "o_bias": False,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "local_layer_ids": [0, 2],
    "sliding_window_size": 6,
    "block_size": 8,
    "markov_rank": 16,
    "markov_head_type": "rnn_hidden",
    # 16 layer visits in the main model below (8 layers, 4 of them run 3
    # times); the last id stands for its final-norm output.
    "target_layer_ids": [3, 7, 13, 16],
}
CONFIG = {
    "dtype": "bfloat16",
    "architectures": ["Inkling2LoopForConditionalGeneration"],
    "model_type": "inkling2_loop",
    # Only what the drafter reads of the main model.
    "text_config": {
        "model_type": "inkling2_loop_text",
        "hidden_size": 64,
        "num_hidden_layers": 8,
        "vocab_size": 160,
        "unpadded_vocab_size": 150,
        "final_logit_softcapping": 30.0,
        "loop_config": {"core_start": 2, "core_end": 6, "num_passes": 3, "gate_rank": 8, "is_varloop": False},
    },
    "mtp_config": MTP_CONFIG,
}


def bf16(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(torch.bfloat16).to(torch.float32)


def tensor_bytes(dtype: str, tensor: torch.Tensor) -> bytes:
    if dtype == "BF16":
        return tensor.to(torch.bfloat16).contiguous().view(torch.int16).numpy().tobytes()
    if dtype == "F32":
        return tensor.to(torch.float32).contiguous().numpy().astype("<f4").tobytes()
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


def build_weights(generator: torch.Generator) -> dict[str, torch.Tensor]:
    """The drafter's tensors, already rounded to BF16 (held as F32)."""
    c = MTP_CONFIG
    hidden, rank = c["hidden_size"], c["markov_rank"]
    query = c["num_attention_heads"] * c["head_dim"]
    key_value = c["num_key_value_heads"] * c["head_dim"]
    padded = CONFIG["text_config"]["vocab_size"]

    def normal(*shape, scale=1.0):
        return torch.randn(*shape, generator=generator) * scale

    def gain(width):
        return 0.5 + torch.rand(width, generator=generator)

    w = {
        "fuse_weight": normal(c["num_hidden_layers"] + 1, len(c["target_layer_ids"])),
        "mask_embed": normal(hidden),
        "norm.weight": gain(hidden),
        "markov_head.dflare_norm.weight": gain(hidden),
        "markov_head.markov_w1.weight": normal(padded, rank),
        "markov_head.rnn_proj.weight": normal(3 * rank, 2 * rank + 2 * hidden, scale=0.12),
        "markov_head.rnn_proj.bias": normal(3 * rank, scale=0.3),
        "markov_head.hidden_out.weight": normal(hidden, rank, scale=0.08),
        "markov_head.hidden_out.bias": normal(hidden, scale=0.03),
    }
    for layer in range(c["num_hidden_layers"]):
        p = f"layers.{layer}."
        w[p + "dflare_norm.weight"] = gain(hidden)
        w[p + "input_layernorm.weight"] = gain(hidden)
        w[p + "post_attention_layernorm.weight"] = gain(hidden)
        # Gains above one sharpen the attention, so that a row's position
        # (its rotation) and the context it reads change what it predicts.
        w[p + "self_attn.q_norm.weight"] = 2.0 * gain(c["head_dim"])
        w[p + "self_attn.k_norm.weight"] = 2.0 * gain(c["head_dim"])
        w[p + "self_attn.k_norm_ctx.weight"] = 2.0 * gain(c["head_dim"])
        w[p + "self_attn.q_proj.weight"] = normal(query, hidden, scale=hidden**-0.5)
        w[p + "self_attn.k_proj.weight"] = normal(key_value, hidden, scale=hidden**-0.5)
        w[p + "self_attn.v_proj.weight"] = normal(key_value, hidden, scale=hidden**-0.5)
        w[p + "self_attn.k_proj_ctx.weight"] = normal(key_value, hidden, scale=hidden**-0.5)
        w[p + "self_attn.v_proj_ctx.weight"] = normal(key_value, hidden, scale=hidden**-0.5)
        w[p + "self_attn.o_proj.weight"] = normal(hidden, query, scale=query**-0.5)
        w[p + "mlp.gate_proj.weight"] = normal(c["intermediate_size"], hidden, scale=hidden**-0.5)
        w[p + "mlp.up_proj.weight"] = normal(c["intermediate_size"], hidden, scale=hidden**-0.5)
        w[p + "mlp.down_proj.weight"] = normal(hidden, c["intermediate_size"], scale=c["intermediate_size"] ** -0.5)
    return {name: bf16(tensor) for name, tensor in w.items()}


class Reference:
    """One request, dense tensors, no cache: the drafter's arithmetic only."""

    def __init__(self, w: dict[str, torch.Tensor]):
        self.w = w
        c, text = MTP_CONFIG, CONFIG["text_config"]
        self.layers = c["num_hidden_layers"]
        self.heads, self.kv_heads, self.head_dim = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        self.eps, self.theta = c["rms_norm_eps"], c["rope_theta"]
        self.local, self.window = c["local_layer_ids"], c["sliding_window_size"]
        self.block, self.rank = c["block_size"], c["markov_rank"]
        self.cap, self.vocab = text["final_logit_softcapping"], text["unpadded_vocab_size"]

    def rms(self, x: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * gain

    def rotate(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """`x` is `[rows, heads, head_dim]`; the first half of a head pairs
        with its second half, and the whole head is rotated."""
        d = x.shape[-1]
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
        angle = positions.to(torch.float32)[:, None] * inv_freq[None, :]
        cos, sin = angle.cos()[:, None, :], angle.sin()[:, None, :]
        first, second = x[..., : d // 2], x[..., d // 2 :]
        return torch.cat([first * cos - second * sin, second * cos + first * sin], dim=-1)

    def fuse(self, taps: torch.Tensor, row: int, gain: torch.Tensor) -> torch.Tensor:
        """`taps` is `[positions, taps, hidden]`. One row of `fuse_weight` per
        consumer; softmax over the taps, weighted sum, RMS norm."""
        alpha = torch.softmax(self.w["fuse_weight"], dim=-1)[row]
        mixed = torch.zeros_like(taps[:, 0])
        for tap in range(taps.shape[1]):
            mixed = mixed + alpha[tap] * taps[:, tap]
        return self.rms(mixed, gain)

    def context(self, layer: int, fused: torch.Tensor, positions: torch.Tensor):
        p = f"layers.{layer}.self_attn."
        keys = (fused @ self.w[p + "k_proj_ctx.weight"].T).view(-1, self.kv_heads, self.head_dim)
        keys = self.rotate(self.rms(keys, self.w[p + "k_norm_ctx.weight"]), positions)
        values = (fused @ self.w[p + "v_proj_ctx.weight"].T).view(-1, self.kv_heads, self.head_dim)
        return keys, values

    def attention(self, layer: int, x: torch.Tensor, positions: torch.Tensor, ctx_keys, ctx_values) -> torch.Tensor:
        p = f"layers.{layer}.self_attn."
        rows = x.shape[0]
        q = (x @ self.w[p + "q_proj.weight"].T).view(rows, self.heads, self.head_dim)
        k = (x @ self.w[p + "k_proj.weight"].T).view(rows, self.kv_heads, self.head_dim)
        v = (x @ self.w[p + "v_proj.weight"].T).view(rows, self.kv_heads, self.head_dim)
        q = self.rotate(self.rms(q, self.w[p + "q_norm.weight"]), positions)
        k = self.rotate(self.rms(k, self.w[p + "k_norm.weight"]), positions)
        committed = ctx_keys.shape[0]
        first = max(0, committed - self.window) if layer in self.local else 0
        keys = torch.cat([ctx_keys[first:], k], dim=0)
        values = torch.cat([ctx_values[first:], v], dim=0)
        group = self.heads // self.kv_heads
        out = torch.zeros(rows, self.heads, self.head_dim)
        for head in range(self.heads):
            shared = head // group  # key/value head j serves query heads j*group ..
            scores = q[:, head] @ keys[:, shared].T * self.head_dim**-0.5
            out[:, head] = scores.softmax(-1) @ values[:, shared]
        return out.reshape(rows, -1) @ self.w[p + "o_proj.weight"].T

    def mlp(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        p = f"layers.{layer}.mlp."
        gate = x @ self.w[p + "gate_proj.weight"].T
        up = x @ self.w[p + "up_proj.weight"].T
        return (gate * torch.sigmoid(gate) * up) @ self.w[p + "down_proj.weight"].T

    def run_block(self, anchor_embedding: torch.Tensor, context) -> torch.Tensor:
        """Final hidden states of the block's rows, `[block, hidden]`."""
        committed = context[0][0].shape[0]
        positions = committed + torch.arange(self.block)
        x = self.w["mask_embed"].repeat(self.block, 1)
        x[0] = anchor_embedding
        for layer in range(self.layers):
            p = f"layers.{layer}."
            normed = self.rms(x, self.w[p + "input_layernorm.weight"])
            x = x + self.attention(layer, normed, positions, *context[layer])
            x = x + self.mlp(layer, self.rms(x, self.w[p + "post_attention_layernorm.weight"]))
        return self.rms(x, self.w["norm.weight"])

    def chain(self, hidden, base, f_rnn, anchor_token, unembed, chosen=None):
        """The sequential head over the mask rows. `chosen` replaces the
        argmax with given tokens. Returns `(tokens, logits [rows, vocab])`."""
        w = self.w
        state = torch.zeros(self.rank)
        previous = anchor_token
        tokens, rows = [], []
        for step in range(hidden.shape[0]):
            inputs = torch.cat([state, w["markov_head.markov_w1.weight"][previous], hidden[step], f_rnn])
            raw = w["markov_head.rnn_proj.weight"] @ inputs + w["markov_head.rnn_proj.bias"]
            gate_raw, candidate_raw, out_raw = raw.chunk(3)
            gate = torch.sigmoid(gate_raw)
            # The latent comes from the same `raw`, so from the old state.
            latent = torch.tanh(out_raw)
            state = gate * state + (1.0 - gate) * torch.tanh(candidate_raw)
            correction = w["markov_head.hidden_out.weight"] @ latent + w["markov_head.hidden_out.bias"]
            logits = self.cap * torch.tanh((base[step] + unembed @ correction) / self.cap)
            logits = logits[: self.vocab]
            previous = int(logits.argmax()) if chosen is None else chosen[step]
            tokens.append(previous)
            rows.append(logits)
        return tokens, torch.stack(rows)


def check_transcription(path: Path, w, taps, anchor_embedding, unembed, greedy, driven) -> None:
    spec = importlib.util.spec_from_file_location("drafter_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = dict(MTP_CONFIG)
    cfg["final_logit_softcapping"] = CONFIG["text_config"]["final_logit_softcapping"]
    cfg["unpadded_vocab_size"] = CONFIG["text_config"]["unpadded_vocab_size"]
    with torch.no_grad():
        tokens, logits = module.draft(w, cfg, taps, anchor_embedding, ANCHOR_TOKEN, unembed)
        picks = iter(DRIVEN_TOKENS)
        _, driven_logits = module.draft(
            w, cfg, taps, anchor_embedding, ANCHOR_TOKEN, unembed, pick=lambda _row: next(picks)
        )
    errors = {
        "greedy": (logits - greedy[1]).abs().max().item(),
        "driven": (driven_logits - driven[1]).abs().max().item(),
    }
    print("against the transcription:", json.dumps(errors))
    if tokens != greedy[0] or max(errors.values()) > 1e-4:
        raise SystemExit("this reference and the transcription disagree")


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(SEED)
    w = build_weights(generator)
    c, text = MTP_CONFIG, CONFIG["text_config"]
    hidden = c["hidden_size"]

    # The taps differ in size, as the main model's partial states and its
    # final-norm output do, so the fusion weights matter.
    taps = torch.randn(COMMITTED, len(c["target_layer_ids"]), hidden, generator=generator)
    taps = taps * torch.tensor([1.0, 2.0, 0.5, 1.5])[None, :, None]
    unembed = bf16(torch.randn(text["vocab_size"], hidden, generator=generator))
    anchor_embedding = torch.randn(hidden, generator=generator)

    model = Reference(w)
    with torch.no_grad():
        positions = torch.arange(COMMITTED)
        fused = [model.fuse(taps, layer, w[f"layers.{layer}.dflare_norm.weight"]) for layer in range(model.layers)]
        fused_head = model.fuse(taps, model.layers, w["markov_head.dflare_norm.weight"])
        context = [model.context(layer, fused[layer], positions) for layer in range(model.layers)]
        block_hidden = model.run_block(anchor_embedding, context)
        base = block_hidden[1:] @ unembed.T
        f_rnn = fused_head[-1]
        greedy = model.chain(block_hidden[1:], base, f_rnn, ANCHOR_TOKEN, unembed)
        driven = model.chain(block_hidden[1:], base, f_rnn, ANCHOR_TOKEN, unembed, chosen=DRIVEN_TOKENS)
        # What the block computes when a local layer wrongly sees everything.
        model.window = COMMITTED
        unwindowed = model.run_block(anchor_embedding, context)
        model.window = c["sliding_window_size"]

    for tensor in (block_hidden, greedy[1], driven[1]):
        if not torch.isfinite(tensor).all():
            raise SystemExit("non-finite reference values")
    # The fixture must exercise what it claims to.
    stats = {
        "max_abs_base_logit": base.abs().max().item(),
        "max_abs_logit": greedy[1].abs().max().item(),
        "head_correction": (greedy[1] - model.cap * torch.tanh(base[:, : model.vocab] / model.cap)).abs().max().item(),
        "window_effect": (unwindowed - block_hidden).abs().max().item(),
        "greedy_tokens": greedy[0],
        "top_two_gap": min((row.topk(2).values[0] - row.topk(2).values[1]).item() for row in greedy[1]),
    }
    print(json.dumps(stats))
    if stats["max_abs_base_logit"] < 15.0:
        raise SystemExit("the soft cap hardly acts")
    if stats["head_correction"] < 1.0:
        raise SystemExit("the head hardly corrects the base logits")
    if stats["window_effect"] < 1e-2:
        raise SystemExit("the window of the local layers does not matter")
    if len(set(greedy[0])) < 4:
        raise SystemExit("the greedy chain repeats itself")
    if stats["top_two_gap"] < 0.1:
        raise SystemExit("a greedy choice is too close to call")
    if any(a == b for a, b in zip(DRIVEN_TOKENS, greedy[0])) or ANCHOR_TOKEN == 0:
        raise SystemExit("the driven chain must differ from the greedy one at every step")

    write_safetensors(out / "mtp.safetensors", {name: ("BF16", tensor) for name, tensor in w.items()})
    (out / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    golden = {
        "taps": ("F32", taps),
        "unembed": ("F32", unembed),
        "anchor_embedding": ("F32", anchor_embedding),
        "anchor_token": ("I64", torch.tensor([ANCHOR_TOKEN])),
        "fused.head": ("F32", fused_head),
        "block_hidden": ("F32", block_hidden),
        "base_logits": ("F32", base),
        "greedy.tokens": ("I64", torch.tensor(greedy[0])),
        "greedy.logits": ("F32", greedy[1]),
        "driven.tokens": ("I64", torch.tensor(driven[0])),
        "driven.logits": ("F32", driven[1]),
    }
    for layer in range(model.layers):
        golden[f"fused.layer.{layer}"] = ("F32", fused[layer])
        golden[f"context.{layer}.keys"] = ("F32", context[layer][0])
        golden[f"context.{layer}.values"] = ("F32", context[layer][1])
    write_safetensors(out / "golden.safetensors", golden)
    print("wrote", out, "tensors:", len(w), "golden:", len(golden))

    if "--check-transcription" in sys.argv:
        path = Path(sys.argv[sys.argv.index("--check-transcription") + 1])
        check_transcription(path, w, taps, anchor_embedding, unembed, greedy, driven)


if __name__ == "__main__":
    main()
