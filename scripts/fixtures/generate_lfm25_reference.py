#!/usr/bin/env python3
"""Generate the LFM2.5-1.2B chat reference fixture from the published GGUFs.

An independent F32 forward pass of the LFM2 hybrid (ShortConv + GQA attention)
backbone, written from HF transformers `modeling_lfm2.py` and llama.cpp
`src/models/lfm2.cpp`, run on the dequantized tensors of the real checkpoint:

    operator_norm -> (GQA attention with per-head q/k RMSNorm before NEOX RoPE
    | ShortConv: in_proj -> B, C, x; y = C * causal_conv(B * x)) -> residual
    -> ffn_norm -> SwiGLU -> residual; final norm `token_embd_norm`;
    tied LM head (`token_embd`). No `+1` on any norm gain.

RoPE tables are computed in F64 on the host. Prompts are rendered exactly like
the checkpoint's chat template (one BOS, `<|im_start|>role\\n...<|im_end|>\\n`,
past assistant turns stripped up to `</think>`) and tokenized without adding
special tokens.

For every prompt the script records the prompt ids and, for a teacher-forced
continuation (the reference's own greedy tokens, stopping at EOS), each step's
next token, its top-2 margin and the top-8 logits. The izwi test feeds the same
tokens, so one early disagreement cannot cascade, and asserts:
  * prompt ids are identical (template + tokenizer),
  * the top-8 reference logits match within a tolerance,
  * izwi's argmax equals the reference token wherever the reference margin is
    not a near tie.

Prompts are rendered with the checkpoint's own GGUF chat template (jinja2,
HF's sandboxed environment settings) and must equal the hand-written
`render` below. `TEMPLATE_CASES` add prompt-id-only cases (whitespace,
empty turns, think-only past turns, a mid-conversation system message) that
pin izwi's prompt builder to the template.

Before writing, the script checks itself: a two-chunk prefill must reproduce
the one-shot prefill logits.

Usage (needs numpy, torch, gguf, tokenizers, jinja2; ~6 GB RSS, ~1 min per
model):

    python3 -I scripts/fixtures/generate_lfm25_reference.py \
        "$HOME/Library/Application Support/izwi/models" \
        crates/izwi-core/tests/fixtures/lfm25_chat_reference.json

The Rust test is `lfm2::chat::reference_tests` (ignored; it needs the GGUFs).
"""
import json
import math
import sys

import jinja2.sandbox
import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize
from tokenizers import Tokenizer

torch.set_grad_enabled(False)

MODELS = {
    "instruct": ("LFM2.5-1.2B-Instruct-GGUF", "LFM2.5-1.2B-Instruct-Q4_K_M.gguf", 48),
    "thinking": ("LFM2.5-1.2B-Thinking-GGUF", "LFM2.5-1.2B-Thinking-Q4_K_M.gguf", 64),
}

# Each prompt is a list of (role, content); contents avoid leading/trailing
# whitespace so the fixture does not depend on trimming behaviour.
PROMPTS = {
    "capital": [("user", "What is the capital of France? Answer in one sentence.")],
    "multi_turn": [
        ("system", "You are a concise assistant. Prefer short, factual answers."),
        ("user", "Name the largest planet in our solar system."),
        (
            "assistant",
            "<think>The user wants the largest planet.</think>\n\nJupiter is the largest planet in our solar system.",
        ),
        (
            "user",
            "How many Earths would fit across its diameter? Give a number and one sentence of explanation.",
        ),
    ],
}

# Prompt-id-only cases: (role, content) lists rendered by the template.
TEMPLATE_CASES = {
    "untrimmed_user": [("user", "  Hi there \n")],
    "code_block": [("user", "```python\nprint('x')\n```\n")],
    "leading_newlines": [("user", "\n\nleading newlines")],
    "empty_past_assistant": [("user", "Hi"), ("assistant", ""), ("user", "Again")],
    "think_only_past_assistant": [
        ("user", "Q"),
        ("assistant", "<think>only thinking</think>"),
        ("user", "Next"),
    ],
    "unclosed_think_past_assistant": [
        ("user", "Q"),
        ("assistant", " <think>never closed "),
        ("user", "Next"),
    ],
    "empty_system": [("system", ""), ("user", "Hi")],
    "last_assistant_keeps_thinking": [
        ("user", "Q"),
        ("assistant", "<think>r</think>\n\nA"),
    ],
    "second_system_message": [
        ("system", "S1"),
        ("user", "u"),
        ("system", "S2"),
        ("user", "v"),
    ],
}

TOP_K = 8


def template_renderer(gguf_path):
    """The checkpoint's chat template, rendered like HF `apply_chat_template`."""
    reader = GGUFReader(gguf_path)
    field = reader.fields["tokenizer.chat_template"]
    source = bytes(field.parts[field.data[0]]).decode("utf-8")
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
    )
    env.filters["tojson"] = lambda value, **_: json.dumps(value, ensure_ascii=False)
    template = env.from_string(source)

    def render_template(messages):
        return template.render(
            messages=[{"role": role, "content": content} for role, content in messages],
            bos_token="<|startoftext|>",
            add_generation_prompt=True,
        )

    return render_template


def render(messages):
    last_assistant = max(
        (i for i, (role, _) in enumerate(messages) if role == "assistant"), default=-1
    )
    out = "<|startoftext|>"
    for i, (role, content) in enumerate(messages):
        if role == "assistant" and i != last_assistant and "</think>" in content:
            content = content.split("</think>")[-1].strip()
        out += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return out + "<|im_start|>assistant\n"


class Model:
    def __init__(self, path):
        r = GGUFReader(path)

        def meta(key):
            f = r.fields[key]
            v = [f.parts[i] for i in f.data]
            if len(v) > 1:
                return [x.tolist()[0] for x in v]
            return v[0].tolist()[0]

        self.n_layer = meta("lfm2.block_count")
        self.hidden = meta("lfm2.embedding_length")
        self.n_head = meta("lfm2.attention.head_count")
        self.kv = meta("lfm2.attention.head_count_kv")
        self.eps = float(meta("lfm2.attention.layer_norm_rms_epsilon"))
        theta = float(meta("lfm2.rope.freq_base"))
        self.L = meta("lfm2.shortconv.l_cache")
        self.hd = self.hidden // self.n_head
        self.inv = 1.0 / (theta ** (torch.arange(0, self.hd, 2, dtype=torch.float64) / self.hd))
        self.T = {}
        for t in r.tensors:
            a = dequantize(t.data, t.tensor_type).astype(np.float32)
            shape = [int(x) for x in reversed(t.shape.tolist())]
            self.T[t.name] = torch.from_numpy(a.reshape(shape).copy())

    def rms(self, x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * w

    def rope(self, x, pos):
        ang = torch.tensor(pos, dtype=torch.float64)[:, None] * self.inv[None]
        cos = torch.cat([ang.cos(), ang.cos()], -1).float()
        sin = torch.cat([ang.sin(), ang.sin()], -1).float()
        half = self.hd // 2
        x1, x2 = x[..., :half], x[..., half:]
        return x * cos + torch.cat([-x2, x1], -1) * sin

    def new_state(self):
        return {
            "k": [None] * self.n_layer,
            "v": [None] * self.n_layer,
            "conv": [torch.zeros(self.hidden, self.L - 1) for _ in range(self.n_layer)],
            "pos": 0,
        }

    def forward(self, ids, st):
        T, H, hd, nh = self.T, self.hidden, self.hd, self.n_head
        x = T["token_embd.weight"][torch.tensor(ids)]
        t = len(ids)
        pos = list(range(st["pos"], st["pos"] + t))
        for i in range(self.n_layer):
            p = f"blk.{i}."
            h = self.rms(x, T[p + "attn_norm.weight"])
            if self.kv[i] > 0:
                kvh = self.kv[i]
                q = (h @ T[p + "attn_q.weight"].T).view(t, nh, hd)
                k = (h @ T[p + "attn_k.weight"].T).view(t, kvh, hd)
                v = (h @ T[p + "attn_v.weight"].T).view(t, kvh, hd)
                q = self.rope(self.rms(q, T[p + "attn_q_norm.weight"]).transpose(0, 1), pos)
                k = self.rope(self.rms(k, T[p + "attn_k_norm.weight"]).transpose(0, 1), pos)
                v = v.transpose(0, 1)
                st["k"][i] = k if st["k"][i] is None else torch.cat([st["k"][i], k], 1)
                st["v"][i] = v if st["v"][i] is None else torch.cat([st["v"][i], v], 1)
                K = st["k"][i].repeat_interleave(nh // kvh, 0)
                V = st["v"][i].repeat_interleave(nh // kvh, 0)
                s = (q @ K.transpose(1, 2)) / math.sqrt(hd)
                total = K.shape[1]
                mask = torch.arange(total)[None, :] > (torch.arange(t)[:, None] + st["pos"])
                s = s.masked_fill(mask, float("-inf")).softmax(-1)
                o = (s @ V).transpose(0, 1).reshape(t, H) @ T[p + "attn_output.weight"].T
            else:
                bcx = (h @ T[p + "shortconv.in_proj.weight"].T).T
                b, c, xx = bcx[:H], bcx[H : 2 * H], bcx[2 * H :]
                bx = torch.cat([st["conv"][i], b * xx], 1)
                w = T[p + "shortconv.conv.weight"]
                conv = sum(bx[:, k : k + t] * w[:, k : k + 1] for k in range(self.L))
                st["conv"][i] = bx[:, -(self.L - 1) :].clone()
                o = (c * conv).T @ T[p + "shortconv.out_proj.weight"].T
            x = x + o
            h = self.rms(x, T[p + "ffn_norm.weight"])
            g = h @ T[p + "ffn_gate.weight"].T
            u = h @ T[p + "ffn_up.weight"].T
            x = x + (torch.nn.functional.silu(g) * u) @ T[p + "ffn_down.weight"].T
        st["pos"] += t
        x = self.rms(x[-1], T["token_embd_norm.weight"])
        return x @ T["token_embd.weight"].T


def step_record(logits, next_id):
    top = torch.topk(logits, TOP_K)
    values = [float(v) for v in top.values]
    return {
        "next": int(next_id),
        "margin": values[0] - values[1],
        "top": [[int(i), round(v, 5)] for i, v in zip(top.indices, values)],
    }


def main():
    root, out_path = sys.argv[1], sys.argv[2]
    fixture = {"generator": "scripts/fixtures/generate_lfm25_reference.py", "models": {}}
    for key, (folder, gguf, max_steps) in MODELS.items():
        render_template = template_renderer(f"{root}/{folder}/{gguf}")
        tok = Tokenizer.from_file(f"{root}/{folder}/tokenizer.json")
        template_cases = {}
        for name, messages in TEMPLATE_CASES.items():
            text = render_template(messages)
            template_cases[name] = {
                "messages": [[role, content] for role, content in messages],
                "prompt_ids": tok.encode(text, add_special_tokens=False).ids,
            }
        model = Model(f"{root}/{folder}/{gguf}")
        prompts = {}
        for name, messages in PROMPTS.items():
            text = render(messages)
            assert text == render_template(messages), f"{key}/{name}: render drifts from the template"
            ids = tok.encode(text, add_special_tokens=False).ids
            st = model.new_state()
            logits = model.forward(ids, st)
            split = model.new_state()
            model.forward(ids[:5], split)
            chunked = model.forward(ids[5:], split)
            drift = float((chunked - logits).abs().max())
            assert drift < 1e-3, f"{key}/{name}: chunked prefill drifts by {drift}"
            steps = []
            for _ in range(max_steps):
                nxt = int(logits.argmax())
                steps.append(step_record(logits, nxt))
                if nxt in (7, 2):
                    break
                logits = model.forward([nxt], st)
            prompts[name] = {
                "messages": [[role, content] for role, content in messages],
                "prompt_ids": ids,
                "steps": steps,
                "greedy_text": tok.decode(
                    [s["next"] for s in steps if s["next"] not in (7, 2)],
                    skip_special_tokens=False,
                ),
            }
            print(f"{key}/{name}: {len(ids)} prompt ids, {len(steps)} steps, chunk drift {drift:.2e}")
        fixture["models"][key] = {
            "folder": folder,
            "gguf": gguf,
            "prompts": prompts,
            "template_cases": template_cases,
        }
        del model
    with open(out_path, "w") as f:
        json.dump(fixture, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
