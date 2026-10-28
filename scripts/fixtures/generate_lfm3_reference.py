#!/usr/bin/env python3
"""Generate the LFM3-1.7B chat reference fixture from the published GGUF.

An independent F32 forward pass of the LFM3 tri-hybrid trunk, written from HF
`modeling_lfm3.py` (LiquidAI/LFM3-1.7B @ 40ebe0aa) and llama.cpp
`src/models/lfm3.cpp` (#31252), run on the dequantized tensors of the real
checkpoint:

    operator_norm -> (gated ShortConv of width 7: in_proj -> B, C, x;
    y = C * causal_conv(B * x) | GQA attention: per-head q/k RMSNorm, NEOX RoPE,
    a 256-token causal window on sliding layers, the head output scaled by
    sigmoid(attn_gate(x))) -> post_attention_norm -> residual -> ffn_norm ->
    SwiGLU -> post_ffw_norm -> residual; final `output_norm`; tied LM head.

Global layers rotate at base 1e6 and only their first half of frequency pairs
(the GGUF's `rope_freqs` carries 1e30 on the rest, which HF writes as an
inverse frequency of 0); sliding layers rotate fully at base 1e4. RoPE angles
are computed in F64. Norm gains are plain `weight * x`.

Prompts are rendered with the checkpoint's own `chat_template.jinja` (HF's
sandboxed settings, `{% generation %}` passed through) and tokenized without
adding special tokens. Each prompt records its ids and, for a teacher-forced
continuation (the reference's own greedy tokens, stopping at `<|im_end|>`),
each step's next token, top-2 margin and top-8 logits. `TEMPLATE_CASES` are
prompt-id-only cases that pin izwi's prompt builder to the template.

Before writing, the script checks itself: a two-span prefill must reproduce
the one-shot prefill logits.

Usage (numpy, torch, gguf, tokenizers, jinja2; ~9 GB RSS, a few minutes):

    python3 -I scripts/fixtures/generate_lfm3_reference.py \
        "$HOME/Library/Application Support/izwi/models" \
        crates/izwi-core/tests/fixtures/lfm3_chat_reference.json

The Rust test is `lfm2::chat::lfm3_reference_tests` (ignored; it needs the GGUF).
"""
import json
import math
import sys

import jinja2.ext
import jinja2.nodes
import jinja2.sandbox
import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize
from tokenizers import Tokenizer

torch.set_grad_enabled(False)

FOLDER = "LFM3-1.7B-GGUF"
GGUF = "LFM3-1.7B-QAD-Q4_0.gguf"
IM_END = 124900
END_OF_TEXT = 124895
TOP_K = 8

# Tool definitions keep their keys sorted at every level, so serde_json's
# sorted maps and Python's insertion order serialize identically.
WEATHER_TOOL = {
    "function": {
        "description": "Get the current weather for a city.",
        "name": "get_weather",
        "parameters": {
            "properties": {"city": {"description": "City name.", "type": "string"}},
            "required": ["city"],
            "type": "object",
        },
    },
    "type": "function",
}
TIME_TOOL = {
    "function": {"description": "Current UTC time.", "name": "get_time", "parameters": {}},
    "type": "function",
}


def long_document():
    """About 800 tokens of distinct facts, so the question at the end needs
    the global layers to reach past the 256-token sliding window."""
    colors = ["red", "blue", "green", "amber", "violet", "silver", "orange", "teal"]
    lines = []
    for i in range(1, 41):
        lines.append(
            f"Record {i}: the {colors[i % len(colors)]} crate in aisle {i * 3} holds "
            f"{i * 7} copper fittings and ships on day {i % 9 + 1}."
        )
    return "\n".join(lines)


# name -> (messages, tools, enable_thinking, max_steps); a message is
# (role, content).
PROMPTS = {
    "capital_thinking": (
        [("user", "What is the capital of France? Answer in one sentence.")],
        None,
        True,
        64,
    ),
    "capital_direct": (
        [("user", "What is the capital of France? Answer in one sentence.")],
        None,
        False,
        32,
    ),
    "long_context": (
        [
            (
                "user",
                long_document()
                + "\n\nHow many copper fittings does the crate of Record 2 hold? "
                "Answer with the number only.",
            )
        ],
        None,
        False,
        16,
    ),
    "tool_call": (
        [("system", "Answer briefly."), ("user", "What's the weather in Paris right now?")],
        [WEATHER_TOOL, TIME_TOOL],
        False,
        48,
    ),
}

WEATHER_CALL = (
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
    "</function>\n</tool_call>"
)

# Prompt-id-only cases: name -> (messages, tools, template kwargs).
TEMPLATE_CASES = {
    "untrimmed_user": ([("user", "  Hi there \n")], None, {}),
    "code_block": ([("user", "```python\nprint('x')\n```\n")], None, {}),
    "thinking_off": ([("user", "Hi")], None, {"enable_thinking": False}),
    "past_reasoning_stripped": (
        [
            ("user", "Q"),
            ("assistant", "<think>plan</think>\n\n  Answer one.  "),
            ("user", "Next"),
        ],
        None,
        {},
    ),
    "past_reasoning_preserved": (
        [
            ("user", "Q"),
            ("assistant", "<think>plan</think>\n\nAnswer one."),
            ("user", "Next"),
        ],
        None,
        {"preserve_thinking": True},
    ),
    "empty_past_assistant": ([("user", "Hi"), ("assistant", ""), ("user", "Again")], None, {}),
    "last_assistant_keeps_reasoning": (
        [("user", "Q"), ("assistant", "<think>r</think>\n\nA")],
        None,
        {},
    ),
    "second_system_message": (
        [("system", "S1"), ("user", "u"), ("system", "S2"), ("user", "v")],
        None,
        {},
    ),
    "tools_without_system": ([("user", "Weather in Paris?")], [WEATHER_TOOL], {}),
    "tools_with_system": (
        [("system", "Answer briefly."), ("user", "Weather in Paris?")],
        [WEATHER_TOOL, TIME_TOOL],
        {},
    ),
    "tool_round_trip": (
        [
            ("system", "Answer briefly."),
            ("user", "Weather in Paris?"),
            ("assistant", WEATHER_CALL),
            ("tool", '{"temperature_c": 21}'),
        ],
        [WEATHER_TOOL],
        {},
    ),
    "tool_round_trip_with_text": (
        [
            ("user", "Weather in Paris?"),
            ("assistant", "Let me check." + WEATHER_CALL),
            ("tool", '{"temperature_c": 21}'),
            ("assistant", "It is 21 C."),
            ("user", "Thanks"),
        ],
        [WEATHER_TOOL],
        {},
    ),
}


class Generation(jinja2.ext.Extension):
    """HF's `{% generation %}` block marks assistant spans for training masks;
    rendering passes its body through."""

    tags = {"generation"}

    def parse(self, parser):
        next(parser.stream)
        return parser.parse_statements(["name:endgeneration"], drop_needle=True)


def template_renderer(path):
    """The checkpoint's chat template, rendered like HF `apply_chat_template`."""
    with open(path, encoding="utf-8") as f:
        source = f.read()
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=["jinja2.ext.loopcontrols", Generation],
    )
    env.filters["tojson"] = lambda value, indent=None, **_: json.dumps(
        value, ensure_ascii=False, indent=indent
    )

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    env.globals["raise_exception"] = raise_exception
    template = env.from_string(source)

    def render_raw(messages, tools=None, **kwargs):
        return template.render(
            messages=messages,
            tools=tools,
            bos_token="<|startoftext|>",
            add_generation_prompt=True,
            **kwargs,
        )

    def render(messages, tools=None, **kwargs):
        return render_raw(
            [{"role": role, "content": content} for role, content in messages], tools, **kwargs
        )

    return render, render_raw


class Model:
    def __init__(self, path):
        r = GGUFReader(path)

        def meta(key):
            f = r.fields[key]
            v = [f.parts[i] for i in f.data]
            if len(v) > 1:
                return [x.tolist()[0] for x in v]
            return v[0].tolist()[0]

        assert bytes(r.fields["general.architecture"].parts[-1]).decode() == "lfm3"
        self.n_layer = meta("lfm3.block_count")
        self.hidden = meta("lfm3.embedding_length")
        self.n_head = meta("lfm3.attention.head_count")
        self.kv = meta("lfm3.attention.head_count_kv")
        self.swa = [bool(x) for x in meta("lfm3.attention.sliding_window_pattern")]
        self.window = meta("lfm3.attention.sliding_window")
        self.eps = float(meta("lfm3.attention.layer_norm_rms_epsilon"))
        self.L = meta("lfm3.shortconv.l_cache")
        self.hd = self.hidden // self.n_head
        self.T = {}
        for t in r.tensors:
            a = dequantize(t.data, t.tensor_type).astype(np.float32)
            shape = [int(x) for x in reversed(t.shape.tolist())]
            self.T[t.name] = torch.from_numpy(a.reshape(shape).copy())
        pairs = torch.arange(0, self.hd, 2, dtype=torch.float64) / self.hd
        factors = self.T["rope_freqs.weight"].double()
        assert all(f == 1.0 or f >= 1e29 for f in factors.tolist())
        theta = float(meta("lfm3.rope.freq_base"))
        theta_swa = float(meta("lfm3.rope.freq_base_swa"))
        # HF zero-tail partial RoPE: unrotated pairs have an inverse frequency of 0.
        self.inv_full = torch.where(factors == 1.0, 1.0 / theta**pairs, torch.zeros_like(pairs))
        self.inv_swa = 1.0 / theta_swa**pairs

    def rms(self, x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * w

    def rope(self, x, pos, inv):
        ang = torch.tensor(pos, dtype=torch.float64)[:, None] * inv[None]
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
                inv = self.inv_swa if self.swa[i] else self.inv_full
                q = (h @ T[p + "attn_q.weight"].T).view(t, nh, hd)
                k = (h @ T[p + "attn_k.weight"].T).view(t, kvh, hd)
                v = (h @ T[p + "attn_v.weight"].T).view(t, kvh, hd)
                q = self.rope(self.rms(q, T[p + "attn_q_norm.weight"]).transpose(0, 1), pos, inv)
                k = self.rope(self.rms(k, T[p + "attn_k_norm.weight"]).transpose(0, 1), pos, inv)
                v = v.transpose(0, 1)
                st["k"][i] = k if st["k"][i] is None else torch.cat([st["k"][i], k], 1)
                st["v"][i] = v if st["v"][i] is None else torch.cat([st["v"][i], v], 1)
                K = st["k"][i].repeat_interleave(nh // kvh, 0)
                V = st["v"][i].repeat_interleave(nh // kvh, 0)
                s = (q @ K.transpose(1, 2)) / math.sqrt(hd)
                key_pos = torch.arange(K.shape[1])[None, :]
                query_pos = torch.arange(t)[:, None] + st["pos"]
                mask = key_pos > query_pos
                if self.swa[i]:
                    mask = mask | (query_pos - key_pos >= self.window)
                s = s.masked_fill(mask, float("-inf")).softmax(-1)
                o = (s @ V).transpose(0, 1)  # [t, heads, hd]
                gate = torch.sigmoid(h @ T[p + "attn_gate.weight"].T)  # [t, heads]
                o = (o * gate[..., None]).reshape(t, H) @ T[p + "attn_output.weight"].T
            else:
                bcx = (h @ T[p + "shortconv.in_proj.weight"].T).T
                b, c, xx = bcx[:H], bcx[H : 2 * H], bcx[2 * H :]
                bx = torch.cat([st["conv"][i], b * xx], 1)
                w = T[p + "shortconv.conv.weight"]
                conv = sum(bx[:, k : k + t] * w[:, k : k + 1] for k in range(self.L))
                st["conv"][i] = bx[:, -(self.L - 1) :].clone()
                o = (c * conv).T @ T[p + "shortconv.out_proj.weight"].T
            x = x + self.rms(o, T[p + "post_attention_norm.weight"])
            h = self.rms(x, T[p + "ffn_norm.weight"])
            g = h @ T[p + "ffn_gate.weight"].T
            u = h @ T[p + "ffn_up.weight"].T
            f = (torch.nn.functional.silu(g) * u) @ T[p + "ffn_down.weight"].T
            x = x + self.rms(f, T[p + "post_ffw_norm.weight"])
        st["pos"] += t
        x = self.rms(x[-1], T["output_norm.weight"])
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
    render, render_raw = template_renderer(f"{root}/{FOLDER}/chat_template.jinja")
    tok = Tokenizer.from_file(f"{root}/{FOLDER}/tokenizer.json")

    # The gateway writes a turn's calls into its content right after the
    # text; that must render exactly like the template's own `tool_calls`,
    # for the current tool loop and for a past turn.
    call = {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
    for text in ("", "Let me check."):
        for past in (False, True):
            user = {"role": "user", "content": "Weather in Paris?"}
            tail = [{"role": "user", "content": "Thanks"}] if past else []
            structured = [user, {"role": "assistant", "content": text, "tool_calls": [call]}] + tail
            embedded = [user, {"role": "assistant", "content": text + WEATHER_CALL}] + tail
            assert render_raw(structured, [WEATHER_TOOL]) == render_raw(embedded, [WEATHER_TOOL]), (
                f"embedded calls drift from tool_calls (text={text!r}, past={past})"
            )

    def template_case(messages, tools, kwargs):
        text = render(messages, tools, **kwargs)
        case = {
            "messages": [[role, content] for role, content in messages],
            "prompt_ids": tok.encode(text, add_special_tokens=False).ids,
        }
        if tools:
            case["tools"] = tools
        if kwargs:
            case["template_kwargs"] = kwargs
        return case

    fixture = {
        "generator": "scripts/fixtures/generate_lfm3_reference.py",
        "folder": FOLDER,
        "gguf": GGUF,
        "template_cases": {
            name: template_case(*case) for name, case in TEMPLATE_CASES.items()
        },
        "prompts": {},
    }
    model = Model(f"{root}/{FOLDER}/{GGUF}")
    for name, (messages, tools, thinking, max_steps) in PROMPTS.items():
        text = render(messages, tools, enable_thinking=thinking)
        ids = tok.encode(text, add_special_tokens=False).ids
        st = model.new_state()
        logits = model.forward(ids, st)
        split_at = len(ids) // 2
        split = model.new_state()
        model.forward(ids[:split_at], split)
        chunked = model.forward(ids[split_at:], split)
        drift = float((chunked - logits).abs().max())
        assert drift < 1e-3, f"{name}: chunked prefill drifts by {drift}"
        steps = []
        for _ in range(max_steps):
            nxt = int(logits.argmax())
            steps.append(step_record(logits, nxt))
            if nxt in (IM_END, END_OF_TEXT):
                break
            logits = model.forward([nxt], st)
        greedy = tok.decode(
            [s["next"] for s in steps if s["next"] not in (IM_END, END_OF_TEXT)],
            skip_special_tokens=False,
        )
        fixture["prompts"][name] = {
            "messages": [[role, content] for role, content in messages],
            "tools": tools or [],
            "enable_thinking": thinking,
            "prompt_ids": ids,
            "steps": steps,
            "greedy_text": greedy,
        }
        print(f"{name}: {len(ids)} prompt ids, {len(steps)} steps, chunk drift {drift:.2e}")
        print(f"  {greedy!r}")
    with open(out_path, "w") as f:
        json.dump(fixture, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
