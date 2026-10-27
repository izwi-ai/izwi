#!/usr/bin/env python3
"""Generate Maya1 reference fixtures with upstream's own code.

The reference is maya-research/maya1 at HF revision b1b9b38c (Apache-2.0)
run through transformers 4.57.1 / torch 2.6.0 on the CPU, and SNAC 24 kHz
(hubertsiuzdak/snac_24khz at d73ad176, MIT) through the `snac` 1.2.1
package. The prompt is the card's Transformers framing
(`SOH BOS <description="..."> text EOT EOH SOA SOS`, no added special
tokens).

Each part writes its own JSON file into OUT_DIR, so the parts that need only
the tokenizer and SNAC run before the 6.6 GB checkpoint is on disk:

  tokenizer  prompt ids for a set of descriptions and texts (`prompts.json`)
  sampler    Transformers' logits processors (repetition penalty, minimum
             length, temperature, top-k 50, top-p) on head-only logits
             (`sampler.json`)
  snac       SNAC decoding of fixed codes with izwi's noise: the whole
             utterance and izwi's incremental windows (`snac.json`)
  lm         the decoder: prefill and teacher-forced head logits of a
             4-layer truncation in F32, and of the full model in BF16, on
             tokens the full model sampled itself (`lm.json`)

SNAC adds `randn * conv1x1(x)` noise in each decoder block. To compare the
decoder exactly, the noise is regenerated here from izwi's portable
SplitMix64 streams (`maya1/noise.rs`, mirrored below) and patched into
`snac.layers.NoiseBlock`. The LM's sampling cannot be shared (torch draws
differently), so the decoder is checked under teacher forcing.

The reference tokenizer needs the repository's `tokenizer_config.json` and
`special_tokens_map.json` beside `tokenizer.json`: without them transformers
falls back to `LlamaTokenizerFast`'s legacy handling, which tokenizes the
prompt differently from upstream (izwi needs only `tokenizer.json`).

Usage (the `lm` part needs about 10 GB RSS and some minutes):

    python3 -I scripts/fixtures/generate_maya1_reference.py \\
        path/to/Maya1-3B crates/izwi-core/tests/fixtures/maya1 \\
        --parts tokenizer,sampler,snac,lm
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ID = "maya-research/maya1"
REVISION = "b1b9b38ca9decbfcfd6c98ea93dd502bfd0f9572"
SNAC_REVISION = "d73ad176a12188fcf4f360ba3bf2c2fbbe8f58ec"
SAMPLE_POINTS = 64
SEED = 20261027

SOH, EOH, SOA, SOS, EOS = 128259, 128260, 128261, 128257, 128258
BOS, TEXT_EOT = 128000, 128009
CODE_START, CODEBOOK = 128266, 4096
HEAD_FIRST, HEAD_ROWS = EOS, 156938 - EOS

PROMPTS = [
    (
        "Realistic female voice in the 30s age with a american accent. Normal pitch, warm "
        "timbre, conversational pacing, neutral tone delivery at med intensity, podcast "
        "domain, podcast_host role, neutral delivery.",
        "Hello! This is a test of the Maya one voice model, recorded on the twenty seventh.",
    ),
    (
        "Creative, dark_villain character. Male voice in their 40s with a british accent. "
        "Low pitch, gravelly timbre, slow pacing, angry tone at high intensity.",
        "The darkness isn't coming... <angry> it's already here! <laugh_harder>",
    ),
    (
        "casual random speaker, casual talk, male 40s.",
        "And you? <sigh> <whisper> I never knew.",
    ),
]

# ---------------------------------------------------------- izwi's RNG mirror
MASK = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15
CODEC_NOISE = 2


def mix(z):
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


def stream_seed(seed, stream):
    return mix(seed ^ mix((stream * GOLDEN) & MASK))


def open01(state, count):
    """`count` SplitMix64 uniforms in (0, 1) after `state`, vectorised."""
    with np.errstate(over="ignore"):
        k = np.arange(1, count + 1, dtype=np.uint64)
        z = np.uint64(state) + k * np.uint64(GOLDEN)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return ((z >> np.uint64(11)).astype(np.float64) + 0.5) * (1.0 / (1 << 53))


def normals_from(state, count):
    pairs = (count + 1) // 2
    u = open01(state, 2 * pairs)
    radius = np.sqrt(-2.0 * np.log(u[0::2]))
    angle = 2.0 * np.pi * u[1::2]
    out = np.empty(2 * pairs, dtype=np.float64)
    out[0::2] = radius * np.cos(angle)
    out[1::2] = radius * np.sin(angle)
    return out[:count].astype(np.float32)


def block_noise(seed, block, first, frames, steps_per_frame):
    """`maya1/noise.rs::block_noise`: one stream per (block, absolute frame)."""
    base = stream_seed(seed, CODEC_NOISE)
    out = []
    for frame in range(first, first + frames):
        key = ((block << 40) ^ (frame + 1)) & MASK
        out.append(normals_from(mix(base ^ mix(key)), steps_per_frame))
    return np.concatenate(out)


BLOCK_STEPS = [32, 256, 1024, 2048]
SAMPLES_PER_FRAME = 2048


# ------------------------------------------------------------------ helpers
def sample(values):
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    n = flat.size
    indices = sorted({int(i * (n - 1) / (SAMPLE_POINTS - 1)) for i in range(SAMPLE_POINTS)})
    return {
        "shape": list(np.shape(values)),
        "indices": indices,
        "values": [float(flat[i]) for i in indices],
        "mean": float(flat.mean()),
        "rms": float(np.sqrt((flat ** 2).mean())),
    }


def head_record(logits):
    """Top-8 head rows of a full-vocabulary logits row, plus sample points."""
    head = logits[HEAD_FIRST:HEAD_FIRST + HEAD_ROWS].to(torch.float32)
    values, rows = torch.topk(head, 8)
    return {
        "top": [[int(r), float(v)] for r, v in zip(rows.tolist(), values.tolist())],
        "sample": sample(head.numpy()),
    }


def write(out_dir, name, payload):
    path = out_dir / name
    path.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {path}")


def prompt_ids(tokenizer, description, text):
    body = tokenizer.encode(f'<description="{description}"> {text}', add_special_tokens=False)
    return [SOH, BOS] + body + [TEXT_EOT, EOH, SOA, SOS]


def readme_prompt_ids(tokenizer, description, text):
    """The card's own construction: control tokens decoded to text."""
    d = tokenizer.decode
    prompt = (
        d([SOH]) + tokenizer.bos_token + f'<description="{description}"> {text}' + d([TEXT_EOT])
        + d([EOH]) + d([SOA]) + d([SOS])
    )
    return tokenizer(prompt, add_special_tokens=False)["input_ids"]


# -------------------------------------------------------------------- parts
def part_tokenizer(model_dir, out_dir):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    cases = []
    for description, text in PROMPTS:
        ids = prompt_ids(tokenizer, description, text)
        readme = readme_prompt_ids(tokenizer, description, text)
        assert ids == readme, "segment encoding differs from the card's string construction"
        cases.append({
            "description": description,
            "text": text,
            "ids": ids,
            "text_tokens": len(tokenizer.encode(text, add_special_tokens=False)),
        })
    write(out_dir, "prompts.json", {"revision": REVISION, "cases": cases})


def part_sampler(out_dir):
    from transformers.generation.logits_process import (
        LogitsProcessorList, MinNewTokensLengthLogitsProcessor,
        RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper, TopKLogitsWarper,
        TopPLogitsWarper)

    rng = np.random.default_rng(SEED)
    cases = []
    for case, (generated, prompt_len) in enumerate([(10, 40), (64, 40)]):
        # Regenerable in Rust: SplitMix64 normals from state `SEED + case`,
        # scaled so tens of rows survive top-k and top-p.
        head = (normals_from(SEED + case, HEAD_ROWS) * np.float32(0.5)).astype(np.float32)
        head[0] = 2.5  # the stop row, likely enough to matter
        # Penalise some of the likeliest rows, positive and negative alike.
        likely = [int(r) for r in np.argsort(-head)[:12] if r >= 8][:6]
        history = likely + rng.integers(8, HEAD_ROWS, generated - len(likely)).tolist()
        vocab = 156960
        scores = torch.full((1, vocab), float("-inf"))
        scores[0, HEAD_FIRST:HEAD_FIRST + HEAD_ROWS] = torch.from_numpy(head)
        # The masked control rows (1..=7 of the head) take no part.
        scores[0, HEAD_FIRST + 1:HEAD_FIRST + 8] = float("-inf")
        input_ids = torch.tensor([[1] * prompt_len + [HEAD_FIRST + h for h in history]])
        processors = LogitsProcessorList([
            RepetitionPenaltyLogitsProcessor(1.1),
            MinNewTokensLengthLogitsProcessor(prompt_len, 28, [EOS]),
            TemperatureLogitsWarper(0.4),
            TopKLogitsWarper(50),
            TopPLogitsWarper(0.9),
        ])
        processed = processors(input_ids, scores)[0, HEAD_FIRST:HEAD_FIRST + HEAD_ROWS]
        live = torch.nonzero(torch.isfinite(processed)).flatten().tolist()
        cases.append({
            "logits_state": SEED + case,
            "history": history,
            "generated": generated,
            "live": [[int(r), float(processed[r])] for r in live],
        })
    write(out_dir, "sampler.json", {"cases": cases})


def snac_codes(seed, frames):
    """Band-respecting codes with some structure (not uniform noise)."""
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.integers(-40, 41, (frames, 7)), axis=0) % CODEBOOK
    return [int(c) for row in walk for c in row]


def unpack(tokens_or_codes):
    c = tokens_or_codes
    n = len(c) // 7
    l1 = [c[7 * i] for i in range(n)]
    l2 = [x for i in range(n) for x in (c[7 * i + 1], c[7 * i + 4])]
    l3 = [x for i in range(n) for x in (c[7 * i + 2], c[7 * i + 3], c[7 * i + 5], c[7 * i + 6])]
    return l1, l2, l3


class NoiseFeed:
    """Replace NoiseBlock's randn with izwi's per-frame streams."""

    def __init__(self, seed, first):
        self.seed, self.first, self.block = seed, first, 0

    def __call__(self, module, x):
        b, _, t = x.shape
        frames = t // BLOCK_STEPS[self.block]
        noise = block_noise(self.seed, self.block, self.first, frames, BLOCK_STEPS[self.block])
        self.block += 1
        noise = torch.from_numpy(noise).reshape(b, 1, t).to(x.dtype)
        return x + noise * module.linear(x)


def snac_decode(model, codes, seed, first):
    import snac.layers as layers

    feed = NoiseFeed(seed, first)
    original = layers.NoiseBlock.forward
    layers.NoiseBlock.forward = lambda module, x: feed(module, x)
    try:
        levels = [torch.tensor(level, dtype=torch.long).unsqueeze(0) for level in unpack(codes)]
        with torch.inference_mode():
            audio = model.decoder(model.quantizer.from_codes(levels))
    finally:
        layers.NoiseBlock.forward = original
    assert feed.block == 4
    return audio[0, 0].numpy()


def windows(available_schedule, plan):
    """izwi's `Maya1AudioStream` windows as frames arrive per the schedule."""
    first_frames, max_frames, left, right = plan
    emitted, count, out = 1, 0, []
    for available, stopped in available_schedule:
        while True:
            start = emitted
            if stopped:
                end, decoded_end = max(available, start), available
            else:
                end = start + min(max(first_frames << min(count, 16), 1), max_frames)
                decoded_end = end + right
                if available < decoded_end:
                    break
            out.append((max(start - left, 0), start, end, decoded_end))
            emitted, count = end, count + 1
            if stopped:
                return out
    return out


def part_snac(model_dir, out_dir):
    from snac import SNAC

    model = SNAC.from_config(str(Path(__file__).with_name("snac_24khz_config.json")))
    state = torch.load(model_dir / "snac_24khz.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    frames = 24
    codes = snac_codes(SEED, frames)
    seed = 7
    whole = snac_decode(model, codes, seed, 0)
    plan = (2, 16, 3, 3)
    schedule = [(n, False) for n in range(1, frames + 1)] + [(frames, True)]
    pieces, spans = [], []
    for first, start, end, decoded_end in windows(schedule, plan):
        spans.append([first, start, end, decoded_end])
        if end == start:
            continue
        audio = snac_decode(model, codes[7 * first:7 * decoded_end], seed, first)
        pieces.append(audio[(start - first) * SAMPLES_PER_FRAME:(end - first) * SAMPLES_PER_FRAME])
    windowed = np.concatenate(pieces)
    trimmed = whole[SAMPLES_PER_FRAME:]
    error = float(np.abs(windowed - trimmed).max() / np.sqrt((trimmed ** 2).mean()))
    print(f"windowed vs whole: max |diff| / rms = {error:.3e}")
    # The reference streamer: last 4 frames, middle 2,048 samples, per frame.
    streamer = np.concatenate([
        snac_decode(model, codes[7 * (n - 4):7 * n], seed, n - 4)[2048:4096]
        for n in range(4, frames + 1)
    ])
    streamer_error = float(
        np.abs(streamer - trimmed[:len(streamer)]).max() / np.sqrt((trimmed ** 2).mean()))
    print(f"upstream streamer vs whole: max |diff| / rms = {streamer_error:.3e}")
    write(out_dir, "snac.json", {
        "revision": SNAC_REVISION,
        "seed": seed,
        "codes": codes,
        "whole": sample(whole),
        "windows": spans,
        "windowed": sample(windowed),
        "windowed_vs_whole_error": error,
        "upstream_streamer_vs_whole_error": streamer_error,
        "noise_block0_first4": block_noise(seed, 0, 0, 1, 32)[:4].tolist(),
    })


def census(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return {k: [v["dtype"], v["shape"]] for k, v in sorted(header.items())}


def part_lm(model_dir, out_dir):
    from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    description, text = PROMPTS[0]
    ids = prompt_ids(tokenizer, description, text)
    tensors = {}
    for shard in sorted(model_dir.glob("model-0000*-of-00002.safetensors")):
        tensors.update(census(shard))

    # The full model in BF16 samples the sequence everything is forced on.
    full = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
    full.eval()
    torch.manual_seed(1234)
    with torch.inference_mode():
        out = full.generate(
            torch.tensor([ids]), max_new_tokens=140, min_new_tokens=28, temperature=0.4,
            top_p=0.9, repetition_penalty=1.1, do_sample=True, eos_token_id=EOS,
            pad_token_id=tokenizer.pad_token_id)
    sampled = out[0, len(ids):].tolist()
    forced = ids + sampled
    steps = [0, 1, 7, 28, len(sampled) - 1]

    def forced_logits(model):
        with torch.inference_mode():
            logits = model(torch.tensor([forced])).logits[0].to(torch.float32)
        return {str(s): head_record(logits[len(ids) - 1 + s]) for s in steps}

    full_record = forced_logits(full)
    del full
    cfg = AutoConfig.from_pretrained(model_dir)
    cfg.num_hidden_layers = 4
    truncated = LlamaForCausalLM.from_pretrained(model_dir, config=cfg, torch_dtype=torch.float32)
    truncated.eval()
    # Cached decode must equal the one-shot forward (validates the reference).
    with torch.inference_mode():
        one_shot = truncated(torch.tensor([forced[:len(ids) + 3]])).logits[0, -1]
        past = truncated(torch.tensor([ids]), use_cache=True)
        cached = past.logits
        cache = past.past_key_values
        for token in sampled[:3]:
            step = truncated(torch.tensor([[token]]), past_key_values=cache, use_cache=True)
            cache, cached = step.past_key_values, step.logits
    assert torch.allclose(one_shot, cached[0, -1], atol=1e-4), "cached != one-shot"
    write(out_dir, "lm.json", {
        "revision": REVISION,
        "tensors": tensors,
        "prompt": ids,
        "sampled": sampled,
        "steps": steps,
        "truncated_layers": 4,
        "truncated_f32": forced_logits(truncated),
        "full_bf16": full_record,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--parts", default="tokenizer,sampler,snac,lm")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(torch.get_num_threads(), 1))
    for part in args.parts.split(","):
        if part == "tokenizer":
            part_tokenizer(args.model_dir, args.out_dir)
        elif part == "sampler":
            part_sampler(args.out_dir)
        elif part == "snac":
            part_snac(args.model_dir, args.out_dir)
        elif part == "lm":
            part_lm(args.model_dir, args.out_dir)
        else:
            sys.exit(f"unknown part {part}")


if __name__ == "__main__":
    main()
