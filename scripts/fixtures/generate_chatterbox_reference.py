#!/usr/bin/env python3
"""Generate the Chatterbox reference fixture with upstream's PyTorch code.

The reference is resemble-ai/chatterbox at 5de7a54a (installed from source,
`pip install --no-deps`), with s3tokenizer at 9bf5d845 and resemble-perth at
ff1c8ac5, torch 2.6.0 and transformers 5.2.0, F32 on the CPU, loading the
English checkpoint at HF revision 5bb1f6ee.

Upstream draws its noise from torch's RNG. To compare the stochastic stages
exactly, this script drives them with izwi's portable SplitMix64 streams
(`chatterbox/noise.rs`, mirrored below): the flow's initial noise, the
vocoder's harmonic phases and its source noise are generated here from the
same seed izwi uses and patched into upstream's own modules. T3's sampling
cannot be shared (torch's multinomial draws differently), so T3 is checked
under teacher forcing on upstream's own sampled tokens, and the sampler's
logit processing is checked separately.

For the default voice (`conds.pt`) the fixture records, as strided sample
points (indices, values, mean, RMS, shape):
  * text normalisation and token ids for a set of strings;
  * the 34 conditioning embeddings and both prefill rows (CFG);
  * prefill logits (top-8 of the conditional and unconditional rows);
  * upstream's own sampled speech tokens, and for the first steps under
    teacher forcing each row's top-8 logits;
  * upstream's logit processing (CFG, repetition penalty, temperature,
    min-p, top-p) on the first step;
  * S3Gen with the shared noise: the encoder output `mu`, the sample after
    Euler steps 1 and 5, and the final mel;
  * HiFT with the shared phases and noise: F0, the mixed source and the
    waveform, before and after the trim fade.

Before writing, the script checks itself: one-shot prefill logits equal the
KV-cached ones, and its stepwise Euler loop reproduces upstream's decoder.

It also writes `tensor_census.json`: every tensor of the three safetensors
files (name, dtype, shape) and the structure of `conds.pt`.

Usage (about 8 GB RSS, a few minutes):

    python3 -I scripts/fixtures/generate_chatterbox_reference.py \\
        path/to/Chatterbox-500M crates/izwi-core/tests/fixtures/chatterbox
"""
import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ID = "ResembleAI/chatterbox"
REVISION = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
SOURCE = "resemble-ai/chatterbox@5de7a54aa4e5e2baadb0182dde554908b48b85c2"
SAMPLE_POINTS = 64
TEACHER_FORCED_STEPS = 48
SEED = 20261026
T3_TORCH_SEED = 1234
MAX_NEW_TOKENS = 1000

TEXT_CASES = [
    "",
    "hello world",
    "The quick brown fox jumps over the lazy dog",
    "Wait... what? It's 3:45 — really; “quoted” text.",
    "Dr. Smith paid $12.50 for 3 apples, didn't he?",
]
MAIN_TEXT = "Hello! This is a quick test of the Chatterbox voice, running natively in Izwi."

# --------------------------------------------------- portable noise (noise.rs)
MASK = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15


def mix(z):
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


STREAMS = {"sampling": 1, "flow_noise": 2, "harmonic_phase": 3, "source_noise": 4}


def stream_seed(seed, stream):
    return mix(seed ^ mix((STREAMS[stream] * GOLDEN) & MASK))


def open01(state, count):
    """`count` SplitMix64 uniforms in (0, 1) after `state`, vectorised."""
    with np.errstate(over="ignore"):
        k = np.arange(1, count + 1, dtype=np.uint64)
        z = np.uint64(state) + k * np.uint64(GOLDEN)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return ((z >> np.uint64(11)).astype(np.float64) + 0.5) * (1.0 / (1 << 53))


def standard_normals(seed, stream, count):
    pairs = (count + 1) // 2
    u = open01(stream_seed(seed, stream), 2 * pairs)
    radius = np.sqrt(-2.0 * np.log(u[0::2]))
    angle = 2.0 * np.pi * u[1::2]
    out = np.empty(2 * pairs, dtype=np.float64)
    out[0::2] = radius * np.cos(angle)
    out[1::2] = radius * np.sin(angle)
    return out[:count].astype(np.float32)


def uniforms(seed, stream, count):
    return open01(stream_seed(seed, stream), count).astype(np.float32)


# ------------------------------------------------------------------ helpers
def sample(tensor):
    flat = tensor.detach().reshape(-1).to(torch.float64).cpu().numpy()
    n = flat.size
    indices = sorted({int(i * (n - 1) / (SAMPLE_POINTS - 1)) for i in range(SAMPLE_POINTS)})
    return {
        "shape": list(tensor.shape),
        "indices": indices,
        "values": [float(flat[i]) for i in indices],
        "mean": float(flat.mean()),
        "rms": float(np.sqrt((flat ** 2).mean())),
    }


def top(logits, k=8):
    values, ids = torch.topk(logits.to(torch.float32), k)
    return [[int(i), float(v)] for i, v in zip(ids.tolist(), values.tolist())]


def census(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    tensors = {k: [v["dtype"], v["shape"]] for k, v in sorted(header.items())}
    digest = hashlib.sha256(
        "\n".join(f"{k}:{d}:{s}" for k, (d, s) in tensors.items()).encode()
    ).hexdigest()
    return {"file_bytes": path.stat().st_size, "count": len(tensors), "digest": digest, "tensors": tensors}


class Patch:
    """Temporarily replace an attribute."""

    def __init__(self, owner, name, value):
        self.owner, self.name, self.value = owner, name, value

    def __enter__(self):
        self.saved = getattr(self.owner, self.name)
        setattr(self.owner, self.name, self.value)

    def __exit__(self, *exc):
        setattr(self.owner, self.name, self.saved)


# --------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    import transformers
    from chatterbox import tts as tts_module
    from chatterbox.models.s3gen import flow_matching
    from chatterbox.models.s3gen.s3gen import S3Token2Wav
    from transformers.generation.logits_process import (
        MinPLogitsWarper,
        RepetitionPenaltyLogitsProcessor,
        TopPLogitsWarper,
    )

    model = tts_module.ChatterboxTTS.from_local(args.model_dir, "cpu")
    t3, s3gen, tokenizer = model.t3, model.s3gen, model.tokenizer
    hp = t3.hp

    out = {
        "source": f"{SOURCE}, torch {torch.__version__}, transformers {transformers.__version__}, F32 CPU",
        "repo_id": REPO_ID,
        "revision": REVISION,
        "seed": SEED,
        "noise_check": {
            "flow_noise": standard_normals(SEED, "flow_noise", 4).tolist(),
            "harmonic_phase": uniforms(SEED, "harmonic_phase", 4).tolist(),
            "source_noise": standard_normals(SEED, "source_noise", 4).tolist(),
        },
    }

    # Text frontend.
    out["text_cases"] = [
        {
            "text": text,
            "normalized": tts_module.punc_norm(text),
            "ids": tokenizer.text_to_tokens(tts_module.punc_norm(text))[0].tolist(),
        }
        for text in TEXT_CASES
    ]

    # T3 prefill with the default voice, guided.
    cfg_weight, exaggeration = 0.5, 0.5
    conds = model.conds
    t3_cond = tts_module.T3Cond(
        speaker_emb=conds.t3.speaker_emb,
        cond_prompt_speech_tokens=conds.t3.cond_prompt_speech_tokens,
        emotion_adv=exaggeration * torch.ones(1, 1, 1),
    )
    normalized = tts_module.punc_norm(MAIN_TEXT)
    text_ids = tokenizer.text_to_tokens(normalized)
    text_ids = torch.nn.functional.pad(text_ids, (1, 0), value=hp.start_text_token)
    text_ids = torch.nn.functional.pad(text_ids, (0, 1), value=hp.stop_text_token)
    text_pair = torch.cat([text_ids, text_ids], dim=0).long()
    conditioning = t3.prepare_conditioning(
        tts_module.T3Cond(
            speaker_emb=t3_cond.speaker_emb,
            cond_prompt_speech_tokens=t3_cond.cond_prompt_speech_tokens,
            emotion_adv=t3_cond.emotion_adv,
        )
    )
    bos = hp.start_speech_token * torch.ones_like(text_pair[:, :1])
    embeds, _ = t3.prepare_input_embeds(
        t3_cond=tts_module.T3Cond(
            speaker_emb=t3_cond.speaker_emb,
            cond_prompt_speech_tokens=t3_cond.cond_prompt_speech_tokens,
            emotion_adv=t3_cond.emotion_adv,
        ),
        text_tokens=text_pair,
        speech_tokens=bos,
        cfg_weight=cfg_weight,
    )
    bos_embed = t3.speech_emb(bos[:1]) + t3.speech_pos_emb.get_fixed_embedding(0)
    prefill = torch.cat([embeds, torch.cat([bos_embed, bos_embed])], dim=1)

    backbone = t3.tfmr
    one_shot = backbone(inputs_embeds=prefill, use_cache=True, output_hidden_states=True)
    last_hidden = one_shot.hidden_states[-1]
    assert torch.equal(last_hidden, one_shot.last_hidden_state), "hidden_states[-1] is not the normed output"
    prefill_logits = t3.speech_head(last_hidden[:, -1])

    # Upstream's own sampled tokens (torch RNG), via its inference loop.
    torch.manual_seed(T3_TORCH_SEED)
    sampled = t3.inference(
        t3_cond=tts_module.T3Cond(
            speaker_emb=t3_cond.speaker_emb,
            cond_prompt_speech_tokens=t3_cond.cond_prompt_speech_tokens,
            emotion_adv=t3_cond.emotion_adv,
        ),
        text_tokens=text_pair,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.8,
        cfg_weight=cfg_weight,
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
    )[0]
    sampled_list = sampled.tolist()

    # Teacher-forced decode on those tokens with the KV cache, checked against
    # a one-shot forward over the whole sequence.
    past = one_shot.past_key_values
    steps = []
    logits = prefill_logits
    tokens = [hp.start_speech_token]
    for i, token in enumerate(sampled_list[:TEACHER_FORCED_STEPS]):
        steps.append({
            "next": int(token),
            "cond_top": top(logits[0]),
            "uncond_top": top(logits[1]),
        })
        tokens.append(int(token))
        embed = t3.speech_emb(torch.tensor([[token]])) + t3.speech_pos_emb.get_fixed_embedding(i + 1)
        step = backbone(inputs_embeds=torch.cat([embed, embed]), past_key_values=past, use_cache=True)
        past = step.past_key_values
        logits = t3.speech_head(step.last_hidden_state[:, -1])
    forced = [
        t3.speech_emb(torch.tensor([[tok]])) + t3.speech_pos_emb.get_fixed_embedding(i + 1)
        for i, tok in enumerate(sampled_list[:TEACHER_FORCED_STEPS])
    ]
    full = torch.cat([prefill, torch.cat([torch.cat(forced, 1)] * 2)], dim=1)
    full_logits = t3.speech_head(backbone(inputs_embeds=full).last_hidden_state[:, -1])
    gap = (full_logits - logits).abs().max().item()
    assert gap < 1e-3, f"cached decode drifted from the one-shot forward by {gap}"

    # Upstream's logit processing on the first step.
    cond, uncond = prefill_logits[0:1], prefill_logits[1:2]
    guided = cond + torch.as_tensor(cfg_weight) * (cond - uncond)
    history = torch.tensor([[hp.start_speech_token]])
    processed = RepetitionPenaltyLogitsProcessor(penalty=1.2)(history, guided)
    processed = processed / 0.8
    processed = MinPLogitsWarper(min_p=0.05)(history, processed)
    processed = TopPLogitsWarper(top_p=1.0)(history, processed)
    kept = torch.isfinite(processed[0]).nonzero().flatten().tolist()

    out["t3"] = {
        "text": MAIN_TEXT,
        "normalized": normalized,
        "text_ids": text_ids[0].tolist(),
        "cfg_weight": cfg_weight,
        "exaggeration": exaggeration,
        "conditioning": sample(conditioning[0]),
        "prefill_cond": sample(prefill[0]),
        "prefill_uncond": sample(prefill[1]),
        "prefill_len": int(prefill.shape[1]),
        "prefill_cond_top": top(prefill_logits[0]),
        "prefill_uncond_top": top(prefill_logits[1]),
        "torch_seed": T3_TORCH_SEED,
        "sampled": sampled_list,
        "steps": steps,
        "processed_first_step": {
            "kept": kept,
            "top": top(processed[0]),
        },
    }

    # S3Gen with izwi's noise.
    speech_tokens = tts_module.drop_invalid_tokens(sampled)
    speech_tokens = speech_tokens[speech_tokens < 6561]
    ref = conds.gen
    flow = s3gen.flow
    prompt_tokens = ref["prompt_token"].shape[1]
    frames = 2 * (prompt_tokens + speech_tokens.shape[0])
    noise = torch.from_numpy(standard_normals(SEED, "flow_noise", 80 * frames)).reshape(1, 80, frames)

    def fixed_randn_like(tensor, *args, **kwargs):
        assert tuple(tensor.shape) == (1, 80, frames), tensor.shape
        return noise.clone()

    with Patch(flow_matching.torch, "randn_like", fixed_randn_like):
        mel = s3gen.flow_inference(speech_tokens, ref_dict=ref, finalize=True)

    # Our own stepwise Euler loop, checked against upstream's output above.
    token = torch.cat([ref["prompt_token"], speech_tokens[None]], dim=1)
    token_len = torch.tensor([token.shape[1]])
    embedding = flow.spk_embed_affine_layer(torch.nn.functional.normalize(ref["embedding"], dim=1))
    h, _ = flow.encoder(flow.input_embedding(token.long()), token_len)
    mu = flow.encoder_proj(h).transpose(1, 2)
    cond_mel = torch.zeros(1, frames, 80)
    cond_mel[:, : ref["prompt_feat"].shape[1]] = ref["prompt_feat"]
    cond_mel = cond_mel.transpose(1, 2)
    mask = torch.ones(1, 1, frames)
    decoder = flow.decoder
    t_span = torch.linspace(0, 1, 11)
    t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
    x = noise.clone()
    snapshots = {}
    for i, (t, r) in enumerate(zip(t_span[:-1], t_span[1:])):
        dxdt = decoder.estimator(
            x=torch.cat([x, x]),
            mask=torch.cat([mask, mask]),
            mu=torch.cat([mu, torch.zeros_like(mu)]),
            t=torch.stack([t, t]),
            spks=torch.cat([embedding, torch.zeros_like(embedding)]),
            cond=torch.cat([cond_mel, torch.zeros_like(cond_mel)]),
        )
        dxdt = (1.0 + decoder.inference_cfg_rate) * dxdt[:1] - decoder.inference_cfg_rate * dxdt[1:]
        x = x + (r - t) * dxdt
        if i in (0, 4):
            snapshots[f"x_step{i + 1}"] = sample(x[0])
    mine = x[:, :, ref["prompt_feat"].shape[1]:]
    gap = (mine - mel).abs().max().item()
    assert gap < 1e-4, f"stepwise Euler loop differs from upstream by {gap}"

    out["s3gen"] = {
        "speech_tokens": speech_tokens.tolist(),
        "frames": frames,
        "prompt_frames": int(ref["prompt_feat"].shape[1]),
        "mu": sample(mu[0]),
        **snapshots,
        "mel": sample(mel[0]),
    }

    # HiFT with izwi's phases and source noise.
    hift = s3gen.mel2wav
    mel_frames = mel.shape[2]
    length = mel_frames * 480
    phases = uniforms(SEED, "harmonic_phase", 9)
    source_noise = torch.from_numpy(standard_normals(SEED, "source_noise", 9 * length)).reshape(1, 9, length)

    from torch.distributions.uniform import Uniform

    def fixed_uniform(self, sample_shape=torch.Size()):
        assert tuple(sample_shape) == (1, 9, 1), sample_shape
        return (self.low + (self.high - self.low) * torch.from_numpy(phases)).reshape(1, 9, 1)

    def fixed_source_randn_like(tensor, *args, **kwargs):
        if tuple(tensor.shape) == (1, 9, length):
            return source_noise.clone()
        return torch.zeros_like(tensor)

    from chatterbox.models.s3gen import hifigan

    with Patch(Uniform, "sample", fixed_uniform), Patch(hifigan.torch, "randn_like", fixed_source_randn_like):
        f0 = hift.f0_predictor(mel)
        wav, source = hift.inference(speech_feat=mel, cache_source=torch.zeros(1, 1, 0))
    faded = wav.clone()
    faded[:, : len(s3gen.trim_fade)] *= s3gen.trim_fade

    out["hift"] = {
        "f0": sample(f0[0]),
        "source": sample(source[0, 0]),
        "wav": sample(wav[0]),
        "wav_faded": sample(faded[0]),
        "samples": int(wav.shape[1]),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "reference.json").write_text(json.dumps(out, indent=1) + "\n")

    conds_raw = torch.load(args.model_dir / "conds.pt", map_location="cpu", weights_only=True)
    census_out = {
        "source": f"safetensors headers and conds.pt of {REPO_ID}@{REVISION}",
        "files": {
            name: census(args.model_dir / name)
            for name in ["t3_cfg.safetensors", "s3gen.safetensors", "ve.safetensors"]
        },
        "conds.pt": {
            group: {
                key: None if value is None else [str(value.dtype).replace("torch.", ""), list(value.shape)]
                for key, value in entries.items()
            }
            for group, entries in conds_raw.items()
        },
    }
    (args.out_dir / "tensor_census.json").write_text(json.dumps(census_out, indent=1) + "\n")
    print(f"T3 sampled {len(sampled_list)} tokens; {speech_tokens.shape[0]} renderable; "
          f"{wav.shape[1] / 24000:.2f} s of audio", file=sys.stderr)


if __name__ == "__main__":
    main()
