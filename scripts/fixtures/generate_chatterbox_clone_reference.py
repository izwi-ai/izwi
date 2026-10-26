#!/usr/bin/env python3
"""Generate the Chatterbox voice-cloning fixture with upstream's PyTorch code.

Same pins as `generate_chatterbox_reference.py`: resemble-ai/chatterbox at
5de7a54a, s3tokenizer at 9bf5d845, torch 2.6.0, F32 on the CPU, the English
checkpoint at HF revision 5bb1f6ee.

The reference audio is synthetic and bit-reproducible: a glottal pulse train
with a scheduled integer period through two fixed resonators, unvoiced
SplitMix64 noise between syllables, and silent margins for the trim. It is
built only from IEEE additions and multiplications in a fixed order, so
`chatterbox::reference` regenerates the same F32 samples in Rust. No audio is
committed.

Upstream's `prepare_conditionals` resamples the reference with librosa's SoXR
for T3. izwi resamples with torchaudio's `Resample` throughout, as upstream
already does inside `S3Gen.embed_ref`, so this script feeds upstream's
modules the torchaudio-resampled signal and records each stage of the voice
izwi builds:
  * the 16 kHz signal;
  * S3Gen: the 80-bin reference mel, the prompt tokens (trimmed to half the
    mel frames) and the CAMPPlus x-vector, with its mean-normalised fbank;
  * T3: the first 6 s tokenised to at most 150 tokens, with the tokenizer's
    log-mel, and the voice encoder's embedding of the whole 16 kHz signal
    with its trimmed span and power mel.

Usage (about 2 GB RSS, under a minute):

    python3 -I scripts/fixtures/generate_chatterbox_clone_reference.py \\
        path/to/Chatterbox-500M crates/izwi-core/tests/fixtures/chatterbox
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

SOURCE = "resemble-ai/chatterbox@5de7a54aa4e5e2baadb0182dde554908b48b85c2"
REVISION = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
SAMPLE_POINTS = 64
SEED = 20261026

# ------------------------------------------------------- reference signal
MASK = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15


def splitmix_open01(state):
    """One SplitMix64 draw in (0, 1); returns (value, next state)."""
    state = (state + GOLDEN) & MASK
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    z ^= z >> 31
    return ((z >> 11) + 0.5) * (1.0 / (1 << 53)), state


SAMPLE_RATE = 24_000
LEAD = 6_000  # 0.25 s of silence before and after
SYLLABLE = 6_000  # 0.25 s voiced
GAP = 1_200  # 0.05 s unvoiced
PERIODS = [160, 166, 172, 178, 172, 166, 154, 148]  # 150–162 Hz
AMPLITUDES = [0.9, 0.6, 1.0, 0.7, 0.8, 0.5, 0.95, 0.65]
SYLLABLES = 14  # 4.2 s voiced and unvoiced, 4.7 s with the margins
# Two resonators y = x + a1·y[-1] + a2·y[-2] (formants near 700 and 1,200 Hz).
F1 = (1.7832, -0.8836)
F2 = (1.5174, -0.8464)


def reference_signal():
    out = [0.0] * LEAD
    state = SEED
    y1 = [0.0, 0.0]
    y2 = [0.0, 0.0]
    for syllable in range(SYLLABLES):
        period = PERIODS[syllable % len(PERIODS)]
        amplitude = AMPLITUDES[syllable % len(AMPLITUDES)]
        for n in range(SYLLABLE + GAP):
            if n < SYLLABLE:
                x = amplitude if n % period == 0 else 0.0
            else:
                u, state = splitmix_open01(state)
                x = (u - 0.5) * 0.08
            a = x + F1[0] * y1[0] + F1[1] * y1[1]
            y1 = [a, y1[0]]
            b = a + F2[0] * y2[0] + F2[1] * y2[1]
            y2 = [b, y2[0]]
            out.append(b * 0.05)
    out.extend([0.0] * LEAD)
    return np.array(out, dtype=np.float64).astype(np.float32)


# ------------------------------------------------------------------ helpers
def sample(tensor):
    tensor = torch.as_tensor(tensor)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()

    import librosa
    import torchaudio
    from chatterbox import tts as tts_module
    from chatterbox.models.s3gen import xvector
    from chatterbox.models.voice_encoder import melspec

    torch.set_grad_enabled(False)
    model = tts_module.ChatterboxTTS.from_local(str(args.model_dir), "cpu")
    s3gen = model.s3gen

    wav24 = torch.from_numpy(reference_signal())
    # The module `S3Gen.embed_ref` uses (`get_resampler`); its kernel is
    # not the functional form's, which rounds differently.
    wav16 = torchaudio.transforms.Resample(24_000, 16_000)(wav24)

    # S3Gen: the reference is under 10 s, so the 16 kHz crop is the whole.
    ref = s3gen.embed_ref(wav24.unsqueeze(0), 24_000, device="cpu")
    mel = ref["prompt_feat"][0]
    s3gen_tokens = ref["prompt_token"][0]
    xvec = ref["embedding"][0]
    fbank = xvector.extract_feature(wav16.unsqueeze(0))[0][0]

    # T3: upstream's ENC_COND_LEN crop (6 s) and speech_cond_prompt_len.
    tokenizer = s3gen.tokenizer
    t3_input = wav16[: 6 * 16_000]
    t3_tokens, _ = tokenizer.forward([t3_input], max_len=model.t3.hp.speech_cond_prompt_len)
    tokenizer_mel = tokenizer.log_mel_spectrogram(t3_input)

    wav16_np = wav16.numpy()
    trimmed, span = librosa.effects.trim(wav16_np, top_db=20)
    ve_mel = melspec.melspectrogram(trimmed, model.ve.hp).T
    ve = model.ve.embeds_from_wavs([wav16_np], sample_rate=16_000).mean(axis=0)

    fixture = {
        "source": SOURCE,
        "revision": REVISION,
        "signal": {"sample_rate": 24_000, **sample(wav24)},
        "wav16": sample(wav16),
        "s3gen": {
            "mel": sample(mel),
            "tokens": [int(t) for t in s3gen_tokens.tolist()],
            "fbank": sample(fbank),
            "xvector": [float(v) for v in xvec.tolist()],
        },
        "t3": {
            "tokenizer_mel": sample(tokenizer_mel),
            "tokens": [int(t) for t in t3_tokens.reshape(-1).tolist()],
            "trim": [int(span[0]), int(span[1])],
            "ve_mel": sample(torch.from_numpy(np.ascontiguousarray(ve_mel))),
            "ve": [float(v) for v in ve.tolist()],
        },
    }
    out = args.out_dir / "clone_reference.json"
    out.write_text(json.dumps(fixture, indent=1) + "\n")
    print(f"wrote {out}: {len(fixture['s3gen']['tokens'])} S3Gen tokens, "
          f"{len(fixture['t3']['tokens'])} T3 tokens, mel {list(mel.shape)}")


if __name__ == "__main__":
    main()
