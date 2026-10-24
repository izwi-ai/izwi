#!/usr/bin/env python3
"""Generate the Granite Speech 5.0 2B reference fixture with transformers.

The reference is IBM's own model code: transformers'
`GraniteSpeechForConditionalGeneration` (the YaRN language model included),
loaded from the published checkpoint in F32 on the CPU.

Both sides read the same 16 kHz audio, so the comparison covers the model
and not two resamplers: the script resamples `test.wav` to 16 kHz once and
writes it, with the multilingual sample, to the audio directory the izwi
test reads.

For every case (an audio file and an instruction) the script records:
  * the prompt ids, rendered by the checkpoint's chat template;
  * a strided sample of the stacked log-mel features and of the projected
    audio embeddings (shape, values at fixed flat indices, mean and RMS);
  * the reference's own greedy continuation (stopping at EOS) and, for each
    of its first steps under teacher forcing, the next token, its top-2
    margin and the top-8 logits (at transformers' scale, divided by
    `logits_scaling`);
  * the decoded greedy text.

The izwi test (`granite_speech5::asr::reference_tests`, ignored: it needs the
checkpoint and the audio directory) renders each case's prompt from izwi's
own request options and compares every stage.

Before writing, the script checks itself: a two-chunk prefill through the
language model must reproduce the one-shot prefill logits.

Usage (needs torch, torchaudio, transformers 4.57.x with
huggingface_hub<1.0, soundfile; about 12 GB RSS, a few minutes):

    python3 -I scripts/fixtures/generate_granite_speech5_reference.py \\
        "$HOME/Library/Application Support/izwi/models/Granite-Speech-5.0-2B" \\
        data/test.wav path/to/multilingual_sample.wav path/to/audio-dir \\
        crates/izwi-core/tests/fixtures/granite_speech5/reference.json

`multilingual_sample.wav` is the sample in the model repository at the pinned
revision (0555e70346a8992da6b5df98ae26ec50c43958b8).
"""
import hashlib
import json
import sys
from pathlib import Path

import soundfile
import torch
import torchaudio
from transformers import AutoProcessor, GraniteSpeechForConditionalGeneration

torch.set_grad_enabled(False)
torch.manual_seed(0)

SAMPLE_RATE = 16_000
SAMPLE_POINTS = 64
TEACHER_FORCED_STEPS = 48
MAX_NEW_TOKENS = 200
PUNCTUATED = "transcribe the speech with proper punctuation and capitalization."

CASES = [
    # (case name, audio key, instruction after the audio token)
    ("test_punctuated", "test", PUNCTUATED),
    ("test_raw", "test", "can you transcribe the speech into a written format?"),
    (
        "test_keywords",
        "test",
        "transcribe the speech to text. Keywords: Agentic AI, sovereign data networks",
    ),
    (
        "multilingual_language_id",
        "multilingual",
        f"The speech is in French. Transcribe it exactly; do not translate. {PUNCTUATED}",
    ),
    ("multilingual_punctuated", "multilingual", PUNCTUATED),
    (
        "multilingual_to_english",
        "multilingual",
        "translate the speech to English with proper punctuation and capitalization.",
    ),
    (
        "test_to_german",
        "test",
        "translate the speech to German with proper punctuation and capitalization.",
    ),
]

AUDIO_FILES = {"test": "test_16k.wav", "multilingual": "multilingual_sample.wav"}


def load_mono(path):
    data, rate = soundfile.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()).mean(dim=0, keepdim=True), rate


def write_16k(source, target):
    """Write `source` as 16 kHz mono float WAV and return its samples."""
    wav, rate = load_mono(source)
    if rate != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, rate, SAMPLE_RATE)
    soundfile.write(target, wav[0].numpy(), SAMPLE_RATE, subtype="FLOAT")
    # Read back what izwi will read.
    wav, rate = load_mono(target)
    assert rate == SAMPLE_RATE
    return wav


def sample(tensor):
    flat = tensor.detach().to(torch.float32).reshape(-1)
    count = flat.numel()
    indices = [(i * (count - 1)) // (SAMPLE_POINTS - 1) for i in range(SAMPLE_POINTS)]
    return {
        "shape": list(tensor.shape),
        "indices": indices,
        "values": [float(flat[i]) for i in indices],
        "mean": float(flat.mean()),
        "rms": float(flat.pow(2).mean().sqrt()),
    }


def top(logits, k=8):
    values, ids = torch.topk(logits.to(torch.float32), k)
    return [[int(i), float(v)] for i, v in zip(ids, values)]


def main():
    model_dir, test_wav, multilingual_wav, audio_dir, out = sys.argv[1:6]
    audio_dir = Path(audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio = {
        "test": write_16k(test_wav, audio_dir / AUDIO_FILES["test"]),
        "multilingual": write_16k(multilingual_wav, audio_dir / AUDIO_FILES["multilingual"]),
    }
    audio_sha256 = {
        key: hashlib.sha256((audio_dir / name).read_bytes()).hexdigest()
        for key, name in AUDIO_FILES.items()
    }

    processor = AutoProcessor.from_pretrained(model_dir)
    tokenizer = processor.tokenizer
    model = GraniteSpeechForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.float32
    ).eval()
    scaling = float(model.config.text_config.logits_scaling)
    eos = model.config.text_config.eos_token_id

    cases = {}
    for name, key, instruction in CASES:
        chat = [{"role": "user", "content": f"<|audio|>{instruction}"}]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = processor(prompt, audio[key], return_tensors="pt")
        features = inputs["input_features"]
        projected = model.get_audio_features(features)
        embeds = model.get_merged_audio_embeddings(
            input_ids=inputs["input_ids"],
            audio_features=projected,
            input_features_mask=inputs.get("input_features_mask"),
        )
        prompt_len = embeds.shape[1]

        # Self-check: two-chunk prefill reproduces one-shot prefill.
        one_shot = model.language_model(inputs_embeds=embeds, use_cache=True)
        split = prompt_len // 2
        first = model.language_model(inputs_embeds=embeds[:, :split], use_cache=True)
        second = model.language_model(
            inputs_embeds=embeds[:, split:],
            past_key_values=first.past_key_values,
            use_cache=True,
        )
        drift = (one_shot.logits[0, -1] - second.logits[0, -1]).abs().max().item()
        assert drift < 1e-3, f"{name}: chunked prefill drifted by {drift}"

        generated = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1
        )[0, inputs["input_ids"].shape[1]:].tolist()
        if eos in generated:
            generated = generated[: generated.index(eos) + 1]

        steps = []
        past = one_shot.past_key_values
        logits = one_shot.logits[0, -1]
        for token in generated[:TEACHER_FORCED_STEPS]:
            scaled = logits.to(torch.float32)
            best = torch.topk(scaled, 2).values
            steps.append(
                {
                    "next": int(torch.argmax(scaled)),
                    "margin": float(best[0] - best[1]),
                    "top": top(scaled),
                }
            )
            if token == eos:
                break
            step = model.language_model(
                input_ids=torch.tensor([[token]]), past_key_values=past, use_cache=True
            )
            past, logits = step.past_key_values, step.logits[0, -1]

        text = tokenizer.decode(generated, skip_special_tokens=True)
        cases[name] = {
            "audio": key,
            "instruction": instruction,
            "prompt": prompt,
            "prompt_ids": inputs["input_ids"][0].tolist(),
            "features": sample(features),
            "projected": sample(projected),
            "generated_ids": generated,
            "steps": steps,
            "text": text,
        }
        print(f"{name}: {len(generated)} tokens: {text!r}", file=sys.stderr)

    fixture = {
        "source": "transformers 4.57 GraniteSpeechForConditionalGeneration, F32 CPU",
        "repo_id": "ibm-granite/granite-speech-5.0-2b",
        "revision": "0555e70346a8992da6b5df98ae26ec50c43958b8",
        "logits_scaling": scaling,
        "audio": AUDIO_FILES,
        "audio_sha256": audio_sha256,
        "cases": cases,
    }
    Path(out).write_text(json.dumps(fixture, indent=1) + "\n")


if __name__ == "__main__":
    main()
