#!/usr/bin/env python3
"""Writes the SOV-Transcribe-1 parity reference for izwi's ignored
real-checkpoint test (`sov_transcribe::reference_tests`).

Runs the published model with its own remote code (Transformers, BF16, on a
CUDA GPU with at least 64 GB) over the WAV files in a directory and records,
per file: the prompt ids, the mel features' summary, the audio embeddings'
summary, the final hidden state of the prompt's last token, the first 16
greedy tokens with their logits' top-5, and the greedy transcript.

    pip install "transformers>=5.19,<5.23" torch librosa accelerate
    python scripts/fixtures/generate_sov_transcribe_reference.py \
        --model /models/SOV-Transcribe-1 --audio-dir clips/ \
        --out crates/izwi-core/tests/fixtures/sov_transcribe/reference.json

Then on the same machine:

    IZWI_SOV_TRANSCRIBE_MODEL_DIR=/models/SOV-Transcribe-1 \
    IZWI_SOV_TRANSCRIBE_AUDIO_DIR=clips/ \
    cargo test -p izwi-core --features cuda --lib sov_transcribe::reference_tests -- --ignored
"""

import argparse
import json
import pathlib

import librosa
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

REVISION = "505d8fae243924c0d8acdc622f7f0abab344eba3"
DEFAULT_INSTRUCTION = (
    "Transcribe the audio verbatim in the language's standard writing system, "
    "using punctuation and capitalization. Output only the transcript."
)


def summary(tensor):
    values = tensor.detach().float().flatten().cpu()
    return {
        "shape": list(tensor.shape),
        "mean": values.mean().item(),
        "abs_mean": values.abs().mean().item(),
        "first": values[:16].tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(
        args.model, trust_remote_code=True, revision=REVISION
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        revision=REVISION,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()

    cases = []
    for path in sorted(pathlib.Path(args.audio_dir).glob("*.wav")):
        audio, _ = librosa.load(path, sr=16000, mono=True)
        inputs = processor.apply_transcription_request(
            audio, prompt=DEFAULT_INSTRUCTION
        ).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            audio_embeds = model.get_audio_features(inputs["input_features"])
            prefill = model(**inputs, output_hidden_states=True)
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        prompt_len = inputs["input_ids"].shape[1]
        tokens = generated.sequences[0, prompt_len:].tolist()
        top5 = [
            torch.topk(score[0].float(), 5).indices.tolist()
            for score in generated.scores[:16]
        ]
        cases.append(
            {
                "file": path.name,
                "instruction": DEFAULT_INSTRUCTION,
                "prompt_ids": inputs["input_ids"][0].tolist(),
                "mel": summary(inputs["input_features"]),
                "audio_embeddings": summary(audio_embeds),
                "last_hidden": summary(prefill.hidden_states[-1][0, -1]),
                "tokens": tokens,
                "top5": top5,
                "text": processor.decode(tokens, skip_special_tokens=True).strip(),
            }
        )
        print(path.name, cases[-1]["text"][:80])

    pathlib.Path(args.out).write_text(
        json.dumps({"revision": REVISION, "cases": cases}, indent=1) + "\n"
    )


if __name__ == "__main__":
    main()
