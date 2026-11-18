#!/usr/bin/env python3
"""Generate the Rumble-2-Flash reference fixture with transformers.

The reference is the published model code: transformers'
`Qwen3ASRForConditionalGeneration` and its processor, loaded from the
checkpoint in F32 on the CPU.

Both sides read the same audio, so the comparison covers the model and not
two decoders or resamplers. For each input file the script converts to
16 kHz mono, trims to a whole number of seconds (at most 30) and writes the
result to the audio directory the izwi test reads.

For every case (an audio file and request options) the script records:
  * the prompt ids the processor renders, audio placeholders expanded;
  * the number of mel frames and audio tokens;
  * a strided sample of the log-mel features and of the projected audio
    embeddings (shape, values at fixed flat indices, RMS);
  * the reference's greedy continuation (stopping at EOS) and, for each of
    its first steps under teacher forcing, the next token and its top-2
    margin;
  * the decoded text and the parsed language.

The izwi test (`rumble::reference_tests`, ignored: it needs the checkpoint
and the audio directory) builds each case's prompt from izwi's own request
options and compares every stage.

Usage (needs torch, transformers >= 5.21 with Qwen3-ASR support, soundfile,
torchaudio; about 4 GB RSS, a minute or two):

    python3 -I scripts/fixtures/generate_rumble_reference.py \\
        "$HOME/Library/Application Support/izwi/models/Rumble-2-Flash" \\
        path/to/audio-dir \\
        crates/izwi-core/tests/fixtures/rumble/reference.json \\
        data/fox.wav data/test.wav

The checkpoint is Reson8/Rumble-2-Flash at the pinned revision
(92ec9141fb1178c5e77fd0020fe86a33c355060b).
"""
import hashlib
import json
import sys
from pathlib import Path

import soundfile
import torch
import torchaudio
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

torch.set_grad_enabled(False)
torch.manual_seed(0)

SAMPLE_RATE = 16_000
MAX_SECONDS = 30
SAMPLE_POINTS = 64
TEACHER_FORCED_STEPS = 48
MAX_NEW_TOKENS = 256

# (name suffix, language, context prompt, digits, filler, verbatim); `None`
# leaves an option at the template's default.
OPTION_SETS = [
    ("auto", None, None, None, None, None),
    ("english", "English", None, None, None, None),
    ("words_fillers", "English", None, False, True, None),
    ("collapsed_context", "English", "Izwi, Rumble", None, None, False),
]


def prepare_audio(source: Path, audio_dir: Path) -> tuple[str, torch.Tensor]:
    wave, rate = torchaudio.load(str(source))
    wave = wave.mean(dim=0)
    if rate != SAMPLE_RATE:
        wave = torchaudio.functional.resample(wave, rate, SAMPLE_RATE)
    seconds = min(wave.shape[0] // SAMPLE_RATE, MAX_SECONDS)
    if seconds == 0:
        raise SystemExit(f"{source} is shorter than one second")
    wave = wave[: seconds * SAMPLE_RATE].contiguous()
    name = f"{source.stem}_{seconds}s.wav"
    soundfile.write(str(audio_dir / name), wave.numpy(), SAMPLE_RATE, subtype="FLOAT")
    # Read back what izwi will read.
    data, rate = soundfile.read(str(audio_dir / name), dtype="float32")
    assert rate == SAMPLE_RATE
    return name, torch.from_numpy(data)


def sample(tensor: torch.Tensor, limit: int | None = None) -> dict:
    flat = tensor.detach().to(torch.float32).reshape(-1)
    count = flat.numel() if limit is None else min(limit, flat.numel())
    stride = max(count // SAMPLE_POINTS, 1)
    indices = list(range(0, count, stride))[:SAMPLE_POINTS]
    return {
        "shape": list(tensor.shape),
        "indices": indices,
        "values": [float(flat[i]) for i in indices],
        "rms": float(flat[:count].pow(2).mean().sqrt()),
    }


def main() -> None:
    if len(sys.argv) < 5:
        raise SystemExit(__doc__)
    model_dir, audio_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    sources = [Path(arg) for arg in sys.argv[4:]]
    audio_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(str(model_dir))
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        str(model_dir), dtype=torch.float32
    ).eval()
    config = model.config
    eos_ids = config.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids])

    fixture = {
        "generator": "scripts/fixtures/generate_rumble_reference.py",
        "transformers": __import__("transformers").__version__,
        "torch": torch.__version__,
        "audio_sha256": {},
        "cases": {},
    }
    for source in sources:
        name, wave = prepare_audio(source, audio_dir)
        fixture["audio_sha256"][name] = hashlib.sha256(
            (audio_dir / name).read_bytes()
        ).hexdigest()
        for suffix, language, prompt, digits, filler, verbatim in OPTION_SETS:
            template_kwargs = {
                key: value
                for key, value in (("digits", digits), ("filler", filler), ("verbatim", verbatim))
                if value is not None
            }
            inputs = processor.apply_transcription_request(
                audio=wave.numpy(),
                language=language,
                prompt=prompt,
                **template_kwargs,
            )
            input_ids = inputs["input_ids"]
            features = inputs["input_features"].to(torch.float32)
            mask = inputs["input_features_mask"]
            mel_frames = int(mask.sum())
            audio = model.get_audio_features(features, mask, return_dict=True).pooler_output
            audio_tokens = int((input_ids == config.audio_token_id).sum())
            assert audio.shape[0] == audio_tokens, (audio.shape, audio_tokens)

            generated = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
            )[0, input_ids.shape[-1] :].tolist()
            while generated and generated[-1] in eos_ids:
                generated.pop()

            # Teacher forcing: one forward over prompt + generated prefix.
            steps = []
            forced = generated[:TEACHER_FORCED_STEPS]
            if forced:
                ids = torch.cat([input_ids, torch.tensor([forced[:-1]], dtype=input_ids.dtype)], dim=1)
                logits = model(
                    input_ids=ids,
                    input_features=inputs["input_features"],
                    input_features_mask=mask,
                ).logits[0, input_ids.shape[-1] - 1 :]
                for row, token in zip(logits, forced):
                    top = torch.topk(row.to(torch.float32), 2)
                    assert int(top.indices[0]) == token, "teacher forcing left the greedy path"
                    steps.append(
                        {"token": token, "margin": float(top.values[0] - top.values[1])}
                    )

            parsed = processor.decode([generated], return_format="parsed")[0]
            # [1, n_mels, padded frames] -> frame-major rows of the valid
            # frames.
            frame_major = features[0, :, :mel_frames].transpose(0, 1).contiguous()
            fixture["cases"][f"{Path(name).stem}_{suffix}"] = {
                "audio": name,
                "language": language,
                "prompt": prompt,
                "digits": digits,
                "filler": filler,
                "verbatim": verbatim,
                "prompt_ids": input_ids[0].tolist(),
                "mel_frames": mel_frames,
                "audio_tokens": audio_tokens,
                "features": sample(frame_major),
                "projected": sample(audio),
                "generated_ids": generated,
                "steps": steps,
                "text": parsed["transcription"],
                "detected_language": parsed["language"],
            }
            print(f"{name} [{suffix}]: {parsed['language']!r} {parsed['transcription']!r}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fixture, indent=1) + "\n")
    print(f"wrote {out_path} ({len(fixture['cases'])} cases)")


if __name__ == "__main__":
    main()
