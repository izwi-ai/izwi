#!/usr/bin/env python3
"""Generate Audio8 TTS Realistic Air reference fixtures with upstream's code.

The reference is Edge0/Audio8-TTS-Realistic-Air at HF revision 6bf7a980
(Apache-2.0): its own `ArkttsProcessor`, `ArkttsModel` and `ArkttsCodec`,
run in F32 on the CPU. Their source is the four Python files at the root of
the model repository, which izwi does not download:

    configuration_arktts.py  modeling_arktts.py
    modeling_arktts_codec.py processing_arktts.py

Put them in a directory of their own (CODE_DIR) and review them before
running this: they execute. This script imports them as a package, builds
the model from MODEL_DIR's `config.json`, `model.safetensors` and
`codec.pth`, and needs torch, numpy, safetensors and tokenizers. It uses
`transformers` for the base classes those files import when it is
installed, and small stand-ins for them when it is not (the model's math is
all in the four files).

Each part writes its own JSON file into OUT_DIR:

  prompt   prompt token ids for speech with and without a reference
           (`prompts.json`). Needs only the tokenizer.
  codec    the codec alone: codes of a synthetic 44.1 kHz clip (written as
           `codec_clip.wav`), and the waveform decoded from fixed codes
           (`codec.json`). Needs `codec.pth`.
  greedy   greedy generation (`do_sample=False`) with no reference: the
           prompt, every generated frame's ten codes and the decoded
           waveform (`greedy.json`). Needs the whole checkpoint, about 6 GB
           of RSS and a few minutes.

Sampling cannot be compared across frameworks (torch draws differently), so
the model is checked greedy. izwi's ignored tests in
`crates/izwi-core/src/models/architectures/audio8/reference_tests.rs` read
these files:

    IZWI_AUDIO8_MODEL_DIR=path/to/Audio8-TTS-Realistic-Air \\
    cargo test -p izwi-core --lib audio8::reference_tests -- --ignored

Usage:

    python3 -I scripts/fixtures/generate_audio8_reference.py \\
        path/to/Audio8-TTS-Realistic-Air path/to/code-dir \\
        crates/izwi-core/tests/fixtures/audio8 --parts prompt,codec,greedy
"""
import argparse
import importlib
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import torch

REPO_ID = "Edge0/Audio8-TTS-Realistic-Air"
REVISION = "6bf7a980759d8022a66922ee975dab99229a2dd9"
SAMPLE_RATE = 44_100
FRAME_SAMPLES = 2_048
NUM_CODEBOOKS = 10
SEMANTIC_BEGIN = 151_678
SAMPLE_POINTS = 256

# (target text, reference transcript or None). Whitespace is left untidy on
# purpose: the processor collapses it.
PROMPTS = [
    ("Hello, this is a test of the new speech model.", None),
    ("  Two  lines,\nand a tab\there.  ", None),
    ("今天天气很好，我们一起去公园散步吧。", None),
    ("The quick brown fox jumps over the lazy dog.", "A reference transcript, as spoken."),
    ("Second speaker tag kept.", "<|speaker:3|>Already tagged transcript."),
]
GREEDY_TEXT = "Hello, this is a test of the new speech model."
GREEDY_MAX_FRAMES = 160


def synthetic_clip(seconds: float = 1.5) -> np.ndarray:
    """A deterministic chirp with harmonics (no RNG), as 16-bit PCM values.

    The clip is written to `codec_clip.wav` and encoded from the same
    quantized values, so izwi's test reads exactly what the codec saw.
    """
    count = int(seconds * SAMPLE_RATE)
    t = np.arange(count, dtype=np.float64) / SAMPLE_RATE
    sweep = 110.0 + 330.0 * t / seconds
    phase = 2.0 * math.pi * np.cumsum(sweep) / SAMPLE_RATE
    wave = 0.35 * np.sin(phase) + 0.2 * np.sin(2.0 * phase) + 0.1 * np.sin(3.0 * phase + 0.5)
    fade = np.minimum(t * 20.0, 1.0)
    return np.round(wave * fade * 32767.0).astype(np.int16)


def write_wav(path: Path, pcm: np.ndarray) -> None:
    import wave

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.astype("<i2").tobytes())
    print(f"wrote {path}")


def fixed_codes(frames: int = 24) -> np.ndarray:
    """Codes from integer arithmetic only, inside each book's range."""
    codes = np.zeros((NUM_CODEBOOKS, frames), dtype=np.int64)
    for book in range(NUM_CODEBOOKS):
        size = 4096 if book == 0 else 1024
        for frame in range(frames):
            codes[book, frame] = (1_103_515_245 * (book * 131 + frame * 17 + 7) + 12_345) % size
    return codes


def sample_points(wave: np.ndarray) -> dict:
    wave = np.asarray(wave, dtype=np.float32)
    index = np.linspace(0, len(wave) - 1, num=min(SAMPLE_POINTS, len(wave))).astype(np.int64)
    return {
        "samples": int(len(wave)),
        "index": index.tolist(),
        "values": [float(v) for v in wave[index]],
        "rms": float(np.sqrt(np.mean(np.square(wave.astype(np.float64))))),
        "peak": float(np.max(np.abs(wave))),
    }


def write(out_dir: Path, name: str, payload: dict) -> None:
    payload = {"repo_id": REPO_ID, "revision": REVISION, **payload}
    path = out_dir / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def install_transformers_stand_ins() -> None:
    """Minimal versions of the `transformers` names the four files import.

    They hold configuration and call order only; every tensor operation is
    in upstream's files.
    """

    class PretrainedConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class PreTrainedModel(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

        def post_init(self):
            pass

        @property
        def device(self):
            return next(self.parameters()).device

        @property
        def dtype(self):
            return next(self.parameters()).dtype

    class GenerationConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class LogitsProcessor:
        pass

    class LogitsProcessorList(list):
        def __call__(self, input_ids, scores):
            for processor in self:
                scores = processor(input_ids, scores)
            return scores

    class StoppingCriteriaList(list):
        def __call__(self, input_ids, scores):
            return any(bool(criterion(input_ids, scores)) for criterion in self)

    class ModelOutput:
        pass

    class BatchFeature(dict):
        def __init__(self, data=None):
            super().__init__(data or {})

    class ProcessorMixin:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

    def cached_file(source, filename, **_):
        return str(Path(source) / filename)

    def module(name, **names):
        created = types.ModuleType(name)
        for key, value in names.items():
            setattr(created, key, value)
        sys.modules[name] = created
        return created

    module(
        "transformers",
        PretrainedConfig=PretrainedConfig,
        PreTrainedModel=PreTrainedModel,
        GenerationConfig=GenerationConfig,
        AutoTokenizer=None,
    )
    module(
        "transformers.generation",
        LogitsProcessor=LogitsProcessor,
        LogitsProcessorList=LogitsProcessorList,
        StoppingCriteriaList=StoppingCriteriaList,
    )
    module("transformers.modeling_outputs", ModelOutput=ModelOutput)
    module("transformers.utils")
    module("transformers.utils.hub", cached_file=cached_file)
    module("transformers.feature_extraction_utils", BatchFeature=BatchFeature)
    module("transformers.processing_utils", ProcessorMixin=ProcessorMixin)


class Reference:
    """Upstream's classes, imported from CODE_DIR as one package."""

    def __init__(self, model_dir: Path, code_dir: Path):
        try:
            import transformers  # noqa: F401

            self.transformers = "transformers " + transformers.__version__
        except ImportError:
            install_transformers_stand_ins()
            self.transformers = "stand-ins (transformers not installed)"
        package = types.ModuleType("arktts_reference")
        package.__path__ = [str(code_dir)]
        sys.modules["arktts_reference"] = package
        self.configuration = importlib.import_module("arktts_reference.configuration_arktts")
        self.modeling = importlib.import_module("arktts_reference.modeling_arktts")
        self.processing = importlib.import_module("arktts_reference.processing_arktts")
        self.model_dir = model_dir
        self._model = None

    def provenance(self) -> dict:
        return {
            "torch": torch.__version__,
            "base_classes": self.transformers,
            "dtype": "float32",
            "device": "cpu",
        }

    def processor(self):
        tokenizer_file = str(self.model_dir / "tokenizer.json")
        if self.transformers.startswith("transformers "):
            from transformers import PreTrainedTokenizerFast

            tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=tokenizer_file, pad_token="<|endoftext|>", eos_token="<|endoftext|>"
            )
        else:
            from tokenizers import Tokenizer

            backend = Tokenizer.from_file(tokenizer_file)

            class FastTokenizer:
                """`PreTrainedTokenizerFast.encode` over the same backend."""

                pad_token_id = backend.token_to_id("<|endoftext|>")

                def encode(self, text, add_special_tokens=False):
                    return backend.encode(text, add_special_tokens=add_special_tokens).ids

            tokenizer = FastTokenizer()
        return self.processing.ArkttsProcessor(tokenizer=tokenizer)

    def model(self):
        if self._model is not None:
            return self._model
        from safetensors.torch import load_file

        values = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
        for key in ("architectures", "auto_map", "model_type", "transformers_version", "dtype", "torch_dtype"):
            values.pop(key, None)
        config = self.configuration.ArkttsConfig(**values)
        # The codec is read from `codec.pth` beside the checkpoint.
        config._name_or_path = str(self.model_dir)
        model = self.modeling.ArkttsModel(config)
        state = load_file(str(self.model_dir / "model.safetensors"))
        model.load_state_dict({name: tensor.float() for name, tensor in state.items()}, strict=True)
        self._model = model.float().eval().to("cpu")
        return self._model


def part_prompt(reference: Reference, out_dir: Path) -> None:
    processor = reference.processor()
    cases = []
    for text, reference_text in PROMPTS:
        prefix, suffix = processor._prompt_segments(text, reference_text, reference_text is not None)
        cases.append(
            {
                "text": text,
                "reference_text": reference_text,
                # With no reference the whole prompt is the prefix. With one,
                # the reference frames go between prefix and suffix.
                "prefix_ids": prefix.tolist(),
                "suffix_ids": suffix.tolist(),
            }
        )
    write(out_dir, "prompts.json", {"generated_with": reference.provenance(), "cases": cases})


def part_codec(reference: Reference, out_dir: Path) -> None:
    model = reference.model()
    pcm = synthetic_clip()
    write_wav(out_dir / "codec_clip.wav", pcm)
    clip = pcm.astype(np.float32) / 32768.0
    audio = torch.from_numpy(clip)[None, None, :]
    lengths = torch.tensor([clip.shape[0]], dtype=torch.long)
    with torch.inference_mode():
        codes, code_lengths = model.encode_audio(audio, lengths)
        frames = int(code_lengths[0])
        encoded = codes[0, :, :frames].cpu().numpy()
        fixed = fixed_codes()
        waveforms, waveform_lengths = model.decode_audio(torch.from_numpy(fixed)[None])
        decoded = waveforms[0, : int(waveform_lengths[0])].float().cpu().numpy()
    write(
        out_dir,
        "codec.json",
        {
            "generated_with": reference.provenance(),
            "encode": {
                "clip": "codec_clip.wav",
                "clip_samples": int(clip.shape[0]),
                "frames": frames,
                "codes": encoded.tolist(),
            },
            "decode": {"codes": fixed.tolist(), "waveform": sample_points(decoded)},
        },
    )


def part_greedy(reference: Reference, out_dir: Path) -> None:
    processor = reference.processor()
    model = reference.model()
    inputs = processor(text=[GREEDY_TEXT], return_tensors="pt")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=GREEDY_MAX_FRAMES,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            do_sample=False,
            return_dict_in_generate=True,
        )
        frames = int(output.code_lengths[0])
        codes = output.codes[0, :, :frames]
        waveforms, waveform_lengths = model.decode_audio(output.codes)
        decoded = waveforms[0, : int(waveform_lengths[0])].float().cpu().numpy()
    write(
        out_dir,
        "greedy.json",
        {
            "generated_with": reference.provenance(),
            "text": GREEDY_TEXT,
            "max_frames": GREEDY_MAX_FRAMES,
            "prompt_ids": inputs["prefix_input_ids"][0].tolist(),
            "finished": bool(output.finished[0]),
            "frames": frames,
            "codes": codes.cpu().numpy().tolist(),
            "waveform": sample_points(decoded),
        },
    )


PARTS = {"prompt": part_prompt, "codec": part_codec, "greedy": part_greedy}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", type=Path, help="config.json, tokenizer.json, model.safetensors, codec.pth")
    parser.add_argument("code_dir", type=Path, help="the model repository's four Python files, reviewed")
    parser.add_argument("out_dir", type=Path, help="where the fixture files go")
    parser.add_argument("--parts", default="prompt,codec,greedy", help="comma-separated: prompt, codec, greedy")
    args = parser.parse_args()
    parts = [part.strip() for part in args.parts.split(",") if part.strip()]
    unknown = [part for part in parts if part not in PARTS]
    if unknown:
        parser.error(f"unknown parts: {', '.join(unknown)}")
    if not (args.model_dir / "config.json").is_file():
        parser.error(f"{args.model_dir} has no config.json")
    for name in ("configuration_arktts", "modeling_arktts", "modeling_arktts_codec", "processing_arktts"):
        if not (args.code_dir / f"{name}.py").is_file():
            parser.error(f"{args.code_dir} has no {name}.py")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    reference = Reference(args.model_dir.resolve(), args.code_dir.resolve())
    for part in parts:
        PARTS[part](reference, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
