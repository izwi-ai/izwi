#!/usr/bin/env python3
"""Generate the Ming-3.0-tts-0.5B reference fixture with the published code.

The reference is upstream's own implementation
(https://github.com/inclusionAI/Ming-3.0-tts at c88c023e), loaded from the
checkpoint in F32 on the CPU. Nothing here samples with torch's generator:
every noise tensor is written by this script from a xorshift stream the izwi
test reproduces, so the two sides compare the same computation.

Recorded:
  * `prompts`: the token ids, speaker rows and latent rows upstream builds
    for each voice mode and control (built-in voice, voice design,
    free-form, emotion, prosody, dialect, a named language, a reference
    with and without its transcript);
  * `lm`: for one built-in-voice prompt, a strided sample of the language
    model's final hidden state and the stop head's probability;
  * `aggregator`: a strided sample of `linear_proj_audio` over fixed latents;
  * `flow`: the latent `cfm.sample` gives for a fixed hidden state, history
    and starting noise (guidance 2.0, ten steps);
  * `vae_decode`: a strided sample of the waveform decoded from fixed
    latents, once per attention implementation available (`sdpa` always,
    `flash` when flash-attn is installed): the two differ by one key in the
    VAE's 64-frame window, and the fixture says which one izwi matches;
  * `vae_encode`: the posterior mean and scale of a short clip;
  * `speaker`: the CAM++ embedding of that clip from upstream's
    `campplus.onnx` (izwi runs the same network from the PyTorch
    checkpoint).

The izwi tests (`ming_tts::reference_tests`, ignored: they need the
checkpoint) compare every stage and skip while the fixture is absent.

Usage (needs torch 2.6, transformers 4.46.3, x_transformers 2.9.2,
diffusers 0.33.0, onnxruntime, torchaudio, soundfile; about 8 GB RSS):

    python3 -I scripts/fixtures/generate_ming_tts_reference.py \\
        path/to/Ming-3.0-tts            # upstream's git checkout \\
        path/to/Ming-3.0-tts-0.5B       # the HF download, campplus.onnx included \\
        crates/izwi-core/tests/fixtures/ming_tts/reference.json \\
        [clip.wav]                      # a 2-10 s mono clip; a tone when omitted

The checkpoint is inclusionAI/Ming-3.0-tts-0.5B at the pinned revision
(ffd36ab5764c39c7e5565dfed66a3aeeced58ab9).
"""
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch

torch.set_grad_enabled(False)

SAMPLE_POINTS = 64
LATENT_DIM = 64
HIDDEN = 896
SAMPLE_RATE = 44_100
LATENT_SAMPLES = 3_528


def xorshift(count, seed, scale=1.0):
    """izwi's `kernels::numeric_check::xorshift_values`."""
    mask = (1 << 64) - 1
    state = ((seed * 0x9E3779B97F4A7C15) & mask) | 1
    out = []
    for _ in range(count):
        state ^= (state << 13) & mask
        state ^= state >> 7
        state ^= (state << 17) & mask
        unit = (state >> 40) / float(1 << 24)
        out.append((unit * 2.0 - 1.0) * scale)
    return out


def sample(tensor):
    flat = tensor.detach().float().flatten()
    count = flat.numel()
    indices = sorted({int(i * (count - 1) / (SAMPLE_POINTS - 1)) for i in range(SAMPLE_POINTS)})
    return {
        "shape": list(tensor.shape),
        "indices": indices,
        "values": [float(flat[i]) for i in indices],
        "rms": float(flat.pow(2).mean().sqrt()),
    }


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    repo, model_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    clip_path = Path(sys.argv[4]) if len(sys.argv) > 4 else None
    sys.path.insert(0, str(repo))

    import prompt_contract
    import tts_tasks
    from modeling_bailingmm import BailingMMNativeForConditionalGeneration
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = BailingMMNativeForConditionalGeneration.from_pretrained(
        str(model_dir), torch_dtype=torch.float32, low_cpu_mem_usage=False
    ).float().eval()
    model.tokenizer = tokenizer
    weights = model_dir / "model.safetensors"
    fixture = {
        "revision": "ffd36ab5764c39c7e5565dfed66a3aeeced58ab9",
        "weights_bytes": weights.stat().st_size,
        "flash_attn": importlib.util.find_spec("flash_attn") is not None,
    }

    # ---- prompts ---------------------------------------------------------
    def prompt_case(name, text, *, language=None, speakers=0, instruction=None,
                    prompt_text=None, latents=0):
        ids, speaker_rows, latent_row = prompt_contract.build_input_ids(
            tokenizer, "dense", tts_tasks.build_tts_prompt(language), text,
            num_speakers=speakers, instruction=instruction,
            prompt_text=prompt_text, prompt_latent_steps=latents,
        )
        return {
            "name": name, "text": text, "language": language,
            "speaker": speakers == 1, "instruction": instruction,
            "reference_text": prompt_text, "reference_latents": latents,
            "ids": ids, "speaker_rows": speaker_rows, "latent_row": latent_row,
        }

    caption = tts_tasks.build_task_instruction
    fixture["prompts"] = [
        prompt_case("builtin_en", "Hello there.", speakers=1,
                    instruction=caption("IP", ip="US_Emma")),
        prompt_case("builtin_zh", "你好，今天天气不错。", language="zh", speakers=1,
                    instruction=caption("IP", ip="灵小柔")),
        prompt_case("design", "Once upon a time.", speakers=1,
                    instruction=caption("VOICE_DESIGN", style="A warm, low storyteller voice.")),
        prompt_case("freeform", "We did it!", speakers=1,
                    instruction="Speak with bright, excited energy."),
        prompt_case("emotion", "所有测试都顺利通过了。", speakers=1,
                    instruction=caption("EMOTION", emotion="高兴")),
        prompt_case("prosody", "请耐心等待下一次通知。", speakers=1,
                    instruction=caption("BASIC", speed="慢速", pitch="中", volume="高")),
        prompt_case("dialect", "今日天气几好。", speakers=1,
                    instruction=caption("DIALECT", dialect="cantonese")),
        prompt_case("german", "Guten Tag, wie geht es Ihnen", language="de", speakers=1),
        prompt_case("clone_text", "A new sentence.", speakers=1,
                    prompt_text="The reference words.", latents=5),
        prompt_case("clone_audio_only", "A new sentence.", speakers=1),
        prompt_case("no_speaker", "Plain text.", speakers=0),
    ]

    # ---- language model and stop head -------------------------------------
    case = fixture["prompts"][0]
    ids = torch.tensor([case["ids"]])
    embeds = model.get_input_embeddings()(ids).clone()
    embeds[0, case["speaker_rows"][0]] = 0.0
    outputs = model.model(
        inputs_embeds=embeds,
        position_ids=torch.arange(ids.shape[1])[None],
        output_hidden_states=True,
        use_cache=False,
    )
    hidden = outputs.hidden_states[-1][:, -1:, :]
    stop = model.stop_head(hidden)[0, 0].softmax(dim=-1)
    fixture["lm"] = {
        "prompt": case["name"],
        "hidden": sample(hidden),
        "hidden_full": [float(v) for v in hidden.flatten()],
        "stop_probability": float(stop[1]),
    }

    # ---- aggregator -------------------------------------------------------
    latents = torch.tensor(xorshift(3 * LATENT_DIM, 101, 1.5)).reshape(3, 1, LATENT_DIM)
    fixture["aggregator"] = {
        "latent_seed": 101, "latent_scale": 1.5, "rows": 3,
        "output": sample(model.linear_proj_audio(latents)),
    }

    # ---- flow head --------------------------------------------------------
    noise = torch.tensor(xorshift(LATENT_DIM, 202, 1.0)).reshape(1, 1, LATENT_DIM)
    history = torch.tensor(xorshift(LATENT_DIM, 203, 1.0)).reshape(1, 1, LATENT_DIM)
    latent = model.flowloss.cfm.sample(
        noise=noise, c=hidden, latent_history=history,
        sde_noise=torch.zeros(10, 1, 1, LATENT_DIM),
        cfg_scale=2.0, patch_size=1, sigma=0.25, temperature=0.0,
    )
    fixture["flow"] = {
        "noise_seed": 202, "history_seed": 203, "guidance": 2.0,
        "latent": [float(v) for v in latent.flatten()],
    }

    # ---- VAE decode, per attention implementation --------------------------
    decode_latents = torch.tensor(xorshift(40 * LATENT_DIM, 304, 1.0)).reshape(1, 40, LATENT_DIM)
    fixture["vae_decode"] = {"latent_seed": 304, "latents": 40, "waveforms": {}}
    implementations = ["sdpa"] + (["flash_attention_2"] if fixture["flash_attn"] else [])
    stack = model.audio.decoder.decoder
    for implementation in implementations:
        stack.config._attn_implementation = implementation
        if implementation == "flash_attention_2" and not torch.cuda.is_available():
            continue
        device = "cuda" if implementation == "flash_attention_2" else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        audio = model.audio.to(device=device, dtype=dtype)
        waveform = audio.decode(
            decode_latents.to(device=device, dtype=dtype),
            past_key_values=None, use_cache=False, last_chunk=True,
        )[0]
        fixture["vae_decode"]["waveforms"][implementation] = sample(waveform.float().cpu())
        model.audio.to(device="cpu", dtype=torch.float32)
    stack.config._attn_implementation = "sdpa"

    # ---- VAE encode and the speaker embedding of one clip ------------------
    if clip_path is not None:
        import soundfile
        import torchaudio

        data, rate = soundfile.read(str(clip_path), dtype="float32", always_2d=True)
        clip = torch.from_numpy(data).mean(dim=1)[None]
        if rate != SAMPLE_RATE:
            clip = torchaudio.transforms.Resample(rate, SAMPLE_RATE)(clip)
    else:
        t = torch.arange(3 * SAMPLE_RATE) / SAMPLE_RATE
        clip = (0.3 * torch.sin(2 * math.pi * 220 * t) * (0.5 + 0.5 * torch.sin(2 * math.pi * 3 * t)))[None]
    whole = (clip.shape[-1] // LATENT_SAMPLES) * LATENT_SAMPLES
    clip = clip[:, -whole:].contiguous()
    encoded, _ = model.audio.encoder(clip)
    fixture["vae_encode"] = {
        "clip_samples": [float(v) for v in clip.flatten()] if clip_path is None else None,
        "clip_sha256": hashlib.sha256(clip.numpy().tobytes()).hexdigest(),
        "clip_len": int(clip.shape[-1]),
        "mean": sample(encoded[..., :LATENT_DIM]),
        "scale": sample(encoded[..., LATENT_DIM:]),
    }
    onnx = model_dir / "campplus.onnx"
    if onnx.is_file():
        import torchaudio
        from spkemb_extractor import SpkembExtractor

        clip16 = torchaudio.transforms.Resample(SAMPLE_RATE, 16_000)(clip)
        embedding = SpkembExtractor(str(onnx))(clip16)
        fixture["speaker"] = {"embedding": [float(v) for v in embedding.flatten()]}
    if clip_path is not None:
        # The clip itself, for the izwi side to read back.
        fixture["vae_encode"]["clip_samples"] = [float(v) for v in clip.flatten()]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fixture, ensure_ascii=False))
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
