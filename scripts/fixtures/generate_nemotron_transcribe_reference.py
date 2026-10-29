#!/usr/bin/env python3
"""Generate the L2 real-weight Nemotron 3.5 Transcribe reference (GPU box).

Model: nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8 at revision
006d73cd8385237ed5603fc6c7e76200e50abbd2 (fetch it with
`scripts/e2e/nemotron-transcribe/fetch.sh`).

Clips: converted ONCE with ffmpeg to 16 kHz mono float32 and written beside
the fixture (`--audio-dir`), both as `<clip>.wav` (IEEE float WAV) and as
`<clip>.f32` (raw little-endian f32), so izwi, Leg A and Leg B read the same
samples (lessons 2026-10-25). Defaults read `data/` of this repo:
  fox          data/fox.wav                       (3.6 s)
  two_speakers data/diarization.wav               (27.3 s, 2 speakers)
  clip_95s     data/diarization-2.mp3, first 95 s (three online windows)
  ceo_600s     data/ceo.mp3, first 600 s          (15 windows)
Cases: fox, two_speakers, clip_95s and ceo_600s with izwi's default
"readable" instruction, plus two_speakers with the card's speaker-tag
instruction.

Leg A (default): an F32 tensor reference with dequantized FP8 weights (W8A16:
fp8_code * weight_scale in F32), built from upstream modules, the same
machinery as the L1 tiny generator (imported from
`generate_nemotron_transcribe_tiny.py`): NeMo Speech's perception (online
windows, streaming Sortformer) and transformers' NemotronH blocks patched to
vLLM semantics (no dt clamp, torch kernels, F32). transformers cannot load
this FP8 repo, so weights stream from the safetensors via mmap, layer by
layer; routed experts stay as FP8 codes (on the device with
`--expert-store device`, or read per use with `lazy`) and are dequantized
when a token routes to them (`RestatedExperts`, a verbatim restatement of
`NemotronHExperts.forward`). Leg A validates itself: cached decode equals the
full forward for the first steps, span prefill equals one-shot prefill, the
first Mamba layer equals an F64 recurrence and the first MoE layer an F64
restatement. With NeMo, it also runs the CTC timestamp head (log-prob slices)
and NeMo's aligner on its own transcripts.

Leg B (`--leg-b`): the official runtime, vLLM 0.32 + the NeMo plugin
(`VLLM_PLUGINS=nemo_speechlm`), started as `vllm serve` with the card's
flags minus speculative decoding and ReplaySSM, plus `--enforce-eager`;
temperature 0, `skip_special_tokens: false`, timestamps requested. It records
transcripts, token ids and top-8 logprobs (when the server allows them with
timestamps), and `ctc_timestamps`. The GPU name and SM tell which FP8 path
ran: SM89+ (L40S, H100, Blackwell) is W8A8 with FP8 KV, so compare izwi's
`IZWI_NEMOTRON_TRANSCRIBE_FP8_MODE=vllm-fp8`; SM80/86 (A100, A6000) is Marlin
W8A16, so compare izwi's default `w8a16`. Leg B runs before Leg A (the GPU
cannot hold both).

Leg A and Leg B may need different Python environments (transformers main
with NemotronH vs the NeMo/vLLM venv from the card): run one leg per
invocation with `--update` to merge into the same reference.json, and point
`--vllm-python` at the vLLM venv's interpreter.

Usage:
    python3 -I scripts/fixtures/generate_nemotron_transcribe_reference.py \
        --model-dir "$MODELS/Nemotron-3.5-Transcribe-30B-A4B-FP8" \
        [--out crates/izwi-core/tests/fixtures/nemotron_transcribe/reference.json] \
        [--audio-dir <out dir>/reference-audio] [--leg-b [--vllm-python PATH]] \
        [--skip-leg-a] [--update] [--device cuda|cpu] [--expert-store auto|device|lazy]

reference.json schema (`schema` = "izwi.nemotron_transcribe.reference",
`schema_version` = 1); floats are F32 shortest round-trip decimals:

{
  "schema", "schema_version", "generator", "created_at",
  "model": {"repo_id", "revision", "files": {name: bytes}},
  "versions": {"leg_a": {...}, "leg_b": {...}},
  "reference_semantics": {...},              # Leg A, as in the tiny fixture
  "instructions": {"readable": str, "speaker": str},   # izwi prompt.rs
  "clips": [{
    "name", "source", "source_sha256", "seconds_limit": float|null,
    "ffmpeg": {"version", "args"},
    "audio_file": "<name>.wav", "raw_file": "<name>.f32",   # in --audio-dir
    "audio_format": "wav pcm_f32le mono 16000 Hz (and the same samples raw)",
    "samples", "duration_secs", "sha256_wav", "sha256_f32",
    "perception": {                          # Leg A, same keys as the tiny
      "mel", "audio_tokens", "windows", "diar_near_threshold",   # fixture's
      "fused_rows", "embeddings_rows",                           # per-clip
      "ctc_log_probs": {"shape", "valid_frames", "blank_id",     # fields
                        "head_dtype", "residual_scale",
                        "frames": [{"frame", "blank_logprob",    # 40 ms frames
                                    "top8": [{"id", "logprob"}]}],   # 0, 1/4,
                        "greedy_nonblank_ids": [first 128]}      # 1/2, 3/4,
                       | {"error"}}                              # last valid
  }],
  "cases": [{
    "name", "clip", "instruction_kind": "readable"|"speaker", "instruction",
    "leg_a": {
      "prompt_ids", "audio_offset", "prompt_tokens",
      "layer_hidden",                        # layers 0 (mamba), 1 (moe),
                                             # 5 (attention); see tiny schema
      "routes",                              # layer 1, sampled rows, near
                                             # ties per MoE layer
      "prefill_logits_top8", "prefill_margin",
      "greedy_ids" (first 64), "greedy_margins" (first 64),
      "generated_tokens", "finish_reason", "stop_token",
      "decoded", "text",                     # the full greedy transcript
      "ctc_timestamps": {...} | {"error"},   # NeMo aligner on Leg A's text
      "self_checks", "timings"},
    "leg_b": {
      "content", "reasoning_content", "text",   # text = guardrail(content)
      "finish_reason", "prompt_tokens", "completion_tokens",
      "token_ids": [...]|null, "top_logprobs": [[{"id", "logprob"}]] (16),
      "ctc_timestamps": {words, diarization, speaker_tag_to_diarization_
                         speaker, error} | null,
      "request_notes": [...], "elapsed_s"}
  }],
  "leg_b_meta": {"gpu_name", "sm", "fp8_path", "vllm_version", "command",
                 "env", "log_file", "log_matches"},
  "suggested_tolerances": {...},
  "self_checks": {...}
}
"""
from __future__ import annotations

import argparse
import base64
import dataclasses
import datetime
import hashlib
import importlib.util
import json
import os
import shlex
import signal
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_tiny():
    spec = importlib.util.spec_from_file_location(
        "nemotron_transcribe_tiny", HERE / "generate_nemotron_transcribe_tiny.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


T = _load_tiny()
np, torch = T.np, T.torch
log = T.log

ROOT = T.ROOT
DEFAULT_OUT = T.FIXTURE_DIR / "reference.json"
SCHEMA = "izwi.nemotron_transcribe.reference"
SCHEMA_VERSION = 1
REPO_ID = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8"
REVISION = "006d73cd8385237ed5603fc6c7e76200e50abbd2"
NEMO_COMMIT = "743beaf4276e3f28740f3cb1281440d3d6f7eb76"
SERVED_NAME = "nemotron-transcribe"
# Exact sizes at the pinned revision (plan §2.1).
FILE_SIZES = {
    "model-00001-of-00004.safetensors": 9_997_859_176,
    "model-00002-of-00004.safetensors": 10_000_407_944,
    "model-00003-of-00004.safetensors": 9_998_941_520,
    "model-00004-of-00004.safetensors": 2_296_485_296,
    "perception.safetensors": 838_256_800,
    "model-ctc-bf16.safetensors": 201_897_998,
    "model-mtp-fp8.safetensors": 1_335_778_712,
    "model.safetensors.index.json": 1_975_374,
    "config.json": 433_812,
    "hf_quant_config.json": 2_275,
    "generation_config.json": 92,
    "tokenizer.json": 17_078_144,
    "tokenizer_config.json": 491,
    "chat_template.jinja": 12_482,
}
CLIPS = [
    # name, file under --clips-dir, seconds to keep (None: all)
    ("fox", "fox.wav", None),
    ("two_speakers", "diarization.wav", None),
    ("clip_95s", "diarization-2.mp3", 95.0),
    ("ceo_600s", "ceo.mp3", 600.0),
]
CASES = [
    ("fox_readable", "fox", "readable"),
    ("two_speakers_readable", "two_speakers", "readable"),
    ("two_speakers_speakers", "two_speakers", "speaker"),
    ("clip_95s_readable", "clip_95s", "readable"),
    ("ceo_600s_readable", "ceo_600s", "readable"),
]
INSTRUCTIONS = {"readable": T.READABLE_INSTRUCTION, "speaker": T.SPEAKER_INSTRUCTION}
GREEDY_RECORD = 64
TOP_LOGPROB_STEPS = 16

VERSION_SNIPPET = r"""
import importlib.metadata as m, importlib.util, json, pathlib, subprocess
out = {}
for name, dist in [("vllm", "vllm"), ("nemo", "nemo_toolkit"), ("torch", "torch"),
                   ("transformers", "transformers"), ("flashinfer", "flashinfer-python")]:
    try:
        out[name] = m.version(dist)
    except Exception:
        out[name] = None
spec = importlib.util.find_spec("nemo")
if spec and spec.origin:
    r = subprocess.run(["git", "-C", str(pathlib.Path(spec.origin).parent), "rev-parse", "HEAD"],
                       capture_output=True, text=True)
    out["nemo_git"] = r.stdout.strip() or None
try:
    import torch
    if torch.cuda.is_available():
        out["gpu_name"] = torch.cuda.get_device_name(0)
        out["sm"] = "%d.%d" % torch.cuda.get_device_capability(0)
        out["torch_cuda"] = torch.version.cuda
except Exception as error:
    out["gpu_error"] = str(error)
print(json.dumps(out))
"""


# ------------------------------------------------------------------ audio

def read_float_wav(path):
    """(rate, mono float32 samples) of an IEEE-float WAV (ffmpeg pcm_f32le)."""
    data = Path(path).read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise RuntimeError(f"{path} is not a RIFF/WAVE file")
    pos, fmt, samples = 12, None, None
    while pos + 8 <= len(data):
        chunk = data[pos:pos + 4]
        size = int.from_bytes(data[pos + 4:pos + 8], "little")
        body = data[pos + 8:pos + 8 + size]
        if chunk == b"fmt ":
            tag, channels, rate = struct.unpack("<HHI", body[:8])
            bits = struct.unpack("<H", body[14:16])[0]
            if tag == 0xFFFE:
                tag = struct.unpack("<H", body[24:26])[0]
            fmt = (tag, channels, rate, bits)
        elif chunk == b"data":
            samples = np.frombuffer(body, dtype="<f4")
        pos += 8 + size + (size & 1)
    if fmt is None or samples is None or fmt[0] != 3 or fmt[1] != 1 or fmt[3] != 32:
        raise RuntimeError(f"{path}: expected mono 32-bit float WAV, found {fmt}")
    return fmt[2], samples.astype(np.float32)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def prepare_clips(args):
    ffmpeg = args.ffmpeg
    version = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True)
    if version.returncode != 0:
        raise RuntimeError(f"{ffmpeg} -version failed: {version.stderr}")
    ffmpeg_version = version.stdout.splitlines()[0]
    args.audio_dir.mkdir(parents=True, exist_ok=True)
    overrides = dict(item.split("=", 1) for item in args.clip)
    clips = {}
    for name, file, seconds in CLIPS:
        source = Path(overrides.get(name, args.clips_dir / file))
        if not source.is_file():
            raise RuntimeError(f"clip {name}: {source} does not exist (use --clip {name}=PATH)")
        wav = args.audio_dir / f"{name}.wav"
        raw = args.audio_dir / f"{name}.f32"
        cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source)]
        if seconds:
            cmd += ["-t", f"{seconds:g}"]
        cmd += ["-ac", "1", "-ar", "16000", "-c:a", "pcm_f32le", "-f", "wav", str(wav)]
        if not wav.is_file() or args.reconvert:
            subprocess.run(cmd, check=True)
        rate, samples = read_float_wav(wav)
        if rate != 16000:
            raise RuntimeError(f"{wav}: rate {rate}")
        samples.astype("<f4").tofile(raw)
        clips[name] = {
            "name": name,
            "source": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
            "source_sha256": T.sha256_file(source),
            "seconds_limit": seconds,
            "ffmpeg": {"version": ffmpeg_version, "args": cmd[1:-1] + ["<out>"]},
            "audio_file": wav.name,
            "raw_file": raw.name,
            "audio_format": "wav pcm_f32le mono 16000 Hz (and the same samples raw in .f32)",
            "samples": int(len(samples)),
            "duration_secs": len(samples) / 16000,
            "sha256_wav": T.sha256_file(wav),
            "sha256_f32": T.sha256_file(raw),
            "_samples": samples,
        }
        log(f"clip {name}: {len(samples) / 16000:.2f} s")
    return clips


# ------------------------------------------------------------------ Leg B

def http_json(url, body=None, timeout=3600):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method="GET" if body is None else "POST",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read() or b"null")


def leg_b(args, clips, cases):
    env_versions = json.loads(subprocess.run(
        [args.vllm_python, "-c", VERSION_SNIPPET], capture_output=True, text=True,
        check=True).stdout.strip().splitlines()[-1])
    sm = env_versions.get("sm")
    if sm is None:
        fp8_path = "unknown (no CUDA device visible to the vLLM environment)"
    elif tuple(int(v) for v in sm.split(".")) >= (8, 9):
        fp8_path = ("w8a8: FP8 tensor cores with static per-tensor activation scales, q/k/v "
                    "requantized to the max weight scale, FP8 KV cache; compare izwi with "
                    "IZWI_NEMOTRON_TRANSCRIBE_FP8_MODE=vllm-fp8")
    else:
        fp8_path = ("marlin-w8a16: weight-only FP8, no activation quantization (KV cache "
                    "still FP8 when supported); compare izwi's default w8a16")
    env = dict(os.environ, VLLM_PLUGINS="nemo_speechlm", VLLM_MAX_AUDIO_DECODE_DURATION_S="3600",
               VLLM_MAX_AUDIO_CLIP_FILESIZE_MB="180", VLLM_USE_V2_MODEL_RUNNER="0",
               VLLM_DISABLED_KERNELS="FlashInferFP8ScaledMMLinearKernel")
    serve = [
        "serve", str(args.model_dir), "--served-model-name", SERVED_NAME, "--trust-remote-code",
        "--dtype", "bfloat16", "--kv-cache-dtype", "auto",
        "--gpu-memory-utilization", str(args.vllm_gpu_memory_utilization),
        "--max-model-len", "65536", "--max-num-batched-tokens", "4096", "--max-num-seqs", "4",
        "--enable-chunked-prefill", "--no-enable-prefix-caching", "--mamba-cache-mode", "none",
        "--mm-processor-cache-gb", "0", "--no-async-scheduling",
        "--limit-mm-per-prompt", json.dumps({"audio": {"count": 1, "length": 57_600_000}}),
        "--reasoning-parser", "nemotron_v3", "--trust-request-mm-kwargs",
        "--middleware", "nemo.collections.speechlm2.vllm.salm.ctc_serving.ctc_timestamp_middleware",
        "--enforce-eager", "--host", "127.0.0.1", "--port", str(args.vllm_port),
    ] + shlex.split(args.vllm_extra_args)
    launcher = ("import sys; from vllm.entrypoints.cli.main import main; "
                "sys.argv = ['vllm'] + sys.argv[1:]; main()")
    command = [args.vllm_python, "-c", launcher] + serve
    log_file = args.out.parent / "vllm-leg-b.log"
    base = f"http://127.0.0.1:{args.vllm_port}"
    log(f"Leg B: starting vLLM ({env_versions.get('vllm')}) on {env_versions.get('gpu_name')} "
        f"SM {sm}; log {log_file}")
    with open(log_file, "w") as handle:
        process = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    results = {}
    try:
        deadline = time.time() + args.vllm_startup_timeout
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"vLLM exited with {process.returncode}; see {log_file}")
            try:
                urllib.request.urlopen(base + "/health", timeout=5).read()
                break
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                if time.time() > deadline:
                    raise RuntimeError(f"vLLM not healthy after {args.vllm_startup_timeout} s")
                time.sleep(5)
        for name, clip_name, kind in cases:
            clip = clips[clip_name]
            audio = base64.b64encode((args.audio_dir / clip["audio_file"]).read_bytes()).decode()
            results[name] = leg_b_case(base, INSTRUCTIONS[kind], audio, args.max_new_tokens)
            log(f"Leg B {name}: {results[name]['finish_reason']} "
                f"{results[name]['completion_tokens']} tokens: {results[name]['text'][:80]!r}")
    finally:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=120)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
    matches = []
    for line in log_file.read_text(errors="replace").splitlines():
        lower = line.lower()
        if any(key in lower for key in ("quantization=", "modelopt", "fp8", "kv cache dtype",
                                        "kv_cache_dtype", "marlin", "cutlass")):
            matches.append(line.strip()[:400])
    meta = {
        "gpu_name": env_versions.get("gpu_name"), "sm": sm, "fp8_path": fp8_path,
        "vllm_version": env_versions.get("vllm"),
        "command": [args.vllm_python, "-c", "<vllm cli>"] + serve,
        "env": {k: env[k] for k in ("VLLM_PLUGINS", "VLLM_MAX_AUDIO_DECODE_DURATION_S",
                                    "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB", "VLLM_USE_V2_MODEL_RUNNER",
                                    "VLLM_DISABLED_KERNELS")},
        "log_file": log_file.name, "log_matches": matches[:200],
    }
    return results, meta, env_versions


def leg_b_case(base, instruction, audio_b64, max_tokens):
    body = {
        "model": SERVED_NAME,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": instruction},
            {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"}},
        ]}],
        "temperature": 0, "top_p": 1.0, "max_tokens": max_tokens, "seed": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "skip_special_tokens": False, "spaces_between_special_tokens": False,
        "logprobs": True, "top_logprobs": 8, "return_tokens_as_token_ids": True,
        "mm_processor_kwargs": {"capture_ctc_timestamps": True},
    }
    notes = []
    attempts = [body,
                {k: v for k, v in body.items() if k not in ("logprobs", "top_logprobs",
                                                            "return_tokens_as_token_ids")},
                {k: v for k, v in body.items() if k not in ("logprobs", "top_logprobs",
                                                            "return_tokens_as_token_ids",
                                                            "mm_processor_kwargs")}]
    response, started = None, time.time()
    for attempt in attempts:
        try:
            started = time.time()
            response = http_json(base + "/v1/chat/completions", attempt, timeout=7200)
            break
        except urllib.error.HTTPError as error:
            notes.append(f"{error.code}: {error.read().decode(errors='replace')[:500]}")
    if response is None:
        raise RuntimeError(f"every Leg B request failed: {notes}")
    elapsed = time.time() - started
    choice = response["choices"][0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    token_ids, top = None, []
    logprobs = (choice.get("logprobs") or {}).get("content")
    if logprobs:
        token_ids = [_token_id(entry.get("token")) for entry in logprobs]
        for entry in logprobs[:TOP_LOGPROB_STEPS]:
            top.append([{"id": _token_id(t.get("token")), "logprob": T.f32(t.get("logprob"))}
                        for t in entry.get("top_logprobs") or []])
    usage = response.get("usage") or {}
    return {
        "content": content,
        "reasoning_content": message.get("reasoning_content") or message.get("reasoning"),
        "text": T.guardrail(content.strip()),
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "token_ids": token_ids,
        "top_logprobs": top,
        "ctc_timestamps": response.get("ctc_timestamps"),
        "request_notes": notes,
        "elapsed_s": round(elapsed, 3),
    }


def _token_id(token):
    if isinstance(token, str) and token.startswith("token_id:"):
        return int(token.split(":", 1)[1])
    return token


# ------------------------------------------------------------------ Leg A

class ExpertStore:
    """Routed experts as FP8 codes: all on the device (`device`) or read from
    the mapped safetensors per use (`lazy`); dequantized to F32 per call."""

    def __init__(self, src, device, mode):
        self.src, self.device, self.mode = src, device, mode
        self.codes = {}

    def fetcher(self, layer):
        prefix = f"backbone.layers.{layer}.mixer.experts"
        if self.mode == "device":
            index = 0
            while self.src.has(f"{prefix}.{index}.up_proj.weight"):
                entry = []
                for which in ("up_proj", "down_proj"):
                    module = f"{prefix}.{index}.{which}"
                    entry.append(self.src.tensor(f"{module}.weight").to(self.device))
                    entry.append(self.src.tensor(f"{module}.weight_scale")
                                 .to(self.device, torch.float32))
                self.codes[(layer, index)] = entry
                index += 1

        def fetch(expert):
            if self.mode == "device":
                up, up_scale, down, down_scale = self.codes[(layer, expert)]
                return up.to(torch.float32) * up_scale, down.to(torch.float32) * down_scale
            return (T.dequantized(self.src, f"{prefix}.{expert}.up_proj", self.device),
                    T.dequantized(self.src, f"{prefix}.{expert}.down_proj", self.device))

        return fetch


def to_host(value):
    """A CTCTimestampInputs (or any dataclass of tensors) moved to the CPU."""
    if dataclasses.is_dataclass(value):
        changes = {f.name: to_host(getattr(value, f.name)) for f in dataclasses.fields(value)}
        return dataclasses.replace(value, **changes)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def jsonable(value):
    if isinstance(value, torch.Tensor):
        return jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return round(v, 6) if np.isfinite(v) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def ctc_log_prob_slices(pee, inputs):
    from nemo.collections.speechlm2.parts.ctc_timestamp_utils import get_ctc_timestamp_aligner

    aligner = get_ctc_timestamp_aligner(pee, pee.ctc_timestamp_model_path,
                                        inputs.asr_encoded.device)
    logp = aligner.ctc_log_probs_from_inputs(inputs)
    valid = int(aligner._ctc_frames(inputs.asr_encoded_lengths)[0])
    blank = logp.shape[-1] - 1
    frames = sorted({0, valid // 4, valid // 2, 3 * valid // 4, max(valid - 1, 0)})
    rows = []
    for frame in frames:
        top = torch.topk(logp[0, frame].to(torch.float32), 8)
        rows.append({"frame": frame, "blank_logprob": T.f32(logp[0, frame, blank]),
                     "top8": [{"id": int(i), "logprob": T.f32(v)}
                              for v, i in zip(top.values, top.indices)]})
    greedy = torch.argmax(logp[0, :valid], dim=-1).tolist()
    collapsed = [t for i, t in enumerate(greedy) if t != blank and (i == 0 or greedy[i - 1] != t)]
    scale = None
    for module in aligner.ctc_decoder.modules():
        value = getattr(module, "residual_scale", None)
        if isinstance(value, torch.Tensor):
            scale = float(value.reshape(-1)[0])
        elif isinstance(value, float):
            scale = value
    head = next(aligner.ctc_decoder.parameters())
    record = {"shape": list(logp.shape), "valid_frames": valid, "blank_id": blank,
              "head_dtype": str(head.dtype), "residual_scale": scale, "frames": rows,
              "greedy_nonblank_ids": collapsed[:128]}
    del logp
    return record


def choose_expert_store(mode, device):
    if mode != "auto":
        return mode
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        # 27.4 GiB of FP8 expert codes + about 12 GiB of F32 trunk + work.
        return "device" if free > 46 * (1 << 30) else "lazy"
    return "lazy"


def leg_a(args, clips, cases):
    from tokenizers import Tokenizer

    T.disable_tf32()
    device = torch.device(args.device)
    cfg = json.loads((args.model_dir / "config.json").read_text())
    g = T.Geometry(T.decode_nemo_floats(cfg))
    src = T.SafetensorsSource(args.model_dir)
    tokenizer = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    if tokenizer.token_to_id(T.AUDIO_TOKEN) != g.vocab:
        raise RuntimeError("<|audio|> is not one past the embedding table")
    timings = {}

    started = time.time()
    if args.audio_impl == "nemo":
        perception = T.build_nemo_perception(cfg, device)
        runner = T.run_nemo_perception
    else:
        perception = T.build_sglang_perception(cfg, device)
        runner = T.run_sglang_perception_impl
    T.load_perception(perception, src, device)
    ctc_path = args.model_dir / "model-ctc-bf16.safetensors"
    use_ctc = args.audio_impl == "nemo" and not args.no_ctc and ctc_path.is_file()
    if use_ctc:
        perception.encoder.ctc_timestamp_model_path = str(ctc_path)
    timings["perception_load_s"] = round(time.time() - started, 2)
    perceived = {}
    for name, clip in clips.items():
        started = time.time()
        if use_ctc:
            result = runner(perception, clip["_samples"], device, ctc=True)
        else:
            result = runner(perception, clip["_samples"], device)
        record = T.perception_record(result, g)
        if use_ctc:
            try:
                record["ctc_log_probs"] = ctc_log_prob_slices(perception.encoder,
                                                              result.ctc_inputs)
            except Exception as error:  # noqa: BLE001
                record["ctc_log_probs"] = {"error": f"{type(error).__name__}: {error}"}
            result.ctc_inputs = to_host(result.ctc_inputs)
        perceived[name] = (result, record)
        timings[f"perception_{name}_s"] = round(time.time() - started, 2)
        log(f"Leg A perception {name}: {record['audio_tokens']} audio tokens, "
            f"{len(record['windows'])} windows")
    perception.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    store_mode = choose_expert_store(args.expert_store, device)
    started = time.time()
    store = ExpertStore(src, device, store_mode)
    decoder = T.build_decoder(cfg, src, device, store.fetcher, log_every=4)
    timings["decoder_load_s"] = round(time.time() - started, 2)
    log(f"Leg A decoder loaded ({store_mode} experts) in {timings['decoder_load_s']} s")
    checks = {
        "mamba_time_step_limits": sorted({str(tuple(b.mixer.time_step_limit))
                                          for b in decoder.blocks
                                          if b.block_type == "linear_attention"}),
        "moe_f64_rel_l2": T.moe_f64_check(decoder),
    }
    failures = []
    if checks["mamba_time_step_limits"] != [str((0.0, float("inf")))]:
        failures.append("Mamba dt clamp was not patched")
    if not checks["moe_f64_rel_l2"] < 1e-5:
        failures.append(f"MoE restatement mismatch {checks['moe_f64_rel_l2']:.3e}")

    trace_layers = [0, 1, g.first("attention")]
    route_layer = g.first("moe")
    span = args.prefill_span or (4096 if store_mode == "lazy" else 1024)
    results = {}
    for index, (name, clip_name, kind) in enumerate(cases):
        result, _ = perceived[clip_name]
        started = time.time()
        first = index == 0
        case, case_checks, (embeds, _) = T.run_decoder_case(
            decoder, tokenizer, INSTRUCTIONS[kind], result.embeddings, GREEDY_RECORD,
            args.max_new_tokens, trace_layers, route_layer, positions_cap=96, span=span,
            cached_checks=(1, 4, 16) if first else ())
        case_timings = {"decode_s": round(time.time() - started, 2)}
        if first and embeds.shape[1] <= 4096:
            case_checks["mamba_f64_rel_l2"] = T.mamba_f64_check(decoder, embeds)
            _, one_shot = decoder.prefill(embeds)
            _, spans = decoder.prefill(embeds, span=17)
            case_checks["span17_vs_one_shot_prefill_max_abs"] = float(
                (one_shot - spans).abs().max())
            case_checks["prefill_logit_scale"] = float(one_shot.abs().max())
            if not case_checks["mamba_f64_rel_l2"] < 1e-5:
                failures.append(f"{name}: Mamba F64 mismatch {case_checks['mamba_f64_rel_l2']:.3e}")
            if not case_checks["span17_vs_one_shot_prefill_max_abs"] <= \
                    1e-4 * max(case_checks["prefill_logit_scale"], 1.0):
                failures.append(f"{name}: span prefill differs from one-shot prefill")
        for key, value in list(case_checks.items()):
            if key.startswith("cached_vs_full") and key.endswith("_max_abs"):
                scale = case_checks[key.replace("_max_abs", "_scale")]
                if not value <= 1e-4 * max(scale, 1.0):
                    failures.append(f"{name}: {key} = {value:.3e}")
        if use_ctc and result.ctc_inputs is not None:
            try:
                aligned = perception.encoder.generate_ctc_timestamps(
                    result.ctc_inputs, [case["decoded"]], [clips[clip_name]["duration_secs"]])
                case["ctc_timestamps"] = jsonable(aligned[0])
            except Exception as error:  # noqa: BLE001
                case["ctc_timestamps"] = {"error": f"{type(error).__name__}: {error}"}
        case["self_checks"] = case_checks
        case["timings"] = case_timings
        case.pop("instruction")
        results[name] = case
        log(f"Leg A {name}: {case['generated_tokens']} tokens ({case['finish_reason']}) in "
            f"{case_timings['decode_s']} s: {case['text'][:80]!r}")
    semantics = {
        "weights": "W8A16: E4M3 code * F32 per-tensor weight_scale, multiplied in F32; BF16 "
                   "tensors upcast exactly; input/k/v scales unused",
        "compute": "float32 throughout, TF32 off, eager attention; residual stream F32",
        "bf16_rounding_points": ["log-mel output", "per-feature normalization (torch BF16 "
                                 "semantics)", "normalized ASR-encoder input",
                                 "RoPE cos/sin tables", "speaker kernel"],
        "mamba": "transformers torch path; time_step_limit (0, inf); dt, y, state F32",
        "experts": f"RestatedExperts over FP8 codes ({store_mode}); router and shared expert "
                   "are transformers' modules",
        "prefill": f"spans of {span} tokens (izwi-like resumable prefill)",
        "ctc": "NeMo TransformerCTCDecoder at the encoder's F32 dtype (upstream serving runs "
               "BF16 with residual scale bf16(0.1) = 0.10009765625; this reference keeps 0.1)",
        "audio_reference": args.audio_impl,
    }
    return results, perceived, checks, failures, semantics, timings


# ------------------------------------------------------------------- main

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--audio-dir", type=Path,
                        help="where the converted clips go (default <out dir>/reference-audio)")
    parser.add_argument("--clips-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--clip", action="append", default=[], metavar="NAME=PATH",
                        help="override one clip's source (fox, two_speakers, clip_95s, ceo_600s)")
    parser.add_argument("--cases", help="comma-separated case names (default all)")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--reconvert", action="store_true",
                        help="re-run ffmpeg even when the converted WAV exists")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--audio-impl", choices=("nemo", "sglang"), default="nemo")
    parser.add_argument("--expert-store", choices=("auto", "device", "lazy"), default="auto")
    parser.add_argument("--prefill-span", type=int, default=0,
                        help="Leg A prefill span (default 1024, 4096 with lazy experts)")
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--no-ctc", action="store_true", help="skip Leg A's CTC head")
    parser.add_argument("--skip-leg-a", action="store_true")
    parser.add_argument("--leg-b", action="store_true", help="run the vLLM + NeMo plugin leg")
    parser.add_argument("--vllm-python", default=sys.executable,
                        help="Python of the vLLM 0.32 + NeMo Speech environment")
    parser.add_argument("--vllm-port", type=int, default=8017)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--vllm-startup-timeout", type=int, default=3600)
    parser.add_argument("--vllm-extra-args", default="", help="appended to `vllm serve`")
    parser.add_argument("--update", action="store_true",
                        help="merge into an existing --out instead of starting afresh")
    parser.add_argument("--allow-size-mismatch", action="store_true")
    parser.add_argument("--no-strict", action="store_true")
    args = parser.parse_args(argv)
    args.audio_dir = args.audio_dir or args.out.parent / "reference-audio"
    return args


def check_model_dir(args):
    files, problems = {}, []
    for name, size in FILE_SIZES.items():
        path = args.model_dir / name
        found = path.stat().st_size if path.is_file() else None
        files[name] = found
        if found != size:
            problems.append(f"{name}: {found} bytes, pinned {size}")
    if problems:
        message = "model directory differs from the pinned revision: " + "; ".join(problems)
        if not args.allow_size_mismatch:
            raise RuntimeError(message)
        log(message)
    return files


def main(argv=None):
    args = parse_args(argv)
    files = check_model_dir(args)
    selected = [c for c in CASES if not args.cases or c[0] in args.cases.split(",")]
    if not selected:
        raise SystemExit("no case selected")
    clip_names = {c[1] for c in selected}
    clips = {k: v for k, v in prepare_clips(args).items() if k in clip_names}

    fixture = {}
    if args.update and args.out.is_file():
        fixture = json.loads(args.out.read_text())
        if fixture.get("schema") != SCHEMA:
            raise RuntimeError(f"{args.out} is not a {SCHEMA} file")
    fixture.update({
        "schema": SCHEMA, "schema_version": SCHEMA_VERSION,
        "generator": "scripts/fixtures/generate_nemotron_transcribe_reference.py",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "model": {"repo_id": REPO_ID, "revision": REVISION, "files": files},
        "instructions": INSTRUCTIONS,
    })
    fixture.setdefault("versions", {})
    fixture.setdefault("self_checks", {})
    clip_records = {c["name"]: c for c in fixture.get("clips", [])}
    for name, clip in clips.items():
        merged = clip_records.get(name, {})
        merged.update({k: v for k, v in clip.items() if not k.startswith("_")})
        clip_records[name] = merged
    case_records = {c["name"]: c for c in fixture.get("cases", [])}
    for name, clip_name, kind in selected:
        case_records.setdefault(name, {}).update(
            {"name": name, "clip": clip_name, "instruction_kind": kind,
             "instruction": INSTRUCTIONS[kind]})

    failures = []
    if args.leg_b:
        results, meta, env_versions = leg_b(args, clips, selected)
        for name, record in results.items():
            case_records[name]["leg_b"] = record
        fixture["leg_b_meta"] = meta
        fixture["versions"]["leg_b"] = env_versions
        if env_versions.get("nemo_git") and env_versions["nemo_git"] != NEMO_COMMIT:
            log(f"note: Leg B ran NeMo {env_versions['nemo_git']}, the plan pins {NEMO_COMMIT}")
    if not args.skip_leg_a:
        results, perceived, checks, leg_failures, semantics, timings = leg_a(args, clips, selected)
        failures += leg_failures
        for name, record in results.items():
            case_records[name]["leg_a"] = record
        for name, (_, record) in perceived.items():
            clip_records[name]["perception"] = record
        fixture["versions"]["leg_a"] = T.collect_versions()
        fixture["reference_semantics"] = semantics
        fixture["self_checks"]["leg_a"] = dict(checks, failures=leg_failures, timings=timings)
        nemo_git = fixture["versions"]["leg_a"].get("nemo_git")
        if nemo_git and nemo_git != NEMO_COMMIT:
            log(f"note: Leg A ran NeMo {nemo_git}, the plan pins {NEMO_COMMIT}")
    order = [c[0] for c in CASES]
    fixture["clips"] = [clip_records[n] for n, _, _ in CLIPS if n in clip_records]
    fixture["cases"] = sorted(case_records.values(),
                              key=lambda c: order.index(c["name"]) if c["name"] in order else 99)
    fixture["suggested_tolerances"] = T.suggested_tolerances("L2")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    T.dump_json(fixture, args.out)
    log(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB); audio in {args.audio_dir}")
    for case in fixture["cases"]:
        a = (case.get("leg_a") or {}).get("text")
        b = (case.get("leg_b") or {}).get("text")
        if a is not None and b is not None:
            log(f"{case['name']}: Leg A {'==' if a == b else '!='} Leg B")
    if failures:
        log("SELF-CHECK FAILURES:\n  " + "\n  ".join(failures))
        if not args.no_strict:
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
