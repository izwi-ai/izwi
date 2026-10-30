#!/usr/bin/env python3
"""Shared helpers for the Fish Audio Drama 3 parity-fixture scripts.

The scripts import this file by path (they run under `python3 -I`, which keeps
the script directory off `sys.path`):

  generate_fish_drama3_tiny.py        L1   tiny upstream fixture
  fetch_fish_drama3_slices.py         L1.5 range-download of real-weight slices
  generate_fish_drama3_slices.py      L1.5 slice outputs
  generate_fish_drama3_reference.py   L2   whole-model reference

Importing it needs only the standard library: torch and fish-speech are
imported inside the functions that use them, so the fetch script runs without
torch.

Shared definitions (each script's docstring refers to these)
------------------------------------------------------------
xorshift (`xorshift_values`): izwi's `kernels::numeric_check::xorshift_values`,
bit for bit:

    state = ((seed * 0x9E3779B97F4A7C15) mod 2^64) | 1
    repeat len times:
        state ^= (state << 13) mod 2^64
        state ^= state >> 7
        state ^= (state << 17) mod 2^64
        unit  = float32(state >> 40) / 2^24              # exact
        value = float32((unit * 2 - 1) * float32(scale)) # one round-half-even
    values lie in [-scale, scale)

checksum (`checksum`), over every element of an F32 tensor, accumulated in
F64 (the order of the sum is not part of the contract; compare with a
relative tolerance):

    {"count": n, "sum": Σx, "abs_sum": Σ|x|, "sum_sq": Σx², "max_abs": max|x|,
     "rms": sqrt(sum_sq / n)}

trace entry (`trace_entry`): one recorded activation `[rows, width]`:

    {"shape": [rows, width],
     "rows": [r...],                 # the rows whose values are stored
     "values": <tensor name>,        # F32 [len(rows), width]
     "row_rms": <tensor name>,       # F32 [rows]: sqrt(mean_j x[r, j]^2) per row
     "checksum": {...}}              # over every row

Tensor names point into the script's sidecar safetensors file (F32 or I64).

Recorded noise (`NoiseTape`, `UpstreamSampler`): upstream's sampler
(`fish_omni/tower.py:47-73`) draws `Exp(1)` noise in RANK order (after its
descending sort); izwi's race reads one exponential per candidate in CANDIDATE
order (`sampling.rs`). While installed, `UpstreamSampler` replaces
`tower.sample_top_k_top_p` with the same body (`tower.py:61-69` verbatim) whose
noise line reads `E[ids]`, where `E` holds one value per candidate in
candidate order from a seeded `torch.Generator` (CPU, `exponential_`). So
`argmax(p_rank / E[ids])` picks exactly the candidate izwi's
`argmax(p_candidate / E_candidate)` picks. The tape keeps only the values izwi
consumes, in its order. Per decode step:

    E_first[C]   main token, request policy; candidates eos, then the band
    E_retry[C]   the RAS draw (1.0, top-k 30, top-p 0.95), always drawn
    E_head1[V] .. E_head9[V]   codebook heads 1..9, codes 0..V-1

A masked candidate still has its value. On the step whose main token is eos,
upstream still runs the tower and draws the 9 heads; izwi stops after the two
main draws, so those 9*V values are cut from the tape (they come last in the
generator's stream, and nothing after them is used): `noise` length is
`steps * 2C + frames * 9V`. Replays run with izwi's first-frame eos mask OFF
(upstream has none).
"""

from __future__ import annotations

import datetime
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import struct
import subprocess
import sys
from array import array
from pathlib import Path
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# Pins
# ---------------------------------------------------------------------------

REPO_ID = "fishaudio/drama-3"
CHECKPOINT_REVISION = "78e5ab79e49aa9a25abe8799eff2c4d73a1c3216"
FISH_SPEECH_COMMIT = "0f8a12e8a651062fdee9c33a894bdccd3a8513f1"
FISH_SPEECH_REPO = "https://github.com/fishaudio/fish-speech"
# sha256 of the small pinned files (LFS sha for tokenizer.json).
PINNED_SHA256 = {
    "config.json": "aea53f0f2b9a15023fc0f8eb0f42360663671c99814037bada6867d508010111",
    "tokenizer.json": "0e9b9db76b2f0389b04b6ef057a2ac3b89100b7e96f2dd7034d19580a09654d6",
    "model.safetensors.index.json": "a8e9aca4e9fddca51f03fad63f52c82c470e98e28b6b0b1b1562dbe6456b50bb",
}
# Shard sizes and LFS sha256 at the pin (HF API `siblings`).
SHARDS = {
    "model-00001-of-00015.safetensors": (2862881576, "a1b7af15210d2eda2d75a2cf976295f80e345855b745b109aa3ce49d26a44497"),
    "model-00002-of-00015.safetensors": (2628610248, "a95e113c6f32f8fcb23ecb618cd4384fad6358f3c0ac31cfbf352daf5370db94"),
    "model-00003-of-00015.safetensors": (2628610224, "a34d0952d6369f5c73e6ae06138ef96bc6a3af9bb259c4e41425a33706cf5b68"),
    "model-00004-of-00015.safetensors": (2628611792, "13ac368d87441a18752e424e485b711a115ea1084e3a18ab13276e73eedef8b4"),
    "model-00005-of-00015.safetensors": (2641525264, "c432fa0dac97471f0d625bc67eb94fe084ca2e5ecd8865f457b751479b50e363"),
    "model-00006-of-00015.safetensors": (2628614896, "661a31433747fa9d029632a61350ea9eb105bf7276d013ff0901b6a939fb2dcb"),
    "model-00007-of-00015.safetensors": (2628614880, "4a7574b5c238f424b37aa945a736da1994a26bd38b86bf3ec4abf524c0969cc8"),
    "model-00008-of-00015.safetensors": (2628614888, "7fac6d1e899939eb03c9cf28a33b0bfd9c7a599298662f26d9a59d64c96d007f"),
    "model-00009-of-00015.safetensors": (2641525264, "fcf1313744ea5978e363b2bb49c3b66748d886d79f3ec102f4346a462385de4b"),
    "model-00010-of-00015.safetensors": (2628614896, "6db3f7cd9d49806ae4f17b6f6d17a0c8dd9686aee57728da1b50ec0c6139a629"),
    "model-00011-of-00015.safetensors": (2628614880, "a9027ef4db30bb1875bd2bc93b80bae785c5943807efe63bf09834b66e49fc16"),
    "model-00012-of-00015.safetensors": (2628614888, "c16fea955e973010064ff7bc907a6fa67821956e0de23f74807c31fb48339f3a"),
    "model-00013-of-00015.safetensors": (2641525264, "64861c8e1ee13a0cbb5ec9431f402cee33edcbcbc744ff63091fe3a997f0da19"),
    "model-00014-of-00015.safetensors": (3749111392, "ba5319475a6258984143b69af8380f166df8179af90378f2f4e02694cad87baf"),
    "model-00015-of-00015.safetensors": (1409461016, "1e2ddf01c1be73fb19f3022c28e90d24e730c1bc759e5de78083afb39aab3fb7"),
}
CODEC_FILE = ("codec.pth", 1871099728, "74fc41c5a7151c6f350af8bd7e5d6e3accfcc7f3dfbfac23afd35af07052bb2f")

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "crates/izwi-core/tests/fixtures/fish_drama3"
PUBLISHED_CONFIG = FIXTURE_DIR / "config.json"
EVIDENCE_DIR = REPO_ROOT / "tasks/fish-drama3-evidence-2026-10-29/hf-78e5ab79"

FRAME_CODEBOOKS = 10
RAS_WINDOW = 16
PREFILL_CHUNK_ROWS = 128
DRAW_KINDS = ["main_first", "main_retry"] + [f"head_{k}" for k in range(1, 10)]

REFERENCE_SEMANTICS = {
    "dtype": "float32 activations, residual and weights (dense BF16 tensors upcast exactly)",
    "experts": "W8A16: W = e4m3(code) * weight_scale_inv[r // 128, c // 128] in F32 (multiplied); "
    "no activation quantization",
    "router_weights": "fp32 softmax, top-k, renormalised, then rounded to BF16 (izwi and every "
    "BF16 deployment of upstream); an fp32 upstream run would keep fp32 weights, so the "
    "generators round them at MoEExperts.forward entry (--router-weights fp32 turns it off)",
    "tf32": "disabled",
    "prefill_chunk_rows": PREFILL_CHUNK_ROWS,
    "cuda_graphs": "disabled (FISH_DRAMA3_CUDA_GRAPH=0), eager decode",
    "sampler": "tower.sample_top_k_top_p with candidate-ordered recorded noise; RAS as upstream",
    "eos_mask_first_frame": False,
}

# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(obj, path: Path) -> None:
    """Indented JSON, but every list of scalars on one line."""

    def compact(value, depth):
        pad = "  " * depth
        if isinstance(value, dict):
            if not value:
                return "{}"
            items = [f"{pad}  {json.dumps(k)}: {compact(v, depth + 1)}" for k, v in value.items()]
            return "{\n" + ",\n".join(items) + f"\n{pad}}}"
        if isinstance(value, list):
            if all(not isinstance(v, (dict, list)) for v in value) or all(
                isinstance(v, list) and all(not isinstance(x, (dict, list)) for x in v) for v in value
            ):
                return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
            items = [f"{pad}  {compact(v, depth + 1)}" for v in value]
            return "[\n" + ",\n".join(items) + f"\n{pad}]"
        return json.dumps(value, ensure_ascii=False)

    Path(path).write_text(compact(obj, 0) + "\n", encoding="utf-8")


def package_versions(names) -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def check_pinned_file(path: Path, name: str, strict: bool = True) -> str:
    digest = sha256_file(path)
    expected = PINNED_SHA256[name]
    if digest != expected:
        message = f"{path}: sha256 {digest} is not the pinned {name} {expected}"
        if strict:
            sys.exit(message)
        print(f"warning: {message}", file=sys.stderr)
    return digest


# ---------------------------------------------------------------------------
# F32 arithmetic and izwi's xorshift
# ---------------------------------------------------------------------------

_MASK64 = (1 << 64) - 1
_XORSHIFT_MULTIPLIER = 0x9E3779B97F4A7C15
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3

assert sys.byteorder == "little", "fixtures are little-endian"
assert array("f").itemsize == 4 and array("I").itemsize == 4 and array("H").itemsize == 2


def f32(x: float) -> float:
    """Round a double to the nearest F32 (ties to even)."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def xorshift_values(count: int, seed: int, scale: float) -> array:
    """izwi `numeric_check::xorshift_values(count, seed, scale)` as an F32 array."""
    state = ((seed * _XORSHIFT_MULTIPLIER) & _MASK64) | 1
    scale = f32(scale)
    unit = 1.0 / (1 << 24)
    out = [0.0] * count
    for i in range(count):
        state ^= (state << 13) & _MASK64
        state ^= state >> 7
        state ^= (state << 17) & _MASK64
        # Exact in F64 up to the product, whose 48 bits are exact too; the
        # array rounds it to F32 once, as izwi's F32 multiply does.
        out[i] = ((state >> 40) * unit * 2.0 - 1.0) * scale
    return array("f", out)


def xorshift_tensor(rows: int, width: int, seed: int, scale: float):
    import torch

    values = xorshift_values(rows * width, seed, scale)
    return torch.frombuffer(bytearray(values.tobytes()), dtype=torch.float32).reshape(rows, width)


def fnv1a_seed(name: str, seed: int) -> int:
    """izwi `synthetic::name_seed`: FNV-1a over the name, offset basis XOR seed."""
    state = _FNV_OFFSET ^ seed
    for byte in name.encode():
        state = ((state ^ byte) * _FNV_PRIME) & _MASK64
    return state


def bf16_bytes(values: array) -> bytes:
    """F32 values to BF16 bytes, round to nearest even (half::bf16::from_f32 on finite values)."""
    bits = array("I")
    bits.frombytes(values.tobytes())
    out = array("H", [0]) * len(bits)
    for i, b in enumerate(bits):
        if (b >> 23) & 0xFF == 0xFF:
            raise ValueError("a non-finite value reached the BF16 writer")
        out[i] = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) & 0xFFFF
    return out.tobytes()


def encode_e4m3fn(value: float) -> int:
    """izwi `fp8::encode_e4m3fn`: nearest E4M3FN byte, ties to even, saturating at 448."""
    if value != value:
        return 0x7F
    sign = 0x80 if math.copysign(1.0, value) < 0 else 0
    magnitude = abs(value)
    if magnitude >= 448.0:
        return sign | 0x7E
    if magnitude < 2.0**-6:
        return sign | int(round(magnitude * 512.0))
    exponent = math.frexp(magnitude)[1] - 1
    steps = int(round(magnitude * 2.0 ** (3 - exponent)))
    biased = exponent + 7
    if steps == 16:
        steps, biased = 8, biased + 1
    return sign | (biased << 3) | (steps - 8)


def decode_e4m3fn(byte: int) -> float:
    e, m = (byte >> 3) & 15, byte & 7
    if e == 0:
        v = m * 2.0**-9
    elif e == 15 and m == 7:
        v = float("nan")
    else:
        v = (1.0 + m / 8.0) * 2.0 ** (e - 7)
    return -v if byte & 0x80 else v


# ---------------------------------------------------------------------------
# safetensors (raw bytes; no torch needed)
# ---------------------------------------------------------------------------

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "F8_E4M3": 1, "I64": 8, "I32": 4, "U8": 1}


def safetensors_header(entries, metadata=None):
    """Header bytes (length prefix included) and {name: (start, end)} relative
    to the data section, for `entries` = [(name, dtype, shape)] laid out in the
    given order."""
    header = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    offsets = {}
    offset = 0
    for name, dtype, shape in entries:
        size = DTYPE_BYTES[dtype] * math.prod(shape)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offsets[name] = (offset, offset + size)
        offset += size
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    return struct.pack("<Q", len(raw)) + raw, offsets


def write_safetensors(path: Path, tensors: dict, metadata=None) -> int:
    """tensors: {name: (dtype, shape, bytes)}, written sorted by name. Returns tensor bytes."""
    names = sorted(tensors)
    head, _ = safetensors_header([(n, tensors[n][0], tensors[n][1]) for n in names], metadata)
    total = 0
    with Path(path).open("wb") as handle:
        handle.write(head)
        for name in names:
            dtype, shape, data = tensors[name]
            if len(data) != DTYPE_BYTES[dtype] * math.prod(shape):
                raise ValueError(f"{name}: {len(data)} bytes for {dtype} {shape}")
            handle.write(data)
            total += len(data)
    return total


def read_safetensors_header(path: Path) -> tuple[dict, int]:
    with Path(path).open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    return header, 8 + length


def tensor_digest(tensors: dict) -> str:
    """sha256 over the tensors sorted by name (UTF-8 bytes): for each,
    f"{name}\\0{dtype}\\0{','.join(shape)}\\0{len(bytes)}\\0" then the raw bytes."""
    digest = hashlib.sha256()
    for name in sorted(tensors, key=lambda n: n.encode()):
        dtype, shape, data = tensors[name]
        digest.update(f"{name}\0{dtype}\0{','.join(map(str, shape))}\0{len(data)}\0".encode())
        digest.update(data)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# fish-speech
# ---------------------------------------------------------------------------


def git_head(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def import_fish_speech(root: Path, skip_commit_check: bool, extra=()) -> SimpleNamespace:
    """Put the fish-speech checkout first on sys.path and import the fish_omni modules.

    `extra` names further modules (e.g. "fish_speech.models.text2semantic.inference").
    """
    root = Path(root).resolve()
    if not (root / "fish_speech/models/text2semantic/fish_omni/generate.py").is_file():
        sys.exit(f"{root} is not a fish-speech checkout (no fish_omni/generate.py)")
    head = git_head(root)
    if head != FISH_SPEECH_COMMIT:
        message = f"{root} is at {head or 'an unknown commit'}, expected {FISH_SPEECH_COMMIT}"
        if not skip_commit_check:
            sys.exit(message + " (pass --skip-commit-check to record it as unverified)")
        print(f"warning: {message}", file=sys.stderr)
    sys.path.insert(0, str(root))
    os.environ.setdefault("TQDM_DISABLE", "1")
    base = "fish_speech.models.text2semantic.fish_omni"
    modules = {
        name: importlib.import_module(f"{base}.{name}")
        for name in ("config", "layers", "moe", "backbone", "tower", "model", "generate")
    }
    for name in extra:
        modules[name.rsplit(".", 1)[-1]] = importlib.import_module(name)
    commit = head if head == FISH_SPEECH_COMMIT else f"unverified ({head})"
    return SimpleNamespace(root=root, commit=commit, **modules)


def set_torch_reference_mode() -> None:
    import torch

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    os.environ["FISH_DRAMA3_CUDA_GRAPH"] = "0"


def install_expert_patches(moe_module, *, bf16_router: bool, force_w8a16: bool):
    """Wrap `moe.MoEExperts.forward` (returns a function that undoes it).

    * bf16_router: round the renormalised top-k weights to BF16 at entry, as a
      BF16 deployment does with `topk_weights.to(flat.dtype)` (moe.py:220).
    * force_w8a16: FP8-resident experts on CUDA take upstream's own dequantize
      path (moe.py:156-158) instead of the W8A8 Triton kernel (moe.py:146-155).
    """
    import torch

    original = moe_module.MoEExperts.forward
    limit = moe_module.GATHERED_EXPERT_MAX_ROWS

    def forward(self, x, topk_ids, topk_weights):
        if bf16_router:
            topk_weights = topk_weights.to(torch.bfloat16).to(topk_weights.dtype)
        if force_w8a16 and self.fp8 and x.is_cuda:
            if topk_ids.numel() <= limit:
                return self.forward_gathered(x, topk_ids, topk_weights)
            return self.forward_grouped(x, topk_ids, topk_weights)
        return original(self, x, topk_ids, topk_weights)

    moe_module.MoEExperts.forward = forward

    def undo():
        moe_module.MoEExperts.forward = original

    return undo


# ---------------------------------------------------------------------------
# Output tensors, checksums, traces
# ---------------------------------------------------------------------------


class TensorSink:
    """Named F32 / I64 tensors for the sidecar safetensors file."""

    def __init__(self) -> None:
        self.tensors = {}

    def put(self, name: str, tensor) -> str:
        import torch

        if name in self.tensors:
            raise KeyError(f"duplicate sidecar tensor {name}")
        tensor = tensor.detach().cpu()
        tensor = tensor.to(torch.int64) if not tensor.is_floating_point() else tensor.to(torch.float32)
        self.tensors[name] = tensor.contiguous().clone()
        return name

    def save(self, path: Path, metadata: dict) -> str:
        from safetensors.torch import save_file

        save_file(self.tensors, str(path), metadata={k: str(v) for k, v in metadata.items()})
        return sha256_file(path)


def checksum(tensor) -> dict:
    d = tensor.detach().double().cpu()
    count = d.numel()
    sum_sq = float((d * d).sum())
    return {
        "count": count,
        "sum": float(d.sum()),
        "abs_sum": float(d.abs().sum()),
        "sum_sq": sum_sq,
        "max_abs": float(d.abs().max()) if count else 0.0,
        "rms": math.sqrt(sum_sq / count) if count else 0.0,
    }


def trace_entry(sink: TensorSink, name: str, tensor, rows) -> dict:
    t = tensor.detach().float().cpu()
    t = t.reshape(-1, t.shape[-1])
    rows = [r for r in rows if 0 <= r < t.shape[0]]
    return {
        "shape": list(t.shape),
        "rows": rows,
        "values": sink.put(f"{name}/values", t[rows]),
        "row_rms": sink.put(f"{name}/row_rms", t.double().pow(2).mean(-1).sqrt().float()),
        "checksum": checksum(t),
    }


def tower_kind(index: int, shared_layers: int = 4) -> dict:
    """Tower layer `index` of the 13: shared layers inject c0; residual layer j
    injects c_j and head j after it predicts c_{j+1} (tower.py:463-466)."""
    if index < shared_layers:
        return {"index": index, "kind": "shared", "injects": "c0", "head_predicts": None}
    j = index - shared_layers
    return {"index": index, "kind": "residual", "injects": f"c{j}", "head_predicts": f"c{j + 1}"}


def route_entry(router_logits, top_k: int, rows, tie: float) -> dict:
    """Routing of `[n, experts]` router logits as upstream computes it
    (moe.py:216-219), the weights rounded to BF16, and its near ties."""
    import torch

    probabilities = torch.softmax(router_logits.float(), dim=-1)
    values, ids = probabilities.topk(min(top_k + 1, probabilities.shape[-1]), dim=-1)
    kept = values[:, :top_k]
    weights = (kept / kept.sum(-1, keepdim=True)).to(torch.bfloat16).float()
    margins = (values[:, top_k - 1] - values[:, top_k]) if values.shape[-1] > top_k else torch.ones(len(values))
    rows = [r for r in rows if 0 <= r < len(ids)]
    return {
        "rows": rows,
        "experts": ids[rows, :top_k].tolist(),
        "weights": weights[rows].tolist(),
        "margins": margins[rows].tolist(),
        "near_tie_threshold": tie,
        "near_ties": int((margins < tie).sum()),
        "min_margin": float(margins.min()),
    }


# ---------------------------------------------------------------------------
# Recorded-noise sampler
# ---------------------------------------------------------------------------


class NoiseTape:
    """Exp(1) noise from a seeded CPU generator, one value per candidate."""

    def __init__(self, seed: int) -> None:
        import torch

        self.seed = int(seed)
        self.generator = torch.Generator(device="cpu").manual_seed(self.seed)
        self.chunks = []
        self.count = 0

    def take(self, n: int):
        import torch

        values = torch.empty(n, dtype=torch.float32).exponential_(generator=self.generator)
        self.chunks.append(values)
        self.count += n
        return values

    def truncate(self, count: int) -> None:
        while self.count > count:
            self.count -= self.chunks.pop().numel()
        if self.count != count:
            raise RuntimeError("the tape can only drop whole draws")

    def tensor(self):
        import torch

        return torch.cat(self.chunks) if self.chunks else torch.zeros(0, dtype=torch.float32)


class UpstreamSampler:
    """Replaces `tower.sample_top_k_top_p` while installed (see the module doc).

    mode "noise":  draws race against `tape` noise read per candidate;
    mode "forced": returns `forced[step][call]` (a candidate index) and records
                   the logits it was handed.
    """

    def __init__(self, tower_module) -> None:
        self.tm = tower_module
        self.original = tower_module.sample_top_k_top_p
        self.mode = None
        self.tape = None
        self.forced = None
        self.step = None
        self.steps = []

    def install(self, mode: str, *, tape=None, forced=None) -> None:
        self.mode, self.tape, self.forced = mode, tape, forced
        self.step, self.steps = None, []
        self.tm.sample_top_k_top_p = self.sample

    def uninstall(self) -> None:
        self.tm.sample_top_k_top_p = self.original
        self.mode = None

    def begin_step(self, position: int, window) -> None:
        offset = self.tape.count if self.tape is not None else None
        self.step = {"position": position, "window": window, "noise_offset": offset, "calls": []}

    def end_step(self, frame, eos_id: int) -> dict:
        step = self.step
        step["frame"] = frame
        if frame[0] == eos_id:
            heads = step["calls"][2:]
            step["dropped_head_draws"] = len(heads)
            if self.mode == "noise":
                self.tape.truncate(step["calls"][1]["noise_end"])
            step["calls"] = step["calls"][:2]
        self.steps.append(step)
        self.step = None
        return step

    def sample(self, logits, temperature, top_k, top_p, *, check_finite=True):
        import torch

        tm = self.tm
        # ---- fish_omni/tower.py:61-69, verbatim ----
        if isinstance(temperature, torch.Tensor):
            temperature = temperature.clamp(min=tm.MIN_TEMPERATURE)
        else:
            temperature = max(temperature, tm.MIN_TEMPERATURE)
        if check_finite and not torch.isfinite(logits).all():
            raise FloatingPointError("non-finite logits; refusing to sample invalid audio")
        values, ids = torch.sort(logits.float() / temperature, descending=True, dim=-1)
        ranks = torch.arange(values.shape[-1], device=values.device)
        dropped = (values.softmax(-1).cumsum(-1) > top_p) | (ranks >= top_k)
        dropped[..., 0] = False
        probabilities = values.masked_fill(dropped, -torch.inf).softmax(-1)
        # ---- tower.py:70-73 with candidate-ordered noise ----
        if self.step is None:
            raise RuntimeError("the sampler ran outside a recorded decode step")
        rows, width = logits.shape
        if rows != 1:
            raise RuntimeError("fixtures decode one row at a time")
        call = {"kind": DRAW_KINDS[len(self.step["calls"])], "width": width}
        if self.mode == "noise":
            call["noise_start"] = self.tape.count
            per_candidate = self.tape.take(width).view(1, width).to(values.device)
            call["noise_end"] = self.tape.count
            noise = per_candidate.gather(-1, ids)
            scores = probabilities / noise
            choice = scores.argmax(-1, keepdim=True)
            picked = ids.gather(-1, choice).squeeze(-1)
            top2 = scores[0].topk(min(2, width)).values.tolist()
            call["race_gap"] = 1.0 if len(top2) < 2 or top2[0] <= 0 else (top2[0] - top2[1]) / top2[0]
        else:
            index = self.forced[len(self.steps)][len(self.step["calls"])]
            picked = torch.tensor([index], device=logits.device, dtype=torch.long)
            call["logits"] = logits[0].detach().float().cpu().clone()
        kept = int((~dropped[0]).sum())
        cumulative = values.softmax(-1).cumsum(-1)[0]
        by_p = max(1, int((cumulative <= top_p).sum()))
        by_k = min(int(top_k), width)
        call["kept"] = kept
        call["cut"] = "top_p" if by_p < by_k else ("top_k" if by_k < width else "none")
        if call["cut"] == "top_p":
            gaps = [abs(float(cumulative[by_p]) - float(top_p)), abs(float(cumulative[by_p - 1]) - float(top_p))]
            call["top_p_gap"] = min(gaps)
        else:
            call["top_p_gap"] = None
        call["choice"] = int(picked[0])
        self.step["calls"].append(call)
        return picked


def verify_sampler_patch(tower_module, trials: int = 64) -> dict:
    """The patched body picks what upstream's own function picks when its rank
    noise is the candidate noise read in rank order."""
    import torch

    sampler = UpstreamSampler(tower_module)
    generator = torch.Generator().manual_seed(1234)
    mismatches = 0
    for trial in range(trials):
        width = 65 if trial % 2 else 4097
        logits = torch.randn(1, width, generator=generator) * (1 + trial % 5)
        temperature, top_k, top_p = [(1.0, 30, 0.9), (0.7, width, 0.7), (1.0, 30, 0.95), (1e-6, 5, 1.0)][trial % 4]
        torch.manual_seed(trial)
        expected = int(sampler.original(logits, temperature, top_k, top_p)[0])
        torch.manual_seed(trial)
        rank_noise = torch.empty(1, width).exponential_()
        _, ids = torch.sort(logits.float() / max(temperature, tower_module.MIN_TEMPERATURE), descending=True, dim=-1)
        candidate_noise = torch.empty(width)
        candidate_noise[ids[0]] = rank_noise[0]
        tape = NoiseTape(0)
        tape.take = lambda n, values=candidate_noise: values  # type: ignore[method-assign]
        tape.count = 0
        sampler.install("noise", tape=tape)
        sampler.begin_step(0, [])
        try:
            got = int(sampler.sample(logits, temperature, top_k, top_p)[0])
        finally:
            sampler.uninstall()
        mismatches += got != expected
    return {"trials": trials, "mismatches": mismatches}


# ---------------------------------------------------------------------------
# Recording an upstream generation
# ---------------------------------------------------------------------------


class GenerationRecorder:
    """Hooks a FishOmniModel for one `generate_drama3` call.

    Prefill: residual stream after each listed trunk layer (DecoderLayer
    output), mixer / MoE outputs and router logits of `component_layers`, the
    backbone input embeddings and final-norm hidden, and each listed tower
    layer's output (TowerLayer output, after `final_layernorm`). Decode: the
    final-norm hidden and logits handed to `compute_logits` each step, and the
    sampler's per-step record.
    """

    def __init__(self, fish, model, sampler: UpstreamSampler, *, trunk_layers, component_layers,
                 route_layers, tower_layers, near_tie_layers=()) -> None:
        import torch

        self.torch = torch
        self.fish, self.model, self.sampler = fish, model, sampler
        self.phase = "off"
        self.captures = {}
        self.handles = []
        self.decode_hidden, self.decode_logits = [], []
        self.first_full_logits = None
        backbone, tower = model.backbone, model.audio_decoder
        config = model.config
        self.eos = config.eos_token_id
        self.candidates = torch.tensor(
            [config.eos_token_id] + list(range(config.semantic_start_token_id, config.semantic_end_token_id + 1))
        )
        layers = backbone.model.layers
        for index in trunk_layers:
            self._hook(layers[index], f"trunk/{index}")
        for index in component_layers:
            layer = layers[index]
            mixer = layer.self_attn if layer.layer_type == "full_attention" else layer.linear_attn
            self._hook(mixer, f"mixer/{index}")
            self._hook(layer.mlp, f"moe/{index}")
        for index in sorted(set(route_layers) | set(near_tie_layers)):
            self._hook(layers[index].mlp.gate, f"router/{index}")
        self._hook(backbone.model.norm, "final_norm")
        tower_stack = list(tower.shared_layers) + list(tower.residual_layers)
        for index in tower_layers:
            self._hook(tower_stack[index], f"tower/{index}")
        self._wrap(backbone, "compute_logits", self._compute_logits)
        self._wrap(tower, "embed_backbone", self._embed_backbone)
        self._wrap(tower, "decode_step", self._decode_step)

    def _hook(self, module, key):
        def hook(_module, _inputs, output):
            if self.phase == "prefill":
                out = output[0] if isinstance(output, tuple) else output
                self.captures.setdefault(key, []).append(out.detach().float().reshape(-1, out.shape[-1]).cpu().clone())

        self.handles.append(module.register_forward_hook(hook))

    def _wrap(self, obj, name, replacement):
        original = getattr(obj, name)
        setattr(obj, name, lambda *a, **k: replacement(original, *a, **k))
        self.handles.append(SimpleNamespace(remove=lambda: delattr(obj, name)))

    def _compute_logits(self, original, hidden):
        self.phase = "decode"
        logits = original(hidden)
        self.decode_hidden.append(hidden.detach().float().reshape(-1).cpu().clone())
        full = logits.detach().float().reshape(-1).cpu()
        if self.first_full_logits is None:
            self.first_full_logits = full.clone()
        self.decode_logits.append(full[self.candidates])
        return logits

    def _embed_backbone(self, original, frames, text_embeddings):
        out = original(frames, text_embeddings)
        if self.phase == "prefill":
            self.captures.setdefault("input", []).append(out.detach().float().reshape(-1, out.shape[-1]).cpu().clone())
        return out

    def _decode_step(self, original, logits, hidden, positions, frames, history, **kwargs):
        self.phase = "decode"
        self.sampler.begin_step(int(positions[0]), history[-RAS_WINDOW:].tolist())
        sampled = original(logits, hidden, positions, frames, history, **kwargs)
        self.sampler.end_step([int(v) for v in sampled[0].tolist()], self.eos)
        return sampled

    def run(self, prompt, *, generate=None, **kwargs):
        """`generate(model, prompt, **kwargs)`: upstream `generate_drama3` unless
        the caller hands over the original it has wrapped elsewhere."""
        generate = generate or self.fish.generate.generate_drama3
        self.phase = "prefill"
        try:
            frames, reason = generate(self.model, prompt, **kwargs)
        finally:
            self.phase = "off"
        return frames, reason

    def captured(self, key):
        chunks = self.captures.get(key)
        return self.torch.cat(chunks) if chunks else None

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def describe_steps(sampler: UpstreamSampler, candidates, config) -> tuple[list, int]:
    """JSON per decode step, and the RAS redraw count. Checks the RAS rule."""
    start, end, eos = config.semantic_start_token_id, config.semantic_end_token_id, config.eos_token_id
    out, redraws = [], 0
    for step in sampler.steps:
        first = int(candidates[step["calls"][0]["choice"]])
        retry = int(candidates[step["calls"][1]["choice"]])
        redrawn = start <= first <= end and first in step["window"]
        token = retry if redrawn else first
        if token != step["frame"][0]:
            raise RuntimeError(f"RAS replay disagrees with upstream at position {step['position']}")
        redraws += redrawn
        calls = []
        for call in step["calls"]:
            calls.append({k: call[k] for k in ("kind", "width", "choice", "kept", "cut", "top_p_gap")}
                         | ({"race_gap": call["race_gap"]} if "race_gap" in call else {}))
        out.append({
            "position": step["position"],
            "window": step["window"],
            "main": {"first": first, "retry": retry, "redrawn": redrawn, "token": token},
            "frame": None if token == eos else step["frame"],
            "noise_offset": step["noise_offset"],
            "dropped_head_draws": step.get("dropped_head_draws", 0),
            "draws": calls,
        })
    return out, redraws


def cached_vs_full(model, prompt, frames, decode_hidden) -> dict:
    """Upstream's cached decode against one uncached forward over prompt +
    frames: the final-norm hidden of every row a decode step sampled from."""
    import torch

    backbone, tower = model.backbone, model.audio_decoder
    rows = torch.cat((prompt, frames)) if len(frames) else prompt
    saved = backbone.max_seq_len
    backbone.max_seq_len = 0
    try:
        embeddings = tower.embed_backbone(rows, backbone.embed(rows[:, 0]))
        full = backbone(inputs_embeds=embeddings[None])[0].float().cpu()
    finally:
        backbone.max_seq_len = saved
    p = len(prompt)
    cached = torch.stack(decode_hidden)
    reference = full[p - 1 : p - 1 + len(cached)]
    difference = (cached - reference).double()
    return {
        "rows": len(cached),
        "max_abs": float(difference.abs().max()),
        "relative_l2": float(difference.norm() / reference.double().norm()),
    }


def frame_rows(rows) -> list:
    return [[int(v) for v in row] for row in rows.tolist()]
