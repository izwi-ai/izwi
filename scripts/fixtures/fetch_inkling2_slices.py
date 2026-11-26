#!/usr/bin/env python3
"""Fetch the Inkling 2 Mini header census and a few real expert rows.

What it does
------------
Reads `thinkingmachines/Inkling-2-Mini` at the pinned revision with HTTP
range requests (no full download; the checkpoint is 19.5 GB) and writes into
`crates/izwi-core/tests/fixtures/inkling2/`:

  tensor_census.txt
    One line per tensor of both safetensors files: `file name dtype shape`.
    The Rust loader's plan is tested against this list.
  real_slices.safetensors (about 100 KB)
    The first rows of one NVFP4 expert and one MXFP8 expert, stored exactly
    as published (packed codes, block scales, per-expert factors), plus the
    values they decode to (`*.expected`, F32), computed here with the same
    arithmetic as transformers' `Inkling2LoopDequantizeExperts`:

      NVFP4: two E2M1 codes per byte (even element in the low nibble), one
             FP8 E4M3 scale per 16 values, times `amax / outer_scale_denominator`
             of the expert.
      MXFP8: FP8 E4M3 values, one scale `2 ** (s - 127)` per 32 values.

The Rust dequantizers and kernels are tested against these real values.

Usage (needs only the Python standard library and numpy):

    python3 scripts/fixtures/fetch_inkling2_slices.py \
        crates/izwi-core/tests/fixtures/inkling2
"""

import json
import os
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np

REPO = "thinkingmachines/Inkling-2-Mini"
REVISION = "c9d028302cd0f035fdf8e7e91f089a4ce94d2195"
FILES = {
    "model-00001-of-00001.safetensors": 19_526_019_168,
    "mtp.safetensors": 635_801_416,
}
ROWS = 4
E2M1 = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def e4m3_table() -> np.ndarray:
    """All 256 FP8 E4M3 (`float8_e4m3fn`) values as F32; 0x7F and 0xFF are NaN."""
    table = np.zeros(256, dtype=np.float32)
    for byte in range(256):
        sign = -1.0 if byte & 0x80 else 1.0
        exponent = (byte >> 3) & 0xF
        mantissa = byte & 0x7
        if exponent == 0xF and mantissa == 0x7:
            value = float("nan")
        elif exponent == 0:
            value = mantissa / 8.0 * 2.0**-6
        else:
            value = (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
        table[byte] = sign * value
    return table


E4M3 = e4m3_table()


def url(name: str) -> str:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    return f"{endpoint}/{REPO}/resolve/{REVISION}/{name}"


def fetch(name: str, start: int, end: int) -> bytes:
    """Bytes `start..end` (exclusive) of a repo file."""
    request = urllib.request.Request(url(name), headers={"Range": f"bytes={start}-{end - 1}"})
    with urllib.request.urlopen(request) as response:
        total = int(response.headers["Content-Range"].rsplit("/", 1)[1])
        if total != FILES[name]:
            raise SystemExit(f"{name}: server reports {total} bytes, pinned {FILES[name]}")
        data = response.read()
    if len(data) != end - start:
        raise SystemExit(f"{name}: short read {len(data)} of {end - start}")
    return data


def header(name: str) -> tuple[dict, int]:
    length = struct.unpack("<Q", fetch(name, 0, 8))[0]
    entries = json.loads(fetch(name, 8, 8 + length))
    entries.pop("__metadata__", None)
    return entries, 8 + length


def rows(name: str, entry: dict, base: int, expert: int, count: int) -> np.ndarray:
    """The first `count` rows of one expert of an `[experts, rows, columns]` tensor."""
    _, n_rows, n_columns = entry["shape"]
    start = base + entry["data_offsets"][0] + expert * n_rows * n_columns
    data = fetch(name, start, start + count * n_columns)
    return np.frombuffer(data, dtype=np.uint8).reshape(count, n_columns)


def scalar_f32(name: str, entry: dict, base: int, index: int) -> np.ndarray:
    start = base + entry["data_offsets"][0] + 4 * index
    return np.frombuffer(fetch(name, start, start + 4), dtype="<f4").copy()


def write_safetensors(path: Path, tensors: dict[str, tuple[str, np.ndarray]], metadata: dict) -> None:
    entries, blobs, offset = {"__metadata__": metadata}, [], 0
    for key in sorted(tensors):
        dtype, array = tensors[key]
        blob = np.ascontiguousarray(array).tobytes()
        entries[key] = {"dtype": dtype, "shape": list(array.shape), "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    head = json.dumps(entries, separators=(",", ":")).encode()
    head += b" " * (-len(head) % 8)
    path.write_bytes(struct.pack("<Q", len(head)) + head + b"".join(blobs))


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    headers = {name: header(name) for name in FILES}

    lines = [
        f"# {REPO} @ {REVISION}",
        "# file name dtype shape (both safetensors headers, read over HTTP range requests)",
    ]
    for name, (entries, _) in headers.items():
        for key in sorted(entries):
            shape = ",".join(str(d) for d in entries[key]["shape"])
            lines.append(f"{name} {key} {entries[key]['dtype']} [{shape}]")
    (out / "tensor_census.txt").write_text("\n".join(lines) + "\n")
    print(f"census: {len(lines) - 2} tensors")

    name = "model-00001-of-00001.safetensors"
    entries, base = headers[name]
    tensors: dict[str, tuple[str, np.ndarray]] = {}

    # NVFP4: layer 0, expert 3, both projections.
    for layer, expert, projection in ((0, 3, "w13_weight"), (0, 3, "w2_weight")):
        prefix = f"model.llm.layers.{layer}.mlp.experts.{projection}"
        codes = rows(name, entries[prefix], base, expert, ROWS)
        scales = rows(name, entries[f"{prefix}_scale"], base, expert, ROWS)
        amax = scalar_f32(name, entries[f"{prefix}_amax"], base, expert)
        denominator = scalar_f32(name, entries[f"{prefix}_outer_scale_denominator"], base, 0)
        values = np.stack([E2M1[codes & 0xF], E2M1[codes >> 4]], axis=-1).reshape(ROWS, -1, 16)
        values = values * E4M3[scales][..., None] * (amax / denominator)
        key = f"nvfp4.layer{layer}.expert{expert}.{projection}"
        tensors[f"{key}.codes"] = ("U8", codes)
        tensors[f"{key}.scale"] = ("F8_E4M3", scales)
        tensors[f"{key}.amax"] = ("F32", amax)
        tensors[f"{key}.outer_scale_denominator"] = ("F32", denominator)
        tensors[f"{key}.expected"] = ("F32", values.reshape(ROWS, -1).astype("<f4"))
        print(key, "amax", amax, "denominator", denominator, "max", np.abs(values).max())

    # MXFP8: layer 20, expert 5, both projections.
    for layer, expert, projection in ((20, 5, "w13_weight"), (20, 5, "w2_weight")):
        prefix = f"model.llm.layers.{layer}.mlp.experts.{projection}"
        codes = rows(name, entries[prefix], base, expert, ROWS)
        scales = rows(name, entries[f"{prefix}_scale"], base, expert, ROWS)
        values = E4M3[codes].reshape(ROWS, -1, 32) * np.exp2(scales.astype(np.float32) - 127.0)[..., None]
        key = f"mxfp8.layer{layer}.expert{expert}.{projection}"
        tensors[f"{key}.values"] = ("F8_E4M3", codes)
        tensors[f"{key}.scale"] = ("U8", scales)
        tensors[f"{key}.expected"] = ("F32", values.reshape(ROWS, -1).astype("<f4"))
        print(key, "scale bytes", scales.min(), scales.max(), "max", np.abs(values).max())

    for key, (_, array) in tensors.items():
        if key.endswith(".expected") and not np.isfinite(array).all():
            raise SystemExit(f"{key}: non-finite decoded values")
    write_safetensors(
        out / "real_slices.safetensors",
        tensors,
        {"repo": REPO, "revision": REVISION, "rows": f"first {ROWS} rows of the named expert"},
    )
    print("wrote", out / "real_slices.safetensors")


if __name__ == "__main__":
    main()
