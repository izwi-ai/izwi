#!/usr/bin/env python3
"""Range-download the Fish Audio Drama 3 real-weight slices (plan §10.3, L1.5).

What it does
------------
Fetches single tensors (and row ranges of two tensors) of
`fishaudio/drama-3` at revision 78e5ab79e49aa9a25abe8799eff2c4d73a1c3216
with HTTP range requests, and writes one safetensors file per slice:

  slice_a.safetensors (about 2.31 GiB)
    * every `audio_decoder*` tensor: the codebook embeddings
      `audio_decoder.codebook_embeddings.weight` [40960, 2048], the 4 shared
      and 9 residual tower layers and the 9 heads (shards 14, 15);
    * `model.language_model.norm.weight` [2048] (the final norm the main
      head reads);
    * `lm_head.weight.compact_rows` [4097, 2048] BF16: rows 248046 (eos), then
      248099..252194 (the semantic band) of `lm_head.weight`, in that order,
      i.e. the compact head's candidates;
    * `model.language_model.embed_tokens.weight.semantic_rows` [4096, 2048]
      BF16: rows 248099..252194 of `embed_tokens`.
    Row ids come from the pinned config.json (eos_token_id,
    semantic_start_token_id, semantic_end_token_id).
  slice_b.safetensors (about 1.63 GiB)
    * every `model.language_model.layers.0.*` tensor (Gated DeltaNet) and every
      `model.language_model.layers.3.*` tensor (gated full attention), each
      with its norms, router, 256 E4M3 experts + F32 `weight_scale_inv` and the
      shared expert (shards 1, 2).

Tensor names, dtypes and shapes are the checkpoint's, except the two row
slices above (their `__metadata__` entry says which rows). The files'
`__metadata__` also records the repo, revision, slice and source shards.

Pinning and integrity
---------------------
* URLs are `{HF_ENDPOINT or https://huggingface.co}/fishaudio/drama-3/resolve/
  <revision>/<file>`, so the bytes are immutable.
* `config.json` and `model.safetensors.index.json` are downloaded whole and
  must hash to the pinned sha256.
* Each needed shard's header is read live (8-byte length, then the JSON). Its
  entries for the fetched tensors must agree with the saved evidence headers
  when `tasks/fish-drama3-evidence-2026-10-29/hf-78e5ab79/headers/` exists
  (local only), and the shard size reported by the server (Content-Range
  total) must equal the pinned LFS size.
* A slice cannot be checked against the LFS sha256 (that covers whole
  shards), so each finished slice gets: per-tensor sha256 and a value sanity
  check (BF16 / F32 finite, E4M3 bytes never the NaN code, scales > 0) in
  `manifest.json`, and the whole file's sha256 in `<slice>.safetensors.sha256`.
  A later run skips a slice whose file still matches its `.sha256`
  (`--force` re-downloads).

Resume
------
A slice is written in place as `<slice>.safetensors.partial` (header first,
then each range at its offset); `<slice>.state.json` lists the finished
requests. Interrupted runs continue where they stopped. Adjacent tensors in a
shard are fetched together (gaps <= 4 MiB, requests <= 64 MiB).

Disk
----
Before downloading, free space at `--out` (shutil.disk_usage) must exceed the
bytes of the unfinished slices plus `--min-free-gib` (default 2).

Usage (stdlib only; any Python >= 3.10; HF_TOKEN / HUGGING_FACE_HUB_TOKEN is
sent when set, and only to the Hugging Face host itself, never on redirects):

    python3 -I scripts/fixtures/fetch_fish_drama3_slices.py \\
        [--out target/fish-drama3-slices] [--slices a,b] [--force] \\
        [--min-free-gib 2] [--retries 5]

Then point `IZWI_FISH_DRAMA3_SLICE_DIR` at `--out` and run
`generate_fish_drama3_slices.py` (same directory). Expect about 4.2 GB of
downloads; runtime is bandwidth-bound (a few minutes at 100 Mbit/s and up).

manifest.json (`schema` = "izwi.fish_drama3.slices_manifest", `schema_version` = 1):

{
  "schema", "schema_version", "generator", "created_at",
  "repo_id", "revision", "endpoint",
  "config_sha256", "index_sha256",
  "slices": {"a" | "b": {
      "file": "slice_a.safetensors", "bytes": int, "sha256": hex,
      "tensors": {name: {"dtype", "shape", "sha256",
                         "source": [{"shard", "tensor", "rows": [first, last] | null,
                                     "byte_start", "byte_end"}]}},  # absolute,
                                                                      # end exclusive
      "sanity": {"bf16_nonfinite": 0, "f32_nonfinite": 0,
                 "e4m3_nan_bytes": 0, "nonpositive_scales": 0}}}
}
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import struct
import sys
import time
import urllib.error
import urllib.request
from array import array
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_common():
    spec = importlib.util.spec_from_file_location("fish_drama3_fixture_common", HERE / "fish_drama3_fixture_common.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


common = load_common()

GENERATOR = "scripts/fixtures/fetch_fish_drama3_slices.py"
DEFAULT_OUT = common.REPO_ROOT / "target/fish-drama3-slices"
SLICE_B_LAYERS = (0, 3)
MAX_GAP = 4 << 20
MAX_REQUEST = 64 << 20
BLOCK = 1 << 20
USER_AGENT = "izwi-fixtures/1 (fish-drama3 slices)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--slices", default="a,b", help="comma-separated: a, b")
    parser.add_argument("--force", action="store_true", help="re-download finished slices")
    parser.add_argument("--min-free-gib", type=float, default=2.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--evidence-headers", type=Path, default=common.EVIDENCE_DIR / "headers")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class Remote:
    def __init__(self, args) -> None:
        self.endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        self.token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        self.retries = max(1, args.retries)
        self.timeout = args.timeout

    def url(self, name: str) -> str:
        return f"{self.endpoint}/{common.REPO_ID}/resolve/{common.CHECKPOINT_REVISION}/{name}"

    def _request(self, name: str, byte_range=None):
        request = urllib.request.Request(self.url(name), headers={"User-Agent": USER_AGENT})
        if byte_range is not None:
            # Redirected with the request (urllib keeps non-content headers).
            request.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1] - 1}")
        if self.token:
            # Not forwarded to the CDN a resolve URL redirects to.
            request.add_unredirected_header("Authorization", f"Bearer {self.token}")
        return urllib.request.urlopen(request, timeout=self.timeout)

    def _retry(self, what, action):
        for attempt in range(1, self.retries + 1):
            try:
                return action()
            except (urllib.error.URLError, OSError, ValueError) as error:
                if isinstance(error, urllib.error.HTTPError) and error.code in (401, 403, 404):
                    sys.exit(f"{what}: HTTP {error.code} ({error.reason}); check HF_TOKEN / access")
                if attempt == self.retries:
                    raise
                wait = min(60, 2**attempt)
                print(f"  {what}: {error}; retry {attempt}/{self.retries - 1} in {wait}s", file=sys.stderr)
                time.sleep(wait)

    def whole(self, name: str) -> bytes:
        def action():
            with self._request(name) as response:
                return response.read()

        return self._retry(name, action)

    def ranged(self, name: str, start: int, end: int, sink) -> int:
        """Stream bytes [start, end) of `name` into sink(offset_in_range, data).
        Returns the file size from Content-Range."""

        def action():
            with self._request(name, (start, end)) as response:
                if response.status != 206:
                    raise ValueError(f"server ignored the range (HTTP {response.status})")
                content_range = response.headers.get("Content-Range", "")
                unit, _, spec = content_range.partition(" ")
                span, _, total = spec.partition("/")
                first, _, last = span.partition("-")
                if unit != "bytes" or int(first) != start or int(last) != end - 1:
                    raise ValueError(f"unexpected Content-Range {content_range!r} for {start}-{end - 1}")
                offset = 0
                while offset < end - start:
                    data = response.read(min(BLOCK, end - start - offset))
                    if not data:
                        raise ValueError(f"short read at {offset} of {end - start}")
                    sink(offset, data)
                    offset += len(data)
                return int(total)

        return self._retry(f"{name} [{start}, {end})", action)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def shard_header(remote: Remote, shard: str, evidence: Path) -> tuple[dict, int]:
    head = bytearray(8)

    def into(buffer):
        def sink(offset, data):
            buffer[offset : offset + len(data)] = data

        return sink

    size = remote.ranged(shard, 0, 8, into(head))
    if size != common.SHARDS[shard][0]:
        sys.exit(f"{shard}: the server reports {size} bytes; the pin says {common.SHARDS[shard][0]}")
    (length,) = struct.unpack("<Q", bytes(head))
    raw = bytearray(length)
    remote.ranged(shard, 8, 8 + length, into(raw))
    header = json.loads(bytes(raw))
    saved = evidence / f"{shard}.header.json"
    if saved.is_file():
        reference = {k: v for k, v in json.loads(saved.read_text()).items() if k != "__metadata__"}
        live = {k: v for k, v in header.items() if k != "__metadata__"}
        if live != reference:
            differing = sorted(k for k in live.keys() | reference.keys() if live.get(k) != reference.get(k))
            sys.exit(f"{shard}: the live header differs from {saved} (e.g. {differing[:3]})")
    return header, 8 + length


def plan_slices(index: dict, config: dict, headers: dict, wanted: list[str]) -> dict:
    """{slice: [(out_name, dtype, shape, [piece])]}; piece = (shard, tensor, rows, start, end)."""
    weight_map = index["weight_map"]
    eos = config["eos_token_id"]
    first, last = config["semantic_start_token_id"], config["semantic_end_token_id"]

    def info(name):
        shard = weight_map[name]
        header, data_start = headers[shard]
        entry = header[name]
        start, end = entry["data_offsets"]
        return shard, entry["dtype"], entry["shape"], data_start + start, data_start + end

    def whole(name):
        shard, dtype, shape, start, end = info(name)
        return (name, dtype, shape, [(shard, name, None, start, end)])

    def rows(name, out_name, ranges):
        shard, dtype, shape, start, _ = info(name)
        row_bytes = common.DTYPE_BYTES[dtype] * shape[1]
        pieces = [(shard, name, [a, b], start + a * row_bytes, start + (b + 1) * row_bytes) for a, b in ranges]
        count = sum(b - a + 1 for a, b in ranges)
        return (out_name, dtype, [count, shape[1]], pieces)

    plans = {}
    if "a" in wanted:
        names = sorted(n for n in weight_map if n.startswith("audio_decoder"))
        plans["a"] = [whole(n) for n in names] + [
            whole("model.language_model.norm.weight"),
            rows("lm_head.weight", "lm_head.weight.compact_rows", [(eos, eos), (first, last)]),
            rows("model.language_model.embed_tokens.weight",
                 "model.language_model.embed_tokens.weight.semantic_rows", [(first, last)]),
        ]
    if "b" in wanted:
        prefixes = tuple(f"model.language_model.layers.{layer}." for layer in SLICE_B_LAYERS)
        plans["b"] = [whole(n) for n in sorted(weight_map) if n.startswith(prefixes)]
    return plans


def requests_for(entries, offsets) -> list[dict]:
    """Coalesce the pieces per shard into ranged requests; each part says where
    its bytes go in the output's data section."""
    pieces = []
    for out_name, _, _, parts in entries:
        position = offsets[out_name][0]
        for shard, _, _, start, end in parts:
            # Large tensors go in MAX_REQUEST pieces, so a retry or resume
            # repeats at most one of them.
            for lo in range(start, end, MAX_REQUEST):
                hi = min(end, lo + MAX_REQUEST)
                pieces.append((shard, lo, hi, position + (lo - start)))
            position += end - start
    pieces.sort()
    requests = []
    for shard, start, end, position in pieces:
        last = requests[-1] if requests else None
        if (
            last is not None
            and last["shard"] == shard
            and start - last["end"] <= MAX_GAP
            and start >= last["end"]
            and end - last["start"] <= MAX_REQUEST
        ):
            last["end"] = end
            last["parts"].append((start, end, position))
        else:
            requests.append({"shard": shard, "start": start, "end": end, "parts": [(start, end, position)]})
    return requests


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def sanity(dtype: str, data: bytes) -> dict:
    out = {"bf16_nonfinite": 0, "f32_nonfinite": 0, "e4m3_nan_bytes": 0, "nonpositive_scales": 0}
    if dtype == "BF16":
        # Non-finite: exponent bits all set, i.e. high byte 0x7F / 0xFF and the
        # low byte's top bit. Screen the high bytes first (C speed).
        high = data[1::2]
        if high.count(b"\x7f") + high.count(b"\xff"):
            high_hits = high.translate(bytes(1 if b & 0x7F == 0x7F else 0 for b in range(256)))
            low_hits = data[0::2].translate(bytes(1 if b & 0x80 else 0 for b in range(256)))
            both = int.from_bytes(high_hits, "little") & int.from_bytes(low_hits, "little")
            out["bf16_nonfinite"] = bin(both).count("1")
    elif dtype == "F32":
        values = array("f")
        values.frombytes(data)
        out["f32_nonfinite"] = sum(1 for v in values if v != v or v in (float("inf"), float("-inf")))
        out["nonpositive_scales"] = sum(1 for v in values if not v > 0)
    elif dtype == "F8_E4M3":
        out["e4m3_nan_bytes"] = data.count(b"\x7f") + data.count(b"\xff")
    return out


def fetch_slice(remote: Remote, out: Path, key: str, entries: list, force: bool) -> dict:
    final = out / f"slice_{key}.safetensors"
    sha_file = final.with_name(final.name + ".sha256")
    partial = final.with_name(final.name + ".partial")
    state_file = out / f"slice_{key}.state.json"
    metadata = {
        "repo_id": common.REPO_ID,
        "revision": common.CHECKPOINT_REVISION,
        "slice": key,
        "shards": ",".join(sorted({p[0] for e in entries for p in e[3]})),
    }
    for out_name, _, _, parts in entries:
        if parts[0][2] is not None:
            metadata[out_name] = f"rows {' + '.join(f'{a}..={b}' for _, _, (a, b), _, _ in parts)} of {parts[0][1]}"
    head, offsets = common.safetensors_header([(e[0], e[1], e[2]) for e in entries], metadata)
    requests = requests_for(entries, offsets)
    plan_digest = hashlib.sha256(head + json.dumps(requests).encode()).hexdigest()
    total = len(head) + sum(end - start for start, end in offsets.values())

    if not force and final.is_file() and sha_file.is_file():
        recorded = sha_file.read_text().split()[0]
        if final.stat().st_size == total and common.sha256_file(final) == recorded:
            print(f"slice {key}: {final} is complete, skipping", file=sys.stderr)
            return {"file": final.name, "bytes": total, "sha256": recorded, "skipped": True}

    state = {}
    if state_file.is_file() and partial.is_file() and not force:
        state = json.loads(state_file.read_text())
    if state.get("plan") != plan_digest:
        state = {"plan": plan_digest, "done": []}
        with partial.open("wb") as handle:
            handle.write(head)
            handle.truncate(total)
    done = set(state["done"])
    print(f"slice {key}: {len(requests)} requests, {total / 2**30:.2f} GiB, {len(done)} done", file=sys.stderr)
    with partial.open("r+b") as handle:
        for number, request in enumerate(requests):
            if number in done:
                continue

            def sink(offset, data, request=request):
                absolute = request["start"] + offset
                for start, end, position in request["parts"]:
                    lo, hi = max(start, absolute), min(end, absolute + len(data))
                    if lo < hi:
                        handle.seek(len(head) + position + (lo - start))
                        handle.write(data[lo - absolute : hi - absolute])

            size = remote.ranged(request["shard"], request["start"], request["end"], sink)
            if size != common.SHARDS[request["shard"]][0]:
                sys.exit(f"{request['shard']}: size {size} is not the pinned one")
            handle.flush()
            os.fsync(handle.fileno())
            done.add(number)
            state["done"] = sorted(done)
            state_file.write_text(json.dumps(state))
            megabytes = (request["end"] - request["start"]) / 2**20
            print(f"  [{len(done)}/{len(requests)}] {request['shard']} {megabytes:.1f} MiB", file=sys.stderr)

    tensors, totals = {}, {"bf16_nonfinite": 0, "f32_nonfinite": 0, "e4m3_nan_bytes": 0, "nonpositive_scales": 0}
    with partial.open("rb") as handle:
        for out_name, dtype, shape, parts in entries:
            start, end = offsets[out_name]
            handle.seek(len(head) + start)
            data = handle.read(end - start)
            checks = sanity(dtype, data)
            for k, v in checks.items():
                totals[k] += v
            tensors[out_name] = {
                "dtype": dtype,
                "shape": shape,
                "sha256": common.sha256_bytes(data),
                "source": [
                    {"shard": s, "tensor": t, "rows": r, "byte_start": a, "byte_end": b}
                    for s, t, r, a, b in parts
                ],
            }
    if any(totals.values()):
        sys.exit(f"slice {key}: value sanity failed {totals}; delete {partial} and retry")
    digest = common.sha256_file(partial)
    partial.replace(final)
    sha_file.write_text(f"{digest}  {final.name}\n")
    state_file.unlink(missing_ok=True)
    return {"file": final.name, "bytes": total, "sha256": digest, "tensors": tensors, "sanity": totals}


def main() -> int:
    args = parse_args()
    wanted = [s.strip() for s in args.slices.split(",") if s.strip()]
    if not wanted or any(s not in ("a", "b") for s in wanted):
        sys.exit("--slices takes a, b or a,b")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    remote = Remote(args)

    small = {}
    for name in ("config.json", "model.safetensors.index.json"):
        path = out / name
        if not path.is_file() or common.sha256_file(path) != common.PINNED_SHA256[name]:
            path.write_bytes(remote.whole(name))
        small[name] = common.check_pinned_file(path, name)
    config = json.loads((out / "config.json").read_text())
    index = json.loads((out / "model.safetensors.index.json").read_text())

    needed_shards = set()
    for name, shard in index["weight_map"].items():
        if ("a" in wanted and (name.startswith("audio_decoder") or name in (
                "model.language_model.norm.weight", "lm_head.weight", "model.language_model.embed_tokens.weight"))) or (
                "b" in wanted and name.startswith(tuple(f"model.language_model.layers.{l}." for l in SLICE_B_LAYERS))):
            needed_shards.add(shard)
    headers = {shard: shard_header(remote, shard, args.evidence_headers) for shard in sorted(needed_shards)}
    plans = plan_slices(index, config, headers, wanted)

    remaining = 0
    for key, entries in plans.items():
        final = out / f"slice_{key}.safetensors"
        if args.force or not (final.is_file() and final.with_name(final.name + ".sha256").is_file()):
            remaining += sum(sum(p[4] - p[3] for p in e[3]) for e in entries)
    free = shutil.disk_usage(out).free
    need = remaining + int(args.min_free_gib * 2**30)
    if free < need:
        sys.exit(f"{out}: {free / 2**30:.1f} GiB free; need {need / 2**30:.1f} GiB "
                 f"({remaining / 2**30:.2f} to download + {args.min_free_gib} margin)")

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    manifest.update({
        "schema": "izwi.fish_drama3.slices_manifest",
        "schema_version": 1,
        "generator": GENERATOR,
        "created_at": common.now_utc(),
        "repo_id": common.REPO_ID,
        "revision": common.CHECKPOINT_REVISION,
        "endpoint": remote.endpoint,
        "config_sha256": small["config.json"],
        "index_sha256": small["model.safetensors.index.json"],
    })
    manifest.setdefault("slices", {})
    for key, entries in plans.items():
        result = fetch_slice(remote, out, key, entries, args.force)
        if result.pop("skipped", False) and key in manifest["slices"]:
            continue
        if "tensors" not in result:
            # Skipped without a manifest entry: describe it from the file.
            header, data_start = common.read_safetensors_header(out / result["file"])
            result["tensors"] = {}
            with (out / result["file"]).open("rb") as handle:
                for out_name, dtype, shape, parts in entries:
                    start, end = header[out_name]["data_offsets"]
                    handle.seek(data_start + start)
                    result["tensors"][out_name] = {
                        "dtype": dtype, "shape": shape,
                        "sha256": common.sha256_bytes(handle.read(end - start)),
                        "source": [{"shard": s, "tensor": t, "rows": r, "byte_start": a, "byte_end": b}
                                   for s, t, r, a, b in parts],
                    }
        manifest["slices"][key] = result
        manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"slices in {out}: {', '.join(sorted(plans))}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
