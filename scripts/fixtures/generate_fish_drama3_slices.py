#!/usr/bin/env python3
"""Generate the L1.5 real-weight slice outputs for Fish Audio Drama 3 (plan §10.3).

What it does
------------
Reads the slices `fetch_fish_drama3_slices.py` wrote (`slice_a.safetensors`,
`slice_b.safetensors`, `config.json`, each checked against its recorded
sha256) and runs them through upstream fish-speech modules (0f8a12e8) on the
CPU in float32, one slice at a time:

1. Trunk layers 0 (Gated DeltaNet) and 3 (gated full attention), slice B.
   Each layer is upstream's `DecoderLayer` inside a `FishOmniBackbone` built on
   the meta device whose layer list is replaced by that one real layer (and
   whose unused embedding / lm_head are dropped), loaded by upstream's
   `load_backbone_weights` (the FP8 experts dequantized by
   `dequantize_fp8_blocks` in F32: W8A16), router weights rounded to BF16.
   Input: X = xorshift(304 * 2048, seed 1000 + layer, scale 1.0) as
   [304, 2048], fed as the residual stream at positions 0..303: a 300-row
   history in prefill chunks [0, 128), [128, 256), [256, 300) (upstream's
   128-row prefill, so the DeltaNet chunk rule continues its state and the
   attention reads its cache), then 4 decode steps at 300, 301, 302, 303 (the
   DeltaNet recurrent rule). Recorded: the layer output (residual after the
   layer), the mixer output and the MoE block output at the 4 decode
   positions, the same over the history (trace entries), and the routing.
2. Tower, slice A: upstream `FishOmniAudioDecoder` (all 13 layers, 9 heads,
   the codebook embeddings) loaded by `load_audio_decoder_weights`.
   Input: H = xorshift(301 * 2048, seed 2000, scale 1.0) as [301, 2048], the
   stand-in final-norm hidden of rows 0..300, and FRAMES (301 rows, below).
   Prefill rows 0..299 teacher-forced (row p labelled with row p + 1) in
   128-row chunks, then one decode step at position 300 (input frame row
   300) under the forced frame FORCED_FRAME: upstream's `decode_step` with
   the sampler forced, recording each head's logits. Recorded: every tower
   layer's output over the prefill (trace entries) and at the step, and the
   9 head logits [9, 4096].
3. Main head, slice A: logits = lm_head_compact(final_norm(X3)) with
   X3 = xorshift(4 * 2048, seed 3000, scale 1.0), `final_norm` upstream's
   ZeroCenteredRMSNorm with `model.language_model.norm.weight`, the head the
   4,097 compact rows (eos, then the band) [4, 4097].
4. Backbone embedding, slice A: upstream `embed_backbone` of FRAMES rows
   40..47 (semantic) with E_lm rows from the semantic slice:
   (E_lm[main] + CB[1 * 4096 + c1]) / sqrt(2) [8, 2048].

FRAMES (row r = 0..300; also written to the JSON):
    r < 40:  [1000 + r, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]          (text rows)
    r >= 40: c0 = (r * 7919 + 17) % 4096
             c_k = (r * 977 + k * 131 + r * k * 31) % 1024    for k = 1..9
             if r % 37 == 0: c5 = 1024 + (r * 29) % 3072       (a head-range code)
             [248099 + c0, c0, c1, ..., c9]
FORCED_FRAME (the predicted frame of the step at 300):
    [248099 + 1234, 1234, 17, 1023, 1024, 4095, 0, 512, 2048, 77, 999]

xorshift is izwi's `numeric_check::xorshift_values` (defined in
`fish_drama3_fixture_common.py`); izwi's test regenerates X, H and X3 and
checks them against the input checksums below before comparing.

Usage (CPU; this Mac works; < 8 GB RAM, a few minutes; needs torch >= 2.4,
safetensors >= 0.4.3, tokenizers, loguru, tqdm and a fish-speech checkout at
0f8a12e8, as for the tiny generator):

    python3 -I scripts/fixtures/fetch_fish_drama3_slices.py --out DIR
    python3 -I scripts/fixtures/generate_fish_drama3_slices.py \\
        --fish-speech /path/to/fish-speech --slice-dir DIR \\
        [--router-weights bf16|fp32] [--skip-commit-check]

It writes `slice_expected.json` and `slice_expected.safetensors` into
`--slice-dir` (override with --out); `IZWI_FISH_DRAMA3_SLICE_DIR=DIR` then
runs izwi's ignored `fish_drama3::reference_tests::slice_*`.

slice_expected.json (`schema` = "izwi.fish_drama3.slice_expected", `schema_version` = 1)
--------------------------------------------------------------------------------------
`<t>` names a tensor in `slice_expected.safetensors` (F32). `trace` and
`checksum` as defined in the common module. H = 2048.

{
  "schema", "schema_version", "generator", "created_at",
  "provenance": {...}, "reference_semantics": {...},
  "model": {"repo_id", "revision", "config_sha256",
            "slices": {"a" | "b": {"file", "sha256"}}},
  "tensors_file": "slice_expected.safetensors", "tensors_sha256": hex,
  "trunk_layers": [{
    "layer": 0 | 3, "kind": "linear_attention" | "full_attention",
    "input": {"seed": 1000 + layer, "rows": 304, "width": H, "scale": 1.0,
              "checksum": checksum},
    "prefill_chunks": [[0, 128], [128, 256], [256, 300]],
    "decode_positions": [300, 301, 302, 303],
    "output" | "mixer" | "moe": {
        "decode": <t> [4, H], "decode_checksum": checksum,
        "history": trace},             # rows 0..299
    "routes": {"decode": {"rows": [300..303], "experts": [[8]],
                          "weights": [[8]] (BF16-rounded), "margins": [4], ...},
               "history": {"near_ties", "near_tie_threshold", "min_margin"}}}],
  "tower": {
    "hidden": {"seed": 2000, "rows": 301, "width": H, "scale": 1.0, "checksum"},
    "frames": [[main, c0..c9]] (301 rows), "prefill_rows": 300,
    "prefill_chunks": [...], "position": 300, "forced_frame": [...],
    "layers": [{"index", "kind", "injects", "head_predicts",
                "prefill": trace, "step": <t> [H]}],
    "head_logits": <t> [9, 4096]},     # head j under forced c0..cj
  "main_head": {"input": {"seed": 3000, "rows": 4, "width": H, "scale": 1.0,
                          "checksum"},
                "candidates": "eos 248046, then 248099..252194",
                "logits": <t> [4, 4097]},
  "embedding": {"rows": [40..47], "frames": [[...]], "values": <t> [8, H]},
  "self_checks": {...}
}
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_common():
    spec = importlib.util.spec_from_file_location("fish_drama3_fixture_common", HERE / "fish_drama3_fixture_common.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


common = load_common()

SCHEMA = "izwi.fish_drama3.slice_expected"
GENERATOR = "scripts/fixtures/generate_fish_drama3_slices.py"
DEFAULT_SLICE_DIR = Path(os.environ.get("IZWI_FISH_DRAMA3_SLICE_DIR") or common.REPO_ROOT / "target/fish-drama3-slices")
TRUNK_LAYERS = (0, 3)
HISTORY, DECODE = 300, 4
CHUNKS = [(0, 128), (128, 256), (256, 300)]
HISTORY_PICKS = [0, 1, 63, 64, 65, 127, 128, 255, 256, 299]
TRUNK_SEED, TOWER_SEED, HEAD_SEED = 1000, 2000, 3000
TOWER_ROWS = 301
TEXT_ROWS = 40
EMBED_ROWS = list(range(40, 48))
SEMANTIC_START = 248099
FORCED_FRAME = [SEMANTIC_START + 1234, 1234, 17, 1023, 1024, 4095, 0, 512, 2048, 77, 999]
TOWER_PICKS = [0, 39, 40, 41, 127, 128, 255, 256, 299]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--fish-speech", type=Path, required=True, help="fish-speech checkout at 0f8a12e8")
    parser.add_argument("--slice-dir", type=Path, default=DEFAULT_SLICE_DIR)
    parser.add_argument("--out", type=Path, default=None, help="default: --slice-dir")
    parser.add_argument("--router-weights", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--route-tie", type=float, default=1e-5)
    parser.add_argument("--skip-commit-check", action="store_true")
    return parser.parse_args()


def tower_frames() -> list[list[int]]:
    rows = []
    for r in range(TOWER_ROWS):
        if r < TEXT_ROWS:
            rows.append([1000 + r] + [0] * 10)
            continue
        codes = [(r * 7919 + 17) % 4096] + [(r * 977 + k * 131 + r * k * 31) % 1024 for k in range(1, 10)]
        if r % 37 == 0:
            codes[5] = 1024 + (r * 29) % 3072
        rows.append([SEMANTIC_START + codes[0]] + codes)
    return rows


def check_slices(slice_dir: Path) -> dict:
    out = {}
    for key in ("a", "b"):
        path = slice_dir / f"slice_{key}.safetensors"
        sha_file = path.with_name(path.name + ".sha256")
        if not path.is_file() or not sha_file.is_file():
            sys.exit(f"{path} (and its .sha256) missing: run fetch_fish_drama3_slices.py --out {slice_dir}")
        recorded = sha_file.read_text().split()[0]
        digest = common.sha256_file(path)
        if digest != recorded:
            sys.exit(f"{path}: sha256 {digest} is not the recorded {recorded}; re-fetch with --force")
        out[key] = {"file": path.name, "sha256": digest}
    return out


def hooked(module, store: list):
    def hook(_module, _inputs, output):
        out = output[0] if isinstance(output, tuple) else output
        store.append(out.detach().float().reshape(-1, out.shape[-1]).cpu().clone())

    return module.register_forward_hook(hook)


def input_record(seed: int, rows: int, tensor) -> dict:
    return {"seed": seed, "rows": rows, "width": tensor.shape[-1], "scale": 1.0, "checksum": common.checksum(tensor)}


def run_trunk_layer(fish, config, slice_path: Path, layer_index: int, sink, args) -> dict:
    import torch
    from safetensors import safe_open

    text = config.text
    hidden = text.hidden_size
    with torch.device("meta"):
        backbone = fish.backbone.FishOmniBackbone(text, fp8_experts=False, dtype=torch.float32)
    layer = fish.backbone.DecoderLayer(text, layer_index, fp8=False, dtype=torch.float32, device="cpu")
    backbone.model.layers = torch.nn.ModuleList([layer])
    backbone.model.embed_tokens = None
    backbone.lm_head = None
    backbone.model.norm = fish.layers.ZeroCenteredRMSNorm(hidden, text.rms_norm_eps, dtype=torch.float32, device="cpu")
    prefix = f"model.language_model.layers.{layer_index}."

    def weights():
        with safe_open(str(slice_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(prefix):
                    # The layer sits at index 0 of the one-layer list.
                    yield "model.language_model.layers.0." + key[len(prefix):], handle.get_tensor(key)
        yield "model.language_model.norm.weight", torch.zeros(hidden)

    fish.backbone.load_backbone_weights(backbone, weights())
    backbone.eval()
    total = HISTORY + DECODE
    backbone.setup_caches(max_batch_size=1, max_seq_len=total)
    x = common.xorshift_tensor(total, hidden, TRUNK_SEED + layer_index, 1.0)
    mixer = layer.self_attn if layer.layer_type == "full_attention" else layer.linear_attn
    stores = {"output": [], "mixer": [], "moe": [], "router": []}
    handles = [hooked(layer, stores["output"]), hooked(mixer, stores["mixer"]),
               hooked(layer.mlp, stores["moe"]), hooked(layer.mlp.gate, stores["router"])]
    try:
        with torch.inference_mode():
            spans = CHUNKS + [(p, p + 1) for p in range(HISTORY, total)]
            for start, end in spans:
                backbone(inputs_embeds=x[None, start:end], input_pos=torch.arange(start, end))
    finally:
        for handle in handles:
            handle.remove()
    base = f"trunk/{layer_index}"
    entry = {
        "layer": layer_index,
        "kind": layer.layer_type,
        "input": input_record(TRUNK_SEED + layer_index, total, x),
        "prefill_chunks": [list(c) for c in CHUNKS],
        "decode_positions": list(range(HISTORY, total)),
    }
    for part in ("output", "mixer", "moe"):
        values = torch.cat(stores[part])
        if values.shape[0] != total:
            raise RuntimeError(f"layer {layer_index} {part}: {values.shape[0]} rows recorded")
        entry[part] = {
            "decode": sink.put(f"{base}/{part}/decode", values[HISTORY:]),
            "decode_checksum": common.checksum(values[HISTORY:]),
            "history": common.trace_entry(sink, f"{base}/{part}/history", values[:HISTORY], HISTORY_PICKS),
        }
    router = torch.cat(stores["router"])
    top_k = text.num_experts_per_tok
    decode_routes = common.route_entry(router, top_k, list(range(HISTORY, total)), args.route_tie)
    history_routes = common.route_entry(router[:HISTORY], top_k, [], args.route_tie)
    entry["routes"] = {
        "decode": decode_routes,
        "history": {k: history_routes[k] for k in ("near_ties", "near_tie_threshold", "min_margin")},
    }
    del backbone, layer
    gc.collect()
    return entry


def run_tower(fish, config, slice_path: Path, sink) -> tuple[dict, dict, dict]:
    import torch
    from safetensors import safe_open

    hidden_size = config.text.hidden_size
    with torch.device("meta"):
        decoder = fish.tower.FishOmniAudioDecoder(config, dtype=torch.float32)
    decoder.to_empty(device="cpu")

    def tower_weights():
        with safe_open(str(slice_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith("audio_decoder"):
                    yield key, handle.get_tensor(key)

    fish.tower.load_audio_decoder_weights(decoder, tower_weights())
    decoder.eval()
    with safe_open(str(slice_path), framework="pt", device="cpu") as handle:
        norm_weight = handle.get_tensor("model.language_model.norm.weight").float()
        compact = handle.get_tensor("lm_head.weight.compact_rows").float()
        semantic_rows = handle.get_tensor("model.language_model.embed_tokens.weight.semantic_rows").float()
    candidates = 1 + config.semantic_end_token_id - config.semantic_start_token_id + 1
    if compact.shape[0] != candidates or semantic_rows.shape[0] != candidates - 1:
        raise RuntimeError(f"slice A rows: compact {tuple(compact.shape)}, semantic {tuple(semantic_rows.shape)}")

    frames_list = tower_frames()
    frames = torch.tensor(frames_list, dtype=torch.long)
    h = common.xorshift_tensor(TOWER_ROWS, hidden_size, TOWER_SEED, 1.0)
    positions = torch.arange(TOWER_ROWS)
    stack = list(decoder.shared_layers) + list(decoder.residual_layers)
    stores = [[] for _ in stack]
    handles = [hooked(layer, store) for layer, store in zip(stack, stores, strict=True)]
    sampler = common.UpstreamSampler(fish.tower)
    main_index = FORCED_FRAME[0] - config.semantic_start_token_id + 1
    sampler.install("forced", forced=[[main_index, main_index] + FORCED_FRAME[2:]])
    try:
        with torch.inference_mode():
            decoder.setup_caches(TOWER_ROWS)
            for start, end in CHUNKS:
                decoder.prefill(h[start:end], positions[start:end], frames[start:end], frames[start + 1 : end + 1])
            sampler.begin_step(HISTORY, frames[: HISTORY + 1, 0][-common.RAS_WINDOW:].tolist())
            logits = torch.zeros(1, config.text.vocab_size)
            sampled = decoder.decode_step(
                logits, h[HISTORY : HISTORY + 1], positions[HISTORY : HISTORY + 1],
                frames[HISTORY : HISTORY + 1], frames[: HISTORY + 1, 0],
                temperature=1.0, top_k=config.audio_decoder.vocab_size, top_p=1.0,
            )
            step = sampler.end_step([int(v) for v in sampled[0].tolist()], config.eos_token_id)
    finally:
        sampler.uninstall()
        for handle in handles:
            handle.remove()
    if step["frame"] != FORCED_FRAME:
        raise RuntimeError(f"forced tower step returned {step['frame']}")
    layers = []
    for index, store in enumerate(stores):
        values = torch.cat(store)
        if values.shape[0] != HISTORY + 1:
            raise RuntimeError(f"tower layer {index}: {values.shape[0]} rows recorded")
        layers.append(common.tower_kind(index) | {
            "prefill": common.trace_entry(sink, f"tower/{index}/prefill", values[:HISTORY], TOWER_PICKS),
            "step": sink.put(f"tower/{index}/step", values[HISTORY]),
        })
    head_logits = torch.stack([call["logits"] for call in step["calls"][2:]])
    tower = {
        "hidden": input_record(TOWER_SEED, TOWER_ROWS, h),
        "frames": frames_list,
        "prefill_rows": HISTORY,
        "prefill_chunks": [list(c) for c in CHUNKS],
        "position": HISTORY,
        "forced_frame": FORCED_FRAME,
        "layers": layers,
        "head_logits": sink.put("tower/head_logits", head_logits),
    }

    x3 = common.xorshift_tensor(4, hidden_size, HEAD_SEED, 1.0)
    norm = fish.layers.ZeroCenteredRMSNorm(hidden_size, config.text.rms_norm_eps, dtype=torch.float32)
    with torch.inference_mode():
        norm.weight.copy_(norm_weight)
        head = torch.nn.Linear(hidden_size, candidates, bias=False, dtype=torch.float32)
        head.weight.copy_(compact)
        main_logits = head(norm(x3))
        embed_frames = frames[EMBED_ROWS]
        text_embeddings = semantic_rows[embed_frames[:, 0] - config.semantic_start_token_id]
        embedded = decoder.embed_backbone(embed_frames, text_embeddings)
    main_head = {
        "input": input_record(HEAD_SEED, 4, x3),
        "candidates": f"eos {config.eos_token_id}, then "
        f"{config.semantic_start_token_id}..{config.semantic_end_token_id}",
        "logits": sink.put("main_head/logits", main_logits),
    }
    embedding = {
        "rows": EMBED_ROWS,
        "frames": [frames_list[r] for r in EMBED_ROWS],
        "rule": "(E_lm[main] + CB[1 * 4096 + c1]) / sqrt(2) on semantic rows",
        "values": sink.put("embedding/values", embedded),
    }
    del decoder
    gc.collect()
    return tower, main_head, embedding


def main() -> int:
    args = parse_args()
    import torch

    common.set_torch_reference_mode()
    slice_dir = args.slice_dir.resolve()
    out = (args.out or slice_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    slices = check_slices(slice_dir)
    config_sha = common.check_pinned_file(slice_dir / "config.json", "config.json")
    fish = common.import_fish_speech(args.fish_speech, args.skip_commit_check)
    config = fish.config.FishOmniConfig.from_dict(json.loads((slice_dir / "config.json").read_text()))
    undo = common.install_expert_patches(fish.moe, bf16_router=args.router_weights == "bf16", force_w8a16=False)
    checks = {"sampler_patch": common.verify_sampler_patch(fish.tower)}
    sink = common.TensorSink()
    try:
        trunk = []
        for layer in TRUNK_LAYERS:
            print(f"trunk layer {layer} (slice B) ...", file=sys.stderr)
            trunk.append(run_trunk_layer(fish, config, slice_dir / "slice_b.safetensors", layer, sink, args))
        print("tower, main head and embedding (slice A) ...", file=sys.stderr)
        tower, main_head, embedding = run_tower(fish, config, slice_dir / "slice_a.safetensors", sink)
    finally:
        undo()
    tensors_sha = sink.save(out / "slice_expected.safetensors", {"schema": SCHEMA, "generator": GENERATOR})
    provenance = common.package_versions(["torch", "safetensors", "tokenizers", "loguru"])
    provenance |= {"fish_speech_commit": fish.commit, "torch_threads": torch.get_num_threads()}
    result = {
        "schema": SCHEMA,
        "schema_version": 1,
        "generator": GENERATOR,
        "created_at": common.now_utc(),
        "provenance": provenance,
        "reference_semantics": common.REFERENCE_SEMANTICS
        | {"experts_loader": "load_backbone_weights into dense F32 experts (dequantize_fp8_blocks)",
           "router_weights_mode": args.router_weights},
        "model": {"repo_id": common.REPO_ID, "revision": common.CHECKPOINT_REVISION,
                  "config_sha256": config_sha, "slices": slices},
        "tensors_file": "slice_expected.safetensors",
        "tensors_sha256": tensors_sha,
        "trunk_layers": trunk,
        "tower": tower,
        "main_head": main_head,
        "embedding": embedding,
        "self_checks": checks,
    }
    common.dump_json(result, out / "slice_expected.json")
    print(f"wrote {out / 'slice_expected.json'}", file=sys.stderr)
    return 3 if checks["sampler_patch"]["mismatches"] else 0


if __name__ == "__main__":
    sys.exit(main())
