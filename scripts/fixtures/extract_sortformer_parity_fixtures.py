#!/usr/bin/env python3
"""Compact the 2026-10-14 NeMo reference runs into Sortformer parity fixtures.

The references came from two faithful PyTorch ports of NeMo main's streaming
``SortformerEncLabelModel`` (CPU F32): ``ref_sortformer.py`` for
diar_streaming_sortformer_4spk-v2.1 and ``ref_nemotron.py`` for
Nemotron-3-Diarization (cross-checked against HF transformers 5.19.0,
max |dp| <= 1.5e-6). Those generators were not kept; their output JSONs are
the source here, and the README next to the fixtures records their
provenance. This script only selects and renames fields, so the fixtures
carry the reference values unchanged.

Usage (run from the repo root):

    python3 -I scripts/fixtures/extract_sortformer_parity_fixtures.py \
        <evidence dir> crates/izwi-core/tests/fixtures/sortformer_parity
"""
import json
import pathlib
import sys

V21_STREAMING_KEYS = (
    "chunk_len", "chunk_right_context", "fifo_len", "spkcache_update_period", "spkcache_len",
)
NEMOTRON_STREAMING_KEYS = V21_STREAMING_KEYS + ("chunk_left_context",)


def v21_run(run, with_probs):
    out = {
        "streaming": {key: run["streaming"][key] for key in V21_STREAMING_KEYS},
        "chunks": run["chunks"],
        "compressions": run["compressions"],
        "output_frames": run["output_frames"],
        "probs_summary": run["probs_summary"],
        "segments": run["segments_legacy_loop_10ms"],
    }
    if with_probs:
        out["probs_80ms"] = run["probs_80ms"]
    return out


def nemotron_run(run, probs_key):
    out = {
        "streaming": {key: run["streaming"][key] for key in NEMOTRON_STREAMING_KEYS},
        "segments": [[s["start"], s["end"], s["speaker"]] for s in run["segments"]],
    }
    if probs_key == "probs_10ms":
        out["probs_10ms"] = run["probs_10ms"]
    else:
        out["probs_100ms_active_speakers"] = run["probs_100ms_active_speakers"]
        out["summary"] = run["summary"]
    return out


def write(out_dir, name, payload):
    path = out_dir / name
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"{name}: {path.stat().st_size} bytes")


def main():
    evidence, out_dir = map(pathlib.Path, sys.argv[1:3])
    out_dir.mkdir(parents=True, exist_ok=True)
    load = lambda name: json.loads((evidence / name).read_text())

    for fixture, source, audio in (
        ("diarization", "02-sortformer-reference-diarization.json", "diarization.soxr16k.f32"),
        ("diarization-2", "02-sortformer-reference-diarization-2.json",
         "diarization-2.soxr16k.f32"),
    ):
        ref = load(source)
        runs = ref["runs"]
        write(out_dir, f"v21-{fixture}.json", {
            "source": ref["reference"],
            "audio": audio,
            "frame_secs": 0.08,
            "postprocessing": "NeMo ts_vad_post_processing, onset=offset=0.5, no padding or "
                              "minimum durations, raw channel ids",
            "runs": {
                "model": v21_run(runs["model_profile_izwi_default"], True),
                "card_high": v21_run(runs["card_very_high_latency"], True),
                "card_low": v21_run(runs["card_low_latency"], False),
            },
        })

    sparse = load("02-sortformer-reference-sparse-ordering.json")
    write(out_dir, "v21-sparse2.json", {
        "source": "ref_sortformer.py (see v21-diarization.json)",
        "audio": "sparse2.soxr16k.f32",
        "construction": sparse["construction"],
        "frame_secs": 0.08,
        "runs": {
            "model": v21_run(sparse["model_nemo_order"], False),
            "card_high": v21_run(sparse["card_high_nemo_order"], False),
            "card_high_izwi_order_before_srt1": v21_run(sparse["card_high_izwi_order"], False),
        },
    })

    parity = load("03-nemotron-diar-reference-parity-fixture-10s.json")
    write(out_dir, "nemotron3-diarization-10s.json", {
        "source": parity["generator"],
        "audio": "diarization_10s.ffmpeg16k_s16.f32",
        "input": parity["input"],
        "frame_secs": 0.01,
        "runs": {
            name: nemotron_run(run, "probs_10ms") for name, run in parity["runs"].items()
        },
    })

    diar2 = load("03-nemotron-diar-reference-diarization-2.json")
    write(out_dir, "nemotron3-diarization-2.json", {
        "source": diar2["reference"],
        "audio": "diarization-2.ffmpeg16k.f32",
        "input": diar2["input_16k"]["ffmpeg16k"],
        "frame_secs": 0.01,
        "runs": {
            name: nemotron_run(diar2["runs"][name], "probs_100ms_active_speakers")
            for name in ("yaml", "card_offline")
        },
    })


if __name__ == "__main__":
    main()
