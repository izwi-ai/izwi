# Qwen3.8 27B L40S validation handoff

This is the manual CUDA validation protocol for `Qwen3.8-27B-FP8`. The
optimization candidates described here are scoped to the separate Qwen3.8
model family. They are all disabled by default and have not been validated on
an NVIDIA GPU in this repository.

The development host used for the implementation had no usable NVIDIA device.
It could verify builds, CPU reference behavior, policy tests, runner routing,
and fail-closed behavior, but it could not establish CUDA correctness,
throughput, latency, VRAM use, kernel selection, or numerical parity. Do not
interpret compilation or an `unsupported` certificate as device evidence.

## Required deployment

Use an NVIDIA L40S host with the pinned
`Qwen/Qwen3.8-27B-FP8` checkpoint revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`. The checked-out repository, the
release CLI, and the running server must all use the same Git SHA. Build the
CLI and server with the intended CUDA feature set, start the server with
`--backend cuda`, preload or load `Qwen3.8-27B-FP8`, and warm it before
collecting a bundle.

The server process owns the candidate switches. Restart the server between
cells so each bundle has exactly the intended environment:

| Cell | Server environment | Default | CUDA status |
|---|---|---|---|
| Baseline | All five variables unset or `0` | Active fallback | Required comparison cell; not a certification of a candidate |
| Packed projections | `IZWI_QWEN38_PACKED_PROJECTIONS=1` | Off | Implemented candidate; unvalidated |
| BF16 KV storage | `IZWI_QWEN38_CUDA_BF16_KV=1` | Off; F16 fallback remains active | Implemented candidate; unvalidated |
| Causal-convolution decode kernel | `IZWI_QWEN38_CAUSAL_CONV_DECODE=1` | Off | Implemented candidate; unvalidated |
| DeltaNet decode kernel | `IZWI_QWEN38_DELTANET_DECODE=1` | Off | Implemented candidate; unvalidated |
| Decode epilogues | `IZWI_QWEN38_DECODE_EPILOGUES=1` | Off | Implemented SiLU×mul, L2Norm, and gated-RMSNorm candidate; unvalidated |

The five variables accept the normal true values (`1`, `true`, `yes`, or
`on`). Use one candidate at a time before testing combinations. Projection
packing and KV selection occur while loading the model, so changing their
environment without reloading does not change the loaded instance. A missing,
unsupported, or failed candidate kernel retains the existing fallback path;
the runtime telemetry must prove which path actually executed.

For a baseline server, the relevant startup shape is:

```bash
IZWI_BACKEND=cuda \
IZWI_PRELOAD_MODELS=Qwen3.8-27B-FP8 \
IZWI_WARMUP_PRELOADED_MODELS=1 \
target/release/izwi serve --backend cuda
```

For a candidate cell, add exactly one variable from the table to that server
environment. Keep the terminal and server log with the evidence bundle.

## Exact evidence command

From the root of the same checkout, with the warmed server listening on port
8080, run exactly:

```bash
scripts/bench/run-qwen38-l40s-evidence.sh \
  --server http://127.0.0.1:8080 \
  --izwi-bin target/release/izwi \
  --output target/qwen38-l40s-evidence
```

Move or rename that output directory before the next cell; every cell needs a
separate complete bundle. The runner requires `bash`, `git`, `jq`, `curl`, a
working `nvidia-smi`, an executable release CLI, a loopback Izwi server, and
the versioned workload at
`benchmarks/manifests/qwen38-l40s-evidence.json`. The optional runner-only
variables `IZWI_QWEN38_EVIDENCE_IZWI`,
`IZWI_QWEN38_EVIDENCE_NVIDIA_SMI`, and
`IZWI_QWEN38_EVIDENCE_CUDA_RUNNER` override tool paths; they do not enable a
model optimization.

Retain the entire output directory, including:

- `certificate.json`, with the workload hash, checkpoint revision, device,
  Git SHA, and measured per-case results;
- `imported-manifest.toml`;
- `nvidia-smi.csv`, `nvidia-smi-q.txt`, `uname.txt`, and `nvcc-version.txt`
  when `nvcc` is available;
- `cuda-evidence/certificate.json`, `cuda-evidence/health.json`, and
  `cuda-evidence/runner.log`;
- `cuda-evidence/benchmark/report.json`, `metadata.json`,
  `observability.json`, and `manifest.toml`;
- the matching server log and the exact server environment used for the cell.

The outer certificate reports both end-to-end completion throughput and
decode throughput derived from the server-reported generation interval. Keep
those metrics separate. The workload also records observed prompt tokens; its
`prompt_words` value is only a deterministic input-size target.

## Candidate review and promotion

Review the baseline first, then each isolated candidate, then only combinations
whose isolated cells passed. A candidate can be considered for default-on
promotion only when all of the following are true on the exact L40S deployment:

1. The runner exits successfully and both certificates say `passed`; the
   device is an observed L40S, the server selects CUDA, and every Git SHA and
   checkpoint revision matches.
2. Every workload case passes strict quality gates with no panic, worker
   restart, request-failure increase, non-finite output, or context/admission
   regression relative to the default-off baseline.
3. Qwen3.8 runtime diagnostics and before/after telemetry prove that the
   requested provider or kernel was selected. An attempt followed only by
   fallback does not validate a candidate.
4. Numerical output is checked against the default-off CUDA path for short,
   sustained, and long-context cases using a tolerance appropriate to the
   candidate dtype. BF16 KV additionally requires attention/KV parity and
   stable long-context generation.
5. Peak VRAM and host memory remain within the deployment budget, without
   reducing the resource-fitted context below the deployment requirement.
6. The candidate produces a repeatable material improvement in its intended
   metric beyond run-to-run variation, without an unacceptable regression in
   TTFT, sustained decode, long-context latency, or concurrency 1/2/4/8.
   Record the acceptance threshold before collecting the promotion runs; this
   repository contains no synthetic threshold or result.
7. The explicit `0` kill switch is exercised after the candidate run and is
   shown to restore the current fallback behavior.

Keep candidates default-off when evidence is incomplete, contradictory, or
shows only compilation/dispatch eligibility. Promotion should be a separate,
reviewable change bound to the retained bundles.

## Deferred evidence-gated phases

The following advanced work is intentionally not part of the initial
default-off candidate set:

- Q8 projection-kernel tuning;
- whole-token CUDA graph capture and replay;
- native block-scaled FP8 execution;
- multi-token prediction (MTP);
- continuous batching.

These phases change numerical behavior, memory ownership, scheduling, or the
execution graph broadly enough that they cannot be enabled safely before the
current baseline and isolated candidates have real L40S evidence. Each phase
needs its own design, kill switch, telemetry, correctness comparison, resource
accounting, and versioned hardware workload. No result for these phases has
been measured or synthesized here.

On a host without an NVIDIA GPU, use `--dry-run` to validate manifest import or
`--allow-unsupported` only to exercise unsupported reporting. Neither path is
eligible for promotion.
