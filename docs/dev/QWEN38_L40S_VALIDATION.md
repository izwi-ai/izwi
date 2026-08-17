# Qwen3.8 27B CUDA validation profiles

This is the manual CUDA validation protocol for `Qwen3.8-27B-FP8`. The
optimization candidates are scoped to the separate Qwen3.8 model family, but
they are not tied to the L40S: production dispatch selects the CUDA backend,
not a GPU product name. The L40S is the first strict evidence profile because
it is the deployment that motivated this work. The legacy CUDA candidates in
the table below remain disabled by default. Qwen3.8 MTP has a separate
default-on policy and four-cell evidence protocol; it has not yet been
validated on an NVIDIA GPU in this repository.

CUDA correctness and performance evidence is hardware-profile-specific. A
passing L40S bundle certifies only the versioned `nvidia-l40s-48gb` profile. It
does not certify, establish a throughput expectation for, or authorize a
global default on other NVIDIA architectures.

The development host used for the implementation had no usable NVIDIA device.
It could verify builds, CPU reference behavior, policy tests, runner routing,
and fail-closed behavior, but it could not establish CUDA correctness,
throughput, latency, VRAM use, kernel selection, or numerical parity. Do not
interpret compilation or an `unsupported` certificate as device evidence.

## Initial strict deployment profile

Use an NVIDIA L40S host with the pinned
`Qwen/Qwen3.8-27B-FP8` checkpoint revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`. The checked-out repository, the
release CLI, and the running server must all use the same Git SHA. Build the
CLI and server with the intended CUDA feature set, start the server with
`--backend cuda`, preload or load `Qwen3.8-27B-FP8`, and warm it before
collecting a bundle.

The L40S manifest records and enforces the selected device name, compute
capability 8.9, minimum VRAM, and a well-formed driver version. To test another
NVIDIA GPU, copy the versioned workload, assign a new `hardware_profile.id`,
and set that device's name regex, compute-capability regex, minimum VRAM, and
driver-version regex. Run the generic
`scripts/bench/run-qwen38-cuda-evidence.sh` command against the new manifest.
Do not weaken the L40S profile to make a different device pass.

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

## Exact L40S evidence command

From the root of the same checkout, with the warmed server listening on port
8080, run exactly:

```bash
scripts/bench/run-qwen38-l40s-evidence.sh \
  --mtp-depth 3 \
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
- `nvidia-smi.csv`, `nvidia-memory-samples.csv`, `nvidia-smi-q.txt`,
  `uname.txt`, and `nvcc-version.txt` when `nvcc` is available;
- `cuda-evidence/certificate.json`, `cuda-evidence/health.json`, and
  `cuda-evidence/runner.log`;
- `cuda-evidence/benchmark/report.json`, `metadata.json`,
  `observability.json`, and `manifest.toml`;
- the matching server log and the exact server environment used for the cell.

The outer certificate reports both end-to-end completion throughput and
decode throughput derived from the server-reported generation interval. Keep
those metrics separate. The workload also records observed prompt tokens; its
`prompt_words` value is only a deterministic input-size target.

For another versioned NVIDIA hardware profile, use:

```bash
scripts/bench/run-qwen38-cuda-evidence.sh \
  --workload benchmarks/manifests/qwen38-<profile>-evidence.json \
  --mtp-depth 3 \
  --server http://127.0.0.1:8080 \
  --izwi-bin target/release/izwi \
  --output target/qwen38-<profile>-evidence
```

The certificate records the exact device UUID/name, compute capability, total
and sampled peak-used/free VRAM, driver version, Git SHA, checkpoint revision,
observed provider, and resolved MTP policy. The selected server device must
match every constraint in the manifest's hardware profile.

## Paired MTP validation

Collect four otherwise identical runs from fresh server processes. Do not rely
on the default when recording evidence; set the process variables explicitly:

| Cell | Required process environment | Evidence argument |
|---|---|---|
| Disabled baseline | `IZWI_QWEN38_MTP=0` | `--mtp-depth 0` |
| Depth 1 | `IZWI_QWEN38_MTP=1 IZWI_QWEN38_MTP_DRAFT_TOKENS=1` | `--mtp-depth 1` |
| Depth 2 | `IZWI_QWEN38_MTP=1 IZWI_QWEN38_MTP_DRAFT_TOKENS=2` | `--mtp-depth 2` |
| Depth 3 | `IZWI_QWEN38_MTP=1 IZWI_QWEN38_MTP_DRAFT_TOKENS=3` | `--mtp-depth 3` |

The runner option labels expected state; it does not mutate the server. A real
cell passes only when loaded-model diagnostics prove the matching enabled flag
and draft depth and run-local counters prove disabled or active MTP execution.
After collecting the four retained bundles, run:

```bash
scripts/bench/certify-qwen38-mtp-evidence.sh \
  --baseline target/qwen38-mtp-disabled \
  --depth-1 target/qwen38-mtp-depth-1 \
  --depth-2 target/qwen38-mtp-depth-2 \
  --depth-3 target/qwen38-mtp-depth-3 \
  --output target/qwen38-mtp-paired
```

The certifier fails closed unless every cell has the same 40-character Git SHA
and checkpoint revision, workload hash, hardware profile, device identity,
hardware/compute/KV provider, and complete case names/configuration. Each cell
must have measured positive TTFT and completion throughput plus sampled peak
device memory. A dry-run pairing emits only `implemented_unvalidated`; four
measured cells emit `runtime_validated`. `performance_certified` additionally
requires a profile-scoped predeclared threshold object with:

```json
{
  "mtp": {
    "minimum_completion_tps_p50_speedup_ratio": 1.05,
    "maximum_ttft_p95_regression_ratio": 1.05,
    "maximum_peak_device_memory_ratio": 1.15
  }
}
```

These numbers are an example of the schema, not recommended L40S thresholds.
Set profile-specific values before measurement. With the checked-in null
thresholds the strongest possible result is `runtime_validated`; tooling never
turns missing thresholds into a performance claim.
Even a performance certificate is eligible only for its bound hardware
profile/device, checkpoint revision, and Git SHA; it cannot promote a global
CUDA default.

## Candidate review and profile-scoped promotion

Review the baseline first, then each isolated candidate, then only combinations
whose isolated cells passed. Declare acceptance thresholds in that hardware
profile before promotion runs; do not reuse absolute throughput thresholds
from a different GPU class. A candidate can be considered for an explicitly
profile-gated provider only when all of the following are true on the matching
deployment:

1. The runner exits successfully and both certificates say `passed`; the
   selected device matches the manifest's name, compute capability, VRAM, and
   driver constraints; the server selects CUDA; and every Git SHA and
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
reviewable change bound to the retained bundles and an explicit runtime
hardware capability/profile gate.

An L40S pass alone must never promote a candidate to the global CUDA default.
A global default requires retained correctness and performance evidence across
the repository's representative supported CUDA architecture matrix, including
each materially different compute capability, plus a fail-closed capability
policy for unobserved devices. Until that exists, promotion remains limited to
the tested profile or the candidate remains default-off.

## Deferred evidence-gated phases

The following advanced work is intentionally not part of the initial
default-off candidate set:

- Q8 projection-kernel tuning;
- whole-token CUDA graph capture and replay;
- native block-scaled FP8 execution;
- continuous batching.

These phases change numerical behavior, memory ownership, scheduling, or the
execution graph broadly enough that they cannot be enabled safely before the
current baseline and isolated candidates have real CUDA device evidence. Each
phase needs its own design, kill switch, telemetry, correctness comparison,
resource accounting, and versioned hardware workload. No result for these
phases has been measured or synthesized here.

On a host without an NVIDIA GPU, use `--dry-run` to validate manifest import or
`--allow-unsupported` only to exercise unsupported reporting. Neither path is
eligible for promotion.
