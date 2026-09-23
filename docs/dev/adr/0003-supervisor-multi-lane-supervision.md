# ADR 0003: Supervisor supervises CPU, Metal, and CUDA lanes

Status: Accepted
Date: 2026-09-23

## Context

The serving supervisor originally refused every non-CPU assignment
(`require_cpu_only`) while its node configuration, launch environment, and the
core's assigned-device selection already validated Metal and CUDA lanes. The
serving plan's Phase 3 requires "explicit, independently supervised
CPU/Metal/CUDA worker assignments", and the distributed-serving plan (DS0.6)
makes three-lane supervision a requirement rather than a decision. The serving
support matrix previously recorded "the supervisor executable rejects Metal
assignments".

## Decision

1. The supervisor accepts worker configurations for all three backends. The
   `require_cpu_only` gate is removed.
2. Every binary flavor referenced by the node configuration must be supplied
   (`--cpu-worker-binary`, `--metal-worker-binary`, `--cuda-worker-binary`);
   a missing flavor fails against the worker that references it
   (`MissingFlavorBinary`), and flavor/backend mismatches still fail config
   validation.
3. Device inventory remains **operator-declared** (`--metal-devices
   id@index`, `--cuda-devices uuid@host_index@total_memory_bytes`). The
   supervisor never probes or initializes accelerators, keeping the
   accelerator-free boundary (`scripts/ci/check-serving-boundary.sh`)
   intact. Declared Metal devices are treated as unified-memory; config
   validation rejects non-unified Metal declarations.
4. Honest lane evidence is preserved and separated:
   - launch/supervision proof is process-level with fake lane workers
     (`tests/multi_lane_launch.rs`) that verify the exact device environment
     (`IZWI_BACKEND`, expected device identity, device visibility, model-load
     slots) and graceful supervisor stop;
   - real model execution on Metal or CUDA remains a separately recorded
     hardware-gated lane and is never implied by supervision support.

## Consequences

- A Metal/CUDA deployment is now fully supervisable end to end on a host with
  that accelerator, using the same lifecycle (readiness, restart backoff,
  quarantine, drain) as CPU workers.
- Wrong device declarations fail closed: either config validation (unknown
  device, index mismatch, non-unified Metal) or the worker's own
  assigned-device selection at startup, which the supervisor's readiness gate
  treats as a failed worker.
- Real Metal/CUDA inference evidence and performance claims remain open items
  in the support matrix until executed on the respective hardware.
