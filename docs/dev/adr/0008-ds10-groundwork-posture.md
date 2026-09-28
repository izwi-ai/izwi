# ADR 0008: DS10 groundwork is best-effort buildable now; activation criteria unchanged

Status: Accepted
Date: 2026-09-28

## Context

DS10 of the distributed-serving plan is a deferred register — "no build
without entry criteria; maintained as decisions, not tasks" — holding four
entries: MoE architectures + expert parallelism, prefill/decode
disaggregation (PD), in-engine tensor parallelism (permanently rejected in
favor of the DS8 vLLM lane), and cross-node/cross-process KV transfer
(rejected; DS4 stays node-local). The register exists to prevent speculative
distributed-systems builds whose value depends on hardware or models we do
not have: no MoE chat family in `models/architectures/`, no validated
RDMA-class fabric, no measured ITL-SLO violation.

A blanket build freeze has a cost the register did not price in: when the
entry criteria are eventually met, the first build pays for engine seams,
fixtures, and admission correctness from scratch, and the measurement the
criteria demand (e.g. ITL p99 under long-prompt saturation) has no
pre-registered workload to run. Separately, one register item is not
speculative at all: the admission-time load-peak formula in
`estimate_from_tensor_inventory` assumes a checkpoint's dominant weight
tensor dominates its load scratch, which is false for expert-parallel
weights and under-reserves INV-10 headroom for any future MoE load.

The operator has directed best-effort groundwork so that "when we have MoE
models they will just work", while keeping the register's protective intent.

## Decision

1. **Two-tier amendment.** Each DS10 entry splits into a *groundwork tier*
   (buildable now, best effort, without satisfying entry criteria) and an
   *activation tier* (gated by exactly the criteria the register already
   records). The full plan lives in
   `docs/dev/DS10_BEST_EFFORT_GROUNDWORK_PLAN.md`; this ADR fixes its
   boundaries.
2. **Groundwork may build:** the MoE runtime core behind a single-device
   sparse-expert dispatch seam, synthetic tiny-model fixtures (no real MoE
   download — that is activation), capability cells with evidence-gated
   NotRun/Disabled posture, the inventory-aware admission scratch fix, and
   expert-activation telemetry. It may also produce: PD and page-transfer
   design specifications, a test-only mock-transport rig, in-process KV
   codec round-trip tests, and pre-registered benchmark workloads.
3. **Groundwork must not:** enable any cross-process or cross-node KV path
   in the engine (the serving-plan exclusion stands), add collective or
   multi-device execution (the TP rejection stands), add protocol fields
   with no producer (role tags, expert shards, KV handles are specified in
   the design docs and land with their features), or advertise an unvalidated
   real model as served-ready (a MoE catalog variant lands disabled until
   activation evidence exists).
4. **Verification posture is unchanged:** every groundwork commit is proven
   on CPU (and Metal where the surface demands it) on this host with
   synthetic fixtures; CUDA stays code-complete + unit-tested with execution
   evidence `not run`; mock ≠ evidence and not-run ≠ passed (INV-08 and the
   plan's evidence rules apply to groundwork verbatim).

## Consequences

- When a MoE model is added to the catalog, activation is a data task
  (metadata, capability cells with real evidence, download manifests) plus
  validation, not an engine project — the family, fixtures, admission math,
  and telemetry already exist and are fixture-proven.
- The DS10 activation gates are now *cheap to evaluate honestly*: the
  long-prompt workload pre-registers the ITL measurement; the mock rig
  pre-proves the KV handoff contract; nothing waits on research that could
  have been done as tests.
- The rejected entries remain rejected in production behavior; they gain
  only documentation and in-process tests, which the boundary gate and the
  register's text continue to enforce.
- Groundwork commits consume review and CI capacity on code whose activation
  date is unknown; accepted deliberately, because the alternative —
  engine-wide seams designed under activation pressure — is the failure mode
  the register was written to avoid.
