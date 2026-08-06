# KV cache certification benchmark

`run_kv_cache_matrix.sh` exercises the public managed-KV arena ABI without
loading a model. Its default matrix covers:

- default features on CPU, plus Metal/CUDA feature lanes when a compatible
  host device is detected;
- 16- and 32-token physical pages;
- ragged short contexts and configurable long contexts;
- paged prefill, paged decode, page copy, page zero, and slot writes.

Run the supported matrix and save its JSON Lines output:

```bash
scripts/bench/run_kv_cache_matrix.sh --output target/kv-cache-matrix.jsonl
```

Run a single lane or reduce the iteration count during development:

```bash
scripts/bench/run_kv_cache_matrix.sh --lane default --iterations 3 --warmup 1
```

The harness synchronizes after every measured operation. Reported latency is
therefore dispatch-to-completion latency at the arena boundary, not an
end-to-end model latency or throughput result. Dispatch counts come from the
arena's monotonic operation counters. The ABI currently has no dedicated
prefill counter: fused accelerator prefill advances the paged-decode counter,
while CPU prefill reports zero dispatches. Preserve that distinction when
comparing paths.

Unsupported accelerator lanes emit `status: "unsupported"` records and are not
compiled or presented as measurements. A visible accelerator can still reject
a case at runtime; that produces `status: "failed"` and a non-zero matrix exit.
The runner deliberately does not infer CUDA availability from feature
compilation alone.

The CUDA lane builds with `--features flash-attn` (which implies `cuda`) and
uses F16 tensors. Its zero-offset page-32 cases are eligible for Candle FA2;
page-16 cases exercise Izwi's native CUDA fallback. Each measurement records
the intended `kernel_path`. Treat this as a long-running, nightly production
certification lane: the first CUDA/FA2 build can be substantial and requires a
compatible CUDA toolchain as well as a device. This script does not claim or
substitute CUDA measurements on hosts lacking either prerequisite.

Validate argument routing and capability classification without compiling:

```bash
scripts/bench/test-run-kv-cache-matrix.sh
```
