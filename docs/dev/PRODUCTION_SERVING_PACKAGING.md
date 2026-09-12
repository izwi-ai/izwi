# Production serving packaging evidence

This document records the narrow native-packaging gate for Izwi's separated
gateway and inference worker. It does not certify the complete production
serving roadmap or any accelerator runtime.

## Packaged contract

The existing `izwi-server` executable provides the hardware-independent
`--role gateway` mode. The terminal release archive now also contains
`izwi-serving-worker` and `izwi-serving-node.example.toml`. The archive verifier
requires all three serving artifacts, as well as the existing CLI and desktop
binary, before publication.

The example node configuration is parser-compatible and intentionally bounded,
but it is not a certified deployment profile. Operators must replace its
absolute paths, memory/thread budgets, pinned artifact revision, and secret
environment variable. It is safe to publish because it contains no credential.

There is no packaged supervisor executable yet. `izwi-serving-supervisor`
currently provides validated configuration, assignment locks, child launch
specifications, readiness/drain behavior, and restart policy as a Rust library.
Until a real operator entry point owns that library and passes process-level
packaging tests, the archive must not claim one-command standalone supervision.

## Platform evidence

| Platform/profile | Packaged worker | Evidence in this slice | Not established |
|---|---|---|---|
| Linux x86_64 native release | CPU-only | Release workflow builds the default worker and the POSIX verifier requires it in the terminal archive. | This checkout did not execute a Linux release build or installer. |
| Windows x86_64 native release | CPU-only | Release workflow includes the worker and example in the terminal zip; the PowerShell verifier requires both. | PowerShell and a Windows runner were unavailable locally, so the zip/installer lane was not executed here. |
| macOS Apple Silicon native release | Metal-capable worker, with the existing local compatibility behavior | Release workflow builds the worker with `metal`, checks its macOS 12 deployment floor, and requires it in the terminal archive. The POSIX verifier was exercised with a synthetic archive on Apple Silicon. | No release binary was built in this low-resource session, and no real Metal inference was executed by this slice. |
| NVIDIA CUDA worker | Not in native GitHub release archives | CUDA remains an explicit source/container build profile. | No CUDA compilation, CUDA device execution, or performance evidence was produced on this machine. |

The macOS terminal bundle still uses a Metal-capable `izwi-server` to preserve
the existing local server workflow. Gateway mode itself does not initialize the
runtime, but there is not yet a separately named CPU-only gateway artifact.
Linux and Windows native artifacts remain CPU-only.

## Reproducible static smoke check

Run:

```sh
bash -n scripts/release/verify-native-artifacts.sh
bash -n scripts/release/test-native-serving-package.sh
scripts/release/test-native-serving-package.sh
cargo test -p izwi-serving-supervisor --test packaged_config -- --test-threads=1
```

The package script creates bounded temporary tar and zip archives, proves that
the valid gateway/worker/config contract passes, and proves that tar archives
missing either the worker or configuration example fail. The focused Rust test
passes the shipped example through the same bounded parser used by the
supervisor library. None of these checks starts a server, loads a model,
compiles an accelerator backend, or measures performance.
