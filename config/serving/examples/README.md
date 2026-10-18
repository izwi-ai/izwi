# Izwi node configuration examples

These files are parser-compatible examples for schema version 2 of the local
node supervisor configuration. They are not ready-to-run hardware profiles.
Every path, budget, device identity, artifact revision, and credential
environment variable must be replaced and validated against the target host.

- `one-device-cpu.toml` assigns one CPU resource group to one worker.
- `one-device-metal.toml` assigns one discovered Apple Metal device to one
  worker. Its `metal:` device identity is a placeholder for the registry ID
  reported by Izwi on the target Mac.
- `multi-device-cuda.toml` assigns one NVIDIA GPU UUID to each worker. Two
  workers advertise the same deployment identity as independent replicas; a
  third worker demonstrates a distinct deployment on another device.
- `docker-compose.serving.yml` (in `config/serving/`) demonstrates the
  separated gateway and worker containers under Docker Compose, in the
  single-node `standalone` topology (the gateway shares the worker's network
  namespace and reaches it over loopback).

The current node schema configures workers and exactly one task-specific
deployment/capability contract per worker. It does not accept gateway settings,
`profile`, `topology`, worker-pool declarations, remote endpoints, or
artifact-provider configuration. These examples declare chat
deployments; speech workers (`speech_to_text`, `text_to_speech`) use the same
schema with their own task and capability. A distinct deployment can be assigned to a separate worker, as
shown in the CUDA example, but multi-stage execution is not implied.

All worker binds are loopback-only because this supervisor generation is for a
single trusted node. The `bearer_token_env` values name environment variables;
the files intentionally contain no bearer tokens or other secret material.
CUDA process-local device indexes remain `0`: the supervisor isolates each
process to its assigned UUID instead of changing global device state inside a
live process.

Parsing proves only that an example matches the bounded schema. Successful
validation and launch additionally require real directories, executable worker
binaries, available CPU resources, and exact accelerator inventory matches.
The `izwi-serving-supervisor` command launches an example only when given an
explicit config path, the worker binary for each lane it uses, the trusted
device inventory (`--cpu-ids`; `--metal-devices device_id@index`;
`--cuda-devices uuid@host_index@total_memory_bytes`), and an allocatable
host-memory ceiling. A device that is not declared fails validation without
fallback. Nothing in these examples is hardware validation, a
certified memory profile, or a performance claim.

For the exact single-node startup, readiness, security, drain, rollback, and
evidence boundaries, see the
[single-node operations runbook](../../../docs/dev/PRODUCTION_SERVING_SINGLE_NODE_RUNBOOK.md).
