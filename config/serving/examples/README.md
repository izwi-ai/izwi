# Izwi node configuration examples

These files are parser-compatible examples for schema version 1 of the local
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

The current node schema configures workers and exactly one deployment per
worker. It does not yet accept a `task` key, gateway settings, `profile`,
`topology`, worker-pool declarations, remote endpoints, or artifact-provider
configuration. The implemented private worker route is chat-only, so these
examples must not be read as support for speech or other task types. A distinct
deployment can be assigned to a separate worker, as shown in the CUDA example,
but multi-stage execution is not implied.

All worker binds are loopback-only because this supervisor generation is for a
single trusted node. The `bearer_token_env` values name environment variables;
the files intentionally contain no bearer tokens or other secret material.
CUDA process-local device indexes remain `0`: the supervisor isolates each
process to its assigned UUID instead of changing global device state inside a
live process.

Parsing proves only that an example matches the bounded schema. Successful
validation and launch additionally require real directories, executable worker
binaries, available CPU resources, and exact accelerator inventory matches.
The `izwi-serving-supervisor` command can launch the CPU example only when given
an explicit config path, CPU worker binary, trusted CPU-ID list, and allocatable
host-memory ceiling. Its current command-line inventory rejects Metal and CUDA
assignments without fallback; those lanes still require a future trusted device
inventory integration. Nothing in these examples is hardware validation, a
certified memory profile, or a performance claim.
