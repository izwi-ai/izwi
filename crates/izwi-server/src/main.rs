//! Izwi server binary for the community build. The entry point is
//! synchronous: `run_from_cli_blocking` starts the Tokio runtime after the
//! process-wide startup that must happen while the process is single-threaded.

fn main() -> anyhow::Result<()> {
    izwi_server::run_from_cli_blocking(izwi_hooks::EnterpriseHooks::noop())
}
