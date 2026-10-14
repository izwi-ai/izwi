//! Izwi server binary for the community build.

fn main() -> anyhow::Result<()> {
    izwi_server::run_server_process(izwi_hooks::EnterpriseHooks::noop())
}
