//! Izwi server binary for the community build.

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    izwi_server::run_from_cli(izwi_hooks::EnterpriseHooks::noop()).await
}
