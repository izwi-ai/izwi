//! The standalone Izwi gateway. It links no inference engine; see the
//! izwi-gateway library for the flags and environment it reads.

use izwi_hooks::EnterpriseHooks;

fn main() -> anyhow::Result<()> {
    izwi_gateway::run_from_cli_blocking(EnterpriseHooks::noop())
}
