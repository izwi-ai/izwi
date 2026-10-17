//! Shutdown signals and the ordered shutdown of the HTTP server, the batch
//! worker and the runtime. The signals and the HTTP shutdown plan are
//! izwi-gateway's, which both roles share.

use std::time::{Duration, Instant};
use tracing::{info, warn};

pub(crate) use izwi_gateway::shutdown::*;

use crate::state::AppState;

pub(crate) async fn cleanup_runtime_for_shutdown(state: &AppState) {
    const CLEANUP_TIMEOUT: Duration = Duration::from_secs(20);
    let started = Instant::now();
    if let Err(err) = state.runtime.wait_for_drain(CLEANUP_TIMEOUT).await {
        let snapshot = state.runtime.coordinator_snapshot();
        warn!(
            active_jobs = snapshot.active_jobs,
            active_executions = snapshot.active_executions,
            "Runtime drain failed: {err}; skipping model unload"
        );
        return;
    }
    let remaining = CLEANUP_TIMEOUT.saturating_sub(started.elapsed());
    match tokio::time::timeout(remaining, state.runtime.unload_all_models()).await {
        Ok(Ok(unloaded)) => {
            info!(
                "Runtime shutdown cleanup completed; unloaded {} model(s)",
                unloaded
            );
        }
        Ok(Err(err)) => {
            warn!("Runtime shutdown cleanup failed: {}", err);
        }
        Err(_) => {
            warn!(
                "Runtime shutdown cleanup timed out after {}s; continuing shutdown",
                CLEANUP_TIMEOUT.as_secs()
            );
        }
    }
}

pub(crate) async fn shutdown_worker_then_cleanup<W, C>(
    worker_shutdown: W,
    cleanup: C,
) -> anyhow::Result<()>
where
    W: std::future::Future<Output = anyhow::Result<()>>,
    C: std::future::Future<Output = ()>,
{
    worker_shutdown.await?;
    cleanup.await;
    Ok(())
}
