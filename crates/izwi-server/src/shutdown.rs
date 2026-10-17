//! Shutdown signals and the ordered shutdown of the HTTP server, the batch worker and the runtime.

use std::io::Read;
use std::time::{Duration, Instant};
use tokio::signal;
use tokio::sync::oneshot;
use tracing::{info, warn};

use crate::state::AppState;

const DESKTOP_OWNER_PIPE_ENV: &str = "IZWI_DESKTOP_OWNER_PIPE";

/// Resolve on Ctrl+C, SIGTERM or the desktop owner pipe closing.
pub(crate) async fn shutdown_requested() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    let desktop_owner_exit = desktop_owner_exit_signal();

    tokio::select! {
        _ = ctrl_c => {
            info!("Received Ctrl+C, shutting down...");
        },
        _ = terminate => {
            info!("Received SIGTERM, shutting down...");
        },
        _ = desktop_owner_exit => {
            info!("Desktop owner pipe closed, shutting down...");
        },
    }
}

pub(crate) async fn gateway_shutdown_requested() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => info!("Received Ctrl+C, shutting down gateway..."),
        _ = terminate => info!("Received SIGTERM, shutting down gateway..."),
        _ = desktop_owner_exit_signal() => info!("Desktop owner pipe closed, shutting down gateway..."),
    }
}

async fn desktop_owner_exit_signal() {
    if std::env::var_os(DESKTOP_OWNER_PIPE_ENV).as_deref() != Some(std::ffi::OsStr::new("1")) {
        std::future::pending::<()>().await;
        return;
    }

    if let Err(err) = tokio::task::spawn_blocking(|| {
        let stdin = std::io::stdin();
        wait_for_owner_pipe_close(stdin.lock())
    })
    .await
    {
        warn!("Desktop owner-pipe monitor failed: {err}");
    }
}

pub(crate) fn wait_for_owner_pipe_close(mut reader: impl Read) {
    let mut buffer = [0_u8; 1];
    loop {
        match reader.read(&mut buffer) {
            Ok(0) | Err(_) => return,
            Ok(_) => {}
        }
    }
}

/// How a role spends its shutdown budget: a pre-stop window in which
/// readiness is unready but requests are served, then a drain of in-flight
/// requests. Streams still open near the end of the drain are ended with a
/// terminal event, so no client is cut off without one.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ShutdownPlan {
    pub(crate) prestop: Duration,
    pub(crate) budget: Duration,
}

impl ShutdownPlan {
    pub(crate) fn drain(&self) -> Duration {
        self.budget.saturating_sub(self.prestop)
    }

    /// When, into the drain, open streams are told to end: late enough for
    /// most to finish on their own, early enough to deliver the event.
    pub(crate) fn stream_notice(&self) -> Duration {
        let drain = self.drain();
        drain.saturating_sub((drain / 4).min(Duration::from_secs(2)))
    }
}

/// What a role does at each step of its [`ShutdownPlan`].
pub(crate) trait ShutdownPhases: Send + 'static {
    /// Readiness turns unready; requests are still admitted.
    fn begin_prestop(&self);
    /// Admission closes; admitted requests run on.
    fn begin_drain(&self);
    /// Live streams end with a terminal "shutting down" event.
    fn end_streams(&self);
}

/// Serve until `shutdown_requested`, then follow `plan`: the pre-stop
/// window, the drain (the listener stops accepting), the stream notice, and
/// the end of the budget. `serve` runs the HTTP server with the graceful
/// shutdown signal it is given. `None` means the budget ran out with
/// connections still open; they are dropped.
pub(crate) async fn serve_with_shutdown_plan<S, F, P>(
    serve: impl FnOnce(futures::future::BoxFuture<'static, ()>) -> F,
    phases: P,
    shutdown_requested: S,
    plan: ShutdownPlan,
) -> Option<F::Output>
where
    S: std::future::Future<Output = ()> + Send + 'static,
    F: std::future::Future,
    P: ShutdownPhases + Clone,
{
    let (drain_started_tx, drain_started_rx) = oneshot::channel::<()>();
    let signal_phases = phases.clone();
    let server = serve(Box::pin(async move {
        shutdown_requested.await;
        signal_phases.begin_prestop();
        info!(
            prestop_secs = plan.prestop.as_secs_f64(),
            budget_secs = plan.budget.as_secs_f64(),
            "Shutdown pre-stop: readiness is unready, requests are still served"
        );
        tokio::time::sleep(plan.prestop).await;
        signal_phases.begin_drain();
        let _ = drain_started_tx.send(());
    }));
    tokio::pin!(server);
    tokio::select! {
        result = &mut server => Some(result),
        started = drain_started_rx => {
            if started.is_err() {
                return Some(server.await);
            }
            match tokio::time::timeout(plan.stream_notice(), &mut server).await {
                Ok(result) => Some(result),
                Err(_) => {
                    phases.end_streams();
                    let rest = plan.drain().saturating_sub(plan.stream_notice());
                    tokio::time::timeout(rest, &mut server).await.ok()
                }
            }
        }
    }
}

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
