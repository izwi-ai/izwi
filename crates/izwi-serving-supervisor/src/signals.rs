//! Operator signals the supervisor listens to. These are local
//! administrative surfaces: Unix signal permissions gate access, and no
//! unauthenticated network listener is introduced. Off unix only Ctrl+C
//! exists, so the other requests never fire there.

use tokio::sync::{mpsc, watch};

/// Resolves on the first SIGTERM or Ctrl+C.
pub async fn shutdown_requested() {
    #[cfg(unix)]
    {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("install SIGTERM listener");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = terminate.recv() => {}
        }
    }
    #[cfg(not(unix))]
    let _ = tokio::signal::ctrl_c().await;
}

/// Sends one diagnostic request per SIGUSR1, coalescing repeats instead of
/// buffering diagnostics.
pub fn forward_diagnostic_requests(sender: mpsc::Sender<()>) {
    #[cfg(unix)]
    tokio::spawn(async move {
        let Ok(mut signal) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::user_defined1())
        else {
            return;
        };
        while signal.recv().await.is_some() {
            let _ = sender.try_send(());
        }
    });
    #[cfg(not(unix))]
    drop(sender);
}

/// Sends one request per SIGHUP: clear every quarantined worker.
pub fn forward_quarantine_clear_requests(sender: mpsc::Sender<()>) {
    #[cfg(unix)]
    tokio::spawn(async move {
        let Ok(mut signal) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::hangup())
        else {
            return;
        };
        while signal.recv().await.is_some() {
            let _ = sender.try_send(());
        }
    });
    #[cfg(not(unix))]
    drop(sender);
}

/// Becomes true on SIGUSR2: an in-process rollout abort request. The sender
/// lives as long as the runtime, so the receiver never observes a closed
/// channel.
pub fn rollout_abort_requests() -> watch::Receiver<bool> {
    let (sender, receiver) = watch::channel(false);
    tokio::spawn(async move {
        #[cfg(unix)]
        if let Ok(mut signal) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::user_defined2())
        {
            signal.recv().await;
            let _ = sender.send(true);
        }
        std::future::pending::<()>().await;
        drop(sender);
    });
    receiver
}
