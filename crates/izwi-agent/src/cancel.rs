use std::sync::Arc;
use tokio::sync::watch;

/// Stops a turn: the model call in flight, the tools it awaits and the
/// rounds that would follow. Clones share one signal.
#[derive(Debug, Clone)]
pub struct TurnCancel {
    signal: Arc<watch::Sender<bool>>,
}

impl Default for TurnCancel {
    fn default() -> Self {
        Self::new()
    }
}

impl TurnCancel {
    pub fn new() -> Self {
        Self {
            signal: Arc::new(watch::Sender::new(false)),
        }
    }

    pub fn cancel(&self) {
        self.signal.send_replace(true);
    }

    pub fn is_cancelled(&self) -> bool {
        *self.signal.borrow()
    }

    /// Resolves once the turn is cancelled; never otherwise.
    pub async fn cancelled(&self) {
        let mut signal = self.signal.subscribe();
        if signal.wait_for(|cancelled| *cancelled).await.is_err() {
            std::future::pending::<()>().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn clones_share_the_signal() {
        let cancel = TurnCancel::new();
        let waiter = {
            let cancel = cancel.clone();
            tokio::spawn(async move { cancel.cancelled().await })
        };
        assert!(!cancel.is_cancelled());
        cancel.cancel();
        waiter.await.unwrap();
        assert!(cancel.is_cancelled());
        // A turn cancelled before anyone waits still resolves.
        cancel.cancelled().await;
    }
}
