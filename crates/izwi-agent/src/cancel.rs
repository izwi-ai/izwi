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

/// Holds a turn back from anything that cannot be undone until its caller
/// confirms it. A reply started early, on speech that may not be finished,
/// writes its text behind a closed gate: it runs no tool, hands over to no
/// agent and records no task until the gate opens. Clones share one gate.
#[derive(Debug, Clone)]
pub struct TurnGate {
    signal: Arc<watch::Sender<bool>>,
}

impl Default for TurnGate {
    /// A gate that is open: the turn is confirmed from the start.
    fn default() -> Self {
        Self::opened()
    }
}

impl TurnGate {
    pub fn opened() -> Self {
        Self {
            signal: Arc::new(watch::Sender::new(true)),
        }
    }

    pub fn closed() -> Self {
        Self {
            signal: Arc::new(watch::Sender::new(false)),
        }
    }

    pub fn open(&self) {
        self.signal.send_replace(true);
    }

    pub fn is_open(&self) -> bool {
        *self.signal.borrow()
    }

    /// Resolves once the gate is open.
    pub async fn wait_open(&self) {
        let mut signal = self.signal.subscribe();
        if signal.wait_for(|open| *open).await.is_err() {
            std::future::pending::<()>().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_gate_holds_until_it_is_opened() {
        TurnGate::default().wait_open().await;
        let gate = TurnGate::closed();
        let waiter = {
            let gate = gate.clone();
            tokio::spawn(async move { gate.wait_open().await })
        };
        tokio::task::yield_now().await;
        assert!(!waiter.is_finished() && !gate.is_open());
        gate.open();
        waiter.await.unwrap();
        assert!(gate.is_open());
    }

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
