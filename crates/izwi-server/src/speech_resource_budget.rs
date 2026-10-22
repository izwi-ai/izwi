//! Process-wide byte reservations for streaming persistence. These limits are
//! separate from device inference admission and apply across all live requests.
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, OnceLock,
};

#[derive(Debug)]
pub(crate) struct ByteBudget {
    limit: usize,
    used: AtomicUsize,
    available: tokio::sync::Notify,
}

impl ByteBudget {
    pub(crate) fn new(limit: usize) -> Arc<Self> {
        Arc::new(Self {
            limit,
            used: AtomicUsize::new(0),
            available: tokio::sync::Notify::new(),
        })
    }

    pub(crate) fn reserve(self: &Arc<Self>, bytes: usize) -> anyhow::Result<ByteReservation> {
        self.used
            .try_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(bytes).filter(|next| *next <= self.limit)
            })
            .map_err(|_| anyhow::anyhow!("Aggregate speech persistence byte capacity exhausted"))?;
        Ok(ByteReservation {
            budget: self.clone(),
            bytes,
        })
    }
    /// Wait without holding partial capacity. Oversized requests cannot ever
    /// succeed and fail immediately; cancelling this future removes its waiter.
    pub(crate) async fn reserve_wait(
        self: &Arc<Self>,
        bytes: usize,
    ) -> anyhow::Result<ByteReservation> {
        anyhow::ensure!(
            bytes <= self.limit,
            "Speech artifact exceeds aggregate spool capacity ({bytes} > {})",
            self.limit
        );
        loop {
            let notified = self.available.notified();
            tokio::pin!(notified);
            // Register before checking capacity so a concurrent release cannot
            // be lost between the failed reservation and suspension.
            notified.as_mut().enable();
            if let Ok(reservation) = self.reserve(bytes) {
                return Ok(reservation);
            }
            notified.await;
        }
    }
}

pub(crate) struct ByteReservation {
    budget: Arc<ByteBudget>,
    bytes: usize,
}

impl ByteReservation {
    pub(crate) fn grow(&mut self, bytes: usize) -> anyhow::Result<()> {
        let mut added = self.budget.reserve(bytes)?;
        self.bytes += added.bytes;
        added.bytes = 0;
        Ok(())
    }
}

impl Drop for ByteReservation {
    fn drop(&mut self) {
        self.budget.used.fetch_sub(self.bytes, Ordering::AcqRel);
        if self.bytes > 0 {
            self.budget.available.notify_waiters();
        }
    }
}

pub(crate) fn spool_budget() -> &'static Arc<ByteBudget> {
    static BUDGET: OnceLock<Arc<ByteBudget>> = OnceLock::new();
    BUDGET.get_or_init(|| ByteBudget::new(crate::env_config::tuning().tts_total_spool_bytes))
}

pub(crate) fn upload_budget() -> &'static Arc<ByteBudget> {
    static BUDGET: OnceLock<Arc<ByteBudget>> = OnceLock::new();
    BUDGET.get_or_init(|| ByteBudget::new(crate::env_config::tuning().tts_total_upload_bytes))
}

pub(crate) fn event_budget() -> &'static Arc<ByteBudget> {
    static BUDGET: OnceLock<Arc<ByteBudget>> = OnceLock::new();
    BUDGET
        .get_or_init(|| ByteBudget::new(crate::env_config::tuning().audio_stream_total_event_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn waiting_reservations_wake_after_release_and_reject_oversized_requests() {
        let budget = ByteBudget::new(10);
        let first = budget.reserve(10).unwrap();
        assert!(budget.reserve_wait(11).await.is_err());
        let waiting_budget = budget.clone();
        let waiting = tokio::spawn(async move { waiting_budget.reserve_wait(8).await.unwrap() });
        tokio::task::yield_now().await;
        assert!(!waiting.is_finished());
        assert_eq!(budget.used.load(Ordering::Acquire), 10);
        drop(first);
        let second = tokio::time::timeout(std::time::Duration::from_secs(1), waiting)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(budget.used.load(Ordering::Acquire), 8);
        drop(second);
        assert_eq!(budget.used.load(Ordering::Acquire), 0);
    }

    #[tokio::test]
    async fn cancelling_capacity_wait_holds_no_partial_credits() {
        let budget = ByteBudget::new(10);
        let owner = budget.reserve(8).unwrap();
        let waiting_budget = budget.clone();
        let waiting = tokio::spawn(async move { waiting_budget.reserve_wait(5).await });
        tokio::task::yield_now().await;
        waiting.abort();
        assert!(matches!(waiting.await, Err(error) if error.is_cancelled()));
        assert_eq!(budget.used.load(Ordering::Acquire), 8);
        drop(owner);
        assert!(budget.reserve_wait(10).await.is_ok());
    }

    #[test]
    fn aggregate_reservations_reject_without_changing_ownership_and_return_on_drop() {
        let budget = ByteBudget::new(10);
        let mut first = budget.reserve(4).unwrap();
        let second = budget.reserve(5).unwrap();
        assert!(first.grow(2).is_err());
        assert_eq!(budget.used.load(Ordering::Acquire), 9);
        drop(second);
        first.grow(6).unwrap();
        assert!(budget.reserve(1).is_err());
        drop(first);
        assert_eq!(budget.used.load(Ordering::Acquire), 0);
        assert!(budget.reserve(usize::MAX).is_err());
    }

    #[tokio::test]
    async fn abort_releases_live_reservation() {
        let budget = ByteBudget::new(16);
        let task_budget = budget.clone();
        let (tx, rx) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(async move {
            let _reservation = task_budget.reserve(16).unwrap();
            tx.send(()).unwrap();
            std::future::pending::<()>().await;
        });
        rx.await.unwrap();
        assert!(budget.reserve(1).is_err());
        task.abort();
        let _ = task.await;
        assert_eq!(budget.used.load(Ordering::Acquire), 0);
    }
}
