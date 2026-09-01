//! Exact managed-KV counters shared by coordinator and backend runtimes.

use std::sync::atomic::{AtomicU64, Ordering};

use serde::Serialize;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct ManagedKvTelemetrySnapshot {
    pub transaction_commits: u64,
    pub transaction_aborts: u64,
    pub transaction_conflicts: u64,
    pub pages_zeroed: u64,
    pub pages_copied: u64,
    pub slots_written: u64,
    pub prefix_hits: u64,
    pub prefix_misses: u64,
    pub prefix_evictions: u64,
    pub prefix_copy_on_write_pages: u64,
    pub prefix_rejections: u64,
    /// Live gauge projected by the manager's runtime snapshot.
    pub prefix_retained_pages: u64,
    pub reused_tokens: u64,
    pub avoided_prefill_tokens: u64,
    pub decode_dispatches: u64,
    pub host_synchronizations: u64,
    pub backing_allocations: u64,
}

#[derive(Debug, Default)]
pub struct ManagedKvTelemetry {
    transaction_commits: AtomicU64,
    transaction_aborts: AtomicU64,
    transaction_conflicts: AtomicU64,
    pages_zeroed: AtomicU64,
    pages_copied: AtomicU64,
    slots_written: AtomicU64,
    prefix_hits: AtomicU64,
    prefix_misses: AtomicU64,
    prefix_evictions: AtomicU64,
    prefix_copy_on_write_pages: AtomicU64,
    prefix_rejections: AtomicU64,
    reused_tokens: AtomicU64,
    avoided_prefill_tokens: AtomicU64,
    decode_dispatches: AtomicU64,
    host_synchronizations: AtomicU64,
    backing_allocations: AtomicU64,
}

impl ManagedKvTelemetry {
    pub fn record_commit(&self) {
        self.transaction_commits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_abort(&self) {
        self.transaction_aborts.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_conflict(&self) {
        self.transaction_conflicts.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_zero(&self, pages: usize) {
        add_usize(&self.pages_zeroed, pages);
    }

    pub fn record_copy(&self, pages: usize) {
        add_usize(&self.pages_copied, pages);
    }

    pub fn record_write(&self, slots: usize) {
        add_usize(&self.slots_written, slots);
    }

    pub fn record_prefix_hit(&self, reused_tokens: u64) {
        self.prefix_hits.fetch_add(1, Ordering::Relaxed);
        self.reused_tokens
            .fetch_add(reused_tokens, Ordering::Relaxed);
        self.avoided_prefill_tokens
            .fetch_add(reused_tokens, Ordering::Relaxed);
    }

    pub fn record_prefix_miss(&self) {
        self.prefix_misses.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_prefix_eviction(&self) {
        self.prefix_evictions.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_prefix_copy_on_write(&self, pages: usize) {
        add_usize(&self.prefix_copy_on_write_pages, pages);
    }

    pub fn record_prefix_rejection(&self) {
        self.prefix_rejections.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_decode_dispatch(&self) {
        self.decode_dispatches.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_host_synchronization(&self) {
        self.host_synchronizations.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_backing_allocation(&self) {
        self.backing_allocations.fetch_add(1, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> ManagedKvTelemetrySnapshot {
        ManagedKvTelemetrySnapshot {
            transaction_commits: load(&self.transaction_commits),
            transaction_aborts: load(&self.transaction_aborts),
            transaction_conflicts: load(&self.transaction_conflicts),
            pages_zeroed: load(&self.pages_zeroed),
            pages_copied: load(&self.pages_copied),
            slots_written: load(&self.slots_written),
            prefix_hits: load(&self.prefix_hits),
            prefix_misses: load(&self.prefix_misses),
            prefix_evictions: load(&self.prefix_evictions),
            prefix_copy_on_write_pages: load(&self.prefix_copy_on_write_pages),
            prefix_rejections: load(&self.prefix_rejections),
            prefix_retained_pages: 0,
            reused_tokens: load(&self.reused_tokens),
            avoided_prefill_tokens: load(&self.avoided_prefill_tokens),
            decode_dispatches: load(&self.decode_dispatches),
            host_synchronizations: load(&self.host_synchronizations),
            backing_allocations: load(&self.backing_allocations),
        }
    }
}

fn load(value: &AtomicU64) -> u64 {
    value.load(Ordering::Relaxed)
}

fn add_usize(counter: &AtomicU64, value: usize) {
    counter.fetch_add(value.min(u64::MAX as usize) as u64, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_reports_physical_hot_path_work() {
        let metrics = ManagedKvTelemetry::default();
        metrics.record_zero(2);
        metrics.record_copy(1);
        metrics.record_write(17);
        metrics.record_decode_dispatch();
        metrics.record_prefix_hit(16);
        metrics.record_prefix_copy_on_write(2);
        metrics.record_prefix_rejection();
        metrics.record_commit();
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.pages_zeroed, 2);
        assert_eq!(snapshot.pages_copied, 1);
        assert_eq!(snapshot.slots_written, 17);
        assert_eq!(snapshot.decode_dispatches, 1);
        assert_eq!(snapshot.reused_tokens, 16);
        assert_eq!(snapshot.avoided_prefill_tokens, 16);
        assert_eq!(snapshot.prefix_copy_on_write_pages, 2);
        assert_eq!(snapshot.prefix_rejections, 1);
        assert_eq!(snapshot.prefix_retained_pages, 0);
        assert_eq!(snapshot.transaction_commits, 1);
    }
}
