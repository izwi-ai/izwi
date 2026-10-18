//! A server's lifecycle for its readiness and liveness probes: starting,
//! ready, the pre-stop window, draining. Both roles keep one.

use serde::Serialize;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug, Serialize)]
pub struct LifecycleSnapshot {
    pub phase: String,
    pub ready: bool,
    pub draining: bool,
    pub started_at: u64,
    pub updated_at: u64,
    /// Every startup warning, reported for operators.
    pub startup_warnings: Vec<String>,
    /// The subset of startup warnings that keeps readiness failed: a
    /// configured model that did not preload or warm up.
    pub preload_failures: Vec<String>,
}

#[derive(Debug)]
struct LifecycleInner {
    phase: String,
    ready: bool,
    draining: bool,
    /// Shutdown's pre-stop window: unready, but still serving.
    stopping: bool,
    started_at: u64,
    updated_at: u64,
    startup_warnings: Vec<String>,
    preload_failures: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct ServerLifecycle {
    inner: Arc<std::sync::RwLock<LifecycleInner>>,
}

impl ServerLifecycle {
    pub fn new() -> Self {
        let now = now_unix_secs();
        Self {
            inner: Arc::new(std::sync::RwLock::new(LifecycleInner {
                phase: "initializing".to_string(),
                ready: false,
                draining: false,
                stopping: false,
                started_at: now,
                updated_at: now,
                startup_warnings: Vec::new(),
                preload_failures: Vec::new(),
            })),
        }
    }

    /// Shutdown is terminal: a startup step that finishes after shutdown
    /// began must not make the server ready again.
    pub fn mark_ready(&self) {
        self.update(|inner| {
            if inner.draining || inner.stopping {
                return;
            }
            inner.phase = "ready".to_string();
            inner.ready = true;
            inner.draining = false;
        });
    }

    /// Shutdown's pre-stop window: readiness turns unready so load
    /// balancers stop routing here, while requests are still served.
    pub fn mark_stopping(&self) {
        self.update(|inner| {
            if inner.draining {
                return;
            }
            inner.phase = "stopping".to_string();
            inner.ready = false;
            inner.stopping = true;
        });
    }

    /// Whether shutdown has begun (its pre-stop window or its drain).
    pub fn shutdown_started(&self) -> bool {
        let guard = self
            .inner
            .read()
            .unwrap_or_else(|poison| poison.into_inner());
        guard.draining || guard.stopping
    }

    pub fn mark_draining(&self) {
        self.update(|inner| {
            inner.phase = "draining".to_string();
            inner.ready = false;
            inner.draining = true;
        });
    }

    /// Record non-fatal startup warnings (durable-state maintenance). They
    /// are reported by readiness but do not fail it.
    pub fn record_startup_warnings(&self, warnings: impl IntoIterator<Item = String>) {
        self.update(|inner| {
            inner.startup_warnings.extend(warnings);
        });
    }

    /// Record configured models that failed to preload or warm up. They are
    /// reported with the startup warnings and keep readiness failed.
    pub fn record_preload_failures(&self, failures: impl IntoIterator<Item = String>) {
        self.update(|inner| {
            let failures = failures.into_iter().collect::<Vec<_>>();
            inner.startup_warnings.extend(failures.iter().cloned());
            inner.preload_failures.extend(failures);
        });
    }

    pub fn snapshot(&self) -> LifecycleSnapshot {
        let guard = self
            .inner
            .read()
            .unwrap_or_else(|poison| poison.into_inner());
        LifecycleSnapshot {
            phase: guard.phase.clone(),
            ready: guard.ready,
            draining: guard.draining,
            started_at: guard.started_at,
            updated_at: guard.updated_at,
            startup_warnings: guard.startup_warnings.clone(),
            preload_failures: guard.preload_failures.clone(),
        }
    }

    fn update(&self, f: impl FnOnce(&mut LifecycleInner)) {
        let mut guard = self
            .inner
            .write()
            .unwrap_or_else(|poison| poison.into_inner());
        f(&mut guard);
        guard.updated_at = now_unix_secs();
    }
}

impl Default for ServerLifecycle {
    fn default() -> Self {
        Self::new()
    }
}

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_lifecycle_transitions_to_ready_and_draining() {
        let lifecycle = ServerLifecycle::new();
        let initial = lifecycle.snapshot();
        assert!(!initial.ready);
        assert_eq!(initial.phase, "initializing");

        lifecycle.record_startup_warnings(vec!["preload failed".to_string()]);
        lifecycle.mark_ready();
        let ready = lifecycle.snapshot();
        assert!(ready.ready);
        assert_eq!(ready.phase, "ready");
        assert_eq!(ready.startup_warnings, vec!["preload failed"]);

        lifecycle.mark_draining();
        let draining = lifecycle.snapshot();
        assert!(!draining.ready);
        assert!(draining.draining);
        assert_eq!(draining.phase, "draining");
        assert!(draining.updated_at >= ready.updated_at);
        assert!(draining.updated_at >= now_unix_secs().saturating_sub(1));

        // A startup step finishing after shutdown began cannot undo the drain.
        lifecycle.mark_ready();
        let still_draining = lifecycle.snapshot();
        assert!(!still_draining.ready);
        assert!(still_draining.draining);
    }
}
