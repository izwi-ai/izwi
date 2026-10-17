use super::types::{
    ClaimedStage, JobStage, MediaAsset, QueueClass, RuntimeArtifact, RuntimeArtifactKind,
    RuntimeArtifactRole, RuntimeCancellationState, RuntimeJob, RuntimeJobKind, RuntimeJobStatus,
    RuntimeStageStatus, RuntimeWorkerHeartbeat, RuntimeWorkerHeartbeatDetails,
    RuntimeWorkerRegistration, StageLease, StageResourceHints, TextAsset, WorkerResourceCapacity,
    STAGE_RESOURCE_HINTS_VERSION, WORKER_HEARTBEAT_DETAILS_VERSION, WORKER_REGISTRATION_VERSION,
};
#[cfg(test)]
use super::types::{DeviceClass, ResourceTarget, RuntimeBackendClass};
use crate::{
    db::{raw, write_transaction_options, StoreDatabase},
    error::StoreError,
    ids::new_uuid,
    speech_history_store::{
        sanitize_audio_mime_type, sanitize_optional_text, NewSpeechHistoryRecord,
        SpeechHistoryProcessingStatus, SpeechHistoryRecord, SpeechRouteKind,
    },
};
use anyhow::{anyhow, bail, Context};
use izwi_hooks::{HookMetadata, MediaNamespace, MediaWriteRequest};
use sea_orm::{
    ConnectionTrait, DatabaseConnection, DbBackend, QueryResult, TransactionTrait, Value,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    io::{self, Write},
    time::{SystemTime, UNIX_EPOCH},
};

use std::sync::Arc;
#[cfg(test)]
use std::sync::{atomic::AtomicI64, atomic::Ordering};

mod acceptance;
mod artifacts;
mod idempotency;
mod jobs;
mod media;
mod provider_writes;
mod queue;
mod quota;
mod reconcile;
mod retention;
mod speech;
mod sql;
mod transitions;
mod workers;
#[cfg(test)]
mod tests;

pub use acceptance::*;
pub use artifacts::*;
pub use idempotency::*;
pub use jobs::*;
pub use media::*;
pub(crate) use provider_writes::*;
pub use queue::*;
pub(crate) use quota::*;
pub use reconcile::*;
pub use retention::*;
pub(crate) use speech::*;
pub use sql::*;
pub(crate) use transitions::*;
pub use workers::*;

#[derive(Debug, Clone)]
pub struct BatchRuntimeStore {
    db: StoreDatabase,
    /// Woken when a stage may have become claimable, so idle worker slots
    /// in this process claim at once instead of waiting out their poll.
    stage_ready: Arc<tokio::sync::Notify>,
    #[cfg(test)]
    test_clock: Option<Arc<AtomicI64>>,
    #[cfg(test)]
    test_tts_admission_limits: Arc<std::sync::RwLock<Option<(usize, usize)>>>,
    #[cfg(test)]
    test_artifact_cleanup_limits: Option<QuotaLimits>,
    #[cfg(test)]
    test_provider_write_limits: Option<QuotaLimits>,
    #[cfg(test)]
    test_pcm_replay_entry_limit: Option<u64>,
    #[cfg(test)]
    test_durable_acceptance_failpoint: Option<DurableAcceptanceFailpoint>,
    #[cfg(test)]
    test_faults: Option<Arc<StoreFaults>>,
}

/// Store calls a test can make fail transiently, as a dropped connection or
/// pool-acquire timeout would.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum StoreFaultPoint {
    WorkerHeartbeat,
    RenewStageLease,
    StageLeaseState,
    CompleteStage,
}

#[cfg(test)]
#[derive(Debug, Default)]
pub(crate) struct StoreFaults {
    /// Per point: calls still let through, then calls still failed.
    armed: std::sync::Mutex<std::collections::HashMap<StoreFaultPoint, (u32, u32)>>,
}

#[cfg(test)]
impl StoreFaults {
    /// Let the next `skip` calls through, then fail the `fail` calls after.
    pub(crate) fn arm(&self, point: StoreFaultPoint, skip: u32, fail: u32) {
        self.armed
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .insert(point, (skip, fail));
    }

    pub(crate) fn pending_failures(&self, point: StoreFaultPoint) -> u32 {
        self.armed
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .get(&point)
            .map_or(0, |(_, fail)| *fail)
    }

    fn trip(&self, point: StoreFaultPoint) -> bool {
        let mut armed = self
            .armed
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let Some((skip, fail)) = armed.get_mut(&point) else {
            return false;
        };
        if *skip > 0 {
            *skip -= 1;
            return false;
        }
        if *fail > 0 {
            *fail -= 1;
            return true;
        }
        false
    }
}

impl BatchRuntimeStore {
    pub fn initialize_with_database(db: StoreDatabase) -> Self {
        Self {
            db,
            stage_ready: Arc::default(),
            #[cfg(test)]
            test_clock: None,
            #[cfg(test)]
            test_tts_admission_limits: Arc::default(),
            #[cfg(test)]
            test_artifact_cleanup_limits: None,
            #[cfg(test)]
            test_provider_write_limits: None,
            #[cfg(test)]
            test_pcm_replay_entry_limit: None,
            #[cfg(test)]
            test_durable_acceptance_failpoint: None,
            #[cfg(test)]
            test_faults: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn set_test_clock(&mut self, clock: Arc<AtomicI64>) {
        self.test_clock = Some(clock);
    }

    /// TTS admission limits in place of the configured ones, through a
    /// shared store (the router tests hold it inside the app state).
    #[cfg(test)]
    pub(crate) fn set_tts_admission_limits_for_test(&self, limits: Option<(usize, usize)>) {
        *self
            .test_tts_admission_limits
            .write()
            .unwrap_or_else(|poison| poison.into_inner()) = limits;
    }

    #[cfg(test)]
    pub(crate) fn set_artifact_cleanup_capacity_for_test(&mut self, capacity: u64) {
        self.set_artifact_cleanup_limits_for_test(QuotaLimits {
            global: capacity,
            per_tenant: capacity,
        });
    }

    #[cfg(test)]
    pub(crate) fn set_artifact_cleanup_limits_for_test(&mut self, limits: QuotaLimits) {
        self.test_artifact_cleanup_limits = Some(limits);
    }

    #[cfg(test)]
    pub(crate) fn set_provider_write_capacity_for_test(&mut self, capacity: u64) {
        self.set_provider_write_limits_for_test(QuotaLimits {
            global: capacity,
            per_tenant: capacity,
        });
    }

    #[cfg(test)]
    pub(crate) fn set_provider_write_limits_for_test(&mut self, limits: QuotaLimits) {
        self.test_provider_write_limits = Some(limits);
    }

    #[cfg(test)]
    pub(crate) fn set_pcm_replay_entry_limit_for_test(&mut self, limit: u64) {
        self.test_pcm_replay_entry_limit = Some(limit);
    }

    #[cfg(test)]
    fn set_durable_acceptance_failpoint(&mut self, failpoint: Option<DurableAcceptanceFailpoint>) {
        self.test_durable_acceptance_failpoint = failpoint;
    }

    #[cfg(test)]
    pub(crate) fn set_faults_for_test(&mut self, faults: Arc<StoreFaults>) {
        self.test_faults = Some(faults);
    }

    fn inject_fault(&self, point: StoreFaultPoint) -> anyhow::Result<()> {
        #[cfg(test)]
        if self
            .test_faults
            .as_ref()
            .is_some_and(|faults| faults.trip(point))
        {
            bail!("Injected transient store failure at {point:?}");
        }
        #[cfg(not(test))]
        let _ = point;
        Ok(())
    }

    /// The store's time: the database server's clock (see
    /// `StoreDatabase::now_millis`), or a test's clock.
    pub(crate) fn now_millis(&self) -> i64 {
        #[cfg(test)]
        if let Some(clock) = &self.test_clock {
            return clock.load(Ordering::SeqCst);
        }
        self.db.now_millis()
    }

    fn pcm_replay_entry_limit(&self) -> u64 {
        #[cfg(test)]
        if let Some(limit) = self.test_pcm_replay_entry_limit {
            return limit;
        }
        super::speech_progress::MAX_PCM_REPLAY_ENTRIES
    }

    fn inject_durable_acceptance_failure(
        &self,
        failpoint: DurableAcceptanceFailpoint,
    ) -> anyhow::Result<()> {
        #[cfg(test)]
        if self.test_durable_acceptance_failpoint == Some(failpoint) {
            bail!("Injected durable acceptance failure after {failpoint:?}");
        }
        #[cfg(not(test))]
        let _ = failpoint;
        Ok(())
    }

    pub async fn connection(&self) -> anyhow::Result<&DatabaseConnection> {
        self.db.connection().await
    }

    /// Resolves after the next [`Self::wake_claimers`]; register it (with
    /// `Notified::enable`) before looking for work so no wake-up is lost.
    pub(crate) fn stage_ready(&self) -> tokio::sync::futures::Notified<'_> {
        self.stage_ready.notified()
    }

    /// A stage may have become claimable: wake this process's idle claimers.
    fn wake_claimers(&self) {
        self.stage_ready.notify_waiters();
    }
}
