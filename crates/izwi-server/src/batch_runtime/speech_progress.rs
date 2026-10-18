//! Versioned, attempt-fenced speech checkpoints and immutable PCM replay entries.
//!
//! PCM objects must be persisted before `publish_pcm` and only sent to listeners
//! after it succeeds. The checkpoint that records a publication commits in the
//! same transaction as its journal entry, so a crash in either half of
//! publication can never silently regenerate an audible segment.
use super::{
    types::{RuntimeArtifact, RuntimeArtifactKind, RuntimeArtifactRole},
    worker::StageExecutionContext,
};
use crate::artifact_store::{
    ArtifactRetention, ArtifactStore, ArtifactTenant, ArtifactWrite, AttemptArtifactWrite,
};
use anyhow::{bail, ensure, Context};
use izwi_core::AudioChunk;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::time::Instant;

pub const SPEECH_CHECKPOINT_VERSION: u32 = 1;
pub const MAX_PCM_REPLAY_BATCH_BYTES: u64 = 1024 * 1024;
// The default Fish 4,800-sample chunks need 66,150 entries for the documented
// 120-minute qualification target. Keep a fixed ceiling above that target.
pub const MAX_PCM_REPLAY_ENTRIES: u64 = 131_072;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpeechCheckpoint {
    pub version: u32,
    /// Immutable planner/model/tokenizer/reference/settings snapshot.
    pub identity: serde_json::Value,
    pub total_segments: u32,
    pub completed_segments: u32,
    pub completed_text_bytes: u64,
    #[serde(default)]
    pub completed_tokens: usize,
    #[serde(default)]
    pub completed_execution_ms: f32,
    #[serde(default)]
    pub completed_duration_secs: f32,
    pub committed_samples: u64,
    pub sample_rate: Option<u32>,
    pub next_sequence: u64,
    pub journal_bytes: u64,
    pub max_journal_bytes: u64,
    pub active_segment: Option<u32>,
    /// Set before the first journal publication; interrupted publication is incomplete.
    pub publication_started: bool,
    /// Set only by the atomic final-audio settlement transaction.
    #[serde(default)]
    pub final_audio_published: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SpeechPcmBatch {
    pub version: u32,
    pub sequence: u64,
    pub segment: u32,
    pub sample_offset: u64,
    pub sample_count: u64,
    pub sample_rate: u32,
}

pub fn pcm_publication_key(sequence: u64) -> String {
    format!("speech-pcm/{sequence:020}")
}

/// Merges consecutive PCM chunks of one segment into fewer journal entries.
/// Each entry costs a provider write (file and directory fsync) and metadata
/// transactions, so chunks are published at most every `min_interval`, or
/// once `max_chunks` are waiting, and never above one MiB. A chunk arriving
/// after a quiet interval publishes at once, so time to first audio is
/// unchanged; listeners see at most `min_interval` of extra delay.
#[derive(Debug)]
pub struct PcmPublicationBuffer {
    min_interval: Duration,
    max_chunks: usize,
    pending: Option<AudioChunk>,
    pending_chunks: usize,
    last_publication: Option<Instant>,
}

impl PcmPublicationBuffer {
    pub fn new(min_interval: Duration, max_chunks: usize) -> Self {
        Self {
            min_interval,
            max_chunks: max_chunks.max(1),
            pending: None,
            pending_chunks: 0,
            last_publication: None,
        }
    }

    /// `IZWI_TTS_PCM_PUBLISH_INTERVAL_MS` (250; 0 publishes every chunk) and
    /// `IZWI_TTS_PCM_PUBLISH_MAX_CHUNKS` (8).
    pub fn configured() -> Self {
        let tuning = crate::env_config::tuning();
        Self::new(
            tuning.tts_pcm_publish_interval,
            tuning.tts_pcm_publish_max_chunks,
        )
    }

    /// Buffer `chunk`. Returns the waiting batch when `chunk` would push it
    /// past one MiB; that batch must be published before anything after it.
    pub fn push(&mut self, chunk: AudioChunk) -> Option<AudioChunk> {
        let max_samples = (MAX_PCM_REPLAY_BATCH_BYTES / 4) as usize;
        let overflow = self
            .pending
            .as_ref()
            .is_some_and(|pending| pending.samples.len() + chunk.samples.len() > max_samples);
        let flushed = if overflow { self.take_pending() } else { None };
        match self.pending.as_mut() {
            Some(pending) => pending.samples.extend_from_slice(&chunk.samples),
            None => self.pending = Some(chunk),
        }
        self.pending_chunks += 1;
        flushed
    }

    /// The waiting batch, if it is due at `now`.
    pub fn take_due(&mut self, now: Instant) -> Option<AudioChunk> {
        let due = self.pending_chunks >= self.max_chunks
            || self.deadline().is_none_or(|deadline| deadline <= now);
        if !due {
            return None;
        }
        self.take(now)
    }

    /// The waiting batch, due or not: at a segment end or its deadline.
    pub fn take(&mut self, now: Instant) -> Option<AudioChunk> {
        let batch = self.take_pending()?;
        self.last_publication = Some(now);
        Some(batch)
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_none()
    }

    /// When the waiting batch becomes due.
    pub fn deadline(&self) -> Option<Instant> {
        self.pending.as_ref()?;
        self.last_publication
            .map(|published| published + self.min_interval)
    }

    fn take_pending(&mut self) -> Option<AudioChunk> {
        self.pending_chunks = 0;
        self.pending.take()
    }
}

impl SpeechCheckpoint {
    pub fn new(
        identity: serde_json::Value,
        total_segments: u32,
        max_journal_bytes: u64,
    ) -> anyhow::Result<Self> {
        ensure!(total_segments > 0, "Speech plan must contain a segment");
        ensure!(
            max_journal_bytes > 0,
            "Speech replay storage quota must be positive"
        );
        Ok(Self {
            version: SPEECH_CHECKPOINT_VERSION,
            identity,
            total_segments,
            completed_segments: 0,
            completed_text_bytes: 0,
            completed_tokens: 0,
            completed_execution_ms: 0.0,
            completed_duration_secs: 0.0,
            committed_samples: 0,
            sample_rate: None,
            next_sequence: 0,
            journal_bytes: 0,
            max_journal_bytes,
            active_segment: None,
            publication_started: false,
            final_audio_published: false,
        })
    }

    pub fn recover(
        progress: &serde_json::Value,
        identity: &serde_json::Value,
    ) -> anyhow::Result<Self> {
        let mut checkpoint: Self =
            serde_json::from_value(progress.clone()).context("Invalid speech checkpoint")?;
        ensure!(
            checkpoint.version == SPEECH_CHECKPOINT_VERSION,
            "Unsupported speech checkpoint version"
        );
        ensure!(
            &checkpoint.identity == identity,
            "Speech model/reference/plan snapshot changed during recovery"
        );
        ensure!(
            checkpoint.completed_segments <= checkpoint.total_segments,
            "Invalid speech segment progress"
        );
        ensure!(
            checkpoint.journal_bytes <= checkpoint.max_journal_bytes,
            "Speech replay storage quota exceeded"
        );
        ensure!(
            checkpoint.next_sequence <= MAX_PCM_REPLAY_ENTRIES,
            "Speech replay entry quota exceeded"
        );
        if checkpoint.active_segment.is_some() && checkpoint.publication_started {
            bail!(
                "speech_partial_segment_interrupted: published audio cannot be regenerated safely"
            );
        }
        checkpoint.active_segment = None;
        Ok(checkpoint)
    }

    pub async fn save(&self, attempt: &StageExecutionContext) -> anyhow::Result<()> {
        attempt.record_progress(serde_json::to_value(self)?).await
    }

    pub async fn begin_segment(&mut self, attempt: &StageExecutionContext) -> anyhow::Result<u32> {
        ensure!(
            self.active_segment.is_none(),
            "Speech segment already active"
        );
        ensure!(
            self.completed_segments < self.total_segments,
            "Speech plan already completed"
        );
        let index = self.completed_segments;
        self.active_segment = Some(index);
        self.publication_started = false;
        self.save(attempt).await?;
        Ok(index)
    }

    /// The immutable object contains mono little-endian f32 PCM, never base64.
    /// Provider reservation consumption, opaque media publication, the attempt
    /// reference, and the publication marker commit in one metadata transaction.
    pub async fn publish_pcm(
        &mut self,
        attempt: &StageExecutionContext,
        artifact_store: &ArtifactStore,
        tenant: &ArtifactTenant,
        bytes: Vec<u8>,
        sample_count: u64,
        sample_rate: u32,
    ) -> anyhow::Result<RuntimeArtifact> {
        let segment = self
            .active_segment
            .context("Speech PCM has no active segment")?;
        ensure!(
            sample_rate > 0 && sample_count > 0,
            "Invalid speech PCM format"
        );
        ensure!(
            self.sample_rate.is_none_or(|rate| rate == sample_rate),
            "Speech PCM sample rate changed"
        );
        let size_bytes = sample_count
            .checked_mul(4)
            .context("Speech PCM size overflow")?;
        ensure!(
            size_bytes <= MAX_PCM_REPLAY_BATCH_BYTES,
            "Speech PCM replay batch exceeds one MiB"
        );
        ensure!(
            bytes.len() as u64 == size_bytes,
            "Speech PCM replay byte count does not match its samples"
        );
        let total = self
            .journal_bytes
            .checked_add(size_bytes)
            .context("Speech replay size overflow")?;
        ensure!(
            total <= self.max_journal_bytes,
            "speech_storage_limit: replay quota exceeded"
        );
        let next_samples = self
            .committed_samples
            .checked_add(sample_count)
            .context("Speech sample count overflow")?;
        let next_sequence = self
            .next_sequence
            .checked_add(1)
            .context("Speech sequence overflow")?;
        ensure!(
            next_sequence <= MAX_PCM_REPLAY_ENTRIES,
            "speech_storage_limit: replay entry quota exceeded"
        );
        let batch = SpeechPcmBatch {
            version: SPEECH_CHECKPOINT_VERSION,
            sequence: self.next_sequence,
            segment,
            sample_offset: self.committed_samples,
            sample_count,
            sample_rate,
        };
        let metadata = serde_json::to_value(batch)?;
        // The checkpoint after this publication commits with its journal
        // entry; no separate progress write follows.
        let mut published = self.clone();
        published.publication_started = true;
        published.next_sequence = next_sequence;
        published.committed_samples = next_samples;
        published.sample_rate = Some(sample_rate);
        published.journal_bytes = total;
        let artifact = artifact_store
            .put_attempt_artifact(
                tenant,
                attempt.lease(),
                AttemptArtifactWrite {
                    publication_key: pcm_publication_key(self.next_sequence),
                    artifact_kind: RuntimeArtifactKind::Audio,
                    artifact_role: RuntimeArtifactRole::OutputIntermediate,
                    metadata_json: metadata.clone(),
                    runtime_retention_policy: "speech_job_replay".to_string(),
                    object: ArtifactWrite {
                        content_type: "audio/pcm-f32le".to_string(),
                        filename: Some("chunk.f32le".to_string()),
                        bytes,
                        retention: ArtifactRetention::Job,
                    },
                },
                serde_json::to_value(&published)?,
            )
            .await?;
        // A reused publication key is accepted only for the same bytes: the
        // store compares the committed digest with this write's.
        ensure!(
            artifact.size_bytes == Some(size_bytes)
                && artifact.metadata_json == metadata
                && artifact.media_asset_id.is_some()
                && artifact.storage_key.is_none(),
            "Speech replay publication key was reused for different PCM"
        );
        *self = published;
        Ok(artifact)
    }

    pub async fn complete_segment(
        &mut self,
        attempt: &StageExecutionContext,
        text_end: u64,
    ) -> anyhow::Result<()> {
        ensure!(
            self.active_segment == Some(self.completed_segments),
            "Speech segment completion out of order"
        );
        ensure!(
            text_end > self.completed_text_bytes,
            "Speech text progress must advance"
        );
        self.completed_segments += 1;
        self.completed_text_bytes = text_end;
        self.active_segment = None;
        self.publication_started = false;
        self.save(attempt).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn recovery_accepts_committed_boundary_but_rejects_partial_publication() {
        let identity = serde_json::json!({"plan": "sha256-plan", "model": "fish"});
        let mut checkpoint = SpeechCheckpoint::new(identity.clone(), 3, 1024).unwrap();
        checkpoint.completed_segments = 1;
        checkpoint.completed_text_bytes = 100;
        checkpoint.active_segment = Some(1);
        let encoded = serde_json::to_value(&checkpoint).unwrap();
        assert_eq!(
            SpeechCheckpoint::recover(&encoded, &identity)
                .unwrap()
                .completed_segments,
            1
        );
        checkpoint.publication_started = true;
        let error =
            SpeechCheckpoint::recover(&serde_json::to_value(&checkpoint).unwrap(), &identity)
                .unwrap_err();
        assert!(error
            .to_string()
            .contains("speech_partial_segment_interrupted"));
    }
    #[test]
    fn recovery_rejects_identity_version_and_storage_changes() {
        let identity = serde_json::json!({"reference": "stable"});
        let mut checkpoint = SpeechCheckpoint::new(identity.clone(), 1, 1024).unwrap();
        assert!(SpeechCheckpoint::recover(
            &serde_json::to_value(&checkpoint).unwrap(),
            &serde_json::json!({})
        )
        .is_err());
        checkpoint.version += 1;
        assert!(
            SpeechCheckpoint::recover(&serde_json::to_value(&checkpoint).unwrap(), &identity)
                .is_err()
        );
        checkpoint.version = SPEECH_CHECKPOINT_VERSION;
        checkpoint.journal_bytes = 1025;
        assert!(
            SpeechCheckpoint::recover(&serde_json::to_value(&checkpoint).unwrap(), &identity)
                .is_err()
        );
        checkpoint.journal_bytes = 0;
        checkpoint.next_sequence = MAX_PCM_REPLAY_ENTRIES + 1;
        assert!(
            SpeechCheckpoint::recover(&serde_json::to_value(&checkpoint).unwrap(), &identity)
                .is_err()
        );
    }
    #[test]
    fn replay_entry_ceiling_covers_two_hours_of_default_fish_chunks() {
        let samples = 120_u64 * 60 * 44_100;
        let entries = samples.div_ceil(4_800);
        assert_eq!(entries, 66_150);
        assert!(entries < MAX_PCM_REPLAY_ENTRIES);
    }
    fn pcm_chunk(samples: usize) -> AudioChunk {
        let mut chunk = AudioChunk::new("speech".to_string(), 0, vec![0.25; samples]);
        chunk.sample_rate = 44_100;
        chunk
    }

    #[test]
    fn publication_buffer_merges_chunks_between_intervals() {
        let start = Instant::now();
        let interval = Duration::from_millis(250);
        let mut buffer = PcmPublicationBuffer::new(interval, 8);

        // The first chunk publishes at once.
        assert!(buffer.push(pcm_chunk(4_800)).is_none());
        assert_eq!(buffer.take_due(start).unwrap().samples.len(), 4_800);
        assert_eq!(buffer.deadline(), None);

        // Chunks inside the interval wait and merge in order.
        for offset in [50, 100, 150] {
            assert!(buffer.push(pcm_chunk(4_800)).is_none());
            assert!(buffer
                .take_due(start + Duration::from_millis(offset))
                .is_none());
        }
        assert_eq!(buffer.deadline(), Some(start + interval));
        let merged = buffer.take_due(start + interval).unwrap();
        assert_eq!(merged.samples.len(), 3 * 4_800);
        assert_eq!(merged.sample_rate, 44_100);

        // Reaching the chunk limit publishes before the interval.
        let mut buffer = PcmPublicationBuffer::new(interval, 2);
        buffer.push(pcm_chunk(10));
        buffer.take_due(start).unwrap();
        buffer.push(pcm_chunk(10));
        assert!(buffer.take_due(start).is_none());
        buffer.push(pcm_chunk(10));
        assert_eq!(buffer.take_due(start).unwrap().samples.len(), 20);

        // A segment end takes whatever is waiting.
        buffer.push(pcm_chunk(7));
        assert_eq!(buffer.take(start).unwrap().samples.len(), 7);
        assert!(buffer.take(start).is_none());
    }

    #[test]
    fn publication_buffer_never_merges_past_one_mib() {
        let start = Instant::now();
        let mut buffer = PcmPublicationBuffer::new(Duration::from_secs(60), 1_000);
        buffer.push(pcm_chunk(1));
        buffer.take_due(start).unwrap();
        let limit = (MAX_PCM_REPLAY_BATCH_BYTES / 4) as usize;
        assert!(buffer.push(pcm_chunk(limit - 100)).is_none());
        let flushed = buffer.push(pcm_chunk(200)).expect("overflow flushes first");
        assert_eq!(flushed.samples.len(), limit - 100);
        assert_eq!(buffer.take(start).unwrap().samples.len(), 200);
    }

    /// DUR-F7: the checkpoint after a publication commits in the
    /// publication's own transaction; no separate progress write follows.
    #[tokio::test]
    async fn publication_commits_the_next_checkpoint_with_its_journal_entry() {
        use crate::artifact_store::ArtifactStoreLimits;
        use crate::batch_runtime::store::{BatchRuntimeStore, NewJobStage, NewRuntimeJob};
        use crate::batch_runtime::types::{RuntimeJobKind, RuntimeJobStatus, RuntimeStageStatus};
        use std::sync::Arc;

        let root = tempfile::tempdir().unwrap();
        let metadata = Arc::new(BatchRuntimeStore::initialize_with_database(
            crate::db::StoreDatabase::new(root.path().join("speech.sqlite3")),
        ));
        let artifact_store = ArtifactStore::new(
            metadata.clone(),
            Arc::new(crate::persistence::LocalMediaStorageProvider::new(
                root.path().join("media"),
            )),
            ArtifactStoreLimits::default(),
        )
        .unwrap();
        let job = metadata
            .create_job(NewRuntimeJob {
                job_kind: RuntimeJobKind::TtsSpeech,
                status: RuntimeJobStatus::Queued,
                priority: 0,
                model_id: Some("FishAudio-S2-Pro".into()),
                capability: Some("tts".into()),
                route_record_kind: None,
                route_record_id: None,
                input_media_asset_id: None,
                input_text_asset_id: None,
                request_json: serde_json::json!({"tenant_key": null}),
                model_snapshot_json: serde_json::json!({}),
                retry_policy_json: serde_json::json!({}),
                max_attempts: 1,
                idempotency_key: None,
                correlation_id: None,
            })
            .await
            .unwrap();
        metadata
            .create_stage(NewJobStage {
                job_id: job.id,
                sequence: 0,
                stage_kind: "tts_synthesize".into(),
                status: RuntimeStageStatus::Queued,
                capability: Some("tts".into()),
                model_id: None,
                max_attempts: 1,
                input_artifact_ids: vec![],
            })
            .await
            .unwrap();
        let claimed = metadata
            .claim_next_stage("speech-worker", 60_000)
            .await
            .unwrap()
            .unwrap();
        let attempt = StageExecutionContext::for_test(metadata.clone(), claimed);
        let tenant = ArtifactTenant::from_scheduling_key(None);
        let mut checkpoint =
            SpeechCheckpoint::new(serde_json::json!({"plan": "one"}), 1, 1 << 20).unwrap();
        checkpoint.begin_segment(&attempt).await.unwrap();
        for published in 1..=2_u64 {
            checkpoint
                .publish_pcm(
                    &attempt,
                    &artifact_store,
                    &tenant,
                    vec![0; 400],
                    100,
                    24_000,
                )
                .await
                .unwrap();
            let stage = metadata
                .get_stage(&attempt.lease().stage_id)
                .await
                .unwrap()
                .unwrap();
            let stored: SpeechCheckpoint =
                serde_json::from_value(stage.progress_json.unwrap()).unwrap();
            assert_eq!(stored, checkpoint);
            assert_eq!(stored.next_sequence, published);
            assert_eq!(stored.committed_samples, published * 100);
        }
    }

    #[test]
    fn cursor_keys_sort_numerically_including_large_sequences() {
        let values = [0, 9, 10, 99, 100, u64::MAX];
        for pair in values.windows(2) {
            assert!(pcm_publication_key(pair[0]) < pcm_publication_key(pair[1]));
        }
    }
}
