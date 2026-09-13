//! Versioned, attempt-fenced speech checkpoints and immutable PCM replay entries.
//!
//! PCM objects must be persisted before `publish_pcm` and only sent to listeners
//! after it succeeds. A publication intent is checkpointed first, so a crash in
//! either half of publication can never silently regenerate an audible segment.
use super::{
    store::sha256_hex,
    types::{RuntimeArtifact, RuntimeArtifactKind, RuntimeArtifactRole},
    worker::StageExecutionContext,
};
use crate::artifact_store::{
    ArtifactRetention, ArtifactStore, ArtifactTenant, ArtifactWrite, AttemptArtifactWrite,
};
use anyhow::{bail, ensure, Context};
use serde::{Deserialize, Serialize};

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
        let expected_sha256 = sha256_hex(&bytes);
        let mut publication_checkpoint = self.clone();
        publication_checkpoint.publication_started = true;
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
                serde_json::to_value(publication_checkpoint)?,
            )
            .await?;
        ensure!(
            artifact.sha256.as_deref() == Some(expected_sha256.as_str())
                && artifact.size_bytes == Some(size_bytes)
                && artifact.metadata_json == metadata
                && artifact.media_asset_id.is_some()
                && artifact.storage_key.is_none(),
            "Speech replay publication key was reused for different PCM"
        );
        self.publication_started = true;
        self.next_sequence = next_sequence;
        self.committed_samples = next_samples;
        self.sample_rate = Some(sample_rate);
        self.journal_bytes = total;
        self.save(attempt).await?;
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
    #[test]
    fn cursor_keys_sort_numerically_including_large_sequences() {
        let values = [0, 9, 10, 99, 100, u64::MAX];
        for pair in values.windows(2) {
            assert!(pcm_publication_key(pair[0]) < pcm_publication_key(pair[1]));
        }
    }
}
