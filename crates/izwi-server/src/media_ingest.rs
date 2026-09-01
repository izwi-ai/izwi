use std::{
    path::Path,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use izwi_core::audio::{
    decode_and_inspect_audio_bytes, resample_mono_high_quality, AudioEncoder, AudioFormat,
    AudioInspection, AudioSourceMetadata,
};
use izwi_hooks::{HookMetadata, MediaNamespace, MediaStorageProvider, StoredMediaBytes};
use serde::Serialize;
use tokio::sync::Semaphore;
use tracing::{info, warn};

use crate::{
    batch_runtime::{
        store::{sha256_hex, BatchRuntimeStore, NewMediaAsset},
        types::MediaAsset,
    },
    persistence::{
        delete_media_object, persist_audio_object, read_media_object, MediaStorageError,
    },
};

const DEFAULT_MEDIA_DECODE_LANES: usize = 2;
const DEFAULT_MAX_AUDIO_DURATION_SECS: f32 = 60.0 * 60.0;
const DEFAULT_MAX_DECODED_SAMPLES: usize = 57_600_000;
const DEFAULT_MAX_SOURCE_CHANNELS: u16 = 8;
const MEDIA_CONTRACT_VERSION: u32 = 1;
const SECURITY_SCAN_STATUS_NOT_SCANNED: &str = "not_scanned";
const DEFAULT_ALLOWED_CONTAINERS: &[&str] = &[
    "wav", "mp3", "flac", "ogg", "matroska", "mp4", "aiff", "caf", "aac",
];
const DEFAULT_ALLOWED_CODECS: &[&str] = &[
    "pcm_*", "adpcm_*", "mp1", "mp2", "mp3", "aac", "opus", "vorbis", "flac", "alac", "speex",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanonicalAudioProfile {
    SpeechRecognition16KhzMonoWav,
    ReferenceVoiceSourceRateMonoWav,
}

impl CanonicalAudioProfile {
    pub const fn id(self) -> &'static str {
        match self {
            Self::SpeechRecognition16KhzMonoWav => "asr_16khz_mono_pcm16_wav_v1",
            Self::ReferenceVoiceSourceRateMonoWav => {
                "reference_voice_source_rate_mono_pcm16_wav_v1"
            }
        }
    }

    pub const fn target_sample_rate(self, source_sample_rate: u32) -> u32 {
        match self {
            Self::SpeechRecognition16KhzMonoWav => 16_000,
            Self::ReferenceVoiceSourceRateMonoWav => source_sample_rate,
        }
    }

    pub const fn asset_kind(self) -> &'static str {
        match self {
            Self::SpeechRecognition16KhzMonoWav => "audio_canonical_wav",
            Self::ReferenceVoiceSourceRateMonoWav => "audio_reference_canonical_wav",
        }
    }

    pub fn content_type(self) -> &'static str {
        AudioEncoder::content_type(AudioFormat::Wav)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AudioIngestPolicy {
    pub max_source_bytes: usize,
    pub max_decoded_samples: usize,
    pub max_duration_secs: Option<f32>,
    pub max_source_channels: Option<u16>,
    pub allowed_containers: Vec<String>,
    pub allowed_codecs: Vec<String>,
    pub corruption_policy: AudioCorruptionPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioCorruptionPolicy {
    Reject,
}

impl AudioIngestPolicy {
    pub fn media_upload(max_source_bytes: usize) -> Self {
        Self {
            max_source_bytes: max_source_bytes.max(1),
            max_decoded_samples: DEFAULT_MAX_DECODED_SAMPLES,
            max_duration_secs: Some(DEFAULT_MAX_AUDIO_DURATION_SECS),
            max_source_channels: Some(DEFAULT_MAX_SOURCE_CHANNELS),
            allowed_containers: DEFAULT_ALLOWED_CONTAINERS
                .iter()
                .map(|value| (*value).to_string())
                .collect(),
            allowed_codecs: DEFAULT_ALLOWED_CODECS
                .iter()
                .map(|value| (*value).to_string())
                .collect(),
            corruption_policy: AudioCorruptionPolicy::Reject,
        }
    }

    fn validate_source(
        &self,
        source_size_bytes: usize,
        source: &AudioSourceMetadata,
        inspection: &AudioInspection,
    ) -> Result<(), MediaIngestError> {
        if source_size_bytes > self.max_source_bytes {
            return Err(MediaIngestError::InvalidInput(format!(
                "Media payload exceeded the configured source limit ({} > {} bytes)",
                source_size_bytes, self.max_source_bytes
            )));
        }
        if inspection.sample_count > self.max_decoded_samples {
            return Err(MediaIngestError::InvalidInput(format!(
                "Decoded audio sample count exceeded the configured limit ({} > {})",
                inspection.sample_count, self.max_decoded_samples
            )));
        }
        if !value_matches_rules(&source.container, &self.allowed_containers) {
            return Err(MediaIngestError::InvalidInput(format!(
                "Unsupported audio container `{}`",
                source.container
            )));
        }
        if !value_matches_rules(&source.codec, &self.allowed_codecs) {
            return Err(MediaIngestError::InvalidInput(format!(
                "Unsupported audio codec `{}`",
                source.codec
            )));
        }
        if let Some(max_duration_secs) = self.max_duration_secs.filter(|value| *value > 0.0) {
            if inspection.duration_secs > max_duration_secs {
                return Err(MediaIngestError::InvalidInput(format!(
                    "Audio duration exceeded the configured limit ({:.3} > {:.3} seconds)",
                    inspection.duration_secs, max_duration_secs
                )));
            }
        }
        if let Some(max_source_channels) = self.max_source_channels.filter(|value| *value > 0) {
            if source.channel_count > max_source_channels {
                return Err(MediaIngestError::InvalidInput(format!(
                    "Audio channel count exceeded the configured limit ({} > {})",
                    source.channel_count, max_source_channels
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct MediaIngestRequest {
    pub bytes: Vec<u8>,
    pub content_type: String,
    pub filename: Option<String>,
    pub namespace: String,
    pub record_id: String,
    pub route: String,
}

#[derive(Debug)]
pub struct ExistingAudioIngestRequest {
    pub bytes: Vec<u8>,
    pub storage_key: String,
    pub storage_namespace: String,
    pub content_type: String,
    pub filename: Option<String>,
    pub record_id: String,
    pub route: String,
}

#[derive(Debug, Clone)]
pub struct CanonicalMediaIngestResult {
    pub storage_key: String,
    pub asset: MediaAsset,
}

#[derive(Debug, Clone)]
pub struct MediaIngestResult {
    pub original_storage_key: String,
    pub source_size_bytes: u64,
    pub source_asset: Option<MediaAsset>,
    pub canonical: Option<CanonicalMediaIngestResult>,
}

#[derive(Debug, thiserror::Error)]
pub enum MediaIngestError {
    #[error("{0}")]
    InvalidInput(String),
    #[error("media processing task failed: {0}")]
    ProcessingTask(String),
    #[error("media storage operation failed: {0}")]
    Storage(#[source] anyhow::Error),
    #[error("media asset registration failed: {0}")]
    AssetRegistration(#[source] anyhow::Error),
}

impl MediaIngestError {
    pub fn is_invalid_input(&self) -> bool {
        matches!(self, Self::InvalidInput(_))
    }
}

#[derive(Clone)]
pub struct MediaIngestService {
    media_storage: Arc<dyn MediaStorageProvider>,
    batch_store: Arc<BatchRuntimeStore>,
    decode_lanes: Arc<Semaphore>,
    decode_lane_capacity: usize,
    decode_waiters: Arc<AtomicUsize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct MediaIngestLaneSnapshot {
    pub capacity: usize,
    pub active: usize,
    pub available: usize,
    pub queued: usize,
}

impl MediaIngestService {
    pub fn new(
        media_storage: Arc<dyn MediaStorageProvider>,
        batch_store: Arc<BatchRuntimeStore>,
    ) -> Self {
        Self::with_decode_lane_capacity(
            media_storage,
            batch_store,
            resolve_media_decode_lane_capacity(),
        )
    }

    fn with_decode_lane_capacity(
        media_storage: Arc<dyn MediaStorageProvider>,
        batch_store: Arc<BatchRuntimeStore>,
        decode_lane_capacity: usize,
    ) -> Self {
        let decode_lane_capacity = decode_lane_capacity.max(1);
        Self {
            media_storage,
            batch_store,
            decode_lanes: Arc::new(Semaphore::new(decode_lane_capacity)),
            decode_lane_capacity,
            decode_waiters: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn lane_snapshot(&self) -> MediaIngestLaneSnapshot {
        let available = self.decode_lanes.available_permits();
        MediaIngestLaneSnapshot {
            capacity: self.decode_lane_capacity,
            active: self.decode_lane_capacity.saturating_sub(available),
            available,
            queued: self.decode_waiters.load(Ordering::Acquire),
        }
    }

    pub async fn ingest(
        &self,
        request: MediaIngestRequest,
        policy: AudioIngestPolicy,
        canonical_profile: CanonicalAudioProfile,
    ) -> Result<MediaIngestResult, MediaIngestError> {
        self.ingest_inner(request, policy, canonical_profile, false)
            .await
    }

    /// Strictly ingest a payload that the calling route requires to be audio.
    ///
    /// Unlike the general `/v1/media` entry point, this never treats an unrecognized payload as
    /// opaque media. Reference-voice and other model inputs therefore fail before job creation.
    pub async fn ingest_audio(
        &self,
        request: MediaIngestRequest,
        policy: AudioIngestPolicy,
        canonical_profile: CanonicalAudioProfile,
    ) -> Result<MediaIngestResult, MediaIngestError> {
        self.ingest_inner(request, policy, canonical_profile, true)
            .await
    }

    async fn ingest_inner(
        &self,
        request: MediaIngestRequest,
        policy: AudioIngestPolicy,
        canonical_profile: CanonicalAudioProfile,
        require_audio: bool,
    ) -> Result<MediaIngestResult, MediaIngestError> {
        let MediaIngestRequest {
            bytes,
            content_type,
            filename,
            namespace,
            record_id,
            route,
        } = request;
        if bytes.is_empty() {
            return Err(MediaIngestError::InvalidInput(
                "Media payload cannot be empty".to_string(),
            ));
        }
        if bytes.len() > policy.max_source_bytes {
            return Err(MediaIngestError::InvalidInput(format!(
                "Media payload exceeded the configured source limit ({} > {} bytes)",
                bytes.len(),
                policy.max_source_bytes
            )));
        }

        let audio_like = require_audio || is_audio_like(&bytes, &content_type, filename.as_deref());
        let (prepared, opaque_bytes) = if audio_like {
            (
                Some(self.prepare_audio(bytes, policy, canonical_profile).await?),
                None,
            )
        } else {
            (None, Some(bytes))
        };
        let source_bytes = prepared
            .as_ref()
            .map(|prepared| prepared.source_bytes.as_slice())
            .or(opaque_bytes.as_deref())
            .expect("media ingest must retain either prepared or opaque source bytes");
        let source_size_bytes = source_bytes.len() as u64;
        let source_namespace = format!("media/{namespace}");
        let original_storage_key = self
            .persist(
                MediaNamespace::Other(source_namespace.clone()),
                record_id.clone(),
                filename.as_deref(),
                &content_type,
                source_bytes,
                source_storage_metadata(&route, prepared.as_ref()),
            )
            .await?;

        let Some(prepared) = prepared else {
            return Ok(MediaIngestResult {
                original_storage_key,
                source_size_bytes,
                source_asset: None,
                canonical: None,
            });
        };

        let canonical_namespace = format!("media/{namespace}/canonical");
        let canonical_filename = canonical_filename(filename.as_deref());
        let canonical_record_id =
            canonical_derivative_record_id(&original_storage_key, canonical_profile);
        let canonical_storage_key = match self
            .persist(
                MediaNamespace::Other(canonical_namespace.clone()),
                canonical_record_id,
                canonical_filename.as_deref(),
                canonical_profile.content_type(),
                &prepared.canonical_bytes,
                canonical_storage_metadata(&route, canonical_profile),
            )
            .await
        {
            Ok(key) => key,
            Err(err) => {
                self.compensate_delete(&original_storage_key).await;
                return Err(err);
            }
        };

        let source_asset = match self
            .register_source_asset(
                &source_namespace,
                &original_storage_key,
                &content_type,
                filename.as_deref(),
                &route,
                &prepared,
            )
            .await
        {
            Ok(asset) => asset,
            Err(err) => {
                self.compensate_delete(&canonical_storage_key).await;
                self.compensate_delete(&original_storage_key).await;
                return Err(err);
            }
        };

        let canonical_asset = match self
            .register_canonical_asset(
                &canonical_namespace,
                &canonical_storage_key,
                canonical_filename.as_deref(),
                &route,
                canonical_profile,
                &source_asset,
                &prepared,
            )
            .await
        {
            Ok(asset) => asset,
            Err(err) => {
                // The original object remains valid because its asset row was committed. Remove
                // only the unregistered canonical object to avoid leaving an orphan.
                self.compensate_delete(&canonical_storage_key).await;
                return Err(err);
            }
        };
        if canonical_asset.storage_key != canonical_storage_key {
            self.compensate_delete(&canonical_storage_key).await;
        }

        info!(
            target: "izwi.audio",
            route,
            source_container = prepared.source.container,
            source_codec = prepared.source.codec,
            source_sample_rate = prepared.source.sample_rate,
            source_channel_count = prepared.source.channel_count,
            source_duration_secs = prepared.source_inspection.duration_secs,
            canonical_profile = canonical_profile.id(),
            canonical_sample_rate = prepared.canonical_inspection.sample_rate,
            "strict audio media ingest completed"
        );

        Ok(MediaIngestResult {
            original_storage_key,
            source_size_bytes,
            source_asset: Some(source_asset),
            canonical: Some(CanonicalMediaIngestResult {
                storage_key: canonical_asset.storage_key.clone(),
                asset: canonical_asset,
            }),
        })
    }

    /// Register and canonicalize an audio object that is already durably stored.
    ///
    /// This is used when a product projection owns the original download object. The original
    /// bytes are decoded once, its existing key becomes the runtime source asset, and only the
    /// immutable canonical derivative is written.
    pub async fn ingest_existing_audio(
        &self,
        request: ExistingAudioIngestRequest,
        policy: AudioIngestPolicy,
        canonical_profile: CanonicalAudioProfile,
    ) -> Result<MediaIngestResult, MediaIngestError> {
        let ExistingAudioIngestRequest {
            bytes,
            storage_key,
            storage_namespace,
            content_type,
            filename,
            record_id: _,
            route,
        } = request;
        if bytes.is_empty() {
            return Err(MediaIngestError::InvalidInput(
                "Media payload cannot be empty".to_string(),
            ));
        }

        let prepared = self.prepare_audio(bytes, policy, canonical_profile).await?;
        let canonical_namespace = format!("{storage_namespace}/canonical");
        let canonical_filename = canonical_filename(filename.as_deref());
        let canonical_record_id = canonical_derivative_record_id(&storage_key, canonical_profile);
        let canonical_storage_key = self
            .persist(
                MediaNamespace::Other(canonical_namespace.clone()),
                canonical_record_id,
                canonical_filename.as_deref(),
                canonical_profile.content_type(),
                &prepared.canonical_bytes,
                canonical_storage_metadata(&route, canonical_profile),
            )
            .await?;

        let source_asset = match self
            .register_source_asset(
                &storage_namespace,
                &storage_key,
                &content_type,
                filename.as_deref(),
                &route,
                &prepared,
            )
            .await
        {
            Ok(asset) => asset,
            Err(err) => {
                self.compensate_delete(&canonical_storage_key).await;
                return Err(err);
            }
        };
        let canonical_asset = match self
            .register_canonical_asset(
                &canonical_namespace,
                &canonical_storage_key,
                canonical_filename.as_deref(),
                &route,
                canonical_profile,
                &source_asset,
                &prepared,
            )
            .await
        {
            Ok(asset) => asset,
            Err(err) => {
                self.compensate_delete(&canonical_storage_key).await;
                return Err(err);
            }
        };
        if canonical_asset.storage_key != canonical_storage_key {
            self.compensate_delete(&canonical_storage_key).await;
        }

        Ok(MediaIngestResult {
            original_storage_key: storage_key,
            source_size_bytes: prepared.source_bytes.len() as u64,
            source_asset: Some(source_asset),
            canonical: Some(CanonicalMediaIngestResult {
                storage_key: canonical_asset.storage_key.clone(),
                asset: canonical_asset,
            }),
        })
    }

    pub async fn read_object(&self, key: &str) -> Result<StoredMediaBytes, MediaStorageError> {
        read_media_object(&self.media_storage, key).await
    }

    pub async fn delete_object(&self, key: &str) -> anyhow::Result<()> {
        delete_media_object(&self.media_storage, Some(key)).await
    }

    pub async fn persist_generated_audio(
        &self,
        record_id: String,
        filename: Option<&str>,
        content_type: &str,
        bytes: &[u8],
        route: &str,
    ) -> Result<String, MediaIngestError> {
        let mut metadata = HookMetadata::new();
        metadata.insert("route".to_string(), route.to_string());
        self.persist(
            MediaNamespace::GeneratedSpeech,
            record_id,
            filename,
            content_type,
            bytes,
            metadata,
        )
        .await
    }

    async fn prepare_audio(
        &self,
        source_bytes: Vec<u8>,
        policy: AudioIngestPolicy,
        canonical_profile: CanonicalAudioProfile,
    ) -> Result<PreparedAudio, MediaIngestError> {
        let waiting = DecodeWaiterGuard::new(self.decode_waiters.clone());
        let permit = self
            .decode_lanes
            .clone()
            .acquire_owned()
            .await
            .map_err(|err| MediaIngestError::ProcessingTask(err.to_string()))?;
        drop(waiting);
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            prepare_audio_blocking(source_bytes, &policy, canonical_profile)
        })
        .await
        .map_err(|err| MediaIngestError::ProcessingTask(err.to_string()))?
    }

    async fn persist(
        &self,
        namespace: MediaNamespace,
        record_id: String,
        filename: Option<&str>,
        content_type: &str,
        bytes: &[u8],
        metadata: HookMetadata,
    ) -> Result<String, MediaIngestError> {
        persist_audio_object(
            &self.media_storage,
            namespace,
            record_id,
            filename,
            content_type,
            bytes,
            metadata,
        )
        .await
        .map_err(MediaIngestError::Storage)
    }

    async fn register_source_asset(
        &self,
        storage_namespace: &str,
        storage_key: &str,
        content_type: &str,
        filename: Option<&str>,
        route: &str,
        prepared: &PreparedAudio,
    ) -> Result<MediaAsset, MediaIngestError> {
        let source_sha256 = sha256_hex(&prepared.source_bytes);
        let result = self
            .batch_store
            .create_media_asset(NewMediaAsset {
                asset_kind: "audio_original".to_string(),
                storage_namespace: storage_namespace.to_string(),
                storage_key: storage_key.to_string(),
                content_type: content_type.to_string(),
                filename: filename.map(ToOwned::to_owned),
                size_bytes: prepared.source_bytes.len() as u64,
                sha256: Some(source_sha256.clone()),
                duration_secs: Some(prepared.source_inspection.duration_secs as f64),
                sample_rate_hz: Some(prepared.source.sample_rate),
                channel_count: Some(prepared.source.channel_count),
                peak_amplitude: Some(prepared.source_inspection.peak),
                rms_amplitude: Some(prepared.source_inspection.rms),
                source_asset_id: None,
                canonical_profile_version: None,
                scan_status: SECURITY_SCAN_STATUS_NOT_SCANNED.to_string(),
                retention_policy: "default".to_string(),
                metadata_json: serde_json::json!({
                    "media_contract_version": MEDIA_CONTRACT_VERSION,
                    "route": route,
                    "normalized": false,
                    "decode_validation": {
                        "status": "passed",
                        "strict": true
                    },
                    "security_scan": {
                        "status": SECURITY_SCAN_STATUS_NOT_SCANNED
                    },
                    "source": {
                        "container": prepared.source.container,
                        "codec": prepared.source.codec,
                        "sample_rate_hz": prepared.source.sample_rate,
                        "channel_count": prepared.source.channel_count
                    }
                }),
            })
            .await;
        match result {
            Ok(asset) => Ok(asset),
            Err(create_error) => match self
                .batch_store
                .get_media_asset_by_storage_key(storage_key)
                .await
            {
                Ok(Some(asset)) if asset.sha256.as_deref() == Some(source_sha256.as_str()) => {
                    Ok(asset)
                }
                Ok(Some(asset)) => Err(MediaIngestError::AssetRegistration(anyhow::anyhow!(
                    "Media storage key `{storage_key}` is already registered with checksum {:?}; refusing to replace immutable source bytes",
                    asset.sha256
                ))),
                Ok(None) => Err(MediaIngestError::AssetRegistration(create_error)),
                Err(lookup_error) => Err(MediaIngestError::AssetRegistration(anyhow::anyhow!(
                    "{create_error}; failed to resolve the existing media asset: {lookup_error}"
                ))),
            },
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn register_canonical_asset(
        &self,
        storage_namespace: &str,
        storage_key: &str,
        filename: Option<&str>,
        route: &str,
        profile: CanonicalAudioProfile,
        source_asset: &MediaAsset,
        prepared: &PreparedAudio,
    ) -> Result<MediaAsset, MediaIngestError> {
        let canonical_sha256 = sha256_hex(&prepared.canonical_bytes);
        let result = self
            .batch_store
            .create_media_asset(NewMediaAsset {
                asset_kind: profile.asset_kind().to_string(),
                storage_namespace: storage_namespace.to_string(),
                storage_key: storage_key.to_string(),
                content_type: profile.content_type().to_string(),
                filename: filename.map(ToOwned::to_owned),
                size_bytes: prepared.canonical_bytes.len() as u64,
                sha256: Some(canonical_sha256.clone()),
                duration_secs: Some(prepared.canonical_inspection.duration_secs as f64),
                sample_rate_hz: Some(prepared.canonical_inspection.sample_rate),
                channel_count: Some(1),
                peak_amplitude: Some(prepared.canonical_inspection.peak),
                rms_amplitude: Some(prepared.canonical_inspection.rms),
                source_asset_id: Some(source_asset.id.clone()),
                canonical_profile_version: Some(profile.id().to_string()),
                scan_status: SECURITY_SCAN_STATUS_NOT_SCANNED.to_string(),
                retention_policy: "default".to_string(),
                metadata_json: serde_json::json!({
                    "media_contract_version": MEDIA_CONTRACT_VERSION,
                    "route": route,
                    "normalized": true,
                    "canonical_profile": profile.id(),
                    "decode_validation": {
                        "status": "derived_from_validated_source",
                        "strict": true
                    },
                    "security_scan": {
                        "status": SECURITY_SCAN_STATUS_NOT_SCANNED
                    },
                    "source_media_asset_id": source_asset.id,
                    "source_storage_key": source_asset.storage_key,
                    "source": {
                        "container": prepared.source.container,
                        "codec": prepared.source.codec,
                        "sample_rate_hz": prepared.source.sample_rate,
                        "channel_count": prepared.source.channel_count
                    }
                }),
            })
            .await;
        match result {
            Ok(asset) => Ok(asset),
            Err(create_error) => match self
                .batch_store
                .get_canonical_media_asset(&source_asset.id, profile.id())
                .await
            {
                Ok(Some(asset)) if asset.sha256.as_deref() == Some(canonical_sha256.as_str()) => {
                    Ok(asset)
                }
                Ok(Some(asset)) => Err(MediaIngestError::AssetRegistration(anyhow::anyhow!(
                    "Canonical profile `{}` for source asset `{}` already has checksum {:?}; refusing divergent derivative bytes",
                    profile.id(),
                    source_asset.id,
                    asset.sha256
                ))),
                Ok(None) => Err(MediaIngestError::AssetRegistration(create_error)),
                Err(lookup_error) => Err(MediaIngestError::AssetRegistration(anyhow::anyhow!(
                    "{create_error}; failed to resolve the existing canonical asset: {lookup_error}"
                ))),
            },
        }
    }

    async fn compensate_delete(&self, key: &str) {
        if let Err(err) = delete_media_object(&self.media_storage, Some(key)).await {
            warn!(storage_key = key, error = %err, "failed compensating media object write");
        }
    }
}

struct DecodeWaiterGuard {
    waiters: Arc<AtomicUsize>,
}

impl DecodeWaiterGuard {
    fn new(waiters: Arc<AtomicUsize>) -> Self {
        waiters.fetch_add(1, Ordering::AcqRel);
        Self { waiters }
    }
}

impl Drop for DecodeWaiterGuard {
    fn drop(&mut self) {
        self.waiters.fetch_sub(1, Ordering::AcqRel);
    }
}

struct PreparedAudio {
    source_bytes: Vec<u8>,
    source: AudioSourceMetadata,
    source_inspection: AudioInspection,
    canonical_bytes: Vec<u8>,
    canonical_inspection: AudioInspection,
}

fn prepare_audio_blocking(
    source_bytes: Vec<u8>,
    policy: &AudioIngestPolicy,
    profile: CanonicalAudioProfile,
) -> Result<PreparedAudio, MediaIngestError> {
    let decoded = match policy.corruption_policy {
        AudioCorruptionPolicy::Reject => decode_and_inspect_audio_bytes(&source_bytes),
    }
    .map_err(|err| MediaIngestError::InvalidInput(format!("Invalid audio payload: {err}")))?;
    policy.validate_source(source_bytes.len(), &decoded.source, &decoded.inspection)?;
    let target_sample_rate = profile.target_sample_rate(decoded.source.sample_rate);
    let canonical_samples = resample_mono_high_quality(
        &decoded.mono_samples,
        decoded.source.sample_rate,
        target_sample_rate,
    )
    .map_err(|err| {
        MediaIngestError::InvalidInput(format!("Failed to canonicalize audio: {err}"))
    })?;
    let canonical_inspection =
        AudioInspection::from_mono_samples(&canonical_samples, target_sample_rate);
    let canonical_bytes = AudioEncoder::new(target_sample_rate, 1)
        .encode(&canonical_samples, AudioFormat::Wav)
        .map_err(|err| MediaIngestError::ProcessingTask(err.to_string()))?;

    Ok(PreparedAudio {
        source_bytes,
        source: decoded.source,
        source_inspection: decoded.inspection,
        canonical_bytes,
        canonical_inspection,
    })
}

fn source_storage_metadata(route: &str, prepared: Option<&PreparedAudio>) -> HookMetadata {
    let mut metadata = HookMetadata::new();
    metadata.insert("route".to_string(), route.to_string());
    metadata.insert("normalized".to_string(), "false".to_string());
    if let Some(prepared) = prepared {
        metadata.insert("decode_validation_status".to_string(), "passed".to_string());
        metadata.insert(
            "security_scan_status".to_string(),
            SECURITY_SCAN_STATUS_NOT_SCANNED.to_string(),
        );
        metadata.insert(
            "source_container".to_string(),
            prepared.source.container.clone(),
        );
        metadata.insert("source_codec".to_string(), prepared.source.codec.clone());
    }
    metadata
}

fn canonical_storage_metadata(route: &str, profile: CanonicalAudioProfile) -> HookMetadata {
    let mut metadata = HookMetadata::new();
    metadata.insert("route".to_string(), route.to_string());
    metadata.insert("normalized".to_string(), "true".to_string());
    metadata.insert("canonical_profile".to_string(), profile.id().to_string());
    metadata.insert(
        "security_scan_status".to_string(),
        SECURITY_SCAN_STATUS_NOT_SCANNED.to_string(),
    );
    metadata
}

fn resolve_media_decode_lane_capacity() -> usize {
    std::env::var("IZWI_MEDIA_DECODE_LANES")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_MEDIA_DECODE_LANES)
}

fn value_matches_rules(value: &str, rules: &[String]) -> bool {
    let normalized = value.trim().to_ascii_lowercase();
    rules.iter().any(|rule| {
        let rule = rule.trim().to_ascii_lowercase();
        rule.strip_suffix('*')
            .map_or(normalized == rule, |prefix| normalized.starts_with(prefix))
    })
}

fn is_audio_like(bytes: &[u8], content_type: &str, filename: Option<&str>) -> bool {
    content_type
        .split(';')
        .next()
        .map(str::trim)
        .is_some_and(|value| value.to_ascii_lowercase().starts_with("audio/"))
        || sniff_audio_container(bytes).is_some()
        || filename
            .and_then(audio_filename_extension)
            .is_some_and(is_audio_extension)
}

fn sniff_audio_container(bytes: &[u8]) -> Option<&'static str> {
    if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WAVE" {
        Some("wav")
    } else if bytes.starts_with(b"fLaC") {
        Some("flac")
    } else if bytes.starts_with(b"OggS") {
        Some("ogg")
    } else if bytes.starts_with(b"ID3") {
        Some("mp3")
    } else if bytes.starts_with(&[0x1a, 0x45, 0xdf, 0xa3]) {
        Some("matroska")
    } else if bytes.len() >= 12 && &bytes[4..8] == b"ftyp" {
        Some("mp4")
    } else if bytes.starts_with(b"FORM")
        && bytes
            .get(8..12)
            .is_some_and(|kind| kind == b"AIFF" || kind == b"AIFC")
    {
        Some("aiff")
    } else if bytes.starts_with(b"caff") {
        Some("caf")
    } else if looks_like_adts_frame(bytes) {
        Some("aac")
    } else if looks_like_mpeg_audio_frame(bytes) {
        Some("mp3")
    } else {
        None
    }
}

fn looks_like_mpeg_audio_frame(bytes: &[u8]) -> bool {
    bytes.len() >= 2 && bytes[0] == 0xff && bytes[1] & 0xe0 == 0xe0
}

fn looks_like_adts_frame(bytes: &[u8]) -> bool {
    bytes.len() >= 2 && bytes[0] == 0xff && bytes[1] & 0xf6 == 0xf0
}

fn audio_filename_extension(filename: &str) -> Option<&str> {
    Path::new(filename)
        .extension()
        .and_then(|extension| extension.to_str())
}

fn is_audio_extension(extension: &str) -> bool {
    matches!(
        extension.to_ascii_lowercase().as_str(),
        "wav"
            | "wave"
            | "mp3"
            | "flac"
            | "ogg"
            | "oga"
            | "opus"
            | "aac"
            | "m4a"
            | "mp4"
            | "webm"
            | "mkv"
            | "aif"
            | "aiff"
            | "caf"
    )
}

fn canonical_filename(filename: Option<&str>) -> Option<String> {
    let filename = filename?;
    let stem = Path::new(filename)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .map(str::trim)
        .filter(|stem| !stem.is_empty())?;
    Some(format!(
        "{}.canonical.wav",
        stem.chars().take(120).collect::<String>()
    ))
}

fn canonical_derivative_record_id(
    source_storage_key: &str,
    profile: CanonicalAudioProfile,
) -> String {
    let identity = format!("{source_storage_key}\0{}", profile.id());
    let digest = sha256_hex(identity.as_bytes());
    format!("canonical-{}", &digest[..32])
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc, Mutex,
        },
    };

    use izwi_hooks::{
        HookError, HookResult, MediaDeleteRequest, MediaObjectKey, MediaObjectMetadata,
        MediaReadRequest, MediaStorageProvider, MediaWriteRequest, StoredMediaBytes,
        StoredMediaObject,
    };
    use sea_orm::ConnectionTrait;

    use super::*;
    use crate::db::StoreDatabase;

    #[derive(Default)]
    struct RecordingMediaProvider {
        objects: Mutex<HashMap<String, Vec<u8>>>,
        deleted: Mutex<Vec<String>>,
        next_key: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl MediaStorageProvider for RecordingMediaProvider {
        async fn put(
            &self,
            request: MediaWriteRequest,
            bytes: Vec<u8>,
        ) -> HookResult<StoredMediaObject> {
            let suffix = self.next_key.fetch_add(1, Ordering::AcqRel) + 1;
            let key = format!("test/{}/{suffix}", request.record_id);
            self.objects
                .lock()
                .expect("objects")
                .insert(key.clone(), bytes.clone());
            Ok(StoredMediaObject {
                key: MediaObjectKey::new(key),
                metadata: MediaObjectMetadata {
                    content_type: request.content_type,
                    filename: request.preferred_filename,
                    content_length: Some(bytes.len() as u64),
                    sha256: None,
                    tenant_id: None,
                    attributes: request.metadata,
                },
            })
        }

        async fn get(&self, request: MediaReadRequest) -> HookResult<StoredMediaBytes> {
            let objects = self.objects.lock().expect("objects");
            let bytes = objects
                .get(&request.key.key)
                .cloned()
                .ok_or_else(|| HookError::NotFound(request.key.key.clone()))?;
            Ok(StoredMediaBytes {
                metadata: MediaObjectMetadata {
                    content_type: "application/octet-stream".to_string(),
                    filename: None,
                    content_length: Some(bytes.len() as u64),
                    sha256: None,
                    tenant_id: None,
                    attributes: request.metadata,
                },
                bytes,
            })
        }

        async fn delete(&self, request: MediaDeleteRequest) -> HookResult<()> {
            self.objects
                .lock()
                .expect("objects")
                .remove(&request.key.key);
            self.deleted.lock().expect("deleted").push(request.key.key);
            Ok(())
        }
    }

    fn stereo_wav_bytes() -> Vec<u8> {
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 48_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut bytes = Vec::new();
        {
            let mut writer =
                hound::WavWriter::new(std::io::Cursor::new(&mut bytes), spec).expect("writer");
            for frame in 0..4_800 {
                let sample = if frame % 2 == 0 {
                    8_000_i16
                } else {
                    -8_000_i16
                };
                writer.write_sample(sample).expect("left");
                writer.write_sample(sample).expect("right");
            }
            writer.finalize().expect("finalize");
        }
        bytes
    }

    #[test]
    fn decode_lane_snapshot_reports_bounded_capacity() {
        let temp = tempfile::tempdir().expect("tempdir");
        let provider = Arc::new(RecordingMediaProvider::default());
        let store = Arc::new(BatchRuntimeStore::initialize_with_database(
            StoreDatabase::new(temp.path().join("runtime.sqlite")),
        ));
        let service = MediaIngestService::with_decode_lane_capacity(provider, store, 3);

        assert_eq!(
            service.lane_snapshot(),
            MediaIngestLaneSnapshot {
                capacity: 3,
                active: 0,
                available: 3,
                queued: 0,
            }
        );
    }

    fn request(bytes: Vec<u8>) -> MediaIngestRequest {
        MediaIngestRequest {
            bytes,
            content_type: "application/octet-stream".to_string(),
            filename: Some("recording.bin".to_string()),
            namespace: "test".to_string(),
            record_id: "record-1".to_string(),
            route: "/v1/media".to_string(),
        }
    }

    fn source_metadata(container: &str, codec: &str) -> AudioSourceMetadata {
        AudioSourceMetadata {
            container: container.to_string(),
            codec: codec.to_string(),
            sample_rate: 16_000,
            channel_count: 1,
        }
    }

    #[test]
    fn policy_rejects_disallowed_container_and_codec() {
        let inspection = AudioInspection::from_mono_samples(&[0.0, 0.1], 16_000);
        let mut policy = AudioIngestPolicy::media_upload(1_000);
        policy.allowed_containers = vec!["wav".to_string()];
        policy.allowed_codecs = vec!["pcm_*".to_string()];

        let container_error = policy
            .validate_source(32, &source_metadata("mp3", "mp3"), &inspection)
            .expect_err("MP3 container should be rejected");
        assert!(container_error.to_string().contains("container `mp3`"));

        let codec_error = policy
            .validate_source(32, &source_metadata("wav", "flac"), &inspection)
            .expect_err("FLAC codec should be rejected in WAV-only policy");
        assert!(codec_error.to_string().contains("codec `flac`"));
    }

    #[test]
    fn policy_rejects_excess_decoded_samples() {
        let inspection = AudioInspection::from_mono_samples(&[0.0, 0.1, 0.2], 16_000);
        let mut policy = AudioIngestPolicy::media_upload(1_000);
        policy.max_decoded_samples = 2;

        let error = policy
            .validate_source(32, &source_metadata("wav", "pcm_s16le"), &inspection)
            .expect_err("decoded sample limit should be enforced");

        assert!(error.to_string().contains("sample count"));
    }

    #[test]
    fn media_upload_policy_fixes_strict_corruption_rejection() {
        let policy = AudioIngestPolicy::media_upload(1_000);

        assert_eq!(policy.corruption_policy, AudioCorruptionPolicy::Reject);
        assert!(policy.max_decoded_samples > 0);
        assert!(policy.max_duration_secs.is_some());
    }

    #[tokio::test]
    async fn byte_sniffing_canonicalizes_audio_with_source_truth() {
        let temp = tempfile::tempdir().expect("tempdir");
        let provider = Arc::new(RecordingMediaProvider::default());
        let store = Arc::new(BatchRuntimeStore::initialize_with_database(
            StoreDatabase::new(temp.path().join("runtime.sqlite")),
        ));
        let service = MediaIngestService::with_decode_lane_capacity(provider.clone(), store, 1);
        let source_bytes = stereo_wav_bytes();

        let result = service
            .ingest(
                request(source_bytes.clone()),
                AudioIngestPolicy::media_upload(1_000_000),
                CanonicalAudioProfile::SpeechRecognition16KhzMonoWav,
            )
            .await
            .expect("ingest");

        let source_asset = result.source_asset.expect("source asset");
        assert_eq!(source_asset.content_type, "application/octet-stream");
        assert_eq!(source_asset.sample_rate_hz, Some(48_000));
        assert_eq!(source_asset.channel_count, Some(2));
        assert_eq!(source_asset.source_asset_id, None);
        assert_eq!(source_asset.canonical_profile_version, None);
        assert_eq!(source_asset.scan_status, SECURITY_SCAN_STATUS_NOT_SCANNED);
        assert_eq!(
            source_asset.metadata_json["decode_validation"]["status"],
            "passed"
        );
        assert_eq!(source_asset.metadata_json["source"]["container"], "wav");
        let canonical = result.canonical.expect("canonical asset");
        assert_eq!(canonical.asset.sample_rate_hz, Some(16_000));
        assert_eq!(canonical.asset.channel_count, Some(1));
        assert_eq!(
            canonical.asset.source_asset_id.as_deref(),
            Some(source_asset.id.as_str())
        );
        assert_eq!(
            canonical.asset.canonical_profile_version.as_deref(),
            Some(CanonicalAudioProfile::SpeechRecognition16KhzMonoWav.id())
        );
        assert_eq!(
            canonical.asset.metadata_json["canonical_profile"],
            CanonicalAudioProfile::SpeechRecognition16KhzMonoWav.id()
        );
        let objects = provider.objects.lock().expect("objects");
        assert_eq!(
            objects.get(&result.original_storage_key),
            Some(&source_bytes)
        );
        assert!(objects.get(&canonical.storage_key).is_some());
    }

    #[tokio::test]
    async fn existing_audio_reuses_original_storage_key_and_writes_only_derivative() {
        let temp = tempfile::tempdir().expect("tempdir");
        let provider = Arc::new(RecordingMediaProvider::default());
        let store = Arc::new(BatchRuntimeStore::initialize_with_database(
            StoreDatabase::new(temp.path().join("runtime.sqlite")),
        ));
        let service = MediaIngestService::with_decode_lane_capacity(provider.clone(), store, 1);
        let source_bytes = stereo_wav_bytes();

        let result = service
            .ingest_existing_audio(
                ExistingAudioIngestRequest {
                    bytes: source_bytes.clone(),
                    storage_key: "transcription/original.wav".to_string(),
                    storage_namespace: "transcription_upload".to_string(),
                    content_type: "audio/wav".to_string(),
                    filename: Some("original.wav".to_string()),
                    record_id: "record-1".to_string(),
                    route: "speech_to_text".to_string(),
                },
                AudioIngestPolicy::media_upload(1_000_000),
                CanonicalAudioProfile::SpeechRecognition16KhzMonoWav,
            )
            .await
            .expect("existing ingest");

        assert_eq!(
            result.source_asset.expect("source asset").storage_key,
            "transcription/original.wav"
        );
        assert_eq!(provider.objects.lock().expect("objects").len(), 1);
        assert!(result.canonical.is_some());
    }

    #[tokio::test]
    async fn concurrent_existing_audio_ingest_reuses_one_source_profile_derivative() {
        let temp = tempfile::tempdir().expect("tempdir");
        let provider = Arc::new(RecordingMediaProvider::default());
        let store = Arc::new(BatchRuntimeStore::initialize_with_database(
            StoreDatabase::new(temp.path().join("runtime.sqlite")),
        ));
        let service = MediaIngestService::with_decode_lane_capacity(provider.clone(), store, 2);
        let source_bytes = stereo_wav_bytes();
        let ingest = |record_id: &str| ExistingAudioIngestRequest {
            bytes: source_bytes.clone(),
            storage_key: "saved_voice/immutable-reference.wav".to_string(),
            storage_namespace: "saved_voice".to_string(),
            content_type: "audio/wav".to_string(),
            filename: Some("reference.wav".to_string()),
            record_id: record_id.to_string(),
            route: "text_to_speech".to_string(),
        };

        let (first, second) = tokio::join!(
            service.ingest_existing_audio(
                ingest("record-1"),
                AudioIngestPolicy::media_upload(1_000_000),
                CanonicalAudioProfile::ReferenceVoiceSourceRateMonoWav,
            ),
            service.ingest_existing_audio(
                ingest("record-2"),
                AudioIngestPolicy::media_upload(1_000_000),
                CanonicalAudioProfile::ReferenceVoiceSourceRateMonoWav,
            )
        );
        let first = first.expect("first ingest");
        let second = second.expect("second ingest");
        let first_source = first.source_asset.expect("first source");
        let second_source = second.source_asset.expect("second source");
        let first_canonical = first.canonical.expect("first canonical");
        let second_canonical = second.canonical.expect("second canonical");

        assert_eq!(first_source.id, second_source.id);
        assert_eq!(first_canonical.asset.id, second_canonical.asset.id);
        assert_eq!(first_canonical.storage_key, second_canonical.storage_key);
        assert_eq!(
            first_canonical.asset.source_asset_id.as_deref(),
            Some(first_source.id.as_str())
        );
        assert_eq!(
            first_canonical.asset.canonical_profile_version.as_deref(),
            Some(CanonicalAudioProfile::ReferenceVoiceSourceRateMonoWav.id())
        );
        assert_eq!(provider.objects.lock().expect("objects").len(), 1);
    }

    #[tokio::test]
    async fn strict_audio_ingest_rejects_opaque_bytes_before_storage() {
        let temp = tempfile::tempdir().expect("tempdir");
        let provider = Arc::new(RecordingMediaProvider::default());
        let store = Arc::new(BatchRuntimeStore::initialize_with_database(
            StoreDatabase::new(temp.path().join("runtime.sqlite")),
        ));
        let service = MediaIngestService::with_decode_lane_capacity(provider.clone(), store, 1);

        let error = service
            .ingest_audio(
                request(b"this is not audio".to_vec()),
                AudioIngestPolicy::media_upload(1_000_000),
                CanonicalAudioProfile::ReferenceVoiceSourceRateMonoWav,
            )
            .await
            .expect_err("opaque bytes must not pass an audio-only boundary");

        assert!(error.is_invalid_input());
        assert!(provider.objects.lock().expect("objects").is_empty());
    }

    #[tokio::test]
    async fn reference_voice_canonicalization_preserves_source_rate_and_downmixes() {
        let temp = tempfile::tempdir().expect("tempdir");
        let provider = Arc::new(RecordingMediaProvider::default());
        let store = Arc::new(BatchRuntimeStore::initialize_with_database(
            StoreDatabase::new(temp.path().join("runtime.sqlite")),
        ));
        let service = MediaIngestService::with_decode_lane_capacity(provider, store, 1);

        let result = service
            .ingest(
                MediaIngestRequest {
                    bytes: stereo_wav_bytes(),
                    content_type: "audio/wav".to_string(),
                    filename: Some("reference.wav".to_string()),
                    namespace: "tts_reference".to_string(),
                    record_id: "record-1".to_string(),
                    route: "text_to_speech".to_string(),
                },
                AudioIngestPolicy::media_upload(1_000_000),
                CanonicalAudioProfile::ReferenceVoiceSourceRateMonoWav,
            )
            .await
            .expect("reference ingest");

        let canonical = result.canonical.expect("canonical reference").asset;
        assert_eq!(canonical.sample_rate_hz, Some(48_000));
        assert_eq!(canonical.channel_count, Some(1));
        assert_eq!(
            canonical.metadata_json["canonical_profile"],
            CanonicalAudioProfile::ReferenceVoiceSourceRateMonoWav.id()
        );
    }

    #[tokio::test]
    async fn registration_failure_compensates_both_unregistered_objects() {
        let temp = tempfile::tempdir().expect("tempdir");
        let provider = Arc::new(RecordingMediaProvider::default());
        let connection = crate::db::sqlite::connect_path(&temp.path().join("broken.sqlite"))
            .await
            .expect("database connection");
        connection
            .execute_unprepared("CREATE TABLE media_assets (id TEXT PRIMARY KEY)")
            .await
            .expect("incomplete media asset schema");
        let store = Arc::new(BatchRuntimeStore::initialize_with_database(
            StoreDatabase::from_connection(connection),
        ));
        let service = MediaIngestService::with_decode_lane_capacity(provider.clone(), store, 1);

        let error = service
            .ingest(
                request(stereo_wav_bytes()),
                AudioIngestPolicy::media_upload(1_000_000),
                CanonicalAudioProfile::SpeechRecognition16KhzMonoWav,
            )
            .await
            .expect_err("asset registration should fail");

        assert!(matches!(error, MediaIngestError::AssetRegistration(_)));
        assert!(provider.objects.lock().expect("objects").is_empty());
        assert_eq!(provider.deleted.lock().expect("deleted").len(), 2);
    }
}
