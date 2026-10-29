//! Diarization runtime methods.

use crate::catalog::{resolve_diarization_asr_model_variant, resolve_diarization_model_variant};
use crate::engine::{ResourceAmount, ResourceVector, WorkUnit};
use crate::error::{Error, Result};
use crate::models::architectures::sortformer::diarization::{
    production_workspace_authorization, sortformer_uses_selected_model_device, SortformerStream,
    SortformerWorkspaceEstimate, SortformerWorkspaceEvent,
};
use crate::models::architectures::voxtral::realtime::VOXTRAL_REALTIME_OFFLINE_AUDIO_SECONDS_HINT;
use crate::models::registry::{
    NativeAsrGenerationOptions, NativeAsrModel, NativeAsrTranscription, VoxtralModelLease,
};
use crate::runtime::adapters::{CapabilityKind, ExecutionTargetKind};
use crate::runtime::audio_io::{
    base64_decode, decode_audio_bytes_canonical, validate_base64_audio_retained_size,
    MAX_AUDIO_SOURCE_BYTES,
};
use crate::runtime::coordinator::{
    wait_for_held_capacity, InferenceCoordinator, JobLease, JobResourceObservation,
};
use crate::runtime::forced_alignment::{AlignedWindow, AlignmentSource, SampleWindow};
use crate::runtime::service::{
    copy_preparation_bytes, copy_preparation_string, CapabilityLoadOptions, RuntimeService,
};
use crate::runtime::types::{
    DiarizationConfig, DiarizationPipelineOptions, DiarizationResult, DiarizationSegment,
    DiarizationTranscriptResult, DiarizationUtterance, DiarizationWord,
};
use crate::runtime::{CoordinatorLane, RuntimeRequestContext};
use crate::ModelVariant;
use izwi_asr_toolkit::{
    chunk_seam_secs, plan_audio_chunks, spacing_between, AsrLongFormConfig, AudioChunk, TimedWord,
};

mod one_model;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tracing::warn;

const UNKNOWN_SPEAKER: &str = "UNKNOWN";
const MAX_UTTERANCE_GAP_SECS: f32 = 0.9;
const MIN_ALIGNMENT_COVERAGE: f32 = 0.25;
const MAX_UNATTRIBUTED_WORD_RATIO: f32 = 0.4;
const ALIGNMENT_COLLAPSE_TAIL_MS: u32 = 250;
const ALIGNMENT_PREFIX_CLUSTER_MS_CAP: u32 = 1_000;
const MAX_REASONABLE_WORD_SPAN_MS: u32 = 2_500;
const PIPELINE_SAMPLE_RATE: u32 = 16_000;
const MAX_PIPELINE_DURATION_SECONDS: usize = 60 * 60;
const MAX_PIPELINE_SAMPLES: usize = PIPELINE_SAMPLE_RATE as usize * MAX_PIPELINE_DURATION_SECONDS;
// The shared decoder permits up to 256 MiB of mono output. During pipeline
// normalization both that output and up to one hour of 16 kHz f32 audio may be
// live, in addition to decoder/source copies. This floor is authorized before
// any of those allocations occur; the generic input-shaped reservation adds
// another 64 MiB plus eight times the retained source size.
const DIARIZATION_AUDIO_WORKSPACE_BYTES: u64 = 512 * 1024 * 1024;
// Encoded audio length cannot safely bound decoded duration. The model module
// derives a complete peak ceiling from the supported preprocessing,
// Conv-subsampling, Conformer, and transformer topology. The loaded checkpoint
// is checked against that immutable envelope before inference begins.
const MAX_FRAGMENT_WORDS: usize = 3;
const MAX_FRAGMENT_DURATION_SECS: f32 = 1.25;
const FRAGMENT_CONFIDENCE_MARGIN: f32 = 0.08;
const PHRASE_CAPITALIZED_GAP_SECS: f32 = 0.35;
const PHRASE_HARD_GAP_SECS: f32 = 0.75;
/// Audio one Sortformer stage covers (PIPE-8): a 1 h file runs in about 40
/// stages, each a few seconds of compute, so work waiting on the same permit
/// or lane gets in between them.
const SORTFORMER_STAGE_AUDIO_SECS: f32 = 90.0;

/// A diarization model loaded for one job: its residency, contract, state
/// and weights.
struct LoadedDiarizer {
    residency_lease: crate::model::ModelResidencyLease,
    execution_contract: crate::runtime::adapters::LoadedExecutionContract,
    state_binding: crate::runtime::adapters::LoadedCapabilityBinding,
    model: Arc<crate::models::registry::NativeDiarizationModel>,
}

#[derive(Debug, Clone)]
struct PipelineAudio {
    samples: Arc<[f32]>,
    sample_rate: u32,
    duration_secs: f32,
}

impl PipelineAudio {
    /// Canonical 16 kHz mono samples, within the one-hour bound.
    fn from_canonical(samples: Vec<f32>) -> Result<Self> {
        validate_pipeline_sample_count(samples.len())?;
        Ok(Self {
            duration_secs: samples.len() as f32 / PIPELINE_SAMPLE_RATE as f32,
            samples: samples.into(),
            sample_rate: PIPELINE_SAMPLE_RATE,
        })
    }
}

fn diarization_input_observation(
    retained_input_bytes: usize,
    retained_audio_copies: &[usize],
    audio: &PipelineAudio,
) -> Result<JobResourceObservation> {
    let sample_bytes = audio
        .samples
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| Error::Overloaded("diarization sample storage overflowed".to_string()))?;
    let retained_audio_bytes = retained_audio_copies
        .iter()
        .try_fold(0usize, |total, bytes| {
            total.checked_add(*bytes).ok_or_else(|| {
                Error::Overloaded("diarization audio storage overflowed".to_string())
            })
        })?;
    let host_bytes = retained_input_bytes
        .checked_add(retained_audio_bytes)
        .and_then(|total| total.checked_add(sample_bytes))
        .ok_or_else(|| Error::Overloaded("diarization input storage overflowed".to_string()))?;
    Ok(JobResourceObservation::host(
        u64::try_from(host_bytes)
            .map_err(|_| Error::Overloaded("diarization input exceeds u64".to_string()))?,
    ))
}

/// The job's audio allowance, authorized before decode: the decoder output
/// and up to one hour of 16 kHz samples, all host-side (unified on Metal).
/// The Sortformer workspace is not in the job; each diarize stage reserves
/// the loaded model's own estimate for the audio it actually decoded.
fn diarization_decode_resources(backend: crate::backends::BackendKind) -> ResourceVector {
    let mut resources = ResourceVector::zero();
    match backend {
        crate::backends::BackendKind::Cpu | crate::backends::BackendKind::Cuda => {
            resources.host_bytes = ResourceAmount::Known(DIARIZATION_AUDIO_WORKSPACE_BYTES)
        }
        crate::backends::BackendKind::Metal => {
            resources.unified_bytes = ResourceAmount::Known(DIARIZATION_AUDIO_WORKSPACE_BYTES)
        }
    }
    resources
}

/// What a diarization job keeps authorized once its audio is decoded: the
/// retained input and decoded samples it has materialized, plus the generic
/// base workspace every request keeps. The decode allowance is given back,
/// so it is not pending through the diarizer, ASR, aligner and LLM loads.
fn diarization_retained_resources(
    backend: crate::backends::BackendKind,
    retained: JobResourceObservation,
) -> Result<ResourceVector> {
    let shared_bytes = || {
        retained
            .host_bytes
            .checked_add(retained.accelerator_bytes)
            .ok_or_else(|| Error::Overloaded("diarization retained audio overflowed".to_string()))
    };
    let mut resources = ResourceVector::zero();
    match backend {
        crate::backends::BackendKind::Cpu => {
            resources.host_bytes = ResourceAmount::Known(shared_bytes()?)
        }
        crate::backends::BackendKind::Metal => {
            resources.unified_bytes = ResourceAmount::Known(shared_bytes()?)
        }
        crate::backends::BackendKind::Cuda => {
            resources.host_bytes = ResourceAmount::Known(retained.host_bytes);
            resources.device_bytes = ResourceAmount::Known(retained.accelerator_bytes);
        }
    }
    crate::runtime::service::transient_resources(backend, 0).checked_add(resources)
}

/// Settle a diarization job once its audio is decoded: record the decode
/// peak, restore the pending claim for the source copy about to be freed,
/// free it, then give back the decode allowance.
fn settle_decoded_job(
    job: &JobLease,
    backend: crate::backends::BackendKind,
    peak_usage: JobResourceObservation,
    steady_usage: JobResourceObservation,
    retained_source: Option<String>,
) -> Result<()> {
    job.record_materialized_usage(peak_usage)?;
    job.prepare_materialized_release(steady_usage)?;
    drop(retained_source);
    job.shrink_authorization(diarization_retained_resources(backend, steady_usage)?)
}

/// Run one diarize body under a stage-scoped lease on the loaded model's own
/// workspace estimate. The lease is consumed: it is released when this
/// returns, whether the body succeeds or fails, so no workspace claim stays
/// pending after the stage.
fn diarize_under_workspace_lease<T>(
    coordinator: &crate::runtime::coordinator::InferenceCoordinator,
    job: &JobLease,
    workspace: SortformerWorkspaceEstimate,
    body: impl FnOnce(&mut dyn FnMut(SortformerWorkspaceEvent) -> Result<()>) -> Result<T>,
) -> Result<T> {
    let observation =
        JobResourceObservation::new(workspace.host_bytes, workspace.accelerator_bytes);
    let lease = coordinator.reserve_stage_workspace(job, "diarization.infer", observation)?;
    let mut observer = |event| match event {
        SortformerWorkspaceEvent::Materialized {
            workspace: observed,
        } => {
            if observed != workspace {
                return Err(Error::InferenceError(format!(
                    "Sortformer materialized workspace {observed:?} after estimating {workspace:?}"
                )));
            }
            lease.record_materialized_usage(observation)
        }
        SortformerWorkspaceEvent::Releasing {
            workspace: observed,
        } => {
            if observed != workspace {
                return Err(Error::InferenceError(format!(
                    "Sortformer released workspace {observed:?} after estimating {workspace:?}"
                )));
            }
            lease.prepare_materialized_release(JobResourceObservation::default())
        }
    };
    body(&mut observer)
}

/// Run `load`; if it fails while the overlapped diarizer still runs (it may
/// hold the memory the load needs), wait for the diarizer to finish and free
/// its model, then run `load` once more (PIPE-15).
async fn load_after_diarizer_on_failure<T, F, Fut>(
    diarized: &mut tokio::sync::watch::Receiver<bool>,
    mut load: F,
) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    let diarizing = !*diarized.borrow();
    match load().await {
        Err(error) if diarizing => {
            warn!("load failed while diarizing, retrying after the diarizer: {error}");
            if diarized.wait_for(|done| *done).await.is_err() {
                return Err(error);
            }
            load().await
        }
        result => result,
    }
}

fn lock_stream(
    stream: &Mutex<Option<SortformerStream>>,
) -> Result<std::sync::MutexGuard<'_, Option<SortformerStream>>> {
    stream
        .lock()
        .map_err(|_| Error::InferenceError("Sortformer stream lock poisoned".into()))
}

/// Run `step` as one loaded stage at a time until it reports the work
/// finished (PIPE-8). Between stages the job holds no permit, so other work
/// gets the device in between. A `host_model` (weights and compute in host
/// memory) takes the coordinator's host-model lane instead of the device
/// permit on an accelerator. A stage refused memory other work holds is
/// retried as in [`wait_for_held_capacity`]; `step` must leave its work
/// unchanged when it fails before running.
async fn run_bounded_stages<F>(
    coordinator: &Arc<InferenceCoordinator>,
    job: &JobLease,
    contract: &crate::runtime::adapters::LoadedExecutionContract,
    work: WorkUnit,
    host_model: bool,
    step: F,
) -> Result<()>
where
    F: Fn() -> Result<bool> + Clone + Send + 'static,
{
    loop {
        let finished = wait_for_held_capacity(pipeline_capacity_wait(), || {
            let step = step.clone();
            let (contract, work) = (contract.clone(), work.clone());
            async move {
                if host_model {
                    coordinator
                        .run_loaded_host_model_stage(job, contract, work, step)
                        .await
                } else {
                    coordinator
                        .run_loaded_blocking_stage(job, contract, work, step)
                        .await
                }
            }
        })
        .await?;
        if finished {
            return Ok(());
        }
    }
}

#[derive(Debug, Clone)]
struct TranscribedChunk {
    range: AudioChunk,
    text: String,
    language: Option<String>,
    /// The decode stopped at its token budget, so the text may be cut short.
    hit_token_budget: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DiarizationAsrRegistryRoute {
    Native,
    Voxtral,
    AudioChat,
}

fn diarization_asr_registry_route(variant: ModelVariant) -> DiarizationAsrRegistryRoute {
    if variant.family() == crate::catalog::ModelFamily::Voxtral {
        DiarizationAsrRegistryRoute::Voxtral
    } else if variant.is_audio_chat() {
        DiarizationAsrRegistryRoute::AudioChat
    } else {
        DiarizationAsrRegistryRoute::Native
    }
}

/// LFM3-Audio transcribes clips of at most 30 s, the length it was trained
/// on, as 12.5 encoder tokens per second.
const LFM3_AUDIO_ASR_SECONDS: f32 = 30.0;
const LFM3_AUDIO_ASR_TOKEN_RATE: f32 = 12.5;

#[derive(Clone)]
enum DiarizationAsrModel {
    Native(Arc<NativeAsrModel>),
    Voxtral(VoxtralModelLease),
    /// An audio-chat model that transcribes on its invocation state
    /// (LFM3-Audio).
    AudioChat(Arc<crate::models::registry::NativeAudioChatModel>),
}

impl DiarizationAsrModel {
    fn max_audio_seconds_hint(&self) -> Option<f32> {
        match self {
            Self::Native(model) => model.max_audio_seconds_hint(),
            // Chunk Voxtral within its offline frame cap; a longer single
            // pass fails on CPU/Metal instead of truncating.
            Self::Voxtral(_) => Some(VOXTRAL_REALTIME_OFFLINE_AUDIO_SECONDS_HINT),
            Self::AudioChat(_) => Some(LFM3_AUDIO_ASR_SECONDS),
        }
    }

    /// Decoder-prompt tokens per second of audio; Voxtral's geometry is not
    /// exposed here, so its pipeline chunking stays hint-driven.
    fn audio_token_rate(&self) -> Option<f32> {
        match self {
            Self::Native(model) => model.audio_token_rate(),
            Self::Voxtral(_) => None,
            Self::AudioChat(_) => Some(LFM3_AUDIO_ASR_TOKEN_RATE),
        }
    }

    fn transcribe_from_invocation_workspace(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        prompt: Option<&str>,
        options: NativeAsrGenerationOptions,
        leases: &mut crate::kv::v2::InvocationWorkspaceLeaseSetV2,
    ) -> Result<NativeAsrTranscription> {
        match self {
            Self::Native(model) => model
                .transcribe_with_details_and_prompt_and_options_from_invocation_workspace(
                    audio,
                    sample_rate,
                    language,
                    prompt,
                    options,
                    leases,
                ),
            Self::Voxtral(model) => {
                let cache = leases
                    .lease_exact_kind_mut(
                        crate::kv::v2::InvocationStateBackingKindV2::PagedAttention,
                    )?
                    .paged_cache_mut()?;
                let output =
                    model.transcribe_with_details_physical(audio, sample_rate, language, cache)?;
                Ok(NativeAsrTranscription {
                    text: output.text,
                    language: output.language,
                    diagnostics: output.diagnostics,
                })
            }
            // The pipeline's chunks fit one 30 s ASR window; `prompt` is
            // not read by LFM3-Audio.
            Self::AudioChat(model) => model
                .transcribe_with_callback_and_max_tokens_from_invocation_workspace(
                    audio,
                    sample_rate,
                    language,
                    Some(options.max_new_tokens),
                    leases,
                    &mut |_| {},
                ),
        }
    }
}

#[derive(Clone, Copy)]
enum DiarizationAudioInput<'a> {
    Base64(&'a str),
    Bytes(&'a [u8]),
}

/// The audio of one transcript-pipeline job: an encoded upload, or its
/// canonical 16 kHz mono samples decoded by the caller (PIPE-14).
enum TranscriptAudioInput<'a> {
    Encoded(DiarizationAudioInput<'a>),
    Samples(Vec<f32>),
}

impl TranscriptAudioInput<'_> {
    fn validate(&self) -> Result<()> {
        match self {
            Self::Encoded(input) => {
                if input.input_bytes() == 0 {
                    return Err(Error::InvalidInput(
                        "diarization request missing audio input".to_string(),
                    ));
                }
                input.validate_retained_size()
            }
            Self::Samples(samples) => validate_pipeline_sample_count(samples.len()),
        }
    }
}

enum OwnedDiarizationAudioInput {
    Base64(String),
    Bytes(Vec<u8>),
}

impl DiarizationAudioInput<'_> {
    fn input_bytes(self) -> usize {
        match self {
            Self::Base64(audio) => audio.len(),
            Self::Bytes(audio) => audio.len(),
        }
    }

    fn validate_retained_size(self) -> Result<()> {
        match self {
            Self::Base64(audio) => {
                validate_base64_audio_retained_size(audio.len(), MAX_AUDIO_SOURCE_BYTES)
            }
            Self::Bytes([]) => Err(Error::InvalidInput(
                "diarization request missing audio bytes".to_string(),
            )),
            Self::Bytes(audio) if audio.len() > MAX_AUDIO_SOURCE_BYTES => {
                Err(Error::InvalidInput(format!(
                    "diarization encoded audio is {} bytes, exceeding the {MAX_AUDIO_SOURCE_BYTES}-byte source limit",
                    audio.len()
                )))
            }
            Self::Bytes(_) => Ok(()),
        }
    }

    async fn into_owned_for_job(self, job: &JobLease) -> Result<OwnedDiarizationAudioInput> {
        match self {
            Self::Base64(audio) => Ok(OwnedDiarizationAudioInput::Base64(
                copy_preparation_string(job, audio, "diarization base64 audio").await?,
            )),
            Self::Bytes(audio) => Ok(OwnedDiarizationAudioInput::Bytes(
                copy_preparation_bytes(job, audio, "diarization encoded audio").await?,
            )),
        }
    }
}

impl OwnedDiarizationAudioInput {
    fn retained_bytes(&self) -> usize {
        match self {
            Self::Base64(audio) => audio.capacity(),
            Self::Bytes(audio) => audio.capacity(),
        }
    }

    fn decode_bytes(self) -> Result<(Vec<u8>, Option<String>)> {
        match self {
            Self::Base64(audio) => Ok((base64_decode(&audio)?, Some(audio))),
            Self::Bytes(audio) => Ok((audio, None)),
        }
    }
}

impl RuntimeService {
    async fn load_diarizer(&self, job: &JobLease, variant: ModelVariant) -> Result<LoadedDiarizer> {
        let (residency_lease, execution_contract, state_binding) = self
            .load_capability_with_state_for_job(
                job,
                variant,
                CapabilityKind::Diarization,
                false,
                ExecutionTargetKind::PipelineRunner,
                CapabilityLoadOptions::default(),
            )
            .await?;
        let model = self
            .model_registry
            .get_diarization(variant)
            .await
            .ok_or_else(|| Error::ModelNotFound(variant.to_string()))?;
        Ok(LoadedDiarizer {
            residency_lease,
            execution_contract,
            state_binding,
            model,
        })
    }

    /// Diarize `audio` with a loaded diarizer, which is released (and its
    /// model evictable) when this returns.
    async fn run_diarizer(
        &self,
        job: &JobLease,
        diarizer: LoadedDiarizer,
        audio: PipelineAudio,
        config: DiarizationConfig,
    ) -> Result<DiarizationResult> {
        let LoadedDiarizer {
            residency_lease,
            execution_contract,
            state_binding,
            model,
        } = diarizer;
        let workspace = model.workspace_estimate(audio.samples.len())?;
        let authorized_workspace = production_workspace_authorization(
            MAX_PIPELINE_SAMPLES,
            matches!(
                self.backend_context().backend_kind,
                crate::backends::BackendKind::Cuda
            ),
        )?;
        if workspace.host_bytes > authorized_workspace.host_bytes
            || workspace.accelerator_bytes > authorized_workspace.accelerator_bytes
        {
            tracing::warn!(
                "Sortformer workspace {workspace:?} exceeds production authorization {authorized_workspace:?}"
            );
            let (need, limit) = if workspace.host_bytes > authorized_workspace.host_bytes {
                (workspace.host_bytes, authorized_workspace.host_bytes)
            } else {
                (
                    workspace.accelerator_bytes,
                    authorized_workspace.accelerator_bytes,
                )
            };
            return Err(Error::Overloaded(format!(
                "diarization workspace needs {:.2} GB, above the {:.2} GB one request may use",
                need as f64 / 1e9,
                limit as f64 / 1e9
            )));
        }
        // The file runs as bounded stages, each reserving the workspace when
        // it starts and releasing it when it returns. A stage refused memory
        // other work holds gives its permit back and runs again once that
        // memory frees, instead of failing the job (P1V-2). On Metal the model
        // runs on the CPU: its steps take the host-model lane, not the device
        // permit, so chat, TTS and ASR on the GPU are not queued behind the
        // file (PIPE-8, D4). Only the final commit to the invocation state
        // touches the device.
        let host_model =
            !sortformer_uses_selected_model_device(self.backend_context().backend_kind);
        let stream = model.begin_stream(&audio.samples, &config, SORTFORMER_STAGE_AUDIO_SECS)?;
        let finished = stream.is_finished();
        let stream = Arc::new(Mutex::new(Some(stream)));
        let residency_lease = Arc::new(residency_lease);
        let step = {
            let coordinator = self.coordinator.clone();
            let job = job.clone();
            let model = model.clone();
            let audio = audio.clone();
            let stream = stream.clone();
            let residency_lease = residency_lease.clone();
            move || {
                let _residency_lease = &residency_lease;
                diarize_under_workspace_lease(&coordinator, &job, workspace, |observer| {
                    let mut stream = lock_stream(&stream)?;
                    let stream = stream.as_mut().ok_or_else(|| {
                        Error::InferenceError("Sortformer stream already finished".into())
                    })?;
                    model.advance_stream(stream, &audio.samples, observer)?;
                    Ok(stream.is_finished())
                })
            }
        };
        if !finished {
            run_bounded_stages(
                &self.coordinator,
                job,
                &execution_contract,
                WorkUnit::PipelineStage {
                    name: "diarization.infer".to_string(),
                    ordinal: 0,
                },
                host_model,
                step,
            )
            .await?;
        }
        wait_for_held_capacity(pipeline_capacity_wait(), || {
            let residency_lease = residency_lease.clone();
            let coordinator = self.coordinator.clone();
            let stage_job = job.clone();
            let model = model.clone();
            let audio = audio.clone();
            let config = config.clone();
            let stream = stream.clone();
            self.coordinator
                .run_loaded_blocking_stage_with_invocation_workspace(
                    job,
                    execution_contract.clone(),
                    state_binding.clone(),
                    WorkUnit::PipelineStage {
                        name: "diarization.infer".to_string(),
                        ordinal: 0,
                    },
                    move |leases| {
                        let _residency_lease = residency_lease;
                        let state = leases
                            .lease_exact_kind_mut(
                                crate::kv::v2::InvocationStateBackingKindV2::Tensor,
                            )?
                            .typed_mut::<crate::engine::InvocationTensorLease>()?;
                        diarize_under_workspace_lease(
                            &coordinator,
                            &stage_job,
                            workspace,
                            |observer| {
                                let stream = lock_stream(&stream)?.take().ok_or_else(|| {
                                    Error::InferenceError(
                                        "Sortformer stream already finished".into(),
                                    )
                                })?;
                                model.finish_stream_physical(
                                    stream,
                                    &audio.samples,
                                    &config,
                                    state,
                                    observer,
                                )
                            },
                        )
                    },
                )
        })
        .await
    }

    /// Admit one transcript-pipeline job with its audio as 16 kHz samples.
    /// Encoded input is decoded once, in an admitted host stage under the
    /// decode allowance, and its encoded bytes are dropped once decoded;
    /// decoded input is admitted at its retained size and never decoded
    /// again (PIPE-14). Waits while admitted work holds the memory the job
    /// needs (P1V-2), unless it is cancelled meanwhile.
    async fn admit_pipeline_audio(
        &self,
        audio_input: TranscriptAudioInput<'_>,
        cancellation: &crate::runtime::coordinator::JobCancellation,
    ) -> Result<(JobLease, PipelineAudio)> {
        let backend = self.backend_context().backend_kind;
        let (spec, observed_bytes) = match &audio_input {
            TranscriptAudioInput::Encoded(input) => {
                let input_bytes = input.input_bytes();
                let mut spec = self.coordinator_job_for_input(
                    uuid::Uuid::new_v4().to_string(),
                    CoordinatorLane::Pipeline,
                    RuntimeRequestContext::new(crate::engine::WorkloadClass::Background),
                    input_bytes,
                );
                spec.resources = spec
                    .resources
                    .checked_add(diarization_decode_resources(backend))?;
                (spec, input_bytes)
            }
            TranscriptAudioInput::Samples(samples) => {
                let sample_bytes = std::mem::size_of_val(samples.as_slice());
                let mut spec = self.coordinator_job_for_input(
                    uuid::Uuid::new_v4().to_string(),
                    CoordinatorLane::Pipeline,
                    RuntimeRequestContext::new(crate::engine::WorkloadClass::Background),
                    0,
                );
                spec.resources = diarization_retained_resources(
                    backend,
                    JobResourceObservation::host(sample_bytes as u64),
                )?;
                (spec, sample_bytes)
            }
        };
        let observed_bytes = u64::try_from(observed_bytes)
            .map_err(|_| Error::Overloaded("diarization input exceeds u64".to_string()))?;
        // A cancelled job fails its next stage and drops its leases (PIPE-7).
        let pipeline_job = wait_for_held_capacity(pipeline_capacity_wait(), || {
            let spec = spec.clone();
            async move {
                if cancellation.is_cancelled() {
                    return Err(Error::Cancelled(
                        "diarization was cancelled before admission".to_string(),
                    ));
                }
                self.coordinator
                    .admit_observed(spec, JobResourceObservation::host(observed_bytes))
                    .await
            }
        })
        .await?
        .with_cancellation(cancellation.clone());
        let audio_input = match audio_input {
            TranscriptAudioInput::Samples(samples) => {
                return Ok((pipeline_job, PipelineAudio::from_canonical(samples)?));
            }
            TranscriptAudioInput::Encoded(input) => input,
        };
        let input_bytes = audio_input.input_bytes();
        let owned_audio_input = audio_input.into_owned_for_job(&pipeline_job).await?;
        pipeline_job.record_materialized_usage(JobResourceObservation::host(
            u64::try_from(
                input_bytes
                    .checked_add(owned_audio_input.retained_bytes())
                    .ok_or_else(|| {
                        Error::Overloaded("diarization retained input overflowed".to_string())
                    })?,
            )
            .map_err(|_| Error::Overloaded("diarization input exceeds u64".to_string()))?,
        ))?;
        let observation_job = pipeline_job.clone();
        let decodes = self.diarization_audio_decodes.clone();
        let audio = self
            .coordinator
            .run_host_blocking_stage(&pipeline_job, move || {
                let (audio_bytes, retained_source) = owned_audio_input.decode_bytes()?;
                decodes.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let audio = decode_pipeline_audio_bytes(&audio_bytes)?;
                let peak_usage = diarization_input_observation(
                    input_bytes,
                    &[
                        audio_bytes.capacity(),
                        retained_source.as_ref().map_or(0, String::capacity),
                    ],
                    &audio,
                )?;
                // The job's copy of the encoded bytes ends with the decode;
                // only the caller's input and the samples stay.
                drop(audio_bytes);
                let steady_usage = diarization_input_observation(input_bytes, &[], &audio)?;
                settle_decoded_job(
                    &observation_job,
                    backend,
                    peak_usage,
                    steady_usage,
                    retained_source,
                )?;
                Ok(audio)
            })
            .await?;
        Ok((pipeline_job, audio))
    }

    /// Run diarization and produce speaker-attributed transcript artifacts.
    pub async fn diarize_with_transcript(
        &self,
        audio_base64: &str,
        diarization_model_id: Option<&str>,
        asr_model_id: Option<&str>,
        aligner_model_id: Option<&str>,
        config: &DiarizationConfig,
        options: &DiarizationPipelineOptions,
    ) -> Result<DiarizationTranscriptResult> {
        self.diarize_with_transcript_input(
            TranscriptAudioInput::Encoded(DiarizationAudioInput::Base64(audio_base64)),
            diarization_model_id,
            asr_model_id,
            aligner_model_id,
            config,
            options,
        )
        .await
    }

    pub async fn diarize_with_transcript_bytes(
        &self,
        audio_bytes: &[u8],
        diarization_model_id: Option<&str>,
        asr_model_id: Option<&str>,
        aligner_model_id: Option<&str>,
        config: &DiarizationConfig,
        options: &DiarizationPipelineOptions,
    ) -> Result<DiarizationTranscriptResult> {
        self.diarize_with_transcript_input(
            TranscriptAudioInput::Encoded(DiarizationAudioInput::Bytes(audio_bytes)),
            diarization_model_id,
            asr_model_id,
            aligner_model_id,
            config,
            options,
        )
        .await
    }

    /// Run the pipeline over canonical 16 kHz mono `samples` the caller has
    /// already decoded, so the upload is decoded once per job (PIPE-14).
    /// Audio over one hour is refused.
    pub async fn diarize_with_transcript_samples(
        &self,
        samples: Vec<f32>,
        diarization_model_id: Option<&str>,
        asr_model_id: Option<&str>,
        aligner_model_id: Option<&str>,
        config: &DiarizationConfig,
        options: &DiarizationPipelineOptions,
    ) -> Result<DiarizationTranscriptResult> {
        self.diarize_with_transcript_input(
            TranscriptAudioInput::Samples(samples),
            diarization_model_id,
            asr_model_id,
            aligner_model_id,
            config,
            options,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn diarize_with_transcript_input(
        &self,
        audio_input: TranscriptAudioInput<'_>,
        diarization_model_id: Option<&str>,
        asr_model_id: Option<&str>,
        aligner_model_id: Option<&str>,
        config: &DiarizationConfig,
        options: &DiarizationPipelineOptions,
    ) -> Result<DiarizationTranscriptResult> {
        audio_input.validate()?;
        let diarization_variant = resolve_diarization_model_variant(diarization_model_id);
        if diarization_variant.diarizes_in_one_model() {
            if asr_model_id.is_some() || aligner_model_id.is_some() {
                return Err(Error::InvalidInput(format!(
                    "{diarization_variant} diarizes on its own; it takes no pipeline ASR or aligner"
                )));
            }
            return self
                .diarize_in_one_model(audio_input, diarization_variant, config, options)
                .await;
        }
        // Refuse an ASR or aligner the pipeline cannot run before diarizing:
        // the ASR's long-form graph must run as one blocking stage (the
        // predicate the post-load guard applies).
        let asr_variant =
            resolve_diarization_asr_model_variant(asr_model_id).map_err(Error::InvalidInput)?;
        if !self.adapter_registry.supports_atomic_long_form(
            CapabilityKind::Asr,
            asr_variant,
            self.backend_context().backend_kind,
        )? {
            return Err(Error::InvalidInput(format!(
                "{asr_variant} cannot transcribe in the diarization pipeline: it has no atomic long-form graph"
            )));
        }
        let aligner_variant =
            crate::runtime::asr::resolve_forced_aligner_variant(aligner_model_id)?;
        self.observe_broker_capability_request(
            CapabilityKind::Diarization,
            Some(diarization_variant),
            false,
        )?;
        self.record_diarization_transcript_pipeline();
        let (pipeline_job, audio) = self
            .admit_pipeline_audio(audio_input, &options.cancellation)
            .await?;
        let config = config.clone();

        // Sortformer and the ASR are independent until attribution, so they
        // run at once (PIPE-15): on Metal the CPU Sortformer's stages take the
        // host-model lane while the ASR holds the GPU; elsewhere their bounded
        // stages share the execution permits. The diarizer loads first, as
        // it did when the stages ran in turn; an ASR or aligner load that
        // fails while it runs waits for it to finish, and frees its model,
        // then tries once more, so the overlap fails no job that ran in turn.
        let diarizer = self
            .load_diarizer(&pipeline_job, diarization_variant)
            .await?;
        let (diarized_tx, mut diarized) = tokio::sync::watch::channel(false);
        let diarize = async {
            let result = self
                .run_diarizer(&pipeline_job, diarizer, audio.clone(), config.clone())
                .await;
            let _ = diarized_tx.send(true);
            result
        };
        let transcribe = async {
            // Model loads are not stages: do not load the ASR or the aligner
            // for a job that was cancelled.
            pipeline_job.ensure_not_cancelled()?;
            self.observe_broker_capability_request(CapabilityKind::Asr, Some(asr_variant), false)?;
            let asr_target = self
                .adapter_registry
                .require(CapabilityKind::Asr, asr_variant)?
                .execution_target;
            let (asr_lease, asr_contract, asr_state) =
                load_after_diarizer_on_failure(&mut diarized, || {
                    self.load_capability_with_state_for_job(
                        &pipeline_job,
                        asr_variant,
                        CapabilityKind::Asr,
                        false,
                        asr_target,
                        CapabilityLoadOptions {
                            asr_long_form: true,
                        },
                    )
                })
                .await?;
            let asr_model = match diarization_asr_registry_route(asr_variant) {
                DiarizationAsrRegistryRoute::Voxtral => DiarizationAsrModel::Voxtral(
                    self.model_registry
                        .get_voxtral_lease(asr_variant)
                        .await
                        .ok_or_else(|| Error::ModelNotFound(asr_variant.to_string()))?,
                ),
                DiarizationAsrRegistryRoute::Native => DiarizationAsrModel::Native(
                    self.model_registry
                        .get_asr(asr_variant)
                        .await
                        .ok_or_else(|| Error::ModelNotFound(asr_variant.to_string()))?,
                ),
                DiarizationAsrRegistryRoute::AudioChat => DiarizationAsrModel::AudioChat(
                    self.model_registry
                        .get_audio_chat(asr_variant)
                        .await
                        .ok_or_else(|| Error::ModelNotFound(asr_variant.to_string()))?,
                ),
            };

            pipeline_job.ensure_not_cancelled()?;
            let mut aligner_unavailable = None;
            let aligner_binding = match load_after_diarizer_on_failure(&mut diarized, || {
                self.load_capability_for_job(
                    &pipeline_job,
                    aligner_variant,
                    CapabilityKind::ForcedAlignment,
                    false,
                    ExecutionTargetKind::BatchRunner,
                )
            })
            .await
            {
                Ok(binding) => Some(binding),
                Err(err) => {
                    warn!("Forced aligner load failed, using heuristic timings: {err}");
                    aligner_unavailable = Some(format!("{aligner_variant} failed to load: {err}"));
                    None
                }
            };
            let aligner_model = if aligner_binding.is_some() {
                match self.model_registry.get_asr(aligner_variant).await {
                    Some(model) => Some(model),
                    None => {
                        warn!(
                            "Forced aligner {} was loaded but not found in registry",
                            aligner_variant
                        );
                        aligner_unavailable =
                            Some(format!("{aligner_variant} is not in the model registry"));
                        None
                    }
                }
            } else {
                None
            };

            // Compose the ASR family hint with the invocation context so
            // invocation-backed families chunk within the memory-fitted cache.
            let asr_limit_secs = crate::runtime::asr::compose_invocation_audio_limit(
                asr_model.max_audio_seconds_hint(),
                asr_model.audio_token_rate(),
                self.model_registry.effective_context(asr_variant),
                diarization_asr_max_new_tokens,
            )?;
            tracing::info!(
                variant = %asr_variant,
                asr_limit_secs = ?asr_limit_secs,
                "resolved diarization pipeline ASR invocation audio limit"
            );

            let aligner_limit = aligner_model
                .as_ref()
                .and_then(|model| model.max_audio_seconds_hint());
            let chunk_limit = combined_chunk_limit(asr_limit_secs, aligner_limit);
            let use_single_pass_asr =
                should_use_single_pass_diarization_asr(audio.duration_secs, chunk_limit);
            // Every ASR handle (residency lease, model, contract and state) moves
            // into this block and drops at its end, so the ASR is evictable before
            // the aligner runs (ADM-8). Dropping only
            // the lease would let eviction count weights the model still pins.
            let (asr_windows, chunk_texts, single_pass_hit_budget) = {
                let _asr_lease = asr_lease;
                let (asr_model, asr_contract, asr_state) = (asr_model, asr_contract, asr_state);
                if use_single_pass_asr {
                    let audio_for_task = audio.clone();
                    let language = options.language.clone();
                    // Size the decode budget from the duration, as each chunk's
                    // budget covers its own audio on the chunked path.
                    let options = NativeAsrGenerationOptions {
                        max_new_tokens: diarization_asr_max_new_tokens(audio.duration_secs),
                        ..NativeAsrGenerationOptions::default()
                    };
                    let transcription = self
                        .coordinator
                        .run_loaded_blocking_stage_with_invocation_workspace(
                            &pipeline_job,
                            asr_contract,
                            asr_state,
                            WorkUnit::AtomicJob {
                                kind: "diarization.transcribe".to_string(),
                            },
                            move |leases| {
                                asr_model.transcribe_from_invocation_workspace(
                                    &audio_for_task.samples,
                                    PIPELINE_SAMPLE_RATE,
                                    language.as_deref(),
                                    None,
                                    options,
                                    leases,
                                )
                            },
                        )
                        .await?;
                    let hit_budget = asr_hit_token_budget(transcription.diagnostics.as_ref());
                    let window = SampleWindow::new(
                        0,
                        audio.samples.len(),
                        transcription.text,
                        transcription.language.as_deref(),
                    );
                    (vec![window], Vec::new(), hit_budget)
                } else {
                    let (_cfg, chunks) = plan_invocation_asr_chunks(&audio, chunk_limit)?;
                    let mut transcribed = Vec::with_capacity(chunks.len());
                    for chunk in chunks {
                        if chunk.end_sample <= chunk.start_sample
                            || chunk.end_sample > audio.samples.len()
                        {
                            continue;
                        }
                        let model = asr_model.clone();
                        let chunk_audio = audio.clone();
                        let language = options.language.clone();
                        let chunk_start = chunk.start_sample;
                        let chunk_end = chunk.end_sample;
                        let options = NativeAsrGenerationOptions {
                            max_new_tokens: diarization_asr_max_new_tokens(
                                chunk.len_samples() as f32 / PIPELINE_SAMPLE_RATE as f32,
                            ),
                            ..NativeAsrGenerationOptions::default()
                        };
                        let transcription = self
                            .coordinator
                            .run_loaded_blocking_stage_with_invocation_workspace(
                                &pipeline_job,
                                asr_contract.clone(),
                                asr_state.clone(),
                                WorkUnit::AtomicJob {
                                    kind: "diarization.transcribe_chunks".to_string(),
                                },
                                move |leases| {
                                    model.transcribe_from_invocation_workspace(
                                        &chunk_audio.samples[chunk_start..chunk_end],
                                        PIPELINE_SAMPLE_RATE,
                                        language.as_deref(),
                                        None,
                                        options,
                                        leases,
                                    )
                                },
                            )
                            .await?;
                        transcribed.push(TranscribedChunk {
                            range: chunk,
                            hit_token_budget: asr_hit_token_budget(
                                transcription.diagnostics.as_ref(),
                            ),
                            text: transcription.text,
                            language: transcription.language,
                        });
                    }
                    (chunk_windows(&transcribed), transcribed, false)
                }
            };
            Ok::<_, Error>((
                asr_windows,
                chunk_texts,
                single_pass_hit_budget,
                aligner_unavailable,
                aligner_model,
                aligner_binding,
            ))
        };
        let (
            diarization,
            (
                asr_windows,
                chunk_texts,
                single_pass_hit_budget,
                aligner_unavailable,
                aligner_model,
                aligner_binding,
            ),
        ) = tokio::try_join!(diarize, transcribe)?;
        let mut warnings = asr_token_budget_warnings(
            single_pass_hit_budget.then_some((0.0, audio.duration_secs)),
            &chunk_texts,
        );
        warnings.extend(aligner_unavailable.map(|reason| aligner_unavailable_warning(&reason)));

        let has_words = asr_windows.iter().any(|window| !window.units.is_empty());
        let timed_windows = if !has_words {
            Vec::new()
        } else if let (Some(model), Some((aligner_lease, aligner_contract))) =
            (aligner_model, aligner_binding)
        {
            // One aligner stage per window: the whole file after a single ASR
            // pass, otherwise each ASR chunk with its own text.
            let aligned = self
                .align_sample_windows(
                    &pipeline_job,
                    aligner_contract,
                    model,
                    audio.samples.clone(),
                    audio.sample_rate,
                    asr_windows.clone(),
                    "diarization.force_align",
                )
                .await?;
            drop(aligner_lease);
            timed_windows_from_alignment(&asr_windows, aligned)
        } else {
            interval_timed_windows(&asr_windows, audio.sample_rate)
        };

        let AttributedTranscript {
            asr_text,
            words,
            word_texts,
            model_aligned_words,
            unattributed_words,
        } = attribute_transcript(timed_windows, &diarization.segments, &mut warnings);

        let utterances = build_utterances(&words, &word_texts);
        let raw_transcript = if utterances.is_empty() {
            asr_text.clone()
        } else {
            format_utterance_transcript(&utterances)
        };

        let transcript = raw_transcript.clone();
        pipeline_job.ensure_not_cancelled()?;

        let alignment_coverage = if words.is_empty() {
            0.0
        } else {
            (model_aligned_words.min(words.len())) as f32 / words.len() as f32
        };

        Ok(DiarizationTranscriptResult {
            segments: diarization.segments,
            words,
            utterances,
            asr_text,
            raw_transcript,
            transcript,
            duration_secs: diarization.duration_secs,
            speaker_count: diarization.speaker_count,
            alignment_coverage,
            unattributed_words,
            warnings,
        })
    }
}

fn decode_pipeline_audio_bytes(audio_bytes: &[u8]) -> Result<PipelineAudio> {
    let (samples, sample_rate) = decode_audio_bytes_canonical(audio_bytes)?;
    debug_assert_eq!(sample_rate, PIPELINE_SAMPLE_RATE);
    PipelineAudio::from_canonical(samples)
}

/// The pipeline's one-hour bound, on the decoded length (PIPE-14).
fn validate_pipeline_sample_count(samples: usize) -> Result<()> {
    if samples == 0 {
        return Err(Error::InvalidInput(
            "diarization request missing audio input".to_string(),
        ));
    }
    if samples > MAX_PIPELINE_SAMPLES {
        return Err(Error::InvalidInput(format!(
            "diarization audio is {:.1} s long, over the {MAX_PIPELINE_DURATION_SECONDS} s limit",
            samples as f64 / f64::from(PIPELINE_SAMPLE_RATE)
        )));
    }
    Ok(())
}

/// The long-form chunk plan, with its `IZWI_ASR_CHUNK_*` overrides read once
/// per process.
fn pipeline_chunk_config() -> AsrLongFormConfig {
    static CONFIG: crate::env::ReadOnce<AsrLongFormConfig> =
        crate::env::ReadOnce::new(pipeline_chunk_config_from_env);
    CONFIG.get()
}

fn pipeline_chunk_config_from_env() -> AsrLongFormConfig {
    let mut cfg = AsrLongFormConfig::default();
    if let Some(v) = env_positive_f32("IZWI_ASR_CHUNK_TARGET_SECS") {
        cfg.target_chunk_secs = v;
    }
    if let Some(v) = env_positive_f32("IZWI_ASR_CHUNK_MAX_SECS") {
        cfg.hard_max_chunk_secs = v;
    }
    if let Some(v) = env_positive_f32("IZWI_ASR_CHUNK_OVERLAP_SECS") {
        cfg.overlap_secs = v;
    }
    cfg
}

/// How long a diarization job waits for memory admitted work holds, at
/// admission and at its stage workspace, before it fails (P1V-2).
const DEFAULT_PIPELINE_CAPACITY_WAIT: Duration = Duration::from_secs(120);

/// [`DEFAULT_PIPELINE_CAPACITY_WAIT`], or `IZWI_DIARIZATION_CAPACITY_WAIT_SECS`
/// (0 fails at once), read once per process.
fn pipeline_capacity_wait() -> Duration {
    static WAIT: crate::env::ReadOnce<Duration> = crate::env::ReadOnce::new(|| {
        std::env::var("IZWI_DIARIZATION_CAPACITY_WAIT_SECS")
            .ok()
            .and_then(|raw| raw.trim().parse::<f64>().ok())
            .and_then(|secs| Duration::try_from_secs_f64(secs).ok())
            .unwrap_or(DEFAULT_PIPELINE_CAPACITY_WAIT)
    });
    WAIT.get()
}

fn env_positive_f32(key: &str) -> Option<f32> {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.trim().parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
}

fn combined_chunk_limit(asr_limit: Option<f32>, aligner_limit: Option<f32>) -> Option<f32> {
    match (asr_limit, aligner_limit) {
        (Some(asr), Some(aligner)) => Some(asr.min(aligner)),
        (Some(asr), None) => Some(asr),
        (None, Some(aligner)) => Some(aligner),
        (None, None) => None,
    }
}

/// A context-derived limit below the pipeline's minimum chunk size forces
/// graceful degradation: smaller chunks keep the pipeline functional on
/// constrained hosts instead of failing every invocation.
fn degrade_pipeline_chunk_config(
    mut cfg: AsrLongFormConfig,
    chunk_limit: Option<f32>,
) -> AsrLongFormConfig {
    if let Some(limit) = chunk_limit.filter(|value| value.is_finite() && *value > 0.0)
    {
        if limit < cfg.min_chunk_secs {
            tracing::warn!(
                context_limit_secs = limit,
                family_min_chunk_secs = cfg.min_chunk_secs,
                "ASR pipeline invocation context limits chunks below the family minimum; transcription accuracy degrades"
            );
            cfg.min_chunk_secs = limit.max(1.0);
            cfg.target_chunk_secs = cfg.target_chunk_secs.max(cfg.min_chunk_secs);
        }
    }
    cfg
}

/// Decode tokens one diarization ASR invocation may emit for `seconds` of
/// audio: the native default (enough for a chunk) or, for a longer single
/// pass, a duration budget well above dense speech, so the transcript is not
/// cut at a fixed token count.
const DIARIZATION_ASR_TOKENS_PER_SECOND: f32 = 8.0;
const DIARIZATION_ASR_TOKEN_RESERVE: usize = 64;

fn diarization_asr_max_new_tokens(seconds: f32) -> usize {
    let scaled = if seconds.is_finite() && seconds > 0.0 {
        (seconds * DIARIZATION_ASR_TOKENS_PER_SECOND).ceil() as usize
            + DIARIZATION_ASR_TOKEN_RESERVE
    } else {
        0
    };
    scaled.max(NativeAsrGenerationOptions::default().max_new_tokens)
}

/// Whether an ASR decode stopped at its token budget, as reported by a
/// `stop_reason` of `max_tokens` anywhere in its diagnostics.
fn asr_hit_token_budget(diagnostics: Option<&serde_json::Value>) -> bool {
    fn search(value: &serde_json::Value, depth: usize) -> bool {
        match value {
            serde_json::Value::Object(map) => {
                map.get("stop_reason")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|reason| reason == "max_tokens")
                    || (depth < 4 && map.values().any(|child| search(child, depth + 1)))
            }
            serde_json::Value::Array(items) => {
                depth < 4 && items.iter().any(|child| search(child, depth + 1))
            }
            _ => false,
        }
    }
    diagnostics.is_some_and(|value| search(value, 0))
}

/// One warning per ASR pass that stopped at its token budget: the single pass
/// over `(start, end)` seconds, or each such chunk.
fn asr_token_budget_warnings(
    single_pass: Option<(f32, f32)>,
    chunks: &[TranscribedChunk],
) -> Vec<String> {
    let chunk_spans = chunks
        .iter()
        .filter(|chunk| chunk.hit_token_budget)
        .map(|chunk| {
            (
                chunk.range.start_sample as f32 / PIPELINE_SAMPLE_RATE as f32,
                chunk.range.end_sample as f32 / PIPELINE_SAMPLE_RATE as f32,
            )
        });
    single_pass
        .into_iter()
        .chain(chunk_spans)
        .map(|(start, end)| {
            let warning = format!(
                "ASR for {start:.2}-{end:.2} s stopped at its token budget; the transcript may be cut short"
            );
            warn!("{warning}");
            warning
        })
        .collect()
}

/// The warning a transcript carries when its aligner could not be used, so a
/// record with estimated word timings says so instead of looking aligned.
fn aligner_unavailable_warning(reason: &str) -> String {
    format!("Forced aligner unavailable ({reason}); word timings are estimated, not aligned")
}

/// One ASR pass over the whole file is allowed only when it fits the
/// combined ASR/aligner chunk limit (`None` = unbounded). Without an aligner
/// the ASR limit alone still applies: invocation-backed families reject
/// over-bound audio, and the chunked path falls back to per-chunk timings.
fn should_use_single_pass_diarization_asr(duration_secs: f32, chunk_limit: Option<f32>) -> bool {
    match chunk_limit {
        Some(limit) if limit.is_finite() && limit > 0.0 => duration_secs <= limit,
        _ => true,
    }
}

fn plan_invocation_asr_chunks(
    audio: &PipelineAudio,
    chunk_limit: Option<f32>,
) -> Result<(AsrLongFormConfig, Vec<AudioChunk>)> {
    let cfg = degrade_pipeline_chunk_config(pipeline_chunk_config(), chunk_limit);
    let chunks = plan_audio_chunks(&audio.samples, audio.sample_rate, &cfg, chunk_limit);
    if chunks.is_empty() {
        return Err(Error::InvalidInput(
            "ASR chunk planner produced no chunks".to_string(),
        ));
    }
    Ok((cfg, chunks))
}

/// Alignment windows of the transcribed chunks: each chunk's audio with the
/// text heard in it, split into word units for the chunk's language.
fn chunk_windows(chunks: &[TranscribedChunk]) -> Vec<SampleWindow> {
    chunks
        .iter()
        .map(|chunk| {
            SampleWindow::new(
                chunk.range.start_sample,
                chunk.range.end_sample,
                chunk.text.clone(),
                chunk.language.as_deref(),
            )
        })
        .collect()
}

/// A timed word of one window: the aligner's word (the source text of one
/// unit), the transcript text that renders it (the word with the whitespace
/// before it), and its absolute times.
#[derive(Debug, Clone, PartialEq)]
struct WindowWord {
    word: String,
    text: String,
    start_ms: u32,
    end_ms: u32,
}

/// The word timings of one ASR window and where they came from.
#[derive(Debug, Clone)]
struct TimedWindow {
    start_ms: u32,
    end_ms: u32,
    source: AlignmentSource,
    /// Why the aligner did not time the window, when it ran on it.
    note: Option<String>,
    words: Vec<WindowWord>,
}

/// A word of the stitched transcript; `model` when the aligner timed it.
/// `text` is what the word adds to the stitched text: its source text with
/// the spacing before it.
#[derive(Debug, Clone, PartialEq)]
struct StitchedWord {
    word: String,
    text: String,
    start_ms: u32,
    end_ms: u32,
    model: bool,
}

/// The transcript of every window stitched together, and its timed words.
#[derive(Debug, Default)]
struct StitchedTranscript {
    text: String,
    words: Vec<StitchedWord>,
}

/// The aligner's words of `window` paired with their transcript text: each
/// word with the punctuation around it and the whitespace before it, so a run
/// of tokens concatenates back to the text. A gap between two words splits at
/// its first whitespace, or before the next word when it has none (CJK).
fn word_tokens(window: &SampleWindow) -> Vec<(String, String)> {
    let text = window.text.as_str();
    let ranges = window
        .units
        .iter()
        .map(|unit| unit.source.clone())
        .collect::<Vec<_>>();
    let mut tokens = Vec::with_capacity(ranges.len());
    let mut token_start = 0usize;
    for (idx, range) in ranges.iter().enumerate() {
        let token_end = match ranges.get(idx + 1) {
            Some(next) => {
                let gap = &text[range.end..next.start];
                range.end + gap.find(char::is_whitespace).unwrap_or(gap.len())
            }
            None => text.len(),
        };
        tokens.push((
            text[range.clone()].to_string(),
            text[token_start..token_end].trim_end().to_string(),
        ));
        token_start = token_end;
    }
    tokens
}

/// Pair the aligner's windows with the transcript text of their words. The
/// windows hold the words of `sources` in order (a source split into
/// sub-windows hands its words on in sequence), so one cursor walks both; a
/// window whose words do not match keeps its bare words, space-separated.
fn timed_windows_from_alignment(
    sources: &[SampleWindow],
    aligned: Vec<AlignedWindow>,
) -> Vec<TimedWindow> {
    let tokens = sources.iter().flat_map(word_tokens).collect::<Vec<_>>();
    let mut cursor = 0usize;
    aligned
        .into_iter()
        .map(|window| {
            let count = window.words.len();
            let texts = tokens
                .get(cursor..cursor + count)
                .filter(|slice| {
                    slice
                        .iter()
                        .zip(&window.words)
                        .all(|((token_word, _), (word, _, _))| token_word == word)
                })
                .map(|slice| {
                    slice
                        .iter()
                        .map(|(_, text)| text.clone())
                        .collect::<Vec<_>>()
                });
            cursor += count;
            let words = window
                .words
                .into_iter()
                .enumerate()
                .map(|(idx, (word, start_ms, end_ms))| WindowWord {
                    text: match &texts {
                        Some(texts) => texts[idx].clone(),
                        None if idx == 0 => word.clone(),
                        None => format!(" {word}"),
                    },
                    word,
                    start_ms,
                    end_ms,
                })
                .collect();
            TimedWindow {
                start_ms: window.start_ms,
                end_ms: window.end_ms,
                source: window.source,
                note: (window.source != AlignmentSource::Model).then(|| {
                    let kind = match window.source {
                        AlignmentSource::Failed => "alignment failed",
                        _ => "alignment was degenerate",
                    };
                    match window.detail {
                        Some(detail) => format!("{kind} ({detail})"),
                        None => kind.to_string(),
                    }
                }),
                words,
            }
        })
        .collect()
}

/// Windows timed without the aligner: each window's words evenly spaced over
/// its audio.
fn interval_timed_windows(windows: &[SampleWindow], sample_rate: u32) -> Vec<TimedWindow> {
    windows
        .iter()
        .map(|window| {
            let start_ms = samples_to_ms(window.start_sample, sample_rate);
            let end_ms = samples_to_ms(window.end_sample.max(window.start_sample), sample_rate);
            let tokens = word_tokens(window);
            let words = tokens
                .iter()
                .map(|(word, _)| word.clone())
                .collect::<Vec<_>>();
            let duration_secs = end_ms.saturating_sub(start_ms) as f32 / 1000.0;
            let words = tokens
                .into_iter()
                .zip(fallback_word_timings_from_words(&words, duration_secs))
                .map(|((word, text), (_, start, end))| WindowWord {
                    word,
                    text,
                    start_ms: start_ms.saturating_add(start),
                    end_ms: start_ms.saturating_add(end),
                })
                .collect();
            TimedWindow {
                start_ms,
                end_ms,
                source: AlignmentSource::Heuristic,
                note: None,
                words,
            }
        })
        .collect()
}

fn ms_to_secs(value: u32) -> f32 {
    value as f32 / 1000.0
}

/// Stitch consecutive windows into one transcript at the middle of the audio
/// each pair shares ([`chunk_seam_secs`]): a word belongs to the window whose
/// side of the seam holds its centre, so a word heard by both windows is kept
/// once and a word cut by a window edge lands where most of it was heard.
/// The text is built from the kept words' tokens, keeping the transcript's
/// own spacing between words kept in a row, so it holds exactly the stitched
/// words; a start never runs back before the previous word's end.
fn stitch_timed_windows(windows: &[TimedWindow]) -> StitchedTranscript {
    let mut text = String::new();
    let mut words = Vec::new();
    let mut previous_kept: Option<(usize, usize)> = None;
    let mut previous_end_ms = 0u32;
    for (idx, window) in windows.iter().enumerate() {
        let after_secs = idx.checked_sub(1).map(|previous| {
            chunk_seam_secs(
                ms_to_secs(windows[previous].end_ms),
                ms_to_secs(window.start_ms),
            )
        });
        let until_secs = windows
            .get(idx + 1)
            .map(|next| chunk_seam_secs(ms_to_secs(window.end_ms), ms_to_secs(next.start_ms)));
        for (word_idx, word) in window.words.iter().enumerate() {
            let timed = TimedWord {
                text: word.text.clone(),
                start_secs: ms_to_secs(word.start_ms),
                end_secs: ms_to_secs(word.end_ms),
            };
            if word.text.trim().is_empty() || !timed.within_seams(after_secs, until_secs) {
                continue;
            }
            let follows_in_source = word_idx
                .checked_sub(1)
                .is_some_and(|previous| previous_kept == Some((idx, previous)));
            let rendered = if follows_in_source {
                word.text.clone()
            } else {
                let token = word.text.trim_start();
                format!("{}{token}", spacing_between(&text, token))
            };
            text.push_str(&rendered);
            previous_kept = Some((idx, word_idx));
            let start_ms = word.start_ms.max(previous_end_ms);
            let end_ms = word.end_ms.max(start_ms);
            previous_end_ms = end_ms;
            words.push(StitchedWord {
                word: word.word.clone(),
                text: rendered,
                start_ms,
                end_ms,
                model: window.source == AlignmentSource::Model,
            });
        }
    }
    StitchedTranscript { text, words }
}

/// A model-timed word longer than [`MAX_REASONABLE_WORD_SPAN_MS`] keeps its
/// start and ends at most this much later.
const OUTLIER_WORD_SPAN_MS: u32 = 1_000;

/// The transcript's words timed and attributed to speakers, with the text
/// that renders each word in the transcript (see [`StitchedWord`]).
struct AttributedTranscript {
    asr_text: String,
    words: Vec<DiarizationWord>,
    word_texts: Vec<String>,
    model_aligned_words: usize,
    unattributed_words: usize,
}

/// Time the transcript's words from their windows and attribute them to the
/// speaker turns. Bad timing is repaired where it happened: a window the
/// aligner did not time, or whose aligned words fall outside the speaker
/// turns in it, is re-timed over those turns, and a model word stretched over
/// a pause is clamped to its neighbours. Every other window keeps the model's
/// timings, whatever happened elsewhere in the file.
fn attribute_transcript(
    mut windows: Vec<TimedWindow>,
    segments: &[DiarizationSegment],
    warnings: &mut Vec<String>,
) -> AttributedTranscript {
    for window in &mut windows {
        let span = format!(
            "{:.2}-{:.2} s",
            ms_to_secs(window.start_ms),
            ms_to_secs(window.end_ms)
        );
        if window.source != AlignmentSource::Model {
            let retimed = retime_window_over_turns(window, segments);
            if let Some(note) = &window.note {
                let timing = if retimed {
                    "are spread over the speaker turns"
                } else {
                    "are evenly spaced"
                };
                warnings.push(format!("Words in {span} {timing}: {note}"));
            }
            continue;
        }
        let overlapping = window
            .words
            .iter()
            .filter(|word| {
                let start = ms_to_secs(word.start_ms);
                let end = ms_to_secs(word.end_ms.max(word.start_ms + 1));
                segments.iter().any(|segment| {
                    interval_overlap(start, end, segment.start_secs, segment.end_secs) > 0.0
                })
            })
            .count();
        let outside = window.words.len() - overlapping;
        if attribution_requires_fallback(window.words.len(), overlapping, outside)
            && retime_window_over_turns(window, segments)
        {
            window.source = AlignmentSource::Heuristic;
            let warning = format!(
                "Words in {span} are spread over the speaker turns: {outside} of {} aligned words fell outside them",
                window.words.len()
            );
            warn!("{warning}");
            warnings.push(warning);
        }
    }

    let stitched = stitch_timed_windows(&windows);
    let mut alignments = stitched
        .words
        .iter()
        .map(|word| (word.word.clone(), word.start_ms, word.end_ms))
        .collect::<Vec<_>>();
    clamp_outlier_word_spans(&mut alignments, &stitched.words);
    let model_aligned_words = stitched.words.iter().filter(|word| word.model).count();
    let (mut words, _overlap_assigned_words, unattributed_words) =
        attribute_words_to_speakers(&alignments, segments);
    stabilize_word_speakers(&mut words);
    normalize_phrase_speakers(&mut words, segments);
    AttributedTranscript {
        asr_text: stitched.text,
        words,
        word_texts: stitched.words.into_iter().map(|word| word.text).collect(),
        model_aligned_words,
        unattributed_words,
    }
}

/// Re-time `window`'s words over the speaker turns inside it, keeping their
/// order. Returns false, leaving the words as they are, when no turn
/// overlaps the window.
fn retime_window_over_turns(window: &mut TimedWindow, segments: &[DiarizationSegment]) -> bool {
    let window_start = ms_to_secs(window.start_ms);
    let window_end = ms_to_secs(window.end_ms);
    let turns = segments
        .iter()
        .filter_map(|segment| {
            let start_secs = segment.start_secs.max(window_start);
            let end_secs = segment.end_secs.min(window_end);
            (end_secs > start_secs).then(|| DiarizationSegment {
                start_secs,
                end_secs,
                ..segment.clone()
            })
        })
        .collect::<Vec<_>>();
    if turns.is_empty() || window.words.is_empty() {
        return false;
    }
    let words = window
        .words
        .iter()
        .map(|word| word.word.clone())
        .collect::<Vec<_>>();
    let timings = fallback_word_timings_with_segments(&words, &turns, window_end);
    for (word, (_, start_ms, end_ms)) in window.words.iter_mut().zip(timings) {
        word.start_ms = start_ms;
        word.end_ms = end_ms;
    }
    true
}

/// Clamp each model-timed word longer than any word lasts (it absorbed a
/// pause) to [`OUTLIER_WORD_SPAN_MS`] after its start, never past the next
/// word's start.
fn clamp_outlier_word_spans(alignments: &mut [(String, u32, u32)], stitched: &[StitchedWord]) {
    for idx in 0..alignments.len() {
        let (_, start, end) = alignments[idx];
        if !stitched[idx].model || end.saturating_sub(start) < MAX_REASONABLE_WORD_SPAN_MS {
            continue;
        }
        let next_start = alignments
            .get(idx + 1)
            .map_or(u32::MAX, |(_, next_start, _)| *next_start);
        alignments[idx].2 = start
            .saturating_add(OUTLIER_WORD_SPAN_MS)
            .min(next_start)
            .max(start.saturating_add(1));
    }
}

pub(super) fn fallback_word_timings_from_words(
    words: &[String],
    duration_secs: f32,
) -> Vec<(String, u32, u32)> {
    if words.is_empty() {
        return Vec::new();
    }

    let max_duration_ms = secs_to_ms(duration_secs);
    let duration_ms = if max_duration_ms > 0 {
        max_duration_ms
    } else {
        (words.len() as u32).saturating_mul(300).max(1)
    };
    let step = (duration_ms as f32 / words.len() as f32).max(1.0);

    words
        .iter()
        .enumerate()
        .map(|(idx, word)| {
            let start = ((idx as f32) * step).round() as u32;
            let mut end = (((idx + 1) as f32) * step).round() as u32;
            if end <= start {
                end = start.saturating_add(1);
            }
            (word.clone(), start, end)
        })
        .collect()
}

/// Spread `words`, in order, over the time anyone speaks: the time-sorted
/// union of the turns, each stretch taking words in proportion to its length.
/// Overlapping turns are merged first, so a turn nested in another never takes
/// a run of words out of order and the timings are monotonic.
fn fallback_word_timings_with_segments(
    words: &[String],
    segments: &[DiarizationSegment],
    duration_secs: f32,
) -> Vec<(String, u32, u32)> {
    if words.is_empty() {
        return Vec::new();
    }

    let mut turns = segments
        .iter()
        .filter_map(|segment| {
            let start = secs_to_ms(segment.start_secs);
            let mut end = secs_to_ms(segment.end_secs);
            if end <= start {
                end = start.saturating_add(1);
            }
            (end > start).then_some((start, end))
        })
        .collect::<Vec<_>>();
    turns.sort_by_key(|(start, _)| *start);
    let mut windows: Vec<(u32, u32)> = Vec::with_capacity(turns.len());
    for (start, end) in turns {
        match windows.last_mut() {
            Some(last) if start <= last.1 => last.1 = last.1.max(end),
            _ => windows.push((start, end)),
        }
    }

    if windows.is_empty() {
        return fallback_word_timings_from_words(words, duration_secs);
    }

    let durations = windows
        .iter()
        .map(|(start, end)| end.saturating_sub(*start).max(1))
        .collect::<Vec<_>>();
    let total_duration_ms = durations.iter().copied().sum::<u32>();
    if total_duration_ms == 0 {
        return fallback_word_timings_from_words(words, duration_secs);
    }

    let word_count = words.len();
    let mut allocations = vec![0usize; windows.len()];
    let mut assigned = 0usize;
    let mut remainders = Vec::with_capacity(windows.len());

    for (idx, duration_ms) in durations.iter().copied().enumerate() {
        let target = word_count as f32 * duration_ms as f32 / total_duration_ms as f32;
        let base = target.floor() as usize;
        allocations[idx] = base;
        assigned += base;
        remainders.push((target - base as f32, idx));
    }

    let mut remaining = word_count.saturating_sub(assigned);
    remainders.sort_by(|left, right| right.0.total_cmp(&left.0).then(left.1.cmp(&right.1)));
    for (_, idx) in remainders {
        if remaining == 0 {
            break;
        }
        allocations[idx] += 1;
        remaining -= 1;
    }

    let allocated_total = allocations.iter().sum::<usize>();
    if allocated_total < word_count {
        if let Some(last) = allocations.last_mut() {
            *last += word_count - allocated_total;
        }
    } else if allocated_total > word_count {
        let mut excess = allocated_total - word_count;
        for allocation in allocations.iter_mut().rev() {
            if excess == 0 {
                break;
            }
            let delta = (*allocation).min(excess);
            *allocation -= delta;
            excess -= delta;
        }
    }

    let mut alignments = Vec::with_capacity(word_count);
    let mut word_idx = 0usize;
    for ((segment_start, segment_end), allocation) in windows.into_iter().zip(allocations) {
        if allocation == 0 {
            continue;
        }
        let segment_span = segment_end.saturating_sub(segment_start).max(1);
        let step = segment_span as f32 / allocation as f32;

        for local_idx in 0..allocation {
            if word_idx >= word_count {
                break;
            }
            let start = segment_start.saturating_add((local_idx as f32 * step).floor() as u32);
            let mut end = if local_idx + 1 == allocation {
                segment_end
            } else {
                segment_start.saturating_add(((local_idx + 1) as f32 * step).floor() as u32)
            };
            if end <= start {
                end = start.saturating_add(1);
            }
            alignments.push((words[word_idx].clone(), start, end));
            word_idx += 1;
        }
    }

    if word_idx < word_count {
        let remaining_words = &words[word_idx..];
        let mut carry = fallback_word_timings_from_words(remaining_words, duration_secs);
        alignments.append(&mut carry);
    }

    alignments
}

pub(super) fn alignment_is_suspicious(
    alignments: &[(String, u32, u32)],
    expected_word_count: usize,
    duration_secs: f32,
) -> bool {
    if expected_word_count == 0 {
        return false;
    }
    if alignments.is_empty() {
        return true;
    }
    if alignments.len() < expected_word_count.saturating_div(2).max(1) {
        return true;
    }

    let duration_ms = secs_to_ms(duration_secs).max(1);
    let tail_start = duration_ms.saturating_sub(ALIGNMENT_COLLAPSE_TAIL_MS);
    let prefix_window_end = (duration_ms / 50).clamp(250, ALIGNMENT_PREFIX_CLUSTER_MS_CAP);

    let mut min_start = u32::MAX;
    let mut max_end = 0u32;
    let mut tiny_spans = 0usize;
    let mut tail_heavy = 0usize;
    let mut prefix_clustered = 0usize;
    let mut max_word_span_ms = 0u32;

    for (_, start, end) in alignments {
        let bounded_start = (*start).min(duration_ms);
        let bounded_end = (*end).min(duration_ms).max(bounded_start);
        let span_ms = bounded_end.saturating_sub(bounded_start);

        min_start = min_start.min(bounded_start);
        max_end = max_end.max(bounded_end);
        max_word_span_ms = max_word_span_ms.max(span_ms);

        if bounded_end <= bounded_start.saturating_add(1) {
            tiny_spans += 1;
        }
        if bounded_start >= tail_start {
            tail_heavy += 1;
        }
        if bounded_end <= prefix_window_end {
            prefix_clustered += 1;
        }
    }

    let span = max_end.saturating_sub(min_start);
    let len = alignments.len();
    len >= 8
        && (tiny_spans * 10 >= len * 8
            || tail_heavy * 10 >= len * 8
            || max_word_span_ms >= MAX_REASONABLE_WORD_SPAN_MS
            || (prefix_clustered * 10 >= len * 4 && tiny_spans * 10 >= len * 3)
            || span <= (duration_ms / 20).max(1))
}

fn attribution_requires_fallback(
    word_count: usize,
    overlap_assigned_words: usize,
    unattributed_words: usize,
) -> bool {
    if word_count == 0 {
        return false;
    }

    let overlap_coverage = overlap_assigned_words as f32 / word_count as f32;
    let unattributed_ratio = unattributed_words as f32 / word_count as f32;

    overlap_coverage < MIN_ALIGNMENT_COVERAGE || unattributed_ratio > MAX_UNATTRIBUTED_WORD_RATIO
}

/// Attribute each timed word to a speaker. Words stay in ASR order, which is
/// also start order: stitching never lets a start run back before the previous
/// word, so sorting by time could only ever reorder the transcript.
fn attribute_words_to_speakers(
    alignments: &[(String, u32, u32)],
    segments: &[DiarizationSegment],
) -> (Vec<DiarizationWord>, usize, usize) {
    let mut words = Vec::new();
    let mut overlap_assigned_words = 0usize;
    let mut unattributed_words = 0usize;

    for (word, start_ms, end_ms) in alignments {
        let cleaned = word.trim();
        if cleaned.is_empty() {
            continue;
        }

        let start_secs = (*start_ms as f32 / 1000.0).max(0.0);
        let mut end_secs = (*end_ms as f32 / 1000.0).max(start_secs + 0.001);
        if !end_secs.is_finite() {
            end_secs = start_secs + 0.001;
        }

        let (speaker, speaker_confidence, overlaps_segment) =
            assign_speaker_for_span(start_secs, end_secs, segments);
        if overlaps_segment {
            overlap_assigned_words += 1;
        } else {
            unattributed_words += 1;
        }

        words.push(DiarizationWord {
            word: cleaned.to_string(),
            speaker,
            start_secs,
            end_secs,
            speaker_confidence,
            overlaps_segment,
        });
    }

    (words, overlap_assigned_words, unattributed_words)
}

fn assign_speaker_for_span(
    start_secs: f32,
    end_secs: f32,
    segments: &[DiarizationSegment],
) -> (String, Option<f32>, bool) {
    if segments.is_empty() {
        return (UNKNOWN_SPEAKER.to_string(), None, false);
    }

    let word_span = (end_secs - start_secs).max(0.001);

    let mut best_overlap = 0.0f32;
    let mut best_overlap_ratio = 0.0f32;
    let mut best_specificity = 0.0f32;
    let mut best_confidence = f32::MIN;
    let mut best_segment_span = f32::MAX;
    let mut best_segment: Option<&DiarizationSegment> = None;
    for segment in segments {
        let overlap = interval_overlap(start_secs, end_secs, segment.start_secs, segment.end_secs);
        if overlap <= 0.0 {
            continue;
        }

        let segment_span = (segment.end_secs - segment.start_secs).max(0.001);
        let overlap_ratio = (overlap / word_span).clamp(0.0, 1.0);
        let specificity = (overlap / segment_span).clamp(0.0, 1.0);
        let confidence = segment.confidence.unwrap_or(0.0);

        let overlap_ratio_tied = approx_eq_f32(overlap_ratio, best_overlap_ratio);
        let specificity_tied = approx_eq_f32(specificity, best_specificity);
        let overlap_tied = approx_eq_f32(overlap, best_overlap);
        let confidence_tied = approx_eq_f32(confidence, best_confidence);
        let replace = best_segment.is_none()
            || overlap_ratio > best_overlap_ratio
            || (overlap_ratio_tied && specificity > best_specificity)
            || (overlap_ratio_tied && specificity_tied && overlap > best_overlap)
            || (overlap_ratio_tied
                && specificity_tied
                && overlap_tied
                && confidence > best_confidence)
            || (overlap_ratio_tied
                && specificity_tied
                && overlap_tied
                && confidence_tied
                && segment_span < best_segment_span);

        if replace {
            best_overlap = overlap;
            best_overlap_ratio = overlap_ratio;
            best_specificity = specificity;
            best_confidence = confidence;
            best_segment_span = segment_span;
            best_segment = Some(segment);
        }
    }

    if let Some(segment) = best_segment.filter(|_| best_overlap > 0.0) {
        return (segment.speaker.clone(), segment.confidence, true);
    }

    let midpoint = (start_secs + end_secs) * 0.5;
    let nearest = segments
        .iter()
        .min_by(|left, right| {
            span_distance(midpoint, left.start_secs, left.end_secs).total_cmp(&span_distance(
                midpoint,
                right.start_secs,
                right.end_secs,
            ))
        })
        .expect("segments checked non-empty");

    (nearest.speaker.clone(), nearest.confidence, false)
}

fn approx_eq_f32(left: f32, right: f32) -> bool {
    (left - right).abs() <= 1e-6
}

fn span_distance(point: f32, start: f32, end: f32) -> f32 {
    if point < start {
        start - point
    } else if point > end {
        point - end
    } else {
        0.0
    }
}

fn interval_overlap(a_start: f32, a_end: f32, b_start: f32, b_end: f32) -> f32 {
    (a_end.min(b_end) - a_start.max(b_start)).max(0.0)
}

/// Utterances of consecutive same-speaker words. An utterance's text is the
/// transcript's own text for its words: `word_texts[i]` renders word `i`
/// (its source text with the spacing before it), so punctuation, casing and
/// unspaced scripts (CJK) are kept.
fn build_utterances(words: &[DiarizationWord], word_texts: &[String]) -> Vec<DiarizationUtterance> {
    if words.is_empty() {
        return Vec::new();
    }
    let render = |first: usize, last: usize| {
        (first..=last)
            .map(|idx| match word_texts.get(idx) {
                Some(text) => text.clone(),
                None => format!(" {}", words[idx].word),
            })
            .collect::<String>()
            .trim()
            .to_string()
    };

    let mut utterances = Vec::new();
    let mut current = DiarizationUtterance {
        speaker: words[0].speaker.clone(),
        start_secs: words[0].start_secs,
        end_secs: words[0].end_secs,
        text: String::new(),
        word_start: 0,
        word_end: 0,
    };

    for (idx, word) in words.iter().enumerate().skip(1) {
        let gap = (word.start_secs - current.end_secs).max(0.0);
        let same_speaker = word.speaker == current.speaker;

        if same_speaker && gap <= MAX_UTTERANCE_GAP_SECS {
            current.end_secs = current.end_secs.max(word.end_secs);
            current.word_end = idx;
            continue;
        }

        utterances.push(current);
        current = DiarizationUtterance {
            speaker: word.speaker.clone(),
            start_secs: word.start_secs,
            end_secs: word.end_secs,
            text: String::new(),
            word_start: idx,
            word_end: idx,
        };
    }

    utterances.push(current);
    let mut utterances = merge_adjacent_same_speaker_utterances(utterances);
    for utterance in &mut utterances {
        utterance.text = render(utterance.word_start, utterance.word_end);
    }
    utterances
}

/// Merge consecutive utterances of one speaker. Texts are rendered after
/// merging, from the merged word range.
fn merge_adjacent_same_speaker_utterances(
    utterances: Vec<DiarizationUtterance>,
) -> Vec<DiarizationUtterance> {
    let mut merged: Vec<DiarizationUtterance> = Vec::with_capacity(utterances.len());
    for utterance in utterances {
        if let Some(previous) = merged.last_mut() {
            if previous.speaker == utterance.speaker {
                previous.end_secs = previous.end_secs.max(utterance.end_secs);
                previous.word_end = utterance.word_end;
                continue;
            }
        }
        merged.push(utterance);
    }
    merged
}

fn stabilize_word_speakers(words: &mut [DiarizationWord]) {
    if words.len() < 3 {
        return;
    }

    let mut changed = true;
    while changed {
        changed = false;
        let runs = collect_speaker_runs(words);
        if runs.len() < 3 {
            break;
        }

        // From the first run: an edge run has one neighbour, and a leading
        // fragment can only be absorbed into the run after it.
        for run_idx in 0..runs.len() {
            let run = runs[run_idx];
            let avg_confidence = average_run_confidence(words, run.start, run.end);
            let duration = (words[run.end].end_secs - words[run.start].start_secs).max(0.0);
            let word_count = run.end + 1 - run.start;

            let previous = run_idx.checked_sub(1).map(|idx| runs[idx]);
            let next = runs.get(run_idx + 1).copied();

            let target_speaker = if let (Some(left), Some(right)) = (previous, next) {
                if left.speaker(words) == right.speaker(words)
                    && left.speaker(words) != run.speaker(words)
                    && word_count <= MAX_FRAGMENT_WORDS
                    && duration <= MAX_FRAGMENT_DURATION_SECS
                {
                    let neighbor_confidence = average_run_confidence(words, left.start, right.end);
                    if avg_confidence + FRAGMENT_CONFIDENCE_MARGIN < neighbor_confidence {
                        Some(left.speaker(words).to_string())
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else if let Some(left) = previous {
                if word_count <= 2
                    && duration <= 1.0
                    && left.speaker(words) != run.speaker(words)
                    && !run_has_overlap_backed_words(words, run.start, run.end)
                    && avg_confidence + FRAGMENT_CONFIDENCE_MARGIN
                        < average_run_confidence(words, left.start, left.end)
                {
                    Some(left.speaker(words).to_string())
                } else {
                    None
                }
            } else if let Some(right) = next {
                if word_count <= 2
                    && duration <= 1.0
                    && right.speaker(words) != run.speaker(words)
                    && !run_has_overlap_backed_words(words, run.start, run.end)
                    && avg_confidence + FRAGMENT_CONFIDENCE_MARGIN
                        < average_run_confidence(words, right.start, right.end)
                {
                    Some(right.speaker(words).to_string())
                } else {
                    None
                }
            } else {
                None
            };

            if let Some(target_speaker) = target_speaker {
                for word in &mut words[run.start..=run.end] {
                    word.speaker = target_speaker.clone();
                }
                changed = true;
                break;
            }
        }
    }
}

fn normalize_phrase_speakers(words: &mut [DiarizationWord], segments: &[DiarizationSegment]) {
    let phrases = collect_phrase_runs(words);
    if phrases.len() < 2 {
        return;
    }

    let mut phrase_speakers = phrases
        .iter()
        .map(|phrase| choose_phrase_speaker(words, *phrase))
        .collect::<Vec<_>>();
    let phrase_normalizable = phrases
        .iter()
        .map(|phrase| phrase_can_be_normalized(words, *phrase))
        .collect::<Vec<_>>();

    for phrase_idx in 1..phrases.len().saturating_sub(1) {
        if !phrase_normalizable[phrase_idx] {
            continue;
        }
        let previous = &phrase_speakers[phrase_idx - 1];
        let current = &phrase_speakers[phrase_idx];
        let next = &phrase_speakers[phrase_idx + 1];
        if previous != current || next != current {
            continue;
        }

        if let Some(alternate) =
            fully_covering_alternate_speaker(words, phrases[phrase_idx], segments, current)
        {
            phrase_speakers[phrase_idx] = alternate;
        }
    }

    for ((phrase, speaker), normalizable) in phrases
        .iter()
        .zip(phrase_speakers.iter())
        .zip(phrase_normalizable.iter())
    {
        if !*normalizable {
            continue;
        }
        for word in &mut words[phrase.start..=phrase.end] {
            word.speaker = speaker.clone();
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct SpeakerRun {
    start: usize,
    end: usize,
}

impl SpeakerRun {
    fn speaker<'a>(&self, words: &'a [DiarizationWord]) -> &'a str {
        &words[self.start].speaker
    }
}

fn collect_speaker_runs(words: &[DiarizationWord]) -> Vec<SpeakerRun> {
    if words.is_empty() {
        return Vec::new();
    }

    let mut runs = Vec::new();
    let mut start = 0usize;
    for idx in 1..words.len() {
        if words[idx].speaker != words[start].speaker {
            runs.push(SpeakerRun {
                start,
                end: idx - 1,
            });
            start = idx;
        }
    }
    runs.push(SpeakerRun {
        start,
        end: words.len() - 1,
    });
    runs
}

fn collect_phrase_runs(words: &[DiarizationWord]) -> Vec<SpeakerRun> {
    if words.is_empty() {
        return Vec::new();
    }

    let mut phrases = Vec::new();
    let mut start = 0usize;
    for idx in 1..words.len() {
        let previous = &words[idx - 1];
        let current = &words[idx];
        let gap = (current.start_secs - previous.end_secs).max(0.0);
        let starts_capitalized = current
            .word
            .chars()
            .next()
            .map(|ch| ch.is_ascii_uppercase())
            .unwrap_or(false);

        if gap >= PHRASE_HARD_GAP_SECS || (gap >= PHRASE_CAPITALIZED_GAP_SECS && starts_capitalized)
        {
            phrases.push(SpeakerRun {
                start,
                end: idx - 1,
            });
            start = idx;
        }
    }
    phrases.push(SpeakerRun {
        start,
        end: words.len() - 1,
    });
    phrases
}

fn choose_phrase_speaker(words: &[DiarizationWord], phrase: SpeakerRun) -> String {
    let phrase_words = &words[phrase.start..=phrase.end];
    let runs = collect_speaker_runs(phrase_words);

    if runs.len() == 1 {
        return phrase_words[runs[0].start].speaker.clone();
    }

    #[derive(Debug, Clone, Copy)]
    struct PhraseSpeakerStats {
        total_duration: f32,
        confidence_weighted_duration: f32,
        total_words: usize,
        first_run_idx: usize,
    }

    let mut stats = HashMap::<&str, PhraseSpeakerStats>::new();
    for (run_idx, run) in runs.iter().copied().enumerate() {
        let duration = run_duration_secs(phrase_words, run);
        let avg_confidence = average_run_confidence(phrase_words, run.start, run.end);
        let word_count = run.end + 1 - run.start;
        stats
            .entry(run.speaker(phrase_words))
            .and_modify(|entry| {
                entry.total_duration += duration;
                entry.confidence_weighted_duration += duration * avg_confidence;
                entry.total_words += word_count;
            })
            .or_insert(PhraseSpeakerStats {
                total_duration: duration,
                confidence_weighted_duration: duration * avg_confidence,
                total_words: word_count,
                first_run_idx: run_idx,
            });
    }

    stats
        .into_iter()
        .max_by(|left, right| {
            left.1
                .total_duration
                .total_cmp(&right.1.total_duration)
                .then(
                    left.1
                        .confidence_weighted_duration
                        .total_cmp(&right.1.confidence_weighted_duration),
                )
                .then(left.1.total_words.cmp(&right.1.total_words))
                .then(right.1.first_run_idx.cmp(&left.1.first_run_idx))
        })
        .map(|(speaker, _)| speaker.to_string())
        .unwrap_or_else(|| phrase_words[runs[0].start].speaker.clone())
}

fn phrase_can_be_normalized(words: &[DiarizationWord], phrase: SpeakerRun) -> bool {
    let phrase_words = &words[phrase.start..=phrase.end];
    let mut distinct_speakers = HashMap::<&str, usize>::new();
    for word in phrase_words {
        *distinct_speakers.entry(word.speaker.as_str()).or_insert(0) += 1;
        if distinct_speakers.len() > 2 {
            return false;
        }
    }

    let runs = collect_speaker_runs(phrase_words);
    if runs.len() <= 1 {
        return true;
    }

    // A simple two-run phrase is usually a real turn boundary, not a fragment
    // that should be collapsed into one speaker.
    if runs.len() == 2 {
        return false;
    }

    if runs[0].speaker(phrase_words) != runs[runs.len() - 1].speaker(phrase_words) {
        return false;
    }

    runs.iter()
        .copied()
        .skip(1)
        .take(runs.len().saturating_sub(2))
        .all(|run| {
            let word_count = run.end + 1 - run.start;
            let duration = run_duration_secs(phrase_words, run);
            word_count <= MAX_FRAGMENT_WORDS && duration <= MAX_FRAGMENT_DURATION_SECS
        })
}

fn run_duration_secs(words: &[DiarizationWord], run: SpeakerRun) -> f32 {
    (words[run.end].end_secs - words[run.start].start_secs).max(0.0)
}

fn fully_covering_alternate_speaker(
    words: &[DiarizationWord],
    phrase: SpeakerRun,
    segments: &[DiarizationSegment],
    current_speaker: &str,
) -> Option<String> {
    let phrase_start = words[phrase.start].start_secs;
    let phrase_end = words[phrase.end].end_secs;
    let duration = (phrase_end - phrase_start).max(0.001);
    if duration <= 0.0 {
        return None;
    }

    let mut coverage_by_speaker = HashMap::<&str, f32>::new();
    for segment in segments {
        let overlap = interval_overlap(
            phrase_start,
            phrase_end,
            segment.start_secs,
            segment.end_secs,
        );
        if overlap > 0.0 {
            *coverage_by_speaker
                .entry(segment.speaker.as_str())
                .or_insert(0.0) += overlap;
        }
    }

    coverage_by_speaker
        .into_iter()
        .filter(|(speaker, _)| *speaker != current_speaker)
        .filter(|(_, coverage)| *coverage >= duration * 0.9)
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .map(|(speaker, _)| speaker.to_string())
}

fn average_run_confidence(words: &[DiarizationWord], start: usize, end: usize) -> f32 {
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for word in &words[start..=end] {
        if let Some(confidence) = word.speaker_confidence {
            if confidence.is_finite() {
                sum += confidence.clamp(0.0, 1.0);
                count += 1;
            }
        }
    }

    if count == 0 {
        0.0
    } else {
        sum / count as f32
    }
}

fn run_has_overlap_backed_words(words: &[DiarizationWord], start: usize, end: usize) -> bool {
    words[start..=end].iter().any(|word| word.overlaps_segment)
}

fn format_utterance_transcript(utterances: &[DiarizationUtterance]) -> String {
    utterances
        .iter()
        .map(|utterance| {
            format!(
                "{} [{:.2}s - {:.2}s]: {}",
                utterance.speaker, utterance.start_secs, utterance.end_secs, utterance.text
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn secs_to_ms(value: f32) -> u32 {
    if !value.is_finite() || value <= 0.0 {
        0
    } else {
        (value * 1000.0).round() as u32
    }
}

pub(super) fn samples_to_ms(sample_index: usize, sample_rate: u32) -> u32 {
    if sample_rate == 0 {
        return 0;
    }
    ((sample_index as u64 * 1000) / sample_rate as u64) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diarization_resolves_voxtral_through_its_loaded_asr_registry() {
        assert_eq!(
            diarization_asr_registry_route(ModelVariant::VoxtralMini4BRealtime2602),
            DiarizationAsrRegistryRoute::Voxtral
        );
        assert_eq!(
            diarization_asr_registry_route(ModelVariant::WhisperLargeV3Turbo),
            DiarizationAsrRegistryRoute::Native
        );
        assert_eq!(
            diarization_asr_registry_route(ModelVariant::ParakeetTdt06BV3),
            DiarizationAsrRegistryRoute::Native
        );
    }

    async fn admitted_diarization_copy_job(
        runtime: &RuntimeService,
        request_id: &str,
        input_bytes: usize,
        deadline: Option<std::time::Instant>,
    ) -> JobLease {
        let mut context = RuntimeRequestContext::default();
        if let Some(deadline) = deadline {
            context = context.with_deadline(deadline);
        }
        let mut spec = runtime.coordinator_job_for_input(
            request_id,
            CoordinatorLane::Pipeline,
            context,
            input_bytes,
        );
        spec.resources = spec
            .resources
            .checked_add(diarization_decode_resources(
                runtime.backend_context().backend_kind,
            ))
            .expect("combined resources");
        runtime
            .coordinator
            .admit_observed(
                spec,
                JobResourceObservation::host(
                    u64::try_from(input_bytes).expect("test input fits u64"),
                ),
            )
            .await
            .expect("diarization copy admission")
    }

    #[test]
    fn diarization_decode_workspace_maps_to_physical_backend_memory() {
        let cpu = diarization_decode_resources(crate::backends::BackendKind::Cpu);
        let metal = diarization_decode_resources(crate::backends::BackendKind::Metal);
        let cuda = diarization_decode_resources(crate::backends::BackendKind::Cuda);
        let allowance = ResourceAmount::Known(DIARIZATION_AUDIO_WORKSPACE_BYTES);

        assert_eq!(cpu.host_bytes, allowance);
        assert_eq!(metal.unified_bytes, allowance);
        assert_eq!(cuda.host_bytes, allowance);
        assert_eq!(cpu.device_bytes, ResourceAmount::Known(0));
        assert_eq!(cpu.unified_bytes, ResourceAmount::Known(0));
        assert_eq!(metal.host_bytes, ResourceAmount::Known(0));
        assert_eq!(metal.device_bytes, ResourceAmount::Known(0));
        assert_eq!(cuda.unified_bytes, ResourceAmount::Known(0));
        assert_eq!(cuda.device_bytes, ResourceAmount::Known(0));
    }

    /// ADM-1: a diarization job is priced like an ASR job (the generic
    /// transient plus the audio allowance). The Sortformer workspace is not
    /// in it: a 3 s request used to carry the one-hour, max-over-topologies
    /// v2.1 workspace (about 2.9 GiB on Metal) and was refused live.
    #[test]
    fn diarization_job_does_not_carry_the_one_hour_sortformer_workspace() {
        use crate::backends::BackendKind;
        let three_seconds = 3 * PIPELINE_SAMPLE_RATE as usize;
        let input_bytes = 44 + three_seconds * 2;
        for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            let transient = crate::runtime::service::transient_resources(backend, input_bytes);
            let job = transient
                .checked_add(diarization_decode_resources(backend))
                .unwrap();
            let known = |amount: ResourceAmount| match amount {
                ResourceAmount::Known(bytes) => bytes,
                ResourceAmount::Unknown => panic!("{backend:?} job has an unknown amount"),
            };
            for variant in [
                ModelVariant::DiarStreamingSortformer4SpkV21,
                ModelVariant::Nemotron3Diarization,
            ] {
                let workspace =
                    crate::models::architectures::sortformer::diarization::default_profile_workspace_estimate(
                        variant,
                        three_seconds,
                        backend == BackendKind::Cuda,
                    )
                    .unwrap();
                let (job_bytes, transient_bytes, workspace_bytes) = match backend {
                    BackendKind::Cpu => (
                        known(job.host_bytes),
                        known(transient.host_bytes),
                        workspace.host_bytes + workspace.accelerator_bytes,
                    ),
                    BackendKind::Metal => (
                        known(job.unified_bytes),
                        known(transient.unified_bytes),
                        workspace.host_bytes + workspace.accelerator_bytes,
                    ),
                    BackendKind::Cuda => {
                        assert!(
                            known(job.device_bytes)
                                <= known(transient.device_bytes) + workspace.accelerator_bytes,
                            "{variant}: CUDA device"
                        );
                        (
                            known(job.host_bytes),
                            known(transient.host_bytes),
                            workspace.host_bytes,
                        )
                    }
                };
                assert!(
                    job_bytes
                        <= transient_bytes + DIARIZATION_AUDIO_WORKSPACE_BYTES + workspace_bytes,
                    "{backend:?}/{variant}: job {job_bytes} B exceeds the transient \
                     {transient_bytes} B + 512 MiB + the loaded model's 3 s workspace \
                     {workspace_bytes} B"
                );
            }
        }
    }

    /// PIPE-15: a load that fails while the overlapped diarizer runs (it
    /// may hold the memory) waits for the diarizer to finish and tries once
    /// more; a load that fails after the diarizer finished is not retried.
    #[tokio::test]
    async fn a_load_failing_while_diarizing_retries_after_the_diarizer() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let refused = || Error::Overloaded("insufficient live memory".to_string());

        let (done, mut diarized) = tokio::sync::watch::channel(false);
        let attempts = Arc::new(AtomicUsize::new(0));
        let load = {
            let attempts = attempts.clone();
            move || {
                let attempt = attempts.fetch_add(1, Ordering::AcqRel);
                async move {
                    if attempt == 0 {
                        Err(refused())
                    } else {
                        Ok(attempt)
                    }
                }
            }
        };
        let retried =
            tokio::spawn(async move { load_after_diarizer_on_failure(&mut diarized, load).await });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!retried.is_finished(), "the retry waits for the diarizer");
        assert_eq!(attempts.load(Ordering::Acquire), 1);
        done.send(true).unwrap();
        assert_eq!(retried.await.unwrap().unwrap(), 1);
        assert_eq!(attempts.load(Ordering::Acquire), 2);

        let (_done, mut diarized) = tokio::sync::watch::channel(true);
        let error =
            load_after_diarizer_on_failure(&mut diarized, || async { Err::<(), _>(refused()) })
                .await
                .expect_err("no retry once the diarizer finished");
        assert!(crate::engine::resources::is_held_capacity_refusal(&error));
    }

    /// PIPE-8, D4: an interactive job submitted while a long diarization
    /// runs gets the Metal permit before the diarization finishes. With the
    /// CPU Sortformer on the host-model lane it runs at once, mid-stage; on
    /// the device permit (CUDA, CPU) it runs between two bounded stages.
    #[tokio::test]
    async fn interactive_work_gets_the_device_before_diarization_finishes() {
        use crate::engine::{ModelInstanceId, Priority, WorkloadClass};
        use crate::runtime::adapters::{LoadedModelBundle, RuntimeAdapterRegistry};
        use crate::runtime::coordinator::JobSpec;
        use std::sync::atomic::{AtomicUsize, Ordering};

        const STAGES: usize = 3;
        let spec = |id: &str, workload_class| {
            let mut resources = ResourceVector::zero();
            resources.unified_bytes = ResourceAmount::Known(64 * 1024 * 1024);
            JobSpec {
                request_id: id.to_string(),
                lane: CoordinatorLane::Pipeline,
                priority: Priority::Normal,
                workload_class,
                deadline: None,
                resources,
            }
        };
        for host_model in [true, false] {
            let coordinator = Arc::new(InferenceCoordinator::new(
                crate::backends::BackendKind::Metal,
                1,
                8,
            ));
            let adapters = RuntimeAdapterRegistry::built_in_with_execution_limits(1, 4).unwrap();
            // Any scalar loaded stage will do: the stages under test are stubs.
            let contract = LoadedModelBundle::bind(
                &adapters,
                coordinator.execution_group_id(),
                ModelInstanceId::new(7),
                ModelVariant::Kokoro82M,
                crate::backends::BackendKind::Metal,
            )
            .unwrap()
            .contract(CapabilityKind::StreamingTts, false)
            .unwrap();
            let diarization_job = coordinator
                .admit(spec("diarization", WorkloadClass::Background))
                .await
                .unwrap();
            // Each stage blocks until the test releases it.
            let (release, released) = std::sync::mpsc::channel::<()>();
            let released = Arc::new(Mutex::new(released));
            let (started, done) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
            let step = {
                let (started, done) = (started.clone(), done.clone());
                move || {
                    started.fetch_add(1, Ordering::AcqRel);
                    released.lock().unwrap().recv().unwrap();
                    Ok(done.fetch_add(1, Ordering::AcqRel) + 1 == STAGES)
                }
            };
            let diarization = tokio::spawn({
                let coordinator = coordinator.clone();
                async move {
                    run_bounded_stages(
                        &coordinator,
                        &diarization_job,
                        &contract,
                        WorkUnit::AtomicJob {
                            kind: "tts".to_string(),
                        },
                        host_model,
                        step,
                    )
                    .await
                }
            });
            while started.load(Ordering::Acquire) == 0 {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }

            let interactive = tokio::spawn({
                let coordinator = coordinator.clone();
                async move {
                    let job = coordinator
                        .admit(spec("chat", WorkloadClass::Interactive))
                        .await?;
                    coordinator.run_stage(&job, async { Ok(()) }).await
                }
            });
            if host_model {
                // The first stage still blocks: the permit was never taken.
                tokio::time::timeout(Duration::from_secs(10), interactive)
                    .await
                    .expect("interactive work waited on a host-model stage")
                    .unwrap()
                    .unwrap();
                assert_eq!(done.load(Ordering::Acquire), 0);
                assert_eq!(coordinator.snapshot().host_model_stages_total, 1);
            } else {
                tokio::time::sleep(Duration::from_millis(50)).await;
                assert!(!interactive.is_finished(), "the stage holds the permit");
                release.send(()).unwrap();
                tokio::time::timeout(Duration::from_secs(10), interactive)
                    .await
                    .expect("interactive work waited for the whole diarization")
                    .unwrap()
                    .unwrap();
                assert_eq!(done.load(Ordering::Acquire), 1);
                assert_eq!(coordinator.snapshot().host_model_stages_total, 0);
            }
            assert!(!diarization.is_finished());
            for _ in done.load(Ordering::Acquire)..STAGES {
                release.send(()).unwrap();
            }
            diarization.await.unwrap().unwrap();
            assert_eq!(started.load(Ordering::Acquire), STAGES);
        }
    }

    /// The diarize stage reserves the loaded model's workspace, records it
    /// while it is allocated, and releases it when the stage returns,
    /// whether it succeeds or fails.
    #[tokio::test]
    async fn diarize_stage_workspace_lease_is_released_after_the_stage() {
        use crate::runtime::coordinator::{InferenceCoordinator, JobSpec};
        let coordinator = Arc::new(InferenceCoordinator::new(
            crate::backends::BackendKind::Metal,
            1,
            4,
        ));
        let context = RuntimeRequestContext::default();
        let job = coordinator
            .admit(JobSpec {
                request_id: "diarize-stage-lease".to_string(),
                lane: CoordinatorLane::Pipeline,
                priority: context.priority,
                workload_class: context.workload_class,
                deadline: None,
                resources: diarization_decode_resources(crate::backends::BackendKind::Metal),
            })
            .await
            .unwrap();
        let workspace = SortformerWorkspaceEstimate {
            host_bytes: 3 << 20,
            accelerator_bytes: 40 << 20,
        };
        let workspace_bytes = workspace.host_bytes + workspace.accelerator_bytes;
        let authority = coordinator.resource_authority();
        let before = authority.snapshot();
        let plus = |amount: ResourceAmount| match amount {
            ResourceAmount::Known(bytes) => ResourceAmount::Known(bytes + workspace_bytes),
            ResourceAmount::Unknown => panic!("unknown authority amount"),
        };

        diarize_under_workspace_lease(&coordinator, &job, workspace, |observer| {
            let reserved = authority.snapshot();
            assert_eq!(
                reserved.reserved.unified_bytes,
                plus(before.reserved.unified_bytes)
            );
            assert_eq!(
                reserved.pending.unified_bytes,
                plus(before.pending.unified_bytes)
            );
            observer(SortformerWorkspaceEvent::Materialized { workspace })?;
            let allocated = authority.snapshot();
            assert_eq!(
                allocated.materialized.unified_bytes,
                plus(before.materialized.unified_bytes)
            );
            assert_eq!(
                allocated.pending.unified_bytes,
                before.pending.unified_bytes
            );
            observer(SortformerWorkspaceEvent::Releasing { workspace })?;
            let releasing = authority.snapshot();
            assert_eq!(
                releasing.materialized.unified_bytes,
                before.materialized.unified_bytes
            );
            assert_eq!(
                releasing.pending.unified_bytes,
                plus(before.pending.unified_bytes)
            );
            Ok(())
        })
        .unwrap();
        let after = authority.snapshot();
        assert_eq!(after.reserved, before.reserved);
        assert_eq!(after.pending, before.pending);
        assert_eq!(after.reservations, before.reservations);

        // A failed stage, or one whose model allocated a different workspace
        // than it estimated, still releases the claim.
        let failed: Result<()> =
            diarize_under_workspace_lease(&coordinator, &job, workspace, |observer| {
                observer(SortformerWorkspaceEvent::Materialized { workspace })?;
                Err(Error::InferenceError("stage failed".to_string()))
            });
        assert!(failed.is_err());
        let mismatched = diarize_under_workspace_lease(&coordinator, &job, workspace, |observer| {
            observer(SortformerWorkspaceEvent::Materialized {
                workspace: SortformerWorkspaceEstimate {
                    host_bytes: workspace.host_bytes + 1,
                    ..workspace
                },
            })
        });
        assert!(matches!(mismatched, Err(Error::InferenceError(_))));
        assert_eq!(authority.snapshot().reserved, before.reserved);
        assert_eq!(authority.snapshot().reservations, before.reservations);
    }

    /// ADM-2: once the audio is decoded and diarized, the job holds only
    /// what it retains (the decoded audio, the retained input and the
    /// generic base workspace). The pipeline's ASR load then sees live
    /// memory less that base, not less the 512 MiB decode allowance or a
    /// Sortformer workspace.
    #[tokio::test]
    async fn after_diarizing_the_job_leaves_live_memory_to_the_asr_load() {
        use crate::backends::BackendKind;
        use crate::engine::{ReservationClass, ReservationOwner};
        use crate::runtime::coordinator::{InferenceCoordinator, JobSpec};
        const LIVE: u64 = 64 * 1024 * 1024 * 1024;
        let samples = 3 * PIPELINE_SAMPLE_RATE as usize;
        let input_bytes = 44 + samples * 2;
        let steady = JobResourceObservation::host((2 * input_bytes + samples * 4) as u64);
        let peak = JobResourceObservation::host((3 * input_bytes + samples * 4) as u64);
        for backend in [BackendKind::Metal, BackendKind::Cuda] {
            let coordinator = Arc::new(InferenceCoordinator::new(backend, 1, 4));
            let context = RuntimeRequestContext::default();
            let job = coordinator
                .admit_observed(
                    JobSpec {
                        request_id: "diarize-then-load-asr".to_string(),
                        lane: CoordinatorLane::Pipeline,
                        priority: context.priority,
                        workload_class: context.workload_class,
                        deadline: None,
                        resources: crate::runtime::service::transient_resources(
                            backend,
                            input_bytes,
                        )
                        .checked_add(diarization_decode_resources(backend))
                        .unwrap(),
                    },
                    JobResourceObservation::host(input_bytes as u64),
                )
                .await
                .unwrap();
            settle_decoded_job(&job, backend, peak, steady, Some(String::new())).unwrap();
            let workspace =
                crate::models::architectures::sortformer::diarization::default_profile_workspace_estimate(
                    ModelVariant::DiarStreamingSortformer4SpkV21,
                    samples,
                    backend == BackendKind::Cuda,
                )
                .unwrap();
            diarize_under_workspace_lease(&coordinator, &job, workspace, |observer| {
                observer(SortformerWorkspaceEvent::Materialized { workspace })?;
                observer(SortformerWorkspaceEvent::Releasing { workspace })
            })
            .unwrap();

            let authority = coordinator.resource_authority();
            let base = crate::runtime::service::transient_resources(backend, 0);
            let known = |amount: ResourceAmount| match amount {
                ResourceAmount::Known(bytes) => bytes,
                ResourceAmount::Unknown => panic!("{backend:?}: unknown amount"),
            };
            if backend == BackendKind::Metal {
                // The live pre-load check of the pipeline's ASR load.
                let headroom = authority
                    .live_preload_headroom_bytes(backend)
                    .unwrap()
                    .expect("a live sample");
                assert_eq!(headroom, LIVE - known(base.unified_bytes));
            } else {
                // A guarded model reservation of all memory but the retained
                // audio and the base.
                let asr = authority.reserve(
                    ReservationOwner::new(ReservationClass::Model, "asr"),
                    ResourceVector {
                        host_bytes: ResourceAmount::Known(
                            LIVE - steady.host_bytes - known(base.host_bytes),
                        ),
                        device_bytes: ResourceAmount::Known(LIVE - known(base.device_bytes)),
                        ..ResourceVector::zero()
                    },
                );
                assert!(asr.is_ok(), "{asr:?}");
            }
        }
    }

    /// LIVE-11: a finished pipeline job leaves nothing behind. Its admission
    /// (with the capacity wait and a cancellation handle), its decode
    /// settlement and its Sortformer stage lease all return the authority to
    /// where it started, so the memory live refusals reported as held at
    /// idle is not the job's: it is loaded models' reserved, not yet
    /// allocated, claims.
    #[tokio::test]
    async fn a_finished_pipeline_job_releases_every_claim() {
        use crate::backends::BackendKind;
        use crate::runtime::coordinator::{InferenceCoordinator, JobCancellation, JobSpec};
        let samples = 3 * PIPELINE_SAMPLE_RATE as usize;
        let input_bytes = 44 + samples * 2;
        for backend in [BackendKind::Metal, BackendKind::Cuda, BackendKind::Cpu] {
            let coordinator = Arc::new(InferenceCoordinator::new(backend, 1, 4));
            let authority = coordinator.resource_authority();
            let before = authority.snapshot();
            let context = RuntimeRequestContext::default();
            let spec = JobSpec {
                request_id: "finished-pipeline".to_string(),
                lane: CoordinatorLane::Pipeline,
                priority: context.priority,
                workload_class: context.workload_class,
                deadline: None,
                resources: crate::runtime::service::transient_resources(backend, input_bytes)
                    .checked_add(diarization_decode_resources(backend))
                    .unwrap(),
            };
            let job = wait_for_held_capacity(Duration::from_secs(1), || {
                coordinator.admit_observed(
                    spec.clone(),
                    JobResourceObservation::host(input_bytes as u64),
                )
            })
            .await
            .unwrap()
            .with_cancellation(JobCancellation::new());
            let steady = JobResourceObservation::host((2 * input_bytes + samples * 4) as u64);
            let peak = JobResourceObservation::host((3 * input_bytes + samples * 4) as u64);
            settle_decoded_job(&job, backend, peak, steady, Some(String::new())).unwrap();
            let workspace =
                crate::models::architectures::sortformer::diarization::default_profile_workspace_estimate(
                    ModelVariant::DiarStreamingSortformer4SpkV21,
                    samples,
                    backend == BackendKind::Cuda,
                )
                .unwrap();
            let stage_coordinator = coordinator.clone();
            let stage_job = job.clone();
            coordinator
                .run_blocking_stage(&job, move || {
                    diarize_under_workspace_lease(
                        &stage_coordinator,
                        &stage_job,
                        workspace,
                        |observer| {
                            observer(SortformerWorkspaceEvent::Materialized { workspace })?;
                            observer(SortformerWorkspaceEvent::Releasing { workspace })
                        },
                    )
                })
                .await
                .unwrap();
            drop(job);

            let after = authority.snapshot();
            assert_eq!(after.reserved, before.reserved, "{backend:?}");
            assert_eq!(after.pending, before.pending, "{backend:?}");
            assert_eq!(after.materialized, before.materialized, "{backend:?}");
            assert_eq!(after.reservations, before.reservations, "{backend:?}");
            let snapshot = coordinator.snapshot();
            assert_eq!(snapshot.active_jobs, 0, "{backend:?}");
            assert_eq!(snapshot.active_executions, 0, "{backend:?}");
        }
    }

    #[tokio::test]
    async fn pipeline_rejects_an_unrunnable_asr_or_aligner_before_diarizing() {
        // No checkpoints exist here: an id that reached Sortformer would fail
        // with a load or decode error instead of naming the rejected model.
        let runtime = RuntimeService::new(crate::config::EngineConfig {
            models_dir: std::env::temp_dir().join("izwi-diarization-pipeline-validation"),
            ..crate::config::EngineConfig::default()
        })
        .expect("runtime");
        let audio = vec![7_u8; 64];
        let config = DiarizationConfig::default();
        for (asr, aligner, rejected) in [
            (Some("bogus"), None, "bogus"),
            (
                Some("LFM2.5-Audio-1.5B-GGUF"),
                None,
                "LFM2.5-Audio-1.5B-GGUF",
            ),
            (None, Some("Parakeet-TDT-0.6B-v3"), "not a forced aligner"),
        ] {
            let error = runtime
                .diarize_with_transcript_bytes(
                    &audio,
                    None,
                    asr,
                    aligner,
                    &config,
                    &DiarizationPipelineOptions::default(),
                )
                .await
                .expect_err("an unrunnable pipeline model must be rejected");
            assert!(
                matches!(&error, Error::InvalidInput(message) if message.contains(rejected)),
                "{asr:?}/{aligner:?}: {error}"
            );
        }
    }

    #[allow(clippy::await_holding_lock)]
    #[tokio::test(flavor = "current_thread")]
    async fn diarization_admitted_copy_yields_between_bounded_quanta() {
        use std::sync::atomic::{AtomicBool, Ordering};
        // The max-pipeline admission below competes with load-transaction
        // residency reservations on the process-global authority; serialize.
        let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let runtime = RuntimeService::new(crate::config::EngineConfig::default()).expect("runtime");
        let input = vec![11_u8; 2 * 1024 * 1024 + 17];
        let job =
            admitted_diarization_copy_job(&runtime, "yielding-diarization-copy", input.len(), None)
                .await;
        let peer_ran = Arc::new(AtomicBool::new(false));
        let task_peer_ran = peer_ran.clone();
        let peer = tokio::spawn(async move {
            task_peer_ran.store(true, Ordering::Release);
        });

        let owned = DiarizationAudioInput::Bytes(&input)
            .into_owned_for_job(&job)
            .await
            .expect("admitted copy");
        let OwnedDiarizationAudioInput::Bytes(copied) = owned else {
            panic!("byte input must remain bytes");
        };

        assert_eq!(copied, input);
        assert!(
            peer_ran.load(Ordering::Acquire),
            "a multi-quantum diarization copy must yield to another Tokio task"
        );
        peer.await.expect("peer task");
    }

    #[allow(clippy::await_holding_lock)]
    #[tokio::test(flavor = "current_thread")]
    async fn diarization_admitted_copy_stops_at_absolute_deadline() {
        // Same max-pipeline admission reservation as the yielding test.
        let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let runtime = RuntimeService::new(crate::config::EngineConfig::default()).expect("runtime");
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(50);
        let job =
            admitted_diarization_copy_job(&runtime, "expired-diarization-copy", 8, Some(deadline))
                .await;
        tokio::time::sleep(std::time::Duration::from_millis(60)).await;

        let result = DiarizationAudioInput::Bytes(&[0_u8; 8])
            .into_owned_for_job(&job)
            .await;
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("copy must honor its absolute deadline"),
        };

        assert!(
            matches!(error, Error::Timeout(request_id) if request_id == "expired-diarization-copy")
        );
    }

    #[test]
    fn sortformer_workspace_authorization_bounds_all_supported_backends() {
        let one_second = production_workspace_authorization(16_000, false).unwrap();
        let production_limit = production_workspace_authorization(MAX_PIPELINE_SAMPLES, false)
            .expect("production workspace");
        let cuda_limit = production_workspace_authorization(MAX_PIPELINE_SAMPLES, true)
            .expect("CUDA production workspace");

        assert!(production_limit.host_bytes > one_second.host_bytes);
        assert!(production_limit.accelerator_bytes >= one_second.accelerator_bytes);
        assert!(cuda_limit.host_bytes > production_limit.host_bytes);
        assert_eq!(
            cuda_limit.accelerator_bytes,
            production_limit.accelerator_bytes
        );
    }

    /// PIPE-14: the one-hour bound applies to the decoded length (the old
    /// guard sat in a 16 kHz -> 16 kHz resample that returned before it).
    #[test]
    fn pipeline_audio_is_bounded_by_its_decoded_length() {
        assert!(validate_pipeline_sample_count(MAX_PIPELINE_SAMPLES).is_ok());
        for samples in [0, MAX_PIPELINE_SAMPLES + 1] {
            assert!(matches!(
                validate_pipeline_sample_count(samples),
                Err(Error::InvalidInput(_))
            ));
        }
        let audio = PipelineAudio::from_canonical(vec![0.25; 16_000]).unwrap();
        assert_eq!(
            (audio.sample_rate, audio.duration_secs),
            (PIPELINE_SAMPLE_RATE, 1.0)
        );
    }

    /// PIPE-14: each job decodes its upload once. The pipeline decodes an
    /// encoded upload once and never decodes samples it is given; PCM over
    /// one hour is refused on its length.
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn pipeline_decodes_each_upload_once() {
        let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        // No checkpoints exist here: every job fails at the Sortformer load,
        // after its audio is admitted.
        let runtime = RuntimeService::new(crate::config::EngineConfig {
            models_dir: std::env::temp_dir().join("izwi-diarization-decode-once"),
            ..crate::config::EngineConfig::default()
        })
        .expect("runtime");
        let config = DiarizationConfig::default();
        let options = DiarizationPipelineOptions::default();
        let one_second = vec![0.1_f32; 16_000];
        let wav = crate::runtime::asr_no_speech::tests::wav_bytes(&one_second, 16_000);

        runtime
            .diarize_with_transcript_bytes(&wav, None, None, None, &config, &options)
            .await
            .expect_err("no Sortformer checkpoint");
        assert_eq!(runtime.diarization_audio_decodes(), 1);

        runtime
            .diarize_with_transcript_samples(one_second, None, None, None, &config, &options)
            .await
            .expect_err("no Sortformer checkpoint");
        assert_eq!(runtime.diarization_audio_decodes(), 1);

        let error = runtime
            .diarize_with_transcript_samples(
                vec![0.0; MAX_PIPELINE_SAMPLES + 1],
                None,
                None,
                None,
                &config,
                &options,
            )
            .await
            .expect_err("over an hour of audio");
        assert!(
            matches!(&error, Error::InvalidInput(message) if message.contains("3600 s limit")),
            "{error}"
        );
        assert_eq!(runtime.coordinator_snapshot().active_jobs, 0);
    }

    #[test]
    fn fallback_word_timings_generates_monotonic_ranges() {
        let words = ["hello", "world", "from", "test"].map(String::from);
        let timings = fallback_word_timings_from_words(&words, 2.0);
        assert_eq!(timings.len(), 4);
        for (idx, (_, start, end)) in timings.iter().enumerate() {
            assert!(end > start, "word {} should have positive duration", idx);
            if idx > 0 {
                assert!(*start >= timings[idx - 1].1);
            }
        }
    }

    /// A model-timed window of `(word, start_ms, end_ms)` words whose text
    /// is the words, space-separated.
    fn model_window(start_ms: u32, end_ms: u32, words: &[(&str, u32, u32)]) -> TimedWindow {
        TimedWindow {
            start_ms,
            end_ms,
            source: AlignmentSource::Model,
            note: None,
            words: words
                .iter()
                .enumerate()
                .map(|(idx, (word, start, end))| WindowWord {
                    word: word.to_string(),
                    text: if idx == 0 {
                        word.to_string()
                    } else {
                        format!(" {word}")
                    },
                    start_ms: *start,
                    end_ms: *end,
                })
                .collect(),
        }
    }

    fn stitched_words(stitched: &StitchedTranscript) -> Vec<&str> {
        stitched
            .words
            .iter()
            .map(|word| word.word.as_str())
            .collect()
    }

    /// PIPE-6 scenario B: chunk 0 stopped at "near the" and chunk 1
    /// re-transcribed the overlap as "the offices near the ...". A one-word
    /// text match used to splice them into "office near the offices near
    /// the station"; stitching at the overlap midpoint keeps each word once.
    #[test]
    fn stitching_keeps_each_overlap_word_once() {
        let windows = [
            model_window(
                0,
                28_100,
                &[
                    ("we", 25_000, 25_300),
                    ("met", 25_400, 25_800),
                    ("at", 25_900, 26_100),
                    ("the", 26_200, 26_400),
                    ("office", 26_500, 27_100),
                    ("near", 27_300, 27_700),
                    ("the", 27_800, 28_000),
                ],
            ),
            model_window(
                26_000,
                56_000,
                &[
                    ("the", 26_200, 26_400),
                    ("offices", 26_500, 27_100),
                    ("near", 27_300, 27_700),
                    ("the", 27_800, 28_000),
                    ("station", 28_200, 29_000),
                    ("today", 29_200, 29_800),
                ],
            ),
        ];
        let stitched = stitch_timed_windows(&windows);
        assert_eq!(
            stitched_words(&stitched),
            ["we", "met", "at", "the", "office", "near", "the", "station", "today"]
        );
        assert_eq!(stitched.text, "we met at the office near the station today");
    }

    /// PIPE-6 scenario C: chunk 0 ends inside "station" and times the cut
    /// "sta" to its edge. Trimming every later word that starts before that
    /// end used to drop "the station today"; the midpoint keeps them.
    #[test]
    fn stitching_keeps_words_after_a_word_cut_at_the_chunk_edge() {
        let windows = [
            model_window(
                0,
                30_000,
                &[
                    ("near", 27_200, 27_600),
                    ("the", 27_800, 28_000),
                    ("sta", 28_200, 30_000),
                ],
            ),
            model_window(
                27_000,
                57_000,
                &[
                    ("near", 27_200, 27_600),
                    ("the", 27_800, 28_000),
                    ("station", 28_200, 29_000),
                    ("today", 29_200, 29_800),
                    ("and", 30_500, 30_800),
                ],
            ),
        ];
        let stitched = stitch_timed_windows(&windows);
        assert_eq!(
            stitched_words(&stitched),
            ["near", "the", "station", "today", "and"]
        );
        let starts = stitched
            .words
            .iter()
            .map(|word| word.start_ms)
            .collect::<Vec<_>>();
        assert_eq!(starts, [27_200, 27_800, 28_200, 29_200, 30_500]);
    }

    /// The stitched text keeps the transcript's punctuation and holds exactly
    /// the stitched words, across windows and for scripts without spaces.
    #[test]
    fn stitched_text_holds_exactly_the_stitched_words() {
        let sources = [
            SampleWindow::new(
                0,
                10 * PIPELINE_SAMPLE_RATE as usize,
                "Hello, world! It's \"me\" here.".to_string(),
                None,
            ),
            SampleWindow::new(
                9 * PIPELINE_SAMPLE_RATE as usize,
                20 * PIPELINE_SAMPLE_RATE as usize,
                "你好。Then more".to_string(),
                None,
            ),
        ];
        let timed = interval_timed_windows(&sources, PIPELINE_SAMPLE_RATE);
        let stitched = stitch_timed_windows(&timed);
        let restitched = SampleWindow::new(0, 0, stitched.text.clone(), None);
        assert_eq!(
            restitched.words(),
            stitched
                .words
                .iter()
                .map(|word| word.word.clone())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            stitched.text,
            "Hello, world! It's \"me\" here.你好。Then more"
        );
    }

    fn turn(speaker: &str, start_secs: f32, end_secs: f32) -> DiarizationSegment {
        DiarizationSegment {
            speaker: speaker.to_string(),
            start_secs,
            end_secs,
            confidence: None,
        }
    }

    /// PIPE-5 (suspicious_cascade): 19 well-aligned 30 s chunks and one the
    /// aligner failed on, whose 6 words are spread 5 s apart. The failed
    /// window's long spans used to flag the whole file and re-time all 196
    /// words; now only the failed window is re-timed.
    #[test]
    fn one_failed_window_does_not_discard_the_other_windows_alignment() {
        const FAILED: usize = 7;
        let windows = (0..20u32)
            .map(|idx| {
                let start_ms = idx * 30_000;
                if idx as usize == FAILED {
                    let words = (0..6u32)
                        .map(|k| WindowWord {
                            word: format!("f{k}"),
                            text: format!(" f{k}"),
                            start_ms: start_ms + k * 5_000,
                            end_ms: start_ms + (k + 1) * 5_000,
                        })
                        .collect();
                    return TimedWindow {
                        start_ms,
                        end_ms: start_ms + 30_000,
                        source: AlignmentSource::Failed,
                        note: Some("alignment failed (boom)".to_string()),
                        words,
                    };
                }
                let words = (0..10u32)
                    .map(|k| {
                        let word_start = start_ms + 1_000 + k * 2_800;
                        (format!("w{idx}x{k}"), word_start, word_start + 400)
                    })
                    .collect::<Vec<_>>();
                let words = words
                    .iter()
                    .map(|(word, start, end)| (word.as_str(), *start, *end))
                    .collect::<Vec<_>>();
                model_window(start_ms, start_ms + 30_000, &words)
            })
            .collect::<Vec<_>>();
        let expected = windows
            .iter()
            .filter(|window| window.source == AlignmentSource::Model)
            .flat_map(|window| window.words.clone())
            .map(|word| (word.word, word.start_ms, word.end_ms))
            .collect::<Vec<_>>();

        let mut warnings = Vec::new();
        let transcript =
            attribute_transcript(windows, &[turn("SPEAKER_00", 0.0, 600.0)], &mut warnings);

        assert_eq!(transcript.words.len(), 196);
        assert_eq!(transcript.model_aligned_words, 190);
        let kept = transcript
            .words
            .iter()
            .filter(|word| word.word.starts_with('w'))
            .map(|word| {
                (
                    word.word.clone(),
                    secs_to_ms(word.start_secs),
                    secs_to_ms(word.end_secs),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(kept, expected, "model timings must survive");
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].contains("210.00-240.00 s"), "{warnings:?}");
        assert!(warnings[0].contains("boom"), "{warnings:?}");
    }

    /// A window whose aligned words fall outside the speaker turns in it is
    /// re-timed over those turns; the window before it keeps its alignment.
    #[test]
    fn poorly_attributed_window_is_retimed_over_its_own_turns() {
        let good = model_window(
            0,
            10_000,
            &[
                ("a", 1_000, 1_400),
                ("b", 2_000, 2_400),
                ("c", 3_000, 3_400),
            ],
        );
        let stray = model_window(
            10_000,
            30_000,
            &[
                ("d", 11_000, 11_400),
                ("e", 12_000, 12_400),
                ("f", 13_000, 13_400),
            ],
        );
        let turns = [
            turn("SPEAKER_00", 0.0, 10.0),
            turn("SPEAKER_01", 20.0, 30.0),
        ];
        let mut warnings = Vec::new();
        let transcript = attribute_transcript(vec![good, stray], &turns, &mut warnings);

        let times = transcript
            .words
            .iter()
            .map(|word| (secs_to_ms(word.start_secs), secs_to_ms(word.end_secs)))
            .collect::<Vec<_>>();
        assert_eq!(
            &times[..3],
            [(1_000, 1_400), (2_000, 2_400), (3_000, 3_400)]
        );
        for (start, end) in &times[3..] {
            assert!(*start >= 20_000 && *end <= 30_000, "{times:?}");
        }
        assert_eq!(transcript.model_aligned_words, 3);
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].contains("10.00-30.00 s"), "{warnings:?}");
    }

    /// A model word stretched over a pause is clamped instead of discarding
    /// the window.
    #[test]
    fn outlier_model_word_is_clamped_to_its_neighbours() {
        let window = model_window(
            0,
            10_000,
            &[("a", 0, 300), ("b", 400, 4_900), ("c", 5_000, 5_300)],
        );
        let mut warnings = Vec::new();
        let transcript = attribute_transcript(
            vec![window],
            &[turn("SPEAKER_00", 0.0, 10.0)],
            &mut warnings,
        );
        let times = transcript
            .words
            .iter()
            .map(|word| (secs_to_ms(word.start_secs), secs_to_ms(word.end_secs)))
            .collect::<Vec<_>>();
        assert_eq!(times, [(0, 300), (400, 1_400), (5_000, 5_300)]);
        assert_eq!(transcript.model_aligned_words, 3);
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    const COUNTING: [&str; 12] = [
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
        "twelve",
    ];

    /// PIPE-4 (fallback_scramble): a turn nested in another (S1 4-6 s inside
    /// S0 0-10 s) used to take its own run of words, and sorting by time then
    /// gave "four nine five ten six". The fallback must keep ASR order.
    #[test]
    fn turn_guided_fallback_keeps_asr_word_order_over_overlapping_turns() {
        let turns = [
            turn("SPEAKER_00", 0.0, 10.0),
            turn("SPEAKER_01", 4.0, 6.0),
            turn("SPEAKER_00", 10.0, 12.0),
        ];
        let words = COUNTING.map(str::to_string);
        let timings = fallback_word_timings_with_segments(&words, &turns, 12.0);
        assert!(
            timings.windows(2).all(|pair| pair[0].2 <= pair[1].1),
            "{timings:?}"
        );

        let failed = TimedWindow {
            start_ms: 0,
            end_ms: 12_000,
            source: AlignmentSource::Failed,
            note: Some("alignment failed".to_string()),
            words: COUNTING
                .iter()
                .enumerate()
                .map(|(idx, word)| WindowWord {
                    word: word.to_string(),
                    text: if idx == 0 {
                        word.to_string()
                    } else {
                        format!(" {word}")
                    },
                    start_ms: idx as u32 * 1_000,
                    end_ms: idx as u32 * 1_000 + 1_000,
                })
                .collect(),
        };
        let mut warnings = Vec::new();
        let transcript = attribute_transcript(vec![failed], &turns, &mut warnings);
        let order = transcript
            .words
            .iter()
            .map(|word| word.word.as_str())
            .collect::<Vec<_>>();
        assert_eq!(order, COUNTING);
        assert_eq!(transcript.asr_text, COUNTING.join(" "));
    }

    /// Attribution never reorders words, even if their times disagree.
    #[test]
    fn attribution_keeps_asr_order() {
        let alignments = vec![
            ("first".to_string(), 2_000, 2_500),
            ("second".to_string(), 1_000, 1_500),
        ];
        let (words, _, _) =
            attribute_words_to_speakers(&alignments, &[turn("SPEAKER_00", 0.0, 3.0)]);
        let order = words
            .iter()
            .map(|word| word.word.as_str())
            .collect::<Vec<_>>();
        assert_eq!(order, ["first", "second"]);
    }

    #[test]
    fn attribution_prefers_overlap_then_nearest() {
        let segments = vec![
            DiarizationSegment {
                speaker: "SPEAKER_00".to_string(),
                start_secs: 0.0,
                end_secs: 1.0,
                confidence: Some(0.9),
            },
            DiarizationSegment {
                speaker: "SPEAKER_01".to_string(),
                start_secs: 2.0,
                end_secs: 3.0,
                confidence: Some(0.8),
            },
        ];

        let aligned = vec![
            ("hello".to_string(), 100, 400),
            ("there".to_string(), 1200, 1300),
            ("friend".to_string(), 2400, 2800),
        ];

        let (words, overlap_count, unattributed) = attribute_words_to_speakers(&aligned, &segments);
        assert_eq!(words.len(), 3);
        assert_eq!(words[0].speaker, "SPEAKER_00");
        assert_eq!(words[2].speaker, "SPEAKER_01");
        assert_eq!(overlap_count, 2);
        assert_eq!(unattributed, 1);
    }

    #[test]
    fn attribution_prefers_more_specific_segment_when_overlaps_tie() {
        let segments = vec![
            DiarizationSegment {
                speaker: "SPEAKER_00".to_string(),
                start_secs: 0.0,
                end_secs: 10.0,
                confidence: Some(0.9),
            },
            DiarizationSegment {
                speaker: "SPEAKER_01".to_string(),
                start_secs: 0.0,
                end_secs: 1.0,
                confidence: Some(0.2),
            },
        ];

        let aligned = vec![("hello".to_string(), 200, 800)];
        let (words, overlap_count, unattributed) = attribute_words_to_speakers(&aligned, &segments);

        assert_eq!(words.len(), 1);
        assert_eq!(words[0].speaker, "SPEAKER_01");
        assert_eq!(overlap_count, 1);
        assert_eq!(unattributed, 0);
    }

    /// Words of `text` in `language` over `secs` of audio, attributed to
    /// `turns`, and their utterances.
    fn rendered_utterances(
        text: &str,
        language: Option<&str>,
        secs: usize,
        turns: &[DiarizationSegment],
    ) -> (Vec<DiarizationWord>, Vec<DiarizationUtterance>) {
        let rate = PIPELINE_SAMPLE_RATE as usize;
        let window = SampleWindow::new(0, secs * rate, text.to_string(), language);
        let timed = interval_timed_windows(&[window], PIPELINE_SAMPLE_RATE);
        let attributed = attribute_transcript(timed, turns, &mut Vec::new());
        let utterances = build_utterances(&attributed.words, &attributed.word_texts);
        (attributed.words, utterances)
    }

    /// PIPE-11: utterances keep the transcript's punctuation and casing, and
    /// CJK joins without separators.
    #[test]
    fn utterances_render_the_transcript_text() {
        let turns = [turn("SPEAKER_00", 0.0, 4.0)];
        let (words, utterances) = rendered_utterances("你好，世界。", Some("Chinese"), 4, &turns);
        assert_eq!(
            words
                .iter()
                .map(|word| word.word.as_str())
                .collect::<Vec<_>>(),
            ["你", "好，", "世", "界。"]
        );
        assert_eq!(utterances.len(), 1);
        assert_eq!(utterances[0].text, "你好，世界。");

        let (_, utterances) = rendered_utterances("Hello, world.", Some("English"), 4, &turns);
        assert_eq!(utterances[0].text, "Hello, world.");
        assert_eq!(
            format_utterance_transcript(&utterances),
            "SPEAKER_00 [0.00s - 4.00s]: Hello, world."
        );
    }

    /// Rendering does not change attribution: each word keeps the speaker of
    /// its timing, and each utterance the text of its words.
    #[test]
    fn utterance_text_follows_speaker_attribution() {
        let turns = [turn("SPEAKER_00", 0.0, 2.6), turn("SPEAKER_01", 2.6, 8.0)];
        let (words, utterances) =
            rendered_utterances("Hello, world. 你好，世界。", Some("Chinese"), 8, &turns);
        assert_eq!(
            words
                .iter()
                .map(|word| (word.word.as_str(), word.speaker.as_str()))
                .collect::<Vec<_>>(),
            [
                ("Hello,", "SPEAKER_00"),
                ("world.", "SPEAKER_00"),
                ("你", "SPEAKER_01"),
                ("好，", "SPEAKER_01"),
                ("世", "SPEAKER_01"),
                ("界。", "SPEAKER_01"),
            ]
        );
        assert_eq!(
            utterances
                .iter()
                .map(|utterance| (utterance.speaker.as_str(), utterance.text.as_str()))
                .collect::<Vec<_>>(),
            [
                ("SPEAKER_00", "Hello, world."),
                ("SPEAKER_01", "你好，世界。")
            ]
        );
    }

    #[test]
    fn build_utterances_merges_small_gaps_for_same_speaker() {
        let words = vec![
            DiarizationWord {
                word: "hello".to_string(),
                speaker: "SPEAKER_00".to_string(),
                start_secs: 0.0,
                end_secs: 0.4,
                speaker_confidence: None,
                overlaps_segment: true,
            },
            DiarizationWord {
                word: "world".to_string(),
                speaker: "SPEAKER_00".to_string(),
                start_secs: 0.5,
                end_secs: 0.8,
                speaker_confidence: None,
                overlaps_segment: true,
            },
            DiarizationWord {
                word: "next".to_string(),
                speaker: "SPEAKER_01".to_string(),
                start_secs: 1.2,
                end_secs: 1.6,
                speaker_confidence: None,
                overlaps_segment: true,
            },
        ];

        let utterances = build_utterances(&words, &[]);
        assert_eq!(utterances.len(), 2);
        assert_eq!(utterances[0].speaker, "SPEAKER_00");
        assert!(utterances[0].text.contains("hello world"));
        assert_eq!(utterances[1].speaker, "SPEAKER_01");
    }

    #[test]
    fn build_utterances_merges_consecutive_same_speaker_runs_after_long_pauses() {
        let words = vec![
            DiarizationWord {
                word: "first".to_string(),
                speaker: "SPEAKER_00".to_string(),
                start_secs: 0.0,
                end_secs: 0.4,
                speaker_confidence: None,
                overlaps_segment: true,
            },
            DiarizationWord {
                word: "block".to_string(),
                speaker: "SPEAKER_00".to_string(),
                start_secs: 0.4,
                end_secs: 0.8,
                speaker_confidence: None,
                overlaps_segment: true,
            },
            DiarizationWord {
                word: "second".to_string(),
                speaker: "SPEAKER_00".to_string(),
                start_secs: 2.0,
                end_secs: 2.4,
                speaker_confidence: None,
                overlaps_segment: true,
            },
            DiarizationWord {
                word: "block".to_string(),
                speaker: "SPEAKER_00".to_string(),
                start_secs: 2.4,
                end_secs: 2.8,
                speaker_confidence: None,
                overlaps_segment: true,
            },
            DiarizationWord {
                word: "other".to_string(),
                speaker: "SPEAKER_01".to_string(),
                start_secs: 3.2,
                end_secs: 3.6,
                speaker_confidence: None,
                overlaps_segment: true,
            },
        ];

        let utterances = build_utterances(&words, &[]);
        assert_eq!(utterances.len(), 2, "{utterances:#?}");
        assert_eq!(utterances[0].speaker, "SPEAKER_00");
        assert_eq!(utterances[0].start_secs, 0.0);
        assert_eq!(utterances[0].end_secs, 2.8);
        assert_eq!(utterances[0].word_start, 0);
        assert_eq!(utterances[0].word_end, 3);
        assert_eq!(utterances[0].text, "first block second block");
        assert_eq!(utterances[1].speaker, "SPEAKER_01");
    }

    #[test]
    fn stabilize_word_speakers_absorbs_low_confidence_inner_fragment() {
        let mut words = vec![
            DiarizationWord {
                word: "hello".to_string(),
                speaker: "SPEAKER_00".to_string(),
                start_secs: 0.0,
                end_secs: 0.4,
                speaker_confidence: Some(0.82),
                overlaps_segment: true,
            },
            DiarizationWord {
                word: "there".to_string(),
                speaker: "SPEAKER_01".to_string(),
                start_secs: 0.5,
                end_secs: 0.7,
                speaker_confidence: Some(0.22),
                overlaps_segment: true,
            },
            DiarizationWord {
                word: "friend".to_string(),
                speaker: "SPEAKER_00".to_string(),
                start_secs: 0.8,
                end_secs: 1.2,
                speaker_confidence: Some(0.81),
                overlaps_segment: true,
            },
        ];

        stabilize_word_speakers(&mut words);

        assert!(words.iter().all(|word| word.speaker == "SPEAKER_00"));
    }

    #[test]
    fn stabilize_word_speakers_preserves_overlap_backed_edge_turn() {
        let mut words = vec![
            DiarizationWord {
                word: "hello".to_string(),
                speaker: "SPEAKER_00".to_string(),
                start_secs: 0.0,
                end_secs: 0.4,
                speaker_confidence: Some(0.82),
                overlaps_segment: true,
            },
            DiarizationWord {
                word: "world".to_string(),
                speaker: "SPEAKER_00".to_string(),
                start_secs: 0.4,
                end_secs: 0.8,
                speaker_confidence: Some(0.82),
                overlaps_segment: true,
            },
            DiarizationWord {
                word: "yeah".to_string(),
                speaker: "SPEAKER_01".to_string(),
                start_secs: 0.9,
                end_secs: 1.2,
                speaker_confidence: Some(0.22),
                overlaps_segment: true,
            },
        ];

        stabilize_word_speakers(&mut words);

        assert_eq!(words[2].speaker, "SPEAKER_01");
    }

    /// PIPE-17: a low-confidence one-word fragment before the first turn is
    /// absorbed into the run that follows it; the scan used to start at the
    /// second run, so a leading fragment was never considered.
    #[test]
    fn stabilize_word_speakers_absorbs_low_confidence_leading_fragment() {
        let fragment = DiarizationWord {
            overlaps_segment: false,
            ..test_word("um", "SPEAKER_01", 0.0, 0.3, 0.2)
        };
        let mut words = vec![
            fragment,
            test_word("so", "SPEAKER_00", 0.4, 0.6, 0.85),
            test_word("today", "SPEAKER_00", 0.6, 1.0, 0.85),
            test_word("we", "SPEAKER_00", 1.0, 1.2, 0.85),
            test_word("start", "SPEAKER_00", 1.2, 1.8, 0.85),
            test_word("right", "SPEAKER_01", 2.4, 2.7, 0.8),
            test_word("okay", "SPEAKER_01", 2.7, 3.0, 0.8),
        ];

        stabilize_word_speakers(&mut words);

        let speakers = words
            .iter()
            .map(|word| word.speaker.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            speakers,
            [
                "SPEAKER_00",
                "SPEAKER_00",
                "SPEAKER_00",
                "SPEAKER_00",
                "SPEAKER_00",
                "SPEAKER_01",
                "SPEAKER_01",
            ]
        );
    }

    #[test]
    fn choose_phrase_speaker_prefers_total_phrase_support() {
        let words = vec![
            DiarizationWord {
                word: "So".to_string(),
                speaker: "SPEAKER_01".to_string(),
                start_secs: 0.0,
                end_secs: 0.3,
                speaker_confidence: Some(0.25),
                overlaps_segment: true,
            },
            DiarizationWord {
                word: "Aaron".to_string(),
                speaker: "SPEAKER_01".to_string(),
                start_secs: 0.3,
                end_secs: 0.6,
                speaker_confidence: Some(0.25),
                overlaps_segment: true,
            },
            DiarizationWord {
                word: "in".to_string(),
                speaker: "SPEAKER_01".to_string(),
                start_secs: 0.6,
                end_secs: 0.9,
                speaker_confidence: Some(0.25),
                overlaps_segment: true,
            },
            DiarizationWord {
                word: "your".to_string(),
                speaker: "SPEAKER_01".to_string(),
                start_secs: 0.9,
                end_secs: 1.2,
                speaker_confidence: Some(0.25),
                overlaps_segment: true,
            },
            DiarizationWord {
                word: "email".to_string(),
                speaker: "SPEAKER_00".to_string(),
                start_secs: 1.2,
                end_secs: 2.1,
                speaker_confidence: Some(0.8),
                overlaps_segment: true,
            },
            DiarizationWord {
                word: "you".to_string(),
                speaker: "SPEAKER_00".to_string(),
                start_secs: 2.1,
                end_secs: 2.8,
                speaker_confidence: Some(0.8),
                overlaps_segment: true,
            },
        ];

        let phrase = SpeakerRun {
            start: 0,
            end: words.len() - 1,
        };
        let chosen = choose_phrase_speaker(&words, phrase);

        assert_eq!(chosen, "SPEAKER_00");
    }

    #[test]
    fn normalize_phrase_speakers_preserves_two_run_turn_boundary() {
        let segments = vec![
            DiarizationSegment {
                speaker: "SPEAKER_01".to_string(),
                start_secs: 0.0,
                end_secs: 1.44,
                confidence: Some(0.28),
            },
            DiarizationSegment {
                speaker: "SPEAKER_00".to_string(),
                start_secs: 1.52,
                end_secs: 4.08,
                confidence: Some(0.78),
            },
        ];
        let mut words = vec![
            test_word("So", "SPEAKER_01", 0.32, 0.64, 0.28),
            test_word("Aaron", "SPEAKER_01", 0.64, 1.04, 0.28),
            test_word("in", "SPEAKER_01", 1.20, 1.28, 0.28),
            test_word("your", "SPEAKER_01", 1.28, 1.44, 0.28),
            test_word("email", "SPEAKER_00", 1.52, 1.92, 0.78),
            test_word("you", "SPEAKER_00", 1.92, 2.08, 0.78),
            test_word("said", "SPEAKER_00", 2.08, 2.32, 0.78),
            test_word("wanted", "SPEAKER_00", 2.40, 2.72, 0.78),
        ];

        normalize_phrase_speakers(&mut words, &segments);
        let utterances = build_utterances(&words, &[]);

        assert_eq!(utterances.len(), 2);
        assert_eq!(utterances[0].speaker, "SPEAKER_01");
        assert_eq!(utterances[0].text, "So Aaron in your");
        assert_eq!(utterances[1].speaker, "SPEAKER_00");
        assert!(utterances[1].text.starts_with("email you said"));
    }

    #[test]
    fn normalize_phrase_speakers_can_flip_ambiguous_middle_phrase() {
        let segments = vec![
            DiarizationSegment {
                speaker: "SPEAKER_00".to_string(),
                start_secs: 0.0,
                end_secs: 6.0,
                confidence: Some(0.8),
            },
            DiarizationSegment {
                speaker: "SPEAKER_01".to_string(),
                start_secs: 2.0,
                end_secs: 4.0,
                confidence: Some(0.25),
            },
        ];
        let mut words = vec![
            DiarizationWord {
                word: "hello".to_string(),
                speaker: "SPEAKER_00".to_string(),
                start_secs: 0.0,
                end_secs: 1.0,
                speaker_confidence: Some(0.8),
                overlaps_segment: true,
            },
            DiarizationWord {
                word: "How".to_string(),
                speaker: "SPEAKER_00".to_string(),
                start_secs: 2.0,
                end_secs: 2.6,
                speaker_confidence: Some(0.8),
                overlaps_segment: true,
            },
            DiarizationWord {
                word: "review".to_string(),
                speaker: "SPEAKER_00".to_string(),
                start_secs: 2.6,
                end_secs: 4.0,
                speaker_confidence: Some(0.8),
                overlaps_segment: true,
            },
            DiarizationWord {
                word: "Yeah".to_string(),
                speaker: "SPEAKER_00".to_string(),
                start_secs: 4.5,
                end_secs: 5.4,
                speaker_confidence: Some(0.8),
                overlaps_segment: true,
            },
        ];

        normalize_phrase_speakers(&mut words, &segments);

        assert_eq!(words[1].speaker, "SPEAKER_01");
        assert_eq!(words[2].speaker, "SPEAKER_01");
    }

    #[test]
    fn normalize_phrase_speakers_preserves_three_speaker_phrase() {
        let segments = vec![
            DiarizationSegment {
                speaker: "SPEAKER_00".to_string(),
                start_secs: 0.0,
                end_secs: 0.62,
                confidence: Some(0.92),
            },
            DiarizationSegment {
                speaker: "SPEAKER_01".to_string(),
                start_secs: 0.62,
                end_secs: 1.16,
                confidence: Some(0.89),
            },
            DiarizationSegment {
                speaker: "SPEAKER_02".to_string(),
                start_secs: 1.16,
                end_secs: 1.72,
                confidence: Some(0.91),
            },
        ];
        let mut words = vec![
            test_word("Alice", "SPEAKER_00", 0.00, 0.28, 0.92),
            test_word("opens", "SPEAKER_00", 0.28, 0.58, 0.92),
            test_word("Bob", "SPEAKER_01", 0.62, 0.86, 0.89),
            test_word("answers", "SPEAKER_01", 0.86, 1.12, 0.89),
            test_word("Carol", "SPEAKER_02", 1.16, 1.40, 0.91),
            test_word("wraps", "SPEAKER_02", 1.40, 1.70, 0.91),
            test_word("Later", "SPEAKER_00", 2.20, 2.52, 0.92),
            test_word("done", "SPEAKER_00", 2.52, 2.82, 0.92),
        ];

        normalize_phrase_speakers(&mut words, &segments);
        let utterances = build_utterances(&words, &[]);

        assert_eq!(words[0].speaker, "SPEAKER_00");
        assert_eq!(words[2].speaker, "SPEAKER_01");
        assert_eq!(words[4].speaker, "SPEAKER_02");
        assert_eq!(utterances.len(), 4);
        assert_eq!(utterances[0].speaker, "SPEAKER_00");
        assert_eq!(utterances[1].speaker, "SPEAKER_01");
        assert_eq!(utterances[2].speaker, "SPEAKER_02");
    }

    #[test]
    fn phrase_normalization_matches_expected_turns_for_overlap_heavy_sample() {
        let segments = vec![
            DiarizationSegment {
                speaker: "SPEAKER_00".to_string(),
                start_secs: 0.32,
                end_secs: 4.0,
                confidence: Some(0.78),
            },
            DiarizationSegment {
                speaker: "SPEAKER_01".to_string(),
                start_secs: 0.32,
                end_secs: 1.6,
                confidence: Some(0.28),
            },
            DiarizationSegment {
                speaker: "SPEAKER_00".to_string(),
                start_secs: 4.88,
                end_secs: 12.32,
                confidence: Some(0.78),
            },
            DiarizationSegment {
                speaker: "SPEAKER_01".to_string(),
                start_secs: 4.88,
                end_secs: 5.36,
                confidence: Some(0.25),
            },
            DiarizationSegment {
                speaker: "SPEAKER_01".to_string(),
                start_secs: 11.76,
                end_secs: 12.32,
                confidence: Some(0.25),
            },
            DiarizationSegment {
                speaker: "SPEAKER_00".to_string(),
                start_secs: 13.04,
                end_secs: 20.72,
                confidence: Some(0.77),
            },
            DiarizationSegment {
                speaker: "SPEAKER_01".to_string(),
                start_secs: 13.04,
                end_secs: 20.72,
                confidence: Some(0.27),
            },
            DiarizationSegment {
                speaker: "SPEAKER_00".to_string(),
                start_secs: 21.2,
                end_secs: 27.12,
                confidence: Some(0.78),
            },
            DiarizationSegment {
                speaker: "SPEAKER_01".to_string(),
                start_secs: 21.2,
                end_secs: 21.6,
                confidence: Some(0.26),
            },
            DiarizationSegment {
                speaker: "SPEAKER_01".to_string(),
                start_secs: 23.2,
                end_secs: 23.84,
                confidence: Some(0.26),
            },
            DiarizationSegment {
                speaker: "SPEAKER_01".to_string(),
                start_secs: 26.08,
                end_secs: 27.12,
                confidence: Some(0.27),
            },
        ];
        let mut words = vec![
            test_word("So", "SPEAKER_01", 0.32, 0.64, 0.28),
            test_word("Aaron", "SPEAKER_01", 0.64, 1.04, 0.28),
            test_word("in", "SPEAKER_01", 1.20, 1.28, 0.28),
            test_word("your", "SPEAKER_01", 1.28, 1.44, 0.28),
            test_word("email", "SPEAKER_00", 1.52, 1.92, 0.78),
            test_word("you", "SPEAKER_00", 1.92, 2.08, 0.78),
            test_word("said", "SPEAKER_00", 2.08, 2.32, 0.78),
            test_word("you", "SPEAKER_00", 2.32, 2.40, 0.78),
            test_word("wanted", "SPEAKER_00", 2.40, 2.72, 0.78),
            test_word("to", "SPEAKER_00", 2.72, 2.80, 0.78),
            test_word("talk", "SPEAKER_00", 2.80, 3.12, 0.78),
            test_word("about", "SPEAKER_00", 3.12, 3.36, 0.78),
            test_word("the", "SPEAKER_00", 3.36, 3.52, 0.78),
            test_word("exam", "SPEAKER_00", 3.52, 4.08, 0.78),
            test_word("Yeah", "SPEAKER_01", 4.88, 5.28, 0.25),
            test_word("um", "SPEAKER_00", 5.44, 5.92, 0.78),
            test_word("I've", "SPEAKER_00", 6.24, 6.40, 0.79),
            test_word("just", "SPEAKER_00", 6.40, 6.72, 0.79),
            test_word("never", "SPEAKER_00", 6.72, 7.04, 0.79),
            test_word("taken", "SPEAKER_00", 7.04, 7.52, 0.79),
            test_word("a", "SPEAKER_00", 7.52, 7.60, 0.79),
            test_word("class", "SPEAKER_00", 7.60, 8.00, 0.79),
            test_word("with", "SPEAKER_00", 8.00, 8.16, 0.79),
            test_word("so", "SPEAKER_00", 8.16, 8.40, 0.79),
            test_word("many", "SPEAKER_00", 8.40, 8.72, 0.79),
            test_word("different", "SPEAKER_00", 8.72, 8.96, 0.79),
            test_word("readings", "SPEAKER_00", 9.04, 9.68, 0.79),
            test_word("I've", "SPEAKER_01", 10.16, 10.32, 0.26),
            test_word("managed", "SPEAKER_00", 10.32, 10.80, 0.78),
            test_word("to", "SPEAKER_00", 10.80, 10.88, 0.78),
            test_word("keep", "SPEAKER_00", 10.88, 11.12, 0.78),
            test_word("up", "SPEAKER_00", 11.12, 11.36, 0.78),
            test_word("with", "SPEAKER_00", 11.36, 11.44, 0.78),
            test_word("all", "SPEAKER_00", 11.52, 11.60, 0.78),
            test_word("the", "SPEAKER_00", 11.60, 11.76, 0.78),
            test_word("assignments", "SPEAKER_01", 11.76, 12.48, 0.25),
            test_word("but", "SPEAKER_00", 12.96, 13.12, 0.77),
            test_word("I'm", "SPEAKER_00", 13.12, 13.28, 0.77),
            test_word("not", "SPEAKER_00", 13.28, 13.44, 0.77),
            test_word("sure", "SPEAKER_00", 13.44, 13.76, 0.77),
            test_word("how", "SPEAKER_00", 13.76, 13.92, 0.77),
            test_word("to", "SPEAKER_00", 14.00, 14.24, 0.77),
            test_word("how", "SPEAKER_00", 14.96, 15.20, 0.76),
            test_word("to", "SPEAKER_00", 15.20, 15.52, 0.76),
            test_word("How", "SPEAKER_00", 16.16, 16.56, 0.76),
            test_word("to", "SPEAKER_00", 16.72, 16.72, 0.76),
            test_word("review", "SPEAKER_00", 16.88, 17.44, 0.76),
            test_word("everything", "SPEAKER_00", 17.44, 18.08, 0.76),
            test_word("Yeah", "SPEAKER_00", 18.48, 18.96, 0.76),
            test_word("in", "SPEAKER_00", 19.36, 19.52, 0.77),
            test_word("other", "SPEAKER_00", 19.60, 19.84, 0.77),
            test_word("classes", "SPEAKER_00", 19.84, 20.32, 0.77),
            test_word("I've", "SPEAKER_00", 20.32, 20.48, 0.77),
            test_word("had", "SPEAKER_00", 20.48, 20.80, 0.77),
            test_word("there's", "SPEAKER_01", 21.20, 21.44, 0.26),
            test_word("usually", "SPEAKER_00", 21.44, 21.92, 0.78),
            test_word("just", "SPEAKER_00", 21.92, 22.16, 0.78),
            test_word("one", "SPEAKER_00", 22.16, 22.48, 0.78),
            test_word("book", "SPEAKER_00", 22.48, 22.72, 0.78),
            test_word("to", "SPEAKER_00", 22.80, 22.88, 0.78),
            test_word("review", "SPEAKER_00", 22.88, 23.20, 0.78),
            test_word("not", "SPEAKER_01", 23.20, 23.36, 0.26),
            test_word("three", "SPEAKER_01", 23.60, 23.84, 0.26),
            test_word("different", "SPEAKER_00", 23.84, 24.16, 0.78),
            test_word("books", "SPEAKER_00", 24.16, 24.48, 0.78),
            test_word("plus", "SPEAKER_00", 24.48, 24.72, 0.78),
            test_word("all", "SPEAKER_00", 24.80, 24.96, 0.78),
            test_word("those", "SPEAKER_00", 24.96, 25.12, 0.78),
            test_word("other", "SPEAKER_00", 25.12, 25.44, 0.78),
            test_word("text", "SPEAKER_00", 25.44, 25.84, 0.78),
            test_word("excerpts", "SPEAKER_00", 25.84, 26.40, 0.78),
            test_word("and", "SPEAKER_01", 26.40, 26.56, 0.27),
            test_word("videos", "SPEAKER_01", 26.56, 27.28, 0.27),
        ];

        stabilize_word_speakers(&mut words);
        normalize_phrase_speakers(&mut words, &segments);
        let utterances = build_utterances(&words, &[]);

        assert!(utterances.len() >= 4, "{utterances:#?}");
        assert_eq!(utterances[0].speaker, "SPEAKER_01");
        assert_eq!(utterances[0].text, "So Aaron in your");
        assert_eq!(utterances[1].speaker, "SPEAKER_00");
        assert!(utterances[1].text.starts_with("email you said"));
        assert!(utterances
            .iter()
            .any(|utterance| utterance.speaker == "SPEAKER_01"
                && utterance.text == "How to review everything"));
        assert_eq!(utterances.last().unwrap().speaker, "SPEAKER_01");
        assert!(utterances.last().unwrap().text.ends_with("and videos"));
    }

    fn test_word(
        word: &str,
        speaker: &str,
        start_secs: f32,
        end_secs: f32,
        speaker_confidence: f32,
    ) -> DiarizationWord {
        DiarizationWord {
            word: word.to_string(),
            speaker: speaker.to_string(),
            start_secs,
            end_secs,
            speaker_confidence: Some(speaker_confidence),
            overlaps_segment: true,
        }
    }

    #[test]
    fn fallback_word_timings_with_segments_places_words_in_windows() {
        let words = vec![
            "one".to_string(),
            "two".to_string(),
            "three".to_string(),
            "four".to_string(),
        ];
        let segments = vec![
            DiarizationSegment {
                speaker: "SPEAKER_00".to_string(),
                start_secs: 0.0,
                end_secs: 1.0,
                confidence: None,
            },
            DiarizationSegment {
                speaker: "SPEAKER_01".to_string(),
                start_secs: 2.0,
                end_secs: 3.0,
                confidence: None,
            },
        ];

        let alignments = fallback_word_timings_with_segments(&words, &segments, 3.0);
        assert_eq!(alignments.len(), 4);
        assert!(alignments[0].1 < 1_000);
        assert!(alignments[1].1 < 1_000);
        assert!(alignments[2].1 >= 2_000);
        assert!(alignments[3].1 >= 2_000);
    }

    #[test]
    fn alignment_is_suspicious_detects_tail_collapse() {
        let alignments = (0..20)
            .map(|idx| (format!("w{idx}"), 27_302u32, 27_303u32))
            .collect::<Vec<_>>();
        assert!(alignment_is_suspicious(&alignments, 20, 27.303));
    }

    #[test]
    fn alignment_is_suspicious_detects_front_loaded_micro_spans_and_bridge_word() {
        let mut alignments = (0..42)
            .map(|idx| (format!("w{idx}"), idx as u32, idx as u32 + 1))
            .collect::<Vec<_>>();
        alignments.push(("bridge".to_string(), 41, 13_600));
        alignments.extend((43..73).map(|idx| {
            let start = 18_080 + ((idx - 43) as u32 * 320);
            (format!("w{idx}"), start, start + 240)
        }));

        assert!(alignment_is_suspicious(&alignments, 73, 27.303));
    }

    #[test]
    fn attribution_requires_fallback_when_too_many_words_are_unattributed() {
        assert!(attribution_requires_fallback(73, 31, 42));
        assert!(!attribution_requires_fallback(73, 60, 13));
    }

    #[test]
    fn voxtral_diarization_chunks_at_its_offline_audio_hint() {
        // Voxtral reports no audio-token rate, so the hint alone sets the limit.
        let limit = crate::runtime::asr::compose_invocation_audio_limit(
            Some(VOXTRAL_REALTIME_OFFLINE_AUDIO_SECONDS_HINT),
            None,
            Some(8_256),
            |_| NativeAsrGenerationOptions::default().max_new_tokens,
        )
        .unwrap();
        assert_eq!(limit, Some(30.0));
        assert_eq!(combined_chunk_limit(limit, Some(300.0)), Some(30.0));
    }

    #[test]
    fn should_use_single_pass_diarization_asr_respects_combined_limit() {
        assert!(should_use_single_pass_diarization_asr(
            27.3,
            combined_chunk_limit(None, Some(300.0))
        ));
        assert!(should_use_single_pass_diarization_asr(27.3, None));
        assert!(!should_use_single_pass_diarization_asr(
            600.0,
            combined_chunk_limit(None, Some(300.0))
        ));
        // An aligner with a longer limit no longer lifts the ASR limit.
        assert!(!should_use_single_pass_diarization_asr(
            120.0,
            combined_chunk_limit(Some(30.0), Some(300.0))
        ));
    }

    #[test]
    fn diarization_asr_without_aligner_chunks_by_the_asr_limit() {
        let asr_only = combined_chunk_limit(Some(30.0), None);
        assert!(!should_use_single_pass_diarization_asr(116.5, asr_only));
        assert!(!should_use_single_pass_diarization_asr(600.0, asr_only));
        assert!(should_use_single_pass_diarization_asr(27.3, asr_only));
    }

    #[test]
    fn diarization_asr_without_aligner_plans_bounded_chunks_with_fallback_timings() {
        let _env = crate::env_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let sample_rate = PIPELINE_SAMPLE_RATE;
        let audio = PipelineAudio {
            samples: vec![0.0_f32; 60 * sample_rate as usize].into(),
            sample_rate,
            duration_secs: 60.0,
        };
        let chunk_limit = combined_chunk_limit(Some(30.0), None);
        assert!(!should_use_single_pass_diarization_asr(
            audio.duration_secs,
            chunk_limit
        ));

        let (_cfg, chunks) = plan_invocation_asr_chunks(&audio, chunk_limit).expect("plan");
        assert!(chunks.len() >= 2, "60 s must split into >= 2 chunks");
        for chunk in &chunks {
            let secs = chunk.len_samples() as f32 / sample_rate as f32;
            assert!(secs <= 30.0, "chunk of {secs} s exceeds the 30 s ASR limit");
        }

        let transcribed = chunks
            .iter()
            .enumerate()
            .map(|(index, range)| TranscribedChunk {
                range: *range,
                text: format!("alpha{index} beta{index} gamma{index}"),
                language: None,
                hit_token_budget: false,
            })
            .collect::<Vec<_>>();
        let stitched = stitch_timed_windows(&interval_timed_windows(
            &chunk_windows(&transcribed),
            sample_rate,
        ));
        assert!(!stitched.words.is_empty());
        let last_end_ms = stitched.words.iter().map(|word| word.end_ms).max().unwrap();
        assert!(last_end_ms > 30_000, "timings must cover the later chunks");
    }

    /// LIVE-9: an aligner that failed to load must leave a warning naming
    /// the failure, not a record that looks aligned.
    #[test]
    fn aligner_load_failure_becomes_a_warning() {
        let warning = aligner_unavailable_warning(
            "Qwen3-ForcedAligner-0.6B failed to load: needs 2.77 GB, 0.19 GB available",
        );
        assert!(warning.contains("Forced aligner unavailable"), "{warning}");
        assert!(warning.contains("needs 2.77 GB"), "{warning}");
        assert!(warning.contains("estimated, not aligned"), "{warning}");
    }

    #[test]
    fn single_pass_asr_budget_grows_with_duration() {
        let default = NativeAsrGenerationOptions::default().max_new_tokens;
        assert_eq!(diarization_asr_max_new_tokens(0.0), default);
        assert_eq!(diarization_asr_max_new_tokens(30.0), default);
        assert_eq!(diarization_asr_max_new_tokens(f32::NAN), default);
        // 600 s of dense speech needs far more than the 768-token default.
        assert_eq!(diarization_asr_max_new_tokens(600.0), 600 * 8 + 64);
        assert!(diarization_asr_max_new_tokens(180.0) > default);
    }

    #[test]
    fn asr_token_budget_stops_become_warnings() {
        let granite = serde_json::json!({
            "generation": {"stats": {"stop_reason": "max_tokens"}}
        });
        assert!(asr_hit_token_budget(Some(&granite)));
        assert!(asr_hit_token_budget(Some(
            &serde_json::json!({"stop_reason": "max_tokens"})
        )));
        assert!(!asr_hit_token_budget(Some(
            &serde_json::json!({"stop_reason": "stop_token"})
        )));
        assert!(!asr_hit_token_budget(None));

        let chunk = |start: usize, end: usize, hit: bool| TranscribedChunk {
            range: AudioChunk {
                start_sample: start * PIPELINE_SAMPLE_RATE as usize,
                end_sample: end * PIPELINE_SAMPLE_RATE as usize,
            },
            text: "words".to_string(),
            language: None,
            hit_token_budget: hit,
        };
        let warnings = asr_token_budget_warnings(
            Some((0.0, 95.0)),
            &[chunk(0, 30, false), chunk(27, 57, true)],
        );
        assert_eq!(warnings.len(), 2, "{warnings:?}");
        assert!(warnings[0].contains("0.00-95.00 s"), "{warnings:?}");
        assert!(warnings[1].contains("27.00-57.00 s"), "{warnings:?}");
        assert!(asr_token_budget_warnings(None, &[chunk(0, 30, false)]).is_empty());
    }
}
