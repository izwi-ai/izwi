use std::sync::Arc;
use std::time::Instant;

use crate::catalog::ModelFamily;
use crate::engine::InvocationPagedKvLease;
use crate::error::{Error, Result};
use crate::models::architectures::qwen3::tts::{
    PhysicalTtsManagedQuantumCheckpoint, PhysicalTtsPrefillManagedCheckpoint, SpeakerReference,
    TalkerPhysicalCache, TtsGenerationParams, TtsStreamingConfig,
};
use crate::runtime::audio_io::decode_reference_audio_base64;

use super::super::request::EngineCoreRequest;
use super::super::scheduler::ScheduledRequest;
use super::super::types::AudioOutput;
use super::super::SessionKey;
use super::state::{
    ActiveFishS2TtsDecode, ActiveLfm25TtsDecode, ActiveQwenTtsDecode, ActiveVibeVoiceTtsDecode,
    ActiveVoxtralTtsDecode, QwenTtsPhysicalState,
};
use super::{
    ExecutorOutput, ExecutorPhaseTiming, ExecutorStateLease, ModelSessionResult, NativeExecutor,
};

struct ContinuousVoxtralTtsRow<'a> {
    index: usize,
    lease: Option<ExecutorStateLease<'a, ActiveVoxtralTtsDecode>>,
    cache: crate::models::shared::attention::physical::PhysicalPagedKvCache,
    checkpoint:
        Option<crate::models::architectures::voxtral::tts::retained::VoxtralTtsQuantumCheckpoint>,
    prior_frames: usize,
    prior_stream_sequence: usize,
}

impl ContinuousVoxtralTtsRow<'_> {
    fn rollback(&mut self) -> Result<()> {
        let checkpoint = self.checkpoint.take().ok_or_else(|| {
            Error::InferenceError("Voxtral TTS row has no rollback checkpoint".into())
        })?;
        let lease = self.lease.as_mut().expect("armed Voxtral TTS lease");
        let active = lease.require_state_mut()?;
        active
            .state
            .rollback_quantum(&mut self.cache, &checkpoint)?;
        active.last_frames_generated = self.prior_frames;
        active.stream_sequence = self.prior_stream_sequence;
        lease.mark_clean();
        Ok(())
    }
}

struct ContinuousVoxtralTtsBatch<'a> {
    rows: Vec<ContinuousVoxtralTtsRow<'a>>,
    armed: bool,
}

impl Drop for ContinuousVoxtralTtsBatch<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        for row in &mut self.rows {
            if row.checkpoint.is_some() {
                if let Err(error) = row.rollback() {
                    tracing::error!(%error, "Voxtral TTS rollback failed; state remains fenced");
                }
            }
        }
    }
}

struct ContinuousLfm25TtsRow<'a> {
    index: usize,
    session: SessionKey,
    lease: Option<ExecutorStateLease<'a, ActiveLfm25TtsDecode>>,
    main: crate::models::shared::attention::physical::PhysicalPagedKvCache,
    depth: Option<InvocationPagedKvLease>,
    checkpoint: Option<
        crate::models::architectures::lfm25_audio::tts_retained::Lfm25AudioTtsQuantumCheckpoint,
    >,
    prior_tokens: usize,
    prior_stream_sequence: usize,
}

struct ContinuousVibeVoiceTtsRow<'a> {
    index: usize,
    session: SessionKey,
    lease: Option<ExecutorStateLease<'a, ActiveVibeVoiceTtsDecode>>,
    checkpoint:
        Option<crate::models::architectures::vibevoice::tts::VibeVoiceTtsRetainedCheckpoint>,
    prior_frames: usize,
    prior_stream_sequence: usize,
}

impl ContinuousVibeVoiceTtsRow<'_> {
    fn rollback(&mut self) -> Result<()> {
        let checkpoint = self.checkpoint.as_mut().ok_or_else(|| {
            Error::InferenceError("VibeVoice TTS cohort checkpoint is absent".into())
        })?;
        let lease = self
            .lease
            .as_mut()
            .ok_or_else(|| Error::InferenceError("VibeVoice TTS cohort lease is absent".into()))?;
        let active = lease.require_state_mut()?;
        active.state.rollback_managed_quantum(checkpoint)?;
        active.last_frames_generated = self.prior_frames;
        active.stream_sequence = self.prior_stream_sequence;
        self.checkpoint = None;
        lease.mark_clean();
        Ok(())
    }
}

struct ContinuousVibeVoiceTtsBatch<'a> {
    rows: Vec<ContinuousVibeVoiceTtsRow<'a>>,
    armed: bool,
}

impl Drop for ContinuousVibeVoiceTtsBatch<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        for row in &mut self.rows {
            if row.checkpoint.is_some() {
                if let Err(error) = row.rollback() {
                    tracing::error!(request_id = %row.session.request_id, %error, "VibeVoice TTS cohort rollback failed");
                }
            }
        }
    }
}

impl<'a> ContinuousLfm25TtsRow<'a> {
    fn lease_mut(&mut self) -> Result<&mut ExecutorStateLease<'a, ActiveLfm25TtsDecode>> {
        self.lease.as_mut().ok_or_else(|| {
            Error::InferenceError("LFM2.5 Audio TTS cohort state lease is absent".into())
        })
    }

    fn rollback(&mut self) -> Result<()> {
        let checkpoint = self.checkpoint.take().ok_or_else(|| {
            Error::InferenceError("LFM2.5 Audio TTS cohort checkpoint is absent".into())
        })?;
        let lease = self.lease.as_mut().ok_or_else(|| {
            Error::InferenceError("LFM2.5 Audio TTS cohort state lease is absent".into())
        })?;
        let active = lease.require_state_mut()?;
        active.state.rollback_quantum(
            &mut self.main,
            self.depth.as_mut().map(|depth| depth.cache_mut()),
            &checkpoint,
        )?;
        active.last_tokens_generated = self.prior_tokens;
        active.stream_sequence = self.prior_stream_sequence;
        lease.mark_clean();
        Ok(())
    }
}

struct ContinuousLfm25TtsBatch<'a> {
    rows: Vec<ContinuousLfm25TtsRow<'a>>,
    armed: bool,
}

impl<'a> ContinuousLfm25TtsBatch<'a> {
    fn new(rows: Vec<ContinuousLfm25TtsRow<'a>>) -> Self {
        Self { rows, armed: true }
    }

    fn rollback_row(&mut self, row: usize) -> Result<()> {
        self.rows
            .get_mut(row)
            .ok_or_else(|| {
                Error::InferenceError("continuous LFM TTS rollback row is out of range".into())
            })?
            .rollback()
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for ContinuousLfm25TtsBatch<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        for row in &mut self.rows {
            if row.checkpoint.is_none() {
                continue;
            }
            if let Err(error) = row.rollback() {
                tracing::error!(
                    request_id = %row.session.request_id,
                    epoch = row.session.epoch,
                    %error,
                    "continuous LFM2.5 TTS rollback failed; state fenced until cleanup"
                );
            }
        }
    }
}

fn validate_qwen_tts_physical_bindings(
    has_managed_runtime: bool,
    has_talker_cache: bool,
    has_tensor_arena: bool,
    has_tensor_reservation: bool,
) -> Result<()> {
    if has_managed_runtime != has_talker_cache {
        return Err(Error::InferenceError(
            "physical Qwen TTS execution requires its exact talker reservation".to_string(),
        ));
    }
    if !has_talker_cache {
        return Err(Error::InferenceError(
            "Qwen TTS no longer supports model-owned decode caches".to_string(),
        ));
    }
    if has_tensor_arena != has_tensor_reservation {
        return Err(Error::InferenceError(
            "physical Qwen TTS tensor state requires its exact row reservation".into(),
        ));
    }
    Ok(())
}

fn qwen_tts_decode_iterations(scheduled: &ScheduledRequest) -> usize {
    if scheduled.is_prefill {
        0
    } else {
        scheduled.num_tokens.max(1)
    }
}

fn resumable_tts_prefill_span(
    scheduled: &ScheduledRequest,
    prompt_tokens: usize,
) -> Result<(usize, usize)> {
    let start = scheduled.num_computed_tokens;
    let end = start.checked_add(scheduled.num_tokens).ok_or_else(|| {
        Error::InvalidInput("resumable TTS prefill span overflowed prompt accounting".into())
    })?;
    let crate::engine::WorkUnit::SequenceStep { phase, input, .. } = &scheduled.work else {
        return Err(Error::InvalidInput(
            "resumable TTS prefill requires sequence-prefill work".into(),
        ));
    };
    if *phase != crate::engine::SequencePhase::Prefill
        || input.start != start
        || input.end != end
        || start >= end
        || end > prompt_tokens
    {
        return Err(Error::InvalidInput(format!(
            "resumable TTS work [{}, {}) disagrees with scheduler span [{start}, {end}) for {prompt_tokens} prompt tokens",
            input.start, input.end
        )));
    }
    Ok((start, end))
}

fn validate_continuous_tts_batch_shape(scheduled: &[ScheduledRequest]) -> Result<()> {
    if scheduled.is_empty()
        || scheduled
            .iter()
            .any(|scheduled| scheduled.is_prefill || scheduled.num_tokens != 1)
    {
        return Err(Error::InvalidInput(
            "continuous TTS execution requires one decode frame per row".into(),
        ));
    }
    Ok(())
}

fn late_cancelled_tts_rows(cancelled: &[bool], checkpoint_armed: &[bool]) -> Vec<usize> {
    cancelled
        .iter()
        .zip(checkpoint_armed)
        .enumerate()
        .filter_map(|(row, (cancelled, armed))| (*cancelled && *armed).then_some(row))
        .collect()
}

fn continuous_tts_model_call(
    live_kernel_rows: usize,
    tensor_batched: bool,
) -> Option<crate::engine::metrics::EngineModelCall> {
    match live_kernel_rows {
        0 => None,
        1 => Some(crate::engine::metrics::EngineModelCall::ScalarRows {
            envelope: crate::engine::NativeBatchMode::Continuous,
            rows: 1,
        }),
        rows if tensor_batched => Some(crate::engine::metrics::EngineModelCall::NativeTensor {
            mode: crate::engine::NativeBatchMode::Continuous,
            rows,
        }),
        rows => Some(crate::engine::metrics::EngineModelCall::ScalarRows {
            envelope: crate::engine::NativeBatchMode::Continuous,
            rows,
        }),
    }
}

fn retained_tts_batch_model_call(
    mode: crate::engine::NativeBatchMode,
    live_kernel_rows: usize,
) -> Option<crate::engine::metrics::EngineModelCall> {
    match live_kernel_rows {
        0 => None,
        1 => Some(crate::engine::metrics::EngineModelCall::ScalarRows {
            envelope: mode,
            rows: 1,
        }),
        rows => Some(crate::engine::metrics::EngineModelCall::NativeTensor { mode, rows }),
    }
}

fn kokoro_live_row_indices(cancelled: &[bool]) -> Vec<usize> {
    cancelled
        .iter()
        .enumerate()
        .filter_map(|(index, cancelled)| (!cancelled).then_some(index))
        .collect()
}

fn validate_exact_kokoro_model<T>(expected: &mut Option<Arc<T>>, candidate: Arc<T>) -> Result<()> {
    if expected
        .as_ref()
        .is_some_and(|expected| !Arc::ptr_eq(expected, &candidate))
    {
        return Err(Error::InferenceError(
            "Kokoro static cohort mixed exact model instances".into(),
        ));
    }
    if expected.is_none() {
        *expected = Some(candidate);
    }
    Ok(())
}

fn scatter_kokoro_rows<T>(
    width: usize,
    live_indices: &[usize],
    results: Vec<T>,
) -> Result<Vec<Option<T>>> {
    if results.len() != live_indices.len() {
        return Err(Error::InferenceError(format!(
            "Kokoro returned {} results for {} live rows",
            results.len(),
            live_indices.len()
        )));
    }
    let mut scattered = (0..width).map(|_| None).collect::<Vec<_>>();
    for (index, result) in live_indices.iter().copied().zip(results) {
        let slot = scattered.get_mut(index).ok_or_else(|| {
            Error::InferenceError("Kokoro live row index exceeded cohort width".into())
        })?;
        if slot.is_some() {
            return Err(Error::InferenceError(
                "Kokoro live row index was duplicated".into(),
            ));
        }
        *slot = Some(result);
    }
    Ok(scattered)
}

fn validate_voxtral_tts_prefill_step(
    row: usize,
    scheduled: &ScheduledRequest,
    step: &crate::models::architectures::voxtral::tts::retained::VoxtralTtsPrefillStep,
) -> Result<()> {
    let expected_cursor = scheduled
        .num_computed_tokens
        .checked_add(scheduled.num_tokens)
        .ok_or_else(|| Error::InferenceError("Voxtral TTS prefill cursor overflowed".into()))?;
    if step.consumed_tokens != scheduled.num_tokens || step.prefill_cursor != expected_cursor {
        return Err(Error::InferenceError(format!(
            "Voxtral TTS prefill row {row} drifted: consumed {}, cursor {}, expected {} tokens ending at {expected_cursor}",
            step.consumed_tokens, step.prefill_cursor, scheduled.num_tokens,
        )));
    }
    Ok(())
}

fn accepted_tts_talker_tokens(
    start_cursor: usize,
    end_cursor: usize,
    scheduled_tokens: usize,
) -> Result<usize> {
    let accepted = end_cursor
        .checked_sub(start_cursor)
        .ok_or_else(|| Error::InferenceError("Qwen3-TTS talker cursor moved backwards".into()))?;
    if accepted > scheduled_tokens {
        return Err(Error::InferenceError(
            "Qwen3-TTS accepted more frames than were scheduled".into(),
        ));
    }
    Ok(accepted)
}

#[derive(Clone)]
struct ActiveTtsOuterCheckpoint {
    last_frames_generated: usize,
    stream_sequence: usize,
    audio_samples_len: usize,
    sampling_ms: f64,
    prefill_ms: f64,
    decode_ms: f64,
    codec_ms: f64,
    postprocess_ms: f64,
    first_output_ms_since_start: Option<f64>,
    prefill_steps: u32,
    decode_steps: u32,
}

impl ActiveTtsOuterCheckpoint {
    fn capture(state: &ActiveQwenTtsDecode) -> Self {
        Self {
            last_frames_generated: state.last_frames_generated,
            stream_sequence: state.stream_sequence,
            audio_samples_len: state.audio_samples_accum.len(),
            sampling_ms: state.sampling_ms,
            prefill_ms: state.prefill_ms,
            decode_ms: state.decode_ms,
            codec_ms: state.codec_ms,
            postprocess_ms: state.postprocess_ms,
            first_output_ms_since_start: state.first_output_ms_since_start,
            prefill_steps: state.prefill_steps,
            decode_steps: state.decode_steps,
        }
    }

    fn restore(self, state: &mut ActiveQwenTtsDecode) {
        state.last_frames_generated = self.last_frames_generated;
        state.stream_sequence = self.stream_sequence;
        state.audio_samples_accum.truncate(self.audio_samples_len);
        state.sampling_ms = self.sampling_ms;
        state.prefill_ms = self.prefill_ms;
        state.decode_ms = self.decode_ms;
        state.codec_ms = self.codec_ms;
        state.postprocess_ms = self.postprocess_ms;
        state.first_output_ms_since_start = self.first_output_ms_since_start;
        state.prefill_steps = self.prefill_steps;
        state.decode_steps = self.decode_steps;
    }
}

enum TtsManagedCheckpoint {
    Prefill(PhysicalTtsPrefillManagedCheckpoint),
    Decode(PhysicalTtsManagedQuantumCheckpoint),
}

fn rollback_tts_quantum(
    active: &mut ActiveQwenTtsDecode,
    checkpoint: TtsManagedCheckpoint,
    outer: ActiveTtsOuterCheckpoint,
) -> Result<()> {
    match (&mut active.state, checkpoint) {
        (QwenTtsPhysicalState::Prefill(state), TtsManagedCheckpoint::Prefill(checkpoint)) => {
            state.rollback_managed_quantum(checkpoint);
        }
        (QwenTtsPhysicalState::Decode(state), TtsManagedCheckpoint::Decode(checkpoint)) => {
            state.rollback_managed_quantum(checkpoint);
        }
        (state, TtsManagedCheckpoint::Prefill(checkpoint)) => {
            *state = QwenTtsPhysicalState::Prefill(checkpoint.into_state());
        }
        _ => {
            return Err(Error::InferenceError(
                "Qwen3-TTS rollback checkpoint no longer matches active phase".into(),
            ));
        }
    }
    outer.restore(active);
    Ok(())
}

struct ContinuousTtsStateBatch<'a> {
    rows: Vec<(
        usize,
        SessionKey,
        ExecutorStateLease<'a, ActiveQwenTtsDecode>,
        Option<(TtsManagedCheckpoint, ActiveTtsOuterCheckpoint)>,
    )>,
    armed: bool,
}

impl<'a> ContinuousTtsStateBatch<'a> {
    fn new(
        rows: Vec<(
            usize,
            SessionKey,
            ExecutorStateLease<'a, ActiveQwenTtsDecode>,
        )>,
    ) -> Self {
        Self {
            rows: rows
                .into_iter()
                .map(|(index, session, lease)| (index, session, lease, None))
                .collect(),
            armed: true,
        }
    }

    fn rollback_row(&mut self, row: usize) -> Result<usize> {
        let (index, _, lease, checkpoint) = self.rows.get_mut(row).ok_or_else(|| {
            Error::InferenceError("continuous TTS rollback row is out of range".into())
        })?;
        let (checkpoint, outer) = checkpoint.take().ok_or_else(|| {
            Error::InferenceError("continuous TTS row has no armed checkpoint".into())
        })?;
        rollback_tts_quantum(lease.require_state_mut()?, checkpoint, outer)?;
        lease.mark_clean();
        Ok(*index)
    }

    fn commit(
        mut self,
    ) -> Vec<(
        usize,
        SessionKey,
        ExecutorStateLease<'a, ActiveQwenTtsDecode>,
    )> {
        self.armed = false;
        std::mem::take(&mut self.rows)
            .into_iter()
            .map(|(index, session, lease, _)| (index, session, lease))
            .collect()
    }
}

impl Drop for ContinuousTtsStateBatch<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        for (_, session, lease, checkpoint) in &mut self.rows {
            let Some((checkpoint, outer)) = checkpoint.take() else {
                continue;
            };
            match lease
                .require_state_mut()
                .and_then(|state| rollback_tts_quantum(state, checkpoint, outer))
            {
                Ok(()) => lease.mark_clean(),
                Err(error) => tracing::error!(
                    request_id = %session.request_id,
                    epoch = session.epoch,
                    %error,
                    "continuous TTS rollback failed; state fenced until cleanup"
                ),
            }
        }
        self.rows.clear();
    }
}

impl NativeExecutor {
    pub(super) fn kokoro_tts_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
    ) -> Result<ModelSessionResult> {
        let mut outputs = self.kokoro_tts_batch(&[request], std::slice::from_ref(scheduled))?;
        outputs.pop().ok_or_else(|| {
            Error::InferenceError("Kokoro scalar synthesis returned no result".into())
        })
    }

    pub(super) fn kokoro_tts_batch(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ModelSessionResult>> {
        if requests.len() != scheduled.len() || requests.is_empty() {
            return Err(Error::InvalidInput(
                "Kokoro static synthesis requires one non-empty request per scheduled row".into(),
            ));
        }
        if scheduled
            .iter()
            .any(|row| !matches!(row.work, crate::engine::WorkUnit::AtomicJob { .. }))
        {
            return Err(Error::InvalidInput(
                "Kokoro static synthesis requires atomic scheduled work".into(),
            ));
        }

        let mut outputs = vec![None; requests.len()];
        let pre_call_cancelled = requests
            .iter()
            .map(|request| request.is_cancelled())
            .collect::<Vec<_>>();
        let live_indices = kokoro_live_row_indices(&pre_call_cancelled);
        let mut artifacts = Vec::with_capacity(requests.len());
        let mut model = None;
        for (index, request) in requests.iter().enumerate() {
            if request.id != scheduled[index].request_id {
                return Err(Error::InvalidInput(format!(
                    "Kokoro static row {index} request identity differs from its scheduled row"
                )));
            }
            if pre_call_cancelled[index] {
                outputs[index] = Some(ModelSessionResult::cancelled_before_dispatch(
                    ExecutorOutput::cancelled(request.id.clone()),
                ));
                continue;
            }
            let lease = request
                .prepared_kokoro_tts_model_lease_for_executor()?
                .ok_or_else(|| {
                    Error::InferenceError("Kokoro request lost its exact model lease".into())
                })?;
            let model_arc = lease.model_arc();
            validate_exact_kokoro_model(&mut model, model_arc)?;
            let artifact = request
                .prepared_kokoro_tts_artifact_for_executor()?
                .ok_or_else(|| {
                    Error::InferenceError("Kokoro request lost its prepared artifact".into())
                })?;
            artifacts.push((*artifact).clone());
        }

        if !live_indices.is_empty() {
            let model = model.expect("live Kokoro rows establish an exact model");
            let results = Self::run_blocking(|| model.generate_prepared_batch(&artifacts))?;
            let mut results = scatter_kokoro_rows(requests.len(), &live_indices, results)?;
            if let Some(call) = retained_tts_batch_model_call(
                crate::engine::NativeBatchMode::Static,
                live_indices.len(),
            ) {
                crate::engine::metrics::record_engine_model_call(call);
            }
            for index in live_indices {
                let request = requests[index];
                if request.is_cancelled() {
                    outputs[index] = Some(ModelSessionResult::cancelled(
                        ExecutorOutput::cancelled(request.id.clone()),
                    ));
                    continue;
                }
                let result = results[index].take().ok_or_else(|| {
                    Error::InferenceError(format!("Kokoro result row {index} was absent"))
                })?;
                if result.sample_rate == 0 {
                    return Err(Error::InferenceError(format!(
                        "Kokoro result row {index} has a zero sample rate"
                    )));
                }
                let duration_secs = result.samples.len() as f32 / result.sample_rate as f32;
                outputs[index] = Some(ModelSessionResult::atomic(ExecutorOutput {
                    request_id: request.id.clone(),
                    audio: Some(AudioOutput {
                        samples: result.samples,
                        sample_rate: result.sample_rate,
                        duration_secs,
                    }),
                    text: Some(result.phonemes),
                    input_transcription: None,
                    tokens_processed: scheduled[index].num_tokens,
                    tokens_generated: result.tokens_generated,
                    finished: true,
                    phase_timing_override: None,
                    asr_diagnostics: None,
                    error: None,
                }));
            }
        }

        outputs
            .into_iter()
            .enumerate()
            .map(|(index, output)| {
                output.ok_or_else(|| {
                    Error::InferenceError(format!(
                        "Kokoro static synthesis did not resolve row {index}"
                    ))
                })
            })
            .collect()
    }

    pub(super) fn to_tts_params(request: &EngineCoreRequest) -> TtsGenerationParams {
        request.qwen_tts_generation_params()
    }

    pub(super) fn reference_from_request(
        request: &EngineCoreRequest,
    ) -> Result<Option<SpeakerReference>> {
        if !request.has_tts_reference_for_execution() {
            return Ok(None);
        }

        let ref_audio = request.tts_reference_audio_for_execution().ok_or_else(|| {
            Error::InvalidInput(
                "reference_audio and reference_text must both be provided".to_string(),
            )
        })?;
        let ref_text = request.tts_reference_text_for_execution().ok_or_else(|| {
            Error::InvalidInput(
                "reference_audio and reference_text must both be provided".to_string(),
            )
        })?;
        if ref_text.trim().is_empty() {
            return Err(Error::InvalidInput(
                "reference_text cannot be empty".to_string(),
            ));
        }

        // Reference conditioning has a deliberately tighter contract than
        // inference audio (32 MiB decoded and 30 seconds). Keep decoding on
        // the executor's blocking boundary so compressed input never stalls
        // the async engine worker.
        let (audio_samples, sample_rate) =
            Self::run_blocking(|| decode_reference_audio_base64(ref_audio))?;
        Ok(Some(SpeakerReference {
            audio_samples,
            text: ref_text.to_string(),
            sample_rate,
        }))
    }

    pub(super) fn vibevoice_tts_request_with_managed_cache(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        retained: Option<super::RetainedRowManagedState>,
    ) -> Result<ModelSessionResult> {
        let variant = Self::resolve_variant(request)?;
        let model = request
            .prepared_vibevoice_tts_model_lease_for_executor()?
            .ok_or_else(|| Error::InferenceError("VibeVoice TTS lost model residency".into()))?;
        let mut retained = retained.ok_or_else(|| {
            Error::InferenceError("VibeVoice TTS lost retained physical state".into())
        })?;
        let positive = retained
            .take_paged_domain(crate::kv::CacheDomainId::new(1), true)?
            .expect("required VibeVoice positive cache");
        let negative = retained
            .take_paged_domain(crate::kv::CacheDomainId::new(2), true)?
            .expect("required VibeVoice negative cache");
        retained.ensure_all_paged_consumed()?;
        let _tensor = retained.tensor_state.clone().ok_or_else(|| {
            Error::InferenceError("VibeVoice TTS lost tokenizer reservation".into())
        })?;
        let arena = request
            .managed_cache_runtime()
            .and_then(|runtime| runtime.tensor_state())
            .cloned()
            .ok_or_else(|| Error::InferenceError("VibeVoice TTS lost tokenizer arena".into()))?;
        let mut lease = ExecutorStateLease::checkout(
            &self.vibevoice_tts_decode_states,
            scheduled.session_key(),
            variant,
            "VibeVoice TTS decode",
        )?;
        if lease.state().is_some_and(|active| {
            active.variant != variant || !Arc::ptr_eq(&active.model.model_arc(), &model.model_arc())
        }) {
            lease.discard_state();
        }
        let fresh = lease.state().is_none();
        let mut checkpoint = if fresh {
            let artifact = request
                .prepared_vibevoice_tts_artifact_for_executor()?
                .ok_or_else(|| Error::InferenceError("VibeVoice TTS lost prompt".into()))?;
            let params = request
                .vibevoice_tts_generation_params_for_executor()?
                .ok_or_else(|| Error::InferenceError("VibeVoice TTS lost geometry".into()))?;
            let (state, checkpoint) =
                model.new_retained_state_in_quantum(artifact, params, positive, negative)?;
            lease.install_state(ActiveVibeVoiceTtsDecode {
                variant,
                model: model.clone(),
                state,
                last_frames_generated: 0,
                stream_sequence: 0,
            })?;
            checkpoint
        } else {
            lease
                .require_state_mut()?
                .state
                .begin_managed_quantum(positive, negative)?
        };
        lease.mark_dirty();
        let result = (|| {
            let active = lease.require_state_mut()?;
            let step = if scheduled.is_prefill {
                let step = model.retained_prefill_step(&mut active.state, scheduled.num_tokens)?;
                if step.consumed_positive_tokens != scheduled.num_tokens {
                    return Err(Error::InferenceError(
                        "VibeVoice TTS prefill progress differs from scheduler".into(),
                    ));
                }
                None
            } else {
                let transaction =
                    crate::backends::state::PhysicalStateTransactionId::new(scheduled.plan_id)?;
                Some(model.retained_decode_step(
                    &mut active.state,
                    &crate::models::architectures::vibevoice::tts::VibeVoiceTtsTokenizerQuantum {
                        arena: arena.clone(),
                        transaction,
                    },
                )?)
            };
            if request.is_cancelled() {
                return Err(Error::Cancelled(request.id.clone()));
            }
            Ok::<_, Error>(step)
        })();
        let step = match result {
            Ok(step) if !request.is_cancelled() => step,
            result => {
                if fresh {
                    let _ = lease
                        .require_state_mut()?
                        .state
                        .take_managed_write_completions();
                    lease.discard_state();
                } else {
                    lease
                        .require_state_mut()?
                        .state
                        .rollback_managed_quantum(&mut checkpoint)?;
                }
                lease.mark_clean();
                return result.and_then(|_| Err(Error::Cancelled(request.id.clone())));
            }
        };
        let completions = lease
            .require_state_mut()?
            .state
            .take_managed_write_completions();
        lease
            .require_state_mut()?
            .state
            .commit_managed_quantum(&mut checkpoint)?;
        let active = lease.require_state_mut()?;
        let generated = step
            .as_ref()
            .map(|step| {
                step.frames_generated
                    .saturating_sub(active.last_frames_generated)
            })
            .unwrap_or(0);
        if let Some(step) = &step {
            active.last_frames_generated = step.frames_generated;
            let _ = active.state.take_staged_step();
        }
        let finished = step.as_ref().is_some_and(|step| step.finished);
        let (samples, sample_rate) = if finished {
            let output = model.finalize_retained_state(&active.state)?;
            (output.samples, output.sample_rate)
        } else {
            (Vec::new(), 24_000)
        };
        lease.mark_clean();
        if finished {
            lease.release()?;
        } else {
            lease.restore()?;
        }
        Ok(ModelSessionResult::sequence(ExecutorOutput {
            request_id: request.id.clone(),
            audio: Some(AudioOutput::new(samples, sample_rate)),
            text: None,
            input_transcription: None,
            tokens_processed: scheduled.num_tokens,
            tokens_generated: generated,
            finished,
            phase_timing_override: None,
            asr_diagnostics: None,
            error: None,
        })
        .with_managed_cache_completions(completions))
    }

    pub(super) fn voxtral_tts_request_with_managed_cache(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        mut retained: Option<super::RetainedRowManagedState>,
    ) -> Result<ModelSessionResult> {
        let variant = Self::resolve_variant(request)?;
        if variant.family() != ModelFamily::VoxtralTts {
            return Err(Error::InvalidInput("foreign Voxtral TTS request".into()));
        }
        let model = request
            .prepared_voxtral_tts_model_lease_for_executor()?
            .ok_or_else(|| Error::InferenceError("Voxtral TTS lost model residency".into()))?;
        let mut retained = retained.take().ok_or_else(|| {
            Error::InferenceError("Voxtral TTS lost retained physical state".into())
        })?;
        let mut cache = retained.take_only_paged()?;
        retained.ensure_all_paged_consumed()?;
        let mut lease = ExecutorStateLease::checkout(
            &self.voxtral_tts_decode_states,
            scheduled.session_key(),
            variant,
            "Voxtral TTS decode",
        )?;
        if lease.state().is_some_and(|active| {
            active.variant != variant || !Arc::ptr_eq(&active.model.model_arc(), &model.model_arc())
        }) {
            lease.discard_state();
        }
        let fresh = lease.state().is_none();
        if fresh {
            if !scheduled.is_prefill || scheduled.num_computed_tokens != 0 {
                return Err(Error::InferenceError(
                    "Voxtral TTS lost initial state".into(),
                ));
            }
            let artifact = request
                .prepared_voxtral_tts_artifact_for_executor()?
                .ok_or_else(|| Error::InferenceError("Voxtral TTS lost prompt".into()))?;
            let params = request
                .voxtral_tts_generation_params_for_executor()?
                .ok_or_else(|| Error::InferenceError("Voxtral TTS lost geometry".into()))?;
            let state = model.new_retained_state(artifact, params)?;
            lease.install_state(ActiveVoxtralTtsDecode {
                variant,
                model: model.clone(),
                state,
                last_frames_generated: 0,
                stream_sequence: 0,
            })?;
        }
        let prior_stream_sequence = lease.require_state_mut()?.stream_sequence;
        if scheduled.is_prefill {
            let state = &lease.require_state_mut()?.state;
            let (start, _) = resumable_tts_prefill_span(scheduled, state.prompt_tokens())?;
            if state.prefill_cursor() != start {
                return Err(Error::InferenceError(
                    "Voxtral TTS scheduler and retained prefill cursors diverged".into(),
                ));
            }
        }
        let checkpoint = lease.require_state_mut()?.state.begin_quantum(&cache)?;
        lease.mark_dirty();
        let result = (|| -> Result<_> {
            let active = lease.require_state_mut()?;
            if scheduled.is_prefill {
                let step = model.retained_prefill_step(
                    &mut active.state,
                    &mut cache,
                    &checkpoint,
                    scheduled.num_tokens,
                )?;
                if step.consumed_tokens != scheduled.num_tokens {
                    return Err(Error::InferenceError("Voxtral TTS prefill drifted".into()));
                }
                Ok((false, 0usize))
            } else {
                let step =
                    model.retained_decode_step(&mut active.state, &mut cache, &checkpoint)?;
                Ok((step.finished, step.frames_generated))
            }
        })();
        let (codec_ready, frames) = match result {
            Ok(value) if !request.is_cancelled() => value,
            result => {
                let _ = request.take_staged_stream_outputs();
                lease
                    .require_state_mut()?
                    .state
                    .rollback_quantum(&mut cache, &checkpoint)?;
                lease.require_state_mut()?.stream_sequence = prior_stream_sequence;
                lease.mark_clean();
                if fresh {
                    lease.discard_state();
                }
                return result.and_then(|_| Err(Error::Cancelled(request.id.clone())));
            }
        };
        let completions = cache.take_completed_writes();
        lease
            .require_state_mut()?
            .state
            .commit_quantum(&cache, &checkpoint)?;
        let active = lease.require_state_mut()?;
        let generated = frames.saturating_sub(active.last_frames_generated);
        active.last_frames_generated = frames;
        let audio = AudioOutput::new(
            Vec::new(),
            u32::try_from(model.codec_config.sample_rate)
                .map_err(|_| Error::InferenceError("Voxtral TTS sample rate exceeds u32".into()))?,
        );
        lease.mark_clean();
        lease.restore()?;
        let output = ExecutorOutput {
            request_id: request.id.clone(),
            audio: Some(audio),
            text: None,
            input_transcription: None,
            tokens_processed: scheduled.num_tokens,
            tokens_generated: generated,
            finished: false,
            phase_timing_override: None,
            asr_diagnostics: None,
            error: None,
        };
        let result = if codec_ready {
            ModelSessionResult::yielded(output, crate::engine::YieldReason::AwaitingFinalization)
        } else {
            ModelSessionResult::sequence(output)
        };
        Ok(result.with_managed_cache_completions(completions))
    }

    pub(super) fn voxtral_tts_batch_with_managed(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
        mut managed: Vec<Option<super::RetainedRowManagedState>>,
    ) -> Result<Vec<ModelSessionResult>> {
        if requests.len() != scheduled.len() || managed.len() != scheduled.len() {
            return Err(Error::InvalidInput(
                "Voxtral TTS batch rows do not match".into(),
            ));
        }
        let mut outputs = (0..scheduled.len()).map(|_| None).collect::<Vec<_>>();
        let live = requests
            .iter()
            .enumerate()
            .filter_map(|(index, request)| {
                if request.is_cancelled() {
                    outputs[index] = Some(ModelSessionResult::cancelled_before_dispatch(
                        ExecutorOutput::cancelled(request.id.clone()),
                    ));
                    None
                } else {
                    Some(index)
                }
            })
            .collect::<Vec<_>>();
        if live.is_empty() {
            return outputs
                .into_iter()
                .map(|output| {
                    output.ok_or_else(|| {
                        Error::InferenceError("Voxtral TTS cancelled row lost output".into())
                    })
                })
                .collect();
        }
        let is_prefill = scheduled[live[0]].is_prefill;
        if live
            .iter()
            .any(|index| scheduled[*index].is_prefill != is_prefill)
        {
            return Err(Error::InvalidInput("Voxtral TTS batch mixed phases".into()));
        }
        let model = requests[live[0]]
            .prepared_voxtral_tts_model_lease_for_executor()?
            .ok_or_else(|| Error::InferenceError("Voxtral TTS batch lost model".into()))?;
        let model_arc = model.model_arc();
        let mut rows = Vec::with_capacity(live.len());
        for index in live.iter().copied() {
            let request = requests[index];
            let variant = Self::resolve_variant(request)?;
            let row_model = request
                .prepared_voxtral_tts_model_lease_for_executor()?
                .ok_or_else(|| Error::InferenceError("Voxtral TTS row lost model".into()))?;
            if variant.family() != ModelFamily::VoxtralTts
                || !Arc::ptr_eq(&model_arc, &row_model.model_arc())
            {
                return Err(Error::InvalidInput(
                    "Voxtral TTS batch crossed identity".into(),
                ));
            }
            let mut retained = managed[index].take().ok_or_else(|| {
                Error::InferenceError("Voxtral TTS batch lost physical state".into())
            })?;
            let cache = retained.take_only_paged()?;
            retained.ensure_all_paged_consumed()?;
            let mut lease = ExecutorStateLease::checkout(
                &self.voxtral_tts_decode_states,
                scheduled[index].session_key(),
                variant,
                "batched Voxtral TTS",
            )?;
            if lease.state().is_some_and(|active| {
                active.variant != variant || !Arc::ptr_eq(&active.model.model_arc(), &model_arc)
            }) {
                lease.discard_state();
            }
            if lease.state().is_none() {
                if !is_prefill || scheduled[index].num_computed_tokens != 0 {
                    return Err(Error::InferenceError("Voxtral TTS batch lost state".into()));
                }
                let artifact = request
                    .prepared_voxtral_tts_artifact_for_executor()?
                    .ok_or_else(|| Error::InferenceError("Voxtral TTS row lost prompt".into()))?;
                let params = request
                    .voxtral_tts_generation_params_for_executor()?
                    .ok_or_else(|| Error::InferenceError("Voxtral TTS row lost geometry".into()))?;
                lease.install_state(ActiveVoxtralTtsDecode {
                    variant,
                    model: model.clone(),
                    state: model.new_retained_state(artifact, params)?,
                    last_frames_generated: 0,
                    stream_sequence: 0,
                })?;
            }
            let active = lease.require_state_mut()?;
            if is_prefill {
                let (start, _) =
                    resumable_tts_prefill_span(&scheduled[index], active.state.prompt_tokens())?;
                if active.state.prefill_cursor() != start {
                    return Err(Error::InferenceError(
                        "Voxtral TTS batch scheduler and retained prefill cursors diverged".into(),
                    ));
                }
            }
            let prior_frames = active.last_frames_generated;
            let prior_stream_sequence = active.stream_sequence;
            let checkpoint = active.state.begin_quantum(&cache)?;
            lease.mark_dirty();
            rows.push(ContinuousVoxtralTtsRow {
                index,
                lease: Some(lease),
                cache,
                checkpoint: Some(checkpoint),
                prior_frames,
                prior_stream_sequence,
            });
        }
        let mut batch = ContinuousVoxtralTtsBatch { rows, armed: true };
        for row in &mut batch.rows {
            if requests[row.index].is_cancelled() {
                row.rollback()?;
                outputs[row.index] = Some(ModelSessionResult::cancelled(
                    ExecutorOutput::cancelled(requests[row.index].id.clone()),
                ));
            }
        }
        let call_rows = batch
            .rows
            .iter()
            .enumerate()
            .filter_map(|(row, state)| state.checkpoint.is_some().then_some(row))
            .collect::<Vec<_>>();
        let mut state_refs = Vec::new();
        let mut cache_refs = Vec::new();
        let mut checkpoint_refs = Vec::new();
        let mut spans = Vec::new();
        for (row_index, row) in batch.rows.iter_mut().enumerate() {
            if !call_rows.contains(&row_index) {
                continue;
            }
            state_refs.push(
                &mut row
                    .lease
                    .as_mut()
                    .expect("armed Voxtral lease")
                    .require_state_mut()?
                    .state,
            );
            cache_refs.push(&mut row.cache);
            checkpoint_refs.push(row.checkpoint.as_ref().expect("armed Voxtral checkpoint"));
            spans.push(scheduled[row.index].num_tokens);
        }
        let prefill_result = is_prefill
            .then(|| {
                Self::run_blocking(|| {
                    model.retained_prefill_batch(
                        &mut state_refs,
                        &mut cache_refs,
                        &checkpoint_refs,
                        &spans,
                    )
                })
            })
            .transpose()?;
        let decode_result = (!is_prefill)
            .then(|| {
                Self::run_blocking(|| {
                    model.retained_decode_batch(&mut state_refs, &mut cache_refs, &checkpoint_refs)
                })
            })
            .transpose()?;
        drop((state_refs, cache_refs, checkpoint_refs));
        let (steps, launch_widths) = if let Some(result) = prefill_result {
            if result.steps.len() != call_rows.len() {
                return Err(Error::InferenceError(
                    "Voxtral TTS prefill returned the wrong number of rows".into(),
                ));
            }
            for (step, &call_row) in result.steps.iter().zip(&call_rows) {
                let index = batch.rows[call_row].index;
                validate_voxtral_tts_prefill_step(index, &scheduled[index], step)?;
            }
            (
                result
                    .steps
                    .into_iter()
                    .map(|step| (false, step.prefill_cursor))
                    .collect::<Vec<_>>(),
                result.lm_launch_widths,
            )
        } else if let Some(result) = decode_result {
            (
                result
                    .steps
                    .into_iter()
                    .map(|step| (step.finished, step.frames_generated))
                    .collect(),
                vec![result.acoustic_launch_width, result.lm_launch_width],
            )
        } else {
            unreachable!("Voxtral TTS cohort has one physical phase")
        };
        if steps.len() != call_rows.len() {
            return Err(Error::InferenceError(
                "Voxtral TTS batch returned wrong width".into(),
            ));
        }
        let mode = if is_prefill {
            crate::engine::NativeBatchMode::Static
        } else {
            crate::engine::NativeBatchMode::Continuous
        };
        for width in launch_widths {
            if let Some(call) = retained_tts_batch_model_call(mode, width) {
                crate::engine::metrics::record_engine_model_call(call);
            }
        }
        for row in &mut batch.rows {
            let index = row.index;
            if row.checkpoint.is_none() {
                continue;
            }
            if requests[index].is_cancelled() {
                row.rollback()?;
                outputs[index] = Some(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                    requests[index].id.clone(),
                )));
            }
        }
        // Cancellation is checked again immediately before the all-row commit
        // validation. Codec work is deliberately absent here: terminal rows
        // yield onto the scheduler-visible SequenceFinalize stage.
        for row in &mut batch.rows {
            let index = row.index;
            if row.checkpoint.is_some() && requests[index].is_cancelled() {
                let _ = requests[index].take_staged_stream_outputs();
                row.rollback()?;
                outputs[index] = Some(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                    requests[index].id.clone(),
                )));
            }
        }

        // Validate every host/KV commit before consuming any checkpoint.
        for row in &mut batch.rows {
            let Some(checkpoint) = row.checkpoint.as_ref() else {
                continue;
            };
            row.lease
                .as_mut()
                .expect("armed Voxtral lease")
                .require_state_mut()?
                .state
                .commit_quantum(&row.cache, checkpoint)?;
        }

        let sample_rate = u32::try_from(model.codec_config.sample_rate)
            .map_err(|_| Error::InferenceError("Voxtral sample rate exceeds u32".into()))?;
        for (call_row, (codec_ready, progress)) in call_rows.into_iter().zip(steps) {
            let row = &mut batch.rows[call_row];
            let index = row.index;
            if row.checkpoint.is_none() {
                continue;
            }
            let completions = row.cache.take_completed_writes();
            let lease = row.lease.as_mut().expect("armed Voxtral lease");
            let active = lease.require_state_mut()?;
            let generated = if is_prefill {
                0
            } else {
                progress.saturating_sub(active.last_frames_generated)
            };
            if !is_prefill {
                active.last_frames_generated = progress;
            }
            row.checkpoint = None;
            lease.mark_clean();
            let lease = row.lease.take().expect("armed Voxtral lease");
            lease.restore()?;
            let output = ExecutorOutput {
                request_id: requests[index].id.clone(),
                audio: Some(AudioOutput::new(Vec::new(), sample_rate)),
                text: None,
                input_transcription: None,
                tokens_processed: scheduled[index].num_tokens,
                tokens_generated: generated,
                finished: false,
                phase_timing_override: None,
                asr_diagnostics: None,
                error: None,
            };
            let result = if codec_ready {
                ModelSessionResult::yielded(
                    output,
                    crate::engine::YieldReason::AwaitingFinalization,
                )
            } else {
                ModelSessionResult::sequence(output)
            };
            outputs[index] = Some(result.with_managed_cache_completions(completions));
        }
        batch.armed = false;
        outputs
            .into_iter()
            .map(|output| {
                output.ok_or_else(|| Error::InferenceError("Voxtral TTS row lost output".into()))
            })
            .collect()
    }

    pub(super) fn voxtral_tts_finalize_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
    ) -> Result<ModelSessionResult> {
        if !matches!(
            scheduled.work,
            crate::engine::WorkUnit::SequenceFinalize {
                max_output_steps: 1
            }
        ) || scheduled.is_prefill
        {
            return Err(Error::InvalidInput(
                "Voxtral TTS codec requires an exact sequence-finalize quantum".into(),
            ));
        }
        let variant = Self::resolve_variant(request)?;
        if variant.family() != ModelFamily::VoxtralTts {
            return Err(Error::InvalidInput(
                "foreign Voxtral TTS codec request".into(),
            ));
        }
        let model = request
            .prepared_voxtral_tts_model_lease_for_executor()?
            .ok_or_else(|| {
                Error::InferenceError("Voxtral TTS codec lost model residency".into())
            })?;
        let mut lease = ExecutorStateLease::checkout(
            &self.voxtral_tts_decode_states,
            scheduled.session_key(),
            variant,
            "Voxtral TTS codec",
        )?;
        let active = lease.require_state_mut()?;
        if active.variant != variant
            || !Arc::ptr_eq(&active.model.model_arc(), &model.model_arc())
            || !active.state.codec_ready()
        {
            return Err(Error::InferenceError(
                "Voxtral TTS codec state crossed its model or phase fence".into(),
            ));
        }
        let prior_stream_sequence = active.stream_sequence;
        let checkpoint = active.state.begin_codec_quantum()?;
        lease.mark_dirty();
        let result = (|| -> Result<AudioOutput> {
            let active = lease.require_state_mut()?;
            let output = model.retained_codec_finalize(&mut active.state)?;
            let sample_rate = u32::try_from(output.sample_rate)
                .map_err(|_| Error::InferenceError("Voxtral sample rate exceeds u32".into()))?;
            if let Some(tx) = Self::stream_sender(request) {
                Self::stream_audio_with_policy(
                    &tx,
                    request.stream_policy,
                    &request.id,
                    &mut active.stream_sequence,
                    output.samples.clone(),
                    sample_rate,
                    false,
                )?;
                Self::stream_final_marker_with_policy(
                    &tx,
                    request.stream_policy,
                    &request.id,
                    &mut active.stream_sequence,
                )?;
            }
            if request.is_cancelled() {
                return Err(Error::Cancelled(request.id.clone()));
            }
            active.state.commit_codec_quantum(&checkpoint)?;
            Ok(AudioOutput::new(output.samples, sample_rate))
        })();
        let audio = match result {
            Ok(audio) => audio,
            Err(error) => {
                let _ = request.take_staged_stream_outputs();
                let active = lease.require_state_mut()?;
                active.state.rollback_codec_quantum(&checkpoint);
                active.stream_sequence = prior_stream_sequence;
                lease.mark_clean();
                lease.restore()?;
                return Err(error);
            }
        };
        lease.mark_clean();
        lease.release()?;
        Ok(ModelSessionResult::sequence(ExecutorOutput {
            request_id: request.id.clone(),
            audio: Some(audio),
            text: None,
            input_transcription: None,
            tokens_processed: 0,
            tokens_generated: 0,
            finished: true,
            phase_timing_override: None,
            asr_diagnostics: None,
            error: None,
        }))
    }

    pub(super) fn fish_s2_tts_request_with_managed_cache(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        retained: Option<super::RetainedRowManagedState>,
    ) -> Result<ModelSessionResult> {
        use crate::models::architectures::fish_s2::FishS2RetainedStep;

        let variant = Self::resolve_variant(request)?;
        if variant.family() != ModelFamily::FishS2Tts {
            return Err(Error::InvalidInput("foreign Fish S2 TTS request".into()));
        }
        let model = request
            .prepared_fish_s2_tts_model_lease_for_executor()?
            .ok_or_else(|| Error::InferenceError("Fish S2 TTS lost model residency".into()))?;
        let mut retained = retained.ok_or_else(|| {
            Error::InferenceError("Fish S2 TTS lost retained physical state".into())
        })?;
        let slow = retained
            .take_paged_domain(crate::kv::CacheDomainId::new(1), true)?
            .expect("required Fish S2 slow cache");
        let fast = retained
            .take_paged_domain(crate::kv::CacheDomainId::new(2), true)?
            .expect("required Fish S2 fast cache");
        retained.ensure_all_paged_consumed()?;
        let mut lease = ExecutorStateLease::checkout(
            &self.fish_s2_tts_decode_states,
            scheduled.session_key(),
            variant,
            "Fish S2 TTS decode",
        )?;
        if lease.state().is_some_and(|active| {
            active.variant != variant || !Arc::ptr_eq(&active.model.model_arc(), &model.model_arc())
        }) {
            lease.discard_state();
        }
        let fresh = lease.state().is_none();
        let mut checkpoint = if fresh {
            if !scheduled.is_prefill || scheduled.num_computed_tokens != 0 {
                return Err(Error::InferenceError(
                    "Fish S2 TTS lost state before initial prefill".into(),
                ));
            }
            let artifact = request
                .prepared_fish_s2_tts_artifact_for_executor()?
                .ok_or_else(|| Error::InferenceError("Fish S2 TTS lost prompt".into()))?;
            let params = request
                .fish_s2_tts_generation_params_for_executor()?
                .ok_or_else(|| Error::InferenceError("Fish S2 TTS lost geometry".into()))?;
            let (state, checkpoint) =
                model.new_retained_state_in_quantum(artifact, params, slow, fast)?;
            lease.install_state(ActiveFishS2TtsDecode {
                variant,
                model: model.clone(),
                state,
                last_frames_generated: 0,
                stream_sequence: 0,
            })?;
            checkpoint
        } else {
            lease
                .require_state_mut()?
                .state
                .begin_managed_quantum(slow, fast)?
        };
        lease.mark_dirty();
        let result = (|| {
            let active = lease.require_state_mut()?;
            let step = if scheduled.is_prefill {
                let step = model.retained_prefill_step(&mut active.state, scheduled.num_tokens)?;
                if !matches!(
                    step,
                    FishS2RetainedStep::Prefill { consumed, .. }
                        if consumed == scheduled.num_tokens
                ) {
                    return Err(Error::InferenceError(
                        "Fish S2 TTS prefill progress differs from scheduler".into(),
                    ));
                }
                Some(step)
            } else {
                Some(model.retained_decode_step(&mut active.state)?)
            };
            if request.is_cancelled() {
                return Err(Error::Cancelled(request.id.clone()));
            }
            Ok::<_, Error>(step)
        })();
        let step = match result {
            Ok(step) if !request.is_cancelled() => step,
            result => {
                if fresh {
                    let _ = lease
                        .require_state_mut()?
                        .state
                        .take_managed_write_completions();
                    lease.discard_state();
                } else {
                    lease
                        .require_state_mut()?
                        .state
                        .rollback_managed_quantum(&mut checkpoint)?;
                }
                lease.mark_clean();
                return result.and_then(|_| Err(Error::Cancelled(request.id.clone())));
            }
        };
        let completions = lease
            .require_state_mut()?
            .state
            .take_managed_write_completions();
        let staged = lease.require_state_mut()?.state.take_staged_step();
        if step.is_some() && staged != step {
            if fresh {
                lease.discard_state();
            } else {
                lease
                    .require_state_mut()?
                    .state
                    .rollback_managed_quantum(&mut checkpoint)?;
            }
            lease.mark_clean();
            return Err(Error::InferenceError(
                "Fish S2 TTS staged output changed before commit".into(),
            ));
        }
        lease
            .require_state_mut()?
            .state
            .commit_managed_quantum(&mut checkpoint)?;
        let active = lease.require_state_mut()?;
        let frames_generated = active.state.frames_generated();
        let generated = frames_generated.saturating_sub(active.last_frames_generated);
        active.last_frames_generated = frames_generated;
        let finished = matches!(step, Some(FishS2RetainedStep::Finished { .. }));
        let (samples, sample_rate) = if finished {
            let output = model.finalize_retained_state(&active.state)?;
            (output.samples, output.sample_rate)
        } else {
            (Vec::new(), model.diagnostics().sample_rate)
        };
        if generated > 0 {
            active.stream_sequence = active.stream_sequence.saturating_add(1);
        }
        lease.mark_clean();
        if finished {
            lease.release()?;
        } else {
            lease.restore()?;
        }
        Ok(ModelSessionResult::sequence(ExecutorOutput {
            request_id: request.id.clone(),
            audio: Some(AudioOutput::new(samples, sample_rate)),
            text: None,
            input_transcription: None,
            tokens_processed: scheduled.num_tokens,
            tokens_generated: generated,
            finished,
            phase_timing_override: None,
            asr_diagnostics: None,
            error: None,
        })
        .with_managed_cache_completions(completions))
    }

    pub(super) fn lfm25_audio_tts_request_with_managed_cache(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        retained: Option<super::RetainedRowManagedState>,
    ) -> Result<ModelSessionResult> {
        let variant = Self::resolve_variant(request)?;
        if variant.family() != ModelFamily::Lfm25Audio {
            return Err(Error::InvalidInput(
                "foreign LFM2.5 Audio TTS request".into(),
            ));
        }
        let model = request
            .prepared_lfm25_audio_tts_model_lease_for_executor()?
            .ok_or_else(|| Error::InferenceError("LFM2.5 Audio TTS lost model residency".into()))?;
        let mut retained = retained
            .ok_or_else(|| Error::InferenceError("LFM2.5 Audio TTS lost retained state".into()))?;
        let tensor = retained.tensor_state.clone().ok_or_else(|| {
            Error::InferenceError("LFM2.5 Audio TTS lost ShortConv reservation".into())
        })?;
        let arena = request
            .managed_cache_runtime()
            .and_then(|runtime| runtime.tensor_state())
            .ok_or_else(|| Error::InferenceError("LFM2.5 Audio TTS lost ShortConv arena".into()))?;
        let mut main = retained.take_only_paged()?;
        retained.ensure_all_paged_consumed()?;
        let mut lease = ExecutorStateLease::checkout(
            &self.lfm25_tts_decode_states,
            scheduled.session_key(),
            variant,
            "LFM2.5 Audio TTS decode",
        )?;
        if lease.state().is_some_and(|active| {
            active.variant != variant || !Arc::ptr_eq(&active.model.model_arc(), &model.model_arc())
        }) {
            lease.discard_state();
        }
        let fresh = lease.state().is_none();
        if fresh {
            if !scheduled.is_prefill || scheduled.num_computed_tokens != 0 {
                return Err(Error::InferenceError(
                    "LFM2.5 Audio TTS lost state before initial prefill".into(),
                ));
            }
            let artifact = request
                .prepared_lfm25_audio_tts_artifact_for_executor()?
                .ok_or_else(|| Error::InferenceError("LFM2.5 Audio TTS lost prompt".into()))?;
            if artifact.prompt_tokens != request.num_prompt_tokens() {
                return Err(Error::InferenceError(
                    "LFM2.5 Audio TTS prompt differs from admission".into(),
                ));
            }
            let state = model.new_lfm25_audio_retained_tts_state(
                artifact,
                request.params.max_tokens.max(1),
                request.lfm25_audio_tts_generation_config(),
            )?;
            lease.install_state(ActiveLfm25TtsDecode {
                variant,
                model: model.clone(),
                state,
                last_tokens_generated: 0,
                stream_sequence: 0,
            })?;
        }
        let prior = {
            let active = lease.require_state_mut()?;
            active.state.bind_tensor_sequence(tensor.sequence)?;
            active.state.restore_shortconv(arena)?;
            (active.last_tokens_generated, active.stream_sequence)
        };
        let needs_depth =
            !scheduled.is_prefill && lease.require_state_mut()?.state.decode_needs_depthformer();
        let mut depth = needs_depth
            .then(|| super::invocation_paged_lease_for_row(request, scheduled))
            .transpose()?;
        let checkpoint = lease
            .require_state_mut()?
            .state
            .reset_and_begin_quantum(&main, depth.as_mut().map(|depth| depth.cache_mut()))?;
        lease.mark_dirty();
        let step = (|| {
            let active = lease.require_state_mut()?;
            let step = if scheduled.is_prefill {
                if active.state.prefill_cursor() != scheduled.num_computed_tokens {
                    return Err(Error::InferenceError(
                        "LFM2.5 Audio TTS prefill cursor differs from admission".into(),
                    ));
                }
                let step = model.lfm25_audio_tts_prefill_step(
                    &mut active.state,
                    &mut main,
                    &checkpoint,
                    scheduled.num_tokens,
                )?;
                if step.consumed_tokens != scheduled.num_tokens {
                    return Err(Error::InferenceError(
                        "LFM2.5 Audio TTS prefill progress is inconsistent".into(),
                    ));
                }
                None
            } else {
                Some(model.lfm25_audio_tts_decode_step(
                    &mut active.state,
                    &mut main,
                    depth.as_mut().map(|depth| depth.cache_mut()),
                    &checkpoint,
                )?)
            };
            if request.is_cancelled() {
                return Err(Error::Cancelled(request.id.clone()));
            }
            active.state.stage_shortconv(arena, scheduled.plan_id)?;
            Ok::<_, Error>(step)
        })();
        let step = match step {
            Ok(step) if !request.is_cancelled() => step,
            result => {
                let active = lease.require_state_mut()?;
                active.state.rollback_quantum(
                    &mut main,
                    depth.as_mut().map(|depth| depth.cache_mut()),
                    &checkpoint,
                )?;
                active.last_tokens_generated = prior.0;
                active.stream_sequence = prior.1;
                lease.mark_clean();
                if request.is_cancelled() || matches!(result, Err(Error::Cancelled(_))) {
                    lease.release()?;
                    if let Some(depth) = depth {
                        let _ = depth.release()?;
                    }
                    return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                        request.id.clone(),
                    )));
                }
                if fresh {
                    lease.discard_state();
                }
                return Err(result.expect_err("non-cancelled failed TTS quantum"));
            }
        };
        let completions = main.take_completed_writes();
        lease.require_state_mut()?.state.commit_quantum(
            &main,
            depth.as_ref().map(|depth| depth.cache()),
            &checkpoint,
        )?;
        if let Some(depth) = depth {
            let _ = depth.release()?;
        }
        let (text, generated, finished, samples, sample_rate) = {
            let active = lease.require_state_mut()?;
            let generated = step.as_ref().map_or(0, |step| {
                step.tokens_generated
                    .saturating_sub(active.last_tokens_generated)
            });
            if let Some(step) = step.as_ref() {
                active.last_tokens_generated = step.tokens_generated;
            }
            let finished = step.as_ref().is_some_and(|step| step.finished);
            let samples = if finished {
                model.detokenize_lfm25_audio_retained_tts_state(&active.state)?
            } else {
                Vec::new()
            };
            (
                active.state.text().to_string(),
                generated,
                finished,
                samples,
                model.lfm25_audio_tts_output_sample_rate(),
            )
        };
        lease.mark_clean();
        if finished {
            lease.release()?;
        } else {
            lease.restore()?;
        }
        Ok(ModelSessionResult::sequence(ExecutorOutput {
            request_id: request.id.clone(),
            audio: Some(AudioOutput {
                duration_secs: samples.len() as f32 / sample_rate as f32,
                samples,
                sample_rate,
            }),
            text: Some(text),
            input_transcription: None,
            tokens_processed: scheduled.num_tokens,
            tokens_generated: generated,
            finished,
            phase_timing_override: None,
            asr_diagnostics: None,
            error: None,
        })
        .with_managed_cache_completions(completions))
    }

    pub(super) fn qwen_tts_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
    ) -> Result<ModelSessionResult> {
        self.qwen_tts_request_with_managed_cache(request, scheduled, None, None)
    }

    pub(super) fn qwen_tts_request_with_managed_cache(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        mut talker_cache: Option<TalkerPhysicalCache>,
        tensor_reservation: Option<crate::engine::ManagedTensorStateReservation>,
    ) -> Result<ModelSessionResult> {
        let tensor_arena = request
            .managed_cache_runtime()
            .and_then(|runtime| runtime.tensor_state().cloned());
        validate_qwen_tts_physical_bindings(
            request.managed_cache_runtime().is_some(),
            talker_cache.is_some(),
            tensor_arena.is_some(),
            tensor_reservation.is_some(),
        )?;
        if scheduled.is_prefill {
            resumable_tts_prefill_span(scheduled, request.num_prompt_tokens())?;
        }
        let execution_started = Instant::now();
        let stream_tx = Self::stream_sender(request);
        let stream_policy = request.stream_policy;
        let variant = request.model_variant;
        let marker_variant = variant.ok_or_else(|| {
            Error::InvalidInput("Qwen TTS request is missing its model variant".into())
        })?;
        let params = Self::to_tts_params(request);
        let language = request.language.as_deref();
        let session = scheduled.session_key();

        {
            let mut state_lease = ExecutorStateLease::checkout(
                &self.qwen_tts_decode_states,
                session,
                marker_variant,
                "Qwen TTS decode",
            )?;
            if state_lease
                .state()
                .map(|state| state.variant != variant)
                .unwrap_or(false)
            {
                state_lease.discard_state();
            }
            let fresh_state = state_lease.state().is_none();
            let (model, new_model_lease) = if let Some(state) = state_lease.state() {
                (state.model.clone(), None)
            } else {
                let (model, lease) = self.qwen_model_for_request(request)?;
                (model, lease)
            };
            let model_arc = model;
            let model = model_arc.as_ref();

            if state_lease.state().is_none() {
                if request.is_cancelled() {
                    state_lease.release()?;
                    return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                        request.id.clone(),
                    )));
                }
                let normalization_started = Instant::now();
                let text = request
                    .text
                    .as_deref()
                    .ok_or_else(|| Error::InvalidInput("TTS request missing text".to_string()))?;
                let prepared = request.prepared_qwen_tts_input_for_executor()?;
                let stream_config = if stream_tx.is_some() {
                    TtsStreamingConfig::default()
                } else {
                    TtsStreamingConfig::final_only()
                };
                let normalization_ms = normalization_started.elapsed().as_secs_f64() * 1000.0;
                let cache = talker_cache.take().ok_or_else(|| {
                    Error::InferenceError(
                        "Qwen TTS prefill lost its retained talker reservation".to_string(),
                    )
                })?;

                let prefill_state = if let Some(reference) = prepared.reference.as_deref() {
                    Self::run_blocking(|| {
                        model.begin_physical_prefill_with_voice_clone_params(
                            text,
                            reference,
                            language,
                            &params,
                            stream_config,
                            cache,
                        )
                    })?
                } else if let Some(speaker) = prepared.speaker.as_deref() {
                    Self::run_blocking(|| {
                        model.begin_physical_prefill_with_speaker_params(
                            text,
                            speaker,
                            language,
                            request.voice_description.as_deref(),
                            &params,
                            stream_config,
                            cache,
                        )
                    })?
                } else {
                    Self::run_blocking(|| {
                        model.begin_physical_prefill_with_text_params(
                            text,
                            language,
                            request.voice_description.as_deref(),
                            &params,
                            stream_config,
                            cache,
                        )
                    })?
                };
                if prefill_state.prefill_tokens() != prepared.prefill_tokens {
                    return Err(Error::InferenceError(format!(
                        "Qwen TTS runtime prepared {} tokens, but admission authorized {}",
                        prefill_state.prefill_tokens(),
                        prepared.prefill_tokens
                    )));
                }

                let mut active = ActiveQwenTtsDecode {
                    variant,
                    model: model_arc.clone(),
                    _model_lease: new_model_lease,
                    state: QwenTtsPhysicalState::Prefill(prefill_state),
                    last_frames_generated: 0,
                    stream_sequence: 0,
                    audio_samples_accum: Vec::new(),
                    execution_started,
                    normalization_ms,
                    prefill_ms: 0.0,
                    sampling_ms: 0.0,
                    decode_ms: 0.0,
                    codec_ms: 0.0,
                    postprocess_ms: 0.0,
                    first_output_ms_since_start: None,
                    prefill_steps: 0,
                    decode_steps: 0,
                };
                if let Some(ref reservation) = tensor_reservation {
                    let QwenTtsPhysicalState::Prefill(state) = &mut active.state else {
                        unreachable!("fresh Qwen3-TTS state is prefill")
                    };
                    state.bind_tensor_sequence(reservation.sequence)?;
                }
                state_lease.install_state(active)?;
            }

            let outer_checkpoint =
                ActiveTtsOuterCheckpoint::capture(state_lease.require_state_mut()?);
            let mut managed_checkpoint = if fresh_state {
                None
            } else {
                let cache = talker_cache.take().ok_or_else(|| {
                    Error::InferenceError(
                        "active Qwen TTS state lost its talker reservation".into(),
                    )
                })?;
                let active = state_lease.require_state_mut()?;
                Some(match &mut active.state {
                    QwenTtsPhysicalState::Prefill(state) => {
                        TtsManagedCheckpoint::Prefill(state.begin_managed_quantum(cache)?)
                    }
                    QwenTtsPhysicalState::Decode(state) => {
                        TtsManagedCheckpoint::Decode(state.begin_managed_quantum(cache)?)
                    }
                    QwenTtsPhysicalState::Transitioning => {
                        return Err(Error::InferenceError(
                            "Qwen3-TTS state was left in a transition".into(),
                        ));
                    }
                })
            };
            if let Some(reservation) = tensor_reservation {
                let hydration = (|| -> Result<()> {
                    let active = state_lease.require_state_mut()?;
                    match &mut active.state {
                        QwenTtsPhysicalState::Prefill(state) => {
                            state.bind_tensor_sequence(reservation.sequence)?;
                        }
                        QwenTtsPhysicalState::Decode(state) => {
                            state.bind_tensor_sequence(reservation.sequence)?;
                            state.restore_tensor_state(tensor_arena.as_ref().ok_or_else(
                                || {
                                    Error::InferenceError(
                                        "Qwen3-TTS tensor arena disappeared".into(),
                                    )
                                },
                            )?)?;
                        }
                        QwenTtsPhysicalState::Transitioning => {
                            return Err(Error::InferenceError(
                                "Qwen3-TTS state was left in a transition".into(),
                            ));
                        }
                    }
                    Ok(())
                })();
                if let Err(error) = hydration {
                    if let Some(checkpoint) = managed_checkpoint.take() {
                        rollback_tts_quantum(
                            state_lease.require_state_mut()?,
                            checkpoint,
                            outer_checkpoint,
                        )?;
                        state_lease.mark_clean();
                    } else {
                        state_lease.discard_state();
                    }
                    return Err(error);
                }
            }

            if request.is_cancelled() {
                if let Some(checkpoint) = managed_checkpoint.take() {
                    rollback_tts_quantum(
                        state_lease.require_state_mut()?,
                        checkpoint,
                        outer_checkpoint,
                    )?;
                    state_lease.mark_clean();
                } else {
                    state_lease.discard_state();
                }
                state_lease.release()?;
                return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                    request.id.clone(),
                )));
            }

            state_lease.mark_dirty();
            let execution = (|| -> Result<(
                usize,
                usize,
                bool,
                Vec<f32>,
                Option<ExecutorPhaseTiming>,
                Vec<Arc<crate::backends::kv::KvWriteBatchCompletion>>,
            )> {
                let active_state = state_lease.require_state_mut()?;
                if scheduled.is_prefill {
                    let (span_start, span_end) =
                        resumable_tts_prefill_span(scheduled, request.num_prompt_tokens())?;
                    let prefill_started = Instant::now();
                    let complete = {
                        let QwenTtsPhysicalState::Prefill(state) = &mut active_state.state else {
                            return Err(Error::InferenceError(
                                "Qwen3-TTS prefill work reached a decode state".into(),
                            ));
                        };
                        Self::run_blocking(|| {
                            active_state.model.continue_physical_prefill(
                                state,
                                span_start,
                                span_end,
                            )
                        })?
                    };
                    active_state.prefill_ms +=
                        prefill_started.elapsed().as_secs_f64() * 1000.0;
                    active_state.prefill_steps = active_state.prefill_steps.saturating_add(1);
                    if request.is_cancelled() {
                        return Err(Error::Cancelled("Qwen3-TTS prefill cancelled".into()));
                    }
                    if let Some(arena) = tensor_arena.as_ref() {
                        let QwenTtsPhysicalState::Prefill(state) = &active_state.state else {
                            unreachable!("prefill state remains prefill before transition")
                        };
                        state.stage_tensor_state(arena, scheduled.plan_id)?;
                    }
                    if request.is_cancelled() {
                        return Err(Error::Cancelled("Qwen3-TTS prefill cancelled".into()));
                    }
                    let completions = match &mut active_state.state {
                        QwenTtsPhysicalState::Prefill(state) => {
                            state.take_managed_write_completions()
                        }
                        _ => unreachable!("prefill state remains prefill before transition"),
                    };
                    if complete {
                        let prefill = match std::mem::replace(
                            &mut active_state.state,
                            QwenTtsPhysicalState::Transitioning,
                        ) {
                            QwenTtsPhysicalState::Prefill(state) => state,
                            _ => unreachable!("validated prefill transition"),
                        };
                        active_state.state = QwenTtsPhysicalState::Decode(
                            active_state.model.finish_physical_prefill(prefill)?,
                        );
                    }
                    let timing = Some(ExecutorPhaseTiming {
                        normalization_ms: Some(active_state.normalization_ms),
                        prefill_ms: Some(active_state.prefill_ms),
                        prefill_steps: Some(active_state.prefill_steps),
                        ..ExecutorPhaseTiming::default()
                    });
                    return Ok((
                        scheduled.num_tokens,
                        0,
                        false,
                        Vec::new(),
                        timing,
                        completions,
                    ));
                }

                let decode_iterations = qwen_tts_decode_iterations(scheduled);
                let start_cursor = match &active_state.state {
                    QwenTtsPhysicalState::Decode(state) => state.talker_context_len(),
                    _ => {
                        return Err(Error::InferenceError(
                            "Qwen3-TTS decode work reached an unfinished prefill".into(),
                        ));
                    }
                };
                let mut finished = false;
                let mut stream_events = Vec::new();
                for _ in 0..decode_iterations {
                    if request.is_cancelled() {
                        return Err(Error::Cancelled("Qwen3-TTS decode cancelled".into()));
                    }
                    let mut predictor = super::invocation_paged_lease_for_row(request, scheduled)?;
                    let step = Self::run_blocking(|| {
                        let QwenTtsPhysicalState::Decode(state) = &mut active_state.state else {
                            unreachable!("validated Qwen3-TTS decode state")
                        };
                        active_state
                            .model
                            .tts_decode_step_physical(state, predictor.cache_mut())
                    })?;
                    let _ = predictor.release()?;
                    if request.is_cancelled() {
                        return Err(Error::Cancelled("Qwen3-TTS decode cancelled".into()));
                    }
                    active_state.sampling_ms += step.sampling_ms;
                    active_state.decode_ms += step.decode_ms;
                    active_state.codec_ms += step.codec_ms;
                    active_state.decode_steps = active_state.decode_steps.saturating_add(1);
                    active_state.last_frames_generated = step.frames_generated;
                    if !step.samples.is_empty() {
                        if active_state.first_output_ms_since_start.is_none() {
                            active_state.first_output_ms_since_start = Some(
                                active_state.execution_started.elapsed().as_secs_f64() * 1000.0,
                            );
                        }
                        active_state
                            .audio_samples_accum
                            .extend_from_slice(&step.samples);
                        stream_events.push((step.samples, false));
                    }
                    if step.finished {
                        stream_events.push((Vec::new(), true));
                        finished = true;
                        break;
                    }
                }
                let (end_cursor, completions) = match &mut active_state.state {
                    QwenTtsPhysicalState::Decode(state) => {
                        let cursor = state.talker_context_len();
                        if let Some(arena) = tensor_arena.as_ref() {
                            state.stage_tensor_state(arena, scheduled.plan_id)?;
                        }
                        (cursor, state.take_managed_write_completions())
                    }
                    _ => unreachable!("validated Qwen3-TTS decode state"),
                };
                let accepted =
                    accepted_tts_talker_tokens(start_cursor, end_cursor, scheduled.num_tokens)?;
                if let Some(tx) = stream_tx.as_ref() {
                    for (samples, is_final) in stream_events {
                        Self::stream_audio_with_policy(
                            tx,
                            stream_policy,
                            &request.id,
                            &mut active_state.stream_sequence,
                            samples,
                            if is_final { 0 } else { 24_000 },
                            is_final,
                        )?;
                    }
                }
                if request.is_cancelled() {
                    return Err(Error::Cancelled("Qwen3-TTS decode cancelled".into()));
                }
                let postprocess_started = Instant::now();
                let finished_samples = if finished {
                    active_state.audio_samples_accum.clone()
                } else {
                    Vec::new()
                };
                active_state.postprocess_ms +=
                    postprocess_started.elapsed().as_secs_f64() * 1000.0;
                let timing = Some(ExecutorPhaseTiming {
                    normalization_ms: Some(active_state.normalization_ms),
                    prefill_ms: Some(active_state.prefill_ms),
                    decode_ms: Some(active_state.decode_ms),
                    sampling_ms: Some(active_state.sampling_ms),
                    codec_ms: Some(active_state.codec_ms),
                    postprocess_ms: Some(active_state.postprocess_ms),
                    first_output_ms_since_start: active_state.first_output_ms_since_start,
                    prefill_steps: Some(active_state.prefill_steps),
                    decode_steps: Some(active_state.decode_steps),
                    ..ExecutorPhaseTiming::default()
                });
                Ok((
                    accepted,
                    accepted,
                    finished,
                    finished_samples,
                    timing,
                    completions,
                ))
            })();

            let (
                tokens_processed,
                total_tokens_generated,
                finished,
                finished_samples,
                phase_timing_override,
                managed_cache_completions,
            ) = match execution {
                Ok(value) => value,
                Err(error) => {
                    let _ = request.take_staged_stream_outputs();
                    if let Some(checkpoint) = managed_checkpoint.take() {
                        rollback_tts_quantum(
                            state_lease.require_state_mut()?,
                            checkpoint,
                            outer_checkpoint,
                        )?;
                        state_lease.mark_clean();
                    } else {
                        state_lease.discard_state();
                    }
                    return if matches!(error, Error::Cancelled(_)) {
                        state_lease.release()?;
                        Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                            request.id.clone(),
                        )))
                    } else {
                        Err(error)
                    };
                }
            };
            // This is the commit fence for both final prefill and decode. A
            // cancellation may arrive after model/tensor/stream staging, so do
            // not disarm the managed checkpoint or publish cache completions
            // until the request has passed this last gate.
            if request.is_cancelled() {
                let _ = request.take_staged_stream_outputs()?;
                if let Some(checkpoint) = managed_checkpoint.take() {
                    rollback_tts_quantum(
                        state_lease.require_state_mut()?,
                        checkpoint,
                        outer_checkpoint,
                    )?;
                    state_lease.mark_clean();
                } else {
                    state_lease.discard_state();
                }
                state_lease.release()?;
                return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                    request.id.clone(),
                )));
            }
            drop(managed_checkpoint);

            let transition = if finished {
                state_lease.release()
            } else {
                state_lease.restore()
            };
            if let Err(error) = transition {
                let _ = request.take_staged_stream_outputs();
                return Err(error);
            }

            Ok(ModelSessionResult::sequence(ExecutorOutput {
                request_id: request.id.clone(),
                audio: Some(AudioOutput::new(finished_samples, 24_000)),
                text: None,
                input_transcription: None,
                tokens_processed,
                tokens_generated: total_tokens_generated,
                finished,
                phase_timing_override,
                asr_diagnostics: None,
                error: None,
            })
            .with_managed_cache_completions(managed_cache_completions))
        }
    }

    pub(super) fn lfm25_audio_tts_prefill_batch_with_managed(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
        mut managed: Vec<Option<super::RetainedRowManagedState>>,
    ) -> Result<Vec<ModelSessionResult>> {
        if requests.len() != scheduled.len() || managed.len() != scheduled.len() {
            return Err(Error::InvalidInput(
                "LFM2.5 Audio TTS prefill batch rows do not match".into(),
            ));
        }
        let mut outputs = (0..scheduled.len()).map(|_| None).collect::<Vec<_>>();
        let live = requests
            .iter()
            .enumerate()
            .filter_map(|(index, request)| {
                if request.is_cancelled() {
                    outputs[index] = Some(ModelSessionResult::cancelled_before_dispatch(
                        ExecutorOutput::cancelled(request.id.clone()),
                    ));
                    None
                } else {
                    Some(index)
                }
            })
            .collect::<Vec<_>>();
        if live.is_empty() {
            return outputs
                .into_iter()
                .map(|output| {
                    output
                        .ok_or_else(|| Error::InferenceError("LFM TTS prefill lost output".into()))
                })
                .collect();
        }
        let model = requests[live[0]]
            .prepared_lfm25_audio_tts_model_lease_for_executor()?
            .ok_or_else(|| Error::InferenceError("LFM TTS prefill lost model".into()))?;
        let model_arc = model.model_arc();
        let mut rows = Vec::with_capacity(live.len());
        for index in live.iter().copied() {
            if !scheduled[index].is_prefill {
                return Err(Error::InvalidInput(
                    "LFM TTS prefill batch contains a decode row".into(),
                ));
            }
            let variant = Self::resolve_variant(requests[index])?;
            let row_model = requests[index]
                .prepared_lfm25_audio_tts_model_lease_for_executor()?
                .ok_or_else(|| Error::InferenceError("LFM TTS prefill row lost model".into()))?;
            if !Arc::ptr_eq(&model_arc, &row_model.model_arc()) {
                return Err(Error::InferenceError(
                    "LFM TTS prefill crossed model identity".into(),
                ));
            }
            let mut retained = managed[index].take().ok_or_else(|| {
                Error::InferenceError("LFM TTS prefill lost retained state".into())
            })?;
            let tensor = retained.tensor_state.clone().ok_or_else(|| {
                Error::InferenceError("LFM TTS prefill lost ShortConv reservation".into())
            })?;
            let arena = requests[index]
                .managed_cache_runtime()
                .and_then(|runtime| runtime.tensor_state())
                .ok_or_else(|| {
                    Error::InferenceError("LFM TTS prefill lost ShortConv arena".into())
                })?;
            let main = retained.take_only_paged()?;
            retained.ensure_all_paged_consumed()?;
            let mut lease = ExecutorStateLease::checkout(
                &self.lfm25_tts_decode_states,
                scheduled[index].session_key(),
                variant,
                "batched LFM2.5 Audio TTS prefill",
            )?;
            if lease.state().is_some_and(|active| {
                active.variant != variant || !Arc::ptr_eq(&active.model.model_arc(), &model_arc)
            }) {
                lease.discard_state();
            }
            if lease.state().is_none() {
                if scheduled[index].num_computed_tokens != 0 {
                    return Err(Error::InferenceError(
                        "LFM TTS prefill lost state after initial quantum".into(),
                    ));
                }
                let artifact = requests[index]
                    .prepared_lfm25_audio_tts_artifact_for_executor()?
                    .ok_or_else(|| Error::InferenceError("LFM TTS prefill lost prompt".into()))?;
                if artifact.prompt_tokens != requests[index].num_prompt_tokens() {
                    return Err(Error::InferenceError(
                        "LFM TTS prefill prompt differs from admission".into(),
                    ));
                }
                let state = model.new_lfm25_audio_retained_tts_state(
                    artifact,
                    requests[index].params.max_tokens.max(1),
                    requests[index].lfm25_audio_tts_generation_config(),
                )?;
                lease.install_state(ActiveLfm25TtsDecode {
                    variant,
                    model: model.clone(),
                    state,
                    last_tokens_generated: 0,
                    stream_sequence: 0,
                })?;
            }
            let active = lease.require_state_mut()?;
            active.state.bind_tensor_sequence(tensor.sequence)?;
            active.state.restore_shortconv(arena)?;
            if active.state.prefill_cursor() != scheduled[index].num_computed_tokens {
                return Err(Error::InferenceError(
                    "LFM TTS prefill cursor differs from admission".into(),
                ));
            }
            let prior_tokens = active.last_tokens_generated;
            let prior_stream_sequence = active.stream_sequence;
            let checkpoint = active.state.reset_and_begin_quantum(&main, None)?;
            lease.mark_dirty();
            rows.push(ContinuousLfm25TtsRow {
                index,
                session: scheduled[index].session_key(),
                lease: Some(lease),
                main,
                depth: None,
                checkpoint: Some(checkpoint),
                prior_tokens,
                prior_stream_sequence,
            });
        }
        let mut batch = ContinuousLfm25TtsBatch::new(rows);
        for row in 0..batch.rows.len() {
            let index = batch.rows[row].index;
            if requests[index].is_cancelled() {
                batch.rollback_row(row)?;
                outputs[index] = Some(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                    requests[index].id.clone(),
                )));
            }
        }
        let call_rows = batch
            .rows
            .iter()
            .enumerate()
            .filter_map(|(row, state)| state.checkpoint.is_some().then_some(row))
            .collect::<Vec<_>>();
        let native = if call_rows.is_empty() {
            None
        } else {
            let mut states = Vec::with_capacity(call_rows.len());
            let mut mains = Vec::with_capacity(call_rows.len());
            let mut checkpoints = Vec::with_capacity(call_rows.len());
            let mut spans = Vec::with_capacity(call_rows.len());
            for (row_index, row) in batch.rows.iter_mut().enumerate() {
                if !call_rows.contains(&row_index) {
                    continue;
                }
                let ContinuousLfm25TtsRow {
                    lease,
                    main,
                    checkpoint,
                    ..
                } = row;
                states.push(
                    &mut lease
                        .as_mut()
                        .expect("armed LFM TTS prefill state lease")
                        .require_state_mut()?
                        .state,
                );
                mains.push(main);
                checkpoints.push(
                    checkpoint
                        .as_ref()
                        .expect("armed LFM TTS prefill checkpoint"),
                );
                spans.push(scheduled[row.index].num_tokens);
            }
            Some(Self::run_blocking(|| {
                model.lfm25_audio_tts_prefill_batch(&mut states, &mut mains, &checkpoints, &spans)
            })?)
        };
        if let Some(native) = native {
            if native.steps.len() != call_rows.len() {
                return Err(Error::InferenceError(
                    "LFM TTS prefill returned wrong row count".into(),
                ));
            }
            for width in native.launch_widths {
                if let Some(call) =
                    retained_tts_batch_model_call(crate::engine::NativeBatchMode::Static, width)
                {
                    crate::engine::metrics::record_engine_model_call(call);
                }
            }
            for (row, step) in call_rows.iter().copied().zip(native.steps) {
                let index = batch.rows[row].index;
                let expected_end = scheduled[index]
                    .num_computed_tokens
                    .checked_add(scheduled[index].num_tokens)
                    .ok_or_else(|| Error::InvalidInput("LFM TTS prefill span overflowed".into()))?;
                if step.consumed_tokens != scheduled[index].num_tokens
                    || step.prefill_cursor != expected_end
                {
                    return Err(Error::InferenceError(
                        "LFM TTS prefill progress differs from admission".into(),
                    ));
                }
            }
        }
        for row in 0..batch.rows.len() {
            let index = batch.rows[row].index;
            if batch.rows[row].checkpoint.is_none() {
                continue;
            }
            let arena = requests[index]
                .managed_cache_runtime()
                .and_then(|runtime| runtime.tensor_state())
                .ok_or_else(|| {
                    Error::InferenceError("LFM TTS prefill lost ShortConv arena".into())
                })?;
            batch.rows[row]
                .lease_mut()?
                .require_state_mut()?
                .state
                .stage_shortconv(arena, scheduled[index].plan_id)?;
            if requests[index].is_cancelled() {
                batch.rollback_row(row)?;
                outputs[index] = Some(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                    requests[index].id.clone(),
                )));
            }
        }
        for row in &mut batch.rows {
            let index = row.index;
            if row.checkpoint.is_none() {
                row.lease
                    .take()
                    .expect("rolled back LFM TTS prefill lease")
                    .release()?;
                continue;
            }
            let completions = row.main.take_completed_writes();
            let checkpoint = row
                .checkpoint
                .take()
                .expect("armed LFM TTS prefill checkpoint");
            {
                let ContinuousLfm25TtsRow { lease, main, .. } = row;
                lease
                    .as_mut()
                    .expect("armed LFM TTS prefill lease")
                    .require_state_mut()?
                    .state
                    .commit_quantum(main, None, &checkpoint)?;
            }
            row.lease_mut()?.mark_clean();
            row.lease
                .take()
                .expect("committed LFM TTS prefill lease")
                .restore()?;
            outputs[index] = Some(
                ModelSessionResult::sequence(ExecutorOutput {
                    request_id: requests[index].id.clone(),
                    audio: Some(AudioOutput::new(
                        Vec::new(),
                        model.lfm25_audio_tts_output_sample_rate(),
                    )),
                    text: Some(String::new()),
                    input_transcription: None,
                    tokens_processed: scheduled[index].num_tokens,
                    tokens_generated: 0,
                    finished: false,
                    phase_timing_override: None,
                    asr_diagnostics: None,
                    error: None,
                })
                .with_managed_cache_completions(completions),
            );
        }
        batch.disarm();
        outputs
            .into_iter()
            .map(|output| {
                output.ok_or_else(|| Error::InferenceError("LFM TTS prefill lost output".into()))
            })
            .collect()
    }

    fn lfm25_audio_tts_decode_batch_with_managed(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
        mut managed: Vec<Option<super::RetainedRowManagedState>>,
    ) -> Result<Vec<ModelSessionResult>> {
        let mut outputs = (0..scheduled.len()).map(|_| None).collect::<Vec<_>>();
        let mut live_indices = Vec::new();
        let mut audio_by_index = vec![false; scheduled.len()];
        for index in 0..scheduled.len() {
            if requests[index].is_cancelled() {
                outputs[index] = Some(ModelSessionResult::cancelled_before_dispatch(
                    ExecutorOutput::cancelled(requests[index].id.clone()),
                ));
                continue;
            }
            let variant = Self::resolve_variant(requests[index])?;
            let lease = ExecutorStateLease::checkout(
                &self.lfm25_tts_decode_states,
                scheduled[index].session_key(),
                variant,
                "continuous LFM2.5 Audio TTS inspect",
            )?;
            let audio = lease
                .state()
                .ok_or_else(|| Error::InferenceError("continuous LFM TTS row has no state".into()))?
                .state
                .decode_needs_depthformer();
            lease.restore()?;
            audio_by_index[index] = audio;
            live_indices.push(index);
        }
        if live_indices.is_empty() {
            return outputs
                .into_iter()
                .map(|output| {
                    output.ok_or_else(|| {
                        Error::InferenceError("LFM TTS row produced no output".into())
                    })
                })
                .collect();
        }
        let model = requests[live_indices[0]]
            .prepared_lfm25_audio_tts_model_lease_for_executor()?
            .ok_or_else(|| Error::InferenceError("continuous LFM TTS lost model".into()))?;
        let model_arc = model.model_arc();
        let mut rows = Vec::with_capacity(live_indices.len());
        for index in live_indices.iter().copied() {
            let row_model = requests[index]
                .prepared_lfm25_audio_tts_model_lease_for_executor()?
                .ok_or_else(|| Error::InferenceError("continuous LFM TTS row lost model".into()))?;
            if !Arc::ptr_eq(&model_arc, &row_model.model_arc()) {
                return Err(Error::InferenceError(
                    "continuous LFM TTS crossed model identity".into(),
                ));
            }
            let mut retained = managed[index].take().ok_or_else(|| {
                Error::InferenceError("continuous LFM TTS lost managed state".into())
            })?;
            let tensor = retained.tensor_state.clone().ok_or_else(|| {
                Error::InferenceError("continuous LFM TTS lost ShortConv reservation".into())
            })?;
            let arena = requests[index]
                .managed_cache_runtime()
                .and_then(|runtime| runtime.tensor_state())
                .ok_or_else(|| {
                    Error::InferenceError("continuous LFM TTS lost ShortConv arena".into())
                })?;
            let main = retained.take_only_paged()?;
            retained.ensure_all_paged_consumed()?;
            let mut lease = ExecutorStateLease::checkout(
                &self.lfm25_tts_decode_states,
                scheduled[index].session_key(),
                Self::resolve_variant(requests[index])?,
                "continuous LFM2.5 Audio TTS decode",
            )?;
            let active = lease.require_state_mut()?;
            active.state.bind_tensor_sequence(tensor.sequence)?;
            active.state.restore_shortconv(arena)?;
            let prior_tokens = active.last_tokens_generated;
            let prior_stream_sequence = active.stream_sequence;
            let mut depth = audio_by_index[index]
                .then(|| super::invocation_paged_lease_for_row(requests[index], &scheduled[index]))
                .transpose()?;
            let checkpoint = active
                .state
                .reset_and_begin_quantum(&main, depth.as_mut().map(|depth| depth.cache_mut()))?;
            lease.mark_dirty();
            rows.push(ContinuousLfm25TtsRow {
                index,
                session: scheduled[index].session_key(),
                lease: Some(lease),
                main,
                depth,
                checkpoint: Some(checkpoint),
                prior_tokens,
                prior_stream_sequence,
            });
        }
        let mut batch = ContinuousLfm25TtsBatch::new(rows);
        let mut steps = (0..batch.rows.len()).map(|_| None).collect::<Vec<_>>();
        let mut launch_widths = Vec::new();
        let audio_rows = batch
            .rows
            .iter()
            .enumerate()
            .filter_map(|(row, state)| state.depth.is_some().then_some(row))
            .collect::<Vec<_>>();
        if !audio_rows.is_empty() {
            let native = (|| {
                let mut states = Vec::with_capacity(audio_rows.len());
                let mut mains = Vec::with_capacity(audio_rows.len());
                let mut depths = Vec::with_capacity(audio_rows.len());
                let mut checkpoints = Vec::with_capacity(audio_rows.len());
                for (row_index, row) in batch.rows.iter_mut().enumerate() {
                    if !audio_rows.contains(&row_index) {
                        continue;
                    }
                    let ContinuousLfm25TtsRow {
                        lease,
                        main,
                        depth,
                        checkpoint,
                        ..
                    } = row;
                    states.push(
                        &mut lease
                            .as_mut()
                            .ok_or_else(|| {
                                Error::InferenceError(
                                    "continuous LFM TTS state lease is absent".into(),
                                )
                            })?
                            .require_state_mut()?
                            .state,
                    );
                    mains.push(main);
                    depths.push(
                        depth
                            .as_mut()
                            .expect("armed LFM TTS depth lease")
                            .cache_mut(),
                    );
                    checkpoints.push(checkpoint.as_ref().expect("armed LFM TTS checkpoint"));
                }
                Self::run_blocking(|| {
                    model.lfm25_audio_tts_audio_decode_batch(
                        &mut states,
                        &mut mains,
                        &mut depths,
                        &checkpoints,
                    )
                })
            })()?;
            if native.steps.len() != audio_rows.len() {
                return Err(Error::InferenceError(
                    "LFM TTS audio cohort returned wrong width".into(),
                ));
            }
            launch_widths.extend(native.depthformer_width);
            launch_widths.extend(native.main_launch_widths);
            for (row, step) in audio_rows.iter().copied().zip(native.steps) {
                steps[row] = Some(step);
            }
        }
        let text_rows = batch
            .rows
            .iter()
            .enumerate()
            .filter_map(|(row, state)| state.depth.is_none().then_some(row))
            .collect::<Vec<_>>();
        if !text_rows.is_empty() {
            let native = (|| {
                let mut states = Vec::with_capacity(text_rows.len());
                let mut mains = Vec::with_capacity(text_rows.len());
                let mut checkpoints = Vec::with_capacity(text_rows.len());
                for (row_index, row) in batch.rows.iter_mut().enumerate() {
                    if !text_rows.contains(&row_index) {
                        continue;
                    }
                    let ContinuousLfm25TtsRow {
                        lease,
                        main,
                        depth: _,
                        checkpoint,
                        ..
                    } = row;
                    states.push(
                        &mut lease
                            .as_mut()
                            .ok_or_else(|| {
                                Error::InferenceError(
                                    "continuous LFM TTS state lease is absent".into(),
                                )
                            })?
                            .require_state_mut()?
                            .state,
                    );
                    mains.push(main);
                    checkpoints.push(checkpoint.as_ref().expect("armed LFM TTS checkpoint"));
                }
                Self::run_blocking(|| {
                    model.lfm25_audio_tts_text_decode_batch(&mut states, &mut mains, &checkpoints)
                })
            })()?;
            if native.steps.len() != text_rows.len() {
                return Err(Error::InferenceError(
                    "LFM TTS text cohort returned wrong width".into(),
                ));
            }
            launch_widths.extend(native.main_launch_widths);
            for (row, step) in text_rows.iter().copied().zip(native.steps) {
                steps[row] = Some(step);
            }
        }
        for width in launch_widths {
            if let Some(call) = continuous_tts_model_call(width, true) {
                crate::engine::metrics::record_engine_model_call(call);
            }
        }
        for row_index in 0..batch.rows.len() {
            let index = batch.rows[row_index].index;
            let arena = requests[index]
                .managed_cache_runtime()
                .and_then(|runtime| runtime.tensor_state())
                .ok_or_else(|| {
                    Error::InferenceError("continuous LFM TTS lost ShortConv arena".into())
                })?;
            batch.rows[row_index]
                .lease_mut()?
                .require_state_mut()?
                .state
                .stage_shortconv(arena, scheduled[index].plan_id)?;
        }
        for row_index in 0..batch.rows.len() {
            let index = batch.rows[row_index].index;
            if requests[index].is_cancelled() {
                batch.rollback_row(row_index)?;
                outputs[index] = Some(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                    requests[index].id.clone(),
                )));
            }
        }
        for (row_index, step) in steps.into_iter().enumerate() {
            let step = step.ok_or_else(|| {
                Error::InferenceError("LFM TTS cohort lost a decode result".into())
            })?;
            let row = &mut batch.rows[row_index];
            let index = row.index;
            if requests[index].is_cancelled() {
                row.lease
                    .take()
                    .expect("armed LFM TTS state lease")
                    .release()?;
                continue;
            }
            let completions = row.main.take_completed_writes();
            let checkpoint = row.checkpoint.take().expect("armed LFM TTS checkpoint");
            {
                let ContinuousLfm25TtsRow {
                    lease, main, depth, ..
                } = row;
                lease
                    .as_mut()
                    .expect("armed LFM TTS state lease")
                    .require_state_mut()?
                    .state
                    .commit_quantum(main, depth.as_ref().map(|depth| depth.cache()), &checkpoint)?;
            }
            if let Some(depth) = row.depth.take() {
                let _ = depth.release()?;
            }
            let active = row.lease_mut()?.require_state_mut()?;
            let generated = step
                .tokens_generated
                .saturating_sub(active.last_tokens_generated);
            active.last_tokens_generated = step.tokens_generated;
            let samples = if step.finished {
                model.detokenize_lfm25_audio_retained_tts_state(&active.state)?
            } else {
                Vec::new()
            };
            let sample_rate = model.lfm25_audio_tts_output_sample_rate();
            let text = step.text;
            row.lease_mut()?.mark_clean();
            if step.finished {
                row.lease
                    .take()
                    .expect("armed LFM TTS state lease")
                    .release()?;
            } else {
                row.lease
                    .take()
                    .expect("armed LFM TTS state lease")
                    .restore()?;
            }
            outputs[index] = Some(
                ModelSessionResult::sequence(ExecutorOutput {
                    request_id: requests[index].id.clone(),
                    audio: Some(AudioOutput::new(samples, sample_rate)),
                    text: Some(text),
                    input_transcription: None,
                    tokens_processed: 1,
                    tokens_generated: generated,
                    finished: step.finished,
                    phase_timing_override: None,
                    asr_diagnostics: None,
                    error: None,
                })
                .with_managed_cache_completions(completions),
            );
        }
        batch.disarm();
        outputs
            .into_iter()
            .map(|output| {
                output.ok_or_else(|| {
                    Error::InferenceError("continuous LFM TTS row produced no output".into())
                })
            })
            .collect()
    }

    fn vibevoice_tts_decode_batch_with_managed(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
        mut managed: Vec<Option<super::RetainedRowManagedState>>,
    ) -> Result<Vec<ModelSessionResult>> {
        let mut outputs = (0..scheduled.len()).map(|_| None).collect::<Vec<_>>();
        let mut native_indices = Vec::new();
        let mut scalar_rows = 0usize;
        let first_live = requests.iter().position(|request| !request.is_cancelled());
        let Some(first_live) = first_live else {
            return requests
                .iter()
                .map(|request| {
                    Ok(ModelSessionResult::cancelled_before_dispatch(
                        ExecutorOutput::cancelled(request.id.clone()),
                    ))
                })
                .collect();
        };
        let model = requests[first_live]
            .prepared_vibevoice_tts_model_lease_for_executor()?
            .ok_or_else(|| Error::InferenceError("VibeVoice TTS cohort lost model".into()))?;
        let model_arc = model.model_arc();
        let cohort_params = requests[first_live]
            .vibevoice_tts_generation_params_for_executor()?
            .ok_or_else(|| Error::InferenceError("VibeVoice TTS cohort lost parameters".into()))?;
        for index in 0..scheduled.len() {
            if requests[index].is_cancelled() {
                outputs[index] = Some(ModelSessionResult::cancelled_before_dispatch(
                    ExecutorOutput::cancelled(requests[index].id.clone()),
                ));
                continue;
            }
            let row_model = requests[index]
                .prepared_vibevoice_tts_model_lease_for_executor()?
                .ok_or_else(|| Error::InferenceError("VibeVoice TTS row lost model".into()))?;
            let row_params = requests[index]
                .vibevoice_tts_generation_params_for_executor()?
                .ok_or_else(|| Error::InferenceError("VibeVoice TTS row lost parameters".into()))?;
            if !Arc::ptr_eq(&model_arc, &row_model.model_arc()) || row_params != cohort_params {
                scalar_rows += 1;
                outputs[index] = Some(self.vibevoice_tts_request_with_managed_cache(
                    requests[index],
                    &scheduled[index],
                    managed[index].take(),
                )?);
                continue;
            }
            let inspect = ExecutorStateLease::checkout(
                &self.vibevoice_tts_decode_states,
                scheduled[index].session_key(),
                Self::resolve_variant(requests[index])?,
                "VibeVoice TTS cohort inspect",
            )?;
            let eligible = inspect
                .state()
                .map(|active| model.retained_decode_batch_eligible(&active.state))
                .transpose()?
                .unwrap_or(false);
            inspect.restore()?;
            if eligible {
                native_indices.push(index);
            } else {
                scalar_rows += 1;
                outputs[index] = Some(self.vibevoice_tts_request_with_managed_cache(
                    requests[index],
                    &scheduled[index],
                    managed[index].take(),
                )?);
            }
        }
        if native_indices.len() == 1 {
            let index = native_indices.pop().unwrap();
            scalar_rows += 1;
            outputs[index] = Some(self.vibevoice_tts_request_with_managed_cache(
                requests[index],
                &scheduled[index],
                managed[index].take(),
            )?);
        } else if !native_indices.is_empty() {
            let mut rows = Vec::with_capacity(native_indices.len());
            let mut quanta = Vec::with_capacity(native_indices.len());
            for index in native_indices.iter().copied() {
                let mut retained = managed[index].take().ok_or_else(|| {
                    Error::InferenceError("VibeVoice TTS cohort lost retained state".into())
                })?;
                let positive = retained
                    .take_paged_domain(crate::kv::CacheDomainId::new(1), true)?
                    .expect("required positive cache");
                let negative = retained
                    .take_paged_domain(crate::kv::CacheDomainId::new(2), true)?
                    .expect("required negative cache");
                retained.ensure_all_paged_consumed()?;
                let _tensor = retained.tensor_state.clone().ok_or_else(|| {
                    Error::InferenceError("VibeVoice TTS cohort lost tokenizer reservation".into())
                })?;
                let arena = requests[index]
                    .managed_cache_runtime()
                    .and_then(|runtime| runtime.tensor_state())
                    .cloned()
                    .ok_or_else(|| {
                        Error::InferenceError("VibeVoice TTS cohort lost tokenizer arena".into())
                    })?;
                quanta.push(
                    crate::models::architectures::vibevoice::tts::VibeVoiceTtsTokenizerQuantum {
                        arena,
                        transaction: crate::backends::state::PhysicalStateTransactionId::new(
                            scheduled[index].plan_id,
                        )?,
                    },
                );
                let mut lease = ExecutorStateLease::checkout(
                    &self.vibevoice_tts_decode_states,
                    scheduled[index].session_key(),
                    Self::resolve_variant(requests[index])?,
                    "VibeVoice TTS cohort decode",
                )?;
                let active = lease.require_state_mut()?;
                let prior_frames = active.last_frames_generated;
                let prior_stream_sequence = active.stream_sequence;
                let checkpoint = active.state.begin_managed_quantum(positive, negative)?;
                lease.mark_dirty();
                rows.push(ContinuousVibeVoiceTtsRow {
                    index,
                    session: scheduled[index].session_key(),
                    lease: Some(lease),
                    checkpoint: Some(checkpoint),
                    prior_frames,
                    prior_stream_sequence,
                });
            }
            let mut batch = ContinuousVibeVoiceTtsBatch { rows, armed: true };
            let steps = {
                let mut states = batch
                    .rows
                    .iter_mut()
                    .map(|row| {
                        &mut row
                            .lease
                            .as_mut()
                            .expect("armed VibeVoice lease")
                            .require_state_mut()
                            .expect("checked VibeVoice state")
                            .state
                    })
                    .collect::<Vec<_>>();
                Self::run_blocking(|| model.retained_decode_step_batch(&mut states, &quanta))?
            };
            crate::engine::metrics::record_engine_model_call(
                crate::engine::metrics::EngineModelCall::NativeTensor {
                    mode: crate::engine::NativeBatchMode::Continuous,
                    rows: batch.rows.len(),
                },
            );
            for (row_index, step) in steps.into_iter().enumerate() {
                let row = &mut batch.rows[row_index];
                let index = row.index;
                if requests[index].is_cancelled() {
                    row.rollback()?;
                    row.lease
                        .take()
                        .expect("rolled back VibeVoice lease")
                        .release()?;
                    outputs[index] = Some(ModelSessionResult::cancelled(
                        ExecutorOutput::cancelled(requests[index].id.clone()),
                    ));
                    continue;
                }
                let checkpoint = row.checkpoint.as_mut().expect("armed VibeVoice checkpoint");
                let active = row
                    .lease
                    .as_mut()
                    .expect("armed VibeVoice lease")
                    .require_state_mut()?;
                let completions = active.state.take_managed_write_completions();
                active.state.commit_managed_quantum(checkpoint)?;
                row.checkpoint = None;
                let generated = step
                    .frames_generated
                    .saturating_sub(active.last_frames_generated);
                active.last_frames_generated = step.frames_generated;
                let _ = active.state.take_staged_step();
                let (samples, sample_rate) = if step.finished {
                    let output = model.finalize_retained_state(&active.state)?;
                    (output.samples, output.sample_rate)
                } else {
                    (Vec::new(), 24_000)
                };
                row.lease
                    .as_mut()
                    .expect("armed VibeVoice lease")
                    .mark_clean();
                if step.finished {
                    row.lease.take().expect("armed VibeVoice lease").release()?;
                } else {
                    row.lease.take().expect("armed VibeVoice lease").restore()?;
                }
                outputs[index] = Some(
                    ModelSessionResult::sequence(ExecutorOutput {
                        request_id: requests[index].id.clone(),
                        audio: Some(AudioOutput::new(samples, sample_rate)),
                        text: None,
                        input_transcription: None,
                        tokens_processed: 1,
                        tokens_generated: generated,
                        finished: step.finished,
                        phase_timing_override: None,
                        asr_diagnostics: None,
                        error: None,
                    })
                    .with_managed_cache_completions(completions),
                );
            }
            batch.armed = false;
        }
        if scalar_rows > 0 {
            crate::engine::metrics::record_engine_model_call(
                crate::engine::metrics::EngineModelCall::ScalarRows {
                    envelope: crate::engine::NativeBatchMode::Continuous,
                    rows: scalar_rows,
                },
            );
        }
        outputs
            .into_iter()
            .map(|output| {
                output.ok_or_else(|| {
                    Error::InferenceError("VibeVoice TTS row produced no output".into())
                })
            })
            .collect()
    }

    pub(super) fn tts_decode_batch_with_managed(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
        managed_caches: Vec<Option<super::RetainedRowManagedState>>,
    ) -> Result<Vec<ModelSessionResult>> {
        validate_continuous_tts_batch_shape(scheduled)?;
        if managed_caches.len() != scheduled.len() {
            return Err(Error::InvalidInput(
                "continuous TTS managed-cache rows do not match batch width".into(),
            ));
        }
        let ordered_requests = scheduled
            .iter()
            .map(|scheduled| {
                requests
                    .iter()
                    .copied()
                    .find(|request| request.id == scheduled.request_id)
                    .ok_or_else(|| {
                        Error::InferenceError(format!(
                            "continuous TTS request {} is missing its snapshot",
                            scheduled.request_id
                        ))
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        if ordered_requests.first().is_some_and(|request| {
            request
                .model_variant
                .is_some_and(|variant| variant.family() == ModelFamily::Lfm25Audio)
        }) {
            return self.lfm25_audio_tts_decode_batch_with_managed(
                &ordered_requests,
                scheduled,
                managed_caches,
            );
        }
        if ordered_requests.first().is_some_and(|request| {
            request
                .model_variant
                .is_some_and(|variant| variant.family() == ModelFamily::VoxtralTts)
        }) {
            return self.voxtral_tts_batch_with_managed(
                &ordered_requests,
                scheduled,
                managed_caches,
            );
        }
        if ordered_requests.first().is_some_and(|request| {
            request
                .model_variant
                .is_some_and(|variant| variant.family() == ModelFamily::VibeVoiceTts)
        }) {
            return self.vibevoice_tts_decode_batch_with_managed(
                &ordered_requests,
                scheduled,
                managed_caches,
            );
        }
        let live_indices = ordered_requests
            .iter()
            .enumerate()
            .filter_map(|(index, request)| (!request.is_cancelled()).then_some(index))
            .collect::<Vec<_>>();
        let mut outputs = (0..scheduled.len())
            .map(|_| None)
            .collect::<Vec<Option<ModelSessionResult>>>();
        for (index, request) in ordered_requests.iter().enumerate() {
            if request.is_cancelled() {
                outputs[index] = Some(ModelSessionResult::cancelled_before_dispatch(
                    ExecutorOutput::cancelled(request.id.clone()),
                ));
            }
        }
        if live_indices.is_empty() {
            return outputs
                .into_iter()
                .map(|output| {
                    output.ok_or_else(|| {
                        Error::InferenceError("cancelled TTS row produced no result".into())
                    })
                })
                .collect();
        }

        let model = ordered_requests[live_indices[0]]
            .prepared_qwen_tts_model_for_executor()?
            .ok_or_else(|| {
                Error::InferenceError(
                    "continuous TTS request has no exact loaded model identity".into(),
                )
            })?;
        if !model.supports_continuous_decode_batch() {
            return Err(Error::InvalidInput(
                "loaded TTS model has no continuous tensor decode adapter".into(),
            ));
        }
        for index in live_indices.iter().copied().skip(1) {
            let row_model = ordered_requests[index]
                .prepared_qwen_tts_model_for_executor()?
                .ok_or_else(|| {
                    Error::InferenceError(
                        "continuous TTS row has no exact loaded model identity".into(),
                    )
                })?;
            if !Arc::ptr_eq(&model, &row_model) {
                return Err(Error::InferenceError(
                    "continuous TTS batch spans different loaded model instances".into(),
                ));
            }
        }

        let mut checked_out = Vec::with_capacity(live_indices.len());
        for index in live_indices.iter().copied() {
            let request = ordered_requests[index];
            let session = scheduled[index].session_key();
            let marker_variant = request.model_variant.ok_or_else(|| {
                Error::InvalidInput("continuous Qwen TTS row is missing its model variant".into())
            })?;
            let lease = ExecutorStateLease::checkout(
                &self.qwen_tts_decode_states,
                session.clone(),
                marker_variant,
                "continuous Qwen TTS decode",
            )?;
            let state = lease.state().ok_or_else(|| {
                Error::InferenceError(format!(
                    "continuous TTS session {}:{} has no active decode state",
                    session.request_id, session.epoch
                ))
            })?;
            if state.variant != request.model_variant || !Arc::ptr_eq(&state.model, &model) {
                return Err(Error::InferenceError(
                    "continuous TTS state identity does not match its request".into(),
                ));
            }
            if !matches!(state.state, QwenTtsPhysicalState::Decode(_)) {
                return Err(Error::InferenceError(
                    "continuous TTS batch contains an unfinished prefill".into(),
                ));
            }
            checked_out.push((index, session, lease));
        }

        let mut active = ContinuousTtsStateBatch::new(checked_out);
        let mut managed_caches = managed_caches;
        for (index, _, lease, checkpoint) in &mut active.rows {
            let request = ordered_requests[*index];
            let mut views = managed_caches[*index].take().ok_or_else(|| {
                Error::InferenceError(
                    "continuous Qwen3-TTS decode requires retained physical state".into(),
                )
            })?;
            let tensor_reservation = views.tensor_state.clone();
            let tensor_arena = request
                .managed_cache_runtime()
                .and_then(|runtime| runtime.tensor_state());
            if tensor_arena.is_some() != tensor_reservation.is_some() {
                return Err(Error::InferenceError(
                    "continuous Qwen3-TTS row lost its tensor reservation".into(),
                ));
            }
            let cache = views.take_only_paged()?;
            let state = lease.require_state_mut()?;
            let outer = ActiveTtsOuterCheckpoint::capture(state);
            let QwenTtsPhysicalState::Decode(decode) = &mut state.state else {
                unreachable!("validated continuous Qwen3-TTS decode state")
            };
            let native = decode.begin_managed_quantum(cache)?;
            *checkpoint = Some((TtsManagedCheckpoint::Decode(native), outer));
            if let (Some(arena), Some(reservation)) = (tensor_arena, tensor_reservation) {
                decode.bind_tensor_sequence(reservation.sequence)?;
                decode.restore_tensor_state(arena)?;
            }
            lease.mark_dirty();
        }

        let before_cursors = active
            .rows
            .iter_mut()
            .map(|(_, _, lease, _)| {
                let state = lease.require_state_mut()?;
                let QwenTtsPhysicalState::Decode(state) = &state.state else {
                    unreachable!("validated continuous Qwen3-TTS decode state")
                };
                Ok(state.talker_context_len())
            })
            .collect::<Result<Vec<_>>>()?;
        let mut predictor_leases = active
            .rows
            .iter()
            .map(|(index, _, _, _)| {
                super::invocation_paged_lease_for_row(ordered_requests[*index], &scheduled[*index])
            })
            .collect::<Result<Vec<_>>>()?;
        let steps = {
            let mut state_refs = active
                .rows
                .iter_mut()
                .map(|(_, _, lease, _)| {
                    let active = lease.require_state_mut()?;
                    let QwenTtsPhysicalState::Decode(state) = &mut active.state else {
                        unreachable!("validated continuous Qwen3-TTS decode state")
                    };
                    Ok(state)
                })
                .collect::<Result<Vec<_>>>()?;
            let mut predictor_refs = predictor_leases
                .iter_mut()
                .map(|lease| lease.cache_mut())
                .collect::<Vec<_>>();
            Self::run_blocking(|| {
                model.tts_decode_step_batch_physical(&mut state_refs, &mut predictor_refs)
            })?
        };
        for predictor in predictor_leases {
            let _ = predictor.release()?;
        }
        if steps.len() != active.rows.len() {
            return Err(Error::InferenceError(
                "continuous TTS model returned the wrong number of rows".into(),
            ));
        }
        let live_kernel_rows = steps.iter().filter(|step| step.executed_model_row).count();
        if let Some(call) = continuous_tts_model_call(
            live_kernel_rows,
            model.continuous_decode_is_tensor_batched(),
        ) {
            crate::engine::metrics::record_engine_model_call(call);
        }

        let mut continuing = vec![false; scheduled.len()];
        for (row, step) in steps.into_iter().enumerate() {
            let index = active.rows[row].0;
            let request = ordered_requests[index];
            if request.is_cancelled() {
                let _ = request.take_staged_stream_outputs()?;
                let index = active.rollback_row(row)?;
                outputs[index] = Some(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                    request.id.clone(),
                )));
                continue;
            }
            let row_result = (|| -> Result<ModelSessionResult> {
                let active_state = active.rows[row].2.require_state_mut()?;
                let QwenTtsPhysicalState::Decode(state) = &mut active_state.state else {
                    unreachable!("validated continuous Qwen3-TTS decode state")
                };
                if let Some(arena) = request
                    .managed_cache_runtime()
                    .and_then(|runtime| runtime.tensor_state())
                {
                    state.stage_tensor_state(arena, scheduled[index].plan_id)?;
                }
                let accepted =
                    accepted_tts_talker_tokens(before_cursors[row], state.talker_context_len(), 1)?;
                active_state.sampling_ms += step.sampling_ms;
                active_state.decode_ms += step.decode_ms;
                active_state.codec_ms += step.codec_ms;
                active_state.decode_steps = active_state.decode_steps.saturating_add(1);
                active_state.last_frames_generated = step.frames_generated;
                if !step.samples.is_empty() {
                    if active_state.first_output_ms_since_start.is_none() {
                        active_state.first_output_ms_since_start =
                            Some(active_state.execution_started.elapsed().as_secs_f64() * 1000.0);
                    }
                    active_state
                        .audio_samples_accum
                        .extend_from_slice(&step.samples);
                    if let Some(tx) = Self::stream_sender(request) {
                        Self::stream_audio_with_policy(
                            &tx,
                            request.stream_policy,
                            &request.id,
                            &mut active_state.stream_sequence,
                            step.samples.clone(),
                            24_000,
                            false,
                        )?;
                    }
                }
                if step.finished {
                    if let Some(tx) = Self::stream_sender(request) {
                        Self::stream_final_marker_with_policy(
                            &tx,
                            request.stream_policy,
                            &request.id,
                            &mut active_state.stream_sequence,
                        )?;
                    }
                }
                if request.is_cancelled() {
                    return Err(Error::Cancelled(request.id.clone()));
                }
                let completions = state.take_managed_write_completions();
                let audio = if step.finished {
                    active_state.audio_samples_accum.clone()
                } else {
                    Vec::new()
                };
                Ok(ModelSessionResult::sequence(ExecutorOutput {
                    request_id: request.id.clone(),
                    audio: Some(AudioOutput::new(audio, 24_000)),
                    text: None,
                    input_transcription: None,
                    tokens_processed: accepted,
                    tokens_generated: accepted,
                    finished: step.finished,
                    phase_timing_override: None,
                    asr_diagnostics: None,
                    error: None,
                })
                .with_managed_cache_completions(completions))
            })();
            match row_result {
                Ok(result) => {
                    continuing[index] = !step.finished;
                    outputs[index] = Some(result);
                }
                Err(error) => {
                    let _ = request.take_staged_stream_outputs();
                    let index = active.rollback_row(row)?;
                    outputs[index] = Some(if matches!(error, Error::Cancelled(_)) {
                        ModelSessionResult::cancelled(ExecutorOutput::cancelled(request.id.clone()))
                    } else {
                        ModelSessionResult::sequence(ExecutorOutput::error(
                            request.id.clone(),
                            format!("continuous TTS row failed: {error}"),
                        ))
                    });
                }
            }
        }

        let cancelled = active
            .rows
            .iter()
            .map(|(index, _, _, _)| ordered_requests[*index].is_cancelled())
            .collect::<Vec<_>>();
        let checkpoint_armed = active
            .rows
            .iter()
            .map(|(_, _, _, checkpoint)| checkpoint.is_some())
            .collect::<Vec<_>>();
        for row in late_cancelled_tts_rows(&cancelled, &checkpoint_armed) {
            let index = active.rows[row].0;
            let request = ordered_requests[index];
            let _ = request.take_staged_stream_outputs()?;
            let index = active.rollback_row(row)?;
            continuing[index] = false;
            outputs[index] = Some(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                request.id.clone(),
            )));
        }

        for (index, _, lease) in active.commit() {
            let transition = if continuing[index] {
                lease.restore()
            } else {
                lease.release()
            };
            if let Err(error) = transition {
                let _ = ordered_requests[index].take_staged_stream_outputs();
                outputs[index] = Some(ModelSessionResult::sequence(ExecutorOutput::error(
                    ordered_requests[index].id.clone(),
                    format!("continuous TTS state transition failed: {error}"),
                )));
            }
        }
        outputs
            .into_iter()
            .map(|output| {
                output.ok_or_else(|| {
                    Error::InferenceError("continuous TTS row produced no result".into())
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;

    fn scheduled(is_prefill: bool, num_tokens: usize) -> ScheduledRequest {
        ScheduledRequest {
            plan_id: 1,
            request_id: "tts-transition".into(),
            sequence_id: 1,
            num_tokens,
            is_prefill,
            num_computed_tokens: usize::from(!is_prefill),
            work: crate::engine::WorkUnit::SequenceStep {
                phase: if is_prefill {
                    crate::engine::SequencePhase::Prefill
                } else {
                    crate::engine::SequencePhase::Decode
                },
                input: crate::engine::InputRange {
                    start: usize::from(!is_prefill),
                    end: usize::from(!is_prefill) + num_tokens,
                },
                max_output_steps: num_tokens.max(1),
                auxiliary_state: None,
            },
        }
    }

    #[test]
    fn qwen_tts_physical_binding_rejects_legacy_and_partial_ownership() {
        assert!(validate_qwen_tts_physical_bindings(true, true, true, true).is_ok());
        assert!(validate_qwen_tts_physical_bindings(false, false, false, false).is_err());
        assert!(validate_qwen_tts_physical_bindings(true, false, true, true).is_err());
        assert!(validate_qwen_tts_physical_bindings(true, true, true, false).is_err());
    }

    #[test]
    fn qwen_tts_prefill_only_hydrates_and_decode_consumes_the_row_budget() {
        assert_eq!(qwen_tts_decode_iterations(&scheduled(true, 17)), 0);
        assert_eq!(qwen_tts_decode_iterations(&scheduled(false, 0)), 1);
        assert_eq!(qwen_tts_decode_iterations(&scheduled(false, 4)), 4);
    }

    #[test]
    fn continuous_tts_final_sweep_selects_only_newly_cancelled_armed_rows() {
        assert_eq!(
            late_cancelled_tts_rows(
                &[true, false, true, true, false],
                &[true, true, false, true, false]
            ),
            vec![0, 3]
        );
    }

    #[test]
    fn kokoro_static_batch_excludes_cancelled_rows_before_model_entry() {
        assert_eq!(
            kokoro_live_row_indices(&[false, true, false, true]),
            vec![0, 2]
        );
        assert!(kokoro_live_row_indices(&[true, true]).is_empty());
    }

    #[test]
    fn kokoro_static_batch_rechecks_mixed_cancellation_after_model_entry() {
        let entered = kokoro_live_row_indices(&[false, false, false]);
        let publishable = kokoro_live_row_indices(&[false, true, false]);
        assert_eq!(entered, vec![0, 1, 2]);
        assert_eq!(publishable, vec![0, 2]);
    }

    #[test]
    fn kokoro_static_batch_preserves_live_result_order() {
        let scattered = scatter_kokoro_rows(5, &[0, 2, 4], vec!["a", "c", "e"]).unwrap();
        assert_eq!(scattered, vec![Some("a"), None, Some("c"), None, Some("e")]);
        assert!(scatter_kokoro_rows(2, &[1], vec!["x", "y"]).is_err());
        assert!(scatter_kokoro_rows(1, &[1], vec!["x"]).is_err());
    }

    #[test]
    fn kokoro_static_batch_rejects_mixed_exact_model_identity() {
        let first = Arc::new(());
        let same = first.clone();
        let other = Arc::new(());
        let mut expected = None;
        validate_exact_kokoro_model(&mut expected, first).unwrap();
        validate_exact_kokoro_model(&mut expected, same).unwrap();
        assert!(validate_exact_kokoro_model(&mut expected, other).is_err());
    }

    #[test]
    fn kokoro_static_batch_reports_b1_fallback_and_native_batched_rows_truthfully() {
        assert!(matches!(
            retained_tts_batch_model_call(crate::engine::NativeBatchMode::Static, 1),
            Some(crate::engine::metrics::EngineModelCall::ScalarRows {
                envelope: crate::engine::NativeBatchMode::Static,
                rows: 1,
            })
        ));
        assert!(matches!(
            retained_tts_batch_model_call(crate::engine::NativeBatchMode::Static, 2),
            Some(crate::engine::metrics::EngineModelCall::NativeTensor {
                mode: crate::engine::NativeBatchMode::Static,
                rows: 2,
            })
        ));
    }

    #[test]
    fn terminal_tts_rows_report_zero_accepted_talker_tokens() {
        assert_eq!(accepted_tts_talker_tokens(17, 17, 1).unwrap(), 0);
        assert_eq!(accepted_tts_talker_tokens(17, 18, 1).unwrap(), 1);
        assert!(accepted_tts_talker_tokens(18, 17, 1).is_err());
        assert!(accepted_tts_talker_tokens(17, 19, 1).is_err());
    }

    #[test]
    fn continuous_tts_telemetry_uses_live_kernel_width() {
        assert!(continuous_tts_model_call(0, true).is_none());
        assert!(matches!(
            continuous_tts_model_call(1, true),
            Some(crate::engine::metrics::EngineModelCall::ScalarRows { rows: 1, .. })
        ));
        assert!(matches!(
            continuous_tts_model_call(2, true),
            Some(crate::engine::metrics::EngineModelCall::NativeTensor { rows: 2, .. })
        ));
        assert!(matches!(
            retained_tts_batch_model_call(crate::engine::NativeBatchMode::Static, 1),
            Some(crate::engine::metrics::EngineModelCall::ScalarRows {
                envelope: crate::engine::NativeBatchMode::Static,
                rows: 1
            })
        ));
        assert!(matches!(
            retained_tts_batch_model_call(crate::engine::NativeBatchMode::Static, 3),
            Some(crate::engine::metrics::EngineModelCall::NativeTensor {
                mode: crate::engine::NativeBatchMode::Static,
                rows: 3
            })
        ));
    }

    #[test]
    fn voxtral_prefill_telemetry_preserves_unequal_wave_widths() {
        let calls = [3, 2, 1]
            .into_iter()
            .map(|width| {
                retained_tts_batch_model_call(crate::engine::NativeBatchMode::Static, width)
                    .expect("positive launch width")
            })
            .collect::<Vec<_>>();
        assert!(matches!(
            calls[0],
            crate::engine::metrics::EngineModelCall::NativeTensor { rows: 3, .. }
        ));
        assert!(matches!(
            calls[1],
            crate::engine::metrics::EngineModelCall::NativeTensor { rows: 2, .. }
        ));
        assert!(matches!(
            calls[2],
            crate::engine::metrics::EngineModelCall::ScalarRows { rows: 1, .. }
        ));
    }

    #[test]
    fn voxtral_prefill_step_rejects_consumption_or_cursor_drift() {
        let scheduled = scheduled(true, 3);
        let valid = crate::models::architectures::voxtral::tts::retained::VoxtralTtsPrefillStep {
            consumed_tokens: 3,
            prefill_cursor: 3,
            prompt_tokens: 8,
            complete: false,
        };
        validate_voxtral_tts_prefill_step(0, &scheduled, &valid).unwrap();
        let mut drifted = valid.clone();
        drifted.prefill_cursor = 2;
        assert!(validate_voxtral_tts_prefill_step(0, &scheduled, &drifted).is_err());
        drifted.prefill_cursor = 3;
        drifted.consumed_tokens = 2;
        assert!(validate_voxtral_tts_prefill_step(0, &scheduled, &drifted).is_err());
    }

    #[test]
    fn qwen_reference_audio_uses_the_thirty_second_decode_contract() {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 1_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut wav = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut wav);
            let mut writer = hound::WavWriter::new(cursor, spec).expect("WAV writer");
            for _ in 0..30_001 {
                writer.write_sample(0_i16).expect("WAV sample");
            }
            writer.finalize().expect("WAV finalize");
        }

        let mut request = EngineCoreRequest::tts("clone this voice");
        request.reference_audio = Some(base64::engine::general_purpose::STANDARD.encode(wav));
        request.reference_text = Some("reference transcript".to_string());

        let error = NativeExecutor::reference_from_request(&request)
            .expect_err("reference audio above thirty seconds must fail before model encoding");
        assert!(
            error.to_string().contains("production limit"),
            "unexpected reference bound error: {error}"
        );
    }
}
