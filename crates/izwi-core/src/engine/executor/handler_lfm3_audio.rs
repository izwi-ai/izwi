//! LFM3-Audio requests: text to speech and spoken replies, each one atomic
//! invocation on the language model's leased pages and ShortConv ring.
//! Speech streams as the codec decodes it.

use std::cell::{Cell, RefCell};

use crate::error::{Error, Result};
use crate::models::architectures::lfm25_audio::physical::{
    LFM25_MAIN_ATTENTION_STATE_DOMAIN, LFM25_MAIN_SHORTCONV_STATE_DOMAIN,
};
use crate::models::architectures::lfm3_audio::config::Lfm3TextSampling;
use crate::models::architectures::lfm3_audio::model::{
    InvocationLmState, Lfm3AudioModel, Lfm3AudioOutput,
};
use crate::models::registry::NativeAudioChatModel;
use crate::models::shared::chat::ChatMessage;

use super::super::request::EngineCoreRequest;
use super::super::scheduler::ScheduledRequest;
use super::super::types::AudioOutput;
use super::audio::decode_request_audio_with_rate;
use super::{ExecutorOutput, ModelSessionResult, NativeExecutor};
use crate::engine::InvocationTensorLease;
use crate::kv::v2::InvocationWorkspaceLeaseSetV2;

const OUTPUT_SAMPLE_RATE: u32 = 24_000;

/// Run `generate` on the request's leased language-model state.
fn with_lm_state<T>(
    leases: &mut InvocationWorkspaceLeaseSetV2,
    generate: impl FnOnce(&mut InvocationLmState<'_>) -> Result<T>,
) -> Result<T> {
    let (attention, shortconv) = leases.lease_pair_mut(
        LFM25_MAIN_ATTENTION_STATE_DOMAIN,
        LFM25_MAIN_SHORTCONV_STATE_DOMAIN,
    )?;
    let mut state = InvocationLmState {
        cache: attention.paged_cache_mut()?,
        shortconv: shortconv.typed_mut::<InvocationTensorLease>()?,
    };
    generate(&mut state)
}

impl NativeExecutor {
    fn lfm3_audio_model(
        &self,
        request: &EngineCoreRequest,
    ) -> Result<std::sync::Arc<NativeAudioChatModel>> {
        let variant = Self::resolve_variant(request)?;
        let model = self.with_registry(|registry| {
            registry.try_get_audio_chat(variant).ok_or_else(|| {
                Error::ModelNotFound(format!("Audio-chat model {variant} is not loaded"))
            })
        })?;
        match model.as_ref() {
            NativeAudioChatModel::Lfm3Audio(_) => Ok(model),
            _ => Err(Error::InvalidInput(format!(
                "{variant} is not an LFM3-Audio model"
            ))),
        }
    }

    fn lfm3_audio_of(model: &NativeAudioChatModel) -> &Lfm3AudioModel {
        match model {
            NativeAudioChatModel::Lfm3Audio(model) => model,
            _ => unreachable!("checked by lfm3_audio_model"),
        }
    }

    fn lfm3_audio_cancelled(request: &EngineCoreRequest) -> ModelSessionResult {
        let _ = request.take_staged_stream_outputs();
        ModelSessionResult::cancelled(ExecutorOutput::cancelled(request.id.clone()))
    }

    /// Answer a spoken turn with interleaved text and speech, after
    /// transcribing it for the conversation's record.
    pub(super) fn lfm3_audio_chat_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        history: &[ChatMessage],
        system_prompt: Option<&str>,
    ) -> Result<ModelSessionResult> {
        let stream_tx = Self::stream_sender(request);
        let stream_policy = request.stream_policy;
        let max_slots = request.params.max_tokens.max(1);
        // Liquid's `generation_config.json` decodes the reply's text
        // greedily; no voice route sets text sampling of its own.
        let text_sampling = Lfm3TextSampling::default();
        let sampling = request.lfm3_audio_sampling();
        let seed = request
            .params
            .speech
            .seed
            .unwrap_or_else(|| Self::chat_request_seed(&request.id));
        let (samples, sample_rate) =
            Self::run_blocking(|| decode_request_audio_with_rate(request))?;
        let model = self.lfm3_audio_model(request)?;
        let model = Self::lfm3_audio_of(&model);
        let mut leases =
            super::invocation_workspace_leases_for_atomic_scalar_row(request, scheduled)?;

        // The turn is encoded once, for its transcript and for the reply.
        let audio = Self::run_blocking(|| model.encode_audio(&samples, sample_rate))?;
        let transcription = Self::run_blocking(|| {
            with_lm_state(&mut leases, |state| {
                model.transcribe_turn(
                    &samples,
                    sample_rate,
                    &audio,
                    state,
                    &mut || request.is_cancelled(),
                )
            })
        });
        let input_transcription = match leases.fence_on_error(transcription) {
            Err(Error::Cancelled(_)) => return Ok(Self::lfm3_audio_cancelled(request)),
            transcription => transcription?,
        };
        if request.is_cancelled() {
            return Ok(Self::lfm3_audio_cancelled(request));
        }

        let sequence = Cell::new(0usize);
        let stream_error = RefCell::new(None::<Error>);
        let mut on_text = |delta: &str| {
            let Some(tx) = stream_tx.as_ref() else {
                return;
            };
            if delta.is_empty() || stream_error.borrow().is_some() {
                return;
            }
            let mut next = sequence.get();
            match Self::stream_text_with_policy(
                tx,
                stream_policy,
                &request.id,
                &mut next,
                delta.to_string(),
            ) {
                Ok(()) => sequence.set(next),
                Err(error) => *stream_error.borrow_mut() = Some(error),
            }
        };
        let mut on_audio = |samples: &[f32]| {
            let Some(tx) = stream_tx.as_ref() else {
                return;
            };
            if samples.is_empty() || stream_error.borrow().is_some() {
                return;
            }
            let mut next = sequence.get();
            match Self::stream_audio_with_policy(
                tx,
                stream_policy,
                &request.id,
                &mut next,
                samples.to_vec(),
                OUTPUT_SAMPLE_RATE,
                false,
            ) {
                Ok(()) => sequence.set(next),
                Err(error) => *stream_error.borrow_mut() = Some(error),
            }
        };
        let mut should_stop = || request.is_cancelled() || stream_error.borrow().is_some();
        let output: Result<Lfm3AudioOutput> = Self::run_blocking(|| {
            with_lm_state(&mut leases, |state| {
                model.respond(
                    history,
                    audio,
                    system_prompt,
                    max_slots,
                    text_sampling,
                    sampling,
                    seed,
                    state,
                    &mut should_stop,
                    &mut on_text,
                    &mut on_audio,
                )
            })
        });
        let output = leases.fence_on_error(output)?;
        if let Some(error) = stream_error.into_inner() {
            return Err(error);
        }
        if output.stop_reason == "cancelled" {
            return Ok(Self::lfm3_audio_cancelled(request));
        }
        leases.release()?;
        if let Some(tx) = stream_tx.as_ref() {
            let mut next = sequence.get();
            Self::stream_final_marker_with_policy(tx, stream_policy, &request.id, &mut next)?;
        }
        Ok(ModelSessionResult::atomic(ExecutorOutput {
            request_id: request.id.clone(),
            audio: Some(AudioOutput::new(output.samples, OUTPUT_SAMPLE_RATE)),
            text: Some(output.text.trim().to_string()),
            input_transcription: Some(input_transcription),
            tokens_processed: output.prompt_tokens,
            tokens_generated: (output.text_tokens + output.audio_frames).max(1),
            finished: true,
            phase_timing_override: None,
            asr_diagnostics: None,
            error: None,
            logprobs: Vec::new(),
        }))
    }
}
