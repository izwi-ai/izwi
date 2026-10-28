//! LFM3-Audio requests: text to speech and spoken replies, each one atomic
//! invocation on the language model's leased pages and ShortConv ring.
//! Speech streams as the codec decodes it.

use std::cell::{Cell, RefCell};

use crate::error::{Error, Result};
use crate::models::architectures::lfm25_audio::physical::{
    LFM25_MAIN_ATTENTION_STATE_DOMAIN, LFM25_MAIN_SHORTCONV_STATE_DOMAIN,
};
use crate::models::architectures::lfm3_audio::config::{Lfm3AudioSampling, Lfm3TextSampling};
use crate::models::architectures::lfm3_audio::model::{
    lfm3_audio_tts_auto_max_frames, InvocationLmState, Lfm3AudioModel, Lfm3AudioOutput,
};
use crate::models::architectures::lfm3_audio::prompts::Lfm3AudioVoice;
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

/// The request's speech sampling: the model card's (temperature 1, top-k
/// 128, a 1.1 repetition penalty over 4,096 frames) unless it sets its own.
fn audio_sampling(request: &EngineCoreRequest) -> Lfm3AudioSampling {
    let defaults = Lfm3AudioSampling::default();
    let speech = &request.params.speech;
    Lfm3AudioSampling {
        temperature: request
            .params
            .audio_temperature
            .or(speech.temperature)
            .unwrap_or(defaults.temperature)
            .max(0.0),
        top_k: request
            .params
            .audio_top_k
            .or(speech.top_k)
            .unwrap_or(defaults.top_k),
        repetition_penalty: speech
            .repetition_penalty
            .unwrap_or(defaults.repetition_penalty),
        repetition_window: defaults.repetition_window,
    }
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

    /// Speak the request's text in its voice (`female` or `male`) and
    /// language (one of 19 locales).
    pub(super) fn lfm3_audio_tts_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
    ) -> Result<ModelSessionResult> {
        let text = request
            .text
            .as_deref()
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| Error::InvalidInput("LFM3-Audio TTS request is missing text".into()))?;
        let voice = Lfm3AudioVoice::resolve(
            request
                .params
                .voice
                .as_deref()
                .or(request.params.speaker.as_deref()),
            request.language.as_deref(),
        )?;
        let max_frames = match request.params.max_tokens {
            0 => lfm3_audio_tts_auto_max_frames(text),
            frames => frames,
        };
        let sampling = audio_sampling(request);
        let seed = request
            .params
            .speech
            .seed
            .unwrap_or_else(|| Self::chat_request_seed(&request.id));
        let stream_tx = Self::stream_sender(request);
        let stream_policy = request.stream_policy;
        let model = self.lfm3_audio_model(request)?;
        let model = Self::lfm3_audio_of(&model);
        let mut leases =
            super::invocation_workspace_leases_for_atomic_scalar_row(request, scheduled)?;
        if request.is_cancelled() {
            return Ok(Self::lfm3_audio_cancelled(request));
        }

        let sequence = Cell::new(0usize);
        let stream_error = RefCell::new(None::<Error>);
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
        let output = Self::run_blocking(|| {
            with_lm_state(&mut leases, |state| {
                model.synthesize(
                    text.trim(),
                    voice,
                    max_frames,
                    sampling,
                    seed,
                    state,
                    &mut should_stop,
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
        if output.samples.is_empty() {
            return Err(Error::InferenceError(format!(
                "LFM3-Audio spoke no audio for request {}",
                request.id
            )));
        }
        let audio = match stream_tx.as_ref() {
            Some(tx) => {
                let mut next = sequence.get();
                Self::stream_final_marker_with_policy(tx, stream_policy, &request.id, &mut next)?;
                AudioOutput::streamed(output.samples.len(), OUTPUT_SAMPLE_RATE)
            }
            None => AudioOutput::new(output.samples, OUTPUT_SAMPLE_RATE),
        };
        Ok(ModelSessionResult::atomic(ExecutorOutput {
            request_id: request.id.clone(),
            audio: Some(audio),
            text: None,
            input_transcription: None,
            tokens_processed: output.prompt_tokens,
            tokens_generated: output.audio_frames.max(1),
            finished: true,
            phase_timing_override: None,
            asr_diagnostics: None,
            error: None,
            logprobs: Vec::new(),
        }))
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
        let text_sampling = Lfm3TextSampling {
            temperature: request.params.temperature.max(0.0),
            top_k: request.params.top_k,
            top_p: request.params.top_p.clamp(0.0, 1.0),
            ..Lfm3TextSampling::default()
        };
        let sampling = audio_sampling(request);
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

        let transcription = Self::run_blocking(|| {
            with_lm_state(&mut leases, |state| {
                model.transcribe(
                    &samples,
                    sample_rate,
                    None,
                    None,
                    state,
                    &mut || request.is_cancelled(),
                    &mut |_| {},
                )
            })
        });
        let (input_transcription, _) = match leases.fence_on_error(transcription) {
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
                    &samples,
                    sample_rate,
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
