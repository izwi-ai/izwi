use crate::error::{Error, Result};
use crate::models::architectures::lfm25_audio::{
    Lfm25AudioGenerationConfig, Lfm25AudioStreamConfig, Lfm25SamplingConfig,
};
use crate::models::shared::chat::ChatMessage;

use super::super::request::EngineCoreRequest;
use super::super::scheduler::ScheduledRequest;
use super::super::types::AudioOutput;
use super::audio::decode_request_audio_with_rate;
use super::{ExecutorOutput, NativeExecutor};

impl NativeExecutor {
    fn audio_chat_generation_config(request: &EngineCoreRequest) -> Lfm25AudioGenerationConfig {
        Lfm25AudioGenerationConfig {
            text: Lfm25SamplingConfig {
                temperature: request.params.temperature.max(0.0),
                top_k: request.params.top_k,
                top_p: request.params.top_p.clamp(0.0, 1.0),
            },
            audio: Lfm25SamplingConfig {
                temperature: request
                    .params
                    .audio_temperature
                    .unwrap_or(request.params.temperature)
                    .max(0.0),
                top_k: request.params.audio_top_k.unwrap_or(request.params.top_k),
                top_p: request.params.top_p.clamp(0.0, 1.0),
            },
            seed: Self::chat_request_seed(&request.id),
        }
    }

    fn audio_chat_messages(request: &EngineCoreRequest) -> &[ChatMessage] {
        request.chat_messages.as_deref().unwrap_or(&[])
    }

    pub(super) fn audio_chat_request(
        &self,
        request: &EngineCoreRequest,
        _scheduled: &ScheduledRequest,
    ) -> Result<ExecutorOutput> {
        let variant = Self::resolve_variant(request)?;
        let stream_tx = Self::stream_sender(request);
        let generation_config = Self::audio_chat_generation_config(request);
        let stream_config = Lfm25AudioStreamConfig::default();
        let history_messages = Self::audio_chat_messages(request);
        let max_new_tokens = request.params.max_tokens.max(1);

        let (samples, sample_rate) = decode_request_audio_with_rate(request)?;
        let model = self.with_registry(|registry| {
            registry.try_get_audio_chat(variant).ok_or_else(|| {
                Error::ModelNotFound(format!("Audio-chat model {variant} is not loaded"))
            })
        })?;

        let input_transcription = Self::run_blocking(|| model.transcribe(&samples, sample_rate))?;
        let output = Self::run_blocking(|| {
            if let Some(tx) = stream_tx.as_ref() {
                let stream_sequence = std::cell::Cell::new(0usize);
                let stream_err = std::cell::RefCell::new(None::<Error>);
                let mut emit_text = |delta: &str| {
                    if delta.is_empty() || stream_err.borrow().is_some() {
                        return;
                    }
                    let mut sequence = stream_sequence.get();
                    match Self::stream_text(tx, &request.id, &mut sequence, delta.to_string()) {
                        Ok(()) => stream_sequence.set(sequence),
                        Err(err) => {
                            *stream_err.borrow_mut() = Some(err);
                        }
                    }
                };
                let mut emit_audio = |delta: &[f32]| {
                    if delta.is_empty() || stream_err.borrow().is_some() {
                        return;
                    }
                    let mut sequence = stream_sequence.get();
                    match Self::stream_audio(
                        tx,
                        &request.id,
                        &mut sequence,
                        delta.to_vec(),
                        24_000,
                        false,
                    ) {
                        Ok(()) => stream_sequence.set(sequence),
                        Err(err) => {
                            *stream_err.borrow_mut() = Some(err);
                        }
                    }
                };
                let output = model.generate_interleaved_with_config_and_callback(
                    history_messages,
                    &samples,
                    sample_rate,
                    max_new_tokens,
                    request.system_prompt.as_deref(),
                    &generation_config,
                    &stream_config,
                    &mut emit_text,
                    &mut emit_audio,
                )?;
                if let Some(err) = stream_err.into_inner() {
                    return Err(err);
                }
                let mut sequence = stream_sequence.get();
                Self::stream_final_marker(tx, &request.id, &mut sequence)?;
                Ok(output)
            } else {
                let mut no_text = |_delta: &str| {};
                let mut no_audio = |_samples: &[f32]| {};
                model.generate_interleaved_with_config_and_callback(
                    history_messages,
                    &samples,
                    sample_rate,
                    max_new_tokens,
                    request.system_prompt.as_deref(),
                    &generation_config,
                    &stream_config,
                    &mut no_text,
                    &mut no_audio,
                )
            }
        })?;

        Ok(ExecutorOutput {
            request_id: request.id.clone(),
            audio: Some(AudioOutput::new(output.samples, output.sample_rate)),
            text: Some(output.text),
            input_transcription: Some(input_transcription.text),
            tokens_processed: output.prompt_tokens,
            tokens_generated: output.tokens_generated.max(1),
            finished: true,
            error: None,
        })
    }
}
