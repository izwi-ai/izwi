//! Speech-to-speech runtime methods routed through the unified core engine.

use crate::catalog::parse_model_variant;
use crate::engine::{EngineCoreRequest, GenerationParams, StreamingOutput};
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::shared::chat::ChatMessage;
use crate::runtime::service::RuntimeService;
use crate::runtime::types::SpeechToSpeechGeneration;

enum SpeechAudioInput<'a> {
    Base64(&'a str),
    Bytes(&'a [u8]),
}

fn resolve_audio_chat_variant(model_id: Option<&str>) -> Result<ModelVariant> {
    match model_id {
        Some(raw) => {
            let variant = parse_model_variant(raw)
                .map_err(|err| Error::InvalidInput(err.to_string()))?;
            if variant.is_audio_chat() {
                Ok(variant)
            } else {
                Err(Error::InvalidInput(format!(
                    "Model `{raw}` is not an audio-chat model"
                )))
            }
        }
        None => Ok(ModelVariant::Lfm25Audio15BGguf),
    }
}

impl RuntimeService {
    async fn build_speech_to_speech_request(
        &self,
        variant: ModelVariant,
        audio_input: SpeechAudioInput<'_>,
        messages: Vec<ChatMessage>,
        mut params: GenerationParams,
        system_prompt: Option<&str>,
        correlation_id: Option<&str>,
    ) -> Result<EngineCoreRequest> {
        self.load_model(variant).await?;

        let mut request = match audio_input {
            SpeechAudioInput::Base64(audio_base64) => {
                EngineCoreRequest::speech_to_speech(audio_base64.to_string())
            }
            SpeechAudioInput::Bytes(audio_bytes) => {
                EngineCoreRequest::speech_to_speech_bytes(audio_bytes.to_vec())
            }
        };
        request.model_variant = Some(variant);
        request.chat_messages = (!messages.is_empty()).then_some(messages);
        request.system_prompt = system_prompt
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned);
        request.correlation_id = correlation_id.map(|value| value.to_string());
        params.max_tokens = params.max_tokens.max(1);
        request.params = params;
        Ok(request)
    }

    pub async fn speech_to_speech_generate_bytes_with_variant(
        &self,
        variant: ModelVariant,
        audio_bytes: &[u8],
        messages: Vec<ChatMessage>,
        params: GenerationParams,
        system_prompt: Option<&str>,
        correlation_id: Option<&str>,
    ) -> Result<SpeechToSpeechGeneration> {
        let request = self
            .build_speech_to_speech_request(
                variant,
                SpeechAudioInput::Bytes(audio_bytes),
                messages,
                params,
                system_prompt,
                correlation_id,
            )
            .await?;
        let output = self.run_request(request).await?;
        Ok(SpeechToSpeechGeneration {
            text: output.text.unwrap_or_default(),
            samples: output.audio.samples,
            sample_rate: output.audio.sample_rate,
            input_transcription: output.input_transcription,
            generation_time_ms: output.generation_time.as_secs_f64() * 1000.0,
        })
    }

    pub async fn speech_to_speech_generate_with_variant(
        &self,
        variant: ModelVariant,
        audio_base64: &str,
        messages: Vec<ChatMessage>,
        params: GenerationParams,
        system_prompt: Option<&str>,
        correlation_id: Option<&str>,
    ) -> Result<SpeechToSpeechGeneration> {
        let request = self
            .build_speech_to_speech_request(
                variant,
                SpeechAudioInput::Base64(audio_base64),
                messages,
                params,
                system_prompt,
                correlation_id,
            )
            .await?;
        let output = self.run_request(request).await?;
        Ok(SpeechToSpeechGeneration {
            text: output.text.unwrap_or_default(),
            samples: output.audio.samples,
            sample_rate: output.audio.sample_rate,
            input_transcription: output.input_transcription,
            generation_time_ms: output.generation_time.as_secs_f64() * 1000.0,
        })
    }

    pub async fn speech_to_speech_generate_streaming_bytes_with_variant<F>(
        &self,
        variant: ModelVariant,
        audio_bytes: &[u8],
        messages: Vec<ChatMessage>,
        params: GenerationParams,
        system_prompt: Option<&str>,
        correlation_id: Option<&str>,
        mut on_chunk: F,
    ) -> Result<SpeechToSpeechGeneration>
    where
        F: FnMut(StreamingOutput) + Send + 'static,
    {
        let request = self
            .build_speech_to_speech_request(
                variant,
                SpeechAudioInput::Bytes(audio_bytes),
                messages,
                params,
                system_prompt,
                correlation_id,
            )
            .await?;
        let mut streamed_text = String::new();
        let mut streamed_samples = Vec::new();
        let mut streamed_sample_rate = 24_000u32;
        let output = self
            .run_streaming_request(request, |chunk| {
                if let Some(delta) = chunk.text.as_ref() {
                    streamed_text.push_str(delta);
                }
                if !chunk.samples.is_empty() {
                    streamed_sample_rate = chunk.sample_rate.max(1);
                    streamed_samples.extend_from_slice(&chunk.samples);
                }
                on_chunk(chunk);
                std::future::ready(Ok(()))
            })
            .await?;

        Ok(SpeechToSpeechGeneration {
            text: output.text.unwrap_or(streamed_text),
            samples: if output.audio.samples.is_empty() {
                streamed_samples
            } else {
                output.audio.samples
            },
            sample_rate: if output.audio.sample_rate > 0 {
                output.audio.sample_rate
            } else {
                streamed_sample_rate
            },
            input_transcription: output.input_transcription,
            generation_time_ms: output.generation_time.as_secs_f64() * 1000.0,
        })
    }

    pub async fn speech_to_speech_generate_streaming_bytes<F>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        messages: Vec<ChatMessage>,
        params: GenerationParams,
        system_prompt: Option<&str>,
        on_chunk: F,
    ) -> Result<SpeechToSpeechGeneration>
    where
        F: FnMut(StreamingOutput) + Send + 'static,
    {
        let variant = resolve_audio_chat_variant(model_id)?;
        self.speech_to_speech_generate_streaming_bytes_with_variant(
            variant,
            audio_bytes,
            messages,
            params,
            system_prompt,
            None,
            on_chunk,
        )
        .await
    }

    pub async fn speech_to_speech_generate_bytes(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        messages: Vec<ChatMessage>,
        params: GenerationParams,
        system_prompt: Option<&str>,
    ) -> Result<SpeechToSpeechGeneration> {
        let variant = resolve_audio_chat_variant(model_id)?;
        self.speech_to_speech_generate_bytes_with_variant(
            variant,
            audio_bytes,
            messages,
            params,
            system_prompt,
            None,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_audio_chat_variant_defaults_to_lfm25_audio() {
        let variant = resolve_audio_chat_variant(None).expect("default variant");
        assert_eq!(variant, ModelVariant::Lfm25Audio15BGguf);
    }

    #[test]
    fn resolve_audio_chat_variant_rejects_non_audio_chat_models() {
        let err =
            resolve_audio_chat_variant(Some("Qwen3-1.7B-GGUF")).expect_err("expected rejection");
        assert!(err.to_string().contains("not an audio-chat model"));
    }
}
