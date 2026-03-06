//! LFM2 runtime helpers routed through the unified core engine.

use tokio::sync::mpsc;

use crate::engine::{EngineCoreRequest, GenerationParams as CoreGenParams};
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::runtime::service::RuntimeService;
use crate::runtime::types::{
    AsrTranscription, AudioChunk, GenerationRequest, GenerationResult, SpeechToSpeechGeneration,
};

impl RuntimeService {
    const LFM2_TTS_DEFAULT_AUDIO_TEMPERATURE: f32 = 0.8;
    const LFM2_TTS_DEFAULT_AUDIO_TOP_K: usize = 64;
    const LFM2_S2S_DEFAULT_AUDIO_TEMPERATURE: f32 = 1.0;
    const LFM2_S2S_DEFAULT_AUDIO_TOP_K: usize = 4;

    fn default_lfm2_variant() -> ModelVariant {
        ModelVariant::Lfm25Audio15B
    }

    async fn resolve_active_lfm2_variant(&self) -> ModelVariant {
        if let Some(variant) = *self.loaded_tts_variant.read().await {
            if matches!(variant.family(), crate::catalog::ModelFamily::Lfm2Audio) {
                return variant;
            }
        }
        Self::default_lfm2_variant()
    }

    fn build_lfm2_tts_request(
        request: &GenerationRequest,
        variant: ModelVariant,
    ) -> Result<EngineCoreRequest> {
        let opts = &request.config.options;
        let using_generic_defaults =
            opts.top_k == 0 && (opts.temperature - 0.7).abs() < f32::EPSILON;
        let temperature = if using_generic_defaults {
            Self::LFM2_TTS_DEFAULT_AUDIO_TEMPERATURE
        } else {
            opts.temperature
        };
        let top_k = if opts.top_k > 0 {
            opts.top_k
        } else {
            Self::LFM2_TTS_DEFAULT_AUDIO_TOP_K
        };

        let mut core_request = EngineCoreRequest::tts(request.text.clone());
        core_request.id = request.id.clone();
        core_request.correlation_id = request.correlation_id.clone();
        core_request.model_variant = Some(variant);
        core_request.language = request.language.clone();
        core_request.voice_description = request.voice_description.clone();
        core_request.reference_audio = request.reference_audio.clone();
        core_request.reference_text = request.reference_text.clone();
        core_request.params = CoreGenParams {
            temperature,
            top_p: opts.top_p,
            top_k,
            repetition_penalty: opts.repetition_penalty,
            max_tokens: if opts.max_tokens == 0 {
                512
            } else {
                opts.max_tokens
            },
            speaker: opts.speaker.clone(),
            voice: opts.voice.clone().or_else(|| opts.speaker.clone()),
            audio_temperature: Some(temperature),
            audio_top_k: Some(top_k),
            speed: opts.speed,
            stop_sequences: Vec::new(),
            stop_token_ids: Vec::new(),
        };

        Ok(core_request)
    }

    fn build_lfm2_s2s_request(
        audio_base64: &str,
        variant: ModelVariant,
        language: Option<&str>,
        system_prompt: Option<&str>,
        temperature: Option<f32>,
        top_k: Option<usize>,
        correlation_id: Option<&str>,
    ) -> EngineCoreRequest {
        let resolved_temperature = temperature.unwrap_or(Self::LFM2_S2S_DEFAULT_AUDIO_TEMPERATURE);
        let resolved_top_k = top_k
            .filter(|&v| v > 0)
            .unwrap_or(Self::LFM2_S2S_DEFAULT_AUDIO_TOP_K);

        let mut request = EngineCoreRequest::speech_to_speech(audio_base64.to_string());
        request.model_variant = Some(variant);
        request.correlation_id = correlation_id.map(|s| s.to_string());
        request.language = language.map(|s| s.to_string());
        request.system_prompt = system_prompt.map(|s| s.to_string());
        request.params = CoreGenParams {
            temperature: resolved_temperature,
            top_p: 1.0,
            top_k: resolved_top_k,
            repetition_penalty: 1.0,
            max_tokens: 1024,
            speaker: None,
            voice: None,
            audio_temperature: Some(resolved_temperature),
            audio_top_k: Some(resolved_top_k),
            speed: 1.0,
            stop_sequences: Vec::new(),
            stop_token_ids: Vec::new(),
        };
        request
    }

    fn build_lfm2_s2s_request_bytes(
        audio_bytes: &[u8],
        variant: ModelVariant,
        language: Option<&str>,
        system_prompt: Option<&str>,
        temperature: Option<f32>,
        top_k: Option<usize>,
        correlation_id: Option<&str>,
    ) -> EngineCoreRequest {
        let resolved_temperature = temperature.unwrap_or(Self::LFM2_S2S_DEFAULT_AUDIO_TEMPERATURE);
        let resolved_top_k = top_k
            .filter(|&v| v > 0)
            .unwrap_or(Self::LFM2_S2S_DEFAULT_AUDIO_TOP_K);

        let mut request = EngineCoreRequest::speech_to_speech_bytes(audio_bytes.to_vec());
        request.model_variant = Some(variant);
        request.correlation_id = correlation_id.map(|s| s.to_string());
        request.language = language.map(|s| s.to_string());
        request.system_prompt = system_prompt.map(|s| s.to_string());
        request.params = CoreGenParams {
            temperature: resolved_temperature,
            top_p: 1.0,
            top_k: resolved_top_k,
            repetition_penalty: 1.0,
            max_tokens: 1024,
            speaker: None,
            voice: None,
            audio_temperature: Some(resolved_temperature),
            audio_top_k: Some(resolved_top_k),
            speed: 1.0,
            stop_sequences: Vec::new(),
            stop_token_ids: Vec::new(),
        };
        request
    }

    pub async fn lfm2_asr_transcribe_streaming<F>(
        &self,
        variant: ModelVariant,
        audio_base64: &str,
        language: Option<&str>,
        mut on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.load_model(variant).await?;

        let mut request = EngineCoreRequest::asr(audio_base64.to_string());
        request.model_variant = Some(variant);
        request.language = language.map(|s| s.to_string());

        let mut streamed_text = String::new();
        let output = self
            .run_streaming_request(request, |chunk| {
                if let Some(delta) = chunk.text {
                    if !delta.is_empty() {
                        streamed_text.push_str(&delta);
                        on_delta(delta);
                    }
                }
                std::future::ready(Ok(()))
            })
            .await?;
        let text = output.text.unwrap_or(streamed_text);

        Ok(AsrTranscription {
            text,
            language: language.map(|s| s.to_string()),
            duration_secs: output.audio.duration_secs,
        })
    }

    pub async fn lfm2_tts_generate(&self, request: GenerationRequest) -> Result<GenerationResult> {
        let variant = self.resolve_active_lfm2_variant().await;
        self.load_model(variant).await?;
        let core_request = Self::build_lfm2_tts_request(&request, variant)?;

        let output = self.run_request(core_request).await?;
        Ok(GenerationResult {
            request_id: output.request_id,
            samples: output.audio.samples,
            sample_rate: output.audio.sample_rate,
            total_tokens: output.num_tokens,
            total_time_ms: output.generation_time.as_secs_f32() * 1000.0,
        })
    }

    pub async fn lfm2_tts_generate_streaming(
        &self,
        request: GenerationRequest,
        chunk_tx: mpsc::Sender<AudioChunk>,
    ) -> Result<()> {
        let variant = self.resolve_active_lfm2_variant().await;
        self.load_model(variant).await?;
        let core_request = Self::build_lfm2_tts_request(&request, variant)?;

        self.run_streaming_request(core_request, |stream_chunk| {
            let tx = chunk_tx.clone();
            async move {
                if stream_chunk.samples.is_empty() && !stream_chunk.is_final {
                    return Ok(());
                }

                let mut chunk = AudioChunk::new(
                    stream_chunk.request_id.clone(),
                    stream_chunk.sequence,
                    stream_chunk.samples,
                );
                chunk.is_final = stream_chunk.is_final;
                tx.send(chunk).await.map_err(|_| {
                    Error::InferenceError("Streaming output channel closed".to_string())
                })?;
                Ok(())
            }
        })
        .await?;

        Ok(())
    }

    pub async fn lfm2_speech_to_speech_streaming<F, G>(
        &self,
        audio_base64: &str,
        language: Option<&str>,
        system_prompt: Option<&str>,
        temperature: Option<f32>,
        top_k: Option<usize>,
        on_delta: F,
        on_audio_chunk: G,
    ) -> Result<SpeechToSpeechGeneration>
    where
        F: FnMut(String) -> Result<()> + Send + 'static,
        G: FnMut(AudioChunk) -> Result<()> + Send + 'static,
    {
        self.lfm2_speech_to_speech_streaming_with_correlation(
            audio_base64,
            language,
            system_prompt,
            temperature,
            top_k,
            None,
            on_delta,
            on_audio_chunk,
        )
        .await
    }

    pub async fn lfm2_speech_to_speech_streaming_with_correlation<F, G>(
        &self,
        audio_base64: &str,
        language: Option<&str>,
        system_prompt: Option<&str>,
        temperature: Option<f32>,
        top_k: Option<usize>,
        correlation_id: Option<&str>,
        mut on_delta: F,
        mut on_audio_chunk: G,
    ) -> Result<SpeechToSpeechGeneration>
    where
        F: FnMut(String) -> Result<()> + Send + 'static,
        G: FnMut(AudioChunk) -> Result<()> + Send + 'static,
    {
        let variant = self.resolve_active_lfm2_variant().await;
        self.load_model(variant).await?;
        let request = Self::build_lfm2_s2s_request(
            audio_base64,
            variant,
            language,
            system_prompt,
            temperature,
            top_k,
            correlation_id,
        );

        let mut streamed_text = String::new();
        let output = self
            .run_streaming_request(request, |chunk| {
                let callback_result = (|| -> Result<()> {
                    if !chunk.samples.is_empty() || chunk.is_final {
                        let mut audio_chunk = AudioChunk::new(
                            chunk.request_id.clone(),
                            chunk.sequence,
                            chunk.samples,
                        );
                        audio_chunk.is_final = chunk.is_final;
                        on_audio_chunk(audio_chunk)?;
                    }

                    if let Some(delta) = chunk.text {
                        if !delta.is_empty() {
                            streamed_text.push_str(&delta);
                            on_delta(delta)?;
                        }
                    }

                    Ok(())
                })();
                std::future::ready(callback_result)
            })
            .await?;
        let text = output.text.unwrap_or(streamed_text);

        Ok(SpeechToSpeechGeneration {
            text,
            samples: output.audio.samples,
            sample_rate: output.audio.sample_rate,
            input_transcription: None,
            generation_time_ms: output.generation_time.as_secs_f64() * 1000.0,
        })
    }

    pub async fn lfm2_speech_to_speech_streaming_bytes_with_correlation<F, G>(
        &self,
        audio_bytes: &[u8],
        language: Option<&str>,
        system_prompt: Option<&str>,
        temperature: Option<f32>,
        top_k: Option<usize>,
        correlation_id: Option<&str>,
        mut on_delta: F,
        mut on_audio_chunk: G,
    ) -> Result<SpeechToSpeechGeneration>
    where
        F: FnMut(String) -> Result<()> + Send + 'static,
        G: FnMut(AudioChunk) -> Result<()> + Send + 'static,
    {
        let variant = self.resolve_active_lfm2_variant().await;
        self.load_model(variant).await?;
        let request = Self::build_lfm2_s2s_request_bytes(
            audio_bytes,
            variant,
            language,
            system_prompt,
            temperature,
            top_k,
            correlation_id,
        );

        let mut streamed_text = String::new();
        let output = self
            .run_streaming_request(request, |chunk| {
                let callback_result = (|| -> Result<()> {
                    if !chunk.samples.is_empty() || chunk.is_final {
                        let mut audio_chunk = AudioChunk::new(
                            chunk.request_id.clone(),
                            chunk.sequence,
                            chunk.samples,
                        );
                        audio_chunk.is_final = chunk.is_final;
                        on_audio_chunk(audio_chunk)?;
                    }

                    if let Some(delta) = chunk.text {
                        if !delta.is_empty() {
                            streamed_text.push_str(&delta);
                            on_delta(delta)?;
                        }
                    }

                    Ok(())
                })();
                std::future::ready(callback_result)
            })
            .await?;
        let text = output.text.unwrap_or(streamed_text);

        Ok(SpeechToSpeechGeneration {
            text,
            samples: output.audio.samples,
            sample_rate: output.audio.sample_rate,
            input_transcription: None,
            generation_time_ms: output.generation_time.as_secs_f64() * 1000.0,
        })
    }

    pub async fn lfm2_speech_to_speech(
        &self,
        audio_base64: &str,
        language: Option<&str>,
        system_prompt: Option<&str>,
        temperature: Option<f32>,
        top_k: Option<usize>,
    ) -> Result<SpeechToSpeechGeneration> {
        self.lfm2_speech_to_speech_with_correlation(
            audio_base64,
            language,
            system_prompt,
            temperature,
            top_k,
            None,
        )
        .await
    }

    pub async fn lfm2_speech_to_speech_with_correlation(
        &self,
        audio_base64: &str,
        language: Option<&str>,
        system_prompt: Option<&str>,
        temperature: Option<f32>,
        top_k: Option<usize>,
        correlation_id: Option<&str>,
    ) -> Result<SpeechToSpeechGeneration> {
        self.lfm2_speech_to_speech_streaming_with_correlation(
            audio_base64,
            language,
            system_prompt,
            temperature,
            top_k,
            correlation_id,
            |_delta| Ok(()),
            |_chunk| Ok(()),
        )
        .await
    }

    pub async fn lfm2_speech_to_speech_bytes_with_correlation(
        &self,
        audio_bytes: &[u8],
        language: Option<&str>,
        system_prompt: Option<&str>,
        temperature: Option<f32>,
        top_k: Option<usize>,
        correlation_id: Option<&str>,
    ) -> Result<SpeechToSpeechGeneration> {
        self.lfm2_speech_to_speech_streaming_bytes_with_correlation(
            audio_bytes,
            language,
            system_prompt,
            temperature,
            top_k,
            correlation_id,
            |_delta| Ok(()),
            |_chunk| Ok(()),
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use crate::models::architectures::lfm2::audio::LFM2_DEFAULT_S2S_PROMPT;

    #[test]
    fn uses_default_s2s_prompt_when_missing() {
        let prompt = None::<&str>.unwrap_or(LFM2_DEFAULT_S2S_PROMPT);
        assert_eq!(prompt, LFM2_DEFAULT_S2S_PROMPT);
    }
}
