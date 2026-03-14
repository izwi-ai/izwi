//! Text-to-speech runtime methods.

use std::time::Instant;

use tokio::sync::mpsc;
use tracing::info;

use crate::engine::{EngineCoreRequest, GenerationParams as CoreGenParams};
use crate::error::{Error, Result};
use crate::models::shared::chat::{ChatMessage, ChatRole};
use crate::runtime::service::RuntimeService;
use crate::runtime::types::{AudioChunk, GenerationConfig, GenerationRequest, GenerationResult};

const LFM25_AUDIO_DEFAULT_MAX_NEW_TOKENS: usize = 1024;

fn lfm25_audio_prompt_messages(text: &str) -> Vec<ChatMessage> {
    vec![ChatMessage {
        role: ChatRole::User,
        content: text.trim().to_string(),
    }]
}

impl RuntimeService {
    async fn lfm25_audio_tts_generate(
        &self,
        request: GenerationRequest,
    ) -> Result<GenerationResult> {
        let variant = (*self.loaded_tts_variant.read().await)
            .filter(|variant| matches!(variant.family(), crate::catalog::ModelFamily::Lfm25Audio))
            .ok_or_else(|| Error::InferenceError("No LFM2.5 Audio TTS model loaded".to_string()))?;
        self.load_model(variant).await?;

        let text = request.text.trim();
        if text.is_empty() {
            return Err(Error::InvalidInput("TTS request missing text".to_string()));
        }

        let model = self
            .model_registry
            .get_audio_chat(variant)
            .await
            .ok_or_else(|| Error::InferenceError("No LFM2.5 Audio model loaded".to_string()))?;
        let max_new_tokens = if request.config.options.max_tokens == 0 {
            LFM25_AUDIO_DEFAULT_MAX_NEW_TOKENS
        } else {
            request.config.options.max_tokens
        };
        let started = Instant::now();
        let output =
            model.generate_sequential(&lfm25_audio_prompt_messages(text), max_new_tokens)?;
        let total_time_ms = started.elapsed().as_secs_f32() * 1000.0;

        Ok(GenerationResult {
            request_id: request.id,
            samples: output.samples,
            sample_rate: output.sample_rate,
            total_tokens: output.tokens_generated,
            total_time_ms,
        })
    }

    async fn lfm25_audio_tts_generate_streaming(
        &self,
        request: GenerationRequest,
        chunk_tx: mpsc::Sender<AudioChunk>,
    ) -> Result<()> {
        let result = self.lfm25_audio_tts_generate(request).await?;
        let mut chunk = AudioChunk::final_chunk(result.request_id.clone(), 0, result.samples);
        chunk.is_final = true;
        chunk_tx
            .send(chunk)
            .await
            .map_err(|_| Error::InferenceError("Streaming output channel closed".to_string()))?;
        Ok(())
    }
}

impl RuntimeService {
    /// Generate audio from text using the unified core engine.
    pub async fn generate(&self, request: GenerationRequest) -> Result<GenerationResult> {
        let loaded_variant = *self.loaded_tts_variant.read().await;
        if loaded_variant
            .map(|variant| matches!(variant.family(), crate::catalog::ModelFamily::KokoroTts))
            .unwrap_or(false)
        {
            return self.kokoro_tts_generate(request).await;
        }
        if loaded_variant
            .map(|variant| matches!(variant.family(), crate::catalog::ModelFamily::Lfm25Audio))
            .unwrap_or(false)
        {
            return self.lfm25_audio_tts_generate(request).await;
        }

        let mut core_request = EngineCoreRequest::tts(request.text.clone());
        core_request.id = request.id.clone();
        core_request.correlation_id = request.correlation_id.clone();
        core_request.model_variant = loaded_variant;
        core_request.language = request.language.clone();
        core_request.reference_audio = request.reference_audio.clone();
        core_request.reference_text = request.reference_text.clone();
        core_request.voice_description = request.voice_description.clone();
        core_request.params = core_params_from_generation(&request.config);

        let output = self.run_request(core_request).await?;
        let samples = output.audio.samples;
        let sample_rate = output.audio.sample_rate;
        let total_tokens = output.num_tokens;
        let total_time_ms = output.generation_time.as_secs_f32() * 1000.0;

        info!(
            "Generated {} samples in {:.1}ms via core engine",
            samples.len(),
            total_time_ms
        );

        Ok(GenerationResult {
            request_id: output.request_id,
            samples,
            sample_rate,
            total_tokens,
            total_time_ms,
        })
    }

    /// Generate audio with streaming output.
    ///
    /// Streaming is emitted from engine outputs in chunked form so all synthesis
    /// execution still routes through the core engine.
    pub async fn generate_streaming(
        &self,
        request: GenerationRequest,
        chunk_tx: mpsc::Sender<AudioChunk>,
    ) -> Result<()> {
        let loaded_variant = *self.loaded_tts_variant.read().await;
        if loaded_variant
            .map(|variant| matches!(variant.family(), crate::catalog::ModelFamily::KokoroTts))
            .unwrap_or(false)
        {
            return self.kokoro_tts_generate_streaming(request, chunk_tx).await;
        }
        if loaded_variant
            .map(|variant| matches!(variant.family(), crate::catalog::ModelFamily::Lfm25Audio))
            .unwrap_or(false)
        {
            return self
                .lfm25_audio_tts_generate_streaming(request, chunk_tx)
                .await;
        }

        let mut core_request = EngineCoreRequest::tts(request.text.clone());
        core_request.id = request.id.clone();
        core_request.correlation_id = request.correlation_id.clone();
        core_request.model_variant = loaded_variant;
        core_request.language = request.language.clone();
        core_request.reference_audio = request.reference_audio.clone();
        core_request.reference_text = request.reference_text.clone();
        core_request.voice_description = request.voice_description.clone();
        core_request.params = core_params_from_generation(&request.config);

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

        info!("Streaming generation complete via core engine");
        Ok(())
    }
}

fn core_params_from_generation(config: &GenerationConfig) -> CoreGenParams {
    let mut params = config.options.clone();
    if params.voice.is_none() {
        params.voice = params.speaker.clone();
    }
    params
}
