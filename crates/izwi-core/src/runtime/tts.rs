//! Text-to-speech runtime methods.

use std::time::Instant;

use tokio::sync::mpsc;
use tracing::info;

use crate::catalog::ModelFamily;
use crate::engine::GenerationParams as CoreGenParams;
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::architectures::lfm25_audio::lfm25_audio_tts_system_prompt;
use crate::models::architectures::qwen3::tts::qwen_tts_cuda_chunked_codec_stream_enabled;
use crate::models::shared::chat::{ChatMessage, ChatRole};
use crate::runtime::adapters::CapabilityKind;
use crate::runtime::request::TtsRuntimeRequest;
use crate::runtime::service::RuntimeService;
use crate::runtime::types::{
    AudioChunk, ChunkStats, GenerationConfig, GenerationRequest, GenerationResult,
};

const LFM25_AUDIO_DEFAULT_MAX_NEW_TOKENS: usize = 1024;

fn qwen_tts_cuda_streaming_uses_final_only(
    is_cuda: bool,
    variant: ModelVariant,
    chunked_codec_stream_enabled: bool,
) -> bool {
    is_cuda && matches!(variant.family(), ModelFamily::Qwen3Tts) && !chunked_codec_stream_enabled
}

fn lfm25_audio_prompt_messages(text: &str, speaker: Option<&str>) -> Vec<ChatMessage> {
    vec![
        ChatMessage {
            role: ChatRole::System,
            content: lfm25_audio_tts_system_prompt(speaker).to_string(),
        },
        ChatMessage {
            role: ChatRole::User,
            content: text.trim().to_string(),
        },
    ]
}

impl RuntimeService {
    async fn resolve_tts_variant_for_request(
        &self,
        request: &GenerationRequest,
    ) -> Result<ModelVariant> {
        request
            .model_variant
            .or(*self.loaded_tts_variant.read().await)
            .ok_or_else(|| Error::InferenceError("No TTS model loaded".to_string()))
            .and_then(|variant| {
                self.adapter_registry
                    .require(CapabilityKind::Tts, variant)
                    .map(|_| variant)
            })
    }

    async fn lfm25_audio_tts_generate(
        &self,
        request: GenerationRequest,
        variant: ModelVariant,
        streaming_required: bool,
    ) -> Result<GenerationResult> {
        self.observe_broker_capability_request(
            CapabilityKind::Tts,
            Some(variant),
            streaming_required,
        )?;
        self.load_model(variant).await?;
        let _lease = self.acquire_model_residency_lease(variant);

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
        let requested_speaker = request.config.options.speaker.as_deref().or(request
            .config
            .options
            .voice
            .as_deref());
        let started = Instant::now();
        let output = model.generate_sequential(
            &lfm25_audio_prompt_messages(text, requested_speaker),
            max_new_tokens,
        )?;
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
        variant: ModelVariant,
        chunk_tx: mpsc::Sender<AudioChunk>,
    ) -> Result<()> {
        let result = self
            .lfm25_audio_tts_generate(request, variant, true)
            .await?;
        let mut chunk = AudioChunk::final_chunk(result.request_id.clone(), 0, result.samples);
        chunk.is_final = true;
        chunk_tx
            .send(chunk)
            .await
            .map_err(|_| Error::InferenceError("Streaming output channel closed".to_string()))?;
        Ok(())
    }

    async fn qwen_tts_cuda_final_only_streaming(
        &self,
        mut request: GenerationRequest,
        chunk_tx: mpsc::Sender<AudioChunk>,
    ) -> Result<()> {
        request.config.streaming = false;
        let result = self.generate(request).await?;
        let generation_time_ms = result.total_time_ms;
        let tokens_generated = result.total_tokens;
        let rtf = result.rtf();
        let mut chunk = AudioChunk::final_chunk(result.request_id.clone(), 0, result.samples);
        chunk.stats = Some(ChunkStats {
            generation_time_ms,
            tokens_generated,
            rtf,
        });
        chunk_tx
            .send(chunk)
            .await
            .map_err(|_| Error::InferenceError("Streaming output channel closed".to_string()))?;

        info!(
            "Qwen3-TTS CUDA streaming emitted final-only audio in {:.1}ms (RTF {:.3}); enable IZWI_QWEN_TTS_CUDA_CHUNKED_CODEC_STREAM=1 for progressive CUDA codec streaming",
            generation_time_ms, rtf
        );
        Ok(())
    }
}

impl RuntimeService {
    /// Generate audio from text using the unified core engine.
    pub async fn generate(&self, request: GenerationRequest) -> Result<GenerationResult> {
        let resolved_variant = self.resolve_tts_variant_for_request(&request).await?;
        if matches!(resolved_variant.family(), ModelFamily::KokoroTts) {
            return self.kokoro_tts_generate(request).await;
        }
        if matches!(resolved_variant.family(), ModelFamily::Lfm25Audio) {
            return self
                .lfm25_audio_tts_generate(request, resolved_variant, false)
                .await;
        }
        self.load_model(resolved_variant).await?;

        let core_params = core_params_from_generation(&request.config);
        let core_request = TtsRuntimeRequest::from_generation(request, resolved_variant)?
            .into_engine_request(core_params);

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
        let resolved_variant = self.resolve_tts_variant_for_request(&request).await?;
        if matches!(resolved_variant.family(), ModelFamily::KokoroTts) {
            return self.kokoro_tts_generate_streaming(request, chunk_tx).await;
        }
        if matches!(resolved_variant.family(), ModelFamily::Lfm25Audio) {
            return self
                .lfm25_audio_tts_generate_streaming(request, resolved_variant, chunk_tx)
                .await;
        }
        if qwen_tts_cuda_streaming_uses_final_only(
            self.device.kind.is_cuda(),
            resolved_variant,
            qwen_tts_cuda_chunked_codec_stream_enabled(),
        ) {
            return self
                .qwen_tts_cuda_final_only_streaming(request, chunk_tx)
                .await;
        }
        self.load_model(resolved_variant).await?;

        let core_params = core_params_from_generation(&request.config);
        let core_request = TtsRuntimeRequest::from_generation(request, resolved_variant)?
            .into_engine_request(core_params);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_tts_cuda_streaming_defaults_to_final_only_without_chunked_codec() {
        assert!(qwen_tts_cuda_streaming_uses_final_only(
            true,
            ModelVariant::Qwen3Tts12Hz06BCustomVoice,
            false,
        ));
    }

    #[test]
    fn qwen_tts_cuda_streaming_respects_chunked_codec_opt_in() {
        assert!(!qwen_tts_cuda_streaming_uses_final_only(
            true,
            ModelVariant::Qwen3Tts12Hz06BCustomVoice,
            true,
        ));
    }

    #[test]
    fn qwen_tts_final_only_streaming_policy_does_not_affect_non_cuda_or_non_qwen() {
        assert!(!qwen_tts_cuda_streaming_uses_final_only(
            false,
            ModelVariant::Qwen3Tts12Hz06BCustomVoice,
            false,
        ));
        assert!(!qwen_tts_cuda_streaming_uses_final_only(
            true,
            ModelVariant::Kokoro82M,
            false,
        ));
    }
}
