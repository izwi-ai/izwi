//! Text-to-speech runtime methods.

use tokio::sync::mpsc;
use tracing::info;

use crate::engine::{EngineCoreRequest, GenerationParams as CoreGenParams};
use crate::error::{Error, Result};
use crate::runtime::service::RuntimeService;
use crate::runtime::types::{AudioChunk, GenerationConfig, GenerationRequest, GenerationResult};

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
