//! ASR runtime methods routed through the unified core engine.

use std::collections::HashSet;
use std::sync::Arc;

use crate::catalog::{ModelFamily, parse_model_variant, resolve_asr_model_variant};
use crate::engine::{AsrProgress, EngineCoreRequest};
use crate::error::{Error, Result};
use crate::model::{ModelResidencyLease, ModelVariant};
use crate::models::architectures::granite_speech::asr::{
    GraniteSpeechTask, parse_granite_speech_output,
};
use crate::models::registry::{
    NativeAsrGenerationOptions, NativeAsrModel, NativeAsrRealtimeEvent, NativeAsrRealtimeState,
};
use crate::runtime::adapters::CapabilityKind;
use crate::runtime::audio_io::{base64_decode, decode_audio_bytes, wav_duration_seconds_fast};
use crate::runtime::request::{AlignmentRuntimeRequest, AsrRuntimeRequest};
use crate::runtime::service::RuntimeService;
use crate::runtime::types::{
    AsrTranscription, SpeakerAttributedAsrResult, SpeakerAttributedAsrStatus,
    SpeakerAttributedAsrTurn,
};

#[derive(Clone, Copy)]
enum AsrAudioInput<'a> {
    Base64(&'a str),
    Bytes(&'a [u8]),
}

const GRANITE_ASR_AUTO_MIN_TOKENS: usize = 76;
const GRANITE_ASR_AUTO_MAX_TOKENS: usize = 2048;
const GRANITE_ASR_AUTO_BASE_SECONDS: f32 = 28.0;
const GRANITE_ASR_AUTO_TOKENS_PER_SECOND: f32 = 4.0;
const GRANITE_SAA_MAX_NEW_TOKENS: usize = 10_000;
const UNKNOWN_SAA_SPEAKER: &str = "UNKNOWN";

fn resolve_asr_realtime_stream_variant(model_id: Option<&str>) -> Option<ModelVariant> {
    let variant = resolve_asr_model_variant(model_id);
    (variant.family() == ModelFamily::NemotronAsr).then_some(variant)
}

pub(crate) fn granite_auto_asr_max_tokens_for_duration(audio_seconds: f32) -> usize {
    let duration_budget =
        if audio_seconds.is_finite() && audio_seconds > GRANITE_ASR_AUTO_BASE_SECONDS {
            ((audio_seconds - GRANITE_ASR_AUTO_BASE_SECONDS) * GRANITE_ASR_AUTO_TOKENS_PER_SECOND)
                .ceil() as usize
        } else {
            0
        };
    GRANITE_ASR_AUTO_MIN_TOKENS
        .saturating_add(duration_budget)
        .clamp(GRANITE_ASR_AUTO_MIN_TOKENS, GRANITE_ASR_AUTO_MAX_TOKENS)
}

fn granite_auto_asr_max_tokens(
    variant: ModelVariant,
    audio_input: AsrAudioInput<'_>,
) -> Result<Option<usize>> {
    if variant.family() != ModelFamily::GraniteSpeechAsr {
        return Ok(None);
    }
    let audio_bytes = match audio_input {
        AsrAudioInput::Base64(audio_base64) => base64_decode(audio_base64)?,
        AsrAudioInput::Bytes(audio_bytes) => audio_bytes.to_vec(),
    };
    let audio_seconds = if let Some(duration) = wav_duration_seconds_fast(&audio_bytes) {
        duration
    } else {
        let (samples, sample_rate) = decode_audio_bytes(&audio_bytes)?;
        if sample_rate > 0 {
            samples.len() as f32 / sample_rate as f32
        } else {
            0.0
        }
    };
    Ok(Some(granite_auto_asr_max_tokens_for_duration(
        audio_seconds,
    )))
}

pub struct RuntimeAsrRealtimeStream {
    variant: ModelVariant,
    model: Arc<NativeAsrModel>,
    state: NativeAsrRealtimeState,
    _lease: ModelResidencyLease,
}

#[derive(Debug, Clone)]
pub struct RuntimeAsrRealtimeEvent {
    pub delta: String,
    pub text: String,
    pub is_final: bool,
    pub chunk_index: usize,
}

fn map_native_realtime_events(events: Vec<NativeAsrRealtimeEvent>) -> Vec<RuntimeAsrRealtimeEvent> {
    events
        .into_iter()
        .map(|event| RuntimeAsrRealtimeEvent {
            delta: event.delta,
            text: event.text,
            is_final: event.is_final,
            chunk_index: event.chunk_index,
        })
        .collect()
}

impl RuntimeService {
    pub async fn try_start_asr_realtime_stream(
        &self,
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
    ) -> Result<Option<RuntimeAsrRealtimeStream>> {
        let Some(variant) = resolve_asr_realtime_stream_variant(model_id) else {
            return Ok(None);
        };

        self.load_model(variant).await?;
        let lease = self.acquire_model_residency_lease(variant);
        let model =
            self.model_registry.get_asr(variant).await.ok_or_else(|| {
                Error::ModelNotFound(format!("ASR model {variant} is not loaded"))
            })?;
        if !model.supports_realtime_stream_decode() {
            return Ok(None);
        }
        let state = model.start_realtime_stream_state(language, prompt, None)?;

        Ok(Some(RuntimeAsrRealtimeStream {
            variant,
            model,
            state,
            _lease: lease,
        }))
    }

    pub fn push_asr_realtime_samples(
        &self,
        stream: &mut RuntimeAsrRealtimeStream,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<RuntimeAsrRealtimeEvent>> {
        let events =
            stream
                .model
                .push_realtime_stream_samples(&mut stream.state, samples, sample_rate)?;
        Ok(map_native_realtime_events(events))
    }

    pub fn finish_asr_realtime_stream(
        &self,
        stream: &mut RuntimeAsrRealtimeStream,
    ) -> Result<Vec<RuntimeAsrRealtimeEvent>> {
        let events = stream.model.finish_realtime_stream(&mut stream.state)?;
        Ok(map_native_realtime_events(events))
    }

    pub fn asr_realtime_stream_variant(&self, stream: &RuntimeAsrRealtimeStream) -> ModelVariant {
        stream.variant
    }

    async fn asr_transcribe_audio_chat_samples<F>(
        &self,
        variant: ModelVariant,
        samples: Vec<f32>,
        sample_rate: u32,
        mut on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String),
    {
        self.load_model(variant).await?;
        let _lease = self.acquire_model_residency_lease(variant);
        let model = self
            .model_registry
            .get_audio_chat(variant)
            .await
            .ok_or_else(|| {
                Error::ModelNotFound(format!("Audio-chat model {variant} is not loaded"))
            })?;

        let mut delta_sink = |delta: &str| {
            if !delta.is_empty() {
                on_delta(delta.to_string());
            }
        };
        let output = model.transcribe_with_callback(&samples, sample_rate, &mut delta_sink)?;

        Ok(AsrTranscription {
            text: output.text,
            language: output.language,
            duration_secs: if sample_rate > 0 {
                samples.len() as f32 / sample_rate as f32
            } else {
                0.0
            },
            asr_diagnostics: None,
        })
    }

    async fn asr_transcribe_audio_chat_base64<F>(
        &self,
        variant: ModelVariant,
        audio_base64: &str,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String),
    {
        let audio_bytes = base64_decode(audio_base64)?;
        let (samples, sample_rate) = decode_audio_bytes(&audio_bytes)?;
        self.asr_transcribe_audio_chat_samples(variant, samples, sample_rate, on_delta)
            .await
    }

    async fn asr_transcribe_audio_chat_bytes<F>(
        &self,
        variant: ModelVariant,
        audio_bytes: &[u8],
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String),
    {
        let (samples, sample_rate) = decode_audio_bytes(audio_bytes)?;
        self.asr_transcribe_audio_chat_samples(variant, samples, sample_rate, on_delta)
            .await
    }

    async fn build_asr_request(
        &self,
        variant: ModelVariant,
        audio_input: AsrAudioInput<'_>,
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
    ) -> Result<EngineCoreRequest> {
        self.load_model(variant).await?;

        let runtime_request = match audio_input {
            AsrAudioInput::Base64(audio_base64) => AsrRuntimeRequest::from_base64(
                audio_base64,
                variant,
                language.map(ToOwned::to_owned),
                correlation_id.map(ToOwned::to_owned),
            )?,
            AsrAudioInput::Bytes(audio_bytes) => AsrRuntimeRequest::from_bytes(
                audio_bytes.to_vec(),
                variant,
                language.map(ToOwned::to_owned),
                correlation_id.map(ToOwned::to_owned),
            )?,
        }
        .with_prompt(prompt.map(ToOwned::to_owned));
        let mut request = runtime_request.into_engine_request();
        if let Some(max_tokens) = max_tokens {
            request.params.max_tokens = max_tokens;
        } else if let Some(auto_max_tokens) = granite_auto_asr_max_tokens(variant, audio_input)? {
            request.params.max_tokens = auto_max_tokens;
            request.asr_auto_max_tokens = true;
        }
        Ok(request)
    }

    pub(crate) async fn asr_transcribe_with_variant(
        &self,
        variant: ModelVariant,
        audio_base64: &str,
        language: Option<&str>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        self.asr_transcribe_with_variant_and_prompt(
            variant,
            audio_base64,
            language,
            None,
            correlation_id,
        )
        .await
    }

    pub(crate) async fn asr_transcribe_with_variant_and_prompt(
        &self,
        variant: ModelVariant,
        audio_base64: &str,
        language: Option<&str>,
        prompt: Option<&str>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        self.asr_transcribe_with_variant_and_prompt_options(
            variant,
            audio_base64,
            language,
            prompt,
            None,
            correlation_id,
        )
        .await
    }

    pub(crate) async fn asr_transcribe_with_variant_and_prompt_options(
        &self,
        variant: ModelVariant,
        audio_base64: &str,
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        if variant.is_audio_chat() {
            self.observe_broker_capability_request(CapabilityKind::Asr, Some(variant), false)?;
            return self
                .asr_transcribe_audio_chat_base64(variant, audio_base64, |_delta| {})
                .await;
        }

        let request = self
            .build_asr_request(
                variant,
                AsrAudioInput::Base64(audio_base64),
                language,
                prompt,
                max_tokens,
                correlation_id,
            )
            .await?;
        let output = self.run_request(request).await?;
        let text = output.text.unwrap_or_default();

        Ok(AsrTranscription {
            text,
            language: language.map(|s| s.to_string()),
            duration_secs: output.audio.duration_secs,
            asr_diagnostics: output.asr_diagnostics,
        })
    }

    pub(crate) async fn asr_transcribe_with_variant_streaming<F>(
        &self,
        variant: ModelVariant,
        audio_base64: &str,
        language: Option<&str>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_with_variant_streaming_and_prompt(
            variant,
            audio_base64,
            language,
            None,
            correlation_id,
            on_delta,
        )
        .await
    }

    pub(crate) async fn asr_transcribe_with_variant_streaming_and_prompt<F>(
        &self,
        variant: ModelVariant,
        audio_base64: &str,
        language: Option<&str>,
        prompt: Option<&str>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_with_variant_streaming_and_prompt_options(
            variant,
            audio_base64,
            language,
            prompt,
            None,
            correlation_id,
            on_delta,
        )
        .await
    }

    pub(crate) async fn asr_transcribe_with_variant_streaming_and_prompt_options<F>(
        &self,
        variant: ModelVariant,
        audio_base64: &str,
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        mut on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        if variant.is_audio_chat() {
            self.observe_broker_capability_request(CapabilityKind::Asr, Some(variant), true)?;
            return self
                .asr_transcribe_audio_chat_base64(variant, audio_base64, on_delta)
                .await;
        }

        let request = self
            .build_asr_request(
                variant,
                AsrAudioInput::Base64(audio_base64),
                language,
                prompt,
                max_tokens,
                correlation_id,
            )
            .await?;
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
            asr_diagnostics: output.asr_diagnostics,
        })
    }

    pub(crate) async fn asr_transcribe_bytes_with_variant(
        &self,
        variant: ModelVariant,
        audio_bytes: &[u8],
        language: Option<&str>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        self.asr_transcribe_bytes_with_variant_and_prompt(
            variant,
            audio_bytes,
            language,
            None,
            correlation_id,
        )
        .await
    }

    pub(crate) async fn asr_transcribe_bytes_with_variant_and_prompt(
        &self,
        variant: ModelVariant,
        audio_bytes: &[u8],
        language: Option<&str>,
        prompt: Option<&str>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        self.asr_transcribe_bytes_with_variant_and_prompt_options(
            variant,
            audio_bytes,
            language,
            prompt,
            None,
            correlation_id,
        )
        .await
    }

    pub(crate) async fn asr_transcribe_bytes_with_variant_and_prompt_options(
        &self,
        variant: ModelVariant,
        audio_bytes: &[u8],
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        if variant.is_audio_chat() {
            self.observe_broker_capability_request(CapabilityKind::Asr, Some(variant), false)?;
            return self
                .asr_transcribe_audio_chat_bytes(variant, audio_bytes, |_delta| {})
                .await;
        }

        let request = self
            .build_asr_request(
                variant,
                AsrAudioInput::Bytes(audio_bytes),
                language,
                prompt,
                max_tokens,
                correlation_id,
            )
            .await?;
        let output = self.run_request(request).await?;
        let text = output.text.unwrap_or_default();

        Ok(AsrTranscription {
            text,
            language: language.map(|s| s.to_string()),
            duration_secs: output.audio.duration_secs,
            asr_diagnostics: output.asr_diagnostics,
        })
    }

    pub(crate) async fn asr_transcribe_bytes_with_variant_streaming<F>(
        &self,
        variant: ModelVariant,
        audio_bytes: &[u8],
        language: Option<&str>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_bytes_with_variant_streaming_and_prompt(
            variant,
            audio_bytes,
            language,
            None,
            correlation_id,
            on_delta,
        )
        .await
    }

    pub(crate) async fn asr_transcribe_bytes_with_variant_streaming_and_prompt<F>(
        &self,
        variant: ModelVariant,
        audio_bytes: &[u8],
        language: Option<&str>,
        prompt: Option<&str>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_bytes_with_variant_streaming_and_prompt_options(
            variant,
            audio_bytes,
            language,
            prompt,
            None,
            correlation_id,
            on_delta,
        )
        .await
    }

    pub(crate) async fn asr_transcribe_bytes_with_variant_streaming_and_prompt_options<F>(
        &self,
        variant: ModelVariant,
        audio_bytes: &[u8],
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_bytes_with_variant_streaming_and_prompt_options_with_progress(
            variant,
            audio_bytes,
            language,
            prompt,
            max_tokens,
            correlation_id,
            on_delta,
            |_progress| {},
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn asr_transcribe_bytes_with_variant_streaming_and_prompt_options_with_progress<
        F,
        P,
    >(
        &self,
        variant: ModelVariant,
        audio_bytes: &[u8],
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        mut on_delta: F,
        mut on_progress: P,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
        P: FnMut(AsrProgress) + Send + 'static,
    {
        if variant.is_audio_chat() {
            self.observe_broker_capability_request(CapabilityKind::Asr, Some(variant), true)?;
            return self
                .asr_transcribe_audio_chat_bytes(variant, audio_bytes, on_delta)
                .await;
        }

        let request = self
            .build_asr_request(
                variant,
                AsrAudioInput::Bytes(audio_bytes),
                language,
                prompt,
                max_tokens,
                correlation_id,
            )
            .await?;
        let mut streamed_text = String::new();
        let output = self
            .run_streaming_request(request, |chunk| {
                if let Some(progress) = chunk.asr_progress {
                    on_progress(progress);
                }
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
            asr_diagnostics: output.asr_diagnostics,
        })
    }

    /// Transcribe audio with Voxtral through the offline transcription path.
    pub async fn voxtral_transcribe(
        &self,
        audio_base64: &str,
        language: Option<&str>,
    ) -> Result<AsrTranscription> {
        self.asr_transcribe_with_variant(
            ModelVariant::VoxtralMini4BRealtime2602,
            audio_base64,
            language,
            None,
        )
        .await
    }

    /// Transcribe audio with Voxtral and emit incremental deltas from offline decode.
    pub async fn voxtral_transcribe_streaming<F>(
        &self,
        audio_base64: &str,
        language: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_with_variant_streaming(
            ModelVariant::VoxtralMini4BRealtime2602,
            audio_base64,
            language,
            None,
            on_delta,
        )
        .await
    }

    /// Transcribe audio with native ASR models.
    pub async fn asr_transcribe(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
    ) -> Result<AsrTranscription> {
        self.asr_transcribe_with_correlation(audio_base64, model_id, language, None)
            .await
    }

    /// Transcribe audio with request correlation metadata.
    pub async fn asr_transcribe_with_correlation(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        self.asr_transcribe_with_prompt_and_correlation(
            audio_base64,
            model_id,
            language,
            None,
            correlation_id,
        )
        .await
    }

    /// Transcribe audio with optional ASR initial prompt/context metadata.
    pub async fn asr_transcribe_with_prompt_and_correlation(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_with_variant_and_prompt(
            variant,
            audio_base64,
            language,
            prompt,
            correlation_id,
        )
        .await
    }

    /// Transcribe audio with optional ASR prompt and max-token decode budget.
    pub async fn asr_transcribe_with_prompt_max_tokens_and_correlation(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_with_variant_and_prompt_options(
            variant,
            audio_base64,
            language,
            prompt,
            max_tokens,
            correlation_id,
        )
        .await
    }

    pub async fn asr_transcribe_bytes(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
    ) -> Result<AsrTranscription> {
        self.asr_transcribe_bytes_with_prompt(audio_bytes, model_id, language, None)
            .await
    }

    pub async fn asr_transcribe_bytes_with_prompt(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
    ) -> Result<AsrTranscription> {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_bytes_with_variant_and_prompt(
            variant,
            audio_bytes,
            language,
            prompt,
            None,
        )
        .await
    }

    pub async fn asr_transcribe_bytes_with_prompt_max_tokens_and_correlation(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_bytes_with_variant_and_prompt_options(
            variant,
            audio_bytes,
            language,
            prompt,
            max_tokens,
            correlation_id,
        )
        .await
    }

    /// Transcribe audio and emit deltas.
    pub async fn asr_transcribe_streaming<F>(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_streaming_with_correlation(
            audio_base64,
            model_id,
            language,
            None,
            on_delta,
        )
        .await
    }

    pub async fn asr_transcribe_streaming_bytes<F>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_streaming_bytes_with_correlation(
            audio_bytes,
            model_id,
            language,
            None,
            on_delta,
        )
        .await
    }

    /// Transcribe audio and emit deltas with request correlation metadata.
    pub async fn asr_transcribe_streaming_with_correlation<F>(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_streaming_with_prompt_and_correlation(
            audio_base64,
            model_id,
            language,
            None,
            correlation_id,
            on_delta,
        )
        .await
    }

    /// Transcribe audio with optional ASR initial prompt/context metadata and deltas.
    pub async fn asr_transcribe_streaming_with_prompt_and_correlation<F>(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_with_variant_streaming_and_prompt(
            variant,
            audio_base64,
            language,
            prompt,
            correlation_id,
            on_delta,
        )
        .await
    }

    pub async fn asr_transcribe_streaming_with_prompt_max_tokens_and_correlation<F>(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_with_variant_streaming_and_prompt_options(
            variant,
            audio_base64,
            language,
            prompt,
            max_tokens,
            correlation_id,
            on_delta,
        )
        .await
    }

    pub async fn asr_transcribe_streaming_bytes_with_correlation<F>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_streaming_bytes_with_prompt_and_correlation(
            audio_bytes,
            model_id,
            language,
            None,
            correlation_id,
            on_delta,
        )
        .await
    }

    pub async fn asr_transcribe_streaming_bytes_with_prompt_and_correlation<F>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_bytes_with_variant_streaming_and_prompt(
            variant,
            audio_bytes,
            language,
            prompt,
            correlation_id,
            on_delta,
        )
        .await
    }

    pub async fn asr_transcribe_streaming_bytes_with_prompt_max_tokens_and_correlation<F>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_bytes_with_variant_streaming_and_prompt_options(
            variant,
            audio_bytes,
            language,
            prompt,
            max_tokens,
            correlation_id,
            on_delta,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn asr_transcribe_streaming_bytes_with_progress_and_correlation<F, P>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        correlation_id: Option<&str>,
        on_delta: F,
        on_progress: P,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
        P: FnMut(AsrProgress) + Send + 'static,
    {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_bytes_with_variant_streaming_and_prompt_options_with_progress(
            variant,
            audio_bytes,
            language,
            None,
            None,
            correlation_id,
            on_delta,
            on_progress,
        )
        .await
    }

    pub async fn speaker_attributed_asr(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
        min_speakers: Option<usize>,
        max_speakers: Option<usize>,
    ) -> Result<SpeakerAttributedAsrResult> {
        let audio_bytes = base64_decode(audio_base64)?;
        self.speaker_attributed_asr_bytes(
            &audio_bytes,
            model_id,
            language,
            min_speakers,
            max_speakers,
        )
        .await
    }

    pub async fn speaker_attributed_asr_bytes(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        min_speakers: Option<usize>,
        max_speakers: Option<usize>,
    ) -> Result<SpeakerAttributedAsrResult> {
        let variant = resolve_speaker_attributed_asr_variant(model_id)?;
        self.observe_broker_capability_request(CapabilityKind::Asr, Some(variant), false)?;
        self.load_model(variant).await?;
        let _lease = self.acquire_model_residency_lease(variant);
        let model = self
            .model_registry
            .get_asr(variant)
            .await
            .ok_or_else(|| Error::ModelNotFound(variant.to_string()))?;

        let (samples, sample_rate) = decode_audio_bytes(audio_bytes)?;
        let duration_secs = if sample_rate > 0 {
            samples.len() as f32 / sample_rate as f32
        } else {
            0.0
        };
        let language_owned = language.map(ToOwned::to_owned);
        let task_samples = samples.clone();
        let task_model = model.clone();
        let transcription = tokio::task::spawn_blocking(move || {
            task_model.transcribe_with_granite_speech_task_and_options(
                &task_samples,
                sample_rate,
                language_owned.as_deref(),
                GraniteSpeechTask::SpeakerAttributed,
                None,
                NativeAsrGenerationOptions {
                    max_new_tokens: GRANITE_SAA_MAX_NEW_TOKENS,
                    ..NativeAsrGenerationOptions::default()
                },
            )
        })
        .await
        .map_err(|err| {
            Error::InferenceError(format!("Granite speaker attributed ASR task failed: {err}"))
        })??;

        Ok(speaker_attributed_asr_result_from_text(
            transcription.text.as_str(),
            transcription
                .language
                .or_else(|| language.map(ToOwned::to_owned)),
            duration_secs,
            min_speakers,
            max_speakers,
        ))
    }

    /// Force alignment remains a specialized operation not expressed by the
    /// generic engine output type.
    pub async fn force_align(
        &self,
        audio_base64: &str,
        reference_text: &str,
    ) -> Result<Vec<(String, u32, u32)>> {
        self.force_align_with_model_and_language(audio_base64, reference_text, None, None)
            .await
    }

    pub async fn force_align_with_model(
        &self,
        audio_base64: &str,
        reference_text: &str,
        model_id: Option<&str>,
    ) -> Result<Vec<(String, u32, u32)>> {
        self.force_align_with_model_and_language(audio_base64, reference_text, None, model_id)
            .await
    }

    pub async fn force_align_with_model_and_language(
        &self,
        audio_base64: &str,
        reference_text: &str,
        language: Option<&str>,
        model_id: Option<&str>,
    ) -> Result<Vec<(String, u32, u32)>> {
        let audio_bytes = base64_decode(audio_base64)?;
        self.force_align_bytes_with_model_and_language(
            &audio_bytes,
            reference_text,
            language,
            model_id,
        )
        .await
    }

    pub async fn force_align_bytes_with_model_and_language(
        &self,
        audio_bytes: &[u8],
        reference_text: &str,
        language: Option<&str>,
        model_id: Option<&str>,
    ) -> Result<Vec<(String, u32, u32)>> {
        let variant = resolve_forced_aligner_variant(model_id)?;
        let _runtime_request = AlignmentRuntimeRequest::from_bytes(
            variant,
            audio_bytes.to_vec(),
            reference_text,
            language.map(ToOwned::to_owned),
        )?;
        self.observe_broker_capability_request(
            CapabilityKind::ForcedAlignment,
            Some(variant),
            false,
        )?;
        self.load_model(variant).await?;
        let _lease = self.acquire_model_residency_lease(variant);

        let model = self
            .model_registry
            .get_asr(variant)
            .await
            .ok_or_else(|| Error::ModelNotFound(variant.to_string()))?;

        let (samples, sample_rate) = decode_audio_bytes(audio_bytes)?;
        model.force_align(&samples, sample_rate, reference_text, language)
    }
}

fn resolve_speaker_attributed_asr_variant(model_id: Option<&str>) -> Result<ModelVariant> {
    let variant = match model_id {
        Some(raw) => {
            parse_model_variant(raw).map_err(|err| Error::InvalidInput(err.to_string()))?
        }
        None => ModelVariant::GraniteSpeech412BPlus,
    };

    if variant != ModelVariant::GraniteSpeech412BPlus {
        return Err(Error::InvalidInput(format!(
            "Speaker attributed ASR currently requires Granite-Speech-4.1-2B-Plus, got {variant}"
        )));
    }

    Ok(variant)
}

fn speaker_attributed_asr_result_from_text(
    raw_text: &str,
    language: Option<String>,
    duration_secs: f32,
    min_speakers: Option<usize>,
    max_speakers: Option<usize>,
) -> SpeakerAttributedAsrResult {
    let parsed = parse_granite_speech_output(raw_text);
    let mut speakers = HashSet::<String>::new();
    let mut turns = Vec::new();

    for segment in &parsed.segments {
        let speaker = segment
            .speaker
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or(UNKNOWN_SAA_SPEAKER)
            .to_string();
        if speaker != UNKNOWN_SAA_SPEAKER {
            speakers.insert(speaker.clone());
        }
        turns.push(SpeakerAttributedAsrTurn {
            speaker,
            text: segment.text.clone(),
            start_secs: None,
            end_secs: None,
        });
    }

    let speaker_count = speakers.len();
    let mut warnings = Vec::new();
    if let Some(min_speakers) = min_speakers {
        if speaker_count < min_speakers {
            warnings.push(format!(
                "Granite SAA emitted {speaker_count} speaker label(s), below requested minimum {min_speakers}."
            ));
        }
    }
    if let Some(max_speakers) = max_speakers {
        if speaker_count > max_speakers {
            warnings.push(format!(
                "Granite SAA emitted {speaker_count} speaker label(s), above requested maximum {max_speakers}."
            ));
        }
    }
    if parsed.text.trim().is_empty() {
        warnings.push("Granite SAA returned an empty transcript.".to_string());
    }

    SpeakerAttributedAsrResult {
        text: parsed.text,
        language,
        duration_secs,
        speaker_turns: turns,
        speaker_count,
        status: if warnings.is_empty() {
            SpeakerAttributedAsrStatus::Ready
        } else {
            SpeakerAttributedAsrStatus::Warning
        },
        warnings,
    }
}

pub(crate) fn resolve_forced_aligner_variant(model_id: Option<&str>) -> Result<ModelVariant> {
    let variant = match model_id {
        Some(raw) => {
            parse_model_variant(raw).map_err(|err| Error::InvalidInput(err.to_string()))?
        }
        None => ModelVariant::Qwen3ForcedAligner06B,
    };

    if !variant.is_forced_aligner() {
        return Err(Error::InvalidInput(format!(
            "Model {} is not a forced aligner model",
            variant.dir_name()
        )));
    }

    Ok(variant)
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::backends::BackendPreference;
    use crate::config::EngineConfig;
    use std::sync::{Mutex, OnceLock};
    use uuid::Uuid;

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn realtime_stream_variant_resolves_only_nemotron_asr() {
        assert_eq!(
            resolve_asr_realtime_stream_variant(Some("nvidia/nemotron-3.5-asr-streaming-0.6b",)),
            Some(ModelVariant::Nemotron35AsrStreaming06B)
        );
        assert_eq!(
            resolve_asr_realtime_stream_variant(Some("Nemotron 3.5 ASR Streaming 0.6B")),
            Some(ModelVariant::Nemotron35AsrStreaming06B)
        );

        assert_eq!(resolve_asr_realtime_stream_variant(None), None);
        assert_eq!(
            resolve_asr_realtime_stream_variant(Some("Qwen3-ASR-1.7B")),
            None
        );
        assert_eq!(
            resolve_asr_realtime_stream_variant(Some("Whisper-Large-v3-Turbo")),
            None
        );
        assert_eq!(
            resolve_asr_realtime_stream_variant(Some("Parakeet-TDT-0.6B-v3")),
            None
        );
        assert_eq!(
            resolve_asr_realtime_stream_variant(Some("not-a-real-model")),
            None
        );
    }

    #[test]
    fn granite_auto_asr_budget_scales_with_audio_duration() {
        assert_eq!(granite_auto_asr_max_tokens_for_duration(0.0), 76);
        assert_eq!(granite_auto_asr_max_tokens_for_duration(3.6), 76);
        assert_eq!(granite_auto_asr_max_tokens_for_duration(27.303175), 76);
        assert_eq!(granite_auto_asr_max_tokens_for_duration(28.0), 76);
        assert_eq!(granite_auto_asr_max_tokens_for_duration(60.0), 204);
        assert_eq!(granite_auto_asr_max_tokens_for_duration(600.0), 2048);
        assert_eq!(granite_auto_asr_max_tokens_for_duration(1200.0), 2048);
    }

    #[test]
    fn granite_auto_asr_budget_reads_wav_duration_without_full_decode() {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut wav_bytes = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut wav_bytes);
            let mut writer = hound::WavWriter::new(cursor, spec).expect("writer");
            for _ in 0..960_000 {
                writer.write_sample(0i16).unwrap();
            }
            writer.finalize().unwrap();
        }

        let budget = granite_auto_asr_max_tokens(
            ModelVariant::GraniteSpeech412BPlus,
            AsrAudioInput::Bytes(&wav_bytes),
        )
        .expect("auto budget")
        .expect("granite budget");

        assert_eq!(budget, 204);
    }

    #[test]
    fn speaker_attributed_asr_result_preserves_granite_turns() {
        let result = speaker_attributed_asr_result_from_text(
            "[Speaker 1]: hello there [Speaker 2]: hi back",
            Some("English".to_string()),
            4.0,
            Some(2),
            None,
        );

        assert_eq!(result.status, SpeakerAttributedAsrStatus::Ready);
        assert_eq!(result.language.as_deref(), Some("English"));
        assert_eq!(result.speaker_count, 2);
        assert!(result.warnings.is_empty());
        assert_eq!(
            result.speaker_turns,
            vec![
                SpeakerAttributedAsrTurn {
                    speaker: "Speaker 1".to_string(),
                    text: "hello there".to_string(),
                    start_secs: None,
                    end_secs: None,
                },
                SpeakerAttributedAsrTurn {
                    speaker: "Speaker 2".to_string(),
                    text: "hi back".to_string(),
                    start_secs: None,
                    end_secs: None,
                },
            ]
        );
    }

    #[test]
    fn speaker_attributed_asr_warns_when_expected_speakers_are_missing() {
        let result = speaker_attributed_asr_result_from_text(
            "[Speaker 1]: one long turn",
            None,
            2.0,
            Some(2),
            None,
        );

        assert_eq!(result.status, SpeakerAttributedAsrStatus::Warning);
        assert_eq!(result.speaker_count, 1);
        assert_eq!(result.speaker_turns.len(), 1);
        assert!(result.warnings[0].contains("below requested minimum 2"));
    }

    #[test]
    fn speaker_attributed_asr_rejects_non_granite_models() {
        let err = resolve_speaker_attributed_asr_variant(Some("Whisper-Large-v3-Turbo"))
            .expect_err("SAA should be Granite-only");
        assert!(err.to_string().contains("Granite-Speech-4.1-2B-Plus"));
    }

    #[tokio::test]
    async fn parakeet_load_rejects_invalid_nemo_archive() {
        let _guard = env_lock().lock().expect("env lock poisoned");

        let root = std::env::temp_dir().join(format!("izwi-parakeet-runtime-{}", Uuid::new_v4()));
        let model_dir = root.join("Parakeet-TDT-0.6B-v3");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("parakeet-tdt-0.6b-v3.nemo"), b"mock-nemo").unwrap();

        let mut config = EngineConfig::default();
        config.models_dir = root.clone();
        config.backend = BackendPreference::Cpu;

        let engine = RuntimeService::new(config).unwrap();
        let err = engine
            .load_model(ModelVariant::ParakeetTdt06BV3)
            .await
            .expect_err("invalid .nemo archive should fail to load");
        let msg = err.to_string();
        assert!(
            msg.contains(".nemo")
                || msg.contains("archive")
                || msg.contains("Failed to load")
                || msg.contains("invalid")
        );

        let _ = std::fs::remove_dir_all(root);
    }
}
