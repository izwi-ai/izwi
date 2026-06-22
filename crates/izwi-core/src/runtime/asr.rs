//! ASR runtime methods routed through the unified core engine.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::catalog::{ModelFamily, parse_model_variant, resolve_asr_model_variant};
use crate::engine::{AsrProgress, AsrProgressPhase, EngineCoreRequest};
use crate::error::{Error, Result};
use crate::model::{ModelResidencyLease, ModelVariant};
use crate::models::architectures::granite_speech::asr::{
    GraniteSpeechTask, parse_granite_speech_output,
};
use crate::models::registry::{
    NativeAsrGenerationOptions, NativeAsrModel, NativeAsrRealtimeEvent, NativeAsrRealtimeState,
    NativeAsrTranscription,
};
use crate::runtime::adapters::CapabilityKind;
use crate::runtime::audio_io::{base64_decode, decode_audio_bytes, wav_duration_seconds_fast};
use crate::runtime::request::{AlignmentRuntimeRequest, AsrRuntimeRequest};
use crate::runtime::service::RuntimeService;
use crate::runtime::types::{
    AsrTranscription, SpeakerAttributedAsrResult, SpeakerAttributedAsrStatus,
    SpeakerAttributedAsrTurn,
};
use izwi_asr_toolkit::{AsrLongFormConfig, AudioChunk, plan_audio_chunks};

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
const GRANITE_SAA_TARGET_CHUNK_SECS: f32 = 240.0;
const GRANITE_SAA_HARD_MAX_CHUNK_SECS: f32 = 510.0;
const GRANITE_SAA_OVERLAP_SECS: f32 = 12.0;
const GRANITE_SAA_MIN_CHUNK_SECS: f32 = 30.0;
const GRANITE_SAA_SILENCE_SEARCH_SECS: f32 = 12.0;
const GRANITE_SAA_PREFIX_MAX_TURNS: usize = 12;
const GRANITE_SAA_PREFIX_MAX_CHARS: usize = 6_000;
const GRANITE_SAA_MIN_OVERLAP_WORDS: usize = 4;
const GRANITE_SAA_MAX_OVERLAP_WORDS: usize = 80;
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
        self.speaker_attributed_asr_bytes_with_progress(
            audio_bytes,
            model_id,
            language,
            min_speakers,
            max_speakers,
            |_| {},
        )
        .await
    }

    pub async fn speaker_attributed_asr_bytes_with_progress<P>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        min_speakers: Option<usize>,
        max_speakers: Option<usize>,
        mut on_progress: P,
    ) -> Result<SpeakerAttributedAsrResult>
    where
        P: FnMut(AsrProgress) + Send + 'static,
    {
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
        let model_limit_secs = model.max_audio_seconds_hint();
        if granite_saa_should_use_single_pass(duration_secs, model_limit_secs) {
            let language_owned = language.map(ToOwned::to_owned);
            let task_samples = samples.clone();
            let task_model = model.clone();
            let transcription = tokio::task::spawn_blocking(move || {
                granite_saa_transcribe_chunk(
                    &task_model,
                    &task_samples,
                    sample_rate,
                    language_owned.as_deref(),
                    None,
                )
            })
            .await
            .map_err(|err| {
                Error::InferenceError(format!("Granite speaker attributed ASR task failed: {err}"))
            })??;

            return Ok(speaker_attributed_asr_result_from_text_with_warnings(
                transcription.text.as_str(),
                transcription
                    .language
                    .or_else(|| language.map(ToOwned::to_owned)),
                duration_secs,
                min_speakers,
                max_speakers,
                Vec::new(),
            ));
        }

        let language_owned = language.map(ToOwned::to_owned);
        let long_form = granite_saa_long_form_transcribe(
            model,
            &samples,
            sample_rate,
            language_owned.as_deref(),
            model_limit_secs,
            &mut on_progress,
        )
        .await?;

        Ok(speaker_attributed_asr_result_from_text_with_warnings(
            long_form.text.as_str(),
            long_form.language.or(language_owned),
            duration_secs,
            min_speakers,
            max_speakers,
            long_form.warnings,
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

#[derive(Debug, Clone)]
struct GraniteSaaLongFormOutput {
    text: String,
    language: Option<String>,
    warnings: Vec<String>,
}

#[derive(Debug, Default)]
struct GraniteSaaTranscriptAssembler {
    turns: Vec<SpeakerAttributedAsrTurn>,
}

impl GraniteSaaTranscriptAssembler {
    fn push_chunk_text(&mut self, chunk_text: &str, chunk_index: usize) -> Vec<String> {
        let mut warnings = Vec::new();
        let incoming = speaker_attributed_asr_turns_from_text(chunk_text);
        if incoming.is_empty() {
            if !chunk_text.trim().is_empty() {
                warnings.push(format!(
                    "Granite SAA chunk {} returned text without speaker labels.",
                    chunk_index + 1
                ));
            }
            return warnings;
        }

        let mut speaker_aliases = HashMap::<String, String>::new();
        if let (Some(last), Some(first)) = (self.turns.last(), incoming.first()) {
            let overlap = overlap_prefix_word_count(
                last.text.as_str(),
                first.text.as_str(),
                GRANITE_SAA_MIN_OVERLAP_WORDS,
                GRANITE_SAA_MAX_OVERLAP_WORDS,
            );
            if overlap > 0
                && first.speaker != UNKNOWN_SAA_SPEAKER
                && last.speaker != UNKNOWN_SAA_SPEAKER
                && first.speaker != last.speaker
            {
                speaker_aliases.insert(first.speaker.clone(), last.speaker.clone());
                warnings.push(format!(
                    "Granite SAA chunk {} reused speaker label {} across an overlap; mapped it to {}.",
                    chunk_index + 1,
                    first.speaker,
                    last.speaker
                ));
            }
        }
        if !speaker_aliases.is_empty() {
            let mut known_global_speakers = Vec::<String>::new();
            for turn in &self.turns {
                if turn.speaker != UNKNOWN_SAA_SPEAKER
                    && !known_global_speakers.contains(&turn.speaker)
                {
                    known_global_speakers.push(turn.speaker.clone());
                }
            }

            let mut used_global_speakers = speaker_aliases
                .values()
                .cloned()
                .collect::<HashSet<String>>();
            let mut local_speakers = Vec::<String>::new();
            for turn in &incoming {
                if turn.speaker != UNKNOWN_SAA_SPEAKER && !local_speakers.contains(&turn.speaker) {
                    local_speakers.push(turn.speaker.clone());
                }
            }

            for local_speaker in local_speakers {
                if speaker_aliases.contains_key(local_speaker.as_str())
                    || !used_global_speakers.contains(local_speaker.as_str())
                {
                    continue;
                }
                if let Some(global_speaker) = known_global_speakers
                    .iter()
                    .find(|speaker| !used_global_speakers.contains(speaker.as_str()))
                    .cloned()
                {
                    speaker_aliases.insert(local_speaker.clone(), global_speaker.clone());
                    used_global_speakers.insert(global_speaker.clone());
                    warnings.push(format!(
                        "Granite SAA chunk {} remapped local speaker label {} to {} after detecting a label reset.",
                        chunk_index + 1,
                        local_speaker,
                        global_speaker
                    ));
                }
            }
        }

        for mut turn in incoming {
            if let Some(mapped) = speaker_aliases.get(turn.speaker.as_str()) {
                turn.speaker = mapped.clone();
            }
            self.push_turn(turn);
        }

        warnings
    }

    fn push_turn(&mut self, mut turn: SpeakerAttributedAsrTurn) {
        turn.text = turn.text.trim().to_string();
        if turn.text.is_empty() {
            return;
        }

        let Some(last) = self.turns.last_mut() else {
            self.turns.push(turn);
            return;
        };

        let overlap = overlap_prefix_word_count(
            last.text.as_str(),
            turn.text.as_str(),
            GRANITE_SAA_MIN_OVERLAP_WORDS,
            GRANITE_SAA_MAX_OVERLAP_WORDS,
        );
        if overlap > 0 {
            turn.text = drop_prefix_words(turn.text.as_str(), overlap);
            if turn.text.trim().is_empty() {
                return;
            }
        }

        if last.speaker == turn.speaker {
            append_with_spacing(&mut last.text, turn.text.trim());
        } else {
            self.turns.push(turn);
        }
    }

    fn text(&self) -> String {
        format_saa_turns(&self.turns)
    }

    fn prefix_text(&self) -> String {
        let mut selected = Vec::<String>::new();
        let mut selected_chars = 0usize;

        for turn in self.turns.iter().rev().take(GRANITE_SAA_PREFIX_MAX_TURNS) {
            let formatted = format_saa_turn(turn);
            let separator_chars = usize::from(!selected.is_empty());
            let formatted_chars = formatted.chars().count();
            if selected_chars + separator_chars + formatted_chars <= GRANITE_SAA_PREFIX_MAX_CHARS {
                selected.push(formatted);
                selected_chars += separator_chars + formatted_chars;
                continue;
            }

            if selected.is_empty() {
                let label_overhead = turn.speaker.chars().count() + "[]: ".chars().count();
                let text_budget = GRANITE_SAA_PREFIX_MAX_CHARS.saturating_sub(label_overhead);
                let suffix = recent_word_suffix(turn.text.as_str(), text_budget);
                if !suffix.is_empty() {
                    selected.push(format!("[{}]: {}", turn.speaker, suffix));
                }
            }
            break;
        }

        selected.reverse();
        selected.join(" ")
    }
}

fn speaker_attributed_asr_turns_from_text(raw_text: &str) -> Vec<SpeakerAttributedAsrTurn> {
    let parsed = parse_granite_speech_output(raw_text);
    if parsed.segments.is_empty() {
        let text = parsed.text.trim();
        if text.is_empty() {
            return Vec::new();
        }
        return vec![SpeakerAttributedAsrTurn {
            speaker: UNKNOWN_SAA_SPEAKER.to_string(),
            text: text.to_string(),
            start_secs: None,
            end_secs: None,
        }];
    }

    parsed
        .segments
        .into_iter()
        .map(|segment| SpeakerAttributedAsrTurn {
            speaker: segment
                .speaker
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .unwrap_or(UNKNOWN_SAA_SPEAKER)
                .to_string(),
            text: segment.text.trim().to_string(),
            start_secs: None,
            end_secs: None,
        })
        .filter(|turn| !turn.text.is_empty())
        .fold(Vec::<SpeakerAttributedAsrTurn>::new(), |mut turns, turn| {
            if let Some(last) = turns.last_mut() {
                if last.speaker == turn.speaker {
                    append_with_spacing(&mut last.text, turn.text.as_str());
                    return turns;
                }
            }
            turns.push(turn);
            turns
        })
}

fn granite_saa_should_use_single_pass(duration_secs: f32, model_limit_secs: Option<f32>) -> bool {
    let limit = model_limit_secs
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(GRANITE_SAA_HARD_MAX_CHUNK_SECS);
    duration_secs.is_finite() && duration_secs <= limit
}

fn granite_saa_long_form_config(model_limit_secs: Option<f32>) -> AsrLongFormConfig {
    let mut cfg = AsrLongFormConfig::default();
    cfg.target_chunk_secs = GRANITE_SAA_TARGET_CHUNK_SECS;
    cfg.hard_max_chunk_secs = GRANITE_SAA_HARD_MAX_CHUNK_SECS;
    cfg.overlap_secs = GRANITE_SAA_OVERLAP_SECS;
    cfg.min_chunk_secs = GRANITE_SAA_MIN_CHUNK_SECS;
    cfg.silence_search_secs = GRANITE_SAA_SILENCE_SEARCH_SECS;
    cfg.min_word_overlap = GRANITE_SAA_MIN_OVERLAP_WORDS;
    cfg.max_word_overlap = GRANITE_SAA_MAX_OVERLAP_WORDS;
    cfg.min_context_replay_words = GRANITE_SAA_MIN_OVERLAP_WORDS;
    cfg.max_context_replay_words = GRANITE_SAA_MAX_OVERLAP_WORDS;

    if let Some(limit) = model_limit_secs.filter(|value| value.is_finite() && *value > 0.0) {
        cfg.hard_max_chunk_secs = cfg.hard_max_chunk_secs.min(limit * 0.95);
    }
    if let Some(value) = env_positive_f32("IZWI_GRANITE_SAA_CHUNK_TARGET_SECS") {
        cfg.target_chunk_secs = value;
    }
    if let Some(value) = env_positive_f32("IZWI_GRANITE_SAA_CHUNK_MAX_SECS") {
        cfg.hard_max_chunk_secs = value;
    }
    if let Some(value) = env_positive_f32("IZWI_GRANITE_SAA_CHUNK_OVERLAP_SECS") {
        cfg.overlap_secs = value;
    }
    if let Some(value) = env_positive_f32("IZWI_GRANITE_SAA_CHUNK_MIN_SECS") {
        cfg.min_chunk_secs = value;
    }
    if let Some(value) = env_positive_f32("IZWI_GRANITE_SAA_CHUNK_SILENCE_SEARCH_SECS") {
        cfg.silence_search_secs = value;
    }

    if let Some(limit) = model_limit_secs.filter(|value| value.is_finite() && *value > 0.0) {
        cfg.hard_max_chunk_secs = cfg.hard_max_chunk_secs.min(limit * 0.95);
    }
    cfg.hard_max_chunk_secs = cfg
        .hard_max_chunk_secs
        .max(cfg.min_chunk_secs.max(1.0))
        .min(GRANITE_SAA_HARD_MAX_CHUNK_SECS);
    cfg.target_chunk_secs = cfg
        .target_chunk_secs
        .max(cfg.min_chunk_secs.max(1.0))
        .min(cfg.hard_max_chunk_secs);
    cfg.overlap_secs = cfg.overlap_secs.clamp(0.0, cfg.target_chunk_secs * 0.45);
    cfg
}

fn granite_saa_chunk_plan(
    samples: &[f32],
    sample_rate: u32,
    model_limit_secs: Option<f32>,
) -> Vec<AudioChunk> {
    let cfg = granite_saa_long_form_config(model_limit_secs);
    plan_audio_chunks(samples, sample_rate, &cfg, Some(cfg.hard_max_chunk_secs))
}

async fn granite_saa_long_form_transcribe<P>(
    model: Arc<NativeAsrModel>,
    samples: &[f32],
    sample_rate: u32,
    language: Option<&str>,
    model_limit_secs: Option<f32>,
    on_progress: &mut P,
) -> Result<GraniteSaaLongFormOutput>
where
    P: FnMut(AsrProgress) + Send + 'static,
{
    let chunks = granite_saa_chunk_plan(samples, sample_rate, model_limit_secs);
    if chunks.is_empty() {
        return Err(Error::InvalidInput(
            "Granite SAA chunk planner produced no chunks".to_string(),
        ));
    }

    on_progress(granite_saa_processing_progress(&chunks, sample_rate));

    let mut assembler = GraniteSaaTranscriptAssembler::default();
    let mut language_out = language.map(ToOwned::to_owned);
    let mut warnings = vec![format!(
        "Granite SAA processed long audio in {} chunks; speaker label continuity across chunks is best-effort.",
        chunks.len()
    )];

    for (idx, chunk) in chunks.iter().enumerate() {
        if chunk.end_sample <= chunk.start_sample || chunk.end_sample > samples.len() {
            warnings.push(format!(
                "Granite SAA skipped invalid chunk {} with sample range {}..{}.",
                idx + 1,
                chunk.start_sample,
                chunk.end_sample
            ));
            continue;
        }

        on_progress(granite_saa_chunk_progress(
            AsrProgressPhase::ChunkStarted,
            idx,
            chunk,
            &chunks,
            sample_rate,
        ));

        let chunk_audio = samples[chunk.start_sample..chunk.end_sample].to_vec();
        let task_model = model.clone();
        let language_owned = language.map(ToOwned::to_owned);
        let prefix_text = assembler.prefix_text();
        let prefix_text = (!prefix_text.trim().is_empty()).then_some(prefix_text);
        let transcription = tokio::task::spawn_blocking(move || {
            granite_saa_transcribe_chunk(
                &task_model,
                &chunk_audio,
                sample_rate,
                language_owned.as_deref(),
                prefix_text.as_deref(),
            )
        })
        .await
        .map_err(|err| {
            Error::InferenceError(format!(
                "Granite speaker attributed ASR chunk {} failed: {err}",
                idx + 1
            ))
        })??;

        if language_out.is_none() {
            language_out = transcription.language.clone();
        }
        if transcription.text.trim().is_empty() {
            warnings.push(format!(
                "Granite SAA chunk {} returned empty text.",
                idx + 1
            ));
        }
        warnings.extend(assembler.push_chunk_text(transcription.text.as_str(), idx));

        on_progress(granite_saa_chunk_progress(
            AsrProgressPhase::ChunkFinished,
            idx,
            chunk,
            &chunks,
            sample_rate,
        ));
    }

    on_progress(granite_saa_complete_progress(&chunks, sample_rate));

    Ok(GraniteSaaLongFormOutput {
        text: assembler.text(),
        language: language_out,
        warnings,
    })
}

fn granite_saa_transcribe_chunk(
    model: &NativeAsrModel,
    audio: &[f32],
    sample_rate: u32,
    language: Option<&str>,
    prefix_text: Option<&str>,
) -> Result<NativeAsrTranscription> {
    model.transcribe_with_granite_speech_task_and_options(
        audio,
        sample_rate,
        language,
        GraniteSpeechTask::SpeakerAttributed,
        prefix_text,
        NativeAsrGenerationOptions {
            max_new_tokens: GRANITE_SAA_MAX_NEW_TOKENS,
            ..NativeAsrGenerationOptions::default()
        },
    )
}

fn granite_saa_processing_progress(chunks: &[AudioChunk], sample_rate: u32) -> AsrProgress {
    let total_audio_secs = chunks
        .last()
        .map(|chunk| samples_to_seconds_f64(chunk.end_sample, sample_rate));
    AsrProgress {
        phase: AsrProgressPhase::Processing,
        current_chunk: None,
        total_chunks: Some(chunks.len()),
        processed_audio_secs: Some(0.0),
        total_audio_secs,
        percent: Some(0.0),
    }
}

fn granite_saa_complete_progress(chunks: &[AudioChunk], sample_rate: u32) -> AsrProgress {
    let total_audio_secs = chunks
        .last()
        .map(|chunk| samples_to_seconds_f64(chunk.end_sample, sample_rate));
    AsrProgress {
        phase: AsrProgressPhase::Complete,
        current_chunk: Some(chunks.len()),
        total_chunks: Some(chunks.len()),
        processed_audio_secs: total_audio_secs,
        total_audio_secs,
        percent: Some(100.0),
    }
}

fn granite_saa_chunk_progress(
    phase: AsrProgressPhase,
    index: usize,
    chunk: &AudioChunk,
    chunks: &[AudioChunk],
    sample_rate: u32,
) -> AsrProgress {
    let total_audio_secs = chunks
        .last()
        .map(|last| samples_to_seconds_f64(last.end_sample, sample_rate));
    let processed_audio_secs = match phase {
        AsrProgressPhase::ChunkStarted => samples_to_seconds_f64(chunk.start_sample, sample_rate),
        AsrProgressPhase::ChunkFinished => samples_to_seconds_f64(chunk.end_sample, sample_rate),
        AsrProgressPhase::Processing => 0.0,
        AsrProgressPhase::Aligning | AsrProgressPhase::Complete => {
            total_audio_secs.unwrap_or_default()
        }
    };
    let percent = total_audio_secs
        .filter(|total| *total > 0.0)
        .map(|total| ((processed_audio_secs / total) * 100.0).clamp(0.0, 100.0));

    AsrProgress {
        phase,
        current_chunk: Some(index + 1),
        total_chunks: Some(chunks.len()),
        processed_audio_secs: Some(processed_audio_secs),
        total_audio_secs,
        percent,
    }
}

fn samples_to_seconds_f64(samples: usize, sample_rate: u32) -> f64 {
    if sample_rate == 0 {
        0.0
    } else {
        samples as f64 / sample_rate as f64
    }
}

fn env_positive_f32(key: &str) -> Option<f32> {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.trim().parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
}

fn speaker_attributed_asr_result_from_text(
    raw_text: &str,
    language: Option<String>,
    duration_secs: f32,
    min_speakers: Option<usize>,
    max_speakers: Option<usize>,
) -> SpeakerAttributedAsrResult {
    speaker_attributed_asr_result_from_text_with_warnings(
        raw_text,
        language,
        duration_secs,
        min_speakers,
        max_speakers,
        Vec::new(),
    )
}

fn speaker_attributed_asr_result_from_text_with_warnings(
    raw_text: &str,
    language: Option<String>,
    duration_secs: f32,
    min_speakers: Option<usize>,
    max_speakers: Option<usize>,
    mut warnings: Vec<String>,
) -> SpeakerAttributedAsrResult {
    let parsed = parse_granite_speech_output(raw_text);
    let mut speakers = HashSet::<String>::new();
    let mut turns = speaker_attributed_asr_turns_from_text(raw_text);

    for turn in &turns {
        let speaker = turn.speaker.trim();
        if speaker != UNKNOWN_SAA_SPEAKER {
            speakers.insert(speaker.to_string());
        }
    }
    if turns.is_empty() && !parsed.text.trim().is_empty() {
        turns.push(SpeakerAttributedAsrTurn {
            speaker: UNKNOWN_SAA_SPEAKER.to_string(),
            text: parsed.text.trim().to_string(),
            start_secs: None,
            end_secs: None,
        });
    }

    let speaker_count = speakers.len();
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

fn format_saa_turns(turns: &[SpeakerAttributedAsrTurn]) -> String {
    turns
        .iter()
        .map(format_saa_turn)
        .collect::<Vec<_>>()
        .join(" ")
}

fn format_saa_turn(turn: &SpeakerAttributedAsrTurn) -> String {
    format!("[{}]: {}", turn.speaker, turn.text.trim())
}

fn append_with_spacing(target: &mut String, text: &str) {
    let text = text.trim();
    if text.is_empty() {
        return;
    }
    if !target.trim().is_empty() && !target.ends_with(char::is_whitespace) {
        target.push(' ');
    }
    target.push_str(text);
}

fn overlap_prefix_word_count(
    existing: &str,
    incoming: &str,
    min_words: usize,
    max_words: usize,
) -> usize {
    let existing_words = normalized_overlap_words(existing);
    let incoming_words = normalized_overlap_words(incoming);
    let max_words = max_words
        .min(existing_words.len())
        .min(incoming_words.len());
    if max_words < min_words {
        return 0;
    }

    for count in (min_words..=max_words).rev() {
        if existing_words[existing_words.len() - count..] == incoming_words[..count] {
            return count;
        }
    }
    0
}

fn normalized_overlap_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter_map(normalize_overlap_word)
        .collect()
}

fn normalize_overlap_word(word: &str) -> Option<String> {
    let normalized = word
        .chars()
        .filter(|ch| ch.is_alphanumeric() || *ch == '\'')
        .flat_map(char::to_lowercase)
        .collect::<String>();
    (!normalized.is_empty()).then_some(normalized)
}

fn drop_prefix_words(text: &str, words_to_drop: usize) -> String {
    text.split_whitespace()
        .skip(words_to_drop)
        .collect::<Vec<_>>()
        .join(" ")
}

fn recent_word_suffix(text: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }

    let mut selected = Vec::<&str>::new();
    let mut selected_chars = 0usize;
    for word in text.split_whitespace().rev() {
        let separator_chars = usize::from(!selected.is_empty());
        let word_chars = word.chars().count();
        if selected_chars + separator_chars + word_chars > max_chars {
            break;
        }
        selected.push(word);
        selected_chars += separator_chars + word_chars;
    }
    selected.reverse();
    selected.join(" ")
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
    fn granite_saa_chunk_plan_keeps_long_audio_under_model_limit() {
        let sample_rate = 10;
        let samples = vec![0.0f32; 15_565];
        let chunks = granite_saa_chunk_plan(&samples, sample_rate, Some(540.0));

        assert!(chunks.len() > 1);
        assert_eq!(chunks.first().unwrap().start_sample, 0);
        assert_eq!(chunks.last().unwrap().end_sample, samples.len());
        assert!(chunks.iter().all(|chunk| chunk.len_samples()
            <= (GRANITE_SAA_HARD_MAX_CHUNK_SECS * sample_rate as f32) as usize));
    }

    #[test]
    fn granite_saa_assembler_dedupes_overlap_and_preserves_turns() {
        let mut assembler = GraniteSaaTranscriptAssembler::default();
        assert!(
            assembler
                .push_chunk_text(
                    "[Speaker 1]: hello there [Speaker 2]: hi back from me now",
                    0,
                )
                .is_empty()
        );
        assert!(
            assembler
                .push_chunk_text(
                    "[Speaker 2]: hi back from me now and more [Speaker 1]: ok",
                    1,
                )
                .is_empty()
        );

        let text = assembler.text();
        assert_eq!(text.matches("hi back from me now").count(), 1);
        assert_eq!(
            text,
            "[Speaker 1]: hello there [Speaker 2]: hi back from me now and more [Speaker 1]: ok"
        );
    }

    #[test]
    fn granite_saa_assembler_maps_reset_label_when_overlap_proves_continuity() {
        let mut assembler = GraniteSaaTranscriptAssembler::default();
        assembler.push_chunk_text(
            "[Speaker 1]: first person [Speaker 2]: boundary overlap words now",
            0,
        );
        let warnings =
            assembler.push_chunk_text("[Speaker 1]: boundary overlap words now continuing", 1);

        assert!(warnings.iter().any(|warning| warning.contains("mapped it")));
        assert_eq!(
            assembler.text(),
            "[Speaker 1]: first person [Speaker 2]: boundary overlap words now continuing"
        );
    }

    #[test]
    fn granite_saa_prefix_text_is_bounded_on_turn_boundaries() {
        let mut assembler = GraniteSaaTranscriptAssembler::default();
        for idx in 0..32 {
            let speaker = if idx % 2 == 0 { 1 } else { 2 };
            assembler.push_chunk_text(format!("[Speaker {speaker}]: turn {idx}").as_str(), idx);
        }

        let prefix = assembler.prefix_text();
        assert!(prefix.chars().count() <= GRANITE_SAA_PREFIX_MAX_CHARS);
        assert!(prefix.contains("[Speaker 1]:"));
        assert!(!prefix.contains("turn 0"));
    }

    #[test]
    fn granite_saa_progress_events_report_chunks_and_percent() {
        let chunks = vec![
            AudioChunk {
                start_sample: 0,
                end_sample: 100,
            },
            AudioChunk {
                start_sample: 90,
                end_sample: 200,
            },
        ];

        let processing = granite_saa_processing_progress(&chunks, 100);
        assert_eq!(processing.phase, AsrProgressPhase::Processing);
        assert_eq!(processing.total_chunks, Some(2));
        assert_eq!(processing.percent, Some(0.0));

        let finished = granite_saa_chunk_progress(
            AsrProgressPhase::ChunkFinished,
            0,
            &chunks[0],
            &chunks,
            100,
        );
        assert_eq!(finished.current_chunk, Some(1));
        assert_eq!(finished.total_chunks, Some(2));
        assert_eq!(finished.processed_audio_secs, Some(1.0));
        assert_eq!(finished.total_audio_secs, Some(2.0));
        assert_eq!(finished.percent, Some(50.0));
    }

    #[test]
    fn speaker_attributed_asr_result_includes_long_form_warnings() {
        let result = speaker_attributed_asr_result_from_text_with_warnings(
            "[Speaker 1]: hello [Speaker 2]: hi",
            None,
            2.0,
            None,
            None,
            vec!["Granite SAA processed long audio in 2 chunks.".to_string()],
        );

        assert_eq!(result.status, SpeakerAttributedAsrStatus::Warning);
        assert_eq!(result.speaker_count, 2);
        assert_eq!(result.warnings.len(), 1);
        assert!(result.warnings[0].contains("2 chunks"));
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
