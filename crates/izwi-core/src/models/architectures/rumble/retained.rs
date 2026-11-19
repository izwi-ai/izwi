//! Rumble's retained route: a request of up to 30 s is prepared once (its
//! audio encoded into a prompt artifact), prefilled onto retained decoder
//! pages, and decoded beside the other rows, one batched step per token.
//! Longer audio, and a request whose prompt and budget exceed
//! [`RUMBLE_SEQUENCE_TOKENS`], run as atomic invocations.
//!
//! Every row, a lone one included, decodes through the trunk's batched step,
//! so a request takes the same code path alone as beside others.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use candle_core::{DType, Tensor, D};

use crate::backends::{BackendKind, DeviceKind};
use crate::error::{Error, Result};
use crate::models::shared::attention::physical::PhysicalPagedKvCache;
use crate::tokenizer::IncrementalDecoder;

use super::audio_tower::{get_cnn_output_lengths, RUMBLE_AUDIO_CNN_SLICE_BUDGET_BYTES};
use super::frontend::{RumbleAudioFeatures, RUMBLE_MAX_AUDIO_SECONDS, RUMBLE_SAMPLE_RATE};
use super::model::{
    last_row_argmax, RumbleAsrModel, RumbleGenerationOptions, RumbleRequestOptions,
    RUMBLE_SEQUENCE_TOKENS,
};
use super::transcript::split_rumble_output;

static NEXT_DECODE_STATE_ID: AtomicU64 = AtomicU64::new(1);

/// The retained route's switch for Rumble (on by default on CUDA only).
pub(crate) const RUMBLE_RETAINED_DECODE_ENV: &str = "IZWI_RUMBLE_RETAINED_DECODE";

/// Whether a Rumble load on `backend` offers the retained sequence route.
/// Off, every request runs as atomic invocations and the load fits no
/// retained decoder state.
///
/// The route pays about 0.1 s a request for its extra dispatches and earns
/// it back by decoding concurrent rows in one step, so it is on by default
/// where rows batch wide (CUDA) and off on CPU and Metal, whose automatic
/// width is a few rows: there a single request is slower on it, and a
/// concurrent load on the CPU gains nothing. The switch overrides the default on every
/// backend. It is read once per process: the loaded adapter and the model
/// it seals both ask with the load's backend, so their routes agree.
pub(crate) fn rumble_retained_decode_enabled(backend: BackendKind) -> bool {
    static CONFIGURED: crate::env::ReadOnce<Option<bool>> =
        crate::env::ReadOnce::new(|| crate::env::flag(RUMBLE_RETAINED_DECODE_ENV));
    CONFIGURED
        .get()
        .unwrap_or(matches!(backend, BackendKind::Cuda))
}

/// A request's prompt with its audio spliced in, ready to prefill.
#[derive(Debug, Clone)]
pub(crate) struct RumblePreparedPromptArtifact {
    model_identity: u64,
    /// `[1, prompt tokens, hidden]` in the decoder's dtype.
    embeddings: Tensor,
    prompt_tokens: usize,
    audio_tokens: usize,
    /// The language the assistant turn was prefilled with.
    forced_language: Option<String>,
    /// The transcript style the system turn asked for.
    style: super::prompt::RumbleTranscriptStyle,
}

impl RumblePreparedPromptArtifact {
    /// The device holding the prepared prompt embeddings.
    pub(crate) fn device_location(&self) -> candle_core::DeviceLocation {
        self.embeddings.device().location()
    }

    pub(crate) fn prompt_tokens(&self) -> usize {
        self.prompt_tokens
    }

    pub(crate) fn audio_tokens(&self) -> usize {
        self.audio_tokens
    }

    pub(crate) fn resident_tensor_bytes(&self) -> Result<u64> {
        to_u64(self.embeddings.elem_count(), "artifact size")?
            .checked_mul(to_u64(
                self.embeddings.dtype().size_in_bytes(),
                "artifact size",
            )?)
            .ok_or_else(|| overflow("artifact size"))
    }

    pub(crate) fn resident_host_bytes(&self) -> u64 {
        self.forced_language
            .as_ref()
            .map_or(0, |language| language.len() as u64)
    }
}

/// What one request's retained preparation will hold and use, known before
/// its audio is encoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RumblePreparedGeometry {
    pub(crate) audio_samples: usize,
    /// Mel frames the tower is handed (padded to a whole conv chunk).
    pub(crate) mel_frames: usize,
    pub(crate) mel_bins: usize,
    pub(crate) prompt_tokens: usize,
    pub(crate) audio_tokens: usize,
    pub(crate) embedding_elements: u64,
    pub(crate) preparation_workspace_bytes: u64,
    pub(crate) retained_device_bytes: u64,
    /// The forced language's bytes the artifact keeps.
    pub(crate) language_bytes: u64,
}

impl RumblePreparedGeometry {
    /// The row's preparation work: its audio tokens, and every element it
    /// materializes (the preparation is exact, with no padding).
    pub(crate) fn work_cost(self) -> Result<crate::engine::WorkCost> {
        Ok(crate::engine::WorkCost::new(
            self.audio_tokens as u64,
            self.materialized_tensor_elements()?,
            self.preparation_workspace_bytes,
        ))
    }

    /// Elements the row materializes: its features and its prompt.
    pub(crate) fn materialized_tensor_elements(self) -> Result<u64> {
        to_u64(self.mel_frames, "feature geometry")?
            .checked_mul(to_u64(self.mel_bins, "feature geometry")?)
            .and_then(|features| features.checked_add(self.embedding_elements))
            .ok_or_else(|| overflow("materialized elements"))
    }
}

/// The preparation ceilings a load seals into its adapter: one request of
/// the longest audio and prompt the retained route takes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RumbleAsrPreparationStageSeal {
    pub(crate) backend: BackendKind,
    pub(crate) dtype: String,
    pub(crate) max_work_units: u64,
    pub(crate) max_materialized_tensor_elements_per_row: u64,
    pub(crate) max_workspace_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RumbleDecodeStep {
    pub(crate) delta: String,
    pub(crate) text: String,
    pub(crate) tokens_generated: usize,
    pub(crate) finished: bool,
}

/// One retained row: its prepared prompt, its decoder pages for the current
/// scheduler quantum, and its text so far.
pub(crate) struct RumbleDecodeState {
    cache: PhysicalPagedKvCache,
    artifact: Arc<RumblePreparedPromptArtifact>,
    prefill_progress: usize,
    /// The token the next decode step appends: the greedy pick of the last
    /// prefill span or decode step.
    pending_token: Option<u32>,
    pos: usize,
    generated_ids: Vec<u32>,
    incremental_decoder: IncrementalDecoder,
    /// Whether the transcript has begun: at once with a forced language,
    /// else after the model's `<asr_text>`.
    in_transcript: bool,
    rendered: String,
    published_len: usize,
    stop_holdback_bytes: usize,
    stop_tokens: Vec<u32>,
    stop_sequences: Vec<String>,
    max_new_tokens: usize,
    finished: bool,
    /// The final transcript and language, once finished.
    final_text: Option<String>,
    language: Option<String>,
    stop_reason: &'static str,
    stop_token: Option<u32>,
    state_id: u64,
    next_quantum_nonce: u64,
    active_quantum: Option<u64>,
    managed_completions_drained: bool,
}

struct RumbleDecodeCheckpointPayload {
    cache: PhysicalPagedKvCache,
    prefill_progress: usize,
    pending_token: Option<u32>,
    pos: usize,
    generated_ids: Vec<u32>,
    incremental_decoder: IncrementalDecoder,
    in_transcript: bool,
    rendered: String,
    published_len: usize,
    finished: bool,
    final_text: Option<String>,
    language: Option<String>,
    stop_reason: &'static str,
    stop_token: Option<u32>,
    managed_completions_drained: bool,
}

/// A row's state before a scheduler quantum, restored if the quantum rolls
/// back.
pub(crate) struct RumbleDecodeCheckpoint {
    state_id: u64,
    quantum_nonce: u64,
    payload: Option<RumbleDecodeCheckpointPayload>,
}

impl RumbleDecodeState {
    pub(crate) const fn uses_managed_kv(&self) -> bool {
        true
    }

    pub(crate) fn prefill_progress(&self) -> usize {
        self.prefill_progress
    }

    pub(crate) fn prefill_token_count(&self) -> usize {
        self.artifact.prompt_tokens
    }

    pub(crate) fn sequence_position(&self) -> usize {
        self.pos
    }

    /// The transcript's language: the forced one, or the one the model
    /// named once the row has finished.
    pub(crate) fn language(&self) -> Option<&str> {
        self.language
            .as_deref()
            .or(self.artifact.forced_language.as_deref())
    }

    pub(crate) fn finished(&self) -> bool {
        self.finished
    }

    #[cfg(test)]
    pub(crate) fn generated_ids(&self) -> &[u32] {
        &self.generated_ids
    }

    #[cfg(test)]
    pub(crate) fn stop_reason(&self) -> &'static str {
        self.stop_reason
    }

    /// What a finished row reports beside its text.
    pub(crate) fn retained_terminal_diagnostics(&self) -> serde_json::Value {
        serde_json::json!({
            "family": "rumble_asr",
            "route": "retained",
            "language": self.language(),
            "forced_language": self.artifact.forced_language,
            "audio_tokens": self.artifact.audio_tokens,
            "prompt_tokens": self.artifact.prompt_tokens,
            "style": {
                "digits": self.artifact.style.digits,
                "fillers": self.artifact.style.fillers,
                "verbatim": self.artifact.style.verbatim,
            },
            "generated_tokens": self.generated_ids.len(),
            "max_new_tokens": self.max_new_tokens,
            "stop_reason": self.stop_reason,
            "stop_token": self.stop_token,
        })
    }

    pub(crate) fn take_managed_write_completions(
        &mut self,
    ) -> Vec<Arc<crate::backends::kv::KvWriteBatchCompletion>> {
        let completions = self.cache.take_completed_writes();
        self.managed_completions_drained = true;
        completions
    }

    /// Swap in the quantum's cache view and checkpoint everything the
    /// quantum may change.
    pub(crate) fn begin_managed_quantum(
        &mut self,
        cache: PhysicalPagedKvCache,
    ) -> Result<RumbleDecodeCheckpoint> {
        if self.active_quantum.is_some() {
            return Err(Error::InferenceError(
                "a Rumble managed quantum is already active".into(),
            ));
        }
        if !self.managed_completions_drained {
            return Err(Error::InferenceError(
                "Rumble managed KV completions must be drained before the next quantum".into(),
            ));
        }
        if self.cache.sequence_authority() != cache.sequence_authority() {
            return Err(Error::InferenceError(
                "a Rumble session cannot switch managed KV authority".into(),
            ));
        }
        if cache.context_len() != self.pos {
            return Err(Error::InferenceError(format!(
                "managed Rumble reservation starts at {}, but state is at {}",
                cache.context_len(),
                self.pos
            )));
        }
        let quantum_nonce = self.next_quantum_nonce;
        self.next_quantum_nonce = self
            .next_quantum_nonce
            .checked_add(1)
            .ok_or_else(|| Error::InferenceError("Rumble quantum nonce overflow".into()))?;
        self.active_quantum = Some(quantum_nonce);
        Ok(RumbleDecodeCheckpoint {
            state_id: self.state_id,
            quantum_nonce,
            payload: Some(RumbleDecodeCheckpointPayload {
                cache: std::mem::replace(&mut self.cache, cache),
                prefill_progress: self.prefill_progress,
                pending_token: self.pending_token,
                pos: self.pos,
                generated_ids: self.generated_ids.clone(),
                incremental_decoder: self.incremental_decoder.clone(),
                in_transcript: self.in_transcript,
                rendered: self.rendered.clone(),
                published_len: self.published_len,
                finished: self.finished,
                final_text: self.final_text.clone(),
                language: self.language.clone(),
                stop_reason: self.stop_reason,
                stop_token: self.stop_token,
                managed_completions_drained: self.managed_completions_drained,
            }),
        })
    }

    pub(crate) fn commit_managed_quantum(
        &mut self,
        checkpoint: &mut RumbleDecodeCheckpoint,
    ) -> Result<()> {
        self.validate_checkpoint(checkpoint)?;
        checkpoint.payload.take().ok_or_else(|| {
            Error::InferenceError("Rumble checkpoint was already consumed".into())
        })?;
        self.active_quantum = None;
        Ok(())
    }

    pub(crate) fn rollback_managed_quantum(
        &mut self,
        checkpoint: &mut RumbleDecodeCheckpoint,
    ) -> Result<()> {
        self.validate_checkpoint(checkpoint)?;
        let payload = checkpoint.payload.take().ok_or_else(|| {
            Error::InferenceError("Rumble checkpoint was already consumed".into())
        })?;
        self.cache = payload.cache;
        self.prefill_progress = payload.prefill_progress;
        self.pending_token = payload.pending_token;
        self.pos = payload.pos;
        self.generated_ids = payload.generated_ids;
        self.incremental_decoder = payload.incremental_decoder;
        self.in_transcript = payload.in_transcript;
        self.rendered = payload.rendered;
        self.published_len = payload.published_len;
        self.finished = payload.finished;
        self.final_text = payload.final_text;
        self.language = payload.language;
        self.stop_reason = payload.stop_reason;
        self.stop_token = payload.stop_token;
        self.managed_completions_drained = payload.managed_completions_drained;
        self.active_quantum = None;
        Ok(())
    }

    fn validate_checkpoint(&self, checkpoint: &RumbleDecodeCheckpoint) -> Result<()> {
        if checkpoint.state_id != self.state_id
            || self.active_quantum != Some(checkpoint.quantum_nonce)
            || checkpoint.payload.is_none()
        {
            return Err(Error::InferenceError(
                "Rumble checkpoint is foreign, stale, or out of order".into(),
            ));
        }
        Ok(())
    }

    fn step(&self, delta: String) -> RumbleDecodeStep {
        RumbleDecodeStep {
            delta,
            // The final text is the atomic route's (split on the marker
            // token and trimmed); deltas stream raw.
            text: match (&self.final_text, self.finished) {
                (Some(text), true) => text.clone(),
                _ => self.rendered.trim().to_string(),
            },
            tokens_generated: self.generated_ids.len(),
            finished: self.finished,
        }
    }

    /// The text not yet published, short of a possible stop sequence's
    /// prefix (`stop_holdback_bytes`) unless `flush`.
    fn publish_stable_text(&mut self, flush: bool) -> Result<String> {
        if self.rendered.len() < self.published_len {
            return Err(Error::InferenceError(
                "Rumble decoded text rewrote an already published prefix".into(),
            ));
        }
        let mut end = if flush {
            self.rendered.len()
        } else {
            self.rendered.len().saturating_sub(self.stop_holdback_bytes)
        };
        while end > self.published_len && !self.rendered.is_char_boundary(end) {
            end -= 1;
        }
        end = end.max(self.published_len);
        let delta = self.rendered[self.published_len..end].to_string();
        self.published_len = end;
        Ok(delta)
    }
}

fn truncate_at_stop_sequence(text: &mut String, stop_sequences: &[String]) -> bool {
    for stop in stop_sequences.iter().filter(|stop| !stop.is_empty()) {
        if let Some(index) = text.find(stop.as_str()) {
            text.truncate(index);
            return true;
        }
    }
    false
}

fn overflow(what: &str) -> Error {
    Error::Overloaded(format!("Rumble {what} overflow"))
}

fn to_u64(value: usize, what: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| overflow(what))
}

impl RumbleAsrModel {
    /// Whether a request of up to 30 s resumes prefill on the retained route
    /// ([`rumble_retained_decode_enabled`] for the load's backend).
    pub(crate) fn supports_resumable_prefill(&self) -> bool {
        self.retained_decode()
    }

    /// The retained route's decode steps: the same switch, so the two never
    /// disagree.
    pub(crate) fn supports_incremental_decode(&self) -> bool {
        self.retained_decode()
    }

    /// Retained rows decode together, one batched step per token.
    pub(crate) fn supports_continuous_decode_batch(&self) -> bool {
        self.retained_decode()
    }

    /// Whether a request of `prompt_tokens` and a `max_new_tokens` budget
    /// fits one retained sequence; a longer one runs as an atomic
    /// invocation.
    pub(crate) fn retained_sequence_fits(prompt_tokens: usize, max_new_tokens: usize) -> bool {
        prompt_tokens
            .checked_add(max_new_tokens.max(1))
            .is_some_and(|tokens| tokens <= RUMBLE_SEQUENCE_TOKENS)
    }

    /// Bytes one row of a continuous decode step uses beside its retained
    /// state: its hidden row, Q/K/V row and MLP row (priced at F32) and its
    /// F32 logits row over the vocabulary.
    pub(crate) fn continuous_decode_workspace_per_row_bytes(&self) -> Result<u64> {
        let text = &self.config().text_config;
        let head_dim = self.config().decoder_head_dim();
        let elements = [
            text.hidden_size,
            text.num_attention_heads * head_dim,
            2 * text.num_key_value_heads * head_dim,
            2 * text.intermediate_size,
            text.vocab_size,
        ]
        .into_iter()
        .try_fold(0usize, |total, part| total.checked_add(part))
        .ok_or_else(|| overflow("decode workspace"))?;
        to_u64(elements, "decode workspace")?
            .checked_mul(std::mem::size_of::<f32>() as u64)
            .ok_or_else(|| overflow("decode workspace"))
    }

    /// One request's audio features, computed on the host, and the retained
    /// geometry they give.
    pub(crate) fn retained_preparation_features(
        &self,
        audio: &[f32],
        sample_rate: u32,
        request: &RumbleRequestOptions,
    ) -> Result<(RumbleAudioFeatures, RumblePreparedGeometry)> {
        let features = self.prepare_features(audio, sample_rate)?;
        let geometry =
            self.retained_preparation_geometry_for_features(&features, audio.len(), request)?;
        Ok((features, geometry))
    }

    /// The geometry `audio` and `request` will prepare to, from the host
    /// features and the built prompt, before any encoder work.
    pub(crate) fn retained_preparation_geometry(
        &self,
        audio: &[f32],
        sample_rate: u32,
        request: &RumbleRequestOptions,
    ) -> Result<RumblePreparedGeometry> {
        self.retained_preparation_features(audio, sample_rate, request)
            .map(|(_, geometry)| geometry)
    }

    /// The retained geometry of a request of `audio_samples` input samples
    /// whose audio features a host worker already computed.
    pub(crate) fn retained_preparation_geometry_for_features(
        &self,
        features: &RumbleAudioFeatures,
        audio_samples: usize,
        request: &RumbleRequestOptions,
    ) -> Result<RumblePreparedGeometry> {
        let prompt = self.build_prompt(features.audio_tokens, &request.prompt_options())?;
        let prompt_tokens = prompt.ids.len();
        let hidden = self.config().text_config.hidden_size;
        let embedding_elements = to_u64(prompt_tokens, "artifact geometry")?
            .checked_mul(to_u64(hidden, "artifact geometry")?)
            .ok_or_else(|| overflow("artifact geometry"))?;
        let retained_device_bytes = embedding_elements
            .checked_mul(to_u64(self.text_dtype().size_in_bytes(), "artifact bytes")?)
            .ok_or_else(|| overflow("artifact bytes"))?;
        Ok(RumblePreparedGeometry {
            audio_samples,
            mel_frames: features.padded_frames,
            mel_bins: features.n_mels,
            prompt_tokens,
            audio_tokens: features.audio_tokens,
            embedding_elements,
            preparation_workspace_bytes: self.preparation_workspace_bytes(
                features.padded_frames,
                features.audio_tokens,
                prompt_tokens,
            )?,
            retained_device_bytes,
            language_bytes: prompt
                .forced_language
                .map_or(0, |language| language.len() as u64),
        })
    }

    /// A conservative live-set envelope (not an allocation sum) of one row's
    /// preparation: the features and their upload, the conv stem (which runs
    /// by slices under its byte budget), the encoder's residual, Q/K/V and
    /// MLP rows per token plus its windowed attention scores for every
    /// head, the projector, and the prompt's assembly. The same formula
    /// prices a request's reservation and the loaded model's ceiling.
    fn preparation_workspace_bytes(
        &self,
        mel_frames: usize,
        audio_tokens: usize,
        prompt_tokens: usize,
    ) -> Result<u64> {
        let audio = &self.config().audio_config;
        let hidden = self.config().text_config.hidden_size;
        let chunk_frames = audio.n_window.unwrap_or(50) * 2;
        let window_tokens = get_cnn_output_lengths(&[chunk_frames])[0]
            * (audio.n_window_infer.unwrap_or(800) / chunk_frames.max(1)).max(1);
        // Each conv keeps its input and output: [channels, mel/2^k, frames/2^k].
        let conv_chunk_elements = audio.downsample_hidden_size
            * ((audio.num_mel_bins / 2) * (chunk_frames / 2)
                + (audio.num_mel_bins / 4) * (chunk_frames / 4)
                + (audio.num_mel_bins / 8) * chunk_frames.div_ceil(8))
            * 2;
        let chunks = mel_frames.div_ceil(chunk_frames.max(1));
        let element_bytes = std::mem::size_of::<f32>() as u64;
        let conv_bytes = to_u64(chunks, "conv workspace")?
            .checked_mul(to_u64(conv_chunk_elements, "conv workspace")?)
            .and_then(|elements| elements.checked_mul(element_bytes))
            .ok_or_else(|| overflow("conv workspace"))?
            .min(2 * RUMBLE_AUDIO_CNN_SLICE_BUDGET_BYTES);
        let per_token = [
            audio.d_model * 10,
            audio.encoder_ffn_dim * 2,
            4 * audio.encoder_attention_heads * window_tokens,
            audio.d_model * 2 + hidden,
        ]
        .into_iter()
        .try_fold(0usize, |total, part| total.checked_add(part))
        .ok_or_else(|| overflow("encoder workspace"))?;
        let elements = mel_frames
            .checked_mul(audio.num_mel_bins * 2)
            .and_then(|features| {
                audio_tokens
                    .checked_mul(per_token)
                    .and_then(|encoder| features.checked_add(encoder))
            })
            .and_then(|total| {
                prompt_tokens
                    .checked_mul(hidden * 3)
                    .and_then(|prompt| total.checked_add(prompt))
            })
            .ok_or_else(|| overflow("preparation workspace"))?;
        to_u64(elements, "preparation workspace")?
            .checked_mul(element_bytes)
            .and_then(|bytes| bytes.checked_add(conv_bytes))
            .filter(|bytes| *bytes > 0)
            .ok_or_else(|| overflow("preparation workspace"))
    }

    /// The ceilings of one request's preparation on `backend`: 30 s of audio
    /// and the longest prompt a retained sequence holds.
    pub(crate) fn scalar_preparation_stage_seal(
        &self,
        backend: BackendKind,
    ) -> Result<RumbleAsrPreparationStageSeal> {
        let loaded = match self.device().kind {
            DeviceKind::Cpu => BackendKind::Cpu,
            DeviceKind::Metal => BackendKind::Metal,
            DeviceKind::Cuda => BackendKind::Cuda,
        };
        if loaded != backend {
            return Err(Error::ModelLoadError(format!(
                "Rumble ASR preparation backend mismatch: model={loaded:?}, adapter={backend:?}"
            )));
        }
        let mel_frames = self.frontend().max_frames();
        let audio_tokens = get_cnn_output_lengths(&[mel_frames])[0];
        let prompt_tokens = RUMBLE_SEQUENCE_TOKENS;
        let max_materialized_tensor_elements_per_row = to_u64(mel_frames, "materialization")?
            .checked_mul(to_u64(self.frontend().n_mels(), "materialization")?)
            .and_then(|features| {
                to_u64(prompt_tokens, "materialization")
                    .ok()?
                    .checked_mul(
                        to_u64(self.config().text_config.hidden_size, "materialization").ok()?,
                    )
                    .and_then(|prompt| features.checked_add(prompt))
            })
            .ok_or_else(|| overflow("maximum materialization"))?;
        Ok(RumbleAsrPreparationStageSeal {
            backend,
            dtype: format!("{:?}", self.text_dtype()).to_ascii_lowercase(),
            max_work_units: to_u64(audio_tokens, "maximum work units")?,
            max_materialized_tensor_elements_per_row,
            max_workspace_bytes: self.preparation_workspace_bytes(
                mel_frames,
                audio_tokens,
                prompt_tokens,
            )?,
        })
    }

    /// The longest audio a retained row takes, in input samples at 16 kHz.
    pub(crate) fn retained_max_audio_samples() -> usize {
        (RUMBLE_MAX_AUDIO_SECONDS * RUMBLE_SAMPLE_RATE as f32) as usize
    }

    /// Encode one request's features and splice the audio into its prompt.
    pub(crate) fn prepare_prompt_artifact_from_features(
        &self,
        features: &RumbleAudioFeatures,
        request: &RumbleRequestOptions,
    ) -> Result<Arc<RumblePreparedPromptArtifact>> {
        let prompt = self.build_prompt(features.audio_tokens, &request.prompt_options())?;
        let audio = self.audio_embeddings(features)?;
        let embeddings = self.prompt_embeddings(&prompt, &audio.embeddings)?;
        // The row keeps exactly its own storage: a pooled allocation would
        // retain more than admission priced.
        let embeddings =
            crate::models::shared::memory::accounting::deep_copy_tensor_storage(&embeddings)?;
        Ok(Arc::new(RumblePreparedPromptArtifact {
            model_identity: self.model_identity(),
            embeddings,
            prompt_tokens: prompt.ids.len(),
            audio_tokens: prompt.audio_tokens,
            forced_language: prompt.forced_language,
            style: request.style,
        }))
    }

    /// Features and artifact for one request's audio.
    pub(crate) fn prepare_prompt_artifact(
        &self,
        audio: &[f32],
        sample_rate: u32,
        request: &RumbleRequestOptions,
    ) -> Result<Arc<RumblePreparedPromptArtifact>> {
        let features = self.prepare_features(audio, sample_rate)?;
        self.prepare_prompt_artifact_from_features(&features, request)
    }

    /// A retained row for `artifact` on `cache`, the empty view of its first
    /// quantum.
    pub(crate) fn begin_resumable_prefill_managed(
        &self,
        artifact: Arc<RumblePreparedPromptArtifact>,
        options: RumbleGenerationOptions,
        cache: PhysicalPagedKvCache,
    ) -> Result<RumbleDecodeState> {
        if artifact.model_identity != self.model_identity() {
            return Err(Error::InvalidInput(
                "Rumble prepared prompt belongs to another loaded model".into(),
            ));
        }
        if cache.context_len() != 0 {
            return Err(Error::InvalidInput(
                "Rumble retained prefill must begin with an empty cache".into(),
            ));
        }
        // As the atomic route: the budget is narrowed to the room the prompt
        // leaves in one sequence.
        let room = RUMBLE_SEQUENCE_TOKENS.saturating_sub(artifact.prompt_tokens);
        if room == 0 {
            return Err(Error::InvalidInput(format!(
                "Rumble prompt ({} tokens) leaves no decode room in one retained sequence ({RUMBLE_SEQUENCE_TOKENS})",
                artifact.prompt_tokens
            )));
        }
        let mut stop_tokens = BTreeSet::from_iter(self.tokenizer().specials().stop_ids());
        stop_tokens.extend(options.stop_token_ids);
        let in_transcript = artifact.forced_language.is_some();
        Ok(RumbleDecodeState {
            cache,
            artifact,
            prefill_progress: 0,
            pending_token: None,
            pos: 0,
            generated_ids: Vec::new(),
            incremental_decoder: IncrementalDecoder::new(true),
            in_transcript,
            rendered: String::new(),
            published_len: 0,
            stop_holdback_bytes: options
                .stop_sequences
                .iter()
                .map(String::len)
                .max()
                .unwrap_or(0)
                .saturating_sub(1),
            stop_tokens: stop_tokens.into_iter().collect(),
            stop_sequences: options.stop_sequences,
            max_new_tokens: options.max_new_tokens.clamp(1, room),
            finished: false,
            final_text: None,
            language: None,
            stop_reason: "max_tokens",
            stop_token: None,
            state_id: NEXT_DECODE_STATE_ID.fetch_add(1, Ordering::Relaxed),
            next_quantum_nonce: 1,
            active_quantum: None,
            managed_completions_drained: true,
        })
    }

    /// Prefill `[span_start, span_end)` of the row's prompt. The last span
    /// stages the greedy first token. Returns whether prefill is complete.
    pub(crate) fn continue_resumable_prefill(
        &self,
        state: &mut RumbleDecodeState,
        span_start: usize,
        span_end: usize,
    ) -> Result<bool> {
        if state.artifact.model_identity != self.model_identity() {
            return Err(Error::InvalidInput(
                "Rumble retained state belongs to another loaded model".into(),
            ));
        }
        if span_start != state.prefill_progress
            || span_start >= span_end
            || span_end > state.artifact.prompt_tokens
            || state.pos != span_start
            || state.cache.context_len() != span_start
        {
            return Err(Error::InvalidInput(
                "Rumble prefill span is non-monotonic or out of range".into(),
            ));
        }
        let span = state
            .artifact
            .embeddings
            .narrow(1, span_start, span_end - span_start)?;
        let complete = span_end == state.artifact.prompt_tokens;
        if complete {
            let logits = self.text_model().forward_managed_last_logits_with_embeds(
                &span,
                span_start,
                &mut state.cache,
                None,
            )?;
            state.pending_token = Some(last_row_argmax(&logits)?);
        } else {
            self.text_model().forward_managed_prefill_only_with_embeds(
                &span,
                span_start,
                &mut state.cache,
                None,
            )?;
        }
        if state.cache.context_len() != span_end {
            return Err(Error::InferenceError(format!(
                "Rumble prefill committed {} tokens instead of {span_end}",
                state.cache.context_len()
            )));
        }
        state.prefill_progress = span_end;
        state.pos = span_end;
        state.managed_completions_drained = false;
        Ok(complete)
    }

    /// One decode step for a lone row: the batched step with one row, so a
    /// row decodes the same alone as beside others.
    pub(crate) fn decode_step(&self, state: &mut RumbleDecodeState) -> Result<RumbleDecodeStep> {
        if state.finished {
            return Ok(state.step(String::new()));
        }
        self.decode_step_batch(&mut [state])?
            .pop()
            .ok_or_else(|| Error::InferenceError("Rumble decode step lost its row".into()))
    }

    /// One token for each retained row: every row appends its staged token
    /// in one batched trunk step and stages the next. Text assembly stays
    /// per row.
    pub(crate) fn decode_step_batch(
        &self,
        states: &mut [&mut RumbleDecodeState],
    ) -> Result<Vec<RumbleDecodeStep>> {
        if states.is_empty() {
            return Ok(Vec::new());
        }
        for state in states.iter() {
            if state.artifact.model_identity != self.model_identity()
                || state.finished
                || state.prefill_progress != state.artifact.prompt_tokens
                || state.pending_token.is_none()
                || state.cache.context_len() != state.pos
            {
                return Err(Error::InvalidInput(
                    "Rumble decode batch requires one staged input token per completed retained row"
                        .into(),
                ));
            }
            state
                .pos
                .checked_add(1)
                .ok_or_else(|| Error::InferenceError("Rumble decode position overflow".into()))?;
        }
        let appended = states
            .iter()
            .map(|state| state.pending_token.expect("validated pending token"))
            .collect::<Vec<_>>();
        let positions = states.iter().map(|state| state.pos).collect::<Vec<_>>();
        let sampled = {
            let mut caches = states
                .iter_mut()
                .map(|state| &mut state.cache)
                .collect::<Vec<_>>();
            let input = Tensor::from_slice(&appended, (appended.len(), 1), &self.device().device)?;
            let logits =
                self.text_model()
                    .forward_managed_decode_batch(&input, &positions, &mut caches)?;
            batched_argmax(&logits)?
        };
        if sampled.len() != states.len() {
            return Err(Error::InferenceError(
                "Rumble decode batch returned the wrong row count".into(),
            ));
        }
        for state in states.iter_mut() {
            state.pending_token = None;
            state.pos += 1;
            state.managed_completions_drained = false;
        }
        states
            .iter_mut()
            .zip(appended)
            .zip(sampled)
            .map(|((state, appended), next)| {
                self.finish_appended_decode_token(state, appended, next)
            })
            .collect()
    }

    /// Account for `appended` (the token a step just appended) and stage
    /// `next`, the step's greedy pick, as the atomic route's loop accepts
    /// tokens: a stop token ends the row unkept, and the budget counts the
    /// tokens kept.
    fn finish_appended_decode_token(
        &self,
        state: &mut RumbleDecodeState,
        appended: u32,
        next: u32,
    ) -> Result<RumbleDecodeStep> {
        if state.stop_tokens.binary_search(&appended).is_ok() {
            state.stop_reason = "stop_token";
            state.stop_token = Some(appended);
            return self.finish_row(state);
        }
        state.generated_ids.push(appended);
        self.render_token(state, appended)?;
        let stopped_on_sequence =
            truncate_at_stop_sequence(&mut state.rendered, &state.stop_sequences);
        let reached_max_tokens = state.generated_ids.len() >= state.max_new_tokens;
        let stopped_on_token = state.stop_tokens.binary_search(&next).is_ok();
        if stopped_on_sequence || reached_max_tokens || stopped_on_token {
            if stopped_on_sequence {
                state.stop_reason = "stop_sequence";
            } else if stopped_on_token {
                state.stop_reason = "stop_token";
                state.stop_token = Some(next);
            }
            return self.finish_row(state);
        }
        let delta = state.publish_stable_text(false)?;
        state.pending_token = Some(next);
        Ok(state.step(delta))
    }

    /// The text of one transcript token: nothing before the model's
    /// `<asr_text>` when it names the language itself.
    fn render_token(&self, state: &mut RumbleDecodeState, id: u32) -> Result<()> {
        if !state.in_transcript {
            if id == self.tokenizer().specials().asr_text {
                state.in_transcript = true;
            }
            return Ok(());
        }
        let delta = self
            .tokenizer()
            .decode_incrementally(&mut state.incremental_decoder, id)?;
        state.rendered.push_str(&delta);
        Ok(())
    }

    /// End the row: flush its text and fix the final transcript and
    /// language as the atomic route computes them.
    fn finish_row(&self, state: &mut RumbleDecodeState) -> Result<RumbleDecodeStep> {
        let tail = self
            .tokenizer()
            .finish_incremental_decode(&mut state.incremental_decoder)?;
        state.rendered.push_str(&tail);
        let stopped_on_sequence =
            truncate_at_stop_sequence(&mut state.rendered, &state.stop_sequences);
        let output = split_rumble_output(
            self.tokenizer(),
            &state.generated_ids,
            state.artifact.forced_language.as_deref(),
        )?;
        let mut text = output.text;
        let text_cut = truncate_at_stop_sequence(&mut text, &state.stop_sequences);
        if text_cut {
            text = text.trim_end().to_string();
        }
        if (stopped_on_sequence || text_cut) && state.stop_reason == "max_tokens" {
            state.stop_reason = "stop_sequence";
        }
        state.language = output
            .language
            .or_else(|| state.artifact.forced_language.clone());
        state.final_text = Some(text);
        state.finished = true;
        state.pending_token = None;
        let delta = state.publish_stable_text(true)?;
        Ok(state.step(delta))
    }
}

/// The argmax of each row's last position of `[rows, sequence, vocab]`.
fn batched_argmax(logits: &Tensor) -> Result<Vec<u32>> {
    let (rows, sequence, _) = logits.dims3()?;
    if rows == 0 || sequence == 0 {
        return Err(Error::InferenceError(format!(
            "Rumble batched decode output expects [rows, sequence, vocab], got {:?}",
            logits.dims()
        )));
    }
    logits
        .narrow(1, sequence - 1, 1)?
        .squeeze(1)?
        .to_dtype(DType::F32)?
        .argmax(D::Minus1)?
        .to_dtype(DType::U32)?
        .to_vec1::<u32>()
        .map_err(Error::from)
}
