//! Granite Speech 5.0's retained route: a single-segment request is prepared
//! once (its audio encoded, possibly beside other requests', into a prompt
//! artifact), prefilled onto retained decoder pages, and decoded beside the
//! other rows, one batched step per token
//! (`GraniteLanguageModel::forward_decode_rows`). Long-form chunks and
//! requests whose prompt and budget exceed
//! [`GRANITE_SPEECH5_RETAINED_SEQUENCE_TOKENS`] run as atomic invocations.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use candle_core::Tensor;

use crate::backends::{BackendKind, DeviceKind};
use crate::error::{Error, Result};
use crate::models::shared::attention::physical::PhysicalPagedKvCache;
use crate::tokenizer::IncrementalDecoder;

use super::{
    clean_granite_speech5_output, GraniteSpeech5AsrModel, GraniteSpeech5AudioFeatures,
    GraniteSpeech5GenerationOptions, GraniteSpeech5RequestOptions, GraniteSpeech5Task,
    GRANITE_SPEECH5_MAX_AUDIO_SECONDS, GRANITE_SPEECH5_RETAINED_SEQUENCE_TOKENS,
};

static NEXT_DECODE_STATE_ID: AtomicU64 = AtomicU64::new(1);

/// A request's prompt with its audio spliced in, ready to prefill.
#[derive(Debug, Clone)]
pub(crate) struct GraniteSpeech5PreparedPromptArtifact {
    model_identity: u64,
    embeddings: Tensor,
    prompt_tokens: usize,
    audio_tokens: usize,
    /// The output's language: the translation target, or the spoken
    /// language the request named.
    language: Option<String>,
}

impl GraniteSpeech5PreparedPromptArtifact {
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
        u64::try_from(self.embeddings.elem_count())
            .ok()
            .and_then(|elements| {
                elements.checked_mul(u64::try_from(self.embeddings.dtype().size_in_bytes()).ok()?)
            })
            .ok_or_else(|| {
                Error::Overloaded("Granite Speech 5.0 prepared artifact size overflow".into())
            })
    }

    pub(crate) fn resident_host_bytes(&self) -> u64 {
        self.language
            .as_ref()
            .map_or(0, |language| language.len() as u64)
    }
}

/// What one request's retained preparation will hold and use, known before
/// its audio is encoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GraniteSpeech5PreparedGeometry {
    pub(crate) audio_samples: usize,
    pub(crate) encoder_frames: usize,
    pub(crate) encoder_dim: usize,
    pub(crate) prompt_tokens: usize,
    pub(crate) audio_tokens: usize,
    pub(crate) embedding_elements: u64,
    pub(crate) preparation_workspace_bytes: u64,
    pub(crate) retained_device_bytes: u64,
    /// The output language's bytes the artifact keeps.
    pub(crate) language_bytes: u64,
}

impl GraniteSpeech5PreparedGeometry {
    pub(crate) fn work_cost(self) -> crate::engine::WorkCost {
        crate::engine::WorkCost::new(
            self.audio_tokens as u64,
            self.embedding_elements,
            self.preparation_workspace_bytes,
        )
    }

    fn batch_useful_tensor_elements(self) -> Result<u64> {
        u64::try_from(self.encoder_frames)
            .ok()
            .and_then(|frames| frames.checked_mul(u64::try_from(self.encoder_dim).ok()?))
            .and_then(|encoder| encoder.checked_add(self.embedding_elements))
            .ok_or_else(|| {
                Error::Overloaded("Granite Speech 5.0 useful preparation elements overflow".into())
            })
    }
}

/// One request of a preparation batch.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GraniteSpeech5PreparationBatchRow<'a> {
    pub(crate) audio: &'a [f32],
    pub(crate) sample_rate: u32,
    pub(crate) request: &'a GraniteSpeech5RequestOptions,
}

/// One request of a preparation batch whose audio features a host worker
/// computed.
pub(crate) struct GraniteSpeech5FeatureRow<'a> {
    pub(crate) features: GraniteSpeech5AudioFeatures,
    pub(crate) request: &'a GraniteSpeech5RequestOptions,
}

/// A preparation batch priced at its padded geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GraniteSpeech5PreparationBatchGeometry {
    pub(crate) batch_size: usize,
    pub(crate) padded_audio_samples_per_row: usize,
    pub(crate) padded_encoder_frames_per_row: usize,
    pub(crate) encoder_dim: usize,
    pub(crate) materialized_tensor_elements_per_row: u64,
    pub(crate) workspace_per_row_bytes: u64,
    pub(crate) max_workspace_bytes: u64,
}

/// The preparation ceilings a load seals into its adapter: one request of
/// the longest audio and prompt the retained route takes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GraniteSpeech5AsrPreparationStageSeal {
    pub(crate) backend: BackendKind,
    pub(crate) dtype: String,
    pub(crate) max_work_units: u64,
    pub(crate) max_materialized_tensor_elements_per_row: u64,
    pub(crate) max_workspace_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GraniteSpeech5DecodeStep {
    pub(crate) delta: String,
    pub(crate) text: String,
    pub(crate) tokens_generated: usize,
    pub(crate) finished: bool,
}

/// One retained row: its prepared prompt, its decoder pages for the current
/// scheduler quantum, and its text so far.
pub(crate) struct GraniteSpeech5DecodeState {
    cache: PhysicalPagedKvCache,
    artifact: Arc<GraniteSpeech5PreparedPromptArtifact>,
    prefill_progress: usize,
    /// The token the next decode step appends: the greedy pick of the last
    /// prefill span or decode step.
    pending_token: Option<u32>,
    pos: usize,
    generated_ids: Vec<u32>,
    incremental_decoder: IncrementalDecoder,
    rendered: String,
    published_len: usize,
    stop_holdback_bytes: usize,
    stop_tokens: Vec<u32>,
    stop_sequences: Vec<String>,
    max_new_tokens: usize,
    finished: bool,
    stop_reason: &'static str,
    stop_token: Option<u32>,
    state_id: u64,
    next_quantum_nonce: u64,
    active_quantum: Option<u64>,
    managed_completions_drained: bool,
}

struct GraniteSpeech5DecodeCheckpointPayload {
    cache: PhysicalPagedKvCache,
    prefill_progress: usize,
    pending_token: Option<u32>,
    pos: usize,
    generated_ids: Vec<u32>,
    incremental_decoder: IncrementalDecoder,
    rendered: String,
    published_len: usize,
    finished: bool,
    stop_reason: &'static str,
    stop_token: Option<u32>,
    managed_completions_drained: bool,
}

/// A row's state before a scheduler quantum, restored if the quantum rolls
/// back.
pub(crate) struct GraniteSpeech5DecodeCheckpoint {
    state_id: u64,
    quantum_nonce: u64,
    payload: Option<GraniteSpeech5DecodeCheckpointPayload>,
}

impl GraniteSpeech5DecodeState {
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

    pub(crate) fn language(&self) -> Option<&str> {
        self.artifact.language.as_deref()
    }

    /// Whether the row has finished (a stop token or sequence, or its
    /// budget).
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
    ) -> Result<GraniteSpeech5DecodeCheckpoint> {
        if self.active_quantum.is_some() {
            return Err(Error::InferenceError(
                "a Granite Speech 5.0 managed quantum is already active".into(),
            ));
        }
        if !self.managed_completions_drained {
            return Err(Error::InferenceError(
                "Granite Speech 5.0 managed KV completions must be drained before the next quantum"
                    .into(),
            ));
        }
        if self.cache.sequence_authority() != cache.sequence_authority() {
            return Err(Error::InferenceError(
                "a Granite Speech 5.0 session cannot switch managed KV authority".into(),
            ));
        }
        if cache.context_len() != self.pos {
            return Err(Error::InferenceError(format!(
                "managed Granite Speech 5.0 reservation starts at {}, but state is at {}",
                cache.context_len(),
                self.pos
            )));
        }
        let quantum_nonce = self.next_quantum_nonce;
        self.next_quantum_nonce = self.next_quantum_nonce.checked_add(1).ok_or_else(|| {
            Error::InferenceError("Granite Speech 5.0 quantum nonce overflow".into())
        })?;
        self.active_quantum = Some(quantum_nonce);
        Ok(GraniteSpeech5DecodeCheckpoint {
            state_id: self.state_id,
            quantum_nonce,
            payload: Some(GraniteSpeech5DecodeCheckpointPayload {
                cache: std::mem::replace(&mut self.cache, cache),
                prefill_progress: self.prefill_progress,
                pending_token: self.pending_token,
                pos: self.pos,
                generated_ids: self.generated_ids.clone(),
                incremental_decoder: self.incremental_decoder.clone(),
                rendered: self.rendered.clone(),
                published_len: self.published_len,
                finished: self.finished,
                stop_reason: self.stop_reason,
                stop_token: self.stop_token,
                managed_completions_drained: self.managed_completions_drained,
            }),
        })
    }

    pub(crate) fn commit_managed_quantum(
        &mut self,
        checkpoint: &mut GraniteSpeech5DecodeCheckpoint,
    ) -> Result<()> {
        self.validate_checkpoint(checkpoint)?;
        checkpoint.payload.take().ok_or_else(|| {
            Error::InferenceError("Granite Speech 5.0 checkpoint was already consumed".into())
        })?;
        self.active_quantum = None;
        Ok(())
    }

    pub(crate) fn rollback_managed_quantum(
        &mut self,
        checkpoint: &mut GraniteSpeech5DecodeCheckpoint,
    ) -> Result<()> {
        self.validate_checkpoint(checkpoint)?;
        let payload = checkpoint.payload.take().ok_or_else(|| {
            Error::InferenceError("Granite Speech 5.0 checkpoint was already consumed".into())
        })?;
        self.cache = payload.cache;
        self.prefill_progress = payload.prefill_progress;
        self.pending_token = payload.pending_token;
        self.pos = payload.pos;
        self.generated_ids = payload.generated_ids;
        self.incremental_decoder = payload.incremental_decoder;
        self.rendered = payload.rendered;
        self.published_len = payload.published_len;
        self.finished = payload.finished;
        self.stop_reason = payload.stop_reason;
        self.stop_token = payload.stop_token;
        self.managed_completions_drained = payload.managed_completions_drained;
        self.active_quantum = None;
        Ok(())
    }

    fn validate_checkpoint(&self, checkpoint: &GraniteSpeech5DecodeCheckpoint) -> Result<()> {
        if checkpoint.state_id != self.state_id
            || self.active_quantum != Some(checkpoint.quantum_nonce)
            || checkpoint.payload.is_none()
        {
            return Err(Error::InferenceError(
                "Granite Speech 5.0 checkpoint is foreign, stale, or out of order".into(),
            ));
        }
        Ok(())
    }

    fn step(&self, delta: String) -> GraniteSpeech5DecodeStep {
        GraniteSpeech5DecodeStep {
            delta,
            // The final text is cleaned as the atomic route's is; deltas
            // stream raw, as the atomic route streams them.
            text: if self.finished {
                clean_granite_speech5_output(&self.rendered)
            } else {
                self.rendered.clone()
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
                "Granite Speech 5.0 decoded text rewrote an already published prefix".into(),
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
    Error::Overloaded(format!("Granite Speech 5.0 {what} overflow"))
}

fn to_u64(value: usize, what: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| overflow(what))
}

impl GraniteSpeech5AsrModel {
    /// Whether a single-segment ASR request resumes prefill on the retained
    /// route ([`super::granite_speech5_retained_decode_enabled`], read at
    /// load).
    pub(crate) const fn supports_resumable_prefill(&self) -> bool {
        self.retained_decode
    }

    /// The retained route's decode steps: the same switch as
    /// [`Self::supports_resumable_prefill`], so the two never disagree.
    pub(crate) const fn supports_incremental_decode(&self) -> bool {
        self.retained_decode
    }

    /// Retained rows decode together, one batched step per token.
    pub(crate) const fn supports_continuous_decode_batch(&self) -> bool {
        self.retained_decode
    }

    /// Whether a request of `prompt_tokens` and a `max_new_tokens` budget
    /// fits one retained sequence; a longer one runs as an atomic
    /// invocation.
    pub(crate) fn retained_sequence_fits(prompt_tokens: usize, max_new_tokens: usize) -> bool {
        prompt_tokens
            .checked_add(max_new_tokens.max(1))
            .is_some_and(|tokens| tokens <= GRANITE_SPEECH5_RETAINED_SEQUENCE_TOKENS)
    }

    /// Bytes one row of a continuous decode step uses beside its retained
    /// state: its hidden row, Q/K/V row and MLP row (priced at F32) and its
    /// F32 logits row over the 100k vocabulary.
    pub(crate) fn continuous_decode_workspace_per_row_bytes(&self) -> Result<u64> {
        let text = &self.config.text_config;
        let head_dim = self.config.decoder_head_dim();
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

    /// The output language of `request`: its translation target, or the
    /// spoken language it named.
    fn output_language(request: &GraniteSpeech5RequestOptions) -> Result<Option<String>> {
        let options = request.prompt_options()?;
        Ok(match options.task {
            GraniteSpeech5Task::Translate { target } => Some(target.code().to_string()),
            GraniteSpeech5Task::Transcribe => options.source_language,
        })
    }

    /// The geometry `audio` and `request` will prepare to, from the host
    /// features and the rendered prompt, before any encoder work.
    pub(crate) fn retained_preparation_geometry(
        &self,
        audio: &[f32],
        sample_rate: u32,
        request: &GraniteSpeech5RequestOptions,
    ) -> Result<GraniteSpeech5PreparedGeometry> {
        let features = self.preprocessor.prepare(audio, sample_rate)?;
        self.retained_preparation_geometry_for_features(&features, audio.len(), request)
    }

    /// One request's audio features, computed on the host, and the retained
    /// geometry they give.
    pub(crate) fn retained_preparation_features(
        &self,
        audio: &[f32],
        sample_rate: u32,
        request: &GraniteSpeech5RequestOptions,
    ) -> Result<(GraniteSpeech5AudioFeatures, GraniteSpeech5PreparedGeometry)> {
        let features = self.preprocessor.prepare(audio, sample_rate)?;
        let geometry =
            self.retained_preparation_geometry_for_features(&features, audio.len(), request)?;
        Ok((features, geometry))
    }

    /// The retained geometry of a request of `audio_samples` input samples
    /// whose audio features a host worker already computed.
    pub(crate) fn retained_preparation_geometry_for_features(
        &self,
        features: &GraniteSpeech5AudioFeatures,
        audio_samples: usize,
        request: &GraniteSpeech5RequestOptions,
    ) -> Result<GraniteSpeech5PreparedGeometry> {
        if features.audio_seconds > GRANITE_SPEECH5_MAX_AUDIO_SECONDS {
            return Err(Error::InvalidInput(format!(
                "Granite Speech 5.0 transcribes up to {GRANITE_SPEECH5_MAX_AUDIO_SECONDS:.0}s per invocation, got {:.1}s",
                features.audio_seconds
            )));
        }
        let audio_tokens = self
            .window_count(features.encoder_frames)
            .checked_mul(self.config.queries_per_window())
            .ok_or_else(|| overflow("audio token geometry"))?;
        if audio_tokens == 0 {
            return Err(Error::InvalidInput(
                "Granite Speech 5.0 projected zero audio tokens".into(),
            ));
        }
        let prompt = self.build_prompt(&request.prompt_options()?)?;
        let prompt_tokens = prompt
            .input_ids
            .len()
            .checked_add(audio_tokens - 1)
            .ok_or_else(|| overflow("prompt geometry"))?;
        let embedding_elements = to_u64(prompt_tokens, "artifact geometry")?
            .checked_mul(to_u64(
                self.config.text_config.hidden_size,
                "artifact geometry",
            )?)
            .ok_or_else(|| overflow("artifact geometry"))?;
        let retained_device_bytes = embedding_elements
            .checked_mul(to_u64(
                self.runtime.embedding_dtype().size_in_bytes(),
                "artifact bytes",
            )?)
            .ok_or_else(|| overflow("artifact bytes"))?;
        let language_bytes =
            Self::output_language(request)?.map_or(0, |language| language.len() as u64);
        Ok(GraniteSpeech5PreparedGeometry {
            audio_samples,
            encoder_frames: features.encoder_frames,
            encoder_dim: features.encoder_dim,
            prompt_tokens,
            audio_tokens,
            embedding_elements,
            preparation_workspace_bytes: self.preparation_workspace_bytes(
                features.encoder_frames,
                features.encoder_dim,
                prompt_tokens,
            )?,
            retained_device_bytes,
            language_bytes,
        })
    }

    fn window_count(&self, encoder_frames: usize) -> usize {
        encoder_frames.div_ceil(self.config.window_size.max(1))
    }

    /// One batch of `rows`, priced at its padded geometry: every row as long
    /// as its longest audio and prompt.
    pub(crate) fn preparation_batch_geometry(
        &self,
        rows: &[GraniteSpeech5PreparedGeometry],
    ) -> Result<GraniteSpeech5PreparationBatchGeometry> {
        let encoder_dim = rows.first().map(|row| row.encoder_dim).ok_or_else(|| {
            Error::InvalidInput("Granite Speech 5.0 preparation batch geometry is empty".into())
        })?;
        let padded_audio_samples_per_row =
            rows.iter().map(|row| row.audio_samples).max().unwrap_or(0);
        let padded_encoder_frames_per_row =
            rows.iter().map(|row| row.encoder_frames).max().unwrap_or(0);
        let max_prompt_tokens = rows.iter().map(|row| row.prompt_tokens).max().unwrap_or(0);
        if padded_audio_samples_per_row == 0
            || padded_encoder_frames_per_row == 0
            || encoder_dim == 0
            || rows.iter().any(|row| row.encoder_dim != encoder_dim)
        {
            return Err(Error::InvalidInput(
                "Granite Speech 5.0 preparation rows have incompatible geometry".into(),
            ));
        }
        let input_elements = to_u64(padded_encoder_frames_per_row, "batch input")?
            .checked_mul(to_u64(encoder_dim, "batch input")?)
            .ok_or_else(|| overflow("batch input"))?;
        let prompt_elements = to_u64(max_prompt_tokens, "batch prompt")?
            .checked_mul(to_u64(self.config.text_config.hidden_size, "batch prompt")?)
            .ok_or_else(|| overflow("batch prompt"))?;
        let workspace_per_row_bytes = self.preparation_workspace_bytes(
            padded_encoder_frames_per_row,
            encoder_dim,
            max_prompt_tokens,
        )?;
        Ok(GraniteSpeech5PreparationBatchGeometry {
            batch_size: rows.len(),
            padded_audio_samples_per_row,
            padded_encoder_frames_per_row,
            encoder_dim,
            materialized_tensor_elements_per_row: input_elements
                .checked_add(prompt_elements)
                .ok_or_else(|| overflow("batch materialization"))?,
            workspace_per_row_bytes,
            max_workspace_bytes: workspace_per_row_bytes
                .checked_mul(to_u64(rows.len(), "batch workspace")?)
                .ok_or_else(|| overflow("batch workspace"))?,
        })
    }

    /// Row `index`'s cost inside `batch`, which must be the geometry of
    /// `rows`.
    pub(crate) fn preparation_row_cost_for_batch(
        &self,
        index: usize,
        rows: &[GraniteSpeech5PreparedGeometry],
        batch: GraniteSpeech5PreparationBatchGeometry,
    ) -> Result<crate::engine::WorkCost> {
        let row = rows.get(index).ok_or_else(|| {
            Error::InvalidInput("Granite Speech 5.0 preparation row index is out of range".into())
        })?;
        if batch != self.preparation_batch_geometry(rows)? {
            return Err(Error::InvalidInput(
                "Granite Speech 5.0 preparation batch geometry is stale or foreign".into(),
            ));
        }
        Ok(crate::engine::WorkCost::new(
            row.audio_tokens as u64,
            row.batch_useful_tensor_elements()?,
            batch.workspace_per_row_bytes,
        ))
    }

    /// A conservative live-set envelope (not an allocation sum) of one
    /// row's preparation: host features and their upload, the conformer's
    /// residual, Q/K/V, MLP and CTC rows per frame plus its block-local
    /// attention (scores, relative bias, mask and F32 softmax over each
    /// frame's 200-frame block for every head), the Q-Former's windows, and
    /// the prompt's assembly. The same formula prices a request's
    /// reservation and the loaded model's ceiling.
    fn preparation_workspace_bytes(
        &self,
        encoder_frames: usize,
        encoder_dim: usize,
        prompt_tokens: usize,
    ) -> Result<u64> {
        let encoder = &self.config.encoder_config;
        let projector = &self.config.projector_config;
        let text_hidden = self.config.text_config.hidden_size;
        let window = self.config.window_size.max(1);
        let queries = self.config.queries_per_window();
        let per_frame = [
            encoder_dim,
            encoder.hidden_dim * 10,
            projector.encoder_hidden_size * 2,
            encoder.output_dim,
            4 * encoder.num_heads * encoder.context_size,
        ]
        .into_iter()
        .try_fold(0usize, |total, part| total.checked_add(part))
        .ok_or_else(|| overflow("encoder workspace"))?;
        let per_window = [
            window * projector.encoder_hidden_size * 2,
            queries * projector.hidden_size * 12,
            queries * projector.intermediate_size * 2,
            queries * text_hidden,
        ]
        .into_iter()
        .try_fold(0usize, |total, part| total.checked_add(part))
        .ok_or_else(|| overflow("projector workspace"))?;
        let elements = encoder_frames
            .checked_mul(per_frame)
            .and_then(|encoder| {
                self.window_count(encoder_frames)
                    .checked_mul(per_window)
                    .and_then(|projector| encoder.checked_add(projector))
            })
            .and_then(|total| {
                prompt_tokens
                    .checked_mul(text_hidden * 3)
                    .and_then(|prompt| total.checked_add(prompt))
            })
            .ok_or_else(|| overflow("preparation workspace"))?;
        let element_bytes = self.dtype.size_in_bytes().max(std::mem::size_of::<f32>());
        to_u64(elements, "preparation workspace")?
            .checked_mul(to_u64(element_bytes, "preparation workspace")?)
            .filter(|bytes| *bytes > 0)
            .ok_or_else(|| overflow("preparation workspace"))
    }

    /// The ceilings of one request's preparation on `backend`: 120 s of
    /// audio and the longest prompt a retained sequence holds.
    pub(crate) fn scalar_preparation_stage_seal(
        &self,
        backend: BackendKind,
    ) -> Result<GraniteSpeech5AsrPreparationStageSeal> {
        let loaded = match self.device.kind {
            DeviceKind::Cpu => BackendKind::Cpu,
            DeviceKind::Metal => BackendKind::Metal,
            DeviceKind::Cuda => BackendKind::Cuda,
        };
        if loaded != backend {
            return Err(Error::ModelLoadError(format!(
                "Granite Speech 5.0 ASR preparation backend mismatch: model={loaded:?}, adapter={backend:?}"
            )));
        }
        let mel = &self.processor.audio_processor.melspec_kwargs;
        let samples = u64::from(self.processor.sample_rate())
            .checked_mul(GRANITE_SPEECH5_MAX_AUDIO_SECONDS as u64)
            .ok_or_else(|| overflow("maximum sample count"))?;
        let encoder_frames =
            usize::try_from(samples / to_u64(mel.hop_length.max(1), "hop")? / 2 + 2)
                .map_err(|_| overflow("maximum frames"))?;
        let encoder_dim = self.config.encoder_config.input_dim;
        let prompt_tokens = GRANITE_SPEECH5_RETAINED_SEQUENCE_TOKENS;
        let audio_tokens = self
            .window_count(encoder_frames)
            .checked_mul(self.config.queries_per_window())
            .ok_or_else(|| overflow("maximum work units"))?;
        let max_materialized_tensor_elements_per_row = to_u64(encoder_frames, "materialization")?
            .checked_mul(to_u64(encoder_dim, "materialization")?)
            .and_then(|input| {
                to_u64(prompt_tokens, "materialization")
                    .ok()?
                    .checked_mul(
                        to_u64(self.config.text_config.hidden_size, "materialization").ok()?,
                    )
                    .and_then(|prompt| input.checked_add(prompt))
            })
            .ok_or_else(|| overflow("maximum materialization"))?;
        Ok(GraniteSpeech5AsrPreparationStageSeal {
            backend,
            dtype: format!("{:?}", self.dtype).to_ascii_lowercase(),
            max_work_units: to_u64(audio_tokens, "maximum work units")?,
            max_materialized_tensor_elements_per_row,
            max_workspace_bytes: self.preparation_workspace_bytes(
                encoder_frames,
                encoder_dim,
                prompt_tokens,
            )?,
        })
    }

    /// Encode `rows` (one batch through the encoder and Q-Former) and splice
    /// each row's audio into its own prompt.
    pub(crate) fn prepare_prompt_artifact_batch(
        &self,
        rows: &[GraniteSpeech5PreparationBatchRow<'_>],
    ) -> Result<Vec<Arc<GraniteSpeech5PreparedPromptArtifact>>> {
        let rows = rows
            .iter()
            .map(|row| {
                Ok(GraniteSpeech5FeatureRow {
                    features: self.preprocessor.prepare(row.audio, row.sample_rate)?,
                    request: row.request,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        self.prepare_prompt_artifact_batch_from_features(rows)
    }

    /// Encode rows whose audio features a host worker computed (one batch
    /// through the encoder and Q-Former) and splice each row's audio into its
    /// own prompt.
    pub(crate) fn prepare_prompt_artifact_batch_from_features(
        &self,
        rows: Vec<GraniteSpeech5FeatureRow<'_>>,
    ) -> Result<Vec<Arc<GraniteSpeech5PreparedPromptArtifact>>> {
        if rows.is_empty() {
            return Err(Error::InvalidInput(
                "Granite Speech 5.0 prompt preparation batch is empty".into(),
            ));
        }
        let mut features = Vec::with_capacity(rows.len());
        let mut prompts = Vec::with_capacity(rows.len());
        for row in rows {
            let row_features = row.features;
            if row_features.audio_seconds > GRANITE_SPEECH5_MAX_AUDIO_SECONDS {
                return Err(Error::InvalidInput(format!(
                    "Granite Speech 5.0 transcribes up to {GRANITE_SPEECH5_MAX_AUDIO_SECONDS:.0}s per invocation, got {:.1}s",
                    row_features.audio_seconds
                )));
            }
            features.push(row_features);
            prompts.push((
                self.build_prompt(&row.request.prompt_options()?)?,
                Self::output_language(row.request)?,
            ));
        }
        let audio = self.runtime.audio_embeddings_batch(&features)?;
        prompts
            .into_iter()
            .zip(audio)
            .map(|((prompt, language), (audio_embeds, _stats))| {
                let (embeddings, prompt_tokens, audio_tokens) = self.runtime.prompt_embeddings(
                    &prompt,
                    self.prompt_tokenizer.special_tokens(),
                    &audio_embeds,
                )?;
                Ok(Arc::new(GraniteSpeech5PreparedPromptArtifact {
                    model_identity: self.model_identity,
                    embeddings,
                    prompt_tokens,
                    audio_tokens,
                    language,
                }))
            })
            .collect()
    }

    /// A retained row for `artifact` on `cache`, the empty view of its first
    /// quantum: the request-wide context was bounded at preparation.
    pub(crate) fn begin_resumable_prefill_managed(
        &self,
        artifact: Arc<GraniteSpeech5PreparedPromptArtifact>,
        options: GraniteSpeech5GenerationOptions,
        cache: PhysicalPagedKvCache,
    ) -> Result<GraniteSpeech5DecodeState> {
        if artifact.model_identity != self.model_identity {
            return Err(Error::InvalidInput(
                "Granite Speech 5.0 prepared prompt belongs to another loaded model".into(),
            ));
        }
        if cache.context_len() != 0 {
            return Err(Error::InvalidInput(
                "Granite Speech 5.0 retained prefill must begin with an empty cache".into(),
            ));
        }
        let special = self.prompt_tokenizer.special_tokens();
        let mut stop_tokens = BTreeSet::from([special.eos_token_id, special.pad_token_id]);
        stop_tokens.extend(options.stop_token_ids);
        Ok(GraniteSpeech5DecodeState {
            cache,
            artifact,
            prefill_progress: 0,
            pending_token: None,
            pos: 0,
            generated_ids: Vec::new(),
            incremental_decoder: self.prompt_tokenizer.incremental_decoder(),
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
            max_new_tokens: options.max_new_tokens.max(1),
            finished: false,
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
        state: &mut GraniteSpeech5DecodeState,
        span_start: usize,
        span_end: usize,
    ) -> Result<bool> {
        if state.artifact.model_identity != self.model_identity {
            return Err(Error::InvalidInput(
                "Granite Speech 5.0 retained state belongs to another loaded model".into(),
            ));
        }
        if span_start != state.prefill_progress
            || span_start >= span_end
            || span_end > state.artifact.prompt_tokens
            || state.pos != span_start
        {
            return Err(Error::InvalidInput(
                "Granite Speech 5.0 prefill span is non-monotonic or out of range".into(),
            ));
        }
        let logits = self.runtime.prefill_prompt_span(
            &state.artifact.embeddings,
            span_start,
            span_end,
            &mut state.cache,
        )?;
        state.prefill_progress = span_end;
        state.pos = span_end;
        let complete = span_end == state.artifact.prompt_tokens;
        if complete {
            state.pending_token = Some(self.runtime.greedy_token(&logits)?);
        }
        state.managed_completions_drained = false;
        Ok(complete)
    }

    /// One decode step for a lone row: the batched step with one row, so a
    /// row decodes the same alone as beside others.
    pub(crate) fn decode_step(
        &self,
        state: &mut GraniteSpeech5DecodeState,
    ) -> Result<GraniteSpeech5DecodeStep> {
        if state.finished {
            return Ok(state.step(String::new()));
        }
        self.decode_step_batch(&mut [state])?.pop().ok_or_else(|| {
            Error::InferenceError("Granite Speech 5.0 decode step lost its row".into())
        })
    }

    /// One token for each retained row: every row appends its staged token
    /// in one batched step (`GraniteSpeech5Runtime::decode_tokens_batch`)
    /// and stages the next. Text assembly stays per row.
    pub(crate) fn decode_step_batch(
        &self,
        states: &mut [&mut GraniteSpeech5DecodeState],
    ) -> Result<Vec<GraniteSpeech5DecodeStep>> {
        if states.is_empty() {
            return Ok(Vec::new());
        }
        for state in states.iter() {
            if state.artifact.model_identity != self.model_identity
                || state.finished
                || state.prefill_progress != state.artifact.prompt_tokens
                || state.pending_token.is_none()
                || state.cache.context_len() != state.pos
            {
                return Err(Error::InvalidInput(
                    "Granite Speech 5.0 decode batch requires one staged input token per completed retained row"
                        .into(),
                ));
            }
            state.pos.checked_add(1).ok_or_else(|| {
                Error::InferenceError("Granite Speech 5.0 decode position overflow".into())
            })?;
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
            self.runtime
                .decode_tokens_batch(&appended, &positions, &mut caches)?
        };
        if sampled.len() != states.len() {
            return Err(Error::InferenceError(
                "Granite Speech 5.0 decode batch returned the wrong row count".into(),
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
    /// `next`, the step's greedy pick, exactly as the atomic route's loop
    /// accepts tokens: a stop token ends the row unappended, and the budget
    /// counts the tokens kept.
    fn finish_appended_decode_token(
        &self,
        state: &mut GraniteSpeech5DecodeState,
        appended: u32,
        next: u32,
    ) -> Result<GraniteSpeech5DecodeStep> {
        if state.stop_tokens.binary_search(&appended).is_ok() {
            state.rendered.push_str(
                &self
                    .prompt_tokenizer
                    .finish_incremental_decode(&mut state.incremental_decoder)?,
            );
            truncate_at_stop_sequence(&mut state.rendered, &state.stop_sequences);
            state.finished = true;
            let delta = state.publish_stable_text(true)?;
            state.pending_token = None;
            state.stop_reason = "stop_token";
            state.stop_token = Some(appended);
            return Ok(state.step(delta));
        }
        state.generated_ids.push(appended);
        state.rendered.push_str(
            &self
                .prompt_tokenizer
                .decode_incrementally(&mut state.incremental_decoder, appended)?,
        );
        let mut stopped_on_sequence =
            truncate_at_stop_sequence(&mut state.rendered, &state.stop_sequences);
        let reached_max_tokens = state.generated_ids.len() >= state.max_new_tokens;
        let stopped_on_token = state.stop_tokens.binary_search(&next).is_ok();
        if (reached_max_tokens || stopped_on_token) && !stopped_on_sequence {
            state.rendered.push_str(
                &self
                    .prompt_tokenizer
                    .finish_incremental_decode(&mut state.incremental_decoder)?,
            );
            stopped_on_sequence =
                truncate_at_stop_sequence(&mut state.rendered, &state.stop_sequences);
        }
        let finished = stopped_on_sequence || reached_max_tokens || stopped_on_token;
        let delta = state.publish_stable_text(finished)?;
        state.finished = finished;
        if stopped_on_sequence {
            state.stop_reason = "stop_sequence";
        } else if stopped_on_token {
            state.stop_reason = "stop_token";
            state.stop_token = Some(next);
        }
        state.pending_token = (!finished).then_some(next);
        Ok(state.step(delta))
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::backends::kv::{CpuKvArena, KvArena, KvArenaConfig, KvLayerConfig};
    use crate::engine::ModelInstanceId;
    use crate::kv::{CacheBlockRef, KvArenaId, KvGroupId, KvLayerBinding};
    use candle_core::DType;

    fn caches() -> (PhysicalPagedKvCache, PhysicalPagedKvCache) {
        let binding = KvLayerBinding {
            model_layer: 0,
            physical_layer: 0,
        };
        let arena: Arc<dyn KvArena> = Arc::new(
            CpuKvArena::new(KvArenaConfig {
                id: KvArenaId {
                    model_instance: ModelInstanceId::new(5_177),
                    backend: BackendKind::Cpu,
                    device_ordinal: None,
                    generation: 1,
                },
                group: KvGroupId::new(0),
                page_tokens: 2,
                capacity_pages: 4,
                growth: None,
                dtype: DType::F32,
                layers: vec![KvLayerConfig {
                    binding,
                    num_kv_heads: 1,
                    key_head_dim: 2,
                    value_head_dim: 2,
                }],
            })
            .unwrap(),
        );
        let cache = || {
            let blocks = (0..4)
                .map(|index| CacheBlockRef {
                    arena: arena.id(),
                    group: arena.config().group,
                    index,
                    slot_generation: 1,
                })
                .collect();
            PhysicalPagedKvCache::new(arena.clone(), vec![binding], blocks, 0).unwrap()
        };
        (cache(), cache())
    }

    fn state(cache: PhysicalPagedKvCache, prompt_tokens: usize) -> GraniteSpeech5DecodeState {
        GraniteSpeech5DecodeState {
            cache,
            artifact: Arc::new(GraniteSpeech5PreparedPromptArtifact {
                model_identity: 9,
                embeddings: Tensor::zeros(
                    (1, prompt_tokens.max(1), 4),
                    DType::F32,
                    &candle_core::Device::Cpu,
                )
                .unwrap(),
                prompt_tokens,
                audio_tokens: 1,
                language: Some("en".into()),
            }),
            prefill_progress: 0,
            pending_token: None,
            pos: 0,
            generated_ids: vec![],
            incremental_decoder: IncrementalDecoder::new(true),
            rendered: String::new(),
            published_len: 0,
            stop_holdback_bytes: 0,
            stop_tokens: vec![2],
            stop_sequences: vec![],
            max_new_tokens: 4,
            finished: false,
            stop_reason: "max_tokens",
            stop_token: None,
            state_id: NEXT_DECODE_STATE_ID.fetch_add(1, Ordering::Relaxed),
            next_quantum_nonce: 1,
            active_quantum: None,
            managed_completions_drained: true,
        }
    }

    #[test]
    fn decode_checkpoint_restores_the_staged_token_and_cursor() {
        let (cache, replacement) = caches();
        let mut state = state(cache, 0);
        state.pending_token = Some(6);
        let mut checkpoint = state.begin_managed_quantum(replacement).unwrap();
        state.pending_token = Some(1);
        state.pos = 1;
        state.managed_completions_drained = false;
        state.rollback_managed_quantum(&mut checkpoint).unwrap();
        assert_eq!(state.pending_token, Some(6));
        assert_eq!(state.sequence_position(), 0);
        assert!(state.managed_completions_drained);
    }

    #[test]
    fn checkpoint_rolls_back_staged_progress_and_is_single_use() {
        let (cache, replacement) = caches();
        let mut state = state(cache, 3);
        let mut checkpoint = state.begin_managed_quantum(replacement).unwrap();
        state.prefill_progress = 3;
        state.pos = 3;
        state.generated_ids.push(7);
        state.pending_token = Some(7);
        state.rendered = "staged".into();
        state.finished = true;
        state.rollback_managed_quantum(&mut checkpoint).unwrap();
        assert_eq!(state.prefill_progress(), 0);
        assert_eq!(state.sequence_position(), 0);
        assert!(state.generated_ids.is_empty());
        assert_eq!(state.pending_token, None);
        assert!(state.rendered.is_empty());
        assert!(!state.finished);
        assert!(state
            .rollback_managed_quantum(&mut checkpoint)
            .unwrap_err()
            .to_string()
            .contains("foreign, stale, or out of order"));
        // A second quantum may not begin before the first is settled.
        let (_, other) = caches();
        let mut checkpoint = state.begin_managed_quantum(other).unwrap();
        let (_, third) = caches();
        assert!(state.begin_managed_quantum(third).is_err());
        state.commit_managed_quantum(&mut checkpoint).unwrap();
        assert!(state.commit_managed_quantum(&mut checkpoint).is_err());
    }

    /// A stop sequence's prefix is held back until it either completes (and
    /// is cut) or cannot; the final step's text is cleaned as the atomic
    /// route's is.
    #[test]
    fn published_text_holds_back_a_stop_sequence_prefix() {
        let (cache, _) = caches();
        let mut state = state(cache, 0);
        state.stop_sequences = vec!["<stop>".into()];
        state.stop_holdback_bytes = "<stop>".len() - 1;
        state.rendered = " hello<st".into();
        let first = state.publish_stable_text(false).unwrap();
        state.rendered.push_str("op>");
        assert!(truncate_at_stop_sequence(
            &mut state.rendered,
            &state.stop_sequences
        ));
        state.finished = true;
        let last = state.publish_stable_text(true).unwrap();
        assert_eq!(format!("{first}{last}"), " hello");
        assert_eq!(state.step(String::new()).text, "hello");
    }

    #[test]
    fn retained_sequences_take_the_prompt_and_budget_within_the_ceiling() {
        assert!(GraniteSpeech5AsrModel::retained_sequence_fits(1_328, 1_200));
        assert!(!GraniteSpeech5AsrModel::retained_sequence_fits(
            1_329, 1_200
        ));
        assert!(!GraniteSpeech5AsrModel::retained_sequence_fits(
            100,
            usize::MAX
        ));
    }
}
