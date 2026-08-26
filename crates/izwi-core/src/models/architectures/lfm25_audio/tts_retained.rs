//! Rollback-safe retained LFM2.5 Audio TTS continuation.

use std::sync::Arc;

use candle_core::{IndexOp, Tensor};

use crate::backends::state::TensorStateArena;
use crate::error::{Error, Result};
use crate::models::architectures::lfm2::backbone::QuantizedLfm2Backbone;
use crate::models::shared::attention::physical::PhysicalPagedKvCache;
use crate::models::shared::chat::ChatMessage;

use super::audio_output::{Lfm25AudioHead, Lfm25SampledAudioFrame};
use super::model::{has_token_repetition_loop, text_delta};
use super::sampling::{sample_from_logits, Lfm25AudioGenerationConfig, SimpleRng};
use super::state::{Lfm25AudioRetainedCheckpoint, Lfm25AudioRetainedState};
use super::tokenizer::{Lfm25SpecialTokenIds, Lfm25TextTokenizer};

#[derive(Debug, Clone)]
pub(crate) struct Lfm25AudioPreparedTtsArtifact {
    pub(crate) model_load_nonce: u64,
    pub(crate) prompt_embeddings: Tensor,
    pub(crate) prompt_tokens: usize,
    pub(crate) source_messages: Arc<[ChatMessage]>,
    pub(crate) materialized_tensor_elements: u64,
    pub(crate) retained_resident_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Lfm25AudioTtsPrefillStep {
    pub(crate) consumed_tokens: usize,
    pub(crate) prefill_cursor: usize,
    pub(crate) prompt_tokens: usize,
    pub(crate) complete: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Lfm25AudioTtsDecodeStep {
    pub(crate) delta: String,
    pub(crate) text: String,
    pub(crate) audio_frame: Option<Vec<u32>>,
    pub(crate) tokens_generated: usize,
    pub(crate) finished: bool,
    pub(crate) in_audio: bool,
}

#[derive(Clone)]
struct Lfm25AudioTtsHostCheckpoint {
    rng: SimpleRng,
    prefill_cursor: usize,
    last_hidden: Option<Tensor>,
    logits: Option<Tensor>,
    visible_text_ids: Vec<u32>,
    visible_text: String,
    audio_codes: Vec<Vec<u32>>,
    tokens_generated: usize,
    in_audio: bool,
    finished: bool,
}

pub(crate) struct Lfm25AudioTtsQuantumCheckpoint {
    retained: Lfm25AudioRetainedCheckpoint,
    host: Lfm25AudioTtsHostCheckpoint,
    has_depthformer: bool,
}

pub(crate) struct Lfm25AudioTtsRetainedState {
    artifact: Arc<Lfm25AudioPreparedTtsArtifact>,
    retained: Lfm25AudioRetainedState,
    generation: Lfm25AudioGenerationConfig,
    rng: SimpleRng,
    specials: Lfm25SpecialTokenIds,
    vocab_limit: usize,
    max_new_tokens: usize,
    prefill_cursor: usize,
    last_hidden: Option<Tensor>,
    logits: Option<Tensor>,
    visible_text_ids: Vec<u32>,
    visible_text: String,
    audio_codes: Vec<Vec<u32>>,
    tokens_generated: usize,
    in_audio: bool,
    finished: bool,
}

impl Lfm25AudioTtsRetainedState {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        artifact: Arc<Lfm25AudioPreparedTtsArtifact>,
        retained: Lfm25AudioRetainedState,
        expected_model_load_nonce: u64,
        generation: Lfm25AudioGenerationConfig,
        specials: Lfm25SpecialTokenIds,
        vocab_limit: usize,
        codebooks: usize,
        requested_max_new_tokens: usize,
        context_limit: usize,
    ) -> Result<Self> {
        if artifact.model_load_nonce != expected_model_load_nonce
            || retained.main_position() != 0
            || vocab_limit == 0
            || codebooks == 0
        {
            return Err(Error::InvalidInput(
                "LFM2.5 Audio retained TTS state has invalid identity".into(),
            ));
        }
        let available = context_limit
            .checked_sub(artifact.prompt_tokens)
            .filter(|available| *available > 0)
            .ok_or_else(|| {
                Error::InvalidInput("LFM2.5 Audio TTS prompt leaves no decode capacity".into())
            })?;
        Ok(Self {
            artifact,
            retained,
            generation,
            rng: SimpleRng::new(generation.seed),
            specials,
            vocab_limit,
            max_new_tokens: requested_max_new_tokens.max(1).min(available),
            prefill_cursor: 0,
            last_hidden: None,
            logits: None,
            visible_text_ids: Vec::new(),
            visible_text: String::new(),
            audio_codes: vec![Vec::new(); codebooks],
            tokens_generated: 0,
            in_audio: false,
            finished: false,
        })
    }

    pub(crate) fn reset_and_begin_quantum(
        &mut self,
        main: &PhysicalPagedKvCache,
        depthformer: Option<&mut PhysicalPagedKvCache>,
    ) -> Result<Lfm25AudioTtsQuantumCheckpoint> {
        let host = self.host_checkpoint();
        let has_depthformer = self.decode_needs_depthformer();
        let retained = if has_depthformer {
            let depthformer = depthformer.ok_or_else(|| {
                Error::InvalidInput("LFM2.5 Audio TTS audio row lacks Depthformer state".into())
            })?;
            self.retained.reset_depthformer_frame(depthformer)?;
            self.retained.begin_tts_frame_quantum(main, depthformer)?
        } else {
            if depthformer.is_some() {
                return Err(Error::InvalidInput(
                    "LFM2.5 Audio TTS text row received Depthformer state".into(),
                ));
            }
            self.retained.begin_main_quantum(main)?
        };
        Ok(Lfm25AudioTtsQuantumCheckpoint {
            retained,
            host,
            has_depthformer,
        })
    }

    pub(crate) fn commit_quantum(
        &mut self,
        main: &PhysicalPagedKvCache,
        depthformer: Option<&PhysicalPagedKvCache>,
        checkpoint: &Lfm25AudioTtsQuantumCheckpoint,
    ) -> Result<()> {
        if checkpoint.has_depthformer != depthformer.is_some() {
            return Err(Error::InferenceError(
                "LFM2.5 Audio TTS commit crossed subphase state".into(),
            ));
        }
        self.retained
            .commit_quantum(main, depthformer, &checkpoint.retained)
    }

    pub(crate) fn rollback_quantum(
        &mut self,
        main: &mut PhysicalPagedKvCache,
        depthformer: Option<&mut PhysicalPagedKvCache>,
        checkpoint: &Lfm25AudioTtsQuantumCheckpoint,
    ) -> Result<()> {
        self.retained
            .rollback_quantum(main, depthformer, &checkpoint.retained)?;
        self.restore_host(&checkpoint.host);
        Ok(())
    }

    pub(crate) fn bind_tensor_sequence(&mut self, sequence: u64) -> Result<()> {
        self.retained.bind_tensor_sequence(sequence)
    }

    pub(crate) fn restore_shortconv(&mut self, arena: &TensorStateArena) -> Result<()> {
        self.retained.restore_shortconv(arena)
    }

    pub(crate) fn stage_shortconv(
        &mut self,
        arena: &TensorStateArena,
        transaction: u64,
    ) -> Result<()> {
        self.retained.stage_shortconv(arena, transaction)
    }

    pub(crate) fn prefill_step(
        &mut self,
        backbone: &QuantizedLfm2Backbone,
        main: &mut PhysicalPagedKvCache,
        checkpoint: &Lfm25AudioTtsQuantumCheckpoint,
        max_tokens: usize,
    ) -> Result<Lfm25AudioTtsPrefillStep> {
        self.authenticate_host(checkpoint)?;
        if checkpoint.has_depthformer
            || self.finished
            || self.prefill_cursor >= self.artifact.prompt_tokens
            || max_tokens == 0
        {
            return Err(Error::InvalidInput(
                "LFM2.5 Audio TTS prefill quantum is invalid".into(),
            ));
        }
        let consumed = max_tokens.min(self.artifact.prompt_tokens - self.prefill_cursor);
        let embeds = self
            .artifact
            .prompt_embeddings
            .narrow(1, self.prefill_cursor, consumed)?;
        let hidden = backbone.forward_embeds_retained(
            &embeds,
            self.prefill_cursor,
            main,
            &mut self.retained.shortconv,
        )?;
        self.prefill_cursor += consumed;
        if self.prefill_cursor == self.artifact.prompt_tokens {
            self.last_hidden = Some(last_hidden(&hidden)?);
            self.logits = Some(backbone.project_last_hidden(&hidden)?);
        }
        Ok(Lfm25AudioTtsPrefillStep {
            consumed_tokens: consumed,
            prefill_cursor: self.prefill_cursor,
            prompt_tokens: self.artifact.prompt_tokens,
            complete: self.prefill_cursor == self.artifact.prompt_tokens,
        })
    }

    pub(crate) fn decode_step(
        &mut self,
        backbone: &QuantizedLfm2Backbone,
        tokenizer: &Lfm25TextTokenizer,
        audio_head: &Lfm25AudioHead,
        main: &mut PhysicalPagedKvCache,
        depthformer: Option<&mut PhysicalPagedKvCache>,
        checkpoint: &Lfm25AudioTtsQuantumCheckpoint,
    ) -> Result<Lfm25AudioTtsDecodeStep> {
        self.authenticate_host(checkpoint)?;
        if self.finished || self.prefill_cursor != self.artifact.prompt_tokens {
            return Err(Error::InvalidInput(
                "LFM2.5 Audio TTS decode is not active".into(),
            ));
        }
        if self.in_audio {
            let depthformer = depthformer.ok_or_else(|| {
                Error::InvalidInput("LFM2.5 Audio TTS audio step lacks Depthformer state".into())
            })?;
            let hidden = self.last_hidden.as_ref().ok_or_else(|| {
                Error::InferenceError("LFM2.5 Audio TTS has no main hidden state".into())
            })?;
            let frame = audio_head.sample_audio_frame_retained(
                hidden,
                &self.generation.audio,
                &mut self.rng,
                depthformer,
            )?;
            self.finish_audio_frame(backbone, audio_head, main, frame)
        } else {
            if depthformer.is_some() || checkpoint.has_depthformer {
                return Err(Error::InvalidInput(
                    "LFM2.5 Audio TTS text step crossed Depthformer state".into(),
                ));
            }
            self.finish_text_token(backbone, tokenizer, main)
        }
    }

    pub(crate) fn decode_audio_batch(
        backbone: &QuantizedLfm2Backbone,
        audio_head: &Lfm25AudioHead,
        states: &mut [&mut Self],
        mains: &mut [&mut PhysicalPagedKvCache],
        depths: &mut [&mut PhysicalPagedKvCache],
        checkpoints: &[&Lfm25AudioTtsQuantumCheckpoint],
    ) -> Result<Vec<Lfm25AudioTtsDecodeStep>> {
        let batch = states.len();
        if batch == 0 || mains.len() != batch || depths.len() != batch || checkpoints.len() != batch
        {
            return Err(Error::InvalidInput(
                "LFM2.5 Audio TTS frame batch rows do not match".into(),
            ));
        }
        if batch == 1 {
            states[0].authenticate_host(checkpoints[0])?;
            if !states[0].in_audio || !checkpoints[0].has_depthformer {
                return Err(Error::InvalidInput(
                    "LFM2.5 Audio scalar frame fallback contains a text row".into(),
                ));
            }
            let hidden = states[0].last_hidden.as_ref().ok_or_else(|| {
                Error::InferenceError("LFM2.5 Audio TTS row has no hidden state".into())
            })?;
            let frame = audio_head.sample_audio_frame_retained(
                hidden,
                &states[0].generation.audio,
                &mut states[0].rng,
                depths[0],
            )?;
            return Ok(vec![
                states[0].finish_audio_frame(backbone, audio_head, mains[0], frame)?
            ]);
        }
        for row in 0..batch {
            states[row].authenticate_host(checkpoints[row])?;
            if !states[row].in_audio || !checkpoints[row].has_depthformer {
                return Err(Error::InvalidInput(
                    "LFM2.5 Audio native frame batch contains a text row".into(),
                ));
            }
        }
        let hidden = Tensor::cat(
            &states
                .iter()
                .map(|state| {
                    state.last_hidden.as_ref().ok_or_else(|| {
                        Error::InferenceError("LFM2.5 Audio TTS row has no hidden state".into())
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            0,
        )?;
        let configs = states
            .iter()
            .map(|state| state.generation.audio)
            .collect::<Vec<_>>();
        let mut rngs = states
            .iter_mut()
            .map(|state| &mut state.rng)
            .collect::<Vec<_>>();
        let frames =
            audio_head.sample_audio_frame_retained_batch(&hidden, &configs, &mut rngs, depths)?;
        drop(rngs);
        frames
            .into_iter()
            .enumerate()
            .map(|(row, frame)| {
                states[row].finish_audio_frame(backbone, audio_head, mains[row], frame)
            })
            .collect()
    }

    fn finish_text_token(
        &mut self,
        backbone: &QuantizedLfm2Backbone,
        tokenizer: &Lfm25TextTokenizer,
        main: &mut PhysicalPagedKvCache,
    ) -> Result<Lfm25AudioTtsDecodeStep> {
        let logits = self
            .logits
            .as_ref()
            .ok_or_else(|| Error::InferenceError("LFM2.5 Audio TTS has no text logits".into()))?;
        let next = sample_from_logits(
            logits,
            self.vocab_limit,
            &self.generation.text,
            &mut self.rng,
        )?;
        self.tokens_generated += 1;
        if next == self.specials.im_end
            || next == self.specials.eos
            || self.specials.eos_alt == Some(next)
        {
            self.finished = true;
            return Ok(self.outcome(String::new(), None));
        }
        let prior = self.visible_text.clone();
        if next == self.specials.audio_start {
            self.in_audio = true;
        } else if next != self.specials.text_end {
            self.visible_text_ids.push(next);
            self.visible_text = tokenizer.decode_text(&self.visible_text_ids)?;
        }
        let embeds = backbone.embed_tokens(&Tensor::from_vec(
            vec![next],
            (1, 1),
            self.artifact.prompt_embeddings.device(),
        )?)?;
        let hidden = backbone.forward_embeds_retained(
            &embeds,
            self.retained.main_position(),
            main,
            &mut self.retained.shortconv,
        )?;
        self.last_hidden = Some(last_hidden(&hidden)?);
        self.logits = Some(backbone.project_last_hidden(&hidden)?);
        if has_token_repetition_loop(&self.visible_text_ids)
            || self.tokens_generated >= self.max_new_tokens
        {
            self.finished = true;
        }
        Ok(self.outcome(text_delta(&prior, &self.visible_text), None))
    }

    fn finish_audio_frame(
        &mut self,
        backbone: &QuantizedLfm2Backbone,
        audio_head: &Lfm25AudioHead,
        main: &mut PhysicalPagedKvCache,
        frame: Lfm25SampledAudioFrame,
    ) -> Result<Lfm25AudioTtsDecodeStep> {
        let tokens = frame.tokens()?;
        let is_end = tokens.first().copied() == Some(audio_head.audio_end_token_id());
        self.tokens_generated += 1;
        if !is_end {
            for (codebook, token) in tokens.iter().copied().enumerate() {
                self.audio_codes[codebook].push(token);
            }
        }
        let hidden = backbone.forward_embeds_retained(
            frame.embedding(),
            self.retained.main_position(),
            main,
            &mut self.retained.shortconv,
        )?;
        self.last_hidden = Some(last_hidden(&hidden)?);
        if is_end {
            self.in_audio = false;
            self.logits = Some(backbone.project_last_hidden(&hidden)?);
        }
        if self.tokens_generated >= self.max_new_tokens {
            self.finished = true;
        }
        Ok(self.outcome(String::new(), (!is_end).then_some(tokens)))
    }

    fn outcome(&self, delta: String, audio_frame: Option<Vec<u32>>) -> Lfm25AudioTtsDecodeStep {
        Lfm25AudioTtsDecodeStep {
            delta,
            text: self.visible_text.trim().to_string(),
            audio_frame,
            tokens_generated: self.tokens_generated,
            finished: self.finished,
            in_audio: self.in_audio,
        }
    }

    fn authenticate_host(&self, checkpoint: &Lfm25AudioTtsQuantumCheckpoint) -> Result<()> {
        let host = &checkpoint.host;
        if self.prefill_cursor != host.prefill_cursor
            || self.tokens_generated != host.tokens_generated
            || self.in_audio != host.in_audio
            || self.finished != host.finished
            || self.visible_text_ids != host.visible_text_ids
            || self.visible_text != host.visible_text
            || self.audio_codes != host.audio_codes
        {
            return Err(Error::InferenceError(
                "LFM2.5 Audio TTS quantum was already advanced".into(),
            ));
        }
        Ok(())
    }

    fn host_checkpoint(&self) -> Lfm25AudioTtsHostCheckpoint {
        Lfm25AudioTtsHostCheckpoint {
            rng: self.rng.clone(),
            prefill_cursor: self.prefill_cursor,
            last_hidden: self.last_hidden.clone(),
            logits: self.logits.clone(),
            visible_text_ids: self.visible_text_ids.clone(),
            visible_text: self.visible_text.clone(),
            audio_codes: self.audio_codes.clone(),
            tokens_generated: self.tokens_generated,
            in_audio: self.in_audio,
            finished: self.finished,
        }
    }

    fn restore_host(&mut self, host: &Lfm25AudioTtsHostCheckpoint) {
        self.rng = host.rng.clone();
        self.prefill_cursor = host.prefill_cursor;
        self.last_hidden = host.last_hidden.clone();
        self.logits = host.logits.clone();
        self.visible_text_ids.clone_from(&host.visible_text_ids);
        self.visible_text.clone_from(&host.visible_text);
        self.audio_codes.clone_from(&host.audio_codes);
        self.tokens_generated = host.tokens_generated;
        self.in_audio = host.in_audio;
        self.finished = host.finished;
    }

    pub(crate) fn prompt_tokens(&self) -> usize {
        self.artifact.prompt_tokens
    }
    pub(crate) fn prefill_cursor(&self) -> usize {
        self.prefill_cursor
    }
    pub(crate) fn main_position(&self) -> usize {
        self.retained.main_position()
    }
    pub(crate) fn decode_needs_depthformer(&self) -> bool {
        self.prefill_cursor == self.artifact.prompt_tokens && self.in_audio && !self.finished
    }
    pub(crate) fn finished(&self) -> bool {
        self.finished
    }
    pub(crate) fn text(&self) -> &str {
        self.visible_text.trim()
    }
    pub(crate) fn tokens_generated(&self) -> usize {
        self.tokens_generated
    }
    pub(crate) fn audio_codes(&self) -> &[Vec<u32>] {
        &self.audio_codes
    }
}

fn last_hidden(hidden: &Tensor) -> Result<Tensor> {
    let (_, seq_len, _) = hidden.dims3()?;
    if seq_len == 0 {
        return Err(Error::InferenceError(
            "LFM2.5 Audio hidden state is empty".into(),
        ));
    }
    hidden
        .i((.., seq_len - 1, ..))?
        .unsqueeze(1)
        .map_err(Error::from)
}
