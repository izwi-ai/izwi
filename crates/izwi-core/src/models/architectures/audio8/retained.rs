//! Rollback-safe staged Audio8 TTS generation.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::backends::kv::KvWriteBatchCompletion;
use crate::error::{Error, Result};
use crate::models::shared::attention::physical::PhysicalPagedKvCache;

use super::{
    append_generated_frame, elapsed_ms, generated_frame_prompt, sample_semantic_rows,
    semantic_rows_on_device, Audio8ConditioningPrompt, Audio8GenerationOutput,
    Audio8GenerationParams, Audio8Reference, Audio8Sampler, Audio8SemanticRow,
    Audio8SemanticSampler, Audio8SlowOutput, Audio8StopReason, Audio8TtsGenerationDiagnostics,
    Audio8TtsModel, Audio8VqCodes, RAS_WIN_SIZE,
};

static NEXT_AUDIO8_STATE_ID: AtomicU64 = AtomicU64::new(1);

/// Host ownership and accelerator workspace required before reference preparation.
/// Heap bytes exclude allocator metadata, consistently with other resource charges.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Audio8PreparationMemory {
    pub host_transient_bytes: u64,
    pub retained_artifact_bytes: u64,
    pub accelerator_workspace_bytes: u64,
}

impl Audio8PreparationMemory {
    pub(crate) fn host_bytes(self) -> Result<u64> {
        checked_bytes_sum(&[self.host_transient_bytes, self.retained_artifact_bytes])
    }
}

fn checked_bytes_sum(values: &[u64]) -> Result<u64> {
    values
        .iter()
        .try_fold(0u64, |sum, value| sum.checked_add(*value))
        .ok_or_else(|| Error::Overloaded("Audio8 preparation byte count overflow".into()))
}

fn allocation_bytes(count: usize, element_size: usize) -> Result<u64> {
    count
        .checked_mul(element_size)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| Error::Overloaded("Audio8 preparation allocation overflow".into()))
}

fn artifact_bound(num_codebooks: usize, tokens: usize) -> Result<u64> {
    let rows = num_codebooks
        .checked_add(1)
        .ok_or_else(|| Error::Overloaded("Audio8 artifact rows overflow".into()))?;
    checked_bytes_sum(&[
        allocation_bytes(rows, std::mem::size_of::<Vec<u32>>())?,
        allocation_bytes(
            rows.checked_mul(tokens)
                .ok_or_else(|| Error::Overloaded("Audio8 artifact shape overflow".into()))?,
            4,
        )?,
        allocation_bytes(tokens, std::mem::size_of::<bool>())?,
        // Include the Arc allocation's object and strong/weak counters once.
        allocation_bytes(
            1,
            std::mem::size_of::<Audio8PreparedArtifact>() + 2 * std::mem::size_of::<usize>(),
        )?,
    ])
}

#[derive(Clone)]
pub(crate) struct Audio8PreparedArtifact {
    model_identity: u64,
    prompt: Audio8ConditioningPrompt,
    reference_encode_ms: f32,
    prompt_build_ms: f32,
}

impl std::fmt::Debug for Audio8PreparedArtifact {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Audio8PreparedArtifact")
            .field("model_identity", &self.model_identity)
            .field("prompt_tokens", &self.prompt.prompt_length)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Audio8RetainedStep {
    Prefill {
        consumed: usize,
        position: usize,
        complete: bool,
    },
    Frame {
        frames_generated: usize,
    },
    Finished {
        frames_generated: usize,
    },
}

pub(crate) struct Audio8RetainedState {
    state_id: u64,
    model_identity: u64,
    artifact: Arc<Audio8PreparedArtifact>,
    params: Audio8GenerationParams,
    slow_cache: PhysicalPagedKvCache,
    slow_position: usize,
    slow_output: Option<Audio8SlowOutput>,
    semantic_sampler: Audio8SemanticSampler,
    fast_sampler: Audio8Sampler,
    generated_codebooks: Vec<Vec<u32>>,
    recent_semantic_tokens: Vec<u32>,
    max_frames: usize,
    prefill_ms: f64,
    decode_ms: f64,
    sampling_ms: f64,
    fast_ar_ms: f64,
    slow_ar_ms: f64,
    prefill_steps: u32,
    decode_steps: u32,
    stop_reason: Audio8StopReason,
    finished: bool,
    active_quantum: Option<u64>,
    next_quantum: u64,
    staged_step: Option<Audio8RetainedStep>,
    completions_drained: bool,
}

pub(crate) struct Audio8RetainedCheckpoint {
    state_id: u64,
    quantum: u64,
    payload: Option<Audio8RetainedCheckpointPayload>,
}

impl Audio8RetainedCheckpoint {
    pub(crate) fn is_initial(&self) -> bool {
        self.payload
            .as_ref()
            .is_some_and(|payload| payload.slow_cache.is_none())
    }
}

struct Audio8RetainedCheckpointPayload {
    slow_cache: Option<PhysicalPagedKvCache>,
    slow_position: usize,
    slow_output: Option<Audio8SlowOutput>,
    semantic_sampler: Audio8SemanticSampler,
    fast_sampler: Audio8Sampler,
    generated_frames: usize,
    recent_semantic_tokens: Vec<u32>,
    stop_reason: Audio8StopReason,
    finished: bool,
    staged_step: Option<Audio8RetainedStep>,
    completions_drained: bool,
}

impl Audio8TtsModel {
    pub(crate) fn preparation_memory(
        &self,
        text: &str,
        reference: Option<&Audio8Reference>,
        context_limit: usize,
    ) -> Result<Audio8PreparationMemory> {
        validate_preparation_inputs(text, reference)?;
        let Some(reference) = reference else {
            return plain_preparation_memory(&self.config, text.len(), context_limit);
        };
        preparation_memory_for_geometry(
            &self.config,
            text.len(),
            reference.text.capacity(),
            reference.audio_samples.len(),
            reference.audio_samples.capacity(),
            reference.sample_rate,
            context_limit,
        )
    }

    pub(crate) fn prepare_retained_artifact(
        &self,
        text: &str,
        reference: Option<Audio8Reference>,
    ) -> Result<Arc<Audio8PreparedArtifact>> {
        self.prepare_retained_artifact_with_cancel(
            text,
            reference,
            self.config.max_seq_len,
            &|| Ok(()),
        )
    }

    /// Tokenize the prompt and, with a `reference`, encode its audio to
    /// codec frames (cached by content). Without one the model speaks in a
    /// voice it picks, which the request's seed makes repeatable.
    pub(crate) fn prepare_retained_artifact_with_cancel(
        &self,
        text: &str,
        reference: Option<Audio8Reference>,
        context_limit: usize,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Arc<Audio8PreparedArtifact>> {
        check()?;
        self.preparation_memory(text, reference.as_ref(), context_limit)?;
        let runtime = self.native_runtime()?;
        let mut reference_encode_ms = 0.0;
        let reference_codes = match &reference {
            Some(reference) => {
                let started = Instant::now();
                let (codes, cache_hit) = self.reference_cache.get_or_encode(
                    &reference.audio_samples,
                    reference.sample_rate,
                    check,
                    || {
                        let encode_started = Instant::now();
                        let codes = runtime.dac.encode_reference_audio_with_cancel(
                            &reference.audio_samples,
                            reference.sample_rate,
                            check,
                        );
                        reference_encode_ms = elapsed_ms(encode_started);
                        codes
                    },
                )?;
                tracing::debug!(
                    reference_cache_hit = cache_hit,
                    reference_lookup_total_ms = elapsed_ms(started),
                    reference_encode_ms,
                    "Audio8 reference code preparation"
                );
                Some(codes)
            }
            None => None,
        };
        let started = Instant::now();
        let prompt = runtime.tokenizer.build_speech_prompt_bounded(
            &self.config,
            reference
                .as_ref()
                .map(|reference| reference.text.as_str())
                .zip(reference_codes),
            text,
            context_limit,
        )?;
        let prompt_build_ms = elapsed_ms(started);
        check()?;
        if prompt.prompt_length >= self.config.max_seq_len {
            return Err(Error::InvalidInput(format!(
                "Audio8 prompt length {} exceeds max_seq_len {}",
                prompt.prompt_length, self.config.max_seq_len
            )));
        }
        Ok(Arc::new(Audio8PreparedArtifact {
            model_identity: self.model_identity,
            prompt,
            reference_encode_ms,
            prompt_build_ms,
        }))
    }

    pub(crate) fn new_retained_state(
        &self,
        artifact: Arc<Audio8PreparedArtifact>,
        params: Audio8GenerationParams,
        slow_cache: PhysicalPagedKvCache,
        max_sequence_tokens: usize,
    ) -> Result<Audio8RetainedState> {
        params.validate()?;
        if artifact.model_identity != self.model_identity {
            return Err(Error::InvalidInput(
                "Audio8 prepared artifact belongs to another model load".into(),
            ));
        }
        if slow_cache.context_len() != 0 {
            return Err(Error::InvalidInput(
                "Audio8 retained caches must begin empty".into(),
            ));
        }
        let max_frames = super::effective_frame_budget(
            artifact.prompt.prompt_length,
            self.config.max_seq_len,
            max_sequence_tokens,
            params.max_frames,
        )?;
        let semantic_sampler = Audio8SemanticSampler::from_params(&params);
        let fast_sampler = Audio8Sampler::with_top_k(
            params.temperature,
            params.top_p,
            params.top_k,
            params.seed.wrapping_add(1),
        );
        Ok(Audio8RetainedState {
            state_id: next_state_id()?,
            model_identity: self.model_identity,
            artifact,
            params,
            slow_cache,
            slow_position: 0,
            slow_output: None,
            semantic_sampler,
            fast_sampler,
            generated_codebooks: vec![Vec::new(); self.config.num_codebooks],
            recent_semantic_tokens: Vec::with_capacity(RAS_WIN_SIZE),
            max_frames,
            prefill_ms: 0.0,
            decode_ms: 0.0,
            sampling_ms: 0.0,
            fast_ar_ms: 0.0,
            slow_ar_ms: 0.0,
            prefill_steps: 0,
            decode_steps: 0,
            stop_reason: Audio8StopReason::FrameLimit,
            finished: false,
            active_quantum: None,
            next_quantum: 1,
            staged_step: None,
            completions_drained: true,
        })
    }

    pub(crate) fn new_retained_state_in_quantum(
        &self,
        artifact: Arc<Audio8PreparedArtifact>,
        params: Audio8GenerationParams,
        slow_cache: PhysicalPagedKvCache,
        max_sequence_tokens: usize,
    ) -> Result<(Audio8RetainedState, Audio8RetainedCheckpoint)> {
        // The cache contains only this quantum's pages, not the full sequence.
        let mut state =
            self.new_retained_state(artifact, params, slow_cache, max_sequence_tokens)?;
        state.active_quantum = Some(1);
        state.next_quantum = 2;
        let checkpoint = Audio8RetainedCheckpoint {
            state_id: state.state_id,
            quantum: 1,
            payload: Some(Audio8RetainedCheckpointPayload {
                slow_cache: None,
                slow_position: 0,
                slow_output: None,
                semantic_sampler: state.semantic_sampler.clone(),
                fast_sampler: state.fast_sampler.clone(),
                generated_frames: 0,
                recent_semantic_tokens: Vec::new(),
                stop_reason: Audio8StopReason::FrameLimit,
                finished: false,
                staged_step: None,
                completions_drained: true,
            }),
        };
        Ok((state, checkpoint))
    }

    pub(crate) fn retained_prefill_step(
        &self,
        state: &mut Audio8RetainedState,
        max_tokens: usize,
    ) -> Result<Audio8RetainedStep> {
        let started = Instant::now();
        let result = self.retained_prefill_step_inner(state, max_tokens);
        state.prefill_ms += started.elapsed().as_secs_f64() * 1000.0;
        state.prefill_steps = state.prefill_steps.saturating_add(1);
        result
    }

    fn retained_prefill_step_inner(
        &self,
        state: &mut Audio8RetainedState,
        max_tokens: usize,
    ) -> Result<Audio8RetainedStep> {
        self.validate_retained_state(state)?;
        state.require_clean_quantum()?;
        if max_tokens == 0 {
            return Err(Error::InvalidInput(
                "Audio8 prefill quantum must be nonzero".into(),
            ));
        }
        let remaining = state.artifact.prompt.prompt_length - state.slow_position;
        let consumed = remaining.min(max_tokens);
        if consumed > 0 {
            let prompt = slice_prompt(&state.artifact.prompt, state.slow_position, consumed)?;
            let runtime = self.native_runtime()?;
            let embeds = runtime.slow.embed_prompt(&prompt)?;
            state.slow_output = Some(runtime.slow.forward_embeds(
                &embeds,
                state.slow_position,
                &mut state.slow_cache,
                false,
            )?);
            state.slow_position += consumed;
            state.completions_drained = false;
        }
        state.stage(Audio8RetainedStep::Prefill {
            consumed,
            position: state.slow_position,
            complete: state.slow_position == state.artifact.prompt.prompt_length,
        })
    }

    pub(crate) fn retained_decode_step(
        &self,
        state: &mut Audio8RetainedState,
        fast_cache: &mut PhysicalPagedKvCache,
    ) -> Result<Audio8RetainedStep> {
        let started = Instant::now();
        let result = self.retained_decode_step_inner(state, fast_cache);
        state.decode_ms += started.elapsed().as_secs_f64() * 1000.0;
        state.decode_steps = state.decode_steps.saturating_add(1);
        result
    }

    fn retained_decode_step_inner(
        &self,
        state: &mut Audio8RetainedState,
        fast_cache: &mut PhysicalPagedKvCache,
    ) -> Result<Audio8RetainedStep> {
        self.validate_retained_state(state)?;
        state.require_clean_quantum()?;
        if state.slow_position < state.artifact.prompt.prompt_length {
            return Err(Error::InferenceError(
                "Audio8 decode cannot run before prefill completes".into(),
            ));
        }
        if state.finished {
            return state.stage_finished();
        }
        let runtime = self.native_runtime()?;
        let slow = state.slow_output.as_ref().ok_or_else(|| {
            Error::InferenceError("Audio8 retained state has no slow output".into())
        })?;
        let sampling_started = Instant::now();
        let device_candidates = semantic_rows_on_device(&slow.logits);
        let allow_im_end = state.frames_generated() > 0;
        // The batched path at width one: a row draws alone as it does beside
        // others.
        let semantic_index = sample_semantic_rows(
            runtime,
            &mut [Audio8SemanticRow {
                logits: &slow.logits,
                allow_im_end,
                previous: &state.recent_semantic_tokens,
                sampler: &mut state.semantic_sampler,
            }],
            device_candidates,
        )?[0];
        state.sampling_ms += sampling_started.elapsed().as_secs_f64() * 1000.0;
        let semantic = runtime.slow.token_id_from_logit(semantic_index)?;
        if semantic == runtime.tokenizer.specials().eos {
            // Complete the slow-token append authorized for this quantum even
            // when EOS terminates generation; every committed row needs its KV receipt.
            let mut values = vec![vec![0]; self.config.num_codebooks + 1];
            values[0][0] = semantic;
            let prompt = Audio8ConditioningPrompt {
                values,
                vq_mask: vec![false],
                prompt_length: 1,
            };
            let slow_started = Instant::now();
            let embeds = runtime.slow.embed_prompt(&prompt)?;
            state.slow_output = Some(runtime.slow.forward_embeds(
                &embeds,
                state.slow_position,
                &mut state.slow_cache,
                false,
            )?);
            state.slow_ar_ms += slow_started.elapsed().as_secs_f64() * 1000.0;
            state.slow_position += 1;
            state.completions_drained = false;
            state.stop_reason = Audio8StopReason::EndOfSpeech;
            state.finished = true;
            return state.stage_finished();
        }
        // The fast clock restarts inside this bounded invocation. Only complete
        // frames cross the scheduler's retained-state commit boundary.
        let fast_started = Instant::now();
        let frame = runtime.fast.generate_frame(
            semantic,
            &slow.hidden_states,
            &mut state.fast_sampler,
            fast_cache,
        )?;
        state.fast_ar_ms += fast_started.elapsed().as_secs_f64() * 1000.0;
        append_generated_frame(&mut state.generated_codebooks, &frame)?;
        let frames_generated = state.frames_generated();
        record_recent_semantic(
            &mut state.recent_semantic_tokens,
            semantic_index,
            frames_generated,
        );
        let slow_started = Instant::now();
        let frame_prompt = generated_frame_prompt(self.config.num_codebooks, &frame)?;
        let embeds = runtime.slow.embed_prompt(&frame_prompt)?;
        state.slow_output = Some(runtime.slow.forward_embeds(
            &embeds,
            state.slow_position,
            &mut state.slow_cache,
            false,
        )?);
        state.slow_ar_ms += slow_started.elapsed().as_secs_f64() * 1000.0;
        state.slow_position += 1;
        state.completions_drained = false;
        if state.frames_generated() >= state.max_frames {
            state.finished = true;
            return state.stage_finished();
        }
        state.stage(Audio8RetainedStep::Frame {
            frames_generated: state.frames_generated(),
        })
    }

    /// Execute a complete frame per row under caller-owned managed checkpoints.
    /// On any shared failure callers must roll back all participating rows.
    pub(crate) fn retained_decode_batch(
        &self,
        states: &mut [&mut Audio8RetainedState],
        fast_caches: &mut [&mut PhysicalPagedKvCache],
    ) -> Result<Vec<Audio8RetainedStep>> {
        if states.is_empty() || states.len() != fast_caches.len() {
            return Err(Error::InvalidInput(
                "Audio8 retained decode batch rows do not match".into(),
            ));
        }
        for state in states.iter() {
            self.validate_retained_state(state)?;
            state.require_clean_quantum()?;
            if state.finished
                || state.slow_position < state.artifact.prompt.prompt_length
                || state.slow_output.is_none()
            {
                return Err(Error::InvalidInput(
                    "Audio8 decode batch contains a non-ready row".into(),
                ));
            }
        }
        let started = Instant::now();
        let runtime = self.native_runtime()?;
        // Every row's semantic token from one device pass and one readback.
        let sampling_started = Instant::now();
        let semantic_indices = {
            let device_candidates = semantic_rows_on_device(
                &states[0]
                    .slow_output
                    .as_ref()
                    .expect("validated slow output")
                    .logits,
            );
            let mut rows = states
                .iter_mut()
                .map(|state| {
                    let state = &mut **state;
                    let allow_im_end = state.frames_generated() > 0;
                    Audio8SemanticRow {
                        allow_im_end,
                        logits: &state
                            .slow_output
                            .as_ref()
                            .expect("validated slow output")
                            .logits,
                        previous: &state.recent_semantic_tokens,
                        sampler: &mut state.semantic_sampler,
                    }
                })
                .collect::<Vec<_>>();
            sample_semantic_rows(runtime, &mut rows, device_candidates)?
        };
        let sampling_ms = sampling_started.elapsed().as_secs_f64() * 1000.0;
        let mut semantics = Vec::with_capacity(states.len());
        for (state, &index) in states.iter_mut().zip(&semantic_indices) {
            state.sampling_ms += sampling_ms;
            semantics.push(runtime.slow.token_id_from_logit(index)?);
        }
        let eos = runtime.tokenizer.specials().eos;
        let fast_started = Instant::now();
        let mut frames = std::iter::repeat_with(|| None)
            .take(states.len())
            .collect::<Vec<_>>();
        let active = semantics
            .iter()
            .enumerate()
            .filter_map(|(row, &token)| (token != eos).then_some(row))
            .collect::<Vec<_>>();
        if !active.is_empty() {
            let tokens = active.iter().map(|&row| semantics[row]).collect::<Vec<_>>();
            let hiddens = active
                .iter()
                .map(|&row| {
                    states[row]
                        .slow_output
                        .as_ref()
                        .expect("validated slow output")
                        .hidden_states
                        .clone()
                })
                .collect::<Vec<_>>();
            let hidden = candle_core::Tensor::cat(&hiddens, 0)?;
            let mut samplers = states
                .iter_mut()
                .enumerate()
                .filter_map(|(row, state)| {
                    (semantics[row] != eos).then_some(&mut state.fast_sampler)
                })
                .collect::<Vec<_>>();
            let mut caches = fast_caches
                .iter_mut()
                .enumerate()
                .filter_map(|(row, cache)| (semantics[row] != eos).then_some(&mut **cache))
                .collect::<Vec<_>>();
            let generated =
                runtime
                    .fast
                    .generate_frames_batch(&tokens, &hidden, &mut samplers, &mut caches)?;
            for (row, frame) in active.into_iter().zip(generated) {
                frames[row] = Some(frame);
            }
        }
        let fast_ms = fast_started.elapsed().as_secs_f64() * 1000.0;
        let mut prompts = Vec::with_capacity(states.len());
        for (row, frame) in frames.iter().enumerate() {
            if let Some(frame) = frame {
                prompts.push(generated_frame_prompt(self.config.num_codebooks, frame)?);
            } else {
                let mut values = vec![vec![0]; self.config.num_codebooks + 1];
                values[0][0] = semantics[row];
                prompts.push(Audio8ConditioningPrompt {
                    values,
                    vq_mask: vec![false],
                    prompt_length: 1,
                });
            }
        }
        let slow_started = Instant::now();
        let embeds = runtime.slow.embed_prompt_batch(&prompts)?;
        let mut caches = states
            .iter_mut()
            .map(|state| &mut state.slow_cache)
            .collect::<Vec<_>>();
        let outputs = runtime.slow.forward_embeds_batch(&embeds, &mut caches)?;
        let slow_ms = slow_started.elapsed().as_secs_f64() * 1000.0;
        let elapsed = started.elapsed().as_secs_f64() * 1000.0;
        let mut steps = Vec::with_capacity(states.len());
        for (row, ((state, output), frame)) in
            states.iter_mut().zip(outputs).zip(frames).enumerate()
        {
            state.slow_output = Some(output);
            state.slow_position += 1;
            state.completions_drained = false;
            state.decode_ms += elapsed;
            state.decode_steps = state.decode_steps.saturating_add(1);
            state.slow_ar_ms += slow_ms;
            if let Some(frame) = frame {
                state.fast_ar_ms += fast_ms;
                append_generated_frame(&mut state.generated_codebooks, &frame)?;
                let frames_generated = state.frames_generated();
                record_recent_semantic(
                    &mut state.recent_semantic_tokens,
                    semantic_indices[row],
                    frames_generated,
                );
                state.finished = state.frames_generated() >= state.max_frames;
            } else {
                state.stop_reason = Audio8StopReason::EndOfSpeech;
                state.finished = true;
            }
            steps.push(if state.finished {
                state.stage_finished()?
            } else {
                state.stage(Audio8RetainedStep::Frame {
                    frames_generated: state.frames_generated(),
                })?
            });
        }
        Ok(steps)
    }

    /// Pack only each row's admitted prompt chunk; never pad to the longest prompt.
    pub(crate) fn retained_prefill_batch(
        &self,
        states: &mut [&mut Audio8RetainedState],
        max_tokens: &[usize],
    ) -> Result<Vec<Audio8RetainedStep>> {
        if states.is_empty() || states.len() != max_tokens.len() || max_tokens.contains(&0) {
            return Err(Error::InvalidInput(
                "Audio8 prefill batch bounds do not match".into(),
            ));
        }
        let mut consumed = Vec::with_capacity(states.len());
        let mut prompts = Vec::with_capacity(states.len());
        for (state, &bound) in states.iter().zip(max_tokens) {
            self.validate_retained_state(state)?;
            state.require_clean_quantum()?;
            let count = state
                .artifact
                .prompt
                .prompt_length
                .saturating_sub(state.slow_position)
                .min(bound);
            if count == 0 {
                return Err(Error::InvalidInput(
                    "Audio8 prefill batch row has no remaining prompt".into(),
                ));
            }
            prompts.push(slice_prompt(
                &state.artifact.prompt,
                state.slow_position,
                count,
            )?);
            consumed.push(count);
        }
        let started = Instant::now();
        let runtime = self.native_runtime()?;
        let embeds = runtime.slow.embed_prompt_batch(&prompts)?;
        let mut caches = states
            .iter_mut()
            .map(|state| &mut state.slow_cache)
            .collect::<Vec<_>>();
        let outputs = runtime.slow.forward_embeds_batch(&embeds, &mut caches)?;
        let elapsed = started.elapsed().as_secs_f64() * 1000.0;
        states
            .iter_mut()
            .zip(outputs)
            .zip(consumed)
            .map(|((state, output), consumed)| {
                state.slow_output = Some(output);
                state.slow_position += consumed;
                state.completions_drained = false;
                state.prefill_ms += elapsed;
                state.prefill_steps = state.prefill_steps.saturating_add(1);
                state.stage(Audio8RetainedStep::Prefill {
                    consumed,
                    position: state.slow_position,
                    complete: state.slow_position == state.artifact.prompt.prompt_length,
                })
            })
            .collect()
    }

    pub(crate) fn decode_retained_audio_chunk(
        &self,
        state: &Audio8RetainedState,
        codec_state: &mut super::dac::Audio8DacStreamState,
        max_frames: usize,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Vec<f32>> {
        self.validate_retained_state(state)?;
        if state.active_quantum.is_some()
            || max_frames == 0
            || max_frames > super::AUDIO8_AUDIO_CHUNK_FRAMES
        {
            return Err(Error::InvalidInput(
                "Audio8 codec requires bounded committed frames".into(),
            ));
        }
        let start = codec_state.decoded_frames();
        let end = start
            .saturating_add(max_frames)
            .min(state.frames_generated());
        if start >= end {
            return Err(Error::InvalidInput(
                "Audio8 codec has no committed frames to decode".into(),
            ));
        }
        let codes = state
            .generated_codebooks
            .iter()
            .map(|row| row[start..end].to_vec())
            .collect::<Vec<_>>();
        let mut next = codec_state.clone();
        let samples = self
            .native_runtime()?
            .dac
            .push_frames(&mut next, &codes, check)?;
        if samples.len() != (end - start) * 2048 || samples.iter().any(|value| !value.is_finite()) {
            return Err(Error::InferenceError(
                "Audio8 streaming codec produced invalid PCM".into(),
            ));
        }
        if next.retained_tensor_bytes()
            > super::dac::Audio8DacConfig::current().streaming_history_bound_bytes()?
        {
            return Err(Error::InferenceError(
                "Audio8 streaming codec exceeded retained history bound".into(),
            ));
        }
        *codec_state = next;
        Ok(samples)
    }

    pub(crate) fn decode_retained_audio_batch(
        &self,
        states: &[&Audio8RetainedState],
        codecs: &mut [super::dac::Audio8DacStreamState],
        max_frames: &[usize],
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Vec<Vec<f32>>> {
        if states.is_empty() || states.len() != codecs.len() || states.len() != max_frames.len() {
            return Err(Error::InvalidInput(
                "Audio8 retained codec batch rows do not match".into(),
            ));
        }
        // Each row's frames as its chunks ([`super::audio8_codec_chunks`]):
        // a row stops at a chunk boundary unless it is finished, and every
        // chunk is its own push, so a request's pushes, and so its PCM, are
        // the same whenever and beside whichever rows its quanta run.
        let mut chunks = Vec::with_capacity(states.len());
        let mut counts = Vec::with_capacity(states.len());
        for ((state, codec), &max) in states.iter().zip(codecs.iter()).zip(max_frames) {
            check()?;
            self.validate_retained_state(state)?;
            if state.active_quantum.is_some()
                || max == 0
                || max > super::AUDIO8_CODEC_QUANTUM_FRAMES
            {
                return Err(Error::InvalidInput(
                    "Audio8 codec batch requires committed bounded rows".into(),
                ));
            }
            let start = codec.decoded_frames();
            let generated = state.frames_generated();
            // Never ends mid-chunk before the request's last frame.
            let end = super::audio8_codec_quantum_end(start, generated, state.finished, max);
            if start >= end {
                return Err(Error::InvalidInput(
                    "Audio8 codec batch row has no pending frames".into(),
                ));
            }
            chunks.push(super::audio8_codec_chunks(start, end));
            counts.push(end - start);
        }
        let dac = &self.native_runtime()?.dac;
        let mut next = codecs.to_vec();
        let mut outputs = vec![Vec::new(); states.len()];
        // Round `r` pushes every row's `r`-th chunk together.
        for round in 0..chunks.iter().map(Vec::len).max().unwrap_or(0) {
            let rows = (0..states.len())
                .filter(|&row| chunks[row].len() > round)
                .collect::<Vec<_>>();
            let mut round_states = rows
                .iter()
                .map(|&row| next[row].clone())
                .collect::<Vec<_>>();
            let codes = rows
                .iter()
                .map(|&row| {
                    let (from, to) = chunks[row][round];
                    states[row]
                        .generated_codebooks
                        .iter()
                        .map(|book| book[from..to].to_vec())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let pcm = dac.push_frames_batch(&mut round_states, &codes, check)?;
            for ((&row, state), pcm) in rows.iter().zip(round_states).zip(pcm) {
                next[row] = state;
                outputs[row].extend(pcm);
            }
        }
        let history_bound =
            super::dac::Audio8DacConfig::current().streaming_history_bound_bytes()?;
        for ((output, codec), count) in outputs.iter().zip(&next).zip(counts) {
            if output.len() != count * 2048
                || output.iter().any(|sample| !sample.is_finite())
                || codec.retained_tensor_bytes() > history_bound
            {
                return Err(Error::InferenceError(
                    "Audio8 batched codec produced invalid PCM or history usage".into(),
                ));
            }
        }
        check()?;
        codecs.clone_from_slice(&next);
        Ok(outputs)
    }

    pub(crate) fn finalize_retained_state(
        &self,
        state: &Audio8RetainedState,
    ) -> Result<Audio8GenerationOutput> {
        self.finalize_retained_state_with_cancel(state, &|| Ok(()))
    }

    pub(crate) fn finalize_retained_state_with_cancel(
        &self,
        state: &Audio8RetainedState,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Audio8GenerationOutput> {
        check()?;
        self.validate_retained_state(state)?;
        if !state.finished || state.active_quantum.is_some() {
            return Err(Error::InferenceError(
                "Audio8 finalize requires a terminal committed frame boundary".into(),
            ));
        }
        state.require_complete()?;
        let frames_generated = state.frames_generated();
        if frames_generated == 0 {
            return Err(Error::InferenceError(
                "Audio8 generation produced no audio frames".into(),
            ));
        }
        let started = Instant::now();
        let samples = self.native_runtime()?.dac.decode_vq_codes_with_cancel(
            &Audio8VqCodes {
                codebooks: state.generated_codebooks.clone(),
            },
            check,
        )?;
        check()?;
        let dac_decode_ms = elapsed_ms(started);
        if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
            return Err(Error::InferenceError(
                "Audio8 DAC produced empty or non-finite audio".into(),
            ));
        }
        let sample_rate = self.native_runtime()?.dac.config().sample_rate;
        Ok(Audio8GenerationOutput {
            samples,
            sample_rate,
            frames_generated,
            diagnostics: Audio8TtsGenerationDiagnostics {
                model_family: "audio8_tts",
                sample_rate,
                prompt_tokens: state.artifact.prompt.prompt_length,
                max_frames: state.max_frames,
                frames_generated,
                temperature: state.params.temperature,
                top_p: state.params.top_p,
                top_k: state.params.top_k,
                seed: state.params.seed,
                repetition_aware: state.params.repetition_aware,
                stop_reason: state.stop_reason,
                reference_encode_ms: state.artifact.reference_encode_ms,
                prompt_build_ms: state.artifact.prompt_build_ms,
                slow_prefill_ms: state.prefill_ms as f32,
                ar_decode_ms: state.decode_ms as f32,
                dac_decode_ms,
                total_model_ms: state.artifact.reference_encode_ms
                    + state.artifact.prompt_build_ms
                    + state.prefill_ms as f32
                    + state.decode_ms as f32
                    + dac_decode_ms,
            },
        })
    }

    fn native_runtime(&self) -> Result<&super::Audio8NativeRuntime> {
        self.runtime
            .as_ref()
            .ok_or_else(|| Error::ModelLoadError("Audio8 native runtime is not loaded".into()))
    }

    fn validate_retained_state(&self, state: &Audio8RetainedState) -> Result<()> {
        if state.model_identity != self.model_identity
            || state.artifact.model_identity != self.model_identity
        {
            return Err(Error::InvalidInput(
                "Audio8 retained state belongs to another model load".into(),
            ));
        }
        Ok(())
    }
}

impl Audio8PreparedArtifact {
    #[cfg(test)]
    pub(crate) fn test_prompt(rows: usize, tokens: usize) -> Arc<Self> {
        Arc::new(Self {
            model_identity: 1,
            prompt: Audio8ConditioningPrompt {
                values: vec![vec![0; tokens]; rows],
                vq_mask: vec![false; tokens],
                prompt_length: tokens,
            },
            reference_encode_ms: 0.0,
            prompt_build_ms: 0.0,
        })
    }

    pub(crate) const fn prompt_tokens(&self) -> usize {
        self.prompt.prompt_length
    }

    pub(crate) fn retained_bytes(&self) -> Result<u64> {
        let mut bytes = checked_bytes_sum(&[
            allocation_bytes(
                self.prompt.values.capacity(),
                std::mem::size_of::<Vec<u32>>(),
            )?,
            allocation_bytes(self.prompt.vq_mask.capacity(), std::mem::size_of::<bool>())?,
            allocation_bytes(
                1,
                std::mem::size_of::<Self>() + 2 * std::mem::size_of::<usize>(),
            )?,
        ])?;
        for row in &self.prompt.values {
            bytes = checked_bytes_sum(&[bytes, allocation_bytes(row.capacity(), 4)?])?;
        }
        Ok(bytes)
    }
}

impl Audio8RetainedState {
    pub(crate) fn begin_managed_quantum(
        &mut self,
        slow_cache: PhysicalPagedKvCache,
    ) -> Result<Audio8RetainedCheckpoint> {
        if self.active_quantum.is_some() || self.staged_step.is_some() || !self.completions_drained
        {
            return Err(Error::InferenceError(
                "Audio8 retained quantum is not clean".into(),
            ));
        }
        if slow_cache.arena().id() != self.slow_cache.arena().id()
            || slow_cache.context_len() != self.slow_position
        {
            return Err(Error::InvalidInput(
                "Audio8 managed cache authority or position changed".into(),
            ));
        }
        let quantum = self.next_quantum;
        self.next_quantum = quantum
            .checked_add(1)
            .ok_or_else(|| Error::InferenceError("Audio8 quantum overflow".into()))?;
        self.active_quantum = Some(quantum);
        Ok(Audio8RetainedCheckpoint {
            state_id: self.state_id,
            quantum,
            payload: Some(Audio8RetainedCheckpointPayload {
                slow_cache: Some(std::mem::replace(&mut self.slow_cache, slow_cache)),
                slow_position: self.slow_position,
                slow_output: self.slow_output.clone(),
                semantic_sampler: self.semantic_sampler.clone(),
                fast_sampler: self.fast_sampler.clone(),
                generated_frames: self.frames_generated(),
                recent_semantic_tokens: self.recent_semantic_tokens.clone(),
                stop_reason: self.stop_reason,
                finished: self.finished,
                staged_step: self.staged_step.clone(),
                completions_drained: self.completions_drained,
            }),
        })
    }

    pub(crate) fn commit_managed_quantum(
        &mut self,
        checkpoint: &mut Audio8RetainedCheckpoint,
    ) -> Result<()> {
        self.validate_checkpoint(checkpoint)?;
        if !self.completions_drained || self.staged_step.is_some() {
            return Err(Error::InferenceError(
                "Audio8 completions and staged output must be drained before commit".into(),
            ));
        }
        checkpoint.payload.take();
        self.active_quantum = None;
        Ok(())
    }

    pub(crate) fn rollback_managed_quantum(
        &mut self,
        checkpoint: &mut Audio8RetainedCheckpoint,
    ) -> Result<()> {
        self.validate_checkpoint(checkpoint)?;
        let payload = checkpoint.payload.take().ok_or_else(|| {
            Error::InferenceError("Audio8 checkpoint was already consumed".into())
        })?;
        self.slow_cache = payload.slow_cache.ok_or_else(|| {
            Error::InferenceError("initial Audio8 state must be discarded on rollback".into())
        })?;
        self.slow_position = payload.slow_position;
        self.slow_output = payload.slow_output;
        self.semantic_sampler = payload.semantic_sampler;
        self.fast_sampler = payload.fast_sampler;
        for row in &mut self.generated_codebooks {
            row.truncate(payload.generated_frames);
        }
        self.recent_semantic_tokens = payload.recent_semantic_tokens;
        self.stop_reason = payload.stop_reason;
        self.finished = payload.finished;
        self.staged_step = payload.staged_step;
        self.completions_drained = payload.completions_drained;
        self.active_quantum = None;
        Ok(())
    }

    pub(crate) fn take_staged_step(&mut self) -> Option<Audio8RetainedStep> {
        self.staged_step.take()
    }

    pub(crate) fn take_managed_write_completions(&mut self) -> Vec<Arc<KvWriteBatchCompletion>> {
        let completions = self.slow_cache.take_completed_writes();
        self.completions_drained = true;
        completions
    }

    #[cfg(test)]
    pub(crate) fn initial_quantum_for_test() -> (Self, Audio8RetainedCheckpoint) {
        Audio8TtsModel::for_test()
            .new_retained_state_in_quantum(
                Audio8PreparedArtifact::test_prompt(11, 3),
                Audio8GenerationParams {
                    max_frames: 1000,
                    ..Default::default()
                },
                super::physical::test_physical_cache(91, 1, 1, 1, 8),
                8192,
            )
            .unwrap()
    }

    #[cfg(test)]
    pub(crate) fn for_test() -> Self {
        Audio8TtsModel::for_test()
            .new_retained_state(
                Audio8PreparedArtifact::test_prompt(11, 3),
                Audio8GenerationParams {
                    max_frames: 1000,
                    ..Default::default()
                },
                super::physical::test_physical_cache(91, 1, 1, 1, 8),
                8192,
            )
            .unwrap()
    }

    pub(crate) const fn slow_position(&self) -> usize {
        self.slow_position
    }

    pub(crate) fn require_complete(&self) -> Result<()> {
        self.stop_reason.require_complete()
    }

    pub(crate) const fn finished(&self) -> bool {
        self.finished
    }

    pub(crate) fn params(&self) -> &Audio8GenerationParams {
        &self.params
    }

    pub(crate) fn sampling_and_steps(&self) -> (f64, u32, u32) {
        (self.sampling_ms, self.prefill_steps, self.decode_steps)
    }

    pub(crate) fn trace_timings(&self, request_id: &str, codec_ms: f64) {
        tracing::info!(
            request_id,
            timing_clock = "host_wall",
            reference_encode_ms = self.artifact.reference_encode_ms,
            prompt_build_ms = self.artifact.prompt_build_ms,
            prefill_ms = self.prefill_ms,
            ar_decode_ms = self.decode_ms,
            semantic_sampling_ms = self.sampling_ms,
            fast_ar_ms = self.fast_ar_ms,
            slow_ar_ms = self.slow_ar_ms,
            codec_ms,
            frames = self.frames_generated(),
            "Audio8 execution timing (GPU work may cross host intervals)"
        );
    }

    pub(crate) fn phase_timings(&self) -> (f64, f64) {
        (self.prefill_ms, self.decode_ms)
    }

    pub(crate) fn frames_generated(&self) -> usize {
        self.generated_codebooks.first().map(Vec::len).unwrap_or(0)
    }

    fn require_clean_quantum(&self) -> Result<()> {
        if self.active_quantum.is_none() || self.staged_step.is_some() {
            return Err(Error::InferenceError(
                "Audio8 step requires one clean active quantum".into(),
            ));
        }
        Ok(())
    }

    fn stage(&mut self, step: Audio8RetainedStep) -> Result<Audio8RetainedStep> {
        self.staged_step = Some(step.clone());
        Ok(step)
    }

    fn stage_finished(&mut self) -> Result<Audio8RetainedStep> {
        self.stage(Audio8RetainedStep::Finished {
            frames_generated: self.frames_generated(),
        })
    }

    fn validate_checkpoint(&self, checkpoint: &Audio8RetainedCheckpoint) -> Result<()> {
        if checkpoint.state_id != self.state_id
            || self.active_quantum != Some(checkpoint.quantum)
            || checkpoint.payload.is_none()
        {
            return Err(Error::InferenceError(
                "Audio8 checkpoint is foreign, stale, or out of order".into(),
            ));
        }
        Ok(())
    }
}

fn slice_prompt(
    prompt: &Audio8ConditioningPrompt,
    start: usize,
    len: usize,
) -> Result<Audio8ConditioningPrompt> {
    let end = start
        .checked_add(len)
        .ok_or_else(|| Error::InvalidInput("Audio8 prompt slice overflow".into()))?;
    if end > prompt.prompt_length {
        return Err(Error::InvalidInput(
            "Audio8 prompt slice exceeds prepared artifact".into(),
        ));
    }
    Ok(Audio8ConditioningPrompt {
        values: prompt
            .values
            .iter()
            .map(|row| row[start..end].to_vec())
            .collect(),
        vq_mask: prompt.vq_mask[start..end].to_vec(),
        prompt_length: len,
    })
}

fn preparation_memory_for_geometry(
    config: &super::Audio8Config,
    text_bytes: usize,
    reference_text_bytes: usize,
    input_samples: usize,
    input_capacity: usize,
    sample_rate: u32,
    context_limit: usize,
) -> Result<Audio8PreparationMemory> {
    let context = context_limit.min(config.max_seq_len);
    let max_prompt = context
        .checked_sub(1)
        .filter(|limit| *limit > 0)
        .ok_or_else(|| {
            Error::InvalidInput(
                "Audio8 effective context requires a prompt and generation position".into(),
            )
        })?;
    let dac = super::Audio8DacConfig::current();
    let frames = dac.reference_frame_count(input_samples, sample_rate)?;
    if frames > max_prompt {
        return Err(Error::InvalidInput(
            "Audio8 reference frames exceed effective prompt context".into(),
        ));
    }
    let prepared_samples = frames
        .checked_mul(dac.samples_per_frame()?)
        .ok_or_else(|| Error::Overloaded("Audio8 prepared audio length overflow".into()))?;
    let rendered_bytes = text_bytes
        .checked_add(reference_text_bytes)
        .and_then(|n| n.checked_add(512))
        .ok_or_else(|| Error::Overloaded("Audio8 rendered text length overflow".into()))?;
    // The native byte-level tokenizer's Encoding owns ids, offsets, token
    // strings and normalization/pretokenization scratch. Price these separately
    // from the dense prompt, using a conservative per-UTF8-byte envelope plus
    // fixed small-allocation allowance, as for codec workspace.
    let tokenization = allocation_bytes(rendered_bytes, 256)?;
    let code_elements = frames
        .checked_mul(config.num_codebooks)
        .ok_or_else(|| Error::Overloaded("Audio8 code readback shape overflow".into()))?;
    let host_transient_bytes = checked_bytes_sum(&[
        allocation_bytes(input_capacity, 4)?,
        allocation_bytes(reference_text_bytes, 1)?,
        allocation_bytes(prepared_samples, 4 * 3)?,
        super::codec::fft_workspace(
            u64::try_from(input_samples)
                .map_err(|_| Error::Overloaded("Audio8 input samples overflow".into()))?,
            sample_rate,
        )?,
        allocation_bytes(code_elements, 4 * 3)?,
        allocation_bytes(max_prompt, 4)?,
        tokenization,
        16 * 1024 * 1024,
    ])?;
    Ok(Audio8PreparationMemory {
        host_transient_bytes,
        retained_artifact_bytes: artifact_bound(config.num_codebooks, max_prompt)?,
        accelerator_workspace_bytes: super::codec::preparation_workspace_bytes(
            input_samples,
            sample_rate,
        )?,
    })
}

fn validate_preparation_inputs(text: &str, reference: Option<&Audio8Reference>) -> Result<()> {
    if text.trim().is_empty() {
        return Err(Error::InvalidInput(
            "Audio8 TTS text input cannot be empty".into(),
        ));
    }
    if reference.is_some_and(|reference| {
        reference.text.trim().is_empty() || reference.audio_samples.is_empty()
    }) {
        return Err(Error::InvalidInput(
            "Audio8 reference text and audio cannot be empty".into(),
        ));
    }
    Ok(())
}

/// Preparation memory of a request with no reference: the prompt is text
/// only, so nothing is resampled or encoded.
fn plain_preparation_memory(
    config: &super::Audio8Config,
    text_bytes: usize,
    context_limit: usize,
) -> Result<Audio8PreparationMemory> {
    let context = context_limit.min(config.max_seq_len);
    let max_prompt = context
        .checked_sub(1)
        .filter(|limit| *limit > 0)
        .ok_or_else(|| {
            Error::InvalidInput(
                "Audio8 effective context requires a prompt and generation position".into(),
            )
        })?;
    let rendered_bytes = text_bytes
        .checked_add(512)
        .ok_or_else(|| Error::Overloaded("Audio8 rendered text length overflow".into()))?;
    Ok(Audio8PreparationMemory {
        host_transient_bytes: checked_bytes_sum(&[
            allocation_bytes(max_prompt, 4)?,
            allocation_bytes(rendered_bytes, 256)?,
            16 * 1024 * 1024,
        ])?,
        retained_artifact_bytes: artifact_bound(config.num_codebooks, max_prompt)?,
        accelerator_workspace_bytes: 0,
    })
}

/// Keep `token` for the repetition-aware retry. As in the published
/// generation loop, the utterance's first frame is never recorded: the
/// window holds the last [`RAS_WIN_SIZE`] tokens from the second frame on.
pub(super) fn record_recent_semantic(recent: &mut Vec<u32>, token: u32, frames_generated: usize) {
    if frames_generated <= 1 {
        return;
    }
    recent.push(token);
    if recent.len() > RAS_WIN_SIZE {
        recent.remove(0);
    }
}

fn next_state_id() -> Result<u64> {
    NEXT_AUDIO8_STATE_ID
        .try_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .map_err(|_| Error::InferenceError("Audio8 state identity space exhausted".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::architectures::audio8::physical::test_physical_cache;

    fn tiny_native_model() -> Audio8TtsModel {
        let mut model = Audio8TtsModel::for_test();
        model.config.num_codebooks = 2;
        model.config.codebook_size = 8;
        model.config.semantic_start_token_id = 20;
        model.config.semantic_end_token_id = 27;
        model.config.eos_token_id = 1;
        model.config.max_seq_len = 16;
        let slow = super::super::slow::tiny_for_batch_tests();
        let mask = slow.semantic_allowed_mask(1).unwrap();
        model.runtime = Some(super::super::Audio8NativeRuntime {
            tokenizer: super::super::Audio8PromptTokenizer::for_batch_test(&model.config),
            slow,
            fast: super::super::fast::tiny_for_batch_tests(),
            dac: super::super::dac::tiny_for_batch_tests(),
            semantic_allowed_mask: mask,
            semantic_masks: std::sync::OnceLock::new(),
            dtype: candle_core::DType::F32,
            precision: crate::kernels::numeric_check::PrecisionLoad::unchecked(
                &crate::backends::DTypeSelection {
                    dtype: candle_core::DType::F32,
                    reason: "test fixture".into(),
                },
            ),
        });
        model
    }

    fn next_quantum(state: &mut Audio8RetainedState) -> Audio8RetainedCheckpoint {
        let view = PhysicalPagedKvCache::new(
            state.slow_cache.arena().clone(),
            vec![state.slow_cache.layer_binding(0).unwrap()],
            state.slow_cache.blocks.clone(),
            state.slow_position,
        )
        .unwrap();
        state.begin_managed_quantum(view).unwrap()
    }

    /// A step's semantic rows drawn together from batched device candidates
    /// give each row the token it gets drawn alone, step after step (so each
    /// row's RNG stays in step), first frames (no `<|im_end|>`) beside later
    /// ones.
    #[test]
    fn batched_semantic_rows_draw_each_rows_own_token() {
        use crate::kernels::numeric_check::xorshift_tensor;

        let model = tiny_native_model();
        let runtime = model.runtime.as_ref().unwrap();
        let vocab = runtime.semantic_allowed_mask.len();
        let device = candle_core::Device::Cpu;
        let mut samplers = (0..5)
            .map(|row| Audio8SemanticSampler::new(0.9, 0.8, 70 + row))
            .collect::<Vec<_>>();
        let mut alone = samplers.clone();
        let mut previous = vec![Vec::new(); 5];
        for step in 0..6u64 {
            let logits = (0..5u64)
                .map(|row| xorshift_tensor(&[1, 1, vocab], step * 10 + row, 2.0, &device).unwrap())
                .collect::<Vec<_>>();
            let allow = (0..5)
                .map(|row| step > 0 || row % 2 == 1)
                .collect::<Vec<_>>();
            let batched = {
                let mut rows = samplers
                    .iter_mut()
                    .enumerate()
                    .map(|(row, sampler)| Audio8SemanticRow {
                        logits: &logits[row],
                        allow_im_end: allow[row],
                        previous: &previous[row],
                        sampler,
                    })
                    .collect::<Vec<_>>();
                sample_semantic_rows(runtime, &mut rows, true).unwrap()
            };
            for row in 0..5 {
                let token = sample_semantic_rows(
                    runtime,
                    &mut [Audio8SemanticRow {
                        logits: &logits[row],
                        allow_im_end: allow[row],
                        previous: &previous[row],
                        sampler: &mut alone[row],
                    }],
                    true,
                )
                .unwrap()[0];
                assert_eq!(batched[row], token, "step {step} row {row}");
                previous[row].push(token);
            }
        }
        // Six steps of equal tokens: the rows' RNG streams stayed in step.
    }

    #[test]
    fn real_batched_frames_rollback_rng_and_eos_then_retry_in_different_order() {
        let model = tiny_native_model();
        let slow = super::super::physical::test_physical_caches(902, 1, 1, 2, 16, 7);
        let mut fast = super::super::physical::test_physical_caches(903, 1, 1, 2, 2, 7);
        let mut states = slow
            .into_iter()
            .enumerate()
            .map(|(row, cache)| {
                model
                    .new_retained_state(
                        Audio8PreparedArtifact::test_prompt(3, row % 3 + 1),
                        Audio8GenerationParams {
                            max_frames: 4,
                            seed: row as u64 + 77,
                            top_k: 0,
                            ..Default::default()
                        },
                        cache,
                        16,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let mut checkpoints = states.iter_mut().map(next_quantum).collect::<Vec<_>>();
        model
            .retained_prefill_batch(&mut states.iter_mut().collect::<Vec<_>>(), &[3; 7])
            .unwrap();
        for (state, checkpoint) in states.iter_mut().zip(&mut checkpoints) {
            assert!(state.take_staged_step().is_some());
            assert_eq!(state.take_managed_write_completions().len(), 1);
            state.commit_managed_quantum(checkpoint).unwrap();
        }
        // One row has already emitted a frame and may legally sample EOS.
        states[0].generated_codebooks = vec![vec![1], vec![2]];
        let mut logits = vec![-100f32; 32];
        logits[1] = 100.;
        states[0].slow_output.as_mut().unwrap().logits =
            candle_core::Tensor::from_vec(logits, (1, 1, 32), &candle_core::Device::Cpu).unwrap();
        let original = states
            .iter()
            .map(|s| (s.slow_position, s.generated_codebooks.clone()))
            .collect::<Vec<_>>();
        let mut checkpoints = states.iter_mut().map(next_quantum).collect::<Vec<_>>();
        let steps = model
            .retained_decode_batch(
                &mut states.iter_mut().collect::<Vec<_>>(),
                &mut fast.iter_mut().collect::<Vec<_>>(),
            )
            .unwrap();
        assert!(matches!(steps[0], Audio8RetainedStep::Finished { .. }));
        assert_eq!(fast[0].context_len(), 0, "EOS must skip Fast AR");
        let expected = states
            .iter()
            .map(|s| (s.generated_codebooks.clone(), s.finished))
            .collect::<Vec<_>>();
        for ((state, checkpoint), before) in states.iter_mut().zip(&mut checkpoints).zip(original) {
            state.rollback_managed_quantum(checkpoint).unwrap();
            assert_eq!(
                (state.slow_position, state.generated_codebooks.clone()),
                before
            );
        }
        states.reverse();
        fast.reverse();
        let mut checkpoints = states.iter_mut().map(next_quantum).collect::<Vec<_>>();
        model
            .retained_decode_batch(
                &mut states.iter_mut().collect::<Vec<_>>(),
                &mut fast.iter_mut().collect::<Vec<_>>(),
            )
            .unwrap();
        for ((state, checkpoint), expected) in states
            .iter_mut()
            .zip(&mut checkpoints)
            .zip(expected.into_iter().rev())
        {
            assert_eq!(
                (state.generated_codebooks.clone(), state.finished),
                expected
            );
            state.take_staged_step();
            assert_eq!(state.take_managed_write_completions().len(), 1);
            state.commit_managed_quantum(checkpoint).unwrap();
        }
    }

    #[test]
    fn retained_finalization_rejects_frame_limit_and_accepts_eos() {
        let mut state = Audio8RetainedState::for_test();
        state.finished = true;
        assert!(state
            .require_complete()
            .unwrap_err()
            .to_string()
            .contains("generation incomplete: frame_limit"));
        state.stop_reason = Audio8StopReason::EndOfSpeech;
        state.require_complete().unwrap();
    }

    #[test]
    fn managed_generation_rejects_silent_output_budget_shrink() {
        let model = Audio8TtsModel::for_test();
        let error = model
            .new_retained_state_in_quantum(
                Audio8PreparedArtifact::test_prompt(11, 224),
                Audio8GenerationParams {
                    max_frames: 512,
                    ..Default::default()
                },
                test_physical_cache(91, 1, 1, 1, 192),
                256,
            )
            .err()
            .expect("oversized budget must fail");
        assert!(error
            .to_string()
            .contains("segment does not fit effective context"));
    }

    #[test]
    fn managed_generation_budget_uses_logical_context_not_first_chunk() {
        let model = Audio8TtsModel::for_test();
        for (prompt, context, requested, expected) in [
            (224, 8192, 512, 512),
            (160, 8192, 512, 512),
            (224, 256, 31, 31),
        ] {
            let (mut state, mut checkpoint) = model
                .new_retained_state_in_quantum(
                    Audio8PreparedArtifact::test_prompt(11, prompt),
                    Audio8GenerationParams {
                        max_frames: requested,
                        ..Default::default()
                    },
                    test_physical_cache(91, 1, 1, 1, 192),
                    context,
                )
                .unwrap();
            assert_eq!(state.max_frames, expected);
            assert_eq!(state.slow_cache.capacity_tokens(), 192);
            state.take_managed_write_completions();
            state.commit_managed_quantum(&mut checkpoint).unwrap();
            let _checkpoint = state
                .begin_managed_quantum(test_physical_cache(91, 1, 1, 1, 256))
                .unwrap();
            assert_eq!(state.slow_cache.capacity_tokens(), 256);
            assert_eq!(state.max_frames, expected);
        }
    }

    #[test]
    fn managed_generation_rejects_prompt_at_actual_context_limit() {
        let model = Audio8TtsModel::for_test();
        for context in [192, 224] {
            let error = model
                .new_retained_state_in_quantum(
                    Audio8PreparedArtifact::test_prompt(11, 224),
                    Audio8GenerationParams::default(),
                    test_physical_cache(91, 1, 1, 1, 256),
                    context,
                )
                .err()
                .expect("actual context must leave output room");
            assert!(error.to_string().contains("leaves no output room"));
        }
    }

    fn state() -> Audio8RetainedState {
        Audio8RetainedState {
            state_id: 7,
            model_identity: 9,
            artifact: Arc::new(Audio8PreparedArtifact {
                model_identity: 9,
                prompt: Audio8ConditioningPrompt {
                    values: vec![vec![1, 2, 3], vec![4, 5, 6]],
                    vq_mask: vec![false, true, true],
                    prompt_length: 3,
                },
                reference_encode_ms: 0.0,
                prompt_build_ms: 0.0,
            }),
            params: Audio8GenerationParams::default(),
            slow_cache: test_physical_cache(91, 1, 1, 1, 8),
            slow_position: 0,
            slow_output: None,
            semantic_sampler: Audio8SemanticSampler::new(0.8, 0.8, 0),
            fast_sampler: Audio8Sampler::new(0.8, 0.8, 1),
            generated_codebooks: vec![Vec::new(), Vec::new()],
            recent_semantic_tokens: Vec::new(),
            max_frames: 4,
            prefill_ms: 0.0,
            decode_ms: 0.0,
            sampling_ms: 0.0,
            fast_ar_ms: 0.0,
            slow_ar_ms: 0.0,
            prefill_steps: 0,
            decode_steps: 0,
            stop_reason: Audio8StopReason::FrameLimit,
            finished: false,
            active_quantum: None,
            next_quantum: 1,
            staged_step: None,
            completions_drained: true,
        }
    }

    #[test]
    fn artifact_observation_counts_spare_capacity_and_arc_once() {
        let mut artifact = (*Audio8PreparedArtifact::test_prompt(11, 17)).clone();
        artifact.prompt.values[0].reserve(31);
        artifact.prompt.vq_mask.reserve(31);
        let expected = std::mem::size_of::<Audio8PreparedArtifact>()
            + 2 * std::mem::size_of::<usize>()
            + artifact.prompt.values.capacity() * std::mem::size_of::<Vec<u32>>()
            + artifact
                .prompt
                .values
                .iter()
                .map(|row| row.capacity() * 4)
                .sum::<usize>()
            + artifact.prompt.vq_mask.capacity();
        let shared = Arc::new(artifact);
        let clone = shared.clone();
        assert_eq!(shared.retained_bytes().unwrap(), expected as u64);
        assert_eq!(
            clone.retained_bytes().unwrap(),
            shared.retained_bytes().unwrap()
        );
        assert!(shared.retained_bytes().unwrap() > artifact_bound(10, 17).unwrap());
    }

    #[test]
    fn artifact_context_envelope_covers_exact_dense_allocations() {
        let config = super::super::config::current_config();
        for tokens in [1, 17, config.max_seq_len - 1] {
            let artifact = Audio8PreparedArtifact::test_prompt(config.num_codebooks + 1, tokens);
            assert_eq!(
                artifact.retained_bytes().unwrap(),
                artifact_bound(config.num_codebooks, tokens).unwrap()
            );
        }
    }

    #[test]
    fn preparation_contract_prices_native_resampled_and_capacity_bytes() {
        let config = super::super::config::current_config();
        let native =
            preparation_memory_for_geometry(&config, 32, 64, 44_100, 44_100, 44_100, 1024).unwrap();
        let spare =
            preparation_memory_for_geometry(&config, 32, 64, 44_100, 44_200, 44_100, 1024).unwrap();
        let resampled =
            preparation_memory_for_geometry(&config, 32, 64, 16_000, 16_000, 16_000, 1024).unwrap();
        assert_eq!(
            spare.host_transient_bytes - native.host_transient_bytes,
            400
        );
        assert_eq!(
            native.retained_artifact_bytes,
            resampled.retained_artifact_bytes
        );
        assert!(resampled.host_transient_bytes > 16 * 1024 * 1024);
        assert_eq!(
            native.host_bytes().unwrap(),
            native.host_transient_bytes + native.retained_artifact_bytes
        );
    }

    #[test]
    fn preparation_contract_rejects_invalid_geometry_and_overflow() {
        let config = super::super::config::current_config();
        for (samples, rate, context) in [
            (0, 44_100, 1024),
            (1, 0, 1024),
            (1, u32::MAX, 1024),
            (44_100, 44_100, 2),
            (1, 44_100, 1),
            (usize::MAX, 1, 1024),
        ] {
            assert!(preparation_memory_for_geometry(
                &config, 1, 1, samples, samples, rate, context
            )
            .is_err());
        }
        assert!(
            preparation_memory_for_geometry(&config, usize::MAX, 1, 1, 1, 44_100, 1024).is_err()
        );
        assert!(
            preparation_memory_for_geometry(&config, 1, 1, 1, usize::MAX, 44_100, 1024).is_err()
        );
        assert!(artifact_bound(usize::MAX, 1).is_err());
        assert!(artifact_bound(10, usize::MAX).is_err());
        assert!(Audio8PreparationMemory {
            host_transient_bytes: u64::MAX,
            retained_artifact_bytes: 1,
            accelerator_workspace_bytes: 0
        }
        .host_bytes()
        .is_err());
    }

    #[test]
    fn prepared_prompt_slices_preserve_codebook_rows_and_mask() {
        let state = state();
        let slice = slice_prompt(&state.artifact.prompt, 1, 2).unwrap();
        assert_eq!(slice.values, vec![vec![2, 3], vec![5, 6]]);
        assert_eq!(slice.vq_mask, vec![true, true]);
        assert_eq!(slice.prompt_length, 2);
    }

    #[test]
    fn managed_rollback_restores_frame_codes_and_sampler_state() {
        let mut state = state();
        let logits =
            candle_core::Tensor::new(&[0.0f32, 0.1, 0.2, 0.3], &candle_core::Device::Cpu).unwrap();
        let mut original = state.fast_sampler.clone();
        let expected = super::super::fast::sample_logits(&logits, &mut original).unwrap();
        let slow = test_physical_cache(91, 1, 1, 1, 8);
        let mut checkpoint = state.begin_managed_quantum(slow).unwrap();
        state.slow_position = 3;
        state.generated_codebooks = vec![vec![1], vec![2]];
        state.recent_semantic_tokens.push(42);
        let _ = super::super::fast::sample_logits(&logits, &mut state.fast_sampler).unwrap();
        state.staged_step = Some(Audio8RetainedStep::Frame {
            frames_generated: 1,
        });
        state.rollback_managed_quantum(&mut checkpoint).unwrap();
        assert_eq!(state.slow_position(), 0);
        assert_eq!(state.frames_generated(), 0);
        assert!(state.recent_semantic_tokens.is_empty());
        assert_eq!(
            super::super::fast::sample_logits(&logits, &mut state.fast_sampler).unwrap(),
            expected
        );
        assert!(state.take_staged_step().is_none());
    }

    #[test]
    fn commit_requires_staged_output_and_write_receipts_to_be_drained() {
        let mut state = state();
        let slow = test_physical_cache(91, 1, 1, 1, 8);
        let mut checkpoint = state.begin_managed_quantum(slow).unwrap();
        state.staged_step = Some(Audio8RetainedStep::Frame {
            frames_generated: 1,
        });
        assert!(state.commit_managed_quantum(&mut checkpoint).is_err());
        state.take_staged_step();
        state.take_managed_write_completions();
        state.commit_managed_quantum(&mut checkpoint).unwrap();
    }
}
