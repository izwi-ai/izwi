//! Rollback-safe retained Ming generation, quantum by quantum.
//!
//! A request's language-model sequence lives in one engine-owned paged
//! cache. Its host state (prefill cursor, the hidden state of the last
//! forward, the latents so far, the flow head's history and the noise
//! stream) is checkpointed with the cache at the start of each quantum, so
//! a failed or cancelled quantum rolls back to exactly where it began.
//! Prefill may arrive in chunks. Each decode step samples one latent from
//! the pending hidden state, asks the stop head, and runs the latent's
//! embedding through the language model (the last latent needs no forward).
//! Decode rows from many requests share language-model calls, each row
//! getting the hidden state it would alone; the flow head and the
//! aggregator run one request at a time. Audio is decoded in windows
//! between quanta ([`super::stream`]); a failed window discards the request.

use std::sync::Arc;

use candle_core::{DType, Device, Tensor};

use super::config::LATENT_DIM;
use super::stack::MingForwardRow;
use super::stream::{MingAudioStream, MingWindow};
use super::vae::MingVaeDecoder;
use super::{
    MingGenerationParams, MingLatentInput, MingNoiseStream, MingPrompt, MingRuntime,
    MingStopReason, MIN_LATENTS,
};
use crate::error::{Error, Result};
use crate::models::shared::attention::physical::{PhysicalPagedKvCache, PhysicalPagedKvCheckpoint};
use crate::models::shared::noise::SplitMix64;

/// One request, prepared before it reaches the engine: its prompt.
#[derive(Debug)]
pub(crate) struct MingPreparedArtifact {
    pub(crate) prompt: MingPrompt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MingPhase {
    Prefill,
    Decode,
    /// The model has stopped; windows remain.
    Audio,
    Finished,
}

#[derive(Clone)]
struct HostCheckpoint {
    prefill_cursor: usize,
    pending: Option<Vec<f32>>,
    latents: usize,
    history: Vec<f32>,
    rng: SplitMix64,
    phase: MingPhase,
    stop_reason: Option<MingStopReason>,
}

pub(crate) struct MingQuantumCheckpoint {
    cache: PhysicalPagedKvCheckpoint,
    host: HostCheckpoint,
}

pub(crate) struct MingRetainedState {
    pub(crate) artifact: Arc<MingPreparedArtifact>,
    pub(crate) params: MingGenerationParams,
    /// The most latents this request generates: its parameter, bounded by
    /// what the context leaves after the prompt.
    budget: usize,
    prefill_cursor: usize,
    /// The language model's hidden state from the last forward: what the
    /// next latent is sampled from.
    pending: Option<Vec<f32>>,
    /// Every latent so far, 64 values each.
    latents: Vec<f32>,
    /// The flow head's history: the previous latent.
    history: Vec<f32>,
    rng: SplitMix64,
    phase: MingPhase,
    stop_reason: Option<MingStopReason>,
    pub(crate) audio: MingAudioStream,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MingPrefillStep {
    pub(crate) consumed_tokens: usize,
    pub(crate) complete: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MingDecodeStep {
    pub(crate) finished: bool,
}

/// One request's state and cache in a batched quantum.
pub(crate) struct MingStepRow<'a> {
    pub(crate) state: &'a mut MingRetainedState,
    pub(crate) cache: &'a mut PhysicalPagedKvCache,
}

impl MingRetainedState {
    pub(crate) fn new(
        artifact: Arc<MingPreparedArtifact>,
        params: MingGenerationParams,
        context_limit: usize,
    ) -> Result<Self> {
        params.validate()?;
        let positions = artifact.prompt.positions;
        let budget = params
            .max_new_tokens
            .min(MingRuntime::latent_room(positions));
        if positions == 0
            || budget < MIN_LATENTS
            || Self::required_context(positions, budget) > context_limit
        {
            return Err(Error::InvalidInput(format!(
                "Ming request of {positions} prompt positions leaves no room for speech in its \
                 {context_limit}-position context"
            )));
        }
        Ok(Self {
            rng: SplitMix64::for_stream(params.seed, MingNoiseStream::Flow),
            history: artifact.prompt.history.clone(),
            artifact,
            params,
            budget,
            prefill_cursor: 0,
            pending: None,
            latents: Vec::new(),
            phase: MingPhase::Prefill,
            stop_reason: None,
            audio: MingAudioStream::default(),
        })
    }

    /// Positions the sequence reaches: the prompt plus every latent but the
    /// last.
    pub(crate) fn required_context(prompt_positions: usize, max_new_tokens: usize) -> usize {
        prompt_positions + max_new_tokens.saturating_sub(1)
    }

    pub(crate) fn phase(&self) -> MingPhase {
        self.phase
    }

    pub(crate) fn prompt_tokens(&self) -> usize {
        self.artifact.prompt.positions
    }

    pub(crate) fn prefill_cursor(&self) -> usize {
        self.prefill_cursor
    }

    pub(crate) fn latents_generated(&self) -> usize {
        self.latents.len() / LATENT_DIM
    }

    pub(crate) fn budget(&self) -> usize {
        self.budget
    }

    pub(crate) fn stop_reason(&self) -> Option<MingStopReason> {
        self.stop_reason
    }

    /// Whether the model has generated its last latent.
    pub(crate) fn stopped(&self) -> bool {
        matches!(self.phase, MingPhase::Audio | MingPhase::Finished)
    }

    /// The audio window that can run now, if any.
    pub(crate) fn next_window(&self) -> Option<MingWindow> {
        self.audio
            .next_window(self.latents_generated(), self.stopped())
    }

    /// Decode the next window; its samples, or `None` when no window is
    /// due.
    pub(crate) fn decode_window(&mut self, decoder: &MingVaeDecoder) -> Result<Option<Vec<f32>>> {
        let Some(window) = self.next_window() else {
            return Ok(None);
        };
        self.audio.decode(decoder, &self.latents, window).map(Some)
    }

    /// Positions the cache must hold now.
    fn position(&self) -> usize {
        let prompt = self.artifact.prompt.positions;
        match self.phase {
            MingPhase::Prefill => self.prefill_cursor,
            MingPhase::Decode => prompt + self.latents_generated(),
            MingPhase::Audio | MingPhase::Finished => {
                prompt + self.latents_generated().saturating_sub(1)
            }
        }
    }

    fn host_checkpoint(&self) -> HostCheckpoint {
        HostCheckpoint {
            prefill_cursor: self.prefill_cursor,
            pending: self.pending.clone(),
            latents: self.latents_generated(),
            history: self.history.clone(),
            rng: self.rng.clone(),
            phase: self.phase,
            stop_reason: self.stop_reason,
        }
    }

    fn check_cache(&self, cache: &PhysicalPagedKvCache) -> Result<()> {
        if cache.context_len() != self.position() {
            return Err(Error::InvalidInput(
                "Ming retained cache crossed host state".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn begin_quantum(
        &self,
        cache: &PhysicalPagedKvCache,
    ) -> Result<MingQuantumCheckpoint> {
        self.check_cache(cache)?;
        Ok(MingQuantumCheckpoint {
            cache: cache.logical_checkpoint(),
            host: self.host_checkpoint(),
        })
    }

    pub(crate) fn commit_quantum(
        &self,
        cache: &PhysicalPagedKvCache,
        checkpoint: &MingQuantumCheckpoint,
    ) -> Result<()> {
        self.check_cache(cache)?;
        if self.latents_generated() < checkpoint.host.latents
            || self.prefill_cursor < checkpoint.host.prefill_cursor
        {
            return Err(Error::InferenceError(
                "Ming quantum moved its state backwards".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn rollback_quantum(
        &mut self,
        cache: &mut PhysicalPagedKvCache,
        checkpoint: &MingQuantumCheckpoint,
    ) -> Result<()> {
        cache.restore_logical_checkpoint(checkpoint.cache.clone())?;
        let host = &checkpoint.host;
        self.latents.truncate(host.latents * LATENT_DIM);
        self.history = host.history.clone();
        self.prefill_cursor = host.prefill_cursor;
        self.pending = host.pending.clone();
        self.rng = host.rng.clone();
        self.phase = host.phase;
        self.stop_reason = host.stop_reason;
        Ok(())
    }

    pub(crate) fn mark_finished(&mut self) -> Result<()> {
        if self.phase != MingPhase::Audio || !self.audio.done() {
            return Err(Error::InferenceError(
                "Ming finished before its last audio window".into(),
            ));
        }
        self.phase = MingPhase::Finished;
        Ok(())
    }
}

fn hidden_rows(hidden: &Tensor) -> Result<Vec<Vec<f32>>> {
    Ok(hidden
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .to_vec2::<f32>()?)
}

/// Prefill `tokens[i]` more prompt positions of each row, in one call.
pub(crate) fn ming_prefill_batch(
    runtime: &MingRuntime,
    rows: &mut [MingStepRow<'_>],
    tokens: &[usize],
) -> Result<Vec<MingPrefillStep>> {
    if rows.is_empty() || rows.len() != tokens.len() {
        return Err(Error::InvalidInput(
            "Ming prefill batch rows do not match".into(),
        ));
    }
    let hidden = runtime.config().llm.hidden_size;
    let mut steps = Vec::with_capacity(rows.len());
    let mut spans = Vec::with_capacity(rows.len());
    for (row, &count) in rows.iter().zip(tokens) {
        let state = &*row.state;
        let start = state.prefill_cursor;
        let positions = state.artifact.prompt.positions;
        if state.phase != MingPhase::Prefill || count == 0 || start + count > positions {
            return Err(Error::InvalidInput(format!(
                "Ming prefill of {count} positions at {start} exceeds its {positions}-position prompt"
            )));
        }
        steps.push(MingPrefillStep {
            consumed_tokens: count,
            complete: start + count == positions,
        });
        spans.push(runtime.prompt_rows(
            &state.artifact.prompt.embeds[start * hidden..(start + count) * hidden],
        )?);
    }
    let mut forward = rows
        .iter_mut()
        .zip(&spans)
        .map(|(row, embeds)| MingForwardRow {
            embeds,
            cache: &mut *row.cache,
        })
        .collect::<Vec<_>>();
    let states = hidden_rows(&runtime.trunk().forward_batch(&mut forward)?)?;
    drop(forward);
    for ((row, step), state) in rows.iter_mut().zip(&steps).zip(states) {
        row.state.prefill_cursor += step.consumed_tokens;
        if step.complete {
            row.state.phase = MingPhase::Decode;
            row.state.pending = Some(state);
        }
    }
    Ok(steps)
}

/// One decode step for each row: sample a latent from its pending hidden
/// state, then run every row that has not stopped through the language
/// model in one call.
pub(crate) fn ming_decode_batch(
    runtime: &MingRuntime,
    rows: &mut [MingStepRow<'_>],
) -> Result<Vec<MingDecodeStep>> {
    if rows.is_empty() {
        return Err(Error::InvalidInput("Ming decode batch has no rows".into()));
    }
    // Every row's inputs first: its hidden state, history and its own
    // seeded noise. Then one flow-head call for all of them, and one
    // aggregator call for the rows that go on: each row's values are the
    // ones it gets alone (see `flow_group`).
    let mut inputs = Vec::with_capacity(rows.len());
    for row in rows.iter_mut() {
        let state = &mut *row.state;
        if state.phase != MingPhase::Decode {
            return Err(Error::InvalidInput(
                "Ming decode step outside the decode phase".into(),
            ));
        }
        let hidden = state.pending.take().ok_or_else(|| {
            Error::InferenceError("Ming decode step has no pending hidden state".into())
        })?;
        let mut noise = state.rng.standard_normals(LATENT_DIM);
        noise.truncate(LATENT_DIM);
        inputs.push((hidden, noise));
    }
    let latents = runtime.sample_latents(
        &rows
            .iter()
            .zip(&inputs)
            .map(|(row, (hidden, noise))| MingLatentInput {
                hidden,
                history: &row.state.history,
                noise,
                guidance: row.state.params.guidance,
            })
            .collect::<Vec<_>>(),
    )?;
    let mut steps = Vec::with_capacity(rows.len());
    let mut going_on = Vec::with_capacity(rows.len());
    let mut fed_back = Vec::with_capacity(rows.len() * LATENT_DIM);
    for ((row, (hidden, _)), latent) in rows.iter_mut().zip(&inputs).zip(latents) {
        let state = &mut *row.state;
        let step = state.latents_generated();
        state.latents.extend_from_slice(&latent);
        let stop = if runtime.stops_at(hidden, step) {
            Some(MingStopReason::EndOfSpeech)
        } else if step + 1 >= state.budget {
            Some(MingStopReason::LatentBudget)
        } else {
            None
        };
        match stop {
            Some(reason) => {
                // The last latent is kept and never fed back.
                state.phase = MingPhase::Audio;
                state.stop_reason = Some(reason);
            }
            None => {
                fed_back.extend_from_slice(&latent);
                state.history = latent;
            }
        }
        going_on.push(stop.is_none());
        steps.push(MingDecodeStep {
            finished: stop.is_some(),
        });
    }
    let embeds = if fed_back.is_empty() {
        None
    } else {
        Some(runtime.latent_embeddings(&fed_back)?)
    };
    let mut next = Vec::with_capacity(rows.len());
    let mut fed = 0usize;
    for &goes_on in &going_on {
        next.push(match (&embeds, goes_on) {
            (Some(embeds), true) => {
                fed += 1;
                Some(embeds.narrow(0, fed - 1, 1)?)
            }
            _ => None,
        });
    }
    let mut forward = rows
        .iter_mut()
        .zip(&next)
        .filter_map(|(row, embeds)| {
            embeds.as_ref().map(|embeds| MingForwardRow {
                embeds,
                cache: &mut *row.cache,
            })
        })
        .collect::<Vec<_>>();
    if !forward.is_empty() {
        let states = hidden_rows(&runtime.trunk().forward_batch(&mut forward)?)?;
        drop(forward);
        let mut states = states.into_iter();
        for (row, embeds) in rows.iter_mut().zip(&next) {
            if embeds.is_some() {
                row.state.pending = states.next();
            }
        }
    }
    Ok(steps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::architectures::ming_tts::synthetic::{tiny_request, tiny_runtime};
    use crate::models::architectures::ming_tts::DEFAULT_GUIDANCE;

    fn params(max_new_tokens: usize) -> MingGenerationParams {
        MingGenerationParams {
            guidance: DEFAULT_GUIDANCE,
            max_new_tokens,
            seed: 7,
        }
    }

    fn state(runtime: &MingRuntime, text: &str, latents: usize) -> MingRetainedState {
        let prompt = runtime.prompt(&tiny_request(text)).unwrap();
        MingRetainedState::new(
            Arc::new(MingPreparedArtifact { prompt }),
            params(latents),
            super::super::CONTEXT_POSITIONS,
        )
        .unwrap()
    }

    /// Run a state to its stop, prefilling in `chunk`-position spans;
    /// returns its latents.
    fn run(
        runtime: &MingRuntime,
        state: &mut MingRetainedState,
        cache: &mut PhysicalPagedKvCache,
        chunk: usize,
    ) -> Vec<f32> {
        while state.phase() == MingPhase::Prefill {
            let span = chunk.min(state.prompt_tokens() - state.prefill_cursor());
            let checkpoint = state.begin_quantum(cache).unwrap();
            ming_prefill_batch(runtime, &mut [MingStepRow { state, cache }], &[span]).unwrap();
            state.commit_quantum(cache, &checkpoint).unwrap();
        }
        while state.phase() == MingPhase::Decode {
            let checkpoint = state.begin_quantum(cache).unwrap();
            ming_decode_batch(runtime, &mut [MingStepRow { state, cache }]).unwrap();
            state.commit_quantum(cache, &checkpoint).unwrap();
        }
        state.latents.clone()
    }

    #[test]
    fn retained_steps_reproduce_the_one_shot_generation() {
        let runtime = tiny_runtime();
        let request = tiny_request("hello there friend");
        let mut cache = runtime.scratch_cache(256).unwrap();
        let direct = runtime
            .generate_physical(&request, &params(9), &mut cache)
            .unwrap();
        for chunk in [usize::MAX, 3] {
            let mut retained = state(&runtime, "hello there friend", 9);
            let mut cache = runtime.scratch_cache(256).unwrap();
            let latents = run(&runtime, &mut retained, &mut cache, chunk);
            assert_eq!(latents.len() / LATENT_DIM, direct.latents);
            assert_eq!(retained.stop_reason(), Some(direct.stop_reason));
            let mut samples = Vec::new();
            while let Some(window) = retained.decode_window(runtime.decoder()).unwrap() {
                samples.extend(window);
            }
            assert!(retained.audio.done());
            assert_eq!(samples.len(), direct.samples.len());
            let worst = samples
                .iter()
                .zip(&direct.samples)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(worst < 1e-3, "chunk {chunk}: {worst}");
            retained.mark_finished().unwrap();
        }
    }

    #[test]
    fn a_rolled_back_quantum_replays_identically() {
        let runtime = tiny_runtime();
        let mut straight = state(&runtime, "one two three", 8);
        let mut cache = runtime.scratch_cache(256).unwrap();
        let expected = run(&runtime, &mut straight, &mut cache, usize::MAX);

        let mut retained = state(&runtime, "one two three", 8);
        let mut cache = runtime.scratch_cache(256).unwrap();
        let span = retained.prompt_tokens();
        ming_prefill_batch(
            &runtime,
            &mut [MingStepRow {
                state: &mut retained,
                cache: &mut cache,
            }],
            &[span],
        )
        .unwrap();
        // Two steps, the second rolled back and run again.
        for _ in 0..2 {
            ming_decode_batch(
                &runtime,
                &mut [MingStepRow {
                    state: &mut retained,
                    cache: &mut cache,
                }],
            )
            .unwrap();
        }
        let checkpoint = retained.begin_quantum(&cache).unwrap();
        ming_decode_batch(
            &runtime,
            &mut [MingStepRow {
                state: &mut retained,
                cache: &mut cache,
            }],
        )
        .unwrap();
        assert_eq!(retained.latents_generated(), 3);
        retained.rollback_quantum(&mut cache, &checkpoint).unwrap();
        assert_eq!(retained.latents_generated(), 2);
        let latents = run(&runtime, &mut retained, &mut cache, usize::MAX);
        assert_eq!(latents, expected);
    }

    #[test]
    fn rows_decoded_together_match_each_alone() {
        let runtime = tiny_runtime();
        let texts = ["first row here", "another one", "and the third row of text"];
        let cfg = runtime.trunk().config();
        let caches = |rows: usize| {
            crate::models::shared::attention::physical::test_physical_caches_with_dtype_on(
                runtime.device(),
                DType::F32,
                cfg.num_hidden_layers,
                cfg.num_key_value_heads,
                cfg.head_dim,
                256,
                rows,
            )
        };
        let alone = texts
            .iter()
            .map(|text| {
                let mut retained = state(&runtime, text, 7);
                let mut cache = caches(1).pop().unwrap();
                run(&runtime, &mut retained, &mut cache, usize::MAX)
            })
            .collect::<Vec<_>>();
        let mut states = texts
            .iter()
            .map(|text| state(&runtime, text, 7))
            .collect::<Vec<_>>();
        let mut caches = caches(texts.len());
        for (state, cache) in states.iter_mut().zip(caches.iter_mut()) {
            let span = state.prompt_tokens();
            ming_prefill_batch(&runtime, &mut [MingStepRow { state, cache }], &[span]).unwrap();
        }
        loop {
            let mut rows = states
                .iter_mut()
                .zip(caches.iter_mut())
                .filter(|(state, _)| state.phase() == MingPhase::Decode)
                .map(|(state, cache)| MingStepRow { state, cache })
                .collect::<Vec<_>>();
            if rows.is_empty() {
                break;
            }
            ming_decode_batch(&runtime, &mut rows).unwrap();
        }
        // On caches of one kind a row's latents are the bits it gets alone
        // (a scratch cache pages differently and rounds in the sixth place).
        for (state, alone) in states.iter().zip(&alone) {
            assert_eq!(state.latents.len(), alone.len());
            let worst = state
                .latents
                .iter()
                .zip(alone)
                .map(|(a, b)| {
                    let d = (a - b).abs();
                    if d.is_finite() {
                        d
                    } else {
                        f32::INFINITY
                    }
                })
                .fold(0f32, f32::max);
            assert_eq!(worst, 0.0);
        }
    }

    #[test]
    fn the_budget_ends_a_request_that_never_stops() {
        let runtime = tiny_runtime();
        let mut retained = state(&runtime, "keep going", MIN_LATENTS);
        let mut cache = runtime.scratch_cache(256).unwrap();
        run(&runtime, &mut retained, &mut cache, usize::MAX);
        assert!(retained.latents_generated() <= MIN_LATENTS);
        assert!(retained.stopped());
        if retained.latents_generated() == MIN_LATENTS
            && retained.stop_reason() == Some(MingStopReason::LatentBudget)
        {
            // The last latent was not fed back.
            assert_eq!(
                cache.context_len(),
                retained.prompt_tokens() + MIN_LATENTS - 1
            );
        }
    }
}
