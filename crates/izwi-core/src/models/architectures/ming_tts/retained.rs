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
use std::time::Instant;

use candle_core::{DType, Device, Tensor};

use super::config::LATENT_DIM;
use super::stack::MingForwardRow;
use super::stream::{MingAudioStream, MingWindow};
use super::vae::{MingVaeDecoder, MingVaePending};
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
    timings: MingTimings,
}

/// Where a request's host wall time went, for diagnosing plateaus
/// ([`MingRetainedState::trace_timings`]). A dispatch's time counts for
/// every row it carried; device work may cross host intervals.
#[derive(Debug, Clone, Copy, Default)]
struct MingTimings {
    prefill_ms: f64,
    prefill_dispatches: u32,
    decode_ms: f64,
    decode_steps: u32,
    /// Rows of every decode dispatch this request was in, summed.
    decode_rows: u64,
    /// Audio dispatches that ran one of this request's windows.
    vae_ms: f64,
    /// This request's own inverse STFT (host).
    istft_ms: f64,
    windows: u32,
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
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
            timings: MingTimings::default(),
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
    #[cfg(test)]
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

    /// One log line of where this request's time went: prefill, decode
    /// (the shared dispatches' wall time and their mean width), the VAE
    /// windows and their inverse STFTs, and the latents.
    pub(crate) fn trace_timings(&self, request_id: &str) {
        let t = &self.timings;
        tracing::info!(
            request_id,
            timing_clock = "host_wall",
            prefill_ms = t.prefill_ms,
            prefill_dispatches = t.prefill_dispatches,
            decode_ms = t.decode_ms,
            decode_steps = t.decode_steps,
            mean_decode_width = t.decode_rows as f64 / f64::from(t.decode_steps.max(1)),
            vae_ms = t.vae_ms,
            istft_ms = t.istft_ms,
            windows = t.windows,
            latents = self.latents_generated(),
            "Ming execution timing (dispatch wall time counts for every row; GPU work may cross host intervals)"
        );
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

/// The next audio window of each state, `None` for a state with none due.
/// Every due window's stack is queued on the device first, then each is
/// read back and run through the inverse STFT in turn, so the host's work
/// on one window overlaps the device's on the next. Each window decodes on
/// its own (the VAE keeps per-request attention state): a request gets the
/// samples it would alone.
pub(crate) fn ming_audio_batch(
    decoder: &MingVaeDecoder,
    states: &mut [&mut MingRetainedState],
) -> Result<Vec<Option<Vec<f32>>>> {
    let started = Instant::now();
    let mut queued: Vec<Option<(MingWindow, MingVaePending)>> = Vec::with_capacity(states.len());
    for state in states.iter_mut() {
        queued.push(match state.next_window() {
            Some(window) => {
                let pending = state.audio.begin(decoder, &state.latents, window)?;
                Some((window, pending))
            }
            None => None,
        });
    }
    let mut out = Vec::with_capacity(states.len());
    for (state, queued) in states.iter_mut().zip(queued) {
        out.push(match queued {
            Some((window, pending)) => {
                let istft = Instant::now();
                let samples = state.audio.finish(decoder, pending, window)?;
                state.timings.istft_ms += elapsed_ms(istft);
                state.timings.windows += 1;
                Some(samples)
            }
            None => None,
        });
    }
    let elapsed = elapsed_ms(started);
    for (state, samples) in states.iter_mut().zip(&out) {
        if samples.is_some() {
            state.timings.vae_ms += elapsed;
        }
    }
    Ok(out)
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
    let started = Instant::now();
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
    let elapsed = elapsed_ms(started);
    for ((row, step), state) in rows.iter_mut().zip(&steps).zip(states) {
        row.state.timings.prefill_ms += elapsed;
        row.state.timings.prefill_dispatches += 1;
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
///
/// The device is waited on once. Whether a row stops needs only its pending
/// hidden state (the stop head runs on the host) and its step, so that is
/// decided first; the sampled latents stay on the device and feed the
/// aggregator directly, the language model runs, and one readback brings
/// the latents and the new hidden states back together. Nothing is
/// committed before that readback's checks pass.
pub(crate) fn ming_decode_batch(
    runtime: &MingRuntime,
    rows: &mut [MingStepRow<'_>],
) -> Result<Vec<MingDecodeStep>> {
    if rows.is_empty() {
        return Err(Error::InvalidInput("Ming decode batch has no rows".into()));
    }
    let started = Instant::now();
    // Every row's inputs first: its hidden state, history and its own
    // seeded noise, and whether this latent is its last. Then one flow-head
    // call for all of them, and one aggregator call for the rows that go
    // on: each row's values are the ones it gets alone (see `flow_group`).
    let mut inputs = Vec::with_capacity(rows.len());
    let mut stops = Vec::with_capacity(rows.len());
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
        let step = state.latents_generated();
        stops.push(if runtime.stops_at(&hidden, step) {
            Some(MingStopReason::EndOfSpeech)
        } else if step + 1 >= state.budget {
            Some(MingStopReason::LatentBudget)
        } else {
            None
        });
        let mut noise = state.rng.standard_normals(LATENT_DIM);
        noise.truncate(LATENT_DIM);
        inputs.push((hidden, noise));
    }
    let latents = runtime.sample_latents_device(
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
    let going_on = stops
        .iter()
        .enumerate()
        .filter_map(|(index, stop)| stop.is_none().then_some(index as u32))
        .collect::<Vec<_>>();
    // The latents of the rows that go on, through the aggregator and the
    // language model; the last latent is kept and never fed back.
    let hidden = if going_on.is_empty() {
        None
    } else {
        let fed_back = if going_on.len() == rows.len() {
            latents.clone()
        } else {
            let picked = Tensor::from_vec(going_on.clone(), going_on.len(), runtime.device())?;
            latents.index_select(&picked, 0)?
        };
        let embeds = runtime.latent_embeddings_device(&fed_back)?;
        let next = (0..going_on.len())
            .map(|index| embeds.narrow(0, index, 1))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let mut forward = rows
            .iter_mut()
            .zip(&stops)
            .filter(|(_, stop)| stop.is_none())
            .zip(&next)
            .map(|((row, _), embeds)| MingForwardRow {
                embeds,
                cache: &mut *row.cache,
            })
            .collect::<Vec<_>>();
        Some(runtime.trunk().forward_batch(&mut forward)?)
    };
    // The one readback: every latent, then every new hidden state.
    let flat = latents.flatten_all()?;
    let values = match &hidden {
        Some(hidden) => Tensor::cat(&[&flat, &hidden.to_dtype(DType::F32)?.flatten_all()?], 0)?,
        None => flat,
    }
    .to_device(&Device::Cpu)?
    .to_vec1::<f32>()?;
    let (sampled, states) = values.split_at(rows.len() * LATENT_DIM);
    if sampled.iter().any(|value| !value.is_finite()) {
        return Err(Error::InferenceError(
            "Ming flow head produced a non-finite latent".into(),
        ));
    }
    let hidden_size = runtime.config().llm.hidden_size;
    if states.len() != going_on.len() * hidden_size {
        return Err(Error::InferenceError(
            "Ming decode step lost a row's hidden state".into(),
        ));
    }
    let mut states = states.chunks_exact(hidden_size);
    let mut steps = Vec::with_capacity(rows.len());
    let (elapsed, width) = (elapsed_ms(started), rows.len() as u64);
    for ((row, latent), stop) in rows
        .iter_mut()
        .zip(sampled.as_chunks::<LATENT_DIM>().0)
        .zip(stops)
    {
        let state = &mut *row.state;
        state.timings.decode_ms += elapsed;
        state.timings.decode_steps += 1;
        state.timings.decode_rows += width;
        state.latents.extend_from_slice(latent);
        match stop {
            Some(reason) => {
                state.phase = MingPhase::Audio;
                state.stop_reason = Some(reason);
            }
            None => {
                state.history = latent.to_vec();
                state.pending = states.next().map(<[f32]>::to_vec);
            }
        }
        steps.push(MingDecodeStep {
            finished: stop.is_some(),
        });
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

    /// Each text's latents decoded alone, and all of them decoded together
    /// (the rows that are still decoding share each step), on caches of the
    /// runtime's device and dtype.
    fn latents_alone_and_together(
        runtime: &MingRuntime,
        texts: &[&str],
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let cfg = runtime.trunk().config();
        let caches = |rows: usize| {
            crate::models::shared::attention::physical::test_physical_caches_with_dtype_on(
                runtime.device(),
                runtime.lm_dtype(),
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
                let mut retained = state(runtime, text, 7);
                let mut cache = caches(1).pop().unwrap();
                run(runtime, &mut retained, &mut cache, usize::MAX)
            })
            .collect::<Vec<_>>();
        let mut states = texts
            .iter()
            .map(|text| state(runtime, text, 7))
            .collect::<Vec<_>>();
        let mut caches = caches(texts.len());
        for (state, cache) in states.iter_mut().zip(caches.iter_mut()) {
            let span = state.prompt_tokens();
            ming_prefill_batch(runtime, &mut [MingStepRow { state, cache }], &[span]).unwrap();
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
            ming_decode_batch(runtime, &mut rows).unwrap();
        }
        let together = states.into_iter().map(|state| state.latents).collect();
        (alone, together)
    }

    fn bits(rows: &[Vec<f32>]) -> Vec<Vec<u32>> {
        rows.iter()
            .map(|row| row.iter().map(|value| value.to_bits()).collect())
            .collect()
    }

    #[test]
    fn rows_decoded_together_match_each_alone() {
        let runtime = tiny_runtime();
        let texts = ["first row here", "another one", "and the third row of text"];
        // On caches of one kind a row's latents are the bits it gets alone
        // (a scratch cache pages differently and rounds in the sixth place).
        let (alone, together) = latents_alone_and_together(&runtime, &texts);
        assert!(alone.iter().all(|latents| !latents.is_empty()));
        assert_eq!(bits(&together), bits(&alone));
    }

    /// On CUDA, with the decode stretches graphed at their one padded width:
    /// from eleven rows down to one, a row's latents are the bits it gets
    /// alone, and the bits the eager (ungraphed) steps give.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_rows_decoded_together_match_each_alone_and_eager() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        let texts = [
            "first row here",
            "another one",
            "and the third row of text",
            "one two three",
            "keep going",
            "hello there friend",
            "the text",
            "row of text here",
            "two one",
            "three two one friend",
            "first another third",
        ];
        let synthetic = crate::models::architectures::ming_tts::synthetic::tiny_runtime_on;
        let graphed = synthetic(&device, DType::BF16);
        let (alone, together) = latents_alone_and_together(&graphed, &texts);
        assert_eq!(bits(&together), bits(&alone));
        let diagnostics = graphed.decode_graphs_diagnostics();
        assert!(diagnostics["off"].is_null(), "{diagnostics}");
        assert_eq!(
            diagnostics["widths"],
            serde_json::json!([32]),
            "{diagnostics}"
        );
        assert!(
            diagnostics["replays"].as_u64().unwrap_or(0) > 0,
            "{diagnostics}"
        );
        let mut eager = synthetic(&device, DType::BF16);
        eager.turn_off_decode_graphs();
        let (eager_alone, eager_together) = latents_alone_and_together(&eager, &texts);
        assert_eq!(bits(&eager_alone), bits(&alone));
        assert_eq!(bits(&eager_together), bits(&alone));
    }

    /// Every state's windows through [`ming_audio_batch`], `together` in one
    /// batch per round or each state alone; each state's samples as bits.
    fn audio_bits(
        decoder: &MingVaeDecoder,
        states: &mut [MingRetainedState],
        together: bool,
    ) -> Vec<Vec<u32>> {
        let mut out = vec![Vec::new(); states.len()];
        if together {
            loop {
                let mut refs = states.iter_mut().collect::<Vec<_>>();
                let windows = ming_audio_batch(decoder, &mut refs).unwrap();
                if windows.iter().all(Option::is_none) {
                    break;
                }
                for (samples, window) in out.iter_mut().zip(windows) {
                    samples.extend(window.unwrap_or_default().into_iter().map(f32::to_bits));
                }
            }
        } else {
            for (samples, state) in out.iter_mut().zip(states.iter_mut()) {
                while let Some(window) = ming_audio_batch(decoder, &mut [state]).unwrap().remove(0)
                {
                    samples.extend(window.into_iter().map(f32::to_bits));
                }
            }
        }
        assert!(states.iter().all(|state| state.audio.done()));
        out
    }

    /// Windows queued together are each request's samples alone, bit for
    /// bit (on the CPU, at each window's own length).
    #[test]
    fn audio_windows_together_match_each_alone() {
        let runtime = tiny_runtime();
        let fresh = || {
            ["first row here", "another one", "and the third row of text"]
                .iter()
                .map(|text| {
                    let mut retained = state(&runtime, text, 20);
                    let mut cache = runtime.scratch_cache(256).unwrap();
                    run(&runtime, &mut retained, &mut cache, usize::MAX);
                    retained
                })
                .collect::<Vec<_>>()
        };
        let alone = audio_bits(runtime.decoder(), &mut fresh(), false);
        assert!(alone.iter().all(|samples| !samples.is_empty()));
        assert_eq!(audio_bits(runtime.decoder(), &mut fresh(), true), alone);
    }

    /// On CUDA, through the VAE window's graph: windows queued together are
    /// each request's samples alone, bit for bit, and the graph replays.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_audio_windows_together_match_each_alone() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        let mut runtime =
            crate::models::architectures::ming_tts::synthetic::tiny_runtime_on(&device, DType::F32);
        runtime.resolve_graphs();
        let fresh = || {
            ["first row here", "another one", "and the third row of text"]
                .iter()
                .map(|text| {
                    let mut retained = state(&runtime, text, 20);
                    let mut cache = runtime.scratch_cache(256).unwrap();
                    run(&runtime, &mut retained, &mut cache, usize::MAX);
                    retained
                })
                .collect::<Vec<_>>()
        };
        let alone = audio_bits(runtime.decoder(), &mut fresh(), false);
        assert_eq!(audio_bits(runtime.decoder(), &mut fresh(), true), alone);
        let diagnostics = runtime.vae_graphs_diagnostics();
        assert!(diagnostics["off"].is_null(), "{diagnostics}");
        assert!(
            diagnostics["replays"].as_u64().unwrap_or(0) > 0,
            "{diagnostics}"
        );
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
