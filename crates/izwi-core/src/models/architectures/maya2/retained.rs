//! Rollback-safe retained Maya2 generation, quantum by quantum.
//!
//! A request's backbone sequence lives in one engine-owned paged cache. Its
//! host state (prefill cursor, the pending backbone state, emitted codes,
//! the cb0 sampler, the loop guard and both random streams) is checkpointed
//! with the cache at the start of each quantum, so a failed or cancelled
//! quantum rolls back to exactly where it began.
//!
//! - **Prefill** may arrive in chunks of the composed prompt embeddings;
//!   the last chunk leaves the backbone state that yields frame 0.
//! - **Decode** steps every row once, in one batch: the codebook-0 head and
//!   the host draw (with the controlled stop), the depth decoder for the
//!   rows that go on, the frame's read-back, the budget and loop guard, and
//!   one backbone step over the new frames.
//! - **Audio** runs Mimi's decoder over the frames a row has not decoded,
//!   a few at a time ([`MAYA2_AUDIO_PLAN`]); Mimi is causal, so a streamed
//!   request and a final-only one produce the same samples. A failed
//!   window discards the request.

use std::sync::Arc;

use candle_core::{Device, Tensor};

use super::backbone::{Maya2Backbone, Maya2BackboneRow};
use super::config::{CODEBOOKS, CODEBOOK_SIZE, DEPTH_LEVELS, MAYA2_DEFAULTS};
use super::depth::Maya2Depth;
use super::guards::{LoopGuard, Maya2PieceBudget, Maya2StopReason};
use super::inputs::Maya2Inputs;
use super::runtime::{cap_reason, Maya2GenerationParams, Maya2Stream, Maya2Voice};
use super::sampling::{depth_gumbel, sample_cb0, Cb0State};
use crate::error::{Error, Result};
use crate::models::shared::attention::physical::{PhysicalPagedKvCache, PhysicalPagedKvCheckpoint};
use crate::models::shared::codecs::mimi::{MimiDecoder, MimiDecoderState};
use crate::models::shared::noise::SplitMix64;

/// One request, prepared before it reaches the engine.
pub(crate) struct Maya2PreparedArtifact {
    pub(crate) voice: Maya2Voice,
    pub(crate) language: &'static str,
    pub(crate) text: String,
    pub(crate) prompt: super::prompt::Maya2Prompt,
    pub(crate) budget: Maya2PieceBudget,
    pub(crate) suspend: Vec<(usize, usize)>,
}

impl std::fmt::Debug for Maya2PreparedArtifact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Maya2PreparedArtifact")
            .field("language", &self.language)
            .field("prompt_positions", &self.prompt.items.len())
            .field("budget", &self.budget)
            .finish()
    }
}

/// When a row's Mimi windows run: the first after `first_frames` (quick first
/// audio), then every `frames` frames, and once the model stops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Maya2AudioPlan {
    pub first_frames: usize,
    pub frames: usize,
}

/// One frame (80 ms) for first audio, then 4 frames (320 ms) per window.
pub const MAYA2_AUDIO_PLAN: Maya2AudioPlan = Maya2AudioPlan {
    first_frames: 1,
    frames: 4,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Maya2Phase {
    Prefill,
    Decode,
    /// The model has stopped; audio windows remain.
    Audio,
    Finished,
}

#[derive(Clone)]
struct HostCheckpoint {
    prefill_cursor: usize,
    pending: Option<Tensor>,
    frames: usize,
    fed_frames: usize,
    cb0_state: Cb0State,
    guard: LoopGuard,
    cb0_rng: SplitMix64,
    depth_rng: SplitMix64,
    phase: Maya2Phase,
    stop_reason: Option<Maya2StopReason>,
}

pub(crate) struct Maya2QuantumCheckpoint {
    cache: PhysicalPagedKvCheckpoint,
    host: HostCheckpoint,
}

/// The device-side conditioning of a request, built at its first prefill:
/// the prompt embeddings and the depth decoder's cross constants.
struct Conditioning {
    embeds: Tensor,
    cross: Tensor,
}

pub(crate) struct Maya2RetainedState {
    pub(crate) artifact: Arc<Maya2PreparedArtifact>,
    pub(crate) params: Maya2GenerationParams,
    conditioning: Option<Conditioning>,
    prefill_cursor: usize,
    /// The backbone's last state `[1, hidden]`, the next frame's source.
    pending: Option<Tensor>,
    /// Every emitted frame's 32 codes, frame-major.
    codes: Vec<u32>,
    /// Emitted frames fed back through the backbone (all but a final frame
    /// emitted at a cap or loop stop).
    fed_frames: usize,
    cb0_state: Cb0State,
    guard: LoopGuard,
    cb0_rng: SplitMix64,
    depth_rng: SplitMix64,
    phase: Maya2Phase,
    stop_reason: Option<Maya2StopReason>,
    audio: Option<MimiDecoderState>,
    /// Frames Mimi has decoded and samples handed over.
    decoded_frames: usize,
    samples: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Maya2PrefillStep {
    pub(crate) consumed_tokens: usize,
    pub(crate) complete: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Maya2DecodeStep {
    pub(crate) frames_generated: usize,
    pub(crate) finished: bool,
}

/// One request's state and cache in a batched quantum.
pub(crate) struct Maya2StepRow<'a> {
    pub(crate) state: &'a mut Maya2RetainedState,
    pub(crate) cache: &'a mut PhysicalPagedKvCache,
}

/// The networks a quantum runs.
pub(crate) struct Maya2Networks<'a> {
    pub(crate) backbone: &'a Maya2Backbone,
    pub(crate) inputs: &'a Maya2Inputs,
    pub(crate) depth: &'a Maya2Depth,
    pub(crate) conditioning: &'a dyn Fn(&Maya2PreparedArtifact) -> Result<(Tensor, Tensor)>,
}

impl Maya2RetainedState {
    pub(crate) fn new(
        artifact: Arc<Maya2PreparedArtifact>,
        params: Maya2GenerationParams,
        context_limit: usize,
    ) -> Result<Self> {
        params.validate()?;
        let required = Self::required_context(artifact.prompt.items.len(), artifact.budget.cap);
        if artifact.prompt.items.is_empty() || required > context_limit {
            return Err(Error::InvalidInput(format!(
                "Maya2 request needs {required} positions, beyond its {context_limit}-position context"
            )));
        }
        Ok(Self {
            cb0_rng: SplitMix64::for_stream(params.seed, Maya2Stream::Cb0),
            depth_rng: SplitMix64::for_stream(params.seed, Maya2Stream::Depth),
            guard: LoopGuard::new(artifact.suspend.clone()),
            artifact,
            params,
            conditioning: None,
            prefill_cursor: 0,
            pending: None,
            codes: Vec::new(),
            fed_frames: 0,
            cb0_state: Cb0State::default(),
            phase: Maya2Phase::Prefill,
            stop_reason: None,
            audio: None,
            decoded_frames: 0,
            samples: 0,
        })
    }

    /// Positions the sequence reaches: the prompt plus every emitted frame
    /// but the last (which is never fed back).
    pub(crate) fn required_context(prompt_positions: usize, max_frames: usize) -> usize {
        prompt_positions + max_frames.saturating_sub(1)
    }

    pub(crate) fn phase(&self) -> Maya2Phase {
        self.phase
    }

    pub(crate) fn prompt_tokens(&self) -> usize {
        self.artifact.prompt.items.len()
    }

    pub(crate) fn prefill_cursor(&self) -> usize {
        self.prefill_cursor
    }

    pub(crate) fn frames(&self) -> usize {
        self.codes.len() / CODEBOOKS
    }

    pub(crate) fn codes(&self) -> &[u32] {
        &self.codes
    }

    pub(crate) fn stop_reason(&self) -> Option<Maya2StopReason> {
        self.stop_reason
    }

    pub(crate) fn loop_boosts(&self) -> usize {
        self.guard.boosts
    }

    /// Samples handed over so far.
    pub(crate) fn samples(&self) -> usize {
        self.samples
    }

    /// Whether the model has emitted its last frame.
    pub(crate) fn stopped(&self) -> bool {
        matches!(self.phase, Maya2Phase::Audio | Maya2Phase::Finished)
    }

    /// The frames the next audio window decodes, if one is due.
    pub(crate) fn next_window(&self) -> Option<std::ops::Range<usize>> {
        let emitted = self.frames();
        let pending = emitted.saturating_sub(self.decoded_frames);
        if pending == 0 {
            return None;
        }
        let due = if self.decoded_frames == 0 {
            pending >= MAYA2_AUDIO_PLAN.first_frames
        } else {
            pending >= MAYA2_AUDIO_PLAN.frames
        };
        (due || self.stopped()).then_some(self.decoded_frames..emitted)
    }

    /// Whether every emitted frame has been decoded and handed over.
    pub(crate) fn audio_done(&self) -> bool {
        self.stopped() && self.decoded_frames == self.frames()
    }

    /// Decode the next window with `mimi`: its samples (clamped to ±1), or
    /// `None` when no window is due.
    pub(crate) fn decode_window(&mut self, mimi: &MimiDecoder) -> Result<Option<Vec<f32>>> {
        let Some(window) = self.next_window() else {
            return Ok(None);
        };
        let state = match self.audio.as_mut() {
            Some(state) => state,
            None => self.audio.insert(mimi.new_state()?),
        };
        let codes = &self.codes[window.start * CODEBOOKS..window.end * CODEBOOKS];
        let samples = mimi
            .decode_stream(state, codes)?
            .to_vec1::<f32>()?
            .into_iter()
            .map(|sample| sample.clamp(-1.0, 1.0))
            .collect::<Vec<_>>();
        self.decoded_frames = window.end;
        self.samples += samples.len();
        Ok(Some(samples))
    }

    /// Positions the cache must hold now.
    fn position(&self) -> usize {
        match self.phase {
            Maya2Phase::Prefill => self.prefill_cursor,
            _ => self.prompt_tokens() + self.fed_frames,
        }
    }

    fn host_checkpoint(&self) -> HostCheckpoint {
        HostCheckpoint {
            prefill_cursor: self.prefill_cursor,
            pending: self.pending.clone(),
            frames: self.frames(),
            fed_frames: self.fed_frames,
            cb0_state: self.cb0_state.clone(),
            guard: self.guard.clone(),
            cb0_rng: self.cb0_rng.clone(),
            depth_rng: self.depth_rng.clone(),
            phase: self.phase,
            stop_reason: self.stop_reason,
        }
    }

    fn check_cache(&self, cache: &PhysicalPagedKvCache) -> Result<()> {
        if cache.context_len() != self.position() {
            return Err(Error::InvalidInput(
                "Maya2 retained cache crossed host state".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn begin_quantum(
        &self,
        cache: &PhysicalPagedKvCache,
    ) -> Result<Maya2QuantumCheckpoint> {
        self.check_cache(cache)?;
        Ok(Maya2QuantumCheckpoint {
            cache: cache.logical_checkpoint(),
            host: self.host_checkpoint(),
        })
    }

    pub(crate) fn commit_quantum(
        &self,
        cache: &PhysicalPagedKvCache,
        checkpoint: &Maya2QuantumCheckpoint,
    ) -> Result<()> {
        self.check_cache(cache)?;
        if self.frames() < checkpoint.host.frames
            || self.prefill_cursor < checkpoint.host.prefill_cursor
        {
            return Err(Error::InferenceError(
                "Maya2 quantum moved its state backwards".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn rollback_quantum(
        &mut self,
        cache: &mut PhysicalPagedKvCache,
        checkpoint: &Maya2QuantumCheckpoint,
    ) -> Result<()> {
        cache.restore_logical_checkpoint(checkpoint.cache.clone())?;
        let host = checkpoint.host.clone();
        self.codes.truncate(host.frames * CODEBOOKS);
        self.fed_frames = host.fed_frames;
        self.prefill_cursor = host.prefill_cursor;
        self.pending = host.pending;
        self.cb0_state = host.cb0_state;
        self.guard = host.guard;
        self.cb0_rng = host.cb0_rng;
        self.depth_rng = host.depth_rng;
        self.phase = host.phase;
        self.stop_reason = host.stop_reason;
        Ok(())
    }

    fn stop(&mut self, reason: Maya2StopReason) {
        self.stop_reason = Some(reason);
        self.phase = Maya2Phase::Audio;
    }

    pub(crate) fn mark_finished(&mut self) -> Result<()> {
        if !self.audio_done() {
            return Err(Error::InferenceError(
                "Maya2 finished before its last audio window".into(),
            ));
        }
        self.phase = Maya2Phase::Finished;
        Ok(())
    }
}

/// Prefill `tokens[i]` more prompt positions of each row, in one call.
pub(crate) fn maya2_prefill_batch(
    networks: &Maya2Networks<'_>,
    rows: &mut [Maya2StepRow<'_>],
    tokens: &[usize],
) -> Result<Vec<Maya2PrefillStep>> {
    if rows.is_empty() || rows.len() != tokens.len() {
        return Err(Error::InvalidInput(
            "Maya2 prefill batch rows do not match".into(),
        ));
    }
    let mut steps = Vec::with_capacity(rows.len());
    for (row, &count) in rows.iter_mut().zip(tokens) {
        let state = &mut *row.state;
        let start = state.prefill_cursor;
        let total = state.prompt_tokens();
        if state.phase != Maya2Phase::Prefill || count == 0 || start + count > total {
            return Err(Error::InvalidInput(format!(
                "Maya2 prefill of {count} positions at {start} exceeds its {total}-position prompt"
            )));
        }
        if state.conditioning.is_none() {
            let (embeds, cross) = (networks.conditioning)(&state.artifact)?;
            state.conditioning = Some(Conditioning { embeds, cross });
        }
        steps.push(Maya2PrefillStep {
            consumed_tokens: count,
            complete: start + count == total,
        });
    }
    let mut forward = Vec::with_capacity(rows.len());
    for (row, step) in rows.iter_mut().zip(&steps) {
        let start = row.state.prefill_cursor;
        let embeds = row
            .state
            .conditioning
            .as_ref()
            .expect("conditioning built above")
            .embeds
            .narrow(0, start, step.consumed_tokens)?;
        forward.push(Maya2BackboneRow {
            embeds,
            cache: &mut *row.cache,
        });
    }
    let hidden = networks.backbone.forward_batch(&mut forward)?;
    drop(forward);
    for (index, (row, step)) in rows.iter_mut().zip(&steps).enumerate() {
        row.state.prefill_cursor += step.consumed_tokens;
        if step.complete {
            row.state.phase = Maya2Phase::Decode;
            row.state.pending = Some(hidden.narrow(0, index, 1)?);
        }
    }
    Ok(steps)
}

/// One frame for each row: the cb0 draws (and controlled stops), the depth
/// decoder for the rows that go on, then one backbone step over the frames
/// of the rows still decoding.
pub(crate) fn maya2_decode_batch(
    networks: &Maya2Networks<'_>,
    rows: &mut [Maya2StepRow<'_>],
    device: &Device,
) -> Result<Vec<Maya2DecodeStep>> {
    if rows.is_empty() {
        return Err(Error::InvalidInput("Maya2 decode batch is empty".into()));
    }
    let mut hidden = Vec::with_capacity(rows.len());
    for row in rows.iter() {
        if row.state.phase != Maya2Phase::Decode {
            return Err(Error::InvalidInput(
                "Maya2 decode step outside its decode phase".into(),
            ));
        }
        hidden.push(row.state.pending.clone().ok_or_else(|| {
            Error::InferenceError("Maya2 decode step has no pending state".into())
        })?);
    }
    let hidden = Tensor::cat(&hidden, 0)?;
    let logits = networks.inputs.cb0_logits(&hidden)?.to_vec2::<f32>()?;

    // Host draws; rows that stop on the draw emit nothing.
    let mut going = Vec::with_capacity(rows.len());
    let mut cb0 = Vec::with_capacity(rows.len());
    for (index, (row, logits)) in rows.iter_mut().zip(&logits).enumerate() {
        let state = &mut *row.state;
        let draw = sample_cb0(
            logits,
            &state.params.sampling,
            state.guard.aeos_bias(),
            state.frames(),
            state.artifact.budget.minimum,
            &mut state.cb0_state,
            &mut state.cb0_rng,
        )?;
        match draw.stop {
            Some(reason) => {
                state.pending = None;
                state.stop(reason);
            }
            None => {
                going.push(index);
                cb0.push(draw.code);
            }
        }
    }

    if !going.is_empty() {
        let pick = |tensor: &Tensor| -> Result<Tensor> {
            if going.len() == rows.len() {
                return Ok(tensor.clone());
            }
            let index = Tensor::from_vec(
                going.iter().map(|&i| i as u32).collect::<Vec<_>>(),
                going.len(),
                device,
            )?;
            Ok(tensor.index_select(&index, 0)?)
        };
        let h = pick(&hidden)?;
        let cross = going
            .iter()
            .map(|&index| {
                Ok(rows[index]
                    .state
                    .conditioning
                    .as_ref()
                    .ok_or_else(|| Error::InferenceError("Maya2 row lost its conditioning".into()))?
                    .cross
                    .clone())
            })
            .collect::<Result<Vec<_>>>()?;
        let cross = Tensor::stack(&cross, 0)?;
        let gumbel = {
            let mut rngs = Vec::with_capacity(going.len());
            let mut cursor = 0;
            for (index, row) in rows.iter_mut().enumerate() {
                if going.get(cursor) == Some(&index) {
                    rngs.push(&mut row.state.depth_rng);
                    cursor += 1;
                }
            }
            depth_gumbel(&mut rngs, DEPTH_LEVELS, CODEBOOK_SIZE, device)?
        };
        let cb0 = Tensor::from_vec(cb0, going.len(), device)?;
        let frames = networks.depth.generate_frame(
            &h,
            &cb0,
            &cross,
            &gumbel,
            MAYA2_DEFAULTS.depth_temperature,
            MAYA2_DEFAULTS.depth_top_k,
        )?;
        let host = frames.to_vec2::<u32>()?;

        let mut feed = Vec::with_capacity(going.len());
        for (position, &index) in going.iter().enumerate() {
            let state = &mut *rows[index].state;
            state.codes.extend_from_slice(&host[position]);
            let emitted = state.frames();
            if emitted >= state.artifact.budget.cap {
                state.pending = None;
                state.stop(cap_reason(&state.artifact.budget, state.params.max_frames));
            } else if state.guard.observe(host[position][0], emitted) {
                state.pending = None;
                state.stop(Maya2StopReason::LoopBackstop);
            } else {
                feed.push(position);
            }
        }

        if !feed.is_empty() {
            let index = Tensor::from_vec(
                feed.iter().map(|&p| p as u32).collect::<Vec<_>>(),
                feed.len(),
                device,
            )?;
            let embeds = networks
                .inputs
                .audio_frames(&frames.index_select(&index, 0)?)?;
            let feed_rows = feed.iter().map(|&p| going[p]).collect::<Vec<_>>();
            let mut forward = Vec::with_capacity(feed.len());
            let mut cursor = 0;
            for (index, row) in rows.iter_mut().enumerate() {
                if feed_rows.get(cursor) != Some(&index) {
                    continue;
                }
                forward.push(Maya2BackboneRow {
                    embeds: embeds.narrow(0, cursor, 1)?,
                    cache: &mut *row.cache,
                });
                cursor += 1;
            }
            let next = networks.backbone.forward_batch(&mut forward)?;
            drop(forward);
            for (position, &index) in feed_rows.iter().enumerate() {
                let state = &mut *rows[index].state;
                state.pending = Some(next.narrow(0, position, 1)?);
                state.fed_frames += 1;
            }
        }
    }

    Ok(rows
        .iter()
        .map(|row| Maya2DecodeStep {
            frames_generated: row.state.frames(),
            finished: row.state.stopped(),
        })
        .collect())
}
