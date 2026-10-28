//! LFM3-Audio requests as retained engine rows.
//!
//! A row keeps its language-model state between engine quanta: paged KV and
//! the ShortConv ring on [`Lfm25AudioRetainedState`] (the same backbone and
//! ring as LFM2.5-Audio, main-only), and the generate loop's host state in an
//! [`Lfm3AudioRow`]. Each quantum opens a checkpoint, so a failed or
//! cancelled quantum rolls back to the frame boundary it started at. The
//! prompt prefills one row at a time; decode steps every row together, the
//! language model in one batched forward.

use std::sync::Arc;

use candle_core::Tensor;

use crate::backends::state::TensorStateArena;
use crate::error::{Error, Result};
use crate::models::architectures::lfm25_audio::state::{
    Lfm25AudioRetainedCheckpoint, Lfm25AudioRetainedMode, Lfm25AudioRetainedState,
};
use crate::models::shared::attention::physical::PhysicalPagedKvCache;

use super::model::{Lfm3AudioModel, Lfm3AudioOutput, Lfm3AudioRequest, Lfm3AudioRow};

/// A request prepared before it runs: its prompt embeddings and the inputs
/// its row starts from.
pub(crate) struct Lfm3AudioPreparedArtifact {
    pub(crate) request: Lfm3AudioRequest,
    /// `[1, prompt_tokens, 2048]` on the model's device.
    pub(crate) embeds: Tensor,
    pub(crate) prompt_tokens: usize,
}

impl std::fmt::Debug for Lfm3AudioPreparedArtifact {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Lfm3AudioPreparedArtifact")
            .field("prompt_tokens", &self.prompt_tokens)
            .field("max_slots", &self.request.max_slots)
            .finish()
    }
}

/// The state a quantum restores on rollback.
pub(crate) struct Lfm3AudioRetainedCheckpoint {
    retained: Lfm25AudioRetainedCheckpoint,
    prefill_cursor: usize,
    row: Option<Lfm3AudioRow>,
}

/// One request's retained state.
pub(crate) struct Lfm3AudioRetainedState {
    artifact: Arc<Lfm3AudioPreparedArtifact>,
    retained: Lfm25AudioRetainedState,
    /// Positions the row's reservation holds.
    context: usize,
    prefill_cursor: usize,
    /// The generate loop, once the prompt has prefilled.
    row: Option<Lfm3AudioRow>,
}

impl Lfm3AudioRetainedState {
    pub(crate) fn new(
        model: &Lfm3AudioModel,
        artifact: Arc<Lfm3AudioPreparedArtifact>,
        context: usize,
    ) -> Result<Self> {
        let context = context.min(model.max_context_tokens());
        if artifact.prompt_tokens == 0 || artifact.prompt_tokens >= context {
            return Err(Error::InvalidInput(format!(
                "LFM3-Audio prompt of {} tokens does not fit its {context}-token row",
                artifact.prompt_tokens
            )));
        }
        Ok(Self {
            artifact,
            retained: Lfm25AudioRetainedState::new(
                Lfm25AudioRetainedMode::Asr,
                model.new_lm_shortconv_state(),
                0,
            ),
            context,
            prefill_cursor: 0,
            row: None,
        })
    }

    pub(crate) fn prompt_tokens(&self) -> usize {
        self.artifact.prompt_tokens
    }

    pub(crate) fn prefill_cursor(&self) -> usize {
        self.prefill_cursor
    }

    /// Whether the row has stopped: its output is complete.
    pub(crate) fn finished(&self) -> bool {
        self.row.as_ref().is_some_and(Lfm3AudioRow::finished)
    }

    /// The row's output so far (`None` before the prompt has prefilled).
    pub(crate) fn output(&self) -> Option<&Lfm3AudioOutput> {
        self.row.as_ref().map(Lfm3AudioRow::output)
    }

    /// Samples decoded since the last call.
    pub(crate) fn take_samples(&mut self) -> Vec<f32> {
        self.row
            .as_mut()
            .map(Lfm3AudioRow::take_samples)
            .unwrap_or_default()
    }

    /// Text decoded since the last call.
    pub(crate) fn take_text(&mut self) -> String {
        self.row
            .as_mut()
            .map(Lfm3AudioRow::take_text)
            .unwrap_or_default()
    }

    pub(crate) fn begin_quantum(
        &mut self,
        main: &PhysicalPagedKvCache,
    ) -> Result<Lfm3AudioRetainedCheckpoint> {
        Ok(Lfm3AudioRetainedCheckpoint {
            retained: self.retained.begin_main_quantum(main)?,
            prefill_cursor: self.prefill_cursor,
            row: self.row.clone(),
        })
    }

    pub(crate) fn commit_quantum(
        &mut self,
        main: &PhysicalPagedKvCache,
        checkpoint: &Lfm3AudioRetainedCheckpoint,
    ) -> Result<()> {
        self.retained
            .commit_quantum(main, None, &checkpoint.retained)
    }

    pub(crate) fn rollback_quantum(
        &mut self,
        main: &mut PhysicalPagedKvCache,
        checkpoint: &Lfm3AudioRetainedCheckpoint,
    ) -> Result<()> {
        self.prefill_cursor = checkpoint.prefill_cursor;
        self.row = checkpoint.row.clone();
        self.retained
            .rollback_quantum(main, None, &checkpoint.retained)
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

    /// Stop the row (a cancelled request): its text tail and last speech
    /// decode, and it takes no further slot.
    pub(crate) fn stop(&mut self, model: &Lfm3AudioModel, reason: &'static str) -> Result<()> {
        match self.row.as_mut() {
            Some(row) => model.stop_row(row, reason),
            None => Ok(()),
        }
    }
}

/// A row in a batched step: its state and its paged KV.
pub(crate) struct Lfm3AudioStepRow<'a> {
    pub(crate) state: &'a mut Lfm3AudioRetainedState,
    pub(crate) cache: &'a mut PhysicalPagedKvCache,
}

impl Lfm3AudioModel {
    /// Prefill the next `span` prompt tokens of one row. The prompt's last
    /// span starts the row's generate loop.
    pub(crate) fn retained_prefill(
        &self,
        state: &mut Lfm3AudioRetainedState,
        cache: &mut PhysicalPagedKvCache,
        span: usize,
    ) -> Result<()> {
        let start = state.prefill_cursor;
        let prompt_tokens = state.artifact.prompt_tokens;
        if state.row.is_some() || span == 0 || start + span > prompt_tokens {
            return Err(Error::InvalidInput(format!(
                "LFM3-Audio prefill span {start}+{span} does not fit its {prompt_tokens}-token prompt"
            )));
        }
        let embeds = state.artifact.embeds.narrow(1, start, span)?;
        let hidden = self.lm().forward_embeds_retained(
            &embeds,
            start,
            cache,
            state.retained.shortconv_mut(),
        )?;
        state.prefill_cursor = start + span;
        if state.prefill_cursor == prompt_tokens {
            state.row = Some(self.begin_row(
                &state.artifact.request,
                prompt_tokens,
                state.context,
                &hidden,
            )?);
        }
        Ok(())
    }

    /// One decode step for every row: each fills its next slot (the speaking
    /// rows' frames drawn apart, each from its own generator), then every row
    /// that continues feeds its slot back through one batched forward of the
    /// language model. Returns whether each row has finished.
    pub(crate) fn retained_decode_batch(
        &self,
        rows: &mut [Lfm3AudioStepRow<'_>],
    ) -> Result<Vec<bool>> {
        let mut states = rows
            .iter_mut()
            .map(|row| {
                row.state.row.as_mut().ok_or_else(|| {
                    Error::InvalidInput("LFM3-Audio row decodes before its prefill".into())
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let next = self.choose_slots(&mut states)?;
        drop(states);
        let mut embeds = Vec::new();
        let mut positions = Vec::new();
        let mut continuing = Vec::new();
        for (index, slot) in next.into_iter().enumerate() {
            if let Some((embed, position)) = slot {
                embeds.push(embed);
                positions.push(position);
                continuing.push(index);
            }
        }
        if !continuing.is_empty() {
            let mut shortconvs = Vec::with_capacity(continuing.len());
            let mut caches = Vec::with_capacity(continuing.len());
            for (index, row) in rows.iter_mut().enumerate() {
                if continuing.contains(&index) {
                    shortconvs.push(row.state.retained.shortconv_mut());
                    caches.push(&mut *row.cache);
                }
            }
            let output = self.lm().forward_embeds_retained_batch(
                &embeds,
                &positions,
                &mut shortconvs,
                &mut caches,
            )?;
            drop((shortconvs, caches));
            for (&index, hidden) in continuing.iter().zip(output.last_hidden) {
                let row = rows[index].state.row.as_mut().expect("checked above");
                row.accept_hidden(hidden.squeeze(1)?);
            }
        }
        Ok(rows.iter().map(|row| row.state.finished()).collect())
    }
}
