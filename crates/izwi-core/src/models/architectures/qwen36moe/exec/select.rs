//! Device-side token selection for pipelined decode.
//!
//! A pipelined quantum queues the next decode step before the host has seen
//! the current token, so the token is chosen on the device. Greedy decode
//! takes the first finite maximum (`sampling::greedy_rows`). Sampled decode
//! uses the Qwen3.8 device sampler: repetition and presence penalties from
//! device-resident history counts, temperature, top-k and top-p over the full
//! vocabulary, then the row's host-drawn uniform. The token goes straight
//! into the whole-step graph's input slot, is counted in the device history,
//! and reaches the host through a readback slot (`kernels::cuda::readback`),
//! all queued behind the step that produced the logits with no host
//! synchronization.
use candle_core::{DType, Device, Tensor};

use crate::error::{Error, Result};
use crate::kernels::cuda::readback::ChoiceReadback;
use crate::kernels::cuda::sampling::{self, SamplingParams};
use crate::models::shared::chat::ChatGenerationConfig;

/// The choices of two queued steps can be in flight at once.
const READBACK_SLOTS: usize = 2;

pub(super) struct DeviceSelector {
    device: Device,
    vocab: usize,
    /// `None`: greedy.
    params: Option<SamplingParams>,
    /// Occurrences of each token in the penalty history (U32 `[vocab]`).
    counts: Option<Tensor>,
    one: Tensor,
    readback: ChoiceReadback,
    next: usize,
}

/// A queued choice; [`DeviceSelector::wait`] returns its token.
pub(super) struct PendingToken {
    slot: usize,
}

impl DeviceSelector {
    /// A selector for `config` over the first `vocab` logits. Sampled decode
    /// starts its device counts from `history` (the prompt and the tokens
    /// generated so far).
    pub(super) fn new(
        device: &Device,
        vocab: usize,
        config: &ChatGenerationConfig,
        greedy: bool,
        history: &[u32],
    ) -> Result<Self> {
        let readback = ChoiceReadback::new(device, READBACK_SLOTS)?;
        let (params, counts) = if greedy {
            (None, None)
        } else {
            let mut counts = vec![0u32; vocab];
            for &token in history {
                if let Some(count) = counts.get_mut(token as usize) {
                    *count = count.saturating_add(1);
                }
            }
            (
                Some(SamplingParams {
                    temperature: config.temperature,
                    repetition_penalty: config.repetition_penalty,
                    presence_penalty: config.presence_penalty,
                    frequency_penalty: 0.0,
                    top_k: config.top_k,
                    top_p: config.top_p,
                    min_p: 0.0,
                }),
                Some(Tensor::from_vec(counts, vocab, device)?),
            )
        };
        Ok(Self {
            device: device.clone(),
            vocab,
            params,
            counts,
            one: Tensor::ones(1, DType::U32, device)?,
            readback,
            next: 0,
        })
    }

    /// Queue the choice of the next token from `logits` (one row, at least
    /// `vocab` wide): write it into `token_slot` (U32 `[1]`, the next step's
    /// input), count it in the history, and copy `(token, status)` toward the
    /// host. `uniform` is the row's draw; greedy decode ignores it.
    pub(super) fn select(
        &mut self,
        logits: &Tensor,
        token_slot: &Tensor,
        uniform: f32,
    ) -> Result<PendingToken> {
        let row = logits
            .flatten_all()?
            .narrow(0, 0, self.vocab)?
            .reshape((1, self.vocab))?;
        let chosen = match (&self.params, &self.counts) {
            (Some(params), Some(counts)) => {
                let probabilities =
                    sampling::distributions(&row, &counts.reshape((1, self.vocab))?, params)?;
                let uniforms = Tensor::from_slice(&[uniform], 1, &self.device)?;
                sampling::sample_rows(&probabilities, &uniforms)?
            }
            _ => sampling::greedy_rows(&row)?,
        };
        let token = chosen.narrow(1, 0, 1)?.flatten_all()?.contiguous()?;
        token_slot.slice_set(&token, 0, 0)?;
        if let Some(counts) = &mut self.counts {
            *counts = counts.index_add(&token, &self.one, 0)?;
        }
        let slot = self.next;
        self.next ^= 1;
        self.readback.queue(slot, &chosen.contiguous()?)?;
        Ok(PendingToken { slot })
    }

    /// Wait for a queued choice (only the work queued before it) and return
    /// its token.
    pub(super) fn wait(&self, pending: PendingToken) -> Result<u32> {
        let (token, status) = self.readback.read(pending.slot)?;
        if status != 1 {
            return Err(Error::InferenceError(
                "No finite Qwen3.5 logits to sample".into(),
            ));
        }
        Ok(token)
    }
}
