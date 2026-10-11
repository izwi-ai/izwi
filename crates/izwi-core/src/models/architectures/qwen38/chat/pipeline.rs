//! Device-side token selection for pipelined Qwen3.8 decode.
//!
//! A pipelined quantum queues the next decode step before the host has seen
//! the current token, so the token is chosen on the device, exactly as the
//! step-by-step device sampler (`device_sampling::sample`) chooses it:
//! - greedy decode without penalties takes the first finite maximum
//!   (`sampling::greedy_rows`);
//! - otherwise repetition and presence penalties from device-resident
//!   history counts, temperature, top-k and top-p over the vocabulary
//!   (`sampling::distributions`), then the row's host-drawn uniform
//!   (`sampling::sample_rows`). Only a temperature above 1e-5 draws from the
//!   row's RNG, as the step-by-step sampler does.
//!
//! The token goes straight into the whole-step graph's input slot, is counted
//! in the device history (when the request tracks one), and reaches the host
//! through one of two pinned buffers, all queued behind the step that
//! produced the logits with no host synchronization. Other devices, in tests,
//! read the choice back directly.
use candle_core::{DType, Device, Tensor};

use crate::error::{Error, Result};
use crate::kernels::cuda::sampling::{self, SamplingParams};
use crate::models::shared::chat::ChatGenerationConfig;

pub(super) struct DeviceSelector {
    device: Device,
    vocab: usize,
    /// `None`: greedy without penalties.
    params: Option<SamplingParams>,
    /// Occurrences of each token in the penalty history (U32 `[vocab]`),
    /// counted only when the request tracks a history.
    counts: Option<Tensor>,
    track_history: bool,
    draws: bool,
    one: Tensor,
    readback: Readback,
    next: usize,
}

/// Where queued choices land: the choices of two queued steps can be in
/// flight at once.
enum Readback {
    /// Two pinned `(token, status)` buffers, filled by async copies (CUDA).
    #[cfg(feature = "cuda")]
    Pinned([candle_core::cuda_backend::cudarc::driver::PinnedHostSlice<u32>; 2]),
    /// The choice tensors themselves, read on `wait`.
    Tensors([Option<Tensor>; 2]),
}

/// A queued choice; [`DeviceSelector::wait`] returns its token.
pub(super) struct PendingToken {
    slot: usize,
}

impl DeviceSelector {
    /// A selector for `config` over the first `vocab` logits. A request that
    /// tracks a penalty history starts its device counts from `history` (the
    /// prompt and the tokens generated so far).
    pub(super) fn new(
        device: &Device,
        vocab: usize,
        config: &ChatGenerationConfig,
        track_history: bool,
        history: &[u32],
    ) -> Result<Self> {
        let greedy = config.temperature <= 1e-5
            && config.repetition_penalty <= 1.0
            && config.presence_penalty.abs() <= f32::EPSILON;
        let readback = Readback::new(device)?;
        let (params, counts) = if greedy {
            (None, None)
        } else {
            let mut counts = vec![0u32; vocab];
            if track_history {
                for &token in history {
                    if let Some(count) = counts.get_mut(token as usize) {
                        *count = count.saturating_add(1);
                    }
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
            track_history,
            draws: config.temperature > 1e-5,
            one: Tensor::ones(1, DType::U32, device)?,
            readback,
            next: 0,
        })
    }

    /// Whether a choice consumes a uniform from the row's RNG.
    pub(super) fn draws(&self) -> bool {
        self.draws
    }

    /// Queue the choice of the next token from `logits` (one row, at least
    /// `vocab` wide): write it into `token_slot` (U32 `[1]`, the next step's
    /// input), count it in the history, and copy `(token, status)` toward the
    /// host. `uniform` is the row's draw; a choice that does not draw ignores
    /// it.
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
        if self.track_history {
            if let Some(counts) = &mut self.counts {
                *counts = counts.index_add(&token, &self.one, 0)?;
            }
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
                "No finite Qwen3.8 logits to sample".into(),
            ));
        }
        Ok(token)
    }
}

impl Readback {
    fn new(device: &Device) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if let Device::Cuda(cuda) = device {
            let context = cuda.cuda_stream().context().clone();
            let pinned = || {
                // SAFETY: a device copy writes the buffer before every read.
                unsafe { context.alloc_pinned::<u32>(2) }.map_err(|error| {
                    Error::InferenceError(format!("pinned token readback: {error}"))
                })
            };
            return Ok(Self::Pinned([pinned()?, pinned()?]));
        }
        let _ = device;
        Ok(Self::Tensors([None, None]))
    }

    /// Start moving `chosen` (U32 `[1, 2]`) toward the host.
    fn queue(&mut self, slot: usize, chosen: &Tensor) -> Result<()> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Pinned(buffers) => {
                let (storage, layout) = chosen.storage_and_layout();
                let candle_core::Storage::Cuda(cuda) = &*storage else {
                    return Err(Error::InferenceError(
                        "device token selection produced a host tensor".into(),
                    ));
                };
                let (start, end) = layout.contiguous_offsets().ok_or_else(|| {
                    Error::InferenceError("token choice is not contiguous".into())
                })?;
                let view = cuda.as_cuda_slice::<u32>()?.slice(start..end);
                chosen
                    .device()
                    .as_cuda_device()?
                    .cuda_stream()
                    .memcpy_dtoh(&view, &mut buffers[slot])
                    .map_err(|error| Error::InferenceError(format!("token readback: {error}")))
            }
            Self::Tensors(tensors) => {
                tensors[slot] = Some(chosen.clone());
                Ok(())
            }
        }
    }

    /// `(token, status)` of the choice in `slot`, waiting for its copy.
    fn read(&self, slot: usize) -> Result<(u32, u32)> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Pinned(buffers) => {
                let values = buffers[slot]
                    .as_slice()
                    .map_err(|error| Error::InferenceError(format!("token readback: {error}")))?;
                Ok((values[0], values[1]))
            }
            Self::Tensors(tensors) => {
                let values = tensors[slot]
                    .as_ref()
                    .ok_or_else(|| Error::InferenceError("no queued token choice".into()))?
                    .flatten_all()?
                    .to_vec1::<u32>()?;
                Ok((values[0], values[1]))
            }
        }
    }
}
