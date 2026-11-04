//! The host's view of sampled choices queued behind device work: each slot
//! holds one row's `(token, status)` from [`super::sampling::greedy_rows`] or
//! [`super::sampling::sample_rows`], and a pipelined decode keeps a slot per
//! step it lets run ahead of the host.
//!
//! On CUDA a slot is a pinned host buffer and its own completion event.
//! [`ChoiceReadback::queue`] copies the choice into the buffer asynchronously
//! and records the event right behind the copy; [`ChoiceReadback::read`]
//! waits on that event, so it waits only for the work queued before the copy.
//!
//! The event has to be ours. cudarc records a `PinnedHostSlice`'s internal
//! event after a copy only when the context manages stream synchronization
//! (multi-stream mode with event tracking, since cudarc 0.19.10). Candle runs
//! one per-thread stream, so that event is never recorded and `as_slice()`
//! returns at once: the Qwen3.8 and Qwen3.6 pipelines read their buffers
//! before the copies landed, saw a zeroed status, and failed every chat with
//! "No finite ... logits to sample".
//!
//! Other devices keep the choice tensors and read them on `read`.
use candle_core::{DType, Device, Result, Tensor};

/// Values per choice: the token, then its status (1: finite).
const CHOICE_LEN: usize = 2;

/// Queued `(token, status)` choices, one per slot.
pub struct ChoiceReadback {
    slots: Slots,
}

enum Slots {
    #[cfg(feature = "cuda")]
    Pinned(Vec<PinnedSlot>),
    Tensors(Vec<Option<Tensor>>),
}

#[cfg(feature = "cuda")]
struct PinnedSlot {
    buffer: candle_core::cuda_backend::cudarc::driver::PinnedHostSlice<u32>,
    /// Recorded behind the copy that last filled `buffer`.
    done: candle_core::cuda_backend::cudarc::driver::CudaEvent,
    queued: bool,
}

#[cfg(feature = "cuda")]
impl Drop for PinnedSlot {
    fn drop(&mut self) {
        // The buffer is freed after this runs: let a copy still in flight
        // land first.
        if self.queued {
            let _ = self.done.synchronize();
        }
    }
}

fn no_slot(slot: usize) -> candle_core::Error {
    candle_core::Error::Msg(format!("choice readback has no slot {slot}"))
}

fn nothing_queued(slot: usize) -> candle_core::Error {
    candle_core::Error::Msg(format!("choice readback slot {slot} has nothing queued"))
}

impl ChoiceReadback {
    /// `slots` readbacks for choices made on `device`.
    pub fn new(device: &Device, slots: usize) -> Result<Self> {
        if slots == 0 {
            candle_core::bail!("a choice readback needs at least one slot")
        }
        #[cfg(feature = "cuda")]
        if let Device::Cuda(cuda) = device {
            use candle_core::cuda_backend::{
                cudarc::driver::sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC, WrapErr,
            };
            let context = cuda.cuda_stream().context().clone();
            let pinned = (0..slots)
                .map(|_| -> Result<PinnedSlot> {
                    // SAFETY: `read` returns a slot's values only once its own
                    // event shows the copy that filled them complete, and
                    // refuses a slot that was never filled.
                    let buffer = unsafe { context.alloc_pinned::<u32>(CHOICE_LEN) }.w()?;
                    // Blocking, as cudarc's own pinned-buffer event: the host
                    // sleeps rather than spins while it waits.
                    let done = context.new_event(Some(CU_EVENT_BLOCKING_SYNC)).w()?;
                    Ok(PinnedSlot {
                        buffer,
                        done,
                        queued: false,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            return Ok(Self {
                slots: Slots::Pinned(pinned),
            });
        }
        let _ = device;
        Ok(Self {
            slots: Slots::Tensors(vec![None; slots]),
        })
    }

    /// Start moving `choice` (U32, two values: token and status) toward the
    /// host into `slot`, behind the work already queued on its device.
    pub fn queue(&mut self, slot: usize, choice: &Tensor) -> Result<()> {
        if choice.dtype() != DType::U32 || choice.elem_count() != CHOICE_LEN {
            candle_core::bail!(
                "a choice readback takes U32 (token, status), got {:?} {:?}",
                choice.dims(),
                choice.dtype()
            )
        }
        match &mut self.slots {
            #[cfg(feature = "cuda")]
            Slots::Pinned(slots) => {
                use candle_core::cuda_backend::WrapErr;
                let target = slots.get_mut(slot).ok_or_else(|| no_slot(slot))?;
                let (storage, layout) = choice.storage_and_layout();
                let candle_core::Storage::Cuda(cuda) = &*storage else {
                    candle_core::bail!("a CUDA choice readback takes a CUDA choice")
                };
                let (start, end) = layout.contiguous_offsets().ok_or_else(|| {
                    candle_core::Error::Msg("the queued choice is not contiguous".into())
                })?;
                let view = cuda.as_cuda_slice::<u32>()?.slice(start..end);
                let stream = choice.device().as_cuda_device()?.cuda_stream();
                stream.memcpy_dtoh(&view, &mut target.buffer).w()?;
                target.queued = true;
                if let Err(error) = target.done.record(&stream) {
                    // Without the event nothing fences the copy: wait for it
                    // here so the buffer is never read or freed under it.
                    let _ = stream.synchronize();
                    target.queued = false;
                    return Err(error).w();
                }
                Ok(())
            }
            Slots::Tensors(slots) => {
                *slots.get_mut(slot).ok_or_else(|| no_slot(slot))? = Some(choice.clone());
                Ok(())
            }
        }
    }

    /// The `(token, status)` last queued into `slot`, once its copy is done.
    pub fn read(&self, slot: usize) -> Result<(u32, u32)> {
        let values: [u32; CHOICE_LEN] = match &self.slots {
            #[cfg(feature = "cuda")]
            Slots::Pinned(slots) => {
                use candle_core::cuda_backend::WrapErr;
                let source = slots.get(slot).ok_or_else(|| no_slot(slot))?;
                if !source.queued {
                    return Err(nothing_queued(slot));
                }
                source.done.synchronize().w()?;
                let host = source.buffer.as_slice().w()?;
                [host[0], host[1]]
            }
            Slots::Tensors(slots) => {
                let host = slots
                    .get(slot)
                    .ok_or_else(|| no_slot(slot))?
                    .as_ref()
                    .ok_or_else(|| nothing_queued(slot))?
                    .flatten_all()?
                    .to_vec1::<u32>()?;
                [host[0], host[1]]
            }
        };
        Ok((values[0], values[1]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_queued_choice_reads_back_from_its_own_slot() {
        let device = Device::Cpu;
        let mut readback = ChoiceReadback::new(&device, 2).unwrap();
        let first = Tensor::new(&[[42u32, 1]], &device).unwrap();
        let second = Tensor::new(&[7u32, 0], &device).unwrap();
        readback.queue(0, &first).unwrap();
        readback.queue(1, &second).unwrap();
        assert_eq!(readback.read(0).unwrap(), (42, 1));
        assert_eq!(readback.read(1).unwrap(), (7, 0));
        // A slot holds its latest choice.
        readback.queue(0, &second).unwrap();
        assert_eq!(readback.read(0).unwrap(), (7, 0));
    }

    #[test]
    fn empty_missing_and_malformed_slots_are_refused() {
        let device = Device::Cpu;
        assert!(ChoiceReadback::new(&device, 0).is_err());
        let mut readback = ChoiceReadback::new(&device, 2).unwrap();
        assert!(readback
            .read(1)
            .unwrap_err()
            .to_string()
            .contains("nothing queued"));
        let choice = Tensor::new(&[3u32, 1], &device).unwrap();
        assert!(readback.queue(2, &choice).is_err());
        assert!(readback.read(2).is_err());
        let wide = Tensor::new(&[3u32, 1, 0], &device).unwrap();
        assert!(readback.queue(0, &wide).is_err());
        let float = Tensor::new(&[3f32, 1.], &device).unwrap();
        assert!(readback.queue(0, &float).is_err());
    }

    /// The regression this module exists for: a choice made behind tens of
    /// milliseconds of device work, read back straight after it is queued.
    /// Relying on cudarc's own pinned-buffer event read the buffer before
    /// the copy landed (zeros).
    #[cfg(feature = "cuda")]
    #[test]
    fn a_cuda_read_waits_for_the_copy_queued_behind_device_work() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        let vocab = 4096;
        let mut logits = vec![0f32; vocab];
        let mut readback = ChoiceReadback::new(&device, 2).unwrap();
        let work = Tensor::randn(0f32, 1., (4096, 4096), &device).unwrap();
        for round in 0..6usize {
            let slot = round % 2;
            let token = (17 + 997 * round) % vocab;
            logits.fill(0.);
            logits[token] = 1.;
            // Uploaded (synchronously) before the slow work is queued.
            let row = Tensor::from_slice(&logits, (1, vocab), &device).unwrap();
            for _ in 0..8 {
                drop(work.matmul(&work).unwrap());
            }
            let choice = crate::kernels::cuda::sampling::greedy_rows(&row).unwrap();
            readback.queue(slot, &choice).unwrap();
            assert_eq!(
                readback.read(slot).unwrap(),
                (token as u32, 1),
                "round {round}"
            );
        }
    }
}
