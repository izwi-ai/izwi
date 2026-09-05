//! Request-owned, delayed CUDA-event observations. Timer unavailability keeps
//! the configured depth; it never trains adaptation on submission-only time.
use crate::kernels::cuda::timing::{CudaTimer, PendingCudaTimer};
use std::{sync::Arc, time::Duration};

pub(super) struct RoundTimer(CudaTimer);
#[derive(Clone)]
pub(super) struct PendingRound {
    timer: Arc<PendingCudaTimer>,
    pub depth: usize,
    pub committed: usize,
    pub budget: usize,
}
impl RoundTimer {
    pub fn start(device: &candle_core::Device) -> Option<Self> {
        CudaTimer::start(device).ok().flatten().map(Self)
    }
    pub fn finish(self, depth: usize, committed: usize, budget: usize) -> Option<PendingRound> {
        Some(PendingRound {
            timer: Arc::new(self.0.finish().ok()?),
            depth,
            committed,
            budget,
        })
    }
}
impl PendingRound {
    pub fn try_elapsed(&self) -> Option<Duration> {
        self.timer.try_elapsed().ok().flatten()
    }
}
