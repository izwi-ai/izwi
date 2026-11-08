//! Row-invariant Qwen3-TTS decode on CUDA: a seeded request's codes are the
//! same alone as beside up to 31 other requests.
//!
//! cuBLAS picks a GEMM's kernel, tiles and split from its row count, and
//! each rounds a row's sums its own way, so a decode step whose projections
//! take the step's row count gives a row different bits at every width. On
//! CUDA every per-step dense op of the talker and the code predictor
//! therefore runs at [`DECODE_ROWS`] rows, spare rows zero; attention takes
//! the arena's row-invariant paged decode (each row partitioned from its own
//! length); and a lone request runs the batch path, not a single-row one.
//! The CPU and Metal keep their own paths.

use candle_core::{Device, Module, Tensor};

/// The rows a CUDA decode step's dense ops run at: the widest Qwen3-TTS
/// decode step on CUDA (`catalog::tts_decode_width`), and the code
/// predictor's graph rows.
pub(crate) const DECODE_ROWS: usize = 32;

#[cfg(test)]
thread_local! {
    static FORCED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether decode steps on `device` run row-invariant: CUDA, or a test that
/// forces the CUDA path on the CPU ([`ForceRowInvariant`]).
pub(crate) fn row_invariant(device: &Device) -> bool {
    #[cfg(test)]
    if FORCED.with(std::cell::Cell::get) {
        return true;
    }
    device.is_cuda()
}

/// `module` over `x` (`[..., in]`) with its rows padded to [`DECODE_ROWS`]
/// when `padded` (and they fit), so the matmul runs one shape whatever the
/// step's width; the spare rows' outputs are dropped. `x` as is otherwise.
pub(crate) fn forward_rows<M: Module>(
    module: &M,
    x: &Tensor,
    padded: bool,
) -> candle_core::Result<Tensor> {
    let dims = x.dims();
    let Some((&features, leading)) = dims.split_last() else {
        return module.forward(x);
    };
    let rows = leading.iter().product::<usize>();
    if !padded || rows == 0 || rows > DECODE_ROWS {
        return module.forward(x);
    }
    let flat = x
        .reshape((rows, features))?
        .pad_with_zeros(0, 0, DECODE_ROWS - rows)?;
    let out = module.forward(&flat)?.narrow(0, 0, rows)?;
    let mut out_dims = leading.to_vec();
    out_dims.push(out.dim(1)?);
    out.reshape(out_dims)
}

/// Forces [`row_invariant`] on for this thread while it lives (tests run
/// the CUDA path on the CPU with it).
#[cfg(test)]
pub(crate) struct ForceRowInvariant {
    previous: bool,
}

#[cfg(test)]
impl ForceRowInvariant {
    pub(crate) fn new(on: bool) -> Self {
        Self {
            previous: FORCED.with(|forced| forced.replace(on)),
        }
    }
}

#[cfg(test)]
impl Drop for ForceRowInvariant {
    fn drop(&mut self) {
        FORCED.with(|forced| forced.set(self.previous));
    }
}
