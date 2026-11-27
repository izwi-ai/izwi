//! The dtype the large dense weights are held in, and the one product every
//! such weight goes through.
//!
//! The checkpoint stores its dense tensors in BF16. Activations are F32 on
//! every backend, and so are the small tensors the reference keeps in F32
//! (norm gains, scales, the position-bias bank, convolution taps,
//! depth-residual and message queries, message gates, routers, biases, the
//! vision tower's class and position embeddings). The large matrices
//! ([`is_projection`]) are F32 too unless 16 bits are asked for on an
//! accelerator:
//!
//! | Backend | Projection weights | Product with F32 activations |
//! |---|---|---|
//! | CPU | F32 | the plain F32 product |
//! | Metal | F32; F16 on request | F32 accumulation and result: the dense mat-vec kernel up to 16 rows, the weight widened for the call on wider spans |
//! | CUDA | F32; BF16 (the stored bytes) on request | BF16: activations rounded to BF16, a BF16 product, the result widened |
//!
//! F16 on Metal halves the dense weights (7.2 GiB to 3.6 GiB as published)
//! with numbers equal to the F32 path's up to summation order, and a decode
//! step's products get faster. It is a request and not the default because
//! a prefill span wider than 16 rows widens each weight for the call, which
//! makes that product about a quarter slower than with F32 weights
//! (`bench_projection_products`); end-to-end prefill time has not been
//! measured on the real model.
//!
//! Metal takes F16 because BF16 is the slower of the two on Apple GPUs
//! before the M3 and the mixed-dtype kernel reads F16. The checkpoint's BF16
//! values are F16 values unless they overflow or fall under F16's subnormal
//! step, which [`f16_values`] checks per tensor at load; a tensor that
//! fails stays F32. With exact weights and F32 accumulation the Metal
//! numbers are the F32 path's up to summation order.
//!
//! CUDA has no product of F32 activations with a BF16 weight yet, so BF16
//! weights there mean rounding the activations of every projection to
//! BF16. That is the reference's arithmetic, but it is not close to the F32
//! path: on the random test fixture, whose experts route on near-ties, it
//! moves logits by whole units (`tests::projection_dtypes_agree_on_metal`
//! runs the same code on Metal and measures it). Until a mixed-dtype kernel
//! serves CUDA, BF16 weights are a request ([`DENSE_DTYPE_ENV`]`=bf16`) and
//! the default stays F32.
//!
//! A [`Projection`] is the one place a product with such a weight is made,
//! so a fused or batched kernel has one place to go.
//!
//! [`DENSE_DTYPE_ENV`]: `auto` or `f32` (everything in F32 on every
//! backend), `f16` (F16 projection weights on Metal) or `bf16` (BF16
//! projection weights on CUDA).

use std::sync::atomic::{AtomicBool, Ordering};

use candle_core::{DType, Device, Tensor};
use half::f16;

use super::checkpoint::{Inkling2Checkpoint, TEXT_PREFIX, VISION_PREFIX};
use crate::backends::BackendKind;
use crate::error::Result;

/// `auto`, `f32`, `f16` or `bf16`: see the module docs.
pub const DENSE_DTYPE_ENV: &str = "IZWI_INKLING2_DENSE_DTYPE";

/// Largest error an F16 copy of a tensor may have, relative to the tensor's
/// RMS. A BF16 value below 2^-17 can lose its low bits to F16's subnormal
/// step (2^-24), an absolute error of at most 2^-25: nothing beside weights
/// of ordinary size, everything in a tensor whose weights are all that
/// small.
const F16_TOLERANCE: f64 = 1e-3;

/// The backend a tensor on `device` belongs to.
pub fn backend_of(device: &Device) -> BackendKind {
    match device {
        Device::Cpu => BackendKind::Cpu,
        Device::Metal(_) => BackendKind::Metal,
        Device::Cuda(_) => BackendKind::Cuda,
    }
}

/// The dtype a load on `backend` holds its projection weights in, with the
/// process's [`DENSE_DTYPE_ENV`].
pub fn projection_dtype(backend: BackendKind) -> DType {
    projection_dtype_for(backend, crate::env::value(DENSE_DTYPE_ENV).as_deref())
}

/// [`projection_dtype`] for `raw`, the value of [`DENSE_DTYPE_ENV`]. A
/// value the backend does not run is its `auto`.
pub fn projection_dtype_for(backend: BackendKind, raw: Option<&str>) -> DType {
    let asked = raw.map(|raw| raw.trim().to_ascii_lowercase());
    match (backend, asked.as_deref()) {
        (BackendKind::Cuda, Some("bf16" | "bfloat16")) => DType::BF16,
        (BackendKind::Metal, Some("f16" | "fp16" | "float16")) => DType::F16,
        _ => DType::F32,
    }
}

/// Whether the dense tensor `name` is one of the large matrices held in the
/// projection dtype: the embedding and unembedding, the attention
/// projections, the dynamic convolutions' weight projections, and the
/// vision tower's linear weights. The load and the footprint both ask here,
/// so what is priced is what is held.
pub fn is_projection(name: &str) -> bool {
    const TEXT: [&str; 8] = [
        ".attn.wq_du.weight",
        ".attn.wk_dv.weight",
        ".attn.wv_dv.weight",
        ".attn.wo_ud.weight",
        ".attn.gate_attn.weight",
        ".attn.wr_du.weight",
        ".attn.attn_sink_proj.weight",
        "_sconv.weight_proj",
    ];
    const VISION: [&str; 9] = [
        ".embeddings.patch_embedding.weight",
        ".self_attn.q_proj.weight",
        ".self_attn.k_proj.weight",
        ".self_attn.v_proj.weight",
        ".self_attn.out_proj.weight",
        ".mlp.c_fc.weight",
        ".mlp.c_proj.weight",
        ".proj_in.weight",
        ".proj_out.weight",
    ];
    if name.starts_with(VISION_PREFIX) {
        return VISION.iter().any(|suffix| name.ends_with(suffix));
    }
    match name.strip_prefix(TEXT_PREFIX) {
        Some(".embed.weight" | ".unembed.weight") => true,
        Some(_) => TEXT.iter().any(|suffix| name.ends_with(suffix)),
        None => false,
    }
}

/// Input widths the F16 mat-vec kernel reads: whole 16-weight chunks.
const F16_KERNEL_CHUNK: usize = 16;

/// The dtype the dense tensor `name` of `shape` is held in when projections
/// are held in `projection`. An F16 projection whose input width the
/// mat-vec kernel cannot read stays F32, so that no weight is ever widened
/// for a single row; the embedding is only gathered from and has no such
/// limit.
pub fn held_dtype(name: &str, shape: &[usize], projection: DType) -> DType {
    if !is_projection(name) {
        return DType::F32;
    }
    let gathered = name.ends_with(".embed.weight");
    let readable = shape
        .last()
        .is_some_and(|width| width.is_multiple_of(F16_KERNEL_CHUNK));
    if projection == DType::F16 && !gathered && !readable {
        return DType::F32;
    }
    projection
}

/// Bytes per element the dense tensor `name` of `shape` is held in.
pub fn held_element_bytes(name: &str, shape: &[usize], projection: DType) -> u64 {
    held_dtype(name, shape, projection).size_in_bytes() as u64
}

/// `values` in F16, or `None` when F16 cannot hold them: a value overflows,
/// or the rounding error passes [`F16_TOLERANCE`] of the tensor's RMS.
fn f16_values(values: &[f32]) -> Option<Vec<f16>> {
    let mut out = Vec::with_capacity(values.len());
    let (mut worst, mut squares) = (0f32, 0f64);
    for &value in values {
        let half = f16::from_f32(value);
        let error = (half.to_f32() - value).abs();
        // Also false for a NaN, and for the infinity an overflow leaves.
        if !error.is_finite() {
            return None;
        }
        worst = worst.max(error);
        squares += (value as f64) * (value as f64);
        out.push(half);
    }
    let rms = (squares / values.len().max(1) as f64).sqrt();
    (worst as f64 <= F16_TOLERANCE * rms).then_some(out)
}

/// The dense tensor `name` on `device` in `dtype`, its first `rows` rows
/// when given. F32 is the checkpoint's own F32 load and BF16 uploads the
/// stored bytes. F16 converts on the host and is checked there; a tensor
/// F16 cannot hold stays F32.
fn held(
    checkpoint: &Inkling2Checkpoint,
    name: &str,
    rows: Option<usize>,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    debug_assert!(
        dtype == DType::F32 || is_projection(name),
        "{name} is not priced as a projection"
    );
    let dtype = held_dtype(name, checkpoint.shape(name)?, dtype);
    if dtype != DType::F16 {
        return match rows {
            Some(rows) => checkpoint.tensor_rows(name, rows, device, dtype),
            None => checkpoint.tensor(name, device, dtype),
        };
    }
    let shape = checkpoint.shape(name)?.to_vec();
    let (values, shape) = match rows {
        Some(rows) => (
            checkpoint.host_f32_rows(name, rows)?,
            vec![rows, shape[shape.len() - 1]],
        ),
        None => (checkpoint.host_f32(name)?, shape),
    };
    match f16_values(&values) {
        Some(halves) => {
            drop(values);
            Ok(Tensor::from_vec(halves, shape, device)?)
        }
        None => {
            static WARNED: AtomicBool = AtomicBool::new(false);
            if !WARNED.swap(true, Ordering::Relaxed) {
                tracing::warn!(
                    "Inkling 2: `{name}` does not fit F16 and stays F32 (later tensors are not reported)"
                );
            }
            Ok(Tensor::from_vec(values, shape, device)?)
        }
    }
}

/// The embedding table, `[vocab, hidden]`, in `dtype`.
pub(super) fn table(
    checkpoint: &Inkling2Checkpoint,
    name: &str,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    held(checkpoint, name, None, device, dtype)
}

/// A bias-free linear map with F32 activations on both sides and its weight
/// in F32 or in 16 bits.
#[derive(Debug, Clone)]
pub(super) struct Projection {
    weight: Weight,
}

#[derive(Debug, Clone)]
enum Weight {
    /// F32, transposed (`[in, out]`): the plain product.
    Same(Tensor),
    /// 16 bits, transposed: the activations are rounded to the weight's
    /// dtype, multiplied there and the result widened. This is the
    /// reference's arithmetic for a projection.
    Cast(Tensor),
    /// Metal: F16 as stored (`[out, in]`) and the zero bias the mat-vec
    /// kernel adds. Accumulation and result are F32.
    #[cfg(feature = "metal")]
    MetalF16 { weight: Tensor, zero_bias: Tensor },
}

impl Projection {
    /// From the stored `[out, in]` tensor `name`, held in `dtype`.
    pub(super) fn load(
        checkpoint: &Inkling2Checkpoint,
        name: &str,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        Self::new(held(checkpoint, name, None, device, dtype)?)
    }

    /// [`Self::load`] of the first `rows` output rows.
    pub(super) fn load_rows(
        checkpoint: &Inkling2Checkpoint,
        name: &str,
        rows: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        Self::new(held(checkpoint, name, Some(rows), device, dtype)?)
    }

    /// From a `[out, in]` weight as it is held.
    fn new(weight: Tensor) -> Result<Self> {
        #[cfg(feature = "metal")]
        if weight.dtype() == DType::F16 && weight.device().is_metal() {
            let zero_bias = Tensor::zeros(weight.dim(0)?, DType::F16, weight.device())?;
            return Ok(Self {
                weight: Weight::MetalF16 { weight, zero_bias },
            });
        }
        let dtype = weight.dtype();
        let transposed = weight.t()?.contiguous()?;
        Ok(Self {
            weight: match dtype {
                DType::F32 => Weight::Same(transposed),
                _ => Weight::Cast(transposed),
            },
        })
    }

    /// The dtype the weight is held in.
    pub(super) fn dtype(&self) -> DType {
        match &self.weight {
            Weight::Same(weight) | Weight::Cast(weight) => weight.dtype(),
            #[cfg(feature = "metal")]
            Weight::MetalF16 { weight, .. } => weight.dtype(),
        }
    }

    /// `x W^T` for F32 `x` of `[rows, in]`, in F32.
    pub(super) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match &self.weight {
            Weight::Same(weight) => Ok(x.matmul(weight)?),
            // Widening the weight instead would allocate its F32 copy on
            // every call.
            Weight::Cast(weight) => Ok(x
                .to_dtype(weight.dtype())?
                .matmul(weight)?
                .to_dtype(DType::F32)?),
            #[cfg(feature = "metal")]
            Weight::MetalF16 { weight, zero_bias } => {
                use crate::kernels::metal_dense_mv;
                // The kernel reads the weight once per group of rows, so it
                // is the fast path for a decode step and slower than a
                // tiled product on a prefill span. A span widens the weight
                // for its one call instead; both accumulate in F32 from the
                // same weight values.
                if x.dim(0)? <= metal_dense_mv::GROUP_ROWS
                    && metal_dense_mv::serves(x, weight, zero_bias)
                {
                    return Ok(metal_dense_mv::linear(x, weight, zero_bias)?);
                }
                Ok(x.matmul(&weight.to_dtype(DType::F32)?.t()?)?)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::checkpoint::{expected_census, Stored};
    use super::super::config::tests::{fixture, tiny};
    use super::*;
    use half::bf16;

    #[test]
    fn each_backend_holds_projections_in_its_own_dtype() {
        assert_eq!(projection_dtype_for(BackendKind::Cpu, None), DType::F32);
        // Sixteen bits are asked for, on either accelerator.
        assert_eq!(projection_dtype_for(BackendKind::Metal, None), DType::F32);
        assert_eq!(
            projection_dtype_for(BackendKind::Metal, Some("auto")),
            DType::F32
        );
        assert_eq!(
            projection_dtype_for(BackendKind::Metal, Some("F16")),
            DType::F16
        );
        assert_eq!(
            projection_dtype_for(BackendKind::Cuda, Some("f16")),
            DType::F32
        );
        // BF16 on CUDA rounds activations, so it is asked for.
        assert_eq!(projection_dtype_for(BackendKind::Cuda, None), DType::F32);
        assert_eq!(
            projection_dtype_for(BackendKind::Cuda, Some("BF16")),
            DType::BF16
        );
        assert_eq!(
            projection_dtype_for(BackendKind::Metal, Some("bf16")),
            DType::F32
        );
        assert_eq!(
            projection_dtype_for(BackendKind::Cpu, Some("bf16")),
            DType::F32
        );
        for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            assert_eq!(projection_dtype_for(backend, Some(" F32 ")), DType::F32);
        }
    }

    /// The projections are the stored-BF16 matrices and nothing the
    /// reference keeps in F32.
    #[test]
    fn the_projections_are_the_large_matrices() {
        let config = tiny();
        let checkpoint = Inkling2Checkpoint::open(&fixture("inkling2_golden"), &config).unwrap();
        let census = expected_census(&config, checkpoint.expert_formats());
        let projections = census
            .iter()
            .filter(|entry| is_projection(&entry.name))
            .collect::<Vec<_>>();
        let full = config
            .layer_kinds
            .iter()
            .filter(|kind| **kind == super::super::config::Inkling2LayerKind::Full)
            .count();
        let vision = config.vision.as_ref().unwrap();
        assert_eq!(
            projections.len(),
            2 + 7 * config.layers + 2 * full + 3 + 6 * vision.layers
        );
        for entry in &projections {
            assert_eq!(entry.stored, Stored::Dense, "{}", entry.name);
            assert_eq!(entry.shape.len(), 2, "{}", entry.name);
            assert_eq!(
                held_element_bytes(&entry.name, &entry.shape, DType::BF16),
                2
            );
            assert_eq!(held_element_bytes(&entry.name, &entry.shape, DType::F32), 4);
        }
        for name in [
            format!("{TEXT_PREFIX}.embed_norm.weight"),
            format!("{TEXT_PREFIX}.layers.2.attn.rel_logits_proj.proj"),
            format!("{TEXT_PREFIX}.layers.2.attn.k_sconv.static_w"),
            format!("{TEXT_PREFIX}.layers.2.controllers.k.down_weight"),
            format!("{TEXT_PREFIX}.layers.2.mlp.gate.weight"),
            format!("{VISION_PREFIX}.backbone.layers.0.ln_1.weight"),
            format!("{VISION_PREFIX}.backbone.layers.0.mlp.c_fc.bias"),
            format!("{VISION_PREFIX}.backbone.embeddings.position_embedding"),
        ] {
            assert!(!is_projection(&name), "{name}");
            assert_eq!(held_element_bytes(&name, &[64, 64], DType::F16), 4);
        }
        // An F16 projection the mat-vec kernel cannot read stays F32; the
        // embedding is only gathered from.
        let query = format!("{TEXT_PREFIX}.layers.0.attn.wq_du.weight");
        assert_eq!(held_dtype(&query, &[32, 64], DType::F16), DType::F16);
        assert_eq!(held_dtype(&query, &[32, 12], DType::F16), DType::F32);
        assert_eq!(held_dtype(&query, &[32, 12], DType::BF16), DType::BF16);
        let embedding = format!("{TEXT_PREFIX}.embed.weight");
        assert_eq!(held_dtype(&embedding, &[160, 12], DType::F16), DType::F16);
    }

    #[test]
    fn f16_holds_bf16_values_unless_they_leave_its_range() {
        // Every BF16 value of ordinary size is an F16 value.
        let ordinary = (0..4096u32)
            .map(|i| bf16::from_f32((i as f32 - 2048.0) * 3.7e-3).to_f32())
            .collect::<Vec<_>>();
        let halves = f16_values(&ordinary).unwrap();
        assert!(halves
            .iter()
            .zip(&ordinary)
            .all(|(half, value)| half.to_f32() == *value));
        // A few values under F16's subnormal step are rounded and allowed.
        let mut tailed = ordinary.clone();
        tailed[7] = bf16::from_f32(3.1e-7).to_f32();
        let halves = f16_values(&tailed).unwrap();
        assert!((halves[7].to_f32() - tailed[7]).abs() <= 2f32.powi(-25));
        // Overflow, a tensor of nothing but tiny values, and a NaN are not.
        let mut large = ordinary.clone();
        large[7] = 7.0e4;
        assert!(f16_values(&large).is_none());
        assert!(f16_values(&[3.1e-7, -2.3e-7, 1.1e-7]).is_none());
        let mut broken = ordinary;
        broken[7] = f32::NAN;
        assert!(f16_values(&broken).is_none());
    }

    /// On the host the projection is the plain F32 product with the weight
    /// held transposed and contiguous, as the trunk held it before
    /// projections had a dtype: the same bits on every CPU. (A product with
    /// a transposed view sums in another order on x86.)
    #[test]
    fn an_f32_projection_is_the_plain_product() {
        let config = tiny();
        let checkpoint = Inkling2Checkpoint::open(&fixture("inkling2_golden"), &config).unwrap();
        let name = format!("{TEXT_PREFIX}.layers.0.attn.wq_du.weight");
        let device = Device::Cpu;
        let projection = Projection::load(&checkpoint, &name, &device, DType::F32).unwrap();
        assert_eq!(projection.dtype(), DType::F32);
        let x = Tensor::from_vec(
            (0..3 * config.hidden)
                .map(|i| (i as f32 * 0.37).sin())
                .collect::<Vec<_>>(),
            (3, config.hidden),
            &device,
        )
        .unwrap();
        let want = x
            .matmul(
                &checkpoint
                    .tensor(&name, &device, DType::F32)
                    .unwrap()
                    .t()
                    .unwrap()
                    .contiguous()
                    .unwrap(),
            )
            .unwrap();
        let bits = |t: &Tensor| {
            t.flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&projection.forward(&x).unwrap()), bits(&want));
    }

    /// Metal, published widths, microseconds per product: the F32 product,
    /// the 16-bit products through a cast, the two kernels that take F32
    /// activations with an F16 weight, and the F16 weight widened for the
    /// call. Run by hand:
    /// `cargo test -p izwi-core --features metal --lib -- --ignored
    /// --nocapture bench_projection_products`.
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "benchmark"]
    fn bench_projection_products() {
        use crate::kernels::f16_weight_linear::F16WeightLinear;
        use crate::kernels::metal_dense_mv;
        use crate::kernels::numeric_check::{relative_l2, xorshift_tensor};
        use candle_core::Module;

        let Some((device, _lease)) =
            crate::backends::metal_test_device_or_skip("bench_projection_products")
        else {
            return;
        };
        let host = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (k, n, row_counts) in [
            (2304usize, 3072usize, vec![1usize, 8, 256]),
            (2304, 65_536, vec![1, 8]),
        ] {
            let weight = xorshift_tensor(&[n, k], 7, 0.02, &device).unwrap();
            let transposed = |dtype: DType| {
                weight
                    .to_dtype(dtype)
                    .unwrap()
                    .t()
                    .unwrap()
                    .contiguous()
                    .unwrap()
            };
            let (w32, w16, wb16) = (
                transposed(DType::F32),
                transposed(DType::F16),
                transposed(DType::BF16),
            );
            let ggml = F16WeightLinear::new(&weight, None, &device).unwrap();
            let rows16 = weight.to_dtype(DType::F16).unwrap();
            let zero_bias = Tensor::zeros(n, DType::F16, &device).unwrap();
            for rows in row_counts {
                let x = xorshift_tensor(&[rows, k], 11, 1.0, &device).unwrap();
                let want = host(&x.matmul(&w32).unwrap());
                let cast = |w: &Tensor| {
                    x.to_dtype(w.dtype())
                        .unwrap()
                        .matmul(w)
                        .unwrap()
                        .to_dtype(DType::F32)
                        .unwrap()
                };
                let paths: Vec<(&str, Box<dyn Fn() -> Tensor>)> = vec![
                    ("f32", Box::new(|| x.matmul(&w32).unwrap())),
                    ("f16 cast", Box::new(|| cast(&w16))),
                    ("bf16 cast", Box::new(|| cast(&wb16))),
                    ("ggml f16xf32", Box::new(|| ggml.forward(&x).unwrap())),
                    (
                        "widened f16",
                        Box::new(|| {
                            x.matmul(&rows16.to_dtype(DType::F32).unwrap().t().unwrap())
                                .unwrap()
                        }),
                    ),
                    (
                        "dense_mv f16xf32",
                        Box::new(|| {
                            assert!(metal_dense_mv::serves(&x, &rows16, &zero_bias));
                            metal_dense_mv::linear(&x, &rows16, &zero_bias).unwrap()
                        }),
                    ),
                ];
                for (name, path) in &paths {
                    let error = relative_l2(&host(&path()), &want);
                    let iterations = 40;
                    let mut best = f64::INFINITY;
                    for _ in 0..3 {
                        let start = std::time::Instant::now();
                        let mut last = path();
                        for _ in 1..iterations {
                            last = path();
                        }
                        // Reading one value back waits for the device.
                        let _ = host(&last.narrow(1, 0, 1).unwrap());
                        best = best.min(start.elapsed().as_secs_f64() / iterations as f64);
                    }
                    eprintln!(
                        "{k} -> {n}, {rows} rows, {name}: {:.0} us, relative L2 {error:.2e}",
                        best * 1e6
                    );
                }
            }
        }
    }
}
