//! The one Qwen3.8 weight-residency policy.
//!
//! Per backend it decides the resident form of every checkpoint tensor (raw
//! block FP8, Q8_0, or a dense dtype), whether row-compatible projections are
//! packed into one matrix, the LM head's form, and whether the MTP head loads.
//! The loader ([`super::text`], [`super::mtp`], [`super::chat`]) and load
//! admission (`runtime/lifecycle/qwen38_memory.rs` and the static estimate in
//! `runtime/lifecycle/load.rs`) both ask this policy, so the bytes admission
//! charges are the bytes the loader allocates.
//!
//! CUDA resolves native block FP8 before admission, because the admission
//! lease is fixed before any weight is read. Auto turns it on only after three
//! gates pass on the device: every FP8 geometry the checkpoint needs is inside
//! the kernel contract; a synthetic kernel gate at the checkpoint's real inner
//! dimensions (`fp8::verify_decode_kernels_at`); and a real-weight probe that
//! runs the GEMV on the first rows of two layer-0 projections read straight
//! from the checkpoint and compares it with a host reference from the same
//! bytes. The gate results are cached per device and checkpoint, so admission
//! and the loader see one answer. A load-time self-check failure poisons that
//! answer: the load fails closed and the next attempt resolves (and admits)
//! Q8_0 with exact accounting, so both representations are never resident.
//!
//! Qwen3.8 owns this module; Qwen3.6 keeps its own residency rules.

use std::path::PathBuf;
use std::sync::Mutex;

use candle_core::{DType, Device, Tensor};
use safetensors::Dtype as SafeDType;

use super::fast_path::{compare_values, legacy_requested, summarize, Qwen38FastPath};
use super::native::{
    resident_tensor_specs, IndexedSafetensors, ProjectionMaterialization, Qwen38LayerType,
    Qwen38NativeConfig,
};
use super::text::Qwen38ProjectionRepresentation;
use crate::backends::BackendKind;
use crate::error::{Error, Result};
use crate::kernels::cuda::fp8;
use crate::performance::{CudaProjectionBackend, PerformanceConfig};

/// Kill switch for native block-FP8 projections (`legacy`/`off`/`0`/`false`):
/// Qwen3.8 keeps Q8_0 on CUDA. Qwen3.6 is unaffected.
pub(crate) const FP8_PROJECTIONS_ENV: &str = "IZWI_QWEN38_FP8_PROJECTIONS";
/// Kill switch for the vectorized FP8 decode GEMV: the reference GEMV runs.
pub(crate) const FP8_GEMV_ENV: &str = "IZWI_QWEN38_FP8_GEMV";

/// The checkpoint's block-scale geometry the FP8 kernels index.
pub(crate) const FP8_BLOCK: [usize; 2] = [128, 128];

/// Output channels of the synthetic kernel gate.
const KERNEL_GATE_ROWS: usize = 1024;
/// Rows of the real-weight probe (two scale rows).
const PROBE_ROWS: usize = 256;

/// Resident form of one checkpoint tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Qwen38WeightForm {
    /// The checkpoint's E4M3FN bytes plus one F32 scale per 128x128 block.
    RawFp8,
    /// Scale-applied FP8 values requantized to Q8_0.
    Q8_0,
    /// Expanded to a dense dtype.
    Dense(DType),
}

/// What the FP8 projections become on this backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Qwen38ProjectionPlan {
    /// Raw block FP8 resident and executed by the compact FP8 kernels.
    /// `explicit` when `projection_backend = "native_fp8"` asked for it.
    NativeFp8 { explicit: bool },
    /// Requantized Q8_0 (Candle QMatMul).
    Q8,
    /// Expanded to the activation dtype.
    Expanded,
}

/// The FP8 device capability the policy resolved: whether the compact FP8
/// provider can run on this device at all.
#[derive(Debug, Clone)]
enum Fp8Device {
    /// Checked against the real device with `fp8::provider_supported`.
    Probe(Device),
    /// No device can run it (or none was given).
    Unavailable,
    /// The portable reference kernel stands in (tests on the CPU).
    #[cfg(test)]
    Forced,
}

#[cfg(test)]
thread_local! {
    static FORCE_CPU_NATIVE_FP8: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Tests: resolve native block FP8 on the CPU in this thread (the portable
/// reference kernel runs every device gate, the load-time check and the
/// projections) until the guard drops. Admission and the loader both see it.
#[cfg(test)]
pub(crate) fn force_cpu_native_fp8_for_tests() -> ForcedCpuNativeFp8 {
    FORCE_CPU_NATIVE_FP8.with(|forced| forced.set(true));
    ForcedCpuNativeFp8
}

#[cfg(test)]
pub(crate) struct ForcedCpuNativeFp8;

#[cfg(test)]
impl Drop for ForcedCpuNativeFp8 {
    fn drop(&mut self) {
        FORCE_CPU_NATIVE_FP8.with(|forced| forced.set(false));
    }
}

fn cpu_native_forced() -> bool {
    #[cfg(test)]
    {
        FORCE_CPU_NATIVE_FP8.with(std::cell::Cell::get)
    }
    #[cfg(not(test))]
    {
        false
    }
}

fn forced_fp8_device() -> Fp8Device {
    #[cfg(test)]
    {
        Fp8Device::Forced
    }
    #[cfg(not(test))]
    {
        Fp8Device::Unavailable
    }
}

/// Identity of a residency decision: the device and the checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Identity {
    device: String,
    checkpoint: PathBuf,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Poison {
    projections: Option<String>,
}

/// Cached device gate results: `Err` when the reference kernels or the
/// real-weight probe failed; otherwise the vectorized GEMV's own verdict.
type ProbeOutcome = std::result::Result<std::result::Result<(), String>, String>;

static PROBES: Mutex<Vec<((Identity, DType), ProbeOutcome)>> = Mutex::new(Vec::new());
static POISONED: Mutex<Vec<(Identity, Poison)>> = Mutex::new(Vec::new());

fn lock<T>(mutex: &'static Mutex<T>) -> std::sync::MutexGuard<'static, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn poisoned(identity: Option<&Identity>) -> Poison {
    identity
        .and_then(|identity| {
            lock(&POISONED)
                .iter()
                .find(|(key, _)| key == identity)
                .map(|(_, poison)| poison.clone())
        })
        .unwrap_or_default()
}

/// The resolved residency policy. Cheap to clone; carries no tensors.
#[derive(Debug, Clone)]
pub(crate) struct Qwen38ResidencyPolicy {
    backend: BackendKind,
    activation: DType,
    plan: Qwen38ProjectionPlan,
    /// Why the FP8 projections are not native (`None` when they are).
    fallback: Option<String>,
    fp8_device: Fp8Device,
    /// The vectorized decode GEMV (CUDA): verified and not switched off.
    fp8_gemv: Qwen38FastPath,
    /// Pack row-compatible projections into one matrix.
    packing: bool,
    mtp: bool,
    identity: Option<Identity>,
}

pub(crate) fn backend_of(device: &Device) -> BackendKind {
    if device.is_cuda() {
        BackendKind::Cuda
    } else if device.is_metal() {
        BackendKind::Metal
    } else {
        BackendKind::Cpu
    }
}

/// Whether the MTP head loads. On CUDA it follows the master switch and
/// `cuda.mtp`; off CUDA, `cuda.mtp` alone (the CUDA master switch does not
/// govern other backends). This is the rule the loader has always applied;
/// admission now applies it too instead of assuming CUDA's.
pub(crate) fn mtp_loaded_for(backend: BackendKind, performance: &PerformanceConfig) -> bool {
    match backend {
        BackendKind::Cuda => performance.cuda.enabled() && performance.cuda.mtp.enabled(),
        BackendKind::Metal | BackendKind::Cpu => performance.cuda.mtp.enabled(),
    }
}

/// The activation dtype a load on `device` materializes (the loader derives
/// the same from the device profile): F32 on the CPU, F16 on Metal, BF16 on
/// CUDA compute capability 8.0+ and F16 below.
pub(crate) fn admission_activation(device: &Device) -> ProjectionMaterialization {
    match backend_of(device) {
        BackendKind::Cpu => ProjectionMaterialization::F32,
        BackendKind::Metal => ProjectionMaterialization::F16,
        BackendKind::Cuda if fp8::device_is_sm80_or_newer(device) => {
            ProjectionMaterialization::BF16
        }
        BackendKind::Cuda => ProjectionMaterialization::F16,
    }
}

impl Qwen38ResidencyPolicy {
    /// Resolve the policy for a load (or its admission) on `device`. On CUDA
    /// this runs the cached device gates against `tensors`.
    pub(crate) fn resolve(
        device: &Device,
        activation: ProjectionMaterialization,
        performance: &PerformanceConfig,
        tensors: &IndexedSafetensors,
        config: &Qwen38NativeConfig,
    ) -> Self {
        let backend = backend_of(device);
        let identity = Identity {
            device: format!("{:?}", device.location()),
            checkpoint: tensors.model_dir().to_path_buf(),
        };
        let mut policy = Self::base(backend, activation.dtype(), performance);
        policy.identity = Some(identity);
        if backend == BackendKind::Cuda || (backend == BackendKind::Cpu && cpu_native_forced()) {
            policy.resolve_native(device, performance, tensors, config);
        }
        policy
    }

    /// The policy without a device or checkpoint, for the static estimate of
    /// the pinned 27B checkpoint. CUDA cannot run its device gates here and
    /// stays on Q8_0 (the CUDA admission path resolves with the device).
    pub(crate) fn resolve_static(backend: BackendKind, performance: &PerformanceConfig) -> Self {
        let activation = match backend {
            BackendKind::Cpu => DType::F32,
            BackendKind::Metal => DType::F16,
            BackendKind::Cuda => DType::BF16,
        };
        let mut policy = Self::base(backend, activation, performance);
        if backend == BackendKind::Cuda {
            policy.plan = Qwen38ProjectionPlan::Q8;
            policy.fallback = Some(
                "static estimate: native block FP8 resolves only against the CUDA device".into(),
            );
        }
        policy
    }

    fn base(backend: BackendKind, activation: DType, performance: &PerformanceConfig) -> Self {
        let cuda = &performance.cuda;
        let (plan, fallback) = match backend {
            BackendKind::Cuda => (Qwen38ProjectionPlan::Q8, None),
            BackendKind::Metal | BackendKind::Cpu => (
                Qwen38ProjectionPlan::Expanded,
                Some(
                    "native block-FP8 GEMM is not runtime-certified; using the scale-exact expanded path"
                        .to_string(),
                ),
            ),
        };
        Self {
            backend,
            activation,
            plan,
            fallback,
            fp8_device: Fp8Device::Unavailable,
            fp8_gemv: Qwen38FastPath::legacy("native block FP8 not selected"),
            packing: backend == BackendKind::Cuda
                && cuda.enabled()
                && cuda.packed_projections.enabled(),
            mtp: mtp_loaded_for(backend, performance),
            identity: None,
        }
    }

    /// Native block-FP8 resolution: CUDA in production; in tests also the
    /// CPU, where the portable reference kernel runs every gate.
    fn resolve_native(
        &mut self,
        device: &Device,
        performance: &PerformanceConfig,
        tensors: &IndexedSafetensors,
        config: &Qwen38NativeConfig,
    ) {
        let cuda = &performance.cuda;
        if device.is_cpu() {
            self.packing = cuda.enabled() && cuda.packed_projections.enabled();
        }
        let q8 = |policy: &mut Self, reason: String| {
            policy.plan = Qwen38ProjectionPlan::Q8;
            policy.fallback = Some(reason);
        };
        if !cuda.enabled() {
            return q8(self, "cuda.mode=off".into());
        }
        if cuda.projection_backend == CudaProjectionBackend::Q8 {
            return q8(self, "cuda.projection_backend=q8".into());
        }
        if legacy_requested(FP8_PROJECTIONS_ENV) {
            return q8(self, format!("{FP8_PROJECTIONS_ENV}=legacy"));
        }
        if let Some(reason) = poisoned(self.identity.as_ref()).projections {
            return q8(
                self,
                format!("self_check_failed (poisoned by an earlier load): {reason}"),
            );
        }
        if device.is_cuda()
            && (!matches!(self.activation, DType::BF16 | DType::F16)
                || !fp8::device_is_sm80_or_newer(device))
        {
            return q8(
                self,
                "the compact FP8 provider needs compute capability 8.0+ and 16-bit activations"
                    .into(),
            );
        }
        let explicit = cuda.projection_backend == CudaProjectionBackend::NativeFp8;
        self.fp8_device = if device.is_cuda() {
            Fp8Device::Probe(device.clone())
        } else {
            forced_fp8_device()
        };
        let shapes = match fp8_geometries(config, self.mtp, self.packing) {
            Ok(shapes) => shapes,
            Err(error) => return q8(self, format!("FP8 geometry: {error}")),
        };
        let unsupported: Vec<_> = shapes
            .iter()
            .filter(|shape| !self.fp8_fits(**shape, config.block_fp8.block_shape))
            .collect();
        if !unsupported.is_empty() && !explicit {
            return q8(
                self,
                format!("FP8 kernel contract does not cover {unsupported:?}"),
            );
        }
        let block = config.block_fp8.block_shape;
        let fitting: Vec<[usize; 2]> = shapes
            .iter()
            .copied()
            .filter(|shape| self.fp8_fits(*shape, block))
            .collect();
        let identity = self.identity.clone().expect("resolve sets the identity");
        let fits = |shape: [usize; 2]| self.fp8_fits(shape, block);
        let outcome = if fitting.is_empty() {
            // Nothing will run native (explicit request, no fitting shape).
            Ok(Err("no projection fits the FP8 kernel contract".into()))
        } else {
            cached_probe(&identity, self.activation, || {
                probe_device(device, self.activation, config, tensors, &fitting, &fits)
            })
        };
        let vectorized = match outcome {
            Err(reason) => {
                tracing::warn!(%reason, "Qwen3.8 native FP8 device gate failed; keeping Q8_0");
                super::telemetry::record_fast_path_self_check_failure();
                return q8(self, format!("self_check_failed: {reason}"));
            }
            Ok(vectorized) => vectorized,
        };
        self.plan = Qwen38ProjectionPlan::NativeFp8 { explicit };
        self.fallback = None;
        self.fp8_gemv = if legacy_requested(FP8_GEMV_ENV) {
            Qwen38FastPath::legacy(format!("{FP8_GEMV_ENV}=legacy"))
        } else if !cuda.fused_decode.enabled() {
            Qwen38FastPath::legacy("cuda.fused_decode=off")
        } else if let Err(reason) = vectorized {
            Qwen38FastPath::legacy(format!("self_check_failed: {reason}"))
        } else {
            Qwen38FastPath::Fused
        };
    }

    /// A native-FP8 policy on the CPU, where the portable reference kernel
    /// stands in for the device kernels (tests only).
    #[cfg(test)]
    pub(crate) fn forced_native_fp8_for_tests(activation: ProjectionMaterialization) -> Self {
        let mut policy = Self::base(
            BackendKind::Cpu,
            activation.dtype(),
            &PerformanceConfig::default(),
        );
        policy.plan = Qwen38ProjectionPlan::NativeFp8 { explicit: false };
        policy.fallback = None;
        policy.fp8_device = Fp8Device::Forced;
        policy.fp8_gemv = Qwen38FastPath::Fused;
        policy.packing = true;
        policy
    }

    /// Tests: the same policy with packing switched off.
    #[cfg(test)]
    pub(crate) fn without_packing(mut self) -> Self {
        self.packing = false;
        self
    }

    /// Tests: a policy tied to `checkpoint` on the CPU, so poisoning applies.
    #[cfg(test)]
    pub(crate) fn with_identity_for_tests(mut self, checkpoint: &std::path::Path) -> Self {
        self.identity = Some(Identity {
            device: format!("{:?}", Device::Cpu.location()),
            checkpoint: checkpoint.to_path_buf(),
        });
        self
    }

    pub(crate) fn backend(&self) -> BackendKind {
        self.backend
    }

    pub(crate) fn plan(&self) -> Qwen38ProjectionPlan {
        self.plan
    }

    pub(crate) fn mtp_loaded(&self) -> bool {
        self.mtp
    }

    /// Whether the FP8 kernels run the vectorized decode GEMV.
    pub(crate) fn fp8_vectorized(&self) -> bool {
        self.fp8_gemv.is_fused()
    }

    /// Whether a `[n, k]` block-FP8 projection with `block` scales runs on
    /// the compact FP8 kernels here.
    fn fp8_fits(&self, [n, k]: [usize; 2], block: [usize; 2]) -> bool {
        block == FP8_BLOCK
            && match &self.fp8_device {
                Fp8Device::Probe(device) => fp8::provider_supported(device, self.activation, n, k),
                Fp8Device::Unavailable => false,
                #[cfg(test)]
                Fp8Device::Forced => {
                    n > 0 && k > 0 && n.is_multiple_of(64) && k.is_multiple_of(128)
                }
            }
    }

    /// The resident form of a projection tensor stored as `dtype` with
    /// `shape` and `block` scales.
    pub(crate) fn projection_form(
        &self,
        dtype: SafeDType,
        shape: [usize; 2],
        block: [usize; 2],
    ) -> Qwen38WeightForm {
        if dtype != SafeDType::F8_E4M3 {
            return Qwen38WeightForm::Dense(self.activation);
        }
        match self.plan {
            // Shapes outside the kernel contract stay compact on CUDA.
            Qwen38ProjectionPlan::NativeFp8 { .. } if self.fp8_fits(shape, block) => {
                Qwen38WeightForm::RawFp8
            }
            Qwen38ProjectionPlan::NativeFp8 { .. } if self.backend == BackendKind::Cuda => {
                Qwen38WeightForm::Q8_0
            }
            Qwen38ProjectionPlan::NativeFp8 { .. } => Qwen38WeightForm::Dense(self.activation),
            Qwen38ProjectionPlan::Q8 => Qwen38WeightForm::Q8_0,
            Qwen38ProjectionPlan::Expanded => Qwen38WeightForm::Dense(self.activation),
        }
    }

    /// Whether parts with these forms may be packed into one matrix: they
    /// share one form and the backend packs that form.
    pub(crate) fn packing_allowed(&self, forms: &[Qwen38WeightForm]) -> bool {
        let Some(first) = forms.first() else {
            return false;
        };
        self.packing && forms.iter().all(|form| form == first)
    }

    /// The LM head's resident form.
    pub(crate) fn lm_head_form(&self) -> Qwen38WeightForm {
        Qwen38WeightForm::Dense(self.activation)
    }

    pub(crate) fn representation(&self) -> Qwen38ProjectionRepresentation {
        use Qwen38ProjectionRepresentation as R;
        match (self.plan, self.activation) {
            (Qwen38ProjectionPlan::NativeFp8 { .. }, DType::F16) => R::NativeFp8WithQ8FallbackF16,
            (Qwen38ProjectionPlan::NativeFp8 { .. }, DType::F32) => R::ExpandedF32,
            (Qwen38ProjectionPlan::NativeFp8 { .. }, _) => R::NativeFp8WithQ8FallbackBf16,
            (Qwen38ProjectionPlan::Q8, DType::F16) => R::PackedQ8WithDenseF16,
            (Qwen38ProjectionPlan::Q8, DType::F32) => R::ExpandedF32,
            (Qwen38ProjectionPlan::Q8, _) => R::PackedQ8WithDenseBf16,
            (Qwen38ProjectionPlan::Expanded, DType::F32) => R::ExpandedF32,
            (Qwen38ProjectionPlan::Expanded, DType::F16) => R::ExpandedF16,
            (Qwen38ProjectionPlan::Expanded, _) => R::ExpandedBf16,
        }
    }

    /// The FP8 execution mode for diagnostics.
    pub(crate) fn execution_mode(&self) -> String {
        match (self.plan, self.backend) {
            (Qwen38ProjectionPlan::NativeFp8 { explicit: false }, _) => {
                "native_block_fp8_auto_self_checked".into()
            }
            (Qwen38ProjectionPlan::NativeFp8 { explicit: true }, _) => {
                "native_block_fp8_explicit_self_checked".into()
            }
            (Qwen38ProjectionPlan::Q8, _) => format!(
                "q8_0_fallback:{}",
                self.fallback.as_deref().unwrap_or("unspecified")
            ),
            (Qwen38ProjectionPlan::Expanded, _) => "expanded_fallback".into(),
        }
    }

    /// Why the projections run as they do, for diagnostics.
    pub(crate) fn fallback_reason(&self) -> String {
        match self.plan {
            Qwen38ProjectionPlan::NativeFp8 { .. } => "block-FP8 projections stay resident as the checkpoint's E4M3 bytes and run on the compact FP8 kernels (16-bit activations, software FP8 decode) after device gates and a load-time check against the original checkpoint bytes; not runtime-certified on CUDA hardware".into(),
            Qwen38ProjectionPlan::Q8 => format!(
                "CUDA applies weight_scale_inv and requantizes projections to Q8_0: {}",
                self.fallback.as_deref().unwrap_or("unspecified")
            ),
            Qwen38ProjectionPlan::Expanded => self
                .fallback
                .clone()
                .unwrap_or_else(|| "expanded projections".into()),
        }
    }

    /// `fast_paths` entries: the FP8 projections and the decode GEMV.
    pub(crate) fn diagnostics(&self) -> serde_json::Value {
        let projections = match (&self.plan, &self.fallback) {
            (Qwen38ProjectionPlan::NativeFp8 { .. }, _) => Qwen38FastPath::Fused,
            (_, Some(reason)) => Qwen38FastPath::legacy(reason.clone()),
            (_, None) => Qwen38FastPath::legacy("not selected"),
        };
        serde_json::json!({
            "fp8_projections": summarize([&projections]),
            "fp8_gemv": summarize([&self.fp8_gemv]),
            "fp8_execution_mode": self.execution_mode(),
        })
    }

    /// Fail closed after a load-time self-check: every later resolution for
    /// this device and checkpoint keeps the projections off native FP8, so a
    /// retry admits and loads the fallback with exact accounting.
    pub(crate) fn poison_projections(&self, reason: &str) {
        let Some(identity) = &self.identity else {
            return;
        };
        let mut poisoned = lock(&POISONED);
        if let Some((_, poison)) = poisoned.iter_mut().find(|(key, _)| key == identity) {
            poison.projections = Some(reason.to_string());
        } else {
            poisoned.push((
                identity.clone(),
                Poison {
                    projections: Some(reason.to_string()),
                },
            ));
        }
    }

    /// Resident bytes of one checkpoint tensor under this policy. Scale
    /// companions are charged with their projection.
    pub(crate) fn resident_bytes(
        &self,
        name: &str,
        dtype: SafeDType,
        shape: &[usize],
        block: [usize; 2],
    ) -> Result<u64> {
        if name.ends_with(".weight_scale_inv") {
            return Ok(0);
        }
        let count = elements(shape)?;
        let form = if dtype == SafeDType::F8_E4M3 {
            let [n, k] = <[usize; 2]>::try_from(shape).map_err(|_| {
                Error::ModelLoadError(format!("Qwen3.8 FP8 tensor `{name}` is not a matrix"))
            })?;
            self.projection_form(dtype, [n, k], block)
        } else if name == "lm_head.weight" {
            self.lm_head_form()
        } else if keeps_f32(name) {
            Qwen38WeightForm::Dense(DType::F32)
        } else {
            Qwen38WeightForm::Dense(self.activation)
        };
        if !matches!(
            dtype,
            SafeDType::F8_E4M3 | SafeDType::BF16 | SafeDType::F16 | SafeDType::F32
        ) {
            return Err(Error::ModelLoadError(format!(
                "Unsupported Qwen3.8 dtype {dtype:?} for `{name}`"
            )));
        }
        form_bytes(form, shape, count)
    }
}

/// Whether the loader keeps this dense tensor in F32 whatever the plan: the
/// DeltaNet math constants and convolution kernel, and every zero-centered
/// norm gain (stored as `1 + w`).
fn keeps_f32(name: &str) -> bool {
    name.ends_with("norm.weight")
        || name.starts_with("mtp.pre_fc_norm_")
        || (name.contains(".linear_attn.")
            && [".dt_bias", ".A_log", ".conv1d.weight"]
                .iter()
                .any(|suffix| name.ends_with(suffix)))
}

fn overflow() -> Error {
    Error::ModelLoadError("Qwen3.8 resident inventory overflow".into())
}

fn elements(shape: &[usize]) -> Result<u64> {
    if shape.is_empty() || shape.contains(&0) {
        return Err(Error::ModelLoadError("Empty Qwen3.8 weight shape".into()));
    }
    shape.iter().try_fold(1u64, |n, &d| {
        n.checked_mul(u64::try_from(d).map_err(|_| overflow())?)
            .ok_or_else(overflow)
    })
}

fn form_bytes(form: Qwen38WeightForm, shape: &[usize], count: u64) -> Result<u64> {
    match form {
        Qwen38WeightForm::RawFp8 => {
            let (n, k) = (shape[0] as u64, shape[1] as u64);
            let scales = n
                .div_ceil(FP8_BLOCK[0] as u64)
                .checked_mul(k.div_ceil(FP8_BLOCK[1] as u64))
                .and_then(|blocks| blocks.checked_mul(4))
                .ok_or_else(overflow)?;
            count.checked_add(scales).ok_or_else(overflow)
        }
        Qwen38WeightForm::Q8_0 => {
            if shape.len() != 2 || !shape[1].is_multiple_of(32) {
                return Err(Error::ModelLoadError(format!(
                    "Q8_0 residency needs a matrix with K divisible by 32, found {shape:?}"
                )));
            }
            (count / 32).checked_mul(34).ok_or_else(overflow)
        }
        Qwen38WeightForm::Dense(dtype) => count
            .checked_mul(dtype.size_in_bytes() as u64)
            .ok_or_else(overflow),
    }
}

/// Every FP8 projection geometry a load needs: each projection, plus the
/// packed groups when packing is on.
fn fp8_geometries(
    config: &Qwen38NativeConfig,
    mtp: bool,
    packing: bool,
) -> Result<Vec<[usize; 2]>> {
    let mut shapes: Vec<[usize; 2]> = resident_tensor_specs(config, mtp)?
        .into_iter()
        .filter(|spec| spec.dtype == SafeDType::F8_E4M3)
        .filter_map(|spec| <[usize; 2]>::try_from(spec.shape.as_slice()).ok())
        .collect();
    if packing {
        let text = &config.text;
        let hidden = text.embedding_length;
        let conv_dim = text.ssm_state_size * text.ssm_group_count * 2 + text.ssm_inner_size;
        let q = text.attention_head_count * text.attention_key_length * 2;
        let kv = text.attention_head_count_kv * text.attention_key_length;
        shapes.push([2 * text.feed_forward_length, hidden]);
        shapes.push([q + 2 * kv, hidden]);
        if config
            .layer_types
            .contains(&Qwen38LayerType::LinearAttention)
        {
            shapes.push([conv_dim + text.ssm_inner_size, hidden]);
        }
    }
    shapes.sort_unstable();
    shapes.dedup();
    Ok(shapes)
}

fn cached_probe(
    identity: &Identity,
    activation: DType,
    probe: impl FnOnce() -> ProbeOutcome,
) -> ProbeOutcome {
    let key = (identity.clone(), activation);
    if let Some((_, outcome)) = lock(&PROBES).iter().find(|(cached, _)| *cached == key) {
        return outcome.clone();
    }
    let outcome = probe();
    let mut cache = lock(&PROBES);
    if cache.len() >= 16 {
        drop(cache.remove(0));
    }
    cache.push((key, outcome.clone()));
    outcome
}

/// The synthetic kernel gate at every real inner dimension, then the
/// real-weight probe on the first rows of two layer-0 projections.
fn probe_device(
    device: &Device,
    activation: DType,
    config: &Qwen38NativeConfig,
    tensors: &IndexedSafetensors,
    shapes: &[[usize; 2]],
    fits: &dyn Fn([usize; 2]) -> bool,
) -> ProbeOutcome {
    let mut inner: Vec<usize> = shapes.iter().map(|shape| shape[1]).collect();
    inner.sort_unstable();
    inner.dedup();
    // The CPU reference (tests) has no separate vectorized kernel.
    let mut vectorized = if device.is_cuda() {
        fp8::verify_vectorized_decode(device).map_err(|e| e.to_string())
    } else {
        Ok(())
    };
    for k in inner {
        let gate = fp8::verify_decode_kernels_at(device, activation, KERNEL_GATE_ROWS, k)
            .map_err(|error| format!("kernel gate at K={k}: {error}"))?;
        if vectorized.is_ok() {
            vectorized = gate.vectorized;
        }
    }
    let block = config.block_fp8.block_shape;
    let first_linear = config
        .layer_types
        .iter()
        .position(|kind| *kind == Qwen38LayerType::LinearAttention);
    let text = &config.text;
    let hidden = text.embedding_length;
    let mut probes = vec![(
        "model.language_model.layers.0.mlp.down_proj.weight".to_string(),
        [hidden, text.feed_forward_length],
    )];
    if let Some(layer) = first_linear {
        let conv_dim = text.ssm_state_size * text.ssm_group_count * 2 + text.ssm_inner_size;
        probes.push((
            format!("model.language_model.layers.{layer}.linear_attn.in_proj_qkv.weight"),
            [conv_dim, hidden],
        ));
    }
    for (name, shape) in probes.into_iter().filter(|(_, shape)| fits(*shape)) {
        let rows: Vec<usize> = (0..PROBE_ROWS.min(shape[0])).collect();
        for use_vectorized in [false, vectorized.is_ok()] {
            real_weight_probe(
                device,
                activation,
                tensors,
                &name,
                shape,
                block,
                &rows,
                use_vectorized,
            )
            .map_err(|error| format!("real-weight probe on `{name}`: {error}"))?;
        }
    }
    Ok(vectorized)
}

/// Run the compact FP8 GEMV on `rows` of a checkpoint projection uploaded as
/// they are stored, and compare it with the host reference from the same
/// bytes at one and four activation rows.
#[allow(clippy::too_many_arguments)]
fn real_weight_probe(
    device: &Device,
    activation: DType,
    tensors: &IndexedSafetensors,
    name: &str,
    shape: [usize; 2],
    block: [usize; 2],
    rows: &[usize],
    vectorized: bool,
) -> Result<()> {
    let k = shape[1];
    let (bytes, scales) = tensors.block_fp8_rows(name, shape, block, rows)?;
    let scale_cols = k.div_ceil(block[1]);
    let row_blocks = rows.len().div_ceil(block[0]);
    let probe_scales = scales[..row_blocks * scale_cols].to_vec();
    let weights = Tensor::from_vec(bytes.clone(), (rows.len(), k), device)?;
    let scale_tensor = Tensor::from_vec(probe_scales, (row_blocks, scale_cols), device)?;
    for m in [1usize, 4] {
        let x = probe_activations(m, k, activation, 7 + m as u64);
        let input = Tensor::from_vec(x.clone(), (m, k), &Device::Cpu)?
            .to_dtype(activation)?
            .to_device(device)?;
        let actual =
            fp8::block_fp8_projection_selecting(&input, &weights, &scale_tensor, vectorized)?
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
        let mut expected = Vec::with_capacity(m * rows.len());
        for row in x.chunks(k) {
            for (local, _) in rows.iter().enumerate() {
                expected.push(fp8_row_dot(
                    &bytes[local * k..(local + 1) * k],
                    &scales,
                    local / block[0],
                    scale_cols,
                    block[1],
                    row,
                ));
            }
        }
        compare_values(
            &format!("FP8 GEMV m={m} vectorized={vectorized}"),
            &actual,
            &expected,
            0.01,
            0.02,
        )?;
    }
    Ok(())
}

/// Deterministic activations in [-1, 1) with unit RMS per row, rounded to
/// `dtype` (the device multiplies the rounded values).
pub(crate) fn probe_activations(rows: usize, k: usize, dtype: DType, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut values: Vec<f32> = (0..rows * k)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect();
    for row in values.chunks_mut(k) {
        let rms = (row.iter().map(|v| v * v).sum::<f32>() / k as f32)
            .sqrt()
            .max(f32::MIN_POSITIVE);
        for value in row.iter_mut() {
            *value /= rms;
        }
    }
    match dtype {
        DType::BF16 => values
            .iter()
            .map(|v| half::bf16::from_f32(*v).to_f32())
            .collect(),
        DType::F16 => values
            .iter()
            .map(|v| half::f16::from_f32(*v).to_f32())
            .collect(),
        _ => values,
    }
}

/// `sum_k decode(byte[k]) * x[k] * scale[row_block, k / block_cols]`, block
/// partial sums first, as the kernels accumulate.
pub(crate) fn fp8_row_dot(
    bytes: &[u8],
    scales: &[f32],
    row_block: usize,
    scale_cols: usize,
    block_cols: usize,
    x: &[f32],
) -> f32 {
    bytes
        .chunks(block_cols)
        .zip(x.chunks(block_cols))
        .enumerate()
        .map(|(column_block, (bytes, x))| {
            let partial: f32 = bytes
                .iter()
                .zip(x)
                .map(|(b, x)| fp8::decode_e4m3fn(*b) * x)
                .sum();
            partial * scales[row_block * scale_cols + column_block]
        })
        .sum()
}

/// Prices of the pinned 27B checkpoint (static estimate) or any checkpoint.
pub(crate) fn price_specs<'a>(
    policy: &Qwen38ResidencyPolicy,
    specs: impl IntoIterator<Item = (&'a str, SafeDType, &'a [usize])>,
    block: [usize; 2],
) -> Result<u64> {
    specs
        .into_iter()
        .try_fold(0u64, |total, (name, dtype, shape)| {
            total
                .checked_add(policy.resident_bytes(name, dtype, shape, block)?)
                .ok_or_else(overflow)
        })
}
