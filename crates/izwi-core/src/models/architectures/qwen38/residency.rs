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

use super::chat::Qwen38TextConfig;
use super::fast_path::{
    compare_values, legacy_requested, summarize, Qwen38FastPath, DECODE_GRAPHS_ENV,
    FUSED_AB_GEMV_ENV,
};
use super::native::{
    resident_tensor_specs, IndexedSafetensors, ProjectionMaterialization, Qwen38LayerType,
    Qwen38NativeConfig,
};
use super::text::Qwen38ProjectionRepresentation;
use crate::backends::BackendKind;
use crate::error::{Error, Result};
use crate::kernels::cuda::fp8;
use crate::performance::{
    CpuProjectionBackend, CudaProjectionBackend, MetalProjectionBackend, PerformanceConfig,
};

/// Kill switch for native block-FP8 projections (`legacy`/`off`/`0`/`false`):
/// Qwen3.8 keeps Q8_0 on CUDA. Qwen3.6 is unaffected.
pub(crate) const FP8_PROJECTIONS_ENV: &str = "IZWI_QWEN38_FP8_PROJECTIONS";
/// Kill switch for the vectorized FP8 decode GEMV: the reference GEMV runs.
pub(crate) const FP8_GEMV_ENV: &str = "IZWI_QWEN38_FP8_GEMV";
/// Switch for the dequantized prefill GEMM (`auto`, or `legacy`/`off`/`0`/
/// `false`): long spans of CUDA block-FP8 projections run the compact `mm`
/// kernel instead of a bounded BF16/F16 weight tile plus cuBLAS.
pub(crate) const FP8_PREFILL_GEMM_ENV: &str = "IZWI_QWEN38_FP8_PREFILL_GEMM";
pub(crate) use super::lm_head::FP8_LM_HEAD_ENV;
/// The Q8_0 CPU LM head while the CPU projections are Q8_0 (on when unset):
/// a false value or its registered `dense`/`legacy` word keeps the head F32.
pub(crate) const CPU_LM_HEAD_ENV: &str = "IZWI_QWEN38_CPU_LM_HEAD";

fn cpu_dense_head_requested() -> bool {
    legacy_requested(CPU_LM_HEAD_ENV)
}

/// The checkpoint's block-scale geometry the FP8 kernels index.
pub(crate) const FP8_BLOCK: [usize; 2] = [128, 128];

/// Output channels of the synthetic kernel gate.
const KERNEL_GATE_ROWS: usize = 1024;
/// Rows of the real-weight probe (two scale rows).
const PROBE_ROWS: usize = 256;
/// Metal prefill expands a raw block-FP8 projection to a transient F16
/// matrix: two power-of-two allocator buckets for the largest packed group
/// (the 27B gate_up is 34816 x 5120, 356 MB in F16).
pub(crate) const METAL_FP8_PREFILL_DEQUANT_BYTES: u64 = 1024 * 1024 * 1024;

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
    /// Metal without the device at hand (static estimate): the Metal FP8
    /// GEMV's contract, F16 activations, in a Metal build.
    StaticMetal,
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
    lm_head: Option<String>,
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
    /// The dequantized prefill GEMM (CUDA): planned and not switched off.
    /// Each projection kind still passes its own load-time check first.
    fp8_prefill_gemm: Qwen38FastPath,
    /// Pack row-compatible projections into one matrix.
    packing: bool,
    /// The LM head quantized to block FP8 (with native FP8 projections).
    fp8_lm_head: Qwen38FastPath,
    /// CPU: the LM head requantized to Q8_0 with the Q8_0 projections.
    q8_lm_head: bool,
    mtp: bool,
    /// CUDA: the DeltaNet alpha/beta projection may also keep its weights as
    /// F32 rows for the router GEMV (`text.rs::resolve_alpha_beta_gemv`), so
    /// admission charges those rows beside the dense weights.
    alpha_beta_gemv: bool,
    /// CUDA: why a switch keeps graph decode off, if one does.
    graph_switch_off: Option<String>,
    /// How the MTP head's loading was decided, for diagnostics.
    mtp_decision: String,
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

/// Environment switches whose true value (`1`, `true`, `yes`, `on`) asks for
/// the MTP head explicitly: the global CUDA MTP switch and the Qwen3.8 alias.
/// `auto` or unset lets CUDA prefer graph decode.
pub(crate) const MTP_EXPLICIT_ENVS: [&str; 3] = [
    "IZWI_CUDA_MTP",
    "IZWI_PERFORMANCE_CUDA_MTP",
    "IZWI_QWEN38_MTP",
];

fn mtp_explicitly_requested() -> bool {
    MTP_EXPLICIT_ENVS
        .iter()
        .any(|name| mtp_explicit_value(name, crate::env::value(name).as_deref()))
}

/// Whether switch `name`'s value asks for MTP explicitly: a true boolean
/// (not `auto`, not off).
pub(crate) fn mtp_explicit_value(name: &str, value: Option<&str>) -> bool {
    value.is_some_and(|value| matches!(izwi_env::parse_bool(name, value), Ok(true)))
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
        } else if backend == BackendKind::Metal {
            policy.resolve_metal(device, config);
        } else if backend == BackendKind::Cpu {
            policy.resolve_cpu(config);
        }
        policy.resolve_mtp_default(mtp_explicitly_requested());
        policy
    }

    /// CUDA: with `cuda.mtp=auto`, leave the MTP head out when graph decode
    /// would otherwise be planned (its switches on, native block-FP8
    /// projections). Graph decode covers decode without MTP, and MTP's rounds
    /// still run the eager sequence/verify path (about 3,000 launches per
    /// round for about 1.7 tokens), so one graph launch per token is expected
    /// to be faster until MTP verification runs in graphs too. An explicit
    /// `IZWI_QWEN38_MTP=1` (or `IZWI_CUDA_MTP=1`) keeps the head. Admission
    /// and the loader both resolve this policy, so they agree on the head.
    fn resolve_mtp_default(&mut self, explicit: bool) {
        if !self.mtp || self.backend != BackendKind::Cuda {
            return;
        }
        if explicit {
            self.mtp_decision =
                "explicit: on (IZWI_QWEN38_MTP or IZWI_CUDA_MTP asked for the head)".into();
            return;
        }
        if self.graph_switch_off.is_none()
            && matches!(self.plan, Qwen38ProjectionPlan::NativeFp8 { .. })
        {
            self.mtp = false;
            self.mtp_decision = "auto: off on CUDA, which prefers graph decode while MTP verification runs eagerly; set IZWI_QWEN38_MTP=1 to load the head".into();
        } else {
            self.mtp_decision = format!(
                "auto: on (CUDA graph decode is not planned: {})",
                self.graph_switch_off.clone().unwrap_or_else(|| format!(
                    "projections are not native block FP8 ({})",
                    self.fallback.as_deref().unwrap_or("not selected")
                ))
            );
        }
    }

    /// CPU: Q8_0 only when some block-FP8 projection can hold Q8_0 blocks (K
    /// divisible by 32); a checkpoint with none keeps the F32 expansion and
    /// reports it.
    fn resolve_cpu(&mut self, config: &Qwen38NativeConfig) {
        if self.plan != Qwen38ProjectionPlan::Q8 {
            return;
        }
        let eligible = fp8_geometries(config, self.mtp, false)
            .map(|shapes| shapes.iter().any(|[_, k]| k.is_multiple_of(32)))
            .unwrap_or(false);
        if !eligible {
            self.plan = Qwen38ProjectionPlan::Expanded;
            self.fallback = Some("no block-FP8 projection has K divisible by 32 for Q8_0".into());
            self.packing = false;
            self.q8_lm_head = false;
        }
    }

    /// Metal: raw block FP8 unless `metal.projection_backend = "expanded"`,
    /// the kill switch, or an earlier load's failed check says otherwise, and
    /// (on Auto) only when every FP8 geometry fits the Metal FP8 GEMV. There
    /// are no device gates before admission here: the load-time check against
    /// the original checkpoint bytes runs on every projection kind, and a GEMV
    /// mismatch reroutes that kind through the dequantize + matmul path.
    fn resolve_metal(&mut self, device: &Device, config: &Qwen38NativeConfig) {
        let Qwen38ProjectionPlan::NativeFp8 { explicit } = self.plan else {
            return;
        };
        let expanded = |policy: &mut Self, reason: String| {
            policy.plan = Qwen38ProjectionPlan::Expanded;
            policy.fallback = Some(reason);
            policy.packing = false;
            policy.fp8_gemv = Qwen38FastPath::legacy("native block FP8 not selected");
        };
        if let Some(reason) = poisoned(self.identity.as_ref()).projections {
            return expanded(
                self,
                format!("self_check_failed (poisoned by an earlier load): {reason}"),
            );
        }
        self.fp8_device = Fp8Device::Probe(device.clone());
        let block = config.block_fp8.block_shape;
        let shapes = match fp8_geometries(config, self.mtp, self.packing) {
            Ok(shapes) => shapes,
            Err(error) => return expanded(self, format!("FP8 geometry: {error}")),
        };
        let unsupported: Vec<_> = shapes
            .iter()
            .filter(|shape| !self.fp8_fits(**shape, block))
            .collect();
        if unsupported.len() == shapes.len() || (!unsupported.is_empty() && !explicit) {
            expanded(
                self,
                format!("Metal FP8 kernel contract does not cover {unsupported:?}"),
            );
        }
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
        let expanded_reason =
            "native block-FP8 GEMM is not runtime-certified; using the scale-exact expanded path";
        let (plan, fallback) = match backend {
            BackendKind::Cuda => (Qwen38ProjectionPlan::Q8, None),
            BackendKind::Cpu => match performance.cpu.projection_backend {
                CpuProjectionBackend::Expanded => (
                    Qwen38ProjectionPlan::Expanded,
                    Some(format!(
                        "cpu.projection_backend=expanded: {expanded_reason}"
                    )),
                ),
                CpuProjectionBackend::Auto | CpuProjectionBackend::Q8 => {
                    (Qwen38ProjectionPlan::Q8, None)
                }
            },
            BackendKind::Metal => match performance.metal.projection_backend {
                MetalProjectionBackend::Expanded => (
                    Qwen38ProjectionPlan::Expanded,
                    Some("metal.projection_backend=expanded".into()),
                ),
                _ if legacy_requested(FP8_PROJECTIONS_ENV) => (
                    Qwen38ProjectionPlan::Expanded,
                    Some(format!("{FP8_PROJECTIONS_ENV}=legacy")),
                ),
                _ if !cfg!(feature = "metal") || activation != DType::F16 => (
                    Qwen38ProjectionPlan::Expanded,
                    Some(
                        "the Metal block-FP8 kernels need a Metal build and F16 activations".into(),
                    ),
                ),
                requested => (
                    Qwen38ProjectionPlan::NativeFp8 {
                        explicit: requested == MetalProjectionBackend::NativeFp8,
                    },
                    None,
                ),
            },
        };
        let metal_native =
            backend == BackendKind::Metal && matches!(plan, Qwen38ProjectionPlan::NativeFp8 { .. });
        Self {
            backend,
            activation,
            plan,
            fallback,
            fp8_device: if metal_native {
                Fp8Device::StaticMetal
            } else {
                Fp8Device::Unavailable
            },
            fp8_gemv: if !metal_native {
                Qwen38FastPath::legacy("native block FP8 not selected")
            } else if legacy_requested(FP8_GEMV_ENV) {
                Qwen38FastPath::legacy(format!("{FP8_GEMV_ENV}=legacy"))
            } else {
                Qwen38FastPath::Fused
            },
            fp8_prefill_gemm: Qwen38FastPath::legacy(if metal_native {
                "Metal prefill expands to F16 for one matmul"
            } else {
                "native block FP8 not selected"
            }),
            packing: metal_native
                || (backend == BackendKind::Cpu && plan == Qwen38ProjectionPlan::Q8)
                || backend == BackendKind::Cuda
                    && cuda.enabled()
                    && cuda.packed_projections.enabled(),
            q8_lm_head: backend == BackendKind::Cpu
                && plan == Qwen38ProjectionPlan::Q8
                && !cpu_dense_head_requested(),
            fp8_lm_head: Qwen38FastPath::legacy(if metal_native {
                "the Metal LM head stays F16"
            } else {
                "native block FP8 not selected"
            }),
            mtp: mtp_loaded_for(backend, performance),
            alpha_beta_gemv: backend == BackendKind::Cuda
                && cuda.enabled()
                && cuda.fused_decode.enabled()
                && matches!(activation, DType::F16 | DType::BF16)
                && !legacy_requested(FUSED_AB_GEMV_ENV),
            graph_switch_off: if !cuda.enabled() {
                Some("cuda.mode=off".into())
            } else if !cuda.decode_graphs.enabled() {
                Some("cuda.decode_graphs=off".into())
            } else if !cuda.fused_decode.enabled() {
                Some(
                    "CUDA graph decode needs the fused decode kernels (cuda.fused_decode=off)"
                        .into(),
                )
            } else if legacy_requested(DECODE_GRAPHS_ENV) {
                Some(format!("{DECODE_GRAPHS_ENV}=legacy"))
            } else {
                None
            },
            mtp_decision: if mtp_loaded_for(backend, performance) {
                "auto: on".into()
            } else if backend == BackendKind::Cuda && !cuda.enabled() {
                "off: cuda.mode=off".into()
            } else {
                "off: cuda.mtp=off".into()
            },
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
        self.fp8_lm_head = if legacy_requested(FP8_LM_HEAD_ENV) {
            Qwen38FastPath::legacy(format!("{FP8_LM_HEAD_ENV}=legacy"))
        } else if let Some(reason) = poisoned(self.identity.as_ref()).lm_head {
            Qwen38FastPath::legacy(format!(
                "self_check_failed (poisoned by an earlier load): {reason}"
            ))
        } else {
            Qwen38FastPath::Fused
        };
        self.fp8_gemv = if legacy_requested(FP8_GEMV_ENV) {
            Qwen38FastPath::legacy(format!("{FP8_GEMV_ENV}=legacy"))
        } else if !cuda.fused_decode.enabled() {
            Qwen38FastPath::legacy("cuda.fused_decode=off")
        } else if let Err(reason) = vectorized {
            Qwen38FastPath::legacy(format!("self_check_failed: {reason}"))
        } else {
            Qwen38FastPath::Fused
        };
        self.fp8_prefill_gemm = if legacy_requested(FP8_PREFILL_GEMM_ENV) {
            Qwen38FastPath::legacy(format!("{FP8_PREFILL_GEMM_ENV}=legacy"))
        } else if !device.is_cuda() {
            Qwen38FastPath::legacy("the dequantized prefill GEMM runs on CUDA only")
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
        policy.fp8_lm_head = Qwen38FastPath::Fused;
        policy
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

    /// How the MTP head's loading was decided, for diagnostics.
    pub(crate) fn mtp_decision(&self) -> &str {
        &self.mtp_decision
    }

    /// Whether CUDA graph decode may resolve at load (`text/decode_graph.rs`):
    /// CUDA with the master, fused-decode and decode-graph switches on,
    /// native block-FP8 projections (Candle's Q8_0 QMatMul cannot be
    /// captured), no MTP head (graph decode covers the non-MTP decode), and
    /// `IZWI_QWEN38_CUDA_GRAPHS` not `legacy`. Admission reserves the graph
    /// working memory exactly when this holds, and the loader resolves graph
    /// decode only when it holds, so a load never uses memory admission did
    /// not grant. The load-time self-checks can still keep it off.
    pub(crate) fn graph_decode_planned(&self) -> bool {
        self.graph_decode_refusal_reason().is_none()
    }

    /// Why graph decode is not planned, for diagnostics.
    pub(crate) fn graph_decode_refusal(&self) -> String {
        self.graph_decode_refusal_reason()
            .unwrap_or_else(|| "planned".into())
    }

    fn graph_decode_refusal_reason(&self) -> Option<String> {
        if self.backend != BackendKind::Cuda {
            Some("CUDA graph decode runs on CUDA only".into())
        } else if let Some(reason) = &self.graph_switch_off {
            Some(reason.clone())
        } else if self.mtp {
            Some("CUDA graph decode covers decode without MTP, and the MTP head is loaded".into())
        } else if !matches!(self.plan, Qwen38ProjectionPlan::NativeFp8 { .. }) {
            Some(format!(
                "CUDA graph decode needs native block-FP8 projections ({})",
                self.fallback.as_deref().unwrap_or("not selected")
            ))
        } else {
            None
        }
    }

    /// Device bytes the decode graphs may hold: CUDA graph decode's working
    /// memory when this policy plans it (its graphs replace the bounded
    /// `TensorIsland` regions, which retire when it resolves), otherwise the
    /// bounded region cache of each CUDA stack (the trunk, and the MTP head
    /// when it loads). Admission charges it and the chat model reports it.
    pub(crate) fn graph_capacity_bytes(
        &self,
        text: &Qwen38TextConfig,
        performance: &PerformanceConfig,
    ) -> Result<u64> {
        if self.graph_decode_planned() {
            return graph_decode_reserve_bytes(text);
        }
        Ok(
            if self.backend == BackendKind::Cuda
                && performance.cuda.enabled()
                && performance.cuda.decode_graphs.enabled()
            {
                TENSOR_ISLAND_CACHE_BYTES * if self.mtp { 2 } else { 1 }
            } else {
                0
            },
        )
    }

    /// A CUDA policy with native block-FP8 projections, resolved without a
    /// device (tests of the graph-decode plan and its admission).
    #[cfg(test)]
    pub(crate) fn cuda_native_fp8_for_tests(performance: &PerformanceConfig) -> Self {
        Self::cuda_native_fp8_with_mtp_request_for_tests(performance, false)
    }

    /// [`Self::cuda_native_fp8_for_tests`] with the MTP head explicitly
    /// requested (`explicit_mtp`) or not, as the environment would.
    #[cfg(test)]
    pub(crate) fn cuda_native_fp8_with_mtp_request_for_tests(
        performance: &PerformanceConfig,
        explicit_mtp: bool,
    ) -> Self {
        let mut policy = Self::base(BackendKind::Cuda, DType::BF16, performance);
        policy.plan = Qwen38ProjectionPlan::NativeFp8 { explicit: false };
        policy.fallback = None;
        policy.resolve_mtp_default(explicit_mtp);
        policy
    }

    /// Whether the FP8 kernels run the vectorized decode GEMV.
    pub(crate) fn fp8_vectorized(&self) -> bool {
        self.fp8_gemv.is_fused()
    }

    /// Whether CUDA block-FP8 projections may run long spans through the
    /// dequantized prefill GEMM (each kind after its own load-time check).
    pub(crate) fn fp8_prefill_gemm(&self) -> bool {
        matches!(self.plan, Qwen38ProjectionPlan::NativeFp8 { .. })
            && self.fp8_prefill_gemm.is_fused()
    }

    /// Tests: plan the prefill GEMM on the CPU stand-in for the FP8 kernels.
    #[cfg(test)]
    pub(crate) fn force_fp8_prefill_gemm_for_tests(&mut self) {
        self.fp8_prefill_gemm = Qwen38FastPath::Fused;
    }

    /// Metal: whether every FP8 projection skips the GEMV (dequantize +
    /// matmul for every row count), from `IZWI_QWEN38_FP8_GEMV=legacy`.
    pub(crate) fn fp8_dequant_only(&self) -> bool {
        self.backend == BackendKind::Metal
            && matches!(self.plan, Qwen38ProjectionPlan::NativeFp8 { .. })
            && !self.fp8_gemv.is_fused()
    }

    /// Lazily allocated model-owned scratch the resident estimate includes:
    /// the transient F16 expansion of Metal FP8 prefill, or on CUDA the one
    /// dequantized weight tile of the prefill GEMM.
    pub(crate) fn prefill_scratch_bytes(&self) -> u64 {
        if self.backend == BackendKind::Metal
            && matches!(self.plan, Qwen38ProjectionPlan::NativeFp8 { .. })
        {
            METAL_FP8_PREFILL_DEQUANT_BYTES
        } else if self.backend == BackendKind::Cuda && self.fp8_prefill_gemm() {
            fp8::DEQUANT_GEMM_TILE_BYTES as u64
        } else {
            0
        }
    }

    /// Whether a `[n, k]` block-FP8 projection with `block` scales runs on
    /// the compact FP8 kernels here.
    fn fp8_fits(&self, [n, k]: [usize; 2], block: [usize; 2]) -> bool {
        block == FP8_BLOCK
            && match &self.fp8_device {
                Fp8Device::Probe(device) => fp8::provider_supported(device, self.activation, n, k),
                Fp8Device::Unavailable => false,
                Fp8Device::StaticMetal => {
                    cfg!(feature = "metal")
                        && self.activation == DType::F16
                        && n > 0
                        && k > 0
                        && n.is_multiple_of(64)
                        && k.is_multiple_of(128)
                        && n.checked_mul(k).is_some_and(|v| v <= i32::MAX as usize)
                }
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
            // Q8_0 blocks hold 32 elements of one row.
            Qwen38ProjectionPlan::Q8
                if self.backend == BackendKind::Cpu && !shape[1].is_multiple_of(32) =>
            {
                Qwen38WeightForm::Dense(self.activation)
            }
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
        // Metal packs only raw block-FP8 groups (dense groups stay separate,
        // as they always have there).
        self.packing
            && forms.iter().all(|form| form == first)
            && (self.backend != BackendKind::Metal || *first == Qwen38WeightForm::RawFp8)
            && (self.backend != BackendKind::Cpu || !matches!(first, Qwen38WeightForm::Dense(_)))
    }

    /// The resident form of the `[vocab, hidden]` LM head (BF16 in the
    /// checkpoint): block FP8 quantized at load beside native FP8
    /// projections, otherwise the activation dtype.
    pub(crate) fn lm_head_form(&self, shape: [usize; 2]) -> Qwen38WeightForm {
        if matches!(self.plan, Qwen38ProjectionPlan::NativeFp8 { .. })
            && self.fp8_lm_head.is_fused()
            && self.fp8_fits(shape, FP8_BLOCK)
        {
            Qwen38WeightForm::RawFp8
        } else if self.q8_lm_head
            && self.plan == Qwen38ProjectionPlan::Q8
            && shape[1].is_multiple_of(32)
        {
            Qwen38WeightForm::Q8_0
        } else {
            Qwen38WeightForm::Dense(self.activation)
        }
    }

    /// The FP8 LM head's resolution, for diagnostics.
    pub(crate) fn fp8_lm_head(&self) -> &Qwen38FastPath {
        &self.fp8_lm_head
    }

    pub(crate) fn representation(&self) -> Qwen38ProjectionRepresentation {
        use Qwen38ProjectionRepresentation as R;
        match (self.plan, self.activation) {
            (Qwen38ProjectionPlan::NativeFp8 { .. }, DType::F16)
                if self.backend == BackendKind::Metal =>
            {
                R::NativeFp8WithDenseF16
            }
            (Qwen38ProjectionPlan::NativeFp8 { .. }, DType::F16) => R::NativeFp8WithQ8FallbackF16,
            (Qwen38ProjectionPlan::NativeFp8 { .. }, DType::F32) => R::ExpandedF32,
            (Qwen38ProjectionPlan::NativeFp8 { .. }, _) => R::NativeFp8WithQ8FallbackBf16,
            (Qwen38ProjectionPlan::Q8, DType::F16) => R::PackedQ8WithDenseF16,
            (Qwen38ProjectionPlan::Q8, DType::F32) => R::PackedQ8WithDenseF32,
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
            (Qwen38ProjectionPlan::Q8, BackendKind::Cpu) if self.fallback.is_none() => {
                "q8_0_cpu_requantized".into()
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
            Qwen38ProjectionPlan::NativeFp8 { .. } if self.backend == BackendKind::Metal => "block-FP8 projections stay resident on Metal as the checkpoint's E4M3 bytes; up to 8 rows run the FP8 GEMV and longer spans expand to a transient F16 matrix for one matrix multiplication, after a load-time check of every projection kind against the original checkpoint bytes; the LM head and dense tensors stay F16".into(),
            Qwen38ProjectionPlan::NativeFp8 { .. } => "block-FP8 projections stay resident as the checkpoint's E4M3 bytes and run on the compact FP8 kernels (16-bit activations, software FP8 decode) after device gates and a load-time check against the original checkpoint bytes; spans of 16 rows or more expand bounded weight tiles (128 MiB) to the activation dtype for cuBLAS once the projection kind passes its prefill check (IZWI_QWEN38_FP8_PREFILL_GEMM=legacy keeps the compact kernel); not runtime-certified on CUDA hardware".into(),
            Qwen38ProjectionPlan::Q8 if self.backend == BackendKind::Cpu && self.fallback.is_none() => "the CPU keeps block-FP8 projections (K divisible by 32) and the LM head as Q8_0: weight_scale_inv applied, then requantized (activations are quantized to Q8_0 in decode); decode rows run Candle's Q8_0 matmul and longer spans expand row tiles to F32 for one F32 matmul; the embedding and other dense tensors stay F32; IZWI_CPU_PROJECTION_BACKEND=expanded keeps the F32 expansion".into(),
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
            "fp8_prefill_gemm": summarize([&self.fp8_prefill_gemm]),
            "fp8_execution_mode": self.execution_mode(),
        })
    }

    /// Fail closed after a load-time self-check: every later resolution for
    /// this device and checkpoint keeps the projections off native FP8, so a
    /// retry admits and loads the fallback with exact accounting.
    pub(crate) fn poison_projections(&self, reason: &str) {
        self.poison(|poison| poison.projections = Some(reason.to_string()));
    }

    /// Fail closed after the FP8 LM head's checks: later resolutions for this
    /// device and checkpoint keep the head in the activation dtype.
    pub(crate) fn poison_lm_head(&self, reason: &str) {
        self.poison(|poison| poison.lm_head = Some(reason.to_string()));
    }

    fn poison(&self, apply: impl FnOnce(&mut Poison)) {
        let Some(identity) = &self.identity else {
            return;
        };
        let mut poisoned = lock(&POISONED);
        if let Some((_, poison)) = poisoned.iter_mut().find(|(key, _)| key == identity) {
            apply(poison);
        } else {
            let mut poison = Poison::default();
            apply(&mut poison);
            poisoned.push((identity.clone(), poison));
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
            let [n, k] = <[usize; 2]>::try_from(shape)
                .map_err(|_| Error::ModelLoadError("Qwen3.8 LM head is not a matrix".into()))?;
            self.lm_head_form([n, k])
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
        let bytes = form_bytes(form, shape, count)?;
        if self.alpha_beta_gemv && dtype != SafeDType::F8_E4M3 && is_alpha_beta_projection(name) {
            // The router GEMV's F32 copy of the rows, beside the dense weights.
            return count
                .checked_mul(4)
                .and_then(|rows| rows.checked_add(bytes))
                .ok_or_else(overflow);
        }
        Ok(bytes)
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

/// A trunk DeltaNet `in_proj_a`/`in_proj_b` weight: the dense pair the router
/// GEMV also keeps as F32 rows.
fn is_alpha_beta_projection(name: &str) -> bool {
    name.starts_with("model.language_model.layers.")
        && (name.ends_with(".linear_attn.in_proj_a.weight")
            || name.ends_with(".linear_attn.in_proj_b.weight"))
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

/// Fixed device bytes graph decode keeps beside its state slabs: the segment
/// and whole-step graph pools, the device state table and step buffer, and
/// the logits and token-selection buffers.
pub(crate) const GRAPH_DECODE_FIXED_BYTES: u64 = 64 * 1024 * 1024;

/// One graph-decode state slab: every DeltaNet layer's next `[C, 3]` conv
/// history and `[Hv, Dk, Dv]` recurrent state, F32 (156,893,184 B for the
/// 27B trunk, the same bytes as one row's recurrent state).
pub(crate) fn graph_decode_slab_bytes(text: &Qwen38TextConfig) -> Result<u64> {
    let invalid = || Error::ModelLoadError("Qwen3.8 graph decode state geometry is invalid".into());
    if text.full_attention_interval == 0
        || text.ssm_conv_kernel == 0
        || text.ssm_time_step_rank == 0
        || !text.ssm_inner_size.is_multiple_of(text.ssm_time_step_rank)
    {
        return Err(invalid());
    }
    let linear_layers = text.block_count - text.block_count / text.full_attention_interval;
    let conv_dim = text
        .ssm_state_size
        .checked_mul(text.ssm_group_count)
        .and_then(|keys| keys.checked_mul(2))
        .and_then(|keys| keys.checked_add(text.ssm_inner_size))
        .ok_or_else(invalid)?;
    let history = conv_dim
        .checked_mul(text.ssm_conv_kernel - 1)
        .ok_or_else(invalid)?;
    let recurrent = text
        .ssm_time_step_rank
        .checked_mul(text.ssm_state_size)
        .and_then(|rows| rows.checked_mul(text.ssm_inner_size / text.ssm_time_step_rank))
        .ok_or_else(invalid)?;
    history
        .checked_add(recurrent)
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
        .and_then(|bytes| bytes.checked_mul(linear_layers))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(overflow)
}

/// Device working memory CUDA graph decode needs beyond the row's own
/// recurrent state: three state slabs (the next step's slab, written while the
/// row still holds the current one; a scratch slab for capture verification;
/// a second step queued by pipelined decode) plus the fixed graph pools.
pub(crate) fn graph_decode_reserve_bytes(text: &Qwen38TextConfig) -> Result<u64> {
    graph_decode_slab_bytes(text)?
        .checked_mul(3)
        .and_then(|slabs| slabs.checked_add(GRAPH_DECODE_FIXED_BYTES))
        .ok_or_else(overflow)
}

/// The bounded `TensorIsland` region cache of one stack (trunk or MTP head).
pub(crate) const TENSOR_ISLAND_CACHE_BYTES: u64 = 8 * 1024 * 1024;

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
