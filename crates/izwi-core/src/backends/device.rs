//! Device selection for native inference with Metal optimizations.
//!
//! This module provides optimized device selection with Metal-specific improvements:
//! - Optimized dtype selection based on device capabilities
//! - Memory pool integration for reduced allocation overhead
//! - Unified memory awareness for Apple Silicon

use candle_core::{DType, Device};
use std::any::Any;
use std::borrow::Cow;
use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use tracing::{debug, info, warn};

use super::types::{BackendKind, BackendPreference, RuntimeDeviceAssignment};
use crate::catalog::ModelFamily;
use crate::error::{Error, Result};
use crate::models::shared::memory::metal::{metal_pool_for_device, MetalMemoryPool};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Cuda,
    Metal,
    Cpu,
}

impl DeviceKind {
    pub fn is_cpu(&self) -> bool {
        matches!(self, DeviceKind::Cpu)
    }

    pub fn is_metal(&self) -> bool {
        matches!(self, DeviceKind::Metal)
    }

    pub fn is_cuda(&self) -> bool {
        matches!(self, DeviceKind::Cuda)
    }
}

impl From<DeviceKind> for BackendKind {
    fn from(kind: DeviceKind) -> Self {
        match kind {
            DeviceKind::Cpu => BackendKind::Cpu,
            DeviceKind::Metal => BackendKind::Metal,
            DeviceKind::Cuda => BackendKind::Cuda,
        }
    }
}

/// Device capabilities and optimization hints
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    /// Whether the device prefers float32 (Metal on Apple Silicon)
    pub prefers_f32: bool,
    /// Whether the device supports bfloat16
    pub supports_bf16: bool,
    /// Whether the device supports float16 compute/storage for model activations
    pub supports_f16: bool,
    /// Whether the device has int8 tensor cores
    pub supports_int8_tensor_cores: bool,
    /// Whether the device has unified memory (Apple Silicon)
    pub has_unified_memory: bool,
    /// Recommended batch size for this device
    pub recommended_batch_size: usize,
    /// Available memory in bytes (if detectable)
    pub available_memory_bytes: Option<usize>,
    /// Total physical memory for the selected CUDA device.
    pub cuda_total_memory_bytes: Option<usize>,
    /// Ordinal used to select the CUDA device.
    pub cuda_device_ordinal: Option<usize>,
    /// CUDA compute capability for NVIDIA devices
    pub cuda_compute_capability: Option<(u32, u32)>,
    /// CUDA device name when reported by the runtime
    pub cuda_device_name: Option<String>,
}

impl Default for DeviceCapabilities {
    fn default() -> Self {
        Self {
            prefers_f32: false,
            supports_bf16: false,
            supports_f16: false,
            supports_int8_tensor_cores: false,
            has_unified_memory: false,
            recommended_batch_size: 1,
            available_memory_bytes: None,
            cuda_total_memory_bytes: None,
            cuda_device_ordinal: None,
            cuda_compute_capability: None,
            cuda_device_name: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DTypeSelectionPolicy {
    #[default]
    Default,
    PreferF32,
    PreferF16,
    PreferBf16,
}

#[derive(Debug, Clone, Copy)]
pub struct DTypeSelectionRequest<'a> {
    pub requested: Option<&'a str>,
    pub model_family: Option<ModelFamily>,
    pub checkpoint_dtype: Option<DType>,
    pub quantized: bool,
    pub policy: DTypeSelectionPolicy,
}

impl<'a> DTypeSelectionRequest<'a> {
    pub fn new(requested: Option<&'a str>) -> Self {
        Self {
            requested,
            model_family: None,
            checkpoint_dtype: None,
            quantized: false,
            policy: DTypeSelectionPolicy::Default,
        }
    }

    pub fn with_model_family(mut self, model_family: ModelFamily) -> Self {
        self.model_family = Some(model_family);
        self
    }

    pub fn with_checkpoint_dtype(mut self, checkpoint_dtype: DType) -> Self {
        self.checkpoint_dtype = Some(checkpoint_dtype);
        self
    }

    pub fn with_quantized(mut self, quantized: bool) -> Self {
        self.quantized = quantized;
        self
    }

    pub fn with_policy(mut self, policy: DTypeSelectionPolicy) -> Self {
        self.policy = policy;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DTypeSelection {
    pub dtype: DType,
    pub reason: Cow<'static, str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DTypeSelectionError {
    pub requested: String,
    pub reason: Cow<'static, str>,
}

impl std::fmt::Display for DTypeSelectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "unsupported dtype override {:?}: {}",
            self.requested, self.reason
        )
    }
}

impl std::error::Error for DTypeSelectionError {}

#[derive(Debug, Clone)]
pub struct DeviceProfile {
    pub device: Device,
    pub kind: DeviceKind,
    pub capabilities: DeviceCapabilities,
    /// Optional memory pool for this device (Metal only)
    pub memory_pool: Option<Arc<MetalMemoryPool>>,
}

impl DeviceProfile {
    pub fn cpu() -> Self {
        Self {
            device: Device::Cpu,
            kind: DeviceKind::Cpu,
            capabilities: DeviceCapabilities::default(),
            memory_pool: None,
        }
    }

    /// Select optimal dtype based on device kind and requested preference
    ///
    /// # Optimization Notes:
    /// - Metal: each family keeps its own default (F32 unless a family policy
    ///   says otherwise); `IZWI_METAL_DTYPE` replaces that default, and an
    ///   explicit request (F32, F16 or BF16) is honoured.
    /// - CUDA: BF16 is preferred for compute, F16 for memory-constrained scenarios
    /// - CPU: Always F32 since most CPUs don't have efficient F16/BF16 paths
    pub fn select_dtype(&self, requested: Option<&str>) -> DType {
        let request = DTypeSelectionRequest::new(requested);
        let selection = match self.try_resolve_dtype(request) {
            Ok(selection) => selection,
            Err(err) => {
                warn!(
                    "Falling back from unsupported dtype override on {:?}: {}",
                    self.kind, err
                );
                self.resolve_dtype(DTypeSelectionRequest::new(None))
            }
        };

        debug!(
            "Selected dtype {:?} for device {:?} (requested: {:?}, reason: {})",
            selection.dtype, self.kind, requested, selection.reason
        );

        selection.dtype
    }

    pub fn try_select_dtype(
        &self,
        requested: Option<&str>,
    ) -> std::result::Result<DType, DTypeSelectionError> {
        self.try_resolve_dtype(DTypeSelectionRequest::new(requested))
            .map(|selection| selection.dtype)
    }

    pub fn select_model_dtype(&self, model_family: ModelFamily, requested: Option<&str>) -> DType {
        let request = DTypeSelectionRequest::new(requested).with_model_family(model_family);
        let selection = match self.try_resolve_dtype(request) {
            Ok(selection) => selection,
            Err(err) => {
                warn!(
                    "Falling back from unsupported dtype override for {:?} on {:?}: {}",
                    model_family, self.kind, err
                );
                self.resolve_dtype(DTypeSelectionRequest::new(None).with_model_family(model_family))
            }
        };

        debug!(
            "Selected dtype {:?} for {:?} on {:?} (requested: {:?}, reason: {})",
            selection.dtype, model_family, self.kind, requested, selection.reason
        );

        selection.dtype
    }

    pub fn select_model_dtype_with_checkpoint(
        &self,
        model_family: ModelFamily,
        checkpoint_dtype: Option<DType>,
    ) -> DType {
        let mut request = DTypeSelectionRequest::new(None).with_model_family(model_family);
        if let Some(dtype) = checkpoint_dtype {
            request = request.with_checkpoint_dtype(dtype);
        }
        let selection = self.resolve_dtype(request);

        debug!(
            "Selected dtype {:?} for {:?} on {:?} (checkpoint: {:?}, reason: {})",
            selection.dtype, model_family, self.kind, checkpoint_dtype, selection.reason
        );

        selection.dtype
    }

    pub fn try_select_model_dtype(
        &self,
        model_family: ModelFamily,
        requested: Option<&str>,
    ) -> std::result::Result<DType, DTypeSelectionError> {
        self.try_resolve_dtype(
            DTypeSelectionRequest::new(requested).with_model_family(model_family),
        )
        .map(|selection| selection.dtype)
    }

    pub fn select_model_dtype_checked(
        &self,
        model_family: ModelFamily,
        requested: Option<&str>,
        context: &str,
    ) -> Result<DType> {
        let requested = requested.map(str::trim).filter(|raw| !raw.is_empty());
        if self.kind.is_cuda() && requested.is_some() {
            return self
                .try_select_model_dtype(model_family, requested)
                .map_err(|err| {
                    Error::InvalidInput(format!("Invalid CUDA {context} dtype override: {err}"))
                });
        }

        Ok(self.select_model_dtype(model_family, requested))
    }

    pub fn resolve_dtype(&self, request: DTypeSelectionRequest<'_>) -> DTypeSelection {
        self.try_resolve_dtype(request)
            .unwrap_or_else(|_| self.default_dtype_selection(request))
    }

    pub fn try_resolve_dtype(
        &self,
        request: DTypeSelectionRequest<'_>,
    ) -> std::result::Result<DTypeSelection, DTypeSelectionError> {
        if let Some(raw) = request.requested {
            let raw = raw.trim();
            if !raw.is_empty() {
                if let Some(dtype) = parse_dtype_name(raw) {
                    return self.resolve_requested_dtype(request, raw, dtype);
                }

                return Err(DTypeSelectionError {
                    requested: raw.to_string(),
                    reason: "expected one of f32, f16, or bf16".into(),
                });
            }
        }

        Ok(self.default_dtype_selection(request))
    }

    fn resolve_requested_dtype(
        &self,
        request: DTypeSelectionRequest<'_>,
        raw: &str,
        requested: DType,
    ) -> std::result::Result<DTypeSelection, DTypeSelectionError> {
        match self.kind {
            DeviceKind::Cpu => Ok(DTypeSelection {
                dtype: DType::F32,
                reason: "CPU inference uses F32 regardless of lower-precision override".into(),
            }),
            DeviceKind::Metal => self.resolve_requested_metal_dtype(request, raw, requested),
            DeviceKind::Cuda => match requested {
                DType::F32 => Ok(DTypeSelection {
                    dtype: DType::F32,
                    reason: "explicit CUDA F32 override".into(),
                }),
                DType::F16 if self.capabilities.supports_f16 => Ok(DTypeSelection {
                    dtype: DType::F16,
                    reason: "explicit CUDA F16 override".into(),
                }),
                DType::BF16 if self.capabilities.supports_bf16 => Ok(DTypeSelection {
                    dtype: DType::BF16,
                    reason: "explicit CUDA BF16 override supported by device capability".into(),
                }),
                DType::BF16 => Err(DTypeSelectionError {
                    requested: raw.to_string(),
                    reason: "CUDA device does not report BF16 support".into(),
                }),
                DType::F16 => Err(DTypeSelectionError {
                    requested: raw.to_string(),
                    reason: "CUDA device does not report F16 support".into(),
                }),
                _ => Err(DTypeSelectionError {
                    requested: raw.to_string(),
                    reason: "dtype override is not supported for CUDA inference".into(),
                }),
            },
        }
    }

    /// An explicit Metal request is honoured: weights and activations at F16
    /// or BF16 are what llama.cpp, whisper.cpp and MLX run on Apple GPUs, and
    /// Candle's Metal gemm accumulates in F32 for both. BF16 is accepted on
    /// every Apple family (Candle has BF16 kernels everywhere) but runs slower
    /// than F16 before Apple9, so the reason says so.
    fn resolve_requested_metal_dtype(
        &self,
        request: DTypeSelectionRequest<'_>,
        raw: &str,
        requested: DType,
    ) -> std::result::Result<DTypeSelection, DTypeSelectionError> {
        if request
            .model_family
            .is_some_and(|family| METAL_REQUEST_IS_CONFIG_HINT.contains(&family))
        {
            let default = self.metal_default_dtype_selection(request);
            return Ok(DTypeSelection {
                dtype: default.dtype,
                reason: format!(
                    "{:?} passes its checkpoint config dtype rather than a user override; {}",
                    request.model_family, default.reason
                )
                .into(),
            });
        }
        match requested {
            DType::F32 => Ok(DTypeSelection {
                dtype: DType::F32,
                reason: "explicit Metal F32 override".into(),
            }),
            DType::F16 if self.capabilities.supports_f16 => Ok(DTypeSelection {
                dtype: DType::F16,
                reason: "explicit Metal F16 override".into(),
            }),
            DType::F16 => Err(DTypeSelectionError {
                requested: raw.to_string(),
                reason: "Metal device does not report F16 support".into(),
            }),
            DType::BF16 if self.metal_supports_fast_bf16() => Ok(DTypeSelection {
                dtype: DType::BF16,
                reason: "explicit Metal BF16 override on an Apple9+ GPU".into(),
            }),
            DType::BF16 => Ok(DTypeSelection {
                dtype: DType::BF16,
                reason: "explicit Metal BF16 override; GPUs before Apple9 (M1/M2) run BF16 slower than F16".into(),
            }),
            _ => Err(DTypeSelectionError {
                requested: raw.to_string(),
                reason: "dtype override is not supported for Metal inference".into(),
            }),
        }
    }

    /// Weight-only F16 for an F32 checkpoint: projection (linear and
    /// pointwise) weights stored in F16 and applied to F32 activations with
    /// F32 accumulation, the whisper.cpp recipe for NeMo models. Returns F16
    /// when the projections should load in F16 and F32 when every weight stays
    /// F32. `family_override` is the family's `(variable, value)` dtype
    /// override, if set.
    pub fn resolve_metal_weight_only_f16(
        &self,
        family_override: Option<(&str, &str)>,
    ) -> DTypeSelection {
        metal_weight_only_f16_selection(
            self.kind.is_metal(),
            family_override,
            MetalDTypeOverride::from_env(),
        )
    }

    /// The Metal decoder-LM tier for a BF16 checkpoint (report 13): BF16 on
    /// Apple9+ GPUs, F16 on Apple7/8. See [`metal_bf16_checkpoint_lm_tier`].
    pub fn metal_bf16_checkpoint_lm_tier(&self) -> DTypeSelection {
        metal_bf16_checkpoint_lm_tier(
            self.metal_supports_fast_bf16(),
            METAL_PAGED_ATTENTION_SUPPORTS_BF16,
        )
    }

    /// Whether this profile's device is an Apple9+ GPU (M3/M4/M5), the tier
    /// whose BF16 matches F16 throughput. False off Metal and when the family
    /// cannot be read.
    pub fn metal_supports_fast_bf16(&self) -> bool {
        self.kind.is_metal() && metal_supports_fast_bf16(&self.device)
    }

    /// The Metal default: `IZWI_METAL_DTYPE` when it names a dtype, else the
    /// family policy.
    fn metal_default_dtype_selection(&self, request: DTypeSelectionRequest<'_>) -> DTypeSelection {
        self.metal_default_dtype_selection_with(request, MetalDTypeOverride::from_env())
    }

    fn metal_default_dtype_selection_with(
        &self,
        request: DTypeSelectionRequest<'_>,
        global: MetalDTypeOverride,
    ) -> DTypeSelection {
        match global {
            MetalDTypeOverride::Auto => self.metal_family_dtype_selection(request),
            MetalDTypeOverride::F32 => DTypeSelection {
                dtype: DType::F32,
                reason: "IZWI_METAL_DTYPE=f32 forces F32 on Metal".into(),
            },
            MetalDTypeOverride::F16 => DTypeSelection {
                dtype: DType::F16,
                reason: "IZWI_METAL_DTYPE=f16 selects F16 on Metal".into(),
            },
            MetalDTypeOverride::Bf16 => DTypeSelection {
                dtype: DType::BF16,
                reason: if self.metal_supports_fast_bf16() {
                    "IZWI_METAL_DTYPE=bf16 selects BF16 on an Apple9+ GPU".into()
                } else {
                    "IZWI_METAL_DTYPE=bf16 selects BF16; GPUs before Apple9 (M1/M2) run BF16 slower than F16".into()
                },
            },
        }
    }

    fn default_dtype_selection(&self, request: DTypeSelectionRequest<'_>) -> DTypeSelection {
        match self.kind {
            DeviceKind::Cpu => DTypeSelection {
                dtype: DType::F32,
                reason: "CPU default dtype is F32".into(),
            },
            DeviceKind::Metal => self.metal_default_dtype_selection(request),
            DeviceKind::Cuda => self.default_cuda_dtype_selection(request),
        }
    }

    /// The per-family Metal default, before `IZWI_METAL_DTYPE`.
    fn metal_family_dtype_selection(&self, request: DTypeSelectionRequest<'_>) -> DTypeSelection {
        if matches!(
            request.model_family,
            Some(ModelFamily::Voxtral | ModelFamily::VoxtralTts)
        ) && request.checkpoint_dtype != Some(DType::F32)
        {
            // Voxtral's BF16 checkpoints: BF16 on Apple9+ (once Metal paged
            // attention takes BF16), F16 on Apple7/8, which also keeps
            // realtime decoding within Apple GPU memory and latency limits.
            // Voxtral TTS applies the tier to its LM and acoustic
            // transformer behind a load-time self-check; its codec stays F32.
            return self.metal_bf16_checkpoint_lm_tier();
        }
        DTypeSelection {
            dtype: if request.model_family == Some(ModelFamily::VibeVoiceTts)
                || request.model_family == Some(ModelFamily::WhisperAsr)
            {
                DType::F16
            } else {
                DType::F32
            },
            reason: if request.model_family == Some(ModelFamily::VibeVoiceTts) {
                "VibeVoice TTS Metal policy uses F16 to bound dense model residency".into()
            } else if request.model_family == Some(ModelFamily::WhisperAsr) {
                "Whisper Metal policy runs its F16 checkpoint in F16 on every Apple family, with F32 softmax, LayerNorm statistics and logits (whisper.cpp, mlx-whisper, OpenAI Whisper)".into()
            } else {
                "Metal default dtype remains F32".into()
            },
        }
    }

    fn default_cuda_dtype_selection(&self, request: DTypeSelectionRequest<'_>) -> DTypeSelection {
        match request.policy {
            DTypeSelectionPolicy::PreferF32 => {
                return DTypeSelection {
                    dtype: DType::F32,
                    reason: "model policy prefers CUDA F32".into(),
                };
            }
            DTypeSelectionPolicy::PreferF16 if self.capabilities.supports_f16 => {
                return DTypeSelection {
                    dtype: DType::F16,
                    reason: "model policy prefers CUDA F16".into(),
                };
            }
            DTypeSelectionPolicy::PreferBf16 if self.capabilities.supports_bf16 => {
                return DTypeSelection {
                    dtype: DType::BF16,
                    reason: "model policy prefers CUDA BF16 and device supports it".into(),
                };
            }
            _ => {}
        }

        if let Some(selection) = self.cuda_model_family_dtype_selection(request) {
            return selection;
        }

        if request.checkpoint_dtype == Some(DType::F32) {
            return DTypeSelection {
                dtype: DType::F32,
                reason: "checkpoint dtype is F32".into(),
            };
        }

        if request.checkpoint_dtype == Some(DType::BF16) && self.capabilities.supports_bf16 {
            return DTypeSelection {
                dtype: DType::BF16,
                reason: "checkpoint dtype is BF16 and CUDA device supports BF16".into(),
            };
        }

        if request.checkpoint_dtype == Some(DType::F16) && self.capabilities.supports_f16 {
            return DTypeSelection {
                dtype: DType::F16,
                reason: "checkpoint dtype is F16 and CUDA device supports F16".into(),
            };
        }

        if self.capabilities.supports_bf16 {
            DTypeSelection {
                dtype: DType::BF16,
                reason: "CUDA default uses BF16 on devices with BF16 support".into(),
            }
        } else if self.capabilities.supports_f16 {
            DTypeSelection {
                dtype: DType::F16,
                reason: "CUDA default falls back to F16 when BF16 is unavailable".into(),
            }
        } else {
            DTypeSelection {
                dtype: DType::F32,
                reason:
                    "CUDA default falls back to F32 because no half precision support was reported"
                        .into(),
            }
        }
    }

    fn cuda_model_family_dtype_selection(
        &self,
        request: DTypeSelectionRequest<'_>,
    ) -> Option<DTypeSelection> {
        match request.model_family? {
            ModelFamily::WhisperAsr => {
                if self.capabilities.supports_f16 {
                    Some(DTypeSelection {
                        dtype: DType::F16,
                        reason: "Whisper CUDA policy defaults to F16".into(),
                    })
                } else {
                    Some(DTypeSelection {
                        dtype: DType::F32,
                        reason: "Whisper CUDA policy falls back to F32 without F16 support".into(),
                    })
                }
            }
            ModelFamily::KokoroTts | ModelFamily::ParakeetAsr => Some(DTypeSelection {
                dtype: DType::F32,
                reason: "model family policy keeps CUDA default in F32".into(),
            }),
            ModelFamily::VoxtralTts => {
                if self.capabilities.supports_bf16 {
                    Some(DTypeSelection {
                        dtype: DType::BF16,
                        reason: "Voxtral TTS CUDA policy defaults dense LM/acoustic stages to BF16 when supported".into(),
                    })
                } else if self.capabilities.supports_f16 {
                    Some(DTypeSelection {
                        dtype: DType::F16,
                        reason:
                            "Voxtral TTS CUDA policy falls back to F16 when BF16 is unavailable"
                                .into(),
                    })
                } else {
                    Some(DTypeSelection {
                        dtype: DType::F32,
                        reason: "Voxtral TTS CUDA policy falls back to F32 without reported half precision support".into(),
                    })
                }
            }
            ModelFamily::VibeVoiceTts | ModelFamily::VibeVoiceAsr => {
                if self.capabilities.supports_bf16 {
                    Some(DTypeSelection {
                        dtype: DType::BF16,
                        reason: "VibeVoice CUDA policy defaults dense language and speech stages to BF16 when supported".into(),
                    })
                } else if self.capabilities.supports_f16 {
                    Some(DTypeSelection {
                        dtype: DType::F16,
                        reason: "VibeVoice CUDA policy falls back to F16 when BF16 is unavailable"
                            .into(),
                    })
                } else {
                    Some(DTypeSelection {
                        dtype: DType::F32,
                        reason: "VibeVoice CUDA policy falls back to F32 without reported half precision support".into(),
                    })
                }
            }
            ModelFamily::NemotronAsr
            | ModelFamily::GraniteSpeechAsr
            | ModelFamily::GraniteSpeech5Asr => {
                if self.capabilities.supports_bf16 {
                    Some(DTypeSelection {
                        dtype: DType::BF16,
                        reason: "dense ASR CUDA policy defaults native speech stages to BF16 when supported".into(),
                    })
                } else if self.capabilities.supports_f16 {
                    Some(DTypeSelection {
                        dtype: DType::F16,
                        reason: "dense ASR CUDA policy falls back to F16 when BF16 is unavailable"
                            .into(),
                    })
                } else {
                    Some(DTypeSelection {
                        dtype: DType::F32,
                        reason: "dense ASR CUDA policy falls back to F32 without reported half precision support".into(),
                    })
                }
            }
            _ => None,
        }
    }

    /// Get the optimal dtype for this device without any specific request
    pub fn optimal_dtype(&self) -> DType {
        self.select_dtype(None)
    }

    /// Check if this device supports memory pooling (Metal only)
    pub fn supports_memory_pool(&self) -> bool {
        self.kind.is_metal() && self.memory_pool.is_some()
    }

    /// Get memory pool statistics if available
    pub fn memory_pool_stats(
        &self,
    ) -> Option<crate::models::shared::memory::metal::MetalPoolStats> {
        self.memory_pool.as_ref().map(|pool| pool.stats())
    }

    /// Returns true if the device has unified memory architecture (Apple Silicon)
    pub fn has_unified_memory(&self) -> bool {
        self.capabilities.has_unified_memory
    }
}

pub struct DeviceSelector;

static METAL_PROBE_PANICKED: AtomicBool = AtomicBool::new(false);
static METAL_DEVICES: OnceLock<Mutex<HashMap<usize, Option<Device>>>> = OnceLock::new();

/// Whether this process may initialize Candle's Metal backend safely.
///
/// Candle 0.11 constructs a residency-set descriptor introduced in macOS 15.
/// Calling into that path on older releases panics before Candle can return an
/// error, so Izwi deliberately keeps those systems on the CPU backend.
pub fn metal_runtime_supported() -> bool {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        objc2::available!(macos = 15.0)
    }

    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        false
    }
}

fn panic_payload_to_string(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        return (*message).to_string();
    }
    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }
    "unknown panic payload".to_string()
}

fn probe_metal_device_for_runtime<F>(runtime_supported: bool, probe: F) -> Option<Device>
where
    F: FnOnce() -> candle_core::Result<Device>,
{
    if !runtime_supported || METAL_PROBE_PANICKED.load(Ordering::Relaxed) {
        return None;
    }

    match catch_unwind(AssertUnwindSafe(probe)) {
        Ok(Ok(device)) if device.is_metal() => Some(device),
        Ok(Ok(_)) | Ok(Err(_)) => None,
        Err(payload) => {
            METAL_PROBE_PANICKED.store(true, Ordering::Relaxed);
            warn!(
                "Metal probe panicked; disabling Metal for this process: {}",
                panic_payload_to_string(payload.as_ref())
            );
            None
        }
    }
}

/// Returns a Metal device only when the current runtime satisfies Izwi's
/// macOS 15+ Metal contract and Candle initializes it without error. Candle's
/// default-device registry is not safe to initialize concurrently, so devices
/// are constructed once per ordinal and the shared handle is cloned thereafter.
pub fn metal_device_if_available(ordinal: usize) -> Option<Device> {
    if !metal_runtime_supported() {
        return None;
    }
    #[cfg(all(test, feature = "metal"))]
    assert!(
        metal_test_lease::held() || super::DeviceScope::current().is_some(),
        "a test reached the shared Metal device without \
         `backends::metal_test_device_or_skip` (two threads driving one Metal \
         device corrupt readbacks and can crash in its residency set)"
    );
    shared_metal_device(ordinal)
}

fn shared_metal_device(ordinal: usize) -> Option<Device> {
    let devices = METAL_DEVICES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut devices = devices
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(device) = devices.get(&ordinal) {
        return device.clone();
    }

    let device = probe_metal_device_for_runtime(true, || Device::new_metal(ordinal));
    devices.insert(ordinal, device.clone());
    device
}

/// A Metal device with its own command queue, for tests whose fences must not
/// be completed by another test's drain on the shared device. Guarded like
/// [`metal_device_if_available`]: `Ok(None)` when the Metal runtime or this
/// ordinal is unavailable, so Candle's constructor is never entered there.
#[cfg(all(test, feature = "metal"))]
pub(crate) fn private_metal_device_for_tests(
    ordinal: usize,
) -> candle_core::Result<Option<Device>> {
    if !metal_runtime_supported() || shared_metal_device(ordinal).is_none() {
        return Ok(None);
    }
    // Construct under the registry lock: Candle's device constructor is not
    // safe to run concurrently.
    let _devices = METAL_DEVICES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    Device::new_metal(ordinal).map(Some)
}

/// The shared Metal device for a test's Metal leg, held exclusively, or
/// `None` after a skip note written past libtest's output capture.
/// `IZWI_REQUIRE_METAL_PARITY=1` turns the skip into a failure on lanes that
/// own a Metal GPU.
///
/// Production runs every device operation on the device's
/// [`DeviceThread`](super::DeviceThread), because Candle's `MetalDevice` must
/// not be driven from two threads: concurrent allocations add to and commit
/// its residency set outside any common lock (a SIGSEGV in
/// `-[IOGPUMetalResidencySet commit]`), and concurrent command buffers corrupt
/// readbacks (stray zero token ids, denormal garbage). Tests drive Metal on
/// their own threads instead, so every test that reaches the shared device
/// holds this lease for its whole Metal leg, and
/// [`metal_device_if_available`] refuses a test thread without one. The
/// lease is re-entrant on its thread.
#[cfg(all(test, feature = "metal"))]
pub(crate) fn metal_test_device_or_skip(
    test: &str,
) -> Option<(Device, metal_test_lease::MetalTestLease)> {
    use std::io::Write;

    let lease = metal_test_lease::MetalTestLease::acquire();
    let Some(device) = metal_device_if_available(0) else {
        assert!(
            !crate::env::flag_or("IZWI_REQUIRE_METAL_PARITY", false),
            "{test}: IZWI_REQUIRE_METAL_PARITY is set but no Metal device is available"
        );
        let _ = writeln!(std::io::stderr(), "{test}: skipped, no Metal device");
        return None;
    };
    Some((device, lease))
}

/// Whether `device` is the "Apple Paravirtual device" of a macOS VM
/// (GitHub's hosted runners among them). It reports an older family than its
/// host and rounds F16 several times coarser than Apple silicon, so tests
/// hold it to looser bounds.
#[cfg(test)]
pub(crate) fn metal_device_is_paravirtual(device: &Device) -> bool {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        use objc2_metal::MTLDevice;

        if let Device::Metal(metal) = device {
            return metal
                .device()
                .as_ref()
                .name()
                .to_string()
                .contains("Paravirtual");
        }
    }
    let _ = device;
    false
}

#[cfg(all(test, feature = "metal"))]
pub(crate) mod metal_test_lease {
    use std::cell::Cell;
    use std::sync::{Mutex, MutexGuard, PoisonError};

    static SHARED_DEVICE: Mutex<()> = Mutex::new(());

    thread_local! {
        static HELD: Cell<bool> = const { Cell::new(false) };
    }

    /// Whether the calling thread holds the shared Metal device's lease.
    pub(crate) fn held() -> bool {
        HELD.with(Cell::get)
    }

    /// Exclusive use of the shared Metal device by the thread holding it.
    pub(crate) struct MetalTestLease {
        /// `None` for a nested lease on a thread that already holds one.
        serial: Option<MutexGuard<'static, ()>>,
    }

    impl MetalTestLease {
        pub(super) fn acquire() -> Self {
            if held() {
                return Self { serial: None };
            }
            let serial = SHARED_DEVICE
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            HELD.with(|held| held.set(true));
            Self {
                serial: Some(serial),
            }
        }
    }

    impl Drop for MetalTestLease {
        fn drop(&mut self) {
            if self.serial.is_some() {
                HELD.with(|held| held.set(false));
            }
        }
    }
}

impl DeviceSelector {
    fn try_metal_at(ordinal: usize) -> Option<DeviceProfile> {
        if let Some(device) = metal_device_if_available(ordinal) {
            // Initialize memory pool for Metal
            let memory_pool = metal_pool_for_device(&device);

            if memory_pool.is_some() {
                info!("Metal memory pool initialized");
            }

            Some(DeviceProfile {
                device,
                kind: DeviceKind::Metal,
                capabilities: DeviceCapabilities {
                    prefers_f32: true,    // Metal on Apple Silicon prefers F32
                    supports_bf16: false, // Metal doesn't have good BF16 support
                    supports_f16: true,   // Used by model-specific Metal policies.
                    supports_int8_tensor_cores: false,
                    has_unified_memory: true, // Apple Silicon has unified memory
                    recommended_batch_size: 4, // Conservative for unified memory
                    available_memory_bytes: None, // Could be detected via system APIs
                    cuda_total_memory_bytes: None,
                    cuda_device_ordinal: None,
                    cuda_compute_capability: None,
                    cuda_device_name: None,
                },
                memory_pool,
            })
        } else {
            None
        }
    }

    fn try_metal() -> Option<DeviceProfile> {
        Self::try_metal_at(0)
    }

    fn try_cuda_at(ordinal: usize) -> Option<DeviceProfile> {
        let device = std::panic::catch_unwind(|| Device::cuda_if_available(ordinal))
            .ok()?
            .ok()?;
        if device.is_cuda() {
            let cuda_capabilities = Self::detect_cuda_capabilities(ordinal);
            let supports_bf16 = cuda_capabilities
                .compute_capability
                .is_some_and(cuda_compute_capability_supports_bf16);
            let supports_f16 = cuda_capabilities
                .compute_capability
                .is_none_or(cuda_compute_capability_supports_f16);
            let supports_int8_tensor_cores = cuda_capabilities
                .compute_capability
                .is_some_and(cuda_compute_capability_supports_int8_tensor_cores);

            Some(DeviceProfile {
                device,
                kind: DeviceKind::Cuda,
                capabilities: DeviceCapabilities {
                    prefers_f32: false,
                    supports_bf16,
                    supports_f16,
                    supports_int8_tensor_cores,
                    has_unified_memory: false,
                    recommended_batch_size: 8, // CUDA can handle larger batches
                    available_memory_bytes: cuda_capabilities.free_memory_bytes,
                    cuda_total_memory_bytes: cuda_capabilities.total_memory_bytes,
                    cuda_device_ordinal: Some(ordinal),
                    cuda_compute_capability: cuda_capabilities.compute_capability,
                    cuda_device_name: cuda_capabilities.device_name,
                },
                memory_pool: None, // CUDA uses its own memory management
            })
        } else {
            None
        }
    }

    fn try_cuda() -> Option<DeviceProfile> {
        Self::try_cuda_at(configured_cuda_ordinal())
    }

    fn detect_cuda_capabilities(ordinal: usize) -> CudaProbe {
        detect_cuda_capabilities(ordinal)
    }

    pub fn detect() -> Result<DeviceProfile> {
        if cfg!(target_os = "macos") {
            if let Some(profile) = Self::try_metal() {
                info!(
                    "Using Metal device for inference (unified memory: {})",
                    profile.has_unified_memory()
                );
                return Ok(profile);
            }
        } else if let Some(profile) = Self::try_cuda() {
            info!("Using CUDA device for inference");
            return Ok(profile);
        }

        if let Some(profile) = Self::try_metal() {
            info!("Using Metal device for inference");
            return Ok(profile);
        }

        info!("Falling back to CPU for inference");
        Ok(DeviceProfile::cpu())
    }

    pub fn detect_for_preference(preference: BackendPreference) -> Result<DeviceProfile> {
        match preference {
            BackendPreference::Auto => Self::detect(),
            BackendPreference::Cpu => Ok(DeviceProfile::cpu()),
            BackendPreference::Metal => {
                if let Some(profile) = Self::try_metal() {
                    Ok(profile)
                } else {
                    Self::detect()
                }
            }
            BackendPreference::Cuda => {
                if cfg!(target_os = "macos") {
                    return Self::detect();
                }
                if let Some(profile) = Self::try_cuda() {
                    Ok(profile)
                } else {
                    Self::detect()
                }
            }
        }
    }

    /// Select exactly the supervisor-assigned execution device.
    ///
    /// This path never consults backend preference environment variables and
    /// never falls back to another backend or accelerator ordinal.
    pub fn select_assigned(assignment: &RuntimeDeviceAssignment) -> Result<DeviceProfile> {
        assignment
            .validate()
            .map_err(|message| Error::ConfigError(message.to_string()))?;

        match assignment {
            RuntimeDeviceAssignment::Cpu => Ok(DeviceProfile::cpu()),
            RuntimeDeviceAssignment::Metal {
                process_local_device_index,
                expected_device_id,
            } => {
                let profile = Self::try_metal_at(*process_local_device_index).ok_or_else(|| {
                    Error::ConfigError(format!(
                        "assigned Metal device {} is unavailable",
                        process_local_device_index
                    ))
                })?;
                let observed = metal_device_identity(&profile.device).ok_or_else(|| {
                    Error::ConfigError(
                        "selected Metal device did not expose a stable identity".into(),
                    )
                })?;
                if observed != *expected_device_id {
                    return Err(Error::ConfigError(format!(
                        "assigned Metal device identity mismatch: expected {expected_device_id}, observed {observed}"
                    )));
                }
                Ok(profile)
            }
            RuntimeDeviceAssignment::Cuda {
                process_local_device_index,
                expected_device_uuid,
            } => {
                let profile = Self::try_cuda_at(*process_local_device_index).ok_or_else(|| {
                    Error::ConfigError(format!(
                        "assigned CUDA device {} is unavailable",
                        process_local_device_index
                    ))
                })?;
                let observed = cuda_device_uuid(*process_local_device_index).ok_or_else(|| {
                    Error::ConfigError("selected CUDA device did not expose a stable UUID".into())
                })?;
                if !observed.eq_ignore_ascii_case(expected_device_uuid) {
                    return Err(Error::ConfigError(format!(
                        "assigned CUDA device UUID mismatch: expected {expected_device_uuid}, observed {observed}"
                    )));
                }
                Ok(profile)
            }
        }
    }

    pub fn detect_with_preference(preference: Option<&str>) -> Result<DeviceProfile> {
        preference
            .and_then(BackendPreference::parse)
            .map(Self::detect_for_preference)
            .unwrap_or_else(Self::detect)
    }
}

#[cfg(feature = "metal")]
fn metal_device_identity(device: &Device) -> Option<String> {
    match device {
        Device::Metal(device) => Some(format!("metal:{}", device.registry_id())),
        Device::Cpu | Device::Cuda(_) => None,
    }
}

#[cfg(not(feature = "metal"))]
fn metal_device_identity(_device: &Device) -> Option<String> {
    None
}

#[cfg(feature = "cuda")]
fn cuda_device_uuid(ordinal: usize) -> Option<String> {
    use candle_core::cuda_backend::cudarc::driver::CudaContext;

    let uuid = CudaContext::new(ordinal).ok()?.uuid().ok()?;
    let bytes = uuid.bytes.map(|byte| byte as u8);
    Some(format_cuda_uuid(bytes))
}

fn format_cuda_uuid(bytes: [u8; 16]) -> String {
    format!(
        "GPU-{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]
    )
}

#[cfg(not(feature = "cuda"))]
fn cuda_device_uuid(_ordinal: usize) -> Option<String> {
    None
}

#[derive(Debug, Clone, Default)]
struct CudaProbe {
    compute_capability: Option<(u32, u32)>,
    free_memory_bytes: Option<usize>,
    total_memory_bytes: Option<usize>,
    device_name: Option<String>,
}

pub fn parse_dtype_name(raw: &str) -> Option<DType> {
    let normalized = raw.trim().to_ascii_lowercase();
    let normalized = normalized
        .strip_prefix("torch.")
        .unwrap_or(normalized.as_str());
    match normalized {
        "bfloat16" | "bf16" => Some(DType::BF16),
        "float16" | "f16" | "fp16" | "half" => Some(DType::F16),
        "float32" | "float" | "f32" | "fp32" => Some(DType::F32),
        _ => None,
    }
}

/// Families whose loader passes the checkpoint's config dtype (not a user
/// override) through the requested-dtype path. On Metal that hint keeps the
/// family default. Granite Speech resolves its own `IZWI_GRANITE_SPEECH_DTYPE`
/// before it reaches the resolver, so only its config hint arrives here;
/// Granite Speech 5.0 passes only its config hint.
const METAL_REQUEST_IS_CONFIG_HINT: &[ModelFamily] = &[
    ModelFamily::GraniteSpeechAsr,
    ModelFamily::GraniteSpeech5Asr,
];

/// The global Metal dtype override variable.
pub const METAL_DTYPE_ENV: &str = "IZWI_METAL_DTYPE";

/// `IZWI_METAL_DTYPE`: `auto` (or unset) keeps each family's Metal policy;
/// `f32`, `f16` or `bf16` replaces the family default wherever the Metal
/// resolver is consulted. `f32` is the escape hatch from every lower-precision
/// Metal policy, weight-only F16 included. A per-family override such as
/// `IZWI_WHISPER_DTYPE` is more specific and wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MetalDTypeOverride {
    #[default]
    Auto,
    F32,
    F16,
    Bf16,
}

impl MetalDTypeOverride {
    pub fn parse(raw: &str) -> Option<Self> {
        let raw = raw.trim();
        if raw.is_empty() || raw.eq_ignore_ascii_case("auto") {
            return Some(Self::Auto);
        }
        match parse_dtype_name(raw)? {
            DType::F32 => Some(Self::F32),
            DType::F16 => Some(Self::F16),
            DType::BF16 => Some(Self::Bf16),
            _ => None,
        }
    }

    /// Reads [`METAL_DTYPE_ENV`]. An unrecognised value is ignored (`Auto`)
    /// with a warning, once per process.
    pub fn from_env() -> Self {
        let Ok(raw) = std::env::var(METAL_DTYPE_ENV) else {
            return Self::Auto;
        };
        Self::parse(&raw).unwrap_or_else(|| {
            static WARNED: AtomicBool = AtomicBool::new(false);
            if !WARNED.swap(true, Ordering::Relaxed) {
                warn!(
                    "Ignoring invalid {METAL_DTYPE_ENV}={raw:?}; expected f32, f16, bf16 or auto"
                );
            }
            Self::Auto
        })
    }

    pub fn dtype(self) -> Option<DType> {
        match self {
            Self::Auto => None,
            Self::F32 => Some(DType::F32),
            Self::F16 => Some(DType::F16),
            Self::Bf16 => Some(DType::BF16),
        }
    }
}

/// [`DeviceProfile::resolve_metal_weight_only_f16`] with the global override
/// passed in. The family override wins, then `IZWI_METAL_DTYPE`; the Metal
/// default is on.
fn metal_weight_only_f16_selection(
    is_metal: bool,
    family_override: Option<(&str, &str)>,
    global: MetalDTypeOverride,
) -> DTypeSelection {
    let off = |reason: Cow<'static, str>| DTypeSelection {
        dtype: DType::F32,
        reason,
    };
    let on = |reason: Cow<'static, str>| DTypeSelection {
        dtype: DType::F16,
        reason,
    };
    if !is_metal {
        return off("weight-only F16 is a Metal policy; weights stay F32".into());
    }
    let family_override = family_override
        .map(|(name, raw)| (name, raw.trim()))
        .filter(|(_, raw)| !raw.is_empty());
    let family_note = match family_override {
        Some((name, raw)) => match parse_dtype_name(raw) {
            Some(DType::F32) => {
                return off(format!("{name}={raw} keeps every weight in F32").into());
            }
            Some(DType::F16) => {
                return on(
                    format!("{name}={raw}: F16 projection weights with F32 activations").into(),
                );
            }
            _ => format!("; {name}={raw} is not a Metal weight precision and is ignored"),
        },
        None => String::new(),
    };
    match global {
        MetalDTypeOverride::F32 => off(format!(
            "IZWI_METAL_DTYPE=f32 keeps every weight in F32{family_note}"
        )
        .into()),
        MetalDTypeOverride::Auto => on(format!(
            "Metal default: F16 projection weights with F32 activations and accumulation{family_note}"
        )
        .into()),
        MetalDTypeOverride::F16 | MetalDTypeOverride::Bf16 => on(format!(
            "IZWI_METAL_DTYPE={}: F16 projection weights with F32 activations{family_note}",
            if global == MetalDTypeOverride::F16 { "f16" } else { "bf16" }
        )
        .into()),
    }
}

/// The Metal precision tier for a decoder LM whose checkpoint is BF16
/// (report 13, plan Phase 6 P6.4):
/// - Apple9+ (M3/M4/M5, `fast_bf16`): BF16, the checkpoint's own precision,
///   which llama.cpp measured at about F16 throughput there. This tier is
///   selected by GPU capability only; it has not been validated numerically
///   on an Apple9+ GPU yet (plan section 9, open question 1) and needs that
///   WER-parity run before it is trusted.
/// - Apple7/8 (M1/M2): F16, because BF16 ran at 0.81x of F16 there
///   (llama.cpp #29770). F16 narrows BF16's range to 65504, so each family
///   gates it behind a load-time range probe with an F32 fallback.
///
/// The BF16 tier also needs `bf16_paged_attention`: every decoder LM here
/// attends through izwi's Metal paged-attention kernels, which take F32 and
/// F16 only ([`METAL_PAGED_ATTENTION_SUPPORTS_BF16`]). Until they take BF16
/// (or K/V is stored in F16 under BF16 activations, as llama.cpp does), an
/// Apple9+ GPU takes the F16 tier and the reason says why.
pub fn metal_bf16_checkpoint_lm_tier(
    fast_bf16: bool,
    bf16_paged_attention: bool,
) -> DTypeSelection {
    if fast_bf16 && bf16_paged_attention {
        DTypeSelection {
            dtype: DType::BF16,
            reason: "Metal BF16-checkpoint LM tier: BF16 on an Apple9+ GPU (capability-selected; awaiting M3+ validation)".into(),
        }
    } else if fast_bf16 {
        DTypeSelection {
            dtype: DType::F16,
            reason: "Metal BF16-checkpoint LM tier: F16 on an Apple9+ GPU, behind a load-time range probe; the BF16 tier waits for BF16 Metal paged attention".into(),
        }
    } else {
        DTypeSelection {
            dtype: DType::F16,
            reason: "Metal BF16-checkpoint LM tier: F16 on an Apple7/8 GPU, where BF16 runs slower, behind a load-time range probe".into(),
        }
    }
}

/// Whether izwi's Metal paged-attention kernels (`kernels/metal.rs`) take
/// BF16 queries and K/V. They are compiled for F32 and F16 only.
pub const METAL_PAGED_ATTENTION_SUPPORTS_BF16: bool = false;

/// An Apple GPU family (`MTLGPUFamilyApple<N>`): M1 is Apple7, M2 Apple8,
/// M3 and M4 Apple9, M5 Apple10.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct AppleGpuFamily(pub u8);

impl AppleGpuFamily {
    /// The first family whose BF16 throughput matches F16. llama.cpp measured
    /// BF16 at 0.81x on M2 Ultra and about parity on M3+.
    pub const FAST_BF16: Self = Self(9);

    pub fn supports_fast_bf16(self) -> bool {
        self >= Self::FAST_BF16
    }
}

/// The newest Apple GPU family `device` supports, or `None` off Metal.
/// Candle's `supports_bf16()` is true on every Metal device, so the
/// precision policy reads the family itself (`MTLDevice supportsFamily:`).
pub fn metal_apple_gpu_family(device: &Device) -> Option<AppleGpuFamily> {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        use objc2_metal::{MTLDevice, MTLGPUFamily};

        let Device::Metal(metal) = device else {
            return None;
        };
        let raw = metal.device().as_ref();
        [
            (MTLGPUFamily::Apple10, 10),
            (MTLGPUFamily::Apple9, 9),
            (MTLGPUFamily::Apple8, 8),
            (MTLGPUFamily::Apple7, 7),
            (MTLGPUFamily::Apple6, 6),
            (MTLGPUFamily::Apple5, 5),
            (MTLGPUFamily::Apple4, 4),
            (MTLGPUFamily::Apple3, 3),
            (MTLGPUFamily::Apple2, 2),
            (MTLGPUFamily::Apple1, 1),
        ]
        .into_iter()
        .find(|(family, _)| raw.supportsFamily(*family))
        .map(|(_, generation)| AppleGpuFamily(generation))
    }

    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        let _ = device;
        None
    }
}

/// Whether `device` is an Apple9+ GPU (M3/M4/M5), the Metal BF16 tier.
pub fn metal_supports_fast_bf16(device: &Device) -> bool {
    metal_apple_gpu_family(device).is_some_and(AppleGpuFamily::supports_fast_bf16)
}

fn parse_cuda_ordinal(raw: Option<&str>) -> Option<usize> {
    raw.map(str::trim)
        .filter(|raw| !raw.is_empty())
        .and_then(|raw| raw.parse::<usize>().ok())
}

fn configured_cuda_ordinal() -> usize {
    let raw = std::env::var("IZWI_CUDA_DEVICE_ORDINAL").ok();
    match raw.as_deref() {
        None => 0,
        Some(value) => match parse_cuda_ordinal(Some(value)) {
            Some(ordinal) => ordinal,
            None => {
                warn!("Ignoring invalid IZWI_CUDA_DEVICE_ORDINAL={value:?}; using CUDA device 0");
                0
            }
        },
    }
}

pub fn cuda_compute_capability_supports_bf16((major, _minor): (u32, u32)) -> bool {
    major >= 8
}

pub fn cuda_compute_capability_supports_f16((major, minor): (u32, u32)) -> bool {
    major > 5 || (major == 5 && minor >= 3)
}

pub fn cuda_compute_capability_supports_int8_tensor_cores((major, minor): (u32, u32)) -> bool {
    major > 7 || (major == 7 && minor >= 5)
}

#[cfg(feature = "cuda")]
fn detect_cuda_capabilities(ordinal: usize) -> CudaProbe {
    use candle_core::cuda_backend::cudarc::driver::{result, CudaContext};

    match CudaContext::new(ordinal) {
        Ok(context) => {
            let compute_capability = context
                .compute_capability()
                .ok()
                .map(|(major, minor)| (major.max(0) as u32, minor.max(0) as u32));
            let (free_memory_bytes, total_memory_bytes) = result::mem_get_info()
                .map(|(free, total)| (Some(free), Some(total)))
                .unwrap_or_else(|_| {
                    let total = unsafe { result::device::total_mem(context.cu_device()).ok() };
                    (None, total)
                });
            let device_name = context.name().ok().map(|name| name.trim().to_string());

            CudaProbe {
                compute_capability,
                free_memory_bytes,
                total_memory_bytes,
                device_name: device_name.filter(|name| !name.is_empty()),
            }
        }
        Err(err) => {
            warn!("Unable to query CUDA device properties: {err}");
            CudaProbe::default()
        }
    }
}

#[cfg(not(feature = "cuda"))]
fn detect_cuda_capabilities(_ordinal: usize) -> CudaProbe {
    CudaProbe::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tensors outlive the thread that made them: results and session state
    /// cross tokio threads that may never have bound the CUDA context. Candle
    /// runs on the per-thread default stream, and cudarc's `CudaSlice` drop
    /// binds the context only through its event waits. With event tracking
    /// off it frees on the dropping thread's default stream with no current
    /// context, which fails with `CUDA_ERROR_INVALID_CONTEXT`, leaks the
    /// allocation and latches the error on the context for the next caller.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_tensors_drop_cleanly_on_threads_without_the_context() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        let context = device
            .as_cuda_device()
            .unwrap()
            .cuda_stream()
            .context()
            .clone();
        assert!(
            context.is_event_tracking(),
            "the selected CUDA device must keep cudarc's event tracking"
        );
        context
            .check_err()
            .expect("no error latched before the drop");
        let tensor = candle_core::Tensor::zeros(1024, candle_core::DType::F32, &device).unwrap();
        std::thread::spawn(move || drop(tensor)).join().unwrap();
        context
            .check_err()
            .expect("dropping a tensor on a fresh thread must not latch a CUDA error");
    }

    #[test]
    fn assigned_cpu_selects_cpu_without_accelerator_discovery() {
        let profile =
            DeviceSelector::select_assigned(&RuntimeDeviceAssignment::Cpu).expect("assigned CPU");
        assert_eq!(profile.kind, DeviceKind::Cpu);
        assert!(profile.device.is_cpu());
    }

    #[cfg(not(feature = "metal"))]
    #[test]
    fn assigned_metal_fails_closed_when_not_compiled() {
        let error = DeviceSelector::select_assigned(&RuntimeDeviceAssignment::Metal {
            process_local_device_index: 0,
            expected_device_id: "metal:1".into(),
        })
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("assigned Metal device 0 is unavailable"));
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn assigned_cuda_fails_closed_when_not_compiled() {
        let error = DeviceSelector::select_assigned(&RuntimeDeviceAssignment::Cuda {
            process_local_device_index: 0,
            expected_device_uuid: "GPU-00000000-0000-0000-0000-000000000000".into(),
        })
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("assigned CUDA device 0 is unavailable"));
    }

    #[test]
    fn cuda_uuid_uses_nvidia_canonical_shape() {
        assert_eq!(
            format_cuda_uuid([
                0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd,
                0xee, 0xff,
            ]),
            "GPU-00112233-4455-6677-8899-aabbccddeeff"
        );
    }

    #[test]
    fn unsupported_metal_runtime_never_invokes_candle_probe() {
        let invoked = AtomicBool::new(false);
        let device = probe_metal_device_for_runtime(false, || {
            invoked.store(true, Ordering::Relaxed);
            Ok(Device::Cpu)
        });

        assert!(device.is_none());
        assert!(!invoked.load(Ordering::Relaxed));
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn pre_macos15_auto_and_explicit_metal_preferences_fall_back_to_cpu() {
        if metal_runtime_supported() {
            return;
        }

        for preference in [BackendPreference::Auto, BackendPreference::Metal] {
            let profile = DeviceSelector::detect_for_preference(preference).unwrap();
            assert_eq!(profile.kind, DeviceKind::Cpu);
            assert!(profile.device.is_cpu());
        }
    }

    #[test]
    fn cuda_ordinal_parser_is_deterministic() {
        assert_eq!(parse_cuda_ordinal(None), None);
        assert_eq!(parse_cuda_ordinal(Some("")), None);
        assert_eq!(parse_cuda_ordinal(Some(" 2 ")), Some(2));
        assert_eq!(parse_cuda_ordinal(Some("-1")), None);
        assert_eq!(parse_cuda_ordinal(Some("gpu0")), None);
    }

    #[test]
    fn test_detect_with_cpu_preference_returns_cpu() {
        let profile = DeviceSelector::detect_with_preference(Some("cpu")).unwrap();
        assert_eq!(profile.kind, DeviceKind::Cpu);
        assert!(profile.device.is_cpu());
        assert!(!profile.has_unified_memory());
    }

    #[test]
    fn test_detect_for_cpu_preference_returns_cpu() {
        let profile = DeviceSelector::detect_for_preference(BackendPreference::Cpu).unwrap();
        assert_eq!(profile.kind, DeviceKind::Cpu);
        assert!(profile.device.is_cpu());
    }

    #[test]
    fn test_detect_kind_matches_device() {
        #[cfg(feature = "metal")]
        let _lease = metal_test_device_or_skip("test_detect_kind_matches_device");
        let profile = DeviceSelector::detect().unwrap();
        match profile.kind {
            DeviceKind::Cpu => assert!(profile.device.is_cpu()),
            DeviceKind::Metal => {
                assert!(profile.device.is_metal());
                assert!(profile.has_unified_memory());
                assert!(profile.capabilities.prefers_f32);
            }
            DeviceKind::Cuda => assert!(profile.device.is_cuda()),
        }
    }

    #[test]
    fn test_device_kind_maps_to_backend_kind() {
        assert_eq!(BackendKind::from(DeviceKind::Cpu), BackendKind::Cpu);
        assert_eq!(BackendKind::from(DeviceKind::Metal), BackendKind::Metal);
        assert_eq!(BackendKind::from(DeviceKind::Cuda), BackendKind::Cuda);
    }

    #[test]
    fn test_metal_prefers_f32() {
        // Test that Metal devices prefer F32
        let metal_profile = DeviceProfile {
            device: Device::Cpu, // Use CPU for testing
            kind: DeviceKind::Metal,
            capabilities: DeviceCapabilities {
                prefers_f32: true,
                ..Default::default()
            },
            memory_pool: None,
        };

        // Default should be F32 for Metal
        assert_eq!(metal_profile.select_dtype(None), DType::F32);

        // F32 request should give F32
        assert_eq!(metal_profile.select_dtype(Some("f32")), DType::F32);

        // A device without F16 support refuses an F16 override; the lenient
        // selector falls back to the default.
        assert_eq!(
            metal_profile
                .try_select_dtype(Some("f16"))
                .unwrap_err()
                .requested,
            "f16"
        );
        assert_eq!(metal_profile.select_dtype(Some("f16")), DType::F32);
    }

    fn metal_test_profile() -> DeviceProfile {
        DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Metal,
            capabilities: DeviceCapabilities {
                prefers_f32: true,
                supports_f16: true,
                ..Default::default()
            },
            memory_pool: None,
        }
    }

    #[test]
    fn metal_honours_explicit_lower_precision_requests() {
        let metal = metal_test_profile();
        assert_eq!(metal.select_dtype(Some("f16")), DType::F16);
        assert_eq!(metal.select_dtype(Some("bf16")), DType::BF16);
        assert_eq!(metal.select_dtype(Some("f32")), DType::F32);
        for family in [
            ModelFamily::WhisperAsr,
            ModelFamily::VibeVoiceAsr,
            ModelFamily::Qwen3Tts,
        ] {
            assert_eq!(
                metal.select_model_dtype(family, Some("float16")),
                DType::F16
            );
            assert_eq!(
                metal
                    .select_model_dtype_checked(family, Some("f32"), "test")
                    .unwrap(),
                DType::F32
            );
        }
        // A test profile on the CPU device has no Apple family, so the reason
        // warns that BF16 is the slow tier.
        let bf16 = metal
            .try_resolve_dtype(DTypeSelectionRequest::new(Some("bf16")))
            .unwrap();
        assert!(bf16.reason.contains("before Apple9"), "{}", bf16.reason);
    }

    #[test]
    fn metal_config_hints_keep_the_family_default() {
        let metal = metal_test_profile();
        assert_eq!(
            metal.select_model_dtype(ModelFamily::GraniteSpeechAsr, Some("torch.bfloat16")),
            DType::F32
        );
    }

    #[test]
    fn metal_global_override_replaces_family_defaults_only() {
        let metal = metal_test_profile();
        let request = |family| DTypeSelectionRequest::new(None).with_model_family(family);
        let with = |family, global| {
            metal
                .metal_default_dtype_selection_with(request(family), global)
                .dtype
        };

        // `auto` keeps every family default unchanged.
        assert_eq!(
            with(ModelFamily::ParakeetAsr, MetalDTypeOverride::Auto),
            DType::F32
        );
        assert_eq!(
            with(ModelFamily::NemotronAsr, MetalDTypeOverride::Auto),
            DType::F32
        );
        assert_eq!(
            with(ModelFamily::VibeVoiceTts, MetalDTypeOverride::Auto),
            DType::F16
        );
        assert_eq!(
            metal
                .metal_default_dtype_selection_with(
                    request(ModelFamily::Voxtral).with_checkpoint_dtype(DType::BF16),
                    MetalDTypeOverride::Auto
                )
                .dtype,
            DType::F16
        );

        // A named dtype replaces them.
        assert_eq!(
            with(ModelFamily::VibeVoiceTts, MetalDTypeOverride::F32),
            DType::F32
        );
        assert_eq!(
            with(ModelFamily::WhisperAsr, MetalDTypeOverride::F16),
            DType::F16
        );
        assert_eq!(
            with(ModelFamily::WhisperAsr, MetalDTypeOverride::Bf16),
            DType::BF16
        );
    }

    #[test]
    fn metal_dtype_override_parses_names_and_auto() {
        assert_eq!(
            MetalDTypeOverride::parse(""),
            Some(MetalDTypeOverride::Auto)
        );
        assert_eq!(
            MetalDTypeOverride::parse(" AUTO "),
            Some(MetalDTypeOverride::Auto)
        );
        assert_eq!(
            MetalDTypeOverride::parse("fp32"),
            Some(MetalDTypeOverride::F32)
        );
        assert_eq!(
            MetalDTypeOverride::parse("half"),
            Some(MetalDTypeOverride::F16)
        );
        assert_eq!(
            MetalDTypeOverride::parse("bfloat16"),
            Some(MetalDTypeOverride::Bf16)
        );
        assert_eq!(MetalDTypeOverride::parse("int8"), None);
        assert_eq!(MetalDTypeOverride::Auto.dtype(), None);
        assert_eq!(MetalDTypeOverride::F16.dtype(), Some(DType::F16));
    }

    #[test]
    fn metal_weight_only_f16_defaults_on_and_honours_f32_escape_hatches() {
        let select = metal_weight_only_f16_selection;
        let env = Some(("IZWI_PARAKEET_ASR_DTYPE", "f32"));

        assert_eq!(
            select(false, None, MetalDTypeOverride::Auto).dtype,
            DType::F32
        );
        assert_eq!(
            select(false, Some(("X", "f16")), MetalDTypeOverride::F16).dtype,
            DType::F32
        );

        assert_eq!(
            select(true, None, MetalDTypeOverride::Auto).dtype,
            DType::F16
        );
        assert_eq!(
            select(true, None, MetalDTypeOverride::F16).dtype,
            DType::F16
        );
        assert_eq!(
            select(true, None, MetalDTypeOverride::Bf16).dtype,
            DType::F16
        );
        let global_f32 = select(true, None, MetalDTypeOverride::F32);
        assert_eq!(global_f32.dtype, DType::F32);
        assert!(global_f32.reason.contains("IZWI_METAL_DTYPE=f32"));

        // The family override is more specific than the global one.
        let family_f32 = select(true, env, MetalDTypeOverride::Auto);
        assert_eq!(family_f32.dtype, DType::F32);
        assert!(family_f32.reason.contains("IZWI_PARAKEET_ASR_DTYPE=f32"));
        assert_eq!(
            select(true, Some(("X", "fp16")), MetalDTypeOverride::F32).dtype,
            DType::F16
        );
        // A value that is not a Metal weight precision is ignored, visibly.
        let ignored = select(true, Some(("X", "bf16")), MetalDTypeOverride::Auto);
        assert_eq!(ignored.dtype, DType::F16);
        assert!(ignored.reason.contains("ignored"), "{}", ignored.reason);
        assert_eq!(
            select(true, Some(("X", " ")), MetalDTypeOverride::F32).dtype,
            DType::F32
        );
    }

    #[test]
    fn bf16_checkpoint_lm_tier_is_bf16_only_on_apple9_and_newer() {
        let fast = metal_bf16_checkpoint_lm_tier(true, true);
        assert_eq!(fast.dtype, DType::BF16);
        assert!(fast.reason.contains("Apple9+"), "{}", fast.reason);
        // Without BF16 paged attention an Apple9+ GPU takes the F16 tier.
        let held = metal_bf16_checkpoint_lm_tier(true, false);
        assert_eq!(held.dtype, DType::F16);
        assert!(
            held.reason.contains("BF16 Metal paged attention"),
            "{}",
            held.reason
        );
        for bf16_attention in [false, true] {
            let slow = metal_bf16_checkpoint_lm_tier(false, bf16_attention);
            assert_eq!(slow.dtype, DType::F16);
            assert!(slow.reason.contains("range probe"), "{}", slow.reason);
        }
        // A profile that cannot read an Apple9+ family takes the F16 tier.
        assert_eq!(
            metal_test_profile().metal_bf16_checkpoint_lm_tier().dtype,
            DType::F16
        );
        for family in 7..=10 {
            let family = AppleGpuFamily(family);
            assert_eq!(
                metal_bf16_checkpoint_lm_tier(family.supports_fast_bf16(), true).dtype,
                if family >= AppleGpuFamily::FAST_BF16 {
                    DType::BF16
                } else {
                    DType::F16
                }
            );
        }
    }

    #[test]
    fn apple_gpu_family_bf16_tier_starts_at_apple9() {
        assert!(!AppleGpuFamily(7).supports_fast_bf16());
        assert!(!AppleGpuFamily(8).supports_fast_bf16());
        assert!(AppleGpuFamily(9).supports_fast_bf16());
        assert!(AppleGpuFamily(10).supports_fast_bf16());
        assert_eq!(metal_apple_gpu_family(&Device::Cpu), None);
        assert!(!metal_supports_fast_bf16(&Device::Cpu));
        assert!(!DeviceProfile::cpu().metal_supports_fast_bf16());
    }

    /// Every Apple Silicon Mac is Apple7 or newer; a Metal device that reads
    /// no family would leave the BF16 tier silently off.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_device_reports_an_apple_silicon_family() {
        let Some((device, _lease)) =
            metal_test_device_or_skip("metal_device_reports_an_apple_silicon_family")
        else {
            return;
        };
        let family = metal_apple_gpu_family(&device).expect("Apple GPU family");
        let Device::Metal(metal) = &device else {
            unreachable!("metal_test_device_or_skip returns a Metal device");
        };
        let name = {
            use objc2_metal::MTLDevice;
            metal.device().as_ref().name().to_string()
        };
        eprintln!("Metal device family: Apple{} ({name})", family.0);
        if !metal_device_is_paravirtual(&device) {
            assert!(family >= AppleGpuFamily(7), "{family:?} on {name}");
        }
        assert_eq!(
            metal_supports_fast_bf16(&device),
            family >= AppleGpuFamily::FAST_BF16
        );
    }

    #[test]
    fn metal_voxtral_defaults_to_f16_for_realtime_latency() {
        let metal_profile = DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Metal,
            capabilities: DeviceCapabilities {
                prefers_f32: true,
                supports_f16: true,
                ..Default::default()
            },
            memory_pool: None,
        };

        assert_eq!(
            metal_profile
                .select_model_dtype_with_checkpoint(ModelFamily::Voxtral, Some(DType::BF16)),
            DType::F16
        );
        let voxtral = metal_profile.resolve_dtype(
            DTypeSelectionRequest::new(None)
                .with_model_family(ModelFamily::Voxtral)
                .with_checkpoint_dtype(DType::BF16),
        );
        assert!(voxtral.reason.contains("LM tier"), "{}", voxtral.reason);
        assert_eq!(
            metal_profile
                .select_model_dtype_with_checkpoint(ModelFamily::Voxtral, Some(DType::F32)),
            DType::F32
        );
        assert_eq!(
            metal_profile.select_model_dtype(ModelFamily::Qwen3Tts, None),
            DType::F32
        );
        // Voxtral TTS's LM and acoustic stages take the same tier (behind
        // the model's load-time self-check).
        let tts = metal_profile.resolve_dtype(
            DTypeSelectionRequest::new(None).with_model_family(ModelFamily::VoxtralTts),
        );
        assert_eq!(tts.dtype, DType::F16);
        assert!(tts.reason.contains("LM tier"), "{}", tts.reason);
    }

    #[test]
    fn test_cuda_dtype_selection() {
        let cuda_profile = DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Cuda,
            capabilities: DeviceCapabilities {
                supports_bf16: true,
                supports_f16: true,
                ..Default::default()
            },
            memory_pool: None,
        };

        // CUDA should prefer BF16 by default
        assert_eq!(cuda_profile.select_dtype(None), DType::BF16);

        // Explicit requests should be respected
        assert_eq!(cuda_profile.select_dtype(Some("f32")), DType::F32);
        assert_eq!(cuda_profile.select_dtype(Some("f16")), DType::F16);
        assert_eq!(cuda_profile.select_dtype(Some("bf16")), DType::BF16);
    }

    #[test]
    fn test_cuda_without_bf16_defaults_to_f16() {
        let cuda_profile = DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Cuda,
            capabilities: DeviceCapabilities {
                supports_bf16: false,
                supports_f16: true,
                cuda_compute_capability: Some((7, 5)),
                ..Default::default()
            },
            memory_pool: None,
        };

        assert_eq!(cuda_profile.select_dtype(None), DType::F16);
        assert_eq!(
            cuda_profile
                .try_select_dtype(Some("bf16"))
                .unwrap_err()
                .requested,
            "bf16"
        );
    }

    #[test]
    fn cuda_model_family_defaults_are_specific() {
        let cuda_profile = DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Cuda,
            capabilities: DeviceCapabilities {
                supports_bf16: true,
                supports_f16: true,
                ..Default::default()
            },
            memory_pool: None,
        };

        assert_eq!(
            cuda_profile.select_model_dtype(ModelFamily::WhisperAsr, None),
            DType::F16
        );
        assert_eq!(
            cuda_profile.select_model_dtype(ModelFamily::KokoroTts, None),
            DType::F32
        );
        assert_eq!(
            cuda_profile.select_model_dtype(ModelFamily::ParakeetAsr, None),
            DType::F32
        );
        assert_eq!(
            cuda_profile.select_model_dtype(ModelFamily::Qwen3Chat, None),
            DType::BF16
        );
        assert_eq!(
            cuda_profile.select_model_dtype(ModelFamily::Qwen3Tts, None),
            DType::BF16
        );
        assert_eq!(
            cuda_profile.select_model_dtype(ModelFamily::Gemma3Chat, None),
            DType::BF16
        );
        assert_eq!(
            cuda_profile.select_model_dtype(ModelFamily::Voxtral, None),
            DType::BF16
        );
        assert_eq!(
            cuda_profile.select_model_dtype(ModelFamily::VoxtralTts, None),
            DType::BF16
        );
        assert_eq!(
            cuda_profile.select_model_dtype(ModelFamily::VibeVoiceTts, None),
            DType::BF16
        );
        assert_eq!(
            cuda_profile.select_model_dtype(ModelFamily::VibeVoiceAsr, None),
            DType::BF16
        );
        assert_eq!(
            cuda_profile.select_model_dtype(ModelFamily::NemotronAsr, None),
            DType::BF16
        );
    }

    #[test]
    fn parse_dtype_name_handles_config_aliases() {
        assert_eq!(parse_dtype_name("torch.bfloat16"), Some(DType::BF16));
        assert_eq!(parse_dtype_name("torch.float16"), Some(DType::F16));
        assert_eq!(parse_dtype_name("fp16"), Some(DType::F16));
        assert_eq!(parse_dtype_name("float"), Some(DType::F32));
        assert_eq!(parse_dtype_name("float8"), None);
    }

    #[test]
    fn cuda_dense_checkpoint_dtype_beats_generic_bf16_default() {
        let cuda_profile = DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Cuda,
            capabilities: DeviceCapabilities {
                supports_bf16: true,
                supports_f16: true,
                ..Default::default()
            },
            memory_pool: None,
        };

        for family in [
            ModelFamily::Qwen3Chat,
            ModelFamily::Gemma3Chat,
            ModelFamily::Voxtral,
        ] {
            assert_eq!(
                cuda_profile.select_model_dtype_with_checkpoint(family, Some(DType::F16)),
                DType::F16
            );
            assert_eq!(
                cuda_profile.select_model_dtype_with_checkpoint(family, Some(DType::F32)),
                DType::F32
            );
        }
    }

    #[test]
    fn cuda_quantized_checkpoint_dtype_respects_capabilities() {
        let cuda_profile = DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Cuda,
            capabilities: DeviceCapabilities {
                supports_bf16: false,
                supports_f16: true,
                ..Default::default()
            },
            memory_pool: None,
        };

        let selection = cuda_profile.resolve_dtype(
            DTypeSelectionRequest::new(None)
                .with_model_family(ModelFamily::Qwen3Asr)
                .with_checkpoint_dtype(DType::BF16)
                .with_quantized(true),
        );
        assert_eq!(selection.dtype, DType::F16);

        let selection = cuda_profile.resolve_dtype(
            DTypeSelectionRequest::new(None)
                .with_model_family(ModelFamily::Qwen3Asr)
                .with_checkpoint_dtype(DType::F32)
                .with_quantized(true),
        );
        assert_eq!(selection.dtype, DType::F32);
    }

    #[test]
    fn cuda_checked_model_dtype_rejects_bad_overrides() {
        let cuda_profile = DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Cuda,
            capabilities: DeviceCapabilities {
                supports_bf16: false,
                supports_f16: true,
                ..Default::default()
            },
            memory_pool: None,
        };

        let err = cuda_profile
            .select_model_dtype_checked(ModelFamily::Qwen3Chat, Some("bf16"), "Qwen3 chat")
            .unwrap_err();
        assert!(err.to_string().contains("Invalid CUDA Qwen3 chat"));

        let err = cuda_profile
            .select_model_dtype_checked(ModelFamily::Qwen3Chat, Some("float8"), "Qwen3 chat")
            .unwrap_err();
        assert!(err.to_string().contains("expected one of"));
    }

    #[test]
    fn non_cuda_checked_model_dtype_falls_back_or_honours_metal_overrides() {
        let cpu_profile = DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Cpu,
            capabilities: DeviceCapabilities::default(),
            memory_pool: None,
        };
        let metal_profile = DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Metal,
            capabilities: DeviceCapabilities {
                prefers_f32: true,
                ..Default::default()
            },
            memory_pool: None,
        };

        assert_eq!(
            cpu_profile
                .select_model_dtype_checked(ModelFamily::Qwen3Chat, Some("float8"), "Qwen3 chat")
                .unwrap(),
            DType::F32
        );
        assert_eq!(
            metal_profile
                .select_model_dtype_checked(ModelFamily::Qwen3Chat, Some("bf16"), "Qwen3 chat")
                .unwrap(),
            DType::BF16
        );
    }

    #[test]
    fn whisper_model_dtype_defaults_to_f32_on_cpu_and_f16_on_metal() {
        let cpu_profile = DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Cpu,
            capabilities: DeviceCapabilities::default(),
            memory_pool: None,
        };
        let metal_profile = DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Metal,
            capabilities: DeviceCapabilities {
                prefers_f32: true,
                ..Default::default()
            },
            memory_pool: None,
        };

        assert_eq!(
            cpu_profile.select_model_dtype(ModelFamily::WhisperAsr, None),
            DType::F32
        );
        assert_eq!(
            cpu_profile
                .select_model_dtype_checked(ModelFamily::WhisperAsr, Some("f16"), "Whisper")
                .unwrap(),
            DType::F32
        );
        // Whisper's F16 checkpoint runs in F16 on every Apple family.
        let metal_default = metal_profile.metal_default_dtype_selection_with(
            DTypeSelectionRequest::new(None).with_model_family(ModelFamily::WhisperAsr),
            MetalDTypeOverride::Auto,
        );
        assert_eq!(metal_default.dtype, DType::F16);
        assert!(
            metal_default.reason.contains("whisper.cpp"),
            "{}",
            metal_default.reason
        );
        assert_eq!(
            metal_profile
                .metal_default_dtype_selection_with(
                    DTypeSelectionRequest::new(None).with_model_family(ModelFamily::WhisperAsr),
                    MetalDTypeOverride::F32,
                )
                .dtype,
            DType::F32
        );
        assert_eq!(
            metal_profile
                .select_model_dtype_checked(ModelFamily::WhisperAsr, Some("f32"), "Whisper")
                .unwrap(),
            DType::F32
        );
        assert_eq!(
            metal_profile
                .select_model_dtype_checked(ModelFamily::WhisperAsr, Some("bf16"), "Whisper")
                .unwrap(),
            DType::BF16
        );
    }

    #[test]
    fn vibevoice_tts_uses_f16_on_metal_while_cpu_and_asr_stay_f32() {
        let cpu_profile = DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Cpu,
            capabilities: DeviceCapabilities::default(),
            memory_pool: None,
        };
        let metal_profile = DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Metal,
            capabilities: DeviceCapabilities {
                prefers_f32: true,
                supports_f16: true,
                supports_bf16: true,
                ..Default::default()
            },
            memory_pool: None,
        };

        assert_eq!(
            cpu_profile.select_model_dtype(ModelFamily::VibeVoiceTts, None),
            DType::F32
        );
        assert_eq!(
            metal_profile.select_model_dtype(ModelFamily::VibeVoiceTts, None),
            DType::F16
        );
        assert_eq!(
            metal_profile.select_model_dtype(ModelFamily::VibeVoiceAsr, None),
            DType::F32
        );
    }

    #[test]
    fn cpu_and_metal_nemotron_model_dtype_defaults_to_f32() {
        let cpu_profile = DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Cpu,
            capabilities: DeviceCapabilities::default(),
            memory_pool: None,
        };
        let metal_profile = DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Metal,
            capabilities: DeviceCapabilities {
                prefers_f32: true,
                supports_f16: true,
                supports_bf16: true,
                ..Default::default()
            },
            memory_pool: None,
        };

        assert_eq!(
            cpu_profile.select_model_dtype(ModelFamily::NemotronAsr, None),
            DType::F32
        );
        assert_eq!(
            cpu_profile
                .select_model_dtype_checked(ModelFamily::NemotronAsr, Some("bf16"), "Nemotron ASR")
                .unwrap(),
            DType::F32
        );
        assert_eq!(
            metal_profile.select_model_dtype(ModelFamily::NemotronAsr, None),
            DType::F32
        );
        assert_eq!(
            metal_profile
                .select_model_dtype_checked(ModelFamily::NemotronAsr, Some("f16"), "Nemotron ASR")
                .unwrap(),
            DType::F16
        );
    }

    #[test]
    fn cuda_vibevoice_dtype_falls_back_by_capability() {
        let cuda_f16_profile = DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Cuda,
            capabilities: DeviceCapabilities {
                supports_bf16: false,
                supports_f16: true,
                ..Default::default()
            },
            memory_pool: None,
        };
        let cuda_f32_profile = DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Cuda,
            capabilities: DeviceCapabilities {
                supports_bf16: false,
                supports_f16: false,
                ..Default::default()
            },
            memory_pool: None,
        };

        assert_eq!(
            cuda_f16_profile.select_model_dtype(ModelFamily::VibeVoiceTts, None),
            DType::F16
        );
        assert_eq!(
            cuda_f32_profile.select_model_dtype(ModelFamily::VibeVoiceTts, None),
            DType::F32
        );
    }

    #[test]
    fn cuda_nemotron_dtype_falls_back_by_capability() {
        let cuda_bf16_profile = DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Cuda,
            capabilities: DeviceCapabilities {
                supports_bf16: true,
                supports_f16: true,
                ..Default::default()
            },
            memory_pool: None,
        };
        let cuda_f16_profile = DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Cuda,
            capabilities: DeviceCapabilities {
                supports_bf16: false,
                supports_f16: true,
                ..Default::default()
            },
            memory_pool: None,
        };
        let cuda_f32_profile = DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Cuda,
            capabilities: DeviceCapabilities {
                supports_bf16: false,
                supports_f16: false,
                ..Default::default()
            },
            memory_pool: None,
        };

        assert_eq!(
            cuda_bf16_profile.select_model_dtype(ModelFamily::NemotronAsr, None),
            DType::BF16
        );
        assert_eq!(
            cuda_f16_profile.select_model_dtype(ModelFamily::NemotronAsr, None),
            DType::F16
        );
        assert_eq!(
            cuda_f32_profile.select_model_dtype(ModelFamily::NemotronAsr, None),
            DType::F32
        );
    }

    #[test]
    fn cuda_compute_capability_gates_bf16() {
        assert!(!cuda_compute_capability_supports_bf16((7, 5)));
        assert!(cuda_compute_capability_supports_bf16((8, 0)));
        assert!(cuda_compute_capability_supports_bf16((9, 0)));
    }

    #[test]
    fn cuda_compute_capability_gates_int8_tensor_cores() {
        assert!(!cuda_compute_capability_supports_int8_tensor_cores((7, 0)));
        assert!(cuda_compute_capability_supports_int8_tensor_cores((7, 5)));
        assert!(cuda_compute_capability_supports_int8_tensor_cores((8, 0)));
    }

    #[test]
    fn test_cpu_always_f32() {
        let cpu_profile = DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Cpu,
            capabilities: DeviceCapabilities::default(),
            memory_pool: None,
        };

        // CPU should always use F32 regardless of request
        assert_eq!(cpu_profile.select_dtype(None), DType::F32);
        assert_eq!(cpu_profile.select_dtype(Some("bf16")), DType::F32);
        assert_eq!(cpu_profile.select_dtype(Some("f16")), DType::F32);
        assert_eq!(cpu_profile.select_dtype(Some("f32")), DType::F32);
    }
}
