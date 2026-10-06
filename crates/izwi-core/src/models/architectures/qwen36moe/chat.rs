//! Qwen3.5-MoE chat model loader and text generation.
//!
//! The wrapper is deliberately thin: it owns family-specific loading (the
//! native block-FP8 bundle, or the synthetic GGUF fixture for CI) and
//! text-only prompt preparation, while prompt rendering, the hybrid decode
//! state machinery, and sampling come from the shared
//! [`Qwen35ChatExec`](crate::models::architectures::qwen35::chat::Qwen35ChatExec)
//! core the dense `Qwen35ChatModel` drives.

use std::path::Path;

use candle_core::DType;

use crate::backends::{BackendKind, DeviceProfile};
use crate::error::{Error, Result};
use crate::kv::v2::InferenceStateContract;
use crate::kv::{InferenceStateCapability, InferenceStateContractProvider};
use crate::model::ModelVariant;
use crate::models::architectures::qwen35::chat::{
    ChatDecodeState, ChatDecodeStep, Qwen35ChatExec, Qwen35PreparedPrompt, Qwen35TextConfig,
    Qwen35Tokenizer,
};
use crate::models::architectures::qwen35::text::{GgufSource, Qwen35TextModel};
use crate::models::shared::attention::paged::default_kv_page_size;
use crate::models::shared::attention::physical::PhysicalPagedKvCache;
use crate::models::shared::chat::{ChatGenerationConfig, ChatMessage};
use crate::models::shared::moe::ExpertActivationCounters;
use crate::models::shared::weights::gguf::GgufLoader;

use super::gguf::{parse_fixture_gguf_config, QWEN36_MOE_FIXTURE_GGUF_FILENAME};
use super::native::{Qwen36MoeMtpLoadPolicy, Qwen36MoeNativeCheckpoint};
use super::native_model::load_text_model_native;

const CUDA_BF16_KV_ENV: &str = "IZWI_QWEN36_CUDA_BF16_KV";

/// Persistent KV storage selection for one loaded model. CUDA picks BF16 by
/// default (compute capability 8.0+) because BF16 activations narrowed into
/// an F16 cache lose their exponent range above 65504 and can poison
/// attention with infinities; the portable backends keep the F32 cache the
/// fixture and CPU/Metal plans are validated against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Qwen36MoeKvStorageProvider {
    CpuF32,
    MetalF32,
    CudaF16Fallback,
    CudaF16CapabilityFallback,
    CudaBf16,
}

impl Qwen36MoeKvStorageProvider {
    fn select(
        backend: BackendKind,
        cuda_compute_capability: Option<(u32, u32)>,
        cuda_bf16_override: Option<&str>,
    ) -> Self {
        fn bf16_kv_enabled(raw: Option<&str>) -> bool {
            matches!(
                raw.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
                None | Some("1" | "true" | "yes" | "on")
            )
        }
        fn cuda_supports_bf16(capability: Option<(u32, u32)>) -> bool {
            capability.is_some_and(crate::backends::device::cuda_compute_capability_supports_bf16)
        }
        match backend {
            BackendKind::Cpu => Self::CpuF32,
            BackendKind::Metal => Self::MetalF32,
            BackendKind::Cuda
                if bf16_kv_enabled(cuda_bf16_override)
                    && cuda_supports_bf16(cuda_compute_capability) =>
            {
                Self::CudaBf16
            }
            BackendKind::Cuda if bf16_kv_enabled(cuda_bf16_override) => {
                Self::CudaF16CapabilityFallback
            }
            BackendKind::Cuda => Self::CudaF16Fallback,
        }
    }

    const fn dtype(self) -> DType {
        match self {
            Self::CpuF32 | Self::MetalF32 => DType::F32,
            Self::CudaF16Fallback | Self::CudaF16CapabilityFallback => DType::F16,
            Self::CudaBf16 => DType::BF16,
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::CpuF32 => "portable_f32",
            Self::MetalF32 => "metal_f32",
            Self::CudaF16Fallback => "cuda_f16_fallback",
            Self::CudaF16CapabilityFallback => "cuda_f16_capability_fallback",
            Self::CudaBf16 => "cuda_bf16",
        }
    }

    const fn fallback_reason(self) -> Option<&'static str> {
        match self {
            Self::CudaF16Fallback => {
                Some("CUDA BF16 KV disabled by IZWI_QWEN36_CUDA_BF16_KV; using F16")
            }
            Self::CudaF16CapabilityFallback => Some(
                "CUDA BF16 KV requires an observed compute capability 8.0 or newer; using F16",
            ),
            _ => None,
        }
    }
}

pub struct Qwen36MoeChatModel {
    device_kind: BackendKind,
    kv_storage_provider: Qwen36MoeKvStorageProvider,
    performance: crate::performance::CudaPerformanceConfig,
    exec: Qwen35ChatExec,
}

impl InferenceStateContractProvider for Qwen36MoeChatModel {
    fn inference_state_contract(&self) -> Result<InferenceStateCapability> {
        let dtype = self.kv_storage_provider.dtype();
        Ok(InferenceStateCapability::Managed(
            self.managed_composite_cache_contract(dtype, default_kv_page_size())?,
        ))
    }
}

impl Qwen36MoeChatModel {
    /// Load the Qwen3.6-35B-A3B-FP8 family (qwen3_5_moe architecture) from
    /// its native bundle, or from a synthetic GGUF fixture when one is
    /// present (CI-only; the published variant never ships as GGUF).
    pub fn load(model_dir: &Path, variant: ModelVariant, device: DeviceProfile) -> Result<Self> {
        Self::load_with_performance(
            model_dir,
            variant,
            device,
            &crate::performance::PerformanceConfig::default(),
            false,
        )
    }

    pub fn load_with_performance(
        model_dir: &Path,
        variant: ModelVariant,
        device: DeviceProfile,
        performance: &crate::performance::PerformanceConfig,
        _prefix_reuse: bool,
    ) -> Result<Self> {
        performance.validate()?;
        if variant != ModelVariant::Qwen36Moe35BA3BFp8 {
            return Err(Error::ModelLoadError(format!(
                "Unsupported Qwen3.5/3.6-MoE chat variant: {variant}"
            )));
        }
        let device_kind = BackendKind::from(device.kind);
        let kv_storage_provider = Qwen36MoeKvStorageProvider::select(
            device_kind,
            device.capabilities.cuda_compute_capability,
            std::env::var(CUDA_BF16_KV_ENV).ok().as_deref(),
        );
        tracing::info!(
            provider = kv_storage_provider.as_str(),
            "Qwen3.6-MoE KV storage selection"
        );
        if let Some(reason) = kv_storage_provider.fallback_reason() {
            tracing::warn!(reason, "Qwen3.6-MoE KV storage fell back");
        }
        tracing::info!(
            cuda_mode = ?performance.cuda.mode,
            projection_backend = ?performance.cuda.projection_backend,
            "Qwen3.6-MoE performance policy"
        );
        // MTP mirrors the qwen3.8 gating: the master CUDA switch also gates
        // MTP on CUDA devices; CPU/Metal consult the MTP knob alone. The
        // policy both validates the draft manifest and constructs the head.
        let mtp_enabled = if device_kind == BackendKind::Cuda {
            performance.cuda.enabled() && performance.cuda.mtp.enabled()
        } else {
            performance.cuda.mtp.enabled()
        };
        tracing::info!(mtp_enabled, "Qwen3.6-MoE MTP policy");
        let mtp_policy = if mtp_enabled {
            Qwen36MoeMtpLoadPolicy::Enabled
        } else {
            Qwen36MoeMtpLoadPolicy::Disabled
        };
        let _ = mtp_policy;
        let fixture_path = model_dir.join(QWEN36_MOE_FIXTURE_GGUF_FILENAME);
        let exec = if fixture_path.exists() {
            Self::load_fixture_gguf(model_dir, &fixture_path, variant, &device)?
        } else {
            Self::load_native(
                model_dir,
                variant,
                &device,
                &performance.cuda,
                mtp_enabled,
            )?
        };
        Ok(Self {
            device_kind,
            kv_storage_provider,
            performance: performance.cuda.clone(),
            exec,
        })
    }

    fn load_fixture_gguf(
        model_dir: &Path,
        fixture_path: &Path,
        variant: ModelVariant,
        device: &DeviceProfile,
    ) -> Result<Qwen35ChatExec> {
        let loader =
            GgufLoader::from_path_with_backend(fixture_path, BackendKind::from(device.kind))?;
        let text_config = parse_fixture_gguf_config(&loader)?;
        let tokenizer = Qwen35Tokenizer::load(model_dir, variant, &loader)?;
        let text_model = Qwen35TextModel::load_with_source(
            &GgufSource::new(&loader),
            &text_config,
            &device.device,
        )?;
        Ok(Qwen35ChatExec {
            variant,
            tokenizer,
            text_config,
            text_model,
            mtp_head: None,
        })
    }

    fn load_native(
        model_dir: &Path,
        variant: ModelVariant,
        device: &DeviceProfile,
        performance: &crate::performance::CudaPerformanceConfig,
        mtp_enabled: bool,
    ) -> Result<Qwen35ChatExec> {
        let mtp_policy = if mtp_enabled {
            Qwen36MoeMtpLoadPolicy::Enabled
        } else {
            Qwen36MoeMtpLoadPolicy::Disabled
        };
        let checkpoint = Qwen36MoeNativeCheckpoint::open_with_policies(
            model_dir,
            super::native::Qwen36MoeGeometryPolicy::from_env(),
            mtp_policy,
        )?;
        let tokenizer = Qwen35Tokenizer::load_hf(model_dir, variant)?;
        let (text_config, text_model, mtp_head) = load_text_model_native(
            &checkpoint,
            device,
            &device.device,
            performance,
            mtp_enabled,
        )?;
        Ok(Qwen35ChatExec {
            variant,
            tokenizer,
            text_config,
            text_model,
            mtp_head,
        })
    }

    pub fn variant(&self) -> ModelVariant {
        self.exec.variant()
    }

    pub fn text_config(&self) -> &Qwen35TextConfig {
        self.exec.text_config()
    }

    pub fn max_context_tokens(&self) -> Result<usize> {
        self.exec.max_context_tokens()
    }

    pub(crate) fn managed_composite_cache_contract(
        &self,
        attention_dtype: DType,
        preferred_page_tokens: usize,
    ) -> Result<InferenceStateContract> {
        self.exec
            .managed_composite_cache_contract(attention_dtype, preferred_page_tokens)
    }

    pub fn chat_template(&self) -> &str {
        self.exec.chat_template()
    }

    pub fn default_enable_thinking(&self) -> bool {
        self.exec.default_enable_thinking()
    }

    pub fn prompt_token_ids(&self, messages: &[ChatMessage]) -> Result<Vec<u32>> {
        self.exec.prompt_token_ids(messages)
    }

    pub fn prompt_token_ids_with_config(
        &self,
        messages: &[ChatMessage],
        config: &ChatGenerationConfig,
    ) -> Result<Vec<u32>> {
        self.exec.prompt_token_ids_with_config(messages, config)
    }

    /// Text-only prepared prompt. The checkpoint's vision tower is out of
    /// scope for this family: any media input is rejected up front.
    pub fn prepare_prompt_for_execution(
        &self,
        messages: &[ChatMessage],
        config: &ChatGenerationConfig,
    ) -> Result<Qwen35PreparedPrompt> {
        if !config.request.media_inputs.is_empty() {
            return Err(Error::InvalidInput(
                "Qwen3.5/3.6-MoE serving is text-only and does not accept media inputs".to_string(),
            ));
        }
        self.exec.prepare_text_prompt(messages, config)
    }

    pub fn supports_incremental_decode(&self) -> bool {
        self.exec.supports_incremental_decode()
    }

    pub fn supports_continuous_decode_batch(&self) -> bool {
        self.exec.supports_continuous_decode_batch()
    }

    pub fn continuous_decode_batch_workspace_per_row_bytes(&self) -> Result<u64> {
        self.exec.continuous_decode_batch_workspace_per_row_bytes()
    }

    pub fn device_kind(&self) -> BackendKind {
        self.device_kind
    }

    /// Per-sparse-layer expert activation histograms (DS10 A6 posture).
    pub(crate) fn expert_activation_counters(
        &self,
    ) -> Vec<std::sync::Arc<ExpertActivationCounters>> {
        self.exec.text_model.expert_activation_counters()
    }

    pub(crate) fn start_decode_state_physical(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
        config: &ChatGenerationConfig,
        prepared: Option<&Qwen35PreparedPrompt>,
        cache: PhysicalPagedKvCache,
    ) -> Result<ChatDecodeState> {
        let prepared = match prepared {
            Some(prepared) => prepared.clone(),
            None => self.prepare_prompt_for_execution(messages, config)?,
        };
        let mut state = self.exec.begin_resumable_prefill_state_physical(
            &prepared,
            max_new_tokens,
            config,
            cache,
        )?;
        self.exec.continue_resumable_prefill_physical(
            &mut state,
            &prepared,
            0,
            prepared.prompt_ids().len(),
        )?;
        Ok(state)
    }

    pub(crate) fn begin_resumable_prefill_state_physical(
        &self,
        prepared: &Qwen35PreparedPrompt,
        max_new_tokens: usize,
        config: &ChatGenerationConfig,
        cache: PhysicalPagedKvCache,
    ) -> Result<ChatDecodeState> {
        self.exec
            .begin_resumable_prefill_state_physical(prepared, max_new_tokens, config, cache)
    }

    pub(crate) fn continue_resumable_prefill_physical(
        &self,
        state: &mut ChatDecodeState,
        prepared: &Qwen35PreparedPrompt,
        span_start: usize,
        span_end: usize,
    ) -> Result<bool> {
        self.exec
            .continue_resumable_prefill_physical(state, prepared, span_start, span_end)
    }

    pub fn decode_step(&self, state: &mut ChatDecodeState) -> Result<ChatDecodeStep> {
        self.exec.decode_step(state)
    }

    pub fn decode_step_batch(
        &self,
        states: &mut [&mut ChatDecodeState],
    ) -> Result<Vec<ChatDecodeStep>> {
        self.exec.decode_step_batch(states)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::kv::{KvArenaConfig, KvLayerConfig};
    use crate::engine::ModelInstanceId;
    use crate::kv::{CacheBlockRef, KvArenaId, KvGroupId, KvLayerBinding};
    use crate::models::shared::chat::ChatRole;
    use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
    use candle_core::{Device, Tensor};
    use std::path::PathBuf;
    use std::sync::Arc;

    // ------------------------------------------------------------------
    // Tiny synthetic GGUF fixture (the DS10 pattern): a 4-layer hybrid
    // trunk (interval 2 → GDN layers 0/2, gated full attention 1/3) with
    // 4 routed experts (top-2) plus an always-on shared expert, a 256-byte
    // level vocabulary with the ChatML specials, and an optional shared
    // expert gate. Exercises MoE mechanics without the 35B download.
    // ------------------------------------------------------------------

    const FIXTURE_VOCAB: usize = 263;
    const FIXTURE_HIDDEN: usize = 32;
    const FIXTURE_EXPERTS: usize = 4;
    const FIXTURE_EXPERT_FF: usize = 16;
    const FIXTURE_SHARED_FF: usize = 8;

    struct Lcg(u64);

    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / u32::MAX as f32 - 0.5) * 0.2
        }

        fn tensor(&mut self, shape: Vec<usize>) -> QTensor {
            let count: usize = shape.iter().product();
            let values: Vec<f32> = (0..count).map(|_| self.next_f32()).collect();
            QTensor::quantize(
                &Tensor::from_vec(values, shape, &Device::Cpu).unwrap(),
                GgmlDType::F32,
            )
            .unwrap()
        }
    }

    fn byte_level_char(byte: u8) -> char {
        let mut bytes: Vec<u8> = (b'!'..=b'~')
            .chain(0xA1..=0xAC)
            .chain(0xAE..=0xFF)
            .collect();
        let mut codepoints: Vec<u32> = bytes.iter().map(|b| u32::from(*b)).collect();
        let mut next = 0u32;
        for candidate in 0..=u8::MAX {
            if !bytes.contains(&candidate) {
                bytes.push(candidate);
                codepoints.push(256 + next);
                next += 1;
            }
        }
        let index = bytes
            .iter()
            .position(|candidate| *candidate == byte)
            .expect("byte-level alphabet contains every byte");
        char::from_u32(codepoints[index]).expect("byte-level codepoint is valid")
    }

    fn fixture_tokenizer_metadata() -> (Vec<gguf_file::Value>, u32) {
        let mut tokens: Vec<String> = (0..=u8::MAX)
            .map(|byte| byte_level_char(byte).to_string())
            .collect();
        tokens.push("<|im_start|>".to_string());
        tokens.push("<|im_end|>".to_string());
        tokens.push("<|image_pad|>".to_string());
        tokens.push("<|video_pad|>".to_string());
        tokens.push("<|endoftext|>".to_string());
        tokens.push("<think>".to_string());
        tokens.push("</think>".to_string());
        let mut token_types = vec![1u32; 256];
        token_types.extend([3, 3, 3, 3, 3, 4, 4]);
        let eos_token_id = 257u32; // <|im_end|>
        let values = vec![
            gguf_file::Value::Array(tokens.into_iter().map(gguf_file::Value::String).collect()),
            gguf_file::Value::Array(token_types.into_iter().map(gguf_file::Value::U32).collect()),
            gguf_file::Value::Array(Vec::new()),
            gguf_file::Value::String("qwen35".to_string()),
            gguf_file::Value::String(
                "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}<|im_start|>assistant\n".to_string(),
            ),
            gguf_file::Value::U64(u64::from(eos_token_id)),
        ];
        (values, eos_token_id)
    }

    fn write_fixture(dir: &Path) -> PathBuf {
        let mut rng = Lcg(0x5135_8A0E);
        let hidden = FIXTURE_HIDDEN;
        let mut tensors: Vec<(String, QTensor)> = Vec::new();

        let mut push = |rng: &mut Lcg, name: String, shape: Vec<usize>| {
            tensors.push((name, rng.tensor(shape)));
        };

        push(
            &mut rng,
            "token_embd.weight".into(),
            vec![FIXTURE_VOCAB, hidden],
        );
        push(&mut rng, "output_norm.weight".into(), vec![hidden]);

        for layer in 0..4usize {
            let prefix = format!("blk.{layer}");
            push(&mut rng, format!("{prefix}.attn_norm.weight"), vec![hidden]);
            push(
                &mut rng,
                format!("{prefix}.post_attention_norm.weight"),
                vec![hidden],
            );
            if (layer + 1) % 2 == 0 {
                // Gated full attention: q_proj fuses the per-head output
                // gate, so it carries heads × head_dim × 2 rows.
                push(
                    &mut rng,
                    format!("{prefix}.attn_q.weight"),
                    vec![64, hidden],
                );
                push(
                    &mut rng,
                    format!("{prefix}.attn_k.weight"),
                    vec![16, hidden],
                );
                push(
                    &mut rng,
                    format!("{prefix}.attn_v.weight"),
                    vec![16, hidden],
                );
                push(
                    &mut rng,
                    format!("{prefix}.attn_output.weight"),
                    vec![hidden, hidden],
                );
                push(&mut rng, format!("{prefix}.attn_q_norm.weight"), vec![8]);
                push(&mut rng, format!("{prefix}.attn_k_norm.weight"), vec![8]);
            } else {
                // GDN: K heads 2 × state 8 twice + V 4 heads × 8.
                push(
                    &mut rng,
                    format!("{prefix}.attn_qkv.weight"),
                    vec![64, hidden],
                );
                push(
                    &mut rng,
                    format!("{prefix}.attn_gate.weight"),
                    vec![32, hidden],
                );
                push(
                    &mut rng,
                    format!("{prefix}.ssm_beta.weight"),
                    vec![4, hidden],
                );
                push(
                    &mut rng,
                    format!("{prefix}.ssm_alpha.weight"),
                    vec![4, hidden],
                );
                push(&mut rng, format!("{prefix}.ssm_dt.bias"), vec![4]);
                push(&mut rng, format!("{prefix}.ssm_a"), vec![4]);
                push(&mut rng, format!("{prefix}.ssm_conv1d.weight"), vec![64, 4]);
                push(&mut rng, format!("{prefix}.ssm_norm.weight"), vec![8]);
                push(
                    &mut rng,
                    format!("{prefix}.ssm_out.weight"),
                    vec![hidden, 32],
                );
            }
            // Sparse MoE feed-forward (every layer) with shared expert.
            push(
                &mut rng,
                format!("{prefix}.ffn_gate_inp.weight"),
                vec![FIXTURE_EXPERTS, hidden],
            );
            push(
                &mut rng,
                format!("{prefix}.ffn_gate_exps.weight"),
                vec![FIXTURE_EXPERTS, FIXTURE_EXPERT_FF, hidden],
            );
            push(
                &mut rng,
                format!("{prefix}.ffn_up_exps.weight"),
                vec![FIXTURE_EXPERTS, FIXTURE_EXPERT_FF, hidden],
            );
            push(
                &mut rng,
                format!("{prefix}.ffn_down_exps.weight"),
                vec![FIXTURE_EXPERTS, hidden, FIXTURE_EXPERT_FF],
            );
            push(
                &mut rng,
                format!("{prefix}.ffn_gate_shexp.weight"),
                vec![FIXTURE_SHARED_FF, hidden],
            );
            push(
                &mut rng,
                format!("{prefix}.ffn_up_shexp.weight"),
                vec![FIXTURE_SHARED_FF, hidden],
            );
            push(
                &mut rng,
                format!("{prefix}.ffn_down_shexp.weight"),
                vec![hidden, FIXTURE_SHARED_FF],
            );
            push(
                &mut rng,
                format!("{prefix}.ffn_gate_inp_shexp.weight"),
                vec![1, hidden],
            );
        }

        let (tokenizer_values, _eos) = fixture_tokenizer_metadata();
        let metadata: Vec<(&str, gguf_file::Value)> = vec![
            (
                "general.architecture",
                gguf_file::Value::String("qwen36moe".into()),
            ),
            ("qwen36moe.block_count", gguf_file::Value::U64(4)),
            ("qwen36moe.context_length", gguf_file::Value::U64(64)),
            (
                "qwen36moe.embedding_length",
                gguf_file::Value::U64(hidden as u64),
            ),
            (
                "qwen36moe.feed_forward_length",
                gguf_file::Value::U64(FIXTURE_EXPERT_FF as u64),
            ),
            ("qwen36moe.attention.head_count", gguf_file::Value::U64(4)),
            (
                "qwen36moe.attention.head_count_kv",
                gguf_file::Value::U64(2),
            ),
            ("qwen36moe.attention.key_length", gguf_file::Value::U64(8)),
            ("qwen36moe.attention.value_length", gguf_file::Value::U64(8)),
            (
                "qwen36moe.attention.layer_norm_rms_epsilon",
                gguf_file::Value::F64(1e-5),
            ),
            (
                "qwen36moe.rope.dimension_sections",
                gguf_file::Value::Array(vec![
                    gguf_file::Value::U64(2),
                    gguf_file::Value::U64(2),
                    gguf_file::Value::U64(2),
                ]),
            ),
            ("qwen36moe.rope.dimension_count", gguf_file::Value::U64(8)),
            ("qwen36moe.rope.freq_base", gguf_file::Value::F64(10_000.0)),
            ("qwen36moe.ssm.conv_kernel", gguf_file::Value::U64(4)),
            ("qwen36moe.ssm.state_size", gguf_file::Value::U64(8)),
            ("qwen36moe.ssm.group_count", gguf_file::Value::U64(2)),
            ("qwen36moe.ssm.time_step_rank", gguf_file::Value::U64(4)),
            ("qwen36moe.ssm.inner_size", gguf_file::Value::U64(32)),
            (
                "qwen36moe.full_attention_interval",
                gguf_file::Value::U64(2),
            ),
            (
                "qwen36moe.expert_count",
                gguf_file::Value::U64(FIXTURE_EXPERTS as u64),
            ),
            ("qwen36moe.expert_used_count", gguf_file::Value::U64(2)),
            (
                "qwen36moe.expert_feed_forward_length",
                gguf_file::Value::U64(FIXTURE_EXPERT_FF as u64),
            ),
            (
                "qwen36moe.expert_shared_feed_forward_length",
                gguf_file::Value::U64(FIXTURE_SHARED_FF as u64),
            ),
            ("tokenizer.ggml.tokens", tokenizer_values[0].clone()),
            ("tokenizer.ggml.token_type", tokenizer_values[1].clone()),
            ("tokenizer.ggml.merges", tokenizer_values[2].clone()),
            ("tokenizer.ggml.pre", tokenizer_values[3].clone()),
            ("tokenizer.chat_template", tokenizer_values[4].clone()),
            ("tokenizer.ggml.eos_token_id", tokenizer_values[5].clone()),
        ];

        let path = dir.join(QWEN36_MOE_FIXTURE_GGUF_FILENAME);
        let mut file = std::fs::File::create(&path).expect("create fixture gguf");
        let weight_refs: Vec<(&str, &QTensor)> = tensors
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect();
        let metadata_refs: Vec<(&str, &gguf_file::Value)> = metadata
            .iter()
            .map(|(name, value)| (*name, value))
            .collect();
        gguf_file::write(&mut file, &metadata_refs, &weight_refs).expect("write fixture gguf");
        path
    }

    fn fixture_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("izwi-qwen36moe-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn load_fixture(tag: &str) -> (Qwen36MoeChatModel, PathBuf) {
        let dir = fixture_dir(tag);
        write_fixture(&dir);
        let model =
            Qwen36MoeChatModel::load(&dir, ModelVariant::Qwen36Moe35BA3BFp8, DeviceProfile::cpu())
                .expect("load qwen36moe fixture");
        (model, dir)
    }

    fn physical_cache(model: &Qwen36MoeChatModel, device: &DeviceProfile) -> PhysicalPagedKvCache {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        use crate::backends::kv::CandleAcceleratorKvArena;
        use crate::backends::kv::{CpuKvArena, KvArena};
        use candle_core::DeviceLocation;
        let contract = match model.inference_state_contract().expect("contract") {
            InferenceStateCapability::Managed(contract) => contract,
            other => panic!("expected managed contract, got {other:?}"),
        };
        // Build the CPU arena directly from the contract's paged-attention
        // domain: full-attention model layers 1 and 3 → physical 0 and 1.
        let kv_heads = model.text_config().attention_head_count_kv;
        let head_dim = model.text_config().attention_key_length;
        // Mirror the runtime's device tag: Metal arenas require the exact
        // device identity (Candle reports the registry id, not the ordinal).
        let device_ordinal = match device.device.location() {
            DeviceLocation::Cpu => None,
            DeviceLocation::Cuda { gpu_id } => u32::try_from(gpu_id).ok(),
            DeviceLocation::Metal { gpu_id } => {
                let id = gpu_id as u64;
                Some((id ^ (id >> 32)) as u32)
            }
        };
        let id = KvArenaId {
            model_instance: ModelInstanceId::new(4242),
            backend: BackendKind::from(device.kind),
            device_ordinal,
            generation: 1,
        };
        let group = KvGroupId::new(1);
        let arena_config = KvArenaConfig {
            id,
            group,
            page_tokens: 8,
            capacity_pages: 16,
            growth: None,
            dtype: DType::F32,
            layers: vec![
                KvLayerConfig {
                    binding: KvLayerBinding {
                        model_layer: 1,
                        physical_layer: 0,
                    },
                    num_kv_heads: kv_heads as u32,
                    key_head_dim: head_dim as u32,
                    value_head_dim: head_dim as u32,
                },
                KvLayerConfig {
                    binding: KvLayerBinding {
                        model_layer: 3,
                        physical_layer: 1,
                    },
                    num_kv_heads: kv_heads as u32,
                    key_head_dim: head_dim as u32,
                    value_head_dim: head_dim as u32,
                },
            ],
        };
        let is_accelerator = BackendKind::from(device.kind) != BackendKind::Cpu;
        let arena: Arc<dyn KvArena> = if is_accelerator {
            #[cfg(any(feature = "cuda", feature = "metal"))]
            {
                Arc::new(
                    CandleAcceleratorKvArena::new_mutation_only(
                        arena_config,
                        device.device.clone(),
                    )
                    .unwrap(),
                )
            }
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            {
                let _ = arena_config;
                panic!("accelerator KV arenas require the cuda or metal feature")
            }
        } else {
            Arc::new(CpuKvArena::new(arena_config).unwrap())
        };
        let blocks = (0..16)
            .map(|index| CacheBlockRef {
                arena: id,
                group,
                index,
                slot_generation: 1,
            })
            .collect();
        let _ = contract;
        PhysicalPagedKvCache::new(
            arena,
            vec![
                KvLayerBinding {
                    model_layer: 1,
                    physical_layer: 0,
                },
                KvLayerBinding {
                    model_layer: 3,
                    physical_layer: 1,
                },
            ],
            blocks,
            0,
        )
        .unwrap()
    }

    fn generation_config() -> ChatGenerationConfig {
        ChatGenerationConfig::default()
    }

    /// Per-row cache windows over ONE shared arena: the continuous decode
    /// batch requires every row to reference the same arena instance.
    fn shared_physical_caches(
        model: &Qwen36MoeChatModel,
        device: &DeviceProfile,
        rows: usize,
    ) -> Vec<PhysicalPagedKvCache> {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        use crate::backends::kv::CandleAcceleratorKvArena;
        use crate::backends::kv::{CpuKvArena, KvArena};
        use candle_core::DeviceLocation;
        let contract = match model.inference_state_contract().expect("contract") {
            InferenceStateCapability::Managed(contract) => contract,
            other => panic!("expected managed contract, got {other:?}"),
        };
        let kv_heads = model.text_config().attention_head_count_kv;
        let head_dim = model.text_config().attention_key_length;
        let device_ordinal = match device.device.location() {
            DeviceLocation::Cpu => None,
            DeviceLocation::Cuda { gpu_id } => u32::try_from(gpu_id).ok(),
            DeviceLocation::Metal { gpu_id } => {
                let id = gpu_id as u64;
                Some((id ^ (id >> 32)) as u32)
            }
        };
        let id = KvArenaId {
            model_instance: ModelInstanceId::new(4244),
            backend: BackendKind::from(device.kind),
            device_ordinal,
            generation: 1,
        };
        let group = KvGroupId::new(1);
        let pages_per_row = 16usize;
        let arena_config = KvArenaConfig {
            id,
            group,
            page_tokens: 8,
            capacity_pages: (rows * pages_per_row) as u32,
            growth: None,
            dtype: DType::F32,
            layers: vec![
                KvLayerConfig {
                    binding: KvLayerBinding {
                        model_layer: 1,
                        physical_layer: 0,
                    },
                    num_kv_heads: kv_heads as u32,
                    key_head_dim: head_dim as u32,
                    value_head_dim: head_dim as u32,
                },
                KvLayerConfig {
                    binding: KvLayerBinding {
                        model_layer: 3,
                        physical_layer: 1,
                    },
                    num_kv_heads: kv_heads as u32,
                    key_head_dim: head_dim as u32,
                    value_head_dim: head_dim as u32,
                },
            ],
        };
        let is_accelerator = BackendKind::from(device.kind) != BackendKind::Cpu;
        let arena: std::sync::Arc<dyn KvArena> = if is_accelerator {
            #[cfg(any(feature = "cuda", feature = "metal"))]
            {
                std::sync::Arc::new(
                    CandleAcceleratorKvArena::new_mutation_only(arena_config, device.device.clone())
                        .unwrap(),
                )
            }
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            {
                let _ = arena_config;
                panic!("accelerator KV arenas require the cuda or metal feature")
            }
        } else {
            std::sync::Arc::new(CpuKvArena::new(arena_config).unwrap())
        };
        let bindings = vec![
            KvLayerBinding {
                model_layer: 1,
                physical_layer: 0,
            },
            KvLayerBinding {
                model_layer: 3,
                physical_layer: 1,
            },
        ];
        let blocks: Vec<CacheBlockRef> = (0..rows * pages_per_row)
            .map(|index| CacheBlockRef {
                arena: id,
                group,
                index: index as u32,
                slot_generation: 1,
            })
            .collect();
        let _ = contract;
        (0..rows)
            .map(|row| {
                PhysicalPagedKvCache::new(
                    arena.clone(),
                    bindings.clone(),
                    blocks[row * pages_per_row..(row + 1) * pages_per_row].to_vec(),
                    0,
                )
                .unwrap()
            })
            .collect()
    }

    #[test]
    fn fixture_decodes_two_rows_in_one_continuous_batch() {
        let (model, dir) = load_fixture("batch");
        let messages_for = |content: &str| {
            vec![ChatMessage {
                role: ChatRole::User,
                content: content.to_string(),
            }]
        };

        // Two concurrent sessions over one shared arena: different prompt
        // lengths, independent GDN/conv state, one batched decode step for
        // both rows at a time. Row b samples with its own seed/temperature
        // so the batch carries rows with different sampling configurations
        // (the fixture's greedy path converges both rows onto identical
        // tokens otherwise).
        let mut caches = shared_physical_caches(&model, &DeviceProfile::cpu(), 2);
        let mut state_a = model
            .start_decode_state_physical(
                &messages_for("ab"),
                8,
                &generation_config(),
                None,
                caches.remove(0),
            )
            .expect("row a decode state");
        let mut sampled = generation_config();
        sampled.seed = 0x5EED_0002;
        sampled.temperature = 1.0;
        let mut state_b = model
            .start_decode_state_physical(
                &messages_for("abcd"),
                8,
                &sampled,
                None,
                caches.remove(0),
            )
            .expect("row b decode state");

        // The first token of every session is the scalar prefill quantum
        // (it samples the stored prefill logits without a forward) — rows
        // join the continuous batch only after it. Deltas may be empty
        // (incremental UTF-8 buffering); parity is asserted below.
        let first_a = model
            .decode_step(&mut state_a)
            .expect("row a scalar first token")
            .delta;
        let first_b = model
            .decode_step(&mut state_b)
            .expect("row b scalar first token")
            .delta;

        let mut batched_a = Vec::new();
        let mut batched_b = Vec::new();
        for step_index in 0..4 {
            let steps = {
                let mut rows = [&mut state_a, &mut state_b];
                model.decode_step_batch(&mut rows).expect("batched decode step")
            };
            assert_eq!(steps.len(), 2, "one step per row");
            assert!(!steps[0].finished && !steps[1].finished);
            // tokens_generated is cumulative: scalar first token + one per
            // batched step.
            let expected_count = 2 + step_index;
            assert_eq!(steps[0].tokens_generated, expected_count);
            assert_eq!(steps[1].tokens_generated, expected_count);
            batched_a.push(steps[0].delta.clone());
            batched_b.push(steps[1].delta.clone());
        }

        // Batched continuations must agree token-for-token with solo decode
        // of the same prompts (greedy, deterministic fixture).
        let solo = |content: &str, config: &ChatGenerationConfig| -> Vec<String> {
            let cache = physical_cache(&model, &DeviceProfile::cpu());
            let mut state = model
                .start_decode_state_physical(
                    &messages_for(content),
                    8,
                    config,
                    None,
                    cache,
                )
                .expect("solo decode state");
            let mut deltas = Vec::new();
            for _ in 0..5 {
                let step = model.decode_step(&mut state).expect("solo decode step");
                deltas.push(step.delta);
                if step.finished {
                    break;
                }
            }
            deltas
        };
        let solo_a = solo("ab", &generation_config());
        let solo_b = solo("abcd", &sampled);
        assert_eq!(first_a, solo_a[0], "row a first token must match solo");
        assert_eq!(first_b, solo_b[0], "row b first token must match solo");
        assert_eq!(
            batched_a, solo_a[1..],
            "row a batched continuations must match solo"
        );
        assert_eq!(
            batched_b, solo_b[1..],
            "row b batched continuations must match solo"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn kv_storage_provider_keeps_portable_f32_and_gates_cuda_bf16() {
        let select = Qwen36MoeKvStorageProvider::select;
        assert_eq!(select(BackendKind::Cpu, None, None), Qwen36MoeKvStorageProvider::CpuF32);
        assert_eq!(select(BackendKind::Metal, None, None), Qwen36MoeKvStorageProvider::MetalF32);
        // Default CUDA policy: BF16 on capability 8.0+, F16 below.
        assert_eq!(
            select(BackendKind::Cuda, Some((8, 0)), None),
            Qwen36MoeKvStorageProvider::CudaBf16
        );
        assert_eq!(
            select(BackendKind::Cuda, Some((9, 0)), Some("TRUE")),
            Qwen36MoeKvStorageProvider::CudaBf16
        );
        assert_eq!(
            select(BackendKind::Cuda, Some((7, 5)), None),
            Qwen36MoeKvStorageProvider::CudaF16CapabilityFallback
        );
        assert_eq!(
            select(BackendKind::Cuda, None, None),
            Qwen36MoeKvStorageProvider::CudaF16CapabilityFallback
        );
        // The kill switch forces F16 even on capable hardware.
        for raw in ["0", "false", "OFF", "no"] {
            assert_eq!(
                select(BackendKind::Cuda, Some((8, 0)), Some(raw)),
                Qwen36MoeKvStorageProvider::CudaF16Fallback
            );
        }
        // Dtypes: portable plans stay F32, CUDA plans are F16/BF16.
        assert_eq!(Qwen36MoeKvStorageProvider::CpuF32.dtype(), DType::F32);
        assert_eq!(Qwen36MoeKvStorageProvider::MetalF32.dtype(), DType::F32);
        assert_eq!(
            Qwen36MoeKvStorageProvider::CudaF16CapabilityFallback.dtype(),
            DType::F16
        );
        assert_eq!(Qwen36MoeKvStorageProvider::CudaBf16.dtype(), DType::BF16);
    }

    #[test]
    fn fixture_loads_with_sparse_geometry_and_generates_deterministically() {
        let (model, dir) = load_fixture("e2e");

        // Family geometry came from the fixture metadata.
        assert_eq!(model.text_config().block_count, 4);
        assert_eq!(model.text_config().full_attention_interval, 2);
        let moe = model.text_config().moe_ffn.expect("sparse geometry");
        assert_eq!(moe.num_experts, FIXTURE_EXPERTS);
        assert_eq!(moe.num_experts_per_tok, 2);
        assert_eq!(moe.expert_intermediate_size, FIXTURE_EXPERT_FF);
        assert_eq!(moe.shared_expert_intermediate_size, FIXTURE_SHARED_FF);
        // Thinking default-on for the 35B-A3B MoE variant.
        assert!(model.default_enable_thinking());
        assert_eq!(model.max_context_tokens().unwrap(), 64);

        let messages = vec![
            ChatMessage {
                role: ChatRole::System,
                content: "You are tiny.".to_string(),
            },
            ChatMessage {
                role: ChatRole::User,
                content: "ab".to_string(),
            },
        ];
        let prompt_ids = model.prompt_token_ids(&messages).expect("prompt ids");
        assert!(!prompt_ids.is_empty());

        let cache = physical_cache(&model, &DeviceProfile::cpu());
        let mut state = model
            .start_decode_state_physical(&messages, 8, &generation_config(), None, cache)
            .expect("decode state");
        assert_eq!(state.prefill_progress(), prompt_ids.len());

        let mut steps = Vec::new();
        for _ in 0..16 {
            let step = model.decode_step(&mut state).expect("decode step");
            let finished = step.finished;
            steps.push(step);
            if finished {
                break;
            }
        }
        assert!(
            !steps.is_empty(),
            "fixture decode must produce at least one step"
        );
        assert!(
            steps.iter().any(|step| step.tokens_generated > 0),
            "fixture decode must generate tokens"
        );

        // Expert histograms: one counter array per sparse layer, selections
        // recorded for every routed decision (prompt + generated tokens) ×
        // top-2.
        let counters = model.expert_activation_counters();
        assert_eq!(counters.len(), 4);
        assert!(counters.iter().all(|c| c.total_selections() > 0));

        // Re-run and require identical output (seeded rng, greedy decode).
        let first: Vec<String> = steps.iter().map(|step| step.delta.clone()).collect();
        drop(state);
        let cache = physical_cache(&model, &DeviceProfile::cpu());
        let mut state = model
            .start_decode_state_physical(&messages, 8, &generation_config(), None, cache)
            .expect("decode state");
        let mut second = Vec::new();
        for _ in 0..16 {
            let step = model.decode_step(&mut state).expect("decode step");
            let finished = step.finished;
            second.push(step.delta);
            if finished {
                break;
            }
        }
        assert_eq!(first, second, "fixture generation must be deterministic");

        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn thinking_render_reaches_the_moe_prepared_prompt_both_ways() {
        let (model, dir) = load_fixture("thinking");
        // Fixture token layout: 256 byte tokens, then the specials —
        // <|im_start|>=256, <|im_end|>=257, <|endoftext|>=260,
        // <think>=261, </think>=262.
        const THINK_OPEN: u32 = 261;
        const THINK_CLOSE: u32 = 262;
        let messages = vec![ChatMessage {
            role: ChatRole::User,
            content: "ab".to_string(),
        }];
        let config_with = |enable: Option<bool>| ChatGenerationConfig {
            request: crate::models::shared::chat::ChatRequestConfig {
                enable_thinking: enable,
                tools: Vec::new(),
                media_inputs: Vec::new(),
                ..crate::models::shared::chat::ChatRequestConfig::default()
            },
            ..ChatGenerationConfig::default()
        };

        // Thinking default-on for the 35B-A3B MoE variant (no explicit flag).
        let default_ids = model
            .prompt_token_ids_with_config(&messages, &generation_config())
            .expect("default prompt ids");
        let enabled = model
            .prompt_token_ids_with_config(&messages, &config_with(Some(true)))
            .expect("enabled prompt ids");
        let disabled = model
            .prompt_token_ids_with_config(&messages, &config_with(Some(false)))
            .expect("disabled prompt ids");
        assert_eq!(default_ids, enabled, "thinking defaults on for this family");

        let open_count = |ids: &[u32]| ids.iter().filter(|id| **id == THINK_OPEN).count();
        let close_count = |ids: &[u32]| ids.iter().filter(|id| **id == THINK_CLOSE).count();
        assert_eq!(
            open_count(&enabled),
            1,
            "enabled prompt opens one think block"
        );
        assert_eq!(close_count(&enabled), 0, "enabled prompt leaves it open");
        assert_eq!(open_count(&disabled), 1);
        assert_eq!(close_count(&disabled), 1);
        assert!(
            disabled.starts_with(&enabled),
            "the closed block extends the same prompt"
        );
        assert_eq!(
            disabled.len(),
            enabled.len() + 4,
            "the empty block adds exactly the closing delimiters"
        );

        // Assistant history before the last user query renders the visible
        // reply only: the reasoning span is split off and dropped, so the
        // prompt for a thinking history turn equals the prompt for the same
        // turn with any other reasoning span.
        let history = |assistant: &str| {
            vec![
                ChatMessage {
                    role: ChatRole::Assistant,
                    content: assistant.to_string(),
                },
                ChatMessage {
                    role: ChatRole::User,
                    content: "ab".to_string(),
                },
            ]
        };
        let with_reasoning = model
            .prompt_token_ids_with_config(
                &history("<think>hidden chain</think>visible reply"),
                &generation_config(),
            )
            .expect("history prompt ids");
        let with_other_reasoning = model
            .prompt_token_ids_with_config(
                &history("<think>other span</think>visible reply"),
                &generation_config(),
            )
            .expect("history prompt ids");
        assert_eq!(
            with_reasoning, with_other_reasoning,
            "history reasoning spans are stripped from the prompt"
        );
        let with_other_reply = model
            .prompt_token_ids_with_config(
                &history("<think>other span</think>different reply"),
                &generation_config(),
            )
            .expect("history prompt ids");
        assert_ne!(
            with_reasoning, with_other_reply,
            "the visible reply itself is kept"
        );
        assert_eq!(
            open_count(&with_reasoning),
            1,
            "only the final generation block opens a think section"
        );
        assert_eq!(close_count(&with_reasoning), 0);

        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn json_object_constraint_keeps_fixture_decode_grammar_legal() {
        let (model, dir) = load_fixture("json-object");
        let messages = vec![ChatMessage {
            role: ChatRole::User,
            content: "ab".to_string(),
        }];
        let mut config = generation_config();
        config.constrain_json_object = true;

        let cache = physical_cache(&model, &DeviceProfile::cpu());
        let mut state = model
            .start_decode_state_physical(&messages, 24, &config, None, cache)
            .expect("constrained decode state");

        let mut assembled = String::new();
        for _ in 0..24 {
            let step = model.decode_step(&mut state).expect("constrained step");
            assembled.push_str(&step.delta);
            if step.finished {
                break;
            }
        }
        // Every committed token fed the grammar machine, so the assembled
        // text is always a legal JSON prefix (an empty stop-first output
        // included); an unconstrained random decode would emit arbitrary
        // byte-level characters the machine rejects.
        let mut machine = crate::models::shared::grammar::JsonGrammarMachine::new();
        machine
            .feed(&assembled)
            .expect("output stays grammar-legal");

        // The constrained decode still drains logprobs like the plain path:
        // exactly one entry per non-stop step, in the DS9.3 OpenAI shape.
        drop(state);
        let cache = physical_cache(&model, &DeviceProfile::cpu());
        let mut logged_config = config.clone();
        logged_config.logprobs = true;
        logged_config.top_logprobs = 2;
        let mut logged = model
            .start_decode_state_physical(&messages, 8, &logged_config, None, cache)
            .expect("logprob decode state");
        let mut drained = Vec::new();
        let mut steps_taken = 0usize;
        let mut finished = false;
        for _ in 0..8 {
            let step = model.decode_step(&mut logged).expect("logprob step");
            drained.extend(std::mem::take(&mut logged.pending_logprobs));
            steps_taken += 1;
            finished = step.finished;
            if finished {
                break;
            }
        }
        assert_eq!(
            drained.len(),
            steps_taken - usize::from(finished),
            "one logprob entry per non-stop decode step"
        );
        assert!(
            drained
                .iter()
                .all(|entry| entry.logprob.is_finite() && !entry.token.is_empty()),
            "logprob entries keep the DS9.3 OpenAI shape"
        );

        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn fixture_rejects_media_inputs() {
        let (model, dir) = load_fixture("media");
        let messages = vec![ChatMessage {
            role: ChatRole::User,
            content: "ab".to_string(),
        }];
        let mut config = generation_config();
        config
            .request
            .media_inputs
            .push(crate::models::shared::chat::ChatMediaInput {
                kind: crate::models::shared::chat::ChatMediaKind::Image,
                source: "data:image/png;base64,AAAA".to_string(),
            });
        let error = model
            .prepare_prompt_for_execution(&messages, &config)
            .expect_err("media must be rejected");
        assert!(format!("{error}").contains("text-only"));
        std::fs::remove_dir_all(dir).ok();
    }

    #[cfg(feature = "metal")]
    #[test]
    fn fixture_generation_matches_between_cpu_and_metal() {
        let Some(_metal_device) = crate::backends::metal_device_if_available(0) else {
            eprintln!("metal device unavailable; parity leg not run");
            return;
        };
        let run = |device: DeviceProfile| -> Vec<String> {
            let dir = fixture_dir("parity");
            write_fixture(&dir);
            let model =
                Qwen36MoeChatModel::load(&dir, ModelVariant::Qwen36Moe35BA3BFp8, device.clone())
                    .expect("load fixture");
            let messages = vec![ChatMessage {
                role: ChatRole::User,
                content: "ab".to_string(),
            }];
            let cache = physical_cache(&model, &device);
            let mut state = model
                .start_decode_state_physical(&messages, 8, &generation_config(), None, cache)
                .expect("decode state");
            let mut deltas = Vec::new();
            for _ in 0..8 {
                let step = model.decode_step(&mut state).expect("decode step");
                deltas.push(step.delta);
                if step.finished {
                    break;
                }
            }
            std::fs::remove_dir_all(&dir).ok();
            deltas
        };

        let cpu_deltas = run(DeviceProfile::cpu());
        let metal_profile = crate::backends::DeviceSelector::detect_for_preference(
            crate::backends::BackendPreference::Metal,
        )
        .expect("metal profile");
        let metal_deltas = run(metal_profile);
        assert_eq!(cpu_deltas, metal_deltas, "CPU and Metal must agree");
    }
}
