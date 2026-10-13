//! Load admission for the TTS families (P2.1): the residency plan of the ASR
//! admission framework (`asr_memory`), opened to Qwen3-TTS, Kokoro,
//! VibeVoice-TTS and Voxtral-TTS. Each plan prices the header census of the
//! files its loader reads, in the dtype the loader's own resolver picks, with
//! Metal's power-of-two rounding of device-written buffers and the uploads
//! Candle holds until the load fence (see `asr_memory` for the rules). The
//! catalog's `memory_required_gb` stays a display figure for these variants.
//! Fish keeps its own plan; LFM2.5-Audio is priced as the ASR variant it
//! shares.
//!
//! Components a loader never reads are left out: the Qwen3-TTS
//! speech-tokenizer encoder and ECAPA speaker encoder outside voice-clone
//! checkpoints.
use std::path::{Path, PathBuf};

use candle_core::pickle::read_pth_tensor_info;
use candle_core::DType;

use super::asr_memory::{
    by_name, census_of, price_vibevoice, AsrResidencyPlan, Ledger, Source, VibeVoiceLoad,
};
use super::{CensusFile, CensusTensor, ModelResourcePlan, StoredDType};
use crate::backends::{BackendKind, DeviceProfile};
use crate::catalog::ModelFamily;
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::architectures::kokoro::{
    CHECKPOINT_FILE as KOKORO_CHECKPOINT_FILE, CHECKPOINT_SUBMODULE_KEYS as KOKORO_CHECKPOINT_KEYS,
    KOKORO_MODEL_MEMO_MAX_BYTES,
};
use crate::models::architectures::qwen3::core::qwen3_dense_projection_fusion_enabled;
use crate::models::architectures::qwen3::tts::{qwen3_tts_load_layout, Qwen3TtsLoadLayout};
use crate::models::architectures::vibevoice::tts::vibevoice_tts_weight_dtype;
use crate::models::architectures::voxtral::tts::model::{VoxtralTtsDTypePlan, VoxtralTtsModel};

/// Whether `variant` is a TTS family this module prices.
pub(super) fn has_plan(variant: ModelVariant) -> bool {
    matches!(
        variant.family(),
        ModelFamily::Qwen3Tts
            | ModelFamily::KokoroTts
            | ModelFamily::VibeVoiceTts
            | ModelFamily::VoxtralTts
    )
}

/// Admission for a load of the TTS `variant` checkpoint at `model_path` on
/// `device`, or `None` for a variant without a plan (or with weight files
/// missing, which the loader reports): such loads keep the generic estimate.
pub(super) fn resource_plan(
    variant: ModelVariant,
    model_path: &Path,
    device: &DeviceProfile,
) -> Result<Option<ModelResourcePlan>> {
    let Some(plan) = residency_plan(variant, model_path, device)? else {
        return Ok(None);
    };
    let estimate = plan.estimate()?;
    tracing::info!(
        model = %variant,
        backend = ?BackendKind::from(device.kind),
        resident_bytes = estimate.resident_bytes,
        load_peak_bytes = estimate.load_peak_bytes,
        "Priced TTS load from its residency plan"
    );
    plan.resource_plan().map(Some)
}

pub(super) fn residency_plan(
    variant: ModelVariant,
    model_path: &Path,
    device: &DeviceProfile,
) -> Result<Option<AsrResidencyPlan>> {
    let backend = BackendKind::from(device.kind);
    match variant.family() {
        ModelFamily::Qwen3Tts => qwen3_tts_plan(model_path, device, backend),
        ModelFamily::KokoroTts => kokoro_plan(model_path, backend),
        ModelFamily::VibeVoiceTts => vibevoice_tts_plan(model_path, device, backend),
        ModelFamily::VoxtralTts => voxtral_tts_plan(model_path, device, backend),
        _ => Ok(None),
    }
}

fn qwen3_tts_plan(
    model_path: &Path,
    device: &DeviceProfile,
    backend: BackendKind,
) -> Result<Option<AsrResidencyPlan>> {
    if !model_path.join("config.json").is_file() {
        return Ok(None);
    }
    let layout = qwen3_tts_load_layout(model_path, device)?;
    let paths = [
        model_path.join("model.safetensors"),
        model_path
            .join("speech_tokenizer")
            .join("model.safetensors"),
    ];
    let Some(files) = census_of(&paths)? else {
        return Ok(None);
    };
    price_qwen3_tts(&files, backend, &layout).map(Some)
}

/// Qwen3-TTS (`qwen3/tts/mod.rs`): every tensor through
/// `staged::metal_staged_safetensors`, the talker and code predictor at
/// their dtypes, the ECAPA speaker encoder at F32 (voice clone only) and the
/// speech tokenizer at its dtype, its reference encoder for voice clone
/// only. On Metal the BF16 checkpoint is cast to F32 on the host and
/// uploaded at its exact size.
fn price_qwen3_tts(
    files: &[CensusFile],
    backend: BackendKind,
    layout: &Qwen3TtsLoadLayout,
) -> Result<AsrResidencyPlan> {
    let [model, tokenizer] = files else {
        return Err(Error::ModelLoadError(
            "Qwen3-TTS residency plan needs the model and speech-tokenizer censuses".into(),
        ));
    };
    let mut ledger = Ledger::staged(backend);
    for tensor in &model.tensors {
        let name = tensor.name.as_str();
        let dtype = if name.starts_with("talker.code_predictor.") {
            layout.dtypes.code_predictor
        } else if name.starts_with("talker.") {
            layout.dtypes.talker
        } else if name.starts_with("speaker_encoder.") && layout.speaker_encoder {
            DType::F32
        } else {
            continue;
        };
        ledger.var(tensor, Source::Mapped, &[dtype])?;
    }
    for tensor in &tokenizer.tensors {
        if tensor.name.starts_with("encoder.") && !layout.tokenizer_encoder {
            continue;
        }
        ledger.var(tensor, Source::Mapped, &[layout.dtypes.speech_tokenizer])?;
    }
    AsrResidencyPlan::from_ledger(ledger, 0)
}

fn kokoro_plan(model_path: &Path, backend: BackendKind) -> Result<Option<AsrResidencyPlan>> {
    let checkpoint = model_path.join(KOKORO_CHECKPOINT_FILE);
    if !checkpoint.is_file() {
        return Ok(None);
    }
    let files = kokoro_census(&checkpoint)?;
    price_kokoro(&files, backend).map(Some)
}

/// The census of Kokoro's nested checkpoint: one file per submodule state
/// dict, which Candle lists only by key (the top level is a dict of dicts,
/// so an unkeyed read finds no tensors).
fn kokoro_census(checkpoint: &Path) -> Result<Vec<CensusFile>> {
    KOKORO_CHECKPOINT_KEYS
        .iter()
        .map(|key| {
            let infos = read_pth_tensor_info(checkpoint, false, Some(key)).map_err(|error| {
                Error::ModelLoadError(format!(
                    "Cannot read Kokoro checkpoint submodule '{key}' in {}: {error}",
                    checkpoint.display()
                ))
            })?;
            let tensors = infos
                .into_iter()
                .map(|info| {
                    let elements =
                        u64::try_from(info.layout.shape().elem_count()).unwrap_or(u64::MAX);
                    CensusTensor {
                        name: format!("{key}.{}", info.name),
                        dtype: StoredDType::Dense(info.dtype),
                        shape: info.layout.shape().dims().to_vec(),
                        stored_bytes: elements.saturating_mul(info.dtype.size_in_bytes() as u64),
                    }
                })
                .collect();
            Ok(CensusFile {
                path: PathBuf::from(format!("{}#{key}", checkpoint.display())),
                tensors,
            })
        })
        .collect()
}

/// Kokoro (`kokoro/mod.rs`): every submodule read from the PyTorch archive
/// through a host buffer at F32 (its stored dtype), plus the bounded voice
/// memo. Submodules reuse tensor names, so each file is priced whole.
fn price_kokoro(files: &[CensusFile], backend: BackendKind) -> Result<AsrResidencyPlan> {
    let mut ledger = Ledger::new(backend);
    for tensor in files.iter().flat_map(|file| file.tensors.iter()) {
        ledger.var(tensor, Source::Copied, &[DType::F32])?;
    }
    AsrResidencyPlan::from_ledger(ledger, KOKORO_MODEL_MEMO_MAX_BYTES)
}

fn vibevoice_tts_plan(
    model_path: &Path,
    device: &DeviceProfile,
    backend: BackendKind,
) -> Result<Option<AsrResidencyPlan>> {
    let Ok(paths) = crate::models::shared::weights::gguf::safetensors_weight_files(model_path)
    else {
        return Ok(None);
    };
    let Some(files) = census_of(&paths)? else {
        return Ok(None);
    };
    let load = VibeVoiceLoad::tts(
        vibevoice_tts_weight_dtype(device)?,
        qwen3_dense_projection_fusion_enabled(&device.device),
    );
    price_vibevoice(&files, backend, &load).map(Some)
}

fn voxtral_tts_plan(
    model_path: &Path,
    device: &DeviceProfile,
    backend: BackendKind,
) -> Result<Option<AsrResidencyPlan>> {
    let Ok(metadata) = VoxtralTtsModel::load_metadata(model_path, device.clone()) else {
        return Ok(None);
    };
    let Some(files) = census_of(&[model_path.join("consolidated.safetensors")])? else {
        return Ok(None);
    };
    price_voxtral_tts(&files, backend, &metadata.dtype_plan).map(Some)
}

/// Voxtral-TTS (`voxtral/tts/model.rs`): three `VarBuilder`s over the mapped
/// checkpoint, the flow-matching acoustic transformer and the codec at their
/// dtypes and everything else (the LM and its audio embeddings) at the LM
/// dtype.
fn price_voxtral_tts(
    files: &[CensusFile],
    backend: BackendKind,
    dtypes: &VoxtralTtsDTypePlan,
) -> Result<AsrResidencyPlan> {
    let mut ledger = Ledger::new(backend);
    for (name, tensor) in by_name(files) {
        let dtype = if name.starts_with("acoustic_transformer.") {
            dtypes.acoustic_transformer
        } else if name.starts_with("audio_tokenizer.") {
            dtypes.codec
        } else {
            dtypes.language_model
        };
        ledger.var(tensor, Source::Mapped, &[dtype])?;
    }
    AsrResidencyPlan::from_ledger(ledger, 0)
}

#[cfg(test)]
mod tests {
    use super::super::asr_memory::tests::{
        assert_gib, env_lock, fixture, gib, profile, stored, temp_dir, write_vibevoice_checkpoint,
    };
    use super::*;
    use crate::engine::ResourceAmount;
    use crate::models::architectures::qwen3::tts::tests::cache_test_config;

    const BACKENDS: [BackendKind; 3] = [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda];

    macro_rules! census {
        ($file:literal) => {
            fixture(include_str!(concat!(
                "../../../tests/fixtures/tts_residency/",
                $file
            )))
        };
    }

    /// The layout `Qwen3TtsModel::load` resolves for a CustomVoice (preset
    /// speakers) or Base (voice clone, with an ECAPA encoder) checkpoint.
    fn qwen_layout(voice_clone: bool, backend: BackendKind) -> Qwen3TtsLoadLayout {
        let mut config = cache_test_config();
        if voice_clone {
            config.tts_model_type = "base".into();
            config.speaker_encoder_config =
                Some(serde_json::from_value(serde_json::json!({"enc_dim": 1024})).unwrap());
        } else {
            config.talker_config.spk_id.insert("vivian".into(), 3_000);
        }
        Qwen3TtsLoadLayout::resolve(&config, &profile(backend)).unwrap()
    }

    fn qwen(files: &[CensusFile], voice_clone: bool, backend: BackendKind) -> AsrResidencyPlan {
        price_qwen3_tts(files, backend, &qwen_layout(voice_clone, backend)).unwrap()
    }

    fn kokoro(backend: BackendKind) -> AsrResidencyPlan {
        price_kokoro(&census!("kokoro_82m.json"), backend).unwrap()
    }

    fn vibevoice(backend: BackendKind) -> AsrResidencyPlan {
        let device = profile(backend);
        let load = VibeVoiceLoad::tts(
            vibevoice_tts_weight_dtype(&device).unwrap(),
            // `qwen3_dense_projection_fusion_enabled` on each backend's
            // device: fused on Metal.
            backend == BackendKind::Metal,
        );
        price_vibevoice(&census!("vibevoice_1_5b.json"), backend, &load).unwrap()
    }

    /// Report 08 table B, resident weights per backend (CPU, Metal as
    /// allocated, CUDA), with the load peak each must stay under:
    /// - Qwen3-TTS CustomVoice/Base load F32 on CPU and Metal: 0.6B Base is
    ///   4.04 GiB, as the report measured. The staged loader casts on the
    ///   host on Metal, so the 5.29 GiB the report saw from device casts
    ///   into power-of-two buffers (the 1.19 GiB text embedding took 2 GiB)
    ///   is gone. CUDA keeps the talker and predictor BF16 beside an F32
    ///   speech tokenizer. CustomVoice skips the speech-tokenizer encoder
    ///   (0.24 GiB) it never runs.
    /// - Kokoro is its 0.305 GiB F32 checkpoint read from the nested
    ///   `.pth`, not the 2 GiB catalog figure.
    /// - VibeVoice is F32 on CPU (10.07 GiB), F16 on Metal and BF16 on
    ///   CUDA (5.04 GiB exact); on Metal only its fused LM projections are
    ///   device-written (5.41 GiB allocated, from 5.86).
    #[test]
    fn plans_match_the_per_backend_residency_table() {
        let _env = env_lock();
        let qwen06_cv = census!("qwen3_tts_12hz_0_6b_custom_voice.json");
        let qwen06_base = census!("qwen3_tts_12hz_0_6b_base.json");
        let qwen17_cv = census!("qwen3_tts_12hz_1_7b_custom_voice.json");
        type Row<'a> = (
            &'a str,
            Box<dyn Fn(BackendKind) -> AsrResidencyPlan + 'a>,
            [f64; 3],
        );
        let rows: Vec<Row> = vec![
            (
                "Qwen3-TTS-0.6B-CustomVoice",
                Box::new(|b| qwen(&qwen06_cv, false, b)),
                [3.80, 3.80, 2.11],
            ),
            (
                "Qwen3-TTS-0.6B-Base",
                Box::new(|b| qwen(&qwen06_base, true, b)),
                [4.04, 4.04, 2.36],
            ),
            (
                "Qwen3-TTS-1.7B-CustomVoice",
                Box::new(|b| qwen(&qwen17_cv, false, b)),
                [7.57, 7.57, 4.00],
            ),
            ("Kokoro-82M", Box::new(kokoro), [0.30, 0.30, 0.30]),
            ("VibeVoice-1.5B", Box::new(vibevoice), [10.07, 5.41, 5.04]),
        ];
        for (label, plan, [cpu, metal, cuda]) in rows {
            let plans = BACKENDS.map(&plan);
            let [on_cpu, on_metal, on_cuda] = plans.map(|plan| plan.estimate().unwrap());
            eprintln!(
                "{label:28} resident (load peak) GiB: CPU {:.2} ({:.2}), Metal {:.2} ({:.2}), CUDA {:.2} ({:.2})",
                gib(on_cpu.resident_bytes),
                gib(on_cpu.load_peak_bytes),
                gib(on_metal.resident_bytes),
                gib(on_metal.load_peak_bytes),
                gib(on_cuda.resident_bytes),
                gib(on_cuda.load_peak_bytes),
            );
            assert_gib(&format!("{label} CPU"), on_cpu.resident_bytes, cpu);
            assert_gib(&format!("{label} Metal"), on_metal.resident_bytes, metal);
            assert_gib(&format!("{label} CUDA"), on_cuda.resident_bytes, cuda);
            for estimate in [on_cpu, on_metal, on_cuda] {
                assert!(
                    estimate.load_peak_bytes >= estimate.resident_bytes,
                    "{label}"
                );
            }
        }
    }

    /// The reference-audio components load only for voice cloning: a
    /// CustomVoice layout over the Base checkpoint drops exactly the
    /// speech-tokenizer encoder (stored F32) and the ECAPA speaker encoder,
    /// which Base prices at F32 (35 MB from its BF16 checkpoint).
    #[test]
    fn qwen_tts_prices_the_reference_encoders_only_for_voice_clone() {
        let _env = env_lock();
        let base = census!("qwen3_tts_12hz_0_6b_base.json");
        let speaker_encoder = stored(&base, |tensor| tensor.name.starts_with("speaker_encoder."));
        let tokenizer_encoder = stored(&base[1..], |tensor| tensor.name.starts_with("encoder."));
        assert_eq!(speaker_encoder, 8_854_336 * 2);
        let clone = qwen(&base, true, BackendKind::Cpu).estimate().unwrap();
        let preset = price_qwen3_tts(
            &base,
            BackendKind::Cpu,
            &qwen_layout(false, BackendKind::Cpu),
        )
        .unwrap()
        .estimate()
        .unwrap();
        assert_eq!(
            clone.resident_bytes - preset.resident_bytes,
            2 * speaker_encoder + tokenizer_encoder
        );
        let layout = qwen_layout(false, BackendKind::Metal);
        assert!(!layout.speaker_encoder && !layout.tokenizer_encoder);
        let layout = qwen_layout(true, BackendKind::Metal);
        assert!(layout.speaker_encoder && layout.tokenizer_encoder);
    }

    /// Kokoro's nested checkpoint is read per submodule: the plan is its F32
    /// payload plus the voice memo on every backend, where the generic path
    /// found no tensors and charged the 2 GiB catalog figure.
    #[test]
    fn kokoro_prices_its_nested_checkpoint_not_the_catalog() {
        let _env = env_lock();
        let files = census!("kokoro_82m.json");
        let payload = stored(&files, |_| true);
        assert_eq!(payload, 327_053_640);
        let catalog = super::super::model_memory_estimate(ModelVariant::Kokoro82M);
        for backend in BACKENDS {
            let estimate = kokoro(backend).estimate().unwrap();
            assert_eq!(
                estimate.resident_bytes,
                payload + KOKORO_MODEL_MEMO_MAX_BYTES,
                "{backend:?}"
            );
            assert!(estimate.load_peak_bytes < catalog.load_peak_bytes / 5);
        }
    }

    /// Voxtral-TTS splits one checkpoint across `VarBuilder`s: the acoustic
    /// transformer and the codec at their own dtypes, the rest at the LM's.
    /// CPU resolves F32 everywhere; Metal resolves the LM tier (F16) for the
    /// LM and acoustic transformer and F32 for the codec; CUDA resolves BF16.
    #[test]
    fn voxtral_tts_prices_each_stage_at_its_resolved_dtype() {
        use crate::models::architectures::voxtral::tts::model::select_voxtral_tts_dtypes;
        let _env = env_lock();
        let tensor = |name: &str, elements: usize| CensusTensor {
            name: name.into(),
            dtype: StoredDType::Dense(DType::BF16),
            shape: vec![elements],
            stored_bytes: elements as u64 * 2,
        };
        let files = [CensusFile {
            path: PathBuf::from("consolidated.safetensors"),
            tensors: vec![
                tensor("layers.0.attention.wq.weight", 1 << 20),
                tensor("acoustic_transformer.layers.0.wq.weight", 1 << 18),
                tensor("audio_tokenizer.decoder.0.weight", 1 << 16),
            ],
        }];
        let transformer_elements = (1u64 << 20) + (1 << 18);
        let codec_elements = 1u64 << 16;
        for (backend, transformer_bytes, codec_bytes) in [
            (BackendKind::Cpu, 4, 4),
            (BackendKind::Metal, 2, 4),
            (BackendKind::Cuda, 2, 2),
        ] {
            let dtypes = select_voxtral_tts_dtypes(&profile(backend), None).unwrap();
            let estimate = price_voxtral_tts(&files, backend, &dtypes)
                .unwrap()
                .estimate()
                .unwrap();
            // Power-of-two element counts: Metal's rounding changes nothing.
            assert_eq!(
                estimate.resident_bytes,
                transformer_elements * transformer_bytes + codec_elements * codec_bytes,
                "{backend:?}"
            );
        }
    }

    /// TTS loads are reconciled like ASR loads (6f60b6be): the weight load's
    /// observed growth must stay within its authorized peak plus tolerance.
    /// Fed the Qwen3-TTS 0.6B Metal footprint the admission report measured
    /// (about 5.2 GB of weights), the plan passes, while the 2.49 GB the
    /// checkpoint inventory used to charge would have failed the load.
    #[test]
    fn reconciliation_accepts_the_planned_qwen_weights_and_catches_the_old_price() {
        let _env = env_lock();
        let plan = qwen(
            &census!("qwen3_tts_12hz_0_6b_custom_voice.json"),
            false,
            BackendKind::Metal,
        );
        let authorized = plan.estimate().unwrap().load_peak_bytes;
        let sample = |observed| super::super::DeviceUsageSample {
            observed,
            reserved: 0,
        };
        let observed = 5_200_000_000;
        let variant = ModelVariant::Qwen3Tts12Hz06BCustomVoice;
        assert!(super::super::reconciles_device_usage(variant));
        assert_eq!(
            super::super::reconcile_observed_growth(
                variant,
                "weight load",
                sample(0),
                sample(observed),
                authorized,
            )
            .unwrap(),
            observed
        );
        assert!(super::super::reconcile_observed_growth(
            variant,
            "weight load",
            sample(0),
            sample(observed),
            2_493_807_108,
        )
        .is_err());
    }

    /// The runtime's load admission takes the TTS plan.
    #[test]
    fn runtime_admission_prices_tts_loads_by_their_plan() {
        let _authority = crate::engine::resources::test_support::authority_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _env = env_lock();
        let models_dir = temp_dir("tts-runtime");
        let checkpoint = models_dir.join("VibeVoice-1.5B");
        std::fs::create_dir_all(&checkpoint).unwrap();
        let bf16 = write_vibevoice_checkpoint(&checkpoint);
        let runtime = crate::runtime::service::RuntimeService::new(crate::config::EngineConfig {
            models_dir: models_dir.clone(),
            backend: crate::backends::BackendPreference::Cpu,
            ..crate::config::EngineConfig::default()
        })
        .unwrap();
        let admitted = runtime
            .model_lifecycle
            .model_resource_plan(ModelVariant::VibeVoice15BTts, &checkpoint)
            .unwrap();
        let device = runtime
            .model_lifecycle
            .backend_router
            .context()
            .device
            .clone();
        let plan = resource_plan(ModelVariant::VibeVoice15BTts, &checkpoint, &device)
            .unwrap()
            .unwrap();
        assert_eq!(admitted, plan);
        assert_eq!(
            admitted.resident_authorization.host_bytes,
            ResourceAmount::Known(2 * bf16)
        );
        drop(runtime);
        std::fs::remove_dir_all(models_dir).unwrap();
    }

    /// Admission prices a TTS load by its plan, not the catalog, and the
    /// Metal OOM retry judges futility by the same plan.
    #[test]
    fn admission_and_the_oom_retry_price_tts_loads_by_their_plan() {
        let _env = env_lock();
        let dir = temp_dir("tts-retry");
        let bf16 = write_vibevoice_checkpoint(&dir);
        let performance = crate::performance::PerformanceConfig::default();
        let variant = ModelVariant::VibeVoice15BTts;
        for backend in BACKENDS {
            let device = profile(backend);
            let plan = resource_plan(variant, &dir, &device)
                .unwrap()
                .expect("VibeVoice-TTS has a residency plan");
            let admitted = super::super::load_domain_bytes(plan.load_authorization, backend);
            let retry = super::super::retry_load_peak_bytes(
                variant,
                backend,
                &performance,
                Some((&dir, &device)),
            );
            assert_eq!(admitted, ResourceAmount::Known(retry), "{backend:?}");
            assert!(retry < super::super::model_memory_estimate(variant).load_peak_bytes);
            if backend == BackendKind::Cpu {
                // F32 on the CPU: twice the BF16 checkpoint.
                assert_eq!(
                    plan.resident_authorization.host_bytes,
                    ResourceAmount::Known(2 * bf16)
                );
            }
        }
        assert!(has_plan(variant));
        assert!(has_plan(ModelVariant::Kokoro82M));
        assert!(!has_plan(ModelVariant::FishAudioS2Pro));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
