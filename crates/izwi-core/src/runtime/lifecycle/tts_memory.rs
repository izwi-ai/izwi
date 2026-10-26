//! Load admission for the TTS families (P2.1): the residency plan of the ASR
//! admission framework (`asr_memory`), opened to Qwen3-TTS, Kokoro,
//! VibeVoice-TTS, Voxtral-TTS and Chatterbox. Each plan prices the header census of the
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
use crate::models::architectures::chatterbox::config::{
    S3GEN_WEIGHTS_FILE, T3_WEIGHTS_FILE, VOICE_ENCODER_WEIGHTS_FILE,
};
use crate::models::architectures::chatterbox::perth::PERTH_CHECKPOINT_FILE;
use crate::models::architectures::chatterbox::CHATTERBOX_REFERENCE_VOICE_CACHE;
use crate::models::architectures::kokoro::{
    CHECKPOINT_FILE as KOKORO_CHECKPOINT_FILE, CHECKPOINT_SUBMODULE_KEYS as KOKORO_CHECKPOINT_KEYS,
    KOKORO_MODEL_MEMO_MAX_BYTES,
};
use crate::models::architectures::qwen3::core::qwen3_dense_projection_fusion_enabled;
use crate::models::architectures::qwen3::tts::speech_tokenizer::{
    encoder_reads_tensor, read_encoder_acoustic_codebooks,
};
use crate::models::architectures::qwen3::tts::{
    qwen3_tts_load_layout, qwen3_tts_text_embedding_reads_in_place, Qwen3TtsLoadLayout,
};
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
            | ModelFamily::ChatterboxTts
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
        ModelFamily::ChatterboxTts => chatterbox_tts_plan(model_path, backend),
        _ => Ok(None),
    }
}

/// What a TTS load reloads at when a lowered-precision component fails its
/// load-time probe, priced for the lifecycle to grow the load's
/// authorization to: Qwen3-TTS's talker and predictor dense, after their
/// weight-only F16 self-check fails (`load_weight_only_f16`). `None` for
/// every other TTS load.
pub(super) fn precision_reload_plan(
    variant: ModelVariant,
    model_path: &Path,
    device: &DeviceProfile,
) -> Result<Option<ModelResourcePlan>> {
    if variant.family() != ModelFamily::Qwen3Tts {
        return Ok(None);
    }
    let backend = BackendKind::from(device.kind);
    let Some(plan) = qwen3_tts_plan_with(model_path, device, backend, false)? else {
        return Ok(None);
    };
    let estimate = plan.estimate()?;
    tracing::info!(
        model = %variant,
        resident_bytes = estimate.resident_bytes,
        load_peak_bytes = estimate.load_peak_bytes,
        "Priced the dense reload of a weight-only F16 load whose probe failed"
    );
    plan.resource_plan().map(Some)
}

fn qwen3_tts_plan(
    model_path: &Path,
    device: &DeviceProfile,
    backend: BackendKind,
) -> Result<Option<AsrResidencyPlan>> {
    qwen3_tts_plan_with(model_path, device, backend, true)
}

/// [`qwen3_tts_plan`], with weight-only F16 off when `allow_f16` is unset
/// (the dense reload).
fn qwen3_tts_plan_with(
    model_path: &Path,
    device: &DeviceProfile,
    backend: BackendKind,
    allow_f16: bool,
) -> Result<Option<AsrResidencyPlan>> {
    if !model_path.join("config.json").is_file() {
        return Ok(None);
    }
    let mut layout = qwen3_tts_load_layout(model_path, device)?;
    layout.weight_only_f16 &= allow_f16;
    let paths = [
        model_path.join("model.safetensors"),
        model_path
            .join("speech_tokenizer")
            .join("model.safetensors"),
    ];
    let Some(files) = census_of(&paths)? else {
        return Ok(None);
    };
    let encoder_acoustic_codebooks = if layout.tokenizer_encoder {
        read_encoder_acoustic_codebooks(&model_path.join("speech_tokenizer"))?.unwrap_or(0)
    } else {
        0
    };
    price_qwen3_tts(&files, backend, &layout, encoder_acoustic_codebooks).map(Some)
}

/// Whether a Qwen3-TTS tensor loads as weight-only F16 when the layout asks
/// for it: the talker and predictor attention and MLP projections, the text
/// projection, the predictor's codebook heads and codec embeddings, and the
/// text embedding (F16 storage, F32 compute).
fn qwen3_tts_weight_only_f16_tensor(name: &str) -> bool {
    name.starts_with("talker.")
        && (name.ends_with("_proj.weight")
            || name.starts_with("talker.text_projection.")
            || name.starts_with("talker.code_predictor.lm_head.")
            || name.starts_with("talker.code_predictor.model.codec_embedding.")
            || name == QWEN3_TTS_TEXT_EMBEDDING)
}

const QWEN3_TTS_TEXT_EMBEDDING: &str = "talker.model.text_embedding.weight";

/// Qwen3-TTS (`qwen3/tts/mod.rs`): every tensor through
/// `staged::metal_staged_safetensors`, the talker and code predictor at
/// their dtypes, the ECAPA speaker encoder at F32 (voice clone only) and the
/// speech tokenizer at its dtype, its reference encoder for voice clone
/// only, without the acoustic codebooks past the `encoder_acoustic_codebooks`
/// its encode reads. On Metal the BF16 checkpoint is cast to F32 on the host
/// and uploaded at its exact size.
///
/// With weight-only F16 this is the F16 residency and the staged loader's
/// peak for that form. A failed load-time self-check drops the F16 model
/// and reloads dense, which is not priced: pricing every load at the dense
/// reload (2.3 times the F16 residency at 0.6B) refused 1.7B on a 16 GB Mac
/// for a fallback that almost never runs. A fallback that does not fit
/// fails closed at the load's observed-growth reconciliation (or Metal's
/// OOM retry).
fn price_qwen3_tts(
    files: &[CensusFile],
    backend: BackendKind,
    layout: &Qwen3TtsLoadLayout,
    encoder_acoustic_codebooks: usize,
) -> Result<AsrResidencyPlan> {
    let [model, tokenizer] = files else {
        return Err(Error::ModelLoadError(
            "Qwen3-TTS residency plan needs the model and speech-tokenizer censuses".into(),
        ));
    };
    let mut ledger = Ledger::staged(backend);
    for tensor in &model.tensors {
        let name = tensor.name.as_str();
        if layout.weight_only_f16 && qwen3_tts_weight_only_f16_tensor(name) {
            // Read on the host, converted there and uploaded once at F16
            // (biases stay F32). The text embedding stays on the host at F16
            // (only gathered rows are uploaded): a float one is converted
            // straight from the memory map, a quantized one passes through
            // F32.
            if name == QWEN3_TTS_TEXT_EMBEDDING {
                if matches!(tensor.dtype, StoredDType::Dense(dtype) if qwen3_tts_text_embedding_reads_in_place(dtype))
                {
                    ledger.host_kept(tensor, None, DType::F16)?;
                } else {
                    ledger.host_kept(tensor, Some(DType::F32), DType::F16)?;
                }
            } else if name.ends_with(".bias") {
                ledger.host_converted(tensor, DType::F32)?;
            } else {
                ledger.host_converted(tensor, DType::F16)?;
            }
            continue;
        }
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
        if tensor.name.starts_with("encoder.")
            && !(layout.tokenizer_encoder
                && encoder_reads_tensor(&tensor.name, encoder_acoustic_codebooks))
        {
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

/// Device bytes of one prepared Chatterbox voice (a reference mel of up to
/// 10 s, prompt tokens and two speaker vectors), well under this; a model
/// keeps the default voice and a bounded cache of cloned ones.
const CHATTERBOX_VOICE_BYTES: u64 = 1024 * 1024;

fn chatterbox_tts_plan(
    model_path: &Path,
    backend: BackendKind,
) -> Result<Option<AsrResidencyPlan>> {
    let paths = [
        T3_WEIGHTS_FILE,
        S3GEN_WEIGHTS_FILE,
        VOICE_ENCODER_WEIGHTS_FILE,
    ]
    .map(|file| model_path.join(file));
    let Some(mut files) = census_of(&paths)? else {
        return Ok(None);
    };
    let perth = model_path.join(PERTH_CHECKPOINT_FILE);
    if crate::env::flag_or("IZWI_CHATTERBOX_WATERMARK", true) && perth.is_file() {
        files.push(perth_encoder_census(&perth)?);
    }
    price_chatterbox_tts(&files, backend).map(Some)
}

/// The PerTh encoder's tensors in the checkpoint's `model` state dict (the
/// loader never reads the decoder).
fn perth_encoder_census(checkpoint: &Path) -> Result<CensusFile> {
    let infos = read_pth_tensor_info(checkpoint, false, Some("model")).map_err(|error| {
        Error::ModelLoadError(format!(
            "Cannot read the PerTh checkpoint {}: {error}",
            checkpoint.display()
        ))
    })?;
    let tensors = infos
        .into_iter()
        .filter(|info| info.name.starts_with("encoder."))
        .map(|info| {
            let elements = u64::try_from(info.layout.shape().elem_count()).unwrap_or(u64::MAX);
            CensusTensor {
                name: format!("perth.{}", info.name),
                dtype: StoredDType::Dense(info.dtype),
                shape: info.layout.shape().dims().to_vec(),
                stored_bytes: elements.saturating_mul(info.dtype.size_in_bytes() as u64),
            }
        })
        .collect();
    Ok(CensusFile {
        path: checkpoint.to_path_buf(),
        tensors,
    })
}

/// Chatterbox (`chatterbox/model.rs`): T3, S3Gen's flow and HiFT, the
/// reference encoders a cloned voice needs (voice encoder, S3 tokenizer,
/// CAMPPlus) and, when it watermarks, the PerTh encoder, read at their
/// stored F32 and kept F32 on every backend. Left out, as the loader never reads them: T3's
/// text head, the Llama trunk's own token table and CAMPPlus's batch-norm
/// step counters. HiFT's weight-norm halves and CAMPPlus's batch norms are
/// priced whole, a slight overestimate of what they fold into.
fn price_chatterbox_tts(files: &[CensusFile], backend: BackendKind) -> Result<AsrResidencyPlan> {
    const UNREAD: [&str; 2] = ["text_head.", "tfmr.embed_tokens."];
    let mut ledger = Ledger::new(backend);
    for tensor in files
        .iter()
        .flat_map(|file| file.tensors.iter())
        .filter(|tensor| {
            !UNREAD.iter().any(|prefix| tensor.name.starts_with(prefix))
                && !tensor.name.ends_with(".num_batches_tracked")
        })
    {
        // The PerTh encoder comes through a host buffer, like Kokoro's.
        let source = if tensor.name.starts_with("perth.") {
            Source::Copied
        } else {
            Source::Mapped
        };
        ledger.var(tensor, source, &[DType::F32])?;
    }
    let voices = 1 + CHATTERBOX_REFERENCE_VOICE_CACHE as u64;
    AsrResidencyPlan::from_ledger(ledger, voices * CHATTERBOX_VOICE_BYTES)
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
    price_voxtral_tts_as_loaded(&files, backend, &metadata.dtype_plan).map(Some)
}

/// The Voxtral-TTS plan as loaded: a lower-precision Metal tier is kept
/// only if its load-time self-check passes, else the transformers reload at
/// F32, so the load peak also fits that.
fn price_voxtral_tts_as_loaded(
    files: &[CensusFile],
    backend: BackendKind,
    dtypes: &VoxtralTtsDTypePlan,
) -> Result<AsrResidencyPlan> {
    let plan = price_voxtral_tts(files, backend, dtypes)?;
    if backend == BackendKind::Metal
        && (dtypes.language_model != DType::F32 || dtypes.acoustic_transformer != DType::F32)
    {
        let dense = price_voxtral_tts(
            files,
            backend,
            &VoxtralTtsDTypePlan {
                language_model: DType::F32,
                acoustic_transformer: DType::F32,
                codec: dtypes.codec,
            },
        )?;
        return Ok(plan.with_fallback_peak(&dense));
    }
    Ok(plan)
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

    /// Qwen3-TTS-12Hz encodes references with 16 of its 32 quantizers, one
    /// of them semantic (`speech_tokenizer/config.json`).
    const QWEN3_TTS_12HZ_ACOUSTIC_CODEBOOKS: usize = 15;

    fn qwen(files: &[CensusFile], voice_clone: bool, backend: BackendKind) -> AsrResidencyPlan {
        price_qwen3_tts(
            files,
            backend,
            &qwen_layout(voice_clone, backend),
            QWEN3_TTS_12HZ_ACOUSTIC_CODEBOOKS,
        )
        .unwrap()
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

    /// The published Chatterbox headers (`tests/fixtures/chatterbox`), recast
    /// as a residency fixture of the three files its loader reads.
    fn chatterbox_census() -> Vec<CensusFile> {
        let doc: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/chatterbox/tensor_census.json"
        ))
        .unwrap();
        let files = [
            T3_WEIGHTS_FILE,
            S3GEN_WEIGHTS_FILE,
            VOICE_ENCODER_WEIGHTS_FILE,
        ]
        .map(|name| {
            let mut total_bytes = 0u64;
            let tensors = doc["files"][name]["tensors"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(tensor, spec)| {
                    let elements = spec[1]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|dim| dim.as_u64().unwrap())
                        .product::<u64>();
                    total_bytes += elements * if spec[0] == "I64" { 8 } else { 4 };
                    serde_json::json!([tensor, spec[0], spec[1]])
                })
                .collect::<Vec<_>>();
            serde_json::json!({
                "path": name,
                "format": "safetensors",
                "total_bytes": total_bytes,
                "tensor_count": tensors.len(),
                "tensors": tensors,
            })
        });
        fixture(&serde_json::json!({ "files": files }).to_string())
    }

    /// Report 08 table B, resident weights per backend (CPU, Metal as
    /// allocated, CUDA), with the load peak each must stay under:
    /// - Qwen3-TTS CustomVoice/Base load F32 on CPU: 0.6B Base is 4.01 GiB,
    ///   the report's 4.04 less the 16 encoder codebooks its reference
    ///   encode never reads. On Metal the talker and predictor
    ///   projections, the predictor's codebook heads and codec embeddings
    ///   and the text embedding load as weight-only F16 (F32 compute),
    ///   host-cast and uploaded at exact size: 2.34 GiB, against
    ///   the 5.29 GiB the report saw from device F32 casts into
    ///   power-of-two buffers. CUDA keeps the talker and
    ///   predictor BF16 beside an F32 speech tokenizer. CustomVoice skips the
    ///   speech-tokenizer encoder (0.24 GiB) it never runs.
    /// - Kokoro is its 0.305 GiB F32 checkpoint read from the nested
    ///   `.pth`, not the 2 GiB catalog figure.
    /// - VibeVoice is F32 on CPU (10.07 GiB), F16 on Metal and BF16 on
    ///   CUDA (5.04 GiB exact); on Metal its fused LM projections are joined
    ///   at their exact size too (5.41 GiB allocated before, from 5.86).
    /// - Chatterbox is F32 everywhere: T3, S3Gen's flow and HiFT and the
    ///   reference encoders (all of its files but T3's text head), plus the
    ///   default voice and 16 cached clones: 2.99 GiB.
    #[test]
    fn plans_match_the_per_backend_residency_table() {
        let _env = env_lock();
        let chatterbox = chatterbox_census();
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
                [3.80, 2.12, 2.11],
            ),
            (
                "Qwen3-TTS-0.6B-Base",
                Box::new(|b| qwen(&qwen06_base, true, b)),
                [4.01, 2.34, 2.33],
            ),
            (
                "Qwen3-TTS-1.7B-CustomVoice",
                Box::new(|b| qwen(&qwen17_cv, false, b)),
                [7.57, 4.02, 4.00],
            ),
            ("Kokoro-82M", Box::new(kokoro), [0.30, 0.30, 0.30]),
            ("VibeVoice-1.5B", Box::new(vibevoice), [10.07, 5.04, 5.04]),
            (
                "Chatterbox-500M",
                Box::new(|b| price_chatterbox_tts(&chatterbox, b).unwrap()),
                [2.99, 2.99, 2.99],
            ),
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

    /// Weight-only F16 on Metal prices what the staged loader holds: the F16
    /// residency, with the text embedding's F16 table kept on the host where
    /// it was read in place from the BF16 checkpoint (only gathered rows are
    /// uploaded), so no table-sized transient sits above the residency
    /// (every other tensor is converted and uploaded one at a time at its
    /// exact size). `read_f16_embedding` on a BF16 file reports what it
    /// allocated per element. Measured on a 16 GB M1 Pro
    /// (`measure_metal_load_footprint`), 0.6B Base's Metal allocation peaked
    /// at 2,117,189,632 bytes; the price must cover it, without the dense
    /// reload (6.21 GB) a failed self-check falls back to, which refused 1.7B
    /// CustomVoice at 9.99 GB.
    #[test]
    fn qwen_weight_only_f16_prices_the_staged_f16_load() {
        let _env = env_lock();
        let host_bytes_per_element = {
            let (rows, cols) = (300, 16);
            let dir = std::env::temp_dir().join(format!(
                "izwi-qwen3-tts-embedding-price-{}",
                uuid::Uuid::new_v4()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join("model.safetensors");
            let table =
                candle_core::Tensor::ones((rows, cols), DType::BF16, &candle_core::Device::Cpu)
                    .unwrap();
            candle_core::safetensors::save(
                &std::collections::HashMap::from([(QWEN3_TTS_TEXT_EMBEDDING.to_string(), table)]),
                &path,
            )
            .unwrap();
            let read = crate::models::architectures::qwen3::tts::read_qwen3_tts_f16_text_embedding(
                &path,
                QWEN3_TTS_TEXT_EMBEDDING,
                rows,
                cols,
            )
            .unwrap()
            .unwrap();
            std::fs::remove_dir_all(&dir).ok();
            let elements = (rows * cols) as u64;
            assert_eq!(read.host_bytes % elements, 0);
            read.host_bytes / elements
        };
        for (file, files, voice_clone, measured_peak) in [
            (
                "0.6B Base",
                census!("qwen3_tts_12hz_0_6b_base.json"),
                true,
                Some(2_117_189_632u64),
            ),
            (
                "0.6B CustomVoice",
                census!("qwen3_tts_12hz_0_6b_custom_voice.json"),
                false,
                None,
            ),
            (
                "1.7B CustomVoice",
                census!("qwen3_tts_12hz_1_7b_custom_voice.json"),
                false,
                None,
            ),
        ] {
            let layout = qwen_layout(voice_clone, BackendKind::Metal);
            assert!(layout.weight_only_f16);
            let f16 = price_qwen3_tts(
                &files,
                BackendKind::Metal,
                &layout,
                QWEN3_TTS_12HZ_ACOUSTIC_CODEBOOKS,
            )
            .unwrap()
            .estimate()
            .unwrap();
            let dense = price_qwen3_tts(
                &files,
                BackendKind::Metal,
                &Qwen3TtsLoadLayout {
                    weight_only_f16: false,
                    ..layout
                },
                QWEN3_TTS_12HZ_ACOUSTIC_CODEBOOKS,
            )
            .unwrap()
            .estimate()
            .unwrap();
            assert!(f16.resident_bytes * 3 < dense.resident_bytes * 2, "{file}");
            let embedding = files[0]
                .tensors
                .iter()
                .find(|tensor| tensor.name == QWEN3_TTS_TEXT_EMBEDDING)
                .unwrap()
                .elements();
            assert!(
                f16.load_peak_bytes - f16.resident_bytes < embedding * host_bytes_per_element,
                "{file}: a table-sized transient sits above the residency"
            );
            assert!(f16.load_peak_bytes < dense.load_peak_bytes, "{file}");
            if let Some(measured) = measured_peak {
                assert!(f16.load_peak_bytes >= measured, "{file}");
            }
        }
        // 1.7B CustomVoice now asks for 4.58 GB, not 9.99 (dense reload
        // priced), 6.38 (embedding read through F32) or 5.13 (an F16 host
        // copy beside the uploaded table).
        let files = census!("qwen3_tts_12hz_1_7b_custom_voice.json");
        let layout = qwen_layout(false, BackendKind::Metal);
        let peak = price_qwen3_tts(
            &files,
            BackendKind::Metal,
            &layout,
            QWEN3_TTS_12HZ_ACOUSTIC_CODEBOOKS,
        )
        .unwrap()
        .estimate()
        .unwrap()
        .load_peak_bytes;
        assert!(peak <= 5_800_000_000, "{peak}");
    }

    /// The reference-audio components load only for voice cloning: a
    /// CustomVoice layout over the Base checkpoint drops exactly the
    /// speech-tokenizer encoder (stored F32) and the ECAPA speaker encoder,
    /// which Base prices at F32 (35 MB from its BF16 checkpoint). The
    /// encoder leaves out the 16 acoustic codebooks its 16-quantizer encode
    /// never reads (33.7 MB).
    #[test]
    fn qwen_tts_prices_the_reference_encoders_only_for_voice_clone() {
        let _env = env_lock();
        let base = census!("qwen3_tts_12hz_0_6b_base.json");
        let speaker_encoder = stored(&base, |tensor| tensor.name.starts_with("speaker_encoder."));
        let tokenizer_encoder = stored(&base[1..], |tensor| {
            tensor.name.starts_with("encoder.")
                && encoder_reads_tensor(&tensor.name, QWEN3_TTS_12HZ_ACOUSTIC_CODEBOOKS)
        });
        let unused_codebooks = stored(&base[1..], |tensor| {
            tensor.name.starts_with("encoder.")
                && !encoder_reads_tensor(&tensor.name, QWEN3_TTS_12HZ_ACOUSTIC_CODEBOOKS)
        });
        assert_eq!(speaker_encoder, 8_854_336 * 2);
        assert_eq!(unused_codebooks, 16 * (2_048 + 2_048 * 256 + 1) * 4);
        let clone = qwen(&base, true, BackendKind::Cpu).estimate().unwrap();
        let preset = price_qwen3_tts(
            &base,
            BackendKind::Cpu,
            &qwen_layout(false, BackendKind::Cpu),
            QWEN3_TTS_12HZ_ACOUSTIC_CODEBOOKS,
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
    /// The final Metal run's 0.6B CustomVoice weight load grew the device
    /// 2,408,677,376 bytes and passes. A dense reload after a failed F16
    /// self-check, which admission no longer prices, passes only when it
    /// fits what was authorized; since the F16 text embedding is read in
    /// place (no F32 transient), neither 0.6B's nor 1.7B's does, and each
    /// fails the load closed instead of running unpriced.
    #[test]
    fn reconciliation_accepts_the_f16_load_and_fails_an_unpriced_dense_reload() {
        let _env = env_lock();
        let reconcile = |files: &[CensusFile], variant: ModelVariant, dense_observed: bool| {
            let layout = qwen_layout(false, BackendKind::Metal);
            let price = |weight_only_f16| {
                price_qwen3_tts(
                    files,
                    BackendKind::Metal,
                    &Qwen3TtsLoadLayout {
                        weight_only_f16,
                        ..layout
                    },
                    QWEN3_TTS_12HZ_ACOUSTIC_CODEBOOKS,
                )
                .unwrap()
                .estimate()
                .unwrap()
            };
            let observed = if dense_observed {
                price(false).resident_bytes
            } else {
                2_408_677_376
            };
            let sample = |observed| super::super::DeviceUsageSample {
                observed,
                reserved: 0,
            };
            assert!(super::super::reconciles_device_usage(
                variant,
                BackendKind::Metal
            ));
            super::super::reconcile_observed_growth(
                variant,
                "weight load",
                sample(0),
                sample(observed),
                price(true).load_peak_bytes,
            )
        };
        let small = census!("qwen3_tts_12hz_0_6b_custom_voice.json");
        let large = census!("qwen3_tts_12hz_1_7b_custom_voice.json");
        let small_variant = ModelVariant::Qwen3Tts12Hz06BCustomVoice;
        let large_variant = ModelVariant::Qwen3Tts12Hz17BCustomVoice;
        assert_eq!(
            reconcile(&small, small_variant, false).unwrap(),
            2_408_677_376
        );
        assert!(reconcile(&small, small_variant, true).is_err());
        assert!(reconcile(&large, large_variant, true).is_err());
    }

    /// Fish keeps its own plan, and the Metal OOM retry now reads it too
    /// (it used the catalog figure): the shared lookup takes the Fish branch,
    /// which reads the checkpoint, rather than reporting "no plan".
    #[test]
    fn the_oom_retry_reads_fish_from_its_own_plan() {
        let _env = env_lock();
        let missing = temp_dir("fish-missing");
        let device = profile(BackendKind::Cpu);
        assert!(super::super::checkpoint_resource_plan(
            ModelVariant::FishAudioS2Pro,
            &missing,
            &device,
            &Default::default()
        )
        .is_err());
        std::fs::remove_dir_all(missing).unwrap();
    }

    /// P2.3: the TTS request envelope is held like the ASR one: one per
    /// execution slot, in the backend's domains (one memory on CPU and
    /// Metal, host and device on CUDA).
    #[test]
    fn request_envelopes_scale_by_slot_and_map_to_the_backend_domains() {
        use super::super::super::asr_envelope::AsrRequestEnvelope;
        let envelope = AsrRequestEnvelope {
            host_bytes: 300,
            accelerator_bytes: 700,
        }
        .scaled(2)
        .unwrap();
        assert_eq!(
            (envelope.host_bytes, envelope.accelerator_bytes),
            (600, 1_400)
        );
        let cpu = super::super::request_envelope_resources(BackendKind::Cpu, envelope).unwrap();
        assert_eq!(cpu.host_bytes, ResourceAmount::Known(2_000));
        let metal = super::super::request_envelope_resources(BackendKind::Metal, envelope).unwrap();
        assert_eq!(metal.unified_bytes, ResourceAmount::Known(2_000));
        let cuda = super::super::request_envelope_resources(BackendKind::Cuda, envelope).unwrap();
        assert_eq!(
            (cuda.host_bytes, cuda.device_bytes),
            (ResourceAmount::Known(600), ResourceAmount::Known(1_400))
        );
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

    /// Voxtral's F16 Metal tier prices F16 residency with a load peak that
    /// fits the F32 reload its self-check falls back to.
    #[test]
    fn voxtral_metal_f16_tier_prices_the_f32_fallback_peak() {
        let tensor = |name: &str| CensusTensor {
            name: name.into(),
            dtype: StoredDType::Dense(DType::BF16),
            shape: vec![1 << 20],
            stored_bytes: 2 << 20,
        };
        let files = [CensusFile {
            path: PathBuf::from("consolidated.safetensors"),
            tensors: vec![
                tensor("layers.0.attention.wq.weight"),
                tensor("acoustic_transformer.layers.0.wq.weight"),
            ],
        }];
        let f16 = VoxtralTtsDTypePlan {
            language_model: DType::F16,
            acoustic_transformer: DType::F16,
            codec: DType::F32,
        };
        let f32 = VoxtralTtsDTypePlan {
            language_model: DType::F32,
            acoustic_transformer: DType::F32,
            codec: DType::F32,
        };
        let loaded = price_voxtral_tts_as_loaded(&files, BackendKind::Metal, &f16)
            .unwrap()
            .estimate()
            .unwrap();
        let dense = price_voxtral_tts(&files, BackendKind::Metal, &f32)
            .unwrap()
            .estimate()
            .unwrap();
        assert!(loaded.resident_bytes < dense.resident_bytes);
        assert!(loaded.load_peak_bytes >= dense.load_peak_bytes);
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
                Some((&dir, &device, &Default::default())),
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
