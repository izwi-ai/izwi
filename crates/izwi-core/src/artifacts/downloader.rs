//! Model downloading from HuggingFace Hub
//!
//! Features:
//! - Async streaming downloads with byte-level progress
//! - Multi-progress bar terminal display
//! - Non-blocking downloads with background task spawning
//! - Real-time progress via channels

use std::collections::{BTreeSet, HashMap};
use std::io::SeekFrom;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use futures::StreamExt;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use reqwest;
use reqwest::header::{CONTENT_RANGE, ETAG, IF_RANGE, RANGE};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncSeekExt, AsyncWriteExt};
use tokio::sync::{broadcast, RwLock};
use tokio::task::JoinHandle;
use tokio::time::{Duration, Instant};
use tracing::{debug, info, warn};

use crate::catalog::ModelFamily;
use crate::catalog::ModelVariant;
use crate::error::{Error, Result};
use crate::models::architectures::chatterbox::chinese::{
    CANGJIE_FILE, PKUSEG_DICT_FILE, PKUSEG_MODEL_FILE,
};
use crate::models::architectures::chatterbox::perth::PERTH_CHECKPOINT_FILE;
use crate::models::architectures::granite_speech5::asr::{
    GRANITE_SPEECH5_GGUF_LM_FILE, GRANITE_SPEECH5_GGUF_MMPROJ_FILE,
};
use crate::models::architectures::nemotron::asr::nemotron_native_artifacts_complete;
use crate::models::architectures::parakeet::asr::parakeet_native_artifacts_complete;
use crate::models::shared::weights::nemo::ensure_free_disk;

const HF_BASE_URL: &str = "https://huggingface.co";
pub const ARTIFACT_MANIFEST_FILE: &str = "izwi-artifact.json";
/// On-disk cache of resolved expected download sizes, so cold starts do not
/// need a Hugging Face round-trip per catalog variant to render the model
/// list. Keyed by variant dir_name (stable across process restarts).
const EXPECTED_SIZES_CACHE_FILE: &str = "izwi-expected-sizes.json";

const QWEN38_REQUIRED_METADATA_FILES: &[&str] = &[
    "config.json",
    "generation_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "model.safetensors.index.json",
];

fn qwen_chat_gguf_filename(variant: ModelVariant) -> Option<&'static str> {
    match variant {
        ModelVariant::Qwen314BGguf => Some("Qwen3-14B-Q4_K_M.gguf"),
        ModelVariant::Qwen3Moe30bA3bGguf => Some("Qwen3-30B-A3B-Q4_K_M.gguf"),
        _ => None,
    }
}

fn qwen_asr_gguf_filename(variant: ModelVariant) -> Option<&'static str> {
    match variant {
        ModelVariant::Qwen3Asr06BGguf => Some("qwen3_asr_0.6b_q8_0.gguf"),
        ModelVariant::Qwen3Asr17BGguf => Some("qwen3_asr_1.7b_q8_0.gguf"),
        _ => None,
    }
}

fn qwen35_chat_gguf_filename(variant: ModelVariant) -> Option<&'static str> {
    match variant {
        ModelVariant::Qwen3508BGguf => Some("Qwen3.5-0.8B-Q4_K_M.gguf"),
        ModelVariant::Qwen352BGguf => Some("Qwen3.5-2B-Q4_K_M.gguf"),
        ModelVariant::Qwen354BGguf => Some("Qwen3.5-4B-Q4_K_M.gguf"),
        ModelVariant::Qwen359BGguf => Some("Qwen3.5-9B-Q4_K_M.gguf"),
        _ => None,
    }
}

fn lfm2_chat_gguf_filename(variant: ModelVariant) -> Option<&'static str> {
    match variant {
        ModelVariant::Lfm2512BInstructGguf => Some("LFM2.5-1.2B-Instruct-Q4_K_M.gguf"),
        ModelVariant::Lfm2512BThinkingGguf => Some("LFM2.5-1.2B-Thinking-Q4_K_M.gguf"),
        _ => None,
    }
}

fn lfm25_audio_gguf_filenames(variant: ModelVariant) -> Option<[&'static str; 4]> {
    match variant {
        ModelVariant::Lfm25Audio15BGguf => Some([
            "LFM2.5-Audio-1.5B-Q4_0.gguf",
            "mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf",
            "tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf",
            "vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf",
        ]),
        _ => None,
    }
}

const VOXTRAL_TTS_VOICE_EMBEDDINGS: &[&str] = &[
    "voice_embedding/ar_male.pt",
    "voice_embedding/casual_female.pt",
    "voice_embedding/casual_male.pt",
    "voice_embedding/cheerful_female.pt",
    "voice_embedding/de_female.pt",
    "voice_embedding/de_male.pt",
    "voice_embedding/es_female.pt",
    "voice_embedding/es_male.pt",
    "voice_embedding/fr_female.pt",
    "voice_embedding/fr_male.pt",
    "voice_embedding/hi_female.pt",
    "voice_embedding/hi_male.pt",
    "voice_embedding/it_female.pt",
    "voice_embedding/it_male.pt",
    "voice_embedding/neutral_female.pt",
    "voice_embedding/neutral_male.pt",
    "voice_embedding/nl_female.pt",
    "voice_embedding/nl_male.pt",
    "voice_embedding/pt_female.pt",
    "voice_embedding/pt_male.pt",
];

const VIBEVOICE_QWEN_TOKENIZER_FILES: &[&str] = &[
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
];

/// The English loader's files at the pinned revision (5bb1f6ee); the repo
/// also holds the multilingual and legacy checkpoints, which it never reads.
/// The PerTh watermarker comes from its own repository ([`PERTH_CHECKPOINT`]).
const CHATTERBOX_FILES: &[&str] = &[
    "t3_cfg.safetensors",
    "s3gen.safetensors",
    "ve.safetensors",
    "tokenizer.json",
    "conds.pt",
    PERTH_CHECKPOINT_FILE,
];

/// The multilingual edition's files: its T3 and grapheme tokenizer, then
/// the English S3Gen and voice encoder in place of the pickled `s3gen.pt`
/// and `ve.pt` upstream's multilingual loader reads (bit-identical tensor
/// for tensor at this revision; `chatterbox::multilingual_tests` checks),
/// the default voice, the watermark, and the Chinese frontend's Cangjie
/// table and pkuseg segmenter ([`PKUSEG_MODEL`], [`PKUSEG_DICT`]).
const CHATTERBOX_MULTILINGUAL_FILES: &[&str] = &[
    "t3_mtl23ls_v2.safetensors",
    "grapheme_mtl_merged_expanded_v1.json",
    "s3gen.safetensors",
    "ve.safetensors",
    "conds.pt",
    PERTH_CHECKPOINT_FILE,
    CANGJIE_FILE,
    PKUSEG_MODEL_FILE,
    PKUSEG_DICT_FILE,
];

fn chatterbox_files(variant: ModelVariant) -> &'static [&'static str] {
    if variant == ModelVariant::ChatterboxMultilingual500M {
        CHATTERBOX_MULTILINGUAL_FILES
    } else {
        CHATTERBOX_FILES
    }
}

const FISH_S2_PRO_FILES: &[&str] = &[
    "README.md",
    "LICENSE.md",
    "chat_template.jinja",
    "codec.pth",
    "config.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
];

fn vibevoice_tokenizer_repo(variant: ModelVariant) -> Option<&'static str> {
    match variant {
        ModelVariant::VibeVoice15BTts => Some("Qwen/Qwen2.5-1.5B"),
        ModelVariant::VibeVoiceAsr => Some("Qwen/Qwen2.5-7B"),
        _ => None,
    }
}

fn vibevoice_shard_count(variant: ModelVariant) -> Option<usize> {
    match variant {
        ModelVariant::VibeVoice15BTts => Some(3),
        ModelVariant::VibeVoiceAsr => Some(8),
        _ => None,
    }
}

fn vibevoice_safetensor_shards(variant: ModelVariant) -> Vec<String> {
    let total = vibevoice_shard_count(variant).expect("checked by VibeVoice family");
    (1..=total)
        .map(|idx| format!("model-{idx:05}-of-{total:05}.safetensors"))
        .collect()
}

const GRANITE_SPEECH_PLUS_FILES: &[&str] = &[
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
];

/// Base ships its audio settings in `preprocessor_config.json` and no
/// generation config. Its `out_llm.safetensors` (a CTC head IBM uses only
/// for self-speculative decoding), `multilingual_sample.wav` and
/// `.eval_results/` are not needed to serve it.
const GRANITE_SPEECH_BASE_FILES: &[&str] = &[
    "chat_template.jinja",
    "config.json",
    "model.safetensors.index.json",
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
];

/// Granite Speech 5.0 ships one unsharded `model.safetensors` and its audio
/// settings in `preprocessor_config.json`. `out_llm.safetensors` (a BPE CTC
/// head for self-speculative decoding), the sample WAV and the card assets
/// are not needed to serve it; `tokenizer.json` covers the BPE vocabulary.
const GRANITE_SPEECH_5_FILES: &[&str] = &[
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
];

/// The exact size of Granite Speech 5.0's `model.safetensors` at the pinned
/// revision (LFS sha256 `4151269881d2dd82…`).
const GRANITE_SPEECH_5_MODEL_BYTES: u64 = 4_626_528_224;

/// Granite Speech 5.0's GGUF release: the two GGUFs from the GGUF repo, and
/// the configs and tokenizer the loader reads from the safetensors release
/// (the GGUF repo ships neither the processor configs nor `tokenizer.json`).
const GRANITE_SPEECH_5_GGUF_FILES: &[&str] = &[
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    GRANITE_SPEECH5_GGUF_LM_FILE,
    GRANITE_SPEECH5_GGUF_MMPROJ_FILE,
];

/// Exact sizes of the GGUF release's files at the pinned revision.
const GRANITE_SPEECH_5_GGUF_LM_BYTES: u64 = 1_956_155_104;
const GRANITE_SPEECH_5_GGUF_MMPROJ_BYTES: u64 = 1_159_354_752;

fn granite_speech_required_files(variant: ModelVariant) -> &'static [&'static str] {
    if variant.is_granite_speech5_gguf() {
        return GRANITE_SPEECH_5_GGUF_FILES;
    }
    if variant.family() == ModelFamily::GraniteSpeech5Asr {
        return GRANITE_SPEECH_5_FILES;
    }
    match variant {
        ModelVariant::GraniteSpeech412B => GRANITE_SPEECH_BASE_FILES,
        _ => GRANITE_SPEECH_PLUS_FILES,
    }
}

/// Exact shard sizes from the HF trees (base de575db6, plus e69efe37). The
/// shards are named `model-0000N-of-00003.safetensors`, which the generic
/// `model.safetensors` estimate never matched.
fn granite_speech_shard_size(variant: ModelVariant, file: &str) -> Option<u64> {
    let shard = file
        .strip_prefix("model-")?
        .strip_suffix("-of-00003.safetensors")?;
    let sizes: [u64; 3] = match variant {
        ModelVariant::GraniteSpeech412B => [2_143_518_808, 2_143_963_456, 339_045_512],
        ModelVariant::GraniteSpeech412BPlus => [1_994_213_120, 1_996_752_408, 232_907_080],
        _ => return None,
    };
    match shard {
        "00001" => Some(sizes[0]),
        "00002" => Some(sizes[1]),
        "00003" => Some(sizes[2]),
        _ => None,
    }
}

fn granite_speech_files(variant: ModelVariant) -> Vec<String> {
    granite_speech_required_files(variant)
        .iter()
        .map(|file| (*file).to_string())
        .collect()
}

#[derive(Debug, Deserialize)]
struct HfRepoTreeEntry {
    path: String,
    size: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

/// Reproducibility record written only after a revision-pinned bundle has
/// been downloaded completely.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactManifest {
    pub schema_version: u32,
    pub variant: ModelVariant,
    pub repo_id: String,
    pub revision: String,
    pub files: Vec<String>,
}

pub fn read_artifact_manifest(model_dir: &Path) -> Result<Option<ArtifactManifest>> {
    let path = model_dir.join(ARTIFACT_MANIFEST_FILE);
    if !path.exists() {
        return Ok(None);
    }
    let bytes = std::fs::read(path)?;
    serde_json::from_slice(&bytes)
        .map(Some)
        .map_err(Error::from)
}

fn indexed_safetensor_shards(index_bytes: &[u8]) -> Result<Vec<String>> {
    let index: SafetensorsIndex = serde_json::from_slice(index_bytes).map_err(|error| {
        Error::DownloadError(format!("Invalid model.safetensors.index.json: {error}"))
    })?;
    if index.weight_map.is_empty() {
        return Err(Error::DownloadError(
            "model.safetensors.index.json has an empty weight_map".to_string(),
        ));
    }

    let mut shards = BTreeSet::new();
    for shard in index.weight_map.values() {
        validate_relative_safetensor_path(shard)?;
        shards.insert(shard.clone());
    }
    Ok(shards.into_iter().collect())
}

fn validate_relative_safetensor_path(path: &str) -> Result<()> {
    let invalid_segment = path
        .split('/')
        .any(|segment| segment.is_empty() || segment == "." || segment == "..");
    let parsed = Path::new(path);
    let normal_components_only = parsed
        .components()
        .all(|component| matches!(component, Component::Normal(_)));
    let is_safetensors = parsed.extension().is_some_and(|ext| ext == "safetensors");

    if path.is_empty()
        || path.contains('\\')
        || path.contains(':')
        || parsed.is_absolute()
        || invalid_segment
        || !normal_components_only
        || !is_safetensors
    {
        return Err(Error::DownloadError(format!(
            "Unsafe Safetensors shard path in model index: {path:?}"
        )));
    }
    Ok(())
}

fn qwen38_selected_files(index_bytes: &[u8]) -> Result<Vec<String>> {
    let mut files = QWEN38_REQUIRED_METADATA_FILES
        .iter()
        .map(|file| (*file).to_string())
        .collect::<Vec<_>>();
    files.extend(indexed_safetensor_shards(index_bytes)?);
    Ok(files)
}

const QWEN35_MOE_REQUIRED_METADATA_FILES: &[&str] = &[
    "config.json",
    "generation_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "model.safetensors.index.json",
];

fn qwen35_moe_selected_files(index_bytes: &[u8]) -> Result<Vec<String>> {
    let mut files = QWEN35_MOE_REQUIRED_METADATA_FILES
        .iter()
        .map(|file| (*file).to_string())
        .collect::<Vec<_>>();
    files.extend(indexed_safetensor_shards(index_bytes)?);
    Ok(files)
}

fn qwen35_moe_bundle_is_complete(model_dir: &Path) -> bool {
    let index_bytes = match std::fs::read(model_dir.join("model.safetensors.index.json")) {
        Ok(bytes) => bytes,
        Err(_) => return false,
    };
    let mut selected_files = match qwen35_moe_selected_files(&index_bytes) {
        Ok(files) => files,
        Err(_) => return false,
    };
    if !selected_files
        .iter()
        .all(|file| model_dir.join(file).is_file())
    {
        return false;
    }

    let manifest = match read_artifact_manifest(model_dir) {
        Ok(Some(manifest)) => manifest,
        _ => return false,
    };
    let mut recorded_files = manifest.files.clone();
    recorded_files.sort();
    recorded_files.dedup();
    selected_files.sort();
    selected_files.dedup();

    manifest.schema_version == 1
        && manifest.variant == ModelVariant::Qwen36Moe35BA3BFp8
        && manifest.repo_id == ModelVariant::Qwen36Moe35BA3BFp8.repo_id()
        && manifest.revision == ModelVariant::QWEN36_MOE_35B_A3B_FP8_ARTIFACT_REVISION
        && recorded_files == selected_files
}

fn qwen38_bundle_is_complete(model_dir: &Path) -> bool {
    let index_bytes = match std::fs::read(model_dir.join("model.safetensors.index.json")) {
        Ok(bytes) => bytes,
        Err(_) => return false,
    };
    let mut selected_files = match qwen38_selected_files(&index_bytes) {
        Ok(files) => files,
        Err(_) => return false,
    };
    if !selected_files
        .iter()
        .all(|file| model_dir.join(file).is_file())
    {
        return false;
    }

    let manifest = match read_artifact_manifest(model_dir) {
        Ok(Some(manifest)) => manifest,
        _ => return false,
    };
    let mut recorded_files = manifest.files.clone();
    recorded_files.sort();
    recorded_files.dedup();
    selected_files.sort();
    selected_files.dedup();

    manifest.schema_version == 1
        && manifest.variant == ModelVariant::Qwen3827BFp8
        && manifest.repo_id == ModelVariant::Qwen3827BFp8.repo_id()
        && manifest.revision == ModelVariant::QWEN38_27B_FP8_ARTIFACT_REVISION
        && recorded_files == selected_files
}

/// Sent with every Hugging Face request; follows the release version.
const USER_AGENT: &str = concat!("izwi/", env!("CARGO_PKG_VERSION"));

/// The NeMo ASR families unpack their `.nemo` archive beside it on first
/// load, so their checkpoint needs about twice the download's disk.
fn extracts_nemo_on_load(variant: ModelVariant) -> bool {
    matches!(
        variant.family(),
        ModelFamily::ParakeetAsr | ModelFamily::NemotronAsr
    )
}

/// Free disk a model needs for a download of `download_bytes`, counting the
/// checkpoint a `.nemo` unpacks on first load.
pub fn download_disk_required_bytes(variant: ModelVariant, download_bytes: u64) -> u64 {
    if extracts_nemo_on_load(variant) {
        download_bytes.saturating_mul(2)
    } else {
        download_bytes
    }
}

/// Bytes a download still writes: every `(file, expected bytes, present)`
/// not yet present, and for a `.nemo` the family unpacks on first load, the
/// extracted checkpoint (about the archive's size) beside it.
fn download_disk_requirement(files: &[(&str, u64, bool)], extracts_nemo: bool) -> u64 {
    files
        .iter()
        .filter(|(_, _, present)| !*present)
        .map(|(file, bytes, _)| {
            if extracts_nemo && file.ends_with(".nemo") {
                bytes.saturating_mul(2)
            } else {
                *bytes
            }
        })
        .fold(0u64, u64::saturating_add)
}

/// `error` and its causes, without repeating a cause its parent already
/// quotes. reqwest's `Display` leaves out the cause (refused connection, DNS or
/// TLS failure), which is the useful part of a failed download.
/// The proxy from `HTTPS_PROXY` (or `https_proxy`, `ALL_PROXY`), with the
/// `NO_PROXY` exceptions; `None` when none is set.
fn https_proxy_from_env() -> Result<Option<reqwest::Proxy>> {
    let Some(url) = ["HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"]
        .iter()
        .find_map(|name| {
            std::env::var(name)
                .ok()
                .filter(|value| !value.trim().is_empty())
        })
    else {
        return Ok(None);
    };
    // The URL may carry credentials, so the error does not repeat it.
    let proxy = reqwest::Proxy::https(url.trim())
        .map_err(|_| Error::HfHubError("HTTPS_PROXY is not a valid proxy URL".to_string()))?
        .no_proxy(reqwest::NoProxy::from_env());
    Ok(Some(proxy))
}

fn error_chain(error: &dyn std::error::Error) -> String {
    let mut message = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        let cause_text = cause.to_string();
        if !message.contains(&cause_text) {
            message.push_str(": ");
            message.push_str(&cause_text);
        }
        source = cause.source();
    }
    message
}

/// A file served outside the Hub from a fixed URL, checked against its size
/// and SHA-256 once it lands: Chatterbox's PerTh watermarker and its
/// multilingual edition's Chinese segmenter ship only on GitHub.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PinnedFile {
    url: &'static str,
    size: u64,
    sha256: &'static str,
}

/// resemble-ai/Perth at ff1c8ac5 (MIT): the implicit watermarker's
/// checkpoint, a regular Git blob served by commit.
const PERTH_CHECKPOINT: PinnedFile = PinnedFile {
    url: "https://raw.githubusercontent.com/resemble-ai/Perth/\
          ff1c8ac55a976971245cdd53c18d6131ca00d993/src/perth/perth_net/pretrained/implicit/\
          perth_net_250000.pth.tar",
    size: 37_429_684,
    sha256: "a15bce457ebc53ce5e6c9c3f11df78cf7ee2bf9cdab0a798902135b4c4027670",
};

/// spacy-pkuseg's default model, `spacy_ontonotes` (MIT): the release asset
/// `pkuseg()` downloads, at the SHA-256 pkuseg itself checks.
const PKUSEG_MODEL: PinnedFile = PinnedFile {
    url: "https://github.com/explosion/spacy-pkuseg/releases/download/v0.0.26/spacy_ontonotes.zip",
    size: 34_567_143,
    sha256: "b216e7f92de7ae285aeab8feba2faa8ea8216e5995ff6fb3d391cc8356db1bfe",
};

/// spacy-pkuseg's default merge dictionary, `spacy_pkuseg/dicts/default.pkl`,
/// at the 1.0.2 release commit (the same blob since v0.0.26).
const PKUSEG_DICT: PinnedFile = PinnedFile {
    url: "https://raw.githubusercontent.com/explosion/spacy-pkuseg/\
          16995617f3b8f5cf32e5b6a77d7d800e201b255b/spacy_pkuseg/dicts/default.pkl",
    size: 4_525_959,
    sha256: "3cf450d95719d51883fa49291302b9032372064d86805b130bb2ea0223196c67",
};

/// The pinned source of `variant`'s `local_file`, when it is not on the Hub.
fn pinned_file(variant: ModelVariant, local_file: &str) -> Option<PinnedFile> {
    if variant.family() != ModelFamily::ChatterboxTts {
        return None;
    }
    match local_file {
        PERTH_CHECKPOINT_FILE => Some(PERTH_CHECKPOINT),
        PKUSEG_MODEL_FILE if variant == ModelVariant::ChatterboxMultilingual500M => {
            Some(PKUSEG_MODEL)
        }
        PKUSEG_DICT_FILE if variant == ModelVariant::ChatterboxMultilingual500M => {
            Some(PKUSEG_DICT)
        }
        _ => None,
    }
}

/// Whether `path` holds exactly `pinned`'s bytes; a file that does not is
/// removed.
async fn verify_pinned_file(path: &Path, pinned: PinnedFile) -> Result<()> {
    use sha2::{Digest, Sha256};
    let bytes = tokio::fs::read(path).await?;
    let digest = Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    if digest != pinned.sha256 {
        let _ = tokio::fs::remove_file(path).await;
        return Err(Error::DownloadError(format!(
            "{} does not match its pinned SHA-256 (got {digest}, expected {}); removed it",
            path.display(),
            pinned.sha256
        )));
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct FileDownloadPlan {
    source_repo: String,
    source_revision: String,
    source_file: String,
    local_file: String,
    expected_size: u64,
    strict_size_check: bool,
    /// Served from this URL instead of the Hub.
    pinned: Option<PinnedFile>,
}

impl FileDownloadPlan {
    /// A plan for a pinned file: its own URL and exact size.
    fn pinned(spec: &ModelFileSpec, pinned: PinnedFile) -> Self {
        Self {
            source_repo: spec.source_repo.clone(),
            source_revision: spec.source_revision.clone(),
            source_file: spec.source_file.clone(),
            local_file: spec.local_file.clone(),
            expected_size: pinned.size,
            strict_size_check: true,
            pinned: Some(pinned),
        }
    }

    /// The file's `resolve` URL on the Hub at `base`, or its pinned URL.
    fn url(&self, base: &str) -> String {
        if let Some(pinned) = self.pinned {
            return pinned.url.to_string();
        }
        format!(
            "{base}/{}/resolve/{}/{}",
            self.source_repo, self.source_revision, self.source_file
        )
    }

    /// A revision named by commit serves the same bytes forever, as does a
    /// pinned URL.
    fn immutable(&self) -> bool {
        self.pinned.is_some()
            || (self.source_revision.len() == 40
                && self
                    .source_revision
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit()))
    }

    /// The size the file must have, when the plan knows it exactly.
    fn exact_size(&self) -> Option<u64> {
        (self.strict_size_check && self.expected_size > 0).then_some(self.expected_size)
    }
}

/// A connection that delivers nothing for this long is dropped and the
/// download resumed. There is no limit on a whole transfer: a multi-GB file
/// on a slow link may take hours.
const DOWNLOAD_STALL_TIMEOUT: Duration = Duration::from_secs(60);
const DOWNLOAD_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
/// A file download fails after this many attempts in a row add no bytes.
const MAX_STALLED_FILE_ATTEMPTS: usize = 4;
const INITIAL_RETRY_BACKOFF: Duration = Duration::from_millis(400);
const MAX_RETRY_BACKOFF: Duration = Duration::from_secs(30);

/// What the bytes in a `.part` file are, so an attempt resumes them only
/// from the same source: the URL names the repository, revision and file;
/// the ETag, sent back as `If-Range`, catches a file that changed under an
/// unpinned revision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct PartialDownload {
    url: String,
    /// The server's strong validator for the bytes.
    etag: Option<String>,
    /// The whole file's length, when the server reported it.
    total: Option<u64>,
}

/// Where a file transfer reports its bytes.
#[derive(Default)]
struct FileProgress<'a> {
    bar: Option<&'a ProgressBar>,
    /// The channel, and the model-level progress the file's bytes add to.
    channel: Option<(
        &'a broadcast::Sender<DownloadProgress>,
        &'a DownloadProgress,
    )>,
}

/// `dest`'s partial file and the sidecar describing its bytes.
fn partial_paths(dest: &Path) -> Result<(PathBuf, PathBuf)> {
    let name = dest
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            Error::DownloadError(format!("Invalid download destination: {}", dest.display()))
        })?;
    Ok((
        dest.with_file_name(format!(".{name}.part")),
        dest.with_file_name(format!(".{name}.part.json")),
    ))
}

async fn file_len(path: &Path) -> u64 {
    tokio::fs::metadata(path)
        .await
        .map_or(0, |metadata| metadata.len())
}

/// The partial file's length and the ETag its bytes were served with.
async fn partial_state(partial_path: &Path, sidecar_path: &Path) -> (u64, Option<String>) {
    let etag = tokio::fs::read(sidecar_path)
        .await
        .ok()
        .and_then(|bytes| serde_json::from_slice::<PartialDownload>(&bytes).ok())
        .and_then(|sidecar| sidecar.etag);
    (file_len(partial_path).await, etag)
}

/// The byte to resume `url` from, when the partial file holds bytes of the
/// same source that can be checked: under a commit revision, or with a
/// validator the server can compare.
async fn resumable_partial(
    partial_path: &Path,
    sidecar_path: &Path,
    url: &str,
    immutable: bool,
    exact_size: Option<u64>,
) -> Option<(u64, PartialDownload)> {
    let offset = file_len(partial_path).await;
    let sidecar: PartialDownload =
        serde_json::from_slice(&tokio::fs::read(sidecar_path).await.ok()?).ok()?;
    let fits = sidecar
        .total
        .or(exact_size)
        .is_none_or(|total| offset <= total);
    (offset > 0 && sidecar.url == url && (immutable || sidecar.etag.is_some()) && fits)
        .then_some((offset, sidecar))
}

async fn discard_partial(partial_path: &Path, sidecar_path: &Path) {
    let _ = tokio::fs::remove_file(partial_path).await;
    let _ = tokio::fs::remove_file(sidecar_path).await;
}

/// Move a complete partial file to `dest`.
async fn publish_partial(partial_path: &Path, sidecar_path: &Path, dest: &Path) -> Result<()> {
    if let Err(error) = tokio::fs::rename(partial_path, dest).await {
        discard_partial(partial_path, sidecar_path).await;
        return Err(Error::from(error));
    }
    let _ = tokio::fs::remove_file(sidecar_path).await;
    Ok(())
}

/// `Content-Range: bytes <first>-<last>/<total>` as `(Some(first), total)`,
/// and the unsatisfied form `bytes */<total>` as `(None, total)`; the total
/// is `None` when the server writes `*`.
fn parse_content_range(value: &str) -> Option<(Option<u64>, Option<u64>)> {
    let (range, total) = value.trim().strip_prefix("bytes ")?.split_once('/')?;
    let total = match total.trim() {
        "*" => None,
        total => Some(total.parse().ok()?),
    };
    let first = match range.trim() {
        "*" => None,
        range => Some(range.split_once('-')?.0.trim().parse().ok()?),
    };
    Some((first, total))
}

#[derive(Debug, Clone)]
struct ModelFileSpec {
    source_repo: String,
    source_revision: String,
    source_file: String,
    local_file: String,
}

/// Progress update for model downloads
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    pub variant: ModelVariant,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub current_file: String,
    pub current_file_downloaded: u64,
    pub current_file_total: u64,
    pub files_completed: usize,
    pub files_total: usize,
}

impl DownloadProgress {
    /// Overall download percentage (0-100)
    pub fn total_percent(&self) -> f32 {
        if self.total_bytes > 0 {
            (self.downloaded_bytes as f32 / self.total_bytes as f32) * 100.0
        } else {
            0.0
        }
    }

    /// Current file percentage (0-100)
    pub fn file_percent(&self) -> f32 {
        if self.current_file_total > 0 {
            (self.current_file_downloaded as f32 / self.current_file_total as f32) * 100.0
        } else {
            0.0
        }
    }
}

/// Download state for tracking model downloads
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DownloadState {
    NotDownloaded,
    Downloading,
    Downloaded,
    Error,
}

/// Active download tracking
#[derive(Debug)]
struct ActiveDownload {
    handle: JoinHandle<Result<PathBuf>>,
    progress_tx: broadcast::Sender<DownloadProgress>,
}

impl ActiveDownload {
    /// Subscribe to progress updates
    fn subscribe(&self) -> broadcast::Receiver<DownloadProgress> {
        self.progress_tx.subscribe()
    }
}

/// Shared download state across threads
#[derive(Debug, Clone, Default)]
pub struct DownloadStateManager {
    state: Arc<RwLock<std::collections::HashMap<ModelVariant, DownloadState>>>,
    /// Why each failed download failed, until the next attempt.
    errors: Arc<RwLock<std::collections::HashMap<ModelVariant, String>>>,
}

/// In-memory expected-size map backed by a JSON file next to the models dir.
/// Reads consult memory only; fresh Hugging Face resolutions append to the
/// file so the next cold start renders sizes without any network round-trip.
#[derive(Debug, Clone)]
struct ExpectedSizeStore {
    path: PathBuf,
    sizes: Arc<RwLock<HashMap<ModelVariant, u64>>>,
}

impl ExpectedSizeStore {
    fn load(models_dir: &Path) -> Self {
        let path = models_dir.join(EXPECTED_SIZES_CACHE_FILE);
        let sizes = std::fs::read(&path)
            .ok()
            .and_then(|bytes| {
                serde_json::from_slice::<HashMap<String, u64>>(&bytes)
                    .map_err(|error| {
                        warn!(
                            "Ignoring unreadable expected-size cache {}: {}",
                            path.display(),
                            error
                        );
                        error
                    })
                    .ok()
            })
            .map(|raw| {
                raw.into_iter()
                    .filter_map(|(name, size)| {
                        crate::catalog::parse_model_variant(&name)
                            .ok()
                            .map(|variant| (variant, size))
                    })
                    .collect::<HashMap<_, _>>()
            })
            .unwrap_or_default();
        Self {
            path,
            sizes: Arc::new(RwLock::new(sizes)),
        }
    }

    async fn get(&self, variant: ModelVariant) -> Option<u64> {
        self.sizes.read().await.get(&variant).copied()
    }

    async fn insert(&self, variant: ModelVariant, size: u64) {
        let snapshot = {
            let mut sizes = self.sizes.write().await;
            if sizes.get(&variant).copied() == Some(size) {
                return;
            }
            sizes.insert(variant, size);
            sizes
                .iter()
                .map(|(variant, size)| (variant.dir_name().to_string(), *size))
                .collect::<HashMap<_, _>>()
        };
        // Best-effort persistence: a cache write failure must never fail a
        // download or a model-list request.
        let path = self.path.clone();
        let result = tokio::task::spawn_blocking(move || {
            let bytes = serde_json::to_vec_pretty(&snapshot)?;
            let partial = path.with_extension("json.part");
            std::fs::write(&partial, bytes)?;
            std::fs::rename(&partial, &path)?;
            std::result::Result::<(), Error>::Ok(())
        })
        .await;
        if let Err(join) = &result {
            warn!("Expected-size cache write task failed: {join}");
        } else if let Ok(Err(error)) = result {
            warn!("Failed to persist expected-size cache: {error}");
        }
    }
}

impl DownloadStateManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn set_state(&self, variant: ModelVariant, state: DownloadState) {
        let mut map = self.state.write().await;
        map.insert(variant, state);
        if state != DownloadState::Error {
            self.errors.write().await.remove(&variant);
        }
    }

    /// Mark a download failed and keep why, for the model list and the
    /// progress stream.
    pub async fn set_failed(&self, variant: ModelVariant, reason: String) {
        self.state
            .write()
            .await
            .insert(variant, DownloadState::Error);
        self.errors.write().await.insert(variant, reason);
    }

    /// Why the last download of `variant` failed, if it did.
    pub async fn error(&self, variant: ModelVariant) -> Option<String> {
        self.errors.read().await.get(&variant).cloned()
    }

    pub async fn get_state(&self, variant: ModelVariant) -> DownloadState {
        let map = self.state.read().await;
        map.get(&variant)
            .copied()
            .unwrap_or(DownloadState::NotDownloaded)
    }

    pub async fn is_downloading(&self, variant: ModelVariant) -> bool {
        self.get_state(variant).await == DownloadState::Downloading
    }
}

/// Model downloader for HuggingFace Hub
pub struct ModelDownloader {
    pub models_dir: PathBuf,
    http_client: reqwest::Client,
    active_downloads: Arc<RwLock<std::collections::HashMap<ModelVariant, ActiveDownload>>>,
    latest_progress: Arc<RwLock<std::collections::HashMap<ModelVariant, DownloadProgress>>>,
    repo_tree_cache: Arc<RwLock<HashMap<String, HashMap<String, u64>>>>,
    expected_sizes: ExpectedSizeStore,
    /// Cached on-disk size per downloaded variant. Recursive directory scans
    /// are expensive and run off the async executor; the result is memoized
    /// until a download, cancellation, or deletion changes the directory.
    downloaded_sizes: Arc<RwLock<HashMap<ModelVariant, u64>>>,
    multi_progress: MultiProgress,
    state_manager: DownloadStateManager,
    /// `IZWI_OFFLINE`: no request ever goes to Hugging Face.
    offline: bool,
}

impl ModelDownloader {
    /// Create a new downloader
    pub fn new(models_dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&models_dir)?;

        // reqwest's rustls has no built-in crypto provider in this workspace.
        let _ = rustls::crypto::ring::default_provider().install_default();
        let mut http_client = reqwest::Client::builder()
            // No whole-request limit: a stalled connection fails fast and the
            // file download resumes where it stopped.
            .connect_timeout(DOWNLOAD_CONNECT_TIMEOUT)
            .read_timeout(DOWNLOAD_STALL_TIMEOUT)
            // Avoid macOS SystemConfiguration proxy lookups which can fail in
            // restricted runtime environments (sandboxed agents/CI).
            .no_proxy();
        // An explicit HTTPS_PROXY (a corporate or container egress proxy)
        // still applies, with NO_PROXY exceptions.
        if let Some(proxy) = https_proxy_from_env()? {
            http_client = http_client.proxy(proxy);
        }
        let http_client = http_client.build().map_err(|e| {
            Error::HfHubError(format!("Failed to create HTTP client: {}", error_chain(&e)))
        })?;

        let multi_progress = MultiProgress::new();
        multi_progress.set_draw_target(indicatif::ProgressDrawTarget::stderr_with_hz(10));
        let expected_sizes = ExpectedSizeStore::load(&models_dir);

        Ok(Self {
            models_dir,
            http_client,
            active_downloads: Arc::new(RwLock::new(std::collections::HashMap::new())),
            latest_progress: Arc::new(RwLock::new(std::collections::HashMap::new())),
            repo_tree_cache: Arc::new(RwLock::new(HashMap::new())),
            expected_sizes,
            downloaded_sizes: Arc::new(RwLock::new(HashMap::new())),
            multi_progress,
            state_manager: DownloadStateManager::new(),
            offline: crate::env::flag_or("IZWI_OFFLINE", false),
        })
    }

    /// Get download state manager
    pub fn state_manager(&self) -> DownloadStateManager {
        self.state_manager.clone()
    }

    async fn set_latest_progress(&self, progress: DownloadProgress) {
        let mut latest = self.latest_progress.write().await;
        latest.insert(progress.variant, progress);
    }

    pub async fn get_latest_progress(&self, variant: ModelVariant) -> Option<DownloadProgress> {
        let latest = self.latest_progress.read().await;
        latest.get(&variant).cloned()
    }

    async fn clear_latest_progress(&self, variant: ModelVariant) {
        let mut latest = self.latest_progress.write().await;
        latest.remove(&variant);
    }

    async fn get_repo_tree_index(
        &self,
        repo_id: &str,
        revision: &str,
    ) -> Result<HashMap<String, u64>> {
        self.ensure_online()?;
        let cache_key = format!("{repo_id}@{revision}");
        if let Some(cached) = self.repo_tree_cache.read().await.get(&cache_key).cloned() {
            return Ok(cached);
        }

        let url = format!(
            "{}/api/models/{}/tree/{}?recursive=1",
            HF_BASE_URL, repo_id, revision
        );
        let response = self
            .http_client
            .get(&url)
            .header("User-Agent", USER_AGENT)
            .send()
            .await
            .map_err(|e| Error::HfHubError(format!("Repo tree request failed: {}", error_chain(&e))))?;

        if !response.status().is_success() {
            return Err(Error::HfHubError(format!(
                "HTTP {} for repo tree {}",
                response.status(),
                url
            )));
        }

        let entries: Vec<HfRepoTreeEntry> = response
            .json()
            .await
            .map_err(|e| Error::HfHubError(format!("Failed to parse repo tree JSON: {}", e)))?;

        let mut index = HashMap::with_capacity(entries.len());
        for entry in entries {
            index.insert(entry.path, entry.size.unwrap_or(0));
        }

        self.repo_tree_cache
            .write()
            .await
            .insert(cache_key, index.clone());

        Ok(index)
    }

    fn is_not_found_hf_error(err: &Error) -> bool {
        match err {
            Error::HfHubError(msg) | Error::DownloadError(msg) => msg.contains("404"),
            _ => false,
        }
    }

    async fn get_file_bytes(
        &self,
        repo_id: &str,
        revision: &str,
        filename: &str,
    ) -> Result<Vec<u8>> {
        self.ensure_online()?;
        let url = format!(
            "{}/{}/resolve/{}/{}",
            HF_BASE_URL, repo_id, revision, filename
        );
        let response = self
            .http_client
            .get(&url)
            .header("User-Agent", USER_AGENT)
            .send()
            .await
            .map_err(|error| Error::HfHubError(format!("HTTP request failed: {}", error_chain(&error))))?;
        if !response.status().is_success() {
            return Err(Error::HfHubError(format!(
                "HTTP {} for {}",
                response.status(),
                url
            )));
        }
        response
            .bytes()
            .await
            .map(|bytes| bytes.to_vec())
            .map_err(|error| Error::HfHubError(format!("Response body failed: {}", error_chain(&error))))
    }

    /// Download `url` into `dest`, resuming after a dropped connection.
    /// It fails after [`MAX_STALLED_FILE_ATTEMPTS`] attempts in a row that
    /// leave no more of the file on disk than any attempt before (for the
    /// same version of the file); one that does resets the count and the
    /// backoff, so a slow, flaky link still finishes a large file.
    /// Fetch one planned file, then hold a pinned file to its SHA-256.
    async fn fetch_plan(
        &self,
        plan: &FileDownloadPlan,
        dest: &Path,
        label: &str,
        progress: &FileProgress<'_>,
    ) -> Result<u64> {
        let bytes = self
            .fetch_with_retries(
                &plan.url(HF_BASE_URL),
                plan.immutable(),
                plan.exact_size(),
                dest,
                label,
                progress,
            )
            .await?;
        if let Some(pinned) = plan.pinned {
            verify_pinned_file(dest, pinned).await?;
        }
        Ok(bytes)
    }

    async fn fetch_with_retries(
        &self,
        url: &str,
        immutable: bool,
        exact_size: Option<u64>,
        dest: &Path,
        label: &str,
        progress: &FileProgress<'_>,
    ) -> Result<u64> {
        let (partial_path, sidecar_path) = partial_paths(dest)?;
        let mut stalled = 0usize;
        let mut backoff = INITIAL_RETRY_BACKOFF;
        // The most of this version of the file any attempt has kept.
        // Measuring progress against it, not the previous attempt, ends the
        // loop on a server that ignores `Range` and drops the connection at
        // varying points; a new version (ETag) starts the count again.
        let (mut kept, mut version) = partial_state(&partial_path, &sidecar_path).await;
        loop {
            let error = match self
                .fetch_resumable(url, immutable, exact_size, dest, progress)
                .await
            {
                Ok(bytes) => match exact_size {
                    Some(expected) if bytes != expected => {
                        let _ = tokio::fs::remove_file(dest).await;
                        Error::DownloadError(format!(
                            "Downloaded size mismatch for {label}: expected {expected} bytes, got {bytes} bytes"
                        ))
                    }
                    _ => return Ok(bytes),
                },
                Err(error) => error,
            };
            let (after, after_version) = partial_state(&partial_path, &sidecar_path).await;
            if after_version != version {
                version = after_version;
                kept = 0;
            }
            if after > kept {
                kept = after;
                stalled = 0;
                backoff = INITIAL_RETRY_BACKOFF;
            } else {
                stalled += 1;
            }
            if stalled >= MAX_STALLED_FILE_ATTEMPTS {
                return Err(Error::DownloadError(format!(
                    "Failed to download {label} after {stalled} attempts without progress: {error}"
                )));
            }
            warn!("Download of {label} interrupted ({error}); resuming at byte {after} in {backoff:?}");
            if let Some(bar) = progress.bar {
                bar.set_message(format!("{label} (resuming)"));
            }
            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(MAX_RETRY_BACKOFF);
        }
    }

    /// One attempt at `url`: continue the `.part` file an earlier attempt
    /// left when it holds the same source's bytes, else start over. Only a
    /// complete file is renamed to `dest`, so readiness checks never see a
    /// partial one; the `.part` file and its sidecar stay for the next
    /// attempt (or process) to resume.
    async fn fetch_resumable(
        &self,
        url: &str,
        immutable: bool,
        exact_size: Option<u64>,
        dest: &Path,
        progress: &FileProgress<'_>,
    ) -> Result<u64> {
        self.ensure_online()?;
        debug!("Downloading from URL: {}", url);
        if let Some(parent) = dest.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let (partial_path, sidecar_path) = partial_paths(dest)?;
        let resume =
            resumable_partial(&partial_path, &sidecar_path, url, immutable, exact_size).await;

        let mut request = self.http_client.get(url).header("User-Agent", USER_AGENT);
        if let Some((offset, sidecar)) = &resume {
            request = request.header(RANGE, format!("bytes={offset}-"));
            // A changed file answers with all of its bytes instead.
            if let Some(etag) = &sidecar.etag {
                request = request.header(IF_RANGE, etag.as_str());
            }
        }
        let response = request
            .send()
            .await
            .map_err(|e| Error::HfHubError(format!("HTTP request failed: {}", error_chain(&e))))?;
        let status = response.status();
        let content_range = response
            .headers()
            .get(CONTENT_RANGE)
            .and_then(|value| value.to_str().ok())
            .and_then(parse_content_range);
        let resumed_at = resume.as_ref().map(|(offset, _)| *offset);
        let (offset, total) = match (status, resumed_at) {
            (StatusCode::PARTIAL_CONTENT, Some(offset)) => match content_range {
                Some((Some(start), total)) if start == offset => (offset, total),
                _ => {
                    discard_partial(&partial_path, &sidecar_path).await;
                    return Err(Error::HfHubError(format!(
                        "HTTP 206 for {url} does not continue from byte {offset}"
                    )));
                }
            },
            // The partial file already holds the whole file.
            (StatusCode::RANGE_NOT_SATISFIABLE, Some(offset))
                if content_range == Some((None, Some(offset))) =>
            {
                publish_partial(&partial_path, &sidecar_path, dest).await?;
                return Ok(offset);
            }
            (status, _) if status == StatusCode::OK => (0, response.content_length()),
            (status, _) => {
                if status == StatusCode::RANGE_NOT_SATISFIABLE {
                    discard_partial(&partial_path, &sidecar_path).await;
                }
                return Err(Error::HfHubError(format!("HTTP {status} for {url}")));
            }
        };
        let etag = response
            .headers()
            .get(ETAG)
            .and_then(|value| value.to_str().ok())
            .filter(|etag| !etag.starts_with("W/"))
            .map(str::to_string)
            .or_else(|| {
                resume
                    .filter(|_| offset > 0)
                    .and_then(|(_, sidecar)| sidecar.etag)
            });
        let sidecar = PartialDownload {
            url: url.to_string(),
            etag,
            total,
        };
        tokio::fs::write(
            &sidecar_path,
            serde_json::to_vec(&sidecar).map_err(|e| Error::DownloadError(e.to_string()))?,
        )
        .await?;

        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&partial_path)
            .await?;
        file.set_len(offset).await?;
        file.seek(SeekFrom::Start(offset)).await?;
        if offset > 0 {
            info!("Resuming {} at byte {}", url, offset);
        }

        let total_size = total.unwrap_or(0);
        if let Some(bar) = progress.bar {
            bar.set_length(total_size);
        }
        let mut downloaded = offset;
        self.report_file_progress(progress, downloaded, total_size)
            .await;
        let mut stream = response.bytes_stream();
        let mut last_progress_emit = Instant::now();
        let mut last_progress_bytes = downloaded;
        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(error) => {
                    // Keep what arrived for the next attempt.
                    let _ = file.sync_data().await;
                    return Err(Error::HfHubError(format!(
                        "Stream error: {}",
                        error_chain(&error)
                    )));
                }
            };
            file.write_all(&chunk).await?;
            downloaded += chunk.len() as u64;
            if downloaded.saturating_sub(last_progress_bytes) >= 256 * 1024
                || last_progress_emit.elapsed() >= Duration::from_millis(200)
                || (total_size > 0 && downloaded >= total_size)
            {
                self.report_file_progress(progress, downloaded, total_size)
                    .await;
                last_progress_emit = Instant::now();
                last_progress_bytes = downloaded;
            }
        }
        // Always end on the file's final byte count, whatever was throttled.
        self.report_file_progress(progress, downloaded, total_size)
            .await;

        file.sync_all().await?;
        drop(file);
        if let Some(total) = total.filter(|total| downloaded != *total) {
            return Err(Error::HfHubError(format!(
                "Stream for {url} ended at byte {downloaded} of {total}"
            )));
        }
        publish_partial(&partial_path, &sidecar_path, dest).await?;
        debug!("Downloaded {} bytes to {:?}", downloaded, dest);
        Ok(downloaded)
    }

    /// Show `downloaded` of `total` bytes of the current file.
    async fn report_file_progress(&self, progress: &FileProgress<'_>, downloaded: u64, total: u64) {
        if let Some(bar) = progress.bar {
            bar.set_position(downloaded);
        }
        if let Some((tx, template)) = progress.channel {
            let update = DownloadProgress {
                current_file_downloaded: downloaded,
                current_file_total: total.max(downloaded),
                // Bytes of the files already completed, plus this one's.
                downloaded_bytes: template.downloaded_bytes + downloaded,
                ..template.clone()
            };
            self.set_latest_progress(update.clone()).await;
            let _ = tx.send(update);
        }
    }

    /// Get the local path for a model variant
    pub fn model_path(&self, variant: ModelVariant) -> PathBuf {
        self.models_dir.join(variant.dir_name())
    }

    async fn write_pinned_artifact_manifest(
        &self,
        variant: ModelVariant,
        model_dir: &Path,
        plans: &[FileDownloadPlan],
    ) -> Result<()> {
        let Some(revision) = variant.artifact_revision() else {
            return Ok(());
        };
        let mut files = plans
            .iter()
            .map(|plan| plan.local_file.clone())
            .collect::<Vec<_>>();
        files.sort();
        files.dedup();
        for plan in plans {
            let metadata = tokio::fs::metadata(model_dir.join(&plan.local_file))
                .await
                .map_err(|error| {
                    Error::DownloadError(format!(
                        "Refusing to publish incomplete artifact manifest for {variant}: {}: {error}",
                        plan.local_file
                    ))
                })?;
            if !metadata.is_file()
                || (plan.strict_size_check
                    && plan.expected_size > 0
                    && metadata.len() != plan.expected_size)
            {
                return Err(Error::DownloadError(format!(
                    "Refusing to publish incomplete artifact manifest for {variant}: {}",
                    plan.local_file
                )));
            }
        }

        let manifest = ArtifactManifest {
            schema_version: 1,
            variant,
            repo_id: variant.repo_id().to_string(),
            revision: revision.to_string(),
            files,
        };
        let bytes = serde_json::to_vec_pretty(&manifest)?;
        let destination = model_dir.join(ARTIFACT_MANIFEST_FILE);
        let partial = model_dir.join(format!(".{ARTIFACT_MANIFEST_FILE}.part"));
        tokio::fs::write(&partial, bytes).await?;
        tokio::fs::rename(&partial, &destination).await?;
        Ok(())
    }

    /// Check if a model is already downloaded
    pub fn is_downloaded(&self, variant: ModelVariant) -> bool {
        let path = self.model_path(variant);
        if !path.exists() {
            return false;
        }
        if variant.is_qwen38_fp8() {
            return qwen38_bundle_is_complete(&path);
        }
        if variant.is_qwen35_moe_fp8() {
            return qwen35_moe_bundle_is_complete(&path);
        }

        let has_any_safetensors = || {
            std::fs::read_dir(&path)
                .map(|entries| {
                    entries.filter_map(|e| e.ok()).any(|e| {
                        e.path()
                            .extension()
                            .map(|ext| ext == "safetensors")
                            .unwrap_or(false)
                    })
                })
                .unwrap_or(false)
        };

        let has_sharded_model = || {
            std::fs::read_dir(&path)
                .map(|entries| {
                    entries.filter_map(|e| e.ok()).any(|e| {
                        e.file_name()
                            .to_str()
                            .map(|name| {
                                (name.starts_with("model-")
                                    || name.starts_with("model.safetensors-"))
                                    && name.ends_with(".safetensors")
                            })
                            .unwrap_or(false)
                    })
                })
                .unwrap_or(false)
        };

        match variant.family() {
            ModelFamily::KokoroTts => {
                path.join("config.json").exists()
                    && path.join("kokoro-v1_0.pth").exists()
                    && path.join("voices").join("af_heart.pt").exists()
            }
            ModelFamily::ParakeetAsr => {
                if variant.is_parakeet_nemo() {
                    let nemo_file = match variant {
                        ModelVariant::ParakeetTdt06BV3 => "parakeet-tdt-0.6b-v3.nemo",
                        _ => unreachable!("checked by is_parakeet_nemo"),
                    };
                    // The extracted checkpoint is complete without its
                    // archive once a first load has unpacked it (PKT-9).
                    return path.join(nemo_file).exists()
                        || parakeet_native_artifacts_complete(&path, variant);
                }

                let has_config = path.join("config.json").exists();
                let has_model = path.join("model.safetensors").exists();
                let has_tokenizer = path.join("tokenizer.json").exists()
                    || path.join("tokenizer.model").exists()
                    || path.join("vocab.txt").exists()
                    || path.join("tokenizer.vocab").exists();
                has_config && has_model && has_tokenizer
            }
            ModelFamily::WhisperAsr => {
                path.join("config.json").exists()
                    && path.join("generation_config.json").exists()
                    && path.join("preprocessor_config.json").exists()
                    && path.join("tokenizer.json").exists()
                    && path.join("model.safetensors").exists()
            }
            ModelFamily::Qwen3Asr => {
                let gguf_file =
                    qwen_asr_gguf_filename(variant).expect("checked by qwen3-asr family");
                path.join(gguf_file).exists()
            }
            ModelFamily::VibeVoiceAsr => {
                path.join("config.json").exists()
                    && path.join("model.safetensors.index.json").exists()
                    && path.join("tokenizer.json").exists()
                    && path.join("tokenizer_config.json").exists()
                    && vibevoice_safetensor_shards(variant)
                        .iter()
                        .all(|file| path.join(file).exists())
            }
            ModelFamily::NemotronAsr => {
                path.join("nemotron-3.5-asr-streaming-0.6b.nemo").exists()
                    || nemotron_native_artifacts_complete(&path, variant)
            }
            ModelFamily::GraniteSpeechAsr | ModelFamily::GraniteSpeech5Asr => {
                granite_speech_required_files(variant)
                    .iter()
                    .all(|file| path.join(file).exists())
            }
            ModelFamily::SortformerDiarization => {
                let nemo_file = match variant {
                    ModelVariant::DiarStreamingSortformer4SpkV21 => {
                        "diar_streaming_sortformer_4spk-v2.1.nemo"
                    }
                    ModelVariant::Nemotron3Diarization => "Nemotron-3-Diarization.nemo",
                    _ => unreachable!("checked by family"),
                };
                path.join(nemo_file).exists()
            }
            ModelFamily::Qwen3Chat
            | ModelFamily::Qwen3MoeChat
            | ModelFamily::Qwen35Chat
            | ModelFamily::Qwen35MoeChat
            | ModelFamily::Qwen38Chat
            | ModelFamily::Lfm2Chat
            | ModelFamily::Gemma3Chat => {
                if variant.is_qwen_chat_gguf() {
                    let gguf_file =
                        qwen_chat_gguf_filename(variant).expect("checked by is_qwen_chat_gguf");
                    path.join(gguf_file).exists()
                        && path.join("tokenizer.json").exists()
                        && path.join("tokenizer_config.json").exists()
                } else if variant.is_qwen35_chat_gguf() {
                    let gguf_file =
                        qwen35_chat_gguf_filename(variant).expect("checked by is_qwen35_chat_gguf");
                    path.join(gguf_file).exists() && path.join("mmproj-F16.gguf").exists()
                } else if variant.is_lfm2_chat_gguf() {
                    let gguf_file =
                        lfm2_chat_gguf_filename(variant).expect("checked by is_lfm2_chat_gguf");
                    path.join(gguf_file).exists()
                        && path.join("chat_template.jinja").exists()
                        && path.join("config.json").exists()
                        && path.join("generation_config.json").exists()
                        && path.join("special_tokens_map.json").exists()
                        && path.join("tokenizer.json").exists()
                        && path.join("tokenizer_config.json").exists()
                } else {
                    let has_config = path.join("config.json").exists();
                    let has_tokenizer = path.join("tokenizer.json").exists()
                        || path.join("tokenizer.model").exists()
                        || (path.join("vocab.json").exists() && path.join("merges.txt").exists());
                    let has_model = path.join("model.safetensors").exists() || has_sharded_model();
                    has_config && has_tokenizer && has_model
                }
            }
            ModelFamily::Lfm25Audio => lfm25_audio_gguf_filenames(variant)
                .expect("checked by is_lfm25_audio_gguf")
                .into_iter()
                .all(|file| path.join(file).exists()),
            ModelFamily::Qwen3ForcedAligner => {
                path.join("config.json").exists()
                    && path.join("vocab.json").exists()
                    && path.join("model.safetensors").exists()
            }
            ModelFamily::Voxtral => {
                path.join("params.json").exists()
                    && path.join("config.json").exists()
                    && path.join("tekken.json").exists()
                    && path.join("consolidated.safetensors").exists()
            }
            ModelFamily::VoxtralTts => {
                path.join("params.json").exists()
                    && path.join("tekken.json").exists()
                    && path.join("consolidated.safetensors").exists()
                    && VOXTRAL_TTS_VOICE_EMBEDDINGS
                        .iter()
                        .all(|file| path.join(file).exists())
            }
            ModelFamily::VibeVoiceTts => {
                path.join("config.json").exists()
                    && path.join("preprocessor_config.json").exists()
                    && path.join("model.safetensors.index.json").exists()
                    && path.join("tokenizer.json").exists()
                    && path.join("tokenizer_config.json").exists()
                    && vibevoice_safetensor_shards(variant)
                        .iter()
                        .all(|file| path.join(file).exists())
            }
            ModelFamily::FishS2Tts => FISH_S2_PRO_FILES
                .iter()
                .all(|file| path.join(file).exists()),
            // The watermarker is needed only while the watermark is on.
            ModelFamily::ChatterboxTts => chatterbox_files(variant).iter().all(|file| {
                (*file == PERTH_CHECKPOINT_FILE
                    && !crate::env::flag_or("IZWI_CHATTERBOX_WATERMARK", true))
                    || path.join(file).exists()
            }),
            ModelFamily::Tokenizer => {
                path.join("tokenizer.json").exists() || path.join("vocab.json").exists()
            }
            ModelFamily::Qwen3Tts => path.join("config.json").exists() && has_any_safetensors(),
        }
    }

    /// Refuse a download its volume cannot hold before any byte is fetched:
    /// the files still missing, plus, for the NeMo ASR families, the
    /// checkpoint a first load extracts beside the archive (PKT-9, ADM-14).
    fn ensure_download_disk_space(
        &self,
        variant: ModelVariant,
        local_dir: &Path,
        plans: &[FileDownloadPlan],
    ) -> Result<()> {
        let files = plans
            .iter()
            .map(|plan| {
                let present =
                    std::fs::metadata(local_dir.join(&plan.local_file)).is_ok_and(|metadata| {
                        !(plan.strict_size_check
                            && plan.expected_size > 0
                            && metadata.len() != plan.expected_size)
                    });
                (plan.local_file.as_str(), plan.expected_size, present)
            })
            .collect::<Vec<_>>();
        let required = download_disk_requirement(&files, extracts_nemo_on_load(variant));
        ensure_free_disk(
            local_dir,
            required,
            &format!("downloading {}", variant.dir_name()),
        )
    }

    /// Download a model with multi-progress bar display
    pub async fn download(&self, variant: ModelVariant) -> Result<PathBuf> {
        self.ensure_download_allowed(variant)?;
        let local_dir = self.model_path(variant);

        tokio::fs::create_dir_all(&local_dir).await?;
        info!("Downloading {} to {:?}", variant.repo_id(), local_dir);

        // Create overall progress bar
        let file_plans = self.get_file_download_plans(variant).await?;
        self.ensure_download_disk_space(variant, &local_dir, &file_plans)?;
        let total_bytes: u64 = file_plans.iter().map(|plan| plan.expected_size).sum();

        let overall_pb = self.multi_progress.add(ProgressBar::new(total_bytes));
        overall_pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("##-"),
        );
        overall_pb.set_message(variant.display_name().to_string());

        let mut downloaded_bytes: u64 = 0;

        for plan in &file_plans {
            let file = &plan.local_file;
            let dest = local_dir.join(file);

            // Skip if already downloaded
            if dest.exists() {
                let file_size = tokio::fs::metadata(&dest).await?.len();
                if plan.strict_size_check
                    && plan.expected_size > 0
                    && file_size != plan.expected_size
                {
                    warn!(
                        "Existing file size mismatch for {} (expected {} bytes, found {}), re-downloading",
                        file, plan.expected_size, file_size
                    );
                    tokio::fs::remove_file(&dest).await?;
                } else {
                    debug!("File already exists: {:?}", dest);
                    downloaded_bytes += if plan.expected_size > 0 {
                        plan.expected_size
                    } else {
                        file_size
                    };
                    overall_pb.set_position(downloaded_bytes);
                    continue;
                }
            }

            // Create file progress bar
            let file_size = plan.expected_size;
            let file_pb = self.multi_progress.add(ProgressBar::new(file_size));
            file_pb.set_style(
                ProgressStyle::default_bar()
                    .template(
                        "  [{elapsed_precise}] {bar:30.green/white} {bytes}/{total_bytes} {msg}",
                    )
                    .unwrap()
                    .progress_chars("=>-"),
            );
            file_pb.set_message(file.clone());

            // Stream download with progress
            match self
                .fetch_plan(
                    plan,
                    &dest,
                    file,
                    &FileProgress {
                        bar: Some(&file_pb),
                        channel: None,
                    },
                )
                .await
            {
                Ok(bytes_downloaded) => {
                    debug!(
                        "Downloaded: {}/{} -> {:?} ({} bytes)",
                        plan.source_repo, plan.source_file, dest, bytes_downloaded
                    );
                    downloaded_bytes += bytes_downloaded;
                    overall_pb.set_position(downloaded_bytes);
                    file_pb.finish_with_message(format!("{} ✓", file));
                }
                Err(e) => {
                    warn!("Failed to download {}: {}", file, e);
                    file_pb.finish_with_message(format!("{} ✗", file));
                    // A file the repository tree lists is required; only
                    // files planned without it might be optional.
                    if variant.is_qwen38_fp8() || plan.exact_size().is_some() {
                        return Err(Error::DownloadError(format!(
                            "Failed to download required file {file}: {e}"
                        )));
                    }
                }
            }

            self.multi_progress.remove(&file_pb);
        }

        self.write_pinned_artifact_manifest(variant, &local_dir, &file_plans)
            .await?;

        overall_pb.finish_with_message(format!("{} ✓", variant.display_name()));
        self.multi_progress.remove(&overall_pb);

        info!("Model downloaded to {:?}", local_dir);
        Ok(local_dir)
    }

    /// Spawn a non-blocking download in the background
    pub async fn spawn_download(
        &self,
        variant: ModelVariant,
    ) -> Result<broadcast::Receiver<DownloadProgress>> {
        self.ensure_download_allowed(variant)?;
        // Set initial state
        self.clear_latest_progress(variant).await;
        self.state_manager
            .set_state(variant, DownloadState::Downloading)
            .await;

        // Create broadcast channel for progress (allows multiple subscribers)
        let (progress_tx, _progress_rx) = broadcast::channel(512);
        let downloader = self.clone_downloader();
        let progress_tx_clone = progress_tx.clone();

        // Spawn background download task
        let handle = tokio::spawn(async move {
            let result = downloader
                .download_with_progress(variant, progress_tx_clone)
                .await;

            if let Err(err) = &result {
                warn!("Background download failed for {}: {}", variant, err);
            }

            match &result {
                Ok(_) => {
                    downloader
                        .state_manager
                        .set_state(variant, DownloadState::Downloaded)
                        .await
                }
                Err(err) => {
                    downloader
                        .state_manager
                        .set_failed(variant, err.to_string())
                        .await
                }
            }
            // The directory contents changed; drop any memoized size.
            downloader.invalidate_downloaded_size(variant).await;

            // Remove finished task from active-downloads registry so UI/handlers
            // stop treating completed downloads as active.
            {
                let mut downloads = downloader.active_downloads.write().await;
                downloads.remove(&variant);
            }

            result
        });

        // Store active download
        let active = ActiveDownload {
            handle,
            progress_tx: progress_tx.clone(),
        };

        let mut downloads = self.active_downloads.write().await;
        downloads.insert(variant, active);

        // Return the progress receiver for the caller to use
        Ok(progress_tx.subscribe())
    }

    /// Clone the downloader for use in spawned tasks
    fn clone_downloader(&self) -> Self {
        Self {
            models_dir: self.models_dir.clone(),
            http_client: self.http_client.clone(),
            active_downloads: Arc::clone(&self.active_downloads),
            latest_progress: Arc::clone(&self.latest_progress),
            repo_tree_cache: Arc::clone(&self.repo_tree_cache),
            expected_sizes: self.expected_sizes.clone(),
            downloaded_sizes: Arc::clone(&self.downloaded_sizes),
            multi_progress: MultiProgress::new(), // Each spawned task gets its own multi-progress
            state_manager: self.state_manager.clone(),
            offline: self.offline,
        }
    }

    /// `IZWI_OFFLINE` refuses every download, whichever path started it.
    fn ensure_download_allowed(&self, variant: ModelVariant) -> Result<()> {
        if self.offline {
            return Err(Error::HfHubError(format!(
                "IZWI_OFFLINE is set: {variant} cannot be downloaded; copy its files into the models directory"
            )));
        }
        Ok(())
    }

    /// Every request to Hugging Face goes through here first.
    fn ensure_online(&self) -> Result<()> {
        if self.offline {
            return Err(Error::HfHubError(
                "IZWI_OFFLINE is set: Hugging Face is not contacted".to_string(),
            ));
        }
        Ok(())
    }

    /// Check if a download is active
    pub async fn is_download_active(&self, variant: ModelVariant) -> bool {
        let downloads = self.active_downloads.read().await;
        downloads.contains_key(&variant)
    }

    /// Wait for an active download to complete
    pub async fn wait_for_download(&self, variant: ModelVariant) -> Result<Option<PathBuf>> {
        let handle = {
            let mut downloads = self.active_downloads.write().await;
            downloads.remove(&variant).map(|d| d.handle)
        };

        match handle {
            Some(h) => {
                let result = h.await.map_err(|e| Error::DownloadError(e.to_string()))?;
                Some(result).transpose()
            }
            None => Ok(None),
        }
    }

    /// Download model with real-time byte-level progress
    pub async fn download_with_progress(
        &self,
        variant: ModelVariant,
        progress_tx: broadcast::Sender<DownloadProgress>,
    ) -> Result<PathBuf> {
        self.ensure_download_allowed(variant)?;
        let local_dir = self.model_path(variant);

        tokio::fs::create_dir_all(&local_dir).await?;

        info!("Downloading {} to {:?}", variant.repo_id(), local_dir);

        let file_plans = self.get_file_download_plans(variant).await?;
        self.ensure_download_disk_space(variant, &local_dir, &file_plans)?;
        let total_files = file_plans.len();
        let total_bytes: u64 = file_plans.iter().map(|plan| plan.expected_size).sum();
        let mut downloaded_bytes: u64 = 0;

        for (idx, plan) in file_plans.iter().enumerate() {
            let file = &plan.local_file;
            let file_size = plan.expected_size;
            let dest = local_dir.join(file);

            // Skip if already downloaded
            if dest.exists() {
                let actual_size = tokio::fs::metadata(&dest).await?.len();
                let should_redownload =
                    plan.strict_size_check && file_size > 0 && actual_size != file_size;

                if should_redownload {
                    warn!(
                        "Existing file size mismatch for {} (expected {} bytes, found {}), re-downloading",
                        file, file_size, actual_size
                    );
                    tokio::fs::remove_file(&dest).await?;
                } else {
                    debug!("File already exists: {:?} ({} bytes)", dest, actual_size);
                    downloaded_bytes += if file_size > 0 {
                        file_size
                    } else {
                        actual_size
                    };

                    // Send progress update
                    let progress = DownloadProgress {
                        variant,
                        downloaded_bytes,
                        total_bytes,
                        current_file: file.clone(),
                        current_file_downloaded: actual_size,
                        current_file_total: if file_size > 0 {
                            file_size
                        } else {
                            actual_size
                        },
                        files_completed: idx + 1,
                        files_total: total_files,
                    };
                    self.set_latest_progress(progress.clone()).await;
                    let _ = progress_tx.send(progress);
                    continue;
                }
            }

            // Stream download with per-chunk progress updates
            let file_pb = self.multi_progress.add(ProgressBar::new(file_size));
            file_pb.set_style(
                ProgressStyle::default_bar()
                    .template(
                        "  [{elapsed_precise}] {bar:30.green/white} {bytes}/{total_bytes} {msg}",
                    )
                    .unwrap()
                    .progress_chars("=>-"),
            );
            file_pb.set_message(file.clone());

            // Create progress template for real-time updates
            let progress_template = DownloadProgress {
                variant,
                downloaded_bytes, // Base bytes from completed files
                total_bytes,
                current_file: file.clone(),
                current_file_downloaded: 0,
                current_file_total: file_size,
                files_completed: idx,
                files_total: total_files,
            };

            let bytes_downloaded_for_file = match self
                .fetch_plan(
                    plan,
                    &dest,
                    file,
                    &FileProgress {
                        bar: Some(&file_pb),
                        channel: Some((&progress_tx, &progress_template)),
                    },
                )
                .await
            {
                Ok(bytes) => bytes,
                Err(e) => {
                    file_pb.finish_with_message(format!("{} ✗", file));
                    return Err(Error::DownloadError(format!(
                        "Failed to download required file {file}: {e}"
                    )));
                }
            };

            downloaded_bytes += bytes_downloaded_for_file;

            let progress = DownloadProgress {
                variant,
                downloaded_bytes,
                total_bytes,
                current_file: file.clone(),
                current_file_downloaded: bytes_downloaded_for_file,
                current_file_total: file_size,
                files_completed: idx + 1,
                files_total: total_files,
            };
            self.set_latest_progress(progress.clone()).await;
            let _ = progress_tx.send(progress);

            file_pb.finish_with_message(format!("{} ✓", file));

            self.multi_progress.remove(&file_pb);
        }

        self.write_pinned_artifact_manifest(variant, &local_dir, &file_plans)
            .await?;

        // Send final completion progress
        let progress = DownloadProgress {
            variant,
            downloaded_bytes,
            total_bytes,
            current_file: String::new(),
            current_file_downloaded: 0,
            current_file_total: 0,
            files_completed: total_files,
            files_total: total_files,
        };
        self.set_latest_progress(progress.clone()).await;
        let _ = progress_tx.send(progress);

        info!("Model downloaded to {:?}", local_dir);
        Ok(local_dir)
    }

    /// Get list of files to download for a model variant
    /// Based on actual repo structure on HuggingFace
    fn get_model_files(&self, variant: ModelVariant) -> Vec<String> {
        match variant.family() {
            ModelFamily::KokoroTts => vec![
                "config.json".to_string(),
                "kokoro-v1_0.pth".to_string(),
                "voices/af_alloy.pt".to_string(),
                "voices/af_aoede.pt".to_string(),
                "voices/af_bella.pt".to_string(),
                "voices/af_heart.pt".to_string(),
                "voices/af_jessica.pt".to_string(),
                "voices/af_kore.pt".to_string(),
                "voices/af_nicole.pt".to_string(),
                "voices/af_nova.pt".to_string(),
                "voices/af_river.pt".to_string(),
                "voices/af_sarah.pt".to_string(),
                "voices/af_sky.pt".to_string(),
                "voices/am_adam.pt".to_string(),
                "voices/am_echo.pt".to_string(),
                "voices/am_eric.pt".to_string(),
                "voices/am_fenrir.pt".to_string(),
                "voices/am_liam.pt".to_string(),
                "voices/am_michael.pt".to_string(),
                "voices/am_onyx.pt".to_string(),
                "voices/am_puck.pt".to_string(),
                "voices/am_santa.pt".to_string(),
                "voices/bf_alice.pt".to_string(),
                "voices/bf_emma.pt".to_string(),
                "voices/bf_isabella.pt".to_string(),
                "voices/bf_lily.pt".to_string(),
                "voices/bm_daniel.pt".to_string(),
                "voices/bm_fable.pt".to_string(),
                "voices/bm_george.pt".to_string(),
                "voices/bm_lewis.pt".to_string(),
                "voices/ef_dora.pt".to_string(),
                "voices/em_alex.pt".to_string(),
                "voices/em_santa.pt".to_string(),
                "voices/ff_siwis.pt".to_string(),
                "voices/hf_alpha.pt".to_string(),
                "voices/hf_beta.pt".to_string(),
                "voices/hm_omega.pt".to_string(),
                "voices/hm_psi.pt".to_string(),
                "voices/if_sara.pt".to_string(),
                "voices/im_nicola.pt".to_string(),
                "voices/jf_alpha.pt".to_string(),
                "voices/jf_gongitsune.pt".to_string(),
                "voices/jf_nezumi.pt".to_string(),
                "voices/jf_tebukuro.pt".to_string(),
                "voices/jm_kumo.pt".to_string(),
                "voices/pf_dora.pt".to_string(),
                "voices/pm_alex.pt".to_string(),
                "voices/pm_santa.pt".to_string(),
                "voices/zf_xiaobei.pt".to_string(),
                "voices/zf_xiaoni.pt".to_string(),
                "voices/zf_xiaoxiao.pt".to_string(),
                "voices/zf_xiaoyi.pt".to_string(),
                "voices/zm_yunjian.pt".to_string(),
                "voices/zm_yunxi.pt".to_string(),
                "voices/zm_yunxia.pt".to_string(),
                "voices/zm_yunyang.pt".to_string(),
            ],
            ModelFamily::ParakeetAsr => {
                if variant.is_parakeet_nemo() {
                    let nemo_file = match variant {
                        ModelVariant::ParakeetTdt06BV3 => "parakeet-tdt-0.6b-v3.nemo",
                        _ => unreachable!("checked by is_parakeet_nemo"),
                    };
                    vec![nemo_file.to_string(), "README.md".to_string()]
                } else {
                    vec![
                        "config.json".to_string(),
                        "generation_config.json".to_string(),
                        "model.safetensors".to_string(),
                        "tokenizer.json".to_string(),
                        "tokenizer.model".to_string(),
                        "tokenizer_config.json".to_string(),
                        "special_tokens_map.json".to_string(),
                        "vocab.txt".to_string(),
                        "tokenizer.vocab".to_string(),
                    ]
                }
            }
            ModelFamily::WhisperAsr => vec![
                "config.json".to_string(),
                "generation_config.json".to_string(),
                "preprocessor_config.json".to_string(),
                "tokenizer.json".to_string(),
                "tokenizer_config.json".to_string(),
                "special_tokens_map.json".to_string(),
                "vocab.json".to_string(),
                "merges.txt".to_string(),
                "normalizer.json".to_string(),
                "added_tokens.json".to_string(),
                "model.safetensors".to_string(),
            ],
            ModelFamily::Qwen3Asr => vec![qwen_asr_gguf_filename(variant)
                .expect("checked by qwen3-asr family")
                .to_string()],
            ModelFamily::VibeVoiceAsr => {
                let mut files = vec![
                    "config.json".to_string(),
                    "model.safetensors.index.json".to_string(),
                ];
                files.extend(vibevoice_safetensor_shards(variant));
                files.extend(
                    VIBEVOICE_QWEN_TOKENIZER_FILES
                        .iter()
                        .map(|file| (*file).to_string()),
                );
                files
            }
            ModelFamily::NemotronAsr => {
                vec!["nemotron-3.5-asr-streaming-0.6b.nemo".to_string()]
            }
            ModelFamily::GraniteSpeechAsr | ModelFamily::GraniteSpeech5Asr => {
                granite_speech_files(variant)
            }
            ModelFamily::SortformerDiarization => match variant {
                ModelVariant::Nemotron3Diarization => {
                    vec!["Nemotron-3-Diarization.nemo".to_string()]
                }
                _ => vec![
                    "diar_streaming_sortformer_4spk-v2.1.nemo".to_string(),
                    "README.md".to_string(),
                    "bias.md".to_string(),
                    "privacy.md".to_string(),
                    "safety.md".to_string(),
                ],
            },
            ModelFamily::Qwen3Chat
            | ModelFamily::Qwen3MoeChat
            | ModelFamily::Qwen35Chat
            | ModelFamily::Qwen35MoeChat
            | ModelFamily::Qwen38Chat
            | ModelFamily::Lfm2Chat
            | ModelFamily::Gemma3Chat => {
                if variant.is_qwen38_fp8() {
                    return QWEN38_REQUIRED_METADATA_FILES
                        .iter()
                        .map(|file| (*file).to_string())
                        .collect();
                } else if variant.is_qwen35_moe_fp8() {
                    return QWEN35_MOE_REQUIRED_METADATA_FILES
                        .iter()
                        .map(|file| (*file).to_string())
                        .collect();
                } else if variant.is_qwen_chat_gguf() {
                    let gguf_file =
                        qwen_chat_gguf_filename(variant).expect("checked by is_qwen_chat_gguf");
                    return vec![
                        gguf_file.to_string(),
                        "params".to_string(),
                        "README.md".to_string(),
                        "tokenizer.json".to_string(),
                        "tokenizer_config.json".to_string(),
                    ];
                } else if variant.is_qwen35_chat_gguf() {
                    let gguf_file =
                        qwen35_chat_gguf_filename(variant).expect("checked by is_qwen35_chat_gguf");
                    return vec![gguf_file.to_string(), "mmproj-F16.gguf".to_string()];
                } else if variant.is_lfm2_chat_gguf() {
                    let gguf_file =
                        lfm2_chat_gguf_filename(variant).expect("checked by is_lfm2_chat_gguf");
                    return vec![
                        gguf_file.to_string(),
                        "chat_template.jinja".to_string(),
                        "config.json".to_string(),
                        "generation_config.json".to_string(),
                        "special_tokens_map.json".to_string(),
                        "tokenizer.json".to_string(),
                        "tokenizer_config.json".to_string(),
                    ];
                }
                let mut files = vec![
                    "config.json".to_string(),
                    "generation_config.json".to_string(),
                    "chat_template.jinja".to_string(),
                    "added_tokens.json".to_string(),
                    "tokenizer.json".to_string(),
                    "tokenizer.model".to_string(),
                    "tokenizer_config.json".to_string(),
                    "special_tokens_map.json".to_string(),
                    "vocab.json".to_string(),
                    "merges.txt".to_string(),
                    "preprocessor_config.json".to_string(),
                    "model.safetensors".to_string(),
                    "model.safetensors.index.json".to_string(),
                ];
                for total in [2usize, 3, 4] {
                    for idx in 1..=total {
                        files.push(format!("model-{idx:05}-of-{total:05}.safetensors"));
                    }
                }
                for total in [1usize, 2, 3, 4, 5, 6, 7, 8] {
                    for idx in 1..=total {
                        files.push(format!(
                            "model.safetensors-{idx:05}-of-{total:05}.safetensors"
                        ));
                    }
                }
                files
            }
            ModelFamily::Lfm25Audio => lfm25_audio_gguf_filenames(variant)
                .expect("checked by is_lfm25_audio_gguf")
                .into_iter()
                .map(|file| file.to_string())
                .collect(),
            ModelFamily::Qwen3ForcedAligner => {
                let mut files = vec![
                    "config.json".to_string(),
                    "generation_config.json".to_string(),
                    "merges.txt".to_string(),
                    "model.safetensors".to_string(),
                    "preprocessor_config.json".to_string(),
                    "tokenizer_config.json".to_string(),
                    "vocab.json".to_string(),
                ];
                if variant.is_quantized() {
                    files.push("model.safetensors.index.json".to_string());
                }
                files
            }
            ModelFamily::Voxtral => vec![
                "params.json".to_string(),
                "config.json".to_string(),
                "tekken.json".to_string(),
                "consolidated.safetensors".to_string(),
            ],
            ModelFamily::VoxtralTts => {
                let mut files = vec![
                    "params.json".to_string(),
                    "tekken.json".to_string(),
                    "consolidated.safetensors".to_string(),
                ];
                files.extend(
                    VOXTRAL_TTS_VOICE_EMBEDDINGS
                        .iter()
                        .map(|file| (*file).to_string()),
                );
                files
            }
            ModelFamily::VibeVoiceTts => {
                let mut files = vec![
                    "config.json".to_string(),
                    "preprocessor_config.json".to_string(),
                    "model.safetensors.index.json".to_string(),
                ];
                files.extend(vibevoice_safetensor_shards(variant));
                files.extend(
                    VIBEVOICE_QWEN_TOKENIZER_FILES
                        .iter()
                        .map(|file| (*file).to_string()),
                );
                files
            }
            ModelFamily::FishS2Tts => FISH_S2_PRO_FILES
                .iter()
                .map(|file| (*file).to_string())
                .collect(),
            ModelFamily::ChatterboxTts => chatterbox_files(variant)
                .iter()
                .map(|file| (*file).to_string())
                .collect(),
            ModelFamily::Tokenizer => vec![
                "config.json".to_string(),
                "generation_config.json".to_string(),
                "preprocessor_config.json".to_string(),
                "model.safetensors".to_string(),
            ],
            ModelFamily::Qwen3Tts => {
                let mut files = vec![
                    "config.json".to_string(),
                    "generation_config.json".to_string(),
                    "tokenizer_config.json".to_string(),
                    "vocab.json".to_string(),
                    "merges.txt".to_string(),
                    "preprocessor_config.json".to_string(),
                    "model.safetensors".to_string(),
                ];
                if variant.is_quantized() {
                    files.push("model.safetensors.index.json".to_string());
                }
                files.extend([
                    "speech_tokenizer/config.json".to_string(),
                    "speech_tokenizer/configuration.json".to_string(),
                    "speech_tokenizer/model.safetensors".to_string(),
                    "speech_tokenizer/preprocessor_config.json".to_string(),
                ]);
                files
            }
        }
    }

    fn get_model_file_specs(&self, variant: ModelVariant) -> Vec<ModelFileSpec> {
        let default_repo = variant.repo_id().to_string();
        let default_revision = variant.artifact_revision().unwrap_or("main").to_string();

        if let Some(tokenizer_repo) = vibevoice_tokenizer_repo(variant) {
            return self
                .get_model_files(variant)
                .into_iter()
                .map(|file| {
                    let source_repo = if VIBEVOICE_QWEN_TOKENIZER_FILES.contains(&file.as_str()) {
                        tokenizer_repo.to_string()
                    } else {
                        default_repo.clone()
                    };

                    ModelFileSpec {
                        source_revision: if source_repo == default_repo {
                            default_revision.clone()
                        } else {
                            "main".to_string()
                        },
                        source_repo,
                        source_file: file.clone(),
                        local_file: file,
                    }
                })
                .collect();
        }

        if variant.is_qwen_chat_gguf() {
            let tokenizer_repo = match variant {
                ModelVariant::Qwen314BGguf => "Qwen/Qwen3-14B",
                _ => variant.repo_id(),
            };

            return self
                .get_model_files(variant)
                .into_iter()
                .map(|file| {
                    let source_repo = if file == "tokenizer.json" || file == "tokenizer_config.json"
                    {
                        tokenizer_repo.to_string()
                    } else {
                        default_repo.clone()
                    };

                    ModelFileSpec {
                        source_revision: if source_repo == default_repo {
                            default_revision.clone()
                        } else {
                            "main".to_string()
                        },
                        source_repo,
                        source_file: file.clone(),
                        local_file: file,
                    }
                })
                .collect();
        }

        if variant.is_granite_speech5_gguf() {
            // The configs and tokenizer come from the safetensors release at
            // its pinned revision: the same model, so the same files.
            let companion = ModelVariant::GraniteSpeech502B;
            let companion_revision = companion.artifact_revision().unwrap_or("main");
            return self
                .get_model_files(variant)
                .into_iter()
                .map(|file| {
                    let (source_repo, source_revision) = if file.ends_with(".gguf") {
                        (default_repo.clone(), default_revision.clone())
                    } else {
                        (
                            companion.repo_id().to_string(),
                            companion_revision.to_string(),
                        )
                    };
                    ModelFileSpec {
                        source_repo,
                        source_revision,
                        source_file: file.clone(),
                        local_file: file,
                    }
                })
                .collect();
        }

        if variant.is_lfm2_chat_gguf() {
            // The GGUF repos ship no tokenizer, so the companion files come
            // from the safetensors repos at a pinned revision: upstream
            // replaced their chat template on 2026-08-05, and an unpinned
            // download would follow any later tokenizer change too.
            let (tokenizer_repo, companion_revision) = match variant {
                ModelVariant::Lfm2512BInstructGguf => (
                    "LiquidAI/LFM2.5-1.2B-Instruct",
                    "0f604ada3f766f9f257460c4c9f0b5d6f69d431b",
                ),
                ModelVariant::Lfm2512BThinkingGguf => (
                    "LiquidAI/LFM2.5-1.2B-Thinking",
                    "f313478934a7612d22991f752959d7a1a8756fec",
                ),
                _ => (variant.repo_id(), "main"),
            };

            return self
                .get_model_files(variant)
                .into_iter()
                .map(|file| {
                    let source_repo = if file.ends_with(".gguf") {
                        default_repo.clone()
                    } else {
                        tokenizer_repo.to_string()
                    };

                    ModelFileSpec {
                        source_revision: if source_repo == default_repo {
                            default_revision.clone()
                        } else {
                            companion_revision.to_string()
                        },
                        source_repo,
                        source_file: file.clone(),
                        local_file: file,
                    }
                })
                .collect();
        }

        if variant.is_lfm25_audio_gguf() {
            return self
                .get_model_files(variant)
                .into_iter()
                .map(|file| ModelFileSpec {
                    source_repo: default_repo.clone(),
                    source_revision: default_revision.clone(),
                    source_file: file.clone(),
                    local_file: file,
                })
                .collect();
        }

        if variant.is_qwen35_chat_gguf() {
            return self
                .get_model_files(variant)
                .into_iter()
                .map(|file| ModelFileSpec {
                    source_repo: default_repo.clone(),
                    source_revision: default_revision.clone(),
                    source_file: file.clone(),
                    local_file: file,
                })
                .collect();
        }

        self.get_model_files(variant)
            .into_iter()
            .map(|file| ModelFileSpec {
                source_repo: default_repo.clone(),
                source_revision: default_revision.clone(),
                source_file: file.clone(),
                local_file: file,
            })
            .collect()
    }

    /// Get actual file size from HTTP HEAD request
    async fn get_qwen38_indexed_file_specs(&self) -> Result<Vec<ModelFileSpec>> {
        let variant = ModelVariant::Qwen3827BFp8;
        let repo_id = variant.repo_id();
        let revision = variant
            .artifact_revision()
            .expect("Qwen3.8 artifact revision is catalog-pinned");
        let local_index = self
            .model_path(variant)
            .join("model.safetensors.index.json");
        let index_bytes = if local_index.is_file() {
            tokio::fs::read(&local_index).await?
        } else {
            self.get_file_bytes(repo_id, revision, "model.safetensors.index.json")
                .await?
        };

        qwen38_selected_files(&index_bytes).map(|files| {
            files
                .into_iter()
                .map(|file| ModelFileSpec {
                    source_repo: repo_id.to_string(),
                    source_revision: revision.to_string(),
                    source_file: file.clone(),
                    local_file: file,
                })
                .collect()
        })
    }

    /// Index-closure file plan for the Qwen3.6-35B-A3B-FP8 bundle: every
    /// safetensors shard named by the index plus the required metadata files.
    async fn get_qwen35_moe_indexed_file_specs(&self) -> Result<Vec<ModelFileSpec>> {
        let variant = ModelVariant::Qwen36Moe35BA3BFp8;
        let repo_id = variant.repo_id();
        let revision = variant
            .artifact_revision()
            .expect("Qwen3.6 MoE artifact revision is catalog-pinned");
        let local_index = self
            .model_path(variant)
            .join("model.safetensors.index.json");
        let index_bytes = if local_index.is_file() {
            tokio::fs::read(&local_index).await?
        } else {
            self.get_file_bytes(repo_id, revision, "model.safetensors.index.json")
                .await?
        };

        qwen35_moe_selected_files(&index_bytes).map(|files| {
            files
                .into_iter()
                .map(|file| ModelFileSpec {
                    source_repo: repo_id.to_string(),
                    source_revision: revision.to_string(),
                    source_file: file.clone(),
                    local_file: file,
                })
                .collect()
        })
    }

    /// Get actual file size from HTTP HEAD request
    async fn get_actual_file_size(
        &self,
        repo_id: &str,
        revision: &str,
        filename: &str,
    ) -> Result<u64> {
        self.ensure_online()?;
        let url = format!(
            "{}/{}/resolve/{}/{}",
            HF_BASE_URL, repo_id, revision, filename
        );

        let response = self
            .http_client
            .head(&url)
            .header("User-Agent", USER_AGENT)
            .send()
            .await
            .map_err(|e| Error::HfHubError(format!("HEAD request failed: {}", error_chain(&e))))?;

        if !response.status().is_success() {
            return Err(Error::HfHubError(format!(
                "HTTP {} for HEAD {}",
                response.status(),
                url
            )));
        }

        let size = response
            .headers()
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0);

        Ok(size)
    }

    /// Get actual file sizes for all model files
    async fn get_file_download_plans(
        &self,
        variant: ModelVariant,
    ) -> Result<Vec<FileDownloadPlan>> {
        let file_specs = if variant.is_qwen38_fp8() {
            self.get_qwen38_indexed_file_specs().await?
        } else if variant.is_qwen35_moe_fp8() {
            self.get_qwen35_moe_indexed_file_specs().await?
        } else {
            self.get_model_file_specs(variant)
        };
        let local_dir = self.model_path(variant);
        let require_exact_bundle = variant.is_qwen38_fp8() || variant.is_qwen35_moe_fp8();

        let mut repo_tree_indexes: HashMap<(String, String), HashMap<String, u64>> = HashMap::new();
        let mut repo_tree_planning_available = true;
        for spec in &file_specs {
            let repo_key = (spec.source_repo.clone(), spec.source_revision.clone());
            if repo_tree_indexes.contains_key(&repo_key)
                || pinned_file(variant, &spec.local_file).is_some()
            {
                continue;
            }
            match self
                .get_repo_tree_index(&spec.source_repo, &spec.source_revision)
                .await
            {
                Ok(index) => {
                    repo_tree_indexes.insert(repo_key, index);
                }
                Err(err) => {
                    warn!(
                        "Repo tree lookup failed for {} while planning {}: {}",
                        spec.source_repo, variant, err
                    );
                    repo_tree_planning_available = false;
                    break;
                }
            }
        }

        if repo_tree_planning_available {
            let mut plans = Vec::with_capacity(file_specs.len());
            let mut skipped = 0usize;

            for spec in &file_specs {
                if let Some(pinned) = pinned_file(variant, &spec.local_file) {
                    plans.push(FileDownloadPlan::pinned(spec, pinned));
                } else if let Some(size) = repo_tree_indexes
                    .get(&(spec.source_repo.clone(), spec.source_revision.clone()))
                    .and_then(|index| index.get(&spec.source_file))
                {
                    plans.push(FileDownloadPlan {
                        source_repo: spec.source_repo.clone(),
                        source_revision: spec.source_revision.clone(),
                        source_file: spec.source_file.clone(),
                        local_file: spec.local_file.clone(),
                        expected_size: *size,
                        strict_size_check: *size > 0,
                        pinned: None,
                    });
                } else {
                    if require_exact_bundle {
                        return Err(Error::DownloadError(format!(
                            "Required pinned artifact is absent from repository tree: {}@{}/{}",
                            spec.source_repo, spec.source_revision, spec.source_file
                        )));
                    }
                    skipped += 1;
                    debug!(
                        "Skipping unavailable file for {}: {}/{} (not present in repo tree)",
                        variant, spec.source_repo, spec.source_file
                    );
                }
            }

            if plans.is_empty() {
                return Err(Error::DownloadError(format!(
                    "No downloadable files discovered for {}",
                    variant
                )));
            }

            if skipped > 0 {
                info!(
                    "Skipped {} unavailable file(s) for {} based on repo tree inventory",
                    skipped, variant
                );
            }

            return Ok(plans);
        }

        warn!(
            "Falling back to per-file HEAD planning for {} because repo tree planning was unavailable",
            variant
        );

        let mut plans = Vec::with_capacity(file_specs.len());
        for spec in &file_specs {
            if let Some(pinned) = pinned_file(variant, &spec.local_file) {
                plans.push(FileDownloadPlan::pinned(spec, pinned));
                continue;
            }
            let dest = local_dir.join(&spec.local_file);
            match self
                .get_actual_file_size(&spec.source_repo, &spec.source_revision, &spec.source_file)
                .await
            {
                Ok(size) => plans.push(FileDownloadPlan {
                    source_repo: spec.source_repo.clone(),
                    source_revision: spec.source_revision.clone(),
                    source_file: spec.source_file.clone(),
                    local_file: spec.local_file.clone(),
                    expected_size: size,
                    strict_size_check: true,
                    pinned: None,
                }),
                Err(e) => {
                    if Self::is_not_found_hf_error(&e) {
                        if require_exact_bundle {
                            return Err(Error::DownloadError(format!(
                                "Required pinned artifact is unavailable: {}@{}/{}",
                                spec.source_repo, spec.source_revision, spec.source_file
                            )));
                        }
                        debug!(
                            "Skipping unavailable file for {}: {}/{} (HEAD returned 404)",
                            variant, spec.source_repo, spec.source_file
                        );
                        continue;
                    }

                    if require_exact_bundle {
                        return Err(Error::DownloadError(format!(
                            "Could not determine exact size for required pinned artifact {}@{}/{}: {}",
                            spec.source_repo, spec.source_revision, spec.source_file, e
                        )));
                    }

                    // Fall back to local size if present (resume semantics), else estimate.
                    let fallback_size = if dest.exists() {
                        tokio::fs::metadata(&dest)
                            .await
                            .map(|m| m.len())
                            .unwrap_or_else(|_| {
                                self.get_single_file_size_estimate(variant, &spec.local_file)
                            })
                    } else {
                        self.get_single_file_size_estimate(variant, &spec.local_file)
                    };

                    plans.push(FileDownloadPlan {
                        source_repo: spec.source_repo.clone(),
                        source_revision: spec.source_revision.clone(),
                        source_file: spec.source_file.clone(),
                        local_file: spec.local_file.clone(),
                        expected_size: fallback_size,
                        strict_size_check: false,
                        pinned: None,
                    });
                }
            }
        }

        if plans.is_empty() {
            return Err(Error::DownloadError(format!(
                "No downloadable files discovered for {}",
                variant
            )));
        }

        Ok(plans)
    }

    /// Get estimated size for a single file (fallback when HEAD fails)
    fn get_single_file_size_estimate(&self, variant: ModelVariant, file: &str) -> u64 {
        // The published files' sizes (unsloth Qwen3.5 GGUF and LiquidAI
        // LFM2.5-Audio repositories, 2026-10-19; Chatterbox at 5bb1f6ee).
        if variant.family() == ModelFamily::ChatterboxTts {
            match file {
                "t3_cfg.safetensors" => 2_129_653_744,
                "t3_mtl23ls_v2.safetensors" => 2_143_989_752,
                "grapheme_mtl_merged_expanded_v1.json" => 69_989,
                "s3gen.safetensors" => 1_056_484_620,
                "ve.safetensors" => 5_695_784,
                "conds.pt" => 107_374,
                CANGJIE_FILE => 1_920_163,
                PERTH_CHECKPOINT_FILE => PERTH_CHECKPOINT.size,
                PKUSEG_MODEL_FILE => PKUSEG_MODEL.size,
                PKUSEG_DICT_FILE => PKUSEG_DICT.size,
                _ => 25_470,
            }
        } else if file == "mmproj-F16.gguf" {
            match variant {
                ModelVariant::Qwen3508BGguf => 204_987_232,
                ModelVariant::Qwen352BGguf => 668_227_264,
                ModelVariant::Qwen354BGguf => 672_423_616,
                ModelVariant::Qwen359BGguf => 918_166_080,
                _ => 670_000_000,
            }
        } else if variant.is_granite_speech5_gguf() && file == GRANITE_SPEECH5_GGUF_LM_FILE {
            GRANITE_SPEECH_5_GGUF_LM_BYTES
        } else if variant.is_granite_speech5_gguf() && file == GRANITE_SPEECH5_GGUF_MMPROJ_FILE {
            GRANITE_SPEECH_5_GGUF_MMPROJ_BYTES
        } else if file == "mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf" {
            219_511_136
        } else if file == "tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf" {
            50_546_112
        } else if file == "vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf" {
            108_986_560
        } else if file.ends_with(".gguf") {
            let lower = file.to_ascii_lowercase();
            if file.contains("Qwen3-14B") {
                9_200_000_000
            } else if file.contains("Qwen3.5-0.8B") {
                685_000_000
            } else if file.contains("Qwen3.5-2B") {
                1_850_000_000
            } else if file.contains("Qwen3.5-4B") {
                3_250_000_000
            } else if file.contains("Qwen3.5-9B") {
                6_350_000_000
            } else if file.contains("LFM2.5-1.2B-Instruct") {
                730_895_168
            } else if file.contains("LFM2.5-1.2B-Thinking") {
                730_895_360
            } else if file.contains("LFM2.5-Audio-1.5B") {
                740_000_000
            } else if lower.contains("qwen3_asr_0.6b") {
                1_012_824_608
            } else if lower.contains("qwen3_asr_1.7b") {
                2_512_740_320
            } else {
                1_000_000_000
            }
        } else if file == "consolidated.safetensors" {
            match variant {
                ModelVariant::VoxtralMini4BRealtime2602 => 8_859_358_720,
                ModelVariant::Voxtral4BTts2603 => 8_650_000_000,
                _ => 8_000_000_000,
            }
        } else if let Some(size) = granite_speech_shard_size(variant, file) {
            size
        } else if variant.family() == ModelFamily::GraniteSpeech5Asr && file == "model.safetensors"
        {
            GRANITE_SPEECH_5_MODEL_BYTES
        } else if file.contains("model.safetensors") && !file.contains("index") {
            if matches!(
                variant,
                ModelVariant::VibeVoice15BTts | ModelVariant::VibeVoiceAsr
            ) {
                variant.estimated_size() / vibevoice_shard_count(variant).unwrap_or(1) as u64
            } else if file.contains("00001") || file.contains("00002") {
                2_000_000_000
            } else {
                match variant {
                    ModelVariant::Qwen3Tts12Hz06BBase
                    | ModelVariant::Qwen3Tts12Hz06BCustomVoice => 1_800_000_000,
                    ModelVariant::Qwen3Tts12Hz17BBase
                    | ModelVariant::Qwen3Tts12Hz17BCustomVoice
                    | ModelVariant::Qwen3Tts12Hz17BVoiceDesign => 3_850_000_000,
                    ModelVariant::WhisperLargeV3Turbo => 1_617_824_864,
                    ModelVariant::Qwen306B => 1_520_000_000,
                    ModelVariant::Qwen306B4Bit => 800_000_000,
                    ModelVariant::Qwen317B4Bit => 1_115_000_000,
                    ModelVariant::Qwen3ForcedAligner06B => 1_835_544_544,
                    ModelVariant::Qwen314BGguf => 9_200_000_000,
                    ModelVariant::Qwen3508BGguf => 685_000_000,
                    ModelVariant::Qwen352BGguf => 1_850_000_000,
                    ModelVariant::Qwen354BGguf => 3_250_000_000,
                    ModelVariant::Qwen359BGguf => 6_350_000_000,
                    ModelVariant::Gemma34BIt => 2_400_000_000,
                    ModelVariant::VoxtralMini4BRealtime2602 => 8_900_000_000,
                    _ => 1_500_000_000,
                }
            }
        } else if file.ends_with(".pth") && file.contains("kokoro-v1_0") {
            327_212_226
        } else if (file.starts_with("voices/") || file.starts_with("voice_embedding/"))
            && file.ends_with(".pt")
        {
            550_000
        } else if file.ends_with(".nemo") {
            match variant {
                ModelVariant::ParakeetTdt06BV3 => 2_509_332_480,
                ModelVariant::Nemotron35AsrStreaming06B => 2_368_284_501,
                ModelVariant::DiarStreamingSortformer4SpkV21 => 471_367_680,
                ModelVariant::Nemotron3Diarization => 198_676_480,
                _ => 4_000_000_000,
            }
        } else if file.contains("tokenizer") && file.contains("safetensors") {
            300_000_000
        } else if file.contains("speech_tokenizer") && file.contains("safetensors") {
            100_000_000
        } else if file.ends_with(".json") || file.ends_with(".txt") || file.ends_with(".jinja") {
            if file == "tekken.json" {
                15_000_000
            } else {
                100_000
            }
        } else {
            10_000_000
        }
    }

    /// Get the on-disk size of a downloaded model, computing it off the async
    /// executor on first use and memoizing the result. Returns `None` when the
    /// model directory is absent. Callers that mutate the directory must
    /// invalidate via [`Self::invalidate_downloaded_size`].
    pub async fn get_cached_size(&self, variant: ModelVariant) -> Option<u64> {
        if let Some(size) = self.downloaded_sizes.read().await.get(&variant).copied() {
            return Some(size);
        }
        let path = self.model_path(variant);
        if !path.exists() {
            return None;
        }
        let computed = tokio::task::spawn_blocking(move || Self::dir_size(&path).ok())
            .await
            .ok()
            .flatten()?;
        self.downloaded_sizes
            .write()
            .await
            .insert(variant, computed);
        Some(computed)
    }

    /// Drop the memoized on-disk size for a variant after its directory was
    /// created, modified, or removed.
    pub(crate) async fn invalidate_downloaded_size(&self, variant: ModelVariant) {
        self.downloaded_sizes.write().await.remove(&variant);
    }

    /// Resolve the expected total download size for a model variant.
    ///
    /// This uses the same file planning logic as the downloader itself, then caches
    /// results so list APIs do not repeatedly hit Hugging Face.
    pub async fn expected_size_bytes(&self, variant: ModelVariant) -> u64 {
        if let Some(size) = self.expected_sizes.get(variant).await {
            return size;
        }

        if self.offline {
            return variant.estimated_size();
        }
        let resolved = match self.get_file_download_plans(variant).await {
            Ok(plans) => plans.iter().map(|plan| plan.expected_size).sum::<u64>(),
            Err(err) => {
                warn!(
                    "Falling back to built-in size estimate for {}: {}",
                    variant, err
                );
                variant.estimated_size()
            }
        };

        self.expected_sizes.insert(variant, resolved).await;

        resolved
    }

    /// Non-blocking lookup of a previously resolved expected size. Returns
    /// `None` when no Hugging Face resolution has completed yet, so the caller
    /// can fall back to the built-in estimate without any network I/O.
    pub async fn cached_expected_size_bytes(&self, variant: ModelVariant) -> Option<u64> {
        self.expected_sizes.get(variant).await
    }

    /// Subscribe to progress updates for an active download
    pub async fn subscribe_progress(
        &self,
        variant: ModelVariant,
    ) -> Result<broadcast::Receiver<DownloadProgress>> {
        let downloads = self.active_downloads.read().await;
        if let Some(active) = downloads.get(&variant) {
            Ok(active.subscribe())
        } else {
            Err(Error::DownloadError(format!(
                "No active download for {}",
                variant
            )))
        }
    }

    /// Cancel an active download
    pub async fn cancel_download(&self, variant: ModelVariant) -> Result<()> {
        let handle = {
            let mut downloads = self.active_downloads.write().await;
            downloads.remove(&variant).map(|d| d.handle)
        };

        if let Some(h) = handle {
            h.abort();
            // Clean up partial downloads
            let model_path = self.model_path(variant);
            if model_path.exists() {
                let _ = tokio::fs::remove_dir_all(&model_path).await;
            }
            self.clear_latest_progress(variant).await;
            self.invalidate_downloaded_size(variant).await;
            // Update state
            self.state_manager
                .set_state(variant, DownloadState::NotDownloaded)
                .await;
            Ok(())
        } else {
            Err(Error::DownloadError(format!(
                "No active download for {}",
                variant
            )))
        }
    }

    /// Calculate directory size recursively
    pub(crate) fn dir_size(path: &Path) -> Result<u64> {
        let mut size = 0;
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let metadata = entry.metadata()?;
            if metadata.is_file() {
                size += metadata.len();
            } else if metadata.is_dir() {
                size += Self::dir_size(&entry.path())?;
            }
        }
        Ok(size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    /// The multilingual edition takes its own T3 and grapheme tokenizer and
    /// shares the English S3Gen, voice encoder, default voice and watermark.
    #[test]
    fn chatterbox_multilingual_takes_its_own_t3_and_tokenizer() {
        let (downloader, _) = test_downloader();
        let files = downloader.get_model_files(ModelVariant::ChatterboxMultilingual500M);
        for file in [
            "t3_mtl23ls_v2.safetensors",
            "grapheme_mtl_merged_expanded_v1.json",
            "s3gen.safetensors",
            "ve.safetensors",
            "conds.pt",
            PERTH_CHECKPOINT_FILE,
            CANGJIE_FILE,
            PKUSEG_MODEL_FILE,
            PKUSEG_DICT_FILE,
        ] {
            assert!(files.contains(&file.to_string()), "{file}");
        }
        assert!(!files.contains(&"t3_cfg.safetensors".to_string()));
        assert!(!files.contains(&"tokenizer.json".to_string()));
        let total = files
            .iter()
            .map(|file| {
                downloader
                    .get_single_file_size_estimate(ModelVariant::ChatterboxMultilingual500M, file)
            })
            .sum::<u64>();
        assert_eq!(
            total,
            ModelVariant::ChatterboxMultilingual500M.estimated_size()
        );
    }

    /// Chatterbox's PerTh checkpoint comes from its GitHub repository by
    /// commit, at its exact size; every other file of the model stays on the
    /// Hub.
    #[test]
    fn chatterbox_fetches_its_watermarker_from_a_pinned_url() {
        let (downloader, _) = test_downloader();
        let variant = ModelVariant::Chatterbox500M;
        assert!(downloader
            .get_model_files(variant)
            .contains(&PERTH_CHECKPOINT_FILE.to_string()));
        let pinned = pinned_file(variant, PERTH_CHECKPOINT_FILE).expect("pinned");
        assert_eq!(
            pinned.url,
            "https://raw.githubusercontent.com/resemble-ai/Perth/ff1c8ac55a976971245cdd53c18d6131ca00d993/src/perth/perth_net/pretrained/implicit/perth_net_250000.pth.tar"
        );
        assert_eq!(pinned_file(variant, "t3_cfg.safetensors"), None);
        assert_eq!(
            pinned_file(ModelVariant::Kokoro82M, PERTH_CHECKPOINT_FILE),
            None
        );

        let spec = downloader
            .get_model_file_specs(variant)
            .into_iter()
            .find(|spec| spec.local_file == PERTH_CHECKPOINT_FILE)
            .expect("spec");
        let plan = FileDownloadPlan::pinned(&spec, pinned);
        assert_eq!(plan.url(HF_BASE_URL), pinned.url);
        assert!(plan.immutable());
        assert_eq!(plan.exact_size(), Some(37_429_684));
    }

    /// The Chinese frontend's pkuseg model and dictionary come from GitHub
    /// for the multilingual edition only; its Cangjie table stays on the Hub
    /// with the checkpoint.
    #[test]
    fn chatterbox_multilingual_fetches_its_segmenter_from_pinned_urls() {
        let variant = ModelVariant::ChatterboxMultilingual500M;
        let model = pinned_file(variant, PKUSEG_MODEL_FILE).expect("pinned model");
        assert_eq!(
            model.url,
            "https://github.com/explosion/spacy-pkuseg/releases/download/v0.0.26/spacy_ontonotes.zip"
        );
        let dict = pinned_file(variant, PKUSEG_DICT_FILE).expect("pinned dictionary");
        assert_eq!(
            dict.url,
            "https://raw.githubusercontent.com/explosion/spacy-pkuseg/16995617f3b8f5cf32e5b6a77d7d800e201b255b/spacy_pkuseg/dicts/default.pkl"
        );
        assert_eq!(pinned_file(variant, CANGJIE_FILE), None);
        assert_eq!(
            pinned_file(ModelVariant::Chatterbox500M, PKUSEG_MODEL_FILE),
            None
        );
        let (downloader, _) = test_downloader();
        assert!(!downloader
            .get_model_files(ModelVariant::Chatterbox500M)
            .contains(&CANGJIE_FILE.to_string()));
    }

    #[tokio::test]
    async fn a_pinned_file_must_match_its_digest() {
        let dir = std::env::temp_dir().join(format!("izwi-pinned-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("file.bin");
        std::fs::write(&path, b"abc").unwrap();
        // SHA-256 of "abc".
        let pinned = PinnedFile {
            url: "https://example.invalid/file.bin",
            size: 3,
            sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        };
        verify_pinned_file(&path, pinned)
            .await
            .expect("matching bytes");
        let wrong = PinnedFile {
            sha256: "0000000000000000000000000000000000000000000000000000000000000000",
            ..pinned
        };
        let error = verify_pinned_file(&path, wrong).await.unwrap_err();
        assert!(error
            .to_string()
            .contains("does not match its pinned SHA-256"));
        assert!(!path.exists(), "a mismatched file is removed");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A download needs the bytes still missing; a fresh `.nemo` also needs
    /// its extracted checkpoint beside it.
    #[test]
    fn download_disk_requirement_counts_missing_files_and_nemo_extraction() {
        let files = [
            ("model.nemo", 2_500, false),
            ("README.md", 10, false),
            ("done.bin", 1_000, true),
        ];
        assert_eq!(download_disk_requirement(&files, true), 5_010);
        assert_eq!(download_disk_requirement(&files, false), 2_510);
        assert_eq!(
            download_disk_requirement(&[("model.nemo", 2_500, true)], true),
            0
        );
    }

    /// NeMo ASR models are ready from a complete extracted directory alone,
    /// so the 2.5 GB archive can be deleted after the first load.
    #[test]
    fn nemo_asr_models_are_ready_from_their_native_directory() {
        let (downloader, temp_dir) = test_downloader();
        for (variant, native, files) in [
            (
                ModelVariant::ParakeetTdt06BV3,
                "parakeet-native",
                &["model_config.yaml", "tokenizer.vocab", "model_weights.ckpt"][..],
            ),
            (
                ModelVariant::Nemotron35AsrStreaming06B,
                "nemotron-native",
                &["model_config.yaml", "model_weights.ckpt"][..],
            ),
        ] {
            let dir = downloader.model_path(variant).join(native);
            std::fs::create_dir_all(&dir).unwrap();
            assert!(!downloader.is_downloaded(variant), "{variant}");
            for file in files {
                std::fs::write(dir.join(file), b"x").unwrap();
            }
            assert!(downloader.is_downloaded(variant), "{variant}");
        }
        let _ = std::fs::remove_dir_all(temp_dir);
    }

    /// The advertised Kokoro voice count is the voices it speaks: every
    /// downloaded voice except the Japanese and Mandarin ones, which have
    /// no G2P yet and are hidden and refused.
    #[test]
    fn kokoro_advertises_only_the_voices_it_speaks() {
        use crate::models::architectures::kokoro::phonemizer::kokoro_speaker_supported;
        let (downloader, temp_dir) = test_downloader();
        let spoken = downloader
            .get_model_files(ModelVariant::Kokoro82M)
            .iter()
            .filter_map(|file| file.strip_prefix("voices/")?.strip_suffix(".pt"))
            .filter(|voice| kokoro_speaker_supported(voice))
            .count();
        assert_eq!(spoken, ModelVariant::KOKORO_BUILT_IN_VOICE_COUNT);
        let _ = std::fs::remove_dir_all(temp_dir);
    }

    fn test_downloader() -> (ModelDownloader, PathBuf) {
        let temp_dir =
            std::env::temp_dir().join(format!("izwi-downloader-test-{}", Uuid::new_v4()));
        let downloader = ModelDownloader::new(temp_dir.clone()).expect("downloader");
        (downloader, temp_dir)
    }

    /// How the test server answers.
    struct Behaviour {
        body: Arc<Vec<u8>>,
        etag: &'static str,
        /// Honour `Range` (and `If-Range`), as the Hub and its CDN do.
        honor_range: bool,
        /// Body bytes each successive connection sends before it drops;
        /// connections past the list send everything.
        drop_after: Vec<usize>,
    }

    /// Each request's `Range` start and `If-Range` value.
    type RequestLog = Arc<std::sync::Mutex<Vec<(Option<u64>, Option<String>)>>>;

    /// A local HTTP/1.1 server for one file, one request per connection.
    async fn flaky_server(behaviour: Behaviour) -> (String, RequestLog) {
        use tokio::io::AsyncReadExt;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!(
            "http://{}/repo/resolve/main/model.bin",
            listener.local_addr().unwrap()
        );
        let requests = RequestLog::default();
        let log = requests.clone();
        tokio::spawn(async move {
            for connection in 0.. {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                let mut head = Vec::new();
                let mut buf = [0u8; 1024];
                while !head.windows(4).any(|window| window == b"\r\n\r\n") {
                    match socket.read(&mut buf).await {
                        Ok(0) | Err(_) => break,
                        Ok(read) => head.extend_from_slice(&buf[..read]),
                    }
                }
                let head = String::from_utf8_lossy(&head).to_string();
                let header = |name: &str| {
                    head.lines().find_map(|line| {
                        let (key, value) = line.split_once(':')?;
                        key.eq_ignore_ascii_case(name)
                            .then(|| value.trim().to_string())
                    })
                };
                let range = header("range").and_then(|value| {
                    value
                        .strip_prefix("bytes=")?
                        .strip_suffix('-')?
                        .parse::<u64>()
                        .ok()
                });
                let if_range = header("if-range");
                log.lock().unwrap().push((range, if_range.clone()));

                let body = &behaviour.body;
                let total = body.len() as u64;
                let current = if_range.as_deref().is_none_or(|tag| tag == behaviour.etag);
                let (status, content_range, start) = match range {
                    Some(start) if behaviour.honor_range && current && start >= total => (
                        "416 Range Not Satisfiable",
                        Some(format!("bytes */{total}")),
                        total,
                    ),
                    Some(start) if behaviour.honor_range && current => (
                        "206 Partial Content",
                        Some(format!("bytes {start}-{}/{total}", total - 1)),
                        start,
                    ),
                    _ => ("200 OK", None, 0),
                };
                let content = &body[start as usize..];
                let mut response = format!(
                    "HTTP/1.1 {status}\r\nContent-Length: {}\r\nETag: {}\r\nConnection: close\r\n",
                    content.len(),
                    behaviour.etag
                );
                if let Some(content_range) = content_range {
                    response.push_str(&format!("Content-Range: {content_range}\r\n"));
                }
                response.push_str("\r\n");
                let sent = behaviour
                    .drop_after
                    .get(connection)
                    .map_or(content.len(), |limit| (*limit).min(content.len()));
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.write_all(&content[..sent]).await;
                let _ = socket.flush().await;
                // Dropping the socket ends the response, short or not.
            }
        });
        (url, requests)
    }

    fn test_body() -> Arc<Vec<u8>> {
        Arc::new((0..1_000_000u32).map(|i| (i * 31 + i / 7) as u8).collect())
    }

    async fn write_partial(dest: &Path, bytes: &[u8], sidecar: &PartialDownload) {
        let (partial, sidecar_path) = partial_paths(dest).unwrap();
        tokio::fs::create_dir_all(dest.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::write(&partial, bytes).await.unwrap();
        tokio::fs::write(&sidecar_path, serde_json::to_vec(sidecar).unwrap())
            .await
            .unwrap();
    }

    async fn assert_published(dest: &Path, body: &[u8]) {
        assert_eq!(tokio::fs::read(dest).await.unwrap(), body);
        let (partial, sidecar) = partial_paths(dest).unwrap();
        assert!(!partial.exists() && !sidecar.exists());
    }

    #[test]
    fn content_range_headers_parse() {
        assert_eq!(
            parse_content_range("bytes 100-999/1000"),
            Some((Some(100), Some(1000)))
        );
        assert_eq!(
            parse_content_range("bytes */1000"),
            Some((None, Some(1000)))
        );
        assert_eq!(parse_content_range("bytes 0-9/*"), Some((Some(0), None)));
        assert_eq!(parse_content_range("items 0-9/10"), None);
    }

    /// Each dropped connection resumes from the bytes already on disk, and
    /// the attempts that add bytes do not use up the retry budget: six
    /// attempts here, four allowed without progress.
    #[tokio::test]
    async fn a_dropped_download_resumes_where_it_stopped() {
        let body = test_body();
        let (url, requests) = flaky_server(Behaviour {
            body: body.clone(),
            etag: "\"v1\"",
            honor_range: true,
            drop_after: vec![300_000, 0, 200_000, 100_000, 150_000],
        })
        .await;
        let (downloader, temp_dir) = test_downloader();
        let dest = temp_dir.join("model.bin");

        let bytes = downloader
            .fetch_with_retries(
                &url,
                false,
                Some(body.len() as u64),
                &dest,
                "model.bin",
                &FileProgress::default(),
            )
            .await
            .unwrap();

        assert_eq!(bytes, body.len() as u64);
        assert_published(&dest, &body).await;
        let etag = Some("\"v1\"".to_string());
        assert_eq!(
            *requests.lock().unwrap(),
            vec![
                (None, None),
                (Some(300_000), etag.clone()),
                (Some(300_000), etag.clone()),
                (Some(500_000), etag.clone()),
                (Some(600_000), etag.clone()),
                (Some(750_000), etag),
            ]
        );
        let _ = std::fs::remove_dir_all(temp_dir);
    }

    /// A server that answers a range with the whole file is read from the
    /// start again rather than appended to the partial file.
    #[tokio::test]
    async fn a_server_that_ignores_range_restarts_the_file() {
        let body = test_body();
        let (url, requests) = flaky_server(Behaviour {
            body: body.clone(),
            etag: "\"v1\"",
            honor_range: false,
            drop_after: vec![300_000],
        })
        .await;
        let (downloader, temp_dir) = test_downloader();
        let dest = temp_dir.join("model.bin");

        downloader
            .fetch_with_retries(
                &url,
                true,
                None,
                &dest,
                "model.bin",
                &FileProgress::default(),
            )
            .await
            .unwrap();

        assert_published(&dest, &body).await;
        assert_eq!(requests.lock().unwrap()[1].0, Some(300_000));
        let _ = std::fs::remove_dir_all(temp_dir);
    }

    /// Under an unpinned revision the file may have changed since the
    /// partial was written: `If-Range` with the old ETag gets the whole new
    /// file, never old bytes followed by new ones.
    #[tokio::test]
    async fn a_changed_file_is_downloaded_again_not_spliced() {
        let body = test_body();
        let (url, requests) = flaky_server(Behaviour {
            body: body.clone(),
            etag: "\"v2\"",
            honor_range: true,
            drop_after: Vec::new(),
        })
        .await;
        let (downloader, temp_dir) = test_downloader();
        let dest = temp_dir.join("model.bin");
        let sidecar = PartialDownload {
            url: url.clone(),
            etag: Some("\"v1\"".into()),
            total: Some(body.len() as u64),
        };
        write_partial(&dest, &[0xAA; 400_000], &sidecar).await;

        downloader
            .fetch_with_retries(
                &url,
                false,
                None,
                &dest,
                "model.bin",
                &FileProgress::default(),
            )
            .await
            .unwrap();

        assert_published(&dest, &body).await;
        assert_eq!(
            *requests.lock().unwrap(),
            vec![(Some(400_000), Some("\"v1\"".to_string()))]
        );
        let _ = std::fs::remove_dir_all(temp_dir);
    }

    /// A partial file whose bytes cannot be checked (an unpinned revision
    /// with no validator, or another source) is not resumed.
    #[tokio::test]
    async fn an_unverifiable_partial_starts_over() {
        let body = test_body();
        let (url, requests) = flaky_server(Behaviour {
            body: body.clone(),
            etag: "\"v1\"",
            honor_range: true,
            drop_after: Vec::new(),
        })
        .await;
        let (downloader, temp_dir) = test_downloader();
        for (index, sidecar) in [
            PartialDownload {
                url: url.clone(),
                etag: None,
                total: None,
            },
            PartialDownload {
                url: format!("{url}?other"),
                etag: Some("\"v1\"".into()),
                total: None,
            },
        ]
        .iter()
        .enumerate()
        {
            let dest = temp_dir.join(format!("model-{index}.bin"));
            write_partial(&dest, &[0xAA; 400_000], sidecar).await;
            downloader
                .fetch_with_retries(
                    &url,
                    false,
                    None,
                    &dest,
                    "model.bin",
                    &FileProgress::default(),
                )
                .await
                .unwrap();
            assert_published(&dest, &body).await;
        }
        assert!(requests
            .lock()
            .unwrap()
            .iter()
            .all(|(range, if_range)| range.is_none() && if_range.is_none()));
        let _ = std::fs::remove_dir_all(temp_dir);
    }

    /// A partial file that already holds every byte (the process stopped
    /// before renaming it) is published without another transfer.
    #[tokio::test]
    async fn a_complete_partial_is_published_without_a_transfer() {
        let body = test_body();
        let (url, requests) = flaky_server(Behaviour {
            body: body.clone(),
            etag: "\"v1\"",
            honor_range: true,
            drop_after: Vec::new(),
        })
        .await;
        let (downloader, temp_dir) = test_downloader();
        let dest = temp_dir.join("model.bin");
        let sidecar = PartialDownload {
            url: url.clone(),
            etag: Some("\"v1\"".into()),
            total: Some(body.len() as u64),
        };
        write_partial(&dest, &body, &sidecar).await;

        let bytes = downloader
            .fetch_with_retries(
                &url,
                true,
                None,
                &dest,
                "model.bin",
                &FileProgress::default(),
            )
            .await
            .unwrap();

        assert_eq!(bytes, body.len() as u64);
        assert_published(&dest, &body).await;
        assert_eq!(
            *requests.lock().unwrap(),
            vec![(Some(body.len() as u64), Some("\"v1\"".to_string()))]
        );
        let _ = std::fs::remove_dir_all(temp_dir);
    }

    /// A download fails after four attempts in a row that add no bytes.
    #[tokio::test]
    async fn a_download_without_progress_fails_after_four_attempts() {
        let (url, requests) = flaky_server(Behaviour {
            body: test_body(),
            etag: "\"v1\"",
            honor_range: true,
            drop_after: vec![0; 8],
        })
        .await;
        let (downloader, temp_dir) = test_downloader();
        let dest = temp_dir.join("model.bin");

        let error = downloader
            .fetch_with_retries(
                &url,
                true,
                None,
                &dest,
                "model.bin",
                &FileProgress::default(),
            )
            .await
            .unwrap_err();

        assert!(
            error.to_string().contains("4 attempts without progress"),
            "{error}"
        );
        assert_eq!(requests.lock().unwrap().len(), MAX_STALLED_FILE_ATTEMPTS);
        assert!(!dest.exists());
        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[tokio::test]
    async fn offline_mode_refuses_every_download_path_without_network() {
        let (mut downloader, temp_dir) = test_downloader();
        downloader.offline = true;
        let variant = *ModelVariant::all()
            .iter()
            .find(|variant| variant.is_enabled())
            .expect("an enabled model");

        for refused in [
            downloader.download(variant).await.err(),
            downloader.spawn_download(variant).await.err(),
        ] {
            let message = refused.expect("offline downloads are refused").to_string();
            assert!(message.contains("IZWI_OFFLINE"), "{message}");
        }
        let (progress, _) = broadcast::channel(4);
        assert!(downloader
            .download_with_progress(variant, progress)
            .await
            .is_err());
        // Sizes fall back to the built-in estimate instead of asking the hub.
        assert_eq!(
            downloader.expected_size_bytes(variant).await,
            variant.estimated_size()
        );
        let _ = std::fs::remove_dir_all(temp_dir);
    }

    fn synthetic_qwen38_index() -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "metadata": { "total_size": 3 },
            "weight_map": {
                "model.language_model.layers.0.weight": "layers-0.safetensors",
                "model.language_model.layers.0.weight_scale_inv": "layers-0.safetensors",
                "model.language_model.embed_tokens.weight": "outside.safetensors",
                "mtp.layers.0.weight": "mtp.safetensors"
            }
        }))
        .expect("synthetic index")
    }

    #[test]
    fn qwen38_index_closure_is_validated_and_deduplicated() {
        assert_eq!(
            indexed_safetensor_shards(&synthetic_qwen38_index()).expect("valid index"),
            vec![
                "layers-0.safetensors".to_string(),
                "mtp.safetensors".to_string(),
                "outside.safetensors".to_string(),
            ]
        );

        for unsafe_path in [
            "../escape.safetensors",
            "/absolute.safetensors",
            "nested\\windows.safetensors",
            "nested//empty.safetensors",
            "./relative.safetensors",
            "not-a-shard.bin",
        ] {
            let bytes = serde_json::to_vec(&serde_json::json!({
                "weight_map": { "tensor": unsafe_path }
            }))
            .expect("index");
            assert!(
                indexed_safetensor_shards(&bytes).is_err(),
                "unsafe path should fail: {unsafe_path}"
            );
        }
    }

    #[tokio::test]
    async fn qwen38_indexed_specs_use_pinned_revision_and_exact_index_closure() {
        let (downloader, temp_dir) = test_downloader();
        let model_dir = downloader.model_path(ModelVariant::Qwen3827BFp8);
        std::fs::create_dir_all(&model_dir).expect("model dir");
        std::fs::write(
            model_dir.join("model.safetensors.index.json"),
            synthetic_qwen38_index(),
        )
        .expect("index");

        let specs = downloader
            .get_qwen38_indexed_file_specs()
            .await
            .expect("indexed specs");
        let files = specs
            .iter()
            .map(|spec| spec.local_file.as_str())
            .collect::<BTreeSet<_>>();
        assert!(files.contains("layers-0.safetensors"));
        assert!(files.contains("outside.safetensors"));
        assert!(files.contains("mtp.safetensors"));
        assert!(files.contains("chat_template.jinja"));
        assert!(files.contains("video_preprocessor_config.json"));
        assert!(specs.iter().all(|spec| {
            spec.source_repo == "Qwen/Qwen3.8-27B-FP8"
                && spec.source_revision == ModelVariant::QWEN38_27B_FP8_ARTIFACT_REVISION
        }));

        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn qwen38_is_downloaded_requires_index_closure_and_pinned_manifest() {
        let (downloader, temp_dir) = test_downloader();
        let variant = ModelVariant::Qwen3827BFp8;
        let model_dir = downloader.model_path(variant);
        std::fs::create_dir_all(&model_dir).expect("model dir");
        let index = synthetic_qwen38_index();
        for file in QWEN38_REQUIRED_METADATA_FILES {
            let bytes = if *file == "model.safetensors.index.json" {
                index.as_slice()
            } else {
                b"{}"
            };
            std::fs::write(model_dir.join(file), bytes).expect("metadata");
        }
        std::fs::write(model_dir.join("layers-0.safetensors"), [0u8]).expect("layer shard");
        assert!(
            !downloader.is_downloaded(variant),
            "one shard must never make the 66-shard bundle ready"
        );

        std::fs::write(model_dir.join("outside.safetensors"), [0u8]).expect("outside shard");
        std::fs::write(model_dir.join("mtp.safetensors"), [0u8]).expect("mtp shard");
        assert!(
            !downloader.is_downloaded(variant),
            "a complete closure without provenance metadata is not reproducible"
        );

        let manifest = ArtifactManifest {
            schema_version: 1,
            variant,
            repo_id: variant.repo_id().to_string(),
            revision: ModelVariant::QWEN38_27B_FP8_ARTIFACT_REVISION.to_string(),
            files: qwen38_selected_files(&index).expect("selected files"),
        };
        std::fs::write(
            model_dir.join(ARTIFACT_MANIFEST_FILE),
            serde_json::to_vec_pretty(&manifest).expect("manifest"),
        )
        .expect("write manifest");
        assert!(downloader.is_downloaded(variant));

        std::fs::remove_file(model_dir.join("outside.safetensors")).expect("remove shard");
        assert!(!downloader.is_downloaded(variant));
        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn qwen35_model_files_only_include_core_assets() {
        let (downloader, temp_dir) = test_downloader();
        let files = downloader.get_model_files(ModelVariant::Qwen354BGguf);
        assert_eq!(
            files,
            vec![
                "Qwen3.5-4B-Q4_K_M.gguf".to_string(),
                "mmproj-F16.gguf".to_string(),
            ]
        );
        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn qwen35_is_downloaded_with_gguf_and_mmproj_only() {
        let (downloader, temp_dir) = test_downloader();
        let variant = ModelVariant::Qwen354BGguf;
        let model_dir = downloader.model_path(variant);
        std::fs::create_dir_all(&model_dir).expect("model dir");
        std::fs::write(model_dir.join("Qwen3.5-4B-Q4_K_M.gguf"), [0u8]).expect("gguf");
        std::fs::write(model_dir.join("mmproj-F16.gguf"), [0u8]).expect("mmproj");
        assert!(downloader.is_downloaded(variant));
        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn qwen3_asr_gguf_model_files_only_include_single_checkpoint() {
        let (downloader, temp_dir) = test_downloader();
        let files = downloader.get_model_files(ModelVariant::Qwen3Asr06BGguf);
        assert_eq!(files, vec!["qwen3_asr_0.6b_q8_0.gguf".to_string()]);
        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn qwen3_asr_gguf_is_downloaded_when_checkpoint_exists() {
        let (downloader, temp_dir) = test_downloader();
        let variant = ModelVariant::Qwen3Asr17BGguf;
        let model_dir = downloader.model_path(variant);
        std::fs::create_dir_all(&model_dir).expect("model dir");
        assert!(!downloader.is_downloaded(variant));
        std::fs::write(model_dir.join("qwen3_asr_1.7b_q8_0.gguf"), [0u8]).expect("gguf");
        assert!(downloader.is_downloaded(variant));
        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn nemotron_asr_model_files_only_include_nemo_checkpoint() {
        let (downloader, temp_dir) = test_downloader();
        let files = downloader.get_model_files(ModelVariant::Nemotron35AsrStreaming06B);
        assert_eq!(
            files,
            vec!["nemotron-3.5-asr-streaming-0.6b.nemo".to_string()]
        );
        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn nemotron3_diarization_model_files_only_include_nemo_checkpoint() {
        let (downloader, temp_dir) = test_downloader();
        let files = downloader.get_model_files(ModelVariant::Nemotron3Diarization);
        assert_eq!(files, vec!["Nemotron-3-Diarization.nemo".to_string()]);
        std::fs::remove_dir_all(temp_dir).ok();
    }

    /// NDZ-5: the diarizers download the `.nemo` revision their loader was
    /// verified against, not mutable `main`.
    #[test]
    fn sortformer_diarizers_download_their_pinned_revision() {
        let (downloader, temp_dir) = test_downloader();
        for (variant, revision) in [
            (
                ModelVariant::DiarStreamingSortformer4SpkV21,
                ModelVariant::SORTFORMER_4SPK_V21_ARTIFACT_REVISION,
            ),
            (
                ModelVariant::Nemotron3Diarization,
                ModelVariant::NEMOTRON3_DIARIZATION_ARTIFACT_REVISION,
            ),
        ] {
            let specs = downloader.get_model_file_specs(variant);
            assert!(!specs.is_empty(), "{variant}");
            for spec in specs {
                assert_eq!(
                    spec.source_revision, revision,
                    "{variant} {}",
                    spec.source_file
                );
            }
        }
        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn nemotron3_diarization_is_downloaded_when_nemo_checkpoint_exists() {
        let (downloader, temp_dir) = test_downloader();
        let variant = ModelVariant::Nemotron3Diarization;
        let model_dir = downloader.model_path(variant);
        std::fs::create_dir_all(&model_dir).expect("model dir");
        assert!(!downloader.is_downloaded(variant));
        std::fs::write(model_dir.join("Nemotron-3-Diarization.nemo"), [0u8])
            .expect("nemo");
        assert!(downloader.is_downloaded(variant));
        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn nemotron_asr_is_downloaded_when_nemo_checkpoint_exists() {
        let (downloader, temp_dir) = test_downloader();
        let variant = ModelVariant::Nemotron35AsrStreaming06B;
        let model_dir = downloader.model_path(variant);
        std::fs::create_dir_all(&model_dir).expect("model dir");
        assert!(!downloader.is_downloaded(variant));
        std::fs::write(
            model_dir.join("nemotron-3.5-asr-streaming-0.6b.nemo"),
            [0u8],
        )
        .expect("nemo");
        assert!(downloader.is_downloaded(variant));
        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn granite_speech_model_files_include_required_hf_bundle() {
        let (downloader, temp_dir) = test_downloader();
        let files = downloader.get_model_files(ModelVariant::GraniteSpeech412BPlus);
        assert_eq!(
            files,
            GRANITE_SPEECH_PLUS_FILES
                .iter()
                .map(|file| (*file).to_string())
                .collect::<Vec<_>>()
        );
        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn granite_speech_base_files_cover_the_loader_and_skip_unused_assets() {
        let (downloader, temp_dir) = test_downloader();
        let files = downloader.get_model_files(ModelVariant::GraniteSpeech412B);
        assert!(files.iter().any(|file| file == "preprocessor_config.json"));
        for unused in [
            "generation_config.json",
            "out_llm.safetensors",
            "multilingual_sample.wav",
        ] {
            assert!(!files.iter().any(|file| file == unused), "{unused}");
        }
        assert!(!files.iter().any(|file| file.starts_with(".eval_results")));
        let shard_bytes = files
            .iter()
            .filter(|file| file.starts_with("model-0000"))
            .map(|file| {
                downloader.get_single_file_size_estimate(ModelVariant::GraniteSpeech412B, file)
            })
            .sum::<u64>();
        assert_eq!(shard_bytes, 4_626_527_776);
        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn granite_speech_5_files_are_the_unsharded_bundle_without_unused_assets() {
        let (downloader, temp_dir) = test_downloader();
        let files = downloader.get_model_files(ModelVariant::GraniteSpeech502B);
        assert_eq!(
            files,
            GRANITE_SPEECH_5_FILES
                .iter()
                .map(|file| file.to_string())
                .collect::<Vec<_>>()
        );
        for unused in [
            "model.safetensors.index.json",
            "out_llm.safetensors",
            "multilingual_sample.wav",
            "vocab.json",
            "merges.txt",
        ] {
            assert!(!files.iter().any(|file| file == unused), "{unused}");
        }
        assert_eq!(
            downloader.get_single_file_size_estimate(
                ModelVariant::GraniteSpeech502B,
                "model.safetensors"
            ),
            GRANITE_SPEECH_5_MODEL_BYTES
        );
        std::fs::remove_dir_all(temp_dir).ok();
    }

    /// The GGUF release takes its two GGUFs from the GGUF repo and the
    /// configs and tokenizer from the safetensors release, each pinned.
    #[test]
    fn granite_speech_5_gguf_specs_pin_both_repos() {
        let (downloader, temp_dir) = test_downloader();
        let variant = ModelVariant::GraniteSpeech502BGguf;
        let specs = downloader.get_model_file_specs(variant);
        assert_eq!(specs.len(), GRANITE_SPEECH_5_GGUF_FILES.len());
        for spec in &specs {
            let (repo, revision) = if spec.local_file.ends_with(".gguf") {
                (
                    "ibm-granite/granite-speech-5.0-2b-GGUF",
                    ModelVariant::GRANITE_SPEECH_5_0_2B_GGUF_ARTIFACT_REVISION,
                )
            } else {
                (
                    "ibm-granite/granite-speech-5.0-2b",
                    ModelVariant::GRANITE_SPEECH_5_0_2B_ARTIFACT_REVISION,
                )
            };
            assert_eq!(spec.source_repo, repo, "{}", spec.local_file);
            assert_eq!(spec.source_revision, revision, "{}", spec.local_file);
            assert_eq!(spec.source_file, spec.local_file);
        }
        assert!(specs
            .iter()
            .any(|spec| spec.local_file == "granite-speech-5.0-2b-Q8_0.gguf"));
        assert!(!specs
            .iter()
            .any(|spec| spec.local_file == "model.safetensors"));
        assert_eq!(
            downloader.get_single_file_size_estimate(variant, "mmproj-model-f16.gguf"),
            GRANITE_SPEECH_5_GGUF_MMPROJ_BYTES
        );
        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn granite_speech_is_downloaded_only_when_all_required_assets_exist() {
        let (downloader, temp_dir) = test_downloader();
        for variant in [
            ModelVariant::GraniteSpeech412B,
            ModelVariant::GraniteSpeech412BPlus,
            ModelVariant::GraniteSpeech502B,
            ModelVariant::GraniteSpeech502BGguf,
        ] {
            let required = granite_speech_required_files(variant);
            let model_dir = downloader.model_path(variant);
            std::fs::create_dir_all(&model_dir).expect("model dir");

            for file in required.iter().take(required.len() - 1) {
                std::fs::write(model_dir.join(file), [0u8]).expect("partial granite asset");
            }
            assert!(!downloader.is_downloaded(variant));

            std::fs::write(model_dir.join(required.last().unwrap()), [0u8])
                .expect("last granite asset");
            assert!(downloader.is_downloaded(variant));
        }

        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn lfm25_audio_model_files_include_full_q40_bundle() {
        let (downloader, temp_dir) = test_downloader();
        let files = downloader.get_model_files(ModelVariant::Lfm25Audio15BGguf);
        assert_eq!(
            files,
            vec![
                "LFM2.5-Audio-1.5B-Q4_0.gguf".to_string(),
                "mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf".to_string(),
                "tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf".to_string(),
                "vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf".to_string(),
            ]
        );
        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn lfm25_audio_is_downloaded_only_when_all_bundle_files_exist() {
        let (downloader, temp_dir) = test_downloader();
        let variant = ModelVariant::Lfm25Audio15BGguf;
        let model_dir = downloader.model_path(variant);
        std::fs::create_dir_all(&model_dir).expect("model dir");
        std::fs::write(model_dir.join("LFM2.5-Audio-1.5B-Q4_0.gguf"), [0u8]).expect("main");
        std::fs::write(model_dir.join("mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf"), [0u8])
            .expect("mmproj");
        std::fs::write(
            model_dir.join("tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf"),
            [0u8],
        )
        .expect("tokenizer");
        assert!(!downloader.is_downloaded(variant));
        std::fs::write(model_dir.join("vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf"), [0u8])
            .expect("vocoder");
        assert!(downloader.is_downloaded(variant));
        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn voxtral_model_files_include_required_mistral_assets() {
        let (downloader, temp_dir) = test_downloader();
        let files = downloader.get_model_files(ModelVariant::VoxtralMini4BRealtime2602);
        assert_eq!(
            files,
            vec![
                "params.json".to_string(),
                "config.json".to_string(),
                "tekken.json".to_string(),
                "consolidated.safetensors".to_string(),
            ]
        );
        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn voxtral_is_downloaded_only_when_required_mistral_assets_exist() {
        let (downloader, temp_dir) = test_downloader();
        let variant = ModelVariant::VoxtralMini4BRealtime2602;
        let model_dir = downloader.model_path(variant);
        std::fs::create_dir_all(&model_dir).expect("model dir");
        std::fs::write(model_dir.join("params.json"), "{}").expect("params");
        std::fs::write(model_dir.join("config.json"), "{}").expect("config");
        std::fs::write(model_dir.join("tekken.json"), "{}").expect("tekken");
        assert!(!downloader.is_downloaded(variant));
        std::fs::write(model_dir.join("consolidated.safetensors"), [0u8]).expect("weights");
        assert!(downloader.is_downloaded(variant));
        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn voxtral_tts_model_files_include_voice_embeddings_without_config_json() {
        let (downloader, temp_dir) = test_downloader();
        let files = downloader.get_model_files(ModelVariant::Voxtral4BTts2603);

        assert!(files.contains(&"params.json".to_string()));
        assert!(files.contains(&"tekken.json".to_string()));
        assert!(files.contains(&"consolidated.safetensors".to_string()));
        assert!(!files.contains(&"config.json".to_string()));
        assert_eq!(
            files
                .iter()
                .filter(|file| file.starts_with("voice_embedding/") && file.ends_with(".pt"))
                .count(),
            ModelVariant::VOXTRAL_TTS_BUILT_IN_VOICE_COUNT
        );
        assert!(files.contains(&"voice_embedding/casual_male.pt".to_string()));
        assert!(files.contains(&"voice_embedding/hi_female.pt".to_string()));

        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn voxtral_tts_is_downloaded_only_when_all_voice_embeddings_exist() {
        let (downloader, temp_dir) = test_downloader();
        let variant = ModelVariant::Voxtral4BTts2603;
        let model_dir = downloader.model_path(variant);
        std::fs::create_dir_all(model_dir.join("voice_embedding")).expect("model dir");
        std::fs::write(model_dir.join("params.json"), "{}").expect("params");
        std::fs::write(model_dir.join("tekken.json"), "{}").expect("tekken");
        std::fs::write(model_dir.join("consolidated.safetensors"), [0u8]).expect("weights");

        for file in VOXTRAL_TTS_VOICE_EMBEDDINGS
            .iter()
            .take(VOXTRAL_TTS_VOICE_EMBEDDINGS.len() - 1)
        {
            std::fs::write(model_dir.join(file), [0u8]).expect("voice embedding");
        }
        assert!(!downloader.is_downloaded(variant));

        std::fs::write(
            model_dir.join(VOXTRAL_TTS_VOICE_EMBEDDINGS.last().unwrap()),
            [0u8],
        )
        .expect("last voice embedding");
        assert!(downloader.is_downloaded(variant));

        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn vibevoice_tts_files_include_model_shards_and_qwen_tokenizer_assets() {
        let (downloader, temp_dir) = test_downloader();
        let files = downloader.get_model_files(ModelVariant::VibeVoice15BTts);

        assert!(files.contains(&"config.json".to_string()));
        assert!(files.contains(&"preprocessor_config.json".to_string()));
        assert!(files.contains(&"model.safetensors.index.json".to_string()));
        assert!(files.contains(&"model-00001-of-00003.safetensors".to_string()));
        assert!(files.contains(&"model-00003-of-00003.safetensors".to_string()));
        assert!(files.contains(&"tokenizer.json".to_string()));
        assert!(files.contains(&"tokenizer_config.json".to_string()));

        let specs = downloader.get_model_file_specs(ModelVariant::VibeVoice15BTts);
        let tokenizer_spec = specs
            .iter()
            .find(|spec| spec.local_file == "tokenizer.json")
            .expect("tokenizer spec");
        assert_eq!(tokenizer_spec.source_repo, "Qwen/Qwen2.5-1.5B");
        let shard_spec = specs
            .iter()
            .find(|spec| spec.local_file == "model-00001-of-00003.safetensors")
            .expect("shard spec");
        assert_eq!(shard_spec.source_repo, "microsoft/VibeVoice-1.5B");

        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn lfm25_chat_companion_files_come_from_pinned_safetensors_revisions() {
        let (downloader, temp_dir) = test_downloader();
        for (variant, repo, revision) in [
            (
                ModelVariant::Lfm2512BInstructGguf,
                "LiquidAI/LFM2.5-1.2B-Instruct",
                "0f604ada3f766f9f257460c4c9f0b5d6f69d431b",
            ),
            (
                ModelVariant::Lfm2512BThinkingGguf,
                "LiquidAI/LFM2.5-1.2B-Thinking",
                "f313478934a7612d22991f752959d7a1a8756fec",
            ),
        ] {
            let specs = downloader.get_model_file_specs(variant);
            let tokenizer = specs
                .iter()
                .find(|spec| spec.local_file == "tokenizer.json")
                .expect("tokenizer spec");
            assert_eq!(tokenizer.source_repo, repo);
            assert_eq!(tokenizer.source_revision, revision);
            let gguf = specs
                .iter()
                .find(|spec| spec.local_file.ends_with(".gguf"))
                .expect("gguf spec");
            assert_eq!(gguf.source_repo, variant.repo_id());
            assert!(specs
                .iter()
                .filter(|spec| !spec.local_file.ends_with(".gguf"))
                .all(|spec| spec.source_revision == revision));
        }
        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn vibevoice_asr_files_include_eight_model_shards_and_qwen_tokenizer_assets() {
        let (downloader, temp_dir) = test_downloader();
        let files = downloader.get_model_files(ModelVariant::VibeVoiceAsr);

        assert!(files.contains(&"config.json".to_string()));
        assert!(files.contains(&"model.safetensors.index.json".to_string()));
        assert!(files.contains(&"model-00001-of-00008.safetensors".to_string()));
        assert!(files.contains(&"model-00008-of-00008.safetensors".to_string()));
        assert!(files.contains(&"tokenizer.json".to_string()));
        assert!(files.contains(&"tokenizer_config.json".to_string()));

        let specs = downloader.get_model_file_specs(ModelVariant::VibeVoiceAsr);
        let tokenizer_spec = specs
            .iter()
            .find(|spec| spec.local_file == "tokenizer.json")
            .expect("tokenizer spec");
        assert_eq!(tokenizer_spec.source_repo, "Qwen/Qwen2.5-7B");
        let shard_spec = specs
            .iter()
            .find(|spec| spec.local_file == "model-00008-of-00008.safetensors")
            .expect("shard spec");
        assert_eq!(shard_spec.source_repo, "microsoft/VibeVoice-ASR");

        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn vibevoice_is_downloaded_only_when_required_assets_exist() {
        let (downloader, temp_dir) = test_downloader();
        let variant = ModelVariant::VibeVoice15BTts;
        let model_dir = downloader.model_path(variant);
        std::fs::create_dir_all(&model_dir).expect("model dir");
        std::fs::write(model_dir.join("config.json"), "{}").expect("config");
        std::fs::write(model_dir.join("preprocessor_config.json"), "{}").expect("processor");
        std::fs::write(model_dir.join("model.safetensors.index.json"), "{}").expect("index");
        std::fs::write(model_dir.join("tokenizer.json"), "{}").expect("tokenizer");
        std::fs::write(model_dir.join("tokenizer_config.json"), "{}").expect("tokenizer config");
        for file in vibevoice_safetensor_shards(variant).iter().take(2) {
            std::fs::write(model_dir.join(file), [0u8]).expect("partial shard");
        }
        assert!(!downloader.is_downloaded(variant));

        std::fs::write(model_dir.join("model-00003-of-00003.safetensors"), [0u8])
            .expect("last shard");
        assert!(downloader.is_downloaded(variant));

        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn fish_s2_files_include_required_model_and_codec_bundle() {
        let (downloader, temp_dir) = test_downloader();
        let files = downloader.get_model_files(ModelVariant::FishAudioS2Pro);

        assert!(files.contains(&"config.json".to_string()));
        assert!(files.contains(&"chat_template.jinja".to_string()));
        assert!(files.contains(&"codec.pth".to_string()));
        assert!(files.contains(&"model.safetensors.index.json".to_string()));
        assert!(files.contains(&"model-00001-of-00002.safetensors".to_string()));
        assert!(files.contains(&"model-00002-of-00002.safetensors".to_string()));
        assert!(files.contains(&"tokenizer.json".to_string()));
        assert!(files.contains(&"tokenizer_config.json".to_string()));
        assert!(files.contains(&"special_tokens_map.json".to_string()));

        let specs = downloader.get_model_file_specs(ModelVariant::FishAudioS2Pro);
        assert!(specs
            .iter()
            .all(|spec| spec.source_repo == "fishaudio/s2-pro"));
        assert!(specs
            .iter()
            .all(|spec| spec.source_revision == ModelVariant::FISH_S2_PRO_ARTIFACT_REVISION));

        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn fish_s2_is_downloaded_only_when_required_assets_exist() {
        let (downloader, temp_dir) = test_downloader();
        let variant = ModelVariant::FishAudioS2Pro;
        let model_dir = downloader.model_path(variant);
        std::fs::create_dir_all(&model_dir).expect("model dir");

        for file in FISH_S2_PRO_FILES.iter().take(FISH_S2_PRO_FILES.len() - 1) {
            std::fs::write(model_dir.join(file), [0u8]).expect("partial fish s2 asset");
        }
        assert!(!downloader.is_downloaded(variant));

        std::fs::write(model_dir.join(FISH_S2_PRO_FILES.last().unwrap()), [0u8])
            .expect("last fish s2 asset");
        assert!(downloader.is_downloaded(variant));

        std::fs::remove_dir_all(temp_dir).ok();
    }
}
