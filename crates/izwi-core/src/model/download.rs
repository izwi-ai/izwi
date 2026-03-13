//! Model downloading from HuggingFace Hub
//!
//! Features:
//! - Async streaming downloads with byte-level progress
//! - Multi-progress bar terminal display
//! - Non-blocking downloads with background task spawning
//! - Real-time progress via channels

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use futures::StreamExt;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use reqwest;
use tokio::sync::{broadcast, RwLock};
use tokio::task::JoinHandle;
use tokio::time::{Duration, Instant};
use tracing::{debug, info, warn};

use crate::catalog::ModelFamily;
use crate::catalog::ModelVariant;
use crate::error::{Error, Result};

const HF_BASE_URL: &str = "https://huggingface.co";
const CHUNK_SIZE: usize = 8192; // 8KB chunks for streaming

fn qwen_chat_gguf_filename(variant: ModelVariant) -> Option<&'static str> {
    match variant {
        ModelVariant::Qwen306BGguf => Some("Qwen3-0.6B-Q8_0.gguf"),
        ModelVariant::Qwen317BGguf => Some("Qwen3-1.7B-Q8_0.gguf"),
        ModelVariant::Qwen34BGguf => Some("Qwen3-4B-Q4_K_M.gguf"),
        ModelVariant::Qwen38BGguf => Some("Qwen3-8B-Q4_K_M.gguf"),
        ModelVariant::Qwen314BGguf => Some("Qwen3-14B-Q4_K_M.gguf"),
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

#[derive(Debug, Deserialize)]
struct HfRepoTreeEntry {
    path: String,
    size: Option<u64>,
}

#[derive(Debug, Clone)]
struct FileDownloadPlan {
    source_repo: String,
    source_file: String,
    local_file: String,
    expected_size: u64,
    strict_size_check: bool,
}

#[derive(Debug, Clone)]
struct ModelFileSpec {
    source_repo: String,
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

use serde::{Deserialize, Serialize};

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
}

impl DownloadStateManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn set_state(&self, variant: ModelVariant, state: DownloadState) {
        let mut map = self.state.write().await;
        map.insert(variant, state);
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
    expected_size_cache: Arc<RwLock<HashMap<ModelVariant, u64>>>,
    multi_progress: MultiProgress,
    state_manager: DownloadStateManager,
}

impl ModelDownloader {
    /// Create a new downloader
    pub fn new(models_dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&models_dir)?;

        let http_client = reqwest::Client::builder()
            // Large model files can take significant time on slower links.
            .timeout(std::time::Duration::from_secs(6 * 3600))
            // Avoid macOS SystemConfiguration proxy lookups which can fail in
            // restricted runtime environments (sandboxed agents/CI).
            .no_proxy()
            .build()
            .map_err(|e| Error::HfHubError(format!("Failed to create HTTP client: {}", e)))?;

        let multi_progress = MultiProgress::new();
        multi_progress.set_draw_target(indicatif::ProgressDrawTarget::stderr_with_hz(10));

        Ok(Self {
            models_dir,
            http_client,
            active_downloads: Arc::new(RwLock::new(std::collections::HashMap::new())),
            latest_progress: Arc::new(RwLock::new(std::collections::HashMap::new())),
            repo_tree_cache: Arc::new(RwLock::new(HashMap::new())),
            expected_size_cache: Arc::new(RwLock::new(HashMap::new())),
            multi_progress,
            state_manager: DownloadStateManager::new(),
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

    async fn get_repo_tree_index(&self, repo_id: &str) -> Result<HashMap<String, u64>> {
        if let Some(cached) = self.repo_tree_cache.read().await.get(repo_id).cloned() {
            return Ok(cached);
        }

        let url = format!(
            "{}/api/models/{}/tree/main?recursive=1",
            HF_BASE_URL, repo_id
        );
        let response = self
            .http_client
            .get(&url)
            .header("User-Agent", "izwi/0.1.0-beta-7")
            .send()
            .await
            .map_err(|e| Error::HfHubError(format!("Repo tree request failed: {}", e)))?;

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
            .insert(repo_id.to_string(), index.clone());

        Ok(index)
    }

    fn is_not_found_hf_error(err: &Error) -> bool {
        match err {
            Error::HfHubError(msg) | Error::DownloadError(msg) => msg.contains("404"),
            _ => false,
        }
    }

    /// Download a file with streaming and progress bar
    async fn download_file_streaming(
        &self,
        repo_id: &str,
        filename: &str,
        dest: &Path,
        file_pb: Option<ProgressBar>,
        progress_tx: Option<broadcast::Sender<DownloadProgress>>,
        progress_template: Option<DownloadProgress>,
    ) -> Result<u64> {
        let url = format!("{}/{}/resolve/main/{}", HF_BASE_URL, repo_id, filename);
        debug!("Downloading from URL: {}", url);

        let response = self
            .http_client
            .get(&url)
            .header("User-Agent", "izwi/0.1.0")
            .send()
            .await
            .map_err(|e| Error::HfHubError(format!("HTTP request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(Error::HfHubError(format!(
                "HTTP {} for {}",
                response.status(),
                url
            )));
        }

        // Get content length for progress bar
        let total_size = response
            .headers()
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0);

        if let Some(ref pb) = file_pb {
            pb.set_length(total_size);
        }

        // Create parent directories
        if let Some(parent) = dest.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        // Stream download to file
        let mut file = tokio::fs::File::create(dest).await?;
        let mut downloaded = 0u64;
        let mut stream = response.bytes_stream();
        let mut last_progress_emit = Instant::now();
        let mut last_progress_bytes = 0u64;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| Error::HfHubError(format!("Stream error: {}", e)))?;
            tokio::io::AsyncWriteExt::write_all(&mut file, &chunk).await?;
            downloaded += chunk.len() as u64;

            if let Some(ref pb) = file_pb {
                pb.set_position(downloaded);
            }

            // Send real-time progress update
            if let Some(ref tx) = progress_tx {
                if let Some(ref template) = progress_template {
                    let bytes_delta = downloaded.saturating_sub(last_progress_bytes);
                    let should_emit = bytes_delta >= 256 * 1024
                        || last_progress_emit.elapsed() >= Duration::from_millis(200)
                        || (total_size > 0 && downloaded >= total_size);

                    if should_emit {
                        let progress = DownloadProgress {
                            current_file_downloaded: downloaded,
                            current_file_total: total_size.max(downloaded),
                            // Calculate total downloaded: base bytes from completed files + current file progress
                            downloaded_bytes: template.downloaded_bytes + downloaded,
                            ..template.clone()
                        };
                        self.set_latest_progress(progress.clone()).await;
                        let _ = tx.send(progress);
                        last_progress_emit = Instant::now();
                        last_progress_bytes = downloaded;
                    }
                }
            }
        }

        // Ensure we always emit a final per-file progress event, even if throttled updates
        // did not include the final chunk.
        if let Some(ref tx) = progress_tx {
            if let Some(ref template) = progress_template {
                let progress = DownloadProgress {
                    current_file_downloaded: downloaded,
                    current_file_total: total_size.max(downloaded),
                    downloaded_bytes: template.downloaded_bytes + downloaded,
                    ..template.clone()
                };
                self.set_latest_progress(progress.clone()).await;
                let _ = tx.send(progress);
            }
        }

        // Sync file to disk
        file.sync_all().await?;

        debug!("Downloaded {} bytes to {:?}", downloaded, dest);
        Ok(downloaded)
    }

    /// Get the local path for a model variant
    pub fn model_path(&self, variant: ModelVariant) -> PathBuf {
        self.models_dir.join(variant.dir_name())
    }

    /// Check if a model is already downloaded
    pub fn is_downloaded(&self, variant: ModelVariant) -> bool {
        let path = self.model_path(variant);
        if !path.exists() {
            return false;
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
            ModelFamily::Lfm2Audio => {
                let has_base = path.join("model.safetensors").exists()
                    && path.join("config.json").exists()
                    && path.join("tokenizer.json").exists();
                let has_detokenizer = path.join("audio_detokenizer/config.json").exists()
                    && path.join("audio_detokenizer/model.safetensors").exists();
                has_base && has_detokenizer
            }
            ModelFamily::KokoroTts => {
                path.join("config.json").exists()
                    && path.join("kokoro-v1_0.pth").exists()
                    && path.join("voices").join("af_heart.pt").exists()
            }
            ModelFamily::Qwen3Asr => {
                let has_config = path.join("config.json").exists();
                let has_vocab = path.join("vocab.json").exists();
                let has_chat_template = path.join("chat_template.json").exists();
                let has_model =
                    if variant.is_quantized() || matches!(variant, ModelVariant::Qwen3Asr06B) {
                        path.join("model.safetensors").exists()
                    } else {
                        path.join("model-00001-of-00002.safetensors").exists()
                    };
                has_config && has_vocab && has_chat_template && has_model
            }
            ModelFamily::ParakeetAsr => {
                if variant.is_parakeet_nemo() {
                    let nemo_file = match variant {
                        ModelVariant::ParakeetTdt06BV2 => "parakeet-tdt-0.6b-v2.nemo",
                        ModelVariant::ParakeetTdt06BV3 => "parakeet-tdt-0.6b-v3.nemo",
                        _ => unreachable!("checked by is_parakeet_nemo"),
                    };
                    return path.join(nemo_file).exists();
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
            ModelFamily::SortformerDiarization => path
                .join("diar_streaming_sortformer_4spk-v2.1.nemo")
                .exists(),
            ModelFamily::Qwen3Chat
            | ModelFamily::Qwen35Chat
            | ModelFamily::Lfm2Chat
            | ModelFamily::Gemma3Chat => {
                if variant.is_qwen_chat_gguf() {
                    let gguf_file =
                        qwen_chat_gguf_filename(variant).expect("checked by is_qwen_chat_gguf");
                    path.join(gguf_file).exists()
                        && path.join("tokenizer.json").exists()
                        && path.join("tokenizer_config.json").exists()
                } else if variant.is_qwen35_chat_gguf() {
                    let gguf_file = qwen35_chat_gguf_filename(variant)
                        .expect("checked by is_qwen35_chat_gguf");
                    path.join(gguf_file).exists()
                        && path.join("mmproj-F16.gguf").exists()
                        && path.join("README.md").exists()
                        && path.join("tokenizer.json").exists()
                        && path.join("tokenizer_config.json").exists()
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
            ModelFamily::Qwen3ForcedAligner => {
                path.join("config.json").exists()
                    && path.join("vocab.json").exists()
                    && path.join("model.safetensors").exists()
            }
            ModelFamily::Voxtral => {
                path.join("params.json").exists()
                    && path.join("tekken.json").exists()
                    && path.join("consolidated.safetensors").exists()
            }
            ModelFamily::Tokenizer => {
                path.join("tokenizer.json").exists() || path.join("vocab.json").exists()
            }
            ModelFamily::Qwen3Tts => path.join("config.json").exists() && has_any_safetensors(),
        }
    }

    /// Download a model with multi-progress bar display
    pub async fn download(&self, variant: ModelVariant) -> Result<PathBuf> {
        let local_dir = self.model_path(variant);

        tokio::fs::create_dir_all(&local_dir).await?;
        info!("Downloading {} to {:?}", variant.repo_id(), local_dir);

        // Create overall progress bar
        let file_plans = self.get_file_download_plans(variant).await?;
        let total_bytes: u64 = file_plans.iter().map(|plan| plan.expected_size).sum();

        let overall_pb = self.multi_progress.add(ProgressBar::new(total_bytes));
        overall_pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("##-"),
        );
        overall_pb.set_message(format!("{}", variant.display_name()));

        let mut downloaded_bytes: u64 = 0;

        for plan in &file_plans {
            let file = &plan.local_file;
            let dest = local_dir.join(file);

            // Skip if already downloaded
            if dest.exists() {
                debug!("File already exists: {:?}", dest);
                let file_size = tokio::fs::metadata(&dest).await?.len();
                downloaded_bytes += file_size;
                overall_pb.set_position(downloaded_bytes);
                continue;
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
                .download_file_streaming(
                    &plan.source_repo,
                    &plan.source_file,
                    &dest,
                    Some(file_pb.clone()),
                    None,
                    None,
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
                    // Some files might be optional, continue
                }
            }

            self.multi_progress.remove(&file_pb);
        }

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

            // Update state based on result
            let final_state = match &result {
                Ok(_) => DownloadState::Downloaded,
                Err(_) => DownloadState::Error,
            };
            downloader
                .state_manager
                .set_state(variant, final_state)
                .await;

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
            expected_size_cache: Arc::clone(&self.expected_size_cache),
            multi_progress: MultiProgress::new(), // Each spawned task gets its own multi-progress
            state_manager: self.state_manager.clone(),
        }
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
        let local_dir = self.model_path(variant);

        tokio::fs::create_dir_all(&local_dir).await?;

        info!("Downloading {} to {:?}", variant.repo_id(), local_dir);

        let file_plans = self.get_file_download_plans(variant).await?;
        let total_files = file_plans.len();
        let total_bytes: u64 = file_plans.iter().map(|plan| plan.expected_size).sum();
        let mut downloaded_bytes: u64 = 0;
        const MAX_FILE_DOWNLOAD_ATTEMPTS: usize = 4;

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

            let mut last_err: Option<Error> = None;
            let mut bytes_downloaded_for_file = 0u64;
            let mut attempt = 1usize;

            while attempt <= MAX_FILE_DOWNLOAD_ATTEMPTS {
                if attempt > 1 {
                    info!(
                        "Retrying {} (attempt {}/{})",
                        file, attempt, MAX_FILE_DOWNLOAD_ATTEMPTS
                    );
                    file_pb.set_message(format!(
                        "{} (retry {}/{})",
                        file, attempt, MAX_FILE_DOWNLOAD_ATTEMPTS
                    ));
                }

                match self
                    .download_file_streaming(
                        &plan.source_repo,
                        &plan.source_file,
                        &dest,
                        Some(file_pb.clone()),
                        Some(progress_tx.clone()),
                        Some(progress_template.clone()),
                    )
                    .await
                {
                    Ok(bytes_downloaded) => {
                        if plan.strict_size_check && file_size > 0 && bytes_downloaded != file_size
                        {
                            let _ = tokio::fs::remove_file(&dest).await;
                            last_err = Some(Error::DownloadError(format!(
                                "Downloaded size mismatch for {}: expected {} bytes, got {} bytes",
                                file, file_size, bytes_downloaded
                            )));
                        } else {
                            bytes_downloaded_for_file = bytes_downloaded;
                            last_err = None;
                            break;
                        }
                    }
                    Err(e) => {
                        warn!(
                            "Failed to download {} (attempt {}/{}): {}",
                            file, attempt, MAX_FILE_DOWNLOAD_ATTEMPTS, e
                        );
                        let _ = tokio::fs::remove_file(&dest).await;
                        last_err = Some(e);
                    }
                }

                if attempt < MAX_FILE_DOWNLOAD_ATTEMPTS {
                    let backoff_ms = 400u64 * (1u64 << (attempt - 1));
                    tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                }
                attempt += 1;
            }

            if let Some(e) = last_err {
                file_pb.finish_with_message(format!("{} ✗", file));
                return Err(Error::DownloadError(format!(
                    "Failed to download required file {} after {} attempts: {}",
                    file, MAX_FILE_DOWNLOAD_ATTEMPTS, e
                )));
            }

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
            ModelFamily::Lfm2Audio => vec![
                "config.json".to_string(),
                "model.safetensors".to_string(),
                "model.safetensors.index.json".to_string(),
                "tokenizer.json".to_string(),
                "tokenizer_config.json".to_string(),
                "special_tokens_map.json".to_string(),
                // Required LFM2.5 audio detokenizer assets.
                "audio_detokenizer/config.json".to_string(),
                "audio_detokenizer/model.safetensors".to_string(),
                "audio_detokenizer/tokenizer.json".to_string(),
                "audio_detokenizer/tokenizer_config.json".to_string(),
            ],
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
            ModelFamily::Qwen3Asr => {
                let mut files = vec![
                    "config.json".to_string(),
                    "chat_template.json".to_string(),
                    "generation_config.json".to_string(),
                    "merges.txt".to_string(),
                    "preprocessor_config.json".to_string(),
                    "tokenizer_config.json".to_string(),
                    "vocab.json".to_string(),
                ];
                // mlx-community quantized models use model.safetensors + model.safetensors.index.json.
                if variant.is_quantized() {
                    files.push("model.safetensors".to_string());
                    files.push("model.safetensors.index.json".to_string());
                } else if matches!(variant, ModelVariant::Qwen3Asr06B) {
                    files.push("model.safetensors".to_string());
                } else {
                    files.extend([
                        "model-00001-of-00002.safetensors".to_string(),
                        "model-00002-of-00002.safetensors".to_string(),
                        "model.safetensors.index.json".to_string(),
                    ]);
                }
                files
            }
            ModelFamily::ParakeetAsr => {
                if variant.is_parakeet_nemo() {
                    let nemo_file = match variant {
                        ModelVariant::ParakeetTdt06BV2 => "parakeet-tdt-0.6b-v2.nemo",
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
            ModelFamily::SortformerDiarization => vec![
                "diar_streaming_sortformer_4spk-v2.1.nemo".to_string(),
                "README.md".to_string(),
                "bias.md".to_string(),
                "privacy.md".to_string(),
                "safety.md".to_string(),
            ],
            ModelFamily::Qwen3Chat
            | ModelFamily::Qwen35Chat
            | ModelFamily::Lfm2Chat
            | ModelFamily::Gemma3Chat => {
                if variant.is_qwen_chat_gguf() {
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
                    let gguf_file = qwen35_chat_gguf_filename(variant)
                        .expect("checked by is_qwen35_chat_gguf");
                    return vec![
                        gguf_file.to_string(),
                        "mmproj-F16.gguf".to_string(),
                        "README.md".to_string(),
                        "tokenizer.json".to_string(),
                        "tokenizer_config.json".to_string(),
                    ];
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
                "tekken.json".to_string(),
                "consolidated.safetensors".to_string(),
            ],
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

        if variant.is_qwen_chat_gguf() {
            let tokenizer_repo = match variant {
                ModelVariant::Qwen306BGguf => "Qwen/Qwen3-0.6B",
                ModelVariant::Qwen317BGguf => "Qwen/Qwen3-1.7B",
                ModelVariant::Qwen34BGguf => "Qwen/Qwen3-4B",
                ModelVariant::Qwen38BGguf => "Qwen/Qwen3-8B",
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
                        source_repo,
                        source_file: file.clone(),
                        local_file: file,
                    }
                })
                .collect();
        }

        if variant.is_lfm2_chat_gguf() {
            let tokenizer_repo = match variant {
                ModelVariant::Lfm2512BInstructGguf => "LiquidAI/LFM2.5-1.2B-Instruct",
                ModelVariant::Lfm2512BThinkingGguf => "LiquidAI/LFM2.5-1.2B-Thinking",
                _ => variant.repo_id(),
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
                        source_repo,
                        source_file: file.clone(),
                        local_file: file,
                    }
                })
                .collect();
        }

        if variant.is_qwen35_chat_gguf() {
            return self
                .get_model_files(variant)
                .into_iter()
                .map(|file| ModelFileSpec {
                    source_repo: default_repo.clone(),
                    source_file: file.clone(),
                    local_file: file,
                })
                .collect();
        }

        self.get_model_files(variant)
            .into_iter()
            .map(|file| ModelFileSpec {
                source_repo: default_repo.clone(),
                source_file: file.clone(),
                local_file: file,
            })
            .collect()
    }

    /// Get actual file size from HTTP HEAD request
    async fn get_actual_file_size(&self, repo_id: &str, filename: &str) -> Result<u64> {
        let url = format!("{}/{}/resolve/main/{}", HF_BASE_URL, repo_id, filename);

        let response = self
            .http_client
            .head(&url)
            .header("User-Agent", "izwi/0.1.0")
            .send()
            .await
            .map_err(|e| Error::HfHubError(format!("HEAD request failed: {}", e)))?;

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
        let file_specs = self.get_model_file_specs(variant);
        let local_dir = self.model_path(variant);

        let mut repo_tree_indexes: HashMap<String, HashMap<String, u64>> = HashMap::new();
        let mut repo_tree_planning_available = true;
        for spec in &file_specs {
            if repo_tree_indexes.contains_key(&spec.source_repo) {
                continue;
            }
            match self.get_repo_tree_index(&spec.source_repo).await {
                Ok(index) => {
                    repo_tree_indexes.insert(spec.source_repo.clone(), index);
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
                if let Some(size) = repo_tree_indexes
                    .get(&spec.source_repo)
                    .and_then(|index| index.get(&spec.source_file))
                {
                    plans.push(FileDownloadPlan {
                        source_repo: spec.source_repo.clone(),
                        source_file: spec.source_file.clone(),
                        local_file: spec.local_file.clone(),
                        expected_size: *size,
                        strict_size_check: *size > 0,
                    });
                } else {
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
            let dest = local_dir.join(&spec.local_file);
            match self
                .get_actual_file_size(&spec.source_repo, &spec.source_file)
                .await
            {
                Ok(size) => plans.push(FileDownloadPlan {
                    source_repo: spec.source_repo.clone(),
                    source_file: spec.source_file.clone(),
                    local_file: spec.local_file.clone(),
                    expected_size: size,
                    strict_size_check: true,
                }),
                Err(e) => {
                    if Self::is_not_found_hf_error(&e) {
                        debug!(
                            "Skipping unavailable file for {}: {}/{} (HEAD returned 404)",
                            variant, spec.source_repo, spec.source_file
                        );
                        continue;
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
                        source_file: spec.source_file.clone(),
                        local_file: spec.local_file.clone(),
                        expected_size: fallback_size,
                        strict_size_check: false,
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
        if file == "mmproj-F16.gguf" {
            match variant {
                ModelVariant::Qwen3508BGguf => 52_000_000,
                ModelVariant::Qwen352BGguf => 98_000_000,
                ModelVariant::Qwen354BGguf => 140_000_000,
                ModelVariant::Qwen359BGguf => 230_000_000,
                _ => 120_000_000,
            }
        } else if file.ends_with(".gguf") {
            if file.contains("Qwen3-0.6B") {
                1_100_000_000
            } else if file.contains("Qwen3-1.7B") {
                2_400_000_000
            } else if file.contains("Qwen3-4B") {
                2_500_000_000
            } else if file.contains("Qwen3-8B") {
                5_200_000_000
            } else if file.contains("Qwen3-14B") {
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
            } else {
                1_000_000_000
            }
        } else if file.contains("model.safetensors") && !file.contains("index") {
            if file.contains("00001") || file.contains("00002") {
                2_000_000_000
            } else {
                match variant {
                    ModelVariant::Qwen3Tts12Hz06BBase
                    | ModelVariant::Qwen3Tts12Hz06BCustomVoice => 1_800_000_000,
                    ModelVariant::Qwen3Tts12Hz17BBase4Bit
                    | ModelVariant::Qwen3Tts12Hz17BCustomVoice4Bit => 2_330_000_000,
                    ModelVariant::Qwen3Tts12Hz17BBase
                    | ModelVariant::Qwen3Tts12Hz17BCustomVoice
                    | ModelVariant::Qwen3Tts12Hz17BVoiceDesign => 3_850_000_000,
                    ModelVariant::Qwen3Asr06B => 1_800_000_000,
                    ModelVariant::WhisperLargeV3Turbo => 1_617_824_864,
                    ModelVariant::Qwen306B => 1_520_000_000,
                    ModelVariant::Qwen306B4Bit => 800_000_000,
                    ModelVariant::Qwen317B4Bit => 1_115_000_000,
                    ModelVariant::Qwen3ForcedAligner06B4Bit => 703_000_000,
                    ModelVariant::Lfm25Audio15B4Bit => 884_000_000,
                    ModelVariant::Qwen306BGguf => 1_100_000_000,
                    ModelVariant::Qwen317BGguf => 2_400_000_000,
                    ModelVariant::Qwen34BGguf => 2_500_000_000,
                    ModelVariant::Qwen38BGguf => 5_200_000_000,
                    ModelVariant::Qwen314BGguf => 9_200_000_000,
                    ModelVariant::Qwen3508BGguf => 685_000_000,
                    ModelVariant::Qwen352BGguf => 1_850_000_000,
                    ModelVariant::Qwen354BGguf => 3_250_000_000,
                    ModelVariant::Qwen359BGguf => 6_350_000_000,
                    ModelVariant::Gemma31BIt => 2_100_000_000,
                    ModelVariant::Gemma34BIt => 2_400_000_000,
                    ModelVariant::Lfm25Audio15B => 2_900_000_000,
                    ModelVariant::VoxtralMini4BRealtime2602 => 8_900_000_000,
                    _ => 1_500_000_000,
                }
            }
        } else if file.ends_with(".pth") && file.contains("kokoro-v1_0") {
            327_212_226
        } else if file.starts_with("voices/") && file.ends_with(".pt") {
            550_000
        } else if file.ends_with(".nemo") {
            match variant {
                ModelVariant::ParakeetTdt06BV2 => 4_926_457_088,
                ModelVariant::ParakeetTdt06BV3 => 10_036_761_167,
                ModelVariant::DiarStreamingSortformer4SpkV21 => 510_000_000,
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

    /// Get estimated file sizes for a model variant (in bytes)
    /// These are approximate sizes based on model architecture
    fn get_file_sizes(&self, variant: ModelVariant) -> Vec<u64> {
        let files = self.get_model_file_specs(variant);
        files
            .iter()
            .map(|file| self.get_single_file_size_estimate(variant, &file.local_file))
            .collect()
    }

    /// Get download size for a model (if available from cache)
    pub fn get_cached_size(&self, variant: ModelVariant) -> Option<u64> {
        let path = self.model_path(variant);
        if path.exists() {
            Self::dir_size(&path).ok()
        } else {
            None
        }
    }

    /// Resolve the expected total download size for a model variant.
    ///
    /// This uses the same file planning logic as the downloader itself, then caches
    /// results so list APIs do not repeatedly hit Hugging Face.
    pub async fn expected_size_bytes(&self, variant: ModelVariant) -> u64 {
        if let Some(size) = self.expected_size_cache.read().await.get(&variant).copied() {
            return size;
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

        self.expected_size_cache
            .write()
            .await
            .insert(variant, resolved);

        resolved
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
    fn dir_size(path: &Path) -> Result<u64> {
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
