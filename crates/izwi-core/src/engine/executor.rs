//! Model executor - handles forward pass execution.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::sync::RwLock;
use tracing::{debug, error, info};

#[path = "executor/audio.rs"]
mod audio;
#[path = "executor/dispatch.rs"]
mod dispatch;
#[path = "executor/handler_asr.rs"]
mod handler_asr;
#[path = "executor/handler_chat.rs"]
mod handler_chat;
#[path = "executor/handler_s2s.rs"]
mod handler_s2s;
#[path = "executor/handler_tts.rs"]
mod handler_tts;
#[path = "executor/state.rs"]
mod state;
#[path = "executor/streaming.rs"]
mod streaming;

use super::config::EngineCoreConfig;
use super::request::EngineCoreRequest;
use super::scheduler::ScheduledRequest;
use super::types::AudioOutput;
use crate::backends::{
    BackendContext, BackendKind, BackendPreference, BackendRouter, BackendSelectionSource,
};
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::architectures::qwen3::tts::Qwen3TtsModel;
use crate::models::ModelRegistry;
use state::{
    ActiveAsrDecode, ActiveChatDecode, ActiveLfm2TtsDecode, ActiveQwenTtsDecode,
    ActiveSpeechToSpeechDecode,
};

fn panic_payload_to_string(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(msg) = payload.downcast_ref::<&str>() {
        return (*msg).to_string();
    }
    if let Some(msg) = payload.downcast_ref::<String>() {
        return msg.clone();
    }
    "unknown panic payload".to_string()
}

/// Configuration for the model executor.
#[derive(Clone)]
pub struct WorkerConfig {
    /// Path to models directory
    pub models_dir: PathBuf,
    /// Backend to use (cpu, metal, cuda)
    pub backend: BackendKind,
    /// Resolved backend/device context for this worker.
    pub backend_context: BackendContext,
    /// Data type (float32, float16, bfloat16)
    pub dtype: String,
    /// KV cache storage dtype hint (e.g. float16, int8).
    pub kv_cache_dtype: String,
    /// Number of threads
    pub num_threads: usize,
    /// Maximum number of requests to execute in parallel.
    pub request_parallelism: usize,
    /// Decode-time KV cache page size.
    pub kv_page_size: usize,
    /// Optional shared model registry for loaded runtime models.
    pub model_registry: Option<Arc<ModelRegistry>>,
}

impl std::fmt::Debug for WorkerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerConfig")
            .field("models_dir", &self.models_dir)
            .field("backend", &self.backend)
            .field("backend_context", &self.backend_context)
            .field("dtype", &self.dtype)
            .field("kv_cache_dtype", &self.kv_cache_dtype)
            .field("num_threads", &self.num_threads)
            .field("request_parallelism", &self.request_parallelism)
            .field("kv_page_size", &self.kv_page_size)
            .field(
                "model_registry",
                &self.model_registry.as_ref().map(|_| "<shared>"),
            )
            .finish()
    }
}

impl Default for WorkerConfig {
    fn default() -> Self {
        let backend_context = BackendRouter::resolve_context(
            BackendPreference::Auto,
            BackendSelectionSource::Default,
        );
        let backend_kind = backend_context.backend_kind;
        let num_threads = 4;
        Self {
            models_dir: dirs::data_local_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("izwi")
                .join("models"),
            backend: backend_kind,
            backend_context,
            dtype: "float32".to_string(),
            kv_cache_dtype: "float16".to_string(),
            num_threads,
            request_parallelism: Self::request_parallelism_for(backend_kind, num_threads),
            kv_page_size: 64,
            model_registry: None,
        }
    }
}

impl From<&EngineCoreConfig> for WorkerConfig {
    fn from(config: &EngineCoreConfig) -> Self {
        let backend_context =
            BackendRouter::resolve_context_for_kind(config.backend, BackendSelectionSource::Config);
        let backend_kind = backend_context.backend_kind;
        let num_threads = config.num_threads.max(1);
        Self {
            models_dir: config.models_dir.clone(),
            backend: backend_kind,
            backend_context,
            dtype: "float32".to_string(),
            kv_cache_dtype: config.kv_cache_dtype.clone(),
            num_threads,
            request_parallelism: Self::request_parallelism_for(backend_kind, num_threads),
            kv_page_size: config.block_size.max(1),
            model_registry: None,
        }
    }
}

impl WorkerConfig {
    fn request_parallelism_override() -> Option<usize> {
        std::env::var("IZWI_REQUEST_PARALLELISM")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .filter(|value| *value > 0)
    }

    fn resolve_request_parallelism(
        backend: BackendKind,
        num_threads: usize,
        override_value: Option<usize>,
    ) -> usize {
        let default_parallelism = match backend {
            // CPU workloads already use `num_threads` for BLAS/Rayon/intra-op work, so
            // keep inter-request fan-out conservative unless explicitly overridden.
            BackendKind::Cpu | BackendKind::Metal => 1,
            BackendKind::Cuda => num_threads.max(1),
        };

        override_value.unwrap_or(default_parallelism).max(1)
    }

    fn request_parallelism_for(backend: BackendKind, num_threads: usize) -> usize {
        Self::resolve_request_parallelism(
            backend,
            num_threads,
            Self::request_parallelism_override(),
        )
    }
}

/// Output from the executor after a forward pass.
#[derive(Debug, Clone)]
pub struct ExecutorOutput {
    /// Request ID
    pub request_id: String,
    /// Generated audio samples
    pub audio: Option<AudioOutput>,
    /// Generated text (for ASR/chat)
    pub text: Option<String>,
    /// Number of tokens processed
    pub tokens_processed: usize,
    /// Number of tokens generated
    pub tokens_generated: usize,
    /// Whether generation is complete
    pub finished: bool,
    /// Error if any
    pub error: Option<String>,
}

impl ExecutorOutput {
    pub fn error(request_id: String, error: impl Into<String>) -> Self {
        Self {
            request_id,
            audio: None,
            text: None,
            tokens_processed: 0,
            tokens_generated: 0,
            finished: true,
            error: Some(error.into()),
        }
    }
}

/// Model executor trait - abstracts the model inference backend.
pub trait ModelExecutor: Send + Sync {
    /// Execute prefill pass for newly admitted or in-progress prefill requests.
    fn execute_prefill(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorOutput>>;

    /// Execute decode pass for running requests.
    fn execute_decode(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorOutput>>;

    /// Execute forward pass for scheduled requests.
    /// Compatibility helper that executes decode and prefill paths.
    fn execute(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorOutput>> {
        let mut decode = Vec::new();
        let mut prefill = Vec::new();
        for req in scheduled {
            if req.is_prefill {
                prefill.push(req.clone());
            } else {
                decode.push(req.clone());
            }
        }

        let mut outputs = Vec::new();
        if !decode.is_empty() {
            outputs.extend(self.execute_decode(requests, &decode)?);
        }
        if !prefill.is_empty() {
            outputs.extend(self.execute_prefill(requests, &prefill)?);
        }
        Ok(outputs)
    }

    /// Check if the executor is ready.
    fn is_ready(&self) -> bool;

    /// Initialize the executor (load models, etc.)
    fn initialize(&mut self) -> Result<()>;

    /// Shutdown the executor.
    fn shutdown(&mut self) -> Result<()>;

    /// Cleanup transient per-request state held by the executor backend.
    fn cleanup_request(&self, _request_id: &str) {}
}

pub struct NativeExecutor {
    config: WorkerConfig,
    initialized: bool,
    loaded_tts_model: Option<Arc<Qwen3TtsModel>>,
    chat_decode_states: Mutex<HashMap<String, ActiveChatDecode>>,
    asr_decode_states: Mutex<HashMap<String, ActiveAsrDecode>>,
    qwen_tts_decode_states: Mutex<HashMap<String, ActiveQwenTtsDecode>>,
    lfm2_tts_decode_states: Mutex<HashMap<String, ActiveLfm2TtsDecode>>,
    speech_to_speech_decode_states: Mutex<HashMap<String, ActiveSpeechToSpeechDecode>>,
}

impl NativeExecutor {
    /// Create a new native executor.
    pub fn new(config: WorkerConfig) -> Self {
        Self {
            config,
            initialized: false,
            loaded_tts_model: None,
            chat_decode_states: Mutex::new(HashMap::new()),
            asr_decode_states: Mutex::new(HashMap::new()),
            qwen_tts_decode_states: Mutex::new(HashMap::new()),
            lfm2_tts_decode_states: Mutex::new(HashMap::new()),
            speech_to_speech_decode_states: Mutex::new(HashMap::new()),
        }
    }

    fn with_qwen_model<T>(
        &self,
        variant: Option<ModelVariant>,
        f: impl FnOnce(&Qwen3TtsModel) -> Result<T>,
    ) -> Result<T> {
        if let Some(registry) = &self.config.model_registry {
            let variant = variant.ok_or_else(|| {
                Error::InferenceError("Qwen TTS request is missing model variant".to_string())
            })?;
            let model = registry.try_get_qwen_tts(variant).ok_or_else(|| {
                Error::InferenceError(format!("Qwen TTS model {variant} is not loaded"))
            })?;
            return f(model.as_ref());
        }

        let model = self
            .loaded_tts_model
            .as_deref()
            .ok_or_else(|| Error::InferenceError("Executor model not initialized".to_string()))?;
        f(model)
    }

    fn with_registry<T>(&self, f: impl FnOnce(&ModelRegistry) -> Result<T>) -> Result<T> {
        let registry =
            self.config.model_registry.as_ref().ok_or_else(|| {
                Error::InferenceError("Model registry is not configured".to_string())
            })?;
        f(registry)
    }

    fn run_blocking<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
        let run_catching_panic = || {
            let unwind_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
            match unwind_result {
                Ok(result) => result,
                Err(payload) => {
                    let message = panic_payload_to_string(payload.as_ref());
                    error!("Model execution panicked: {message}");
                    Err(Error::InferenceError(format!(
                        "Model execution panicked: {message}"
                    )))
                }
            }
        };

        match tokio::runtime::Handle::try_current() {
            Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
                // Long-running CPU inference should not monopolize Tokio workers; this allows
                // async tasks (including SSE stream forwarding) to continue making progress.
                tokio::task::block_in_place(run_catching_panic)
            }
            _ => run_catching_panic(),
        }
    }
}

impl ModelExecutor for NativeExecutor {
    fn execute_prefill(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorOutput>> {
        if !self.initialized {
            return Err(Error::InferenceError("Executor not initialized".into()));
        }
        self.execute_requests(requests, scheduled)
    }

    fn execute_decode(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorOutput>> {
        if !self.initialized {
            return Err(Error::InferenceError("Executor not initialized".into()));
        }
        self.execute_requests(requests, scheduled)
    }

    fn execute(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorOutput>> {
        if !self.initialized {
            return Err(Error::InferenceError("Executor not initialized".into()));
        }
        self.execute_requests(requests, scheduled)
    }

    fn is_ready(&self) -> bool {
        self.initialized
    }

    fn initialize(&mut self) -> Result<()> {
        info!("Initializing native executor");
        if self.config.model_registry.is_none() {
            let device = self.config.backend_context.device.clone();
            let model = Qwen3TtsModel::load(
                &self.config.models_dir,
                device,
                self.config.kv_page_size.max(1),
                &self.config.kv_cache_dtype,
            )?;
            self.loaded_tts_model = Some(Arc::new(model));
            debug!(
                "Native executor loaded TTS model from {:?}",
                self.config.models_dir
            );
        } else {
            debug!("Native executor will use shared model registry");
        }
        self.initialized = true;
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down native executor");
        self.initialized = false;
        self.loaded_tts_model = None;
        if let Ok(mut guard) = self.chat_decode_states.lock() {
            guard.clear();
        }
        if let Ok(mut guard) = self.asr_decode_states.lock() {
            guard.clear();
        }
        if let Ok(mut guard) = self.qwen_tts_decode_states.lock() {
            guard.clear();
        }
        if let Ok(mut guard) = self.lfm2_tts_decode_states.lock() {
            guard.clear();
        }
        if let Ok(mut guard) = self.speech_to_speech_decode_states.lock() {
            guard.clear();
        }
        Ok(())
    }

    fn cleanup_request(&self, request_id: &str) {
        if let Ok(mut guard) = self.chat_decode_states.lock() {
            guard.remove(request_id);
        }
        if let Ok(mut guard) = self.asr_decode_states.lock() {
            guard.remove(request_id);
        }
        if let Ok(mut guard) = self.qwen_tts_decode_states.lock() {
            guard.remove(request_id);
        }
        if let Ok(mut guard) = self.lfm2_tts_decode_states.lock() {
            guard.remove(request_id);
        }
        if let Ok(mut guard) = self.speech_to_speech_decode_states.lock() {
            guard.remove(request_id);
        }
    }
}

/// Unified executor that wraps a model executor implementation.
pub struct UnifiedExecutor {
    inner: Arc<RwLock<Box<dyn ModelExecutor>>>,
}

impl UnifiedExecutor {
    /// Create a new unified executor with native backend.
    pub fn new_native(config: WorkerConfig) -> Self {
        Self {
            inner: Arc::new(RwLock::new(Box::new(NativeExecutor::new(config)))),
        }
    }

    #[cfg(test)]
    pub(crate) fn new_for_test(executor: Box<dyn ModelExecutor>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(executor)),
        }
    }

    /// Execute requests.
    pub async fn execute(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorOutput>> {
        let executor = self.inner.read().await;
        executor.execute(requests, scheduled)
    }

    /// Execute prefill requests.
    pub async fn execute_prefill(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorOutput>> {
        let executor = self.inner.read().await;
        executor.execute_prefill(requests, scheduled)
    }

    /// Execute decode requests.
    pub async fn execute_decode(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorOutput>> {
        let executor = self.inner.read().await;
        executor.execute_decode(requests, scheduled)
    }

    /// Check if ready.
    pub async fn is_ready(&self) -> bool {
        let executor = self.inner.read().await;
        executor.is_ready()
    }

    /// Initialize.
    pub async fn initialize(&self) -> Result<()> {
        let mut executor = self.inner.write().await;
        executor.initialize()
    }

    /// Shutdown.
    pub async fn shutdown(&self) -> Result<()> {
        let mut executor = self.inner.write().await;
        executor.shutdown()
    }

    /// Cleanup transient backend state for a completed/aborted request.
    pub async fn cleanup_request(&self, request_id: &str) {
        let executor = self.inner.read().await;
        executor.cleanup_request(request_id);
    }
}

/// Decode base64-encoded audio to samples.
pub fn decode_audio_base64(audio_b64: &str, _sample_rate: u32) -> Result<Vec<f32>> {
    let (samples, _) = decode_audio_base64_with_rate(audio_b64)?;
    Ok(samples)
}

fn decode_audio_base64_with_rate(audio_b64: &str) -> Result<(Vec<f32>, u32)> {
    audio::decode_audio_base64_with_rate(audio_b64)
}

#[cfg(test)]
mod tests {
    use super::super::output::StreamingOutput;
    use super::*;
    use crate::model::ModelVariant;
    use base64::Engine;
    use tokio::sync::mpsc;

    #[test]
    fn test_worker_config_default() {
        let config = WorkerConfig::default();
        assert_eq!(config.backend, config.backend_context.backend_kind);
    }

    #[test]
    fn test_worker_config_from_engine_config_uses_backend_context() {
        let mut engine = EngineCoreConfig::default();
        engine.backend = BackendKind::Cpu;

        let config = WorkerConfig::from(&engine);
        assert_eq!(config.backend, config.backend_context.backend_kind);
        assert_eq!(config.request_parallelism, 1);
        assert_eq!(
            config.backend_context.source,
            BackendSelectionSource::Config
        );
    }

    #[test]
    fn test_request_parallelism_defaults_are_backend_aware() {
        assert_eq!(
            WorkerConfig::resolve_request_parallelism(BackendKind::Cpu, 8, None),
            1
        );
        assert_eq!(
            WorkerConfig::resolve_request_parallelism(BackendKind::Metal, 8, None),
            1
        );
        assert_eq!(
            WorkerConfig::resolve_request_parallelism(BackendKind::Cuda, 8, None),
            8
        );
        assert_eq!(
            WorkerConfig::resolve_request_parallelism(BackendKind::Cpu, 8, Some(3)),
            3
        );
    }

    #[test]
    fn test_run_blocking_converts_panic_to_error() {
        let result = NativeExecutor::run_blocking(|| -> Result<()> {
            panic!("executor panic sentinel");
        });

        let Err(Error::InferenceError(message)) = result else {
            panic!("expected inference error from panic");
        };
        assert!(message.contains("executor panic sentinel"));
    }

    #[test]
    fn test_run_blocking_is_safe_inside_current_thread_runtime() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to build runtime");

        let result =
            runtime.block_on(async { NativeExecutor::run_blocking(|| Ok::<_, Error>(())) });
        assert!(result.is_ok());
    }

    #[test]
    fn test_stream_audio_send_is_safe_inside_current_thread_runtime() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to build runtime");

        let result = runtime.block_on(async {
            let (tx, mut rx) = mpsc::channel(4);
            let mut sequence = 0usize;
            NativeExecutor::stream_audio(
                &tx,
                "req-1",
                &mut sequence,
                vec![0.1, -0.1],
                24_000,
                false,
            )?;
            let chunk = rx
                .recv()
                .await
                .ok_or_else(|| Error::InferenceError("missing streamed chunk".to_string()))?;
            if chunk.request_id != "req-1" || chunk.sequence != 0 || chunk.samples.len() != 2 {
                return Err(Error::InferenceError(
                    "unexpected streamed chunk payload".to_string(),
                ));
            }
            Ok::<(), Error>(())
        });
        assert!(result.is_ok());
    }

    #[test]
    fn test_stream_audio_send_returns_error_when_channel_closed() {
        let (tx, rx) = mpsc::channel::<StreamingOutput>(1);
        drop(rx);

        let mut sequence = 0usize;
        let result = NativeExecutor::stream_audio(
            &tx,
            "req-closed",
            &mut sequence,
            vec![0.2],
            24_000,
            false,
        );
        let Err(Error::InferenceError(message)) = result else {
            panic!("expected inference error when streaming channel is closed");
        };
        assert!(message.contains("Streaming output channel closed"));
    }

    #[test]
    fn test_stream_audio_send_returns_backpressure_error_when_queue_full() {
        let (tx, _rx) = mpsc::channel::<StreamingOutput>(1);

        let mut first_sequence = 0usize;
        NativeExecutor::stream_audio(
            &tx,
            "req-full",
            &mut first_sequence,
            vec![0.1],
            24_000,
            false,
        )
        .expect("first chunk should fit");

        let mut second_sequence = 1usize;
        let result = NativeExecutor::stream_audio(
            &tx,
            "req-full",
            &mut second_sequence,
            vec![0.2],
            24_000,
            false,
        );

        let Err(Error::InferenceError(message)) = result else {
            panic!("expected inference error when streaming queue is full");
        };
        assert!(message.contains("backpressure"));
    }

    #[test]
    fn test_to_tts_params_uses_model_native_auto_limit() {
        let mut request = EngineCoreRequest::tts("Long-form synthesis");
        request.model_variant = Some(ModelVariant::Qwen3Tts12Hz17BVoiceDesign);
        request.params.max_tokens = 0;

        let params = NativeExecutor::to_tts_params(&request);
        assert_eq!(params.max_frames, ModelVariant::QWEN3_TTS_MAX_OUTPUT_FRAMES);
    }

    #[test]
    fn test_to_tts_params_clamps_to_model_native_limit() {
        let mut request = EngineCoreRequest::tts("Long-form synthesis");
        request.model_variant = Some(ModelVariant::Qwen3Tts12Hz06BCustomVoice);
        request.params.max_tokens = 50_000;

        let params = NativeExecutor::to_tts_params(&request);
        assert_eq!(params.max_frames, ModelVariant::QWEN3_TTS_MAX_OUTPUT_FRAMES);
    }

    #[test]
    fn decode_audio_base64_with_rate_downmixes_stereo_wav() {
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut wav_bytes = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut wav_bytes);
            let mut writer = hound::WavWriter::new(cursor, spec).expect("writer");
            // 2 stereo frames: [L,R]=[0.25,0.75] then [0.5,-0.5]
            writer.write_sample((0.25f32 * 32767.0) as i16).unwrap();
            writer.write_sample((0.75f32 * 32767.0) as i16).unwrap();
            writer.write_sample((0.5f32 * 32767.0) as i16).unwrap();
            writer.write_sample((-0.5f32 * 32767.0) as i16).unwrap();
            writer.finalize().unwrap();
        }

        let b64 = base64::engine::general_purpose::STANDARD.encode(&wav_bytes);
        let (samples, sample_rate) =
            decode_audio_base64_with_rate(&b64).expect("decode should succeed");

        assert_eq!(sample_rate, 16_000);
        assert_eq!(samples.len(), 2);
        // After downmixing, expected mono values are averages: 0.5 and 0.0.
        assert!(
            (samples[0] - 0.5).abs() < 0.02,
            "first sample was {}",
            samples[0]
        );
        assert!(samples[1].abs() < 0.02, "second sample was {}", samples[1]);
    }

    #[test]
    fn decode_request_audio_with_rate_accepts_raw_audio_bytes() {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut wav_bytes = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut wav_bytes);
            let mut writer = hound::WavWriter::new(cursor, spec).expect("writer");
            writer.write_sample((0.25f32 * 32767.0) as i16).unwrap();
            writer.write_sample((-0.25f32 * 32767.0) as i16).unwrap();
            writer.finalize().unwrap();
        }

        let request = EngineCoreRequest::asr_bytes(wav_bytes);
        let (samples, sample_rate) =
            audio::decode_request_audio_with_rate(&request).expect("decode should succeed");

        assert_eq!(sample_rate, 16_000);
        assert_eq!(samples.len(), 2);
        assert!(
            (samples[0] - 0.25).abs() < 0.02,
            "first sample was {}",
            samples[0]
        );
        assert!(
            (samples[1] + 0.25).abs() < 0.02,
            "second sample was {}",
            samples[1]
        );
    }
}
