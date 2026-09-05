//! Izwi CLI - World-class command-line interface for audio inference
//!
//! Inspired by vLLM, SGlang, Ollama, and llama.cpp CLIs
#![allow(dead_code)]

use clap::{Args, Parser, Subcommand, ValueEnum};
use izwi_core::backends::BackendPreference;
use std::path::PathBuf;

use crate::style;

/// Izwi - High-performance audio inference engine CLI
///
/// A world-class CLI for text-to-speech and speech-to-text inference
/// optimized for Apple Silicon and CUDA devices.
///
/// Examples:
///   izwi serve                    # Start the server
///   izwi models list              # List available models
///   izwi pull qwen3-tts-0.6b      # Download a model
///   izwi tts "Hello world"        # Generate speech
///   izwi transcribe audio.wav     # Transcribe audio
#[derive(Parser)]
#[command(
    name = "izwi",
    about = "High-performance audio inference engine",
    long_about = "Izwi is a world-class audio inference engine for text-to-speech (TTS) and automatic speech recognition (ASR). Optimized for Apple Silicon and CUDA devices.",
    version = env!("CARGO_PKG_VERSION"),
    author = "Izwi <hi@izwiai.com>",
    help_template = style::HELP_TEMPLATE,
    arg_required_else_help = true,
    propagate_version = true,
    disable_colored_help = false,
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    /// Configuration file path
    #[arg(long, global = true, value_name = "PATH")]
    pub config: Option<PathBuf>,

    /// Server URL for API commands
    #[arg(
        long,
        global = true,
        value_name = "URL",
        default_value = "http://localhost:8080"
    )]
    pub server: String,

    /// Output format
    #[arg(
        long = "output-format",
        global = true,
        value_enum,
        default_value = "table"
    )]
    pub output_format: OutputFormat,

    /// Suppress all output except results
    #[arg(long, global = true)]
    pub quiet: bool,

    /// Enable verbose output
    #[arg(long, global = true)]
    pub verbose: bool,

    /// Disable colored output
    #[arg(long, global = true)]
    pub no_color: bool,
}

/// Optional CLI layer for CUDA and loading policies. Every field is optional
/// so a flag cannot erase a sibling from TOML or the environment.
#[derive(Args, Debug, Clone, Default)]
pub struct PerformanceArgs {
    /// CUDA performance master mode (auto/off); off dominates individual features
    #[arg(
        long = "cuda-performance",
        alias = "cuda-mode",
        value_name = "AUTO_OR_OFF"
    )]
    pub cuda_mode: Option<izwi_core::OptimizationMode>,
    /// CUDA projection backend (auto/q8/native_fp8)
    #[arg(long, value_name = "BACKEND")]
    pub cuda_projection_backend: Option<izwi_core::CudaProjectionBackend>,
    /// CUDA packed projections (auto/off)
    #[arg(long, value_name = "AUTO_OR_OFF")]
    pub cuda_packed_projections: Option<izwi_core::OptimizationMode>,
    /// CUDA fused decode (auto/off)
    #[arg(long, value_name = "AUTO_OR_OFF")]
    pub cuda_fused_decode: Option<izwi_core::OptimizationMode>,
    /// CUDA device sampling (auto/off)
    #[arg(long, value_name = "AUTO_OR_OFF")]
    pub cuda_device_sampling: Option<izwi_core::OptimizationMode>,
    /// CUDA decode graphs (auto/off)
    #[arg(long, value_name = "AUTO_OR_OFF")]
    pub cuda_decode_graphs: Option<izwi_core::OptimizationMode>,
    /// CUDA mtp (auto/off)
    #[arg(long, value_name = "AUTO_OR_OFF")]
    pub cuda_mtp: Option<izwi_core::OptimizationMode>,
    /// CUDA mtp quantum (auto/off)
    #[arg(long, value_name = "AUTO_OR_OFF")]
    pub cuda_mtp_quantum: Option<izwi_core::OptimizationMode>,
    /// CUDA mtp draft tokens (1..3; default 1)
    #[arg(long, value_parser = parse_mtp_depth)]
    pub cuda_mtp_draft_tokens: Option<usize>,
    /// CUDA mtp adaptive (true/false; default true)
    #[arg(long, action = clap::ArgAction::Set, value_name = "BOOL")]
    pub cuda_mtp_adaptive: Option<bool>,
    /// LOADING performance master mode (auto/off); off dominates individual features
    #[arg(
        long = "loading-performance",
        alias = "loading-mode",
        value_name = "AUTO_OR_OFF"
    )]
    pub loading_mode: Option<izwi_core::OptimizationMode>,
    /// LOADING derived weight cache (auto/off)
    #[arg(long, value_name = "AUTO_OR_OFF")]
    pub loading_derived_weight_cache: Option<izwi_core::OptimizationMode>,
    /// LOADING parallel conversion (auto/off)
    #[arg(long, value_name = "AUTO_OR_OFF")]
    pub loading_parallel_conversion: Option<izwi_core::OptimizationMode>,
    /// LOADING pinned uploads (auto/off)
    #[arg(long, value_name = "AUTO_OR_OFF")]
    pub loading_pinned_uploads: Option<izwi_core::OptimizationMode>,
    /// LOADING io strategy (auto/mmap/sequential)
    #[arg(long, value_name = "STRATEGY")]
    pub loading_io_strategy: Option<izwi_core::LoadingIoStrategy>,
    /// LOADING workers (0 selects automatically)
    #[arg(long)]
    pub loading_workers: Option<usize>,
    /// LOADING max staging bytes (positive byte limit; default 256 MiB)
    #[arg(long, value_parser = parse_staging_bytes)]
    pub loading_max_staging_bytes: Option<usize>,
    /// LOADING cache max bytes
    #[arg(long)]
    pub loading_cache_max_bytes: Option<u64>,
    /// LOADING cache dir
    #[arg(long)]
    pub loading_cache_dir: Option<PathBuf>,
}

fn parse_mtp_depth(value: &str) -> std::result::Result<usize, String> {
    value
        .parse::<usize>()
        .ok()
        .filter(|value| (1..=3).contains(value))
        .ok_or_else(|| "MTP draft tokens must be in 1..=3".to_string())
}
fn parse_staging_bytes(value: &str) -> std::result::Result<usize, String> {
    value
        .parse::<usize>()
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| "staging byte limit must be positive".to_string())
}

impl PerformanceArgs {
    pub fn into_overrides(self) -> izwi_core::PerformanceConfigOverrides {
        let mut overrides = izwi_core::PerformanceConfigOverrides::default();
        overrides.cuda.mode = self.cuda_mode;
        overrides.cuda.projection_backend = self.cuda_projection_backend;
        overrides.cuda.packed_projections = self.cuda_packed_projections;
        overrides.cuda.fused_decode = self.cuda_fused_decode;
        overrides.cuda.device_sampling = self.cuda_device_sampling;
        overrides.cuda.decode_graphs = self.cuda_decode_graphs;
        overrides.cuda.mtp = self.cuda_mtp;
        overrides.cuda.mtp_quantum = self.cuda_mtp_quantum;
        overrides.cuda.mtp_draft_tokens = self.cuda_mtp_draft_tokens;
        overrides.cuda.mtp_adaptive = self.cuda_mtp_adaptive;
        overrides.loading.mode = self.loading_mode;
        overrides.loading.derived_weight_cache = self.loading_derived_weight_cache;
        overrides.loading.parallel_conversion = self.loading_parallel_conversion;
        overrides.loading.pinned_uploads = self.loading_pinned_uploads;
        overrides.loading.io_strategy = self.loading_io_strategy;
        overrides.loading.workers = self.loading_workers;
        overrides.loading.max_staging_bytes = self.loading_max_staging_bytes;
        overrides.loading.cache_max_bytes = self.loading_cache_max_bytes;
        overrides.loading.cache_dir = self.loading_cache_dir;
        overrides
    }
}

#[derive(Subcommand)]
pub enum Commands {
    /// Start the inference server
    ///
    /// Launches the HTTP API server with optional configuration.
    /// Supports graceful shutdown with Ctrl+C.
    #[command(name = "serve", alias = "server")]
    Serve {
        #[command(flatten)]
        performance: PerformanceArgs,
        /// Startup mode
        ///
        /// - server: Start only the HTTP server
        /// - desktop: Start server and desktop app
        /// - web: Start server and open the web UI in your browser
        #[arg(long, value_enum, default_value = "server", env = "IZWI_SERVE_MODE")]
        mode: ServeMode,

        /// Host to bind to
        #[arg(short = 'H', long)]
        host: Option<String>,

        /// Port to listen on
        #[arg(short, long)]
        port: Option<u16>,

        /// Models directory
        #[arg(short, long)]
        models_dir: Option<PathBuf>,

        /// Physical tensor batch width (`auto` or a positive row count)
        #[arg(long, value_name = "AUTO_OR_ROWS")]
        max_batch_size: Option<izwi_core::BatchSizePreference>,

        /// Physical launch rollout mode (`serial`, `shadow`, `concurrent`)
        #[arg(long, value_name = "MODE")]
        physical_execution_mode: Option<izwi_core::PhysicalExecutionMode>,

        /// Maximum candidate physical launches in flight
        #[arg(long, value_name = "COUNT")]
        max_physical_in_flight: Option<izwi_core::PhysicalInFlightLimit>,

        /// Maximum logical rows selected by one scheduler step
        #[arg(long)]
        max_scheduler_batch_size: Option<usize>,

        /// Maximum simultaneously resident model variants
        #[arg(long)]
        max_loaded_models: Option<usize>,

        /// Maximum retained sequence/session rows
        #[arg(long)]
        max_retained_sequences: Option<usize>,

        /// Maximum simultaneously staged managed-state transactions
        #[arg(long)]
        max_staged_transactions: Option<usize>,

        /// Maximum admitted jobs in the runtime inference queue
        #[arg(long)]
        max_queued_requests: Option<usize>,

        /// Portable context length (`auto` or a positive token count)
        #[arg(long, value_name = "AUTO_OR_TOKENS")]
        max_sequence_length: Option<izwi_core::ContextLengthPreference>,

        /// Backend preference (`auto`, `cpu`, `metal`, `cuda`)
        #[arg(long, value_enum)]
        backend: Option<Backend>,

        /// Number of CPU threads
        #[arg(short, long)]
        threads: Option<usize>,

        /// Maximum concurrent requests
        #[arg(long)]
        max_concurrent: Option<usize>,

        /// Request timeout in seconds
        #[arg(long)]
        timeout: Option<u64>,

        /// Log level
        #[arg(long, default_value = "warn", env = "RUST_LOG")]
        log_level: String,

        /// Log output format
        #[arg(long, value_enum, default_value = "text", env = "IZWI_LOG_FORMAT")]
        log_format: LogFormat,

        /// Enable development mode with hot reload
        #[arg(long, hide = true)]
        dev: bool,

        /// Enable wildcard CORS responses
        #[arg(long)]
        cors: bool,

        /// Disable static web UI serving
        #[arg(long)]
        no_ui: bool,
    },

    /// Manage models
    #[command(name = "models", alias = "model")]
    Models {
        #[command(subcommand)]
        command: ModelCommands,
    },

    /// Download a model from HuggingFace
    ///
    /// Pulls a model from the HuggingFace Hub and caches it locally.
    /// Supports resume on interrupted downloads.
    #[command(name = "pull", alias = "download")]
    Pull {
        /// Model variant to download
        ///
        /// Examples: qwen3-tts-0.6b-base, qwen3-tts-1.7b-customvoice
        model: String,

        /// Force re-download even if model exists
        #[arg(short, long)]
        force: bool,

        /// Download without confirmation
        #[arg(short, long)]
        yes: bool,
    },

    /// Remove a downloaded model
    #[command(name = "rm", alias = "remove")]
    Rm {
        /// Model variant to remove
        model: String,

        /// Remove without confirmation
        #[arg(short, long)]
        yes: bool,
    },

    /// List available and downloaded models
    ///
    /// Shows both locally available models and models that can be downloaded.
    #[command(name = "list", alias = "ls")]
    List {
        /// Show only downloaded models
        #[arg(short, long)]
        local: bool,

        /// Show detailed information
        #[arg(short, long)]
        detailed: bool,
    },

    /// Text-to-speech generation
    ///
    /// Generate speech from text using a TTS model.
    /// Supports streaming output and various audio formats.
    #[command(name = "tts", alias = "speak")]
    Tts {
        /// Text to synthesize (or "-" to read from stdin)
        text: String,

        /// Model to use
        #[arg(short, long, default_value = "qwen3-tts-0.6b-base")]
        model: String,

        /// Speaker voice or VibeVoice speaker label
        #[arg(short, long)]
        speaker: Option<String>,

        /// Saved reference voice ID
        #[arg(long, value_name = "ID")]
        saved_voice_id: Option<String>,

        /// Reference audio file for voice cloning
        #[arg(long, value_name = "PATH")]
        reference_audio: Option<PathBuf>,

        /// Reference transcript for voice cloning
        #[arg(long)]
        reference_text: Option<String>,

        /// File containing the reference transcript
        #[arg(long, value_name = "PATH")]
        reference_text_file: Option<PathBuf>,

        /// Voice direction prompt for supported models
        #[arg(long)]
        instructions: Option<String>,

        /// Output file path
        #[arg(short, long, value_name = "PATH")]
        output: Option<PathBuf>,

        /// Audio format
        #[arg(short, long, value_enum, default_value = "wav")]
        format: AudioFormat,

        /// Speech speed multiplier
        #[arg(short = 'r', long, default_value = "1.0")]
        speed: f32,

        /// Temperature for sampling
        #[arg(short, long, default_value = "0.7")]
        temperature: f32,

        /// Stream output in real-time
        #[arg(long)]
        stream: bool,

        /// Allow the server to return WAV audio when the requested compressed format is unavailable
        #[arg(long)]
        allow_format_fallback: bool,

        /// Play audio immediately after generation
        #[arg(short, long)]
        play: bool,
    },

    /// Speech-to-text transcription
    ///
    /// Transcribe audio to text using an ASR model.
    #[command(name = "transcribe", alias = "asr")]
    Transcribe {
        /// Audio file to transcribe
        file: PathBuf,

        /// Model to use
        #[arg(short, long, default_value = "parakeet-tdt-0.6b-v3")]
        model: String,

        /// Language hint (auto-detect if not specified)
        #[arg(short, long)]
        language: Option<String>,

        /// Initial ASR prompt or keyword guidance
        #[arg(long)]
        prompt: Option<String>,

        /// Maximum number of ASR decoder tokens to generate
        #[arg(long)]
        max_tokens: Option<usize>,

        /// Output format
        #[arg(short, long, value_enum, default_value = "text")]
        format: TranscriptFormat,

        /// Output file (default: stdout)
        #[arg(short, long, value_name = "PATH")]
        output: Option<PathBuf>,

        /// Include word-level timestamps
        #[arg(long)]
        word_timestamps: bool,
    },

    /// Chat with a multimodal model
    ///
    /// Interactive chat with audio understanding capabilities.
    #[command(name = "chat")]
    Chat {
        /// Model to use (for example Qwen3-8B-GGUF, Qwen3.8-27B-FP8, or Gemma-3-1b-it)
        #[arg(short, long, default_value = "qwen3-0.6b-4bit")]
        model: String,

        /// Initial system prompt
        #[arg(short, long)]
        system: Option<String>,

        /// Voice to use for responses
        #[arg(short, long)]
        voice: Option<String>,
    },

    /// Speaker diarization
    ///
    /// Identify and separate multiple speakers in audio recordings.
    #[command(name = "diarize", alias = "diar")]
    Diarize {
        /// Audio file to analyze
        file: PathBuf,

        /// Diarization model to use
        #[arg(short, long, default_value = "sortformer-4spk")]
        model: String,

        /// Expected number of speakers (optional, auto-detect if not specified)
        #[arg(short, long)]
        num_speakers: Option<u32>,

        /// Output format
        #[arg(short, long, value_enum, default_value = "text")]
        format: TranscriptFormat,

        /// Output file (default: stdout)
        #[arg(short, long, value_name = "PATH")]
        output: Option<PathBuf>,

        /// Compatibility flag (transcript output is now included by default)
        #[arg(long)]
        transcribe: bool,

        /// ASR model used for transcript generation
        #[arg(long, default_value = "parakeet-tdt-0.6b-v3")]
        asr_model: String,
    },

    /// Forced alignment
    ///
    /// Align text to audio at word level for precise timing.
    #[command(name = "align")]
    Align {
        /// Audio file to align
        file: PathBuf,

        /// Reference text to align
        text: String,

        /// Model to use
        #[arg(short, long, default_value = "qwen3-forcedaligner-0.6b")]
        model: String,

        /// Output format
        #[arg(short, long, value_enum, default_value = "json")]
        format: TranscriptFormat,

        /// Output file (default: stdout)
        #[arg(short, long, value_name = "PATH")]
        output: Option<PathBuf>,
    },

    /// Run benchmarks
    ///
    /// Performance testing for models and inference engine.
    #[command(name = "bench", alias = "benchmark")]
    Bench {
        /// Benchmark type
        #[command(subcommand)]
        command: BenchCommands,
    },

    /// Show system status and health
    ///
    /// Display server health, loaded models, and resource usage.
    #[command(name = "status", alias = "info")]
    Status {
        /// Show detailed metrics
        #[arg(short, long)]
        detailed: bool,

        /// Watch mode (continuous updates)
        #[arg(short, long, value_name = "SECONDS")]
        watch: Option<u64>,
    },

    /// Show version information
    #[command(name = "version", alias = "v")]
    Version {
        /// Show detailed version info including dependencies
        #[arg(short, long)]
        full: bool,
    },

    /// Manage configuration
    #[command(name = "config")]
    Config {
        #[command(subcommand)]
        command: ConfigCommands,
    },

    /// Generate shell completions
    #[command(name = "completions")]
    Completions {
        /// Shell to generate completions for
        #[arg(value_enum)]
        shell: Shell,
    },
}

#[derive(Subcommand)]
pub enum ModelCommands {
    /// List available models
    List {
        /// Show only downloaded models
        #[arg(short, long)]
        local: bool,

        /// Show detailed information
        #[arg(short, long)]
        detailed: bool,
    },

    /// Show model information
    Info {
        /// Model variant
        model: String,

        /// Show raw JSON
        #[arg(long)]
        json: bool,
    },

    /// Load a model into memory
    Load {
        /// Model variant to load
        model: String,

        /// Wait for model to be fully loaded
        #[arg(short, long)]
        wait: bool,
    },

    /// Unload a model from memory
    Unload {
        /// Model variant to unload (or "all")
        model: String,

        /// Unload without confirmation
        #[arg(short, long)]
        yes: bool,
    },

    /// Show download progress
    Progress {
        /// Model variant
        model: Option<String>,
    },
}

#[derive(Subcommand)]
pub enum BenchCommands {
    /// Benchmark chat inference
    Chat {
        /// Model to benchmark
        #[arg(short, long, default_value = "Qwen3.5-4B")]
        model: String,

        /// Number of iterations
        #[arg(short, long, default_value = "10")]
        iterations: u32,

        /// User prompt to send
        #[arg(
            short,
            long,
            default_value = "Summarize the main trade-offs between chunked prefill and continuous batching in two concise paragraphs."
        )]
        prompt: String,

        /// Optional system prompt
        #[arg(long)]
        system: Option<String>,

        /// Maximum completion tokens
        #[arg(long, default_value = "128")]
        max_tokens: usize,

        /// Maximum concurrent requests
        #[arg(short, long, default_value = "1")]
        concurrent: u32,

        /// Enable warmup iteration
        #[arg(long)]
        warmup: bool,
    },

    /// Benchmark TTS inference
    Tts {
        /// Model to benchmark
        #[arg(short, long, default_value = "qwen3-tts-0.6b-base")]
        model: String,

        /// Number of iterations
        #[arg(short, long, default_value = "10")]
        iterations: u32,

        /// Text to synthesize
        #[arg(
            short,
            long,
            default_value = "Hello, this is a benchmark test for text to speech synthesis."
        )]
        text: String,

        /// Speaker voice or VibeVoice speaker label
        #[arg(short, long)]
        speaker: Option<String>,

        /// Saved reference voice ID
        #[arg(long, value_name = "ID")]
        saved_voice_id: Option<String>,

        /// Reference audio file for voice cloning
        #[arg(long, value_name = "PATH")]
        reference_audio: Option<PathBuf>,

        /// Reference transcript for voice cloning
        #[arg(long)]
        reference_text: Option<String>,

        /// File containing the reference transcript
        #[arg(long, value_name = "PATH")]
        reference_text_file: Option<PathBuf>,

        /// Maximum concurrent requests
        #[arg(short, long, default_value = "1")]
        concurrent: u32,

        /// Enable warmup iteration
        #[arg(long)]
        warmup: bool,

        /// Use SSE streaming and measure first-audio/inter-chunk latency
        #[arg(long)]
        stream: bool,

        /// Explicit maximum generated audio frames
        #[arg(long)]
        max_output_tokens: Option<usize>,

        /// Per-request client timeout in seconds
        #[arg(long, default_value = "900")]
        timeout_secs: u64,
    },

    /// Benchmark ASR inference
    Asr {
        /// Model to benchmark
        #[arg(short, long, default_value = "parakeet-tdt-0.6b-v3")]
        model: String,

        /// Number of iterations
        #[arg(short, long, default_value = "10")]
        iterations: u32,

        /// Audio file to use
        #[arg(short, long)]
        file: Option<PathBuf>,

        /// Optional language hint (for example: en, es)
        #[arg(short = 'l', long)]
        language: Option<String>,

        /// Maximum ASR decode tokens to request
        #[arg(long)]
        max_tokens: Option<usize>,

        /// Maximum concurrent requests
        #[arg(short, long, default_value = "1")]
        concurrent: u32,

        /// Enable warmup iteration
        #[arg(long)]
        warmup: bool,

        /// Use SSE streaming and measure first-transcript/inter-delta latency
        #[arg(long)]
        stream: bool,
    },

    /// Benchmark system throughput
    Throughput {
        /// Duration in seconds
        #[arg(short, long, default_value = "30")]
        duration: u64,

        /// Concurrent requests
        #[arg(short, long, default_value = "1")]
        concurrent: u32,
    },

    /// Run a benchmark manifest
    Run {
        /// Benchmark manifest path (TOML)
        #[arg(value_name = "PATH")]
        manifest: PathBuf,

        /// Directory to write report, manifest, metadata, and observability artifacts
        #[arg(long, value_name = "DIR")]
        artifact_dir: Option<PathBuf>,
    },

    /// Compare benchmark JSON reports and fail on regressions
    Compare {
        /// Current benchmark report JSON
        current: PathBuf,

        /// Baseline benchmark report JSON
        baseline: PathBuf,

        /// Allowed regression tolerance as a percentage
        #[arg(long, default_value = "5.0")]
        tolerance_percent: f64,
    },
}

#[derive(Subcommand)]
pub enum ConfigCommands {
    /// Show current configuration
    Show,

    /// Set a configuration value
    Set {
        /// Configuration key (e.g., runtime.performance.cuda.mode, runtime.performance.loading.mode)
        key: String,
        /// Configuration value
        value: String,
    },

    /// Get a configuration value
    Get {
        /// Configuration key
        key: String,
    },

    /// Edit configuration in default editor
    Edit,

    /// Reset configuration to defaults
    Reset {
        /// Reset without confirmation
        #[arg(short, long)]
        yes: bool,
    },

    /// Show configuration file path
    Path,
}

#[derive(Clone, ValueEnum)]
pub enum OutputFormat {
    /// Human-readable table format
    Table,
    /// JSON output
    Json,
    /// Plain text
    Plain,
    /// YAML format
    Yaml,
}

#[derive(Clone, ValueEnum)]
pub enum AudioFormat {
    /// WAV format (PCM)
    Wav,
    /// MP3 format
    Mp3,
    /// OGG Vorbis
    Ogg,
    /// FLAC format
    Flac,
    /// AAC format
    Aac,
}

#[derive(Clone, ValueEnum)]
pub enum TranscriptFormat {
    /// Plain text output
    Text,
    /// JSON format with metadata
    Json,
    /// Verbose JSON format with timing metadata
    VerboseJson,
}

#[derive(Clone, ValueEnum)]
pub enum Shell {
    Bash,
    Zsh,
    Fish,
    PowerShell,
    Elvish,
}

#[derive(Clone, ValueEnum)]
pub enum ServeMode {
    /// Start the API server only
    Server,
    /// Start API server and desktop application
    Desktop,
    /// Start API server and open the web UI in a browser tab
    Web,
}

#[derive(Clone, ValueEnum)]
pub enum LogFormat {
    Text,
    Json,
}

impl LogFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Json => "json",
        }
    }
}

#[derive(Clone, ValueEnum)]
pub enum Backend {
    Auto,
    Cpu,
    Metal,
    Cuda,
}

impl Backend {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }

    pub fn as_preference(&self) -> BackendPreference {
        match self {
            Self::Auto => BackendPreference::Auto,
            Self::Cpu => BackendPreference::Cpu,
            Self::Metal => BackendPreference::Metal,
            Self::Cuda => BackendPreference::Cuda,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Cli, Commands};
    use clap::Parser;
    use izwi_core::PhysicalExecutionMode;

    #[test]
    fn chat_model_accepts_qwen38_catalog_id_as_free_form_input() {
        let cli = Cli::try_parse_from(["izwi", "chat", "--model", "Qwen3.8-27B-FP8"])
            .expect("Qwen3.8 catalog id should parse");

        match cli.command {
            Commands::Chat { model, .. } => assert_eq!(model, "Qwen3.8-27B-FP8"),
            _ => panic!("expected chat command"),
        }
    }

    #[test]
    fn serve_parses_typed_physical_execution_controls() {
        let cli = Cli::try_parse_from([
            "izwi",
            "serve",
            "--physical-execution-mode",
            "shadow",
            "--max-physical-in-flight",
            "3",
        ])
        .expect("physical execution controls should parse");

        match cli.command {
            Commands::Serve {
                physical_execution_mode,
                max_physical_in_flight,
                ..
            } => {
                assert_eq!(physical_execution_mode, Some(PhysicalExecutionMode::Shadow));
                assert_eq!(max_physical_in_flight.map(|limit| limit.get()), Some(3));
            }
            _ => panic!("expected serve command"),
        }
    }

    #[test]
    fn serve_performance_flags_are_partial_and_preserve_explicit_false() {
        let cli = Cli::try_parse_from([
            "izwi",
            "serve",
            "--cuda-performance",
            "off",
            "--cuda-mtp",
            "auto",
            "--cuda-mtp-adaptive",
            "false",
            "--loading-workers",
            "0",
            "--loading-performance",
            "off",
        ])
        .unwrap();
        let Commands::Serve { performance, .. } = cli.command else {
            panic!("serve");
        };
        let overrides = performance.into_overrides();
        assert_eq!(overrides.cuda.mode, Some(izwi_core::OptimizationMode::Off));
        assert_eq!(overrides.cuda.mtp, Some(izwi_core::OptimizationMode::Auto));
        assert_eq!(overrides.cuda.mtp_adaptive, Some(false));
        assert_eq!(overrides.loading.workers, Some(0));
        assert!(overrides.cuda.packed_projections.is_none());
        assert!(overrides.loading.derived_weight_cache.is_none());
    }

    #[test]
    fn performance_help_and_invalid_values_cover_operator_contract() {
        use clap::CommandFactory;
        let mut command = Cli::command();
        let help = command
            .find_subcommand_mut("serve")
            .unwrap()
            .render_long_help()
            .to_string();
        for flag in [
            "--cuda-performance",
            "--cuda-projection-backend",
            "--cuda-mtp-adaptive",
            "--loading-performance",
            "--loading-max-staging-bytes",
            "--loading-cache-dir",
        ] {
            assert!(help.contains(flag), "missing {flag}");
        }
        for (flag, value) in [
            ("--cuda-mtp-draft-tokens", "0"),
            ("--cuda-mtp-draft-tokens", "4"),
            ("--cuda-mtp-adaptive", "maybe"),
            ("--cuda-performance", "on"),
            ("--loading-max-staging-bytes", "0"),
            ("--loading-io-strategy", "random"),
        ] {
            assert!(
                Cli::try_parse_from(["izwi", "serve", flag, value]).is_err(),
                "{flag}={value}"
            );
        }
    }
}
