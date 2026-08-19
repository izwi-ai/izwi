//! Error types for the Izwi TTS engine

use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Model error: {0}")]
    ModelError(String),

    #[error("Model loading failed: {0}")]
    ModelLoadError(String),

    #[error("Download failed: {0}")]
    DownloadError(String),

    #[error("Tokenization error: {0}")]
    TokenizationError(String),

    #[error("Inference error: {0}")]
    InferenceError(String),

    #[error("Request timed out: {0}")]
    Timeout(String),

    #[error("Request cancelled: {0}")]
    Cancelled(String),

    #[error("Inference capacity unavailable: {0}")]
    Overloaded(String),

    #[error("Streaming backpressure: {0}")]
    Backpressure(String),

    #[error("Audio encoding error: {0}")]
    AudioError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Missing runtime dependency: {0}")]
    MissingDependency(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("HTTP error: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("HuggingFace Hub error: {0}")]
    HfHubError(String),

    #[error("Safetensors error: {0}")]
    SafetensorsError(String),

    #[error("Unsupported platform: {0}")]
    UnsupportedPlatform(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<hf_hub::api::sync::ApiError> for Error {
    fn from(e: hf_hub::api::sync::ApiError) -> Self {
        Error::HfHubError(e.to_string())
    }
}

impl From<safetensors::SafeTensorError> for Error {
    fn from(e: safetensors::SafeTensorError) -> Self {
        Error::SafetensorsError(e.to_string())
    }
}

impl From<candle_core::Error> for Error {
    fn from(e: candle_core::Error) -> Self {
        let message = e.to_string();
        if is_allocation_oom(&message) {
            Error::Overloaded(format!(
                "accelerator allocation could not be satisfied after managed-capacity admission: {message}"
            ))
        } else {
            Error::InferenceError(message)
        }
    }
}

fn is_allocation_oom(message: &str) -> bool {
    let normalized = message.to_ascii_lowercase();
    normalized.contains("cuda_error_out_of_memory")
        || normalized.contains("out of memory")
        || normalized.contains("outofmemory")
        || normalized.contains("memory allocation failed")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candle_cuda_oom_becomes_typed_capacity_error() {
        let error = Error::from(candle_core::Error::Msg(
            "DriverError(CUDA_ERROR_OUT_OF_MEMORY, out of memory)".into(),
        ));
        assert!(
            matches!(error, Error::Overloaded(message) if message.contains("CUDA_ERROR_OUT_OF_MEMORY"))
        );
    }

    #[test]
    fn ordinary_candle_failures_remain_inference_errors() {
        let error = Error::from(candle_core::Error::Msg("shape mismatch".into()));
        assert!(matches!(error, Error::InferenceError(message) if message == "shape mismatch"));
    }
}
