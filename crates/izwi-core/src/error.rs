//! Error types for the Izwi TTS engine

use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// The prompt, or the prompt plus an explicit output budget, does not
    /// fit the model's context.
    #[error("{0}")]
    ContextLengthExceeded(String),

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

/// The display prefix of each message-carrying variant. Engine rows carry
/// errors as display text; [`Error::from_display`] turns that text back
/// into its variant with this table. `ContextLengthExceeded` displays its
/// bare message, so it has no prefix to decode.
const DISPLAY_PREFIXES: &[(&str, fn(String) -> Error)] = &[
    ("Invalid input: ", Error::InvalidInput),
    ("Model not found: ", Error::ModelNotFound),
    ("Model error: ", Error::ModelError),
    ("Model loading failed: ", Error::ModelLoadError),
    ("Download failed: ", Error::DownloadError),
    ("Tokenization error: ", Error::TokenizationError),
    ("Inference error: ", Error::InferenceError),
    ("Request timed out: ", Error::Timeout),
    ("Request cancelled: ", Error::Cancelled),
    ("Inference capacity unavailable: ", Error::Overloaded),
    ("Streaming backpressure: ", Error::Backpressure),
    ("Audio encoding error: ", Error::AudioError),
    ("Configuration error: ", Error::ConfigError),
    ("Missing runtime dependency: ", Error::MissingDependency),
    ("HuggingFace Hub error: ", Error::HfHubError),
    ("Safetensors error: ", Error::SafetensorsError),
    ("Unsupported platform: ", Error::UnsupportedPlatform),
];

impl Error {
    /// The error whose display text is `message`, as an engine row carries
    /// it. Text without a variant's prefix is the row's own failure message
    /// and becomes an `InferenceError`, so no message gains a second prefix.
    pub(crate) fn from_display(message: String) -> Self {
        DISPLAY_PREFIXES
            .iter()
            .find_map(|(prefix, variant)| {
                message
                    .strip_prefix(prefix)
                    .map(|detail| variant(detail.to_owned()))
            })
            .unwrap_or(Self::InferenceError(message))
    }

    /// An error of the same kind with the same message, for reporting one
    /// failure on several rows. Kinds that wrap a foreign error (I/O,
    /// serialization, HTTP) become an `InferenceError` with its text.
    pub(crate) fn replicate(&self) -> Self {
        match self {
            Self::InvalidInput(message) => Self::InvalidInput(message.clone()),
            Self::ContextLengthExceeded(message) => Self::ContextLengthExceeded(message.clone()),
            Self::ModelNotFound(message) => Self::ModelNotFound(message.clone()),
            Self::ModelError(message) => Self::ModelError(message.clone()),
            Self::ModelLoadError(message) => Self::ModelLoadError(message.clone()),
            Self::DownloadError(message) => Self::DownloadError(message.clone()),
            Self::TokenizationError(message) => Self::TokenizationError(message.clone()),
            Self::InferenceError(message) => Self::InferenceError(message.clone()),
            Self::Timeout(message) => Self::Timeout(message.clone()),
            Self::Cancelled(message) => Self::Cancelled(message.clone()),
            Self::Overloaded(message) => Self::Overloaded(message.clone()),
            Self::Backpressure(message) => Self::Backpressure(message.clone()),
            Self::AudioError(message) => Self::AudioError(message.clone()),
            Self::ConfigError(message) => Self::ConfigError(message.clone()),
            Self::MissingDependency(message) => Self::MissingDependency(message.clone()),
            Self::HfHubError(message) => Self::HfHubError(message.clone()),
            Self::SafetensorsError(message) => Self::SafetensorsError(message.clone()),
            Self::UnsupportedPlatform(message) => Self::UnsupportedPlatform(message.clone()),
            Self::IoError(_) | Self::SerializationError(_) | Self::HttpError(_) => {
                Self::InferenceError(self.to_string())
            }
        }
    }
}

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

    /// Every prefixed variant survives the trip through an engine row's
    /// display text with its variant and message, and text that never had a
    /// prefix gains exactly one.
    #[test]
    fn display_text_round_trips_every_variant() {
        let detail = || "detail".to_string();
        for error in [
            Error::InvalidInput(detail()),
            Error::ModelNotFound(detail()),
            Error::ModelError(detail()),
            Error::ModelLoadError(detail()),
            Error::DownloadError(detail()),
            Error::TokenizationError(detail()),
            Error::InferenceError(detail()),
            Error::Timeout(detail()),
            Error::Cancelled(detail()),
            Error::Overloaded(detail()),
            Error::Backpressure(detail()),
            Error::AudioError(detail()),
            Error::ConfigError(detail()),
            Error::MissingDependency(detail()),
            Error::HfHubError(detail()),
            Error::SafetensorsError(detail()),
            Error::UnsupportedPlatform(detail()),
        ] {
            let text = error.to_string();
            let decoded = Error::from_display(text.clone());
            assert_eq!(decoded.to_string(), text);
            assert_eq!(
                std::mem::discriminant(&decoded),
                std::mem::discriminant(&error),
                "{text}"
            );
        }
        let bare = Error::from_display("unsupported const-set f64".into());
        assert_eq!(
            bare.to_string(),
            "Inference error: unsupported const-set f64"
        );
    }

    #[test]
    fn ordinary_candle_failures_remain_inference_errors() {
        let error = Error::from(candle_core::Error::Msg("shape mismatch".into()));
        assert!(matches!(error, Error::InferenceError(message) if message == "shape mismatch"));
    }
}
