use std::fmt;
use std::io;

pub type Result<T> = std::result::Result<T, CliError>;

/// Longest server error text the CLI repeats; the rest is cut.
const MAX_API_MESSAGE_CHARS: usize = 500;

#[derive(Debug)]
pub enum CliError {
    /// API request failed; `message` is the server's own explanation
    ApiError { status: u16, message: String },
    /// Server connection failed
    ConnectionError(String),
    /// Model not found
    ModelNotFound(String),
    /// Invalid input
    InvalidInput(String),
    /// Configuration error
    ConfigError(String),
    /// I/O error
    Io(io::Error),
    /// Serialization error
    Serialization(serde_json::Error),
    /// Other errors
    Other(String),
}

impl CliError {
    /// The error for a response with an error status: the server's
    /// `error.message` when the body carries one, else the body text.
    pub async fn from_response(response: reqwest::Response) -> Self {
        let status = response.status().as_u16();
        let body = response.text().await.unwrap_or_default();
        CliError::ApiError {
            status,
            message: api_error_message(&body),
        }
    }
}

/// The human message in an error body: `{"error": {"message": ...}}` (the
/// server's and OpenAI's shape), `{"error": "..."}` or `{"message": ...}`,
/// else the body itself on one line.
fn api_error_message(body: &str) -> String {
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(body) {
        let error = value.get("error");
        let message = error
            .and_then(|error| error.get("message"))
            .or(error)
            .or_else(|| value.get("message"))
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|message| !message.is_empty());
        if let Some(message) = message {
            return message.to_string();
        }
    }
    let text = body.split_whitespace().collect::<Vec<_>>().join(" ");
    if text.is_empty() {
        return "the server gave no details".to_string();
    }
    match text.char_indices().nth(MAX_API_MESSAGE_CHARS) {
        Some((cut, _)) => format!("{}...", &text[..cut]),
        None => text,
    }
}

/// `error` and its causes, without repeating a cause its parent already
/// quotes.
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

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CliError::ApiError { status, message } => {
                write!(f, "{} (HTTP {})", message, status)
            }
            CliError::ConnectionError(msg) => write!(f, "{}", msg),
            CliError::ModelNotFound(model) => write!(
                f,
                "Model not found: {}. Run `izwi list` to see the available models.",
                model
            ),
            CliError::InvalidInput(msg) => write!(f, "{}", msg),
            CliError::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
            CliError::Io(e) => write!(f, "I/O error: {}", e),
            CliError::Serialization(e) => write!(f, "Serialization error: {}", e),
            CliError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for CliError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CliError::Io(e) => Some(e),
            CliError::Serialization(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for CliError {
    fn from(e: io::Error) -> Self {
        CliError::Io(e)
    }
}

impl From<serde_json::Error> for CliError {
    fn from(e: serde_json::Error) -> Self {
        CliError::Serialization(e)
    }
}

impl From<reqwest::Error> for CliError {
    fn from(e: reqwest::Error) -> Self {
        let server = e
            .url()
            .map(|url| url.origin().ascii_serialization())
            .unwrap_or_else(|| "the server".to_string());
        if e.is_connect() {
            CliError::ConnectionError(format!(
                "Could not connect to the Izwi server at {server}. Is it running? \
                 Start it with `izwi serve`, or point at another one with --server <URL>."
            ))
        } else if e.is_timeout() {
            CliError::ConnectionError(format!("The request to {server} timed out."))
        } else {
            CliError::Other(error_chain(&e))
        }
    }
}

impl From<anyhow::Error> for CliError {
    fn from(e: anyhow::Error) -> Self {
        CliError::Other(format!("{e:#}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_errors_show_the_server_message_on_one_line() {
        let body = r#"{"error":{"code":"model_not_found","message":"Model not found: Qwen3 0.6B 4-bit not downloaded.","param":"model","type":"not_found_error"}}"#;
        let error = CliError::ApiError {
            status: 404,
            message: api_error_message(body),
        };
        assert_eq!(
            error.to_string(),
            "Model not found: Qwen3 0.6B 4-bit not downloaded. (HTTP 404)"
        );
        assert_eq!(api_error_message(r#"{"error":"busy"}"#), "busy");
        assert_eq!(api_error_message(r#"{"message":"nope"}"#), "nope");
        assert_eq!(
            api_error_message("Failed to buffer\nthe body: length limit exceeded\n"),
            "Failed to buffer the body: length limit exceeded"
        );
        assert_eq!(api_error_message(""), "the server gave no details");
        assert_eq!(api_error_message(&"x".repeat(600)).len(), 503);
    }

    #[tokio::test]
    async fn an_unreachable_server_is_named_with_a_hint() {
        let error: CliError = reqwest::Client::builder()
            .no_proxy()
            .build()
            .unwrap()
            .get("http://127.0.0.1:1/v1/health")
            .send()
            .await
            .unwrap_err()
            .into();
        let message = error.to_string();
        assert!(
            message.starts_with("Could not connect to the Izwi server at http://127.0.0.1:1."),
            "{message}"
        );
        assert!(message.contains("izwi serve"), "{message}");
    }
}
