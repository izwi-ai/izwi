use crate::error::{CliError, Result};
use std::time::Duration;

/// API key the server requires when it is bound to a non-loopback address.
pub const API_KEY_ENV: &str = "IZWI_API_KEY";

pub fn client(timeout: Option<Duration>) -> Result<reqwest::Client> {
    let mut builder = reqwest::Client::builder()
        .no_proxy()
        .user_agent(format!("izwi-cli/{}", env!("CARGO_PKG_VERSION")))
        .default_headers(auth_headers(std::env::var(API_KEY_ENV).ok().as_deref())?);

    if let Some(timeout) = timeout {
        builder = builder.timeout(timeout);
    }

    builder
        .build()
        .map_err(|e| CliError::Other(format!("Failed to initialize HTTP client: {}", e)))
}

fn auth_headers(api_key: Option<&str>) -> Result<reqwest::header::HeaderMap> {
    let mut headers = reqwest::header::HeaderMap::new();
    if let Some(key) = api_key.filter(|key| !key.is_empty()) {
        let mut value = reqwest::header::HeaderValue::from_str(&format!("Bearer {key}"))
            .map_err(|_| CliError::Other(format!("{API_KEY_ENV} is not a valid header value")))?;
        value.set_sensitive(true);
        headers.insert(reqwest::header::AUTHORIZATION, value);
    }
    Ok(headers)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_key_becomes_a_sensitive_bearer_header() {
        assert!(auth_headers(None).unwrap().is_empty());
        assert!(auth_headers(Some("")).unwrap().is_empty());
        let headers = auth_headers(Some("cli-test-api-key-123456")).unwrap();
        let value = &headers[reqwest::header::AUTHORIZATION];
        assert_eq!(value, "Bearer cli-test-api-key-123456");
        assert!(value.is_sensitive());
        assert!(auth_headers(Some("bad\nkey")).is_err());
    }
}
