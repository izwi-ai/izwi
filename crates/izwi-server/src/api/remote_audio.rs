//! Reference audio fetched by URL, for `reference_audio` (and SGLang-Omni's
//! `ref_audio`) on `/v1/audio/speech` and `audio_sample` URLs.
//!
//! Off unless `IZWI_ALLOWED_MEDIA_DOMAINS` names the hosts that may serve it
//! (an entry `example.com` matches that host; `.example.com` also matches its
//! subdomains). Every hop, redirects included (at most five), must be an
//! allowed host whose addresses are all public (`izwi_core::media_policy`);
//! the connection is pinned to the addresses that were checked, no proxy is
//! used, and the body is bounded (10 MiB) and timed (30 s).

use std::net::SocketAddr;
use std::time::Duration;

use crate::error::ApiError;

/// The largest reference clip fetched.
pub(crate) const MAX_REMOTE_AUDIO_BYTES: usize = 10 * 1024 * 1024;
const MAX_REDIRECTS: usize = 5;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// Whether `value` is an http(s) URL rather than base64 or a data URL.
pub(crate) fn is_remote_audio_url(value: &str) -> bool {
    let value = value.trim_start();
    let lower = value.get(..8).unwrap_or(value).to_ascii_lowercase();
    lower.starts_with("http://") || lower.starts_with("https://")
}

/// Whether `host` is on `allowed`.
pub(crate) fn host_allowed(host: &str, allowed: &[String]) -> bool {
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    allowed.iter().any(|entry| match entry.strip_prefix('.') {
        Some(apex) => host == apex || host.ends_with(entry.as_str()),
        None => host == *entry,
    })
}

fn refused(message: impl Into<String>) -> ApiError {
    ApiError::invalid_param("reference_audio", message)
}

/// The bytes at `url`, when policy allows it.
pub(crate) async fn fetch_reference_audio(url: &str) -> Result<Vec<u8>, ApiError> {
    let allowed = &crate::env_config::tuning().speech_api.allowed_media_domains;
    fetch_with_policy(url, allowed, resolve_host).await
}

async fn resolve_host(host: String, port: u16) -> std::io::Result<Vec<SocketAddr>> {
    Ok(tokio::net::lookup_host((host.as_str(), port))
        .await?
        .collect())
}

async fn fetch_with_policy<R, F>(
    url: &str,
    allowed: &[String],
    resolve: R,
) -> Result<Vec<u8>, ApiError>
where
    R: Fn(String, u16) -> F,
    F: std::future::Future<Output = std::io::Result<Vec<SocketAddr>>>,
{
    if allowed.is_empty() {
        return Err(refused(
            "Reference audio URLs are disabled on this server; send the audio as base64, or set IZWI_ALLOWED_MEDIA_DOMAINS.",
        ));
    }
    // reqwest's rustls has no built-in crypto provider in this workspace.
    let _ = rustls::crypto::ring::default_provider().install_default();
    let mut current = reqwest::Url::parse(url.trim())
        .map_err(|err| refused(format!("Invalid reference audio URL: {err}")))?;
    for _ in 0..=MAX_REDIRECTS {
        if !matches!(current.scheme(), "http" | "https") {
            return Err(refused("Reference audio URLs must use http or https"));
        }
        if !current.username().is_empty() || current.password().is_some() {
            return Err(refused("Reference audio URLs cannot carry credentials"));
        }
        let host = current
            .host_str()
            .ok_or_else(|| refused("The reference audio URL has no host"))?
            .trim_matches(['[', ']'])
            .to_string();
        if !host_allowed(&host, allowed) {
            return Err(refused(format!(
                "Host {host} is not in IZWI_ALLOWED_MEDIA_DOMAINS"
            )));
        }
        let port = current
            .port_or_known_default()
            .ok_or_else(|| refused("The reference audio URL has no port"))?;
        let addresses = match host.parse::<std::net::IpAddr>() {
            Ok(address) => vec![SocketAddr::new(address, port)],
            Err(_) => resolve(host.clone(), port)
                .await
                .map_err(|err| refused(format!("Could not resolve {host}: {err}")))?,
        };
        if addresses.is_empty()
            || addresses
                .iter()
                .any(|address| !izwi_core::media_policy::is_public_ip(address.ip()))
        {
            return Err(refused(format!("Host {host} is not publicly routable")));
        }
        let client = reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(REQUEST_TIMEOUT)
            .redirect(reqwest::redirect::Policy::none())
            // A proxy would bypass the checked, pinned addresses.
            .no_proxy()
            .resolve_to_addrs(&host, &addresses)
            .build()
            .map_err(|err| ApiError::internal(format!("HTTP client failed: {err}")))?;
        let mut response = client
            .get(current.clone())
            .send()
            .await
            .map_err(|err| refused(format!("Fetching the reference audio failed: {err}")))?;
        if response.status().is_redirection() {
            let location = response
                .headers()
                .get(reqwest::header::LOCATION)
                .and_then(|value| value.to_str().ok())
                .ok_or_else(|| refused("A reference audio redirect has no Location"))?;
            current = current
                .join(location)
                .map_err(|err| refused(format!("Bad reference audio redirect: {err}")))?;
            continue;
        }
        if !response.status().is_success() {
            return Err(refused(format!(
                "Fetching the reference audio returned HTTP {}",
                response.status().as_u16()
            )));
        }
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("application/octet-stream")
            .to_ascii_lowercase();
        if !(content_type.starts_with("audio/")
            || content_type.starts_with("video/")
            || content_type.starts_with("application/octet-stream"))
        {
            return Err(refused(format!(
                "The reference audio URL served {content_type}, not audio"
            )));
        }
        if response
            .content_length()
            .is_some_and(|length| length > MAX_REMOTE_AUDIO_BYTES as u64)
        {
            return Err(refused("The reference audio is larger than 10 MiB"));
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|err| refused(format!("Reading the reference audio failed: {err}")))?
        {
            if bytes.len() + chunk.len() > MAX_REMOTE_AUDIO_BYTES {
                return Err(refused("The reference audio is larger than 10 MiB"));
            }
            bytes.extend_from_slice(&chunk);
        }
        return Ok(bytes);
    }
    Err(refused(format!(
        "The reference audio URL redirected more than {MAX_REDIRECTS} times"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hosts_match_exactly_or_by_dotted_suffix() {
        let allowed = vec!["cdn.example.com".to_string(), ".media.test".to_string()];
        assert!(host_allowed("CDN.example.com.", &allowed));
        assert!(!host_allowed("evil-cdn.example.com", &allowed));
        assert!(host_allowed("media.test", &allowed));
        assert!(host_allowed("a.b.media.test", &allowed));
        assert!(!host_allowed("notmedia.test", &allowed));
        assert!(is_remote_audio_url(" HTTPS://x/y.wav"));
        assert!(!is_remote_audio_url("UklGRg=="));
        assert!(!is_remote_audio_url("data:audio/wav;base64,AAAA"));
    }

    #[tokio::test]
    async fn urls_are_refused_without_an_allowlist_or_off_it_or_off_public_addresses() {
        let none: Vec<String> = Vec::new();
        let public = |_: String, port: u16| async move {
            Ok(vec![SocketAddr::from(([93, 184, 216, 34], port))])
        };
        let err = fetch_with_policy("https://a.test/x.wav", &none, public)
            .await
            .unwrap_err();
        assert!(err.message.contains("IZWI_ALLOWED_MEDIA_DOMAINS"));

        let allowed = vec!["a.test".to_string()];
        let err = fetch_with_policy("https://b.test/x.wav", &allowed, public)
            .await
            .unwrap_err();
        assert!(err.message.contains("not in"));

        let private =
            |_: String, port: u16| async move { Ok(vec![SocketAddr::from(([10, 0, 0, 5], port))]) };
        let err = fetch_with_policy("https://a.test/x.wav", &allowed, private)
            .await
            .unwrap_err();
        assert!(err.message.contains("not publicly routable"));
        let err = fetch_with_policy("http://127.0.0.1/x.wav", &["127.0.0.1".to_string()], public)
            .await
            .unwrap_err();
        assert!(err.message.contains("not publicly routable"));
        let err = fetch_with_policy("ftp://a.test/x.wav", &allowed, public)
            .await
            .unwrap_err();
        assert!(err.message.contains("http or https"));
    }
}
