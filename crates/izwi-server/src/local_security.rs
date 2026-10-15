//! Network security for the local (single-node) server role.
//!
//! The local role binds loopback by default. Binding any other address needs
//! the built-in API key (`IZWI_API_KEY`) unless the operator explicitly opts
//! out with `--allow-unauthenticated-network`. Like the gateway credentials,
//! the key is read only from the environment so it never appears in argv, and
//! only its SHA-256 digest is kept in memory.
//!
//! The key is enforced through the enterprise `AuthProvider` seam: it wraps
//! whatever provider the build supplies, so the principal (and therefore the
//! tenant) still comes from that provider.

use async_trait::async_trait;
use izwi_hooks::{
    AuthProvider, EnterpriseHooks, HookError, HookResult, Principal, RequestEnvelope,
};
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::fmt;
use std::sync::Arc;

use crate::gateway_security::{constant_time_eq, is_loopback_host, validate_secret};

pub(crate) const API_KEY_ENV: &str = "IZWI_API_KEY";
const API_KEY_HEADER: &str = "x-api-key";
/// Same-site cookie the web UI sets after asking for the key, so browser
/// requests that cannot carry headers (EventSource, WebSocket, media URLs)
/// authenticate too.
const API_KEY_COOKIE: &str = "izwi_api_key";

/// Probes stay reachable without a key so supervisors and container health
/// checks keep working.
const PUBLIC_PROBE_PATHS: &[&str] = &[
    "/livez",
    "/readyz",
    "/v1/live",
    "/v1/ready",
    "/internal/live",
    "/internal/ready",
];

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub(crate) enum LocalSecurityError {
    #[error("IZWI_API_KEY must be 16-4096 printable ASCII characters without spaces")]
    InvalidApiKey,
    #[error(
        "refusing to bind {0} without an API key: set IZWI_API_KEY, bind 127.0.0.1, or pass \
         --allow-unauthenticated-network to expose the server without authentication"
    )]
    UnauthenticatedNetworkBind(String),
}

/// Digest of the configured local API key.
#[derive(Clone)]
pub(crate) struct LocalApiKey {
    digest: [u8; 32],
}

impl LocalApiKey {
    pub(crate) fn from_env() -> Result<Option<Self>, LocalSecurityError> {
        match std::env::var(API_KEY_ENV) {
            Ok(value) if !value.is_empty() => Self::new(&value).map(Some),
            _ => Ok(None),
        }
    }

    pub(crate) fn new(secret: &str) -> Result<Self, LocalSecurityError> {
        validate_secret(secret).map_err(|_| LocalSecurityError::InvalidApiKey)?;
        Ok(Self {
            digest: Sha256::digest(secret.as_bytes()).into(),
        })
    }

    fn matches(&self, candidate: &str) -> bool {
        let digest: [u8; 32] = Sha256::digest(candidate.as_bytes()).into();
        constant_time_eq(&digest, &self.digest)
    }
}

impl fmt::Debug for LocalApiKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("LocalApiKey(<redacted>)")
    }
}

/// Refuse a non-loopback bind without a key unless explicitly allowed.
pub(crate) fn validate_local_bind(
    host: &str,
    api_key: Option<&LocalApiKey>,
    allow_unauthenticated_network: bool,
) -> Result<(), LocalSecurityError> {
    if is_loopback_host(host) || api_key.is_some() || allow_unauthenticated_network {
        return Ok(());
    }
    Err(LocalSecurityError::UnauthenticatedNetworkBind(
        host.to_string(),
    ))
}

/// Install the API key check in front of the build's auth provider.
pub(crate) fn with_api_key(
    mut hooks: EnterpriseHooks,
    api_key: Option<LocalApiKey>,
) -> EnterpriseHooks {
    if let Some(key) = api_key {
        hooks.auth = Arc::new(LocalApiKeyAuth {
            key,
            inner: hooks.auth.clone(),
        });
    }
    hooks
}

struct LocalApiKeyAuth {
    key: LocalApiKey,
    inner: Arc<dyn AuthProvider>,
}

#[async_trait]
impl AuthProvider for LocalApiKeyAuth {
    async fn authenticate(&self, request: &RequestEnvelope) -> HookResult<Principal> {
        if !is_public_request(&request.method, &request.path)
            && !presented_api_keys(request).any(|candidate| self.key.matches(&candidate))
        {
            return Err(HookError::Denied(
                "a valid API key is required (Authorization: Bearer <key> or x-api-key)"
                    .to_string(),
            ));
        }
        self.inner.authenticate(request).await
    }
}

/// Probes, and GET/HEAD requests outside the API (the web UI's static files
/// and docs), need no key. Everything under `/v1` and `/internal` does.
fn is_public_request(method: &str, path: &str) -> bool {
    if !matches!(method, "GET" | "HEAD") {
        return false;
    }
    PUBLIC_PROBE_PATHS.contains(&path) || !(is_under(path, "/v1") || is_under(path, "/internal"))
}

fn is_under(path: &str, prefix: &str) -> bool {
    path.strip_prefix(prefix)
        .is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
}

fn presented_api_keys(request: &RequestEnvelope) -> impl Iterator<Item = Cow<'_, str>> {
    request
        .headers
        .iter()
        .filter_map(|header| match header.name.as_str() {
            "authorization" => header
                .value
                .split_once(' ')
                .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"))
                .map(|(_, token)| Cow::Borrowed(token.trim())),
            API_KEY_HEADER => Some(Cow::Borrowed(header.value.trim())),
            "cookie" => header.value.split(';').find_map(|pair| {
                pair.trim()
                    .strip_prefix(API_KEY_COOKIE)
                    .and_then(|rest| rest.strip_prefix('='))
                    .and_then(percent_decode)
                    .map(Cow::Owned)
            }),
            _ => None,
        })
}

/// Decode the `encodeURIComponent` form the web UI stores in the cookie.
fn percent_decode(value: &str) -> Option<String> {
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            let hex = std::str::from_utf8(bytes.get(index + 1..index + 3)?).ok()?;
            decoded.push(u8::from_str_radix(hex, 16).ok()?);
            index += 3;
        } else {
            decoded.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8(decoded).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use izwi_hooks::HeaderPair;

    const KEY: &str = "local-test-api-key-123456";

    fn envelope(method: &str, path: &str, headers: &[(&str, &str)]) -> RequestEnvelope {
        RequestEnvelope {
            correlation_id: "test".to_string(),
            method: method.to_string(),
            path: path.to_string(),
            headers: headers
                .iter()
                .map(|(name, value)| HeaderPair {
                    name: name.to_string(),
                    value: value.to_string(),
                })
                .collect(),
            remote_addr: None,
        }
    }

    fn auth() -> Arc<dyn AuthProvider> {
        with_api_key(
            EnterpriseHooks::noop(),
            Some(LocalApiKey::new(KEY).unwrap()),
        )
        .auth
    }

    #[test]
    fn non_loopback_bind_requires_a_key_or_explicit_opt_out() {
        let key = LocalApiKey::new(KEY).unwrap();
        for host in ["127.0.0.1", "localhost", "::1", "[::1]"] {
            assert_eq!(validate_local_bind(host, None, false), Ok(()), "{host}");
        }
        for host in ["0.0.0.0", "::", "192.168.1.20", "my-mac.local"] {
            assert!(matches!(
                validate_local_bind(host, None, false),
                Err(LocalSecurityError::UnauthenticatedNetworkBind(_))
            ));
            assert_eq!(validate_local_bind(host, Some(&key), false), Ok(()));
            assert_eq!(validate_local_bind(host, None, true), Ok(()));
        }
    }

    #[test]
    fn api_key_is_validated_and_redacted() {
        assert_eq!(
            LocalApiKey::new("short").unwrap_err(),
            LocalSecurityError::InvalidApiKey
        );
        assert_eq!(
            LocalApiKey::new("has a space in the key").unwrap_err(),
            LocalSecurityError::InvalidApiKey
        );
        let debug = format!("{:?}", LocalApiKey::new(KEY).unwrap());
        assert!(!debug.contains(KEY));
    }

    #[tokio::test]
    async fn api_routes_require_the_key_in_a_header_or_ui_cookie() {
        let auth = auth();
        for path in ["/v1/admin/models", "/v1", "/internal/metrics"] {
            assert!(matches!(
                auth.authenticate(&envelope("GET", path, &[])).await,
                Err(HookError::Denied(_))
            ));
        }
        let wrong = format!("Bearer {KEY}x");
        assert!(auth
            .authenticate(&envelope(
                "POST",
                "/v1/admin/models/m/load",
                &[("authorization", &wrong)]
            ))
            .await
            .is_err());

        let bearer = format!("Bearer {KEY}");
        let cookie = format!("theme=dark; izwi_api_key={}", KEY.replace('-', "%2D"));
        let accepted: [(&str, &str); 3] = [
            ("authorization", &bearer),
            ("x-api-key", KEY),
            ("cookie", &cookie),
        ];
        for header in accepted {
            let principal = auth
                .authenticate(&envelope("POST", "/v1/admin/models/m/load", &[header]))
                .await
                .unwrap_or_else(|err| panic!("{} should authenticate: {err}", header.0));
            assert_eq!(principal, Principal::local_anonymous());
        }
    }

    #[tokio::test]
    async fn probes_and_ui_assets_stay_public() {
        let auth = auth();
        for path in [
            "/livez",
            "/readyz",
            "/v1/ready",
            "/internal/live",
            "/",
            "/assets/a.js",
        ] {
            assert!(
                auth.authenticate(&envelope("GET", path, &[])).await.is_ok(),
                "{path} should not need a key"
            );
        }
        assert!(auth
            .authenticate(&envelope("POST", "/", &[]))
            .await
            .is_err());
        assert!(auth
            .authenticate(&envelope("POST", "/livez", &[]))
            .await
            .is_err());
    }

    #[test]
    fn cookie_values_are_percent_decoded() {
        assert_eq!(percent_decode("a%3Bb%25c").as_deref(), Some("a;b%c"));
        assert_eq!(percent_decode("bad%2"), None);
        assert_eq!(percent_decode("bad%zz"), None);
    }
}
