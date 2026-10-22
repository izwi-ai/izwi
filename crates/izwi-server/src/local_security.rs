//! Network security for the local (single-node) server role.
//!
//! The local role binds loopback by default. Binding any other address needs
//! the built-in API key (`IZWI_API_KEY`) unless the operator explicitly opts
//! out with `--allow-unauthenticated-network`, or declares with
//! `IZWI_LOOPBACK_PUBLISHED` that a container publishes the port on the
//! host's loopback only, in which case the bind is treated as loopback. Like
//! the gateway credentials, the key is read only from the environment so it
//! never appears in argv, and only its SHA-256 digest is kept in memory.
//!
//! The key is enforced through the enterprise `AuthProvider` seam: it wraps
//! whatever provider the build supplies, so the principal (and therefore the
//! tenant) still comes from that provider.
//!
//! Independently of the key, [`guard_local_request`] protects browsers'
//! access to the server: a loopback-only server only answers loopback `Host`
//! names and `IZWI_ALLOWED_HOSTS` (DNS rebinding), and writes and WebSocket
//! upgrades from another site's page are rejected by `Origin`, plus form
//! content types that a cross-site form can send without a CORS preflight.

use async_trait::async_trait;
use axum::extract::{Request, State};
use axum::http::{header, HeaderMap, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use izwi_core::ServeRuntimeConfig;
use izwi_hooks::{
    AuthProvider, EnterpriseHooks, HookError, HookResult, Principal, RequestEnvelope,
};
use izwi_serving_protocol::secret_file::{read_secret_file, SecretFileError};
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::ffi::OsString;
use std::fmt;
use std::path::Path;
use std::sync::Arc;

use crate::api::DESKTOP_CORS_ORIGINS;
use crate::error::ApiError;
use crate::gateway_security::{constant_time_eq, is_loopback_host, validate_secret};

pub(crate) const API_KEY_ENV: &str = "IZWI_API_KEY";
/// A file holding the key instead (a Docker or systemd secret).
pub(crate) const API_KEY_FILE_ENV: &str = "IZWI_API_KEY_FILE";
/// The key's 4096-byte maximum plus a trailing CRLF.
const MAX_API_KEY_FILE_BYTES: u64 = 4098;
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
    #[error("set IZWI_API_KEY or IZWI_API_KEY_FILE, not both")]
    ConflictingApiKeySources,
    #[error(
        "CORS allows every origin, which would let any web page read this server's API \
         without a key: list the origins in IZWI_CORS_ORIGINS, or set IZWI_API_KEY"
    )]
    UnauthenticatedWildcardCors,
    #[error("IZWI_API_KEY_FILE {0}: {1}")]
    ApiKeyFile(String, SecretFileError),
    #[error(
        "refusing to bind {0} without an API key: set IZWI_API_KEY, bind 127.0.0.1, set \
         IZWI_LOOPBACK_PUBLISHED=1 if a container publishes the port on the host's loopback \
         only, or pass --allow-unauthenticated-network to expose the server without \
         authentication"
    )]
    UnauthenticatedNetworkBind(String),
}

/// How clients reach the port, beyond what the bind address says
/// (`IZWI_LOOPBACK_PUBLISHED`, `IZWI_ALLOWED_HOSTS`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct LocalExposure {
    /// A container publishes the port on the host's loopback only, so a
    /// non-loopback bind is treated as a loopback one.
    pub(crate) loopback_published: bool,
    /// Extra lower-cased `Host` names to answer. On a network bind, a
    /// non-empty list turns the Host check on.
    pub(crate) allowed_hosts: Vec<String>,
}

/// Digest of the configured local API key.
#[derive(Clone)]
pub(crate) struct LocalApiKey {
    digest: [u8; 32],
}

impl LocalApiKey {
    /// `IZWI_API_KEY`, or the file `IZWI_API_KEY_FILE` names. Empty values
    /// count as unset.
    pub(crate) fn from_env() -> Result<Option<Self>, LocalSecurityError> {
        Self::from_sources(
            std::env::var(API_KEY_ENV).ok(),
            std::env::var_os(API_KEY_FILE_ENV),
        )
    }

    fn from_sources(
        value: Option<String>,
        file: Option<OsString>,
    ) -> Result<Option<Self>, LocalSecurityError> {
        match (
            value.filter(|value| !value.is_empty()),
            file.filter(|path| !path.is_empty()),
        ) {
            (Some(_), Some(_)) => Err(LocalSecurityError::ConflictingApiKeySources),
            (Some(value), None) => Self::new(&value).map(Some),
            (None, Some(path)) => Self::from_file(Path::new(&path)).map(Some),
            (None, None) => Ok(None),
        }
    }

    fn from_file(path: &Path) -> Result<Self, LocalSecurityError> {
        let file = read_secret_file(path, MAX_API_KEY_FILE_BYTES)
            .map_err(|error| LocalSecurityError::ApiKeyFile(path.display().to_string(), error))?;
        let text =
            std::str::from_utf8(&file.bytes).map_err(|_| LocalSecurityError::InvalidApiKey)?;
        // `echo key > file` and most secret stores end the file with a newline.
        Self::new(text.trim_end_matches(['\r', '\n']))
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

/// Refuse a non-loopback bind without a key unless it is published on
/// loopback only or explicitly allowed.
pub(crate) fn validate_local_bind(
    host: &str,
    exposure: &LocalExposure,
    api_key: Option<&LocalApiKey>,
    allow_unauthenticated_network: bool,
) -> Result<(), LocalSecurityError> {
    if is_loopback_host(host)
        || exposure.loopback_published
        || api_key.is_some()
        || allow_unauthenticated_network
    {
        return Ok(());
    }
    Err(LocalSecurityError::UnauthenticatedNetworkBind(
        host.to_string(),
    ))
}

/// Wildcard CORS lets any web page read API responses. Without a key that
/// stays allowed only on a native loopback bind (the documented `--cors`);
/// a container's or the opt-out's network bind needs listed origins.
pub(crate) fn validate_local_cors(
    serve_config: &ServeRuntimeConfig,
    api_key: Option<&LocalApiKey>,
) -> Result<(), LocalSecurityError> {
    if api_key.is_none() && !is_loopback_host(&serve_config.host) && allows_any_origin(serve_config)
    {
        return Err(LocalSecurityError::UnauthenticatedWildcardCors);
    }
    Ok(())
}

fn allows_any_origin(serve_config: &ServeRuntimeConfig) -> bool {
    serve_config.cors_enabled
        && (serve_config.cors_origins.is_empty()
            || serve_config
                .cors_origins
                .iter()
                .any(|origin| origin.trim() == "*"))
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

/// Host and Origin policy for browser-reachable local requests.
#[derive(Clone, Debug)]
pub(crate) struct LocalRequestGuard {
    /// Bound to loopback, or published on loopback only.
    loopback_only: bool,
    bound_host: String,
    allowed_hosts: Arc<[String]>,
    any_origin: bool,
    allowed_origins: Arc<[String]>,
}

impl LocalRequestGuard {
    pub(crate) fn new(serve_config: &ServeRuntimeConfig, exposure: &LocalExposure) -> Self {
        let any_origin = allows_any_origin(serve_config);
        let allowed_origins = if serve_config.cors_enabled {
            serve_config
                .cors_origins
                .iter()
                .map(|origin| origin.trim().to_ascii_lowercase())
                .collect()
        } else {
            Arc::from([])
        };
        Self {
            loopback_only: is_loopback_host(&serve_config.host) || exposure.loopback_published,
            bound_host: host_name(&serve_config.host).to_ascii_lowercase(),
            allowed_hosts: exposure.allowed_hosts.iter().cloned().collect(),
            any_origin,
            allowed_origins,
        }
    }

    /// Loopback-only servers check every `Host`; network binds only once
    /// `IZWI_ALLOWED_HOSTS` names the hosts they serve.
    fn host_allowed(&self, host: &str) -> bool {
        if !self.loopback_only && self.allowed_hosts.is_empty() {
            return true;
        }
        let name = host_name(host);
        is_loopback_name(name)
            || name.eq_ignore_ascii_case(&self.bound_host)
            || self
                .allowed_hosts
                .iter()
                .any(|allowed| name.eq_ignore_ascii_case(allowed))
    }

    fn check(
        &self,
        method: &Method,
        headers: &HeaderMap,
        authority: Option<&str>,
    ) -> Result<(), ApiError> {
        let host = headers
            .get(header::HOST)
            .and_then(|value| value.to_str().ok())
            .or(authority);
        if host.is_some_and(|host| !self.host_allowed(host)) {
            return Err(ApiError::forbidden(
                "Host header is not allowed; add the name to IZWI_ALLOWED_HOSTS",
            ));
        }

        let writes = !matches!(*method, Method::GET | Method::HEAD | Method::OPTIONS);
        let upgrade = headers.contains_key(header::UPGRADE);
        if !writes && !upgrade {
            return Ok(());
        }
        // Browsers always send Origin on cross-origin writes and WebSocket
        // handshakes; non-browser clients usually send none.
        if let Some(origin) = headers.get(header::ORIGIN) {
            let allowed = origin
                .to_str()
                .is_ok_and(|origin| self.origin_allowed(origin, host));
            if !allowed {
                return Err(ApiError::forbidden("Cross-origin request rejected"));
            }
        }
        if writes && has_form_content_type(headers) {
            return Err(ApiError {
                status: StatusCode::UNSUPPORTED_MEDIA_TYPE,
                message: "Use application/json or multipart/form-data".to_string(),
                detail: None,
            });
        }
        Ok(())
    }

    fn origin_allowed(&self, origin: &str, host: Option<&str>) -> bool {
        let origin = origin.trim().to_ascii_lowercase();
        if origin == "null" {
            return false;
        }
        if self.any_origin
            || DESKTOP_CORS_ORIGINS.contains(&origin.as_str())
            || self.allowed_origins.contains(&origin)
        {
            return true;
        }
        let Some(authority) = origin
            .strip_prefix("http://")
            .or_else(|| origin.strip_prefix("https://"))
        else {
            return false;
        };
        // Pages served from this machine (the Vite dev server's proxy keeps
        // its own Origin) and same-origin pages are trusted.
        is_loopback_name(host_name(authority))
            || host.is_some_and(|host| authority.eq_ignore_ascii_case(host.trim()))
    }
}

pub(crate) async fn guard_local_request(
    State(guard): State<LocalRequestGuard>,
    req: Request,
    next: Next,
) -> Response {
    let authority = req.uri().authority().map(|authority| authority.as_str());
    if let Err(err) = guard.check(req.method(), req.headers(), authority) {
        return err.into_response();
    }
    crate::error::showing_server_faults(guard.loopback_only, next.run(req)).await
}

/// Host name without port or IPv6 brackets.
pub(crate) fn host_name(host: &str) -> &str {
    let host = host.trim();
    if let Some(rest) = host.strip_prefix('[') {
        return rest.split(']').next().unwrap_or(rest);
    }
    match host.rsplit_once(':') {
        Some((name, port)) if !name.contains(':') && port.bytes().all(|b| b.is_ascii_digit()) => {
            name
        }
        _ => host,
    }
}

fn is_loopback_name(name: &str) -> bool {
    is_loopback_host(name) || name.to_ascii_lowercase().ends_with(".localhost")
}

/// Content types a cross-site HTML form can send without a CORS preflight,
/// other than multipart (which upload routes accept). No route takes them.
fn has_form_content_type(headers: &HeaderMap) -> bool {
    headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .is_some_and(|essence| {
            let essence = essence.trim();
            essence.eq_ignore_ascii_case("text/plain")
                || essence.eq_ignore_ascii_case("application/x-www-form-urlencoded")
        })
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
        let network = LocalExposure::default();
        let published = LocalExposure {
            loopback_published: true,
            ..LocalExposure::default()
        };
        for host in ["127.0.0.1", "localhost", "::1", "[::1]"] {
            assert_eq!(
                validate_local_bind(host, &network, None, false),
                Ok(()),
                "{host}"
            );
        }
        for host in ["0.0.0.0", "::", "192.168.1.20", "my-mac.local"] {
            assert!(matches!(
                validate_local_bind(host, &network, None, false),
                Err(LocalSecurityError::UnauthenticatedNetworkBind(_))
            ));
            assert_eq!(
                validate_local_bind(host, &network, Some(&key), false),
                Ok(())
            );
            assert_eq!(validate_local_bind(host, &network, None, true), Ok(()));
            assert_eq!(validate_local_bind(host, &published, None, false), Ok(()));
        }
    }

    #[test]
    fn wildcard_cors_needs_a_key_off_loopback() {
        let key = LocalApiKey::new(KEY).unwrap();
        let config = |host: &str, origins: &[&str]| ServeRuntimeConfig {
            host: host.to_string(),
            cors_enabled: true,
            cors_origins: origins.iter().map(|origin| origin.to_string()).collect(),
            ..ServeRuntimeConfig::default()
        };
        for origins in [&[][..], &["*"][..]] {
            assert_eq!(
                validate_local_cors(&config("0.0.0.0", origins), None),
                Err(LocalSecurityError::UnauthenticatedWildcardCors)
            );
            assert_eq!(
                validate_local_cors(&config("0.0.0.0", origins), Some(&key)),
                Ok(())
            );
            assert_eq!(
                validate_local_cors(&config("127.0.0.1", origins), None),
                Ok(())
            );
        }
        assert_eq!(
            validate_local_cors(&config("0.0.0.0", &["http://localhost:3000"]), None),
            Ok(())
        );
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

    #[test]
    fn api_key_can_come_from_a_secret_file() {
        let dir = std::env::temp_dir().join(format!("izwi-api-key-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("izwi_api_key");
        std::fs::write(&file, format!("{KEY}\n")).unwrap();
        let path = || Some(file.clone().into_os_string());

        let key = LocalApiKey::from_sources(None, path()).unwrap().unwrap();
        assert!(key.matches(KEY));
        assert_eq!(
            LocalApiKey::from_sources(Some(String::new()), path())
                .unwrap()
                .map(|key| key.matches(KEY)),
            Some(true)
        );
        assert_eq!(
            LocalApiKey::from_sources(Some(KEY.to_string()), path()).unwrap_err(),
            LocalSecurityError::ConflictingApiKeySources
        );
        assert!(LocalApiKey::from_sources(None, Some(OsString::new()))
            .unwrap()
            .is_none());
        assert!(matches!(
            LocalApiKey::from_sources(None, Some("relative/key".into())),
            Err(LocalSecurityError::ApiKeyFile(
                _,
                SecretFileError::NotAbsolute
            ))
        ));
        std::fs::write(&file, "short\n").unwrap();
        assert_eq!(
            LocalApiKey::from_sources(None, path()).unwrap_err(),
            LocalSecurityError::InvalidApiKey
        );
        std::fs::remove_dir_all(&dir).unwrap();
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

    fn guard_for(host: &str, cors_enabled: bool, cors_origins: &[&str]) -> LocalRequestGuard {
        guard_exposed(host, cors_enabled, cors_origins, &LocalExposure::default())
    }

    fn guard_exposed(
        host: &str,
        cors_enabled: bool,
        cors_origins: &[&str],
        exposure: &LocalExposure,
    ) -> LocalRequestGuard {
        LocalRequestGuard::new(
            &ServeRuntimeConfig {
                host: host.to_string(),
                cors_enabled,
                cors_origins: cors_origins
                    .iter()
                    .map(|origin| origin.to_string())
                    .collect(),
                ..ServeRuntimeConfig::default()
            },
            exposure,
        )
    }

    fn check(
        guard: &LocalRequestGuard,
        method: Method,
        headers: &[(header::HeaderName, &str)],
    ) -> Option<StatusCode> {
        let mut map = HeaderMap::new();
        for (name, value) in headers {
            map.insert(name.clone(), value.parse().unwrap());
        }
        guard.check(&method, &map, None).err().map(|err| err.status)
    }

    #[test]
    fn loopback_bind_rejects_foreign_host_headers() {
        let guard = guard_for("127.0.0.1", false, &[]);
        for host in [
            "localhost:8080",
            "127.0.0.1:8080",
            "[::1]:8080",
            "tauri.localhost",
            "LOCALHOST",
        ] {
            assert_eq!(
                check(&guard, Method::GET, &[(header::HOST, host)]),
                None,
                "{host}"
            );
        }
        assert_eq!(check(&guard, Method::GET, &[]), None);
        for host in [
            "attacker.example:8080",
            "attacker.example",
            "192.168.1.5:8080",
        ] {
            assert_eq!(
                check(&guard, Method::GET, &[(header::HOST, host)]),
                Some(StatusCode::FORBIDDEN),
                "{host}"
            );
        }
        // A network bind is reachable by any name; the API key guards it.
        let network = guard_for("0.0.0.0", false, &[]);
        assert_eq!(
            check(
                &network,
                Method::GET,
                &[(header::HOST, "my-mac.local:8080")]
            ),
            None
        );
    }

    #[test]
    fn loopback_published_container_keeps_the_loopback_host_check() {
        // A container binds 0.0.0.0 but publishes 127.0.0.1 only: DNS
        // rebinding must still fail, while compose service names can opt in.
        let guard = guard_exposed(
            "0.0.0.0",
            false,
            &[],
            &LocalExposure {
                loopback_published: true,
                allowed_hosts: vec!["izwi".to_string()],
            },
        );
        for host in [
            "localhost:8080",
            "127.0.0.1:8080",
            "[::1]:8080",
            "IZWI:8080",
        ] {
            assert_eq!(
                check(&guard, Method::GET, &[(header::HOST, host)]),
                None,
                "{host}"
            );
        }
        for host in ["attacker.example:8080", "192.168.1.5:8080", "izwi.example"] {
            assert_eq!(
                check(&guard, Method::GET, &[(header::HOST, host)]),
                Some(StatusCode::FORBIDDEN),
                "{host}"
            );
        }
        // The rebinding page's Origin matches its Host, so the Host check is
        // what stops it.
        assert_eq!(
            check(
                &guard,
                Method::POST,
                &[
                    (header::HOST, "attacker.example:8080"),
                    (header::ORIGIN, "http://attacker.example:8080"),
                ]
            ),
            Some(StatusCode::FORBIDDEN)
        );
    }

    #[test]
    fn allowed_hosts_turn_the_host_check_on_for_a_network_bind() {
        let guard = guard_exposed(
            "0.0.0.0",
            false,
            &[],
            &LocalExposure {
                loopback_published: false,
                allowed_hosts: vec!["izwi.example.com".to_string()],
            },
        );
        for host in ["izwi.example.com", "izwi.example.com:443", "127.0.0.1:8080"] {
            assert_eq!(
                check(&guard, Method::GET, &[(header::HOST, host)]),
                None,
                "{host}"
            );
        }
        assert_eq!(
            check(&guard, Method::GET, &[(header::HOST, "attacker.example")]),
            Some(StatusCode::FORBIDDEN)
        );
    }

    #[test]
    fn cross_site_writes_and_websockets_are_rejected_by_origin() {
        let guard = guard_for("0.0.0.0", false, &[]);
        let host = (header::HOST, "192.168.1.5:8080");
        for origin in [
            "http://192.168.1.5:8080",
            "tauri://localhost",
            "http://tauri.localhost",
            "http://localhost:3000",
            "http://127.0.0.1:5173",
        ] {
            assert_eq!(
                check(
                    &guard,
                    Method::POST,
                    &[host.clone(), (header::ORIGIN, origin)]
                ),
                None,
                "{origin}"
            );
        }
        assert_eq!(
            check(&guard, Method::POST, std::slice::from_ref(&host)),
            None
        );
        for origin in [
            "https://attacker.example",
            "null",
            "http://192.168.1.5:9999",
        ] {
            assert_eq!(
                check(
                    &guard,
                    Method::DELETE,
                    &[host.clone(), (header::ORIGIN, origin)]
                ),
                Some(StatusCode::FORBIDDEN),
                "{origin}"
            );
        }
        let evil = (header::ORIGIN, "https://attacker.example");
        assert_eq!(
            check(&guard, Method::GET, &[host.clone(), evil.clone()]),
            None
        );
        assert_eq!(
            check(
                &guard,
                Method::GET,
                &[host.clone(), evil.clone(), (header::UPGRADE, "websocket")]
            ),
            Some(StatusCode::FORBIDDEN)
        );

        let configured = guard_for("0.0.0.0", true, &["https://App.Example"]);
        assert_eq!(
            check(
                &configured,
                Method::POST,
                &[(header::ORIGIN, "https://app.example")]
            ),
            None
        );
        assert_eq!(
            check(&configured, Method::POST, std::slice::from_ref(&evil)),
            Some(StatusCode::FORBIDDEN)
        );
        let wildcard = guard_for("0.0.0.0", true, &["*"]);
        assert_eq!(check(&wildcard, Method::POST, &[evil]), None);
    }

    #[test]
    fn writes_with_form_content_types_are_rejected() {
        let guard = guard_for("127.0.0.1", false, &[]);
        for content_type in [
            "text/plain",
            "application/x-www-form-urlencoded; charset=utf-8",
        ] {
            assert_eq!(
                check(
                    &guard,
                    Method::POST,
                    &[(header::CONTENT_TYPE, content_type)]
                ),
                Some(StatusCode::UNSUPPORTED_MEDIA_TYPE),
                "{content_type}"
            );
        }
        for content_type in ["application/json", "multipart/form-data; boundary=x"] {
            assert_eq!(
                check(
                    &guard,
                    Method::POST,
                    &[(header::CONTENT_TYPE, content_type)]
                ),
                None,
                "{content_type}"
            );
        }
        assert_eq!(check(&guard, Method::POST, &[]), None);
        assert_eq!(
            check(&guard, Method::GET, &[(header::CONTENT_TYPE, "text/plain")]),
            None
        );
    }

    #[test]
    fn cookie_values_are_percent_decoded() {
        assert_eq!(percent_decode("a%3Bb%25c").as_deref(), Some("a;b%c"));
        assert_eq!(percent_decode("bad%2"), None);
        assert_eq!(percent_decode("bad%zz"), None);
    }
}
