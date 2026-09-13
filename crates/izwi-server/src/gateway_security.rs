//! Fail-closed authentication and ingress policy for the public gateway.
//!
//! Local/desktop serving does not use this module. Gateway credentials are
//! resolved from an environment reference so bearer secrets never become
//! command-line arguments or part of a derived `Debug` representation.

use axum::http::{header, HeaderMap};
use izwi_core::ServeRuntimeConfig;
use izwi_hooks::{HookMetadata, Principal};
use std::fmt;
use std::net::IpAddr;
use std::sync::Arc;

pub const DEFAULT_GATEWAY_MAX_CHAT_BODY_BYTES: usize = 1024 * 1024;
pub const MAX_GATEWAY_CHAT_BODY_BYTES: usize = 32 * 1024 * 1024;
pub const MAX_GATEWAY_REQUEST_ID_BYTES: usize = 128;

const MIN_GATEWAY_CHAT_BODY_BYTES: usize = 1024;
const MIN_API_KEY_BYTES: usize = 16;
const MAX_API_KEY_BYTES: usize = 4096;
const MAX_SECRET_REF_BYTES: usize = 128;
const MAX_IDENTITY_BYTES: usize = 128;
const DEFAULT_API_KEY_REF: &str = "env:IZWI_GATEWAY_API_KEY";
const DEFAULT_PRINCIPAL_ID: &str = "gateway-api-key";
const DEFAULT_TENANT_ID: &str = "default";

const API_KEY_REF_ENV: &str = "IZWI_GATEWAY_API_KEY_REF";
const METRICS_API_KEY_REF_ENV: &str = "IZWI_GATEWAY_METRICS_API_KEY_REF";
const PRINCIPAL_ID_ENV: &str = "IZWI_GATEWAY_API_PRINCIPAL_ID";
const TENANT_ID_ENV: &str = "IZWI_GATEWAY_TENANT_ID";
const MAX_CHAT_BODY_ENV: &str = "IZWI_GATEWAY_MAX_CHAT_BODY_BYTES";
const TRUSTED_INGRESS_TLS_ENV: &str = "IZWI_GATEWAY_TRUSTED_INGRESS_TLS";

#[derive(Clone)]
pub struct GatewayPerimeterConfig {
    api_key: Arc<[u8]>,
    api_key_ref: Arc<str>,
    metrics_api_key: Option<Arc<[u8]>>,
    metrics_api_key_ref: Option<Arc<str>>,
    principal: Principal,
    max_chat_body_bytes: usize,
    trusted_ingress_tls: bool,
}

impl fmt::Debug for GatewayPerimeterConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GatewayPerimeterConfig")
            .field("api_key", &"[REDACTED]")
            .field("api_key_ref", &self.api_key_ref)
            .field(
                "metrics_api_key",
                &self.metrics_api_key.as_ref().map(|_| "[REDACTED]"),
            )
            .field("metrics_api_key_ref", &self.metrics_api_key_ref)
            .field("principal", &"[REDACTED]")
            .field("max_chat_body_bytes", &self.max_chat_body_bytes)
            .field("trusted_ingress_tls", &self.trusted_ingress_tls)
            .finish()
    }
}

impl GatewayPerimeterConfig {
    /// Resolve the public bearer credential and ingress policy from bounded
    /// environment configuration. The secret reference currently supports
    /// only `env:VARIABLE`, keeping secret bytes out of argv and config files.
    pub fn from_env() -> Result<Self, GatewayPerimeterConfigError> {
        let secret_ref =
            std::env::var(API_KEY_REF_ENV).unwrap_or_else(|_| DEFAULT_API_KEY_REF.to_string());
        if secret_ref.is_empty() || secret_ref.len() > MAX_SECRET_REF_BYTES {
            return Err(GatewayPerimeterConfigError::InvalidSecretReference);
        }
        let variable = secret_ref
            .strip_prefix("env:")
            .filter(|name| valid_env_name(name))
            .ok_or(GatewayPerimeterConfigError::InvalidSecretReference)?;
        let api_key = std::env::var_os(variable)
            .ok_or(GatewayPerimeterConfigError::MissingSecret)?
            .into_string()
            .map_err(|_| GatewayPerimeterConfigError::InvalidSecret)?;
        let metrics_credential = optional_metrics_credential_from_env()?;
        let principal_id = bounded_identity_env(PRINCIPAL_ID_ENV, DEFAULT_PRINCIPAL_ID)?;
        let tenant_id = bounded_identity_env(TENANT_ID_ENV, DEFAULT_TENANT_ID)?;
        let max_chat_body_bytes = bounded_body_limit_from_env()?;
        let trusted_ingress_tls = bool_from_env(TRUSTED_INGRESS_TLS_ENV)?;
        Self::new(
            api_key,
            secret_ref,
            principal_id,
            tenant_id,
            max_chat_body_bytes,
            trusted_ingress_tls,
            metrics_credential,
        )
    }

    /// Construct an isolated perimeter for router and process-boundary tests.
    /// Production startup always uses [`Self::from_env`].
    #[doc(hidden)]
    pub fn new_for_test(
        api_key: impl Into<String>,
        max_chat_body_bytes: usize,
    ) -> Result<Self, GatewayPerimeterConfigError> {
        Self::new(
            api_key.into(),
            "test-only".to_string(),
            "test-gateway-principal".to_string(),
            "test-tenant".to_string(),
            max_chat_body_bytes,
            true,
            None,
        )
    }

    #[doc(hidden)]
    pub fn with_metrics_api_key_for_test(
        mut self,
        metrics_api_key: impl Into<String>,
    ) -> Result<Self, GatewayPerimeterConfigError> {
        let metrics_api_key = metrics_api_key.into();
        validate_secret(&metrics_api_key)
            .map_err(|_| GatewayPerimeterConfigError::InvalidMetricsSecret)?;
        if constant_time_eq(metrics_api_key.as_bytes(), self.api_key.as_ref()) {
            return Err(GatewayPerimeterConfigError::ReusedMetricsSecret);
        }
        self.metrics_api_key = Some(Arc::from(metrics_api_key.into_bytes()));
        self.metrics_api_key_ref = Some(Arc::from("test-only"));
        Ok(self)
    }

    fn new(
        api_key: String,
        api_key_ref: String,
        principal_id: String,
        tenant_id: String,
        max_chat_body_bytes: usize,
        trusted_ingress_tls: bool,
        metrics_credential: Option<(String, String)>,
    ) -> Result<Self, GatewayPerimeterConfigError> {
        validate_secret(&api_key)?;
        if metrics_credential
            .as_ref()
            .is_some_and(|(metrics_api_key, _)| {
                constant_time_eq(metrics_api_key.as_bytes(), api_key.as_bytes())
            })
        {
            return Err(GatewayPerimeterConfigError::ReusedMetricsSecret);
        }
        if !(MIN_GATEWAY_CHAT_BODY_BYTES..=MAX_GATEWAY_CHAT_BODY_BYTES)
            .contains(&max_chat_body_bytes)
        {
            return Err(GatewayPerimeterConfigError::InvalidBodyLimit);
        }
        let mut attributes = HookMetadata::new();
        attributes.insert("scope".to_string(), "inference".to_string());
        attributes.insert("credential_type".to_string(), "api_key".to_string());
        Ok(Self {
            api_key: Arc::from(api_key.into_bytes()),
            api_key_ref: Arc::from(api_key_ref),
            metrics_api_key: metrics_credential
                .as_ref()
                .map(|(secret, _)| Arc::from(secret.as_bytes())),
            metrics_api_key_ref: metrics_credential.map(|(_, secret_ref)| Arc::from(secret_ref)),
            principal: Principal {
                id: principal_id,
                display_name: None,
                tenant_id: Some(tenant_id),
                roles: vec!["inference".to_string()],
                attributes,
            },
            max_chat_body_bytes,
            trusted_ingress_tls,
        })
    }

    pub(crate) fn authenticate(&self, headers: &HeaderMap) -> Option<Principal> {
        if !authenticate_bearer(headers, self.api_key.as_ref()) {
            return None;
        }
        Some(self.principal.clone())
    }

    pub(crate) fn metrics_enabled(&self) -> bool {
        self.metrics_api_key.is_some()
    }

    pub(crate) fn authenticate_metrics(&self, headers: &HeaderMap) -> bool {
        self.metrics_api_key
            .as_deref()
            .is_some_and(|expected| authenticate_bearer(headers, expected))
    }

    pub fn max_chat_body_bytes(&self) -> usize {
        self.max_chat_body_bytes
    }

    pub fn validate_public_ingress(
        &self,
        serve_config: &ServeRuntimeConfig,
    ) -> Result<(), GatewayPerimeterConfigError> {
        if serve_config.cors_enabled
            && (serve_config.cors_origins.is_empty()
                || serve_config
                    .cors_origins
                    .iter()
                    .any(|origin| origin.trim() == "*"))
        {
            return Err(GatewayPerimeterConfigError::WildcardCors);
        }
        if serve_config.cors_enabled
            && serve_config
                .cors_origins
                .iter()
                .any(|origin| origin.parse::<axum::http::HeaderValue>().is_err())
        {
            return Err(GatewayPerimeterConfigError::InvalidCorsOrigin);
        }
        if !is_loopback_host(&serve_config.host) && !self.trusted_ingress_tls {
            return Err(GatewayPerimeterConfigError::UntrustedPlaintextBind);
        }
        Ok(())
    }
}

fn authenticate_bearer(headers: &HeaderMap, expected: &[u8]) -> bool {
    let mut authorization_values = headers.get_all(header::AUTHORIZATION).iter();
    let Some(value) = authorization_values
        .next()
        .and_then(|value| value.to_str().ok())
    else {
        return false;
    };
    if authorization_values.next().is_some() {
        return false;
    }
    if value.len() > MAX_API_KEY_BYTES.saturating_add(7) {
        return false;
    }
    let Some((scheme, supplied)) = value.split_once(' ') else {
        return false;
    };
    if !scheme.eq_ignore_ascii_case("bearer")
        || supplied.is_empty()
        || supplied.len() > MAX_API_KEY_BYTES
        || supplied.chars().any(char::is_whitespace)
        || !constant_time_eq(supplied.as_bytes(), expected)
    {
        return false;
    }
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum GatewayPerimeterConfigError {
    #[error("gateway API key reference must be a bounded env:VARIABLE reference")]
    InvalidSecretReference,
    #[error("gateway API key environment reference is not set")]
    MissingSecret,
    #[error("gateway API key does not satisfy the bounded credential policy")]
    InvalidSecret,
    #[error("gateway metrics API key reference must be a bounded env:VARIABLE reference")]
    InvalidMetricsSecretReference,
    #[error("gateway metrics API key environment reference is not set")]
    MissingMetricsSecret,
    #[error("gateway metrics API key does not satisfy the bounded credential policy")]
    InvalidMetricsSecret,
    #[error("gateway metrics API key must differ from the public inference key")]
    ReusedMetricsSecret,
    #[error("gateway principal or tenant identity is invalid")]
    InvalidIdentity,
    #[error("gateway chat body limit is outside the supported range")]
    InvalidBodyLimit,
    #[error("gateway trusted-ingress acknowledgement must be a boolean")]
    InvalidTrustedIngress,
    #[error("production gateway CORS requires an explicit origin allowlist")]
    WildcardCors,
    #[error("production gateway CORS contains an invalid origin")]
    InvalidCorsOrigin,
    #[error("non-loopback plaintext gateway binding requires IZWI_GATEWAY_TRUSTED_INGRESS_TLS=1")]
    UntrustedPlaintextBind,
}

fn optional_metrics_credential_from_env(
) -> Result<Option<(String, String)>, GatewayPerimeterConfigError> {
    let Some(secret_ref) = std::env::var_os(METRICS_API_KEY_REF_ENV) else {
        return Ok(None);
    };
    let secret_ref = secret_ref
        .into_string()
        .map_err(|_| GatewayPerimeterConfigError::InvalidMetricsSecretReference)?;
    if secret_ref.is_empty() || secret_ref.len() > MAX_SECRET_REF_BYTES {
        return Err(GatewayPerimeterConfigError::InvalidMetricsSecretReference);
    }
    let variable = secret_ref
        .strip_prefix("env:")
        .filter(|name| valid_env_name(name))
        .ok_or(GatewayPerimeterConfigError::InvalidMetricsSecretReference)?;
    let secret = std::env::var_os(variable)
        .ok_or(GatewayPerimeterConfigError::MissingMetricsSecret)?
        .into_string()
        .map_err(|_| GatewayPerimeterConfigError::InvalidMetricsSecret)?;
    validate_secret(&secret).map_err(|_| GatewayPerimeterConfigError::InvalidMetricsSecret)?;
    Ok(Some((secret, secret_ref)))
}

fn validate_secret(secret: &str) -> Result<(), GatewayPerimeterConfigError> {
    if !(MIN_API_KEY_BYTES..=MAX_API_KEY_BYTES).contains(&secret.len())
        || !secret.as_bytes().iter().all(|byte| byte.is_ascii_graphic())
    {
        return Err(GatewayPerimeterConfigError::InvalidSecret);
    }
    Ok(())
}

fn bounded_identity_env(name: &str, default: &str) -> Result<String, GatewayPerimeterConfigError> {
    let value = std::env::var(name).unwrap_or_else(|_| default.to_string());
    if value.is_empty()
        || value.len() > MAX_IDENTITY_BYTES
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
    {
        return Err(GatewayPerimeterConfigError::InvalidIdentity);
    }
    Ok(value)
}

fn bounded_body_limit_from_env() -> Result<usize, GatewayPerimeterConfigError> {
    match std::env::var(MAX_CHAT_BODY_ENV) {
        Ok(value) if value.len() <= 16 => value
            .parse::<usize>()
            .ok()
            .filter(|limit| {
                (MIN_GATEWAY_CHAT_BODY_BYTES..=MAX_GATEWAY_CHAT_BODY_BYTES).contains(limit)
            })
            .ok_or(GatewayPerimeterConfigError::InvalidBodyLimit),
        Ok(_) => Err(GatewayPerimeterConfigError::InvalidBodyLimit),
        Err(_) => Ok(DEFAULT_GATEWAY_MAX_CHAT_BODY_BYTES),
    }
}

fn bool_from_env(name: &str) -> Result<bool, GatewayPerimeterConfigError> {
    match std::env::var(name) {
        Ok(value) if matches!(value.as_str(), "1" | "true" | "TRUE") => Ok(true),
        Ok(value) if matches!(value.as_str(), "0" | "false" | "FALSE") => Ok(false),
        Ok(_) => Err(GatewayPerimeterConfigError::InvalidTrustedIngress),
        Err(_) => Ok(false),
    }
}

fn valid_env_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= MAX_SECRET_REF_BYTES.saturating_sub(4)
        && name.bytes().enumerate().all(|(index, byte)| {
            byte == b'_' || byte.is_ascii_alphabetic() || (index > 0 && byte.is_ascii_digit())
        })
}

fn is_loopback_host(host: &str) -> bool {
    let host = host.trim().trim_start_matches('[').trim_end_matches(']');
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<IpAddr>()
            .is_ok_and(|address| address.is_loopback())
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    let mut difference = left.len() ^ right.len();
    for index in 0..MAX_API_KEY_BYTES {
        difference |= usize::from(
            left.get(index).copied().unwrap_or_default()
                ^ right.get(index).copied().unwrap_or_default(),
        );
    }
    difference == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    const TEST_API_KEY: &str = "test-public-api-key-123456";
    const TEST_METRICS_KEY: &str = "test-metrics-api-key-654321";

    fn clear_gateway_security_env() {
        for name in [
            API_KEY_REF_ENV,
            METRICS_API_KEY_REF_ENV,
            "IZWI_TEST_GATEWAY_SECRET",
            "IZWI_TEST_GATEWAY_METRICS_SECRET",
            PRINCIPAL_ID_ENV,
            TENANT_ID_ENV,
            MAX_CHAT_BODY_ENV,
            TRUSTED_INGRESS_TLS_ENV,
        ] {
            std::env::remove_var(name);
        }
    }

    #[test]
    fn gateway_api_key_is_resolved_only_through_bounded_environment_reference() {
        let _guard = crate::test_support::env_lock();
        clear_gateway_security_env();
        std::env::set_var(API_KEY_REF_ENV, "env:IZWI_TEST_GATEWAY_SECRET");
        std::env::set_var("IZWI_TEST_GATEWAY_SECRET", TEST_API_KEY);
        std::env::set_var(PRINCIPAL_ID_ENV, "service-principal");
        std::env::set_var(TENANT_ID_ENV, "tenant-a");
        std::env::set_var(MAX_CHAT_BODY_ENV, "4096");
        let config = GatewayPerimeterConfig::from_env().unwrap();
        assert_eq!(config.max_chat_body_bytes(), 4096);
        let debug = format!("{config:?}");
        assert!(!debug.contains(TEST_API_KEY));
        assert!(!debug.contains("service-principal"));
        assert!(!debug.contains("tenant-a"));

        std::env::set_var(API_KEY_REF_ENV, TEST_API_KEY);
        assert_eq!(
            GatewayPerimeterConfig::from_env().unwrap_err(),
            GatewayPerimeterConfigError::InvalidSecretReference
        );
        clear_gateway_security_env();
    }

    #[test]
    fn gateway_api_key_debug_is_secret_safe() {
        let config =
            GatewayPerimeterConfig::new_for_test(TEST_API_KEY, DEFAULT_GATEWAY_MAX_CHAT_BODY_BYTES)
                .unwrap();
        let debug = format!("{config:?}");
        assert!(debug.contains("[REDACTED]"));
        assert!(!debug.contains(TEST_API_KEY));
    }

    #[test]
    fn gateway_metrics_key_is_optional_separate_and_redacted() {
        let _guard = crate::test_support::env_lock();
        clear_gateway_security_env();
        std::env::set_var(API_KEY_REF_ENV, "env:IZWI_TEST_GATEWAY_SECRET");
        std::env::set_var("IZWI_TEST_GATEWAY_SECRET", TEST_API_KEY);
        std::env::set_var(
            METRICS_API_KEY_REF_ENV,
            "env:IZWI_TEST_GATEWAY_METRICS_SECRET",
        );
        std::env::set_var("IZWI_TEST_GATEWAY_METRICS_SECRET", TEST_METRICS_KEY);
        let config = GatewayPerimeterConfig::from_env().unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer test-metrics-api-key-654321"),
        );
        assert!(config.metrics_enabled());
        assert!(config.authenticate_metrics(&headers));
        headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer test-public-api-key-123456"),
        );
        assert!(!config.authenticate_metrics(&headers));
        let debug = format!("{config:?}");
        assert!(debug.contains("[REDACTED]"));
        assert!(!debug.contains(TEST_API_KEY));
        assert!(!debug.contains(TEST_METRICS_KEY));

        std::env::set_var("IZWI_TEST_GATEWAY_METRICS_SECRET", TEST_API_KEY);
        assert_eq!(
            GatewayPerimeterConfig::from_env().unwrap_err(),
            GatewayPerimeterConfigError::ReusedMetricsSecret
        );
        clear_gateway_security_env();
    }

    #[test]
    fn gateway_metrics_key_absence_keeps_internal_scrape_disabled() {
        let config =
            GatewayPerimeterConfig::new_for_test(TEST_API_KEY, DEFAULT_GATEWAY_MAX_CHAT_BODY_BYTES)
                .unwrap();
        assert!(!config.metrics_enabled());
        assert!(!config.authenticate_metrics(&HeaderMap::new()));
    }

    #[test]
    fn gateway_api_key_rejects_values_that_cannot_be_an_authorization_token() {
        assert_eq!(
            GatewayPerimeterConfig::new_for_test(
                "test public api key",
                DEFAULT_GATEWAY_MAX_CHAT_BODY_BYTES,
            )
            .unwrap_err(),
            GatewayPerimeterConfigError::InvalidSecret
        );
    }

    #[test]
    fn gateway_bearer_authentication_is_exact_and_server_authored() {
        let config =
            GatewayPerimeterConfig::new_for_test(TEST_API_KEY, DEFAULT_GATEWAY_MAX_CHAT_BODY_BYTES)
                .unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer test-public-api-key-123456"),
        );
        headers.insert("x-tenant-id", HeaderValue::from_static("forged-tenant"));
        headers.insert(
            "x-principal-id",
            HeaderValue::from_static("forged-principal"),
        );
        headers.insert("x-scopes", HeaderValue::from_static("admin"));

        let principal = config.authenticate(&headers).expect("key should match");
        assert_eq!(principal.id, "test-gateway-principal");
        assert_eq!(principal.tenant_id.as_deref(), Some("test-tenant"));
        assert_eq!(principal.roles, vec!["inference"]);
        assert_eq!(
            principal.attributes.get("scope").map(String::as_str),
            Some("inference")
        );

        headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer test-public-api-key-123457"),
        );
        assert!(config.authenticate(&headers).is_none());
    }

    #[test]
    fn production_gateway_rejects_wildcard_cors_and_unacknowledged_public_http() {
        let strict = GatewayPerimeterConfig::new(
            TEST_API_KEY.to_string(),
            "test".to_string(),
            "principal".to_string(),
            "tenant".to_string(),
            DEFAULT_GATEWAY_MAX_CHAT_BODY_BYTES,
            false,
            None,
        )
        .unwrap();
        let defaults = ServeRuntimeConfig::default();
        let wildcard_cors = ServeRuntimeConfig {
            cors_enabled: true,
            cors_origins: vec!["*".to_string()],
            ..defaults.clone()
        };
        assert_eq!(
            strict.validate_public_ingress(&wildcard_cors),
            Err(GatewayPerimeterConfigError::WildcardCors)
        );

        let explicit_cors = ServeRuntimeConfig {
            cors_enabled: true,
            cors_origins: vec!["https://api.example.test".to_string()],
            ..defaults
        };
        assert_eq!(
            strict.validate_public_ingress(&explicit_cors),
            Err(GatewayPerimeterConfigError::UntrustedPlaintextBind)
        );

        let trusted_ingress = GatewayPerimeterConfig::new(
            TEST_API_KEY.to_string(),
            "test".to_string(),
            "principal".to_string(),
            "tenant".to_string(),
            DEFAULT_GATEWAY_MAX_CHAT_BODY_BYTES,
            true,
            None,
        )
        .unwrap();
        trusted_ingress
            .validate_public_ingress(&explicit_cors)
            .unwrap();

        let loopback = ServeRuntimeConfig {
            host: "127.0.0.1".to_string(),
            ..explicit_cors
        };
        strict.validate_public_ingress(&loopback).unwrap();
    }
}
