//! Accelerator-free HTTP transport between an Izwi gateway and inference workers.
//!
//! The client deliberately does not retry invocation POSTs. Once acceptance is uncertain, only
//! attempt query/cancellation can safely resolve ownership.

use bytes::Bytes;
use futures::{Stream, StreamExt};
use izwi_serving_protocol::{
    AttemptId, AttemptIdentity, AttemptQueryResponse, CancelAttemptRequest, CancelAttemptResponse,
    InvocationEvent, InvocationEventKind, InvocationRejection, InvocationRequest,
    NdjsonDecodeError, NdjsonDecoder, NdjsonLimits, ServiceCredentials, WorkerDescriptor,
    WorkerStatus, INVOCATIONS_PATH, NDJSON_MEDIA_TYPE, PROTOCOL_V1, SERVICE_AUTHORIZATION_HEADER,
    SERVICE_AUTH_SCHEME, SERVICE_CREDENTIAL_ID_HEADER, WORKER_DESCRIPTOR_PATH, WORKER_STATUS_PATH,
};
use reqwest::{redirect::Policy, StatusCode};
use serde::de::DeserializeOwned;
use std::{collections::VecDeque, fmt, io::Cursor, pin::Pin, sync::Arc, time::Duration};
use tokio::{sync::OwnedSemaphorePermit, time::Instant};

#[cfg(any(test, feature = "mock-worker"))]
pub mod mock;

pub const DEFAULT_MAX_REQUEST_JSON_BYTES: usize = 1024 * 1024;
pub const DEFAULT_MAX_CONTROL_BODY_BYTES: usize = 512 * 1024;
pub const DEFAULT_MAX_ERROR_BODY_BYTES: usize = 16 * 1024;
pub const MAX_PRIVATE_CA_ROOTS: usize = 16;
pub const MAX_TLS_PEM_BYTES: usize = 256 * 1024;
pub const MAX_TOTAL_PRIVATE_CA_BYTES: usize = 1024 * 1024;

/// Optional private trust roots and client identity for HTTPS worker links.
/// PEM bytes are deliberately omitted from `Debug` output.
#[derive(Clone, Default)]
pub struct WorkerClientTlsConfig {
    private_ca_roots_pem: Vec<Arc<[u8]>>,
    client_identity_pem: Option<Arc<[u8]>>,
}

impl fmt::Debug for WorkerClientTlsConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WorkerClientTlsConfig")
            .field("private_ca_root_count", &self.private_ca_roots_pem.len())
            .field(
                "client_identity",
                &self.client_identity_pem.as_ref().map(|_| "[REDACTED]"),
            )
            .finish()
    }
}

impl WorkerClientTlsConfig {
    /// Build bounded TLS material. A client certificate and private key must
    /// always be provided together.
    pub fn from_pem(
        private_ca_roots_pem: Vec<Vec<u8>>,
        client_certificate_pem: Option<Vec<u8>>,
        client_private_key_pem: Option<Vec<u8>>,
    ) -> Result<Self, WorkerClientError> {
        if private_ca_roots_pem.len() > MAX_PRIVATE_CA_ROOTS {
            return Err(WorkerClientError::InvalidConfiguration(
                "too many private CA roots",
            ));
        }
        let total_ca_bytes = private_ca_roots_pem
            .iter()
            .try_fold(0_usize, |total, pem| total.checked_add(pem.len()))
            .ok_or(WorkerClientError::InvalidConfiguration(
                "private CA roots exceed the total size limit",
            ))?;
        if total_ca_bytes > MAX_TOTAL_PRIVATE_CA_BYTES
            || private_ca_roots_pem
                .iter()
                .any(|pem| pem.is_empty() || pem.len() > MAX_TLS_PEM_BYTES)
        {
            return Err(WorkerClientError::InvalidConfiguration(
                "private CA roots exceed the bounded PEM policy",
            ));
        }
        for pem in &private_ca_roots_pem {
            let items = parse_pem_items(pem, "private CA root is not valid PEM")?;
            if items.is_empty()
                || items
                    .iter()
                    .any(|item| !matches!(item, rustls_pemfile::Item::X509Certificate(_)))
            {
                return Err(WorkerClientError::InvalidConfiguration(
                    "private CA root must contain only certificate PEM blocks",
                ));
            }
        }
        let client_identity_pem = match (client_certificate_pem, client_private_key_pem) {
            (None, None) => None,
            (Some(_), None) | (None, Some(_)) => {
                return Err(WorkerClientError::InvalidConfiguration(
                    "mTLS requires both a client certificate and private key",
                ));
            }
            (Some(certificate), Some(private_key)) => {
                if certificate.is_empty()
                    || private_key.is_empty()
                    || certificate.len() > MAX_TLS_PEM_BYTES
                    || private_key.len() > MAX_TLS_PEM_BYTES
                {
                    return Err(WorkerClientError::InvalidConfiguration(
                        "mTLS identity exceeds the bounded PEM policy",
                    ));
                }
                let certificate_items =
                    parse_pem_items(&certificate, "mTLS client certificate is not valid PEM")?;
                if certificate_items.is_empty()
                    || certificate_items
                        .iter()
                        .any(|item| !matches!(item, rustls_pemfile::Item::X509Certificate(_)))
                {
                    return Err(WorkerClientError::InvalidConfiguration(
                        "mTLS client certificate must contain only certificate PEM blocks",
                    ));
                }
                let private_key_items =
                    parse_pem_items(&private_key, "mTLS client private key is not valid PEM")?;
                if private_key_items.len() != 1
                    || !private_key_items.iter().all(|item| {
                        matches!(
                            item,
                            rustls_pemfile::Item::RSAKey(_)
                                | rustls_pemfile::Item::PKCS8Key(_)
                                | rustls_pemfile::Item::ECKey(_)
                        )
                    })
                {
                    return Err(WorkerClientError::InvalidConfiguration(
                        "mTLS client private key must contain exactly one supported key PEM block",
                    ));
                }
                let capacity = certificate
                    .len()
                    .checked_add(private_key.len())
                    .and_then(|size| size.checked_add(1))
                    .ok_or(WorkerClientError::InvalidConfiguration(
                        "mTLS identity exceeds the bounded PEM policy",
                    ))?;
                let mut identity = Vec::with_capacity(capacity);
                identity.extend_from_slice(&certificate);
                if !certificate.ends_with(b"\n") {
                    identity.push(b'\n');
                }
                identity.extend_from_slice(&private_key);
                Some(Arc::from(identity))
            }
        };
        Ok(Self {
            private_ca_roots_pem: private_ca_roots_pem.into_iter().map(Arc::from).collect(),
            client_identity_pem,
        })
    }

    pub fn is_configured(&self) -> bool {
        !self.private_ca_roots_pem.is_empty() || self.client_identity_pem.is_some()
    }

    pub fn has_client_identity(&self) -> bool {
        self.client_identity_pem.is_some()
    }
}

fn parse_pem_items(
    pem: &[u8],
    error: &'static str,
) -> Result<Vec<rustls_pemfile::Item>, WorkerClientError> {
    rustls_pemfile::read_all(&mut Cursor::new(pem))
        .map_err(|_| WorkerClientError::InvalidConfiguration(error))
}

#[derive(Debug, Clone)]
pub struct WorkerClientConfig {
    pub max_in_flight: usize,
    pub connect_timeout: Duration,
    /// Maximum wait for response headers and, separately once they arrive, the
    /// first contract-valid admission event.
    pub request_timeout: Duration,
    /// Maximum wait from worker acceptance to the first non-empty model output
    /// or a legitimate terminal event.
    pub first_output_timeout: Duration,
    /// Maximum idle time between non-empty model output events. Raw HTTP
    /// chunks, empty deltas, and usage-only events do not reset it.
    pub progress_timeout: Duration,
    pub max_request_json_bytes: usize,
    pub max_control_body_bytes: usize,
    pub max_error_body_bytes: usize,
    pub ndjson_limits: NdjsonLimits,
    pub tls: WorkerClientTlsConfig,
}

impl Default for WorkerClientConfig {
    fn default() -> Self {
        Self {
            max_in_flight: 32,
            connect_timeout: Duration::from_secs(2),
            request_timeout: Duration::from_secs(10),
            first_output_timeout: Duration::from_secs(60),
            progress_timeout: Duration::from_secs(30),
            max_request_json_bytes: DEFAULT_MAX_REQUEST_JSON_BYTES,
            max_control_body_bytes: DEFAULT_MAX_CONTROL_BODY_BYTES,
            max_error_body_bytes: DEFAULT_MAX_ERROR_BODY_BYTES,
            ndjson_limits: NdjsonLimits::default(),
            tls: WorkerClientTlsConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeadlinePhase {
    InFlightPermit,
    ResponseHeaders,
    InvocationAdmission,
    FirstOutput,
    StreamProgress,
    TotalInvocation,
}

impl std::fmt::Display for DeadlinePhase {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::InFlightPermit => "in-flight permit",
            Self::ResponseHeaders => "response headers",
            Self::InvocationAdmission => "invocation admission",
            Self::FirstOutput => "first output",
            Self::StreamProgress => "stream progress",
            Self::TotalInvocation => "total invocation",
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum WorkerClientError {
    #[error("invalid worker client configuration: {0}")]
    InvalidConfiguration(&'static str),
    #[error("invalid worker endpoint: {0}")]
    InvalidEndpoint(String),
    #[error("failed to construct HTTP client: {0}")]
    Build(#[source] reqwest::Error),
    #[error("invocation contract is invalid: {0}")]
    InvalidInvocation(#[from] izwi_serving_protocol::ContractValidationError),
    #[error("failed to encode request: {0}")]
    Encode(#[from] serde_json::Error),
    #[error("request JSON is {actual} bytes; maximum is {limit}")]
    RequestTooLarge { actual: usize, limit: usize },
    #[error("worker operation exceeded its {0} deadline")]
    Deadline(DeadlinePhase),
    #[error("worker connection was not established before the invocation could be sent: {0}")]
    ConnectionNotEstablished(#[source] reqwest::Error),
    #[error("worker transport failed: {0}")]
    Transport(#[source] reqwest::Error),
    #[error("worker returned HTTP {status}: {body}")]
    HttpStatus { status: StatusCode, body: String },
    #[error("worker rejected invocation with {rejection:?}")]
    Rejected { rejection: InvocationRejection },
    #[error("worker response body exceeded {limit} bytes")]
    ResponseTooLarge { limit: usize },
    #[error("invalid worker JSON response: {0}")]
    InvalidJson(#[source] serde_json::Error),
    #[error("invalid invocation event stream: {0}")]
    Ndjson(#[from] NdjsonDecodeError),
    #[error("invocation stream violated the protocol: {0}")]
    Protocol(String),
    #[error("invocation stream ended without a terminal event; execution state is unknown")]
    InterruptedUnknown,
}

impl WorkerClientError {
    /// True only when the invocation could not have reached authoritative
    /// worker admission. Every other failure must be reconciled against the
    /// exact attempt before external capacity ownership can be released.
    pub const fn proves_attempt_unaccepted(&self) -> bool {
        match self {
            Self::InvalidInvocation(_)
            | Self::Encode(_)
            | Self::RequestTooLarge { .. }
            | Self::Deadline(DeadlinePhase::InFlightPermit)
            | Self::ConnectionNotEstablished(_) => true,
            Self::Rejected { rejection } => !rejection.accepted,
            _ => false,
        }
    }
}

#[derive(Clone)]
pub struct WorkerClient {
    inner: Arc<WorkerClientInner>,
}

struct WorkerClientInner {
    http: reqwest::Client,
    endpoint: reqwest::Url,
    credentials: ServiceCredentials,
    permits: Arc<tokio::sync::Semaphore>,
    config: WorkerClientConfig,
}

impl std::fmt::Debug for WorkerClient {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WorkerClient")
            .field("endpoint", &self.inner.endpoint)
            .field("credentials", &self.inner.credentials)
            .field("config", &self.inner.config)
            .finish_non_exhaustive()
    }
}

impl WorkerClient {
    pub fn new(
        endpoint: &str,
        credentials: ServiceCredentials,
        config: WorkerClientConfig,
    ) -> Result<Self, WorkerClientError> {
        validate_config(&config)?;
        let mut endpoint = reqwest::Url::parse(endpoint)
            .map_err(|error| WorkerClientError::InvalidEndpoint(error.to_string()))?;
        if endpoint.cannot_be_a_base() || endpoint.host_str().is_none() {
            return Err(WorkerClientError::InvalidConfiguration(
                "endpoint must be an absolute hierarchical URL",
            ));
        }
        let numeric_loopback = endpoint
            .host_str()
            .map(|host| host.trim_start_matches('[').trim_end_matches(']'))
            .and_then(|host| host.parse::<std::net::IpAddr>().ok())
            .is_some_and(|host| host.is_loopback());
        if endpoint.scheme() != "https" && !(endpoint.scheme() == "http" && numeric_loopback) {
            return Err(WorkerClientError::InvalidConfiguration(
                "worker endpoints must use certificate-verified HTTPS or numeric loopback HTTP",
            ));
        }
        if !endpoint.username().is_empty()
            || endpoint.password().is_some()
            || endpoint.query().is_some()
            || endpoint.fragment().is_some()
        {
            return Err(WorkerClientError::InvalidConfiguration(
                "worker endpoints must not contain user info, query parameters, or fragments",
            ));
        }
        if !endpoint.path().ends_with('/') {
            endpoint.set_path(&format!("{}/", endpoint.path()));
        }
        if endpoint.scheme() != "https" && config.tls.is_configured() {
            return Err(WorkerClientError::InvalidConfiguration(
                "custom TLS trust or mTLS identity requires an HTTPS worker endpoint",
            ));
        }
        // Additional roots augment reqwest/rustls defaults. Certificate and
        // hostname verification remain enabled; redirects remain forbidden.
        let mut http_builder = reqwest::Client::builder()
            .redirect(Policy::none())
            .connect_timeout(config.connect_timeout);
        for pem in &config.tls.private_ca_roots_pem {
            let certificate = reqwest::Certificate::from_pem(pem).map_err(|_| {
                WorkerClientError::InvalidConfiguration("private CA root is not valid PEM")
            })?;
            http_builder = http_builder.add_root_certificate(certificate);
        }
        if let Some(pem) = &config.tls.client_identity_pem {
            let identity = reqwest::Identity::from_pem(pem).map_err(|_| {
                WorkerClientError::InvalidConfiguration("mTLS identity is not valid PEM")
            })?;
            http_builder = http_builder.identity(identity);
        }
        let tls_configured = config.tls.is_configured();
        let http = http_builder.build().map_err(|error| {
            if tls_configured {
                WorkerClientError::InvalidConfiguration(
                    "TLS client configuration could not be constructed",
                )
            } else {
                WorkerClientError::Build(error)
            }
        })?;
        Ok(Self {
            inner: Arc::new(WorkerClientInner {
                http,
                endpoint,
                credentials,
                permits: Arc::new(tokio::sync::Semaphore::new(config.max_in_flight)),
                config,
            }),
        })
    }

    pub fn uses_https(&self) -> bool {
        self.inner.endpoint.scheme() == "https"
    }

    pub fn uses_numeric_loopback_http(&self) -> bool {
        self.inner.endpoint.scheme() == "http"
            && self
                .inner
                .endpoint
                .host_str()
                .map(|host| host.trim_start_matches('[').trim_end_matches(']'))
                .and_then(|host| host.parse::<std::net::IpAddr>().ok())
                .is_some_and(|host| host.is_loopback())
    }

    pub async fn descriptor(&self) -> Result<WorkerDescriptor, WorkerClientError> {
        let descriptor: WorkerDescriptor = self.get_json(WORKER_DESCRIPTOR_PATH).await?;
        if descriptor.schema_version.major != PROTOCOL_V1.major
            || !descriptor
                .supported_protocol_versions
                .iter()
                .any(|version| version.major == PROTOCOL_V1.major)
        {
            return Err(WorkerClientError::Protocol(
                "worker does not advertise a compatible protocol major version".into(),
            ));
        }
        Ok(descriptor)
    }

    pub async fn status(&self) -> Result<WorkerStatus, WorkerClientError> {
        let status: WorkerStatus = self.get_json(WORKER_STATUS_PATH).await?;
        if status.schema_version.major != PROTOCOL_V1.major {
            return Err(WorkerClientError::Protocol(
                "worker status uses an incompatible protocol major version".into(),
            ));
        }
        Ok(status)
    }

    pub async fn query_attempt(
        &self,
        identity: &AttemptIdentity,
    ) -> Result<AttemptQueryResponse, WorkerClientError> {
        let response = self
            .get_json(&format!("{INVOCATIONS_PATH}/{}", identity.attempt_id))
            .await?;
        validate_query_response(response, identity)
    }

    pub async fn cancel_attempt(
        &self,
        identity: &AttemptIdentity,
    ) -> Result<CancelAttemptResponse, WorkerClientError> {
        let encoded = serde_json::to_vec(&CancelAttemptRequest {
            schema_version: PROTOCOL_V1,
            identity: identity.clone(),
        })?;
        if encoded.len() > self.inner.config.max_request_json_bytes {
            return Err(WorkerClientError::RequestTooLarge {
                actual: encoded.len(),
                limit: self.inner.config.max_request_json_bytes,
            });
        }
        let permit = self.acquire_permit().await?;
        let response = tokio::time::timeout(
            self.inner.config.request_timeout,
            self.authorized(self.inner.http.post(self.url(&format!(
                "{INVOCATIONS_PATH}/{}/cancel",
                identity.attempt_id
            ))))
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(encoded)
            .send(),
        )
        .await
        .map_err(|_| WorkerClientError::Deadline(DeadlinePhase::ResponseHeaders))?
        .map_err(WorkerClientError::Transport)?;
        let result: Result<CancelAttemptResponse, _> = self
            .decode_json_response(response, self.inner.config.max_control_body_bytes)
            .await;
        drop(permit);
        validate_cancel_response(result?, identity)
    }

    /// Starts exactly one invocation POST. This method never retries.
    pub async fn invoke(
        &self,
        mut request: InvocationRequest,
    ) -> Result<InvocationStream, WorkerClientError> {
        request.validate()?;
        let total_deadline = Instant::now() + Duration::from_millis(request.remaining_time_ms);
        let encoded = serde_json::to_vec(&request)?;
        if encoded.len() > self.inner.config.max_request_json_bytes {
            return Err(WorkerClientError::RequestTooLarge {
                actual: encoded.len(),
                limit: self.inner.config.max_request_json_bytes,
            });
        }
        let permit = self.acquire_invocation_permit(total_deadline).await?;
        let remaining_time_ms = u64::try_from(
            total_deadline
                .saturating_duration_since(Instant::now())
                .as_millis(),
        )
        .unwrap_or(u64::MAX);
        if remaining_time_ms == 0 {
            return Err(WorkerClientError::Deadline(DeadlinePhase::InFlightPermit));
        }
        // Queue and total budgets are transport metadata, not logical request content, and are
        // deliberately excluded from the caller-supplied request digest. Preserve that digest
        // while forwarding only the budget that remains after local client admission.
        request.remaining_time_ms = remaining_time_ms;
        request.max_queue_wait_ms = request.max_queue_wait_ms.min(remaining_time_ms);
        request.validate()?;
        let encoded = serde_json::to_vec(&request)?;
        if encoded.len() > self.inner.config.max_request_json_bytes {
            return Err(WorkerClientError::RequestTooLarge {
                actual: encoded.len(),
                limit: self.inner.config.max_request_json_bytes,
            });
        }
        // From this point until an explicit rejection or a live response stream, dropping this
        // future leaves admission uncertain. A tombstoned cancel also closes the race where the
        // cancellation reaches the worker just before the invocation POST.
        let mut admission_guard =
            PendingInvocationGuard::new(self.clone(), AttemptIdentity::from(&request));
        let header_budget = self
            .inner
            .config
            .request_timeout
            .min(total_deadline.saturating_duration_since(Instant::now()));
        let response = tokio::time::timeout(
            header_budget,
            self.authorized(self.inner.http.post(self.url(INVOCATIONS_PATH)))
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .body(encoded)
                .send(),
        )
        .await
        .map_err(|_| WorkerClientError::Deadline(DeadlinePhase::ResponseHeaders))?;
        let response = match response {
            Ok(response) => response,
            Err(error) if error.is_connect() => {
                // Reqwest reached no HTTP peer, so this attempt cannot have been admitted. All
                // later failures remain acceptance-unknown and keep the cancellation guard armed.
                admission_guard.disarm();
                return Err(WorkerClientError::ConnectionNotEstablished(error));
            }
            Err(error) => return Err(WorkerClientError::Transport(error)),
        };

        if !response.status().is_success() {
            let result = self.decode_invocation_rejection(response, &request).await;
            if matches!(&result, WorkerClientError::Rejected { .. }) {
                admission_guard.disarm();
            }
            drop(permit);
            return Err(result);
        }
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default();
        if !content_type
            .split(';')
            .next()
            .is_some_and(|value| value.trim().eq_ignore_ascii_case(NDJSON_MEDIA_TYPE))
        {
            return Err(WorkerClientError::Protocol(format!(
                "expected {NDJSON_MEDIA_TYPE}, received {content_type:?}"
            )));
        }

        let now = Instant::now();
        let phase_deadline = now
            .checked_add(self.inner.config.request_timeout)
            .unwrap_or(total_deadline)
            .min(total_deadline);
        let mut stream = InvocationStream {
            client: self.clone(),
            body: Box::pin(response.bytes_stream()),
            decoder: NdjsonDecoder::new(self.inner.config.ndjson_limits)?,
            pending: VecDeque::new(),
            identity: AttemptIdentity::from(&request),
            last_sequence: None,
            terminal_seen: false,
            eof_seen: false,
            permit: Some(permit),
            total_deadline,
            phase_deadline,
            deadline_phase: DeadlinePhase::InvocationAdmission,
        };
        // The live stream now owns best-effort cancellation for every exit path.
        admission_guard.disarm();
        stream.fill_pending().await?;
        let first = stream.pending.front().cloned().ok_or_else(|| {
            stream.schedule_cancel();
            WorkerClientError::InterruptedUnknown
        })?;
        stream.validate_first_accepted(&first)?;
        Ok(stream)
    }

    pub async fn invoke_collect(
        &self,
        request: InvocationRequest,
    ) -> Result<Vec<InvocationEvent>, WorkerClientError> {
        let mut stream = self.invoke(request).await?;
        let mut events = Vec::new();
        while let Some(event) = stream.next_event().await? {
            events.push(event);
        }
        Ok(events)
    }

    async fn get_json<T: DeserializeOwned>(&self, path: &str) -> Result<T, WorkerClientError> {
        let permit = self.acquire_permit().await?;
        let response = tokio::time::timeout(
            self.inner.config.request_timeout,
            self.authorized(self.inner.http.get(self.url(path))).send(),
        )
        .await
        .map_err(|_| WorkerClientError::Deadline(DeadlinePhase::ResponseHeaders))?
        .map_err(WorkerClientError::Transport)?;
        let result = self
            .decode_json_response(response, self.inner.config.max_control_body_bytes)
            .await;
        drop(permit);
        result
    }

    async fn acquire_permit(&self) -> Result<OwnedSemaphorePermit, WorkerClientError> {
        tokio::time::timeout(
            self.inner.config.request_timeout,
            Arc::clone(&self.inner.permits).acquire_owned(),
        )
        .await
        .map_err(|_| WorkerClientError::Deadline(DeadlinePhase::InFlightPermit))?
        .map_err(|_| WorkerClientError::InvalidConfiguration("client semaphore is closed"))
    }

    async fn acquire_invocation_permit(
        &self,
        total_deadline: Instant,
    ) -> Result<OwnedSemaphorePermit, WorkerClientError> {
        let remaining = total_deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(WorkerClientError::Deadline(DeadlinePhase::InFlightPermit));
        }
        tokio::time::timeout(
            self.inner.config.request_timeout.min(remaining),
            Arc::clone(&self.inner.permits).acquire_owned(),
        )
        .await
        .map_err(|_| WorkerClientError::Deadline(DeadlinePhase::InFlightPermit))?
        .map_err(|_| WorkerClientError::InvalidConfiguration("client semaphore is closed"))
    }

    fn authorized(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        request
            .header(
                SERVICE_AUTHORIZATION_HEADER,
                format!(
                    "{SERVICE_AUTH_SCHEME} {}",
                    self.inner.credentials.bearer_token.expose_secret()
                ),
            )
            .header(
                SERVICE_CREDENTIAL_ID_HEADER,
                self.inner.credentials.credential_id.as_str(),
            )
    }

    fn url(&self, path: &str) -> reqwest::Url {
        self.inner
            .endpoint
            .join(path.trim_start_matches('/'))
            .expect("validated base URL and constant relative path")
    }

    async fn decode_json_response<T: DeserializeOwned>(
        &self,
        response: reqwest::Response,
        limit: usize,
    ) -> Result<T, WorkerClientError> {
        let status = response.status();
        let body_limit = if status.is_success() {
            limit
        } else {
            self.inner.config.max_error_body_bytes
        };
        let bytes =
            collect_bounded(response, body_limit, self.inner.config.progress_timeout).await?;
        if !status.is_success() {
            return Err(WorkerClientError::HttpStatus {
                status,
                body: lossy_bounded(&bytes),
            });
        }
        serde_json::from_slice(&bytes).map_err(WorkerClientError::InvalidJson)
    }

    async fn decode_invocation_rejection(
        &self,
        response: reqwest::Response,
        request: &InvocationRequest,
    ) -> WorkerClientError {
        let status = response.status();
        match collect_bounded(
            response,
            self.inner.config.max_error_body_bytes,
            self.inner.config.progress_timeout,
        )
        .await
        {
            Ok(bytes) => match serde_json::from_slice::<InvocationRejection>(&bytes) {
                Ok(rejection)
                    if rejection.is_valid()
                        && rejection.schema_version.major == PROTOCOL_V1.major
                        && rejection.request_id == request.request_id
                        && rejection.attempt_id == request.attempt_id =>
                {
                    WorkerClientError::Rejected { rejection }
                }
                _ => WorkerClientError::HttpStatus {
                    status,
                    body: lossy_bounded(&bytes),
                },
            },
            Err(error) => error,
        }
    }
}

struct PendingInvocationGuard {
    client: WorkerClient,
    identity: AttemptIdentity,
    armed: bool,
}

impl PendingInvocationGuard {
    fn new(client: WorkerClient, identity: AttemptIdentity) -> Self {
        Self {
            client,
            identity,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for PendingInvocationGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        spawn_cancel(self.client.clone(), self.identity.clone());
    }
}

fn validate_config(config: &WorkerClientConfig) -> Result<(), WorkerClientError> {
    if config.max_in_flight == 0 {
        return Err(WorkerClientError::InvalidConfiguration(
            "max_in_flight must be non-zero",
        ));
    }
    if config.connect_timeout.is_zero()
        || config.request_timeout.is_zero()
        || config.first_output_timeout.is_zero()
        || config.progress_timeout.is_zero()
    {
        return Err(WorkerClientError::InvalidConfiguration(
            "deadlines must be non-zero",
        ));
    }
    if config.max_request_json_bytes == 0
        || config.max_control_body_bytes == 0
        || config.max_error_body_bytes == 0
    {
        return Err(WorkerClientError::InvalidConfiguration(
            "body limits must be non-zero",
        ));
    }
    NdjsonDecoder::<InvocationEvent>::new(config.ndjson_limits)?;
    Ok(())
}

async fn collect_bounded(
    response: reqwest::Response,
    limit: usize,
    progress_timeout: Duration,
) -> Result<Vec<u8>, WorkerClientError> {
    let mut output = Vec::with_capacity(limit.min(8192));
    let mut stream = response.bytes_stream();
    loop {
        let next = tokio::time::timeout(progress_timeout, stream.next())
            .await
            .map_err(|_| WorkerClientError::Deadline(DeadlinePhase::StreamProgress))?;
        match next {
            Some(Ok(chunk)) => {
                if output.len().saturating_add(chunk.len()) > limit {
                    return Err(WorkerClientError::ResponseTooLarge { limit });
                }
                output.extend_from_slice(&chunk);
            }
            Some(Err(error)) => return Err(WorkerClientError::Transport(error)),
            None => return Ok(output),
        }
    }
}

fn lossy_bounded(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

fn validate_query_response(
    response: AttemptQueryResponse,
    expected: &AttemptIdentity,
) -> Result<AttemptQueryResponse, WorkerClientError> {
    if response.schema_version.major != PROTOCOL_V1.major || response.identity != *expected {
        return Err(WorkerClientError::Protocol(
            "attempt query response owner identity did not match".into(),
        ));
    }
    Ok(response)
}

fn validate_cancel_response(
    response: CancelAttemptResponse,
    expected: &AttemptIdentity,
) -> Result<CancelAttemptResponse, WorkerClientError> {
    if response.schema_version.major != PROTOCOL_V1.major || response.identity != *expected {
        return Err(WorkerClientError::Protocol(
            "attempt cancellation response owner identity did not match".into(),
        ));
    }
    Ok(response)
}

type ExpectedInvocationIdentity = AttemptIdentity;

type ResponseByteStream =
    Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send + Sync + 'static>>;

pub struct InvocationStream {
    client: WorkerClient,
    body: ResponseByteStream,
    decoder: NdjsonDecoder<InvocationEvent>,
    pending: VecDeque<InvocationEvent>,
    identity: ExpectedInvocationIdentity,
    last_sequence: Option<u64>,
    terminal_seen: bool,
    eof_seen: bool,
    permit: Option<OwnedSemaphorePermit>,
    total_deadline: Instant,
    phase_deadline: Instant,
    deadline_phase: DeadlinePhase,
}

impl std::fmt::Debug for InvocationStream {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("InvocationStream")
            .field("identity", &self.identity)
            .field("last_sequence", &self.last_sequence)
            .field("terminal_seen", &self.terminal_seen)
            .field("eof_seen", &self.eof_seen)
            .finish_non_exhaustive()
    }
}

impl InvocationStream {
    pub async fn next_event(&mut self) -> Result<Option<InvocationEvent>, WorkerClientError> {
        if self.eof_seen && self.pending.is_empty() {
            return Ok(None);
        }
        if self.pending.is_empty() {
            if let Err(error) = self.fill_pending().await {
                self.schedule_cancel();
                return Err(error);
            }
        }
        let Some(event) = self.pending.pop_front() else {
            if !self.terminal_seen {
                self.schedule_cancel();
                return Err(WorkerClientError::InterruptedUnknown);
            }
            self.permit.take();
            return Ok(None);
        };
        if event.is_terminal() {
            if let Err(error) = self.drain_after_terminal().await {
                self.schedule_cancel();
                return Err(error);
            }
            self.permit.take();
        }
        Ok(Some(event))
    }

    pub fn attempt_id(&self) -> &AttemptId {
        &self.identity.attempt_id
    }

    pub fn attempt_identity(&self) -> &AttemptIdentity {
        &self.identity
    }

    fn validate_first_accepted(&self, event: &InvocationEvent) -> Result<(), WorkerClientError> {
        match &event.event {
            InvocationEventKind::Accepted {
                incarnation_id,
                deployment_id,
                model_generation,
                ..
            } if incarnation_id == &self.identity.incarnation_id
                && deployment_id == &self.identity.deployment_id
                && model_generation == &self.identity.model_generation =>
            {
                Ok(())
            }
            InvocationEventKind::Accepted { .. } => Err(WorkerClientError::Protocol(
                "accepted event owner/deployment identity did not match the request".into(),
            )),
            _ => Err(WorkerClientError::Protocol(
                "the first invocation event was not accepted".into(),
            )),
        }
    }

    async fn fill_pending(&mut self) -> Result<(), WorkerClientError> {
        while self.pending.is_empty() && !self.eof_seen {
            let now = Instant::now();
            let total_remaining = self.total_deadline.saturating_duration_since(now);
            if total_remaining.is_zero() {
                return Err(WorkerClientError::Deadline(DeadlinePhase::TotalInvocation));
            }
            let phase_remaining = self.phase_deadline.saturating_duration_since(now);
            if phase_remaining.is_zero() {
                return Err(WorkerClientError::Deadline(self.deadline_phase));
            }
            // This deadline is deliberately fixed between complete useful
            // events. Repeated body fragments cannot keep an invocation alive
            // without accepted model output.
            let budget = total_remaining.min(phase_remaining);
            let next = tokio::time::timeout(budget, self.body.next())
                .await
                .map_err(|_| {
                    if Instant::now() >= self.total_deadline {
                        WorkerClientError::Deadline(DeadlinePhase::TotalInvocation)
                    } else {
                        WorkerClientError::Deadline(self.deadline_phase)
                    }
                })?;
            match next {
                Some(Ok(chunk)) => {
                    let events = self.decoder.push(&chunk)?;
                    for event in events {
                        self.validate_event(&event)?;
                        self.pending.push_back(event);
                    }
                }
                Some(Err(error)) => return Err(WorkerClientError::Transport(error)),
                None => {
                    for event in self.decoder.finish()? {
                        self.validate_event(&event)?;
                        self.pending.push_back(event);
                    }
                    self.eof_seen = true;
                }
            }
        }
        Ok(())
    }

    async fn drain_after_terminal(&mut self) -> Result<(), WorkerClientError> {
        if !self.pending.is_empty() {
            return Err(WorkerClientError::Protocol(
                "event appeared after the terminal event".into(),
            ));
        }
        while !self.eof_seen {
            self.fill_pending().await?;
            if !self.pending.is_empty() {
                return Err(WorkerClientError::Protocol(
                    "event appeared after the terminal event".into(),
                ));
            }
        }
        Ok(())
    }

    fn validate_event(&mut self, event: &InvocationEvent) -> Result<(), WorkerClientError> {
        if event.schema_version.major != PROTOCOL_V1.major
            || event.request_id != self.identity.request_id
            || event.attempt_id != self.identity.attempt_id
        {
            return Err(WorkerClientError::Protocol(
                "event version or request identity did not match".into(),
            ));
        }
        let expected_sequence = self
            .last_sequence
            .map_or(0, |value| value.saturating_add(1));
        if event.sequence != expected_sequence {
            return Err(WorkerClientError::Protocol(format!(
                "expected event sequence {expected_sequence}, received {}",
                event.sequence
            )));
        }
        if self.last_sequence.is_none()
            && !matches!(event.event, InvocationEventKind::Accepted { .. })
        {
            return Err(WorkerClientError::Protocol(
                "the first invocation event was not accepted".into(),
            ));
        }
        if self.last_sequence.is_some()
            && matches!(event.event, InvocationEventKind::Accepted { .. })
        {
            return Err(WorkerClientError::Protocol(
                "accepted event appeared more than once".into(),
            ));
        }
        if self.terminal_seen {
            return Err(WorkerClientError::Protocol(
                "event appeared after the terminal event".into(),
            ));
        }
        self.last_sequence = Some(event.sequence);
        match &event.event {
            InvocationEventKind::Accepted { .. } => {
                self.reset_phase_deadline(
                    DeadlinePhase::FirstOutput,
                    self.client.inner.config.first_output_timeout,
                );
            }
            InvocationEventKind::TextDelta { text } if !text.is_empty() => {
                self.reset_phase_deadline(
                    DeadlinePhase::StreamProgress,
                    self.client.inner.config.progress_timeout,
                );
            }
            InvocationEventKind::Completed { .. }
            | InvocationEventKind::Error { .. }
            | InvocationEventKind::Cancelled { .. } => {
                // Give protocol EOF a bounded drain window after a legitimate
                // terminal event. This does not change the total deadline.
                self.reset_phase_deadline(
                    DeadlinePhase::StreamProgress,
                    self.client.inner.config.progress_timeout,
                );
            }
            InvocationEventKind::TextDelta { .. } | InvocationEventKind::Usage { .. } => {}
        }
        if event.is_terminal() {
            self.terminal_seen = true;
        }
        Ok(())
    }

    fn reset_phase_deadline(&mut self, phase: DeadlinePhase, timeout: Duration) {
        let now = Instant::now();
        self.deadline_phase = phase;
        self.phase_deadline = now
            .checked_add(timeout)
            .unwrap_or(self.total_deadline)
            .min(self.total_deadline);
    }

    fn schedule_cancel(&mut self) {
        if self.terminal_seen {
            return;
        }
        self.permit.take();
        spawn_cancel(self.client.clone(), self.identity.clone());
    }
}

fn spawn_cancel(client: WorkerClient, identity: AttemptIdentity) {
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        handle.spawn(async move {
            let _ = client.cancel_attempt(&identity).await;
        });
    }
}

impl Drop for InvocationStream {
    fn drop(&mut self) {
        self.schedule_cancel();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use izwi_serving_protocol::{CredentialId, RejectionCode, RequestId, ServiceBearerToken};

    const TEST_CA_PEM: &str = r#"-----BEGIN CERTIFICATE-----
MIIDFjCCAf6gAwIBAgITN3i31kcwBKhGVxjuERUWxATTNTANBgkqhkiG9w0BAQsF
ADAbMRkwFwYDVQQDDBBpendpLXdvcmtlci10ZXN0MB4XDTI2MDkxMzAxMjQzNVoX
DTI2MDkxNDAxMjQzNVowGzEZMBcGA1UEAwwQaXp3aS13b3JrZXItdGVzdDCCASIw
DQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAK7WXUQXz+AiQlFw/O317NnGjz2O
pe0PehytgYUBrejeOg1R9uAHFGhPITKrTufgzQPCCnTAm2i3081sXGUY4AHfs2EU
8SxW/NVdVpTvCUA+ZM1fiyqx8YLskRxA+OWp5GcLvUfPnYadv0wSpziQ7FGYmZT9
l4al3NTKH/80ArPioAHkzh8nUVcaz4YjV9TJF076PZTBeTrlaTD2DMwSVKX2+wJf
qsiUKI/02FsSDvZnhf7pJsrTYP/kjehjBA2WGKGVGRzMeamyc3mdZqj0YDLKrnaN
v3nOMHM1NsQUCOzS1gn+GmRsGg9arOcSIf/N+RqEUPVMFEOITwAqCbSwuOMCAwEA
AaNTMFEwHQYDVR0OBBYEFEJCsPVccB709Z9mqgcNZyr8n1L3MB8GA1UdIwQYMBaA
FEJCsPVccB709Z9mqgcNZyr8n1L3MA8GA1UdEwEB/wQFMAMBAf8wDQYJKoZIhvcN
AQELBQADggEBAFsnPLVGo06kqcvVjfcv/7QbxdjtDVJ8hrffhpebUXJbXu/3i+Of
eKsjYANUc6Tqwff5N1dN6DwYpEf2ICuXyj9Mm10HqUefUyfeotPdUG2xJbKoyT3+
1blm4veHhd61dgQ3TsCSmxv1rYA8HFEhZfxqV9Mkma4Tf2B6OKnyS1KMmBpwvzre
IiAYl1UuUQ4fSp7V8uSs7/8RdLTgOHcQ+Zmcs9sYkkhNHRWN9Ib0hMEOV7xhgAD1
hVWwzOxiHGfjKpGYB7L1cCo1NhVeAsmVbb5SX9IOxDWgvPhvlvzKHRpyRBWeB9rF
V90f5fOFccvW2990PS3ow99ME6x1AUPnVeo=
-----END CERTIFICATE-----"#;
    const TEST_KEY_PEM: &str = r#"-----BEGIN PRIVATE KEY-----
MAECAQ==
-----END PRIVATE KEY-----"#;

    fn credentials() -> ServiceCredentials {
        ServiceCredentials {
            credential_id: CredentialId::new("worker-client-test").expect("static credential ID"),
            bearer_token: ServiceBearerToken::new("worker-client-secret")
                .expect("static bearer token"),
        }
    }

    #[test]
    fn only_pre_admission_failures_prove_an_attempt_was_unaccepted() {
        let rejection = InvocationRejection::new(
            RequestId::new("request-1").expect("static request ID"),
            AttemptId::new("attempt-1").expect("static attempt ID"),
            RejectionCode::CapacityExhausted,
            "capacity exhausted",
        );
        assert!(WorkerClientError::Rejected { rejection }.proves_attempt_unaccepted());
        assert!(
            WorkerClientError::Deadline(DeadlinePhase::InFlightPermit).proves_attempt_unaccepted()
        );

        let mut accepted_rejection = InvocationRejection::new(
            RequestId::new("request-2").expect("static request ID"),
            AttemptId::new("attempt-2").expect("static attempt ID"),
            RejectionCode::CapacityExhausted,
            "invalid accepted rejection fixture",
        );
        accepted_rejection.accepted = true;
        assert!(!WorkerClientError::Rejected {
            rejection: accepted_rejection,
        }
        .proves_attempt_unaccepted());
        assert!(!WorkerClientError::Deadline(DeadlinePhase::ResponseHeaders)
            .proves_attempt_unaccepted());
        assert!(!WorkerClientError::InterruptedUnknown.proves_attempt_unaccepted());
    }

    #[test]
    fn worker_client_allows_verified_https_and_restricts_plaintext_to_numeric_loopback() {
        let config = WorkerClientConfig::default();
        let loopback =
            WorkerClient::new("http://127.0.0.1:9470", credentials(), config.clone()).unwrap();
        assert!(loopback.uses_numeric_loopback_http());
        assert!(!loopback.uses_https());
        assert!(WorkerClient::new("http://[::1]:9470", credentials(), config.clone()).is_ok());
        let https = WorkerClient::new(
            "https://worker.example.test:9470/private/",
            credentials(),
            config.clone(),
        )
        .unwrap();
        assert!(https.uses_https());
        assert!(!https.uses_numeric_loopback_http());
        assert!(WorkerClient::new("https://192.0.2.1:9470", credentials(), config.clone()).is_ok());

        for endpoint in [
            "http://localhost:9470",
            "http://192.0.2.1:9470",
            "http://user:password@127.0.0.1:9470",
            "http://127.0.0.1:9470?token=secret",
            "http://127.0.0.1:9470#fragment",
        ] {
            assert!(matches!(
                WorkerClient::new(endpoint, credentials(), config.clone()),
                Err(WorkerClientError::InvalidConfiguration(_))
            ));
        }
    }

    #[test]
    fn tls_configuration_is_bounded_and_redacted() {
        assert!(!WorkerClientTlsConfig::default().has_client_identity());
        let tls = WorkerClientTlsConfig::from_pem(
            vec![TEST_CA_PEM.as_bytes().to_vec()],
            Some(TEST_CA_PEM.as_bytes().to_vec()),
            Some(TEST_KEY_PEM.as_bytes().to_vec()),
        )
        .expect("bounded material");
        assert!(tls.has_client_identity());
        let debug = format!("{tls:?}");
        assert!(debug.contains("private_ca_root_count: 1"));
        assert!(debug.contains("[REDACTED]"));
        assert!(!debug.contains("BEGIN CERTIFICATE"));
        assert!(!debug.contains("BEGIN PRIVATE KEY"));

        assert!(matches!(
            WorkerClientTlsConfig::from_pem(
                Vec::new(),
                Some(TEST_CA_PEM.as_bytes().to_vec()),
                None,
            ),
            Err(WorkerClientError::InvalidConfiguration(_))
        ));
        assert!(matches!(
            WorkerClientTlsConfig::from_pem(vec![vec![b'x'; MAX_TLS_PEM_BYTES + 1]], None, None,),
            Err(WorkerClientError::InvalidConfiguration(_))
        ));
        assert!(matches!(
            WorkerClientTlsConfig::from_pem(
                vec![b"x".to_vec(); MAX_PRIVATE_CA_ROOTS + 1],
                None,
                None,
            ),
            Err(WorkerClientError::InvalidConfiguration(_))
        ));
    }

    #[test]
    fn worker_client_rejects_malformed_tls_material_without_exposing_it() {
        const SECRET_MARKER: &str = "malformed-private-ca-secret-marker";
        let error =
            WorkerClientTlsConfig::from_pem(vec![SECRET_MARKER.as_bytes().to_vec()], None, None)
                .expect_err("malformed CA root must fail TLS configuration");
        assert!(matches!(&error, WorkerClientError::InvalidConfiguration(_)));
        assert!(!error.to_string().contains(SECRET_MARKER));

        let tls =
            WorkerClientTlsConfig::from_pem(vec![TEST_CA_PEM.as_bytes().to_vec()], None, None)
                .expect("valid test CA");
        let error = WorkerClient::new(
            "http://127.0.0.1:9470",
            credentials(),
            WorkerClientConfig {
                tls,
                ..WorkerClientConfig::default()
            },
        )
        .expect_err("TLS material must not be silently ignored for plaintext development");
        assert!(matches!(&error, WorkerClientError::InvalidConfiguration(_)));

        let tls = WorkerClientTlsConfig::from_pem(
            Vec::new(),
            Some(TEST_CA_PEM.as_bytes().to_vec()),
            Some(TEST_KEY_PEM.as_bytes().to_vec()),
        )
        .expect("bounded identity material");
        let error = WorkerClient::new(
            "https://worker.example.test:9470",
            credentials(),
            WorkerClientConfig {
                tls,
                ..WorkerClientConfig::default()
            },
        )
        .expect_err("malformed client identity must fail construction");
        assert!(matches!(&error, WorkerClientError::InvalidConfiguration(_)));
        assert!(!error.to_string().contains("BEGIN CERTIFICATE"));
        assert!(!error.to_string().contains("BEGIN PRIVATE KEY"));
    }

    #[test]
    fn worker_client_constructs_with_an_additional_private_ca_root() {
        let tls =
            WorkerClientTlsConfig::from_pem(vec![TEST_CA_PEM.as_bytes().to_vec()], None, None)
                .expect("bounded CA root");
        let config = WorkerClientConfig {
            tls,
            ..WorkerClientConfig::default()
        };
        assert!(
            WorkerClient::new("https://worker.example.test:9470", credentials(), config,).is_ok()
        );
    }
}
