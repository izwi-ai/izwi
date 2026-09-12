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
use std::{collections::VecDeque, pin::Pin, sync::Arc, time::Duration};
use tokio::{sync::OwnedSemaphorePermit, time::Instant};

#[cfg(any(test, feature = "mock-worker"))]
pub mod mock;

pub const DEFAULT_MAX_REQUEST_JSON_BYTES: usize = 1024 * 1024;
pub const DEFAULT_MAX_CONTROL_BODY_BYTES: usize = 512 * 1024;
pub const DEFAULT_MAX_ERROR_BODY_BYTES: usize = 16 * 1024;

#[derive(Debug, Clone)]
pub struct WorkerClientConfig {
    pub max_in_flight: usize,
    pub connect_timeout: Duration,
    pub request_timeout: Duration,
    pub progress_timeout: Duration,
    pub max_request_json_bytes: usize,
    pub max_control_body_bytes: usize,
    pub max_error_body_bytes: usize,
    pub ndjson_limits: NdjsonLimits,
}

impl Default for WorkerClientConfig {
    fn default() -> Self {
        Self {
            max_in_flight: 32,
            connect_timeout: Duration::from_secs(2),
            request_timeout: Duration::from_secs(10),
            progress_timeout: Duration::from_secs(30),
            max_request_json_bytes: DEFAULT_MAX_REQUEST_JSON_BYTES,
            max_control_body_bytes: DEFAULT_MAX_CONTROL_BODY_BYTES,
            max_error_body_bytes: DEFAULT_MAX_ERROR_BODY_BYTES,
            ndjson_limits: NdjsonLimits::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeadlinePhase {
    InFlightPermit,
    ResponseHeaders,
    StreamProgress,
    TotalInvocation,
}

impl std::fmt::Display for DeadlinePhase {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::InFlightPermit => "in-flight permit",
            Self::ResponseHeaders => "response headers",
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
        if !endpoint.path().ends_with('/') {
            endpoint.set_path(&format!("{}/", endpoint.path()));
        }
        let http = reqwest::Client::builder()
            .redirect(Policy::none())
            .connect_timeout(config.connect_timeout)
            .build()
            .map_err(WorkerClientError::Build)?;
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

    pub async fn descriptor(&self) -> Result<WorkerDescriptor, WorkerClientError> {
        self.get_json(WORKER_DESCRIPTOR_PATH).await
    }

    pub async fn status(&self) -> Result<WorkerStatus, WorkerClientError> {
        self.get_json(WORKER_STATUS_PATH).await
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
        request: InvocationRequest,
    ) -> Result<InvocationStream, WorkerClientError> {
        request.validate()?;
        let encoded = serde_json::to_vec(&request)?;
        if encoded.len() > self.inner.config.max_request_json_bytes {
            return Err(WorkerClientError::RequestTooLarge {
                actual: encoded.len(),
                limit: self.inner.config.max_request_json_bytes,
            });
        }
        let permit = self.acquire_permit().await?;
        // From this point until an explicit rejection or a live response stream, dropping this
        // future leaves admission uncertain. A tombstoned cancel also closes the race where the
        // cancellation reaches the worker just before the invocation POST.
        let mut admission_guard =
            PendingInvocationGuard::new(self.clone(), AttemptIdentity::from(&request));
        let total_deadline = Instant::now() + Duration::from_millis(request.remaining_time_ms);
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
        .map_err(|_| WorkerClientError::Deadline(DeadlinePhase::ResponseHeaders))?
        .map_err(WorkerClientError::Transport)?;

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
            let remaining = self
                .total_deadline
                .saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(WorkerClientError::Deadline(DeadlinePhase::TotalInvocation));
            }
            let budget = remaining.min(self.client.inner.config.progress_timeout);
            let next = tokio::time::timeout(budget, self.body.next())
                .await
                .map_err(|_| {
                    if Instant::now() >= self.total_deadline {
                        WorkerClientError::Deadline(DeadlinePhase::TotalInvocation)
                    } else {
                        WorkerClientError::Deadline(DeadlinePhase::StreamProgress)
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
        if event.is_terminal() {
            self.terminal_seen = true;
        }
        Ok(())
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
