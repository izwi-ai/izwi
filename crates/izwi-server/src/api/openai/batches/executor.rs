//! Runs batches: validates a batch's input file, runs its lines, and writes
//! its output and error files.
//!
//! A batch is one durable record leased by the executor running it, not a
//! job per line: the first-party job kinds carry transcription and speech
//! history records a batch line has none of, and 50,000 lines would be
//! 50,000 job, stage and lease rows. Instead a finished line is one result
//! row, and lines still to run are read again from the input file, so
//! memory holds only the lines in flight.
//!
//! The lease is renewed while the batch runs. An executor that dies leaves
//! it to lapse, and the next executor to claim the batch skips the lines
//! with a result row and runs the rest. A cancel or the end of the 24 hour
//! window stops the lines in flight; what finished is written out.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use futures::future::BoxFuture;
use futures::stream::{FuturesUnordered, StreamExt};
use serde_json::{json, Map, Value};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, BufReader};
use tokio::sync::{watch, Notify, OwnedSemaphorePermit, Semaphore};
use tracing::{info, warn};

use super::BatchEndpoint;
use crate::artifact_store::ArtifactTenant;
use crate::batch_api_store::{
    BatchApiStore, BatchLineResult, BatchOutputKind, BatchRecord, BatchStatus,
};
use crate::file_store::{FilePurpose, FileSpool, FileStore};
use crate::state::AppState;

/// Batches one server runs at a time; the rest wait their turn, oldest
/// first.
const MAX_ACTIVE_BATCHES: usize = 4;
/// Validation stops at this many errors: enough to fix a file from.
const MAX_VALIDATION_ERRORS: usize = 100;
/// Finished lines read per query while writing an output file.
const RESULTS_PAGE: usize = 500;
/// Lines validated between lease renewals.
const VALIDATION_RENEW_EVERY: usize = 1000;

/// One line to run: its request, and whose batch it belongs to.
#[derive(Debug)]
pub(crate) struct LineJob {
    pub(crate) tenant: ArtifactTenant,
    pub(crate) endpoint: BatchEndpoint,
    pub(crate) custom_id: String,
    /// The line's `batch_req_…` id, also its correlation id in the engine.
    pub(crate) request_id: String,
    pub(crate) body: Value,
}

/// What a line's request answered: the synchronous route's status and JSON.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LineOutcome {
    pub(crate) status_code: u16,
    pub(crate) body: Value,
}

/// Runs one line. The server's runner calls the audio routes' own code;
/// tests put their own in its place.
pub(crate) type LineRunner =
    Arc<dyn Fn(Arc<LineJob>) -> BoxFuture<'static, LineOutcome> + Send + Sync>;

#[derive(Debug, Clone)]
pub(crate) struct ExecutorConfig {
    /// How long a batch stays claimed without a renewal.
    pub(crate) lease: Duration,
    /// How often an idle executor looks for a batch to run.
    pub(crate) poll: Duration,
    /// Lines in flight at once, across this server's batches.
    pub(crate) line_concurrency: usize,
    /// `IZWI_BATCH_MAX_REQUESTS`.
    pub(crate) max_requests: usize,
    pub(crate) max_line_bytes: usize,
    /// The wait before a line refused for capacity is run again.
    pub(crate) capacity_retry: Duration,
}

impl ExecutorConfig {
    fn for_server(state: &AppState) -> Self {
        let runtime = state.runtime.config();
        let audio_bytes = crate::api::openai::audio::resolve_audio_upload_limit_bytes();
        Self {
            lease: Duration::from_secs(30),
            poll: Duration::from_secs(5),
            line_concurrency: crate::roles::local::batch_worker_concurrency(
                runtime
                    .max_retained_sequences
                    .min(runtime.max_queued_requests),
                std::env::var("IZWI_BATCH_WORKER_CONCURRENCY")
                    .ok()
                    .as_deref(),
            ) as usize,
            max_requests: crate::env_config::tuning().batch_max_requests,
            // A line carries at most one upload of audio as base64, plus
            // its other fields.
            max_line_bytes: audio_bytes.saturating_mul(4) / 3 + 1024 * 1024,
            capacity_retry: Duration::from_secs(2),
        }
    }
}

pub(crate) struct BatchExecutor {
    batches: Arc<BatchApiStore>,
    files: Arc<FileStore>,
    runner: LineRunner,
    config: ExecutorConfig,
    /// This executor's name on the leases it holds.
    owner: String,
    wake: Arc<Notify>,
    shutdown: watch::Receiver<bool>,
    line_slots: Arc<Semaphore>,
}

/// Why a batch's lines stopped running.
enum RunEnd {
    /// Every line has a result.
    Finished,
    /// The batch was cancelled, ran out of time or changed hands, or the
    /// server is shutting down. The lines in flight were dropped.
    Stopped,
    /// The input file is gone.
    InputGone,
}

/// Start the server's batch executor; it runs until shutdown.
pub fn spawn(state: AppState) -> tokio::task::JoinHandle<()> {
    let executor = BatchExecutor::new(
        state.batch_api_store.clone(),
        state.file_store.clone(),
        super::runner::audio_line_runner(state.clone()),
        ExecutorConfig::for_server(&state),
        state.batch_api_wake.clone(),
        state.stream_shutdown.subscribe(),
    );
    tokio::spawn(executor.run())
}

impl BatchExecutor {
    pub(crate) fn new(
        batches: Arc<BatchApiStore>,
        files: Arc<FileStore>,
        runner: LineRunner,
        config: ExecutorConfig,
        wake: Arc<Notify>,
        shutdown: watch::Receiver<bool>,
    ) -> Arc<Self> {
        Arc::new(Self {
            batches,
            files,
            runner,
            line_slots: Arc::new(Semaphore::new(config.line_concurrency.max(1))),
            config,
            owner: format!("batch-executor-{}", crate::ids::new_uuid()),
            wake,
            shutdown,
        })
    }

    /// Whether the server is shutting down (or its state is gone).
    fn stopping(&self) -> bool {
        *self.shutdown.borrow() || self.shutdown.has_changed().is_err()
    }

    fn lease_ms(&self) -> i64 {
        i64::try_from(self.config.lease.as_millis()).unwrap_or(i64::MAX)
    }

    /// Claim batches and run them until shutdown. Dropping this future
    /// drops the batches it runs; their leases then lapse.
    pub(crate) async fn run(self: Arc<Self>) {
        let mut shutdown = self.shutdown.clone();
        let mut running = tokio::task::JoinSet::new();
        while !self.stopping() {
            // Registered before looking for work, so a wake-up sent while
            // looking is not lost.
            let wake = self.wake.notified();
            tokio::pin!(wake);
            wake.as_mut().enable();
            while running.len() < MAX_ACTIVE_BATCHES {
                match self.batches.claim_next(&self.owner, self.lease_ms()).await {
                    Ok(Some(id)) => {
                        let executor = self.clone();
                        running.spawn(async move { executor.drive(&id).await });
                    }
                    Ok(None) => break,
                    Err(err) => {
                        warn!(error = %format!("{err:#}"), "Failed to claim a batch");
                        break;
                    }
                }
            }
            tokio::select! {
                _ = &mut wake => {}
                _ = running.join_next(), if !running.is_empty() => {}
                _ = shutdown.changed() => {}
                _ = tokio::time::sleep(self.config.poll) => {}
            }
        }
        // Each batch sees the shutdown and gives its lease back.
        while running.join_next().await.is_some() {}
    }

    async fn drive(&self, id: &str) {
        if let Err(err) = self.advance(id).await {
            // The lease is left to lapse, which spaces out the retries.
            warn!(
                batch_id = id,
                error = %format!("{err:#}"),
                "A batch run failed; it resumes when its lease lapses"
            );
        }
    }

    /// Take the batch from its current status as far as it goes: to a
    /// terminal status, to shutdown, or to the loss of its lease.
    async fn advance(&self, id: &str) -> anyhow::Result<()> {
        loop {
            if self.stopping() {
                return self.batches.release_lease(id, &self.owner).await;
            }
            // Also where a finished batch leaves: its lease went with it.
            if !self
                .batches
                .renew_lease(id, &self.owner, self.lease_ms())
                .await?
            {
                return Ok(());
            }
            let Some(batch) = self.batches.load(id).await? else {
                return Ok(());
            };
            let tenant = ArtifactTenant::parse(batch.tenant_scope.clone())
                .map_err(|err| anyhow::anyhow!("Batch {id} has an invalid tenant: {err}"))?;
            let due = self.batches.now_millis() >= batch.expires_at;
            match batch.status {
                BatchStatus::Validating | BatchStatus::InProgress if due => {
                    self.finalize(&batch, &tenant, BatchStatus::Expired, None)
                        .await?;
                }
                BatchStatus::Validating => self.validate(&batch, &tenant).await?,
                BatchStatus::InProgress => match self.run_lines(&batch, &tenant).await? {
                    RunEnd::Finished => {
                        self.batches.begin_finalizing(id, &self.owner).await?;
                    }
                    RunEnd::Stopped => {}
                    RunEnd::InputGone => {
                        let errors = error_list(vec![input_gone(&batch)]);
                        self.finalize(&batch, &tenant, BatchStatus::Failed, Some(&errors))
                            .await?;
                    }
                },
                BatchStatus::Finalizing => {
                    self.finalize(&batch, &tenant, BatchStatus::Completed, None)
                        .await?;
                }
                BatchStatus::Cancelling => {
                    self.finalize(&batch, &tenant, BatchStatus::Cancelled, None)
                        .await?;
                }
                BatchStatus::Failed
                | BatchStatus::Completed
                | BatchStatus::Expired
                | BatchStatus::Cancelled => return Ok(()),
            }
        }
    }

    /// The batch's input file, line by line; `None` when it was deleted.
    async fn open_input(
        &self,
        batch: &BatchRecord,
        tenant: &ArtifactTenant,
    ) -> anyhow::Result<Option<LineReader>> {
        Ok(self
            .files
            .open(tenant, &batch.input_file_id)
            .await?
            .map(|(_, content)| LineReader::new(content.reader, self.config.max_line_bytes)))
    }

    /// Check every line of the input as OpenAI does, then start the batch
    /// or fail it with the errors found.
    async fn validate(&self, batch: &BatchRecord, tenant: &ArtifactTenant) -> anyhow::Result<()> {
        let endpoint = BatchEndpoint::parse(&batch.endpoint)
            .with_context(|| format!("Batch {} has an unknown endpoint", batch.id))?;
        let Some(mut lines) = self.open_input(batch, tenant).await? else {
            let errors = error_list(vec![input_gone(batch)]);
            self.batches
                .finish(
                    &batch.id,
                    &self.owner,
                    BatchStatus::Validating,
                    BatchStatus::Failed,
                    Some(&errors),
                )
                .await?;
            return Ok(());
        };
        let mut errors = Vec::new();
        let mut custom_ids = HashSet::new();
        let mut total = 0usize;
        while let Some((index, line)) = lines.next().await? {
            total += 1;
            if total > self.config.max_requests {
                errors.push(json!({
                    "code": "too_many_requests",
                    "message": format!(
                        "The input file has more than {} requests (IZWI_BATCH_MAX_REQUESTS).",
                        self.config.max_requests
                    ),
                    "param": null,
                    "line": null,
                }));
                break;
            }
            match parse_line(line.as_deref(), endpoint) {
                Ok(request) => {
                    if !custom_ids.insert(request.custom_id.clone()) {
                        errors.push(
                            LineError::new(
                                "duplicate_custom_id",
                                format!(
                                    "The custom_id `{}` is used by an earlier line.",
                                    request.custom_id
                                ),
                                Some("custom_id"),
                            )
                            .at(index),
                        );
                    }
                }
                Err(error) => errors.push(error.at(index)),
            }
            if errors.len() >= MAX_VALIDATION_ERRORS {
                break;
            }
            if total.is_multiple_of(VALIDATION_RENEW_EVERY)
                && !self
                    .batches
                    .renew_lease(&batch.id, &self.owner, self.lease_ms())
                    .await?
            {
                return Ok(());
            }
        }
        if total == 0 {
            errors.push(json!({
                "code": "empty_file",
                "message": "The input file has no requests.",
                "param": null,
                "line": null,
            }));
        }
        if errors.is_empty() {
            self.batches
                .start(&batch.id, &self.owner, i64::try_from(total)?)
                .await?;
        } else {
            info!(
                batch_id = %batch.id,
                errors = errors.len(),
                "A batch failed validation"
            );
            self.batches
                .finish(
                    &batch.id,
                    &self.owner,
                    BatchStatus::Validating,
                    BatchStatus::Failed,
                    Some(&error_list(errors)),
                )
                .await?;
        }
        Ok(())
    }

    /// Run the lines that have no result yet, a bounded number at a time,
    /// until all have one or the batch must stop.
    async fn run_lines(
        &self,
        batch: &BatchRecord,
        tenant: &ArtifactTenant,
    ) -> anyhow::Result<RunEnd> {
        let endpoint = BatchEndpoint::parse(&batch.endpoint)
            .with_context(|| format!("Batch {} has an unknown endpoint", batch.id))?;
        let finished = self.batches.finished_lines(&batch.id).await?;
        let Some(mut lines) = self.open_input(batch, tenant).await? else {
            return Ok(RunEnd::InputGone);
        };
        let mut shutdown = self.shutdown.clone();
        let mut heartbeat = tokio::time::interval(self.config.lease / 3);
        heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let wake = self.wake.notified();
        tokio::pin!(wake);
        let mut running = FuturesUnordered::new();
        let mut exhausted = false;
        loop {
            if exhausted && running.is_empty() {
                return Ok(RunEnd::Finished);
            }
            tokio::select! {
                // A slot is taken before the next line is read, so lines
                // wait in the file, not in memory.
                slot = self.line_slots.clone().acquire_owned(), if !exhausted => {
                    let slot = slot.context("The batch line limiter closed")?;
                    let mut next = None;
                    while let Some((index, line)) = lines.next().await? {
                        if !finished.contains(&index) {
                            next = Some((index, line));
                            break;
                        }
                    }
                    match next {
                        Some((index, line)) => {
                            running.push(self.run_line(tenant, endpoint, index, line, slot));
                        }
                        None => exhausted = true,
                    }
                }
                Some(result) = running.next(), if !running.is_empty() => {
                    self.batches.record_result(&batch.id, &result).await?;
                }
                // A cancel request wakes the executor; so does other news.
                _ = &mut wake => {
                    wake.set(self.wake.notified());
                    if self.must_stop(batch).await? {
                        return Ok(RunEnd::Stopped);
                    }
                }
                _ = heartbeat.tick() => {
                    if self.must_stop(batch).await? {
                        return Ok(RunEnd::Stopped);
                    }
                }
                _ = shutdown.changed() => return Ok(RunEnd::Stopped),
            }
        }
    }

    /// Renew the lease, and say whether the batch may no longer run lines:
    /// it changed hands, was cancelled, or its 24 hours are up.
    async fn must_stop(&self, batch: &BatchRecord) -> anyhow::Result<bool> {
        if !self
            .batches
            .renew_lease(&batch.id, &self.owner, self.lease_ms())
            .await?
        {
            return Ok(true);
        }
        Ok(self.batches.load(&batch.id).await?.is_none_or(|batch| {
            batch.status != BatchStatus::InProgress || self.batches.now_millis() >= batch.expires_at
        }))
    }

    /// Run one line to its result. A line the server had no capacity for
    /// (`429`, `503`) is not the request's failure: it waits and runs again
    /// until it is served or the batch stops.
    async fn run_line(
        &self,
        tenant: &ArtifactTenant,
        endpoint: BatchEndpoint,
        line_index: i64,
        line: Option<Vec<u8>>,
        _slot: OwnedSemaphorePermit,
    ) -> BatchLineResult {
        let request_id = crate::ids::new_prefixed_id("batch_req_");
        let request = match parse_line(line.as_deref(), endpoint) {
            Ok(request) => request,
            // Validation passed this line, so this is not expected.
            Err(error) => {
                return BatchLineResult {
                    line_index,
                    custom_id: String::new(),
                    request_id,
                    status_code: 400,
                    body: json!({"error": {
                        "message": error.message,
                        "type": "invalid_request_error",
                        "param": error.param,
                        "code": error.code,
                    }}),
                }
            }
        };
        drop(line);
        let job = Arc::new(LineJob {
            tenant: tenant.clone(),
            endpoint,
            custom_id: request.custom_id,
            request_id,
            body: Value::Object(request.body),
        });
        loop {
            let outcome = (self.runner)(job.clone()).await;
            if matches!(outcome.status_code, 429 | 503) {
                tokio::time::sleep(self.config.capacity_retry).await;
                continue;
            }
            return BatchLineResult {
                line_index,
                custom_id: job.custom_id.clone(),
                request_id: job.request_id.clone(),
                status_code: outcome.status_code,
                body: outcome.body,
            };
        }
    }

    /// Write the output and error files from the finished lines and end the
    /// batch as `status`. A step done before a crash is not done again: a
    /// file already recorded on the batch is kept.
    async fn finalize(
        &self,
        batch: &BatchRecord,
        tenant: &ArtifactTenant,
        status: BatchStatus,
        errors: Option<&Value>,
    ) -> anyhow::Result<()> {
        for (kind, written, suffix) in [
            (BatchOutputKind::Output, &batch.output_file_id, "output"),
            (BatchOutputKind::Errors, &batch.error_file_id, "error"),
        ] {
            if written.is_some() {
                continue;
            }
            let mut spool = FileSpool::new()?;
            let mut after = -1;
            loop {
                let page = self
                    .batches
                    .results_page(&batch.id, kind, after, RESULTS_PAGE)
                    .await?;
                let Some(last) = page.last() else {
                    break;
                };
                after = last.line_index;
                for result in &page {
                    let mut line = output_line(result).to_string();
                    line.push('\n');
                    spool.write(line.as_bytes()).await?;
                }
            }
            if spool.size_bytes() == 0 {
                continue;
            }
            let filename = format!("{}_{suffix}.jsonl", batch.id);
            let file = self
                .files
                .create(tenant, FilePurpose::BatchOutput, filename, spool)
                .await?;
            self.batches
                .set_output_file(&batch.id, kind, &file.id)
                .await?;
        }
        let ended = self
            .batches
            .finish(&batch.id, &self.owner, batch.status, status, errors)
            .await?;
        if ended {
            info!(batch_id = %batch.id, status = status.as_str(), "A batch ended");
            // A batch slot is free.
            self.wake.notify_waiters();
        }
        Ok(())
    }
}

/// A finished line as its output or error file says it, in OpenAI's shape.
/// A failed line carries the synchronous route's error body as its
/// response, and that error's code and message as `error`.
fn output_line(result: &BatchLineResult) -> Value {
    let error = result.failed().then(|| {
        let error = &result.body["error"];
        json!({
            "code": error.get("code").cloned().unwrap_or(Value::Null),
            "message": error.get("message").cloned().unwrap_or(Value::Null),
        })
    });
    json!({
        "id": result.request_id,
        "custom_id": result.custom_id,
        "response": {
            "status_code": result.status_code,
            "request_id": result.request_id,
            "body": result.body,
        },
        "error": error,
    })
}

fn error_list(errors: Vec<Value>) -> Value {
    json!({"object": "list", "data": errors})
}

fn input_gone(batch: &BatchRecord) -> Value {
    json!({
        "code": "input_file_not_found",
        "message": format!("The input file {} was deleted.", batch.input_file_id),
        "param": "input_file_id",
        "line": null,
    })
}

/// A line's request once it has passed the checks every line gets.
struct LineRequest {
    custom_id: String,
    body: Map<String, Value>,
}

/// Why a line is refused, as an entry of the batch's `errors`.
struct LineError {
    code: &'static str,
    message: String,
    param: Option<&'static str>,
}

impl LineError {
    fn new(code: &'static str, message: impl Into<String>, param: Option<&'static str>) -> Self {
        Self {
            code,
            message: message.into(),
            param,
        }
    }

    /// The error on the line at `index`, numbered from one as OpenAI does.
    fn at(self, index: i64) -> Value {
        json!({
            "code": self.code,
            "message": self.message,
            "param": self.param,
            "line": index + 1,
        })
    }
}

/// The checks every line gets: OpenAI's line shape, the batch's endpoint,
/// and no streaming. `None` is a line longer than the limit. What the body
/// asks for is checked when the line runs, by the route's own code.
fn parse_line(line: Option<&[u8]>, endpoint: BatchEndpoint) -> Result<LineRequest, LineError> {
    let Some(line) = line else {
        return Err(LineError::new(
            "line_too_long",
            "This line is longer than one request's audio allows.",
            None,
        ));
    };
    let Ok(Value::Object(mut line)) = serde_json::from_slice::<Value>(line) else {
        return Err(LineError::new(
            "invalid_json_line",
            "This line is not parseable as a JSON object.",
            None,
        ));
    };
    let custom_id = match line.remove("custom_id") {
        Some(Value::String(custom_id)) if !custom_id.trim().is_empty() => custom_id,
        _ => {
            return Err(LineError::new(
                "missing_custom_id",
                "`custom_id` is required: a string unique within the file.",
                Some("custom_id"),
            ))
        }
    };
    if line.get("method").and_then(Value::as_str) != Some("POST") {
        return Err(LineError::new(
            "invalid_method",
            "`method` must be `POST`.",
            Some("method"),
        ));
    }
    if line.get("url").and_then(Value::as_str) != Some(endpoint.path()) {
        return Err(LineError::new(
            "invalid_url",
            format!("`url` must be the batch's endpoint, `{}`.", endpoint.path()),
            Some("url"),
        ));
    }
    let Some(Value::Object(body)) = line.remove("body") else {
        return Err(LineError::new(
            "invalid_body",
            "`body` is required: the request's JSON body.",
            Some("body"),
        ));
    };
    if let Err(error) = crate::api::openai::audio::speech_batch::reject_streaming(&body, "body.") {
        return Err(LineError::new(
            "streaming_not_supported",
            error.message,
            Some("body"),
        ));
    }
    Ok(LineRequest { custom_id, body })
}

/// Reads an input file one line at a time, never holding more than a line.
pub(crate) struct LineReader {
    reader: BufReader<std::pin::Pin<Box<dyn AsyncRead + Send>>>,
    next_index: i64,
    max_line_bytes: usize,
}

impl LineReader {
    pub(crate) fn new(
        reader: std::pin::Pin<Box<dyn AsyncRead + Send>>,
        max_line_bytes: usize,
    ) -> Self {
        Self {
            reader: BufReader::with_capacity(64 * 1024, reader),
            next_index: 0,
            max_line_bytes,
        }
    }

    /// The next line that is not blank, with its index in the file. A line
    /// longer than the limit is skipped and returned as `None`.
    pub(crate) async fn next(&mut self) -> std::io::Result<Option<(i64, Option<Vec<u8>>)>> {
        loop {
            let mut line = Vec::new();
            let limit = self.max_line_bytes as u64 + 1;
            let read = (&mut self.reader)
                .take(limit)
                .read_until(b'\n', &mut line)
                .await?;
            if read == 0 {
                return Ok(None);
            }
            let index = self.next_index;
            self.next_index += 1;
            if line.last() != Some(&b'\n') && line.len() > self.max_line_bytes {
                self.skip_to_line_end().await?;
                return Ok(Some((index, None)));
            }
            if line.iter().all(u8::is_ascii_whitespace) {
                continue;
            }
            return Ok(Some((index, Some(line))));
        }
    }

    async fn skip_to_line_end(&mut self) -> std::io::Result<()> {
        loop {
            let buffered = self.reader.fill_buf().await?;
            if buffered.is_empty() {
                return Ok(());
            }
            match buffered.iter().position(|byte| *byte == b'\n') {
                Some(end) => {
                    self.reader.consume(end + 1);
                    return Ok(());
                }
                None => {
                    let length = buffered.len();
                    self.reader.consume(length);
                }
            }
        }
    }
}
