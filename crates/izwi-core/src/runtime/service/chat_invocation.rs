//! Chat invocations: the admitted chat stream handed to the HTTP edge, its
//! events and teardown, and the driver that turns engine output into deltas.

use super::*;

/// Fully owned input for one worker-facing chat invocation.
///
/// The runtime applies coordinator admission, model lifecycle fencing and
/// exact Engine session admission before returning a [`RuntimeChatInvocation`].
#[derive(Debug, Clone)]
pub struct RuntimeChatInvocationRequest {
    pub variant: ModelVariant,
    pub messages: Vec<ChatMessage>,
    pub params: GenerationParams,
    pub chat_config: ChatRequestConfig,
    pub correlation_id: Option<String>,
    pub runtime_context: RuntimeRequestContext,
    /// Require native model streaming and expose committed text deltas.
    pub streaming: bool,
    /// A cap on the output budget that is not part of the request, such as
    /// a gateway's per-deployment ceiling. The budget is resolved from
    /// `params.max_tokens` first, so an explicit budget that does not fit
    /// the context is still rejected (D8); `None` leaves it uncapped.
    pub max_tokens_ceiling: Option<usize>,
}

/// Ordered output emitted by an admitted chat invocation.
#[derive(Debug, Clone)]
pub enum RuntimeChatInvocationEvent {
    /// DS9.3: a text delta with its per-token logprob entries (empty unless
    /// the request asked for logprobs).
    TextDelta {
        text: String,
        logprobs: Vec<crate::engine::TokenLogprob>,
    },
    Completed(ChatGeneration),
}

/// Why an invocation reached confirmed teardown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeChatTeardownDisposition {
    Completed,
    Cancelled,
    Failed,
}

/// Proof that the invocation no longer owns coordinator or model residency.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeChatTeardown {
    pub request_id: String,
    pub disposition: RuntimeChatTeardownDisposition,
}

pub(super) const RUNTIME_CHAT_INVOCATION_EVENT_CAPACITY: usize = 16;

/// Coalesced text a lagging consumer may owe before the invocation fails.
pub(super) const RUNTIME_CHAT_MAX_PENDING_DELTA_BYTES: usize = 1024 * 1024;

/// How long output may wait on a stalled consumer before the invocation
/// fails; `IZWI_STREAM_SLOW_CONSUMER_TIMEOUT_MS` overrides it.
pub const DEFAULT_STREAM_SLOW_CONSUMER_TIMEOUT: Duration = Duration::from_secs(30);

pub(super) const STREAM_SLOW_CONSUMER_TIMEOUT_ENV: &str = "IZWI_STREAM_SLOW_CONSUMER_TIMEOUT_MS";

/// The configured slow-consumer timeout for streamed text output, shared by
/// the runtime driver and the serving worker's NDJSON relay. Read once per
/// process.
pub fn stream_slow_consumer_timeout() -> Duration {
    static TIMEOUT: crate::env::ReadOnce<Duration> = crate::env::ReadOnce::new(|| {
        std::env::var(STREAM_SLOW_CONSUMER_TIMEOUT_ENV)
            .ok()
            .and_then(|raw| raw.trim().parse::<u64>().ok())
            .filter(|millis| *millis > 0)
            .map(Duration::from_millis)
            .unwrap_or(DEFAULT_STREAM_SLOW_CONSUMER_TIMEOUT)
    });
    TIMEOUT.get()
}

/// Opaque transport-facing handle for an authoritatively admitted chat request.
///
/// A successful constructor return is the acceptance boundary: the request
/// already owns coordinator capacity, an exact model generation and an exact
/// Engine session. The background driver, not this handle, owns those leases.
/// Dropping the handle requests cancellation, while the driver retains all
/// capacity until normal completion or exact-session cleanup is confirmed.
pub struct RuntimeChatInvocation {
    pub(super) request_id: String,
    pub(super) cancellation: Arc<AtomicBool>,
    pub(super) cancellation_wakeup: Arc<Notify>,
    pub(super) events: mpsc::Receiver<Result<RuntimeChatInvocationEvent>>,
    pub(super) teardown: Option<oneshot::Receiver<RuntimeChatTeardown>>,
    pub(super) cancel_on_drop: bool,
}

impl std::fmt::Debug for RuntimeChatInvocation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RuntimeChatInvocation")
            .field("request_id", &self.request_id)
            .field(
                "cancellation_requested",
                &self.cancellation.load(Ordering::Acquire),
            )
            .finish_non_exhaustive()
    }
}

impl RuntimeChatInvocation {
    pub(super) fn from_failed_post_admission(
        request_id: String,
        cancellation: Arc<AtomicBool>,
        mut guard: PendingRequestGuard,
        error: Error,
    ) -> Self {
        cancellation.store(true, Ordering::Release);
        let (event_sender, events) = mpsc::channel(1);
        let _ = event_sender.try_send(Err(error));
        drop(event_sender);
        let (teardown_sender, teardown) = oneshot::channel();
        let cleanup_request_id = request_id.clone();
        tokio::spawn(async move {
            if guard.confirm_cleanup().await.is_ok() {
                let _ = teardown_sender.send(RuntimeChatTeardown {
                    request_id: cleanup_request_id,
                    disposition: RuntimeChatTeardownDisposition::Failed,
                });
            }
            // A failed cleanup leaves the guard armed. Its Drop path retries
            // exact-session cleanup while retaining admission fail closed; the
            // closed teardown channel tells the worker it has no proof yet.
        });
        Self {
            request_id,
            cancellation,
            cancellation_wakeup: Arc::new(Notify::new()),
            events,
            teardown: Some(teardown),
            cancel_on_drop: false,
        }
    }

    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    /// Cooperatively request cancellation without waiting for an Engine lock.
    ///
    /// `true` means this call installed the signal. It is not teardown proof;
    /// callers that need that proof must await [`Self::wait_for_teardown`].
    pub fn request_cancel(&self) -> bool {
        let newly_requested = self
            .cancellation
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok();
        self.cancellation_wakeup.notify_one();
        newly_requested
    }

    /// Receive the next committed text delta or terminal generation.
    pub async fn next_event(&mut self) -> Result<Option<RuntimeChatInvocationEvent>> {
        match self.events.recv().await {
            Some(event) => event.map(Some),
            None => Ok(None),
        }
    }

    /// Stop consuming output and wait for confirmed physical teardown.
    ///
    /// Closing the event receiver is treated like a disconnected transport and
    /// therefore requests cancellation if execution has not already finished.
    pub async fn wait_for_teardown(mut self) -> Result<RuntimeChatTeardown> {
        self.events.close();
        self.request_cancel();
        self.cancel_on_drop = false;
        let teardown = self.teardown.take().ok_or_else(|| {
            Error::InferenceError(format!(
                "chat invocation {} lost its teardown waiter",
                self.request_id
            ))
        })?;
        teardown.await.map_err(|_| {
            Error::InferenceError(format!(
                "chat invocation {} ended without teardown confirmation",
                self.request_id
            ))
        })
    }
}

impl Drop for RuntimeChatInvocation {
    fn drop(&mut self) {
        if self.cancel_on_drop {
            self.cancellation.store(true, Ordering::Release);
            self.cancellation_wakeup.notify_one();
        }
    }
}

pub(super) struct RuntimeChatInvocationDriver {
    pub(super) request_id: String,
    pub(super) deadline: Option<Instant>,
    pub(super) observation_context: RuntimeObservationContext,
    pub(super) admission_ms: Option<f64>,
    pub(super) telemetry: Arc<RuntimeTelemetryCollector>,
    pub(super) cancellation: Arc<AtomicBool>,
    pub(super) cancellation_wakeup: Arc<Notify>,
    pub(super) events: mpsc::Sender<Result<RuntimeChatInvocationEvent>>,
    pub(super) teardown: Option<oneshot::Sender<RuntimeChatTeardown>>,
    pub(super) completion: oneshot::Receiver<Result<EngineOutput>>,
    pub(super) stream: Option<mpsc::Receiver<StreamingOutput>>,
    pub(super) stream_order: Option<StreamOutputOrder>,
    pub(super) streamed_text: String,
    /// Text the event consumer has not accepted yet. While it exists, later
    /// deltas merge into it instead of queueing, so a lagging consumer costs
    /// bytes, not messages, and never cancels the generation by itself.
    pub(super) pending_delta: Option<PendingTextDelta>,
    pub(super) slow_consumer_timeout: Duration,
    pub(super) guard: PendingRequestGuard,
}

pub(super) struct PendingTextDelta {
    pub(super) text: String,
    pub(super) logprobs: Vec<crate::engine::TokenLogprob>,
    pub(super) since: Instant,
}

impl PendingTextDelta {
    pub(super) fn into_event(self) -> Result<RuntimeChatInvocationEvent> {
        Ok(RuntimeChatInvocationEvent::TextDelta {
            text: self.text,
            logprobs: self.logprobs,
        })
    }
}

pub(super) enum DeltaOffer {
    Accepted,
    ConsumerClosed,
    Overflow,
}

pub(super) enum RuntimeChatCompletion {
    Delivered(Result<EngineOutput>),
    ChannelClosed,
}

impl RuntimeChatInvocationDriver {
    /// Best-effort event delivery for cancellation and failure paths, which
    /// must not wait on the consumer. Pending text goes first so events stay
    /// ordered; if it cannot be delivered it is discarded with the failed
    /// generation rather than reordered after the error.
    pub(super) fn try_send_event(&mut self, event: Result<RuntimeChatInvocationEvent>) -> bool {
        if let Some(pending) = self.pending_delta.take() {
            let _ = self.events.try_send(pending.into_event());
        }
        self.events.try_send(event).is_ok()
    }

    /// Queue a text delta, or merge it into the pending delta while the
    /// consumer lags.
    pub(super) fn offer_delta(
        &mut self,
        text: String,
        logprobs: Vec<crate::engine::TokenLogprob>,
    ) -> DeltaOffer {
        if let Some(pending) = self.pending_delta.as_mut() {
            pending.text.push_str(&text);
            pending.logprobs.extend(logprobs);
        } else {
            match self
                .events
                .try_send(Ok(RuntimeChatInvocationEvent::TextDelta { text, logprobs }))
            {
                Ok(()) => return DeltaOffer::Accepted,
                Err(mpsc::error::TrySendError::Closed(_)) => return DeltaOffer::ConsumerClosed,
                Err(mpsc::error::TrySendError::Full(event)) => {
                    let Ok(RuntimeChatInvocationEvent::TextDelta { text, logprobs }) = event else {
                        unreachable!("only a text delta was offered");
                    };
                    self.pending_delta = Some(PendingTextDelta {
                        text,
                        logprobs,
                        since: Instant::now(),
                    });
                }
            }
        }
        if self
            .pending_delta
            .as_ref()
            .is_some_and(|pending| pending.text.len() > RUNTIME_CHAT_MAX_PENDING_DELTA_BYTES)
        {
            DeltaOffer::Overflow
        } else {
            DeltaOffer::Accepted
        }
    }

    /// When a stalled consumer exhausts its slow-consumer budget.
    pub(super) fn slow_consumer_deadline(&self) -> Option<Instant> {
        self.pending_delta
            .as_ref()
            .map(|pending| pending.since + self.slow_consumer_timeout)
    }

    /// Deliver a terminal event after any pending text, waiting for the
    /// consumer up to the slow-consumer budget. Terminal events are never
    /// dropped for a full channel and never overtake text.
    pub(super) async fn send_terminal_event(&mut self, event: Result<RuntimeChatInvocationEvent>) -> bool {
        let deadline = self
            .slow_consumer_deadline()
            .unwrap_or_else(|| Instant::now() + self.slow_consumer_timeout);
        if let Some(pending) = self.pending_delta.take() {
            if !matches!(
                tokio::time::timeout_at(deadline.into(), self.events.send(pending.into_event()))
                    .await,
                Ok(Ok(()))
            ) {
                return false;
            }
        }
        matches!(
            tokio::time::timeout_at(deadline.into(), self.events.send(event)).await,
            Ok(Ok(()))
        )
    }

    pub(super) async fn finish_with_output(&mut self, output: EngineOutput) -> RuntimeChatTeardownDisposition {
        let generation = match invocation_chat_generation(&mut self.streamed_text, &output) {
            Ok(generation) => generation,
            Err(error) => {
                self.record_error(&error);
                self.guard.disarm();
                let _ = self.send_terminal_event(Err(error)).await;
                return RuntimeChatTeardownDisposition::Failed;
            }
        };
        self.telemetry
            .record_stage_observation(engine_output_observation(
                self.observation_context.clone(),
                self.admission_ms,
                &output,
            ));
        self.guard.disarm();
        let _ = self
            .send_terminal_event(Ok(RuntimeChatInvocationEvent::Completed(generation)))
            .await;
        // Engine completion is already physical teardown. A disconnected or
        // slow output consumer cannot turn it back into cancellation.
        RuntimeChatTeardownDisposition::Completed
    }

    pub(super) async fn finish_with_terminal_error(&mut self, error: Error) -> RuntimeChatTeardownDisposition {
        self.record_error(&error);
        self.guard.disarm();
        let disposition = if matches!(error, Error::Cancelled(_)) {
            RuntimeChatTeardownDisposition::Cancelled
        } else {
            RuntimeChatTeardownDisposition::Failed
        };
        let _ = self.send_terminal_event(Err(error)).await;
        disposition
    }

    pub(super) fn record_error(&self, error: &Error) {
        self.telemetry
            .record_stage_observation(engine_error_observation(
                self.observation_context.clone(),
                self.admission_ms,
                error.to_string(),
            ));
    }

    pub(super) async fn cancel_and_confirm(&mut self) -> Option<RuntimeChatTeardownDisposition> {
        self.cancellation.store(true, Ordering::Release);
        if let Some(stream) = self.stream.as_mut() {
            stream.close();
        }
        match self.guard.confirm_cleanup().await {
            Ok(_) => Some(RuntimeChatTeardownDisposition::Cancelled),
            Err(error) => {
                // A failed exact abort is not teardown proof. Best-effort error
                // delivery must not block the guard's fail-closed cleanup path.
                let _ = self.try_send_event(Err(error));
                None
            }
        }
    }

    pub(super) async fn fail_and_confirm(&mut self, error: Error) -> Option<RuntimeChatTeardownDisposition> {
        match self.cancel_and_confirm().await {
            Some(_) => {
                self.record_error(&error);
                let _ = self.try_send_event(Err(error));
                Some(RuntimeChatTeardownDisposition::Failed)
            }
            None => None,
        }
    }

    pub(super) async fn finish_completion(
        &mut self,
        completion: RuntimeChatCompletion,
    ) -> Option<RuntimeChatTeardownDisposition> {
        match completion {
            RuntimeChatCompletion::Delivered(Ok(output)) => {
                Some(self.finish_with_output(output).await)
            }
            RuntimeChatCompletion::Delivered(Err(error)) => {
                Some(self.finish_with_terminal_error(error).await)
            }
            RuntimeChatCompletion::ChannelClosed => {
                self.fail_and_confirm(Error::InferenceError(format!(
                    "Request {} completion channel closed unexpectedly",
                    self.request_id
                )))
                .await
            }
        }
    }

    pub(super) async fn drive(mut self) {
        if let Some(disposition) = self.drive_until_terminal().await {
            let _ = self.teardown.take().map(|sender| {
                sender.send(RuntimeChatTeardown {
                    request_id: self.request_id.clone(),
                    disposition,
                })
            });
        }
    }

    pub(super) async fn drive_until_terminal(&mut self) -> Option<RuntimeChatTeardownDisposition> {
        let mut completion_result = None;
        let invocation_deadline = self.deadline;
        let deadline_wait = async move {
            match invocation_deadline {
                Some(deadline) => tokio::time::sleep_until(deadline.into()).await,
                None => std::future::pending::<()>().await,
            }
        };
        tokio::pin!(deadline_wait);

        loop {
            if self.cancellation.load(Ordering::Acquire) || self.events.is_closed() {
                return self.cancel_and_confirm().await;
            }
            let slow_consumer_deadline = self.slow_consumer_deadline();

            if self.stream_order.is_none() && completion_result.is_some() {
                return self
                    .finish_completion(completion_result.take().expect("checked completion"))
                    .await;
            } else if self.stream.is_none() {
                if let Some(completion) = completion_result.take() {
                    let Some(stream_order) = self.stream_order.as_ref() else {
                        unreachable!("non-streaming invocation handled above");
                    };
                    // A model that fails mid-stream closes its stream without
                    // a final marker; its own error is the one to report.
                    let failed = matches!(completion, RuntimeChatCompletion::Delivered(Err(_)));
                    if !failed {
                        if let Err(error) = stream_order.require_final(&self.request_id) {
                            return self.fail_and_confirm(error).await;
                        }
                    }
                    return self.finish_completion(completion).await;
                }
            }

            tokio::select! {
                _ = self.cancellation_wakeup.notified() => {
                    if self.cancellation.load(Ordering::Acquire) || self.events.is_closed() {
                        return self.cancel_and_confirm().await;
                    }
                }
                _ = &mut deadline_wait => {
                    self.cancellation.store(true, Ordering::Release);
                    let error = Error::Timeout(self.request_id.clone());
                    let disposition = self.cancel_and_confirm().await;
                    if disposition.is_some() {
                        let _ = self.try_send_event(Err(error));
                    }
                    return disposition;
                }
                completion = &mut self.completion, if completion_result.is_none() => {
                    completion_result = Some(match completion {
                        Ok(result) => RuntimeChatCompletion::Delivered(result),
                        Err(_) => RuntimeChatCompletion::ChannelClosed,
                    });
                }
                capacity = async { self.events.reserve().await.is_ok() },
                    if self.pending_delta.is_some() =>
                {
                    // The driver is the only sender, so observed capacity
                    // stays available for the flush.
                    let pending = self.pending_delta.take().expect("checked pending delta");
                    if !capacity || self.events.try_send(pending.into_event()).is_err() {
                        return self.cancel_and_confirm().await;
                    }
                }
                _ = async {
                    match slow_consumer_deadline {
                        Some(deadline) => tokio::time::sleep_until(deadline.into()).await,
                        None => std::future::pending().await,
                    }
                }, if slow_consumer_deadline.is_some() => {
                    return self
                        .fail_and_confirm(Error::Backpressure(format!(
                            "chat output consumer for {} stalled for {:?}",
                            self.request_id, self.slow_consumer_timeout
                        )))
                        .await;
                }
                chunk = async {
                    match self.stream.as_mut() {
                        Some(stream) => stream.recv().await,
                        None => std::future::pending().await,
                    }
                }, if self.stream.is_some() => {
                    match chunk {
                        Some(chunk) => {
                            let observed = self
                                .stream_order
                                .as_mut()
                                .expect("streaming invocation has output order")
                                .observe(&self.request_id, &chunk);
                            if let Err(error) = observed {
                                return self.fail_and_confirm(error).await;
                            }
                            if let Some(delta) = chunk
                                .text
                                .filter(|delta| !delta.is_empty() || !chunk.logprobs.is_empty())
                            {
                                self.streamed_text.push_str(&delta);
                                match self.offer_delta(delta, chunk.logprobs) {
                                    DeltaOffer::Accepted => {}
                                    DeltaOffer::ConsumerClosed => {
                                        return self.cancel_and_confirm().await;
                                    }
                                    DeltaOffer::Overflow => {
                                        return self
                                            .fail_and_confirm(Error::Backpressure(format!(
                                                "chat output consumer for {} owes more than {} bytes",
                                                self.request_id,
                                                RUNTIME_CHAT_MAX_PENDING_DELTA_BYTES
                                            )))
                                            .await;
                                    }
                                }
                            }
                        }
                        None => {
                            self.stream = None;
                        }
                    }
                }
            }
        }
    }
}

pub(super) fn invocation_chat_generation(
    streamed_text: &mut String,
    output: &EngineOutput,
) -> Result<ChatGeneration> {
    let terminal_text = output.text.clone();
    let text = if streamed_text.is_empty() {
        terminal_text.unwrap_or_default()
    } else if terminal_text
        .as_ref()
        .is_none_or(|terminal| terminal.is_empty() || terminal == streamed_text)
    {
        std::mem::take(streamed_text)
    } else {
        return Err(Error::InferenceError(format!(
            "Streaming chat text did not match terminal output (streamed {} bytes, terminal {} bytes)",
            streamed_text.len(),
            terminal_text.as_ref().map_or(0, String::len)
        )));
    };
    Ok(ChatGeneration {
        latency_breakdown: output.latency_breakdown.clone(),
        finish_reason: output.finish_reason,
        text,
        prompt_tokens: output.token_stats.prompt_tokens,
        tokens_generated: output.num_tokens,
        generation_time_ms: output.generation_time.as_secs_f64() * 1000.0,
        cached_prompt_tokens: output.token_stats.cached_prefix_tokens.map(u64::from),
        logprobs: output.logprobs.clone(),
    })
}

impl RuntimeService {
    /// Establish an exact Engine session for a prepared chat invocation.
    ///
    /// Returning from this method is the worker acceptance boundary. All
    /// admission and model residency is transferred to a background driver so
    /// dropping the returned transport handle cannot release capacity early.
    pub(crate) async fn start_admitted_chat_invocation(
        &self,
        admitted: AdmittedEngineRequest,
    ) -> Result<RuntimeChatInvocation> {
        let AdmittedEngineRequest {
            mut request,
            job,
            residency_lease,
        } = admitted;
        let streaming = request.streaming;
        self.observe_broker_request(&request)?;
        let (prepared, job) = self
            .prepare_request_for_binding(request, job, Some(&residency_lease))
            .await?;
        request = prepared;
        let loaded_bundle = self
            .model_lifecycle
            .try_get_ready_bundle(residency_lease.variant());
        bind_request_to_residency(
            &mut request,
            Some(&residency_lease),
            loaded_bundle.as_deref(),
            streaming,
        )?;
        if job.spec.request_id != request.id || job.spec.deadline != request.deadline {
            return Err(Error::InvalidInput(
                "chat invocation does not match its coordinator admission".to_string(),
            ));
        }
        request.runtime_admitted_at = Some(job.admitted_at());

        let observation_context = self.engine_observation_context(&request, streaming);
        let admission_ms = request.admission_ms;
        self.ensure_step_driver_started().await;
        let request_id = request.id.clone();
        let deadline = request.deadline;
        let stream_policy = request.stream_policy;
        let cancellation = Arc::new(AtomicBool::new(false));
        request.set_cancellation_signal(cancellation.clone());
        let (waiter_registration_id, completion) = self.register_waiter(&request_id).await?;
        let mut waiter_guard = WaiterRegistrationGuard::new(
            request_id.clone(),
            waiter_registration_id,
            self.completion_waiters.clone(),
        );

        let (session, stream) = if streaming {
            match self
                .await_engine_admission_for_job(
                    &job,
                    self.core_engine.generate_streaming_with_session(request),
                )
                .await
            {
                Ok((session, stream)) => (session, Some(stream)),
                Err(error) => {
                    self.remove_waiter(&request_id, waiter_registration_id)
                        .await;
                    waiter_guard.disarm();
                    return Err(error);
                }
            }
        } else {
            match self
                .await_engine_admission_for_job(
                    &job,
                    self.core_engine.add_request_with_session(request),
                )
                .await
            {
                Ok(session) => (session, None),
                Err(error) => {
                    self.remove_waiter(&request_id, waiter_registration_id)
                        .await;
                    waiter_guard.disarm();
                    return Err(error);
                }
            }
        };

        let mut guard = PendingRequestGuard::new(
            session,
            self.core_engine.clone(),
            self.completion_waiters.clone(),
            waiter_registration_id,
            self.telemetry.clone(),
            job,
            Some(residency_lease),
        );
        if deadline.is_some_and(|deadline| deadline <= Instant::now()) {
            let timeout = Error::Timeout(request_id.clone());
            return match guard.confirm_cleanup().await {
                Ok(_) => Err(timeout),
                Err(cleanup_error) => Ok(RuntimeChatInvocation::from_failed_post_admission(
                    request_id,
                    cancellation,
                    guard,
                    Error::InferenceError(format!(
                        "{timeout}; exact-session cleanup was not confirmed: {cleanup_error}"
                    )),
                )),
            };
        }
        if let Err(bind_error) = self
            .bind_waiter(&request_id, waiter_registration_id, guard.session.epoch)
            .await
        {
            waiter_guard.disarm();
            return match guard.confirm_cleanup().await {
                Ok(_) => Err(bind_error),
                Err(cleanup_error) => Ok(RuntimeChatInvocation::from_failed_post_admission(
                    request_id,
                    cancellation,
                    guard,
                    Error::InferenceError(format!(
                        "{bind_error}; exact-session cleanup was not confirmed: {cleanup_error}"
                    )),
                )),
            };
        }
        waiter_guard.disarm();
        self.telemetry.record_request_queued(&request_id).await;

        let (event_sender, events) = mpsc::channel(RUNTIME_CHAT_INVOCATION_EVENT_CAPACITY);
        let (teardown_sender, teardown) = oneshot::channel();
        let cancellation_wakeup = Arc::new(Notify::new());
        let driver = RuntimeChatInvocationDriver {
            request_id: request_id.clone(),
            deadline,
            observation_context,
            admission_ms,
            telemetry: self.telemetry.clone(),
            cancellation: cancellation.clone(),
            cancellation_wakeup: cancellation_wakeup.clone(),
            events: event_sender,
            teardown: Some(teardown_sender),
            completion,
            stream,
            stream_order: streaming.then(|| StreamOutputOrder::new(stream_policy)),
            streamed_text: String::new(),
            pending_delta: None,
            slow_consumer_timeout: stream_slow_consumer_timeout(),
            guard,
        };
        tokio::spawn(driver.drive());
        self.step_driver_wakeup.notify_one();

        Ok(RuntimeChatInvocation {
            request_id,
            cancellation,
            cancellation_wakeup,
            events,
            teardown: Some(teardown),
            cancel_on_drop: true,
        })
    }
}
