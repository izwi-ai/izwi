use std::thread;

use tracing::error;

use crate::backends::can_parallelize_requests;
use crate::error::{Error, Result};
use crate::model::ModelVariant;

use super::super::request::EngineCoreRequest;
use super::super::scheduler::ScheduledRequest;
use super::super::types::TaskType;
use super::{
    ExecutorOutput, ExecutorStepResult, ModelExecutor, ModelSessionResult, NativeExecutor,
};
use crate::engine::ReadyQuantum;
use crate::engine::{BatchDispatch, BatchDispatchKind};

type RouteHandler =
    fn(&NativeExecutor, &EngineCoreRequest, &ScheduledRequest) -> Result<ModelSessionResult>;
type VariantMatcher = fn(ModelVariant) -> bool;

struct DispatchRoute {
    name: &'static str,
    task: TaskType,
    variant_matcher: Option<VariantMatcher>,
    handler: RouteHandler,
}

impl DispatchRoute {
    fn matches(&self, task: TaskType, variant: Option<ModelVariant>) -> bool {
        if self.task != task {
            return false;
        }

        match self.variant_matcher {
            Some(matcher) => variant.map(matcher).unwrap_or(false),
            None => true,
        }
    }
}

const DISPATCH_ROUTES: &[DispatchRoute] = &[
    DispatchRoute {
        name: "lfm25_audio_tts",
        task: TaskType::TTS,
        variant_matcher: Some(|variant| {
            variant.family() == crate::catalog::ModelFamily::Lfm25Audio
        }),
        handler: |executor, request, scheduled| {
            executor.lfm25_audio_tts_request_with_managed_cache(request, scheduled, None)
        },
    },
    DispatchRoute {
        name: "tts",
        task: TaskType::TTS,
        variant_matcher: None,
        handler: NativeExecutor::qwen_tts_request,
    },
    DispatchRoute {
        name: "asr",
        task: TaskType::ASR,
        variant_matcher: None,
        handler: NativeExecutor::transcribe_request,
    },
    DispatchRoute {
        name: "speech_to_speech",
        task: TaskType::SpeechToSpeech,
        variant_matcher: None,
        handler: NativeExecutor::audio_chat_request,
    },
    DispatchRoute {
        name: "chat",
        task: TaskType::Chat,
        variant_matcher: None,
        handler: NativeExecutor::chat_request,
    },
];

fn vibevoice_prefill_dispatch(
    used_native_tokenizer_batch: bool,
    scheduled_width: usize,
) -> BatchDispatch {
    if used_native_tokenizer_batch {
        BatchDispatch::new(BatchDispatchKind::TensorStatic, scheduled_width)
    } else {
        BatchDispatch::serial()
    }
}

impl NativeExecutor {
    fn find_request<'a>(
        requests: &'a [&EngineCoreRequest],
        scheduled: &ScheduledRequest,
    ) -> Option<&'a EngineCoreRequest> {
        requests
            .iter()
            .copied()
            .find(|r| r.id == scheduled.request_id)
    }

    pub(super) fn resolve_variant(request: &EngineCoreRequest) -> Result<ModelVariant> {
        request.model_variant.ok_or_else(|| {
            Error::InvalidInput(format!(
                "Request {} is missing model variant routing information",
                request.id
            ))
        })
    }

    fn resolve_route(
        task: TaskType,
        variant: Option<ModelVariant>,
    ) -> Option<&'static DispatchRoute> {
        DISPATCH_ROUTES
            .iter()
            .find(|route| route.matches(task, variant))
    }

    fn execute_single_request(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled_req: &ScheduledRequest,
        managed_cache: Option<&crate::engine::ManagedCacheReservation>,
    ) -> ModelSessionResult {
        let Some(request) = Self::find_request(requests, scheduled_req) else {
            return ModelSessionResult::atomic(ExecutorOutput::error(
                scheduled_req.request_id.clone(),
                "Scheduled request not found in batch",
            ));
        };

        if request.is_cancelled() {
            return ModelSessionResult::cancelled_before_dispatch(ExecutorOutput::cancelled(
                request.id.clone(),
            ));
        }

        let Some(route) = Self::resolve_route(request.task_type, request.model_variant) else {
            return ModelSessionResult::atomic(ExecutorOutput::error(
                request.id.clone(),
                format!(
                    "No executor route for task {:?} (variant {:?})",
                    request.task_type, request.model_variant
                ),
            ));
        };

        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match managed_cache {
                Some(reservation) if request.task_type == TaskType::Chat => {
                    if request.model_variant.is_some_and(|variant| {
                        variant.family() == crate::catalog::ModelFamily::Qwen38Chat
                    }) {
                        let caches = super::qwen38_managed_caches_for_row(
                            request,
                            scheduled_req,
                            reservation,
                        )?;
                        self.chat_request_with_managed_cache(
                            request,
                            scheduled_req,
                            Some(caches.target),
                            caches.mtp,
                            reservation.clocked_state.clone(),
                        )
                    } else {
                        let cache = super::qwen3_managed_cache_for_row(
                            request,
                            scheduled_req,
                            reservation,
                        )?;
                        self.chat_request_with_managed_cache(
                            request,
                            scheduled_req,
                            Some(cache),
                            None,
                            reservation.clocked_state.clone(),
                        )
                    }
                }
                Some(reservation) if request.task_type == TaskType::ASR => {
                    let state = super::retained_row_managed_state_for_row(
                        request,
                        scheduled_req,
                        reservation,
                    )?;
                    self.transcribe_request_with_managed_cache(request, scheduled_req, Some(state))
                }
                Some(reservation)
                    if request.task_type == TaskType::TTS
                        && request.model_variant.is_some_and(|variant| {
                            variant.family() == crate::catalog::ModelFamily::Lfm25Audio
                        }) =>
                {
                    let state = super::retained_row_managed_state_for_row(
                        request,
                        scheduled_req,
                        reservation,
                    )?;
                    self.lfm25_audio_tts_request_with_managed_cache(
                        request,
                        scheduled_req,
                        Some(state),
                    )
                }
                Some(reservation)
                    if request.task_type == TaskType::TTS
                        && request.model_variant.is_some_and(|variant| {
                            variant.family() == crate::catalog::ModelFamily::VibeVoiceTts
                        }) =>
                {
                    let state = super::retained_row_managed_state_for_row(
                        request,
                        scheduled_req,
                        reservation,
                    )?;
                    self.vibevoice_tts_request_with_managed_cache(
                        request,
                        scheduled_req,
                        Some(state),
                    )
                }
                Some(reservation)
                    if request.task_type == TaskType::TTS
                        && request.model_variant.is_some_and(|variant| {
                            variant.family() == crate::catalog::ModelFamily::Qwen3Tts
                        }) =>
                {
                    let cache =
                        super::qwen3_managed_cache_for_row(request, scheduled_req, reservation)?;
                    self.qwen_tts_request_with_managed_cache(
                        request,
                        scheduled_req,
                        Some(cache),
                        reservation.clocked_state.clone(),
                    )
                }
                Some(_) => Err(Error::InferenceError(
                    "managed paged cache was routed to an unsupported executor".to_string(),
                )),
                None => (route.handler)(self, request, scheduled_req),
            }));

        let result = match result {
            Ok(result) => result,
            Err(payload) => {
                let message = super::panic_payload_to_string(payload.as_ref());
                error!(
                    request_id = %request.id,
                    task = ?request.task_type,
                    route = route.name,
                    "Executor request handling panicked: {message}"
                );
                std::panic::resume_unwind(payload)
            }
        };

        match result {
            Ok(output) => output,
            Err(err) => ModelSessionResult::atomic(ExecutorOutput::error(
                request.id.clone(),
                err.to_string(),
            )),
        }
    }

    fn can_parallelize_requests(&self, scheduled_len: usize) -> bool {
        if scheduled_len <= 1 || self.config.request_parallelism <= 1 {
            return false;
        }
        // Keep Metal execution serialized to avoid command-queue contention.
        can_parallelize_requests(self.config.backend)
    }

    fn execute_requests_parallel(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
        rows: Option<&[ReadyQuantum]>,
    ) -> Result<Vec<ModelSessionResult>> {
        let worker_count = self.config.request_parallelism.min(scheduled.len()).max(1);
        let mut partitions: Vec<Vec<(usize, ScheduledRequest)>> = vec![Vec::new(); worker_count];
        for (idx, item) in scheduled.iter().enumerate() {
            partitions[idx % worker_count].push((idx, item.clone()));
        }

        let (tx, rx) = std::sync::mpsc::channel::<Vec<(usize, ModelSessionResult)>>();
        thread::scope(|scope| {
            for chunk in partitions {
                if chunk.is_empty() {
                    continue;
                }
                let tx = tx.clone();
                scope.spawn(move || {
                    let mut local = Vec::with_capacity(chunk.len());
                    for (idx, scheduled_req) in chunk {
                        let managed_cache = rows.and_then(|rows| {
                            rows.iter()
                                .find(|row| row.plan_id == scheduled_req.plan_id)
                                .and_then(|row| row.managed_cache.as_ref())
                        });
                        let output =
                            self.execute_single_request(requests, &scheduled_req, managed_cache);
                        local.push((idx, output));
                    }
                    let _ = tx.send(local);
                });
            }
        });
        drop(tx);

        let mut ordered: Vec<Option<ModelSessionResult>> = vec![None; scheduled.len()];
        while let Ok(batch_outputs) = rx.recv() {
            for (idx, output) in batch_outputs {
                if idx < ordered.len() {
                    ordered[idx] = Some(output);
                }
            }
        }

        let outputs = ordered
            .into_iter()
            .enumerate()
            .map(|(idx, output)| {
                output.unwrap_or_else(|| {
                    ModelSessionResult::atomic(ExecutorOutput::error(
                        scheduled[idx].request_id.clone(),
                        "Parallel executor worker failed to produce output",
                    ))
                })
            })
            .collect();
        Ok(outputs)
    }

    pub(super) fn execute_requests(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorStepResult>> {
        self.execute_requests_with_rows(requests, scheduled, None)
    }

    pub(super) fn execute_requests_with_rows(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
        rows: Option<&[ReadyQuantum]>,
    ) -> Result<Vec<ExecutorStepResult>> {
        let (outputs, dispatch) = if self.can_parallelize_requests(scheduled.len()) {
            (
                self.execute_requests_parallel(requests, scheduled, rows)?,
                BatchDispatch::new(BatchDispatchKind::RequestParallel, scheduled.len()),
            )
        } else {
            (
                scheduled
                    .iter()
                    .map(|scheduled_req| {
                        let managed_cache = rows.and_then(|rows| {
                            rows.iter()
                                .find(|row| row.plan_id == scheduled_req.plan_id)
                                .and_then(|row| row.managed_cache.as_ref())
                        });
                        self.execute_single_request(requests, scheduled_req, managed_cache)
                    })
                    .collect(),
                BatchDispatch::serial(),
            )
        };
        self.finish_scheduled_execution(requests, scheduled, outputs, dispatch, rows)
    }

    pub(super) fn execute_continuous_chat_requests(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorStepResult>> {
        self.execute_continuous_chat_requests_with_rows(requests, scheduled, None)
    }

    pub(super) fn execute_continuous_chat_requests_with_rows(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
        rows: Option<&[ReadyQuantum]>,
    ) -> Result<Vec<ExecutorStepResult>> {
        let managed_caches = scheduled
            .iter()
            .map(|scheduled| {
                let reservation = rows
                    .and_then(|rows| rows.iter().find(|row| row.plan_id == scheduled.plan_id))
                    .and_then(|row| row.managed_cache.as_ref());
                let request = Self::find_request(requests, scheduled).ok_or_else(|| {
                    Error::InferenceError(
                        "continuous managed-cache row has no request snapshot".to_string(),
                    )
                })?;
                reservation
                    .map(|reservation| {
                        super::retained_row_managed_state_for_row(request, scheduled, reservation)
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?;
        let outputs = self.chat_decode_batch_with_managed(requests, scheduled, managed_caches)?;
        self.finish_scheduled_execution(
            requests,
            scheduled,
            outputs,
            BatchDispatch::new(BatchDispatchKind::TensorContinuous, scheduled.len()),
            rows,
        )
    }

    pub(super) fn execute_continuous_asr_requests_with_rows(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
        rows: Option<&[ReadyQuantum]>,
    ) -> Result<Vec<ExecutorStepResult>> {
        let managed_caches = scheduled
            .iter()
            .map(|scheduled| {
                let reservation = rows
                    .and_then(|rows| rows.iter().find(|row| row.plan_id == scheduled.plan_id))
                    .and_then(|row| row.managed_cache.as_ref());
                let request = Self::find_request(requests, scheduled).ok_or_else(|| {
                    Error::InferenceError(
                        "continuous ASR managed-cache row has no request snapshot".to_string(),
                    )
                })?;
                reservation
                    .map(|reservation| {
                        super::retained_row_managed_state_for_row(request, scheduled, reservation)
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?;
        let outputs = self.asr_decode_batch_with_managed(requests, scheduled, managed_caches)?;
        self.finish_scheduled_execution(
            requests,
            scheduled,
            outputs,
            BatchDispatch::new(BatchDispatchKind::TensorContinuous, scheduled.len()),
            rows,
        )
    }

    pub(super) fn execute_static_vibevoice_prefill_requests_with_rows(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
        rows: Option<&[ReadyQuantum]>,
    ) -> Result<Vec<ExecutorStepResult>> {
        let managed_caches = scheduled
            .iter()
            .map(|scheduled| {
                let reservation = rows
                    .and_then(|rows| rows.iter().find(|row| row.plan_id == scheduled.plan_id))
                    .and_then(|row| row.managed_cache.as_ref());
                let request = Self::find_request(requests, scheduled).ok_or_else(|| {
                    Error::InferenceError(
                        "static VibeVoice prefill row has no request snapshot".into(),
                    )
                })?;
                reservation
                    .map(|reservation| {
                        super::retained_row_managed_state_for_row(request, scheduled, reservation)
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?;
        let (outputs, used_native_tokenizer_batch) =
            self.vibevoice_prefill_batch_with_managed(requests, scheduled, managed_caches)?;
        let dispatch = vibevoice_prefill_dispatch(used_native_tokenizer_batch, scheduled.len());
        self.finish_scheduled_execution(requests, scheduled, outputs, dispatch, rows)
    }

    pub(super) fn execute_static_lfm25_asr_prefill_requests_with_rows(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
        rows: Option<&[ReadyQuantum]>,
    ) -> Result<Vec<ExecutorStepResult>> {
        let ordered_requests = scheduled
            .iter()
            .map(|scheduled| {
                Self::find_request(requests, scheduled).ok_or_else(|| {
                    Error::InferenceError(
                        "static LFM2.5 ASR prefill row has no request snapshot".into(),
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let managed = scheduled
            .iter()
            .zip(&ordered_requests)
            .map(|(scheduled, request)| {
                let reservation = rows
                    .and_then(|rows| rows.iter().find(|row| row.plan_id == scheduled.plan_id))
                    .and_then(|row| row.managed_cache.as_ref());
                reservation
                    .map(|reservation| {
                        super::retained_row_managed_state_for_row(request, scheduled, reservation)
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?;
        let outputs =
            self.lfm25_audio_asr_prefill_batch_with_managed(&ordered_requests, scheduled, managed)?;
        self.finish_scheduled_execution(
            requests,
            scheduled,
            outputs,
            BatchDispatch::new(BatchDispatchKind::TensorStatic, scheduled.len()),
            rows,
        )
    }

    pub(super) fn execute_continuous_tts_requests_with_rows(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
        rows: Option<&[ReadyQuantum]>,
    ) -> Result<Vec<ExecutorStepResult>> {
        let managed_caches = scheduled
            .iter()
            .map(|scheduled| {
                let reservation = rows
                    .and_then(|rows| rows.iter().find(|row| row.plan_id == scheduled.plan_id))
                    .and_then(|row| row.managed_cache.as_ref());
                let request = Self::find_request(requests, scheduled).ok_or_else(|| {
                    Error::InferenceError(
                        "continuous TTS managed-cache row has no request snapshot".to_string(),
                    )
                })?;
                reservation
                    .map(|reservation| {
                        super::retained_row_managed_state_for_row(request, scheduled, reservation)
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?;
        let outputs = self.tts_decode_batch_with_managed(requests, scheduled, managed_caches)?;
        self.finish_scheduled_execution(
            requests,
            scheduled,
            outputs,
            BatchDispatch::new(BatchDispatchKind::TensorContinuous, scheduled.len()),
            rows,
        )
    }

    pub(super) fn execute_static_lfm25_tts_prefill_requests_with_rows(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
        rows: Option<&[ReadyQuantum]>,
    ) -> Result<Vec<ExecutorStepResult>> {
        let ordered_requests = scheduled
            .iter()
            .map(|scheduled| {
                Self::find_request(requests, scheduled).ok_or_else(|| {
                    Error::InferenceError(
                        "static LFM2.5 TTS prefill row has no request snapshot".into(),
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let managed = scheduled
            .iter()
            .zip(&ordered_requests)
            .map(|(scheduled, request)| {
                let reservation = rows
                    .and_then(|rows| rows.iter().find(|row| row.plan_id == scheduled.plan_id))
                    .and_then(|row| row.managed_cache.as_ref());
                reservation
                    .map(|reservation| {
                        super::retained_row_managed_state_for_row(request, scheduled, reservation)
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?;
        let outputs =
            self.lfm25_audio_tts_prefill_batch_with_managed(&ordered_requests, scheduled, managed)?;
        self.finish_scheduled_execution(
            requests,
            scheduled,
            outputs,
            BatchDispatch::new(BatchDispatchKind::TensorStatic, scheduled.len()),
            rows,
        )
    }

    pub(super) fn finish_scheduled_execution(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
        outputs: Vec<ModelSessionResult>,
        dispatch: BatchDispatch,
        rows: Option<&[ReadyQuantum]>,
    ) -> Result<Vec<ExecutorStepResult>> {
        if outputs.len() != scheduled.len() {
            return Err(Error::InferenceError(format!(
                "native executor produced {} session results for {} scheduled rows",
                outputs.len(),
                scheduled.len()
            )));
        }
        let dispatch = if !outputs.is_empty()
            && outputs
                .iter()
                .all(|output| output.provenance.dispatch_state == super::DispatchState::NotStarted)
        {
            BatchDispatch::not_dispatched(scheduled.len().max(1))
        } else {
            dispatch
        };
        Ok(scheduled
            .iter()
            .zip(outputs)
            .map(|(scheduled, output)| {
                let Some(request) = requests
                    .iter()
                    .copied()
                    .find(|request| request.id == scheduled.request_id)
                else {
                    return ExecutorStepResult::from_session(
                        scheduled,
                        ModelSessionResult::atomic(ExecutorOutput::error(
                            scheduled.request_id.clone(),
                            "Scheduled request not found during execution finalization",
                        )),
                    )
                    .with_dispatch(dispatch);
                };
                let staged = request.take_staged_stream_outputs();
                let mut result = match staged {
                    Ok(staged) => ExecutorStepResult::from_session(
                        scheduled,
                        output.with_staged_stream_outputs(staged),
                    )
                    .with_dispatch(dispatch)
                    .with_observed_resources(crate::engine::ResourceVector::zero()),
                    Err(err) => {
                        let _ = ModelExecutor::cleanup_session(self, &scheduled.session_key());
                        ExecutorStepResult::from_session(
                            scheduled,
                            ModelSessionResult::atomic(ExecutorOutput::error(
                                scheduled.request_id.clone(),
                                format!("staged stream publication failed: {err}"),
                            )),
                        )
                        .with_dispatch(dispatch)
                        .with_observed_resources(crate::engine::ResourceVector::zero())
                    }
                };
                if matches!(
                    result.disposition,
                    super::ExecutionDisposition::RestartSequence(_)
                ) && (!result.staged_stream_outputs.is_empty()
                    || !result.managed_cache_completions.is_empty()
                    || result.clocked_state_completion.is_some()
                    || result.output.audio.is_some()
                    || result.output.text.is_some()
                    || result.output.input_transcription.is_some()
                    || result.output.phase_timing_override.is_some()
                    || result.output.asr_diagnostics.is_some())
                {
                    result = ExecutorStepResult::from_session(
                        scheduled,
                        ModelSessionResult::atomic(ExecutorOutput::error(
                            scheduled.request_id.clone(),
                            "sequence restart cannot publish stream output or managed-cache completions",
                        )),
                    )
                    .with_dispatch(dispatch)
                    .with_observed_resources(result.observed_resources);
                }
                if result.output.error.is_none()
                    && result.provenance.dispatch_state == super::DispatchState::ProducedOutput
                    && matches!(
                        result.disposition,
                        super::ExecutionDisposition::Progress
                            | super::ExecutionDisposition::Yielded(_)
                            | super::ExecutionDisposition::Finished(
                                super::FinishReason::Completed,
                            )
                    )
                {
                    if let Some(reservation) = rows
                        .and_then(|rows| rows.iter().find(|row| row.plan_id == scheduled.plan_id))
                        .and_then(|row| row.managed_cache.as_ref())
                    {
                        let selected_clocked_state = reservation
                            .clocked_state
                            .as_ref()
                            .is_some_and(|state| state.selections().is_some());
                        let receipt = if let Some(appended) = result.managed_cache_append {
                            reservation
                                .domains
                                .first()
                                .ok_or_else(|| {
                                    Error::InferenceError(
                                        "managed-cache reservation has no domains".into(),
                                    )
                                })
                                .and_then(|domain| {
                                    let appended = u32::try_from(appended).map_err(|_| {
                                        Error::InferenceError(
                                            "managed-cache append exceeds u32".into(),
                                        )
                                    })?;
                                    let committed = domain
                                        .execution_start_tokens
                                        .checked_add(appended)
                                        .ok_or_else(|| {
                                            Error::InferenceError(
                                                "managed-cache accepted cursor overflowed".into(),
                                            )
                                        })?;
                                    reservation.completed_write_receipt_for_prefix(
                                        &result.managed_cache_completions,
                                        committed,
                                    )
                                })
                        } else if selected_clocked_state
                            && result.output.tokens_processed != scheduled.num_tokens
                        {
                            Err(Error::InferenceError(
                                "selected auxiliary state requires exact scheduled progress"
                                    .into(),
                            ))
                        } else if result.output.tokens_processed == scheduled.num_tokens {
                            reservation.completed_write_receipt(&result.managed_cache_completions)
                        } else {
                            reservation
                                .domains
                                .first()
                                .ok_or_else(|| {
                                    Error::InferenceError(
                                        "managed-cache reservation has no domains".into(),
                                    )
                                })
                                .and_then(|domain| {
                                    let accepted = u32::try_from(result.output.tokens_processed)
                                        .map_err(|_| {
                                            Error::InferenceError(
                                                "accepted cache prefix exceeds u32".into(),
                                            )
                                        })?;
                                    let committed = domain
                                        .execution_start_tokens
                                        .checked_add(accepted)
                                        .ok_or_else(|| {
                                            Error::InferenceError(
                                                "accepted cache cursor overflowed".into(),
                                            )
                                        })?;
                                    reservation.completed_write_receipt_for_prefix(
                                        &result.managed_cache_completions,
                                        committed,
                                    )
                                })
                        }
                        .and_then(|receipt| {
                            match result.clocked_state_completion.clone() {
                                Some(completion) => {
                                    receipt.with_clocked_state_completion(completion)
                                }
                                None => Ok(receipt),
                            }
                        });
                        match receipt {
                            Ok(receipt) => result.managed_cache = Some(receipt),
                            Err(error) => {
                                result = ExecutorStepResult::from_session(
                                    scheduled,
                                    ModelSessionResult::atomic(ExecutorOutput::error(
                                        scheduled.request_id.clone(),
                                        format!(
                                            "physical cache completion reconciliation failed: {error}"
                                        ),
                                    )),
                                )
                                .with_dispatch(dispatch)
                                .with_observed_resources(result.observed_resources);
                            }
                        }
                    } else if !result.managed_cache_completions.is_empty()
                        || result.clocked_state_completion.is_some()
                    {
                        result = ExecutorStepResult::from_session(
                            scheduled,
                            ModelSessionResult::atomic(ExecutorOutput::error(
                                scheduled.request_id.clone(),
                                "executor returned an unplanned physical cache completion",
                            )),
                        )
                        .with_dispatch(dispatch)
                        .with_observed_resources(result.observed_resources);
                    }
                }
                result.managed_cache_completions.clear();
                result.clocked_state_completion = None;
                result
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::kv::{
        CpuKvBackendRuntime, KvArenaConfig, KvBackendRuntime, KvLayerConfig, KvWriteArgs,
        KvWriteCompletionCollector,
    };
    use crate::backends::state::{
        PhysicalStateSequenceId, PhysicalStateTransactionId, TensorStateBatchCompletion,
        TensorStateSelection,
    };
    use crate::backends::BackendKind;
    use crate::engine::cache::coordinator::GroupBlockTable;
    use crate::engine::{
        AdapterAbiRevision, AdapterInstanceId, BatchDispatchKind, BatchLaneKey, ClockedStateSpan,
        ExecutionGroupId, InputRange, ManagedCacheDomainReservation, ManagedCacheReservation,
        ManagedClockedStateReservation, ModelInstanceId, SequencePhase, StageId, WorkCost,
        WorkUnit,
    };
    use crate::kv::v2::{StateClock, StateGroupId};
    use crate::kv::{CacheBlockRef, CacheDomainId, KvArenaId, KvGroupId};
    use candle_core::{DType, Device, Tensor};
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    fn scheduled(request_id: &str, plan_id: u64) -> ScheduledRequest {
        ScheduledRequest {
            plan_id,
            request_id: request_id.to_string(),
            sequence_id: plan_id,
            num_tokens: 1,
            is_prefill: true,
            num_computed_tokens: 0,
            work: WorkUnit::SequenceStep {
                phase: SequencePhase::Prefill,
                input: InputRange { start: 0, end: 1 },
                max_output_steps: 1,
                auxiliary_state: None,
            },
        }
    }

    #[test]
    fn cancelled_request_is_rejected_before_model_dispatch() {
        let executor = NativeExecutor::new(super::super::WorkerConfig::default());
        let mut request = EngineCoreRequest::tts("cancelled");
        request.id = "cancelled".to_string();
        let signal = Arc::new(AtomicBool::new(true));
        request.set_cancellation_signal(signal);
        let scheduled = ScheduledRequest {
            plan_id: 1,
            request_id: request.id.clone(),
            sequence_id: 1,
            num_tokens: 1,
            is_prefill: true,
            num_computed_tokens: 0,
            work: crate::engine::WorkUnit::SequenceStep {
                phase: crate::engine::SequencePhase::Prefill,
                input: crate::engine::InputRange { start: 0, end: 1 },
                max_output_steps: 1,
                auxiliary_state: None,
            },
        };

        let result = executor.execute_single_request(&[&request], &scheduled, None);
        assert!(result.output.finished);
        assert!(result.output.error.is_none());
        assert_eq!(
            result.disposition,
            crate::engine::ExecutionDisposition::Finished(crate::engine::FinishReason::Cancelled)
        );
    }

    #[test]
    fn backend_policy_controls_real_request_parallel_dispatch() {
        for (backend, expected) in [
            (BackendKind::Cpu, BatchDispatchKind::RequestParallel),
            (BackendKind::Metal, BatchDispatchKind::Serial),
            (BackendKind::Cuda, BatchDispatchKind::RequestParallel),
        ] {
            let config = super::super::WorkerConfig {
                backend,
                request_parallelism: 2,
                ..Default::default()
            };
            let executor = NativeExecutor::new(config);
            let mut first = EngineCoreRequest::tts("first");
            first.id = "parallel-first".to_string();
            let mut second = EngineCoreRequest::tts("second");
            second.id = "parallel-second".to_string();
            let scheduled = vec![scheduled(&first.id, 1), scheduled(&second.id, 2)];

            let outputs = executor
                .execute_requests(&[&first, &second], &scheduled)
                .expect("dispatch should return per-request results");

            assert_eq!(outputs.len(), 2);
            assert!(outputs
                .iter()
                .all(|output| output.dispatch.kind == expected));
            let expected_width = if expected == BatchDispatchKind::RequestParallel {
                2
            } else {
                1
            };
            assert!(outputs
                .iter()
                .all(|output| output.dispatch.width == expected_width));
        }
    }

    #[test]
    fn vibevoice_prefill_dispatch_reports_the_call_that_survived_cancellation() {
        assert_eq!(
            vibevoice_prefill_dispatch(true, 2),
            BatchDispatch::new(BatchDispatchKind::TensorStatic, 2)
        );
        assert_eq!(
            vibevoice_prefill_dispatch(false, 2),
            BatchDispatch::serial()
        );
    }

    #[test]
    fn successful_native_row_emits_receipt_for_its_exact_reserved_blocks() {
        let executor = NativeExecutor::new(super::super::WorkerConfig::default());
        let mut request = EngineCoreRequest::tts("receipt");
        request.id = "managed-receipt".to_string();
        let scheduled = scheduled(&request.id, 7);
        let arena = KvArenaId {
            model_instance: ModelInstanceId::new(4),
            backend: BackendKind::Cpu,
            device_ordinal: None,
            generation: 2,
        };
        let block = CacheBlockRef {
            arena,
            group: KvGroupId::new(0),
            index: 3,
            slot_generation: 5,
        };
        let reservation = ManagedCacheReservation {
            txn_id: scheduled.plan_id,
            session: scheduled.session_key(),
            session_generation: crate::engine::ManagedSessionGeneration::INITIAL,
            domains: vec![ManagedCacheDomainReservation {
                arena,
                domain: CacheDomainId::new(0),
                expected_version: 0,
                expected_committed_tokens: 0,
                execution_start_tokens: 0,
                target_committed_tokens: 1,
                target_window_start: 0,
                first_page_offset: 0,
                provisional_groups: vec![GroupBlockTable {
                    group: KvGroupId::new(0),
                    blocks: vec![block],
                }],
                writable_blocks: vec![block],
            }],
            clocked_state: None,
            allow_unchanged_prefix: false,
        };
        let lane = BatchLaneKey {
            execution_group: ExecutionGroupId::new(1),
            model_instance: ModelInstanceId::new(4),
            adapter_instance: AdapterInstanceId::new(1),
            adapter_abi: AdapterAbiRevision::new(1),
            capability_id: "chat".to_string(),
            stage_id: StageId::new(1),
            backend: BackendKind::Cpu,
            device_ordinal: None,
            compute_dtype: "f32".to_string(),
            state_dtype: "f32".to_string(),
            tensor_layout: "paged".to_string(),
            quantization: "none".to_string(),
            state_schema: "qwen3".to_string(),
            kernel_mode: "reference".to_string(),
            semantic_mode: "greedy".to_string(),
            shape_bucket: "tokens.1".to_string(),
        };
        let rows = vec![ReadyQuantum {
            plan_id: scheduled.plan_id,
            session: scheduled.session_key(),
            lane,
            work: scheduled.work.clone(),
            cost: WorkCost::new(1, 1, 0),
            managed_cache: Some(reservation.clone()),
        }];
        let binding = crate::kv::KvLayerBinding {
            model_layer: 0,
            physical_layer: 0,
        };
        let arena_config = KvArenaConfig {
            id: arena,
            group: block.group,
            page_tokens: 4,
            capacity_pages: 4,
            growth: None,
            dtype: DType::F32,
            layers: vec![KvLayerConfig {
                binding,
                num_kv_heads: 1,
                key_head_dim: 2,
                value_head_dim: 2,
            }],
        };
        let physical_arena = CpuKvBackendRuntime
            .allocate_arena(arena_config.clone())
            .unwrap();
        let slots = physical_arena
            .lower_slots(&[crate::kv::KvSlotRef { block, offset: 0 }])
            .unwrap();
        let keys = Tensor::zeros((1, 1, 2), DType::F32, &Device::Cpu).unwrap();
        let values = Tensor::zeros((1, 1, 2), DType::F32, &Device::Cpu).unwrap();
        let completion = physical_arena
            .write_slots(
                binding,
                KvWriteArgs {
                    keys: &keys,
                    values: &values,
                    slots: slots.as_ref(),
                },
            )
            .unwrap();
        let mut collector =
            KvWriteCompletionCollector::new(&arena_config, slots.logical_slots()).unwrap();
        collector.collect(completion).unwrap();
        let completion = Arc::new(collector.seal().unwrap());
        let output = ModelSessionResult::yielded(
            ExecutorOutput {
                request_id: request.id.clone(),
                audio: None,
                text: Some("done".to_string()),
                input_transcription: None,
                tokens_processed: 1,
                tokens_generated: 0,
                finished: false,
                phase_timing_override: None,
                asr_diagnostics: None,
                error: None,
            },
            crate::engine::YieldReason::QuantumExhausted,
        )
        .with_managed_cache_completions(vec![completion.clone()]);

        let result = executor
            .finish_scheduled_execution(
                &[&request],
                std::slice::from_ref(&scheduled),
                vec![output],
                BatchDispatch::serial(),
                Some(&rows),
            )
            .unwrap();
        let receipt = result[0]
            .managed_cache
            .as_ref()
            .expect("successful managed row must acknowledge writes");
        assert_eq!(receipt.reservation, reservation);
        assert_eq!(receipt.domains[0].written_blocks, vec![block]);

        let terminal = ModelSessionResult::atomic(ExecutorOutput {
            request_id: request.id.clone(),
            audio: None,
            text: Some("done".to_string()),
            input_transcription: None,
            tokens_processed: 1,
            tokens_generated: 0,
            finished: true,
            phase_timing_override: None,
            asr_diagnostics: None,
            error: None,
        })
        .with_managed_cache_completions(vec![completion]);
        let terminal = executor
            .finish_scheduled_execution(
                &[&request],
                std::slice::from_ref(&scheduled),
                vec![terminal],
                BatchDispatch::serial(),
                Some(&rows),
            )
            .unwrap();
        let terminal_receipt = terminal[0]
            .managed_cache
            .as_ref()
            .expect("completed terminal row must acknowledge its appended KV writes");
        assert_eq!(terminal_receipt.reservation, reservation);
        assert_eq!(terminal_receipt.domains[0].written_blocks, vec![block]);

        let restart = executor
            .finish_scheduled_execution(
                &[&request],
                std::slice::from_ref(&scheduled),
                vec![ModelSessionResult::restart_sequence(
                    request.id.clone(),
                    crate::engine::SequenceRestartReason::ModelFallback,
                )],
                BatchDispatch::serial(),
                Some(&rows),
            )
            .unwrap();
        assert!(matches!(
            restart[0].disposition,
            crate::engine::ExecutionDisposition::RestartSequence(_)
        ));
        assert!(restart[0].managed_cache.is_none());
        assert!(restart[0].managed_cache_completions.is_empty());
        assert!(restart[0].staged_stream_outputs.is_empty());
        assert_eq!(restart[0].output.tokens_processed, 0);
        assert_eq!(restart[0].output.tokens_generated, 0);

        let mut invalid_restart = ModelSessionResult::restart_sequence(
            request.id.clone(),
            crate::engine::SequenceRestartReason::ModelFallback,
        );
        invalid_restart.output.text = Some("must not escape".to_string());
        let invalid_restart = executor
            .finish_scheduled_execution(
                &[&request],
                std::slice::from_ref(&scheduled),
                vec![invalid_restart],
                BatchDispatch::serial(),
                Some(&rows),
            )
            .unwrap();
        assert!(matches!(
            invalid_restart[0].disposition,
            crate::engine::ExecutionDisposition::Failed(_)
        ));
        assert!(invalid_restart[0].managed_cache.is_none());
    }

    #[test]
    fn selected_tensor_only_completion_crosses_production_dispatch_exactly() {
        let executor = NativeExecutor::new(super::super::WorkerConfig::default());
        let mut request = EngineCoreRequest::tts("clocked receipt");
        request.id = "clocked-receipt".to_string();
        let mut scheduled = scheduled(&request.id, 71);
        let group = StateGroupId::new(2);
        let clock = StateClock::AudioSamples;
        scheduled.work = WorkUnit::SequenceStep {
            phase: SequencePhase::Prefill,
            input: InputRange::new(0, 1).unwrap(),
            max_output_steps: 1,
            auxiliary_state: Some(Arc::from([ClockedStateSpan::new(
                group,
                clock.clone(),
                InputRange::new(0, 3_200).unwrap(),
            )
            .unwrap()])),
        };
        let physical_selection = TensorStateSelection {
            group,
            clock,
            expected_cursor: 0,
            target_cursor: 3_200,
        };
        let model_instance = ModelInstanceId::new(4);
        let sequence = PhysicalStateSequenceId::new(9).unwrap();
        let transaction = PhysicalStateTransactionId::new(scheduled.plan_id).unwrap();
        let completion = TensorStateBatchCompletion::for_dispatch_test(
            transaction,
            sequence,
            Arc::from([physical_selection.clone()]),
        );
        let reservation = ManagedCacheReservation {
            txn_id: scheduled.plan_id,
            session: scheduled.session_key(),
            session_generation: crate::engine::ManagedSessionGeneration::INITIAL,
            domains: vec![],
            clocked_state: Some(
                ManagedClockedStateReservation::selected(
                    model_instance,
                    sequence.get(),
                    Arc::from([physical_selection]),
                )
                .unwrap(),
            ),
            allow_unchanged_prefix: false,
        };
        let rows = vec![ReadyQuantum {
            plan_id: scheduled.plan_id,
            session: scheduled.session_key(),
            lane: BatchLaneKey {
                execution_group: ExecutionGroupId::new(1),
                model_instance,
                adapter_instance: AdapterInstanceId::new(1),
                adapter_abi: AdapterAbiRevision::new(1),
                capability_id: "asr".into(),
                stage_id: StageId::new(1),
                backend: BackendKind::Cpu,
                device_ordinal: None,
                compute_dtype: "f32".into(),
                state_dtype: "f32".into(),
                tensor_layout: "tensor".into(),
                quantization: "none".into(),
                state_schema: "clocked".into(),
                kernel_mode: "reference".into(),
                semantic_mode: "greedy".into(),
                shape_bucket: "audio.3200".into(),
            },
            work: scheduled.work.clone(),
            cost: WorkCost::new(1, 1, 0),
            managed_cache: Some(reservation),
        }];
        let output = ModelSessionResult::yielded(
            ExecutorOutput {
                request_id: request.id.clone(),
                audio: None,
                text: None,
                input_transcription: None,
                tokens_processed: 1,
                tokens_generated: 0,
                finished: false,
                phase_timing_override: None,
                asr_diagnostics: None,
                error: None,
            },
            crate::engine::YieldReason::QuantumExhausted,
        )
        .with_clocked_state_completion(completion.clone());
        let exact = executor
            .finish_scheduled_execution(
                &[&request],
                std::slice::from_ref(&scheduled),
                vec![output],
                BatchDispatch::serial(),
                Some(&rows),
            )
            .unwrap();
        assert_eq!(
            exact[0]
                .managed_cache
                .as_ref()
                .and_then(|receipt| receipt.clocked_state())
                .map(|receipt| receipt.completion()),
            Some(&completion)
        );

        scheduled.num_tokens = 2;
        let partial = ModelSessionResult::yielded(
            ExecutorOutput {
                request_id: request.id.clone(),
                audio: None,
                text: None,
                input_transcription: None,
                tokens_processed: 1,
                tokens_generated: 0,
                finished: false,
                phase_timing_override: None,
                asr_diagnostics: None,
                error: None,
            },
            crate::engine::YieldReason::QuantumExhausted,
        )
        .with_clocked_state_completion(completion);
        let partial = executor
            .finish_scheduled_execution(
                &[&request],
                std::slice::from_ref(&scheduled),
                vec![partial],
                BatchDispatch::serial(),
                Some(&rows),
            )
            .unwrap();
        assert!(partial[0].output.error.as_deref().is_some_and(|message| {
            message.contains("selected auxiliary state requires exact scheduled progress")
        }));
        assert!(partial[0].managed_cache.is_none());
    }

    #[test]
    fn one_backend_batch_completion_authenticates_each_ragged_row_subset() {
        let arena = KvArenaId {
            model_instance: ModelInstanceId::new(9),
            backend: BackendKind::Cpu,
            device_ordinal: None,
            generation: 1,
        };
        let binding = crate::kv::KvLayerBinding {
            model_layer: 0,
            physical_layer: 0,
        };
        let config = KvArenaConfig {
            id: arena,
            group: KvGroupId::new(2),
            page_tokens: 4,
            capacity_pages: 2,
            growth: None,
            dtype: DType::F32,
            layers: vec![KvLayerConfig {
                binding,
                num_kv_heads: 1,
                key_head_dim: 2,
                value_head_dim: 2,
            }],
        };
        let physical_arena = CpuKvBackendRuntime.allocate_arena(config.clone()).unwrap();
        let blocks = (0..2)
            .map(|index| CacheBlockRef {
                arena,
                group: config.group,
                index,
                slot_generation: 1,
            })
            .collect::<Vec<_>>();
        let slot_refs = blocks
            .iter()
            .map(|block| crate::kv::KvSlotRef {
                block: *block,
                offset: 0,
            })
            .collect::<Vec<_>>();
        let slots = physical_arena.lower_slots(&slot_refs).unwrap();
        let keys = Tensor::zeros((2, 1, 2), DType::F32, &Device::Cpu).unwrap();
        let values = Tensor::zeros((2, 1, 2), DType::F32, &Device::Cpu).unwrap();
        let completion = physical_arena
            .write_slots(
                binding,
                KvWriteArgs {
                    keys: &keys,
                    values: &values,
                    slots: slots.as_ref(),
                },
            )
            .unwrap();
        let mut collector =
            KvWriteCompletionCollector::new(&config, slots.logical_slots()).unwrap();
        collector.collect(completion).unwrap();
        let completion = Arc::new(collector.seal().unwrap());

        for (row, block) in blocks.into_iter().enumerate() {
            let reservation = ManagedCacheReservation {
                txn_id: row as u64 + 1,
                session: crate::engine::SessionKey {
                    request_id: format!("row-{row}"),
                    epoch: 1,
                },
                session_generation: crate::engine::ManagedSessionGeneration::INITIAL,
                domains: vec![ManagedCacheDomainReservation {
                    arena,
                    domain: CacheDomainId::new(1),
                    expected_version: 0,
                    expected_committed_tokens: 0,
                    execution_start_tokens: 0,
                    target_committed_tokens: 1,
                    target_window_start: 0,
                    first_page_offset: 0,
                    provisional_groups: vec![GroupBlockTable {
                        group: config.group,
                        blocks: vec![block],
                    }],
                    writable_blocks: vec![block],
                }],
                clocked_state: None,
                allow_unchanged_prefix: false,
            };
            let receipt = reservation
                .completed_write_receipt(std::slice::from_ref(&completion))
                .unwrap();
            assert_eq!(receipt.domains[0].written_blocks, vec![block]);
        }
    }
}
