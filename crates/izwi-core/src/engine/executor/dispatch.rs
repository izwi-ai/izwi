use std::thread;

use tracing::error;

use crate::backends::can_parallelize_requests;
use crate::error::{Error, Result};
use crate::model::ModelVariant;

use super::super::request::EngineCoreRequest;
use super::super::scheduler::ScheduledRequest;
use super::super::types::TaskType;
use super::{ExecutorOutput, ExecutorStepResult, ModelSessionResult, NativeExecutor};
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
    ) -> ModelSessionResult {
        let Some(request) = Self::find_request(requests, scheduled_req) else {
            return ModelSessionResult::atomic(ExecutorOutput::error(
                scheduled_req.request_id.clone(),
                "Scheduled request not found in batch",
            ));
        };

        if request.is_cancelled() {
            return ModelSessionResult::cancelled(ExecutorOutput::cancelled(request.id.clone()));
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

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            (route.handler)(self, request, scheduled_req)
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
                Err(Error::InferenceError(format!(
                    "Executor request handling panicked: {message}"
                )))
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
                        let output = self.execute_single_request(requests, &scheduled_req);
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
        // Reserve model-owned cache growth before any tensor state can be
        // allocated. The lease remains attached to the exact session until
        // finish, abort, failure cleanup, or recompute preemption.
        self.reserve_scheduled_cache(requests, scheduled)?;
        let (outputs, dispatch) = if let Some(result) = self.try_qwen_tts_batch(requests, scheduled)
        {
            let result = result?;
            let dispatch = if result.tensor_width > 1 {
                BatchDispatch::new(BatchDispatchKind::TensorStatic, result.tensor_width)
            } else {
                BatchDispatch::serial()
            };
            (
                result
                    .outputs
                    .into_iter()
                    .map(ModelSessionResult::atomic)
                    .collect(),
                dispatch,
            )
        } else if self.can_parallelize_requests(scheduled.len()) {
            (
                self.execute_requests_parallel(requests, scheduled)?,
                BatchDispatch::new(BatchDispatchKind::RequestParallel, scheduled.len()),
            )
        } else {
            (
                scheduled
                    .iter()
                    .map(|scheduled_req| self.execute_single_request(requests, scheduled_req))
                    .collect(),
                BatchDispatch::serial(),
            )
        };
        Ok(scheduled
            .iter()
            .zip(outputs)
            .map(|(scheduled, output)| {
                ExecutorStepResult::from_session(scheduled, output).with_dispatch(dispatch)
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

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
            block_ids: Vec::new(),
            num_computed_tokens: 0,
            work: crate::engine::WorkUnit::SequenceStep {
                phase: crate::engine::SequencePhase::Prefill,
                input: crate::engine::InputRange { start: 0, end: 1 },
                max_output_steps: 1,
            },
        };

        let result = executor.execute_single_request(&[&request], &scheduled);
        assert!(result.output.finished);
        assert!(result.output.error.is_none());
        assert_eq!(
            result.disposition,
            crate::engine::ExecutionDisposition::Finished(crate::engine::FinishReason::Cancelled)
        );
    }
}
