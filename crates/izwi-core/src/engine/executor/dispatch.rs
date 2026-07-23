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
                    let cache =
                        super::qwen3_managed_cache_for_row(request, scheduled_req, reservation)?;
                    self.chat_request_with_managed_cache(
                        request,
                        scheduled_req,
                        Some(cache),
                        reservation.tensor_state,
                    )
                }
                Some(reservation) if request.task_type == TaskType::ASR => {
                    let cache =
                        super::qwen3_managed_cache_for_row(request, scheduled_req, reservation)?;
                    self.transcribe_request_with_managed_cache(request, scheduled_req, Some(cache))
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
                        reservation.tensor_state,
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
                        super::qwen3_managed_cache_for_row(request, scheduled, reservation)
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

    fn finish_scheduled_execution(
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
                if result.output.error.is_none()
                    && result.provenance.dispatch_state == super::DispatchState::ProducedOutput
                    && matches!(
                        result.disposition,
                        super::ExecutionDisposition::Progress
                            | super::ExecutionDisposition::Yielded(_)
                    )
                {
                    if let Some(reservation) = rows
                        .and_then(|rows| rows.iter().find(|row| row.plan_id == scheduled.plan_id))
                        .and_then(|row| row.managed_cache.as_ref())
                    {
                        match reservation
                            .completed_write_receipt(&result.managed_cache_completions)
                        {
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
                    } else if !result.managed_cache_completions.is_empty() {
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
    use crate::backends::BackendKind;
    use crate::engine::cache::coordinator::GroupBlockTable;
    use crate::engine::{
        AdapterAbiRevision, AdapterInstanceId, BatchDispatchKind, BatchLaneKey, ExecutionGroupId,
        InputRange, ManagedCacheDomainReservation, ManagedCacheReservation, ModelInstanceId,
        SequencePhase, StageId, WorkCost, WorkUnit,
    };
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
            let mut config = super::super::WorkerConfig::default();
            config.backend = backend;
            config.request_parallelism = 2;
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
            tensor_state: None,
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
            tokens_processed: 0,
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
        assert!(terminal[0].managed_cache.is_none());
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
                tensor_state: None,
            };
            let receipt = reservation
                .completed_write_receipt(std::slice::from_ref(&completion))
                .unwrap();
            assert_eq!(receipt.domains[0].written_blocks, vec![block]);
        }
    }
}
