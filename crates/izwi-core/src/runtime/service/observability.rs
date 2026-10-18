//! Runtime observability: broker and engine observations, telemetry snapshots,
//! Prometheus export and voice session metrics.

use super::*;

pub(super) fn managed_kv_used_bytes(snapshot: &crate::engine::ManagedKvRuntimeSnapshot) -> u64 {
    snapshot
        .models
        .iter()
        .flat_map(|model| &model.arenas)
        .map(|arena| {
            arena
                .coordinator
                .allocated_pages
                .saturating_mul(arena.bytes_per_page)
        })
        .fold(0_u64, u64::saturating_add)
}

/// Per-arena page gauges labeled by model, backend, domain and storage dtype,
/// so pools of different models are never summed into one utilization.
pub(super) fn push_managed_kv_arena_metrics(
    payload: &mut String,
    snapshot: &crate::engine::ManagedKvRuntimeSnapshot,
    model_label: impl Fn(crate::engine::ModelInstanceId) -> String,
) {
    let arenas = snapshot
        .models
        .iter()
        .flat_map(|model| {
            let label = model_label(model.model_instance);
            model.arenas.iter().map(move |arena| {
                let labels = vec![
                    ("model", label.clone()),
                    ("backend", model.backend.as_str().to_string()),
                    ("domain", arena.domain_id.to_string()),
                    ("storage_format", arena.storage_dtype.clone()),
                ];
                (labels, arena)
            })
        })
        .collect::<Vec<_>>();
    type ArenaGauge = fn(&crate::engine::ManagedKvArenaRuntimeSnapshot) -> u64;
    let gauges: [(&str, ArenaGauge); 5] = [
        (ENGINE_KV_MODEL_CAPACITY_PAGES, |arena| {
            arena.coordinator.capacity_pages
        }),
        (ENGINE_KV_MODEL_ALLOCATED_PAGES, |arena| {
            arena.coordinator.allocated_pages
        }),
        (ENGINE_KV_MODEL_PREFIX_RETAINED_PAGES, |arena| {
            arena.coordinator.prefix_refs
        }),
        (ENGINE_KV_MODEL_ADMISSION_CLAIMED_PAGES, |arena| {
            arena.coordinator.admission_claimed_pages
        }),
        (ENGINE_KV_MODEL_ADMISSION_AVAILABLE_PAGES, |arena| {
            arena.coordinator.admission_available_pages
        }),
    ];
    for (name, value) in gauges {
        let series = arenas
            .iter()
            .map(|(labels, arena)| (labels.clone(), value(arena)))
            .collect::<Vec<_>>();
        push_engine_multi_labeled_metric(payload, name, &series);
    }
}

pub(super) fn managed_kv_device_pages(snapshot: &crate::engine::ManagedKvRuntimeSnapshot) -> u64 {
    snapshot
        .models
        .iter()
        .filter(|model| model.backend != BackendKind::Cpu)
        .flat_map(|model| &model.arenas)
        .map(|arena| arena.coordinator.allocated_pages)
        .fold(0_u64, u64::saturating_add)
}

pub(super) fn engine_output_observation(
    context: RuntimeObservationContext,
    admission_ms: Option<f64>,
    output: &EngineOutput,
) -> RuntimeStageObservation {
    let mut timing = RuntimeStageTiming {
        admission_ms,
        total_ms: Some(output.generation_time.as_secs_f64() * 1000.0),
        ..RuntimeStageTiming::default()
    };
    if let Some(latency) = output.latency_breakdown.as_ref() {
        // Queue, first output and total run from runtime admission, so
        // they include the coordinator and model-residency time before the
        // engine saw the request.
        let pre_engine_ms = latency.pre_engine_ms.unwrap_or(0.0);
        timing.queue_wait_ms = Some(pre_engine_ms + latency.queue_wait_ms);
        timing.media_decode_ms = latency.media_decode_ms;
        timing.normalization_ms = latency.normalization_ms;
        timing.prefill_ms = Some(latency.prefill_ms);
        timing.decode_ms = Some(latency.decode_ms);
        timing.ttft_ms = latency.ttft_ms.map(|ttft_ms| pre_engine_ms + ttft_ms);
        timing.sampling_ms = latency.sampling_ms;
        timing.codec_ms = latency.codec_ms;
        timing.postprocess_ms = latency.postprocess_ms;
        timing.total_ms = Some(pre_engine_ms + latency.total_ms);
    }

    let outcome = if output.error.is_some() {
        RuntimeStageOutcome::Failed
    } else {
        RuntimeStageOutcome::Completed
    };
    let mut observation = RuntimeStageObservation::new(context, outcome);
    observation.timing = timing;
    observation.outputs = RuntimeStageOutputCounters {
        prompt_tokens: Some(output.token_stats.prompt_tokens as u64),
        generated_tokens: Some(output.token_stats.generated_tokens as u64),
        audio_samples: Some(output.audio.samples.len() as u64),
        transcript_chars: output.text.as_ref().map(|text| text.chars().count() as u64),
        stop_reason: output.finish_reason.map(|reason| format!("{reason:?}")),
        ..RuntimeStageOutputCounters::default()
    };
    if let Some(error) = output.error.as_ref() {
        observation.error_kind = Some(error.clone());
    }
    observation
}

pub(super) fn engine_error_observation(
    context: RuntimeObservationContext,
    admission_ms: Option<f64>,
    error_kind: impl Into<String>,
) -> RuntimeStageObservation {
    let mut observation = RuntimeStageObservation::new(context, RuntimeStageOutcome::Failed)
        .with_error_kind(error_kind);
    observation.timing.admission_ms = admission_ms;
    observation
}

impl RuntimeService {
    pub fn record_stage_observation(&self, observation: RuntimeStageObservation) {
        self.telemetry.record_stage_observation(observation);
    }

    pub(super) fn observe_broker_request(&self, request: &EngineCoreRequest) -> Result<()> {
        self.observe_broker_request_with_streaming_required(request, request.streaming)
    }

    pub(super) fn observe_broker_request_with_streaming_required(
        &self,
        request: &EngineCoreRequest,
        streaming_required: bool,
    ) -> Result<()> {
        let Some(observation) = self
            .inference_broker
            .observe_engine_request_with_streaming_required(
                request,
                streaming_required,
                &self.adapter_registry,
                &self.backend_router,
            )
        else {
            return Ok(());
        };

        self.record_broker_observation(observation)
    }

    pub(super) fn observe_broker_request_with_transport_streaming(
        &self,
        request: &EngineCoreRequest,
    ) -> Result<()> {
        self.observe_broker_request_with_streaming_required(request, false)
    }

    pub(crate) fn observe_broker_capability_request(
        &self,
        capability: CapabilityKind,
        model_variant: Option<ModelVariant>,
        streaming_required: bool,
    ) -> Result<()> {
        let Some(observation) = self.inference_broker.observe_capability_request(
            RouteSource::InternalRuntime,
            capability,
            model_variant,
            streaming_required,
            &self.adapter_registry,
            &self.backend_router,
        ) else {
            return Ok(());
        };

        self.record_broker_observation(observation)
    }

    pub(super) fn record_broker_observation(&self, observation: InferenceBrokerObservation) -> Result<()> {
        if observation.shadow_enabled {
            self.telemetry.record_broker_shadow_request();
        }
        if observation.execution_enabled {
            self.telemetry.record_broker_execution_request();
        }
        if observation.routing_decision.is_some() {
            self.telemetry.record_broker_route_decision();
        }

        if let Some(message) = observation.validation_error {
            self.telemetry.record_broker_validation_failure();
            self.telemetry.record_stage_observation(
                RuntimeStageObservation::new(
                    RuntimeObservationContext {
                        route_source: Some(format!("{:?}", observation.source)),
                        capability: Some(format!("{:?}", observation.capability)),
                        model_variant: observation
                            .model_variant
                            .map(|variant| variant.dir_name().to_string()),
                        pipeline_stage: Some("runtime.routing".to_string()),
                        ..RuntimeObservationContext::default()
                    },
                    RuntimeStageOutcome::Failed,
                )
                .with_error_kind("routing_validation_failed"),
            );
            if observation.execution_enabled {
                return Err(Error::InvalidInput(message));
            }
            debug!(
                source = ?observation.source,
                capability = ?observation.capability,
                model_variant = ?observation.model_variant,
                "Inference broker shadow validation failed: {message}"
            );
        } else if let Some(decision) = observation.routing_decision {
            self.telemetry
                .record_stage_observation(RuntimeStageObservation::new(
                    RuntimeObservationContext {
                        route_source: Some(format!("{:?}", observation.source)),
                        capability: Some(format!("{:?}", observation.capability)),
                        model_variant: Some(decision.selected_model_variant.dir_name().to_string()),
                        backend_kind: Some(decision.backend_kind.as_str().to_string()),
                        execution_target: Some(format!(
                            "{:?}",
                            decision.execution_plan.execution_target
                        )),
                        streaming_mode: Some(format!(
                            "{:?}",
                            decision.execution_plan.streaming_mode
                        )),
                        pipeline_stage: Some("runtime.routing".to_string()),
                        ..RuntimeObservationContext::default()
                    },
                    RuntimeStageOutcome::Observed,
                ));
            debug!(
                source = ?observation.source,
                capability = ?observation.capability,
                requested_model_variant = ?observation.model_variant,
                selected_model_variant = ?decision.selected_model_variant,
                execution_target = ?decision.execution_plan.execution_target,
                backend_kind = ?decision.backend_kind,
                "Inference broker route decision recorded"
            );
        }

        Ok(())
    }

    pub(super) fn engine_observation_context(
        &self,
        request: &EngineCoreRequest,
        streaming: bool,
    ) -> RuntimeObservationContext {
        RuntimeObservationContext {
            route_source: Some(format!("{:?}", RouteSource::InternalEngine)),
            capability: Some(request.task_type.as_str().to_string()),
            model_variant: request
                .model_variant
                .map(|variant| variant.dir_name().to_string()),
            backend_kind: Some(
                self.backend_router
                    .default_backend()
                    .kind()
                    .as_str()
                    .to_string(),
            ),
            pipeline_stage: Some(
                if streaming {
                    ENGINE_STREAMING_REQUEST_STAGE
                } else {
                    ENGINE_REQUEST_STAGE
                }
                .to_string(),
            ),
            workload_class: Some(request.workload_class.as_str().to_string()),
            request_id: Some(request.id.clone()),
            correlation_id: request.correlation_id.clone(),
            ..RuntimeObservationContext::default()
        }
    }

    pub(super) fn record_engine_output_observation(
        &self,
        request: &EngineCoreRequest,
        output: &EngineOutput,
        streaming: bool,
    ) {
        self.telemetry
            .record_stage_observation(engine_output_observation(
                self.engine_observation_context(request, streaming),
                request.admission_ms,
                output,
            ));
    }

    pub(super) fn record_engine_error_observation(
        &self,
        request: &EngineCoreRequest,
        streaming: bool,
        error_kind: impl Into<String>,
    ) {
        self.telemetry
            .record_stage_observation(engine_error_observation(
                self.engine_observation_context(request, streaming),
                request.admission_ms,
                error_kind,
            ));
    }

    /// Snapshot of runtime/engine telemetry (queue/prefill/decode/worker health).
    pub async fn telemetry_snapshot(&self) -> RuntimeTelemetrySnapshot {
        let mut snapshot = self.telemetry.snapshot().await;
        snapshot.engine = self.engine_telemetry_snapshot().await;
        snapshot.coordinator = self.coordinator.snapshot();
        snapshot.models = self.loaded_model_diagnostics().await;
        snapshot
    }

    /// Prometheus exposition format telemetry payload.
    pub async fn telemetry_prometheus(&self) -> String {
        let mut payload = self.telemetry.prometheus().await;
        self.push_engine_prometheus_metrics(&mut payload).await;
        self.push_coordinator_prometheus_metrics(&mut payload);
        payload
    }

    pub fn coordinator_snapshot(&self) -> CoordinatorSnapshot {
        self.coordinator.snapshot()
    }

    /// Uploads the diarization pipeline has decoded itself; a job given
    /// decoded samples adds none (PIPE-14).
    pub fn diarization_audio_decodes(&self) -> u64 {
        self.diarization_audio_decodes.load(Ordering::Relaxed)
    }

    /// Physical execution health: admission poison and latched device faults.
    /// Cheap enough for liveness and worker status polling.
    pub fn health(&self) -> RuntimeHealth {
        self.coordinator.health()
    }

    /// Fault injection for probe and status tests in dependent crates: latch
    /// `error` through the same classification device execution uses. It
    /// fails this runtime only, never the process-wide resource authority
    /// that runtimes running alongside it share.
    #[doc(hidden)]
    pub fn latch_device_fault_for_test(&self, error: &Error) -> bool {
        self.coordinator.latch_fatal_device_fault_for_test(error)
    }

    pub(super) async fn engine_telemetry_snapshot(&self) -> EngineRuntimeTelemetrySnapshot {
        let queue_depth = self.core_engine.pending_requests().await as u64;
        let running_requests = self.core_engine.running_requests().await as u64;
        let kv_cache = self.core_engine.kv_cache_snapshot().await;
        let stream = engine_stream_metrics_snapshot();

        let batch = engine_batch_metrics_snapshot();
        EngineRuntimeTelemetrySnapshot {
            chat_concurrency_policy: self.chat_concurrency_policy(),
            scheduler_queue_depth: queue_depth,
            scheduler_running_requests: running_requests,
            incremental_prefill_quanta_committed_total: batch
                .incremental_prefill_quanta_committed_total,
            incremental_prefill_tokens_committed_total: batch
                .incremental_prefill_tokens_committed_total,
            multispan_prefill_requests_total: batch.multispan_prefill_requests_total,
            stream_backpressure_total: stream.backpressure_total,
            stream_checkpoints_committed_total: stream.checkpoints_committed_total,
            stream_checkpoint_rejections_total: stream.checkpoint_rejections_total,
            stream_delivery_failures_total: stream.delivery_failures_total,
            tensor_batches_total: batch.tensor_batches_total,
            tensor_static_batches_total: batch.tensor_static_batches_total,
            tensor_continuous_batches_total: batch.tensor_continuous_batches_total,
            tensor_continuous_multirow_batches_total: batch
                .tensor_continuous_multirow_batches_total,
            request_parallel_batches_total: batch.request_parallel_batches_total,
            physical_batch_rejections_total: batch.physical_batch_rejections_total,
            tensor_batch_max_width: batch.tensor_batch_max_width,
            tensor_batch_rows_total: batch.tensor_batch_rows_total,
            tensor_batch_capacity_rows_total: batch.tensor_batch_capacity_rows_total,
            tensor_batch_useful_elements_total: batch.tensor_batch_useful_elements_total,
            tensor_batch_materialized_elements_total: batch
                .tensor_batch_materialized_elements_total,
            batch_workspace_bytes_total: batch.batch_workspace_bytes_total,
            dispatch_states: batch.dispatch_states,
            failure_origins: batch.failure_origins,
            deadline_phases: batch.deadline_phases,
            workspace_domains: batch.workspace_domains,
            tensor_batch_fill_ratio: batch.tensor_batch_fill_ratio,
            tensor_batch_padding_ratio: batch.tensor_batch_padding_ratio,
            model_tensor_batches_total: batch.model_tensor_batches_total,
            model_tensor_batch_rows_total: batch.model_tensor_batch_rows_total,
            model_tensor_batch_max_width: batch.model_tensor_batch_max_width,
            model_scalar_row_dispatches_total: batch.model_scalar_row_dispatches_total,
            model_decode_calls_total: batch.model_decode_calls_total,
            model_tensor_multirow_calls_total: batch.model_tensor_multirow_calls_total,
            model_tensor_batch_width_counts: batch.model_tensor_batch_width_counts,
            capacity_suspensions_total: batch.capacity_suspensions_total,
            capacity_replay_tokens_total: batch.capacity_replay_tokens_total,
            capacity_replay_capped_total: batch.capacity_replay_capped_total,
            recompute_preemptions_total: batch.recompute_preemptions_total,
            claim_deadlock_aborts_total: batch.claim_deadlock_aborts_total,
            kv_blocked_requests: batch.kv_blocked_requests,
            continuous_envelope_scalar_fallbacks_total: batch
                .continuous_envelope_scalar_fallbacks_total,
            physical_execution: batch.physical_execution,
            kv_cache,
        }
    }

    pub(super) async fn push_engine_prometheus_metrics(&self, payload: &mut String) {
        let snapshot = self.engine_telemetry_snapshot().await;
        crate::engine::push_engine_histograms(payload);
        push_engine_metric(
            payload,
            ENGINE_SCHEDULER_QUEUE_DEPTH,
            snapshot.scheduler_queue_depth,
        );
        push_engine_metric(
            payload,
            ENGINE_SCHEDULER_RUNNING_REQUESTS,
            snapshot.scheduler_running_requests,
        );
        push_engine_metric(
            payload,
            ENGINE_SCHEDULER_INCREMENTAL_PREFILL_QUANTA_COMMITTED_TOTAL,
            snapshot.incremental_prefill_quanta_committed_total,
        );
        push_engine_metric(
            payload,
            ENGINE_SCHEDULER_INCREMENTAL_PREFILL_TOKENS_COMMITTED_TOTAL,
            snapshot.incremental_prefill_tokens_committed_total,
        );
        push_engine_metric(
            payload,
            ENGINE_SCHEDULER_MULTISPAN_PREFILL_REQUESTS_TOTAL,
            snapshot.multispan_prefill_requests_total,
        );
        push_engine_metric(
            payload,
            ENGINE_KV_CACHE_HITS_TOTAL,
            snapshot.kv_cache.counters.prefix_hits,
        );
        push_engine_metric(
            payload,
            ENGINE_KV_CACHE_MISSES_TOTAL,
            snapshot.kv_cache.counters.prefix_misses,
        );
        push_engine_metric(
            payload,
            ENGINE_KV_CACHE_EVICTIONS_TOTAL,
            snapshot.kv_cache.counters.prefix_evictions,
        );
        push_engine_metric(
            payload,
            ENGINE_KV_HOST_PAGES,
            snapshot.kv_cache.counters.kv_host_pages,
        );
        push_engine_metric(
            payload,
            ENGINE_KV_DEMOTIONS_TOTAL,
            snapshot.kv_cache.counters.demotions_total,
        );
        push_engine_metric(
            payload,
            ENGINE_KV_PROMOTIONS_TOTAL,
            snapshot.kv_cache.counters.promotions_total,
        );
        push_engine_metric_f64(
            payload,
            ENGINE_KV_PROMOTION_LATENCY_AVG_SECONDS,
            if snapshot.kv_cache.counters.promotions_total > 0 {
                snapshot.kv_cache.counters.promotion_latency_ns_total as f64
                    / snapshot.kv_cache.counters.promotions_total as f64
                    / 1e9
            } else {
                0.0
            },
        );
        push_engine_labeled_metric(
            payload,
            ENGINE_KV_CACHE_ALLOCATED_BLOCKS,
            "accounting",
            &[(
                "physical_pages",
                snapshot.kv_cache.totals.coordinator.allocated_pages,
            )],
        );
        push_engine_labeled_metric(
            payload,
            ENGINE_KV_CACHE_FREE_BLOCKS,
            "accounting",
            &[(
                "physical_pages",
                snapshot.kv_cache.totals.coordinator.free_pages,
            )],
        );
        push_engine_labeled_metric_f64(
            payload,
            ENGINE_KV_CACHE_UTILIZATION_RATIO,
            "accounting",
            &[(
                "physical_pages",
                if snapshot.kv_cache.totals.coordinator.capacity_pages == 0 {
                    0.0
                } else {
                    snapshot.kv_cache.totals.coordinator.allocated_pages as f64
                        / snapshot.kv_cache.totals.coordinator.capacity_pages as f64
                },
            )],
        );
        push_engine_labeled_metric(
            payload,
            ENGINE_KV_CACHE_MEMORY_USED_BYTES,
            "accounting",
            &[(
                "physical_managed_pages",
                managed_kv_used_bytes(&snapshot.kv_cache),
            )],
        );
        push_engine_labeled_metric(
            payload,
            ENGINE_KV_CACHE_MEMORY_CAPACITY_BYTES,
            "accounting",
            &[(
                snapshot.kv_cache.memory_accounting,
                snapshot.kv_cache.totals.physical_bytes,
            )],
        );
        push_engine_labeled_metric(
            payload,
            ENGINE_KV_CACHE_GPU_RESIDENT_BLOCKS,
            "accounting",
            &[(
                "physical_device_pages",
                managed_kv_device_pages(&snapshot.kv_cache),
            )],
        );
        push_engine_metric(
            payload,
            ENGINE_KV_PREFIX_QUERY_TOKENS_TOTAL,
            snapshot.kv_cache.counters.prefix_query_tokens,
        );
        push_engine_metric(
            payload,
            ENGINE_KV_PREFIX_HIT_TOKENS_TOTAL,
            snapshot.kv_cache.counters.prefix_hit_tokens,
        );
        let model_label = |model_instance: crate::engine::ModelInstanceId| {
            self.model_registry
                .kv_instance_variant(model_instance)
                .map_or_else(
                    || format!("instance-{}", model_instance.get()),
                    |variant| variant.to_string(),
                )
        };
        push_managed_kv_arena_metrics(payload, &snapshot.kv_cache, model_label);
        push_engine_metric(
            payload,
            ENGINE_STREAM_BACKPRESSURE_TOTAL,
            snapshot.stream_backpressure_total,
        );
        push_engine_metric(
            payload,
            ENGINE_STREAM_CHECKPOINTS_COMMITTED_TOTAL,
            snapshot.stream_checkpoints_committed_total,
        );
        push_engine_metric(
            payload,
            ENGINE_STREAM_CHECKPOINT_REJECTIONS_TOTAL,
            snapshot.stream_checkpoint_rejections_total,
        );
        push_engine_metric(
            payload,
            ENGINE_STREAM_DELIVERY_FAILURES_TOTAL,
            snapshot.stream_delivery_failures_total,
        );
        push_engine_metric(
            payload,
            ENGINE_EXECUTOR_TENSOR_BATCHES_TOTAL,
            snapshot.tensor_batches_total,
        );
        push_engine_metric(
            payload,
            ENGINE_EXECUTOR_REQUEST_PARALLEL_BATCHES_TOTAL,
            snapshot.request_parallel_batches_total,
        );
        push_engine_metric(
            payload,
            ENGINE_EXECUTOR_TENSOR_BATCH_MAX_WIDTH,
            snapshot.tensor_batch_max_width,
        );
        push_engine_metric(
            payload,
            ENGINE_EXECUTOR_TENSOR_STATIC_BATCHES_TOTAL,
            snapshot.tensor_static_batches_total,
        );
        push_engine_metric(
            payload,
            ENGINE_EXECUTOR_TENSOR_CONTINUOUS_BATCHES_TOTAL,
            snapshot.tensor_continuous_batches_total,
        );
        push_engine_metric(
            payload,
            ENGINE_EXECUTOR_TENSOR_CONTINUOUS_MULTIROW_BATCHES_TOTAL,
            snapshot.tensor_continuous_multirow_batches_total,
        );
        push_engine_metric(
            payload,
            ENGINE_EXECUTOR_MODEL_TENSOR_BATCHES_TOTAL,
            snapshot.model_tensor_batches_total,
        );
        push_engine_metric(
            payload,
            ENGINE_EXECUTOR_MODEL_TENSOR_BATCH_ROWS_TOTAL,
            snapshot.model_tensor_batch_rows_total,
        );
        push_engine_metric(
            payload,
            ENGINE_EXECUTOR_MODEL_TENSOR_BATCH_MAX_WIDTH,
            snapshot.model_tensor_batch_max_width,
        );
        push_engine_metric(
            payload,
            ENGINE_EXECUTOR_MODEL_SCALAR_ROW_DISPATCHES_TOTAL,
            snapshot.model_scalar_row_dispatches_total,
        );
        push_engine_metric(
            payload,
            ENGINE_EXECUTOR_MODEL_DECODE_CALLS_TOTAL,
            snapshot.model_decode_calls_total,
        );
        push_engine_metric(
            payload,
            ENGINE_EXECUTOR_MODEL_TENSOR_MULTIROW_CALLS_TOTAL,
            snapshot.model_tensor_multirow_calls_total,
        );
        let width_labels = snapshot
            .model_tensor_batch_width_counts
            .iter()
            .map(|(width, count)| (width.to_string(), *count))
            .collect::<Vec<_>>();
        let width_values = width_labels
            .iter()
            .map(|(label, count)| (label.as_str(), *count))
            .collect::<Vec<_>>();
        push_engine_labeled_metric(
            payload,
            ENGINE_EXECUTOR_MODEL_TENSOR_BATCH_WIDTH_CALLS_TOTAL,
            "width",
            &width_values,
        );
        push_engine_metric(
            payload,
            ENGINE_SCHEDULER_CAPACITY_SUSPENSIONS_TOTAL,
            snapshot.capacity_suspensions_total,
        );
        push_engine_metric(
            payload,
            ENGINE_SCHEDULER_CAPACITY_REPLAY_TOKENS_TOTAL,
            snapshot.capacity_replay_tokens_total,
        );
        push_engine_labeled_metric(
            payload,
            ENGINE_SCHEDULER_PREEMPTIONS_TOTAL,
            "kind",
            &[
                ("replay", snapshot.capacity_suspensions_total),
                ("replay_capped", snapshot.capacity_replay_capped_total),
                ("recompute", snapshot.recompute_preemptions_total),
                ("abort", snapshot.claim_deadlock_aborts_total),
            ],
        );
        push_engine_metric(
            payload,
            ENGINE_SCHEDULER_KV_BLOCKED_REQUESTS,
            snapshot.kv_blocked_requests,
        );
        push_engine_metric(
            payload,
            ENGINE_EXECUTOR_CONTINUOUS_ENVELOPE_SCALAR_FALLBACKS_TOTAL,
            snapshot.continuous_envelope_scalar_fallbacks_total,
        );
        push_engine_metric(
            payload,
            ENGINE_EXECUTOR_PHYSICAL_BATCH_REJECTIONS_TOTAL,
            snapshot.physical_batch_rejections_total,
        );
        push_engine_metric(
            payload,
            ENGINE_EXECUTOR_TENSOR_BATCH_ROWS_TOTAL,
            snapshot.tensor_batch_rows_total,
        );
        push_engine_metric(
            payload,
            ENGINE_EXECUTOR_TENSOR_BATCH_CAPACITY_ROWS_TOTAL,
            snapshot.tensor_batch_capacity_rows_total,
        );
        push_engine_metric(
            payload,
            ENGINE_EXECUTOR_TENSOR_BATCH_USEFUL_ELEMENTS_TOTAL,
            snapshot.tensor_batch_useful_elements_total,
        );
        push_engine_metric(
            payload,
            ENGINE_EXECUTOR_TENSOR_BATCH_MATERIALIZED_ELEMENTS_TOTAL,
            snapshot.tensor_batch_materialized_elements_total,
        );
        push_engine_metric(
            payload,
            ENGINE_EXECUTOR_BATCH_WORKSPACE_BYTES_TOTAL,
            snapshot.batch_workspace_bytes_total,
        );
        push_engine_labeled_metric(
            payload,
            ENGINE_EXECUTOR_DISPATCH_STATE_ROWS_TOTAL,
            "state",
            &snapshot.dispatch_states.labeled_values(),
        );
        push_engine_labeled_metric(
            payload,
            ENGINE_EXECUTOR_FAILURE_ORIGIN_ROWS_TOTAL,
            "origin",
            &snapshot.failure_origins.labeled_values(),
        );
        push_engine_labeled_metric(
            payload,
            ENGINE_EXECUTOR_DEADLINE_PHASE_ROWS_TOTAL,
            "phase",
            &snapshot.deadline_phases.labeled_values(),
        );
        push_engine_labeled_metric(
            payload,
            ENGINE_EXECUTOR_BATCH_WORKSPACE_DOMAIN_BYTES_TOTAL,
            "domain",
            &snapshot.workspace_domains.labeled_values(),
        );
        push_engine_metric_f64(
            payload,
            ENGINE_EXECUTOR_TENSOR_BATCH_FILL_RATIO,
            snapshot.tensor_batch_fill_ratio,
        );
        push_engine_metric_f64(
            payload,
            ENGINE_EXECUTOR_TENSOR_BATCH_PADDING_RATIO,
            snapshot.tensor_batch_padding_ratio,
        );
        push_engine_physical_execution_metrics(payload, &snapshot.physical_execution);
    }

    pub(super) fn push_coordinator_prometheus_metrics(&self, payload: &mut String) {
        let snapshot = self.coordinator.snapshot();
        payload.push_str(&format!(
            "# TYPE izwi_inference_coordinator_capacity gauge\n\
izwi_inference_coordinator_capacity {}\n\
# TYPE izwi_inference_coordinator_active_jobs gauge\n\
izwi_inference_coordinator_active_jobs {}\n\
# TYPE izwi_inference_coordinator_active_executions gauge\n\
izwi_inference_coordinator_active_executions {}\n\
# TYPE izwi_inference_coordinator_reserved_memory_bytes gauge\n\
izwi_inference_coordinator_reserved_memory_bytes {}\n\
# TYPE izwi_inference_coordinator_reserved_host_memory_bytes gauge\n\
izwi_inference_coordinator_reserved_host_memory_bytes {}\n\
# TYPE izwi_inference_coordinator_reserved_device_memory_bytes gauge\n\
izwi_inference_coordinator_reserved_device_memory_bytes {}\n\
# TYPE izwi_inference_coordinator_reserved_unified_memory_bytes gauge\n\
izwi_inference_coordinator_reserved_unified_memory_bytes {}\n\
# TYPE izwi_inference_coordinator_materialized_device_memory_bytes gauge\n\
izwi_inference_coordinator_materialized_device_memory_bytes {}\n\
# TYPE izwi_inference_coordinator_pending_device_memory_bytes gauge\n\
izwi_inference_coordinator_pending_device_memory_bytes {}\n\
# TYPE izwi_inference_coordinator_untracked_device_memory_bytes gauge\n\
izwi_inference_coordinator_untracked_device_memory_bytes {}\n\
# TYPE izwi_inference_coordinator_admitted_total counter\n\
izwi_inference_coordinator_admitted_total {}\n\
# TYPE izwi_inference_coordinator_rejected_total counter\n\
izwi_inference_coordinator_rejected_total {}\n\
# TYPE izwi_inference_coordinator_expired_total counter\n\
izwi_inference_coordinator_expired_total {}\n\
# TYPE izwi_inference_coordinator_draining gauge\n\
izwi_inference_coordinator_draining {}\n\
# TYPE izwi_inference_coordinator_poisoned gauge\n\
izwi_inference_coordinator_poisoned {}\n",
            snapshot.capacity,
            snapshot.active_jobs,
            snapshot.active_executions,
            snapshot.reserved_memory_bytes,
            snapshot.reserved_host_memory_bytes,
            snapshot.reserved_device_memory_bytes,
            snapshot.reserved_unified_memory_bytes,
            snapshot.materialized_device_memory_bytes,
            snapshot.pending_device_memory_bytes,
            snapshot.untracked_device_memory_bytes,
            snapshot.admitted_total,
            snapshot.rejected_total,
            snapshot.expired_total,
            u8::from(snapshot.draining),
            u8::from(snapshot.poisoned),
        ));
    }

    pub fn record_voice_session_started(&self) {
        self.telemetry.record_voice_session_started();
        self.record_voice_stage_observation("voice.session_started");
    }

    pub fn record_voice_session_closed(&self) {
        self.telemetry.record_voice_session_closed();
        self.record_voice_stage_observation("voice.session_closed");
    }

    pub fn record_voice_interruption(&self) {
        self.telemetry.record_voice_interruption();
        self.record_voice_stage_observation("voice.interruption");
    }

    pub fn record_voice_barge_in(&self) {
        self.telemetry.record_voice_barge_in();
        self.record_voice_stage_observation("voice.barge_in");
    }

    pub fn record_voice_speech_rejected(&self) {
        self.telemetry.record_voice_speech_rejected();
        self.record_voice_stage_observation("voice.speech_rejected");
    }

    pub fn record_voice_latency(
        &self,
        latency: crate::runtime::VoiceLatency,
        elapsed: std::time::Duration,
    ) {
        self.telemetry.record_voice_latency(latency, elapsed);
    }

    pub fn record_voice_egress_underrun(&self) {
        self.telemetry.record_voice_egress_underrun();
    }

    pub fn record_voice_ingress_dropped_frames(&self, frames: u64) {
        self.telemetry.record_voice_ingress_dropped_frames(frames);
    }

    pub fn record_voice_stream_backpressure(&self) {
        self.telemetry.record_voice_stream_backpressure();
        self.record_voice_stage_observation("voice.stream_backpressure");
    }

    pub fn record_transcription_stream_backpressure(&self) {
        self.telemetry.record_transcription_stream_backpressure();
        self.telemetry
            .record_stage_observation(RuntimeStageObservation::new(
                RuntimeObservationContext {
                    route_source: Some("realtime_transcription".to_string()),
                    capability: Some("asr".to_string()),
                    pipeline_kind: Some("realtime_transcription".to_string()),
                    pipeline_stage: Some("transcription.stream_backpressure".to_string()),
                    ..RuntimeObservationContext::default()
                },
                RuntimeStageOutcome::Observed,
            ));
    }

    pub fn record_modular_voice_pipeline_turn(&self) {
        let graph = PipelineGraph::modular_voice_turn();
        let summary = PipelineExecutor.execute_contract(&graph);
        self.telemetry.record_pipeline_execution(&summary);
    }

    pub fn record_unified_voice_pipeline_turn(&self) {
        let graph = PipelineGraph::unified_voice_turn();
        let summary = PipelineExecutor.execute_contract(&graph);
        self.telemetry.record_pipeline_execution(&summary);
    }

    pub(crate) fn record_diarization_transcript_pipeline(&self, enable_llm_refinement: bool) {
        let graph = PipelineGraph::diarization_transcript(enable_llm_refinement);
        let summary = PipelineExecutor.execute_contract(&graph);
        self.telemetry.record_pipeline_execution(&summary);
    }

    pub fn record_batch_asr_pipeline_job(&self) {
        let graph = PipelineGraph::batch_asr_transcription();
        let summary = PipelineExecutor.execute_contract(&graph);
        self.telemetry.record_pipeline_execution(&summary);
    }

    pub fn record_batch_tts_pipeline_job(&self) {
        let graph = PipelineGraph::batch_tts_speech();
        let summary = PipelineExecutor.execute_contract(&graph);
        self.telemetry.record_pipeline_execution(&summary);
    }

    pub(super) fn record_voice_stage_observation(&self, pipeline_stage: &'static str) {
        self.telemetry
            .record_stage_observation(RuntimeStageObservation::new(
                RuntimeObservationContext {
                    route_source: Some(format!("{:?}", RouteSource::RealtimeVoice)),
                    capability: Some(format!("{:?}", CapabilityKind::SpeechToSpeech)),
                    pipeline_kind: Some("realtime_voice".to_string()),
                    pipeline_stage: Some(pipeline_stage.to_string()),
                    ..RuntimeObservationContext::default()
                },
                RuntimeStageOutcome::Observed,
            ));
    }
}
