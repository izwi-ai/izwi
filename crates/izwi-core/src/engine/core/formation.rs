//! Physical batch formation: work costs, lanes and budgets, and the batches
//! formed from one scheduler step.

use super::*;

impl PhysicalBatchAssembly {
    /// An assembly holding the validated single-row `physical_batch`.
    pub(super) fn new(
        physical_batch: PhysicalBatch,
        request: Arc<EngineCoreRequest>,
        scheduled: super::super::scheduler::ScheduledRequest,
        output_visibility: super::super::OutputVisibility,
        shape_policy: StageShapePolicy,
        workspace_base_bytes: u64,
        launch_policy: PhysicalLaunchPolicy,
    ) -> Result<Self> {
        debug_assert_eq!(physical_batch.rows.len(), 1);
        let mut row_keys = HashSet::new();
        let row_cost = physical_batch.validate_appended(&mut row_keys, WorkCost::default())?;
        Ok(Self {
            physical_batch,
            requests: vec![request],
            scheduled: vec![scheduled],
            output_visibility,
            shape_policy,
            workspace_base_bytes,
            launch_policy,
            row_keys,
            row_cost,
        })
    }

    pub(super) fn materialized_tensor_elements(
        shape_policy: StageShapePolicy,
        rows: &[ReadyQuantum],
    ) -> Option<u64> {
        if shape_policy == StageShapePolicy::Padded {
            let maximum = rows.iter().map(|row| row.cost.tensor_elements).max()?;
            maximum.checked_mul(u64::try_from(rows.len()).ok()?)
        } else {
            rows.iter().try_fold(0u64, |total, row| {
                total.checked_add(row.cost.tensor_elements)
            })
        }
    }

    pub(super) fn workspace_resources(
        backend: BackendKind,
        shape_policy: StageShapePolicy,
        base_bytes: u64,
        rows: &[ReadyQuantum],
    ) -> Option<ResourceVector> {
        let generic = if shape_policy == StageShapePolicy::Padded {
            let width = u64::try_from(rows.len()).ok()?;
            let maximum = |select: fn(ResourceVector) -> ResourceAmount| {
                rows.iter()
                    .map(|row| match select(row.cost.workspace) {
                        ResourceAmount::Known(value) => Some(value),
                        ResourceAmount::Unknown => None,
                    })
                    .try_fold(0u64, |maximum, value| Some(maximum.max(value?)))?
                    .checked_mul(width)
            };
            ResourceVector {
                host_bytes: ResourceAmount::Known(maximum(|value| value.host_bytes)?),
                device_bytes: ResourceAmount::Known(maximum(|value| value.device_bytes)?),
                unified_bytes: ResourceAmount::Known(maximum(|value| value.unified_bytes)?),
                kv_bytes: ResourceAmount::Known(maximum(|value| value.kv_bytes)?),
                temporary_bytes: ResourceAmount::Known(
                    base_bytes.checked_add(maximum(|value| value.temporary_bytes)?)?,
                ),
                compute_slots: ResourceAmount::Known(maximum(|value| value.compute_slots)?),
            }
        } else {
            rows.iter().try_fold(
                ResourceVector {
                    temporary_bytes: ResourceAmount::Known(base_bytes),
                    ..ResourceVector::zero()
                },
                |total, row| total.checked_add(row.cost.workspace).ok(),
            )?
        };
        if generic.kv_bytes != ResourceAmount::Known(0)
            || generic.compute_slots != ResourceAmount::Known(0)
        {
            return None;
        }
        let known = |amount| match amount {
            ResourceAmount::Known(value) => Some(value),
            ResourceAmount::Unknown => None,
        };
        let host = known(generic.host_bytes)?;
        let accelerator = known(generic.device_bytes)?
            .checked_add(known(generic.unified_bytes)?)?
            .checked_add(known(generic.temporary_bytes)?)?;
        let mut workspace = ResourceVector::zero();
        match backend {
            BackendKind::Cpu => {
                workspace.host_bytes = ResourceAmount::Known(host.checked_add(accelerator)?)
            }
            BackendKind::Metal => {
                workspace.unified_bytes = ResourceAmount::Known(host.checked_add(accelerator)?)
            }
            BackendKind::Cuda => {
                workspace.host_bytes = ResourceAmount::Known(host);
                workspace.device_bytes = ResourceAmount::Known(accelerator);
            }
        }
        Some(workspace)
    }

    pub(super) fn matches_shape(
        &self,
        lane: &BatchLaneKey,
        mode: NativeBatchMode,
        budget: &BatchBudget,
        output_visibility: super::super::OutputVisibility,
        shape_policy: StageShapePolicy,
        launch_policy: PhysicalLaunchPolicy,
    ) -> bool {
        self.physical_batch.lane == *lane
            && self.physical_batch.mode == mode
            && self.physical_batch.budget == *budget
            && self.output_visibility == output_visibility
            && self.shape_policy == shape_policy
            && self.launch_policy == launch_policy
    }

    /// Structural precondition of `try_push` for a row of this work and
    /// cost. Element and workspace budgets need the prepared row; `try_push`
    /// checks them.
    pub(super) fn may_accept(&self, work: &WorkUnit, cost: &WorkCost) -> bool {
        self.physical_batch.rows.len() < self.physical_batch.budget.max_rows
            && !self.isolates_multi_token_quanta(work, cost)
    }

    /// A model-authored multi-token decode quantum (MTP drafts and verify):
    /// continuous envelopes are N rows by exactly one token, and a mixed
    /// batch holds one-token decode rows beside prefill spans of any length.
    pub(super) fn is_isolated_quantum(mode: NativeBatchMode, work: &WorkUnit, cost: &WorkCost) -> bool {
        cost.logical_units > 1
            && match mode {
                NativeBatchMode::Continuous => true,
                NativeBatchMode::Mixed => matches!(
                    work,
                    WorkUnit::SequenceStep {
                        phase: SequencePhase::Decode,
                        ..
                    }
                ),
                NativeBatchMode::None | NativeBatchMode::Static => false,
            }
    }

    pub(super) fn isolates_multi_token_quanta(&self, work: &WorkUnit, cost: &WorkCost) -> bool {
        // Isolated quanta run alone so membership can change only after
        // their bounded transaction commits.
        let mode = self.physical_batch.mode;
        !self.physical_batch.rows.is_empty()
            && (Self::is_isolated_quantum(mode, work, cost)
                || self.physical_batch.rows.iter().any(|existing| {
                    Self::is_isolated_quantum(mode, &existing.work, &existing.cost)
                }))
    }

    /// Append `row` when the batch stays valid. The row is validated in
    /// place against the rows already accepted; a refused row is returned
    /// and the batch is left as it was.
    pub(super) fn try_push(
        &mut self,
        request: Arc<EngineCoreRequest>,
        scheduled: super::super::scheduler::ScheduledRequest,
        row: ReadyQuantum,
    ) -> std::result::Result<(), Box<RefusedBatchRow>> {
        if self.isolates_multi_token_quanta(&row.work, &row.cost) {
            return Err(Box::new((request, scheduled, row)));
        }
        let previous = (
            self.physical_batch.materialized_tensor_elements,
            self.physical_batch.workspace,
        );
        self.physical_batch.rows.push(row);
        let accepted = (|| {
            let batch = &mut self.physical_batch;
            batch.materialized_tensor_elements =
                Self::materialized_tensor_elements(self.shape_policy, &batch.rows)?;
            batch.workspace = Self::workspace_resources(
                batch.lane.backend,
                self.shape_policy,
                self.workspace_base_bytes,
                &batch.rows,
            )?;
            batch
                .validate_appended(&mut self.row_keys, self.row_cost)
                .ok()
        })();
        match accepted {
            Some(cost) => {
                self.row_cost = cost;
                self.requests.push(request);
                self.scheduled.push(scheduled);
                Ok(())
            }
            None => {
                let row = self
                    .physical_batch
                    .rows
                    .pop()
                    .expect("the refused row was pushed above");
                (
                    self.physical_batch.materialized_tensor_elements,
                    self.physical_batch.workspace,
                ) = previous;
                Err(Box::new((request, scheduled, row)))
            }
        }
    }
}

impl EngineCore {
    pub(super) fn work_cost(
        request: &EngineCoreRequest,
        work: &WorkUnit,
        stage: Option<&super::super::StageDescriptor>,
        backend: crate::backends::BackendKind,
    ) -> Result<WorkCost> {
        if let WorkUnit::RealtimePreparation { operation_id, .. } = work {
            return request.realtime_asr_preparation_cost(*operation_id);
        }
        if let Some(prepared) = stage.and_then(|stage| request.prepared_stage_cost(stage.id)) {
            if stage.is_some_and(|stage| stage.batch_mode == NativeBatchMode::Continuous) {
                if let WorkUnit::SequenceStep {
                    phase: SequencePhase::Decode,
                    input,
                    ..
                } = work
                {
                    let input_units = u64::try_from(input.len()).map_err(|_| {
                        Error::Overloaded(
                            "continuous decode input exceeds work accounting".to_string(),
                        )
                    })?;
                    if input_units > 1 {
                        let tensor_elements = prepared
                            .tensor_elements
                            .checked_mul(input_units)
                            .ok_or_else(|| {
                                Error::Overloaded(
                                    "continuous decode tensor cost overflowed".to_string(),
                                )
                            })?;
                        return Ok(WorkCost::with_workspace(
                            input_units,
                            tensor_elements,
                            prepared.workspace,
                        ));
                    }
                }
            }
            return Ok(prepared);
        }
        let logical_units = match work {
            WorkUnit::PreSequencePreparation { .. } => 1,
            WorkUnit::SequenceStep {
                input,
                max_output_steps,
                ..
            } => {
                let input = u64::try_from(input.len()).map_err(|_| {
                    Error::Overloaded("execution input length exceeds work accounting".to_string())
                })?;
                let output = u64::try_from(*max_output_steps).map_err(|_| {
                    Error::Overloaded("execution output bound exceeds work accounting".to_string())
                })?;
                input.max(output).max(1)
            }
            WorkUnit::SequenceAudioDecode {
                max_frames: max_output_steps,
            }
            | WorkUnit::SequenceFinalize { max_output_steps } => u64::try_from(*max_output_steps)
                .map_err(|_| {
                    Error::Overloaded(
                        "sequence finalization output bound exceeds work accounting".into(),
                    )
                })?
                .max(1),
            WorkUnit::RealtimePush {
                input,
                max_output_steps,
                max_cache_append,
                ..
            } => {
                let input = u64::try_from(input.len()).map_err(|_| {
                    Error::Overloaded("realtime input length exceeds work accounting".to_string())
                })?;
                let output = u64::try_from(*max_output_steps).map_err(|_| {
                    Error::Overloaded("realtime output bound exceeds work accounting".to_string())
                })?;
                let cache_append = u64::try_from(*max_cache_append).map_err(|_| {
                    Error::Overloaded(
                        "realtime cache append bound exceeds work accounting".to_string(),
                    )
                })?;
                input.max(output).max(cache_append).max(1)
            }
            WorkUnit::RealtimeFinish {
                max_output_steps,
                max_cache_append,
                ..
            } => {
                let output = u64::try_from(*max_output_steps).map_err(|_| {
                    Error::Overloaded(
                        "realtime finish output bound exceeds work accounting".to_string(),
                    )
                })?;
                let cache_append = u64::try_from(*max_cache_append).map_err(|_| {
                    Error::Overloaded(
                        "realtime cache append bound exceeds work accounting".to_string(),
                    )
                })?;
                output.max(cache_append).max(1)
            }
            WorkUnit::RealtimePreparation { input, .. } => u64::try_from(input.len())
                .map_err(|_| {
                    Error::Overloaded(
                        "realtime preparation input exceeds work accounting".to_string(),
                    )
                })?
                .max(1),
            WorkUnit::RealtimePromptPrefill {
                max_output_steps,
                cache_append,
                ..
            } => u64::try_from((*max_output_steps).max(*cache_append))
                .map_err(|_| {
                    Error::Overloaded("realtime prompt prefill exceeds work accounting".to_string())
                })?
                .max(1),
            WorkUnit::RealtimeDecodeContinuation { .. } | WorkUnit::RealtimeCompletion { .. } => 1,
            WorkUnit::AtomicJob { .. } | WorkUnit::PipelineStage { .. } => 1,
        };
        // Fish codec finalization is scalar. Native tensor-stage costs require
        // a bound batchable stage, so derive this workspace from the sealed
        // generation geometry only after the execution plan has its binding.
        if request.task_type == super::super::TaskType::TTS
            && request.model_variant == Some(ModelVariant::FishAudioS2Pro)
            && matches!(
                work,
                WorkUnit::SequenceFinalize { .. } | WorkUnit::SequenceAudioDecode { .. }
            )
        {
            let binding = request.execution_adapter_binding().ok_or_else(|| {
                Error::InvalidInput("Fish S2 finalization requires a loaded adapter binding".into())
            })?;
            let stage = stage
                .filter(|stage| {
                    stage.selector.matches(work)
                        && (stage.batch_mode == NativeBatchMode::None
                            || (matches!(work, WorkUnit::SequenceAudioDecode { .. })
                                && stage.batch_mode == NativeBatchMode::Static))
                        && binding.stages.contains(stage)
                })
                .ok_or_else(|| {
                    Error::InvalidInput(
                        "Fish S2 codec requires its exact bound execution stage".into(),
                    )
                })?;
            let params = request
                .fish_s2_tts_generation_params_for_executor()?
                .ok_or_else(|| {
                    Error::InvalidInput(
                        "Fish S2 finalization requires sealed generation parameters".into(),
                    )
                })?;
            params.validate()?;
            if params.max_frames > ModelVariant::FISH_S2_PRO_MAX_OUTPUT_FRAMES {
                return Err(Error::InvalidInput(
                    "Fish S2 output exceeds the codec frame limit".into(),
                ));
            }
            let frames = match work {
                WorkUnit::SequenceAudioDecode { max_frames } => {
                    (*max_frames).min(params.max_frames)
                }
                _ => params.max_frames,
            };
            let bytes = if matches!(work, WorkUnit::SequenceAudioDecode { .. }) {
                crate::models::architectures::fish_s2::codec::streaming_decode_workspace_bytes(
                    frames,
                )?
            } else {
                // Terminal flush only moves retained PCM; neural work is complete.
                0
            };
            if logical_units > stage.max_work_units || bytes > stage.max_workspace_bytes {
                return Err(Error::Overloaded(
                    "Fish S2 finalization exceeds its loaded stage workspace or work limit".into(),
                ));
            }
            let mut workspace = super::super::ResourceVector::zero();
            match backend {
                crate::backends::BackendKind::Cpu => {
                    workspace.host_bytes = ResourceAmount::Known(bytes)
                }
                crate::backends::BackendKind::Metal => {
                    workspace.unified_bytes = ResourceAmount::Known(bytes)
                }
                crate::backends::BackendKind::Cuda => {
                    workspace.device_bytes = ResourceAmount::Known(bytes)
                }
            }
            return Ok(WorkCost::with_workspace(
                logical_units,
                logical_units,
                workspace,
            ));
        }
        let workspace_bytes = stage.map_or(Ok(0), |stage| {
            stage
                .workspace_per_work_unit_bytes
                .checked_mul(logical_units)
                .and_then(|bytes| bytes.checked_add(stage.workspace_per_row_bytes))
                .ok_or_else(|| Error::Overloaded("batch workspace estimate overflow".to_string()))
        })?;
        Ok(WorkCost::new(logical_units, logical_units, workspace_bytes))
    }

    pub(super) fn batch_lane(plan: &ExecutionPlan, cost: WorkCost) -> BatchLaneKey {
        let adapter = plan.batch_key.adapter.as_ref();
        let stage = plan.stage.as_ref();
        let shape_policy = stage
            .map(|stage| stage.shape_policy)
            .unwrap_or(StageShapePolicy::Exact);
        let shape_bucket = match shape_policy {
            StageShapePolicy::Independent => "independent".to_string(),
            StageShapePolicy::Exact => format!("exact.{}", cost.tensor_elements),
            StageShapePolicy::Bucketed => format!(
                "bucket.{}",
                cost.tensor_elements
                    .checked_next_power_of_two()
                    .unwrap_or(u64::MAX)
            ),
            StageShapePolicy::Padded => match &plan.work {
                WorkUnit::SequenceStep {
                    auxiliary_state: Some(spans),
                    ..
                } if !spans.is_empty() => "padded.auxiliary".to_string(),
                WorkUnit::SequenceStep { .. } => "padded.primary".to_string(),
                _ => "padded".to_string(),
            },
            StageShapePolicy::Ragged => "ragged".to_string(),
        };
        BatchLaneKey {
            execution_group: adapter
                .map(|adapter| adapter.execution_group_id)
                .unwrap_or(ExecutionGroupId::new(0)),
            model_instance: adapter
                .map(|adapter| adapter.model_instance_id)
                .unwrap_or(ModelInstanceId::new(0)),
            adapter_instance: adapter
                .map(|adapter| adapter.adapter_instance_id)
                .unwrap_or(AdapterInstanceId::new(0)),
            adapter_abi: adapter
                .map(|adapter| adapter.adapter_abi_revision)
                .unwrap_or(AdapterAbiRevision::new(0)),
            capability_id: adapter
                .map(|adapter| adapter.capability_id.clone())
                .unwrap_or_else(|| "compatibility".to_string()),
            stage_id: adapter
                .map(|adapter| adapter.stage_id)
                .unwrap_or(StageId::new(0)),
            backend: plan.batch_key.backend,
            device_ordinal: None,
            compute_dtype: plan.batch_key.compute_dtype.clone(),
            state_dtype: plan.batch_key.kv_dtype.clone(),
            tensor_layout: format!("{shape_policy:?}").to_ascii_lowercase(),
            quantization: "adapter-owned".to_string(),
            state_schema: plan.batch_key.cache_namespace.clone(),
            kernel_mode: stage
                .map(|stage| stage.name.clone())
                .unwrap_or_else(|| "compatibility".to_string()),
            semantic_mode: format!(
                "{:?}.{}",
                plan.batch_key.task_type, plan.batch_key.work_kind
            )
            .to_ascii_lowercase(),
            shape_bucket,
        }
    }

    pub(super) fn batch_budget(plan: &ExecutionPlan) -> Result<(BatchBudget, StageShapePolicy)> {
        let Some(stage) = plan.stage.as_ref() else {
            if plan.batch_mode == NativeBatchMode::None {
                return Ok((BatchBudget::width_one(), StageShapePolicy::Exact));
            }
            return Err(Error::InferenceError(
                "native tensor batch plan is missing its loaded stage contract".to_string(),
            ));
        };
        if stage.concurrency == ConcurrencyClass::Exclusive {
            return Ok((BatchBudget::width_one(), stage.shape_policy));
        }
        let budget = BatchBudget {
            max_rows: plan.max_batch_size.min(stage.max_batch_size).max(1),
            max_logical_units: stage.max_work_units,
            max_tensor_elements: u64::MAX,
            max_workspace_bytes: stage.max_workspace_bytes,
            max_padding_basis_points: stage.max_padding_basis_points,
            max_formation_delay: stage.max_formation_delay,
        };
        budget.validate()?;
        Ok((budget, stage.shape_policy))
    }

    /// The scope of a tracked request's lane step driver; `None` (the whole
    /// engine) for an unknown request.
    pub(in crate::engine) fn request_lane_scope(&self, request_id: &RequestId) -> LaneScope {
        self.requests
            .get(request_id)
            .map(|request| request.model_instance_id())
    }

    /// Lanes of the tracked requests that use `variant`.
    pub(in crate::engine) fn variant_lanes(&self, variant: ModelVariant) -> Vec<ExecutionLane> {
        let mut lanes = self
            .requests
            .values()
            .filter(|request| request.model_variant == Some(variant))
            .map(|request| request.model_instance_id())
            .collect::<Vec<_>>();
        lanes.sort_unstable();
        lanes.dedup();
        lanes
    }

    /// Lanes that hold scheduler work.
    pub(in crate::engine) fn lanes_with_work(&self) -> Vec<ExecutionLane> {
        self.scheduler.lanes_with_work()
    }

    /// The execution lane of a tracked request.
    pub(super) fn request_lane(&self, request_id: &RequestId) -> ExecutionLane {
        self.requests
            .get(request_id)
            .and_then(|request| request.model_instance_id())
    }

    /// Lanes with a dispatch in flight.
    pub(super) fn busy_lanes(&self) -> HashSet<ExecutionLane> {
        self.in_flight_dispatches
            .values()
            .flat_map(|dispatch| dispatch.lanes.iter().copied())
            .collect()
    }

    /// Dispatches in flight in `lanes`: during a prepare, the ones this
    /// step registered, since a busy lane is never admitted.
    pub(super) fn dispatches_in_flight_in(&self, lanes: &LaneSelection) -> usize {
        self.in_flight_dispatches
            .values()
            .filter(|dispatch| dispatch.lanes.iter().all(|lane| lanes.admits(*lane)))
            .count()
    }

    pub(super) fn allocate_batch_id(&mut self) -> Result<BatchId> {
        let batch_id = BatchId::new(self.next_batch_id);
        self.next_batch_id = self.next_batch_id.checked_add(1).ok_or_else(|| {
            Error::InferenceError("physical batch identity space was exhausted".to_string())
        })?;
        Ok(batch_id)
    }

    pub(super) fn register_prepared_physical_dispatch(
        &mut self,
        dispatch: &PreparedPhysicalDispatch,
    ) -> Result<()> {
        let mut lanes = dispatch
            .scheduled()
            .iter()
            .map(|row| self.request_lane(&row.request_id))
            .collect::<Vec<_>>();
        lanes.sort_unstable();
        lanes.dedup();
        let in_flight = InFlightPhysicalDispatch::from_prepared(dispatch, lanes);
        let batch_id = in_flight.physical_batch.batch_id;
        let candidate_limit = self
            .config
            .resolved_physical_execution_capacity()
            .candidate_dispatch_limit
            .get();
        // The candidate limit is per lane: another lane's dispatch in
        // flight does not own this lane's slots.
        let lane_in_flight = self
            .in_flight_dispatches
            .values()
            .filter(|active| {
                active
                    .lanes
                    .iter()
                    .any(|lane| in_flight.lanes.contains(lane))
            })
            .count();
        if lane_in_flight >= candidate_limit {
            return Err(Error::InferenceError(
                "physical dispatch candidate limit is already fully owned".to_string(),
            ));
        }
        if self.in_flight_dispatches.contains_key(&batch_id) {
            return Err(Error::InferenceError(
                "physical dispatch batch identity was registered twice".to_string(),
            ));
        }

        for row in &in_flight.physical_batch.rows {
            let plan = self.active_plans.get(&row.plan_id).ok_or_else(|| {
                Error::InferenceError(
                    "physical dispatch row has no active execution plan".to_string(),
                )
            })?;
            if plan.session != row.session || plan.work != row.work {
                return Err(Error::InferenceError(
                    "physical dispatch row crossed its active plan fence".to_string(),
                ));
            }
            let sealed_launch_policy = plan
                .stage
                .as_ref()
                .map(|stage| stage.physical_launch_policy)
                .unwrap_or(PhysicalLaunchPolicy::ExecutionGroupExclusive);
            if sealed_launch_policy != in_flight.launch_policy {
                return Err(Error::InferenceError(
                    "physical dispatch launch policy differs from its sealed stage contract"
                        .to_string(),
                ));
            }
            if self.in_flight_dispatches.values().any(|active| {
                active.contains_plan(row.plan_id) || active.contains_session(&row.session)
            }) {
                return Err(Error::InferenceError(
                    "physical dispatch duplicated an in-flight session or plan quantum".to_string(),
                ));
            }
            match (
                row.managed_cache.as_ref(),
                self.active_managed_cache.get(&row.plan_id),
            ) {
                (None, None) => {}
                (Some(row), Some(active)) if row == active => {}
                _ => {
                    return Err(Error::InferenceError(
                        "physical dispatch row crossed its active managed-cache fence".to_string(),
                    ));
                }
            }
        }

        let stream_fence = in_flight.stream_fence();
        if stream_fence.is_some() && self.active_stream_batches.contains_key(&batch_id) {
            return Err(Error::InferenceError(
                "physical stream batch identity was registered twice".to_string(),
            ));
        }
        self.in_flight_dispatches.insert(batch_id, in_flight);
        if let Some(stream_fence) = stream_fence {
            self.active_stream_batches.insert(batch_id, stream_fence);
        }
        Ok(())
    }

    /// Form the decode side of a step: the mixed batch first (decode rows
    /// and prefill spans of mixed-step models), then split decode rows in the
    /// candidates it left.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn form_decode_side(
        &mut self,
        mixed_requests: &[Arc<EngineCoreRequest>],
        mixed_scheduled: &[super::super::scheduler::ScheduledRequest],
        decode_requests: &[Arc<EngineCoreRequest>],
        decode_scheduled: &[super::super::scheduler::ScheduledRequest],
        max_candidates: usize,
        capacity_blocked: &mut Vec<super::super::scheduler::ScheduledRequest>,
        transaction_deferred: &mut Vec<super::super::scheduler::ScheduledRequest>,
        prepare_failures: &mut Vec<ManagedPrepareFailure>,
    ) -> Result<Vec<PreparedPhysicalDispatch>> {
        let mut batches = if mixed_scheduled.is_empty() {
            Vec::new()
        } else {
            self.form_physical_batches(
                ExecutionPhase::Mixed,
                mixed_requests,
                mixed_scheduled,
                max_candidates,
                capacity_blocked,
                transaction_deferred,
                prepare_failures,
            )?
        };
        if !decode_scheduled.is_empty() {
            let decode = self.form_physical_batches(
                ExecutionPhase::Decode,
                decode_requests,
                decode_scheduled,
                max_candidates.saturating_sub(batches.len()),
                capacity_blocked,
                transaction_deferred,
                prepare_failures,
            )?;
            batches.extend(decode);
        }
        Ok(batches)
    }

    pub(super) fn form_physical_batches(
        &mut self,
        phase: ExecutionPhase,
        requests: &[Arc<EngineCoreRequest>],
        scheduled: &[super::super::scheduler::ScheduledRequest],
        max_candidates: usize,
        capacity_blocked: &mut Vec<super::super::scheduler::ScheduledRequest>,
        transaction_deferred: &mut Vec<super::super::scheduler::ScheduledRequest>,
        prepare_failures: &mut Vec<ManagedPrepareFailure>,
    ) -> Result<Vec<PreparedPhysicalDispatch>> {
        let available = requests
            .iter()
            .map(|request| request.id.clone())
            .collect::<HashSet<_>>();
        let mut assemblies: Vec<PhysicalBatchAssembly> = Vec::new();
        // Preparing a row begins its managed-state transaction, but only the
        // first `available_candidates` assemblies launch this step. A row that
        // could only open a later assembly is deferred before preparation, so
        // formation never stages more transactions than one step launches
        // (candidate limit x batch width), the bound automatic staged rows use.
        // Callers net the tickets this step already registered out of
        // `max_candidates`; other lanes' tickets own no slot of this step.
        let candidate_limit = self
            .config
            .resolved_physical_execution_capacity()
            .candidate_dispatch_limit
            .get();
        let available_candidates = candidate_limit.min(max_candidates);

        for scheduled in scheduled {
            if !available.contains(&scheduled.request_id) {
                continue;
            }
            let request = self
                .requests
                .get(&scheduled.request_id)
                .cloned()
                .ok_or_else(|| {
                    Error::InferenceError(format!(
                        "scheduled request {} disappeared during batch formation",
                        scheduled.request_id
                    ))
                })?;
            let plan = self
                .active_plans
                .get(&scheduled.plan_id)
                .cloned()
                .ok_or_else(|| {
                    Error::InferenceError(format!(
                        "scheduled plan {} disappeared during batch formation",
                        scheduled.plan_id
                    ))
                })?;
            let cost = Self::work_cost(
                &request,
                &plan.work,
                plan.stage.as_ref(),
                plan.batch_key.backend,
            )?;
            let lane = Self::batch_lane(&plan, cost);
            let (mut budget, shape_policy) = Self::batch_budget(&plan)?;
            if let Some(cap) = self.workspace_row_limits.get(&plan.session) {
                budget.max_rows = budget.max_rows.min(*cap);
            }
            let output_visibility = plan
                .stage
                .as_ref()
                .map_or(super::super::OutputVisibility::AfterQuantumCommit, |stage| {
                    stage.output_visibility
                });
            let launch_policy = plan
                .stage
                .as_ref()
                .map(|stage| stage.physical_launch_policy)
                .unwrap_or(PhysicalLaunchPolicy::ExecutionGroupExclusive);
            if plan.batch_mode == NativeBatchMode::None {
                if self.config.physical_execution_mode
                    == crate::config::PhysicalExecutionMode::Concurrent
                    && matches!(launch_policy, PhysicalLaunchPolicy::Concurrent { .. })
                {
                    let launch_limit = self
                        .config
                        .resolved_physical_execution_capacity()
                        .physical_launch_limit;
                    budget.max_rows = budget
                        .max_rows
                        .min(launch_policy.effective_max_in_flight_per_model(launch_limit));
                } else {
                    // Physical serial/shadow modes are authoritative for
                    // scalar calls. Native B>1 tensor stages remain one model
                    // entry and are governed by their independent rollout.
                    budget.max_rows = 1;
                }
            }
            let joins_launchable_assembly = budget.max_rows > 1
                && assemblies
                    .iter()
                    .take(available_candidates)
                    .any(|assembly| {
                        assembly.matches_shape(
                            &lane,
                            plan.batch_mode,
                            &budget,
                            output_visibility,
                            shape_policy,
                            launch_policy,
                        ) && assembly.may_accept(&plan.work, &cost)
                    });
            if assemblies.len() >= available_candidates && !joins_launchable_assembly {
                transaction_deferred.push(scheduled.clone());
                continue;
            }
            let planned_work = plan.work.clone();
            let claim_policy = self.managed_claim_policy(&request);
            let prefix_attach = self.scheduler.admits_prefix_attach(&scheduled.request_id);
            let managed_cache = match request.managed_cache_runtime().map(|runtime| {
                self.managed_kv_cache.prepare_row(
                    runtime,
                    plan.plan_id,
                    &plan.session,
                    &planned_work,
                    Some(&request),
                    claim_policy,
                    prefix_attach,
                )
            }) {
                Some(Ok(reservation)) => reservation,
                None => None,
                Some(Err(Error::Backpressure(reason)))
                    if reason
                        .starts_with(crate::engine::cache::managed::MANAGED_PREFIX_CURSOR_LOST) =>
                {
                    // The probed prefix vanished before execution. Drop the
                    // cursor and replan this request as a zero-start prefill.
                    if !self
                        .scheduler
                        .reset_managed_prefix_cursor(&scheduled.request_id)
                    {
                        warn!(
                            request_id = %scheduled.request_id,
                            "prefix cursor lost without scheduler cursor state"
                        );
                    }
                    capacity_blocked.push(scheduled.clone());
                    continue;
                }
                Some(Err(Error::Backpressure(reason))) => {
                    super::super::metrics::record_engine_physical_defer(
                        super::super::metrics::EnginePhysicalDeferReason::ManagedCacheCapacity,
                    );
                    capacity_blocked.push(scheduled.clone());
                    debug!(
                        request_id = %scheduled.request_id,
                        plan_id = scheduled.plan_id,
                        reason = %reason,
                        "Withholding only the row blocked by managed KV capacity"
                    );
                    continue;
                }
                Some(Err(error)) => {
                    // Only this row failed; keep forming the others. The
                    // caller rolls the row back and either retries it on a
                    // later step or fails it terminally.
                    prepare_failures.push(ManagedPrepareFailure {
                        scheduled: scheduled.clone(),
                        session: plan.session.clone(),
                        error,
                    });
                    continue;
                }
            };
            self.managed_prepare_attempts.remove(&plan.session);
            if let Some(reservation) = managed_cache.as_ref() {
                if self
                    .active_managed_cache
                    .insert(plan.plan_id, reservation.clone())
                    .is_some()
                {
                    let _ = self.managed_kv_cache.finalize(reservation, None, false);
                    return Err(Error::InferenceError(
                        "managed KV transaction was prepared twice".to_string(),
                    ));
                }
            }
            let row = ReadyQuantum {
                plan_id: plan.plan_id,
                session: plan.session.clone(),
                lane: lane.clone(),
                work: planned_work.clone(),
                cost,
                managed_cache,
            };
            let mut executable = scheduled.clone();
            executable.work = planned_work;

            let mut pending = Some((request, executable, row));
            if budget.max_rows > 1 {
                for assembly in &mut assemblies {
                    if !assembly.matches_shape(
                        &lane,
                        plan.batch_mode,
                        &budget,
                        output_visibility,
                        shape_policy,
                        launch_policy,
                    ) {
                        continue;
                    }
                    let Some((request, scheduled, row)) = pending.take() else {
                        break;
                    };
                    match assembly.try_push(request, scheduled, row) {
                        Ok(()) => break,
                        Err(refused) => pending = Some(*refused),
                    }
                }
            }

            if let Some((request, scheduled, row)) = pending {
                let materialized_tensor_elements = row.cost.tensor_elements;
                let workspace_base_bytes = plan
                    .stage
                    .as_ref()
                    .map(|stage| stage.workspace_base_bytes)
                    .unwrap_or(0);
                let workspace = PhysicalBatchAssembly::workspace_resources(
                    lane.backend,
                    shape_policy,
                    workspace_base_bytes,
                    std::slice::from_ref(&row),
                )
                .ok_or_else(|| {
                    Error::Overloaded("batch workspace estimate overflow".to_string())
                })?;
                let physical_batch = PhysicalBatch {
                    batch_id: self.allocate_batch_id()?,
                    lane,
                    mode: plan.batch_mode,
                    budget,
                    rows: vec![row],
                    materialized_tensor_elements,
                    workspace,
                };
                assemblies.push(PhysicalBatchAssembly::new(
                    physical_batch,
                    request,
                    scheduled,
                    output_visibility,
                    shape_policy,
                    workspace_base_bytes,
                    launch_policy,
                )?);
            }
        }

        let mut prepared = Vec::with_capacity(assemblies.len().min(available_candidates));
        for (index, assembly) in assemblies.into_iter().enumerate() {
            if index >= available_candidates {
                transaction_deferred.extend(assembly.scheduled);
                continue;
            }
            let reservations = assembly
                .physical_batch
                .rows
                .iter()
                .filter_map(|row| row.managed_cache.clone())
                .collect();
            let dispatch = PreparedPhysicalDispatch::new(
                phase,
                assembly.physical_batch,
                assembly.requests,
                assembly.scheduled,
                assembly.output_visibility,
                reservations,
                assembly.launch_policy,
            )?;
            self.register_prepared_physical_dispatch(&dispatch)?;
            prepared.push(dispatch);
        }
        Ok(prepared)
    }

    #[cfg(test)]
    pub(super) fn merge_audio_output(
        existing: Option<AudioOutput>,
        current: Option<AudioOutput>,
    ) -> Option<AudioOutput> {
        // Stream completion carries cumulative committed sample metadata while
        // intentionally retaining no waveform. Its empty Vec is not no output.
        if current
            .as_ref()
            .is_some_and(|audio| audio.streamed_samples.is_some())
        {
            return current;
        }
        match (existing, current) {
            (None, None) => None,
            (Some(existing), None) => Some(existing),
            (None, Some(current)) => Some(current),
            (Some(mut existing), Some(current)) => {
                if existing.sample_rate != current.sample_rate {
                    return Some(current);
                }
                if current.samples.is_empty() {
                    return Some(existing);
                }
                if existing.samples.is_empty() {
                    return Some(current);
                }

                let looks_cumulative = current.samples.len() >= existing.samples.len()
                    && current
                        .samples
                        .iter()
                        .zip(existing.samples.iter())
                        .all(|(cur, prev)| cur == prev);

                if looks_cumulative {
                    Some(current)
                } else {
                    existing.append(&current);
                    Some(existing)
                }
            }
        }
    }

    #[cfg(test)]
    pub(super) fn merge_executor_output(
        existing: Option<ExecutorOutput>,
        current: ExecutorOutput,
    ) -> ExecutorOutput {
        let Some(mut merged) = existing else {
            return current;
        };

        if merged.request_id != current.request_id {
            return current;
        }
        if merged.finished || merged.error.is_some() {
            return merged;
        }

        let ExecutorOutput {
            request_id: _,
            audio,
            text,
            input_transcription,
            tokens_processed,
            tokens_generated,
            finished,
            phase_timing_override,
            asr_diagnostics,
            error,
            logprobs,
        } = current;

        merged.logprobs.extend(logprobs);
        merged.audio = Self::merge_audio_output(merged.audio.take(), audio);
        if text.is_some() {
            merged.text = text;
        }
        if input_transcription.is_some() {
            merged.input_transcription = input_transcription;
        }
        merged.tokens_processed = merged.tokens_processed.saturating_add(tokens_processed);
        merged.tokens_generated = merged.tokens_generated.saturating_add(tokens_generated);
        merged.finished |= finished;
        merged.phase_timing_override = phase_timing_override.or(merged.phase_timing_override);
        merged.asr_diagnostics = asr_diagnostics.or(merged.asr_diagnostics);
        if error.is_some() {
            merged.error = error;
        }

        merged
    }

    pub(super) fn has_user_visible_output(exec_output: &ExecutorOutput) -> bool {
        exec_output.tokens_generated > 0
            || exec_output
                .text
                .as_ref()
                .is_some_and(|text| !text.is_empty())
            || exec_output
                .audio
                .as_ref()
                .is_some_and(|audio| !audio.samples.is_empty())
    }
}
