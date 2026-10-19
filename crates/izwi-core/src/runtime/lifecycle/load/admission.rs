//! Load admission: the residency budget, portable and CUDA memory, the request
//! envelope and the resources a load reserves.

use super::*;

/// Recurrent state the CUDA resident fit must hold right after the weights:
/// its fewest rows (the staged width when retained rows are automatic, the
/// operator's rows otherwise) plus the committed prefix snapshots reserved
/// beside them, each priced at the model's per-row bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CudaStateFloor {
    pub(super) retained_rows: usize,
    pub(super) staged_rows: usize,
    /// Prefix snapshot budget when reuse engages for the model (an upper
    /// bound: a contract that keeps its state private reserves none).
    pub(super) snapshot_rows: usize,
    pub(super) per_sequence_bytes: u64,
    pub(super) automatic_rows: bool,
}

impl CudaStateFloor {
    pub(super) fn bytes(self) -> Result<u64> {
        self.retained_rows
            .checked_add(self.staged_rows)
            .and_then(|rows| rows.checked_add(self.snapshot_rows))
            .and_then(|rows| u64::try_from(rows).ok())
            .and_then(|rows| rows.checked_mul(self.per_sequence_bytes))
            .ok_or_else(|| Error::ModelLoadError("CUDA recurrent state floor overflow".into()))
    }
}

/// Price the recurrent state floor before any weight streams. A plan whose
/// weights, floor state and safety reserve exceed the device can never pass
/// the resident fit, so it fails here with the fit's guidance. Otherwise the
/// load authorization also covers the floor, so a resident model in the way
/// is evicted, or the load refused, before the weights are read.
pub(super) fn admit_cuda_state_floor(
    variant: ModelVariant,
    plan: &ModelResourcePlan,
    floor: CudaStateFloor,
    safety_reserve_bytes: u64,
    device_capacity: ResourceAmount,
) -> Result<ResourceVector> {
    let (ResourceAmount::Known(weights_bytes), ResourceAmount::Known(load_peak_bytes)) = (
        plan.resident_authorization.device_bytes,
        plan.load_authorization.device_bytes,
    ) else {
        return Err(Error::ModelLoadError(format!(
            "{variant} CUDA weight plan must be known before state admission"
        )));
    };
    let minimum_state_bytes = floor.bytes()?;
    let required_bytes = weights_bytes
        .checked_add(minimum_state_bytes)
        .and_then(|bytes| bytes.checked_add(safety_reserve_bytes))
        .ok_or_else(|| Error::ModelLoadError("CUDA state floor overflow".into()))?;
    if let ResourceAmount::Known(device_bytes) = device_capacity {
        if required_bytes > device_bytes {
            let guidance = if floor.automatic_rows {
                crate::engine::CUDA_AUTOMATIC_ROWS_GUIDANCE
            } else {
                crate::engine::CUDA_EXPLICIT_ROWS_GUIDANCE
            };
            return Err(Error::ModelLoadError(format!(
                "CUDA managed context cannot fit the model-authored minimum before loading {variant}: retained_sequences={}, staged_transactions={}, prefix_snapshots={}, per_sequence_state_bytes={}, minimum_state_bytes={minimum_state_bytes}, weights_bytes={weights_bytes}, safety_reserve={safety_reserve_bytes}, device_capacity={device_bytes}; {guidance}",
                floor.retained_rows, floor.staged_rows, floor.snapshot_rows, floor.per_sequence_bytes,
            )));
        }
    }
    Ok(ResourceVector {
        device_bytes: ResourceAmount::Known(load_peak_bytes.max(required_bytes)),
        ..plan.load_authorization
    })
}

/// Fold the request floor into an ASR load's authorization before anything is
/// evicted for it (ADM-10): the weights must leave room for one request
/// envelope per execution slot, or no request could run beside the loaded
/// model. The minimal state itself (one page per row) is small beside it.
pub(super) fn admit_asr_request_floor(
    plan: &ModelResourcePlan,
    backend: BackendKind,
    floor_bytes: u64,
) -> Result<ResourceVector> {
    let (ResourceAmount::Known(resident), ResourceAmount::Known(peak)) = (
        load_domain_bytes(plan.resident_authorization, backend),
        load_domain_bytes(plan.load_authorization, backend),
    ) else {
        return Ok(plan.load_authorization);
    };
    let required = resident
        .checked_add(floor_bytes)
        .ok_or_else(|| Error::ModelLoadError("ASR request floor overflow".into()))?
        .max(peak);
    let mut authorization = plan.load_authorization;
    match backend {
        BackendKind::Cpu => authorization.host_bytes = ResourceAmount::Known(required),
        BackendKind::Metal => authorization.unified_bytes = ResourceAmount::Known(required),
        BackendKind::Cuda => authorization.device_bytes = ResourceAmount::Known(required),
    }
    Ok(authorization)
}

/// The load authorization admission checks for `variant`: the plan's, with
/// an ASR model's per-slot request floor folded in ([`admit_asr_request_floor`]).
pub(super) fn load_authorization_with_request_floor(
    variant: ModelVariant,
    plan: &ModelResourcePlan,
    backend: BackendKind,
    request_slots: usize,
) -> Result<ResourceVector> {
    if !asr_memory::is_asr_variant(variant) {
        return Ok(plan.load_authorization);
    }
    let Some(envelope) =
        super::super::asr_envelope::deferred_request_envelope(variant, backend, request_slots)?
    else {
        return Ok(plan.load_authorization);
    };
    let floor_bytes = match backend {
        BackendKind::Cuda => envelope.accelerator_bytes,
        BackendKind::Cpu | BackendKind::Metal => envelope.total_bytes()?,
    };
    admit_asr_request_floor(plan, backend, floor_bytes)
}

/// A request envelope in `backend`'s ledger domains: host and accelerator
/// share one memory on CPU and Metal; CUDA splits them.
pub(super) fn request_envelope_resources(
    backend: BackendKind,
    envelope: super::super::asr_envelope::AsrRequestEnvelope,
) -> Result<ResourceVector> {
    let mut resources = ResourceVector::zero();
    match backend {
        BackendKind::Cpu => {
            resources.host_bytes = ResourceAmount::Known(envelope.total_bytes()?);
        }
        BackendKind::Metal => {
            resources.unified_bytes = ResourceAmount::Known(envelope.total_bytes()?);
        }
        BackendKind::Cuda => {
            resources.host_bytes = ResourceAmount::Known(envelope.host_bytes);
            resources.device_bytes = ResourceAmount::Known(envelope.accelerator_bytes);
        }
    }
    Ok(resources)
}

/// The short refusal of a load whose per-slot request envelope does not
/// fit beside its weights: what the envelope needs in the load's domain and
/// what planning has left (a 503; the terms are logged by the caller).
pub(super) fn request_envelope_shortfall(
    variant: ModelVariant,
    backend: BackendKind,
    envelope: super::super::asr_envelope::AsrRequestEnvelope,
    available_bytes: u64,
) -> Result<Error> {
    let need = match backend {
        BackendKind::Cuda => envelope.accelerator_bytes,
        BackendKind::Cpu | BackendKind::Metal => envelope.total_bytes()?,
    };
    Ok(Error::Overloaded(capacity_shortfall_reason(
        format_args!("not enough memory to run {variant} requests"),
        need,
        available_bytes,
    )))
}

/// Why a load cannot start once nothing more can be evicted for it, in the
/// terms a client can act on: what it needs, what is free, and the one thing
/// that would make room (unloading named models only when that would cover
/// the gap; otherwise what is actually short).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct LoadShortfall {
    pub(super) backend: BackendKind,
    pub(super) admission: crate::engine::MetalLoadAdmission,
    pub(super) required_bytes: u64,
    /// Ledger headroom: the memory budget less every reservation.
    pub(super) ledger_headroom: u64,
    /// The memory budget the ledger is held to.
    pub(super) ledger_ceiling: Option<u64>,
    /// The live sample's headroom (the GPU working set; host memory too
    /// under `strict`), when the backend has one.
    pub(super) live_headroom: Option<u64>,
    /// Bytes the resident models and their state hold.
    pub(super) held_by_residents: u64,
    /// Residents the load may not evict: loaded explicitly (pinned) or
    /// serving a request.
    pub(super) unevictable: Vec<ModelVariant>,
}

impl LoadShortfall {
    fn available(&self) -> u64 {
        self.live_headroom
            .map_or(self.ledger_headroom, |live| live.min(self.ledger_headroom))
    }

    pub(super) fn message(&self, variant: ModelVariant) -> String {
        let gb = |bytes: u64| bytes as f64 / 1e9;
        let available = self.available();
        let mut message = capacity_shortfall_reason(
            format_args!("Cannot load {variant}"),
            self.required_bytes,
            available,
        );
        let freeable = available.saturating_add(self.held_by_residents);
        let live_binds = self
            .live_headroom
            .is_some_and(|live| live < self.ledger_headroom);
        if !self.unevictable.is_empty() && self.required_bytes <= freeable {
            message.push_str("; ");
            message.push_str(&unload_advice(
                &self.unevictable,
                Some(self.held_by_residents),
            ));
        } else if live_binds && self.backend == BackendKind::Metal {
            if self.admission == crate::engine::MetalLoadAdmission::Strict {
                message.push_str(
                    "; macOS reports less memory available because other apps are using it \
                     (IZWI_METAL_LOAD_ADMISSION=strict): close some, or set it to advisory to \
                     let macOS page them out",
                );
            } else {
                message.push_str(
                    "; that is all the GPU working set (this Mac's limit for GPU memory) has \
                     free",
                );
            }
        } else if let Some(ceiling) = self
            .ledger_ceiling
            .filter(|ceiling| self.required_bytes > *ceiling)
        {
            message.push_str(&format!(
                "; the model is larger than the {:.2} GB memory budget ({})",
                gb(ceiling),
                match self.backend {
                    BackendKind::Cpu => "IZWI_CPU_MEMORY_BUDGET_BYTES",
                    BackendKind::Metal => {
                        "IZWI_METAL_MEMORY_BUDGET_BYTES, by default 90% of the GPU working set"
                    }
                    BackendKind::Cuda => "IZWI_CUDA_MEMORY_BUDGET_BYTES",
                },
            ));
        } else {
            message.push_str(
                "; the memory budget is held by running requests and model state: retry when \
                 they finish",
            );
        }
        message
    }
}

/// The residents a load of `requested` may not evict: loaded explicitly
/// (pinned) or serving a request.
pub(super) fn unevictable_residents(
    residents: &[ModelVariant],
    requested: ModelVariant,
    active: &HashSet<ModelVariant>,
    pinned: &HashSet<ModelVariant>,
) -> Vec<ModelVariant> {
    let mut unevictable: Vec<ModelVariant> = residents
        .iter()
        .copied()
        .filter(|variant| {
            *variant != requested && (active.contains(variant) || pinned.contains(variant))
        })
        .collect();
    unevictable.sort_by_key(ToString::to_string);
    unevictable
}

/// "X is loaded explicitly or serving requests (and holds N GB): unload it to
/// make room", for the residents a load may not evict.
fn unload_advice(unevictable: &[ModelVariant], held_bytes: Option<u64>) -> String {
    let one = unevictable.len() == 1;
    let held = held_bytes.map_or(String::new(), |bytes| {
        format!(
            " and {} {:.2} GB",
            if one { "holds" } else { "hold" },
            bytes as f64 / 1e9
        )
    });
    format!(
        "{} {} loaded explicitly or serving requests{held}: unload {} to make room",
        join_variants(unevictable),
        if one { "is" } else { "are" },
        if one { "it" } else { "one of them" },
    )
}

fn join_variants(variants: &[ModelVariant]) -> String {
    let names: Vec<String> = variants.iter().map(ToString::to_string).collect();
    match names.as_slice() {
        [] => String::new(),
        [one] => one.clone(),
        [rest @ .., last] => format!("{} and {last}", rest.join(", ")),
    }
}

/// Fixed part of the observed-materialization tolerance. Candle's Metal pool
/// keeps a buffer per size class it has served even after a synchronization
/// drops released ones, and the plan leaves out derived tensors under about
/// 1 MiB per family; a CUDA context lazily allocates module and stream
/// memory on its first kernels. A few hundred MiB covers these.
pub(super) const MATERIALIZATION_TOLERANCE_FLOOR_BYTES: u64 = 256 * 1024 * 1024;

/// How far observed growth may exceed what the ledger authorized before the
/// load fails: an eighth of the authorization (the plans' own conservative
/// approximations, such as the Metal pool serving a smaller tensor from a
/// larger freed buffer, are well inside that), and never less than
/// [`MATERIALIZATION_TOLERANCE_FLOOR_BYTES`]. An excess beyond it means the
/// plan prices a different representation than the loader allocates, which
/// is the defect class the reconciliation exists to catch.
pub(super) fn materialization_tolerance_bytes(authorized_bytes: u64) -> u64 {
    (authorized_bytes / 8).max(MATERIALIZATION_TOLERANCE_FLOOR_BYTES)
}

/// The device growth observed between two checkpoints of a load, failing the
/// load when it exceeds what the ledger authorized over the same interval
/// plus [`materialization_tolerance_bytes`]. `own_authorized_bytes` is what
/// this load held before `before` was sampled (its weight peak); every lease
/// reserved in between, the load's own state leases and any concurrent
/// request workspace alike, is authorized through the ledger growth, so a
/// concurrent allocation counts on both sides.
pub(super) fn reconcile_observed_growth(
    variant: ModelVariant,
    stage: &str,
    before: DeviceUsageSample,
    after: DeviceUsageSample,
    own_authorized_bytes: u64,
) -> Result<u64> {
    let observed_growth = after.observed.saturating_sub(before.observed);
    let authorized_bytes =
        own_authorized_bytes.saturating_add(after.reserved.saturating_sub(before.reserved));
    let tolerance_bytes = materialization_tolerance_bytes(authorized_bytes);
    if observed_growth > authorized_bytes.saturating_add(tolerance_bytes) {
        return Err(Error::ModelLoadError(format!(
            "{variant} {stage} allocated more device memory than its admission authorized: observed_bytes={observed_growth}, authorized_bytes={authorized_bytes}, tolerance_bytes={tolerance_bytes}; the residency plan under-prices this load"
        )));
    }
    Ok(observed_growth)
}

/// The resource plan load admission prices for `variant` at `model_path` on
/// `backend` (`device` is that backend's device). The lifecycle and the Metal
/// footprint harness both price through this one function.
pub(super) fn admission_resource_plan(
    variant: ModelVariant,
    model_path: &Path,
    backend: BackendKind,
    device: &crate::backends::DeviceProfile,
    performance: &crate::performance::PerformanceConfig,
    granite_speech: &crate::config::GraniteSpeechOptions,
) -> Result<ModelResourcePlan> {
    if variant == ModelVariant::Qwen3827BFp8 {
        if backend != BackendKind::Cuda
            && crate::models::architectures::qwen38::native::synthetic_geometry_enabled()
        {
            // Fixture load (benchmark/CI): price the actual checkpoint
            // instead of the pinned 27B constants.
            let estimate = qwen38_synthetic_fixture_estimate(model_path)?;
            return Ok(model_resource_plan(backend, estimate));
        }
        // Every backend prices the checkpoint under the residency policy
        // the loader resolves (CUDA runs its cached device gates here).
        return qwen38_memory::resource_plan(model_path, &device.device, performance);
    }
    if variant == ModelVariant::Qwen36Moe35BA3BFp8 {
        if crate::models::architectures::qwen36moe::native::synthetic_geometry_enabled() {
            // Fixture load (benchmark/CI): price the actual checkpoint
            // instead of the pinned 35B constants.
            let estimate = qwen36moe_memory::synthetic_fixture_estimate(model_path)?;
            return Ok(model_resource_plan(backend, estimate));
        }
        return qwen36moe_memory::resource_plan(backend, performance);
    }
    if let Some(plan) = checkpoint_resource_plan(variant, model_path, device, granite_speech)? {
        return Ok(plan);
    }
    let estimate = if backend == BackendKind::Cuda {
        model_memory_estimate(variant)
    } else {
        portable_model_memory_estimate(backend, variant, model_path)?
    };
    Ok(model_resource_plan(backend, estimate))
}

impl ModelLifecycleController {
    pub(in crate::runtime::lifecycle) async fn ensure_model_budget_before_load(
        &self,
        requested_variant: ModelVariant,
        max_loaded_models: Option<usize>,
    ) -> Result<()> {
        let Some(max_loaded_models) = max_loaded_models else {
            return Ok(());
        };

        loop {
            let resident_variants = self.known_resident_variants().await;
            if residency_budget_has_capacity(
                &resident_variants,
                requested_variant,
                max_loaded_models,
            ) {
                return Ok(());
            }

            let mut active_variants = self.core_engine.active_model_variants().await;
            active_variants.extend(
                resident_variants
                    .iter()
                    .copied()
                    .filter(|variant| self.model_manager.active_residency_leases(*variant) > 0),
            );
            let mut ready_variants = Vec::with_capacity(resident_variants.len());
            for variant in &resident_variants {
                if self.resident_phase(*variant)
                    == Some(crate::runtime::lifecycle::controller::ResidentPhase::Ready)
                    || self.model_manager.is_ready(*variant).await
                {
                    ready_variants.push(*variant);
                }
            }
            let last_used = self.model_last_used.lock().await.clone();
            let pinned_variants = self.pinned_variants.lock().await.clone();
            let Some(victim) = select_lru_eviction_candidate(
                &ready_variants,
                requested_variant,
                &active_variants,
                &pinned_variants,
                &last_used,
            ) else {
                let unevictable = unevictable_residents(
                    &resident_variants,
                    requested_variant,
                    &active_variants,
                    &pinned_variants,
                );
                let advice = if unevictable.is_empty() {
                    "the other models are still loading: retry when they finish".to_string()
                } else {
                    unload_advice(&unevictable, None)
                };
                return Err(Error::Overloaded(format!(
                    "Cannot load {requested_variant}: the {max_loaded_models}-model residency budget (IZWI_MAX_LOADED_MODELS) is full and {advice}"
                )));
            };

            info!(
                requested_variant = %requested_variant,
                victim = %victim,
                max_loaded_models,
                "Evicting idle model before loading its replacement"
            );
            self.unload_model_locked(victim).await?;
        }
    }

    pub(super) fn model_resource_plan(
        &self,
        variant: ModelVariant,
        model_path: &Path,
    ) -> Result<ModelResourcePlan> {
        let context = self.backend_router.context();
        admission_resource_plan(
            variant,
            model_path,
            context.backend_kind,
            &context.device,
            &self.config.performance,
            &self.config.granite_speech,
        )
    }

    /// The recurrent state floor of a model whose CUDA context is fitted
    /// against resident headroom; `None` for every other load.
    pub(super) fn cuda_state_floor(
        &self,
        variant: ModelVariant,
        model_path: &Path,
    ) -> Result<Option<CudaStateFloor>> {
        let policy =
            managed_chat_capacity_policy(variant, self.backend_router.context().backend_kind);
        if policy.cuda_resident_pool == crate::engine::CudaResidentPool::Disabled {
            return Ok(None);
        }
        let per_sequence_bytes = match variant {
            ModelVariant::Qwen3827BFp8 => qwen38_memory::per_sequence_state_bytes(model_path)?,
            ModelVariant::Qwen36Moe35BA3BFp8 => {
                use crate::models::architectures::qwen36moe::native as qwen36moe;
                if qwen36moe::synthetic_geometry_enabled() {
                    // Fixture loads do not have the pinned 35B geometry.
                    return Ok(None);
                }
                hybrid_recurrent_state_bytes_per_sequence(qwen36moe_state_geometry(
                    &qwen36moe::pinned_native_config().text,
                ))?
            }
            _ => return Ok(None),
        };
        let engine = self.core_engine.config();
        let staged_rows = match policy.staged_transaction_rows {
            Some(rows) => usize::try_from(rows).unwrap_or(usize::MAX),
            None => engine.max_staged_transactions,
        };
        let automatic_rows = engine.automatic_row_limits.retained_sequences;
        let retained_rows = if automatic_rows {
            staged_rows.min(engine.max_retained_sequences)
        } else {
            engine.max_retained_sequences
        };
        // The fit reserves the prefix snapshot budget beside the rows when
        // the model shares hybrid prefixes (see `prefix_snapshot_rows_for`).
        let snapshot_rows = if engine.enable_prefix_caching
            && engine.prefix_cache_pages_or_default() > 0
            && crate::catalog::prefix_reuse_engages(
                variant,
                BackendKind::Cuda,
                self.model_registry.prefix_reuse_mode(),
            ) {
            usize::try_from(engine.max_prefix_snapshots).unwrap_or(usize::MAX)
        } else {
            0
        };
        Ok(Some(CudaStateFloor {
            retained_rows,
            staged_rows: staged_rows.min(retained_rows),
            snapshot_rows,
            per_sequence_bytes,
            automatic_rows,
        }))
    }

    /// A Metal load the GPU working set admits but the host's available
    /// memory does not cover goes ahead under the advisory host policy:
    /// macOS compresses and swaps other processes to make room. Record why it
    /// may be slow for the caller, and log it.
    async fn note_advisory_host_shortfall(
        &self,
        variant: ModelVariant,
        backend: BackendKind,
        required_bytes: u64,
    ) {
        let authority = self.coordinator.resource_authority();
        // The host sample runs `sysctl`; keep it off the async workers.
        let available = tokio::task::spawn_blocking(move || {
            authority.advisory_host_shortfall(backend, required_bytes)
        })
        .await
        .ok()
        .flatten();
        let warning = available.map(|available_bytes| {
            warn!(
                model = %variant,
                required_bytes,
                available_bytes,
                "Loading beyond the host memory macOS reports available; it will compress or swap other processes' memory"
            );
            format!(
                "{variant} needs {:.2} GB and macOS reports {:.2} GB available, so it will compress or swap other apps' memory to make room; loads may be slow",
                required_bytes as f64 / 1e9,
                available_bytes as f64 / 1e9,
            )
        });
        self.set_load_warning(variant, warning);
    }

    pub(super) async fn ensure_portable_memory_before_load(
        &self,
        requested_variant: ModelVariant,
        required_bytes: u64,
    ) -> Result<()> {
        let backend = self.backend_router.context().backend_kind;
        if backend == BackendKind::Cuda {
            return Ok(());
        }
        self.set_load_warning(requested_variant, None);
        let mut pooled_scratch_flushed = false;
        loop {
            let ResourceAmount::Known(planning_headroom) = self
                .coordinator
                .resource_authority()
                .planning_headroom_bytes(backend)?
            else {
                return Ok(());
            };
            // Metal loads allocate the full working set at the load fence, so
            // the pre-load check must also respect probed live availability —
            // static ledger estimates alone under-count GGUF materialization
            // and pooled scratch. What the live sample bounds (the GPU working
            // set, host memory too under `strict`) follows
            // `IZWI_METAL_LOAD_ADMISSION`. An unavailable live probe never
            // blocks a load; planning headroom remains the contract then.
            let live_headroom = self
                .coordinator
                .resource_authority()
                .live_preload_headroom_bytes(backend)?;
            let headroom = live_headroom
                .map(|live| planning_headroom.min(live))
                .unwrap_or(planning_headroom);
            if required_bytes <= headroom {
                self.note_advisory_host_shortfall(requested_variant, backend, required_bytes)
                    .await;
                return Ok(());
            }
            let resident_variants = self.known_resident_variants().await;
            let mut active_variants = self.core_engine.active_model_variants().await;
            active_variants.extend(
                resident_variants
                    .iter()
                    .copied()
                    .filter(|variant| self.model_manager.active_residency_leases(*variant) > 0),
            );
            let last_used = self.model_last_used.lock().await.clone();
            let pinned_variants = self.pinned_variants.lock().await.clone();
            let Some(victim) = select_lru_eviction_candidate(
                &resident_variants,
                requested_variant,
                &active_variants,
                &pinned_variants,
                &last_used,
            ) else {
                if backend == BackendKind::Metal && !pooled_scratch_flushed {
                    // The pool retains released activation scratch until the
                    // last model unloads, which can hide gigabytes of
                    // reclaimable headroom from the live sample. Flushing it
                    // is safe (only unchecked-out buffers are dropped) and
                    // beats failing a load that would fit. One flush per
                    // admission decision; persistent shortage still errors.
                    pooled_scratch_flushed = true;
                    info!(
                        requested_variant = %requested_variant,
                        planning_headroom,
                        live_headroom,
                        "Reclaiming pooled Metal scratch before rejecting the load"
                    );
                    self.flush_metal_pools().await;
                    self.coordinator
                        .resource_authority()
                        .refresh_physical_capacity_after_release();
                    continue;
                }
                tracing::warn!(
                    "Cannot fit {requested_variant} model tensors before state allocation: load_peak_bytes={required_bytes}, planning_headroom={planning_headroom}, live_headroom={live_headroom:?}, backend={backend:?}"
                );
                let authority = self.coordinator.resource_authority();
                let shortfall = LoadShortfall {
                    backend,
                    admission: authority.metal_load_admission(),
                    required_bytes,
                    ledger_headroom: planning_headroom,
                    ledger_ceiling: authority.planning_ceiling_bytes(backend),
                    live_headroom,
                    held_by_residents: authority.reserved_bytes_for_classes(
                        backend,
                        &[ReservationClass::Model, ReservationClass::Cache],
                    ),
                    unevictable: unevictable_residents(
                        &resident_variants,
                        requested_variant,
                        &active_variants,
                        &pinned_variants,
                    ),
                };
                return Err(Error::Overloaded(shortfall.message(requested_variant)));
            };
            info!(
                requested_variant = %requested_variant,
                victim = %victim,
                required_bytes,
                planning_headroom = headroom,
                "Evicting idle model before portable tensor allocation"
            );
            self.unload_model_locked(victim).await?;
        }
    }

    /// One reconciliation checkpoint for a load priced by a residency plan
    /// (ASR, the forced aligner, the diarizers and TTS), or `None` for other
    /// families, for weights the observer cannot see, and for backends
    /// without a device observer (CPU).
    pub(super) fn sample_device_usage(&self, variant: ModelVariant) -> Option<DeviceUsageSample> {
        let backend = self.backend_router.context().backend_kind;
        if !reconciles_device_usage(variant, backend) {
            return None;
        }
        let authority = self.coordinator.resource_authority();
        let observed = authority.observed_device_usage_bytes()?;
        Some(DeviceUsageSample {
            observed,
            reserved: authority.reserved_bytes_in_domain(backend),
        })
    }

    /// Staged rows for a chat model on CPU or Metal when the engine's are
    /// automatic: one step's launch at the model's own decode width, which
    /// may exceed the engine rows' portable width (`catalog::batch_width`).
    /// `None` keeps the engine rows (CUDA, or an operator's explicit limit).
    pub(super) fn portable_chat_staged_rows(
        &self,
        bundle_draft: &LoadedModelBundleDraft,
        backend: BackendKind,
    ) -> Result<Option<u32>> {
        let engine = self.core_engine.config();
        if backend == BackendKind::Cuda || !engine.automatic_row_limits.staged_transactions {
            return Ok(None);
        }
        let width = bundle_draft
            .execution_contracts(CapabilityKind::Chat)?
            .iter()
            .flat_map(|contract| contract.stages.iter())
            .map(|stage| stage.max_batch_size)
            .max()
            .unwrap_or(1);
        let rows = crate::runtime::service::automatic_staged_transaction_rows(
            width,
            engine
                .resolved_physical_execution_capacity()
                .candidate_dispatch_limit
                .get(),
            engine.max_retained_sequences,
        );
        u32::try_from(rows)
            .map(Some)
            .map_err(|_| Error::InvalidInput("managed state transaction limit exceeds u32".into()))
    }

    pub(super) fn asr_state_rows(&self) -> (usize, usize) {
        let engine = self.core_engine.config();
        asr_state_rows(
            engine.automatic_row_limits.retained_sequences,
            engine.max_retained_sequences,
            engine.max_staged_transactions,
        )
    }

    /// Reserve one request envelope per execution slot for the state fits
    /// to leave free (ASR: one full chunk; TTS: one request at the largest
    /// automatic budget), or `None` for a variant without one.
    pub(super) async fn hold_request_envelope(
        &self,
        variant: ModelVariant,
        backend: BackendKind,
    ) -> Result<Option<ResourceLease>> {
        let slots = self.coordinator.physical_execution_admission().capacity();
        let (envelope, label) =
            match super::super::asr_envelope::deferred_request_envelope(variant, backend, slots)? {
                Some(envelope) => (envelope, "asr"),
                None => match self.tts_request_envelope(variant).await? {
                    Some(envelope) => (envelope.scaled(slots)?, "tts"),
                    None => return Ok(None),
                },
            };
        let resources = request_envelope_resources(backend, envelope)?;
        let authority = self.coordinator.resource_authority();
        let lease = match authority.reserve(
            ReservationOwner::new(
                ReservationClass::Model,
                format!("{variant}:{label}-request-envelope"),
            ),
            resources,
        ) {
            Ok(lease) => lease,
            Err(Error::Overloaded(detail)) => {
                let available = match authority.planning_headroom_bytes(backend)? {
                    ResourceAmount::Known(bytes) => bytes,
                    ResourceAmount::Unknown => 0,
                };
                tracing::warn!(
                    model = %variant,
                    request_slots = slots,
                    host_bytes = envelope.host_bytes,
                    accelerator_bytes = envelope.accelerator_bytes,
                    planning_headroom = available,
                    %detail,
                    "The {label} request envelope does not fit beside the loaded weights"
                );
                return Err(request_envelope_shortfall(
                    variant, backend, envelope, available,
                )?);
            }
            Err(error) => return Err(error),
        };
        info!(
            model = %variant,
            request_slots = slots,
            host_bytes = envelope.host_bytes,
            accelerator_bytes = envelope.accelerator_bytes,
            "Holding the {label} request envelope out of the state fits"
        );
        Ok(Some(lease))
    }

    /// One TTS request's envelope, priced by the loaded model (P2.3), for
    /// the families whose load fits state beside it: Qwen3-TTS.
    pub(super) async fn tts_request_envelope(
        &self,
        variant: ModelVariant,
    ) -> Result<Option<super::super::asr_envelope::AsrRequestEnvelope>> {
        if variant.family() != crate::catalog::ModelFamily::Qwen3Tts {
            return Ok(None);
        }
        let Some(model) = self.model_registry.get_qwen_tts(variant).await else {
            return Ok(None);
        };
        let (host_bytes, accelerator_bytes) = model.request_envelope_bytes()?;
        Ok(Some(super::super::asr_envelope::AsrRequestEnvelope {
            host_bytes,
            accelerator_bytes,
        }))
    }

    /// Grows a loading slot's authorization to the precision reload plan
    /// of `variant` before a component whose load-time probe failed reloads
    /// at `request.dtype`, through the same live check as the load itself
    /// (pooled-scratch flush and eviction included). Returns the reload plan
    /// that now prices the load, or `None` when the load has none (its plan
    /// already covers the reload). Refuses, with the shortfall, when the
    /// reload cannot fit.
    pub(super) async fn admit_precision_reload(
        &self,
        variant: ModelVariant,
        model_path: &Path,
        current: &ModelResourcePlan,
        request: &PrecisionReloadRequest,
    ) -> Result<Option<ModelResourcePlan>> {
        let device = &self.backend_router.context().device;
        let Some(reload) = asr_memory::precision_reload_plan(variant, model_path, device)? else {
            return Ok(None);
        };
        self.grow_load_authorization_for_reload(variant, current, reload, request)
            .await
            .map(Some)
    }

    pub(super) async fn grow_load_authorization_for_reload(
        &self,
        variant: ModelVariant,
        current: &ModelResourcePlan,
        reload: ModelResourcePlan,
        request: &PrecisionReloadRequest,
    ) -> Result<ModelResourcePlan> {
        let backend = self.backend_router.context().backend_kind;
        let refused = |error: Error| {
            Error::ModelLoadError(format!(
                "Cannot reload {variant} {} at {:?} after its load-time probe failed: {error}",
                request.component, request.dtype
            ))
        };
        let growth = match (
            load_domain_bytes(current.load_authorization, backend),
            load_domain_bytes(reload.load_authorization, backend),
        ) {
            (ResourceAmount::Known(current), ResourceAmount::Known(reload)) => {
                reload.saturating_sub(current)
            }
            _ => 0,
        };
        warn!(
            model = %variant,
            component = %request.component,
            dtype = ?request.dtype,
            growth_bytes = growth,
            "A lowered-precision load failed its probe; admitting the reload"
        );
        // The loader synchronized the device after dropping the lowered
        // component, so a fresh sample sees its memory free.
        self.coordinator
            .resource_authority()
            .refresh_physical_capacity_after_release();
        self.ensure_portable_memory_before_load(variant, growth)
            .await
            .map_err(refused)?;
        self.grow_loading_slot_authorization(variant, reload.load_authorization)
            .map_err(refused)?;
        Ok(reload)
    }

    pub(super) async fn reserve_model_resources(
        &self,
        requested_variant: ModelVariant,
        load_authorization: ResourceVector,
    ) -> Result<ResourceLease> {
        loop {
            match self.coordinator.resource_authority().reserve(
                ReservationOwner::new(ReservationClass::Model, requested_variant.to_string()),
                load_authorization,
            ) {
                Ok(lease) => return Ok(lease),
                Err(Error::Overloaded(resource_error)) => {
                    let resident_variants = self.known_resident_variants().await;
                    let mut active_variants = self.core_engine.active_model_variants().await;
                    active_variants.extend(resident_variants.iter().copied().filter(|variant| {
                        self.model_manager.active_residency_leases(*variant) > 0
                    }));
                    let mut ready_variants = Vec::new();
                    for variant in &resident_variants {
                        if self.resident_phase(*variant)
                            == Some(crate::runtime::lifecycle::controller::ResidentPhase::Ready)
                            || self.model_manager.is_ready(*variant).await
                        {
                            ready_variants.push(*variant);
                        }
                    }
                    let last_used = self.model_last_used.lock().await.clone();
                    let pinned_variants = self.pinned_variants.lock().await.clone();
                    let Some(victim) = select_lru_eviction_candidate(
                        &ready_variants,
                        requested_variant,
                        &active_variants,
                        &pinned_variants,
                        &last_used,
                    ) else {
                        let unevictable = unevictable_residents(
                            &resident_variants,
                            requested_variant,
                            &active_variants,
                            &pinned_variants,
                        );
                        let advice = if unevictable.is_empty() {
                            "retry when running requests finish".to_string()
                        } else {
                            unload_advice(&unevictable, None)
                        };
                        return Err(Error::Overloaded(format!(
                            "Cannot reserve memory for {requested_variant}: {resource_error}; {advice}"
                        )));
                    };
                    info!(
                        requested_variant = %requested_variant,
                        victim = %victim,
                        "Evicting idle model to satisfy the physical memory budget"
                    );
                    self.unload_model_locked(victim).await?;
                }
                Err(err) => return Err(err),
            }
        }
    }
}
