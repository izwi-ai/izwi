//! Runtime construction, backend selection, model management and drain.

use super::*;

impl RuntimeService {
    pub fn backend_context(&self) -> crate::backends::BackendContext {
        self.backend_router.context().clone()
    }

    pub(super) fn ensure_requested_backend_available(
        backend_context: &crate::backends::BackendContext,
    ) -> Result<()> {
        if backend_context.matches_preference() {
            return Ok(());
        }

        Err(Error::InferenceError(
            requested_backend_unavailable_message(backend_context),
        ))
    }

    /// Create a new inference engine.
    pub fn new(mut config: EngineConfig) -> Result<Self> {
        crate::env::validate_process_env()?;
        config.performance = config.performance.resolve_env()?;
        let backend_context =
            BackendRouter::resolve_context(config.backend, BackendSelectionSource::Config);
        Self::ensure_requested_backend_available(&backend_context)?;
        Self::new_with_backend_context(config, backend_context)
    }

    /// Create a production worker runtime on exactly the supervisor-assigned device.
    pub fn new_assigned(
        mut config: EngineConfig,
        assignment: RuntimeDeviceAssignment,
    ) -> Result<Self> {
        crate::env::validate_process_env()?;
        config.performance = config.performance.resolve_env()?;
        let backend_context =
            BackendRouter::resolve_assigned_context(&assignment, BackendSelectionSource::Config)?;
        config.backend = BackendPreference::from(assignment.backend_kind());
        Self::new_with_backend_context(config, backend_context)
    }

    pub(super) fn new_with_backend_context(
        mut config: EngineConfig,
        backend_context: crate::backends::BackendContext,
    ) -> Result<Self> {
        // Resolve catalog-auto prefix reuse (DS1.6) once, before the fail-closed
        // cache-policy boundary: explicit operator enablement keeps its salt
        // contract, catalog-auto engages with the operator namespace or a
        // generated per-process one, and the registry consults the catalog cell
        // of each loaded family on the active backend.
        let prefix_reuse_mode = if config.enable_prefix_caching {
            crate::catalog::PrefixReuseMode::Explicit
        } else if config.prefix_reuse_catalog_auto {
            crate::catalog::PrefixReuseMode::CatalogAuto
        } else {
            crate::catalog::PrefixReuseMode::Disabled
        };
        let (prefix_engaged, prefix_namespace) = crate::config::resolve_prefix_engagement(
            config.enable_prefix_caching,
            config.prefix_reuse_catalog_auto,
            config.managed_prefix_cache_salt.as_deref(),
        );
        if prefix_engaged {
            config.enable_prefix_caching = true;
            if let Some(namespace) = prefix_namespace {
                config.managed_prefix_cache_salt = Some(namespace);
            }
        }
        // The auto flag stays set for catalog-auto engagements so the policy
        // resolver can degrade instead of failing startup on a zero budget.
        // Reject unsupported or unsafe cache policy before any model registry,
        // device arena, or readiness state can be created.
        let cache_policy = config.resolved_kv_cache_policy(startup_policy_capacity_pages(
            backend_context.backend_kind,
            &config,
        ))?;
        if prefix_engaged
            && matches!(
                cache_policy.effective.prefix,
                crate::config::PrefixCachePolicy::Disabled
            )
        {
            // Catalog-auto degraded on this runtime's page budget; the
            // resolved policy is the truth for the engine core as well.
            config.enable_prefix_caching = false;
            config.managed_prefix_cache_salt = None;
        }
        // The engine never writes the process environment: binaries size the
        // CPU compute libraries once, single-threaded, through
        // `crate::env::configure_process_threads`.
        config.granite_speech = config.granite_speech.with_env_fallback();
        let model_manager = Arc::new(ModelManager::new(config.clone())?);

        let device = backend_context.device.clone();
        let selected_backend_kind = backend_context.backend_kind;
        // Zero means automatic administrative capacity, not zero usable rows.
        // This is only an upper bound: each loaded model fits exact physical state.
        // Automatic staged rows are resolved below from the launch width.
        let automatic_retained_sequences = config.max_retained_sequences == 0;
        let automatic_staged_transactions = config.max_staged_transactions == 0;
        // `max_num_seqs` sizes every row limit left automatic, and an
        // automatic tensor width (each chat family still caps it at its
        // certified width).
        let max_num_seqs = config.max_num_seqs.filter(|seqs| *seqs > 0);
        // Only an operator's own width may run above the certified widths;
        // one derived from `max_num_seqs` keeps them.
        let explicit_tensor_width = matches!(config.max_batch_size, BatchSizePreference::Fixed(_));
        let automatic_rows = match max_num_seqs {
            Some(seqs) => seqs,
            None if selected_backend_kind == BackendKind::Cuda => backend_context
                .device
                .capabilities
                .cuda_total_memory_bytes
                .map(|bytes| (bytes / (256 * 1024 * 1024)).max(1))
                .unwrap_or(1),
            None => 8,
        };
        if let (Some(seqs), BatchSizePreference::Auto) = (max_num_seqs, config.max_batch_size) {
            config.max_batch_size = BatchSizePreference::fixed(seqs)?;
        }
        for rows in [
            &mut config.max_scheduler_batch_size,
            &mut config.max_retained_sequences,
        ] {
            if *rows == 0 {
                *rows = automatic_rows;
            }
        }

        let model_registry = Arc::new(
            ModelRegistry::new_with_performance(
                config.models_dir.clone(),
                device.clone(),
                config.performance.clone(),
            )
            .with_prefix_reuse_mode(prefix_reuse_mode)
            .with_granite_speech(config.granite_speech.clone()),
        );

        let mut core_config = EngineCoreConfig::for_qwen3_tts();
        core_config.portable_context_auto = config.max_sequence_length.explicit_tokens().is_none();
        core_config.max_num_seqs = config.max_num_seqs.filter(|seqs| *seqs > 0);
        core_config.portable_context_reserve_bytes = config.portable_context_reserve_bytes;
        core_config.models_dir = config.models_dir.clone();
        core_config.performance = config.performance.clone();
        core_config.max_batch_size = config.max_scheduler_batch_size.max(1);
        core_config.max_tensor_batch_size = config.max_batch_size;
        core_config.explicit_tensor_width = explicit_tensor_width;
        core_config.physical_execution_mode = config.physical_execution_mode;
        core_config.max_physical_in_flight = config.max_physical_in_flight;
        core_config.max_retained_sequences = config.max_retained_sequences.max(1);
        core_config.max_staged_transactions = config.max_staged_transactions.max(1);
        core_config.automatic_row_limits.retained_sequences = automatic_retained_sequences;
        core_config.max_queued_requests = config.max_queued_requests.max(1);
        core_config.backend = selected_backend_kind;
        core_config.num_threads = config.num_threads.max(1);
        core_config.block_size = config.kv_page_size.max(1);
        // Model contracts read the page grid when they load; publish the one
        // the engine uses so the two can never disagree.
        crate::models::shared::attention::paged::publish_configured_kv_page_size(
            core_config.block_size,
        );
        core_config.kv_cache_dtype = cache_policy.effective.dtype.to_string();
        core_config.enable_prefix_caching = config.enable_prefix_caching;
        core_config.managed_prefix_cache_salt = config.managed_prefix_cache_salt.clone();
        // Prefix engagement is fully resolved above (the engine core reuses
        // the resolved namespace verbatim), but the auto marker is carried so
        // the core's own budget check degrades instead of failing startup.
        core_config.prefix_reuse_catalog_auto = config.prefix_reuse_catalog_auto;
        core_config.max_prefix_cache_pages = engine_core_prefix_cache_pages(
            config.max_prefix_cache_pages,
            &cache_policy.effective.prefix,
        );
        // After the prefix settings: CUDA sizes its page cap for a maximum-
        // length request plus the prefix budget.
        core_config.apply_backend_context_capacity(config.portable_context_ceiling());
        core_config.enable_chunked_prefill = config.enable_chunked_prefill;
        core_config
            .resolve_step_sizes(config.max_tokens_per_step, config.chunked_prefill_threshold);
        // DS4 hierarchical offload: explicit opt-in through the host pool
        // budget; the manager materializes pools only where committed prefix
        // pages exist.
        core_config.kv_host_pool_budget_bytes = config.kv_host_pool_budget_bytes;
        core_config.kv_offload_high_watermark = config.kv_offload_high_watermark;
        core_config.kv_offload_low_watermark = config.kv_offload_low_watermark;
        core_config.kv_offload_max_in_flight_pages = config.kv_offload_max_in_flight_pages;
        core_config.kv_offload_max_promotion_pages = config.kv_offload_max_promotion_pages;

        let mut worker_config = WorkerConfig::from(&core_config);
        let non_chat_row_width = crate::catalog::non_chat_row_width(
            selected_backend_kind,
            worker_config.max_tensor_batch_size,
            explicit_tensor_width,
        );
        if non_chat_row_width
            > crate::catalog::engine_row_width(
                selected_backend_kind,
                worker_config.max_tensor_batch_size,
            )
        {
            tracing::warn!(
                width = non_chat_row_width,
                backend = selected_backend_kind.as_str(),
                "explicit batch width exceeds the certified speech-model width on this backend; ASR, TTS and realtime rows run uncertified"
            );
        }
        if automatic_staged_transactions {
            // Engine rows back every model without its own row fit; a chat
            // model sizes its staged rows from its decode width at load.
            let staged = automatic_staged_transaction_rows(
                non_chat_row_width,
                core_config
                    .resolved_physical_execution_capacity()
                    .candidate_dispatch_limit
                    .get(),
                core_config.max_retained_sequences,
            );
            config.max_staged_transactions = staged;
            core_config.max_staged_transactions = staged;
            core_config.automatic_row_limits.staged_transactions = true;
        }
        worker_config.models_dir = config.models_dir.clone();
        worker_config.kv_cache_dtype = cache_policy.effective.dtype.to_string();
        worker_config.kv_page_size = config.kv_page_size.max(1);
        worker_config.model_registry = Some(model_registry.clone());
        worker_config.backend = selected_backend_kind;
        worker_config.backend_context = backend_context.clone();
        // Chat models clamp the engine width to their family's certified
        // width; every other capability runs at the engine rows' width. An
        // operator's explicit width applies as asked.
        let adapter_registry = Arc::new(
            RuntimeAdapterRegistry::built_in_with_execution_limits(
                non_chat_row_width,
                worker_config.request_parallelism,
            )?
            .with_chat_tensor_batch_size(worker_config.max_tensor_batch_size)
            .with_explicit_tensor_width(explicit_tensor_width),
        );
        worker_config.static_tensor_batch_variants =
            Arc::new(adapter_registry.static_tensor_batch_variants(selected_backend_kind));
        let execution_parallelism = effective_physical_execution_parallelism(
            core_config.physical_execution_mode,
            worker_config.request_parallelism,
            core_config
                .resolved_physical_execution_capacity()
                .physical_launch_limit
                .get(),
        );
        let coordinator = Arc::new(InferenceCoordinator::new_with_device(
            selected_backend_kind,
            device.clone(),
            execution_parallelism,
            core_config.max_queued_requests,
        )?);
        let qwen_asr_encoder_batcher = Arc::new(QwenAsrEncoderBatcher::new(coordinator.clone()));
        let whisper_encoder_batcher = Arc::new(WhisperEncoderBatcher::new(coordinator.clone()));
        let vibevoice_encoder_batcher = Arc::new(VibeVoiceEncoderBatcher::new(coordinator.clone()));
        let granite_speech_encoder_batcher =
            Arc::new(GraniteSpeechEncoderBatcher::new(coordinator.clone()));
        let asr_realtime_sessions = RealtimeAsrSessionPolicy::resolve_for_sequences(
            config.asr_realtime,
            selected_backend_kind,
            max_num_seqs,
        )?;
        let realtime_asr_sequence_capacity = asr_realtime_sessions.retained_sequence_capacity()?;
        worker_config.resource_authority = Some(coordinator.resource_authority());
        worker_config.physical_execution_admission =
            Some(coordinator.physical_execution_admission());
        let concurrency = ResolvedConcurrency {
            profile: ConcurrencyProfile::resolve(selected_backend_kind, max_num_seqs),
            max_num_seqs,
            scheduler_rows: core_config.max_batch_size,
            retained_sequences: core_config.max_retained_sequences,
            staged_transactions: core_config.max_staged_transactions,
            tensor_width: worker_config.max_tensor_batch_size,
            non_chat_row_width,
            realtime_asr_sessions: asr_realtime_sessions.max_sessions(),
            chunked_prefill: core_config.effective_chunked_prefill(),
        };
        tracing::info!(
            profile = concurrency.profile.as_str(),
            max_num_seqs = ?concurrency.max_num_seqs,
            scheduler_rows = concurrency.scheduler_rows,
            retained_sequences = concurrency.retained_sequences,
            staged_transactions = concurrency.staged_transactions,
            tensor_width = concurrency.tensor_width,
            non_chat_row_width = concurrency.non_chat_row_width,
            realtime_asr_sessions = concurrency.realtime_asr_sessions,
            chunked_prefill = concurrency.chunked_prefill,
            "resolved runtime concurrency"
        );
        let core_engine = Arc::new(CoreEngine::new_with_worker(core_config, worker_config)?);
        let backend_router = BackendRouter::from_context(backend_context);
        let tokenizer = Arc::new(RwLock::new(None));
        let codec = Arc::new(RwLock::new(AudioCodec::new()));
        let loaded_tts_variant = Arc::new(RwLock::new(None));
        let model_lifecycle = Arc::new(ModelLifecycleController::new(
            config.clone(),
            backend_router.clone(),
            adapter_registry.clone(),
            model_manager.clone(),
            model_registry.clone(),
            core_engine.clone(),
            coordinator.clone(),
            tokenizer,
            codec.clone(),
            loaded_tts_variant.clone(),
            realtime_asr_sequence_capacity,
        ));

        let max_loaded_models = config
            .max_loaded_models
            .or_else(|| positive_usize_env("IZWI_MAX_LOADED_MODELS"));

        Ok(Self {
            config,
            cache_policy,
            backend_router,
            inference_broker: InferenceBroker::from_env(),
            adapter_registry,
            model_manager,
            model_registry,
            codec,
            core_engine,
            coordinator,
            qwen_asr_encoder_batcher,
            whisper_encoder_batcher,
            vibevoice_encoder_batcher,
            granite_speech_encoder_batcher,
            asr_realtime_sessions,
            telemetry: Arc::new(RuntimeTelemetryCollector::new(2048)),
            completion_waiters: Arc::new(Mutex::new(HashMap::new())),
            next_completion_waiter_registration: AtomicU64::new(1),
            step_driver_task: Mutex::new(None),
            step_driver_wakeup: Arc::new(Notify::new()),
            step_driver_started: AtomicBool::new(false),
            diarization_audio_decodes: Arc::new(AtomicU64::new(0)),
            loaded_tts_variant,
            max_loaded_models,
            model_lifecycle,
            concurrency,
        })
    }

    /// Get reference to model manager.
    pub fn model_manager(&self) -> &Arc<ModelManager> {
        &self.model_manager
    }

    /// List available models.
    pub async fn list_models(&self) -> Vec<ModelInfo> {
        self.model_manager.list_models().await
    }

    /// Get explicit artifact and residency state for a specific model.
    pub async fn model_lifecycle_snapshot(
        &self,
        variant: ModelVariant,
    ) -> Option<ModelLifecycleSnapshot> {
        self.model_manager.lifecycle_snapshot(variant).await
    }

    /// Get explicit artifact and residency states for all known models.
    pub async fn model_lifecycle_snapshots(&self) -> Vec<ModelLifecycleSnapshot> {
        self.model_manager.lifecycle_snapshots().await
    }

    /// Download a model.
    pub async fn download_model(&self, variant: ModelVariant) -> Result<()> {
        self.model_manager.download_model(variant).await?;
        Ok(())
    }

    /// Spawn a non-blocking background download.
    pub async fn spawn_download(
        &self,
        variant: ModelVariant,
    ) -> Result<broadcast::Receiver<DownloadProgress>> {
        self.model_manager.spawn_download(variant).await
    }

    /// Check if a download is active.
    pub async fn is_download_active(&self, variant: ModelVariant) -> bool {
        self.model_manager.is_download_active(variant).await
    }

    /// Get runtime configuration.
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// The rows the engine batches TTS (and every other non-chat
    /// capability) at: the resolved tensor width on CUDA, the portable two
    /// rows on CPU and Metal (`catalog::engine_row_width`). Callers that fan
    /// out TTS work size their concurrency from it; more would only queue.
    pub fn tts_row_width(&self) -> usize {
        self.adapter_registry.max_tensor_batch_size().max(1)
    }

    /// The concurrency this runtime resolved at startup (`max_num_seqs` and
    /// every limit it sized).
    pub fn resolved_concurrency(&self) -> &ResolvedConcurrency {
        &self.concurrency
    }

    /// Effective startup concurrency policy; never re-reads rollout environment variables.
    pub fn chat_concurrency_policy(
        &self,
    ) -> crate::engine::metrics::EngineChatConcurrencyPolicySnapshot {
        crate::engine::metrics::EngineChatConcurrencyPolicySnapshot::from_config(
            self.core_engine.config(),
        )
    }

    /// Immutable requested/effective KV cache policy selected at startup.
    pub fn resolved_kv_cache_policy(&self) -> &ResolvedKvCachePolicy {
        &self.cache_policy
    }

    /// Get codec sample rate.
    pub async fn sample_rate(&self) -> u32 {
        self.codec.read().await.sample_rate()
    }

    /// Create audio encoder.
    pub async fn audio_encoder(&self) -> AudioEncoder {
        let codec = self.codec.read().await;
        AudioEncoder::new(codec.sample_rate(), 1)
    }

    /// Get available speakers for loaded TTS model.
    pub async fn available_speakers(&self) -> Result<Vec<String>> {
        let variant = (*self.loaded_tts_variant.read().await)
            .ok_or_else(|| Error::InferenceError("No TTS model loaded".to_string()))?;
        let _lease = self
            .model_lifecycle
            .try_acquire_ready_lease(variant)
            .ok_or_else(|| Error::InferenceError("No TTS model loaded".to_string()))?;

        match variant.family() {
            crate::catalog::ModelFamily::Qwen3Tts => {
                let model = self
                    .model_registry
                    .get_qwen_tts(variant)
                    .await
                    .ok_or_else(|| Error::InferenceError("No TTS model loaded".to_string()))?;
                Ok(model.available_speakers().into_iter().cloned().collect())
            }
            crate::catalog::ModelFamily::KokoroTts => {
                let model = self
                    .model_registry
                    .get_kokoro(variant)
                    .await
                    .ok_or_else(|| Error::InferenceError("No TTS model loaded".to_string()))?;
                model.available_speakers()
            }
            crate::catalog::ModelFamily::VoxtralTts => {
                let model = self
                    .model_registry
                    .get_voxtral_tts(variant)
                    .await
                    .ok_or_else(|| Error::InferenceError("No TTS model loaded".to_string()))?;
                Ok(model.available_speakers())
            }
            crate::catalog::ModelFamily::VibeVoiceTts => {
                let model = self
                    .model_registry
                    .get_vibevoice_tts(variant)
                    .await
                    .ok_or_else(|| Error::InferenceError("No TTS model loaded".to_string()))?;
                Ok(model.available_speakers())
            }
            crate::catalog::ModelFamily::Lfm25Audio => Ok(
                crate::models::architectures::lfm25_audio::LFM25_AUDIO_BUILT_IN_SPEAKERS
                    .iter()
                    .map(|speaker| (*speaker).to_string())
                    .collect(),
            ),
            _ => Err(Error::InferenceError(format!(
                "Model {variant} does not expose TTS speakers"
            ))),
        }
    }

    /// Content identity of the loaded Fish weights, codec and tokenizer. Native
    /// loading computes this once; segment admission only clones the small hash.
    pub async fn fish_s2_artifact_fingerprint(&self) -> Option<String> {
        self.model_registry
            .get_fish_s2_tts(ModelVariant::FishAudioS2Pro)
            .await?
            .artifact_fingerprint()
            .map(str::to_owned)
    }

    /// Machine-readable diagnostics for the currently loaded direct TTS model.
    pub async fn loaded_tts_model_diagnostics(&self) -> Option<serde_json::Value> {
        let variant = (*self.loaded_tts_variant.read().await)?;
        match variant.family() {
            crate::catalog::ModelFamily::Qwen3Tts => {
                let model = self.model_registry.get_qwen_tts(variant).await?;
                serde_json::to_value(model.diagnostics()).ok()
            }
            crate::catalog::ModelFamily::VibeVoiceTts => {
                let model = self.model_registry.get_vibevoice_tts(variant).await?;
                serde_json::to_value(model.diagnostics()).ok()
            }
            crate::catalog::ModelFamily::FishS2Tts => {
                let model = self.model_registry.get_fish_s2_tts(variant).await?;
                serde_json::to_value(model.diagnostics()).ok()
            }
            _ => None,
        }
    }

    /// Registry-backed diagnostics for native model handles loaded in memory.
    pub async fn loaded_model_diagnostics(&self) -> Vec<LoadedModelDiagnostics> {
        self.model_registry.loaded_model_diagnostics().await
    }

    pub fn is_draining(&self) -> bool {
        self.coordinator.is_draining()
    }

    pub fn begin_drain(&self) {
        self.coordinator.begin_drain();
        self.step_driver_wakeup.notify_waiters();
    }

    pub async fn wait_for_drain(&self, timeout: Duration) -> Result<()> {
        self.begin_drain();
        self.coordinator
            .wait_for_idle(Instant::now() + timeout)
            .await
    }
}
