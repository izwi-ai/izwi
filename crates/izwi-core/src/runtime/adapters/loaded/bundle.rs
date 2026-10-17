//! Loaded model bundles: building a draft from the registry's factories and
//! sealing it.

use super::*;

impl RuntimeAdapterRegistry {
    pub(in crate::runtime::adapters) fn loaded_adapter_factory(
        &self,
        metadata: AdapterMetadata,
        backend_kind: BackendKind,
    ) -> Result<&dyn LoadedExecutionAdapterFactory> {
        let mut matches = self
            .loaded_adapter_factories
            .iter()
            .filter(|factory| factory.supports(metadata, backend_kind));
        let Some(selected) = matches.next() else {
            return Err(Error::ModelLoadError(format!(
                "loaded model {} capability {:?} has no execution adapter factory for {backend_kind:?}",
                metadata.model_variant, metadata.capability,
            )));
        };
        if let Some(ambiguous) = matches.next() {
            return Err(Error::ModelLoadError(format!(
                "loaded model {} capability {:?} matches both execution adapter factories `{}` and `{}`",
                metadata.model_variant,
                metadata.capability,
                selected.id(),
                ambiguous.id(),
            )));
        }
        Ok(selected.as_ref())
    }

    pub(in crate::runtime::adapters) fn loaded_native_variants(
        &self,
        backend_kind: BackendKind,
        batch_mode: NativeBatchMode,
    ) -> std::collections::HashSet<ModelVariant> {
        ModelVariant::all()
            .iter()
            .copied()
            .filter(|variant| {
                self.capabilities_for(*variant).into_iter().any(|metadata| {
                    self.loaded_adapter_factory(metadata, backend_kind)
                        .expect("factory ambiguity is rejected when the registry is built")
                        .batch_mode()
                        == batch_mode
                })
            })
            .collect()
    }

    pub(super) fn create_loaded_adapter(
        &self,
        execution_group_id: ExecutionGroupId,
        model_instance_id: ModelInstanceId,
        metadata: AdapterMetadata,
        backend_kind: BackendKind,
    ) -> Result<Arc<dyn LoadedExecutionAdapter>> {
        let context = LoadedAdapterFactoryContext {
            execution_group_id,
            model_instance_id,
            backend_kind,
            // A chat model runs at most its family's certified width.
            max_tensor_batch_size: if metadata.capability == CapabilityKind::Chat {
                self.max_chat_tensor_batch_size()
                    .min(crate::catalog::certified_decode_width(
                        metadata.model_variant.family(),
                        backend_kind,
                    ))
            } else {
                self.max_tensor_batch_size()
            },
            request_parallelism: self.request_parallelism(),
        };
        let adapter = self
            .loaded_adapter_factory(metadata, backend_kind)?
            .create(context, metadata)?;
        if adapter.metadata() != metadata {
            return Err(Error::ModelLoadError(format!(
                "loaded adapter factory returned mismatched metadata for {} capability {:?}",
                metadata.model_variant, metadata.capability
            )));
        }
        Ok(adapter)
    }
}

impl fmt::Debug for LoadedModelBundleDraft {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LoadedModelBundleDraft")
            .field("execution_group_id", &self.execution_group_id)
            .field("model_instance_id", &self.model_instance_id)
            .field("model_variant", &self.model_variant)
            .field("backend_kind", &self.backend_kind)
            .field("capability_count", &self.capabilities.len())
            .finish()
    }
}

impl LoadedModelBundleDraft {
    pub(crate) fn build(
        registry: &RuntimeAdapterRegistry,
        execution_group_id: ExecutionGroupId,
        model_instance_id: ModelInstanceId,
        model_variant: ModelVariant,
        backend_kind: BackendKind,
    ) -> Result<Self> {
        let metadata = registry.capabilities_for(model_variant);
        if metadata.is_empty() {
            return Err(Error::ModelLoadError(format!(
                "loaded model {model_variant} has no executable capability adapter"
            )));
        }
        let mut capabilities = HashMap::with_capacity(metadata.len());
        for metadata in metadata {
            let adapter = registry.create_loaded_adapter(
                execution_group_id,
                model_instance_id,
                metadata,
                backend_kind,
            )?;
            if capabilities.insert(metadata.capability, adapter).is_some() {
                return Err(Error::ModelLoadError(format!(
                    "loaded model {model_variant} has duplicate {:?} adapters",
                    metadata.capability
                )));
            }
        }
        Ok(Self {
            execution_group_id,
            model_instance_id,
            model_variant,
            backend_kind,
            capabilities,
        })
    }

    pub(crate) fn execution_contracts(
        &self,
        capability: CapabilityKind,
    ) -> Result<Vec<LoadedExecutionContract>> {
        let execution = self.capabilities.get(&capability).ok_or_else(|| {
            Error::InvalidInput(format!(
                "loaded model {} does not expose capability {:?}",
                self.model_variant, capability
            ))
        })?;
        loaded_execution_contracts(execution.as_ref())
    }

    pub(crate) fn seal_fish_capacity(
        &self,
        capacity: crate::runtime::fish_capacity::ResolvedFishServingCapacity,
    ) -> Result<()> {
        for capability in [CapabilityKind::Tts, CapabilityKind::StreamingTts] {
            self.capabilities
                .get(&capability)
                .ok_or_else(|| {
                    Error::ModelLoadError("Fish capability missing during capacity seal".into())
                })?
                .seal_fish_capacity(capacity.clone())?;
        }
        Ok(())
    }

    pub(crate) fn seal_chat_workspace(&self, accelerator_bytes: u64) -> Result<()> {
        let execution = self
            .capabilities
            .get(&CapabilityKind::Chat)
            .ok_or_else(|| {
                Error::ModelLoadError("loaded bundle has no chat capability to seal".into())
            })?;
        execution.seal_chat_workspace(accelerator_bytes)
    }

    pub(crate) fn seal_chat_decode_quantum(&self, tokens: u64) -> Result<()> {
        self.capabilities
            .get(&CapabilityKind::Chat)
            .ok_or_else(|| {
                Error::ModelLoadError("loaded bundle has no chat capability to seal".into())
            })?
            .seal_chat_decode_quantum(tokens)
    }

    pub(crate) fn seal_chat_prefill_workspace(
        &self,
        per_token_bytes: u64,
        max_span_tokens: u64,
    ) -> Result<()> {
        self.capabilities
            .get(&CapabilityKind::Chat)
            .ok_or_else(|| {
                Error::ModelLoadError("loaded bundle has no chat capability to seal".into())
            })?
            .seal_chat_prefill_workspace(per_token_bytes, max_span_tokens)
    }

    pub(crate) fn seal_chat_mixed_step(&self, token_budget: u64) -> Result<()> {
        self.capabilities
            .get(&CapabilityKind::Chat)
            .ok_or_else(|| {
                Error::ModelLoadError("loaded bundle has no chat capability to seal".into())
            })?
            .seal_chat_mixed_step(token_budget)
    }

    pub(crate) fn seal_qwen3_asr_audio_preparation(
        &self,
        model: &crate::models::architectures::qwen3::asr::Qwen3AsrModel,
    ) -> Result<()> {
        let execution = self.capabilities.get(&CapabilityKind::Asr).ok_or_else(|| {
            Error::ModelLoadError("Qwen3 ASR loaded bundle has no ASR capability to seal".into())
        })?;
        execution.seal_qwen3_asr_audio_preparation(model)
    }

    pub(crate) fn seal_whisper_audio_preparation(
        &self,
        model: &crate::models::architectures::whisper::asr::WhisperTurboAsrModel,
    ) -> Result<()> {
        let execution = self.capabilities.get(&CapabilityKind::Asr).ok_or_else(|| {
            Error::ModelLoadError("Whisper loaded bundle has no ASR capability to seal".into())
        })?;
        execution.seal_whisper_audio_preparation(model)
    }

    pub(crate) fn seal_vibevoice_asr_preparation(
        &self,
        model: &crate::models::architectures::vibevoice::asr::VibeVoiceAsrModel,
    ) -> Result<()> {
        let execution = self.capabilities.get(&CapabilityKind::Asr).ok_or_else(|| {
            Error::ModelLoadError("VibeVoice loaded bundle has no ASR capability to seal".into())
        })?;
        execution.seal_vibevoice_asr_preparation(model)
    }

    pub(crate) fn seal_granite_speech_asr_preparation(
        &self,
        model: &crate::models::architectures::granite_speech::asr::GraniteSpeechAsrModel,
    ) -> Result<()> {
        let execution = self.capabilities.get(&CapabilityKind::Asr).ok_or_else(|| {
            Error::ModelLoadError(
                "Granite Speech loaded bundle has no ASR capability to seal".into(),
            )
        })?;
        execution.seal_granite_speech_asr_preparation(model)
    }

    pub(crate) fn seal_lfm25_audio_asr_preparation(
        &self,
        model: &crate::models::registry::NativeAudioChatModel,
    ) -> Result<()> {
        let execution = self.capabilities.get(&CapabilityKind::Asr).ok_or_else(|| {
            Error::ModelLoadError("LFM2.5 Audio loaded bundle has no ASR capability to seal".into())
        })?;
        execution.seal_lfm25_audio_asr_preparation(model)
    }

    pub(crate) fn seal_lfm25_audio_tts_preparation(
        &self,
        model: &crate::models::registry::NativeAudioChatModel,
    ) -> Result<()> {
        let execution = self.capabilities.get(&CapabilityKind::Tts).ok_or_else(|| {
            Error::ModelLoadError("LFM2.5 Audio loaded bundle has no TTS capability to seal".into())
        })?;
        execution.seal_lfm25_audio_tts_preparation(model)
    }

    pub(crate) fn seal_voxtral_realtime_preparation(
        &self,
        model: &crate::models::architectures::voxtral::realtime::VoxtralRealtimeModel,
    ) -> Result<()> {
        let execution = self
            .capabilities
            .get(&CapabilityKind::RealtimeAsr)
            .ok_or_else(|| {
                Error::ModelLoadError(
                    "Voxtral loaded bundle has no realtime ASR capability to seal".into(),
                )
            })?;
        execution.seal_voxtral_realtime_preparation(model)
    }

    pub(crate) fn seal(
        self,
        mut state_publications: HashMap<CapabilityKind, LoadedStatePublication>,
    ) -> Result<LoadedModelBundle> {
        let mut unmatched = state_publications
            .keys()
            .copied()
            .filter(|capability| !self.capabilities.contains_key(capability))
            .map(CapabilityKind::as_str)
            .collect::<Vec<_>>();
        if !unmatched.is_empty() {
            unmatched.sort_unstable();
            return Err(Error::ModelLoadError(format!(
                "loaded model {} published cache truth for unregistered capabilities: {}",
                self.model_variant,
                unmatched.join(", ")
            )));
        }

        let mut capabilities = HashMap::with_capacity(self.capabilities.len());
        for (capability, execution) in self.capabilities {
            let state = state_publications.remove(&capability);
            let descriptor = LoadedCapabilityDescriptor::new(execution, state, self.backend_kind)?;
            capabilities.insert(capability, descriptor);
        }
        debug_assert!(state_publications.is_empty());
        Ok(LoadedModelBundle {
            execution_group_id: self.execution_group_id,
            model_instance_id: self.model_instance_id,
            model_variant: self.model_variant,
            backend_kind: self.backend_kind,
            capabilities,
        })
    }
}

impl fmt::Debug for LoadedModelBundle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LoadedModelBundle")
            .field("execution_group_id", &self.execution_group_id)
            .field("model_instance_id", &self.model_instance_id)
            .field("model_variant", &self.model_variant)
            .field("backend_kind", &self.backend_kind)
            .field("capability_count", &self.capabilities.len())
            .finish()
    }
}

impl LoadedModelBundle {
    #[cfg(test)]
    pub(crate) fn bind(
        registry: &RuntimeAdapterRegistry,
        execution_group_id: ExecutionGroupId,
        model_instance_id: ModelInstanceId,
        model_variant: ModelVariant,
        backend_kind: BackendKind,
    ) -> Result<Self> {
        Self::bind_with_state_publications(
            registry,
            execution_group_id,
            model_instance_id,
            model_variant,
            backend_kind,
            HashMap::new(),
        )
    }

    /// Bind adapter metadata and exact loaded-model state truth into one sealed
    /// descriptor per capability. Only capabilities explicitly classified as
    /// stateless may omit a physical state publication.
    #[cfg(test)]
    pub(crate) fn bind_with_state_publications(
        registry: &RuntimeAdapterRegistry,
        execution_group_id: ExecutionGroupId,
        model_instance_id: ModelInstanceId,
        model_variant: ModelVariant,
        backend_kind: BackendKind,
        state_publications: HashMap<CapabilityKind, LoadedStatePublication>,
    ) -> Result<Self> {
        LoadedModelBundleDraft::build(
            registry,
            execution_group_id,
            model_instance_id,
            model_variant,
            backend_kind,
        )?
        .seal(state_publications)
    }

    pub(crate) fn execution_group_id(&self) -> ExecutionGroupId {
        self.execution_group_id
    }

    pub(crate) fn model_instance_id(&self) -> ModelInstanceId {
        self.model_instance_id
    }

    pub(crate) fn model_variant(&self) -> ModelVariant {
        self.model_variant
    }

    pub(crate) fn backend_kind(&self) -> BackendKind {
        self.backend_kind
    }

    #[cfg(test)]
    pub(crate) fn adapter_count(&self) -> usize {
        self.capabilities.len()
    }

    pub(super) fn require_capability(
        &self,
        capability: CapabilityKind,
    ) -> Result<&LoadedCapabilityDescriptor> {
        self.capabilities.get(&capability).ok_or_else(|| {
            Error::InvalidInput(format!(
                "loaded model {} does not expose capability {:?}",
                self.model_variant, capability
            ))
        })
    }

    #[cfg(test)]
    pub(crate) fn contract(
        &self,
        capability: CapabilityKind,
        streaming_required: bool,
    ) -> Result<LoadedExecutionContract> {
        self.contract_for_streaming(
            capability,
            StreamingRequirements::native(streaming_required),
        )
    }

    pub(crate) fn contract_for_streaming(
        &self,
        capability: CapabilityKind,
        streaming: StreamingRequirements,
    ) -> Result<LoadedExecutionContract> {
        self.require_capability(capability)?.contract(streaming)
    }

    pub(crate) fn capability_binding_for_streaming(
        &self,
        capability: CapabilityKind,
        streaming: StreamingRequirements,
    ) -> Result<LoadedCapabilityBinding> {
        self.require_capability(capability)?.binding(streaming)
    }
}
