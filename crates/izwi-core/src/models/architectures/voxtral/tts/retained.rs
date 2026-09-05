//! Rollback-safe retained Voxtral TTS generation state.

use std::sync::Arc;

use candle_core::Tensor;

use crate::backends::BackendKind;
use crate::error::{Error, Result};
use crate::models::shared::attention::physical::{PhysicalPagedKvCache, PhysicalPagedKvCheckpoint};

use super::sampling::VoxtralTtsGenerationParams;

#[derive(Clone)]
pub(crate) struct VoxtralTtsPreparedArtifact {
    pub(crate) prompt_embeddings: Tensor,
    pub(crate) prompt_tokens: usize,
    pub(crate) source_text: Arc<str>,
    pub(crate) voice: Arc<str>,
    pub(crate) retained_resident_bytes: u64,
}

impl std::fmt::Debug for VoxtralTtsPreparedArtifact {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VoxtralTtsPreparedArtifact")
            .field("prompt_tokens", &self.prompt_tokens)
            .field("source_text", &self.source_text)
            .field("voice", &self.voice)
            .field("retained_resident_bytes", &self.retained_resident_bytes)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VoxtralTtsRetainedPhase {
    Prefill,
    Decode,
    Codec,
    Finished,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VoxtralTtsPrefillStep {
    pub(crate) consumed_tokens: usize,
    pub(crate) prefill_cursor: usize,
    pub(crate) prompt_tokens: usize,
    pub(crate) complete: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VoxtralTtsPrefillBatch {
    pub(crate) steps: Vec<VoxtralTtsPrefillStep>,
    /// One entry for every physical LM wave. Unequal prompt spans can shrink a
    /// cohort over time, so a single maximum is not truthful call telemetry.
    pub(crate) lm_launch_widths: Vec<usize>,
    pub(crate) max_lm_launch_width: usize,
    pub(crate) scalar_lm_launches: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VoxtralTtsDecodeStep {
    pub(crate) frame: Option<Vec<u32>>,
    pub(crate) frames_generated: usize,
    pub(crate) finished: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VoxtralTtsDecodeBatch {
    pub(crate) steps: Vec<VoxtralTtsDecodeStep>,
    /// Physical width of the acoustic-transformer launch. A value greater
    /// than one is only reported after a real batched tensor invocation.
    pub(crate) acoustic_launch_width: usize,
    pub(crate) lm_launch_width: usize,
    pub(crate) scalar_lm_launches: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VoxtralTtsStageResourceEnvelope {
    pub(crate) backend: BackendKind,
    pub(crate) work_units: u64,
    pub(crate) materialized_tensor_elements: u64,
    pub(crate) host_workspace_bytes: u64,
    pub(crate) device_workspace_bytes: u64,
    pub(crate) unified_workspace_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VoxtralTtsStageCeiling {
    pub(crate) max_prompt_tokens: usize,
    pub(crate) max_frames: usize,
    pub(crate) hidden_size: usize,
    pub(crate) num_codebooks: usize,
}

#[derive(Clone)]
struct VoxtralTtsHostCheckpoint {
    prefill_cursor: usize,
    lm_position: usize,
    last_hidden: Option<Tensor>,
    frames: Vec<Vec<u32>>,
    phase: VoxtralTtsRetainedPhase,
}

pub(crate) struct VoxtralTtsQuantumCheckpoint {
    cache: PhysicalPagedKvCheckpoint,
    host: VoxtralTtsHostCheckpoint,
}

pub(crate) struct VoxtralTtsCodecCheckpoint {
    host: VoxtralTtsHostCheckpoint,
}

pub(crate) struct VoxtralTtsRetainedState {
    pub(crate) artifact: Arc<VoxtralTtsPreparedArtifact>,
    pub(crate) params: VoxtralTtsGenerationParams,
    pub(crate) prefill_cursor: usize,
    pub(crate) lm_position: usize,
    pub(crate) last_hidden: Option<Tensor>,
    pub(crate) frames: Vec<Vec<u32>>,
    pub(crate) phase: VoxtralTtsRetainedPhase,
}

impl VoxtralTtsRetainedState {
    pub(crate) fn new(
        artifact: Arc<VoxtralTtsPreparedArtifact>,
        params: VoxtralTtsGenerationParams,
        context_limit: usize,
    ) -> Result<Self> {
        let max_frames = params.max_frames.max(1);
        let required = artifact
            .prompt_tokens
            .checked_add(max_frames.saturating_sub(1))
            .ok_or_else(|| Error::InvalidInput("Voxtral TTS retained context overflowed".into()))?;
        if artifact.prompt_tokens == 0 || required > context_limit {
            return Err(Error::InvalidInput(
                "Voxtral TTS retained request exceeds its context".into(),
            ));
        }
        Ok(Self {
            artifact,
            params: VoxtralTtsGenerationParams {
                max_frames,
                ..params
            },
            prefill_cursor: 0,
            lm_position: 0,
            last_hidden: None,
            frames: Vec::new(),
            phase: VoxtralTtsRetainedPhase::Prefill,
        })
    }

    pub(crate) fn begin_quantum(
        &self,
        cache: &PhysicalPagedKvCache,
    ) -> Result<VoxtralTtsQuantumCheckpoint> {
        if cache.context_len() != self.lm_position {
            return Err(Error::InvalidInput(
                "Voxtral TTS retained cache cursor crossed host state".into(),
            ));
        }
        Ok(VoxtralTtsQuantumCheckpoint {
            cache: cache.logical_checkpoint(),
            host: self.host_checkpoint(),
        })
    }

    pub(crate) fn commit_quantum(
        &self,
        cache: &PhysicalPagedKvCache,
        checkpoint: &VoxtralTtsQuantumCheckpoint,
    ) -> Result<()> {
        if cache.context_len() != self.lm_position || self.lm_position < checkpoint.host.lm_position
        {
            return Err(Error::InferenceError(
                "Voxtral TTS retained quantum made invalid cache progress".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn rollback_quantum(
        &mut self,
        cache: &mut PhysicalPagedKvCache,
        checkpoint: &VoxtralTtsQuantumCheckpoint,
    ) -> Result<()> {
        cache.restore_logical_checkpoint(checkpoint.cache.clone())?;
        self.restore_host(&checkpoint.host);
        Ok(())
    }

    pub(crate) fn begin_codec_quantum(&self) -> Result<VoxtralTtsCodecCheckpoint> {
        if self.phase != VoxtralTtsRetainedPhase::Codec || self.frames.is_empty() {
            return Err(Error::InvalidInput(
                "Voxtral TTS codec quantum requires committed acoustic frames".into(),
            ));
        }
        Ok(VoxtralTtsCodecCheckpoint {
            host: self.host_checkpoint(),
        })
    }

    pub(crate) fn commit_codec_quantum(
        &self,
        _checkpoint: &VoxtralTtsCodecCheckpoint,
    ) -> Result<()> {
        if self.phase != VoxtralTtsRetainedPhase::Finished {
            return Err(Error::InferenceError(
                "Voxtral TTS codec quantum did not reach its terminal state".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn rollback_codec_quantum(&mut self, checkpoint: &VoxtralTtsCodecCheckpoint) {
        self.restore_host(&checkpoint.host);
    }

    fn host_checkpoint(&self) -> VoxtralTtsHostCheckpoint {
        VoxtralTtsHostCheckpoint {
            prefill_cursor: self.prefill_cursor,
            lm_position: self.lm_position,
            last_hidden: self.last_hidden.clone(),
            frames: self.frames.clone(),
            phase: self.phase,
        }
    }

    fn restore_host(&mut self, checkpoint: &VoxtralTtsHostCheckpoint) {
        self.prefill_cursor = checkpoint.prefill_cursor;
        self.lm_position = checkpoint.lm_position;
        self.last_hidden = checkpoint.last_hidden.clone();
        self.frames.clone_from(&checkpoint.frames);
        self.phase = checkpoint.phase;
    }

    pub(crate) fn prompt_tokens(&self) -> usize {
        self.artifact.prompt_tokens
    }

    pub(crate) fn prefill_cursor(&self) -> usize {
        self.prefill_cursor
    }

    pub(crate) fn finished(&self) -> bool {
        self.phase == VoxtralTtsRetainedPhase::Finished
    }

    pub(crate) fn codec_ready(&self) -> bool {
        self.phase == VoxtralTtsRetainedPhase::Codec
    }
}

pub(crate) fn voxtral_tts_stage_resource_envelope(
    backend: BackendKind,
    work_units: usize,
    tensor_elements: u64,
    workspace_bytes: u64,
) -> Result<VoxtralTtsStageResourceEnvelope> {
    let work_units = u64::try_from(work_units)
        .map_err(|_| Error::InvalidInput("Voxtral TTS work units exceed u64".into()))?;
    let (host_workspace_bytes, device_workspace_bytes, unified_workspace_bytes) = match backend {
        BackendKind::Cpu => (workspace_bytes, 0, 0),
        BackendKind::Metal => (0, 0, workspace_bytes),
        BackendKind::Cuda => (0, workspace_bytes, 0),
    };
    Ok(VoxtralTtsStageResourceEnvelope {
        backend,
        work_units,
        materialized_tensor_elements: tensor_elements,
        host_workspace_bytes,
        device_workspace_bytes,
        unified_workspace_bytes,
    })
}

pub(crate) fn validate_acoustic_cohort(states: &[&mut VoxtralTtsRetainedState]) -> Result<()> {
    let Some(first) = states.first() else {
        return Err(Error::InvalidInput(
            "Voxtral TTS acoustic cohort cannot be empty".into(),
        ));
    };
    let cfg_alpha = first.params.cfg_alpha.to_bits();
    let decoding_steps = first.params.n_decoding_steps;
    let allow_end_audio = !first.frames.is_empty();
    for (row, state) in states.iter().enumerate() {
        if state.phase != VoxtralTtsRetainedPhase::Decode
            || state.last_hidden.is_none()
            || state.params.cfg_alpha.to_bits() != cfg_alpha
            || state.params.n_decoding_steps != decoding_steps
            || state.frames.is_empty() == allow_end_audio
        {
            return Err(Error::InvalidInput(format!(
                "Voxtral TTS acoustic cohort row {row} is not shape/config compatible"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn stage_workspace_maps_to_exact_backend_domain() {
        let cpu = voxtral_tts_stage_resource_envelope(BackendKind::Cpu, 2, 3, 5).unwrap();
        let metal = voxtral_tts_stage_resource_envelope(BackendKind::Metal, 2, 3, 5).unwrap();
        let cuda = voxtral_tts_stage_resource_envelope(BackendKind::Cuda, 2, 3, 5).unwrap();
        assert_eq!(
            (cpu.host_workspace_bytes, cpu.device_workspace_bytes),
            (5, 0)
        );
        assert_eq!(metal.unified_workspace_bytes, 5);
        assert_eq!(
            (cuda.host_workspace_bytes, cuda.device_workspace_bytes),
            (0, 5)
        );
    }

    #[test]
    fn retained_state_rejects_prompt_and_frame_budget_beyond_context() {
        let artifact = Arc::new(VoxtralTtsPreparedArtifact {
            prompt_embeddings: Tensor::zeros((1, 4, 8), DType::F32, &Device::Cpu).unwrap(),
            prompt_tokens: 4,
            source_text: Arc::from("hello"),
            voice: Arc::from("default"),
            retained_resident_bytes: 128,
        });
        let params = VoxtralTtsGenerationParams {
            max_frames: 4,
            ..Default::default()
        };
        let err = VoxtralTtsRetainedState::new(artifact, params, 6)
            .err()
            .expect("context must reject the retained request");
        assert!(err.to_string().contains("exceeds its context"));
    }

    fn decode_state(cfg_alpha: f32, frames: usize) -> VoxtralTtsRetainedState {
        let artifact = Arc::new(VoxtralTtsPreparedArtifact {
            prompt_embeddings: Tensor::zeros((1, 2, 8), DType::F32, &Device::Cpu).unwrap(),
            prompt_tokens: 2,
            source_text: Arc::from("hello"),
            voice: Arc::from("default"),
            retained_resident_bytes: 64,
        });
        let mut state = VoxtralTtsRetainedState::new(
            artifact,
            VoxtralTtsGenerationParams {
                cfg_alpha,
                max_frames: 8,
                ..Default::default()
            },
            16,
        )
        .unwrap();
        state.phase = VoxtralTtsRetainedPhase::Decode;
        state.last_hidden = Some(Tensor::zeros((1, 8), DType::F32, &Device::Cpu).unwrap());
        state.frames = vec![vec![1, 2]; frames];
        state
    }

    #[test]
    fn acoustic_cohort_requires_matching_shape_and_generation_config() {
        let mut left = decode_state(1.2, 0);
        let mut right = decode_state(1.2, 0);
        validate_acoustic_cohort(&[&mut left, &mut right]).unwrap();

        right.params.cfg_alpha = 1.1;
        assert!(validate_acoustic_cohort(&[&mut left, &mut right]).is_err());
        right.params.cfg_alpha = 1.2;
        right.frames.push(vec![1, 2]);
        assert!(validate_acoustic_cohort(&[&mut left, &mut right]).is_err());
    }

    #[test]
    fn codec_checkpoint_rolls_back_terminal_host_mutation_without_kv() {
        let mut state = decode_state(1.2, 1);
        state.phase = VoxtralTtsRetainedPhase::Codec;
        let checkpoint = state.begin_codec_quantum().unwrap();
        state.phase = VoxtralTtsRetainedPhase::Finished;
        state.frames.clear();
        assert!(state.commit_codec_quantum(&checkpoint).is_ok());

        state.rollback_codec_quantum(&checkpoint);
        assert!(state.codec_ready());
        assert_eq!(state.frames, vec![vec![1, 2]]);
    }
}
