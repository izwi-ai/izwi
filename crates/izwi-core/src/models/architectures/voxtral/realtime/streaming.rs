//! Streaming audio buffer for Voxtral Realtime.
//!
//! Manages audio buffering with look-ahead and look-back for streaming transcription.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use candle_core::Tensor;

use crate::error::{Error, Result};
use crate::kv::KvArenaId;
use crate::models::shared::attention::physical::PhysicalPagedKvSequenceAuthority;

static NEXT_VOXTRAL_REALTIME_STATE_ID: AtomicU64 = AtomicU64::new(1);

/// Host-owned continuation state for one Voxtral realtime stream.
///
/// Audio preparation deliberately retains the original source samples.  Until
/// the encoder exposes a checkpointed convolution/attention continuation, a
/// push can recompute the causal encoder prefix and replace `audio_embeds`
/// without changing the already-consumed LM/KV cursor.
#[derive(Debug)]
pub(crate) struct VoxtralRealtimeState {
    pub(super) state_id: u64,
    pub(super) next_quantum_nonce: u64,
    pub(super) active_quantum: Option<u64>,
    pub(super) active_cache_arena: Option<KvArenaId>,
    pub(super) active_cache_view_id: Option<u64>,
    /// Monotonic artifact authority. This deliberately is not part of a host
    /// checkpoint: rolling back a cancelled install must not make its artifact
    /// replayable by a later operation.
    pub(super) preparation_generation: u64,
    bound_cache_authority: Option<PhysicalPagedKvSequenceAuthority>,
    pub(super) language: Option<String>,
    pub(super) source_sample_rate: Option<u32>,
    pub(super) source_samples: Arc<Vec<f32>>,
    pub(super) audio_embeds: Option<Tensor>,
    pub(super) prepared_audio_frames: usize,
    pub(super) next_audio_frame: usize,
    pub(super) prompt_initialized: bool,
    pub(super) pending_input_token: Option<u32>,
    pub(super) generated: Vec<u32>,
    pub(super) assembled: String,
    pub(super) input_closed: bool,
    pub(super) final_padding_applied: bool,
    pub(super) finished: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VoxtralRealtimeResourceUsage {
    pub(crate) host_bytes: u64,
    pub(crate) tensor_bytes: u64,
}

#[derive(Debug, Clone)]
pub(super) struct VoxtralRealtimeHostCheckpoint {
    source_sample_rate: Option<u32>,
    source_samples: Arc<Vec<f32>>,
    audio_embeds: Option<Tensor>,
    prepared_audio_frames: usize,
    next_audio_frame: usize,
    prompt_initialized: bool,
    pending_input_token: Option<u32>,
    generated_len: usize,
    assembled: String,
    input_closed: bool,
    final_padding_applied: bool,
    finished: bool,
}

impl VoxtralRealtimeState {
    pub(crate) fn new(language: Option<&str>) -> Self {
        Self {
            state_id: NEXT_VOXTRAL_REALTIME_STATE_ID.fetch_add(1, Ordering::Relaxed),
            next_quantum_nonce: 1,
            active_quantum: None,
            active_cache_arena: None,
            active_cache_view_id: None,
            preparation_generation: 1,
            bound_cache_authority: None,
            language: language.map(ToOwned::to_owned),
            source_sample_rate: None,
            source_samples: Arc::new(Vec::new()),
            audio_embeds: None,
            prepared_audio_frames: 0,
            next_audio_frame: 0,
            prompt_initialized: false,
            pending_input_token: None,
            generated: Vec::new(),
            assembled: String::new(),
            input_closed: false,
            final_padding_applied: false,
            finished: false,
        }
    }

    pub(crate) fn text(&self) -> &str {
        self.assembled.trim()
    }

    pub(crate) fn is_finished(&self) -> bool {
        self.finished
    }

    pub(crate) fn input_closed(&self) -> bool {
        self.input_closed
    }

    pub(crate) fn prepared_audio_frames(&self) -> usize {
        self.prepared_audio_frames
    }

    pub(crate) fn next_audio_frame(&self) -> usize {
        self.next_audio_frame
    }

    pub(crate) fn source_sample_count(&self) -> usize {
        self.source_samples.len()
    }

    pub(crate) fn language(&self) -> Option<&str> {
        self.language.as_deref()
    }

    pub(crate) fn tokens_generated(&self) -> usize {
        self.generated.len()
    }

    pub(crate) fn resource_usage(&self) -> Result<VoxtralRealtimeResourceUsage> {
        let host_bytes = self
            .source_samples
            .capacity()
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|bytes| {
                self.generated
                    .capacity()
                    .checked_mul(std::mem::size_of::<u32>())
                    .and_then(|tokens| bytes.checked_add(tokens))
            })
            .and_then(|bytes| bytes.checked_add(self.assembled.capacity()))
            .and_then(|bytes| bytes.checked_add(self.language.as_ref().map_or(0, String::capacity)))
            .ok_or_else(|| Error::InferenceError("Voxtral host usage overflow".into()))?;
        let tensor_bytes = self.audio_embeds.as_ref().map_or(Ok(0usize), |tensor| {
            tensor
                .elem_count()
                .checked_mul(tensor.dtype().size_in_bytes())
                .ok_or_else(|| Error::InferenceError("Voxtral tensor usage overflow".into()))
        })?;
        Ok(VoxtralRealtimeResourceUsage {
            host_bytes: u64::try_from(host_bytes)
                .map_err(|_| Error::InferenceError("Voxtral host usage exceeds u64".into()))?,
            tensor_bytes: u64::try_from(tensor_bytes)
                .map_err(|_| Error::InferenceError("Voxtral tensor usage exceeds u64".into()))?,
        })
    }

    pub(super) fn append_source_samples(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<()> {
        if self.input_closed || self.finished {
            return Err(Error::InvalidInput(
                "Voxtral realtime input is already closed or finished".into(),
            ));
        }
        if sample_rate == 0 {
            return Err(Error::InvalidInput(
                "Voxtral realtime input sample rate must be non-zero".into(),
            ));
        }
        if self
            .source_sample_rate
            .is_some_and(|current| current != sample_rate)
        {
            return Err(Error::InvalidInput(
                "Voxtral realtime input sample rate changed within one stream".into(),
            ));
        }
        let _ = self
            .source_samples
            .len()
            .checked_add(samples.len())
            .ok_or_else(|| Error::InvalidInput("Voxtral realtime audio length overflow".into()))?;
        self.source_sample_rate.get_or_insert(sample_rate);
        Arc::make_mut(&mut self.source_samples).extend_from_slice(samples);
        Ok(())
    }

    pub(super) fn bind_cache_authority(
        &mut self,
        authority: PhysicalPagedKvSequenceAuthority,
    ) -> Result<()> {
        match self.bound_cache_authority {
            Some(bound) if bound != authority => Err(Error::InferenceError(
                "Voxtral realtime state belongs to another retained cache sequence".into(),
            )),
            Some(_) => Ok(()),
            None => {
                self.bound_cache_authority = Some(authority);
                Ok(())
            }
        }
    }

    pub(super) fn active_cache_matches(&self, arena: KvArenaId, view_id: u64) -> bool {
        self.active_quantum.is_some()
            && self.active_cache_arena == Some(arena)
            && self.active_cache_view_id == Some(view_id)
    }

    pub(super) fn checkpoint(&self) -> VoxtralRealtimeHostCheckpoint {
        VoxtralRealtimeHostCheckpoint {
            source_sample_rate: self.source_sample_rate,
            source_samples: self.source_samples.clone(),
            audio_embeds: self.audio_embeds.clone(),
            prepared_audio_frames: self.prepared_audio_frames,
            next_audio_frame: self.next_audio_frame,
            prompt_initialized: self.prompt_initialized,
            pending_input_token: self.pending_input_token,
            generated_len: self.generated.len(),
            assembled: self.assembled.clone(),
            input_closed: self.input_closed,
            final_padding_applied: self.final_padding_applied,
            finished: self.finished,
        }
    }

    pub(super) fn restore_checkpoint(&mut self, checkpoint: VoxtralRealtimeHostCheckpoint) {
        self.source_sample_rate = checkpoint.source_sample_rate;
        self.source_samples = checkpoint.source_samples;
        self.audio_embeds = checkpoint.audio_embeds;
        self.prepared_audio_frames = checkpoint.prepared_audio_frames;
        self.next_audio_frame = checkpoint.next_audio_frame;
        self.prompt_initialized = checkpoint.prompt_initialized;
        self.pending_input_token = checkpoint.pending_input_token;
        self.generated.truncate(checkpoint.generated_len);
        self.assembled = checkpoint.assembled;
        self.input_closed = checkpoint.input_closed;
        self.final_padding_applied = checkpoint.final_padding_applied;
        self.finished = checkpoint.finished;
    }
}

/// Buffer for realtime streaming audio
pub struct VoxtralRealtimeBuffer {
    sampling_rate: usize,
    look_ahead: usize,
    look_back: usize,
    streaming_size: usize,
    start: usize,
    end: usize,
    buffer: Vec<f32>,
    filled_len: usize,
    pre_allocate_size: usize,
}

impl VoxtralRealtimeBuffer {
    /// Create new buffer with audio config
    pub fn new(
        sampling_rate: usize,
        streaming_look_ahead_ms: f32,
        streaming_look_back_ms: f32,
        transcription_delay_ms: f32,
        frame_rate: f32,
    ) -> Self {
        let look_ahead = ((sampling_rate as f32 * streaming_look_ahead_ms) / 1000.0) as usize;
        let look_back = ((sampling_rate as f32 * streaming_look_back_ms) / 1000.0) as usize;
        let streaming_size = if frame_rate > 0.0 {
            ((sampling_rate as f32 * 1000.0) / (frame_rate * 1000.0)) as usize
        } else {
            0
        }
        .max(1);
        let streaming_delay = ((sampling_rate as f32 * transcription_delay_ms) / 1000.0) as usize;

        let pre_allocate_size = 30 * sampling_rate; // 30 seconds
        let buffer = vec![0.0f32; pre_allocate_size];

        Self {
            sampling_rate,
            look_ahead,
            look_back,
            streaming_size,
            start: 0,
            end: streaming_delay + streaming_size,
            buffer,
            filled_len: 0,
            pre_allocate_size,
        }
    }

    fn get_len_in_samples(&self, len_in_ms: f32) -> usize {
        ((self.sampling_rate as f32 * len_in_ms) / 1000.0) as usize
    }

    /// Start index including look-back
    pub fn start_idx(&self) -> usize {
        self.start.saturating_sub(self.look_back)
    }

    /// End index including look-ahead
    pub fn end_idx(&self) -> usize {
        self.end.saturating_add(self.look_ahead)
    }

    /// Check if enough audio is available for processing
    pub fn is_audio_complete(&self) -> bool {
        self.filled_len >= self.end_idx()
    }

    /// Write audio chunk to buffer
    pub fn write_audio(&mut self, audio: &[f32]) {
        if audio.is_empty() {
            return;
        }

        self.ensure_capacity_for(audio.len());
        self.buffer[self.filled_len..self.filled_len + audio.len()].copy_from_slice(audio);
        self.filled_len += audio.len();
    }

    fn ensure_capacity_for(&mut self, incoming_len: usize) {
        if self.filled_len.saturating_add(incoming_len) <= self.buffer.len() {
            return;
        }

        self.allocate_new_buffer();
        let required_len = self.filled_len.saturating_add(incoming_len);
        if required_len <= self.buffer.len() {
            return;
        }

        let mut new_len = self.buffer.len().max(1);
        while new_len < required_len {
            new_len = new_len.saturating_mul(2);
            if new_len == usize::MAX {
                break;
            }
        }
        self.buffer.resize(new_len, 0.0);
        self.pre_allocate_size = new_len;
    }

    fn allocate_new_buffer(&mut self) {
        let old_start_idx = self.start_idx();
        let retained_start = self.start.saturating_sub(old_start_idx);
        let retained_end = self.end.saturating_sub(old_start_idx);
        let mut new_buffer = vec![0.0f32; self.pre_allocate_size];
        let left_to_copy = self.filled_len.saturating_sub(old_start_idx);

        if left_to_copy > 0 {
            new_buffer[..left_to_copy]
                .copy_from_slice(&self.buffer[old_start_idx..self.filled_len]);
        }

        self.buffer = new_buffer;
        self.filled_len = left_to_copy;
        self.start = retained_start.min(self.filled_len);
        self.end = retained_end.max(self.start);
    }

    /// Read audio chunk for processing (with look-ahead/look-back)
    pub fn read_audio(&mut self) -> Option<Vec<f32>> {
        if !self.is_audio_complete() {
            return None;
        }

        let audio = self.buffer[self.start_idx()..self.end_idx()].to_vec();
        self.start = self.end;
        self.end += self.streaming_size;

        Some(audio)
    }
}

#[cfg(test)]
mod tests {
    use super::{VoxtralRealtimeBuffer, VoxtralRealtimeState};
    use crate::backends::BackendKind;
    use crate::engine::ModelInstanceId;
    use crate::kv::KvArenaId;

    fn tiny_buffer() -> VoxtralRealtimeBuffer {
        let mut buffer = VoxtralRealtimeBuffer::new(10, 100.0, 200.0, 0.0, 10.0);
        buffer.pre_allocate_size = 8;
        buffer.buffer = vec![0.0; buffer.pre_allocate_size];
        buffer
    }

    #[test]
    fn realtime_buffer_reallocation_preserves_window_offsets() {
        let mut buffer = tiny_buffer();
        buffer.buffer = (0..8).map(|value| value as f32).collect();
        buffer.filled_len = 8;
        buffer.start = 5;
        buffer.end = 6;

        buffer.allocate_new_buffer();

        assert_eq!(buffer.filled_len, 5);
        assert_eq!(buffer.start, 2);
        assert_eq!(buffer.end, 3);
        assert_eq!(
            &buffer.buffer[..buffer.filled_len],
            &[3.0, 4.0, 5.0, 6.0, 7.0]
        );
    }

    #[test]
    fn realtime_buffer_grows_for_large_audio_chunk() {
        let mut buffer = tiny_buffer();

        buffer.write_audio(&[1.0; 20]);

        assert_eq!(buffer.filled_len, 20);
        assert!(buffer.buffer.len() >= 20);
        assert!(buffer.buffer[..20].iter().all(|sample| *sample == 1.0));
    }

    #[test]
    fn realtime_buffer_uses_nonzero_streaming_size_when_frame_rate_is_zero() {
        let buffer = VoxtralRealtimeBuffer::new(16_000, 0.0, 0.0, 0.0, 0.0);

        assert_eq!(buffer.streaming_size, 1);
        assert_eq!(buffer.end, 1);
    }

    #[test]
    fn retained_state_rejects_sample_rate_changes_without_consuming_audio() {
        let mut state = VoxtralRealtimeState::new(Some("en"));
        state.append_source_samples(&[0.1, 0.2], 16_000).unwrap();
        let checkpoint = state.checkpoint();

        let error = state
            .append_source_samples(&[0.3], 48_000)
            .expect_err("one stream must have one source rate");

        assert!(format!("{error}").contains("sample rate changed"));
        assert_eq!(state.source_samples, checkpoint.source_samples);
        assert_eq!(state.source_sample_rate, checkpoint.source_sample_rate);
    }

    #[test]
    fn retained_state_checkpoint_restores_finish_and_cursors() {
        let mut state = VoxtralRealtimeState::new(None);
        state.append_source_samples(&[0.1, 0.2], 16_000).unwrap();
        let checkpoint = state.checkpoint();
        state.append_source_samples(&[0.3], 16_000).unwrap();
        state.input_closed = true;
        state.final_padding_applied = true;
        state.next_audio_frame = 9;

        state.restore_checkpoint(checkpoint);

        assert!(!state.input_closed);
        assert!(!state.final_padding_applied);
        assert_eq!(state.next_audio_frame, 0);
        assert_eq!(state.source_samples.as_ref(), &vec![0.1, 0.2]);
    }

    #[test]
    fn retained_state_rollback_does_not_rearm_consumed_preparation_generation() {
        let mut state = VoxtralRealtimeState::new(None);
        state.append_source_samples(&[0.1, 0.2], 16_000).unwrap();
        let checkpoint = state.checkpoint();
        let prior_source = state.source_samples.clone();
        let consumed_generation = state.preparation_generation;

        state.preparation_generation += 1;
        state.source_samples = std::sync::Arc::new(vec![0.3, 0.4]);
        state.restore_checkpoint(checkpoint);

        assert!(std::sync::Arc::ptr_eq(&prior_source, &state.source_samples));
        assert_ne!(state.preparation_generation, consumed_generation);
    }

    #[test]
    fn retained_state_active_quantum_authenticates_exact_cache_view() {
        let mut state = VoxtralRealtimeState::new(None);
        let arena = KvArenaId {
            model_instance: ModelInstanceId::new(7),
            backend: BackendKind::Cpu,
            device_ordinal: None,
            generation: 3,
        };
        state.active_quantum = Some(1);
        state.active_cache_arena = Some(arena);
        state.active_cache_view_id = Some(11);

        assert!(state.active_cache_matches(arena, 11));
        assert!(!state.active_cache_matches(arena, 12));
        state.active_quantum = None;
        assert!(!state.active_cache_matches(arena, 11));
    }
}
