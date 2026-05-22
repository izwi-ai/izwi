//! Streaming audio buffer for Voxtral Realtime.
//!
//! Manages audio buffering with look-ahead and look-back for streaming transcription.

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
    use super::VoxtralRealtimeBuffer;

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

        buffer.write_audio(&vec![1.0; 20]);

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
}
