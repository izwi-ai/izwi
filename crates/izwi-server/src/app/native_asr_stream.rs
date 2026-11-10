//! A native streaming ASR session as the realtime sockets drive it: PCM16
//! in, and every operation under a real-time admission permit. The legacy
//! transcription socket and `/v1/realtime` live transcripts share it.

use izwi_core::{RuntimeAsrRealtimeEvent, RuntimeAsrRealtimeStream, WorkloadClass};

use crate::app::realtime_transport::pcm16_i16_to_f32;
use crate::state::AppState;

/// Open a session on `model_id`, or `None` when the model has no native
/// streaming session. A full session quota is an error of kind
/// `CapacityExhausted`.
pub(crate) async fn start(
    state: &AppState,
    model_id: Option<&str>,
    language: Option<&str>,
    prompt: Option<&str>,
) -> izwi_core::Result<Option<RuntimeAsrRealtimeStream>> {
    let _permit = state.acquire_workload_permit(WorkloadClass::Realtime).await;
    state
        .runtime
        .try_start_asr_realtime_stream(model_id, language, prompt)
        .await
}

/// Feed the session the next audio of its stream.
pub(crate) async fn push(
    state: &AppState,
    stream: &mut RuntimeAsrRealtimeStream,
    samples: &[i16],
    sample_rate: u32,
) -> izwi_core::Result<Vec<RuntimeAsrRealtimeEvent>> {
    let samples = pcm16_i16_to_f32(samples);
    let _permit = state.acquire_workload_permit(WorkloadClass::Realtime).await;
    state
        .runtime
        .push_asr_realtime_samples(stream, &samples, sample_rate)
        .await
}

/// End the session: its last events carry the final transcript.
pub(crate) async fn finish(
    state: &AppState,
    stream: &mut RuntimeAsrRealtimeStream,
) -> izwi_core::Result<Vec<RuntimeAsrRealtimeEvent>> {
    let _permit = state.acquire_workload_permit(WorkloadClass::Realtime).await;
    state.runtime.finish_asr_realtime_stream(stream).await
}
