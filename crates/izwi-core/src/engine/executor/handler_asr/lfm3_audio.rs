//! LFM3-Audio long-form ASR: the planned chunks, one per dispatch, each a
//! clip of at most 30 s on the main backbone's invocation workspace.

use super::*;

impl NativeExecutor {
    /// Run audio that LFM3-Audio preparation planned as several chunks,
    /// resuming the run the committed chunks left.
    pub(super) fn lfm3_audio_asr_long_form_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
    ) -> Result<ModelSessionResult> {
        let model = request
            .prepared_lfm3_audio_model_lease_for_executor()?
            .ok_or_else(|| {
                Error::InferenceError("LFM3-Audio long-form ASR lost model residency".into())
            })?;
        let session = scheduled.session_key();
        let resumed = self.long_form_asr.committed(&session);
        let (execution_audio, sample_rate, audio_decode_ms) = match resumed.as_ref() {
            Some(resumed) => (
                AsrExecutionAudio::Prepared(resumed.samples.clone()),
                resumed.sample_rate,
                resumed.audio_decode_ms,
            ),
            None => resolve_asr_execution_audio(request, ModelFamily::Lfm3Audio, || {
                Self::run_blocking(|| decode_request_audio_with_rate(request))
            })?,
        };
        let samples = execution_audio.samples();
        let language = request.lfm3_audio_asr_language();
        let token_limit = request.lfm3_audio_asr_token_limit();
        let stream_tx = Self::stream_sender(request);
        let stream_policy = request.stream_policy;
        let mut leases =
            super::super::invocation_workspace_leases_for_atomic_scalar_row(request, scheduled)?;
        let chunk_plan = asr_execution_route_plan(request, samples, sample_rate, || {
            Ok(model.asr_route_capabilities())
        })?;
        if chunk_plan.chunks.is_empty() {
            return Err(Error::InferenceError(
                "LFM3-Audio long-form ASR planned no chunks".into(),
            ));
        }
        let transcription = Self::run_blocking(|| {
            let transcribe_chunk =
                |chunk_audio: &[f32],
                 sr: u32,
                 _prefix_text: &str,
                 partial: Option<&mut dyn FnMut(&str)>| {
                    let mut no_partials = |_: &str| {};
                    let emit: &mut dyn FnMut(&str) = match partial {
                        Some(emit) => emit,
                        None => &mut no_partials,
                    };
                    let output = model.lfm3_audio_transcribe_clip_from_invocation_workspace(
                        chunk_audio,
                        sr,
                        language,
                        token_limit,
                        &mut leases,
                        &mut || request.is_cancelled(),
                        emit,
                    )?;
                    Ok(AsrChunkTranscription {
                        text: output.text,
                        diagnostics: output.diagnostics,
                    })
                };
            Self::run_long_form_chunk(
                request,
                resumed,
                &execution_audio,
                sample_rate,
                audio_decode_ms,
                stream_tx.as_ref(),
                stream_policy,
                &chunk_plan,
                transcribe_chunk,
            )
        });
        let transcription = leases.fence_on_error(transcription);
        let Some(transcription) = settle_atomic_asr_cancellation(request, transcription)? else {
            return Ok(ModelSessionResult::cancelled(ExecutorOutput::cancelled(
                request.id.clone(),
            )));
        };
        let completions = leases.release()?;
        if completions.len() != 2 {
            return Err(Error::InferenceError(
                "LFM3-Audio long-form ASR returned an incomplete physical-state completion set"
                    .into(),
            ));
        }
        let (text, diagnostics) = match transcription {
            AsrTranscriptionStep::Complete(text, diagnostics) => (text, diagnostics),
            AsrTranscriptionStep::Yielded(state) => {
                return self.yield_long_form_chunk(request, scheduled, session, *state);
            }
        };
        let samples_len = samples.len();

        Ok(ModelSessionResult::atomic(ExecutorOutput {
            request_id: request.id.clone(),
            audio: Some(AudioOutput {
                streamed_samples: None,
                samples: Vec::new(),
                sample_rate,
                duration_secs: samples_len as f32 / sample_rate as f32,
            }),
            text: Some(text),
            input_transcription: None,
            tokens_processed: request.num_prompt_tokens(),
            tokens_generated: (samples_len / 256).max(1),
            finished: true,
            phase_timing_override: Some(ExecutorPhaseTiming::with_media_decode_ms(audio_decode_ms)),
            asr_diagnostics: Self::with_audio_decode_timing(diagnostics, audio_decode_ms),
            error: None,
            logprobs: Vec::new(),
        }))
    }
}
