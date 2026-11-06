//! Speaking a reply while its text streams: sentence segments, one segment
//! of TTS lookahead, joins and the spoken ledger, sent through a reply sink.

use super::*;

/// The voice a reply is spoken in: a model's built-in speaker, or a
/// reference clip (base64 audio and its transcript) the model clones.
#[derive(Debug, Clone, Default)]
pub(crate) struct TtsVoice {
    pub(crate) speaker: Option<String>,
    pub(crate) reference_audio: Option<String>,
    pub(crate) reference_text: Option<String>,
}

impl TtsVoice {
    pub(crate) fn speaker(speaker: Option<String>) -> Self {
        Self {
            speaker,
            ..Self::default()
        }
    }
}

/// One reply segment being synthesized: its audio arrives on `chunks` while
/// the synthesis task, holding its own admission permit, runs ahead.
pub(crate) struct SegmentSynthesis {
    pub(crate) text: String,
    pub(crate) chunks: tokio::sync::mpsc::Receiver<izwi_core::AudioChunk>,
    pub(crate) task: AbortOnDropTask<Result<(), String>>,
}

impl SegmentSynthesis {
    pub(crate) fn start(
        state: &AppState,
        correlation_id: &str,
        variant: izwi_core::ModelVariant,
        voice: TtsVoice,
        model_leases: &SessionModelLeases,
        text: String,
    ) -> Self {
        let TtsVoice {
            speaker,
            reference_audio,
            reference_text,
        } = voice;
        let mut request = GenerationRequest {
            id: uuid::Uuid::new_v4().to_string(),
            model_variant: Some(variant),
            correlation_id: Some(correlation_id.to_string()),
            runtime_context: RuntimeRequestContext::default(),
            text: text.clone(),
            config: GenerationConfig {
                streaming: true,
                options: GenerationParams {
                    max_tokens: 0,
                    speaker: speaker.clone(),
                    voice: speaker,
                    ..Default::default()
                },
            },
            language: None,
            reference_audio,
            reference_text,
            reference_language: None,
            voice_description: None,
            speaker_references: Vec::new(),
            chunk_history: Vec::new(),
        };
        let (chunk_tx, chunks) = tokio::sync::mpsc::channel(32);
        let state = state.clone();
        let model_leases = model_leases.clone();
        let task = AbortOnDropTask::new(tokio::spawn(async move {
            model_leases
                .hold(&state.runtime, variant)
                .await
                .map_err(|err| model_load_error("TTS", err))?;
            let permit = acquire_stage_permit(&state).await?;
            request.runtime_context = permit.runtime_context();
            state
                .runtime
                .generate_streaming(request, chunk_tx)
                .await
                .map_err(|err| format!("TTS failed: {err}"))
        }));
        Self { text, chunks, task }
    }

    pub(crate) async fn finish(self) -> Result<(), String> {
        drop(self.chunks);
        self.task
            .join()
            .await
            .map_err(|err| format!("TTS streaming task failed: {err}"))?
    }
}

/// The next audio chunk of the segment playing, if one is.
pub(crate) async fn next_chunk(
    segment: Option<&mut SegmentSynthesis>,
) -> Option<izwi_core::AudioChunk> {
    match segment {
        Some(segment) => segment.chunks.recv().await,
        None => std::future::pending().await,
    }
}

/// Where a spoken reply's speech and text go: the wire protocol it is sent
/// in. `speak_reply` decides what is said and when; a sink decides how it is
/// encoded and framed (the voice socket's binary frames, OpenAI Realtime's
/// base64 deltas).
pub(crate) trait ReplySink: Send {
    /// The reply's speech starts, at `model_rate`.
    fn speech_started(&mut self, model_rate: u32) -> Result<(), String>;
    /// Speech samples at the announced rate; `is_final` on the reply's last.
    /// Resolves to whether they went out (`false` when the encoder holds
    /// them back for the next samples).
    fn speech(
        &mut self,
        samples: &[f32],
        model_rate: u32,
        is_final: bool,
    ) -> impl std::future::Future<Output = Result<bool, String>> + Send;
    /// The reply's visible text so far (its reasoning left out).
    fn text(&mut self, visible: &str);
    /// The reply's speech has all gone out.
    fn speech_done(&mut self);
}

/// The voice socket's reply sink: `assistant_*` JSON events and binary
/// assistant audio frames in the session's codec, for the legacy and typed
/// protocols alike (the typed protocol maps the JSON events).
pub(crate) struct VoiceSocketReplySink<'a> {
    out_tx: &'a OutboundTx,
    utterance_id: &'a str,
    utterance_seq: u64,
    output_generation: u64,
    format: OutputAudioFormat,
    encoder: Option<AudioOutputEncoder>,
    chunk_seq: u32,
}

impl<'a> VoiceSocketReplySink<'a> {
    pub(crate) fn new(
        out_tx: &'a OutboundTx,
        commit: &'a PendingAudioCommit,
        output_generation: u64,
    ) -> Self {
        Self {
            out_tx,
            utterance_id: &commit.utterance_id,
            utterance_seq: commit.utterance_seq,
            output_generation,
            format: commit.turn_config.output_format(),
            encoder: None,
            chunk_seq: 0,
        }
    }
}

impl ReplySink for VoiceSocketReplySink<'_> {
    fn speech_started(&mut self, model_rate: u32) -> Result<(), String> {
        self.encoder = Some(AudioOutputEncoder::new(self.format, model_rate)?);
        send_turn_json(
            self.out_tx,
            self.output_generation,
            json!({
                "type": "assistant_audio_start",
                "utterance_id": self.utterance_id,
                "utterance_seq": self.utterance_seq,
                "sample_rate": self.format.wire_rate(model_rate),
                "audio_format": self.format.as_str(),
            }),
        );
        Ok(())
    }

    async fn speech(
        &mut self,
        samples: &[f32],
        model_rate: u32,
        is_final: bool,
    ) -> Result<bool, String> {
        let Some(encoder) = self.encoder.as_mut() else {
            return Ok(false);
        };
        let mut encoded = encoder.encode(samples)?;
        if is_final {
            encoded.extend(encoder.finish()?);
        } else if encoded.is_empty() {
            // The resampler holds this audio for the next chunk.
            return Ok(false);
        }
        let frame = encode_assistant_audio_binary_frame(
            self.format,
            self.utterance_seq,
            self.chunk_seq,
            self.format.wire_rate(model_rate),
            is_final,
            &encoded,
        );
        self.chunk_seq = self.chunk_seq.saturating_add(1);
        let duration = Duration::from_secs_f64(samples.len() as f64 / model_rate.max(1) as f64);
        self.out_tx
            .send_audio(
                self.output_generation,
                Message::Binary(frame.into()),
                encoded.len(),
                duration.as_millis() as u64,
            )
            .await
            .map_err(|err| format!("Outbound audio delivery failed: {err:?}"))?;
        Ok(true)
    }

    fn text(&mut self, visible: &str) {
        send_turn_text_snapshot(
            self.out_tx,
            self.output_generation,
            format!("assistant-text-{}", self.utterance_seq),
            json!({
                "type": "assistant_text_snapshot",
                "utterance_id": self.utterance_id,
                "utterance_seq": self.utterance_seq,
                "text": visible,
            }),
        );
    }

    fn speech_done(&mut self) {
        send_turn_json(
            self.out_tx,
            self.output_generation,
            json!({
                "type": "assistant_audio_done",
                "utterance_id": self.utterance_id,
                "utterance_seq": self.utterance_seq,
            }),
        );
    }
}

/// Assistant audio of one turn as it goes out: the rate it is announced
/// at, the sink it goes through, and the turn's latency accounting.
pub(crate) struct ReplyAudio<'a, S: ReplySink> {
    pub(crate) state: &'a AppState,
    pub(crate) control: &'a TurnControl,
    pub(crate) sink: &'a mut S,
    pub(crate) announced_rate: Option<u32>,
    pub(crate) first_text_at: Option<Instant>,
}

impl<S: ReplySink> ReplyAudio<'_, S> {
    /// The rate `chunk_rate` audio goes out at, announcing the first.
    pub(crate) fn rate(&mut self, chunk_rate: u32) -> Result<u32, String> {
        let (sample_rate, first) = realtime_chunk_rate(&mut self.announced_rate, chunk_rate)?;
        if first {
            self.sink.speech_started(sample_rate)?;
        }
        Ok(sample_rate)
    }

    pub(crate) async fn send(&mut self, samples: &[f32], is_final: bool) -> Result<(), String> {
        let Some(sample_rate) = self.announced_rate else {
            return Ok(());
        };
        if samples.is_empty() && !is_final {
            return Ok(());
        }
        if !self.sink.speech(samples, sample_rate, is_final).await? {
            return Ok(());
        }
        let duration = Duration::from_secs_f64(samples.len() as f64 / sample_rate.max(1) as f64);
        if !samples.is_empty() {
            if let Some(first_text_at) = self.first_text_at.take() {
                self.state
                    .runtime
                    .record_voice_latency(VoiceLatency::TtsFirstAudio, first_text_at.elapsed());
            }
            self.control
                .reply()
                .ledger
                .audio_sent(duration.as_secs_f64() * 1000.0);
            self.control.audio_sent(&self.state.runtime, duration);
        }
        Ok(())
    }
}

/// Speak a reply while the text model writes it (SPK-F2): its text, minus
/// any reasoning, is cut into segments as sentences complete, and the next
/// segment is synthesized while the one before it plays. Segments are
/// joined with the long-form pause and fades (SPK-F20), and each spoken
/// segment is entered in the turn's ledger, so an interruption knows what
/// was heard (SPK-F13).
pub(crate) async fn speak_reply<S: ReplySink>(
    state: &AppState,
    sink: &mut S,
    control: &TurnControl,
    mut deltas: tokio::sync::mpsc::UnboundedReceiver<String>,
    text_requested_at: Instant,
    synthesize: impl Fn(String) -> SegmentSynthesis,
) -> Result<(), String> {
    let fallback_sample_rate = state.runtime.sample_rate().await;
    let mut reasoning = ReasoningFilter::default();
    let mut segmenter = SpeechSegmenter::default();
    let mut joiner = SegmentJoiner::default();
    let mut visible = String::new();
    let mut text_done = false;
    let mut spoke = false;
    let mut playing: Option<SegmentSynthesis> = None;
    let mut lookahead: Option<SegmentSynthesis> = None;
    let mut audio = ReplyAudio {
        state,
        control,
        sink,
        announced_rate: None,
        first_text_at: None,
    };

    loop {
        if lookahead.is_none() {
            if let Some(text) = segmenter.next_segment(text_done) {
                if !spoke {
                    spoke = true;
                    state.runtime.record_voice_latency(
                        VoiceLatency::LlmFirstToken,
                        text_requested_at.elapsed(),
                    );
                    audio.first_text_at = Some(Instant::now());
                }
                lookahead = Some(synthesize(text));
            }
        }
        if playing.is_none() {
            if let Some(next) = lookahead.take() {
                joiner.begin_segment();
                control.reply().ledger.begin_segment(&next.text);
                playing = Some(next);
                continue;
            }
            if text_done {
                break;
            }
        }

        tokio::select! {
            biased;
            _ = control.cancelled() => return Err("Turn cancelled".to_string()),
            chunk = next_chunk(playing.as_mut()) => match chunk {
                Some(chunk) => {
                    if chunk.samples.is_empty() {
                        continue;
                    }
                    let sample_rate = audio.rate(chunk.sample_rate_or(fallback_sample_rate))?;
                    let samples = joiner.push(&chunk.samples, sample_rate);
                    audio.send(&samples, false).await?;
                }
                None => {
                    if let Some(finished) = playing.take() {
                        finished.finish().await?;
                    }
                    if let Some(sample_rate) = audio.announced_rate {
                        let tail = joiner.end_segment(sample_rate);
                        audio.send(&tail, false).await?;
                    }
                }
            },
            delta = deltas.recv(), if !text_done => {
                let text = match delta {
                    Some(delta) => reasoning.push(&delta),
                    None => {
                        text_done = true;
                        reasoning.finish()
                    }
                };
                if !text.is_empty() {
                    visible.push_str(&text);
                    segmenter.push(&text);
                    audio.sink.text(visible.trim_start());
                }
            }
        }
    }

    if audio.announced_rate.is_none() {
        if !spoke {
            // Nothing to say.
            return Ok(());
        }
        audio.rate(fallback_sample_rate)?;
    }
    audio.send(&[], true).await?;
    audio.sink.speech_done();
    Ok(())
}

/// The rate a realtime utterance's audio is labelled with: its first chunk's
/// (`true` when this chunk announces it). A later chunk at another rate is
/// refused rather than mislabelled.
pub(crate) fn realtime_chunk_rate(
    announced: &mut Option<u32>,
    chunk_rate: u32,
) -> Result<(u32, bool), String> {
    let chunk_rate = chunk_rate.max(1);
    match *announced {
        None => {
            *announced = Some(chunk_rate);
            Ok((chunk_rate, true))
        }
        Some(rate) if rate == chunk_rate => Ok((rate, false)),
        Some(rate) => Err(format!(
            "TTS audio changed sample rate mid-utterance ({rate} Hz to {chunk_rate} Hz)"
        )),
    }
}
