//! Kokoro TTS runtime helpers (isolated from generic runtime routing).

use tokio::sync::mpsc;
use tracing::info;

use crate::backends::BackendKind;
use crate::engine::WorkUnit;
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::architectures::kokoro::admission::{
    kokoro_chunk_frame_bound, kokoro_chunk_price,
};
use crate::models::architectures::kokoro::planning::{KokoroChunkPolicy, KokoroPlannedChunk};
use crate::models::architectures::kokoro::{
    kokoro_output_budget, KokoroPreparedRequest, KokoroSynthesisResult, KokoroTtsModel,
};
use crate::runtime::adapters::{CapabilityKind, ExecutionTargetKind};
use crate::runtime::coordinator::{JobLease, JobResourceObservation};
use crate::runtime::service::RuntimeService;
use crate::runtime::tts::direct_tts_retained_input_bytes;
use crate::runtime::types::{AudioChunk, GenerationRequest};

pub(crate) const KOKORO_STREAM_TARGET_CHARS: usize = 180;

/// Planned frames of the heaviest chunk the stream planner can produce for
/// `text`: its chunks never exceed `KOKORO_STREAM_TARGET_CHARS`.
pub(crate) fn kokoro_stream_chunk_frames(text: &str, speed: f32) -> usize {
    kokoro_chunk_frame_bound(text, KOKORO_STREAM_TARGET_CHARS, speed)
}

/// The audio one streamed chunk of `text` may produce. Admission reserved
/// one chunk of output at the planning pace for the largest chunk the
/// planner can produce (KOK-8); each chunk's audio is handed on before the
/// next is synthesized.
fn kokoro_stream_chunk_sample_limit(text: &str, speed: f32, backend: BackendKind) -> Result<usize> {
    let samples =
        kokoro_chunk_price(kokoro_stream_chunk_frames(text, speed), backend)?.output_samples;
    usize::try_from(samples)
        .map_err(|_| Error::Overloaded("Kokoro chunk sample limit exceeds usize".into()))
}

fn kokoro_streaming_capability() -> CapabilityKind {
    CapabilityKind::StreamingTts
}

impl RuntimeService {
    fn default_kokoro_variant() -> ModelVariant {
        ModelVariant::Kokoro82M
    }

    async fn resolve_kokoro_variant_for_request(
        &self,
        request: &GenerationRequest,
    ) -> ModelVariant {
        if let Some(variant) = request.model_variant {
            if matches!(variant.family(), crate::catalog::ModelFamily::KokoroTts) {
                return variant;
            }
        }
        if let Some(variant) = *self.loaded_tts_variant.read().await {
            if matches!(variant.family(), crate::catalog::ModelFamily::KokoroTts) {
                return variant;
            }
        }
        Self::default_kokoro_variant()
    }
    pub(crate) async fn kokoro_tts_generate_streaming(
        &self,
        job: &JobLease,
        request: GenerationRequest,
        chunk_tx: mpsc::Sender<AudioChunk>,
    ) -> Result<()> {
        let retained_input_bytes = u64::try_from(direct_tts_retained_input_bytes(&request)?)
            .map_err(|_| Error::InvalidInput("Kokoro retained input exceeds u64".to_string()))?;
        let output_budget = kokoro_output_budget(&request.text, request.config.options.speed)?;
        let request_id = request.id.clone();
        let variant = self.resolve_kokoro_variant_for_request(&request).await;
        let capability = kokoro_streaming_capability();
        self.observe_broker_capability_request(capability, Some(variant), true)?;
        let (residency_lease, execution_contract) = self
            .load_capability_for_job(
                job,
                variant,
                capability,
                true,
                ExecutionTargetKind::DirectModel,
            )
            .await?;
        let model = self
            .model_registry
            .get_kokoro(variant)
            .await
            .ok_or_else(|| Error::InferenceError("Kokoro model not loaded".to_string()))?;
        let text = request.text;
        let text_for_limit = text.clone();
        let speaker = request
            .config
            .options
            .speaker
            .or(request.config.options.voice);
        let language = request.language;
        let speed = request.config.options.speed;
        let model_for_plan = model.clone();
        let speaker_for_plan = speaker.clone();
        let language_for_plan = language.clone();

        let (mut residency_lease, stream_chunks) = self
            .coordinator
            .run_loaded_blocking_stage(
                job,
                execution_contract.clone(),
                WorkUnit::AtomicJob {
                    kind: capability.as_str().to_string(),
                },
                move || {
                    // One G2P pass over the whole text; a short first chunk
                    // for time to first audio, then chunks of up to
                    // KOKORO_STREAM_TARGET_CHARS (KOK-11/KOK-12).
                    let stream_chunks = model_for_plan.plan(
                        &text,
                        speaker_for_plan.as_deref(),
                        language_for_plan.as_deref(),
                        KokoroChunkPolicy::streaming(KOKORO_STREAM_TARGET_CHARS),
                    )?;
                    if stream_chunks.is_empty() {
                        return Err(Error::InvalidInput(
                            "Kokoro streaming planner produced no chunks".to_string(),
                        ));
                    }
                    Ok((residency_lease, stream_chunks))
                },
            )
            .await?;
        let total_chunks = stream_chunks.len();
        if total_chunks > output_budget.max_chunks {
            return Err(Error::InferenceError(format!(
                "Kokoro streaming planner exceeded its hard chunk contract: {total_chunks} > {}",
                output_budget.max_chunks
            )));
        }
        let mut expected_sample_rate: Option<u32> = None;
        let mut emitted_samples = 0usize;
        let mut materialized_output_samples = 0usize;
        let chunk_sample_limit = kokoro_stream_chunk_sample_limit(
            &text_for_limit,
            speed,
            self.backend_router.context().backend_kind,
        )?;

        for (sequence, planned) in stream_chunks.into_iter().enumerate() {
            let model_for_task = model.clone();
            let speaker_for_task = speaker.clone();
            let language_for_task = language.clone();
            let (returned_lease, synthesis) = self
                .coordinator
                .run_loaded_blocking_stage(
                    job,
                    execution_contract.clone(),
                    WorkUnit::AtomicJob {
                        kind: capability.as_str().to_string(),
                    },
                    move || {
                        let synthesis = synthesize_planned_kokoro(
                            model_for_task.as_ref(),
                            &planned,
                            speaker_for_task.as_deref(),
                            language_for_task.as_deref(),
                            speed,
                            chunk_sample_limit,
                        )?;
                        Ok((residency_lease, synthesis))
                    },
                )
                .await?;
            residency_lease = returned_lease;
            let current_sample_rate = synthesis.sample_rate;
            match expected_sample_rate {
                Some(expected) if expected != current_sample_rate => {
                    return Err(Error::InferenceError(format!(
                        "Kokoro streaming sample rate mismatch: expected {}, got {}",
                        expected, current_sample_rate
                    )));
                }
                None => expected_sample_rate = Some(current_sample_rate),
                _ => {}
            }

            emitted_samples = checked_kokoro_sample_growth(
                emitted_samples,
                synthesis.samples.len(),
                output_budget.max_samples,
                "streamed samples",
            )?;
            materialized_output_samples = record_kokoro_materialized_output(
                job,
                retained_input_bytes,
                materialized_output_samples,
                synthesis.samples.capacity(),
                chunk_sample_limit,
            )?;

            let mut chunk = AudioChunk::new(request_id.clone(), sequence, synthesis.samples)
                .with_sample_rate(current_sample_rate);
            chunk.is_final = sequence + 1 == total_chunks;
            match job.spec.deadline {
                Some(deadline) => tokio::time::timeout_at(deadline.into(), chunk_tx.send(chunk))
                    .await
                    .map_err(|_| Error::Timeout(job.spec.request_id.clone()))?
                    .map_err(|_| {
                        Error::InferenceError("Streaming output channel closed".to_string())
                    })?,
                None => chunk_tx.send(chunk).await.map_err(|_| {
                    Error::InferenceError("Streaming output channel closed".to_string())
                })?,
            }
        }
        Ok(())
    }
}

/// Synthesize one planned chunk from its phonemes (no G2P).
fn synthesize_planned_kokoro(
    model: &KokoroTtsModel,
    chunk: &KokoroPlannedChunk,
    speaker: Option<&str>,
    language: Option<&str>,
    speed: f32,
    max_samples: usize,
) -> Result<KokoroSynthesisResult> {
    let prepared = model.prepare_planned(chunk, speaker, language, speed)?;
    let result = model
        .generate_prepared_batch(&[prepared])?
        .pop()
        .ok_or_else(|| {
            Error::InferenceError("Kokoro returned an empty scalar synthesis batch".to_string())
        })?;
    validate_kokoro_samples(&result.samples, max_samples, "synthesis")?;
    Ok(result)
}

/// Plans and prepares text against the voice pack: the loaded model, or a
/// fake in tests of the engine preparation.
pub(crate) trait KokoroPreparer {
    /// The text's chunks, phonemized once (see `KokoroTtsModel::plan_chunks`).
    fn plan(
        &self,
        text: &str,
        speaker: Option<&str>,
        language: Option<&str>,
        policy: KokoroChunkPolicy,
    ) -> Result<Vec<KokoroPlannedChunk>>;

    /// One planned chunk's model inputs, from its phonemes.
    fn prepare_planned(
        &self,
        chunk: &KokoroPlannedChunk,
        speaker: Option<&str>,
        language: Option<&str>,
        speed: f32,
    ) -> Result<KokoroPreparedRequest>;
}

impl KokoroPreparer for KokoroTtsModel {
    fn plan(
        &self,
        text: &str,
        speaker: Option<&str>,
        language: Option<&str>,
        policy: KokoroChunkPolicy,
    ) -> Result<Vec<KokoroPlannedChunk>> {
        self.plan_chunks(text, speaker, language, policy)
    }

    fn prepare_planned(
        &self,
        chunk: &KokoroPlannedChunk,
        speaker: Option<&str>,
        language: Option<&str>,
        speed: f32,
    ) -> Result<KokoroPreparedRequest> {
        KokoroTtsModel::prepare_planned(self, chunk, speaker, language, speed)
    }
}

/// Prepare a Kokoro request for engine synthesis: the whole text when it
/// fits the voice pack's 510-phoneme limit, otherwise the chunks the
/// planner splits it into (the streaming route's planner), each prepared.
/// A `seed` seeds each chunk's decoder noise (offset by its position), so
/// the request repeats exactly; without one every synthesis draws fresh noise.
pub(crate) fn prepare_kokoro_text(
    model: &impl KokoroPreparer,
    text: &str,
    speaker: Option<&str>,
    language: Option<&str>,
    speed: f32,
    seed: Option<u64>,
) -> Result<crate::engine::KokoroPreparedText> {
    // One G2P pass; one chunk when the text fits the voice pack (KOK-11).
    let planned = model.plan(text, speaker, language, KokoroChunkPolicy::WHOLE)?;
    if planned.len() > 1 {
        info!(
            chunks = planned.len(),
            "Kokoro text exceeds the voice-pack limit; synthesizing planned chunks"
        );
    }
    let chunks = planned
        .iter()
        .zip(0u64..)
        .map(|(chunk, position)| {
            let mut prepared = model.prepare_planned(chunk, speaker, language, speed)?;
            prepared.rng_seed = seed.map(|seed| seed.wrapping_add(position));
            Ok(prepared)
        })
        .collect::<Result<Vec<_>>>()?;
    if chunks.is_empty() {
        return Err(Error::InvalidInput(
            "Kokoro text produced no synthesizable chunks".to_string(),
        ));
    }
    Ok(crate::engine::KokoroPreparedText {
        source_text: text.to_string(),
        requested_speaker: speaker.map(str::to_string),
        requested_language: language.map(str::to_string),
        requested_speed: speed,
        chunks,
    })
}

fn checked_kokoro_sample_growth(
    current: usize,
    additional: usize,
    max_samples: usize,
    label: &str,
) -> Result<usize> {
    let next = current
        .checked_add(additional)
        .ok_or_else(|| Error::Overloaded(format!("Kokoro {label} count overflowed")))?;
    if next > max_samples {
        return Err(Error::InferenceError(format!(
            "Kokoro {label} exceeded the admitted hard sample contract: {next} > {max_samples}"
        )));
    }
    Ok(next)
}

fn validate_kokoro_samples(samples: &Vec<f32>, max_samples: usize, label: &str) -> Result<()> {
    checked_kokoro_sample_growth(0, samples.len(), max_samples, label)?;
    checked_kokoro_sample_growth(
        0,
        samples.capacity(),
        max_samples,
        &format!("{label} capacity"),
    )?;
    Ok(())
}

/// Records the stream's output as materialized: the input plus the largest
/// chunk buffer so far (`recorded`, returned updated). Each chunk's buffer is
/// handed to the client before the next is synthesized into a new one, and
/// a later chunk is often smaller (the stream starts with a short chunk, and
/// its last is partial), so recording each chunk's own size would lower the
/// observation, which the authority refuses: freed memory must go back to
/// pending first. The largest chunk stays inside the one-chunk reservation.
fn record_kokoro_materialized_output(
    job: &JobLease,
    retained_input_bytes: u64,
    recorded: usize,
    output_capacity_samples: usize,
    max_samples: usize,
) -> Result<usize> {
    checked_kokoro_sample_growth(
        0,
        output_capacity_samples,
        max_samples,
        "materialized sample capacity",
    )?;
    let largest = recorded.max(output_capacity_samples);
    let output_bytes = u64::try_from(largest)
        .ok()
        .and_then(|samples| samples.checked_mul(std::mem::size_of::<f32>() as u64))
        .ok_or_else(|| Error::Overloaded("Kokoro materialized output overflowed".to_string()))?;
    let materialized_host_bytes =
        retained_input_bytes
            .checked_add(output_bytes)
            .ok_or_else(|| {
                Error::Overloaded("Kokoro materialized host usage overflowed".to_string())
            })?;
    // The output ceiling is part of the job's immutable pre-admission
    // reservation. This only reclassifies authorized pending bytes after the
    // bounded Vec exists; it never grows the lease.
    job.record_materialized_usage(JobResourceObservation::host(materialized_host_bytes))?;
    Ok(largest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{ExecutionGroupId, ModelInstanceId};
    use crate::runtime::adapters::{LoadedModelBundle, RuntimeAdapterRegistry};

    /// A 45-word stream: a short first chunk, two full ones and a partial
    /// last chunk, recorded through the job's real resource accounting.
    /// Recording each chunk's own size lowered the observation at the
    /// third chunk, and the authority's refusal ended the stream with a
    /// 400-class "Invalid input" after most of the audio was sent.
    #[tokio::test]
    async fn a_stream_of_uneven_chunks_records_its_output_without_lowering_it() {
        use crate::engine::{Priority, ResourceAmount, ResourceVector, WorkloadClass};
        use crate::runtime::coordinator::{CoordinatorLane, InferenceCoordinator, JobSpec};
        let (input, limit) = (4_096u64, 96_000usize);
        let coordinator = std::sync::Arc::new(InferenceCoordinator::new(BackendKind::Cpu, 1, 2));
        let mut resources = ResourceVector::zero();
        resources.host_bytes = ResourceAmount::Known(input + limit as u64 * 4);
        let job = coordinator
            .admit(JobSpec {
                request_id: "kokoro-stream".to_string(),
                lane: CoordinatorLane::Atomic,
                priority: Priority::Normal,
                workload_class: WorkloadClass::Streaming,
                deadline: None,
                resources,
            })
            .await
            .unwrap();
        let mut recorded = 0;
        for chunk in [18_000, 96_000, 90_000, 31_000] {
            recorded =
                record_kokoro_materialized_output(&job, input, recorded, chunk, limit).unwrap();
        }
        assert_eq!(recorded, 96_000);
        // Lowering an observation is the owner's accounting bug: a server
        // error, not the client's invalid input.
        let lowered = job.record_materialized_usage(
            crate::runtime::coordinator::JobResourceObservation::host(input),
        );
        assert!(
            matches!(lowered, Err(Error::InferenceError(_))),
            "{lowered:?}"
        );
    }

    /// A voice pack that takes one phoneme per non-space character and
    /// refuses more than 510, as the model does. Its G2P splits sentences at
    /// `.` and uses the text as phonemes; packing is the model's planner.
    pub(crate) struct FakeKokoroPreparer;

    impl KokoroPreparer for FakeKokoroPreparer {
        fn plan(
            &self,
            text: &str,
            _speaker: Option<&str>,
            _language: Option<&str>,
            policy: KokoroChunkPolicy,
        ) -> Result<Vec<KokoroPlannedChunk>> {
            use crate::models::architectures::kokoro::phonemizer::{
                KokoroBoundary, KokoroPhonemeUnit,
            };
            let units = text
                .split_inclusive('.')
                .map(str::trim)
                .filter(|sentence| !sentence.is_empty())
                .map(|sentence| KokoroPhonemeUnit {
                    text: sentence.to_string(),
                    phonemes: sentence.to_string(),
                    boundary: KokoroBoundary::Sentence,
                })
                .collect();
            Ok(crate::models::architectures::kokoro::planning::plan_chunks(
                units, policy, 510,
            ))
        }

        fn prepare_planned(
            &self,
            chunk: &KokoroPlannedChunk,
            speaker: Option<&str>,
            language: Option<&str>,
            speed: f32,
        ) -> Result<KokoroPreparedRequest> {
            let phonemes = chunk.phonemes.chars().count();
            if phonemes > 510 {
                return Err(Error::InvalidInput(format!(
                    "Kokoro phoneme sequence length {phonemes} exceeds voice-pack limit (510)"
                )));
            }
            Ok(KokoroPreparedRequest {
                source_text: chunk.text.clone(),
                requested_speaker: speaker.map(str::to_string),
                requested_language: language.map(str::to_string),
                requested_speed: speed,
                phonemes: chunk.phonemes.clone(),
                token_ids: vec![1; phonemes.max(1)],
                ref_style: candle_core::Tensor::zeros(
                    (1, 256),
                    candle_core::DType::F32,
                    &candle_core::Device::Cpu,
                )
                .unwrap(),
                speed,
                rng_seed: None,
            })
        }
    }

    /// KOK-1/TPIPE-1: non-stream text beyond the voice pack's 510 phonemes
    /// returned 500 because only the streaming route planned chunks. The
    /// engine route now prepares the same planned chunks: each fits, they
    /// keep every word in order, and short text stays one chunk.
    #[test]
    fn engine_kokoro_preparation_plans_chunks_for_text_beyond_the_voice_pack() {
        let sentence = "The quick brown fox jumps over the lazy dog near the river bank. ";
        let long = sentence.repeat(25);
        assert!(long.split_whitespace().count() >= 250);
        let prepared = prepare_kokoro_text(
            &FakeKokoroPreparer,
            &long,
            Some("af_heart"),
            None,
            1.0,
            None,
        )
        .unwrap();
        assert!(
            prepared.chunks.len() > 1,
            "{} chunks",
            prepared.chunks.len()
        );
        assert!(prepared
            .chunks
            .iter()
            .all(|chunk| chunk.token_ids.len() <= 510));
        let rejoined = prepared
            .chunks
            .iter()
            .flat_map(|chunk| chunk.source_text.split_whitespace())
            .collect::<Vec<_>>();
        assert_eq!(rejoined, long.split_whitespace().collect::<Vec<_>>());
        assert_eq!(prepared.source_text, long);
        assert_eq!(prepared.requested_speaker.as_deref(), Some("af_heart"));

        let short =
            prepare_kokoro_text(&FakeKokoroPreparer, sentence, None, None, 1.0, None).unwrap();
        assert_eq!(short.chunks.len(), 1);
        assert_eq!(short.chunks[0].rng_seed, None, "no seed draws fresh noise");
    }

    /// A request seed reaches every chunk, offset by its position, so a
    /// seeded request repeats exactly however its chunks are batched.
    #[test]
    fn a_request_seed_seeds_every_chunk_in_order() {
        let long = "The quick brown fox jumps over the lazy dog near the river bank. ".repeat(25);
        let prepared =
            prepare_kokoro_text(&FakeKokoroPreparer, &long, None, None, 1.0, Some(40)).unwrap();
        let seeds = prepared
            .chunks
            .iter()
            .map(|chunk| chunk.rng_seed)
            .collect::<Vec<_>>();
        let expected = (40..)
            .take(prepared.chunks.len())
            .map(Some)
            .collect::<Vec<_>>();
        assert!(seeds.len() > 1);
        assert_eq!(seeds, expected);
    }

    /// KOK-12: the streaming policy starts with a short chunk for time
    /// to first audio, and no chunk exceeds the stream target the admission
    /// price (`kokoro_stream_chunk_frames`) assumes.
    #[test]
    fn streaming_plan_starts_short_and_stays_within_the_stream_target() {
        let text = "Good morning. Today we are testing a speech synthesis system on a laptop. \
                    The goal is to measure how fast it runs, how much memory it uses, and whether \
                    every word in this paragraph comes out clearly. Then we stop.";
        let chunks = FakeKokoroPreparer
            .plan(
                text,
                None,
                None,
                KokoroChunkPolicy::streaming(KOKORO_STREAM_TARGET_CHARS),
            )
            .unwrap();
        // The first chunk is at most 80 characters of whole sentences.
        assert_eq!(
            chunks[0].text,
            "Good morning. Today we are testing a speech synthesis system on a laptop."
        );
        assert!(chunks.len() >= 2);
        assert!(chunks
            .iter()
            .all(|chunk| chunk.text.chars().count() <= KOKORO_STREAM_TARGET_CHARS));
        let rejoined = chunks
            .iter()
            .flat_map(|chunk| chunk.text.split_whitespace())
            .collect::<Vec<_>>();
        assert_eq!(rejoined, text.split_whitespace().collect::<Vec<_>>());
    }

    #[test]
    fn kokoro_routes_streaming_requests_through_the_published_streaming_capability() {
        let registry = RuntimeAdapterRegistry::built_in();

        let streaming = registry
            .require(kokoro_streaming_capability(), ModelVariant::Kokoro82M)
            .expect("Kokoro streaming TTS capability");
        assert_eq!(streaming.capability, CapabilityKind::StreamingTts);
        assert_eq!(streaming.execution_target, ExecutionTargetKind::DirectModel);

        let bundle = LoadedModelBundle::bind(
            &registry,
            ExecutionGroupId::new(1),
            ModelInstanceId::new(1),
            ModelVariant::Kokoro82M,
            BackendKind::Cpu,
        )
        .expect("loaded Kokoro bundle");
        let contract = bundle
            .contract(kokoro_streaming_capability(), true)
            .expect("loaded Kokoro streaming contract");
        assert_eq!(
            contract.metadata.execution_target,
            ExecutionTargetKind::DirectModel
        );
    }

    /// Short replies ("Sure.", "Call 911.") and text the G2P expands
    /// (numbers, currency) speak within the chunk audio a stream admits for
    /// them, on the real checkpoint with its slowest measured voice. The
    /// character-only price admitted a one-character text at 2,400 samples,
    /// and the voice pipeline failed "Sure." (34,200 > 2,400).
    #[test]
    fn short_and_expanded_texts_fit_their_admitted_chunk_audio_if_env_set() {
        use crate::backends::DeviceSelector;
        use std::io::Write;
        let Some(model_dir) = std::env::var_os("IZWI_KOKORO_MODEL_DIR") else {
            let _ = writeln!(std::io::stderr(), "IZWI_KOKORO_MODEL_DIR unset; not run");
            return;
        };
        let device = DeviceSelector::detect_with_preference(Some("cpu")).expect("cpu device");
        let model = KokoroTtsModel::load(std::path::Path::new(&model_dir), device)
            .expect("load local Kokoro model");
        let voice = Some("am_michael");
        let mut failures = Vec::new();
        for text in [
            "A",
            "1",
            "OK.",
            "Sure.",
            "Wait, really?",
            "Call 911.",
            "That costs $1,234,567.89.",
            "Dr. Lee, 221B Baker St.",
        ] {
            let limit = kokoro_stream_chunk_sample_limit(text, 1.0, BackendKind::Cpu).unwrap();
            let policy = KokoroChunkPolicy::streaming(KOKORO_STREAM_TARGET_CHARS);
            for chunk in model.plan(text, voice, None, policy).unwrap() {
                let samples =
                    synthesize_planned_kokoro(&model, &chunk, voice, None, 1.0, usize::MAX)
                        .unwrap()
                        .samples
                        .len();
                if samples > limit {
                    failures.push(format!("{:?}: {samples} > {limit}", chunk.text));
                }
            }
        }
        assert!(failures.is_empty(), "{failures:#?}");
    }
}
