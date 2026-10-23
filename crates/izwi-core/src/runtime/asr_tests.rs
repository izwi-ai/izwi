use super::*;
use crate::backends::BackendPreference;
use crate::config::EngineConfig;
use crate::engine::{EngineCoreRequest, ModelInstanceId, Priority, WorkloadClass};
use crate::runtime::adapters::{LoadedModelBundleDraft, RuntimeAdapterRegistry};
use crate::runtime::coordinator::JobSpec;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};
use uuid::Uuid;

/// The shared automatic ASR ceiling: prompt, one chunk of audio tokens and
/// the decode budget per sequence; one per retained row, under the native
/// context. Qwen3-ASR's 1,307 and VibeVoice's retained 1,938 come from it.
#[test]
fn shared_asr_ceiling_bounds_one_chunk_per_sequence() {
    assert_eq!(
        asr_sequence_ceiling(ASR_TEXT_PROMPT_TOKEN_RESERVE, 13.0, 30.0 / 0.95, 768),
        1_307
    );
    assert_eq!(
        asr_sequence_ceiling(
            512,
            7.5,
            60.0,
            crate::models::architectures::vibevoice::asr::vibevoice_auto_asr_max_tokens_for_duration(60.0)
        ),
        1_938
    );
    assert_eq!(asr_sequence_ceiling(128, 13.0, -1.0, 768), 896);
    assert_eq!(asr_automatic_context(65_536, 1_938, 8), (1_938, 15_504));
    assert_eq!(asr_automatic_context(65_536, 1_938, 0), (1_938, 1_938));
    assert_eq!(asr_automatic_context(1_024, 1_938, 8), (1_024, 1_024));
}

#[test]
fn offline_only_asr_models_stream_on_the_offline_route() {
    let adapters = RuntimeAdapterRegistry::built_in();
    for variant in [
        ModelVariant::ParakeetTdt06BV3,
        ModelVariant::Nemotron35AsrStreaming06B,
    ] {
        assert!(
            !asr_streams_natively(&adapters, variant),
            "{variant} has offline ASR only, so its streams take the offline route"
        );
    }
    let qwen = ModelVariant::Qwen3Asr06BGguf;
    assert!(asr_streams_natively(&adapters, qwen));
}

#[test]
fn stream_deltas_end_with_the_unstreamed_transcript_tail() {
    // Final-only: nothing streamed, the whole transcript is one delta.
    assert_eq!(
        unstreamed_transcript_tail("", "hello world"),
        Some("hello world")
    );
    assert_eq!(
        unstreamed_transcript_tail("hello ", "hello world"),
        Some("world")
    );
    assert_eq!(unstreamed_transcript_tail("hello", "hello"), None);
    assert_eq!(
        unstreamed_transcript_tail(" hello", "hello world"),
        Some(" world")
    );
    assert_eq!(unstreamed_transcript_tail("", ""), None);
    // Cleanup rewrote text already sent: appending cannot repair it.
    assert_eq!(unstreamed_transcript_tail("aaaaaaaa", "aaa b"), None);
}

fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

/// SPK-F9: realtime sessions default to production limits (an 8-hour
/// lifetime instead of 10 minutes, a 90 s idle timeout instead of 30 s,
/// 16 sessions on CUDA), and the engine config overrides the
/// environment and the defaults.
#[test]
fn realtime_session_limits_default_for_production_and_follow_the_config() {
    // serve_runtime's tests set these variables under this lock.
    let _env = crate::env_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let env_set = [
        ENV_ASR_REALTIME_MAX_SESSIONS,
        ENV_ASR_REALTIME_MAX_LIFETIME_SECS,
        ENV_ASR_REALTIME_IDLE_TIMEOUT_SECS,
    ]
    .iter()
    .any(|name| std::env::var_os(name).is_some());
    if !env_set {
        let resolve = |backend| {
            RealtimeAsrSessionLimits::resolve(RealtimeAsrSessionConfig::default(), backend)
                .unwrap()
        };
        let cuda = resolve(BackendKind::Cuda);
        assert_eq!(cuda.max_sessions, 16);
        assert_eq!(cuda.max_lifetime, Duration::from_secs(8 * 60 * 60));
        assert_eq!(cuda.idle_timeout, Duration::from_secs(90));
        for backend in [BackendKind::Cpu, BackendKind::Metal] {
            let limits = resolve(backend);
            assert_eq!(limits.max_sessions, 4, "{backend:?}");
            assert_eq!(limits.max_lifetime, cuda.max_lifetime);
            assert_eq!(limits.idle_timeout, cuda.idle_timeout);
        }
    }

    let configured = RealtimeAsrSessionLimits::resolve(
        RealtimeAsrSessionConfig {
            max_sessions: Some(9),
            max_lifetime_secs: Some(3_600),
            idle_timeout_secs: Some(120),
        },
        BackendKind::Metal,
    )
    .unwrap();
    assert_eq!(configured.max_sessions, 9);
    assert_eq!(configured.max_lifetime, Duration::from_secs(3_600));
    assert_eq!(configured.idle_timeout, Duration::from_secs(120));
    for zero in [
        RealtimeAsrSessionConfig {
            max_sessions: Some(0),
            ..Default::default()
        },
        RealtimeAsrSessionConfig {
            max_lifetime_secs: Some(0),
            ..Default::default()
        },
        RealtimeAsrSessionConfig {
            idle_timeout_secs: Some(0),
            ..Default::default()
        },
    ] {
        assert!(matches!(
            RealtimeAsrSessionLimits::resolve(zero, BackendKind::Cpu),
            Err(Error::ConfigError(_))
        ));
    }
}

fn saa_decode_policy(secs: f32) -> usize {
    (GRANITE_SAA_NEW_TOKEN_RESERVE
        + (secs * GRANITE_SAA_NEW_TOKENS_PER_SECOND).ceil() as usize)
        .clamp(GRANITE_SAA_MIN_NEW_TOKENS, GRANITE_SAA_MAX_NEW_TOKENS)
}

fn saa_demand_fits(context_tokens: usize, secs: f32) -> bool {
    let audio_tokens = (secs * 10.0).ceil() as usize;
    ASR_TEXT_PROMPT_TOKEN_RESERVE
        .saturating_add(audio_tokens)
        .saturating_add(saa_decode_policy(secs))
        <= context_tokens
}

#[test]
fn invocation_solver_bounds_chunk_seconds_to_context() {
    let InvocationAudioLimit::Limited(limit) =
        solve_invocation_audio_limit(4_096, 10.0, ASR_TEXT_PROMPT_TOKEN_RESERVE, saa_decode_policy)
    else {
        panic!("a 4096-token context must admit audio chunks");
    };
    assert!(
        (205.0..=207.0).contains(&limit),
        "unexpected solved limit {limit}"
    );
    assert!(saa_demand_fits(4_096, limit));
    assert!(!saa_demand_fits(4_096, limit + 0.5));

    // A 1024-token context cannot fit the 30 s family minimum: the solver
    // degrades to roughly 9.6 s so constrained hosts stay functional.
    let InvocationAudioLimit::Limited(small) =
        solve_invocation_audio_limit(1_024, 10.0, ASR_TEXT_PROMPT_TOKEN_RESERVE, saa_decode_policy)
    else {
        panic!("a 1024-token context must still admit minimal chunks");
    };
    assert!((9.0..=9.65).contains(&small), "unexpected limit {small}");
    assert!(small < GRANITE_SAA_MIN_CHUNK_SECS);
}

#[test]
fn invocation_solver_fails_closed_when_minimum_does_not_fit() {
    let outcome =
        solve_invocation_audio_limit(512, 10.0, ASR_TEXT_PROMPT_TOKEN_RESERVE, saa_decode_policy);
    assert_eq!(outcome, InvocationAudioLimit::Infeasible);
    let outcome = solve_invocation_audio_limit(4_096, 0.0, ASR_TEXT_PROMPT_TOKEN_RESERVE, saa_decode_policy);
    assert_eq!(outcome, InvocationAudioLimit::Infeasible);
}

#[test]
fn compose_keeps_hint_when_audio_never_reaches_the_decoder() {
    let composed = compose_invocation_audio_limit(Some(30.0), None, Some(1_024), saa_decode_policy)
        .expect("no-rate families compose to their hint");
    assert_eq!(composed, Some(30.0));

    let composed =
        compose_invocation_audio_limit(Some(30.0), Some(10.0), None, saa_decode_policy)
            .expect("no context resolves to the hint alone");
    assert_eq!(composed, Some(30.0));
}

#[test]
fn compose_applies_the_smaller_of_hint_and_context_budget() {
    let composed = compose_invocation_audio_limit(
        Some(540.0),
        Some(10.0),
        Some(4_096),
        saa_decode_policy,
    )
    .expect("feasible context composes");
    let InvocationAudioLimit::Limited(context_limit) =
        solve_invocation_audio_limit(4_096, 10.0, ASR_TEXT_PROMPT_TOKEN_RESERVE, saa_decode_policy)
    else {
        panic!("feasible context expected");
    };
    assert_eq!(composed, Some(540.0_f32.min(context_limit)));

    let error = compose_invocation_audio_limit(Some(540.0), Some(10.0), Some(512), saa_decode_policy)
        .expect_err("infeasible context must fail closed");
    assert!(error.to_string().contains("cannot fit even a minimal"));
}

#[test]
fn realtime_stream_variant_resolves_both_engine_backed_families() {
    assert_eq!(
        resolve_asr_realtime_stream_variant(Some("nvidia/nemotron-3.5-asr-streaming-0.6b",)),
        Some(ModelVariant::Nemotron35AsrStreaming06B)
    );
    assert_eq!(
        resolve_asr_realtime_stream_variant(Some("Nemotron 3.5 ASR Streaming 0.6B")),
        Some(ModelVariant::Nemotron35AsrStreaming06B)
    );
    assert_eq!(
        resolve_asr_realtime_stream_variant(Some("Voxtral-Mini-4B-Realtime-2602")),
        Some(ModelVariant::VoxtralMini4BRealtime2602)
    );
    assert_eq!(
        realtime_asr_execution_path(ModelVariant::Nemotron35AsrStreaming06B),
        RealtimeAsrPath::Direct
    );
    assert_eq!(
        realtime_asr_execution_path(ModelVariant::VoxtralMini4BRealtime2602),
        RealtimeAsrPath::Engine
    );
    assert_eq!(
        realtime_asr_execution_path(ModelVariant::WhisperLargeV3Turbo),
        RealtimeAsrPath::Direct
    );

    assert_eq!(resolve_asr_realtime_stream_variant(None), None);
    assert_eq!(
        resolve_asr_realtime_stream_variant(Some("Qwen3-ASR-1.7B")),
        None
    );
    assert_eq!(
        resolve_asr_realtime_stream_variant(Some("Whisper-Large-v3-Turbo")),
        None
    );
    assert_eq!(
        resolve_asr_realtime_stream_variant(Some("Parakeet-TDT-0.6B-v3")),
        None
    );
    assert_eq!(
        resolve_asr_realtime_stream_variant(Some("not-a-real-model")),
        None
    );
}

#[tokio::test]
async fn nemotron_public_push_and_finish_select_the_direct_native_path() {
    let runtime = RuntimeService::new(EngineConfig {
        backend: BackendPreference::Cpu,
        ..Default::default()
    })
    .unwrap();
    let now = Instant::now();
    let resources = Arc::new(StdMutex::new(RuntimeAsrRealtimeResources {
        model: None,
        state: None,
        execution_contract: None,
        residency_lease: None,
        job: None,
        session_lease: None,
        engine_session: None,
        voxtral_model: None,
        absolute_deadline: now + Duration::from_secs(60),
        idle_timeout: Duration::from_secs(60),
        last_activity: now,
        active_operations: 0,
        closed: false,
        timeout_reason: None,
    }));
    let mut stream = RuntimeAsrRealtimeStream {
        variant: ModelVariant::Nemotron35AsrStreaming06B,
        noise_gate: None,
        resources,
        activity: Arc::new(Notify::new()),
        operation_gate: Arc::new(tokio::sync::Mutex::new(())),
        max_samples: 16,
        committed_samples: 0,
        engine_sample_rate: None,
        engine_text: String::new(),
        engine_chunk_index: 0,
        segment: RealtimeEngineSegment::default(),
    };

    // Nemotron routes to the direct native streaming path: the fixture
    // omits its loaded model, so both operations fail closed there.
    let push = runtime
        .push_asr_realtime_samples(&mut stream, &[0.0], 16_000)
        .await
        .expect_err("fixture intentionally omits the native stream model");
    assert!(push
        .to_string()
        .contains("realtime ASR stream model is unavailable"));
    assert!(!push.to_string().contains("Engine realtime ASR session"));

    let finish = runtime
        .finish_asr_realtime_stream(&mut stream)
        .await
        .expect_err("fixture intentionally omits the native stream model");
    assert!(finish
        .to_string()
        .contains("realtime ASR stream model is unavailable"));
    assert!(!finish.to_string().contains("Engine realtime ASR session"));
}

/// Only a model that hallucinates on non-speech pays the gate's lookahead:
/// Voxtral is gated, the Nemotron transducer is not.
#[test]
fn realtime_noise_gate_covers_hallucinating_families_only() {
    let gate = realtime_noise_gate(ModelVariant::VoxtralMini4BRealtime2602)
        .expect("Voxtral realtime is gated");
    assert_eq!(gate.lookahead_ms(), izwi_vad::DEFAULT_NOISE_GATE_LOOKAHEAD_MS);
    assert!(realtime_noise_gate(ModelVariant::Nemotron35AsrStreaming06B).is_none());
}

fn fixture_realtime_stream(
    max_samples: usize,
    session_lease: Option<Arc<RealtimeAsrSessionLease>>,
) -> RuntimeAsrRealtimeStream {
    let now = Instant::now();
    let resources = Arc::new(StdMutex::new(RuntimeAsrRealtimeResources {
        model: None,
        state: None,
        execution_contract: None,
        residency_lease: None,
        job: None,
        session_lease,
        engine_session: None,
        voxtral_model: None,
        absolute_deadline: now + Duration::from_secs(60),
        idle_timeout: Duration::from_secs(60),
        last_activity: now,
        active_operations: 0,
        closed: false,
        timeout_reason: None,
    }));
    RuntimeAsrRealtimeStream {
        variant: ModelVariant::VoxtralMini4BRealtime2602,
        noise_gate: None,
        resources,
        activity: Arc::new(Notify::new()),
        operation_gate: Arc::new(tokio::sync::Mutex::new(())),
        max_samples,
        committed_samples: 0,
        engine_sample_rate: None,
        engine_text: String::new(),
        engine_chunk_index: 0,
        segment: RealtimeEngineSegment::new(Some("en"), None),
    }
}

fn text_output(text: &str, is_final: bool) -> StreamingOutput {
    let mut output = StreamingOutput::new("rollover".to_string(), 0, Vec::new(), 16_000);
    output.text = Some(text.to_string());
    output.is_final = is_final;
    output
}

/// SPK-F1: Voxtral prepares at most 30 s per session. Drive 70 s of
/// synthetic speech with pauses, then 70 s without any, through the
/// segment clock: no segment may exceed the ceiling, a segment ends in a
/// pause when the speaker leaves one near the ceiling, and continuous
/// speech still rolls over exactly at the ceiling.
#[test]
fn realtime_segment_clock_rolls_over_at_a_pause_or_the_ceiling() {
    const RATE: u32 = 16_000;
    const MAX: usize = 480_000;
    const FRAME: usize = 320;
    let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
    let mut speech = move || {
        (0..FRAME)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.6
            })
            .collect::<Vec<f32>>()
    };
    let silence = vec![0.0_f32; FRAME];
    let pause_samples = (RATE as usize) * REALTIME_SEGMENT_PAUSE_MS as usize / 1_000;
    let soft_start = MAX - MAX / REALTIME_SEGMENT_SOFT_ROLLOVER_DIVISOR;

    let run = |pattern: &mut dyn FnMut(usize) -> Vec<f32>| {
        let (mut committed, mut quiet, mut boundaries) = (0_usize, 0_usize, Vec::new());
        for frame_index in 0..(70 * RATE as usize / FRAME) {
            let frame = pattern(frame_index);
            if realtime_segment_rollover_due(committed, frame.len(), quiet, MAX, RATE) {
                boundaries.push((committed, quiet));
                committed = 0;
                quiet = 0;
            }
            committed += frame.len();
            assert!(committed <= MAX, "a segment exceeded the session ceiling");
            quiet = realtime_trailing_quiet_samples(quiet, &frame);
        }
        boundaries
    };

    // 1.6 s of speech, then 0.4 s of silence, repeated.
    let paused = run(&mut |index| {
        if index % 100 < 80 {
            speech()
        } else {
            silence.clone()
        }
    });
    assert_eq!(paused.len(), 2, "70 s of audio spans three segments");
    for (committed, quiet) in paused {
        assert!(committed >= soft_start && committed < MAX);
        assert!(quiet >= pause_samples, "the boundary must fall in a pause");
    }

    let continuous = run(&mut |_| speech());
    assert_eq!(continuous, vec![(MAX, 0), (MAX, 0)]);
}

#[test]
fn realtime_segment_clock_never_rolls_over_an_empty_segment() {
    let due = |committed, incoming, quiet| {
        realtime_segment_rollover_due(committed, incoming, quiet, 480_000, 16_000)
    };
    assert!(!due(0, 480_000, 0));
    assert!(!due(0, 320, 1_000_000));
    // Quiet audio before the soft window is not a boundary.
    assert!(!due(100_000, 320, 16_000));
    assert!(due(479_900, 320, 0));
}

/// A rollover keeps one continuous transcript: chunk numbering carries
/// over, the segment flush is not reported as the stream's final event,
/// and a missing word separator is restored at the boundary.
#[tokio::test]
async fn engine_segment_continuation_keeps_one_transcript_and_quota_slot() {
    let policy = RealtimeAsrSessionPolicy::new(RealtimeAsrSessionLimits {
        max_sessions: 1,
        max_lifetime: Duration::from_secs(60),
        idle_timeout: Duration::from_secs(60),
    })
    .expect("session policy");
    let lease = policy.try_acquire().expect("session slot");
    let mut stream = fixture_realtime_stream(480_000, Some(lease));
    stream.committed_samples = 480_000;
    stream.engine_sample_rate = Some(16_000);
    stream.segment.trailing_quiet_samples = 3_200;

    let first = stream.map_engine_outputs(vec![
        text_output("Hello", false),
        text_output(" world", false),
    ]);
    assert_eq!(first[1].text, "Hello world");

    let (session_lease, _, _) = stream.segment_authorities().expect("open stream");
    release_realtime_asr_cleanup(stream.detach())
        .await
        .expect("release the finished segment");
    assert!(stream.is_closed());
    assert_eq!(
        policy.permits.available_permits(),
        0,
        "the session keeps its quota slot across the rollover"
    );

    let mut next = fixture_realtime_stream(240_000, Some(session_lease));
    stream.continue_on(&mut next);
    drop(next);
    assert!(!stream.is_closed());
    assert_eq!(stream.max_samples, 240_000);
    assert_eq!(stream.committed_samples, 0);
    assert_eq!(stream.engine_sample_rate, None);
    assert_eq!(stream.segment.trailing_quiet_samples, 0);
    assert_eq!(stream.segment.language.as_deref(), Some("en"));
    assert_eq!(policy.permits.available_permits(), 0);

    let second = stream.map_engine_outputs(vec![
        text_output("", false),
        text_output("again", false),
        text_output(" today", true),
    ]);
    assert_eq!(second[1].delta, " again");
    assert_eq!(second[2].text, "Hello world again today");
    assert_eq!(second[2].chunk_index, 4);
    assert!(second[2].is_final);

    drop(stream);
    tokio::time::timeout(Duration::from_secs(1), async {
        while policy.permits.available_permits() != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("closing the stream releases its one slot");
}

/// SPK-F1 regression: a push past the Engine session ceiling used to be
/// rejected while the stream stayed open, so every later frame failed
/// the same way. It must now roll over, or end the stream if the
/// rollover cannot run (this fixture has no Engine session to finish).
#[tokio::test]
async fn push_past_the_engine_ceiling_never_leaves_a_stream_that_rejects_every_frame() {
    let runtime = RuntimeService::new(EngineConfig {
        backend: BackendPreference::Cpu,
        ..Default::default()
    })
    .unwrap();
    let policy = RealtimeAsrSessionPolicy::new(RealtimeAsrSessionLimits {
        max_sessions: 1,
        max_lifetime: Duration::from_secs(60),
        idle_timeout: Duration::from_secs(60),
    })
    .expect("session policy");
    let mut stream =
        fixture_realtime_stream(16, Some(policy.try_acquire().expect("session slot")));
    stream.committed_samples = 16;
    stream.engine_sample_rate = Some(16_000);

    let error = runtime
        .push_asr_realtime_samples(&mut stream, &[0.25; 4], 16_000)
        .await
        .expect_err("the fixture cannot open a new segment");
    assert!(
        !error.to_string().contains("exceeds the stream reservation"),
        "the ceiling must trigger a rollover, not a per-frame rejection: {error}"
    );
    assert!(stream.is_closed(), "a failed rollover must end the stream");
}

#[test]
fn realtime_stream_reservation_routes_to_backend_memory_domains() {
    let cpu = realtime_stream_resource_vector(BackendKind::Cpu, 11, 29).unwrap();
    let metal = realtime_stream_resource_vector(BackendKind::Metal, 11, 29).unwrap();
    let cuda = realtime_stream_resource_vector(BackendKind::Cuda, 11, 29).unwrap();

    assert_eq!(cpu.host_bytes, ResourceAmount::Known(40));
    assert_eq!(cpu.device_bytes, ResourceAmount::Known(0));
    assert_eq!(cpu.unified_bytes, ResourceAmount::Known(0));
    assert_eq!(metal.host_bytes, ResourceAmount::Known(0));
    assert_eq!(metal.device_bytes, ResourceAmount::Known(0));
    assert_eq!(metal.unified_bytes, ResourceAmount::Known(40));
    assert_eq!(cuda.host_bytes, ResourceAmount::Known(11));
    assert_eq!(cuda.device_bytes, ResourceAmount::Known(29));
    assert_eq!(cuda.unified_bytes, ResourceAmount::Known(0));
}

#[test]
fn voxtral_operation_cost_uses_preparation_geometry_not_raw_samples() {
    let geometry = VoxtralRealtimePreparationGeometry {
        source_samples: 160,
        resampled_samples: 160,
        padded_samples: 320,
        mel_frames: 8,
        conv1_frames: 4,
        conv2_frames: 2,
        pooled_frames: 1,
        stable_frames: 1,
        embedding_elements: 32,
    };
    let batch = VoxtralRealtimePreparationBatchGeometry {
        width: 1,
        padded_mel_frames: 8,
        padded_conv_frames: 2,
        materialized_tensor_elements_per_row: 704,
        workspace_per_row_bytes: 4096,
    };
    let seal = VoxtralRealtimePreparationStageSeal {
        max_source_samples: 1_600,
        max_work_units: 1_600,
        max_materialized_tensor_elements_per_row: 1_024,
        max_workspace_bytes: 8192,
    };

    let cost = voxtral_realtime_preparation_cost_from_geometry(geometry, batch, seal).unwrap();

    assert_eq!(cost.logical_units, 160);
    assert_eq!(cost.tensor_elements, 704);
    assert_eq!(cost.workspace.temporary_bytes, ResourceAmount::Known(4096));
    assert_ne!(cost.tensor_elements, geometry.source_samples as u64);

    let undersized_seal = VoxtralRealtimePreparationStageSeal {
        max_materialized_tensor_elements_per_row: 703,
        ..seal
    };
    assert!(
        voxtral_realtime_preparation_cost_from_geometry(geometry, batch, undersized_seal)
            .is_err()
    );
}

#[test]
fn voxtral_committed_observation_excludes_transient_workspace() {
    let observed = voxtral_realtime_committed_observation(
        11,
        VoxtralRealtimePreparedResourceUsage {
            host_bytes: 40,
            tensor_bytes: 96,
        },
    )
    .unwrap();

    assert_eq!(observed, JobResourceObservation::new(62, 96));
}

#[test]
fn realtime_session_quota_is_held_until_all_in_flight_clones_exit() {
    let policy = RealtimeAsrSessionPolicy::new(RealtimeAsrSessionLimits {
        max_sessions: 1,
        max_lifetime: Duration::from_secs(60),
        idle_timeout: Duration::from_secs(10),
    })
    .expect("session policy");
    let stream_lease = policy.try_acquire().expect("first session");
    let in_flight_lease = stream_lease.clone();

    drop(stream_lease);
    assert!(matches!(policy.try_acquire(), Err(Error::Overloaded(_))));

    drop(in_flight_lease);
    assert!(policy.try_acquire().is_ok());
}

#[test]
fn realtime_session_policy_rejects_quota_that_would_panic_semaphore() {
    let result = RealtimeAsrSessionPolicy::new(RealtimeAsrSessionLimits {
        max_sessions: Semaphore::MAX_PERMITS + 1,
        max_lifetime: Duration::from_secs(60),
        idle_timeout: Duration::from_secs(10),
    });

    assert!(matches!(result, Err(Error::ConfigError(_))));
}

#[test]
fn realtime_session_expiration_distinguishes_idle_and_absolute_limits() {
    let started_at = Instant::now();
    let idle_deadline = started_at + Duration::from_secs(2);
    let absolute_deadline = started_at + Duration::from_secs(10);

    assert_eq!(
        realtime_asr_session_expiration(idle_deadline, absolute_deadline, idle_deadline, 0,),
        Some("realtime ASR stream exceeded its idle timeout")
    );
    assert_eq!(
        realtime_asr_session_expiration(idle_deadline, absolute_deadline, idle_deadline, 1,),
        None
    );
    assert_eq!(
        realtime_asr_session_expiration(absolute_deadline, absolute_deadline, idle_deadline, 1,),
        Some("realtime ASR stream exceeded its absolute lifetime")
    );
}

#[tokio::test(flavor = "current_thread")]
async fn realtime_watchdog_releases_idle_session_quota() {
    let policy = RealtimeAsrSessionPolicy::new(RealtimeAsrSessionLimits {
        max_sessions: 1,
        max_lifetime: Duration::from_secs(60),
        idle_timeout: Duration::from_millis(1),
    })
    .expect("session policy");
    let session_lease = policy.try_acquire().expect("first session");
    let now = Instant::now();
    let resources = Arc::new(StdMutex::new(RuntimeAsrRealtimeResources {
        model: None,
        state: None,
        execution_contract: None,
        residency_lease: None,
        job: None,
        session_lease: Some(session_lease),
        engine_session: None,
        voxtral_model: None,
        absolute_deadline: now + Duration::from_secs(60),
        idle_timeout: Duration::from_millis(1),
        last_activity: now - Duration::from_secs(1),
        active_operations: 0,
        closed: false,
        timeout_reason: None,
    }));
    let activity = Arc::new(Notify::new());
    spawn_realtime_asr_watchdog(&resources, activity);

    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            let closed = resources.lock().expect("resource lock").closed;
            if closed && policy.permits.available_permits() == 1 {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("watchdog cleanup");

    let resources = resources.lock().expect("resource lock");
    assert_eq!(
        resources.timeout_reason,
        Some("realtime ASR stream exceeded its idle timeout")
    );
    assert!(resources.session_lease.is_none());
    drop(resources);
    assert!(policy.try_acquire().is_ok());
}

#[tokio::test(flavor = "current_thread")]
async fn realtime_stream_drop_releases_detached_session_quota() {
    let policy = RealtimeAsrSessionPolicy::new(RealtimeAsrSessionLimits {
        max_sessions: 1,
        max_lifetime: Duration::from_secs(60),
        idle_timeout: Duration::from_secs(60),
    })
    .expect("session policy");
    let session_lease = policy.try_acquire().expect("first session");
    let now = Instant::now();
    let resources = Arc::new(StdMutex::new(RuntimeAsrRealtimeResources {
        model: None,
        state: None,
        execution_contract: None,
        residency_lease: None,
        job: None,
        session_lease: Some(session_lease),
        engine_session: None,
        voxtral_model: None,
        absolute_deadline: now + Duration::from_secs(60),
        idle_timeout: Duration::from_secs(60),
        last_activity: now,
        active_operations: 0,
        closed: false,
        timeout_reason: None,
    }));
    let stream = RuntimeAsrRealtimeStream {
        variant: ModelVariant::VoxtralMini4BRealtime2602,
        noise_gate: None,
        resources,
        activity: Arc::new(Notify::new()),
        operation_gate: Arc::new(tokio::sync::Mutex::new(())),
        max_samples: 1,
        committed_samples: 0,
        engine_sample_rate: None,
        engine_text: String::new(),
        engine_chunk_index: 0,
        segment: RealtimeEngineSegment::default(),
    };

    drop(stream);
    tokio::time::timeout(Duration::from_secs(1), async {
        while policy.permits.available_permits() != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("drop cleanup");
    assert!(policy.try_acquire().is_ok());
}

#[tokio::test(flavor = "current_thread")]
async fn aborted_cleanup_task_retains_detached_authorities_fail_closed() {
    let policy = RealtimeAsrSessionPolicy::new(RealtimeAsrSessionLimits {
        max_sessions: 1,
        max_lifetime: Duration::from_secs(60),
        idle_timeout: Duration::from_secs(60),
    })
    .expect("session policy");
    let session_lease = policy.try_acquire().expect("first session");
    let cleanup = RealtimeAsrDetachedResources {
        state: None,
        model: None,
        execution_contract: None,
        residency_lease: None,
        job: None,
        session_lease: Some(session_lease),
        engine_session: None,
        voxtral_model: None,
    };
    let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
    let task = tokio::spawn(async move {
        let _guard = RealtimeAsrCleanupGuard::for_test(cleanup);
        let _ = entered_tx.send(());
        std::future::pending::<()>().await;
    });
    entered_rx.await.expect("cleanup task entered");

    task.abort();
    let _ = task.await;
    assert_eq!(policy.permits.available_permits(), 0);
    assert!(policy.try_acquire().is_err());
}

fn realtime_test_job(id: &str, backend: BackendKind, deadline: Option<Instant>) -> JobSpec {
    let mut resources = ResourceVector::zero();
    match backend {
        BackendKind::Cpu => resources.host_bytes = ResourceAmount::Known(1024 * 1024),
        BackendKind::Metal => resources.unified_bytes = ResourceAmount::Known(1024 * 1024),
        BackendKind::Cuda => {
            resources.host_bytes = ResourceAmount::Known(64 * 1024);
            resources.device_bytes = ResourceAmount::Known(1024 * 1024);
        }
    }
    JobSpec {
        request_id: id.to_string(),
        lane: CoordinatorLane::Realtime,
        priority: Priority::Normal,
        workload_class: WorkloadClass::Realtime,
        deadline,
        resources,
    }
}

fn realtime_test_contract(
    coordinator: &InferenceCoordinator,
    backend: BackendKind,
) -> LoadedExecutionContract {
    let draft = LoadedModelBundleDraft::build(
        &RuntimeAdapterRegistry::built_in(),
        coordinator.execution_group_id(),
        ModelInstanceId::new(1),
        ModelVariant::Nemotron35AsrStreaming06B,
        backend,
    )
    .expect("realtime test bundle draft");
    draft
        .execution_contracts(CapabilityKind::RealtimeAsr)
        .expect("realtime test contracts")
        .into_iter()
        .next()
        .expect("realtime adapter publishes a contract")
}

#[test]
fn oversized_realtime_packet_is_rejected_before_copy_or_model_work() {
    let side_effects = AtomicUsize::new(0);
    let result = validate_realtime_input_copy(4_801, 4_800).map(|()| {
        side_effects.fetch_add(1, Ordering::Relaxed);
        let _owned = vec![0.0_f32; 4_801];
        side_effects.fetch_add(1, Ordering::Relaxed);
    });

    assert!(matches!(result, Err(Error::InvalidInput(_))));
    assert_eq!(side_effects.load(Ordering::Relaxed), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn realtime_blocking_work_respects_deadline_without_blocking_tokio() {
    let coordinator = Arc::new(InferenceCoordinator::new(BackendKind::Cpu, 1, 4));
    let job = coordinator
        .admit(realtime_test_job(
            "realtime-blocking-deadline",
            BackendKind::Cpu,
            Some(Instant::now() + std::time::Duration::from_millis(200)),
        ))
        .await
        .unwrap();
    let runner_job = job.clone();
    drop(job);
    let gate = Arc::new(tokio::sync::Mutex::new(()));
    let release = Arc::new((Mutex::new(false), Condvar::new()));
    let task_release = release.clone();
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let task_coordinator = coordinator.clone();
    let execution_contract = realtime_test_contract(&coordinator, BackendKind::Cpu);
    let runner = tokio::spawn(async move {
        run_realtime_blocking_operation(
            &task_coordinator,
            &runner_job,
            execution_contract,
            gate,
            "asr.realtime.test",
            move || {
                Ok(move || {
                    let _ = started_tx.send(());
                    let (lock, wake) = &*task_release;
                    let mut released = lock.lock().unwrap_or_else(|poison| poison.into_inner());
                    while !*released {
                        released = wake
                            .wait(released)
                            .unwrap_or_else(|poison| poison.into_inner());
                    }
                    Ok(())
                })
            },
        )
        .await
    });

    started_rx.await.unwrap();
    let heartbeat = tokio::time::timeout(
        std::time::Duration::from_millis(50),
        tokio::time::sleep(std::time::Duration::from_millis(5)),
    )
    .await;
    let result = tokio::time::timeout(std::time::Duration::from_secs(1), runner).await;
    let while_blocked = coordinator.snapshot();
    {
        let (lock, wake) = &*release;
        *lock.lock().unwrap_or_else(|poison| poison.into_inner()) = true;
        wake.notify_all();
    }
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while coordinator.snapshot().active_executions != 0
            || coordinator.snapshot().active_jobs != 0
        {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();

    assert!(heartbeat.is_ok());
    assert!(matches!(
        result,
        Ok(Ok(Err(Error::Timeout(id)))) if id == "realtime-blocking-deadline"
    ));
    assert_eq!(while_blocked.active_executions, 1);
    assert_eq!(while_blocked.active_jobs, 1);
    assert_eq!(coordinator.snapshot().expired_total, 1);
}

#[tokio::test]
async fn cancelled_realtime_call_retains_stream_operation_order() {
    let coordinator = Arc::new(InferenceCoordinator::new(BackendKind::Cuda, 2, 4));
    let job = coordinator
        .admit(realtime_test_job(
            "realtime-cancellation-order",
            BackendKind::Cuda,
            None,
        ))
        .await
        .unwrap();
    let gate = Arc::new(tokio::sync::Mutex::new(()));
    let release = Arc::new((Mutex::new(false), Condvar::new()));
    let stage = Arc::new(AtomicUsize::new(0));
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();

    let first_coordinator = coordinator.clone();
    let first_job = job.clone();
    let first_gate = gate.clone();
    let first_release = release.clone();
    let first_stage = stage.clone();
    let first_contract = realtime_test_contract(&coordinator, BackendKind::Cuda);
    let first = tokio::spawn(async move {
        run_realtime_blocking_operation(
            &first_coordinator,
            &first_job,
            first_contract,
            first_gate,
            "asr.realtime.test",
            move || {
                Ok(move || {
                    assert_eq!(first_stage.fetch_add(1, Ordering::SeqCst), 0);
                    let _ = started_tx.send(());
                    let (lock, wake) = &*first_release;
                    let mut released = lock.lock().unwrap_or_else(|poison| poison.into_inner());
                    while !*released {
                        released = wake
                            .wait(released)
                            .unwrap_or_else(|poison| poison.into_inner());
                    }
                    assert_eq!(first_stage.fetch_add(1, Ordering::SeqCst), 1);
                    Ok(())
                })
            },
        )
        .await
    });
    started_rx.await.unwrap();
    first.abort();
    assert!(first.await.unwrap_err().is_cancelled());

    let second_coordinator = coordinator.clone();
    let second_job = job.clone();
    let second_gate = gate.clone();
    let second_stage = stage.clone();
    let second_contract = realtime_test_contract(&coordinator, BackendKind::Cuda);
    let second = tokio::spawn(async move {
        run_realtime_blocking_operation(
            &second_coordinator,
            &second_job,
            second_contract,
            second_gate,
            "asr.realtime.test",
            move || {
                Ok(move || {
                    assert_eq!(second_stage.fetch_add(1, Ordering::SeqCst), 2);
                    Ok(())
                })
            },
        )
        .await
    });

    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    assert_eq!(stage.load(Ordering::SeqCst), 1);
    {
        let (lock, wake) = &*release;
        *lock.lock().unwrap_or_else(|poison| poison.into_inner()) = true;
        wake.notify_all();
    }
    tokio::time::timeout(std::time::Duration::from_secs(1), second)
        .await
        .unwrap()
        .unwrap()
        .unwrap();

    assert_eq!(stage.load(Ordering::SeqCst), 3);
    assert_eq!(coordinator.snapshot().active_executions, 0);
    drop(job);
    assert_eq!(coordinator.snapshot().active_jobs, 0);
}

#[test]
fn granite_auto_asr_budget_follows_the_ibm_reference_formula() {
    assert_eq!(granite_auto_asr_max_tokens_for_duration(0.0), 200);
    assert_eq!(granite_auto_asr_max_tokens_for_duration(3.0), 200);
    assert_eq!(granite_auto_asr_max_tokens_for_duration(27.303175), 200);
    assert_eq!(granite_auto_asr_max_tokens_for_duration(30.0), 200);
    assert_eq!(granite_auto_asr_max_tokens_for_duration(31.0), 207);
    assert_eq!(granite_auto_asr_max_tokens_for_duration(60.0), 400);
    assert_eq!(granite_auto_asr_max_tokens_for_duration(540.0), 3_600);
    assert_eq!(granite_auto_asr_max_tokens_for_duration(f32::NAN), 200);
}

#[test]
fn granite_planned_asr_budget_honours_the_request_ceiling() {
    let granite = ModelVariant::GraniteSpeech412BPlus;
    let mut auto = EngineCoreRequest::asr_bytes(vec![1]).with_model_variant(granite);
    apply_asr_max_tokens(&mut auto, granite, None);
    assert_eq!(granite_asr_planned_max_tokens(&auto, 3.0), 200);
    assert_eq!(granite_asr_planned_max_tokens(&auto, 60.0), 400);
    assert_eq!(
        granite_asr_planned_max_tokens(&auto, 540.0),
        GRANITE_ASR_AUTO_MAX_TOKENS
    );

    let mut explicit = EngineCoreRequest::asr_bytes(vec![1]).with_model_variant(granite);
    apply_asr_max_tokens(&mut explicit, granite, Some(96));
    assert_eq!(granite_asr_planned_max_tokens(&explicit, 60.0), 96);
}

#[test]
fn granite_timestamp_budget_covers_tagged_words_and_a_reserve() {
    assert_eq!(granite_timestamp_max_tokens_for_duration(0.0), 64);
    assert_eq!(granite_timestamp_max_tokens_for_duration(1.0), 90);
    assert_eq!(granite_timestamp_max_tokens_for_duration(30.0), 844);
    assert_eq!(granite_timestamp_max_tokens_for_duration(60.0), 1_624);
    assert_eq!(granite_timestamp_max_tokens_for_duration(f32::NAN), 64);
}

/// Only plus runs the timestamp task, so only plus gets its budget; a
/// base request asking for word timestamps keeps the ASR budget and is
/// timed by the aligner.
#[test]
fn granite_planned_budget_follows_the_task_the_request_runs() {
    let auto = |variant, word_timestamps| {
        let mut request = EngineCoreRequest::asr_bytes(vec![1])
            .with_model_variant(variant)
            .with_asr_native_word_timestamps(word_timestamps);
        apply_asr_max_tokens(&mut request, variant, None);
        request
    };
    let plus = ModelVariant::GraniteSpeech412BPlus;
    let base = ModelVariant::GraniteSpeech412B;
    assert_eq!(granite_asr_planned_max_tokens(&auto(plus, true), 30.0), 844);
    assert_eq!(
        granite_asr_planned_max_tokens(&auto(plus, true), 60.0),
        1_624
    );
    assert_eq!(
        granite_asr_planned_max_tokens(&auto(plus, true), 210.0),
        GRANITE_ASR_AUTO_MAX_TOKENS
    );
    assert_eq!(
        granite_asr_planned_max_tokens(&auto(plus, false), 30.0),
        200
    );
    assert_eq!(granite_asr_planned_max_tokens(&auto(base, true), 30.0), 200);

    let mut explicit = EngineCoreRequest::asr_bytes(vec![1])
        .with_model_variant(plus)
        .with_asr_native_word_timestamps(true);
    apply_asr_max_tokens(&mut explicit, plus, Some(300));
    assert_eq!(granite_asr_planned_max_tokens(&explicit, 60.0), 300);
}

#[test]
fn invocation_decode_room_matches_the_planner_demand_model() {
    // 30 s at 10 tokens/s: the 128-token text reserve and 300 audio tokens
    // leave 3,668 of 4,096 for decoding; a small context leaves none.
    assert_eq!(invocation_decode_room(4_096, 10.0, 30.0), 3_668);
    assert_eq!(invocation_decode_room(400, 10.0, 30.0), 0);

    // At the solved single-pass limit Granite's budget still fits the room.
    let InvocationAudioLimit::Limited(limit) = solve_invocation_audio_limit(
        4_096,
        10.0,
        ASR_TEXT_PROMPT_TOKEN_RESERVE,
        granite_auto_asr_max_tokens_for_duration,
    ) else {
        panic!("a 4096-token context must admit Granite audio");
    };
    assert!((237.0..=239.0).contains(&limit), "unexpected limit {limit}");
    assert!(
        granite_auto_asr_max_tokens_for_duration(limit)
            <= invocation_decode_room(4_096, 10.0, limit)
    );
}

#[test]
fn granite_auto_asr_request_uses_executor_adjusted_ceiling() {
    assert_eq!(
        granite_auto_asr_token_ceiling(ModelVariant::GraniteSpeech412BPlus),
        Some(GRANITE_ASR_AUTO_MAX_TOKENS)
    );
    assert_eq!(
        granite_auto_asr_token_ceiling(ModelVariant::WhisperLargeV3Turbo),
        None
    );
}

#[test]
fn single_window_and_chunked_transcripts_get_the_same_cleanup() {
    let raw = "  Total:   1000000000 units.............\n\nNext  line  ";
    let mut assembler =
        izwi_asr_toolkit::TranscriptAssembler::new(AsrLongFormConfig::default());
    assembler.push_chunk_text(raw);
    let chunked = assembler.finish();

    assert_eq!(
        asr_transcript_text(raw),
        "Total: 1000000000 units........\nNext line"
    );
    assert_eq!(asr_transcript_text(raw), chunked);
    assert_eq!(asr_transcript_text(&chunked), chunked);
}

#[test]
fn asr_response_language_prefers_the_hint_then_the_detected_language() {
    let detected = serde_json::json!({ "language": "de" });
    assert_eq!(
        asr_response_language(Some("en"), None, Some(&detected)).as_deref(),
        Some("en")
    );
    assert_eq!(
        asr_response_language(None, None, Some(&detected)).as_deref(),
        Some("de")
    );
    assert_eq!(asr_response_language(None, None, None), None);
    let undetected = serde_json::json!({ "language": null });
    assert_eq!(asr_response_language(None, None, Some(&undetected)), None);
    // A translation is in its target language.
    assert_eq!(
        asr_response_language(Some("fr"), Some("English"), Some(&detected)).as_deref(),
        Some("English")
    );
}

#[test]
fn only_granite_speech_5_takes_a_translation_target_or_keywords() {
    let granite5 = ModelVariant::GraniteSpeech502B;
    let translate = AsrTaskOptions::new(Some("English".to_string()), Vec::new());
    let keywords = AsrTaskOptions::new(None, vec!["Izwi".to_string()]);

    validate_asr_task(granite5, Some("fr"), None, &translate).unwrap();
    validate_asr_task(granite5, None, None, &keywords).unwrap();
    validate_asr_task(granite5, None, None, &AsrTaskOptions::default()).unwrap();
    for variant in [
        ModelVariant::WhisperLargeV3Turbo,
        ModelVariant::GraniteSpeech412B,
    ] {
        validate_asr_task(variant, None, None, &AsrTaskOptions::default()).unwrap();
        assert!(matches!(
            validate_asr_task(variant, None, None, &translate),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            validate_asr_task(variant, None, None, &keywords),
            Err(Error::InvalidInput(_))
        ));
    }
    // 5.0's own limits: an unsupported pair and an oversized keyword.
    assert!(matches!(
        validate_asr_task(
            granite5,
            Some("ja"),
            None,
            &AsrTaskOptions::new(Some("German".to_string()), Vec::new()),
        ),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        validate_asr_task(
            granite5,
            None,
            None,
            &AsrTaskOptions::new(None, vec!["x".repeat(129)]),
        ),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn vibevoice_omitted_budget_is_admitted_at_the_ceiling_and_planned_by_duration() {
    let vibevoice = ModelVariant::VibeVoiceAsr;
    let mut auto = EngineCoreRequest::asr_bytes(vec![1]).with_model_variant(vibevoice);
    apply_asr_max_tokens(&mut auto, vibevoice, None);
    assert!(auto.asr_auto_max_tokens);
    assert_eq!(auto.params.max_tokens, 32_768);
    assert_eq!(vibevoice_asr_planned_max_tokens(&auto, 10.0), 512);
    assert_eq!(vibevoice_asr_planned_max_tokens(&auto, 60.0), 976);
    assert_eq!(vibevoice_asr_planned_max_tokens(&auto, 600.0), 7_456);
    assert_eq!(vibevoice_asr_planned_max_tokens(&auto, 3_600.0), 32_768);

    let mut explicit = EngineCoreRequest::asr_bytes(vec![1]).with_model_variant(vibevoice);
    apply_asr_max_tokens(&mut explicit, vibevoice, Some(200));
    assert!(!explicit.asr_auto_max_tokens);
    assert_eq!(vibevoice_asr_planned_max_tokens(&explicit, 600.0), 200);

    // The duration budget never exceeds what admission priced.
    auto.params.max_tokens = 700;
    assert_eq!(vibevoice_asr_planned_max_tokens(&auto, 60.0), 700);
}

#[test]
fn asr_transcription_duration_prefers_prepared_input_over_executor_audio() {
    let mut output = crate::engine::OutputProcessor::new(16_000).process(
        crate::engine::ExecutorOutput {
            audio: Some(crate::engine::AudioOutput::new(Vec::new(), 16_000)),
            ..crate::engine::ExecutorOutput::terminal("asr-duration".to_string())
        },
        1,
        Duration::from_millis(300),
    );
    assert_eq!(output.audio.duration_secs, 0.0);
    output.input_audio_duration_secs = Some(1.5);
    assert_eq!(asr_transcription_duration_secs(&output), 1.5);

    // Families without preparation report the decoded input themselves.
    output.input_audio_duration_secs = None;
    output.audio.duration_secs = 2.0;
    assert_eq!(asr_transcription_duration_secs(&output), 2.0);
}

#[test]
fn omitted_asr_max_tokens_is_automatic_and_explicit_budget_is_kept() {
    let default_max_tokens = EngineCoreRequest::asr_bytes(vec![1]).params.max_tokens;
    let lfm = ModelVariant::Lfm25Audio15BGguf;

    let mut omitted = EngineCoreRequest::asr_bytes(vec![1]).with_model_variant(lfm);
    apply_asr_max_tokens(&mut omitted, lfm, None);
    assert!(omitted.asr_auto_max_tokens);
    assert_eq!(omitted.params.max_tokens, default_max_tokens);

    let mut explicit = EngineCoreRequest::asr_bytes(vec![1]).with_model_variant(lfm);
    apply_asr_max_tokens(&mut explicit, lfm, Some(96));
    assert!(!explicit.asr_auto_max_tokens);
    assert_eq!(explicit.params.max_tokens, 96);

    let granite = ModelVariant::GraniteSpeech412BPlus;
    let mut granite_auto = EngineCoreRequest::asr_bytes(vec![1]).with_model_variant(granite);
    apply_asr_max_tokens(&mut granite_auto, granite, None);
    assert!(granite_auto.asr_auto_max_tokens);
    assert_eq!(granite_auto.params.max_tokens, GRANITE_ASR_AUTO_MAX_TOKENS);
}

#[test]
fn speaker_attributed_asr_result_preserves_granite_turns() {
    let result = speaker_attributed_asr_result_from_text(
        "[Speaker 1]: hello there [Speaker 2]: hi back",
        Some("English".to_string()),
        4.0,
        Some(2),
        None,
    );

    assert_eq!(result.status, SpeakerAttributedAsrStatus::Ready);
    assert_eq!(result.language.as_deref(), Some("English"));
    assert_eq!(result.speaker_count, 2);
    assert!(result.warnings.is_empty());
    assert_eq!(
        result.speaker_turns,
        vec![
            SpeakerAttributedAsrTurn {
                speaker: "Speaker 1".to_string(),
                text: "hello there".to_string(),
                start_secs: None,
                end_secs: None,
            },
            SpeakerAttributedAsrTurn {
                speaker: "Speaker 2".to_string(),
                text: "hi back".to_string(),
                start_secs: None,
                end_secs: None,
            },
        ]
    );
}

#[test]
fn speaker_attributed_asr_warns_when_expected_speakers_are_missing() {
    let result = speaker_attributed_asr_result_from_text(
        "[Speaker 1]: one long turn",
        None,
        2.0,
        Some(2),
        None,
    );

    assert_eq!(result.status, SpeakerAttributedAsrStatus::Warning);
    assert_eq!(result.speaker_count, 1);
    assert_eq!(result.speaker_turns.len(), 1);
    assert!(result.warnings[0].contains("below requested minimum 2"));
}

#[test]
fn granite_saa_chunk_plan_keeps_long_audio_under_model_limit() {
    let sample_rate = 10;
    let samples = vec![0.0f32; 15_565];
    let chunks = granite_saa_chunk_plan(&samples, sample_rate, Some(540.0));

    assert!(chunks.len() > 1);
    assert_eq!(chunks.first().unwrap().start_sample, 0);
    assert_eq!(chunks.last().unwrap().end_sample, samples.len());
    assert!(chunks.iter().all(|chunk| chunk.len_samples()
        <= (GRANITE_SAA_HARD_MAX_CHUNK_SECS * sample_rate as f32) as usize));
}

#[test]
fn granite_saa_decode_budget_uses_saa_sized_limit() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    let previous = std::env::var("IZWI_GRANITE_SAA_MAX_NEW_TOKENS").ok();
    std::env::remove_var("IZWI_GRANITE_SAA_MAX_NEW_TOKENS");

    assert_eq!(
        granite_saa_max_new_tokens_for_duration(27.0),
        GRANITE_SAA_MIN_NEW_TOKENS
    );
    assert_eq!(granite_saa_max_new_tokens_for_duration(240.0), 2176);
    assert_eq!(
        granite_saa_max_new_tokens_for_duration(510.0),
        GRANITE_SAA_MAX_NEW_TOKENS
    );
    assert!(granite_saa_max_new_tokens_for_duration(510.0) < 10_000);

    if let Some(previous) = previous {
        std::env::set_var("IZWI_GRANITE_SAA_MAX_NEW_TOKENS", previous);
    } else {
        std::env::remove_var("IZWI_GRANITE_SAA_MAX_NEW_TOKENS");
    }
}

#[test]
fn granite_saa_decode_budget_allows_explicit_override() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    let previous = std::env::var("IZWI_GRANITE_SAA_MAX_NEW_TOKENS").ok();
    std::env::set_var("IZWI_GRANITE_SAA_MAX_NEW_TOKENS", "1234");

    assert_eq!(granite_saa_max_new_tokens_for_duration(240.0), 1234);

    if let Some(previous) = previous {
        std::env::set_var("IZWI_GRANITE_SAA_MAX_NEW_TOKENS", previous);
    } else {
        std::env::remove_var("IZWI_GRANITE_SAA_MAX_NEW_TOKENS");
    }
}

#[test]
fn granite_saa_assembler_dedupes_overlap_and_preserves_turns() {
    let mut assembler = GraniteSaaTranscriptAssembler::default();
    assert!(assembler
        .push_chunk_text(
            "[Speaker 1]: hello there [Speaker 2]: hi back from me now",
            0,
        )
        .is_empty());
    assert!(assembler
        .push_chunk_text(
            "[Speaker 2]: hi back from me now and more [Speaker 1]: ok",
            1,
        )
        .is_empty());

    let text = assembler.text();
    assert_eq!(text.matches("hi back from me now").count(), 1);
    assert_eq!(
        text,
        "[Speaker 1]: hello there [Speaker 2]: hi back from me now and more [Speaker 1]: ok"
    );
}

#[test]
fn granite_saa_assembler_maps_reset_label_when_overlap_proves_continuity() {
    let mut assembler = GraniteSaaTranscriptAssembler::default();
    assembler.push_chunk_text(
        "[Speaker 1]: first person [Speaker 2]: boundary overlap words now",
        0,
    );
    let warnings =
        assembler.push_chunk_text("[Speaker 1]: boundary overlap words now continuing", 1);

    assert!(warnings.iter().any(|warning| warning.contains("mapped it")));
    assert_eq!(
        assembler.text(),
        "[Speaker 1]: first person [Speaker 2]: boundary overlap words now continuing"
    );
}

#[test]
fn granite_saa_prefix_text_is_bounded_on_turn_boundaries() {
    let mut assembler = GraniteSaaTranscriptAssembler::default();
    for idx in 0..32 {
        let speaker = if idx % 2 == 0 { 1 } else { 2 };
        assembler.push_chunk_text(format!("[Speaker {speaker}]: turn {idx}").as_str(), idx);
    }

    let prefix = assembler.prefix_text();
    assert!(prefix.chars().count() <= GRANITE_SAA_PREFIX_MAX_CHARS);
    assert!(prefix.contains("[Speaker 1]:"));
    assert!(!prefix.contains("turn 0"));
}

#[test]
fn granite_saa_chunk_prefix_defaults_to_none() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    let previous = std::env::var("IZWI_GRANITE_SAA_PREFIX_MODE").ok();
    std::env::remove_var("IZWI_GRANITE_SAA_PREFIX_MODE");
    let mut assembler = GraniteSaaTranscriptAssembler::default();
    assembler.push_chunk_text("[Speaker 1]: prior audio text", 0);

    assert_eq!(GraniteSaaPrefixMode::from_env(), GraniteSaaPrefixMode::None);
    assert_eq!(
        granite_saa_chunk_prefix_text(&assembler, GraniteSaaPrefixMode::from_env()),
        None
    );

    if let Some(previous) = previous {
        std::env::set_var("IZWI_GRANITE_SAA_PREFIX_MODE", previous);
    } else {
        std::env::remove_var("IZWI_GRANITE_SAA_PREFIX_MODE");
    }
}

#[test]
fn granite_saa_chunk_prefix_full_mode_is_explicit_and_bounded() {
    let _guard = env_lock().lock().expect("env lock poisoned");
    let previous = std::env::var("IZWI_GRANITE_SAA_PREFIX_MODE").ok();
    std::env::set_var("IZWI_GRANITE_SAA_PREFIX_MODE", "full");
    let mut assembler = GraniteSaaTranscriptAssembler::default();
    for idx in 0..32 {
        assembler.push_chunk_text(format!("[Speaker 1]: turn {idx}").as_str(), idx);
    }

    let prefix = granite_saa_chunk_prefix_text(&assembler, GraniteSaaPrefixMode::from_env())
        .expect("full prefix");
    assert_eq!(
        GraniteSaaPrefixMode::from_env(),
        GraniteSaaPrefixMode::FullTranscript
    );
    assert!(prefix.chars().count() <= GRANITE_SAA_PREFIX_MAX_CHARS);
    assert!(prefix.contains("[Speaker 1]:"));

    if let Some(previous) = previous {
        std::env::set_var("IZWI_GRANITE_SAA_PREFIX_MODE", previous);
    } else {
        std::env::remove_var("IZWI_GRANITE_SAA_PREFIX_MODE");
    }
}

#[test]
fn granite_saa_decode_diagnostics_warn_when_chunk_hits_token_limit() {
    let diagnostics = serde_json::json!({
        "decode": {
            "generated_tokens": 2500,
            "max_new_tokens": 2500,
            "stop_reason": "max_tokens"
        },
        "prompt": {
            "prompt_tokens": 512,
            "prefix_tokens": 0
        }
    });

    let decoded = granite_saa_decode_diagnostics(Some(&diagnostics));
    assert_eq!(
        decoded,
        GraniteSaaChunkDecodeDiagnostics {
            generated_tokens: Some(2500),
            max_new_tokens: Some(2500),
            stop_reason: Some("max_tokens".to_string()),
            prompt_tokens: Some(512),
            prompt_prefix_tokens: Some(0),
        }
    );

    let warning = granite_saa_max_token_warning(2, &decoded).expect("max token limit warning");
    assert!(warning.contains("chunk 3"));
    assert!(warning.contains("2500/2500"));
}

#[test]
fn granite_saa_progress_events_report_chunks_and_percent() {
    let chunks = vec![
        AudioChunk {
            start_sample: 0,
            end_sample: 100,
        },
        AudioChunk {
            start_sample: 90,
            end_sample: 200,
        },
    ];

    let processing = granite_saa_processing_progress(&chunks, 100);
    assert_eq!(processing.phase, AsrProgressPhase::Processing);
    assert_eq!(processing.total_chunks, Some(2));
    assert_eq!(processing.percent, Some(0.0));

    let finished = granite_saa_chunk_progress(
        AsrProgressPhase::ChunkFinished,
        0,
        &chunks[0],
        &chunks,
        100,
    );
    assert_eq!(finished.current_chunk, Some(1));
    assert_eq!(finished.total_chunks, Some(2));
    assert_eq!(finished.processed_audio_secs, Some(1.0));
    assert_eq!(finished.total_audio_secs, Some(2.0));
    assert_eq!(finished.percent, Some(50.0));
}

#[test]
fn speaker_attributed_asr_result_includes_long_form_warnings() {
    let result = speaker_attributed_asr_result_from_text_with_warnings(
        "[Speaker 1]: hello [Speaker 2]: hi",
        None,
        2.0,
        None,
        None,
        vec!["Granite SAA processed long audio in 2 chunks.".to_string()],
    );

    assert_eq!(result.status, SpeakerAttributedAsrStatus::Warning);
    assert_eq!(result.speaker_count, 2);
    assert_eq!(result.warnings.len(), 1);
    assert!(result.warnings[0].contains("2 chunks"));
}

#[test]
fn speaker_attributed_asr_rejects_non_granite_models() {
    let err = resolve_speaker_attributed_asr_variant(Some("Whisper-Large-v3-Turbo"))
        .expect_err("SAA should be Granite-only");
    assert!(err.to_string().contains("Granite-Speech-4.1-2B-Plus"));
}

#[test]
fn offline_asr_observations_include_owned_and_decoded_storage() {
    let input = [1_u8, 2, 3, 4];
    let owned = OwnedAsrAudioInput::Bytes(input.to_vec());
    let owned_bytes = owned.retained_bytes();
    assert!(owned_bytes >= input.len());

    let copied = retained_host_observation(&[input.len(), owned_bytes]).unwrap();
    assert_eq!(copied.host_bytes, (input.len() + owned_bytes) as u64);

    let decoded = decoded_audio_with_scratch_observation(input.len(), 8, 4).unwrap();
    assert_eq!(
        decoded.host_bytes,
        (input.len() + 12 * std::mem::size_of::<f32>()) as u64
    );
}

#[test]
fn qwen_asr_build_route_accounts_retained_prepared_audio_once() {
    let variant = ModelVariant::Qwen3Asr06BGguf;
    let mut request =
        EngineCoreRequest::asr_bytes(vec![1, 2, 3, 4]).with_model_variant(variant);
    let source_only = retained_engine_request_input_bytes(&request).unwrap();
    request
        .install_prepared_asr_audio(variant, vec![0.0; 513], 16_000)
        .unwrap();
    request
        .install_prepared_sequence_input_tokens(48, 4096)
        .unwrap();
    let retained = retained_engine_request_input_bytes(&request).unwrap();
    assert_eq!(retained - source_only, 513 * std::mem::size_of::<f32>());
    assert_eq!(
        retained_host_observation(&[retained]).unwrap().host_bytes,
        retained as u64
    );
}

#[tokio::test(flavor = "current_thread")]
async fn cancelled_offline_asr_stage_retains_job_and_execution_until_physical_exit() {
    let coordinator = Arc::new(InferenceCoordinator::new(BackendKind::Cpu, 1, 4));
    let job = coordinator
        .admit(realtime_test_job(
            "offline-asr-cancel",
            BackendKind::Cpu,
            None,
        ))
        .await
        .unwrap();
    let release = Arc::new((Mutex::new(false), Condvar::new()));
    let task_release = release.clone();
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let task_coordinator = coordinator.clone();
    let task_job = job.clone();
    drop(job);
    let runner = tokio::spawn(async move {
        task_coordinator
            .run_blocking_stage(&task_job, move || {
                let _ = started_tx.send(());
                let (lock, wake) = &*task_release;
                let mut released = lock.lock().unwrap_or_else(|poison| poison.into_inner());
                while !*released {
                    released = wake
                        .wait(released)
                        .unwrap_or_else(|poison| poison.into_inner());
                }
                Ok(())
            })
            .await
    });

    started_rx.await.unwrap();
    runner.abort();
    assert!(runner.await.unwrap_err().is_cancelled());
    assert_eq!(coordinator.snapshot().active_jobs, 1);
    assert_eq!(coordinator.snapshot().active_executions, 1);

    {
        let (lock, wake) = &*release;
        *lock.lock().unwrap_or_else(|poison| poison.into_inner()) = true;
        wake.notify_all();
    }
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while coordinator.snapshot().active_jobs != 0
            || coordinator.snapshot().active_executions != 0
        {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
// The guard intentionally serializes process-global environment access for
// the complete asynchronous load attempt in this test.
#[allow(clippy::await_holding_lock)]
async fn parakeet_load_rejects_invalid_nemo_archive() {
    let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let _guard = env_lock().lock().expect("env lock poisoned");

    let root = std::env::temp_dir().join(format!("izwi-parakeet-runtime-{}", Uuid::new_v4()));
    let model_dir = root.join("Parakeet-TDT-0.6B-v3");
    std::fs::create_dir_all(&model_dir).unwrap();
    std::fs::write(model_dir.join("parakeet-tdt-0.6b-v3.nemo"), b"mock-nemo").unwrap();

    let config = EngineConfig {
        models_dir: root.clone(),
        backend: BackendPreference::Cpu,
        ..Default::default()
    };

    let engine = RuntimeService::new(config).unwrap();
    let err = engine
        .load_model(ModelVariant::ParakeetTdt06BV3)
        .await
        .expect_err("invalid .nemo archive should fail to load");
    let msg = err.to_string();
    assert!(
        msg.contains(".nemo")
            || msg.contains("archive")
            || msg.contains("Failed to load")
            || msg.contains("invalid")
    );

    let _ = std::fs::remove_dir_all(root);
}
