//! Golden wire tests for `/v1/voice/realtime/ws`: the exact messages the
//! legacy (`voice_realtime_v1`) and typed (`voice_realtime` v2) protocols
//! send for scripted sessions. They were recorded before the voice core was
//! made protocol-neutral for the OpenAI Realtime front-end, and pin both
//! protocols byte for byte across that refactor.
//!
//! Volatile values (session and instance ids, wall-clock times, the test's
//! temporary directory) are replaced before comparing; everything else is
//! compared as sent. Set `IZWI_UPDATE_VOICE_GOLDEN=1` to rewrite the files,
//! which should only happen for an intended protocol change.

use super::tests::{fake_synthesis, test_app_state};
use super::*;
use crate::app::realtime_transport::MailboxOutput;
use serde_json::Value;

const GOLDEN_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/voice_realtime");
/// Keys whose values differ between runs.
const VOLATILE_KEYS: &[&str] = &[
    "session_id",
    "owner_instance_id",
    "server_time_ms",
    "timestamp_ms",
    "elapsed_ms",
    "server_timestamp_ms",
];

fn check_golden(name: &str, lines: &[String]) {
    let path = format!("{GOLDEN_DIR}/{name}.txt");
    let actual = format!("{}\n", lines.join("\n"));
    if std::env::var_os("IZWI_UPDATE_VOICE_GOLDEN").is_some() {
        std::fs::create_dir_all(GOLDEN_DIR).expect("golden dir");
        std::fs::write(&path, &actual).expect("write golden");
        return;
    }
    let expected = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("{path}: {err} (record with IZWI_UPDATE_VOICE_GOLDEN=1)"));
    if actual != expected {
        let first = actual
            .lines()
            .zip(expected.lines())
            .position(|(a, e)| a != e)
            .unwrap_or_else(|| actual.lines().count().min(expected.lines().count()));
        panic!(
            "golden `{name}` changed at line {}:\n  actual:   {}\n  expected: {}",
            first + 1,
            actual.lines().nth(first).unwrap_or("<end>"),
            expected.lines().nth(first).unwrap_or("<end>"),
        );
    }
}

fn normalize(value: &mut Value, tmp: &str) -> bool {
    let mut changed = false;
    match value {
        Value::Object(map) => {
            for (key, entry) in map.iter_mut() {
                if VOLATILE_KEYS.contains(&key.as_str()) && !entry.is_null() {
                    *entry = Value::String("<volatile>".to_string());
                    changed = true;
                } else {
                    changed |= normalize(entry, tmp);
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                changed |= normalize(item, tmp);
            }
        }
        Value::String(text) if !tmp.is_empty() && text.contains(tmp) => {
            *text = text.replace(tmp, "<tmp>");
            changed = true;
        }
        _ => {}
    }
    changed
}

/// FNV-1a, enough to pin a payload's bytes in a readable golden line.
fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Every message queued on `out_tx`, one golden line each; the mailbox is
/// closed afterwards.
async fn drain_lines(out_tx: &OutboundTx, tmp: &str) -> Vec<String> {
    out_tx.close();
    let mut lines = Vec::new();
    while let Some(output) = out_tx.mailbox.next().await {
        lines.push(golden_line(output, tmp));
    }
    lines
}

/// The messages queued on `out_tx` so far, leaving the mailbox open.
async fn take_lines(out_tx: &OutboundTx, tmp: &str) -> Vec<String> {
    let mut lines = Vec::new();
    while let Ok(Some(output)) =
        tokio::time::timeout(Duration::from_millis(50), out_tx.mailbox.next()).await
    {
        lines.push(golden_line(output, tmp));
    }
    lines
}

fn golden_line(output: MailboxOutput, tmp: &str) -> String {
    {
        match output {
            MailboxOutput::Message(Message::Text(text)) => {
                let mut value: Value = serde_json::from_str(text.as_str()).expect("json event");
                if normalize(&mut value, tmp) {
                    format!("norm {value}")
                } else {
                    format!("text {}", text.as_str())
                }
            }
            MailboxOutput::Message(Message::Binary(frame)) => {
                let head = frame.len().min(WS_BIN_ASSISTANT_HEADER_LEN);
                format!(
                    "bin len={} head={} payload={:016x}",
                    frame.len(),
                    hex(&frame[..head]),
                    fnv1a(&frame[head..])
                )
            }
            MailboxOutput::Message(other) => format!("other {other:?}"),
            MailboxOutput::Close(reason) => format!("close {reason}"),
        }
    }
}

/// A client PCM16 frame carrying `samples` at `sample_rate`.
fn client_frame(frame_seq: u32, sample_rate: u32, samples: &[i16]) -> Vec<u8> {
    let mut frame = Vec::with_capacity(16 + samples.len() * 2);
    frame.extend_from_slice(WS_BIN_MAGIC);
    frame.push(crate::app::realtime_transport::WS_BIN_VERSION);
    frame.push(1);
    frame.extend_from_slice(&[0, 0]);
    frame.extend_from_slice(&sample_rate.to_le_bytes());
    frame.extend_from_slice(&frame_seq.to_le_bytes());
    for sample in samples {
        frame.extend_from_slice(&sample.to_le_bytes());
    }
    frame
}

fn tone(len: usize, sample_rate: u32, offset: usize) -> Vec<i16> {
    (0..len)
        .map(|index| {
            let t = (offset + index) as f32 / sample_rate as f32;
            ((t * 220.0 * std::f32::consts::TAU).sin() * 8_000.0) as i16
        })
        .collect()
}

async fn send_text(
    state: &AppState,
    out_tx: &OutboundTx,
    conn: &mut ConnectionState,
    value: Value,
) {
    if let Err(err) = handle_text_message(state, "golden", out_tx, conn, &value.to_string()).await {
        send_error(out_tx, None, None, err);
    }
}

/// A scripted session through the receive-side handlers: start, an input
/// stream whose VAD calls every frame speech, 600 ms of tone, a stop that
/// ends the utterance (its turn then fails, as no model is installed), and
/// the control events with no turn behind them.
async fn scripted_session(protocol: Option<(&str, u16)>) -> Vec<String> {
    let (dir, state) = test_app_state(4);
    let tmp = dir.path().to_string_lossy().to_string();
    let out_tx = OutboundTx::new(state.runtime.clone(), "voice golden");
    let mut conn = ConnectionState::default();

    let mut start = json!({"type": "session_start", "system_prompt": "Be brief."});
    if let Some((name, version)) = protocol {
        start["protocol"] = json!(name);
        start["version"] = json!(version);
    }
    send_text(&state, &out_tx, &mut conn, start).await;
    send_text(
        &state,
        &out_tx,
        &mut conn,
        json!({
            "type": "input_stream_start",
            "mode": "modular",
            "asr_model_id": "Parakeet-TDT-0.6B-v3",
            "text_model_id": "Qwen3.5-0.8B",
            "tts_model_id": "Kokoro-82M",
            "vad_threshold": 0.01,
            "vad_snr_margin_db": 0.0,
            "min_speech_ms": 50,
            "silence_duration_ms": 400,
            "input_sample_rate": 16000,
        }),
    )
    .await;
    if let Some(warmup) = conn.model_warmup.take() {
        warmup.join().await.expect("warmup");
    }

    // 600 ms of tone in 20 ms frames, with one frame missing (a gap).
    for frame in 0..30u32 {
        if frame == 12 {
            continue;
        }
        let samples = tone(320, 16_000, frame as usize * 320);
        if let Err(err) = handle_binary_message(
            &state,
            "golden",
            &out_tx,
            &mut conn,
            client_frame(frame + 1, 16_000, &samples),
        )
        .await
        {
            send_error(&out_tx, None, None, err);
        }
    }
    send_text(
        &state,
        &out_tx,
        &mut conn,
        json!({"type": "input_stream_stop"}),
    )
    .await;

    // The turn fails at its first stage; wait for it to finish, and take its
    // events before an interrupt can cut them off.
    let deadline = Instant::now() + Duration::from_secs(20);
    while conn.has_running_turn() {
        assert!(Instant::now() < deadline, "the scripted turn finishes");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let mut lines = take_lines(&out_tx, &tmp).await;

    send_text(
        &state,
        &out_tx,
        &mut conn,
        json!({"type": "conversation_item_truncate", "utterance_seq": 99, "audio_end_ms": 10}),
    )
    .await;
    send_text(
        &state,
        &out_tx,
        &mut conn,
        json!({"type": "conversation.item.truncate", "utterance_seq": 1, "audio_end_ms": 10}),
    )
    .await;
    send_text(&state, &out_tx, &mut conn, json!({"type": "interrupt"})).await;
    send_text(
        &state,
        &out_tx,
        &mut conn,
        json!({"type": "function_call_output", "call_id": "call_x", "output": "{}"}),
    )
    .await;
    send_text(
        &state,
        &out_tx,
        &mut conn,
        json!({"type": "ping", "timestamp_ms": 5}),
    )
    .await;
    send_text(&state, &out_tx, &mut conn, json!({"type": "no_such_event"})).await;
    lines.extend(drain_lines(&out_tx, &tmp).await);
    lines
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn golden_legacy_scripted_session() {
    let lines = scripted_session(None).await;
    check_golden("legacy_scripted_session", &lines);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn golden_typed_scripted_session() {
    let lines = scripted_session(Some(("voice_realtime", VOICE_REALTIME_VERSION))).await;
    check_golden("typed_scripted_session", &lines);
}

fn modular_commit(format: OutputAudioFormat) -> PendingAudioCommit {
    PendingAudioCommit {
        utterance_id: "utt-1".to_string(),
        utterance_seq: 1,
        turn_config: VoiceTurnConfig::Modular(ModularVoiceTurnConfig {
            asr_model_id: "Parakeet-TDT-0.6B-v3".to_string(),
            text_model_id: "Qwen3.5-0.8B".to_string(),
            tts_model_id: "Kokoro-82M".to_string(),
            speaker: None,
            asr_language: None,
            max_output_tokens: 64,
            output_format: format,
        }),
        endpointed_at: Instant::now(),
        endpointing_delay: None,
    }
}

/// A spoken reply streamed through `speak_reply` with the fake TTS.
async fn spoken_reply(format: OutputAudioFormat, typed: bool) -> Vec<String> {
    let (dir, state) = test_app_state(4);
    let tmp = dir.path().to_string_lossy().to_string();
    let out_tx = OutboundTx::new(state.runtime.clone(), "voice golden");
    if typed {
        out_tx.configure_wire_protocol(
            VoiceWireProtocol::TypedV2,
            "session".to_string(),
            "owner".to_string(),
        );
    }
    let control = TurnControl::new(Instant::now());
    let (deltas, delta_rx) = tokio::sync::mpsc::unbounded_channel();
    for delta in [
        "<think>plan</think>Hello there, it is lovely to hear from you.",
        " The second sentence is longer",
        " than the first. Last one.",
    ] {
        deltas.send(delta.to_string()).expect("delta");
    }
    drop(deltas);
    speak_reply(
        &state,
        &mut VoiceSocketReplySink::new(&out_tx, &modular_commit(format), 1),
        &control,
        delta_rx,
        Instant::now(),
        fake_synthesis(Arc::default()),
    )
    .await
    .expect("spoken");
    drain_lines(&out_tx, &tmp).await
}

#[tokio::test]
async fn golden_legacy_spoken_reply_pcm() {
    check_golden(
        "legacy_spoken_reply_pcm",
        &spoken_reply(OutputAudioFormat::PcmI16, false).await,
    );
}

#[tokio::test]
async fn golden_legacy_spoken_reply_ulaw() {
    check_golden(
        "legacy_spoken_reply_ulaw",
        &spoken_reply(OutputAudioFormat::G711Ulaw, false).await,
    );
}

#[tokio::test]
async fn golden_typed_spoken_reply_pcm() {
    check_golden(
        "typed_spoken_reply_pcm",
        &spoken_reply(OutputAudioFormat::PcmI16, true).await,
    );
}

/// A unified reply's stream sink fed text and speech chunks, as LFM-Audio
/// streams them: text deltas, a tool-call round's markup held back, audio.
async fn unified_sink(typed: bool, tools: bool) -> Vec<String> {
    let (dir, state) = test_app_state(4);
    let tmp = dir.path().to_string_lossy().to_string();
    let out_tx = OutboundTx::new(state.runtime.clone(), "voice golden");
    if typed {
        out_tx.configure_wire_protocol(
            VoiceWireProtocol::TypedV2,
            "session".to_string(),
            "owner".to_string(),
        );
    }
    let control = TurnControl::new(Instant::now());
    let stream = Arc::new(Mutex::new(UnifiedReplyStream {
        parser: tools.then(|| ToolCallParser::new(ToolCallFormat::Lfm3Xml)),
        ..UnifiedReplyStream::default()
    }));
    let delivery_error = Arc::new(Mutex::new(None));
    let mut sink = unified_chunk_sink(
        &out_tx,
        "utt-1",
        1,
        OutputAudioFormat::PcmI16,
        &control,
        1,
        state.runtime.clone(),
        stream.clone(),
        delivery_error.clone(),
    );
    let text = |sequence: usize, text: &str| {
        let mut chunk =
            izwi_core::StreamingOutput::new("req".to_string(), sequence, Vec::new(), 24_000);
        chunk.text = Some(text.to_string());
        chunk
    };
    let chunks = vec![
        text(0, "Sure, "),
        text(1, "let me check."),
        izwi_core::StreamingOutput::new("req".to_string(), 2, vec![0.25; 480], 24_000),
        text(3, "<tool_call>\n<function=time>\n</function>\n</tool_call>"),
        izwi_core::StreamingOutput::final_chunk("req".to_string(), 4, vec![-0.5; 240], 24_000),
    ];
    for chunk in chunks {
        sink(chunk).await.expect("chunk");
    }
    assert!(delivery_error.lock().expect("error").is_none());
    drain_lines(&out_tx, &tmp).await
}

#[tokio::test]
async fn golden_legacy_unified_sink() {
    check_golden("legacy_unified_sink", &unified_sink(false, false).await);
}

#[tokio::test]
async fn golden_legacy_unified_sink_with_tools() {
    check_golden(
        "legacy_unified_sink_tools",
        &unified_sink(false, true).await,
    );
}

#[tokio::test]
async fn golden_typed_unified_sink() {
    check_golden("typed_unified_sink", &unified_sink(true, true).await);
}

/// Interrupting a running turn, and a finished one still playing, on both
/// protocols.
async fn interruptions(typed: bool) -> Vec<String> {
    let (dir, state) = test_app_state(1);
    let tmp = dir.path().to_string_lossy().to_string();
    let out_tx = OutboundTx::new(state.runtime.clone(), "voice golden");
    if typed {
        out_tx.configure_wire_protocol(
            VoiceWireProtocol::TypedV2,
            "session".to_string(),
            "owner".to_string(),
        );
    }
    let running = tokio::spawn(tokio::time::sleep(Duration::from_secs(30)));
    let mut active = Some(ActiveTurn {
        utterance_id: "utt-1".to_string(),
        utterance_seq: 1,
        turn_record_id: "turn-1".to_string(),
        state: state.clone(),
        conversation: Arc::default(),
        task: running,
        control: TurnControl::new(Instant::now()),
        output_generation: 1,
    });
    assert!(interrupt_active_turn(&out_tx, &mut active, "barge_in"));

    let finished = tokio::spawn(async {});
    while !finished.is_finished() {
        tokio::task::yield_now().await;
    }
    let mut active = Some(ActiveTurn {
        utterance_id: "utt-2".to_string(),
        utterance_seq: 2,
        turn_record_id: "turn-2".to_string(),
        state: state.clone(),
        conversation: Arc::default(),
        task: finished,
        control: TurnControl::new(Instant::now()),
        output_generation: 2,
    });
    assert!(!interrupt_active_turn(
        &out_tx,
        &mut active,
        "client_interrupt"
    ));
    drain_lines(&out_tx, &tmp).await
}

#[tokio::test]
async fn golden_legacy_interruptions() {
    check_golden("legacy_interruptions", &interruptions(false).await);
}

#[tokio::test]
async fn golden_typed_interruptions() {
    check_golden("typed_interruptions", &interruptions(true).await);
}
