//! Chunk rows on fake rows: the switch and family opt-in, ordered commits,
//! the row width, cancellation and the parent's request metrics. The merge
//! itself matches the sequential chunk route
//! (`engine::executor::audio::tests::committed_chunk_rows_match_the_sequential_run`).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use izwi_asr_toolkit::AsrLongFormConfig;
use tokio::sync::oneshot;

use super::*;
use crate::engine::AsrChunkPlannerKind;

/// Three six-second chunks at 10 Hz, each sharing a second with the next.
fn plan() -> PlannedAsrChunks {
    PlannedAsrChunks {
        config: AsrLongFormConfig::default(),
        chunks: [(0, 60), (50, 110), (100, 160)]
            .map(|(start_sample, end_sample)| AudioChunk {
                start_sample,
                end_sample,
            })
            .to_vec(),
        planner: AsrChunkPlannerKind::Duration,
        speech_plan: None,
        input_samples: 160,
        sample_rate: 10,
        planning_ms: 0.0,
    }
}

const TEXTS: [&str; 3] = [
    "one two three four five six seven",
    "seven eight nine ten eleven twelve thirteen",
    "thirteen fourteen fifteen sixteen seventeen",
];
const TRANSCRIPT: &str =
    "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen \
     sixteen seventeen";

fn row(text: &str) -> ChunkRow {
    ChunkRow {
        text: text.to_string(),
        diagnostics: None,
        transcribe_ms: 1.0,
        work: ChunkRowWork::default(),
    }
}

/// Counts rows started, and rows dropped before they finished.
#[derive(Default)]
struct RowCounts {
    started: AtomicUsize,
    running: AtomicUsize,
    most_running: AtomicUsize,
    dropped_unfinished: AtomicUsize,
}

/// Held by a running row; dropping it unfinished is a cancelled row.
struct RunningRow<'a> {
    counts: &'a RowCounts,
    finished: bool,
}

impl<'a> RunningRow<'a> {
    fn start(counts: &'a RowCounts) -> Self {
        counts.started.fetch_add(1, Ordering::SeqCst);
        let running = counts.running.fetch_add(1, Ordering::SeqCst) + 1;
        counts.most_running.fetch_max(running, Ordering::SeqCst);
        Self {
            counts,
            finished: false,
        }
    }
}

impl Drop for RunningRow<'_> {
    fn drop(&mut self) {
        self.counts.running.fetch_sub(1, Ordering::SeqCst);
        if !self.finished {
            self.counts
                .dropped_unfinished
                .fetch_add(1, Ordering::SeqCst);
        }
    }
}

/// The run's transcript, deltas and progress phases (with chunk numbers)
/// when the rows finish in `finish_order`, all in flight at once.
async fn run_finishing_in(
    finish_order: &[usize],
    streaming: bool,
) -> (String, Vec<String>, Vec<(AsrProgressPhase, Option<usize>)>) {
    let plan = plan();
    let senders = Mutex::new(
        (0..plan.chunks.len())
            .map(|_| None::<oneshot::Sender<String>>)
            .collect::<Vec<_>>(),
    );
    let submit = |index: usize, _chunk: AudioChunk| {
        let (tx, rx) = oneshot::channel::<String>();
        senders.lock().unwrap()[index] = Some(tx);
        async move { Ok(row(&rx.await.expect("the test finishes every row"))) }
    };
    let mut deltas = Vec::new();
    let mut progress = Vec::new();
    let mut on_delta = |delta| deltas.push(delta);
    let mut on_progress = |event: AsrProgress| progress.push((event.phase, event.current_chunk));
    let run = run_chunk_rows(
        &plan,
        plan.chunks.len(),
        streaming.then_some(0),
        Instant::now(),
        submit,
        &mut on_delta,
        &mut on_progress,
    );
    let finish = async {
        while senders.lock().unwrap().iter().any(Option::is_none) {
            tokio::task::yield_now().await;
        }
        for &index in finish_order {
            let tx = senders.lock().unwrap()[index].take().unwrap();
            tx.send(TEXTS[index].to_string()).unwrap();
            tokio::task::yield_now().await;
        }
    };
    let (rows, ()) = futures::future::join(run, finish).await;
    let rows = rows.expect("rows finish");
    assert_eq!(rows.work.len(), plan.chunks.len());
    (rows.transcription.text, deltas, progress)
}

/// The server spawns the transcription entry points, so chunk rows inside
/// them must keep their futures `Send` (checked at compile time).
#[allow(dead_code)]
fn transcriptions_with_chunk_rows_stay_send(runtime: &RuntimeService) {
    fn send<T: Send>(_: T) {}
    send(
        runtime.asr_transcribe_bytes_with_runtime_context_and_word_timestamps(
            &[],
            None,
            None,
            None,
            AsrTaskOptions::default(),
            None,
            None,
            RuntimeRequestContext::default(),
            false,
        ),
    );
    send(runtime.asr_transcribe_bytes_with_progress_and_runtime_context(
        &[],
        None,
        None,
        AsrTaskOptions::default(),
        None,
        RuntimeRequestContext::default(),
        false,
        |_| {},
        |_| {},
    ));
}

#[test]
fn only_plain_transcriptions_of_opted_in_families_run_as_rows() {
    let plain = AsrTaskOptions::default();
    for variant in [
        ModelVariant::Qwen3Asr06BGguf,
        ModelVariant::Qwen3Asr17BGguf,
        ModelVariant::WhisperLargeV3Turbo,
        ModelVariant::VoxtralMini4BRealtime2602,
    ] {
        assert!(chunk_rows_apply(variant, &plain, true), "{variant}");
        // The switch off keeps every file on the request route.
        assert!(!chunk_rows_apply(variant, &plain, false), "{variant}");
        let mut attributed = plain.clone();
        attributed.speaker_attribution = true;
        assert!(!chunk_rows_apply(variant, &attributed, true), "{variant}");
        let translated = AsrTaskOptions::new(Some("de".into()), Vec::new());
        assert!(!chunk_rows_apply(variant, &translated, true), "{variant}");
    }
    assert!(chunk_rows_family(ModelFamily::NemotronAsr));
    for family in [
        ModelFamily::ParakeetAsr,
        ModelFamily::VibeVoiceAsr,
        ModelFamily::NemotronTranscribeAsr,
        ModelFamily::GraniteSpeechAsr,
        ModelFamily::GraniteSpeech5Asr,
        ModelFamily::Lfm25Audio,
        ModelFamily::Lfm3Audio,
    ] {
        assert!(!chunk_rows_family(family), "{family:?}");
    }
    assert!(!chunk_rows_apply(
        ModelVariant::ParakeetTdt06BV3,
        &plain,
        true
    ));
}

/// Files fanned out side by side keep two batches of rows in flight, and
/// never more than half the coordinator queue.
#[test]
fn chunk_rows_in_flight_across_files_stay_within_the_queue() {
    assert_eq!(chunk_row_permit_count(8, 128), 16);
    assert_eq!(chunk_row_permit_count(2, 128), 4);
    assert_eq!(chunk_row_permit_count(32, 128), 64);
    assert_eq!(chunk_row_permit_count(32, 40), 20);
    assert_eq!(chunk_row_permit_count(8, 1), 1);
    assert_eq!(chunk_row_permit_count(0, 128), 2);
}

/// Rows that finish out of order still commit, stream and report progress
/// in plan order, with the transcript of rows finishing in order.
#[tokio::test]
async fn rows_commit_in_plan_order_whatever_order_they_finish() {
    for streaming in [false, true] {
        let (in_order, in_order_deltas, in_order_progress) =
            run_finishing_in(&[0, 1, 2], streaming).await;
        assert_eq!(in_order, TRANSCRIPT);
        for finish_order in [[2, 1, 0], [1, 2, 0], [2, 0, 1]] {
            let (text, deltas, progress) = run_finishing_in(&finish_order, streaming).await;
            assert_eq!(text, TRANSCRIPT, "{finish_order:?}");
            assert_eq!(deltas, in_order_deltas, "{finish_order:?}");
            assert_eq!(progress, in_order_progress, "{finish_order:?}");
        }
        if streaming {
            assert_eq!(in_order_deltas.concat(), TRANSCRIPT);
            assert_eq!(in_order_deltas.len(), 3, "a delta per committed chunk");
        } else {
            assert!(in_order_deltas.is_empty());
        }
        use AsrProgressPhase::*;
        assert_eq!(
            in_order_progress,
            vec![
                (Processing, None),
                (ChunkStarted, Some(1)),
                (ChunkFinished, Some(1)),
                (ChunkStarted, Some(2)),
                (ChunkFinished, Some(2)),
                (ChunkStarted, Some(3)),
                (ChunkFinished, Some(3)),
                (Complete, Some(3)),
            ]
        );
    }
}

/// At most `rows_in_flight` rows run at once, and a file keeps that many
/// running while it has chunks left.
#[tokio::test]
async fn rows_in_flight_bound_the_rows_running_at_once() {
    let mut plan = plan();
    plan.chunks = (0..7)
        .map(|index| AudioChunk {
            start_sample: index * 20,
            end_sample: index * 20 + 20,
        })
        .collect();
    plan.input_samples = 140;
    let counts = RowCounts::default();
    let submit = |index: usize, _chunk: AudioChunk| {
        let counts = &counts;
        async move {
            let mut running = RunningRow::start(counts);
            for _ in 0..(7 - index) {
                tokio::task::yield_now().await;
            }
            running.finished = true;
            Ok(row(&format!("chunk{index}")))
        }
    };
    let rows = run_chunk_rows(
        &plan,
        3,
        None,
        Instant::now(),
        submit,
        &mut |_| {},
        &mut |_| {},
    )
    .await
    .unwrap();
    assert_eq!(
        rows.transcription.text,
        "chunk0 chunk1 chunk2 chunk3 chunk4 chunk5 chunk6"
    );
    assert_eq!(counts.started.load(Ordering::SeqCst), 7);
    assert_eq!(counts.most_running.load(Ordering::SeqCst), 3);
    assert_eq!(counts.dropped_unfinished.load(Ordering::SeqCst), 0);
}

/// Dropping the parent drops (cancels) its rows in flight, and no row
/// starts after it.
#[tokio::test]
async fn dropping_the_parent_cancels_its_rows_in_flight() {
    let mut plan = plan();
    plan.chunks = (0..6)
        .map(|index| AudioChunk {
            start_sample: index * 20,
            end_sample: index * 20 + 20,
        })
        .collect();
    plan.input_samples = 120;
    let counts = RowCounts::default();
    {
        let submit = |_index: usize, _chunk: AudioChunk| {
            let counts = &counts;
            async move {
                let _running = RunningRow::start(counts);
                std::future::pending::<Result<ChunkRow>>().await
            }
        };
        let mut deltas = |_: String| {};
        let mut progress = |_: AsrProgress| {};
        let run = run_chunk_rows(
            &plan,
            4,
            Some(0),
            Instant::now(),
            submit,
            &mut deltas,
            &mut progress,
        );
        tokio::pin!(run);
        for _ in 0..8 {
            assert!(futures::poll!(run.as_mut()).is_pending());
        }
        assert_eq!(counts.running.load(Ordering::SeqCst), 4);
    }
    assert_eq!(counts.started.load(Ordering::SeqCst), 4);
    assert_eq!(counts.running.load(Ordering::SeqCst), 0);
    assert_eq!(counts.dropped_unfinished.load(Ordering::SeqCst), 4);
}

/// A failed row fails its parent, which cancels the rows still running.
#[tokio::test]
async fn a_failed_row_fails_the_parent_and_cancels_the_rest() {
    let counts = RowCounts::default();
    let submit = |index: usize, _chunk: AudioChunk| {
        let counts = &counts;
        async move {
            let mut running = RunningRow::start(counts);
            if index == 1 {
                running.finished = true;
                return Err(Error::InferenceError("row failed".into()));
            }
            std::future::pending::<Result<ChunkRow>>().await
        }
    };
    let error = run_chunk_rows(
        &plan(),
        3,
        None,
        Instant::now(),
        submit,
        &mut |_| {},
        &mut |_| {},
    )
    .await
    .err()
    .expect("the parent fails");
    assert!(error.to_string().contains("row failed"), "{error}");
    // It fails without waiting for the earlier row, which never finishes;
    // every other row that started is cancelled.
    assert_eq!(counts.running.load(Ordering::SeqCst), 0);
    let started = counts.started.load(Ordering::SeqCst);
    assert!(started >= 2, "{started}");
    assert_eq!(
        counts.dropped_unfinished.load(Ordering::SeqCst),
        started - 1
    );
}

/// The parent's output counts the whole file once: its audio, every row's
/// tokens and engine time, queued until its first row was scheduled and
/// answering when its first chunk committed.
#[test]
fn the_parent_output_sums_its_rows() {
    let work = [
        ChunkRowWork {
            prompt_tokens: 100,
            generated_tokens: 10,
            prefill_ms: 5.0,
            decode_ms: 50.0,
            scheduled_ms: Some(30.0),
        },
        ChunkRowWork {
            prompt_tokens: 120,
            generated_tokens: 12,
            prefill_ms: 6.0,
            decode_ms: 60.0,
            scheduled_ms: Some(20.0),
        },
    ];
    let transcription = AsrTranscription {
        text: TRANSCRIPT.to_string(),
        language: None,
        duration_secs: 600.0,
        asr_diagnostics: None,
    };
    let output =
        chunk_rows_parent_output("parent", &transcription, &work, Some(400.0), 12.0, 900.0);
    assert_eq!(output.request_id, "parent");
    assert_eq!(output.text.as_deref(), Some(TRANSCRIPT));
    assert_eq!(output.input_audio_duration_secs, Some(600.0));
    assert_eq!(output.token_stats.prompt_tokens, 220);
    assert_eq!(output.token_stats.generated_tokens, 22);
    assert!(output.error.is_none());
    let latency = output.latency_breakdown.expect("latency");
    assert_eq!(latency.queue_wait_ms, 20.0);
    assert_eq!(latency.prefill_ms, 11.0);
    assert_eq!(latency.decode_ms, 110.0);
    assert_eq!(latency.ttft_ms, Some(400.0));
    assert_eq!(latency.total_ms, 900.0);
    assert_eq!(latency.media_decode_ms, Some(12.0));
}
