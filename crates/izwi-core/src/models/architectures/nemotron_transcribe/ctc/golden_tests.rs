//! Replays `tests/fixtures/nemotron_transcribe/ctc_goldens.json`, which
//! `scripts/fixtures/generate_nemotron_transcribe_ctc_goldens.py` writes by
//! running NeMo's aligner (743beaf4) on synthetic inputs.
//!
//! * Alignment goldens run everywhere: each case's words carry the ids the
//!   real CTC tokenizer gave them, so the targets, speaker mapping, band,
//!   path, score, raw word times and the public `ctc_timestamps` object are
//!   all checked without the checkpoint.
//! * Tokenizer and `prepare` goldens need the real SentencePiece model: they
//!   read it from `$IZWI_NEMOTRON_TRANSCRIBE_MODEL_DIR/model-ctc-bf16.safetensors`
//!   and report themselves as skipped when the variable is unset.
//! * Small goldens: Python's `round(x, 3)`, SciPy's assignment, torch's
//!   resampling and the Unicode name ranges of script detection.

use std::io::Read;
use std::path::{Path, PathBuf};

use base64::Engine;
use serde_json::Value;

use super::align::{
    align, prepare, resample_speaker_probs, round_milliseconds, trace_alignment,
    CtcAlignmentConfig, CtcAlignmentInputs, CtcPrepared, CtcTimestampError, CtcTimestamps,
    SpeakerActivity,
};
use super::assignment::linear_sum_assignment;
use super::text::{character_token_ids, SotWord, SCRIPT_TABLES};
use super::tokenizer::CtcTokenizer;

const MODEL_DIR_ENV: &str = "IZWI_NEMOTRON_TRANSCRIBE_MODEL_DIR";
const PROTO_TENSOR: &str = "ctc_timestamp.tokenizer.model_proto";

fn fixture() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/nemotron_transcribe/ctc_goldens.json");
    let text = std::fs::read_to_string(&path).expect("read ctc_goldens.json");
    serde_json::from_str(&text).expect("parse ctc_goldens.json")
}

/// `base64(zlib(bytes))` → bytes.
fn unpack(text: &str) -> Vec<u8> {
    let compressed = base64::engine::general_purpose::STANDARD
        .decode(text)
        .expect("base64");
    let mut out = Vec::new();
    flate2::read::ZlibDecoder::new(&compressed[..])
        .read_to_end(&mut out)
        .expect("zlib");
    out
}

fn tag(value: &Value) -> Option<String> {
    value.as_str().map(str::to_string)
}

fn u32s(value: &Value) -> Vec<u32> {
    value
        .as_array()
        .expect("array")
        .iter()
        .map(|v| v.as_u64().expect("u32") as u32)
        .collect()
}

/// A case's words with their CTC ids, as upstream tokenized them.
fn tokenized_words(case: &Value) -> Vec<(SotWord, Vec<u32>)> {
    case["words"]
        .as_array()
        .expect("words")
        .iter()
        .map(|entry| {
            (
                SotWord {
                    word: entry[0].as_str().expect("word").to_string(),
                    speaker_tag: tag(&entry[1]),
                    turn_index: None,
                    character: entry[3].as_bool().expect("character"),
                },
                u32s(&entry[2]),
            )
        })
        .collect()
}

/// The case's inputs, owned.
struct Inputs {
    log_probs: Vec<f32>,
    speaker: Option<(Vec<f32>, usize, usize)>,
    diarization: Option<(Vec<f32>, usize, usize)>,
    duration: f64,
}

impl Inputs {
    fn of(case: &Value, columns: usize) -> Self {
        let frames = case["ctc_frames"].as_u64().unwrap() as usize;
        let log_probs: Vec<f32> = unpack(case["logp_codes"].as_str().unwrap())
            .into_iter()
            .map(|code| -(code as f32 / 16.0))
            .collect();
        assert_eq!(log_probs.len(), frames * columns, "{}", case["name"]);
        let speaker = case["speaker"].as_object().map(|speaker| {
            let values = unpack(speaker["codes"].as_str().unwrap())
                .into_iter()
                .map(|code| code as f32 / 256.0)
                .collect();
            (
                values,
                speaker["frames"].as_u64().unwrap() as usize,
                speaker["speakers"].as_u64().unwrap() as usize,
            )
        });
        let diarization = case["diarization"].as_object().map(|diarization| {
            let frames = diarization["frames"].as_u64().unwrap() as usize;
            let speakers = diarization["speakers"].as_u64().unwrap() as usize;
            let mut values = vec![0.1f32; frames * speakers];
            for (speaker, runs) in diarization["runs"].as_array().unwrap().iter().enumerate() {
                for run in runs.as_array().unwrap() {
                    let (a, b) = (
                        run[0].as_u64().unwrap() as usize,
                        run[1].as_u64().unwrap() as usize,
                    );
                    for frame in a..b {
                        values[frame * speakers + speaker] = 0.9;
                    }
                }
            }
            (values, frames, speakers)
        });
        Self {
            log_probs,
            speaker,
            diarization,
            duration: case["audio_duration"].as_f64().unwrap(),
        }
    }

    fn view(&self) -> CtcAlignmentInputs<'_> {
        fn activity(entry: &Option<(Vec<f32>, usize, usize)>) -> Option<SpeakerActivity<'_>> {
            entry
                .as_ref()
                .map(|(values, frames, speakers)| SpeakerActivity {
                    values,
                    frames: *frames,
                    speakers: *speakers,
                })
        }
        CtcAlignmentInputs {
            log_probs: &self.log_probs,
            speaker_probs: activity(&self.speaker),
            diarization: activity(&self.diarization),
            audio_duration: self.duration,
        }
    }
}

fn close(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() <= tolerance
}

/// Compares a result with the fixture's public object; mismatches go to
/// `failures`.
fn compare_public(
    name: &str,
    actual: &CtcTimestamps,
    expected: &Value,
    failures: &mut Vec<String>,
) {
    let error = expected["error"].as_str();
    let actual_error = actual.error.map(|e| serde_json::to_value(e).unwrap());
    if actual_error.as_ref().and_then(Value::as_str) != error {
        failures.push(format!("{name}: error {actual_error:?} != {error:?}"));
    }
    let words = expected["words"].as_array().unwrap();
    if words.len() != actual.words.len() {
        failures.push(format!(
            "{name}: {} words, expected {}",
            actual.words.len(),
            words.len()
        ));
    }
    for (index, (got, want)) in actual.words.iter().zip(words).enumerate() {
        let same = got.word == want["word"].as_str().unwrap()
            && got.speaker == want["speaker"].as_str().unwrap()
            && close(got.start, want["start"].as_f64().unwrap(), 1e-9)
            && close(got.end, want["end"].as_f64().unwrap(), 1e-9);
        if !same {
            failures.push(format!("{name}: word {index}: {got:?} != {want}"));
        }
    }
    let segments = expected["diarization"].as_array().unwrap();
    if segments.len() != actual.diarization.len() {
        failures.push(format!(
            "{name}: {} segments, expected {}",
            actual.diarization.len(),
            segments.len()
        ));
    }
    for (index, (got, want)) in actual.diarization.iter().zip(segments).enumerate() {
        let same = got.speaker as u64 == want["speaker"].as_u64().unwrap()
            && close(got.start, want["start"].as_f64().unwrap(), 1e-9)
            && close(got.end, want["end"].as_f64().unwrap(), 1e-9);
        if !same {
            failures.push(format!("{name}: segment {index}: {got:?} != {want}"));
        }
    }
    let mapping = serde_json::to_value(&actual.speaker_tag_to_diarization_speaker).unwrap();
    if mapping != expected["speaker_tag_to_diarization_speaker"] {
        failures.push(format!(
            "{name}: mapping {mapping} != {}",
            expected["speaker_tag_to_diarization_speaker"]
        ));
    }
}

/// Compares the prepared targets with the fixture's.
fn compare_prepared(name: &str, prepared: &CtcPrepared, case: &Value, failures: &mut Vec<String>) {
    if prepared.columns != u32s(&case["columns"]) {
        failures.push(format!(
            "{name}: columns {:?} != {}",
            prepared.columns, case["columns"]
        ));
    }
    let streams = case["streams"].as_array().unwrap();
    if streams.len() != prepared.streams.len() {
        failures.push(format!(
            "{name}: {} streams, expected {}",
            prepared.streams.len(),
            streams.len()
        ));
    }
    for (got, want) in prepared.streams.iter().zip(streams) {
        let words: Vec<&str> = want["words"]
            .as_array()
            .unwrap()
            .iter()
            .map(|w| w.as_str().unwrap())
            .collect();
        let state_to_word: Vec<Option<usize>> = want["state_to_word"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().map(|w| w as usize))
            .collect();
        let same = got.speaker_tag == tag(&want["speaker_tag"])
            && got.words == words
            && got.labels == u32s(&want["labels"])
            && got.state_to_word == state_to_word;
        if !same {
            failures.push(format!(
                "{name}: stream {:?} differs from {want}",
                got.speaker_tag
            ));
        }
    }
}

/// Starts as the fixture's delta string.
fn delta_string(starts: &[usize]) -> String {
    let mut out = String::with_capacity(starts.len());
    if let Some(&first) = starts.first() {
        out.push_str(&first.to_string());
    }
    for pair in starts.windows(2) {
        out.push_str(&(pair[1] - pair[0]).to_string());
    }
    out
}

fn run_length(path: &[usize]) -> Vec<(usize, usize)> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for &state in path {
        match runs.last_mut() {
            Some((last, count)) if *last == state => *count += 1,
            _ => runs.push((state, 1)),
        }
    }
    runs
}

/// Checks the search trace and raw rows of one prepared case.
fn compare_trace(
    name: &str,
    prepared: &CtcPrepared,
    inputs: &Inputs,
    case: &Value,
    failures: &mut Vec<String>,
) {
    let config = CtcAlignmentConfig::default();
    let Some(trace) = trace_alignment(prepared, &inputs.view(), &config) else {
        failures.push(format!("{name}: the trace failed"));
        return;
    };
    let searches = case["search"].as_array().unwrap();
    if searches.len() != trace.streams.len() {
        failures.push(format!(
            "{name}: {} searches, expected {}",
            trace.streams.len(),
            searches.len()
        ));
    }
    for (index, (got, want)) in trace.streams.iter().zip(searches).enumerate() {
        let column = want["column"].as_u64().map(|c| c as usize);
        if got.column != column || got.retained as u64 != want["retained_frames"].as_u64().unwrap()
        {
            failures.push(format!(
                "{name}: stream {index}: column {:?}/{} retained {} vs {want}",
                got.column, got.retained, want["retained_frames"]
            ));
        }
        let failed = want["failed"].as_bool().unwrap();
        if got.outcome.path.is_none() != failed {
            failures.push(format!(
                "{name}: stream {index}: failed {} vs {failed}",
                got.outcome.path.is_none()
            ));
            continue;
        }
        if failed {
            continue;
        }
        let score = want["score"].as_f64().unwrap();
        if !close(got.outcome.score as f64, score, 1e-5 * score.abs().max(1.0)) {
            failures.push(format!(
                "{name}: stream {index}: score {} != {score}",
                got.outcome.score
            ));
        }
        if delta_string(&got.outcome.starts) != want["starts"].as_str().unwrap() {
            failures.push(format!("{name}: stream {index}: band starts differ"));
        }
        let path: Vec<(usize, usize)> = want["path"]
            .as_array()
            .unwrap()
            .iter()
            .map(|run| {
                (
                    run[0].as_u64().unwrap() as usize,
                    run[1].as_u64().unwrap() as usize,
                )
            })
            .collect();
        if run_length(got.outcome.path.as_deref().unwrap_or(&[])) != path {
            failures.push(format!("{name}: stream {index}: path differs"));
        }
    }
    let raw = case["raw_words"].as_array().unwrap();
    if raw.len() != trace.rows.len() {
        failures.push(format!(
            "{name}: {} raw rows, expected {}",
            trace.rows.len(),
            raw.len()
        ));
    }
    for (index, (got, want)) in trace.rows.iter().zip(raw).enumerate() {
        let speaker = got
            .speaker_tag
            .clone()
            .unwrap_or_else(|| "None".to_string());
        let same = got.word == want["word"].as_str().unwrap()
            && speaker == want["speaker"].as_str().unwrap()
            && close(got.start, want["start"].as_f64().unwrap(), 1e-9)
            && close(got.end, want["end"].as_f64().unwrap(), 1e-9);
        if !same {
            failures.push(format!("{name}: raw row {index}: {got:?} != {want}"));
        }
    }
}

#[test]
fn alignment_goldens_reproduce_upstream() {
    let fixture = fixture();
    let blank = fixture["blank"].as_u64().unwrap() as u32;
    let config = CtcAlignmentConfig::default();
    let mut failures = Vec::new();
    let mut checked = 0;
    for case in fixture["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        if !case["prepare_error"].is_null() {
            continue;
        }
        let frames = case["ctc_frames"].as_u64().unwrap() as usize;
        let prepared = CtcPrepared::from_tokenized(tokenized_words(case), frames, blank)
            .unwrap_or_else(|error| panic!("{name}: {error:?}"));
        compare_prepared(name, &prepared, case, &mut failures);
        let inputs = Inputs::of(case, prepared.columns.len());
        compare_trace(name, &prepared, &inputs, case, &mut failures);
        let result = align(&prepared, inputs.view(), &config);
        compare_public(name, &result, &case["expected"], &mut failures);
        checked += 1;
    }
    assert!(checked >= 10, "only {checked} cases");
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// The CTC SentencePiece proto from the checkpoint's CTC file, by reading
/// only the safetensors header and the one U8 tensor.
fn read_model_proto(dir: &Path) -> Vec<u8> {
    use std::io::{Seek, SeekFrom};
    let path = dir.join("model-ctc-bf16.safetensors");
    let mut file = std::fs::File::open(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut length = [0u8; 8];
    file.read_exact(&mut length).unwrap();
    let length = u64::from_le_bytes(length) as usize;
    let mut header = vec![0u8; length];
    file.read_exact(&mut header).unwrap();
    let header: Value = serde_json::from_slice(&header).expect("safetensors header");
    let entry = &header[PROTO_TENSOR];
    assert_eq!(entry["dtype"], "U8", "{PROTO_TENSOR}");
    let (start, end) = (
        entry["data_offsets"][0].as_u64().unwrap(),
        entry["data_offsets"][1].as_u64().unwrap(),
    );
    file.seek(SeekFrom::Start(8 + length as u64 + start))
        .unwrap();
    let mut proto = vec![0u8; (end - start) as usize];
    file.read_exact(&mut proto).unwrap();
    proto
}

#[test]
fn tokenizer_and_prepare_goldens_with_the_real_proto() {
    let Some(dir) = std::env::var_os(MODEL_DIR_ENV) else {
        eprintln!("skipped: set {MODEL_DIR_ENV} to a directory holding model-ctc-bf16.safetensors");
        return;
    };
    let fixture = fixture();
    let tokenizer =
        CtcTokenizer::from_model_proto(&read_model_proto(Path::new(&dir))).expect("tokenizer");
    let golden = &fixture["tokenizer"];
    assert_eq!(tokenizer.len() as u64, golden["pieces"].as_u64().unwrap());
    assert_eq!(
        tokenizer.unk_id() as u64,
        golden["unk_id"].as_u64().unwrap()
    );
    let blank = fixture["blank"].as_u64().unwrap() as u32;
    let mut failures = Vec::new();
    for entry in golden["encode"].as_array().unwrap() {
        let text = entry["text"].as_str().unwrap();
        let ids = tokenizer.encode(text);
        if ids != u32s(&entry["ids"]) {
            failures.push(format!("encode {text:?}: {ids:?} != {}", entry["ids"]));
        }
    }
    for entry in golden["character_ids"].as_array().unwrap() {
        let character = entry["character"].as_str().unwrap();
        let ids = character_token_ids(&tokenizer, character, blank);
        let want = (!entry["ids"].is_null()).then(|| u32s(&entry["ids"]));
        if ids != want {
            failures.push(format!("character {character:?}: {ids:?} != {want:?}"));
        }
    }
    let config = CtcAlignmentConfig::default();
    for case in fixture["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let frames = case["ctc_frames"].as_u64().unwrap() as usize;
        let transcript = case["transcript"].as_str().unwrap();
        let prepared = prepare(&tokenizer, transcript, None, frames, blank, &config);
        match (prepared, case["prepare_error"].as_str()) {
            (Err(CtcTimestampError::AlignmentFailed), Some("alignment_failed")) => {}
            (Ok(prepared), None) => {
                let expected =
                    CtcPrepared::from_tokenized(tokenized_words(case), frames, blank).unwrap();
                if prepared != expected {
                    failures.push(format!(
                        "{name}: prepare differs from upstream's tokenization"
                    ));
                }
                compare_prepared(name, &prepared, case, &mut failures);
                let inputs = Inputs::of(case, prepared.columns.len());
                let result = align(&prepared, inputs.view(), &config);
                compare_public(name, &result, &case["expected"], &mut failures);
            }
            (got, want) => failures.push(format!("{name}: prepare {got:?}, upstream {want:?}")),
        }
    }
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn small_goldens_match_python_scipy_and_torch() {
    let fixture = fixture();
    for pair in fixture["rounding"].as_array().unwrap() {
        // Strings: serde_json's default float parsing is not correctly
        // rounded for 17-digit decimals; `str::parse` is.
        let value: f64 = pair[0].as_str().unwrap().parse().unwrap();
        let rounded: f64 = pair[1].as_str().unwrap().parse().unwrap();
        assert_eq!(round_milliseconds(value), rounded, "round({value}, 3)");
    }
    for case in fixture["assignment"].as_array().unwrap() {
        let rows = case["rows"].as_u64().unwrap() as usize;
        let cols = case["cols"].as_u64().unwrap() as usize;
        let cost: Vec<f64> = case["cost"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let want: Vec<(usize, usize)> = case["pairs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                (
                    p[0].as_u64().unwrap() as usize,
                    p[1].as_u64().unwrap() as usize,
                )
            })
            .collect();
        assert_eq!(
            linear_sum_assignment(&cost, rows, cols).unwrap(),
            want,
            "{case}"
        );
    }
    for case in fixture["resample"].as_array().unwrap() {
        let frames = case["frames"].as_u64().unwrap() as usize;
        let speakers = case["speakers"].as_u64().unwrap() as usize;
        let target = case["target"].as_u64().unwrap() as usize;
        let values: Vec<f32> = case["codes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c.as_u64().unwrap() as f32 / 256.0)
            .collect();
        let want: Vec<f32> = case["output"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        assert_eq!(
            resample_speaker_probs(&values, frames, speakers, target),
            want,
            "{case}"
        );
    }
    let ranges = &fixture["unicode_name_ranges"];
    for (language, table) in SCRIPT_TABLES {
        let want: Vec<(u32, u32)> = ranges[language]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| (r[0].as_u64().unwrap() as u32, r[1].as_u64().unwrap() as u32))
            .collect();
        assert_eq!(table, want.as_slice(), "{language}");
    }
    let config = CtcAlignmentConfig::default();
    let golden = &fixture["config"];
    assert_eq!(
        config.speaker_activity_threshold as f64,
        golden["speaker_activity_threshold"].as_f64().unwrap()
    );
    assert_eq!(
        config.speaker_logprob_weight as f64,
        golden["speaker_logprob_weight"].as_f64().unwrap()
    );
    assert_eq!(
        config.maximum_token_len,
        golden["maximum_token_len"].as_f64().unwrap()
    );
    assert_eq!(
        config.max_word_frames as u64,
        golden["max_word_frames"].as_u64().unwrap()
    );
    assert_eq!(config.epsilon, golden["epsilon"].as_f64().unwrap() as f32);
}
