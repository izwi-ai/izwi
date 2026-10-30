//! Parity against upstream references, stage by stage.
//!
//! * **L1, the tiny upstream fixture** (`tests/fixtures/nemotron_transcribe/tiny/`,
//!   written once by `scripts/fixtures/generate_nemotron_transcribe_tiny.py`):
//!   a random checkpoint in the published layout and what NeMo's perception
//!   and transformers' NemotronH computed from it with dequantized weights.
//!   Runs on the CPU and Metal whenever the fixture is present, and reports
//!   itself as not run otherwise.
//! * **L2, the real checkpoint** (`#[ignore]`): the reference
//!   `generate_nemotron_transcribe_reference.py` writes on the GPU box, read
//!   with `IZWI_NEMOTRON_TRANSCRIBE_MODEL_DIR` (the checkpoint),
//!   `IZWI_NEMOTRON_TRANSCRIBE_AUDIO_DIR` (the converted clips),
//!   `IZWI_NEMOTRON_TRANSCRIBE_BACKEND` (`cpu`, `metal` or `cuda`) and
//!   optionally `IZWI_NEMOTRON_TRANSCRIBE_REFERENCE` (the JSON; default the
//!   fixture's `reference.json`). `scripts/e2e/nemotron-transcribe/run-parity.sh`
//!   runs it.
//!
//! Every stage compares with the tolerance the fixture suggests for the
//! backend, router decisions only where the reference's margin is not a
//! near-tie, and greedy ids up to the first step whose reference margin is a
//! near-tie. Failures are collected per stage and reported together.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_core::{DType, Device};
use serde_json::Value;

use super::decoder::moe::Routing;
use super::decoder::{DecoderTrace, TraceRequest};
use super::frontend::NemotronMel;
use super::model::{NemotronStepRow, NemotronTranscribeModel, NemotronTranscribeRequest};
use super::perception::{PerceptionStream, PerceptionWindow, TimestampWindow};
use super::prompt::NemotronPromptOptions;
use super::speech_transformer::host_values;
use super::transcript::NemotronFinishReason;
use crate::kernels::numeric_check::relative_l2;
use crate::models::shared::attention::physical::test_physical_caches_for_model_layers_on;

const MODEL_DIR_ENV: &str = "IZWI_NEMOTRON_TRANSCRIBE_MODEL_DIR";
const AUDIO_DIR_ENV: &str = "IZWI_NEMOTRON_TRANSCRIBE_AUDIO_DIR";
const BACKEND_ENV: &str = "IZWI_NEMOTRON_TRANSCRIBE_BACKEND";
const REFERENCE_ENV: &str = "IZWI_NEMOTRON_TRANSCRIBE_REFERENCE";

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/nemotron_transcribe")
}

/// What izwi computed for one request, in the reference's terms.
struct Observed {
    mel: NemotronMel,
    plan: Vec<PerceptionWindow>,
    windows: Vec<TimestampWindow>,
    /// `[audio_tokens, hidden]`.
    embeddings: Vec<f32>,
    prompt_ids: Vec<u32>,
    /// Residual stream after each traced layer, `[prompt, hidden]`.
    hidden: Vec<(usize, Vec<f32>)>,
    /// The first MoE layer's routing of every prompt position.
    routes: Vec<Routing>,
    final_logits: Vec<f32>,
    greedy: Vec<u32>,
    finish: NemotronFinishReason,
    /// The decoded text with tags, after the guardrail (`transcript.raw`).
    text: String,
}

fn observe(
    model: &NemotronTranscribeModel,
    device: &Device,
    samples: &[f32],
    request: NemotronTranscribeRequest,
    traced_layers: &[usize],
) -> Observed {
    let mel = model.frontend().compute(samples).unwrap();
    let mut stream = PerceptionStream::new(mel.clone(), model.perception().stacking(), true);
    let embeddings = host_values(&stream.all_embeddings(model.perception()).unwrap()).unwrap();
    let windows = stream.retained_windows().to_vec();
    let plan = stream.windows().to_vec();

    let artifact = Arc::new(model.prepare(samples, request).unwrap());
    let prompt_ids = artifact.layout.expanded_ids(model.audio_token_id());
    let context = artifact.max_context();
    let a = model.decoder().config().attention;
    let kv = if device.is_cpu() {
        DType::F32
    } else {
        DType::F16
    };
    let mut cache = test_physical_caches_for_model_layers_on(
        device,
        kv,
        model.decoder().attention_layers(),
        a.kv_heads,
        a.head_dim,
        context,
        1,
    )
    .remove(0);
    let mut state = model.new_row(artifact, context).unwrap();
    let trace_request = TraceRequest {
        routes: true,
        hidden_after_layers: traced_layers.to_vec(),
    };
    let moe_layers = model
        .decoder()
        .config()
        .layers_of(super::config::NemotronLayerKind::Moe)
        .len();
    let span = super::prefill_span(match device {
        Device::Cpu => crate::backends::BackendKind::Cpu,
        Device::Metal(_) => crate::backends::BackendKind::Metal,
        Device::Cuda(_) => crate::backends::BackendKind::Cuda,
    });
    let mut hidden: Vec<(usize, Vec<f32>)> =
        traced_layers.iter().map(|&l| (l, Vec::new())).collect();
    let mut routes = Vec::new();
    let mut final_logits = Vec::new();
    while state.prefill_cursor() < state.prompt_tokens() {
        let n = span.min(state.prompt_tokens() - state.prefill_cursor());
        let mut trace = DecoderTrace::default();
        model
            .retained_prefill_traced(
                &mut state,
                &mut cache,
                n,
                Some((&trace_request, &mut trace)),
            )
            .unwrap();
        for (layer, values) in trace.hidden {
            if let Some(slot) = hidden.iter_mut().find(|(l, _)| *l == layer) {
                slot.1.extend(values);
            }
        }
        // Each span records every MoE layer in order; the first is layer 1.
        if trace.routes.len() == moe_layers {
            routes.extend(trace.routes.into_iter().next().unwrap());
        }
        if !trace.final_logits.is_empty() {
            final_logits = trace.final_logits;
        }
    }
    while !state.finished() {
        model
            .retained_decode_batch(&mut [NemotronStepRow {
                state: &mut state,
                cache: &mut cache,
            }])
            .unwrap();
    }
    let transcription = model.finish_row(&state).unwrap();
    Observed {
        mel,
        plan,
        windows,
        embeddings,
        prompt_ids,
        hidden,
        routes,
        final_logits,
        greedy: state.decode_loop().unwrap().generated().to_vec(),
        finish: transcription.finish_reason,
        text: transcription.transcript.raw,
    }
}

/// A stage's verdicts, collected so one run reports every stage.
#[derive(Default)]
struct Report {
    failures: Vec<String>,
    notes: Vec<String>,
}

impl Report {
    fn check(&mut self, ok: bool, message: impl FnOnce() -> String) {
        if !ok {
            self.failures.push(message());
        }
    }

    fn finish(self, what: &str) {
        for note in &self.notes {
            eprintln!("{what}: {note}");
        }
        assert!(
            self.failures.is_empty(),
            "{what}: {} stage failures:\n{}",
            self.failures.len(),
            self.failures.join("\n")
        );
    }
}

fn floats(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_f64)
                .map(|v| v as f32)
                .collect()
        })
        .unwrap_or_default()
}

/// A tolerance from `suggested_tolerances`: `path` then the backend key, or
/// a plain number, or `default`.
fn tolerance(tolerances: &Value, path: &[&str], backend: &str, default: f64) -> f64 {
    let mut value = tolerances;
    for key in path {
        value = &value[*key];
    }
    value
        .get(backend)
        .and_then(Value::as_f64)
        .or_else(|| value.as_f64())
        .unwrap_or(default)
}

fn bf16_ulps(a: f32, b: f32) -> u16 {
    let (a, b) = (half::bf16::from_f32(a), half::bf16::from_f32(b));
    if a == b {
        0
    } else {
        a.to_bits().abs_diff(b.to_bits())
    }
}

/// Compares one clip's perception and one request's decode against the
/// reference objects (`perception` keys and decode keys may live on the
/// same object, as in L1).
#[allow(clippy::too_many_arguments)]
fn compare(
    report: &mut Report,
    label: &str,
    observed: &Observed,
    perception: &Value,
    decode: &Value,
    tolerances: &Value,
    backend: &str,
    hidden_width: usize,
) {
    // Mel.
    let mel = &perception["mel"];
    if mel.is_object() {
        report.check(
            mel["valid_frames"].as_u64() == Some(observed.mel.valid_frames() as u64),
            || {
                format!(
                    "{label}: mel valid frames {:?} vs {}",
                    mel["valid_frames"],
                    observed.mel.valid_frames()
                )
            },
        );
        let (mean, std) = observed.mel.statistics();
        let worst_mean = floats(&mel["mean"])
            .iter()
            .zip(&mean)
            .map(|(r, o)| bf16_ulps(*r, *o))
            .max()
            .unwrap_or(0);
        report.check(worst_mean <= 1, || {
            format!("{label}: mel mean off by {worst_mean} BF16 ulps")
        });
        let std_rel = tolerance(tolerances, &["mel_std", "rel"], backend, 2e-3) as f32;
        let worst_std = floats(&mel["std"])
            .iter()
            .zip(&std)
            .map(|(r, o)| ((r - o) / r.abs().max(1e-6)).abs())
            .fold(0f32, f32::max);
        report.check(worst_std <= std_rel, || {
            format!("{label}: mel std rel {worst_std:e}")
        });
        for row in mel["raw_rows"].as_array().into_iter().flatten() {
            let frame = row["frame"].as_u64().unwrap_or(0) as usize;
            let expected = floats(&row["values"]);
            let ours = observed.mel.raw_window(frame, frame + 1);
            let off = expected
                .iter()
                .zip(&ours)
                .filter(|(r, o)| bf16_ulps(**r, **o) > 0)
                .count();
            let far = expected
                .iter()
                .zip(&ours)
                .filter(|(r, o)| bf16_ulps(**r, **o) > 1)
                .count();
            report.check(far == 0 && off * 50 <= expected.len().max(1), || {
                format!("{label}: mel frame {frame}: {off} values differ, {far} by more than one BF16 ulp")
            });
        }
    }
    // Windows: the plan, the ASR core rows (before asr_norm) and the
    // diarizer's 80 ms probabilities.
    for (index, window) in perception["windows"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
    {
        let Some(plan) = observed.plan.get(index) else {
            report
                .failures
                .push(format!("{label}: izwi has no window {index}"));
            continue;
        };
        report.check(
            window["start"].as_u64() == Some(plan.start as u64)
                && window["end"].as_u64() == Some(plan.end as u64)
                && window["context_start"].as_u64() == Some(plan.context_start as u64)
                && window["context_end"].as_u64() == Some(plan.context_end as u64),
            || format!("{label}: window {index} plan differs: {window:?} vs {plan:?}"),
        );
        let Some(encoded) = observed.windows.get(index) else {
            continue;
        };
        let width = encoded.ctc_input.dims()[1];
        let asr = host_values(&encoded.ctc_input).unwrap();
        let rel = tolerance(tolerances, &["asr_core_rows", "rel_l2"], backend, 2e-2);
        for row in window["asr_core_rows"].as_array().into_iter().flatten() {
            let r = row["row"].as_u64().unwrap_or(0) as usize;
            let expected = floats(&row["values"]);
            let ours = asr.get(r * width..(r + 1) * width).unwrap_or(&[]);
            let error = relative_l2(ours, &expected);
            report.check(error <= rel, || {
                format!("{label}: window {index} ASR row {r} rel {error:e}")
            });
        }
        let follows = window["follows_compression"].as_bool().unwrap_or(false);
        let expected = window["diar_probs_80ms"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        if !follows {
            let bound = tolerance(
                tolerances,
                &["diar_probs_80ms", "before_first_compression", "max_abs"],
                backend,
                5e-3,
            ) as f32;
            let worst = expected
                .iter()
                .zip(&encoded.speaker_preds)
                .flat_map(|(r, o)| {
                    floats(r)
                        .into_iter()
                        .zip(o.clone())
                        .map(|(a, b)| (a - b).abs())
                })
                .fold(0f32, f32::max);
            report.check(worst <= bound, || {
                format!("{label}: window {index} diarizer max abs {worst:e}")
            });
        } else {
            let (mut agree, mut total) = (0usize, 0usize);
            for (r, o) in expected.iter().zip(&encoded.speaker_preds) {
                for (a, b) in floats(r).into_iter().zip(o.clone()) {
                    agree += usize::from((a > 0.5) == (b > 0.5));
                    total += 1;
                }
            }
            let min = tolerance(
                tolerances,
                &[
                    "diar_probs_80ms",
                    "after_first_compression",
                    "binary_agreement_min",
                ],
                backend,
                0.995,
            );
            let rate = agree as f64 / total.max(1) as f64;
            report.check(rate >= min, || {
                format!("{label}: window {index} diarizer activity agreement {rate}")
            });
        }
    }
    // Projected embeddings.
    let rel = tolerance(tolerances, &["embeddings_rows", "rel_l2"], backend, 2e-2);
    for row in perception["embeddings_rows"]
        .as_array()
        .into_iter()
        .flatten()
    {
        let token = row["token"].as_u64().unwrap_or(0) as usize;
        let expected = floats(&row["values"]);
        let ours = observed
            .embeddings
            .get(token * hidden_width..(token + 1) * hidden_width)
            .unwrap_or(&[]);
        let error = relative_l2(ours, &expected);
        report.check(error <= rel, || {
            format!("{label}: embedding token {token} rel {error:e}")
        });
    }
    if !decode.is_object() {
        return;
    }
    // Prompt ids.
    let expected_ids = decode["prompt_ids"].as_array().map(|ids| {
        ids.iter()
            .filter_map(Value::as_u64)
            .map(|v| v as u32)
            .collect::<Vec<_>>()
    });
    if let Some(ids) = expected_ids {
        report.check(ids == observed.prompt_ids, || {
            format!(
                "{label}: prompt ids differ ({} vs {} tokens)",
                ids.len(),
                observed.prompt_ids.len()
            )
        });
    }
    // Layer outputs.
    let rel = tolerance(tolerances, &["layer_hidden", "rel_l2"], backend, 5e-2);
    for layer in decode["layer_hidden"].as_array().into_iter().flatten() {
        let index = layer["layer"].as_u64().unwrap_or(0) as usize;
        let Some((_, ours)) = observed.hidden.iter().find(|(l, _)| *l == index) else {
            report
                .notes
                .push(format!("{label}: layer {index} was not traced"));
            continue;
        };
        for row in layer["rows"].as_array().into_iter().flatten() {
            let position = row["position"].as_u64().unwrap_or(0) as usize;
            let expected = floats(&row["values"]);
            let ours = ours
                .get(position * hidden_width..position * hidden_width + expected.len())
                .unwrap_or(&[]);
            let error = relative_l2(ours, &expected);
            report.check(error <= rel, || {
                format!("{label}: layer {index} position {position} rel {error:e}")
            });
        }
    }
    // Router decisions, tie-aware.
    let near_tie = tolerances["routes"]["expert_set_equal_when_margin_ge"]
        .as_f64()
        .unwrap_or(1e-5) as f32;
    let (mut compared, mut ties) = (0usize, 0usize);
    for row in decode["routes"]["rows"].as_array().into_iter().flatten() {
        let position = row["position"].as_u64().unwrap_or(0) as usize;
        let margin = row["margin"].as_f64().unwrap_or(f64::INFINITY) as f32;
        let Some(ours) = observed.routes.get(position) else {
            continue;
        };
        let mut expected = row["experts"]
            .as_array()
            .map(|e| {
                e.iter()
                    .filter_map(Value::as_u64)
                    .map(|v| v as usize)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let mut mine = ours.experts.clone();
        expected.sort_unstable();
        mine.sort_unstable();
        if margin < near_tie || ours.margin < near_tie {
            ties += 1;
            continue;
        }
        compared += 1;
        report.check(expected == mine, || {
            format!("{label}: position {position} routed {mine:?}, reference {expected:?} (margin {margin:e})")
        });
    }
    report.notes.push(format!(
        "{label}: {compared} routed rows compared, {ties} near-ties skipped"
    ));
    // Prefill logits.
    let bound = tolerance(
        tolerances,
        &["prefill_logits_top8", "max_abs"],
        backend,
        0.1,
    ) as f32;
    let mut worst = 0f32;
    for entry in decode["prefill_logits_top8"]
        .as_array()
        .into_iter()
        .flatten()
    {
        let id = entry["id"].as_u64().unwrap_or(0) as usize;
        let logit = entry["logit"].as_f64().unwrap_or(0.0) as f32;
        let ours = observed.final_logits.get(id).copied().unwrap_or(f32::NAN);
        let diff = (ours - logit).abs();
        worst = if diff.is_finite() {
            worst.max(diff)
        } else {
            f32::INFINITY
        };
    }
    report.check(worst <= bound, || {
        format!("{label}: prefill top-8 logits max abs {worst:e}")
    });
    // Greedy ids: equal up to the first reference near-tie.
    let margin_bound = tolerances["greedy_ids"]["exact_prefix_until_margin_below"]
        .as_f64()
        .unwrap_or(0.02) as f32;
    let expected = decode["greedy_ids"]
        .as_array()
        .map(|ids| {
            ids.iter()
                .filter_map(Value::as_u64)
                .map(|v| v as u32)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let margins = floats(&decode["greedy_margins"]);
    let mut prefix_ok = true;
    for (step, reference) in expected.iter().enumerate() {
        if margins.get(step).is_some_and(|m| *m < margin_bound) {
            report.notes.push(format!(
                "{label}: greedy near-tie at step {step}; later ids may diverge"
            ));
            break;
        }
        if observed.greedy.get(step) != Some(reference) {
            prefix_ok = false;
            report.failures.push(format!(
                "{label}: greedy id {step} is {:?}, reference {reference}",
                observed.greedy.get(step)
            ));
            break;
        }
    }
    if prefix_ok && expected.len() < 64 {
        // The whole short decode matched: the text must too.
        if let Some(text) = decode["text"].as_str() {
            report.check(text == observed.text, || {
                format!("{label}: text {:?} vs {text:?}", observed.text)
            });
        }
        if let Some(finish) = decode["finish_reason"].as_str() {
            report.check(finish == observed.finish.as_str(), || {
                format!("{label}: finish {} vs {finish}", observed.finish.as_str())
            });
        }
    }
}

fn read_pcm16_wav(path: &Path) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).unwrap();
    match reader.spec().sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| f32::from(s.unwrap()) / 32_768.0)
            .collect(),
        hound::SampleFormat::Float => reader.samples::<f32>().map(Result::unwrap).collect(),
    }
}

fn read_f32_le(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn run_tiny_fixture(device: Device, backend: &str) {
    let dir = fixture_dir().join("tiny");
    let Ok(raw) = std::fs::read(dir.join("expected.json")) else {
        eprintln!("tests/fixtures/nemotron_transcribe/tiny/ is absent; L1 parity not run");
        return;
    };
    let expected: Value = serde_json::from_slice(&raw).unwrap();
    let model = NemotronTranscribeModel::load_on(&dir, None, &device).unwrap();
    let hidden = model.decoder().config().hidden;
    let traced = expected["clips"][0]["layer_hidden"]
        .as_array()
        .map(|layers| {
            layers
                .iter()
                .filter_map(|l| l["layer"].as_u64())
                .map(|l| l as usize)
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![0, 1, 2]);
    let mut report = Report::default();
    for clip in expected["clips"].as_array().into_iter().flatten() {
        let name = clip["name"].as_str().unwrap_or("clip");
        let samples = read_pcm16_wav(&dir.join(clip["audio_file"].as_str().unwrap()));
        let request = NemotronTranscribeRequest {
            prompt: NemotronPromptOptions::default(),
            max_new_tokens: expected["max_new_tokens"].as_u64().map(|v| v as usize),
            word_timestamps: false,
        };
        let observed = observe(&model, &device, &samples, request, &traced);
        compare(
            &mut report,
            &format!("L1 {backend} {name}"),
            &observed,
            clip,
            clip,
            &expected["suggested_tolerances"],
            backend,
            hidden,
        );
    }
    report.finish(&format!("L1 tiny fixture on {backend}"));
}

#[test]
fn tiny_fixture_matches_upstream_on_cpu() {
    run_tiny_fixture(Device::Cpu, "cpu");
}

#[cfg(feature = "metal")]
#[test]
fn tiny_fixture_matches_upstream_on_metal() {
    let Some((device, _lease)) =
        crate::backends::metal_test_device_or_skip("tiny_fixture_matches_upstream_on_metal")
    else {
        return;
    };
    run_tiny_fixture(device, "metal");
}

/// The real checkpoint against the GPU box's reference (Leg A tensors and
/// text, Leg B text reported).
#[test]
#[ignore = "needs the published checkpoint, the L2 reference and its audio"]
fn real_checkpoint_matches_the_reference() {
    let model_dir = PathBuf::from(std::env::var(MODEL_DIR_ENV).expect(MODEL_DIR_ENV));
    let reference_path = std::env::var(REFERENCE_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|_| fixture_dir().join("reference.json"));
    let audio_dir = std::env::var(AUDIO_DIR_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|_| fixture_dir().join("reference-audio"));
    let backend = std::env::var(BACKEND_ENV).unwrap_or_else(|_| "cpu".into());
    let device = match backend.as_str() {
        "cpu" => Device::Cpu,
        #[cfg(feature = "metal")]
        "metal" => crate::backends::metal_device_if_available(0)
            .expect("IZWI_NEMOTRON_TRANSCRIBE_BACKEND=metal on a machine without Metal"),
        #[cfg(feature = "cuda")]
        "cuda" => Device::new_cuda(0).unwrap(),
        other => panic!("{BACKEND_ENV}={other}: this build has no such backend"),
    };
    let reference: Value =
        serde_json::from_slice(&std::fs::read(&reference_path).unwrap()).unwrap();
    let model = NemotronTranscribeModel::load_on(&model_dir, None, &device).unwrap();
    eprintln!("{}", model.diagnostics_summary());
    let hidden = model.decoder().config().hidden;
    let clips = reference["clips"].as_array().cloned().unwrap_or_default();
    let mut report = Report::default();
    for case in reference["cases"].as_array().into_iter().flatten() {
        let name = case["name"].as_str().unwrap_or("case");
        let clip_name = case["clip"].as_str().unwrap_or(name);
        let Some(clip) = clips.iter().find(|c| c["name"].as_str() == Some(clip_name)) else {
            report
                .failures
                .push(format!("{name}: reference has no clip {clip_name}"));
            continue;
        };
        let samples = read_f32_le(&audio_dir.join(clip["raw_file"].as_str().unwrap()));
        let speaker = case["instruction_kind"].as_str() == Some("speaker");
        let request = NemotronTranscribeRequest {
            prompt: NemotronPromptOptions {
                speaker_attribution: speaker,
                ..Default::default()
            },
            max_new_tokens: None,
            word_timestamps: false,
        };
        let started = std::time::Instant::now();
        let observed = observe(&model, &device, &samples, request, &[0, 1, 5]);
        eprintln!(
            "{name}: {:.1} s on {backend}: {:?}",
            started.elapsed().as_secs_f64(),
            observed.text
        );
        compare(
            &mut report,
            &format!("L2 {backend} {name}"),
            &observed,
            &clip["perception"],
            &case["leg_a"],
            &reference["suggested_tolerances"],
            &backend,
            hidden,
        );
        // Leg B is W8A8 on SM89+ (or Marlin W8A16 on SM80/86): its text is
        // reported against izwi's, and gated only at one word per clip.
        if let Some(leg_b) = case["leg_b"]["text"].as_str() {
            let ours = super::transcript::parse_transcript(&observed.text).text;
            let theirs = super::transcript::parse_transcript(leg_b).text;
            let a = ours.split_whitespace().collect::<Vec<_>>();
            let b = theirs.split_whitespace().collect::<Vec<_>>();
            let differing =
                a.iter().zip(&b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len());
            report
                .notes
                .push(format!("{name}: Leg B text differs in {differing} words"));
            report.check(differing <= 1, || {
                format!("{name}: Leg B text differs in {differing} words")
            });
        }
    }
    report.finish(&format!("L2 real checkpoint on {backend}"));
}
