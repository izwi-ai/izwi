//! Parity with the published PyTorch implementation, on real weights.
//!
//! The fixtures come from `scripts/fixtures/generate_audio8_reference.py`,
//! which runs the checkpoint's own `ArkttsProcessor`, `ArkttsModel` and
//! `ArkttsCodec` in F32 on the CPU. Every test here is ignored: it needs the
//! pinned bundle (`IZWI_AUDIO8_MODEL_DIR`) and the fixture files, which are
//! not checked in until that script has been run.
//!
//! ```text
//! IZWI_AUDIO8_MODEL_DIR=path/to/Audio8-TTS-Realistic-Air \
//! cargo test -p izwi-core --lib audio8::reference_tests -- --ignored
//! ```

use std::path::PathBuf;

use serde::Deserialize;

use crate::backends::{BackendPreference, DeviceSelector};
use crate::catalog::ModelVariant;
use crate::error::{Error, Result};
use crate::runtime::{GenerationRequest, RuntimeService};
use crate::{ContextLengthPreference, EngineConfig};

use super::{Audio8Config, Audio8PromptTokenizer, Audio8TtsModel, Audio8VqCodes};

const SAMPLES_PER_FRAME: usize = 2048;

fn model_dir() -> Result<PathBuf> {
    std::env::var_os("IZWI_AUDIO8_MODEL_DIR")
        .map(PathBuf::from)
        .filter(|path| path.is_dir())
        .ok_or_else(|| {
            Error::InvalidInput("Set IZWI_AUDIO8_MODEL_DIR to the pinned Audio8 bundle".into())
        })
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/audio8")
}

fn fixture<T: for<'de> Deserialize<'de>>(name: &str) -> Result<T> {
    let path = fixture_dir().join(name);
    let raw = std::fs::read_to_string(&path).map_err(|error| {
        Error::InvalidInput(format!(
            "{}: {error}; run scripts/fixtures/generate_audio8_reference.py first",
            path.display()
        ))
    })?;
    serde_json::from_str(&raw).map_err(|error| Error::InvalidInput(error.to_string()))
}

#[derive(Deserialize)]
struct WaveformPoints {
    samples: usize,
    index: Vec<usize>,
    values: Vec<f32>,
    rms: f64,
}

impl WaveformPoints {
    /// The largest difference between `actual` and the reference at its
    /// sample points, after checking the length and the overall level.
    fn max_abs_difference(&self, actual: &[f32]) -> f32 {
        assert_eq!(actual.len(), self.samples, "waveform length");
        let rms = (actual
            .iter()
            .map(|sample| f64::from(*sample).powi(2))
            .sum::<f64>()
            / actual.len() as f64)
            .sqrt();
        assert!(
            (rms - self.rms).abs() <= 0.02 * self.rms.max(1e-6),
            "rms {rms} vs reference {}",
            self.rms
        );
        self.index
            .iter()
            .zip(&self.values)
            .map(|(index, expected)| (actual[*index] - expected).abs())
            .fold(0.0, f32::max)
    }
}

#[derive(Deserialize)]
struct PromptFixture {
    cases: Vec<PromptCase>,
}

#[derive(Deserialize)]
struct PromptCase {
    text: String,
    reference_text: Option<String>,
    prefix_ids: Vec<u32>,
    suffix_ids: Vec<u32>,
}

/// The prompt's text ids equal the published processor's, with and without
/// a reference: the same parts, tokenized separately, whitespace collapsed.
#[test]
#[ignore = "requires the pinned Audio8 bundle and the generated fixtures"]
fn prompt_ids_match_the_published_processor() -> Result<()> {
    let dir = model_dir()?;
    let config = Audio8Config::load(&dir)?;
    let tokenizer = Audio8PromptTokenizer::load(&dir, &config)?;
    let fixture: PromptFixture = fixture("prompts.json")?;
    assert!(!fixture.cases.is_empty());
    for case in &fixture.cases {
        let (before, after) =
            tokenizer.speech_prompt_text_ids(case.reference_text.as_deref(), &case.text)?;
        assert_eq!(before, case.prefix_ids, "prefix of {:?}", case.text);
        assert_eq!(after, case.suffix_ids, "suffix of {:?}", case.text);
    }
    Ok(())
}

#[derive(Deserialize)]
struct CodecFixture {
    encode: CodecEncode,
    decode: CodecDecode,
}

#[derive(Deserialize)]
struct CodecEncode {
    clip: String,
    clip_samples: usize,
    frames: usize,
    codes: Vec<Vec<u32>>,
}

#[derive(Deserialize)]
struct CodecDecode {
    codes: Vec<Vec<u32>>,
    waveform: WaveformPoints,
}

/// The codec alone: a clip encodes to the reference's codes (a frame may
/// differ where two codebook entries are within rounding of each other),
/// and fixed codes decode to the reference's waveform.
#[test]
#[ignore = "requires the pinned Audio8 bundle and the generated fixtures"]
fn codec_matches_the_published_codec() -> Result<()> {
    let dir = model_dir()?;
    let device = DeviceSelector::detect_for_preference(BackendPreference::Cpu)?;
    let model = Audio8TtsModel::load(&dir, ModelVariant::Audio8TtsRealisticAir, device)?;
    let runtime = model.runtime.as_ref().expect("native modules");
    let fixture: CodecFixture = fixture("codec.json")?;

    let mut reader = hound::WavReader::open(fixture_dir().join(&fixture.encode.clip))
        .map_err(|error| Error::AudioError(error.to_string()))?;
    let spec = reader.spec();
    assert_eq!((spec.channels, spec.bits_per_sample), (1, 16));
    let clip = reader
        .samples::<i16>()
        .map(|sample| sample.map(|sample| f32::from(sample) / 32768.0))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| Error::AudioError(error.to_string()))?;
    assert_eq!(clip.len(), fixture.encode.clip_samples);
    let encoded = runtime
        .dac
        .encode_reference_audio(&clip, spec.sample_rate)?;
    assert_eq!(encoded.codebooks.len(), fixture.encode.codes.len());
    assert_eq!(encoded.frame_count(), fixture.encode.frames);
    let total = fixture.encode.frames * fixture.encode.codes.len();
    let differing = encoded
        .codebooks
        .iter()
        .zip(&fixture.encode.codes)
        .flat_map(|(actual, expected)| actual.iter().zip(expected))
        .filter(|(actual, expected)| actual != expected)
        .count();
    eprintln!("Audio8 codec encode: {differing} of {total} codes differ");
    assert!(
        differing * 50 <= total,
        "{differing} of {total} codes differ from the reference"
    );

    let decoded = runtime.dac.decode_vq_codes(&Audio8VqCodes {
        codebooks: fixture.decode.codes.clone(),
    })?;
    let difference = fixture.decode.waveform.max_abs_difference(&decoded);
    eprintln!("Audio8 codec decode: max abs difference {difference}");
    assert!(difference < 2e-3, "decode differs by {difference}");
    Ok(())
}

#[derive(Deserialize)]
struct GreedyFixture {
    text: String,
    max_frames: usize,
    finished: bool,
    frames: usize,
    waveform: WaveformPoints,
}

/// Greedy generation with no reference, through the runtime: the same
/// number of frames and the same waveform as the reference. A different
/// code anywhere changes every later frame, so this checks the prompt, the
/// slow and fast transformers and the codec together.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires the pinned Audio8 bundle and the generated fixtures"]
async fn greedy_speech_matches_the_published_model() -> Result<()> {
    let variant = ModelVariant::Audio8TtsRealisticAir;
    let dir = model_dir()?;
    let fixture: GreedyFixture = fixture("greedy.json")?;
    assert!(
        fixture.finished,
        "the reference must reach end of speech inside its budget"
    );
    let runtime = RuntimeService::new(EngineConfig {
        models_dir: dir
            .parent()
            .ok_or_else(|| Error::InvalidInput("Missing model parent".into()))?
            .to_path_buf(),
        backend: BackendPreference::Cpu,
        max_sequence_length: ContextLengthPreference::explicit(2048)?,
        ..EngineConfig::default()
    })?;
    let mut request = GenerationRequest::new(fixture.text.clone()).with_model_variant(variant);
    request.config.streaming = false;
    request.config.options.max_tokens = fixture.max_frames;
    request.config.options.temperature = 0.0;
    request.config.options.top_p = 1.0;
    runtime.load_model(variant).await?;
    let generated = runtime.generate(request).await;
    let unloaded = runtime.unload_model(variant).await;
    let output = generated?;
    unloaded?;

    assert_eq!(output.samples.len(), fixture.frames * SAMPLES_PER_FRAME);
    let difference = fixture.waveform.max_abs_difference(&output.samples);
    eprintln!(
        "Audio8 greedy: {} frames, max abs difference {difference}",
        fixture.frames
    );
    assert!(difference < 2e-2, "greedy waveform differs by {difference}");
    Ok(())
}
