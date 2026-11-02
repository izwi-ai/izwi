//! Fish S2's DAC and Fish Audio Drama 3's own copy of it (decision D3) must
//! stay bit-identical: Drama 3 ships the very `codec.pth` S2 Pro does. This
//! is the one module allowed to name both families.
//!
//! Every comparison is exact, on the f32 bits: a whole decode; streaming in
//! uneven chunks, each push preceded by a cancelled one that must roll back;
//! batched pushes of mixed ages beside scalar ones, each step preceded by an
//! aborted batch; the reference encoding of `data/fox.wav` and its decode; and
//! how many cancellation checks each call makes. The configurations, limits,
//! workspace envelopes and memory pricing must agree as well. On the real
//! `codec.pth` (`IZWI_FISH_DRAMA3_CODEC_PATH`) the loaded tensors are compared
//! too; each codec is loaded, run and dropped before the other, so only one is
//! ever resident.

use std::cell::Cell;
use std::collections::hash_map::DefaultHasher;
use std::collections::BTreeSet;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::{Init, VarBuilder};

use super::fish_drama3::codec::{
    decode_workspace_bytes as drama3_decode_workspace_bytes, fft_workspace as drama3_fft_workspace,
    fish_drama3_codec_memory,
    maximum_preparation_workspace_bytes as drama3_maximum_preparation_workspace_bytes,
    preparation_workspace_bytes as drama3_preparation_workspace_bytes,
    streaming_decode_workspace_bytes as drama3_streaming_decode_workspace_bytes,
    FishDrama3CodecMemory, FishDrama3CodecWeights, FishDrama3DacConfig, FishDrama3DacContract,
    FishDrama3DacDecoder, FishDrama3DacStreamState,
};
use super::fish_drama3::prompt::FishDrama3Codes;
use super::fish_s2::codec::{
    decode_workspace_bytes as s2_decode_workspace_bytes, fft_workspace as s2_fft_workspace,
    fish_s2_codec_memory,
    maximum_preparation_workspace_bytes as s2_maximum_preparation_workspace_bytes,
    preparation_workspace_bytes as s2_preparation_workspace_bytes,
    streaming_decode_workspace_bytes as s2_streaming_decode_workspace_bytes, FishS2CodecMemory,
    FishS2CodecWeights,
};
use super::fish_s2::contracts::FishS2DacContract;
use super::fish_s2::dac::{FishS2DacConfig, FishS2DacDecoder, FishS2DacStreamState};
use super::fish_s2::tokenizer::FishS2VqCodes;
use crate::error::{Error, Result};
use crate::models::shared::weights::pytorch::{PthTensorMap, PthTensorSpec};

const CODEC_PATH: &str = "IZWI_FISH_DRAMA3_CODEC_PATH";

type Codes = Vec<Vec<u32>>;
type Cancel<'a> = &'a dyn Fn() -> Result<()>;

/// What the comparison drives on either copy.
trait Codec {
    type State: Clone + Default;
    fn decode(&self, codes: &[Vec<u32>], cancel: Cancel<'_>) -> Result<Vec<f32>>;
    fn push(
        &self,
        state: &mut Self::State,
        codes: &[Vec<u32>],
        cancel: Cancel<'_>,
    ) -> Result<Vec<f32>>;
    fn push_batch(
        &self,
        states: &mut [Self::State],
        codes: &[Codes],
        cancel: Cancel<'_>,
    ) -> Result<Vec<Vec<f32>>>;
    fn encode(&self, samples: &[f32], sample_rate: u32, cancel: Cancel<'_>) -> Result<Codes>;
    /// Frames decoded, samples emitted and bytes retained.
    fn observe(state: &Self::State) -> [u64; 3];
    fn flush(state: &mut Self::State) -> Vec<f32>;
}

impl Codec for FishS2DacDecoder {
    type State = FishS2DacStreamState;

    fn decode(&self, codes: &[Vec<u32>], cancel: Cancel<'_>) -> Result<Vec<f32>> {
        let codes = FishS2VqCodes {
            codebooks: codes.to_vec(),
        };
        self.decode_vq_codes_with_cancel(&codes, cancel)
    }

    fn push(
        &self,
        state: &mut Self::State,
        codes: &[Vec<u32>],
        cancel: Cancel<'_>,
    ) -> Result<Vec<f32>> {
        self.push_frames(state, codes, cancel)
    }

    fn push_batch(
        &self,
        states: &mut [Self::State],
        codes: &[Codes],
        cancel: Cancel<'_>,
    ) -> Result<Vec<Vec<f32>>> {
        self.push_frames_batch(states, codes, cancel)
    }

    fn encode(&self, samples: &[f32], sample_rate: u32, cancel: Cancel<'_>) -> Result<Codes> {
        Ok(self
            .encode_reference_audio_with_cancel(samples, sample_rate, cancel)?
            .codebooks)
    }

    fn observe(state: &Self::State) -> [u64; 3] {
        [
            state.decoded_frames() as u64,
            state.emitted_samples() as u64,
            state.retained_tensor_bytes(),
        ]
    }

    fn flush(state: &mut Self::State) -> Vec<f32> {
        state.flush()
    }
}

impl Codec for FishDrama3DacDecoder {
    type State = FishDrama3DacStreamState;

    fn decode(&self, codes: &[Vec<u32>], cancel: Cancel<'_>) -> Result<Vec<f32>> {
        self.decode_vq_codes_with_cancel(&FishDrama3Codes::new(codes.to_vec()), cancel)
    }

    fn push(
        &self,
        state: &mut Self::State,
        codes: &[Vec<u32>],
        cancel: Cancel<'_>,
    ) -> Result<Vec<f32>> {
        self.push_frames(state, codes, cancel)
    }

    fn push_batch(
        &self,
        states: &mut [Self::State],
        codes: &[Codes],
        cancel: Cancel<'_>,
    ) -> Result<Vec<Vec<f32>>> {
        self.push_frames_batch(states, codes, cancel)
    }

    fn encode(&self, samples: &[f32], sample_rate: u32, cancel: Cancel<'_>) -> Result<Codes> {
        Ok(self
            .encode_reference_audio_with_cancel(samples, sample_rate, cancel)?
            .codebooks)
    }

    fn observe(state: &Self::State) -> [u64; 3] {
        [
            state.decoded_frames() as u64,
            state.emitted_samples() as u64,
            state.retained_tensor_bytes(),
        ]
    }

    fn flush(state: &mut Self::State) -> Vec<f32> {
        state.flush()
    }
}

/// S2's configuration with the same values. Both sides are written out in
/// full, so a field added to either copy fails to compile here until it is
/// mapped.
fn s2_config(config: &FishDrama3DacConfig) -> FishS2DacConfig {
    let FishDrama3DacConfig {
        sample_rate,
        encoder_dim,
        encoder_rates,
        encoder_transformer_layers,
        encoder_window_size,
        latent_dim,
        decoder_dim,
        decoder_rates,
        downsample_factors,
        codebook_dim,
        semantic_codebook_size,
        residual_codebook_size,
        residual_codebooks,
        transformer_layers,
        transformer_heads,
        transformer_kv_heads,
        transformer_head_dim,
        transformer_intermediate,
        transformer_window_size,
        rope_theta,
        rms_norm_eps,
    } = config.clone();
    FishS2DacConfig {
        sample_rate,
        encoder_dim,
        encoder_rates,
        encoder_transformer_layers,
        encoder_window_size,
        latent_dim,
        decoder_dim,
        decoder_rates,
        downsample_factors,
        codebook_dim,
        semantic_codebook_size,
        residual_codebook_size,
        residual_codebooks,
        transformer_layers,
        transformer_heads,
        transformer_kv_heads,
        transformer_head_dim,
        transformer_intermediate,
        transformer_window_size,
        rope_theta,
        rms_norm_eps,
    }
}

/// S2's own synthetic geometry (`tiny_config` in its DAC tests): one stage
/// of everything at unit rates.
fn tiny() -> FishDrama3DacConfig {
    FishDrama3DacConfig {
        sample_rate: 100,
        encoder_dim: 2,
        encoder_rates: vec![1],
        encoder_transformer_layers: vec![0],
        encoder_window_size: 4,
        latent_dim: 4,
        decoder_dim: 4,
        decoder_rates: vec![1],
        downsample_factors: vec![1],
        codebook_dim: 2,
        semantic_codebook_size: 8,
        residual_codebook_size: 8,
        residual_codebooks: 1,
        transformer_layers: 1,
        transformer_heads: 2,
        transformer_kv_heads: 2,
        transformer_head_dim: 2,
        transformer_intermediate: 8,
        transformer_window_size: 4,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-5,
    }
}

/// The published rates and strides (2,048 samples a frame at 44.1 kHz) at
/// toy widths, with an encoder transformer and two grouped-query quantizer
/// layers (four heads over two KV heads, where each copy repeats KV heads its
/// own way), so a reference takes the real resampling and hop.
fn strided() -> FishDrama3DacConfig {
    FishDrama3DacConfig {
        sample_rate: 44_100,
        encoder_dim: 2,
        encoder_rates: vec![2, 4, 8, 8],
        encoder_transformer_layers: vec![0, 0, 0, 1],
        encoder_window_size: 8,
        latent_dim: 8,
        decoder_dim: 32,
        decoder_rates: vec![8, 8, 4, 2],
        downsample_factors: vec![2, 2],
        codebook_dim: 2,
        semantic_codebook_size: 16,
        residual_codebook_size: 8,
        residual_codebooks: 2,
        transformer_layers: 2,
        transformer_heads: 4,
        transformer_kv_heads: 2,
        transformer_head_dim: 2,
        transformer_intermediate: 16,
        transformer_window_size: 4,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-5,
    }
}

/// Deterministic weights keyed by tensor name, so both copies load the same
/// synthetic codec: Snake, norm and scale gains near one, everything else
/// small.
struct XorshiftWeights;

impl candle_nn::var_builder::SimpleBackend for XorshiftWeights {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        _: Init,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        let seed = name.bytes().fold(0xCBF2_9CE4_8422_2325u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01B3)
        });
        let gain =
            name.ends_with("alpha") || name.ends_with("norm.weight") || name.ends_with("gamma");
        let scale = if gain { 0.1 } else { 0.3 };
        let values = crate::kernels::numeric_check::xorshift_tensor(
            shape.dims(),
            seed,
            scale,
            &Device::Cpu,
        )?;
        let values = if gain { (values + 1.0)? } else { values };
        values.to_dtype(dtype)?.to_device(device)
    }

    fn get_unchecked(&self, name: &str, _: DType, _: &Device) -> candle_core::Result<Tensor> {
        candle_core::bail!("unexpected unchecked tensor request {name}")
    }

    /// Every weight, bias and gain exists, so each convolution takes its
    /// fused `conv.weight` form.
    fn contains_tensor(&self, name: &str) -> bool {
        name.ends_with(".weight") || name.ends_with(".bias") || name.ends_with(".gamma")
    }
}

fn synthetic_codecs(
    config: &FishDrama3DacConfig,
    device: &Device,
) -> (FishS2DacDecoder, FishDrama3DacDecoder) {
    let vb = || VarBuilder::from_backend(Box::new(XorshiftWeights), DType::F32, device.clone());
    (
        FishS2DacDecoder::load(s2_config(config), vb()).unwrap(),
        FishDrama3DacDecoder::load(config.clone(), vb()).unwrap(),
    )
}

/// `[codebooks][frames]` xorshift codes, a few past each codebook's end
/// (model output may be; the decoder clamps).
fn xorshift_codes(config: &FishDrama3DacConfig, frames: usize, seed: u64) -> Codes {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..config.num_codebooks())
        .map(|book| {
            let size = if book == 0 {
                config.semantic_codebook_size
            } else {
                config.residual_codebook_size
            };
            let span = (size + size.div_ceil(16)) as u64;
            (0..frames)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state % span) as u32
                })
                .collect()
        })
        .collect()
}

fn fox() -> (Vec<f32>, u32) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/fox.wav");
    crate::audio::decode_audio_bytes_to_mono(&std::fs::read(path).unwrap()).unwrap()
}

/// What both copies are driven through.
struct Workload {
    /// `[codebooks][frames]`.
    codes: Codes,
    /// Streaming push sizes, cycled until every frame is pushed.
    chunks: &'static [usize],
    batch_rows: usize,
    batch_steps: usize,
    /// Mono samples and their rate, to encode.
    reference: Option<(Vec<f32>, u32)>,
}

/// Everything a workload observed, as bits, in order.
#[derive(Default)]
struct Trace(Vec<(String, Vec<u64>)>);

impl Trace {
    fn pcm(&mut self, label: String, samples: &[f32]) {
        assert!(
            samples.iter().all(|sample| sample.is_finite()),
            "{label}: non-finite audio"
        );
        self.values(
            label,
            samples.iter().map(|sample| u64::from(sample.to_bits())),
        );
    }

    fn values(&mut self, label: String, values: impl IntoIterator<Item = u64>) {
        self.0.push((label, values.into_iter().collect()));
    }
}

fn assert_identical(s2: &Trace, drama3: &Trace) {
    assert_eq!(s2.0.len(), drama3.0.len(), "the traces differ in length");
    for ((label, left), (other, right)) in s2.0.iter().zip(&drama3.0) {
        assert_eq!(label, other);
        assert_eq!(left.len(), right.len(), "{label}: lengths differ");
        if let Some(index) = left.iter().zip(right).position(|(a, b)| a != b) {
            panic!(
                "{label}: first difference at {index}: S2 {:#x}, Drama 3 {:#x}",
                left[index], right[index]
            );
        }
    }
}

/// A cancellation check that counts its calls and fails on call `fail_at`.
struct Checks {
    calls: Cell<u64>,
    fail_at: Option<u64>,
}

impl Checks {
    fn counting() -> Self {
        Self {
            calls: Cell::new(0),
            fail_at: None,
        }
    }

    fn failing_at(call: u64) -> Self {
        Self {
            calls: Cell::new(0),
            fail_at: Some(call),
        }
    }

    fn check(&self) -> Result<()> {
        let call = self.calls.get() + 1;
        self.calls.set(call);
        if self.fail_at == Some(call) {
            return Err(Error::InferenceError("cancelled by the test".into()));
        }
        Ok(())
    }
}

fn window(codes: &Codes, start: usize, len: usize) -> Codes {
    let end = (start + len).min(codes[0].len());
    codes.iter().map(|row| row[start..end].to_vec()).collect()
}

fn run<C: Codec>(codec: &C, work: &Workload) -> Trace {
    let mut trace = Trace::default();
    let frames = work.codes[0].len();

    let checks = Checks::counting();
    let whole = codec.decode(&work.codes, &|| checks.check()).unwrap();
    assert!(whole.iter().any(|sample| sample.abs() > 1e-4));
    trace.pcm("whole decode".into(), &whole);
    trace.values("whole decode checks".into(), [checks.calls.get()]);

    let mut state = C::State::default();
    let mut start = 0;
    for (push, &len) in work.chunks.iter().cycle().enumerate() {
        if start == frames {
            break;
        }
        let rows = window(&work.codes, start, len);
        let before = C::observe(&state);
        let cancel = Checks::failing_at([2, 7, 12][push % 3]);
        assert!(
            codec.push(&mut state, &rows, &|| cancel.check()).is_err(),
            "push {push} was not cancelled"
        );
        assert_eq!(C::observe(&state), before, "push {push} did not roll back");
        let checks = Checks::counting();
        let pcm = codec.push(&mut state, &rows, &|| checks.check()).unwrap();
        trace.pcm(format!("push {push}"), &pcm);
        trace.values(
            format!("push {push} state and checks"),
            C::observe(&state).into_iter().chain([checks.calls.get()]),
        );
        start += rows[0].len();
    }
    trace.pcm("flush".into(), &C::flush(&mut state));
    assert!(codec
        .push(&mut state, &window(&work.codes, 0, 1), &|| Ok(()))
        .is_err());

    // Rows of mixed ages, reordered every step, batched beside scalar.
    let mut batch = vec![C::State::default(); work.batch_rows];
    let mut scalar = batch.clone();
    for step in 0..work.batch_steps {
        let codes = (0..work.batch_rows)
            .map(|row| {
                let len = if row == 0 { 1 } else { 3 };
                window(&work.codes, (step * 3 + row * 5) % (frames - 3), len)
            })
            .collect::<Vec<_>>();
        let before = batch.iter().map(C::observe).collect::<Vec<_>>();
        let cancel = Checks::failing_at(17);
        assert!(
            codec
                .push_batch(&mut batch, &codes, &|| cancel.check())
                .is_err(),
            "step {step}: the batch was not cancelled"
        );
        assert_eq!(batch.iter().map(C::observe).collect::<Vec<_>>(), before);
        for (row, (state, codes)) in scalar.iter_mut().zip(&codes).enumerate() {
            let pcm = codec.push(state, codes, &|| Ok(())).unwrap();
            trace.pcm(format!("step {step} scalar row {row}"), &pcm);
        }
        let checks = Checks::counting();
        let outputs = codec
            .push_batch(&mut batch, &codes, &|| checks.check())
            .unwrap();
        for (row, pcm) in outputs.iter().enumerate() {
            trace.pcm(format!("step {step} batch row {row}"), pcm);
        }
        trace.values(
            format!("step {step} batch states and checks"),
            batch
                .iter()
                .flat_map(C::observe)
                .chain([checks.calls.get()]),
        );
        batch.reverse();
        scalar.reverse();
    }

    if let Some((samples, sample_rate)) = &work.reference {
        let checks = Checks::counting();
        let codes = codec
            .encode(samples, *sample_rate, &|| checks.check())
            .unwrap();
        assert!(codes.iter().flatten().collect::<BTreeSet<_>>().len() > 1);
        trace.values(
            "reference shape and checks".into(),
            [
                codes.len() as u64,
                codes[0].len() as u64,
                checks.calls.get(),
            ],
        );
        trace.values(
            "reference codes".into(),
            codes.iter().flatten().map(|&code| u64::from(code)),
        );
        let pcm = codec.decode(&codes, &|| Ok(())).unwrap();
        trace.pcm("reference decode".into(), &pcm);
    }
    trace
}

fn synthetic_workloads(device: &Device) {
    let tiny = tiny();
    let (s2, drama3) = synthetic_codecs(&tiny, device);
    let work = Workload {
        codes: xorshift_codes(&tiny, 103, 7),
        chunks: &[1, 3, 8, 2, 32, 5, 13, 7, 1, 31],
        batch_rows: 7,
        batch_steps: 12,
        reference: None,
    };
    assert_identical(&run(&s2, &work), &run(&drama3, &work));

    let strided = strided();
    let (s2, drama3) = synthetic_codecs(&strided, device);
    let work = Workload {
        codes: xorshift_codes(&strided, 40, 11),
        chunks: &[1, 3, 8, 2, 13, 5, 1, 7],
        batch_rows: 4,
        batch_steps: 4,
        reference: Some(fox()),
    };
    assert_identical(&run(&s2, &work), &run(&drama3, &work));
}

fn spec(name: &str, shape: &[usize], dtype: DType) -> PthTensorSpec {
    PthTensorSpec {
        name: name.to_owned(),
        shape: shape.to_vec(),
        dtype,
        archive_member_path: format!("synthetic/{name}"),
    }
}

fn s2_memory(specs: &[PthTensorSpec]) -> Option<[u64; 5]> {
    let FishS2CodecMemory {
        resident_parameter_bytes,
        raw_load_bytes,
        fused_load_bytes,
        largest_source_tensor_bytes,
        largest_target_tensor_bytes,
    } = fish_s2_codec_memory(specs).ok()?;
    Some([
        resident_parameter_bytes,
        raw_load_bytes,
        fused_load_bytes,
        largest_source_tensor_bytes,
        largest_target_tensor_bytes,
    ])
}

fn drama3_memory(specs: &[PthTensorSpec]) -> Option<[u64; 5]> {
    let FishDrama3CodecMemory {
        resident_parameter_bytes,
        raw_load_bytes,
        fused_load_bytes,
        largest_source_tensor_bytes,
        largest_target_tensor_bytes,
    } = fish_drama3_codec_memory(specs).ok()?;
    Some([
        resident_parameter_bytes,
        raw_load_bytes,
        fused_load_bytes,
        largest_source_tensor_bytes,
        largest_target_tensor_bytes,
    ])
}

/// The loaded tensors by name: shape and a digest of their bits.
fn weight_trace<'a>(tensors: impl Iterator<Item = (&'a str, &'a Tensor)>) -> Trace {
    let mut tensors = tensors.collect::<Vec<_>>();
    tensors.sort_by_key(|(name, _)| *name);
    let mut trace = Trace::default();
    for (name, tensor) in tensors {
        assert_eq!(tensor.dtype(), DType::F32, "{name}");
        let bits = tensor
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        let mut hasher = DefaultHasher::new();
        bits.hash(&mut hasher);
        trace.values(
            format!("weight {name}"),
            tensor
                .dims()
                .iter()
                .map(|&dim| dim as u64)
                .chain([hasher.finish()]),
        );
    }
    trace
}

fn real_workload() -> Workload {
    Workload {
        // Past the post-transformer's 128-frame window.
        codes: xorshift_codes(&FishDrama3DacConfig::current(), 150, 13),
        chunks: &[1, 4, 16, 3, 32, 7, 64, 23],
        batch_rows: 3,
        batch_steps: 3,
        reference: Some(fox()),
    }
}

/// Loads, traces and drops S2's codec, then Drama 3's.
fn real_codecs(path: &Path, device: &Device) {
    let work = real_workload();
    let started = std::time::Instant::now();
    let s2 = {
        let weights = FishS2CodecWeights::load(path, device, DType::F32).unwrap();
        let mut trace = weight_trace(
            weights
                .tensor_names()
                .map(|name| (name, weights.tensor(name).unwrap())),
        );
        let codec =
            FishS2DacDecoder::load(FishS2DacConfig::current(), weights.var_builder()).unwrap();
        drop(weights);
        trace.0.extend(run(&codec, &work).0);
        trace
    };
    let s2_elapsed = started.elapsed();
    let started = std::time::Instant::now();
    let drama3 = {
        let weights = FishDrama3CodecWeights::load(path, device, DType::F32).unwrap();
        let mut trace = weight_trace(
            weights
                .tensor_names()
                .map(|name| (name, weights.tensor(name).unwrap())),
        );
        let codec =
            FishDrama3DacDecoder::load(FishDrama3DacConfig::current(), weights.var_builder())
                .unwrap();
        drop(weights);
        trace.0.extend(run(&codec, &work).0);
        trace
    };
    eprintln!(
        "real codec on {device:?}: S2 {:.1} s, Drama 3 {:.1} s, {} traced entries",
        s2_elapsed.as_secs_f64(),
        started.elapsed().as_secs_f64(),
        s2.0.len()
    );
    assert_identical(&s2, &drama3);
}

fn real_codec_path() -> PathBuf {
    PathBuf::from(
        std::env::var_os(CODEC_PATH)
            .unwrap_or_else(|| panic!("set {CODEC_PATH} to the published codec.pth")),
    )
}

#[test]
fn the_copies_configure_and_price_the_codec_identically() {
    assert_eq!(
        s2_config(&FishDrama3DacConfig::current()),
        FishS2DacConfig::current()
    );
    assert_eq!(
        (
            FishDrama3DacConfig::MAX_ENCODER_FRAMES,
            FishDrama3DacConfig::MAX_QUANTIZER_FRAMES,
            FishDrama3DacConfig::MAX_REFERENCE_SAMPLE_RATE,
        ),
        (
            FishS2DacConfig::MAX_ENCODER_FRAMES,
            FishS2DacConfig::MAX_QUANTIZER_FRAMES,
            FishS2DacConfig::MAX_REFERENCE_SAMPLE_RATE,
        )
    );
    let FishS2DacContract {
        sample_rate,
        encoder_rates,
        decoder_rates,
        semantic_codebooks,
        residual_codebooks,
        semantic_codebook_size,
        residual_codebook_size,
        codebook_dim,
        latent_dim,
    } = FishS2DacContract::CURRENT;
    let s2_contract = (
        sample_rate,
        encoder_rates,
        decoder_rates,
        semantic_codebooks,
        residual_codebooks,
        semantic_codebook_size,
        residual_codebook_size,
        codebook_dim,
        latent_dim,
    );
    let FishDrama3DacContract {
        sample_rate,
        encoder_rates,
        decoder_rates,
        semantic_codebooks,
        residual_codebooks,
        semantic_codebook_size,
        residual_codebook_size,
        codebook_dim,
        latent_dim,
    } = FishDrama3DacContract::CURRENT;
    assert_eq!(
        s2_contract,
        (
            sample_rate,
            encoder_rates,
            decoder_rates,
            semantic_codebooks,
            residual_codebooks,
            semantic_codebook_size,
            residual_codebook_size,
            codebook_dim,
            latent_dim,
        )
    );

    let samples = [0, 1, 2_047, 2_048, 86_400, 1 << 20, 50_000_000];
    let rates = [0, 1, 8_000, 24_000, 44_100, 48_000, 384_000, 384_001];
    for drama3 in [tiny(), strided(), FishDrama3DacConfig::current()] {
        let s2 = s2_config(&drama3);
        assert_eq!(s2.num_codebooks(), drama3.num_codebooks());
        assert_eq!(s2.samples_per_frame().ok(), drama3.samples_per_frame().ok());
        assert_eq!(
            s2.maximum_reference_samples().ok(),
            drama3.maximum_reference_samples().ok()
        );
        assert_eq!(
            s2.rotary_cache_bytes().ok(),
            drama3.rotary_cache_bytes().ok()
        );
        assert_eq!(
            s2.streaming_history_bound_bytes().ok(),
            drama3.streaming_history_bound_bytes().ok()
        );
        for count in samples {
            for rate in rates {
                assert_eq!(
                    s2.reference_frame_count(count, rate).ok(),
                    drama3.reference_frame_count(count, rate).ok(),
                    "{count} samples at {rate} Hz"
                );
            }
        }
    }
    for count in samples {
        for rate in rates {
            assert_eq!(
                s2_preparation_workspace_bytes(count, rate).ok(),
                drama3_preparation_workspace_bytes(count, rate).ok(),
                "{count} samples at {rate} Hz"
            );
            if rate == 0 {
                // Neither prices the FFT of a rate refused earlier.
                continue;
            }
            assert_eq!(
                s2_fft_workspace(count as u64, rate).ok(),
                drama3_fft_workspace(count as u64, rate).ok(),
                "{count} samples at {rate} Hz"
            );
        }
    }
    for frames in [0, 1, 4, 16, 127, 128, 143, 2_048, 4_095, 4_096, 4_097] {
        assert_eq!(
            s2_decode_workspace_bytes(frames).ok(),
            drama3_decode_workspace_bytes(frames).ok(),
            "{frames} frames"
        );
        assert_eq!(
            s2_streaming_decode_workspace_bytes(frames).ok(),
            drama3_streaming_decode_workspace_bytes(frames).ok(),
            "{frames} frames"
        );
    }
    assert_eq!(
        s2_maximum_preparation_workspace_bytes().ok(),
        drama3_maximum_preparation_workspace_bytes().ok()
    );

    let original = |base: &str, index: u8, shape: &[usize], dtype| {
        spec(
            &format!("{base}.parametrizations.weight.original{index}"),
            shape,
            dtype,
        )
    };
    for specs in [
        vec![
            original("generator.encoder.conv", 0, &[2, 1, 1], DType::F32),
            original("generator.encoder.conv", 1, &[2, 3, 1], DType::F16),
            spec("generator.encoder.conv.bias", &[2], DType::F32),
            spec("generator.encoder.freqs_cis", &[2, 2, 2], DType::BF16),
            spec("generator.decoder.causal_mask", &[64, 64], DType::U8),
            spec("discriminator.weight", &[4_096], DType::F32),
            spec(
                "generator.quantizer.in_proj.weight_g",
                &[2, 1, 1],
                DType::F32,
            ),
            spec(
                "generator.quantizer.in_proj.weight_v",
                &[2, 2, 1],
                DType::F32,
            ),
        ],
        vec![
            spec("encoder.conv.weight", &[2, 3, 1], DType::F32),
            spec("encoder.conv.weight_g", &[2, 1, 1], DType::F32),
            spec("encoder.conv.weight_v", &[2, 3, 1], DType::F32),
        ],
        vec![spec("encoder.conv.weight_g", &[2], DType::F32)],
        vec![spec("encoder.weight", &[usize::MAX], DType::F32)],
    ] {
        assert_eq!(s2_memory(&specs), drama3_memory(&specs), "{specs:?}");
    }
}

#[test]
fn synthetic_codecs_are_bit_identical() {
    synthetic_workloads(&Device::Cpu);
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires an available Metal device; never falls back to CPU"]
fn metal_synthetic_codecs_are_bit_identical() {
    let (device, _lease) =
        crate::backends::metal_test_device_or_skip("metal_synthetic_codecs_are_bit_identical")
            .expect("Metal device");
    synthetic_workloads(&device);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an available CUDA device; never falls back to CPU"]
fn cuda_synthetic_codecs_are_bit_identical() {
    synthetic_workloads(&Device::new_cuda(0).expect("CUDA device"));
}

/// Both copies price the published archive alike, from its metadata.
fn real_pricing(path: &Path) {
    let specs =
        PthTensorMap::open_first_non_empty(path, &[None, Some("state_dict"), Some("generator")])
            .unwrap()
            .specs();
    assert!(specs.len() > 500, "{} tensors", specs.len());
    let priced = s2_memory(&specs);
    assert!(priced.is_some());
    assert_eq!(priced, drama3_memory(&specs));
}

#[test]
#[ignore = "needs the published codec.pth in IZWI_FISH_DRAMA3_CODEC_PATH"]
fn real_codecs_are_bit_identical_on_cpu() {
    let path = real_codec_path();
    real_pricing(&path);
    real_codecs(&path, &Device::Cpu);
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "needs a Metal device and the published codec.pth in IZWI_FISH_DRAMA3_CODEC_PATH"]
fn metal_real_codecs_are_bit_identical() {
    let path = real_codec_path();
    let (device, _lease) =
        crate::backends::metal_test_device_or_skip("metal_real_codecs_are_bit_identical")
            .expect("Metal device");
    real_codecs(&path, &device);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a CUDA device and the published codec.pth in IZWI_FISH_DRAMA3_CODEC_PATH"]
fn cuda_real_codecs_are_bit_identical() {
    let path = real_codec_path();
    real_codecs(&path, &Device::new_cuda(0).expect("CUDA device"));
}
