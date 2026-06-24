//! Kokoro-82M native runtime integration scaffolding (Rust-only).
//!
//! This module intentionally isolates Kokoro-specific loading, phonemization,
//! voice-pack handling, and future Candle inference implementation from the
//! generic runtime orchestration layer.

mod albert;
mod config;
mod decoder;
mod phonemizer;
mod prosody;
mod text_encoder;
mod voice;

pub use config::KokoroConfig;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use candle_core::pickle::read_pth_tensor_info;
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use tracing::info;

use crate::backends::DeviceProfile;
use crate::error::{Error, Result};

use self::phonemizer::EspeakPhonemizer;
use self::prosody::{
    build_alignment_matrix, KokoroProsodyDebugOutput, KokoroProsodyOutput, KokoroProsodyPredictor,
};
use self::text_encoder::KokoroTextEncoder;
use self::voice::VoiceLibrary;

const CHECKPOINT_FILE: &str = "kokoro-v1_0.pth";
const CONFIG_FILE: &str = "config.json";
const VOICES_DIR: &str = "voices";
const CHECKPOINT_SUBMODULE_KEYS: &[&str] = &[
    "bert",
    "bert_encoder",
    "predictor",
    "text_encoder",
    "decoder",
];

fn kokoro_profile_enabled() -> bool {
    std::env::var_os("IZWI_KOKORO_PROFILE").is_some()
}

fn log_kokoro_profile(stage: &str, dur: Duration) {
    if kokoro_profile_enabled() {
        eprintln!(
            "kokoro profile: {stage} = {:.2} ms",
            dur.as_secs_f64() * 1_000.0
        );
    }
}

fn kokoro_cpu_predecoder_parallel_enabled() -> bool {
    match std::env::var("IZWI_KOKORO_CPU_PREDECODER") {
        Ok(value) => !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "serial" | "sequential"
        ),
        Err(_) => true,
    }
}

#[derive(Debug, Clone)]
pub struct KokoroPreparedRequest {
    pub phonemes: String,
    pub token_ids: Vec<u32>,
    pub ref_style: Tensor,
    pub speed: f32,
}

#[derive(Debug, Clone)]
pub struct KokoroSynthesisResult {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub tokens_generated: usize,
    pub phonemes: String,
}

#[derive(Debug, Clone)]
pub struct KokoroPredecoderDebugOutput {
    pub prosody: KokoroProsodyDebugOutput,
    pub text_encoder_shape: Vec<usize>,
    pub asr_shape: Vec<usize>,
}

#[derive(Debug, Clone)]
struct KokoroPredecoderOutput {
    prosody: KokoroProsodyOutput,
    text_encoder_shape: Vec<usize>,
    asr: Tensor,
}

#[derive(Debug)]
pub struct KokoroTtsModel {
    model_dir: PathBuf,
    checkpoint_path: PathBuf,
    config: KokoroConfig,
    device: DeviceProfile,
    dtype: DType,
    bert: albert::CustomAlbert,
    bert_encoder: Linear,
    prosody: KokoroProsodyPredictor,
    text_encoder: KokoroTextEncoder,
    decoder: decoder::KokoroDecoder,
    phonemizer: EspeakPhonemizer,
    voices: VoiceLibrary,
    checkpoint_tensor_counts: HashMap<String, usize>,
}

impl KokoroTtsModel {
    pub fn load(model_dir: &Path, device: DeviceProfile) -> Result<Self> {
        let config_path = model_dir.join(CONFIG_FILE);
        let checkpoint_path = model_dir.join(CHECKPOINT_FILE);
        let voices_dir = model_dir.join(VOICES_DIR);

        if !config_path.exists() {
            return Err(Error::ModelNotFound(format!(
                "Missing Kokoro config.json at {}",
                config_path.display()
            )));
        }
        if !checkpoint_path.exists() {
            return Err(Error::ModelNotFound(format!(
                "Missing Kokoro checkpoint at {}",
                checkpoint_path.display()
            )));
        }
        if !voices_dir.exists() {
            return Err(Error::ModelNotFound(format!(
                "Missing Kokoro voices directory at {}",
                voices_dir.display()
            )));
        }

        let config: KokoroConfig =
            serde_json::from_str(&std::fs::read_to_string(&config_path).map_err(|e| {
                Error::ModelLoadError(format!(
                    "Failed reading Kokoro config {}: {}",
                    config_path.display(),
                    e
                ))
            })?)?;

        let dtype = DType::F32;
        let checkpoint_tensor_counts =
            inspect_and_validate_checkpoint(&checkpoint_path, &device.device, dtype)?;

        let phonemizer = EspeakPhonemizer::auto()?;
        let voices = VoiceLibrary::new(voices_dir, device.device.clone(), dtype)?;
        let bert = {
            let vb =
                VarBuilder::from_pth_with_state(&checkpoint_path, dtype, "bert", &device.device)
                    .map_err(|e| {
                        Error::ModelLoadError(format!(
                            "Failed to create Kokoro BERT VarBuilder for {}: {}",
                            checkpoint_path.display(),
                            e
                        ))
                    })?;
            albert::CustomAlbert::load(
                &albert::AlbertModelConfig::from_kokoro(&config),
                vb.pp("module"),
            )?
        };
        let bert_encoder = {
            let vb = VarBuilder::from_pth_with_state(
                &checkpoint_path,
                dtype,
                "bert_encoder",
                &device.device,
            )
            .map_err(|e| {
                Error::ModelLoadError(format!(
                    "Failed to create Kokoro bert_encoder VarBuilder for {}: {}",
                    checkpoint_path.display(),
                    e
                ))
            })?;
            candle_nn::linear(
                config.plbert.hidden_size,
                config.hidden_dim,
                vb.pp("module"),
            )
            .map_err(Error::from)?
        };
        let prosody = {
            let vb = VarBuilder::from_pth_with_state(
                &checkpoint_path,
                dtype,
                "predictor",
                &device.device,
            )
            .map_err(|e| {
                Error::ModelLoadError(format!(
                    "Failed to create Kokoro predictor VarBuilder for {}: {}",
                    checkpoint_path.display(),
                    e
                ))
            })?;
            KokoroProsodyPredictor::load(&config, vb)?
        };
        let text_encoder = {
            let vb = VarBuilder::from_pth_with_state(
                &checkpoint_path,
                dtype,
                "text_encoder",
                &device.device,
            )
            .map_err(|e| {
                Error::ModelLoadError(format!(
                    "Failed to create Kokoro text_encoder VarBuilder for {}: {}",
                    checkpoint_path.display(),
                    e
                ))
            })?;
            KokoroTextEncoder::load(&config, vb)?
        };
        let decoder = {
            let vb =
                VarBuilder::from_pth_with_state(&checkpoint_path, dtype, "decoder", &device.device)
                    .map_err(|e| {
                        Error::ModelLoadError(format!(
                            "Failed to create Kokoro decoder VarBuilder for {}: {}",
                            checkpoint_path.display(),
                            e
                        ))
                    })?;
            decoder::KokoroDecoder::load(&config, vb)?
        };

        info!(
            "Loaded Kokoro scaffolding from {:?} (phonemizer={}, submodules={:?})",
            model_dir,
            phonemizer.bin_path().display(),
            checkpoint_tensor_counts
        );

        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            checkpoint_path,
            config,
            device,
            dtype,
            bert,
            bert_encoder,
            prosody,
            text_encoder,
            decoder,
            phonemizer,
            voices,
            checkpoint_tensor_counts,
        })
    }

    pub fn available_speakers(&self) -> Result<Vec<String>> {
        self.voices.list_speakers()
    }

    pub fn prepare_request(
        &self,
        text: &str,
        speaker: Option<&str>,
        language: Option<&str>,
        speed: f32,
    ) -> Result<KokoroPreparedRequest> {
        let speaker = self.resolve_speaker(speaker)?;
        let phonemes = self.phonemizer.phonemize(text, language, Some(&speaker))?;
        let phoneme_len = phonemes.chars().count();
        if phoneme_len == 0 {
            return Err(Error::InvalidInput(
                "Kokoro phonemizer produced no phonemes".to_string(),
            ));
        }
        if phoneme_len > 510 {
            return Err(Error::InvalidInput(format!(
                "Kokoro phoneme sequence length {} exceeds supported voice-pack limit (510). Chunking is not implemented yet in the native runtime.",
                phoneme_len
            )));
        }

        let token_ids = self.token_ids_from_phonemes(&phonemes)?;
        if token_ids.len() + 2 > self.config.context_length() {
            return Err(Error::InvalidInput(format!(
                "Kokoro phoneme token length {} exceeds context length {}",
                token_ids.len() + 2,
                self.config.context_length()
            )));
        }

        let ref_style = self.voices.style_for_phoneme_len(&speaker, phoneme_len)?;
        let speed = speed.clamp(0.5, 2.0);

        Ok(KokoroPreparedRequest {
            phonemes,
            token_ids,
            ref_style,
            speed,
        })
    }

    pub fn generate(
        &self,
        text: &str,
        speaker: Option<&str>,
        language: Option<&str>,
        speed: f32,
    ) -> Result<KokoroSynthesisResult> {
        let t0 = Instant::now();
        let prepared = self.prepare_request(text, speaker, language, speed)?;
        log_kokoro_profile("tts.prepare_request", t0.elapsed());
        let t1 = Instant::now();
        let predecoder = self.run_predecoder(&prepared)?;
        log_kokoro_profile("tts.predecoder", t1.elapsed());
        let style = prepared
            .ref_style
            .i((.., 0..self.config.style_dim))
            .map_err(Error::from)?;
        let t2 = Instant::now();
        let samples = self.decoder.forward(
            &predecoder.asr,
            &predecoder.prosody.f0,
            &predecoder.prosody.n,
            &style,
        )?;
        log_kokoro_profile("tts.decoder", t2.elapsed());
        log_kokoro_profile("tts.total", t0.elapsed());
        Ok(KokoroSynthesisResult {
            tokens_generated: prepared.token_ids.len(),
            phonemes: prepared.phonemes,
            sample_rate: KokoroConfig::TARGET_SAMPLE_RATE,
            samples,
        })
    }

    #[cfg(test)]
    fn generate_with_seed_for_test(
        &self,
        text: &str,
        speaker: Option<&str>,
        language: Option<&str>,
        speed: f32,
        rng_seed: u64,
    ) -> Result<KokoroSynthesisResult> {
        let t0 = Instant::now();
        let prepared = self.prepare_request(text, speaker, language, speed)?;
        log_kokoro_profile("tts.prepare_request", t0.elapsed());
        let t1 = Instant::now();
        let predecoder = self.run_predecoder(&prepared)?;
        log_kokoro_profile("tts.predecoder", t1.elapsed());
        let style = prepared
            .ref_style
            .i((.., 0..self.config.style_dim))
            .map_err(Error::from)?;
        let t2 = Instant::now();
        let samples = self.decoder.forward_with_seed(
            &predecoder.asr,
            &predecoder.prosody.f0,
            &predecoder.prosody.n,
            &style,
            Some(rng_seed),
        )?;
        log_kokoro_profile("tts.decoder", t2.elapsed());
        log_kokoro_profile("tts.total", t0.elapsed());
        Ok(KokoroSynthesisResult {
            tokens_generated: prepared.token_ids.len(),
            phonemes: prepared.phonemes,
            sample_rate: KokoroConfig::TARGET_SAMPLE_RATE,
            samples,
        })
    }

    pub fn config(&self) -> &KokoroConfig {
        &self.config
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn checkpoint_path(&self) -> &Path {
        &self.checkpoint_path
    }

    pub fn checkpoint_tensor_counts(&self) -> &HashMap<String, usize> {
        &self.checkpoint_tensor_counts
    }

    pub fn run_bert_prosody_debug(
        &self,
        prepared: &KokoroPreparedRequest,
    ) -> Result<KokoroProsodyDebugOutput> {
        let input_ids = self.build_model_input_ids(prepared)?;
        self.run_bert_prosody_debug_for_input(&input_ids, prepared)
    }

    fn run_bert_prosody(&self, prepared: &KokoroPreparedRequest) -> Result<KokoroProsodyOutput> {
        let input_ids = self.build_model_input_ids(prepared)?;
        self.run_bert_prosody_for_input(&input_ids, prepared)
    }

    fn run_bert_prosody_debug_for_input(
        &self,
        input_ids: &Tensor,
        prepared: &KokoroPreparedRequest,
    ) -> Result<KokoroProsodyDebugOutput> {
        let bert_hidden = self.bert.forward(&input_ids, None)?;
        let d_en = self
            .bert_encoder
            .forward(&bert_hidden)
            .map_err(Error::from)?
            .transpose(1, 2)
            .map_err(Error::from)?;
        self.prosody
            .forward_debug(&d_en, &prepared.ref_style, prepared.speed)
    }

    fn run_bert_prosody_for_input(
        &self,
        input_ids: &Tensor,
        prepared: &KokoroPreparedRequest,
    ) -> Result<KokoroProsodyOutput> {
        let bert_hidden = self.bert.forward(input_ids, None)?;
        let d_en = self
            .bert_encoder
            .forward(&bert_hidden)
            .map_err(Error::from)?
            .transpose(1, 2)
            .map_err(Error::from)?;
        self.prosody
            .forward(&d_en, &prepared.ref_style, prepared.speed)
    }

    pub fn run_predecoder_debug(
        &self,
        prepared: &KokoroPreparedRequest,
    ) -> Result<KokoroPredecoderDebugOutput> {
        let out = self.run_predecoder(prepared)?;
        Ok(KokoroPredecoderDebugOutput {
            prosody: KokoroProsodyDebugOutput {
                duration_frames: out.prosody.duration_frames.clone(),
                expanded_frames: out.prosody.expanded_frames,
                f0_shape: out.prosody.f0.shape().dims().to_vec(),
                n_shape: out.prosody.n.shape().dims().to_vec(),
            },
            text_encoder_shape: out.text_encoder_shape,
            asr_shape: out.asr.shape().dims().to_vec(),
        })
    }

    fn run_predecoder(&self, prepared: &KokoroPreparedRequest) -> Result<KokoroPredecoderOutput> {
        let input_ids = self.build_model_input_ids(prepared)?;
        let (prosody, t_en) = self.run_predecoder_branches(&input_ids, prepared)?;
        let pred_aln = build_alignment_matrix(&prosody.duration_frames, &self.device.device)?;
        let asr = t_en
            .contiguous()
            .map_err(Error::from)?
            .matmul(&pred_aln.contiguous().map_err(Error::from)?)
            .map_err(Error::from)?;
        let text_encoder_shape = t_en.shape().dims().to_vec();
        Ok(KokoroPredecoderOutput {
            prosody,
            text_encoder_shape,
            asr,
        })
    }

    fn run_predecoder_branches(
        &self,
        input_ids: &Tensor,
        prepared: &KokoroPreparedRequest,
    ) -> Result<(KokoroProsodyOutput, Tensor)> {
        if input_ids.device().is_cpu() && kokoro_cpu_predecoder_parallel_enabled() {
            return self.run_predecoder_branches_parallel_cpu(input_ids, prepared);
        }

        let prosody = self.run_bert_prosody_for_input(input_ids, prepared)?;
        let t_en = self.text_encoder.forward(input_ids)?;
        Ok((prosody, t_en))
    }

    fn run_predecoder_branches_parallel_cpu(
        &self,
        input_ids: &Tensor,
        prepared: &KokoroPreparedRequest,
    ) -> Result<(KokoroProsodyOutput, Tensor)> {
        thread::scope(|scope| {
            let prosody_handle = scope.spawn(|| {
                self.run_bert_prosody_for_input(input_ids, prepared)
                    .map_err(|e| e.to_string())
            });
            let text_handle = scope.spawn(|| {
                self.text_encoder
                    .forward(input_ids)
                    .map_err(Error::from)
                    .map_err(|e| e.to_string())
            });

            let prosody = match prosody_handle.join() {
                Ok(Ok(t)) => t,
                Ok(Err(msg)) => return Err(Error::InferenceError(msg)),
                Err(_) => {
                    return Err(Error::InferenceError(
                        "Kokoro predecoder prosody worker thread panicked".to_string(),
                    ))
                }
            };
            let t_en = match text_handle.join() {
                Ok(Ok(t)) => t,
                Ok(Err(msg)) => return Err(Error::InferenceError(msg)),
                Err(_) => {
                    return Err(Error::InferenceError(
                        "Kokoro predecoder text encoder worker thread panicked".to_string(),
                    ))
                }
            };

            Ok((prosody, t_en))
        })
    }

    fn build_model_input_ids(&self, prepared: &KokoroPreparedRequest) -> Result<Tensor> {
        let mut input_ids = Vec::with_capacity(prepared.token_ids.len() + 2);
        input_ids.push(0u32);
        input_ids.extend_from_slice(&prepared.token_ids);
        input_ids.push(0u32);
        let seq_len = input_ids.len();
        Tensor::from_vec(input_ids, (1, seq_len), &self.device.device).map_err(Error::from)
    }

    fn resolve_speaker(&self, requested: Option<&str>) -> Result<String> {
        let speakers = self.available_speakers()?;
        if speakers.is_empty() {
            return Err(Error::ModelLoadError(
                "Kokoro voices directory is empty".to_string(),
            ));
        }
        let requested = requested
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or("af_heart");
        if let Some(exact) = speakers.iter().find(|s| s.as_str() == requested) {
            return Ok(exact.clone());
        }
        let requested_lower = requested.to_ascii_lowercase();
        if let Some(casefold) = speakers
            .iter()
            .find(|s| s.to_ascii_lowercase() == requested_lower)
        {
            return Ok(casefold.clone());
        }
        Err(Error::InvalidInput(format!(
            "Unknown Kokoro speaker '{requested}'. Available speakers: {}",
            speakers.join(", ")
        )))
    }

    fn token_ids_from_phonemes(&self, phonemes: &str) -> Result<Vec<u32>> {
        let mut token_ids = Vec::with_capacity(phonemes.chars().count());
        let mut unknown = Vec::new();
        for ch in phonemes.chars() {
            let key = ch.to_string();
            if let Some(id) = self.config.vocab.get(&key) {
                token_ids.push(*id);
            } else if ch.is_whitespace() {
                if let Some(id) = self.config.vocab.get(" ") {
                    token_ids.push(*id);
                }
            } else {
                unknown.push(ch);
            }
        }

        if token_ids.is_empty() {
            return Err(Error::TokenizationError(format!(
                "Kokoro phoneme tokenizer produced zero tokens (unknown chars: {:?})",
                unknown
            )));
        }

        if !unknown.is_empty() {
            tracing::warn!(
                "Kokoro phoneme tokenizer skipped {} unknown symbols: {:?}",
                unknown.len(),
                unknown
            );
        }

        Ok(token_ids)
    }
}

fn inspect_and_validate_checkpoint(
    checkpoint_path: &Path,
    device: &candle_core::Device,
    dtype: DType,
) -> Result<HashMap<String, usize>> {
    let mut counts = HashMap::new();
    for key in CHECKPOINT_SUBMODULE_KEYS {
        let infos = read_pth_tensor_info(checkpoint_path, false, Some(key)).map_err(|e| {
            Error::ModelLoadError(format!(
                "Failed to inspect Kokoro checkpoint submodule '{key}' in {}: {}",
                checkpoint_path.display(),
                e
            ))
        })?;
        if infos.is_empty() {
            return Err(Error::ModelLoadError(format!(
                "Kokoro checkpoint submodule '{key}' in {} has no tensors",
                checkpoint_path.display()
            )));
        }
        let _vb =
            VarBuilder::from_pth_with_state(checkpoint_path, dtype, key, device).map_err(|e| {
                Error::ModelLoadError(format!(
                    "Failed to create Candle VarBuilder for Kokoro submodule '{key}' in {}: {}",
                    checkpoint_path.display(),
                    e
                ))
            })?;
        counts.insert((*key).to_string(), infos.len());
    }
    Ok(counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::{DeviceKind, DeviceSelector};
    use rustfft::num_complex::Complex32;
    use rustfft::FftPlanner;
    use std::path::Path;

    #[test]
    fn kokoro_config_context_length_uses_plbert_positions() {
        let cfg = KokoroConfig {
            istftnet: config::KokoroIstftNetConfig {
                upsample_kernel_sizes: vec![20, 12],
                upsample_rates: vec![10, 6],
                gen_istft_hop_size: 5,
                gen_istft_n_fft: 20,
                resblock_dilation_sizes: vec![vec![1, 3, 5]],
                resblock_kernel_sizes: vec![3],
                upsample_initial_channel: 512,
            },
            dim_in: 64,
            dropout: 0.2,
            hidden_dim: 512,
            max_conv_dim: 512,
            max_dur: 50,
            multispeaker: true,
            n_layer: 3,
            n_mels: 80,
            n_token: 178,
            style_dim: 128,
            text_encoder_kernel_size: 5,
            plbert: config::KokoroPlbertConfig {
                hidden_size: 768,
                num_attention_heads: 12,
                intermediate_size: 2048,
                max_position_embeddings: 512,
                num_hidden_layers: 12,
                dropout: 0.1,
            },
            vocab: HashMap::new(),
        };

        assert_eq!(cfg.context_length(), 512);
    }

    #[test]
    fn kokoro_local_prepare_smoke_if_env_set() {
        let Some(model_dir) = std::env::var_os("IZWI_KOKORO_MODEL_DIR") else {
            return;
        };

        let device = DeviceSelector::detect_with_preference(Some("cpu"))
            .expect("detect cpu device for Kokoro smoke");
        let model =
            KokoroTtsModel::load(Path::new(&model_dir), device).expect("load local Kokoro model");
        let prepared = model
            .prepare_request("Hello world.", Some("af_heart"), Some("en-US"), 1.0)
            .expect("prepare Kokoro request");

        assert!(!prepared.phonemes.is_empty());
        assert!(!prepared.token_ids.is_empty());
        assert_eq!(prepared.ref_style.shape().dims(), &[1, 256]);
    }

    #[test]
    fn kokoro_local_bert_prosody_debug_smoke_if_env_set() {
        let Some(model_dir) = std::env::var_os("IZWI_KOKORO_MODEL_DIR") else {
            return;
        };

        let device = DeviceSelector::detect_with_preference(Some("cpu"))
            .expect("detect cpu device for Kokoro prosody smoke");
        let model =
            KokoroTtsModel::load(Path::new(&model_dir), device).expect("load local Kokoro model");
        let prepared = model
            .prepare_request("Hello world.", Some("af_heart"), Some("en-US"), 1.0)
            .expect("prepare Kokoro request");
        let debug = model
            .run_bert_prosody_debug(&prepared)
            .expect("run Kokoro BERT/prosody debug");

        assert!(!debug.duration_frames.is_empty());
        assert!(debug.expanded_frames > 0);
        assert_eq!(debug.f0_shape.len(), 2);
        assert_eq!(debug.n_shape.len(), 2);
    }

    #[test]
    fn kokoro_local_predecoder_debug_smoke_if_env_set() {
        let Some(model_dir) = std::env::var_os("IZWI_KOKORO_MODEL_DIR") else {
            return;
        };

        let device = DeviceSelector::detect_with_preference(Some("cpu"))
            .expect("detect cpu device for Kokoro predecoder smoke");
        let model =
            KokoroTtsModel::load(Path::new(&model_dir), device).expect("load local Kokoro model");
        let prepared = model
            .prepare_request("Hello world.", Some("af_heart"), Some("en-US"), 1.0)
            .expect("prepare Kokoro request");
        let debug = model
            .run_predecoder_debug(&prepared)
            .expect("run Kokoro predecoder debug");

        assert_eq!(debug.text_encoder_shape.len(), 3);
        assert_eq!(debug.asr_shape.len(), 3);
        assert!(debug.prosody.expanded_frames > 0);
    }

    #[test]
    fn kokoro_local_generate_smoke_if_env_set() {
        let Some(model_dir) = std::env::var_os("IZWI_KOKORO_MODEL_DIR") else {
            return;
        };

        let device = DeviceSelector::detect_with_preference(Some("cpu"))
            .expect("detect cpu device for Kokoro generate smoke");
        let model =
            KokoroTtsModel::load(Path::new(&model_dir), device).expect("load local Kokoro model");
        let result = model
            .generate("Hello world.", Some("af_heart"), Some("en-US"), 1.0)
            .expect("run Kokoro generate");

        assert_eq!(result.sample_rate, KokoroConfig::TARGET_SAMPLE_RATE);
        assert!(!result.samples.is_empty());
        assert!(result.samples.iter().all(|v| v.is_finite()));
        assert!(result.samples.len() > 100);
    }

    #[test]
    fn kokoro_local_generate_metal_smoke_if_env_set() {
        let Some(model_dir) = std::env::var_os("IZWI_KOKORO_MODEL_DIR") else {
            return;
        };

        let Ok(device) = DeviceSelector::detect_with_preference(Some("metal")) else {
            return;
        };
        if device.kind != DeviceKind::Metal {
            return;
        }

        let model = KokoroTtsModel::load(Path::new(&model_dir), device)
            .expect("load local Kokoro model on Metal");
        let result = model
            .generate(
                "Hello my name is Bella",
                Some("af_bella"),
                Some("en-US"),
                1.0,
            )
            .expect("run Kokoro generate on Metal");

        assert_eq!(result.sample_rate, KokoroConfig::TARGET_SAMPLE_RATE);
        assert!(!result.samples.is_empty());
        assert!(result.samples.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn kokoro_local_audio_regression_cpu_if_env_set() {
        let Some(model_dir) = std::env::var_os("IZWI_KOKORO_MODEL_DIR") else {
            return;
        };

        let device = DeviceSelector::detect_with_preference(Some("cpu"))
            .expect("detect cpu device for Kokoro regression");
        let model =
            KokoroTtsModel::load(Path::new(&model_dir), device).expect("load local Kokoro model");
        let result = model
            .generate_with_seed_for_test(
                "Hello my name is Bella",
                Some("af_bella"),
                Some("en-US"),
                1.0,
                0xBEE1_A123_2026u64,
            )
            .expect("run seeded Kokoro regression synthesis");

        let duration_s = result.samples.len() as f32 / result.sample_rate as f32;
        let rms = rms(&result.samples);
        let peak = peak_abs(&result.samples);
        let zcr = zero_crossing_rate(&result.samples);
        let centroid_hz = spectral_centroid_hz(&result.samples, result.sample_rate);

        eprintln!(
            "kokoro regression metrics: len={}, dur={:.3}s, rms={:.6}, peak={:.6}, zcr={:.6}, centroid={:.2}Hz",
            result.samples.len(),
            duration_s,
            rms,
            peak,
            zcr,
            centroid_hz
        );

        assert_eq!(result.sample_rate, KokoroConfig::TARGET_SAMPLE_RATE);
        assert_eq!(result.samples.len(), 52_800, "unexpected sample length");
        assert!((duration_s - 2.2).abs() < 0.02, "duration_s={duration_s}");
        assert!((rms - 0.047_011).abs() < 0.015, "rms={rms}");
        assert!((peak - 0.373_43).abs() < 0.15, "peak={peak}");
        assert!((zcr - 0.222_845).abs() < 0.08, "zcr={zcr}");
        assert!(
            (centroid_hz - 5_955.96).abs() < 900.0,
            "centroid_hz={centroid_hz}"
        );
    }

    fn rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let mean_sq = samples
            .iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>()
            / samples.len() as f64;
        mean_sq.sqrt() as f32
    }

    fn peak_abs(samples: &[f32]) -> f32 {
        samples
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, |a, b| a.max(b))
    }

    fn zero_crossing_rate(samples: &[f32]) -> f32 {
        if samples.len() < 2 {
            return 0.0;
        }
        let mut crossings = 0usize;
        for w in samples.windows(2) {
            let a = w[0];
            let b = w[1];
            if (a >= 0.0 && b < 0.0) || (a < 0.0 && b >= 0.0) {
                crossings += 1;
            }
        }
        crossings as f32 / (samples.len() - 1) as f32
    }

    fn spectral_centroid_hz(samples: &[f32], sample_rate: u32) -> f32 {
        let n = samples.len().clamp(256, 4096).next_power_of_two().min(4096);
        if n < 2 {
            return 0.0;
        }
        let mut frame = vec![Complex32::new(0.0, 0.0); n];
        for i in 0..n {
            let s = *samples.get(i).unwrap_or(&0.0);
            let w = 0.5f32 - 0.5f32 * ((2.0 * std::f32::consts::PI * i as f32) / n as f32).cos();
            frame[i] = Complex32::new(s * w, 0.0);
        }
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n);
        fft.process(&mut frame);
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (k, c) in frame.iter().take(n / 2 + 1).enumerate() {
            let mag = c.norm() as f64;
            let hz = k as f64 * sample_rate as f64 / n as f64;
            num += hz * mag;
            den += mag;
        }
        if den <= 1e-12 {
            0.0
        } else {
            (num / den) as f32
        }
    }
}
