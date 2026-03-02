mod decode;
mod nemo;
mod preprocessor;

use std::path::Path;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::ops;
use candle_nn::{
    batch_norm, layer_norm, Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, LayerNorm, Linear, Module,
    ModuleT, VarBuilder,
};

use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::shared::device::DeviceProfile;
use crate::models::shared::weights::mlx;
use crate::tokenizer::Tokenizer;

use decode::decode_tokens;
use nemo::{ensure_parakeet_artifacts, ParakeetArtifacts};
use preprocessor::ParakeetPreprocessor;

const SAMPLE_RATE: u32 = 16_000;
const ENCODER_LAYERS: usize = 24;
const ENCODER_DIM: usize = 1024;
const ENCODER_HEADS: usize = 8;
const ENCODER_HEAD_DIM: usize = ENCODER_DIM / ENCODER_HEADS;
const FF_DIM: usize = ENCODER_DIM * 4;
const PRED_HIDDEN: usize = 640;
const CONV_SUB_CHANNELS: usize = 256;
const CONV_KERNEL_1D: usize = 9;
const SUBSAMPLING_FACTOR: usize = 8;
const FRAME_HOP_MS: f32 = 10.0;
const DEFAULT_MAX_SYMBOLS: usize = 10;

pub struct ParakeetAsrModel {
    variant: ModelVariant,
    _artifacts: Option<ParakeetArtifacts>,
    decoder: ParakeetDecoder,
    preprocessor: ParakeetPreprocessor,
    network: ParakeetNetwork,
    blank_idx: usize,
    num_durations: usize,
    max_symbols: usize,
}

enum ParakeetDecoder {
    Vocab(Vec<String>),
    HfTokenizer(Tokenizer),
}

impl ParakeetDecoder {
    fn decode(&self, ids: &[usize]) -> String {
        match self {
            Self::Vocab(vocab) => decode_tokens(ids, vocab),
            Self::HfTokenizer(tokenizer) => {
                let token_ids: Vec<u32> = ids.iter().map(|id| *id as u32).collect();
                tokenizer.decode(&token_ids).unwrap_or_default()
            }
        }
    }
}

impl ParakeetAsrModel {
    pub fn load(
        model_dir: &Path,
        variant: ModelVariant,
        device_profile: DeviceProfile,
    ) -> Result<Self> {
        if !variant.is_parakeet() {
            return Err(Error::InvalidInput(format!(
                "Variant {} is not a Parakeet model",
                variant.dir_name()
            )));
        }

        let device = select_device_for_parakeet(&device_profile);
        let (artifacts, decoder, vb) = if variant.is_parakeet_nemo() {
            let artifacts = ensure_parakeet_artifacts(model_dir, variant)?;
            let tokenizer_vocab = decode::load_tokenizer_vocab(&artifacts.tokenizer_vocab_path)?;
            let vb = VarBuilder::from_pth(&artifacts.checkpoint_path, DType::F32, &device)
                .map_err(|e| {
                    Error::ModelLoadError(format!(
                        "Failed to load Parakeet checkpoint {}: {}",
                        artifacts.checkpoint_path.display(),
                        e
                    ))
                })?;
            (Some(artifacts), ParakeetDecoder::Vocab(tokenizer_vocab), vb)
        } else if variant.is_parakeet_mlx() {
            let weights_path = model_dir.join("model.safetensors");
            if !weights_path.exists() {
                return Err(Error::ModelNotFound(format!(
                    "Missing Parakeet MLX weights at {}",
                    weights_path.display()
                )));
            }
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
            }
            .map_err(|e| {
                Error::ModelLoadError(format!("Failed to load Parakeet MLX weights: {e}"))
            })?;
            (None, load_mlx_decoder(model_dir)?, vb)
        } else {
            return Err(Error::InvalidInput(format!(
                "Unsupported Parakeet variant: {}",
                variant.dir_name()
            )));
        };

        let preprocessor = ParakeetPreprocessor::load(&vb, model_dir)?;
        let network = ParakeetNetwork::load(&vb)?;

        let blank_idx = network.blank_idx;
        let num_durations = network.num_durations;

        Ok(Self {
            variant,
            _artifacts: artifacts,
            decoder,
            preprocessor,
            network,
            blank_idx,
            num_durations,
            max_symbols: DEFAULT_MAX_SYMBOLS,
        })
    }

    pub fn transcribe(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
    ) -> Result<String> {
        let mut no_op = |_delta: &str| {};
        self.transcribe_with_callback(audio, sample_rate, language, &mut no_op)
    }

    pub fn transcribe_with_callback(
        &self,
        audio: &[f32],
        sample_rate: u32,
        _language: Option<&str>,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        if audio.is_empty() {
            return Err(Error::InvalidInput("Empty audio input".to_string()));
        }

        let mono_16khz = if sample_rate == SAMPLE_RATE {
            audio.to_vec()
        } else {
            resample_linear(audio, sample_rate, SAMPLE_RATE)
        };

        let (features, feature_frames) = self.preprocessor.compute_features(&mono_16khz)?;
        let (encoded, encoded_len) = self.network.encode(&features, feature_frames)?;

        let mut token_ids = Vec::<usize>::new();
        let mut assembled = String::new();

        let mut on_token = |token_id: usize| {
            token_ids.push(token_id);
            let decoded = self.decoder.decode(&token_ids);
            let delta = text_delta(&assembled, &decoded);
            if !delta.is_empty() {
                on_delta(delta.as_str());
            }
            assembled = decoded;
        };

        self.network.decode_tdt_greedy(
            &encoded,
            encoded_len,
            self.blank_idx,
            self.num_durations,
            self.max_symbols,
            &mut on_token,
        )?;

        if assembled.is_empty() {
            assembled = self.decoder.decode(&token_ids);
        }

        Ok(assembled)
    }
}

fn load_mlx_decoder(model_dir: &Path) -> Result<ParakeetDecoder> {
    if let Ok(tokenizer) = Tokenizer::from_path(model_dir) {
        return Ok(ParakeetDecoder::HfTokenizer(tokenizer));
    }

    // Prefer tokenizer.vocab for MLX Parakeet exports. vocab.txt can be a
    // truncated/auxiliary list with different indexing.
    for candidate in ["tokenizer.vocab", "vocab.txt"] {
        let vocab_path = model_dir.join(candidate);
        if vocab_path.exists() {
            let vocab = decode::load_tokenizer_vocab(&vocab_path)?;
            return Ok(ParakeetDecoder::Vocab(vocab));
        }
    }

    Err(Error::ModelLoadError(format!(
        "Unable to locate tokenizer assets for Parakeet MLX model at {}",
        model_dir.display()
    )))
}

fn select_device_for_parakeet(device_profile: &DeviceProfile) -> Device {
    device_profile.device.clone()
}

struct ParakeetNetwork {
    pre_encode: ConvSubsamplingDw,
    layers: Vec<ConformerLayer>,
    predictor: Predictor,
    joint: Joint,
    blank_idx: usize,
    num_durations: usize,
}

impl ParakeetNetwork {
    fn load(vb: &VarBuilder) -> Result<Self> {
        let pre_encode = ConvSubsamplingDw::load(vb.pp("encoder.pre_encode"))?;

        let mut layers = Vec::with_capacity(ENCODER_LAYERS);
        for idx in 0..ENCODER_LAYERS {
            layers.push(ConformerLayer::load(
                vb.pp(format!("encoder.layers.{idx}")),
            )?);
        }

        let predictor = Predictor::load(vb.pp("decoder.prediction"))?;
        let joint = Joint::load(vb.pp("joint"), ENCODER_DIM, predictor.blank_idx + 1)?;

        let blank_idx = predictor.blank_idx;
        let num_durations = joint.num_durations;

        Ok(Self {
            pre_encode,
            layers,
            predictor,
            joint,
            blank_idx,
            num_durations,
        })
    }

    fn encode(&self, features: &Tensor, feature_frames: usize) -> Result<(Tensor, usize)> {
        let (mut x, encoded_len) = self.pre_encode.forward(features, feature_frames)?;

        let pos_len = x.dim(1)?;
        let pos_emb = build_rel_positional_embedding(pos_len, ENCODER_DIM, x.device())?;
        for layer in &self.layers {
            x = layer.forward(&x, &pos_emb)?;
        }

        Ok((x, encoded_len))
    }

    fn decode_tdt_greedy(
        &self,
        encoded: &Tensor,
        encoded_len: usize,
        blank_idx: usize,
        num_durations: usize,
        max_symbols: usize,
        on_token: &mut dyn FnMut(usize),
    ) -> Result<()> {
        let encoded = encoded.i((0, ..encoded_len, ..))?; // [T, D]

        let mut predictor_state = self.predictor.initial_state(1, encoded.device())?;
        let mut predictor_out =
            self.predictor
                .step(blank_idx, &mut predictor_state, encoded.device())?;

        let mut t = 0usize;
        let mut last_emit_t = usize::MAX;
        let mut emit_count_at_t = 0usize;
        let mut guard_steps = 0usize;
        let guard_limit = encoded_len
            .saturating_mul(max_symbols)
            .saturating_mul(8)
            .max(512);

        while t < encoded_len && guard_steps < guard_limit {
            guard_steps += 1;

            let t_cur = t;
            let enc_t = encoded.i((t_cur, ..))?.unsqueeze(0)?.unsqueeze(0)?; // [1,1,D]
            let logits = self
                .joint
                .joint_after_projection(&enc_t, &predictor_out)?
                .squeeze(0)?
                .squeeze(0)?
                .squeeze(0)?; // [V + 1 + num_durations]

            let token_logits = logits.i(..(blank_idx + 1))?;
            let duration_logits = logits.i((blank_idx + 1)..)?;

            let mut label = argmax_1d(&token_logits)?;
            let duration_idx = argmax_1d(&duration_logits)?;
            let mut jump = duration_idx.min(num_durations.saturating_sub(1));

            if label == blank_idx && jump == 0 {
                jump = 1;
            }

            t = t.saturating_add(jump);

            while label == blank_idx && t < encoded_len {
                let enc_t = encoded.i((t, ..))?.unsqueeze(0)?.unsqueeze(0)?;
                let logits = self
                    .joint
                    .joint_after_projection(&enc_t, &predictor_out)?
                    .squeeze(0)?
                    .squeeze(0)?
                    .squeeze(0)?;

                let token_logits = logits.i(..(blank_idx + 1))?;
                let duration_logits = logits.i((blank_idx + 1)..)?;

                label = argmax_1d(&token_logits)?;
                let duration_idx = argmax_1d(&duration_logits)?;
                jump = duration_idx.min(num_durations.saturating_sub(1));

                if label == blank_idx && jump == 0 {
                    jump = 1;
                }

                t = t.saturating_add(jump);
            }

            if label == blank_idx {
                continue;
            }

            if t_cur == last_emit_t {
                emit_count_at_t = emit_count_at_t.saturating_add(1);
            } else {
                last_emit_t = t_cur;
                emit_count_at_t = 1;
            }

            if emit_count_at_t > max_symbols {
                t = t_cur.saturating_add(1);
                continue;
            }

            on_token(label);
            predictor_out = self
                .predictor
                .step(label, &mut predictor_state, encoded.device())?;
        }

        Ok(())
    }
}

fn text_delta(previous: &str, current: &str) -> String {
    if let Some(delta) = current.strip_prefix(previous) {
        return delta.to_string();
    }
    let common = previous
        .chars()
        .zip(current.chars())
        .take_while(|(a, b)| a == b)
        .count();
    current.chars().skip(common).collect()
}

struct ConvSubsamplingDw {
    conv0: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    conv5: Conv2d,
    conv6: Conv2d,
    out: Linear,
}

impl ConvSubsamplingDw {
    fn load(vb: VarBuilder) -> Result<Self> {
        let stride_cfg = Conv2dConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };
        let point_cfg = Conv2dConfig {
            stride: 1,
            padding: 0,
            ..Default::default()
        };

        let conv0 = mlx::load_conv2d(1, CONV_SUB_CHANNELS, 3, stride_cfg, vb.pp("conv.0"))?;

        let mut dw_stride_cfg = stride_cfg;
        dw_stride_cfg.groups = CONV_SUB_CHANNELS;
        // Depthwise conv tensors are shaped [out_channels, in_channels/groups, k, k].
        // For groups=channels this second dimension is 1.
        let conv2 = mlx::load_conv2d(1, CONV_SUB_CHANNELS, 3, dw_stride_cfg, vb.pp("conv.2"))?;
        let conv3 = mlx::load_conv2d(
            CONV_SUB_CHANNELS,
            CONV_SUB_CHANNELS,
            1,
            point_cfg,
            vb.pp("conv.3"),
        )?;
        let conv5 = mlx::load_conv2d(1, CONV_SUB_CHANNELS, 3, dw_stride_cfg, vb.pp("conv.5"))?;
        let conv6 = mlx::load_conv2d(
            CONV_SUB_CHANNELS,
            CONV_SUB_CHANNELS,
            1,
            point_cfg,
            vb.pp("conv.6"),
        )?;

        let out = mlx::load_linear(CONV_SUB_CHANNELS * 16, ENCODER_DIM, vb.pp("out"))?;

        Ok(Self {
            conv0,
            conv2,
            conv3,
            conv5,
            conv6,
            out,
        })
    }

    fn forward(&self, features: &Tensor, feature_frames: usize) -> Result<(Tensor, usize)> {
        // Input features are [B=1, MELS, T]. Convert to [B, 1, T, MELS]
        let mut x = features.transpose(1, 2)?.unsqueeze(1)?;

        x = self.conv0.forward(&x)?;
        x = x.relu()?;

        x = self.conv2.forward(&x)?;
        x = self.conv3.forward(&x)?;
        x = x.relu()?;

        x = self.conv5.forward(&x)?;
        x = self.conv6.forward(&x)?;
        x = x.relu()?;

        let (b, c, t, f) = x.dims4()?;
        let x = x
            .transpose(1, 2)?
            .reshape((b, t, c * f))?
            .apply(&self.out)?;

        // Padding in the frontend/STFT path can slightly overestimate lengths.
        // Clamp to the actual encoder time dimension to keep positional shapes valid.
        let encoded_len = subsampled_len_3x(feature_frames).min(t);
        Ok((x, encoded_len))
    }
}

fn subsampled_len_3x(mut len: usize) -> usize {
    for _ in 0..3 {
        len = (len + 1) / 2;
    }
    len
}

struct ConformerLayer {
    norm_ff1: LayerNorm,
    ff1: FeedForward,
    norm_self_att: LayerNorm,
    self_attn: RelPosSelfAttention,
    norm_conv: LayerNorm,
    conv: ConformerConv,
    norm_ff2: LayerNorm,
    ff2: FeedForward,
    norm_out: LayerNorm,
}

impl ConformerLayer {
    fn load(vb: VarBuilder) -> Result<Self> {
        let norm_ff1 = layer_norm(ENCODER_DIM, 1e-5, vb.pp("norm_feed_forward1"))?;
        let ff1 = FeedForward::load(vb.pp("feed_forward1"))?;

        let norm_self_att = layer_norm(ENCODER_DIM, 1e-5, vb.pp("norm_self_att"))?;
        let self_attn = RelPosSelfAttention::load(vb.pp("self_attn"))?;

        let norm_conv = layer_norm(ENCODER_DIM, 1e-5, vb.pp("norm_conv"))?;
        let conv = ConformerConv::load(vb.pp("conv"))?;

        let norm_ff2 = layer_norm(ENCODER_DIM, 1e-5, vb.pp("norm_feed_forward2"))?;
        let ff2 = FeedForward::load(vb.pp("feed_forward2"))?;

        let norm_out = layer_norm(ENCODER_DIM, 1e-5, vb.pp("norm_out"))?;

        Ok(Self {
            norm_ff1,
            ff1,
            norm_self_att,
            self_attn,
            norm_conv,
            conv,
            norm_ff2,
            ff2,
            norm_out,
        })
    }

    fn forward(&self, x: &Tensor, pos_emb: &Tensor) -> Result<Tensor> {
        let mut residual = x.clone();

        let ff1 = self.ff1.forward(&self.norm_ff1.forward(&residual)?)?;
        residual = residual.broadcast_add(&ff1.affine(0.5, 0.0)?)?;

        let attn = self
            .self_attn
            .forward(&self.norm_self_att.forward(&residual)?, pos_emb)?;
        residual = residual.broadcast_add(&attn)?;

        let conv = self.conv.forward(&self.norm_conv.forward(&residual)?)?;
        residual = residual.broadcast_add(&conv)?;

        let ff2 = self.ff2.forward(&self.norm_ff2.forward(&residual)?)?;
        residual = residual.broadcast_add(&ff2.affine(0.5, 0.0)?)?;

        self.norm_out
            .forward(&residual)
            .map_err(|e| Error::InferenceError(e.to_string()))
    }
}

struct FeedForward {
    linear1: Linear,
    linear2: Linear,
}

impl FeedForward {
    fn load(vb: VarBuilder) -> Result<Self> {
        let linear1 = mlx::load_linear_no_bias(ENCODER_DIM, FF_DIM, vb.pp("linear1"))?;
        let linear2 = mlx::load_linear_no_bias(FF_DIM, ENCODER_DIM, vb.pp("linear2"))?;
        Ok(Self { linear1, linear2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = swish(&x)?;
        self.linear2
            .forward(&x)
            .map_err(|e| Error::InferenceError(e.to_string()))
    }
}

struct ConformerConv {
    pointwise_conv1: Conv1d,
    depthwise_conv: Conv1d,
    batch_norm: candle_nn::BatchNorm,
    pointwise_conv2: Conv1d,
}

impl ConformerConv {
    fn load(vb: VarBuilder) -> Result<Self> {
        let pointwise_conv1 = mlx::load_conv1d_no_bias(
            ENCODER_DIM,
            ENCODER_DIM * 2,
            1,
            Conv1dConfig::default(),
            vb.pp("pointwise_conv1"),
        )?;

        let depthwise_conv = mlx::load_conv1d_no_bias(
            ENCODER_DIM,
            ENCODER_DIM,
            CONV_KERNEL_1D,
            Conv1dConfig {
                padding: (CONV_KERNEL_1D - 1) / 2,
                groups: ENCODER_DIM,
                ..Default::default()
            },
            vb.pp("depthwise_conv"),
        )?;

        let batch_norm = batch_norm(ENCODER_DIM, 1e-5, vb.pp("batch_norm"))?;

        let pointwise_conv2 = mlx::load_conv1d_no_bias(
            ENCODER_DIM,
            ENCODER_DIM,
            1,
            Conv1dConfig::default(),
            vb.pp("pointwise_conv2"),
        )?;

        Ok(Self {
            pointwise_conv1,
            depthwise_conv,
            batch_norm,
            pointwise_conv2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // [B, T, C] -> [B, C, T]
        let mut x = x.transpose(1, 2)?;

        x = self.pointwise_conv1.forward(&x)?;
        let x_a = x.i((.., ..ENCODER_DIM, ..))?;
        let x_b = x.i((.., ENCODER_DIM.., ..))?;
        x = x_a.broadcast_mul(&ops::sigmoid(&x_b)?)?;

        x = self.depthwise_conv.forward(&x)?;
        x = self.batch_norm.forward_t(&x, false)?;
        x = swish(&x)?;
        x = self.pointwise_conv2.forward(&x)?;

        x.transpose(1, 2).map_err(Error::from)
    }
}

struct RelPosSelfAttention {
    linear_q: Linear,
    linear_k: Linear,
    linear_v: Linear,
    linear_out: Linear,
    linear_pos: Linear,
    pos_bias_u: Tensor,
    pos_bias_v: Tensor,
}

impl RelPosSelfAttention {
    fn load(vb: VarBuilder) -> Result<Self> {
        let linear_q = mlx::load_linear_no_bias(ENCODER_DIM, ENCODER_DIM, vb.pp("linear_q"))?;
        let linear_k = mlx::load_linear_no_bias(ENCODER_DIM, ENCODER_DIM, vb.pp("linear_k"))?;
        let linear_v = mlx::load_linear_no_bias(ENCODER_DIM, ENCODER_DIM, vb.pp("linear_v"))?;
        let linear_out = mlx::load_linear_no_bias(ENCODER_DIM, ENCODER_DIM, vb.pp("linear_out"))?;
        let linear_pos = mlx::load_linear_no_bias(ENCODER_DIM, ENCODER_DIM, vb.pp("linear_pos"))?;

        // In Parakeet checkpoints these are direct tensors under self_attn.*.
        let pos_bias_u = vb.get((ENCODER_HEADS, ENCODER_HEAD_DIM), "pos_bias_u")?;
        let pos_bias_v = vb.get((ENCODER_HEADS, ENCODER_HEAD_DIM), "pos_bias_v")?;

        Ok(Self {
            linear_q,
            linear_k,
            linear_v,
            linear_out,
            linear_pos,
            pos_bias_u,
            pos_bias_v,
        })
    }

    fn forward(&self, x: &Tensor, pos_emb: &Tensor) -> Result<Tensor> {
        let (b, t, _d) = x.dims3()?;

        let q = self
            .linear_q
            .forward(x)?
            .reshape((b, t, ENCODER_HEADS, ENCODER_HEAD_DIM))?
            .transpose(1, 2)?
            .contiguous()?; // [B, H, T, Dh]
        let k = self
            .linear_k
            .forward(x)?
            .reshape((b, t, ENCODER_HEADS, ENCODER_HEAD_DIM))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .linear_v
            .forward(x)?
            .reshape((b, t, ENCODER_HEADS, ENCODER_HEAD_DIM))?
            .transpose(1, 2)?
            .contiguous()?;

        let p = self
            .linear_pos
            .forward(pos_emb)?
            .reshape((1, 2 * t - 1, ENCODER_HEADS, ENCODER_HEAD_DIM))?
            .transpose(1, 2)?; // [1, H, 2T-1, Dh]

        let pos_bias_u = self
            .pos_bias_u
            .reshape((1, ENCODER_HEADS, 1, ENCODER_HEAD_DIM))?;
        let pos_bias_v = self
            .pos_bias_v
            .reshape((1, ENCODER_HEADS, 1, ENCODER_HEAD_DIM))?;

        let q_u = q.broadcast_add(&pos_bias_u)?;
        let q_v = q.broadcast_add(&pos_bias_v)?;

        let k_t = k.transpose(2, 3)?.contiguous()?;
        let p_t = p.transpose(2, 3)?.contiguous()?;
        let matrix_ac = q_u.matmul(&k_t)?;
        let matrix_bd = rel_shift(&q_v.matmul(&p_t)?)?;
        let matrix_bd = matrix_bd.narrow(3, 0, t)?;

        let scores = matrix_ac
            .broadcast_add(&matrix_bd)?
            .affine(1.0 / (ENCODER_HEAD_DIM as f64).sqrt(), 0.0)?;
        let attn = ops::softmax(&scores, 3)?;

        let out = attn.matmul(&v)?;
        let out = out.transpose(1, 2)?.reshape((b, t, ENCODER_DIM))?;

        self.linear_out
            .forward(&out)
            .map_err(|e| Error::InferenceError(e.to_string()))
    }
}

fn rel_shift(x: &Tensor) -> Result<Tensor> {
    // x: [B, H, T, 2T-1]
    let (b, h, qlen, pos_len) = x.dims4()?;
    let x = x.pad_with_zeros(3, 1, 0)?; // [B, H, T, 2T]
    let x = x.reshape((b, h, pos_len + 1, qlen))?;
    let x = x.narrow(2, 1, pos_len)?;
    x.reshape((b, h, qlen, pos_len)).map_err(Error::from)
}

struct Predictor {
    embed: Tensor, // [V+1, H]
    lstm_l0: LstmCell,
    lstm_l1: LstmCell,
    blank_idx: usize,
}

#[derive(Clone)]
struct PredictorState {
    h0: Tensor,
    c0: Tensor,
    h1: Tensor,
    c1: Tensor,
}

impl Predictor {
    fn load(vb: VarBuilder) -> Result<Self> {
        let embed = vb.pp("embed").get_unchecked_dtype("weight", DType::F32)?;
        let vocab_plus_blank = embed.dim(0)?;
        let blank_idx = vocab_plus_blank.saturating_sub(1);

        let lstm_l0 = LstmCell::load(vb.pp("dec_rnn.lstm"), 0)?;
        let lstm_l1 = LstmCell::load(vb.pp("dec_rnn.lstm"), 1)?;

        Ok(Self {
            embed,
            lstm_l0,
            lstm_l1,
            blank_idx,
        })
    }

    fn initial_state(&self, batch: usize, device: &Device) -> Result<PredictorState> {
        let zeros = |dim| Tensor::zeros((batch, dim), DType::F32, device).map_err(Error::from);
        Ok(PredictorState {
            h0: zeros(PRED_HIDDEN)?,
            c0: zeros(PRED_HIDDEN)?,
            h1: zeros(PRED_HIDDEN)?,
            c1: zeros(PRED_HIDDEN)?,
        })
    }

    fn step(&self, label: usize, state: &mut PredictorState, device: &Device) -> Result<Tensor> {
        let x = if label == self.blank_idx {
            Tensor::zeros((1, PRED_HIDDEN), DType::F32, device)?
        } else {
            self.embed.i((label, ..))?.unsqueeze(0)?
        };

        let (h0, c0) = self.lstm_l0.step(&x, &state.h0, &state.c0)?;
        state.h0 = h0;
        state.c0 = c0;

        let (h1, c1) = self.lstm_l1.step(&state.h0, &state.h1, &state.c1)?;
        state.h1 = h1;
        state.c1 = c1;

        Ok(state.h1.unsqueeze(1)?) // [B=1, U=1, H]
    }
}

struct LstmCell {
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
}

impl LstmCell {
    fn load(vb: VarBuilder, layer: usize) -> Result<Self> {
        let w_ih_name = format!("weight_ih_l{layer}");
        let w_hh_name = format!("weight_hh_l{layer}");
        let b_ih_name = format!("bias_ih_l{layer}");
        let b_hh_name = format!("bias_hh_l{layer}");
        if vb.contains_tensor(&w_ih_name) {
            let w_ih = vb.get((PRED_HIDDEN * 4, PRED_HIDDEN), &w_ih_name)?;
            let w_hh = vb.get((PRED_HIDDEN * 4, PRED_HIDDEN), &w_hh_name)?;
            let b_ih = vb.get(PRED_HIDDEN * 4, &b_ih_name)?;
            let b_hh = vb.get(PRED_HIDDEN * 4, &b_hh_name)?;
            return Ok(Self {
                w_ih,
                w_hh,
                b_ih,
                b_hh,
            });
        }

        // MLX Parakeet exports LSTM params as dec_rnn.lstm.{layer}.Wx/.Wh/.bias.
        let w_x_name = format!("{layer}.Wx");
        let w_h_name = format!("{layer}.Wh");
        let bias_name = format!("{layer}.bias");
        if !vb.contains_tensor(&w_x_name) || !vb.contains_tensor(&w_h_name) {
            return Err(Error::ModelLoadError(format!(
                "Missing Parakeet LSTM tensors for layer {layer}: expected either weight_ih/weight_hh or {w_x_name}/{w_h_name}"
            )));
        }

        let w_ih = vb.get((PRED_HIDDEN * 4, PRED_HIDDEN), &w_x_name)?;
        let w_hh = vb.get((PRED_HIDDEN * 4, PRED_HIDDEN), &w_h_name)?;
        let b_ih = if vb.contains_tensor(&bias_name) {
            vb.get(PRED_HIDDEN * 4, &bias_name)?
        } else {
            Tensor::zeros(PRED_HIDDEN * 4, DType::F32, vb.device())?
        };
        let b_hh = Tensor::zeros(PRED_HIDDEN * 4, DType::F32, vb.device())?;

        Ok(Self {
            w_ih,
            w_hh,
            b_ih,
            b_hh,
        })
    }

    fn step(&self, x: &Tensor, h_prev: &Tensor, c_prev: &Tensor) -> Result<(Tensor, Tensor)> {
        let gates = x
            .matmul(&self.w_ih.transpose(0, 1)?)?
            .broadcast_add(&self.b_ih.unsqueeze(0)?)?
            .broadcast_add(&h_prev.matmul(&self.w_hh.transpose(0, 1)?)?)?
            .broadcast_add(&self.b_hh.unsqueeze(0)?)?;

        let i = ops::sigmoid(&gates.i((.., 0..PRED_HIDDEN))?)?;
        let f = ops::sigmoid(&gates.i((.., PRED_HIDDEN..(PRED_HIDDEN * 2)))?)?;
        let g = gates
            .i((.., (PRED_HIDDEN * 2)..(PRED_HIDDEN * 3)))?
            .tanh()?;
        let o = ops::sigmoid(&gates.i((.., (PRED_HIDDEN * 3)..))?)?;

        let c = f
            .broadcast_mul(c_prev)?
            .broadcast_add(&i.broadcast_mul(&g)?)?;
        let h = o.broadcast_mul(&c.tanh()?)?;

        Ok((h, c))
    }
}

struct Joint {
    pred: Linear,
    enc: Linear,
    out: Linear,
    num_classes_with_blank: usize,
    num_durations: usize,
}

impl Joint {
    fn load(vb: VarBuilder, enc_hidden: usize, num_classes_with_blank: usize) -> Result<Self> {
        let pred = mlx::load_linear(PRED_HIDDEN, PRED_HIDDEN, vb.pp("pred"))?;
        let enc = mlx::load_linear(enc_hidden, PRED_HIDDEN, vb.pp("enc"))?;

        let out_bias = vb
            .pp("joint_net.2")
            .get_unchecked_dtype("bias", DType::F32)?;
        let out_dim = out_bias.dim(0)?;

        let out = mlx::load_linear(PRED_HIDDEN, out_dim, vb.pp("joint_net.2"))?;

        // TDT adds extra duration logits after token logits.
        let num_durations = out_dim.saturating_sub(num_classes_with_blank);
        if num_durations == 0 {
            return Err(Error::ModelLoadError(format!(
                "Invalid Parakeet joint output size: out_dim={out_dim}, classes_with_blank={num_classes_with_blank}"
            )));
        }

        Ok(Self {
            pred,
            enc,
            out,
            num_classes_with_blank,
            num_durations,
        })
    }

    fn joint_after_projection(&self, f: &Tensor, g: &Tensor) -> Result<Tensor> {
        // f: [B, T, Denc], g: [B, U, Dpred]
        let f = self.enc.forward(f)?;
        let g = self.pred.forward(g)?;

        let inp = f.unsqueeze(2)?.broadcast_add(&g.unsqueeze(1)?)?;
        let inp = inp.relu()?;
        self.out
            .forward(&inp)
            .map_err(|e| Error::InferenceError(e.to_string()))
    }
}

fn build_rel_positional_embedding(len: usize, d_model: usize, device: &Device) -> Result<Tensor> {
    if len == 0 {
        return Err(Error::InvalidInput(
            "Cannot build positional embedding for empty sequence".to_string(),
        ));
    }

    let pos_len = 2 * len - 1;
    let mut positions = Vec::with_capacity(pos_len);
    for p in (-(len as isize - 1))..=(len as isize - 1) {
        positions.push((-p) as f32);
    }

    let mut emb = vec![0f32; pos_len * d_model];
    let denom = (10_000f32).ln() / d_model as f32;

    for (pi, p) in positions.iter().enumerate() {
        for i in (0..d_model).step_by(2) {
            let div = (-denom * i as f32).exp();
            let angle = p * div;
            emb[pi * d_model + i] = angle.sin();
            if i + 1 < d_model {
                emb[pi * d_model + i + 1] = angle.cos();
            }
        }
    }

    Tensor::from_vec(emb, (1, pos_len, d_model), device).map_err(Error::from)
}

fn swish(x: &Tensor) -> Result<Tensor> {
    x.broadcast_mul(&ops::sigmoid(x)?).map_err(Error::from)
}

fn argmax_1d(x: &Tensor) -> Result<usize> {
    let v = x.to_vec1::<f32>()?;
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &val) in v.iter().enumerate() {
        if val > best_val {
            best_val = val;
            best_idx = i;
        }
    }
    Ok(best_idx)
}

fn resample_linear(audio: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate || audio.len() < 2 {
        return audio.to_vec();
    }

    let ratio = dst_rate as f64 / src_rate as f64;
    let out_len = ((audio.len() as f64) * ratio).round().max(1.0) as usize;

    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = (i as f64) / ratio;
        let left = src_pos.floor() as usize;
        let right = left.saturating_add(1).min(audio.len() - 1);
        let frac = (src_pos - left as f64) as f32;
        let sample = audio[left] * (1.0 - frac) + audio[right] * frac;
        out.push(sample);
    }
    out
}
