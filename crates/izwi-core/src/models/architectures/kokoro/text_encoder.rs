use candle_core::{DType, Tensor};
use candle_nn::{embedding, ops, Embedding, Module, VarBuilder};

use crate::error::{Error, Result};

use super::config::KokoroConfig;
use super::prosody::{load_weight_norm_conv1d, BiLstm1};

#[derive(Debug)]
pub struct KokoroTextEncoder {
    embedding: Embedding,
    cnn: Vec<TextEncoderConvBlock>,
    lstm: BiLstm1,
    channels: usize,
}

impl KokoroTextEncoder {
    pub fn load(cfg: &KokoroConfig, vb: VarBuilder) -> Result<Self> {
        let root = vb.pp("module");
        let channels = cfg.hidden_dim;
        let embedding =
            embedding(cfg.n_token, channels, root.pp("embedding")).map_err(Error::from)?;
        let mut cnn = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            cnn.push(TextEncoderConvBlock::load(
                channels,
                cfg.text_encoder_kernel_size,
                root.pp(format!("cnn.{i}")),
            )?);
        }
        let lstm = BiLstm1::load(channels, channels / 2, root.pp("lstm"))?;
        Ok(Self {
            embedding,
            cnn,
            lstm,
            channels,
        })
    }

    /// Mirrors Kokoro/StyleTTS2 text encoder path used to produce decoder ASR features.
    ///
    /// Input: `[B, T]` token ids
    /// Output: `[B, C, T]` features
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_b, _t) = input_ids.dims2().map_err(Error::from)?;
        let mut x = self.embedding.forward(input_ids).map_err(Error::from)?; // [B,T,C]
        x = x.transpose(1, 2).map_err(Error::from)?; // [B,C,T]
        for block in &self.cnn {
            x = block.forward(&x)?;
        }
        let x_bt = x.transpose(1, 2).map_err(Error::from)?; // [B,T,C]
        let x_bt = self.lstm.forward(&x_bt)?; // [B,T,C]
        let x = x_bt.transpose(1, 2).map_err(Error::from)?; // [B,C,T]
        let (_b, c, _t) = x.dims3().map_err(Error::from)?;
        if c != self.channels {
            return Err(Error::InferenceError(format!(
                "KokoroTextEncoder output channels {} != expected {}",
                c, self.channels
            )));
        }
        Ok(x)
    }
}

#[derive(Debug)]
struct TextEncoderConvBlock {
    conv: candle_nn::Conv1d,
    norm: KokoroChannelLayerNorm,
}

impl TextEncoderConvBlock {
    fn load(channels: usize, kernel_size: usize, vb: VarBuilder) -> Result<Self> {
        let padding = (kernel_size.saturating_sub(1)) / 2;
        let conv = load_weight_norm_conv1d(
            vb.pp("0"),
            candle_nn::Conv1dConfig {
                padding,
                stride: 1,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            },
        )?;
        let norm = KokoroChannelLayerNorm::load(channels, vb.pp("1"))?;
        Ok(Self { conv, norm })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x).map_err(Error::from)?;
        let x = self.norm.forward(&x)?;
        ops::leaky_relu(&x, 0.2).map_err(Error::from)
    }
}

#[derive(Debug)]
struct KokoroChannelLayerNorm {
    channels: usize,
    eps: f64,
    gamma: Tensor,
    beta: Tensor,
}

impl KokoroChannelLayerNorm {
    fn load(channels: usize, vb: VarBuilder) -> Result<Self> {
        let gamma = vb
            .get_unchecked_dtype("gamma", DType::F32)
            .map_err(Error::from)?;
        let beta = vb
            .get_unchecked_dtype("beta", DType::F32)
            .map_err(Error::from)?;
        Ok(Self {
            channels,
            eps: 1e-5,
            gamma,
            beta,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, t) = x.dims3().map_err(Error::from)?;
        if c != self.channels {
            return Err(Error::InferenceError(format!(
                "KokoroChannelLayerNorm expected {} channels, got {}",
                self.channels, c
            )));
        }
        let x_btc = x.transpose(1, 2).map_err(Error::from)?; // [B,T,C]
        let (mean, var) = super::prosody::biased_mean_var_keepdim(&x_btc, 2)?;
        let denom = (var + self.eps)
            .map_err(Error::from)?
            .sqrt()
            .map_err(Error::from)?;
        let xhat = x_btc
            .broadcast_sub(&mean)
            .map_err(Error::from)?
            .broadcast_div(&denom)
            .map_err(Error::from)?;
        let gamma = self
            .gamma
            .reshape((1, 1, self.channels))
            .map_err(Error::from)?
            .broadcast_as((b, t, self.channels))
            .map_err(Error::from)?;
        let beta = self
            .beta
            .reshape((1, 1, self.channels))
            .map_err(Error::from)?
            .broadcast_as((b, t, self.channels))
            .map_err(Error::from)?;
        let y = xhat
            .broadcast_mul(&gamma)
            .map_err(Error::from)?
            .broadcast_add(&beta)
            .map_err(Error::from)?;
        y.transpose(1, 2).map_err(Error::from)
    }
}
