//! Granite Speech 5.0's speech projector: a 2-layer BLIP-2 Q-Former over
//! windows of 15 encoder frames, 3 learned queries per window (a 5x
//! downsample to 10 audio tokens per second), then a linear map into the
//! language model's hidden size.

use candle_core::{DType, Tensor, D};
use candle_nn::{layer_norm, linear, ops, LayerNorm, Linear, Module, VarBuilder};

use crate::error::{Error, Result};
use crate::models::shared::attention::flash::try_fused_self_attention;

use super::config::GraniteSpeech5Config;
use super::precision;

pub(super) struct QFormerProjector {
    query: Tensor,
    qformer: QFormer,
    linear: Linear,
    window_size: usize,
    num_queries: usize,
}

impl QFormerProjector {
    pub(super) fn load(config: &GraniteSpeech5Config, vb: VarBuilder) -> Result<Self> {
        let num_queries = config.queries_per_window();
        Ok(Self {
            query: vb.get(
                (1, num_queries, config.projector_config.hidden_size),
                "query",
            )?,
            qformer: QFormer::load(config, vb.pp("qformer"))?,
            linear: linear(
                config.projector_config.hidden_size,
                config.text_config.hidden_size,
                vb.pp("linear"),
            )?,
            window_size: config.window_size.max(1),
            num_queries,
        })
    }

    /// `(batch, frames, encoder)` → `(batch, windows * queries, text hidden)`;
    /// the final window is zero-padded.
    pub(super) fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let (batch, frames, dim) = hidden.dims3()?;
        let windows = self.window_count(frames);
        let pad = windows * self.window_size - frames;
        let hidden = if pad > 0 {
            hidden.pad_with_zeros(1, 0, pad)?
        } else {
            hidden.clone()
        };
        let windowed = hidden.reshape((batch * windows, self.window_size, dim))?;
        let query =
            self.query
                .broadcast_as((batch * windows, self.num_queries, self.query.dim(2)?))?;
        let output = self.qformer.forward(&query, &windowed)?;
        let output = output.reshape((batch, windows * self.num_queries, output.dim(2)?))?;
        self.linear.forward(&output).map_err(Error::from)
    }

    pub(super) fn window_count(&self, frames: usize) -> usize {
        frames.div_ceil(self.window_size)
    }

    pub(super) fn window_size(&self) -> usize {
        self.window_size
    }

    pub(super) fn num_queries(&self) -> usize {
        self.num_queries
    }

    pub(super) fn layer_count(&self) -> usize {
        self.qformer.layers.len()
    }
}

struct QFormer {
    layernorm: LayerNorm,
    layers: Vec<QFormerLayer>,
}

impl QFormer {
    fn load(config: &GraniteSpeech5Config, vb: VarBuilder) -> Result<Self> {
        let projector = &config.projector_config;
        Ok(Self {
            layernorm: layer_norm(
                projector.hidden_size,
                projector.layer_norm_eps,
                vb.pp("layernorm"),
            )?,
            layers: (0..projector.num_hidden_layers)
                .map(|idx| QFormerLayer::load(config, idx, vb.pp(format!("encoder.layer.{idx}"))))
                .collect::<Result<Vec<_>>>()?,
        })
    }

    fn forward(&self, query: &Tensor, encoder_hidden: &Tensor) -> Result<Tensor> {
        let mut x = self.layernorm.forward(query)?;
        for layer in &self.layers {
            x = layer.forward(&x, encoder_hidden)?;
        }
        Ok(x)
    }
}

struct QFormerLayer {
    attention: QFormerAttention,
    crossattention: Option<QFormerAttention>,
    intermediate_query: Linear,
    output_query_dense: Linear,
    output_query_norm: LayerNorm,
}

impl QFormerLayer {
    fn load(config: &GraniteSpeech5Config, idx: usize, vb: VarBuilder) -> Result<Self> {
        let projector = &config.projector_config;
        let hidden = projector.hidden_size;
        let has_cross = idx.is_multiple_of(projector.cross_attention_frequency.max(1));
        Ok(Self {
            attention: QFormerAttention::load(config, false, vb.pp("attention"))?,
            crossattention: has_cross
                .then(|| QFormerAttention::load(config, true, vb.pp("crossattention")))
                .transpose()?,
            intermediate_query: linear(
                hidden,
                projector.intermediate_size,
                vb.pp("intermediate_query.dense"),
            )?,
            output_query_dense: linear(
                projector.intermediate_size,
                hidden,
                vb.pp("output_query.dense"),
            )?,
            output_query_norm: layer_norm(
                hidden,
                projector.layer_norm_eps,
                vb.pp("output_query.LayerNorm"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor, encoder_hidden: &Tensor) -> Result<Tensor> {
        let mut out = self.attention.forward(x, None)?;
        if let Some(crossattention) = &self.crossattention {
            out = crossattention.forward(&out, Some(encoder_hidden))?;
        }
        let ff = self.intermediate_query.forward(&out)?.gelu_erf()?;
        let ff = self.output_query_dense.forward(&ff)?;
        self.output_query_norm
            .forward(&ff.broadcast_add(&out)?)
            .map_err(Error::from)
    }
}

struct QFormerAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output_dense: Linear,
    output_norm: LayerNorm,
    num_heads: usize,
    head_dim: usize,
}

impl QFormerAttention {
    fn load(config: &GraniteSpeech5Config, cross: bool, vb: VarBuilder) -> Result<Self> {
        let projector = &config.projector_config;
        let hidden = projector.hidden_size;
        let kv_hidden = if cross {
            projector.encoder_hidden_size
        } else {
            hidden
        };
        let head_dim = hidden / projector.num_attention_heads;
        let all_heads = projector.num_attention_heads * head_dim;
        Ok(Self {
            query: linear(hidden, all_heads, vb.pp("attention.query"))?,
            key: linear(kv_hidden, all_heads, vb.pp("attention.key"))?,
            value: linear(kv_hidden, all_heads, vb.pp("attention.value"))?,
            output_dense: linear(all_heads, hidden, vb.pp("output.dense"))?,
            output_norm: layer_norm(hidden, projector.layer_norm_eps, vb.pp("output.LayerNorm"))?,
            num_heads: projector.num_attention_heads,
            head_dim,
        })
    }

    fn forward(&self, x: &Tensor, encoder_hidden: Option<&Tensor>) -> Result<Tensor> {
        let kv_input = encoder_hidden.unwrap_or(x);
        let q = self.heads(&self.query.forward(x)?)?;
        let k = self.heads(&self.key.forward(kv_input)?)?;
        let v = self.heads(&self.value.forward(kv_input)?)?;
        let context = match precision::qformer_fused_attention(q.device())
            .then(|| try_fused_self_attention(&q, &k, &v, None, self.head_dim, false))
            .transpose()?
            .flatten()
        {
            Some(context) => context,
            None => {
                let scores = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
                ops::softmax(&scores.to_dtype(DType::F32)?, D::Minus1)?
                    .to_dtype(q.dtype())?
                    .matmul(&v)?
            }
        };
        let context = context.transpose(1, 2)?.flatten_from(D::Minus2)?;
        let out = self.output_dense.forward(&context)?;
        self.output_norm
            .forward(&out.broadcast_add(x)?)
            .map_err(Error::from)
    }

    fn heads(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        x.reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
            .map_err(Error::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::architectures::granite_speech5::asr::config::tests::fixture_dir;
    use candle_core::Device;
    use candle_nn::VarMap;

    #[test]
    fn projects_padded_windows_to_three_queries_each() {
        let device = Device::Cpu;
        let mut config = GraniteSpeech5Config::load(&fixture_dir()).unwrap();
        // A narrow projector with the published window geometry.
        config.projector_config.hidden_size = 8;
        config.projector_config.intermediate_size = 16;
        config.projector_config.num_attention_heads = 2;
        config.projector_config.encoder_hidden_size = 6;
        config.text_config.hidden_size = 4;
        let vars = VarMap::new();
        let vb = VarBuilder::from_varmap(&vars, DType::F32, &device);
        let projector = QFormerProjector::load(&config, vb).unwrap();
        assert_eq!(projector.window_size(), 15);
        assert_eq!(projector.num_queries(), 3);
        assert_eq!(projector.layer_count(), 2);
        // 31 frames: two full windows and a padded third.
        let hidden = Tensor::ones((1, 31, 6), DType::F32, &device).unwrap();
        let output = projector.forward(&hidden).unwrap();
        assert_eq!(output.dims(), &[1, 9, 4]);
        assert_eq!(projector.window_count(31), 3);
        assert_eq!(projector.window_count(30), 2);
    }
}
