//! Nemotron-H attention: GQA over the paged KV cache, causal, scale
//! `head_dim^-0.5`, no q/k norm, no bias and no RoPE anywhere (the config's
//! `rope_theta` is dead). The KV domain stores raw keys.

use candle_core::{DType, Device, Tensor};

use super::super::checkpoint::NemotronCheckpoint;
use super::super::config::{AttentionConfig, NemotronTranscribeConfig};
use super::super::linear::{fake_quantize_e4m3, NtLinear, Pass};
use super::super::precision::{requantize_to_shared_scale, Fp8Mode, Fp8Shard};
use crate::error::Result;
use crate::models::shared::attention::physical::{PhysicalPagedKvCache, PreparedPhysicalPagedStep};

pub struct AttentionMixer {
    config: AttentionConfig,
    q_proj: NtLinear,
    k_proj: NtLinear,
    v_proj: NtLinear,
    o_proj: NtLinear,
    /// `(k_scale, v_scale)` for the `vllm-fp8` KV round trip.
    kv_scales: Option<(f32, f32)>,
}

impl AttentionMixer {
    pub fn load(
        checkpoint: &NemotronCheckpoint,
        config: &NemotronTranscribeConfig,
        layer: usize,
        device: &Device,
        mode: Fp8Mode,
    ) -> Result<Self> {
        let d = &config.decoder;
        let a = d.attention;
        let x = format!("backbone.layers.{layer}.mixer");
        let kv_rows = a.kv_heads * a.head_dim;
        let shapes = [
            ("q_proj", a.heads * a.head_dim),
            ("k_proj", kv_rows),
            ("v_proj", kv_rows),
        ];
        let k_scale = checkpoint.positive_scalar(&format!("{x}.k_proj.k_scale"))?;
        let v_scale = checkpoint.positive_scalar(&format!("{x}.v_proj.v_scale"))?;
        let [q_proj, k_proj, v_proj] = match mode {
            Fp8Mode::W8A16 => shapes.map(|(name, rows)| {
                NtLinear::load_fp8(
                    checkpoint,
                    &format!("{x}.{name}"),
                    rows,
                    d.hidden,
                    false,
                    device,
                    mode,
                )
            }),
            Fp8Mode::VllmFp8 => {
                // vLLM's fused QKV: one weight scale (the largest) for all
                // three shards, and the largest input scale.
                let weights = shapes
                    .iter()
                    .map(|(name, rows)| checkpoint.fp8(&format!("{x}.{name}"), *rows, d.hidden))
                    .collect::<Result<Vec<_>>>()?;
                let shards = requantize_to_shared_scale(
                    &weights
                        .iter()
                        .map(|w| Fp8Shard {
                            bytes: w.bytes.to_vec(),
                            weight_scale: w.weight_scale,
                            input_scale: w.input_scale,
                        })
                        .collect::<Vec<_>>(),
                );
                let mut built = shards.iter().zip(&shapes).map(|(shard, (_, rows))| {
                    NtLinear::from_fp8(
                        super::super::checkpoint::Fp8Weight {
                            bytes: &shard.bytes,
                            rows: *rows,
                            cols: d.hidden,
                            weight_scale: shard.weight_scale,
                            input_scale: shard.input_scale,
                        },
                        *rows,
                        d.hidden,
                        device,
                        mode,
                    )
                });
                [
                    built.next().expect("three shards"),
                    built.next().expect("three shards"),
                    built.next().expect("three shards"),
                ]
            }
        };
        Ok(Self {
            config: a,
            q_proj: q_proj?,
            k_proj: k_proj?,
            v_proj: v_proj?,
            o_proj: NtLinear::load_fp8(
                checkpoint,
                &format!("{x}.o_proj"),
                d.hidden,
                a.heads * a.head_dim,
                false,
                device,
                mode,
            )?,
            kv_scales: (mode == Fp8Mode::VllmFp8).then_some((k_scale, v_scale)),
        })
    }

    pub fn resident_bytes(&self) -> usize {
        [&self.q_proj, &self.k_proj, &self.v_proj, &self.o_proj]
            .iter()
            .map(|l| l.resident_bytes())
            .sum()
    }

    pub fn projections(&self) -> [&NtLinear; 4] {
        [&self.q_proj, &self.k_proj, &self.v_proj, &self.o_proj]
    }

    /// Projects `h` `[n, hidden]` to `(q [n, heads, d], k, v [n, kv, d])`.
    pub fn project(&self, h: &Tensor, pass: Pass) -> Result<(Tensor, Tensor, Tensor)> {
        let a = self.config;
        let n = h.dim(0)?;
        let q = self
            .q_proj
            .forward(h, pass)?
            .reshape((n, a.heads, a.head_dim))?;
        let mut k = self.k_proj.forward(h, pass)?;
        let mut v = self.v_proj.forward(h, pass)?;
        if let Some((k_scale, v_scale)) = self.kv_scales {
            // vllm-fp8: K and V round-trip through the FP8 KV cache.
            let round_trip = |t: &Tensor, scale: f32| -> Result<Tensor> {
                let mut values = t.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
                fake_quantize_e4m3(&mut values, scale);
                Ok(Tensor::from_vec(values, t.dims(), t.device())?)
            };
            k = round_trip(&k, k_scale)?;
            v = round_trip(&v, v_scale)?;
        }
        Ok((
            q,
            k.reshape((n, a.kv_heads, a.head_dim))?,
            v.reshape((n, a.kv_heads, a.head_dim))?,
        ))
    }

    /// Attends one row's `q`, `k`, `v` (`[tokens, heads|kv, d]`) over its
    /// cache at physical layer `layer`, writing `k` and `v`. Returns
    /// `[tokens, heads * d]` F32.
    pub(crate) fn attend(
        &self,
        cache: &PhysicalPagedKvCache,
        prepared: &mut PreparedPhysicalPagedStep,
        layer: usize,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        kv_dtype: DType,
    ) -> Result<Tensor> {
        let a = self.config;
        let tokens = q.dim(0)?;
        let cast = |t: &Tensor| -> Result<Tensor> { Ok(t.to_dtype(kv_dtype)?.contiguous()?) };
        let out = cache.write_and_attend(
            layer,
            prepared,
            &cast(q)?,
            &cast(k)?,
            &cast(v)?,
            (a.head_dim as f32).powf(-0.5),
        )?;
        Ok(out
            .to_dtype(DType::F32)?
            .reshape((tokens, a.heads * a.head_dim))?)
    }

    pub fn output(&self, attended: &Tensor, pass: Pass) -> Result<Tensor> {
        self.o_proj.forward(attended, pass)
    }
}
