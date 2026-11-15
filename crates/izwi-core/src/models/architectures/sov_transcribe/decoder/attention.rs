//! Gemma 4 attention over the paged KV cache.
//!
//! * `q = rope(q_norm(q_proj(h)))`, `k = rope(k_norm(k_proj(h)))`,
//!   `v = v_norm(v_proj(h))`; the q and k norms have a gain, `v_norm` has
//!   none.
//! * Full layers have no `v_proj`: `v = v_norm(k_proj(h))`, taken **before**
//!   `k_norm` and the rotation.
//! * The softmax scale is 1.0 (the q/k norms take its place) and there is no
//!   attention-logit softcap.
//! * Sliding layers see `window` keys, the query's own position included.
//!
//! The KV domain stores rotated keys.

use candle_core::{DType, Device, Tensor};

use super::super::checkpoint::{SovCheckpoint, DECODER_PREFIX};
use super::super::config::{SovAttentionGeometry, SovTranscribeConfig};
use super::super::encoder::linear;
use super::norm::RmsNorm;
use super::rope::Rope;
use crate::error::Result;
use crate::models::shared::attention::physical::{PhysicalPagedKvCache, PreparedPhysicalPagedStep};

pub struct SovAttention {
    geometry: SovAttentionGeometry,
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Option<Tensor>,
    o_proj: Tensor,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    v_norm: RmsNorm,
    rope: Rope,
    resident_bytes: usize,
}

impl SovAttention {
    pub fn load(
        checkpoint: &SovCheckpoint,
        config: &SovTranscribeConfig,
        layer: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let geometry = *config.decoder.geometry(layer);
        let eps = config.decoder.norm_eps;
        let x = format!("{DECODER_PREFIX}.layers.{layer}.self_attn");
        let weight = |name: &str| checkpoint.tensor(&format!("{x}.{name}.weight"), device, dtype);
        let q_proj = weight("q_proj")?;
        let k_proj = weight("k_proj")?;
        let v_proj = geometry.own_values.then(|| weight("v_proj")).transpose()?;
        let o_proj = weight("o_proj")?;
        let resident_bytes = [Some(&q_proj), Some(&k_proj), v_proj.as_ref(), Some(&o_proj)]
            .into_iter()
            .flatten()
            .map(|t| t.elem_count() * dtype.size_in_bytes())
            .sum::<usize>();
        let q_norm = RmsNorm::load(checkpoint, &format!("{x}.q_norm.weight"), device, eps)?;
        let k_norm = RmsNorm::load(checkpoint, &format!("{x}.k_norm.weight"), device, eps)?;
        let resident_bytes = resident_bytes + q_norm.resident_bytes() + k_norm.resident_bytes();
        Ok(Self {
            geometry,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            v_norm: RmsNorm::without_scale(eps),
            rope: Rope::new(&geometry),
            resident_bytes,
        })
    }

    pub fn geometry(&self) -> &SovAttentionGeometry {
        &self.geometry
    }

    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    /// Projects `h [n, hidden]`, whose tokens sit at `positions`, to
    /// `(q [n, heads, d], k [n, kv_heads, d], v [n, kv_heads, d])`.
    pub fn project(&self, h: &Tensor, positions: &[usize]) -> Result<(Tensor, Tensor, Tensor)> {
        let g = &self.geometry;
        let n = h.dim(0)?;
        let (cos, sin) = self.rope.tables(positions, h.device(), h.dtype())?;
        let q = linear(h, &self.q_proj, None)?.reshape((n, g.heads, g.head_dim))?;
        let q = self.rope.apply(&self.q_norm.forward(&q)?, &cos, &sin)?;
        let raw_k = linear(h, &self.k_proj, None)?.reshape((n, g.kv_heads, g.head_dim))?;
        let v = match &self.v_proj {
            Some(v_proj) => linear(h, v_proj, None)?.reshape((n, g.kv_heads, g.head_dim))?,
            None => raw_k.clone(),
        };
        let v = self.v_norm.forward(&v)?;
        let k = self.rope.apply(&self.k_norm.forward(&raw_k)?, &cos, &sin)?;
        Ok((q, k, v))
    }

    /// Attends one row's `q`, `k`, `v` over its cache at physical layer
    /// `layer`, writing `k` and `v`. Returns `[tokens, heads * d]` in
    /// `dtype`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attend(
        &self,
        cache: &PhysicalPagedKvCache,
        prepared: &mut PreparedPhysicalPagedStep,
        layer: usize,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        dtype: DType,
    ) -> Result<Tensor> {
        let g = &self.geometry;
        let tokens = q.dim(0)?;
        let kv_dtype = cache.arena().config().dtype;
        let cast = |t: &Tensor| -> Result<Tensor> { Ok(t.to_dtype(kv_dtype)?.contiguous()?) };
        let out = cache.write_and_attend_with_semantics(
            layer,
            prepared,
            &cast(q)?,
            &cast(k)?,
            &cast(v)?,
            1.0,
            None,
            g.window,
        )?;
        Ok(out
            .to_dtype(dtype)?
            .reshape((tokens, g.heads * g.head_dim))?)
    }

    pub fn output(&self, attended: &Tensor) -> Result<Tensor> {
        linear(attended, &self.o_proj, None)
    }
}
