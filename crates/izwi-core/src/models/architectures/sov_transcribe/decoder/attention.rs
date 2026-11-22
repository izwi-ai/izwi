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
//!
//! The q, k and v projections are one weight (`[q; k; v]` rows, v absent on
//! full layers), so a step runs one product per layer instead of three; the
//! rotation tables come from the decoder, gathered once per step.

use candle_core::{DType, Device, Tensor};

use super::super::checkpoint::{SovCheckpoint, DECODER_PREFIX};
use super::super::config::{SovAttentionGeometry, SovTranscribeConfig};
use super::super::encoder::linear;
use super::norm::RmsNorm;
use super::rope;
use crate::error::Result;
use crate::models::shared::attention::physical::{PhysicalPagedKvCache, PreparedPhysicalPagedStep};

pub struct SovAttention {
    geometry: SovAttentionGeometry,
    /// `[q; k; v]` rows: `heads * d`, `kv_heads * d` and (sliding layers)
    /// `kv_heads * d`.
    qkv_proj: Tensor,
    o_proj: Tensor,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    v_norm: RmsNorm,
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
        let mut parts = vec![weight("q_proj")?, weight("k_proj")?];
        if geometry.own_values {
            parts.push(weight("v_proj")?);
        }
        let qkv_proj = Tensor::cat(&parts, 0)?;
        drop(parts);
        let o_proj = weight("o_proj")?;
        let resident_bytes = (qkv_proj.elem_count() + o_proj.elem_count()) * dtype.size_in_bytes();
        let q_norm = RmsNorm::load(checkpoint, &format!("{x}.q_norm.weight"), device, eps)?;
        let k_norm = RmsNorm::load(checkpoint, &format!("{x}.k_norm.weight"), device, eps)?;
        let resident_bytes = resident_bytes + q_norm.resident_bytes() + k_norm.resident_bytes();
        Ok(Self {
            geometry,
            qkv_proj,
            o_proj,
            q_norm,
            k_norm,
            v_norm: RmsNorm::without_scale(eps, geometry.head_dim, device)?,
            resident_bytes,
        })
    }

    pub fn geometry(&self) -> &SovAttentionGeometry {
        &self.geometry
    }

    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    /// Projects `h [n, hidden]` to `(q [n, heads, d], k [n, kv_heads, d],
    /// v [n, kv_heads, d])`, q and k rotated by `(cos, sin)`
    /// `[n, 1, d / 2]` (the tokens' rows of this geometry's tables).
    pub fn project(
        &self,
        h: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let g = &self.geometry;
        let n = h.dim(0)?;
        let (q_rows, kv_rows) = (g.heads * g.head_dim, g.kv_heads * g.head_dim);
        let qkv = linear(h, &self.qkv_proj, None)?;
        let q = qkv
            .narrow(1, 0, q_rows)?
            .reshape((n, g.heads, g.head_dim))?;
        let q = rope::apply(&self.q_norm.forward(&q)?, cos, sin)?;
        let raw_k = qkv
            .narrow(1, q_rows, kv_rows)?
            .reshape((n, g.kv_heads, g.head_dim))?;
        let v = if g.own_values {
            qkv.narrow(1, q_rows + kv_rows, kv_rows)?
                .reshape((n, g.kv_heads, g.head_dim))?
        } else {
            raw_k.clone()
        };
        let v = self.v_norm.forward(&v)?;
        let k = rope::apply(&self.k_norm.forward(&raw_k)?, cos, sin)?;
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

    /// [`Self::attend`] for one token of every row of a decode step at once
    /// (`q`, `k`, `v` `[rows, heads or kv_heads, d]`). Returns
    /// `[rows, heads * d]` in `dtype`.
    pub(super) fn attend_batch(
        &self,
        batch: &mut super::batch::SovDecodeBatch,
        cache: &PhysicalPagedKvCache,
        layer: usize,
        [q, k, v]: [&Tensor; 3],
        dtype: DType,
    ) -> Result<Tensor> {
        let g = &self.geometry;
        let rows = q.dim(0)?;
        let kv_dtype = cache.arena().config().dtype;
        let cast = |t: &Tensor| -> Result<Tensor> { Ok(t.to_dtype(kv_dtype)?.contiguous()?) };
        let out = batch.attend(
            cache,
            layer,
            [&cast(q)?, &cast(k)?, &cast(v)?],
            g.window.is_some(),
        )?;
        Ok(out.to_dtype(dtype)?.reshape((rows, g.heads * g.head_dim))?)
    }

    pub fn output(&self, attended: &Tensor) -> Result<Tensor> {
        linear(attended, &self.o_proj, None)
    }
}
