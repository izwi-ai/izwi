//! Maya2's depth decoder (`DepthV2`, ~555M): codebooks 1–31 of a frame from
//! the backbone's hidden state and the frame's codebook-0 code, as upstream's
//! fast path serves it (`serve/fast_depth.py`).
//!
//! Level `L` reads the prefix `[in_proj(h) + lvl₀, emb(cb₀, 0), …,
//! emb(cb_{L−1}, L−1)]` through weight group `GROUP_OF_LEVEL[L]`: 4 pre-LN
//! causal self-attention layers (LayerNorm with bias, 16 heads of 64, GELU
//! FFN), each followed by a cross-attention over the speaker memory, then a
//! final LayerNorm and `heads[L−1]`. Levels 1–8 own private groups; levels
//! 9–16, 17–24 and 25–31 share groups 8, 9 and 10, so within a band the K/V
//! of earlier positions are reused and each later level adds one position.
//!
//! The memory is a single token (the ECAPA projection or the learned null),
//! so softmax over it is exactly 1 and each cross-attention reduces to the
//! constant `o(v(memory))`: four vectors per request, computed once.
//!
//! Rows from any requests run in lockstep (every row is at the same level),
//! and every level is sampled on the device ([`super::sampling`]), so a frame
//! is read back once.

use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{LayerNorm, VarBuilder};

use super::config::{CODEBOOKS, CODEBOOK_SIZE, DEPTH_LEVELS};
use super::sampling::sample_depth_level;
use crate::error::{Error, Result};
use crate::models::shared::attention::flash::try_fused_self_attention;

const LAYERS: usize = 4;
const GROUPS: usize = 11;
const LAYER_NORM_EPS: f64 = 1e-5;

/// The depth decoder's sizes: [`DepthGeometry::RELEASE`] in production,
/// small ones in tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DepthGeometry {
    pub width: usize,
    pub heads: usize,
    pub ffn: usize,
    pub rank: usize,
    pub backbone_width: usize,
    pub ecapa_dim: usize,
    pub codebook_size: usize,
}

impl DepthGeometry {
    pub const RELEASE: Self = Self {
        width: 1_024,
        heads: 16,
        ffn: 3_072,
        rank: 128,
        backbone_width: 2_048,
        ecapa_dim: 192,
        codebook_size: CODEBOOK_SIZE,
    };

    fn head_dim(&self) -> usize {
        self.width / self.heads
    }
}

/// The weight group of each level (index 0, codebook 0, is the backbone's).
pub const GROUP_OF_LEVEL: [usize; CODEBOOKS] = [
    usize::MAX,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    9,
    10,
    10,
    10,
    10,
    10,
    10,
    10,
];

struct DepthLayer {
    n1: LayerNorm,
    qkv: Tensor,
    o: Tensor,
    n2: LayerNorm,
    f1: Tensor,
    f2: Tensor,
}

pub struct Maya2Depth {
    geometry: DepthGeometry,
    dtype: DType,
    device: Device,
    in_proj: Tensor,
    level0: Tensor,
    /// `code_up(code_lo[c + k·2048]) + level_emb[k + 1]`, `[31·2048, 1024]`.
    embed: Tensor,
    groups: Vec<Vec<DepthLayer>>,
    /// The value half of each cross layer's `kv` and its output projection.
    cross_v: Vec<Tensor>,
    cross_o: Vec<Tensor>,
    norm: LayerNorm,
    heads: Vec<Tensor>,
    spk_proj: Tensor,
    null_mem: Tensor,
}

fn linear(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    // One 2-D matmul over every row: `broadcast_matmul` would copy the
    // broadcast weight on each call, about 50× slower on Metal.
    let mut dims = x.dims().to_vec();
    let inner = dims.pop().unwrap_or(0);
    let rows = dims.iter().product::<usize>();
    let out = x.reshape((rows, inner))?.matmul(&weight.t()?)?;
    dims.push(weight.dim(0)?);
    Ok(out.reshape(dims)?)
}

impl Maya2Depth {
    /// Load from `vb` (the `depth` prefix) in `vb`'s dtype.
    pub fn load(vb: VarBuilder, geometry: DepthGeometry) -> Result<Self> {
        let DepthGeometry {
            width,
            ffn,
            rank,
            backbone_width,
            ecapa_dim,
            codebook_size,
            ..
        } = geometry;
        let dtype = vb.dtype();
        let device = vb.device().clone();
        let layer_norm = |vb: VarBuilder| -> Result<LayerNorm> {
            Ok(candle_nn::layer_norm(width, LAYER_NORM_EPS, vb)?)
        };
        let groups = (0..GROUPS)
            .map(|group| {
                (0..LAYERS)
                    .map(|layer| {
                        let vb = vb.pp(format!("groups.{group}.{layer}"));
                        Ok(DepthLayer {
                            n1: layer_norm(vb.pp("n1"))?,
                            qkv: vb.get((3 * width, width), "qkv.weight")?,
                            o: vb.get((width, width), "o.weight")?,
                            n2: layer_norm(vb.pp("n2"))?,
                            f1: vb.get((ffn, width), "f1.weight")?,
                            f2: vb.get((width, ffn), "f2.weight")?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let mut cross_v = Vec::with_capacity(LAYERS);
        let mut cross_o = Vec::with_capacity(LAYERS);
        for layer in 0..LAYERS {
            let cross = vb.pp(format!("cross.{layer}"));
            let kv = cross.get((2 * width, width), "kv.weight")?;
            cross_v.push(kv.narrow(0, width, width)?.to_dtype(DType::F32)?);
            cross_o.push(
                cross
                    .get((width, width), "o.weight")?
                    .to_dtype(DType::F32)?,
            );
        }
        let level_emb = vb.get((CODEBOOKS, width), "level_emb.weight")?;
        // The prefix table in F32, then the model dtype.
        let code_lo = vb.get((DEPTH_LEVELS * codebook_size, rank), "code_lo.weight")?;
        let code_up = vb.get((width, rank), "code_up.weight")?;
        let embed = code_lo
            .to_dtype(DType::F32)?
            .matmul(&code_up.to_dtype(DType::F32)?.t()?)?
            .reshape((DEPTH_LEVELS, codebook_size, width))?
            .broadcast_add(
                &level_emb
                    .narrow(0, 1, DEPTH_LEVELS)?
                    .to_dtype(DType::F32)?
                    .unsqueeze(1)?,
            )?
            .reshape((DEPTH_LEVELS * codebook_size, width))?
            .to_dtype(dtype)?;
        let heads = (0..DEPTH_LEVELS)
            .map(|level| Ok(vb.get((codebook_size, width), &format!("heads.{level}.weight"))?))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            in_proj: vb.get((width, backbone_width), "in_proj.weight")?,
            level0: level_emb.get(0)?,
            embed,
            groups,
            cross_v,
            cross_o,
            norm: layer_norm(vb.pp("norm"))?,
            heads,
            spk_proj: vb
                .get((width, ecapa_dim), "spk_proj_ecapa.weight")?
                .to_dtype(DType::F32)?,
            null_mem: vb
                .get((1, 1, width), "null_mem")?
                .reshape(width)?
                .to_dtype(DType::F32)?,
            geometry,
            dtype,
            device,
        })
    }

    pub fn geometry(&self) -> DepthGeometry {
        self.geometry
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The four cross-attention constants `[LAYERS, 1024]` (model dtype) of a
    /// speaker: its L2-normalised ECAPA embedding `[192]`, or the learned
    /// null memory without one. Computed in F32.
    pub fn cross_constants(&self, ecapa: Option<&Tensor>) -> Result<Tensor> {
        let memory = match ecapa {
            Some(ecapa) => linear(
                &ecapa
                    .to_dtype(DType::F32)?
                    .to_device(&self.device)?
                    .reshape((1, self.geometry.ecapa_dim))?,
                &self.spk_proj,
            )?,
            None => self.null_mem.reshape((1, self.geometry.width))?,
        };
        let rows = self
            .cross_v
            .iter()
            .zip(&self.cross_o)
            .map(|(v, o)| linear(&linear(&memory, v)?, o))
            .collect::<Result<Vec<_>>>()?;
        Ok(Tensor::cat(&rows, 0)?.to_dtype(self.dtype)?)
    }

    /// The prefix embedding of code `codes` `[B]` at level index `k` (the
    /// position after it).
    fn prefix_embedding(&self, codes: &Tensor, k: usize) -> Result<Tensor> {
        let offset = Tensor::new((k * self.geometry.codebook_size) as u32, &self.device)?;
        Ok(self
            .embed
            .index_select(&codes.broadcast_add(&offset)?, 0)?
            .unsqueeze(1)?)
    }

    /// One frame for `B` rows: backbone hidden states `h` `[B, 2048]`, their
    /// codebook-0 codes `cb0` `[B]` (U32), cross constants `cross`
    /// `[B, 4, 1024]` and Gumbel noise `gumbel` `[B, 31, 2048]`. Returns the
    /// frame's 32 codes per row, `[B, 32]` U32, on the device.
    pub fn generate_frame(
        &self,
        h: &Tensor,
        cb0: &Tensor,
        cross: &Tensor,
        gumbel: &Tensor,
        temperature: f32,
        top_k: usize,
    ) -> Result<Tensor> {
        self.run_levels(h, cb0, cross, |level, logits| {
            let noise = gumbel.narrow(1, level - 1, 1)?.squeeze(1)?;
            sample_depth_level(logits, temperature, top_k, &noise)
        })
    }

    /// The levels of one frame, `pick` choosing each level's codes `[B]`
    /// from its logits `[B, 2048]` (F32).
    fn run_levels(
        &self,
        h: &Tensor,
        cb0: &Tensor,
        cross: &Tensor,
        mut pick: impl FnMut(usize, &Tensor) -> Result<Tensor>,
    ) -> Result<Tensor> {
        let batch = h.dim(0)?;
        let width = self.geometry.width;
        if cb0.dims() != [batch] || cross.dims() != [batch, LAYERS, width] {
            return Err(Error::InvalidInput(
                "Maya2 depth rows, codes and speakers disagree".into(),
            ));
        }
        let cross = (0..LAYERS)
            .map(|layer| Ok(cross.narrow(1, layer, 1)?))
            .collect::<Result<Vec<_>>>()?;
        let first = linear(&h.to_dtype(self.dtype)?, &self.in_proj)?
            .broadcast_add(&self.level0)?
            .unsqueeze(1)?;
        let mut prefix = vec![first, self.prefix_embedding(cb0, 0)?];
        let mut codes = vec![cb0.clone()];
        // Per layer, the K/V `[B, H, n, d]` of the current shared band.
        let mut band: Vec<Option<(Tensor, Tensor)>> = vec![None; LAYERS];
        let mut previous = usize::MAX;
        for level in 1..CODEBOOKS {
            let group = GROUP_OF_LEVEL[level];
            let positions = level + 1;
            let incremental = group == previous;
            let reused_later = level + 1 < CODEBOOKS && GROUP_OF_LEVEL[level + 1] == group;
            let mut x = if incremental {
                prefix[level].clone()
            } else {
                Tensor::cat(&prefix[..positions], 1)?
            };
            for (layer, block) in self.groups[group].iter().enumerate() {
                let tokens = x.dim(1)?;
                let (q, k, v) = self.project(block, &x)?;
                let attended = if incremental {
                    let (k, v) = match band[layer].take() {
                        Some((pk, pv)) => {
                            (Tensor::cat(&[&pk, &k], 2)?, Tensor::cat(&[&pv, &v], 2)?)
                        }
                        None => {
                            return Err(Error::InferenceError(
                                "Maya2 depth band lost its keys".into(),
                            ))
                        }
                    };
                    let out = attention(&q, &k, &v, false)?;
                    band[layer] = Some((k, v));
                    out
                } else {
                    if reused_later {
                        band[layer] = Some((k.clone(), v.clone()));
                    }
                    if layer == LAYERS - 1 {
                        // Only the final position is read after the last
                        // layer, and it sees every key.
                        x = x.narrow(1, tokens - 1, 1)?;
                        attention(&q.narrow(2, tokens - 1, 1)?, &k, &v, false)?
                    } else {
                        attention(&q, &k, &v, true)?
                    }
                };
                x = self.finish_layer(block, &x, &attended, &cross[layer])?;
            }
            if !reused_later {
                band = vec![None; LAYERS];
            }
            let code = pick(level, &self.level_logits(&x, level)?)?;
            if positions < CODEBOOKS {
                prefix.push(self.prefix_embedding(&code, level)?);
            }
            codes.push(code);
            previous = group;
        }
        Ok(Tensor::stack(&codes, 1)?)
    }

    /// q, k and v `[B, H, T, d]` of `x` `[B, T, width]`.
    fn project(&self, block: &DepthLayer, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch, tokens, _) = x.dims3()?;
        let qkv = linear(&block.n1.forward(x)?, &block.qkv)?.reshape((
            batch,
            tokens,
            3,
            self.geometry.heads,
            self.geometry.head_dim(),
        ))?;
        let heads = |index: usize| -> Result<Tensor> {
            Ok(qkv
                .narrow(2, index, 1)?
                .squeeze(2)?
                .transpose(1, 2)?
                .contiguous()?)
        };
        Ok((heads(0)?, heads(1)?, heads(2)?))
    }

    /// The rest of a layer after attention: output projection, FFN and the
    /// speaker's cross constant.
    fn finish_layer(
        &self,
        block: &DepthLayer,
        x: &Tensor,
        attended: &Tensor,
        cross: &Tensor,
    ) -> Result<Tensor> {
        let (batch, tokens, width) = x.dims3()?;
        let attended = attended.transpose(1, 2)?.reshape((batch, tokens, width))?;
        let x = (x + linear(&attended, &block.o)?)?;
        let ffn = linear(
            &linear(&block.n2.forward(&x)?, &block.f1)?.gelu_erf()?,
            &block.f2,
        )?;
        Ok((x + ffn)?.broadcast_add(cross)?)
    }

    /// `heads[level − 1]` of the final norm of `x`'s last position, F32.
    fn level_logits(&self, x: &Tensor, level: usize) -> Result<Tensor> {
        let tokens = x.dim(1)?;
        let last = self
            .norm
            .forward(&x.narrow(1, tokens - 1, 1)?.squeeze(1)?)?;
        Ok(linear(&last, &self.heads[level - 1])?.to_dtype(DType::F32)?)
    }
}

/// Scaled dot-product attention over `[B, H, T, d]`: the fused kernel on a
/// GPU, otherwise in F32; `causal` masks keys after each query (query and
/// key lengths equal).
fn attention(q: &Tensor, k: &Tensor, v: &Tensor, causal: bool) -> Result<Tensor> {
    if let Some(out) = try_fused_self_attention(q, k, v, None, q.dim(D::Minus1)?, causal)? {
        return Ok(out);
    }
    let dtype = q.dtype();
    let scale = 1.0 / (q.dim(D::Minus1)? as f64).sqrt();
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;
    let mut scores = (q.matmul(&k.t()?)? * scale)?;
    if causal {
        let length = scores.dim(D::Minus1)?;
        let mask = (0..length)
            .flat_map(|query| {
                (0..length).map(move |key| if key > query { f32::NEG_INFINITY } else { 0.0 })
            })
            .collect::<Vec<_>>();
        let mask = Tensor::from_vec(mask, (length, length), q.device())?;
        scores = scores.broadcast_add(&mask)?;
    }
    let weights = candle_nn::ops::softmax_last_dim(&scores)?;
    Ok(weights.matmul(&v)?.to_dtype(dtype)?)
}

#[cfg(test)]
impl Maya2Depth {
    /// Every level's logits for given `codes` `[B, 32]` through upstream's
    /// per-level loop (`DepthV2._forward_loop`): the whole prefix through
    /// every layer for each level, no band reuse.
    fn reference_logits(&self, h: &Tensor, codes: &Tensor, cross: &Tensor) -> Result<Vec<Tensor>> {
        let cross = (0..LAYERS)
            .map(|layer| Ok(cross.narrow(1, layer, 1)?))
            .collect::<Result<Vec<_>>>()?;
        let mut prefix = vec![linear(&h.to_dtype(self.dtype)?, &self.in_proj)?
            .broadcast_add(&self.level0)?
            .unsqueeze(1)?];
        for k in 0..DEPTH_LEVELS {
            let column = codes.narrow(1, k, 1)?.squeeze(1)?.contiguous()?;
            prefix.push(self.prefix_embedding(&column, k)?);
        }
        (1..CODEBOOKS)
            .map(|level| {
                let mut x = Tensor::cat(&prefix[..level + 1], 1)?;
                for (layer, block) in self.groups[GROUP_OF_LEVEL[level]].iter().enumerate() {
                    let (q, k, v) = self.project(block, &x)?;
                    let attended = attention(&q, &k, &v, true)?;
                    x = self.finish_layer(block, &x, &attended, &cross[layer])?;
                }
                self.level_logits(&x, level)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use candle_nn::VarMap;

    use super::*;
    use crate::models::architectures::maya2::sampling::depth_gumbel;
    use crate::models::shared::noise::{mix, SplitMix64};

    const TINY: DepthGeometry = DepthGeometry {
        width: 16,
        heads: 2,
        ffn: 24,
        rank: 4,
        backbone_width: 12,
        ecapa_dim: 6,
        codebook_size: 32,
    };

    /// Random weights for every tensor a tiny depth decoder reads.
    fn tiny_depth() -> Maya2Depth {
        let varmap = VarMap::new();
        Maya2Depth::load(
            VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu),
            TINY,
        )
        .unwrap();
        let mut seed = 11u64;
        for var in varmap.data().lock().unwrap().values() {
            let shape = var.shape().clone();
            let values = (0..shape.elem_count())
                .map(|_| {
                    seed = mix(seed.wrapping_add(1));
                    ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.8
                })
                .collect::<Vec<_>>();
            var.set(&Tensor::from_vec(values, shape, &Device::Cpu).unwrap())
                .unwrap();
        }
        Maya2Depth::load(
            VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu),
            TINY,
        )
        .unwrap()
    }

    fn random(rows: usize, cols: usize, salt: u64) -> Tensor {
        let values = (0..rows * cols)
            .map(|i| ((mix(i as u64 ^ salt) >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0)
            .collect::<Vec<_>>();
        Tensor::from_vec(values, (rows, cols), &Device::Cpu).unwrap()
    }

    #[test]
    fn band_reuse_matches_the_per_level_loop() {
        let depth = tiny_depth();
        let rows = 3;
        let h = random(rows, TINY.backbone_width, 1);
        let codes = (0..rows * CODEBOOKS)
            .map(|i| (mix(i as u64 + 99) % TINY.codebook_size as u64) as u32)
            .collect::<Vec<_>>();
        let codes = Tensor::from_vec(codes, (rows, CODEBOOKS), &Device::Cpu).unwrap();
        let ecapa = random(1, TINY.ecapa_dim, 2)
            .reshape(TINY.ecapa_dim)
            .unwrap();
        let speakers = [
            depth.cross_constants(Some(&ecapa)).unwrap(),
            depth.cross_constants(None).unwrap(),
            depth
                .cross_constants(Some(&(ecapa * -1.0).unwrap()))
                .unwrap(),
        ];
        let cross = Tensor::stack(&speakers, 0).unwrap();
        let mut fast = Vec::new();
        let cb0 = codes
            .narrow(1, 0, 1)
            .unwrap()
            .squeeze(1)
            .unwrap()
            .contiguous()
            .unwrap();
        let out = depth
            .run_levels(&h, &cb0, &cross, |level, logits| {
                fast.push(logits.clone());
                Ok(codes.narrow(1, level, 1)?.squeeze(1)?.contiguous()?)
            })
            .unwrap();
        assert_eq!(
            out.to_vec2::<u32>().unwrap(),
            codes.to_vec2::<u32>().unwrap()
        );
        let reference = depth.reference_logits(&h, &codes, &cross).unwrap();
        assert_eq!(fast.len(), DEPTH_LEVELS);
        for (level, (a, b)) in fast.iter().zip(&reference).enumerate() {
            let diff = (a - b)
                .unwrap()
                .abs()
                .unwrap()
                .flatten_all()
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(diff < 1e-4, "level {} differs by {diff}", level + 1);
        }
    }

    #[test]
    fn a_rows_frame_does_not_depend_on_its_batch() {
        let depth = tiny_depth();
        let h = random(2, TINY.backbone_width, 5);
        let cb0 = Tensor::new(&[3u32, 17], &Device::Cpu).unwrap();
        let cross = Tensor::stack(
            &[
                depth.cross_constants(None).unwrap(),
                depth.cross_constants(None).unwrap(),
            ],
            0,
        )
        .unwrap();
        let mut a = SplitMix64::new(21);
        let mut b = SplitMix64::new(22);
        let noise = depth_gumbel(
            &mut [&mut a, &mut b],
            DEPTH_LEVELS,
            TINY.codebook_size,
            &Device::Cpu,
        )
        .unwrap();
        let both = depth
            .generate_frame(&h, &cb0, &cross, &noise, 0.7, 4)
            .unwrap()
            .to_vec2::<u32>()
            .unwrap();
        let mut a = SplitMix64::new(21);
        let alone_noise = depth_gumbel(
            &mut [&mut a],
            DEPTH_LEVELS,
            TINY.codebook_size,
            &Device::Cpu,
        )
        .unwrap();
        let alone = depth
            .generate_frame(
                &h.narrow(0, 0, 1).unwrap(),
                &cb0.narrow(0, 0, 1).unwrap(),
                &cross.narrow(0, 0, 1).unwrap(),
                &alone_noise,
                0.7,
                4,
            )
            .unwrap()
            .to_vec2::<u32>()
            .unwrap();
        assert_eq!(alone[0], both[0]);
        assert!(both[0]
            .iter()
            .all(|&code| (code as usize) < TINY.codebook_size));
    }

    #[test]
    fn bands_are_contiguous_runs() {
        assert_eq!(GROUP_OF_LEVEL[1..9], [0, 1, 2, 3, 4, 5, 6, 7]);
        assert!(GROUP_OF_LEVEL[9..17].iter().all(|&g| g == 8));
        assert!(GROUP_OF_LEVEL[17..25].iter().all(|&g| g == 9));
        assert!(GROUP_OF_LEVEL[25..32].iter().all(|&g| g == 10));
    }
}

/// Each op of a depth level timed on Metal at the release shapes, so a slow
/// kernel path shows in `--nocapture` output.
#[cfg(all(test, feature = "metal"))]
mod metal_timing {
    use std::time::Instant;

    use super::*;
    use crate::backends::{BackendPreference, DeviceSelector};

    fn time(label: &str, iterations: usize, device: &Device, mut op: impl FnMut() -> Tensor) {
        let warm = op();
        warm.to_dtype(DType::F32)
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let started = Instant::now();
        let mut last = None;
        for _ in 0..iterations {
            last = Some(op());
        }
        last.unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        device.synchronize().unwrap();
        let per = started.elapsed().as_secs_f64() * 1e6 / iterations as f64;
        eprintln!("{label:<40} {per:>9.1} µs");
    }

    #[test]
    #[ignore = "timing on Metal"]
    fn depth_ops_on_metal_are_timed() {
        let _lease = crate::backends::metal_test_device_or_skip("depth_ops_on_metal_are_timed");
        let profile = DeviceSelector::detect_for_preference(BackendPreference::Metal).unwrap();
        if !matches!(profile.device, Device::Metal(_)) {
            return;
        }
        let device = profile.device;
        let f16 = |shape: &[usize]| {
            Tensor::randn(0f32, 0.02, shape, &device)
                .unwrap()
                .to_dtype(DType::F16)
                .unwrap()
        };
        let n = 200;
        let w_qkv = f16(&[3072, 1024]);
        let w_head = f16(&[2048, 1024]);
        for tokens in [1usize, 9] {
            let x = f16(&[1, tokens, 1024]);
            time(
                &format!("linear qkv x[1,{tokens},1024]"),
                n,
                &device,
                || linear(&x, &w_qkv).unwrap(),
            );
            let x2 = x.reshape((tokens, 1024)).unwrap();
            time(
                &format!("matmul 2-D x[{tokens},1024]·wᵀ"),
                n,
                &device,
                || x2.matmul(&w_qkv.t().unwrap()).unwrap(),
            );
        }
        let ln = candle_nn::LayerNorm::new(f16(&[1024]), f16(&[1024]), LAYER_NORM_EPS);
        let x = f16(&[1, 9, 1024]);
        time("layer_norm [1,9,1024]", n, &device, || {
            ln.forward(&x).unwrap()
        });
        let q = f16(&[1, 16, 9, 64]);
        time("attention causal [1,16,9,64]", n, &device, || {
            attention(&q, &q, &q, true).unwrap()
        });
        time("attention [1,16,1,64]x9", n, &device, || {
            attention(&q.narrow(2, 8, 1).unwrap(), &q, &q, false).unwrap()
        });
        let last = f16(&[1, 1024]);
        time(
            "head logits [1,1024]·[2048]ᵀ → F32",
            n,
            &device,
            || {
                linear(&last, &w_head)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap()
            },
        );
        let logits = Tensor::randn(0f32, 1.0, (1, 2048), &device).unwrap();
        let gumbel = Tensor::randn(0f32, 1.0, (1, 2048), &device).unwrap();
        time("sample_depth_level [1,2048]", n, &device, || {
            sample_depth_level(&logits, 0.7, 64, &gumbel).unwrap()
        });
        time("sort_last_dim [2,1024]", n, &device, || {
            logits
                .reshape((2, 1024))
                .unwrap()
                .sort_last_dim(false)
                .unwrap()
                .0
        });
        let table = f16(&[CODEBOOKS * CODEBOOK_SIZE, 1024]);
        let code = Tensor::new(&[5u32], &device).unwrap();
        time("prefix embedding gather", n, &device, || {
            let offset = Tensor::new(2048u32, &device).unwrap();
            table
                .index_select(&code.broadcast_add(&offset).unwrap(), 0)
                .unwrap()
        });
        time("gelu_erf [1,9,3072]", n, &device, || {
            f16(&[1, 9, 3072]).gelu_erf().unwrap()
        });
        let a = f16(&[1, 9, 1024]);
        time("add [1,9,1024]", n, &device, || (&a + &a).unwrap());
        time("qkv split contiguous", n, &device, || {
            f16(&[1, 9, 3, 16, 64])
                .narrow(2, 0, 1)
                .unwrap()
                .squeeze(2)
                .unwrap()
                .transpose(1, 2)
                .unwrap()
                .contiguous()
                .unwrap()
        });
    }
}
