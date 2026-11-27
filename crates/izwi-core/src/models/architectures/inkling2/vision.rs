//! Image input: pixels to cells, and the vision tower that turns an image's
//! cells into one language-model embedding each.
//!
//! This follows transformers' `Inkling2ImageProcessor` and
//! `Inkling2VisionModel` operation by operation, in F32.
//!
//! * **No resize.** An image is used at its own resolution: one image token
//!   per `cell x cell` pixels (40 as published), rounded up per side.
//!   Transparency is composited over a checkerboard of 8-pixel squares, and
//!   the bottom and right are padded to whole cells with -1 on the 0-255
//!   scale (just below black after normalisation to `[-1, 1]`).
//! * **Patches.** A cell is `merge x merge` patches (2 x 2 of 20 pixels).
//!   Patches are embedded by a linear layer and run through the tower in
//!   block-major order, so a cell's patches are adjacent and the projector
//!   at the end simply concatenates them.
//! * **Position.** A learned square grid of position embeddings is resampled
//!   bilinearly to the image's patch grid, and every attention rotates
//!   queries and keys by the patch's row and column. A class token leads the
//!   image at position (0, 0); patches are 1-indexed.
//! * **Layers.** Pre-norm attention and MLP, each scaled by a learned
//!   per-channel factor before it joins the residual. Images do not attend
//!   to one another: each is encoded on its own.

use candle_core::{DType, Device, Tensor, D};
use candle_nn::ops::softmax;

use super::checkpoint::{Inkling2Checkpoint, VISION_PREFIX};
use super::config::{Inkling2Config, Inkling2VisionConfig};
use crate::error::{Error, Result};

/// transformers' default for this tower; the published config does not set
/// it.
const ROPE_THETA: f32 = 10_000.0;
/// Side of a checkerboard square behind transparent pixels.
const CHECKER: usize = 8;
const CHECKER_LIGHT: i32 = 204;
const CHECKER_DARK: i32 = 153;
/// Most bytes of attention scores computed at once: heads are taken a few at
/// a time so a large image's scores (quadratic in its patches) stay bounded.
pub const ATTENTION_CHUNK_BYTES: usize = 256 << 20;
/// The padding value on the 0-255 scale.
const PAD_VALUE: f32 = -1.0;

fn vision_error(message: impl Into<String>) -> Error {
    Error::InferenceError(format!("Inkling 2 vision: {}", message.into()))
}

/// An image cut into cells: `rows * columns` image tokens.
#[derive(Debug, Clone, PartialEq)]
pub struct Inkling2Image {
    pub rows: usize,
    pub columns: usize,
    /// `[rows * columns, cell, cell, 3]` in `[-1, 1]`, cells in raster order.
    pub pixels: Vec<f32>,
}

impl Inkling2Image {
    /// Image tokens this image occupies in the prompt.
    pub fn tokens(&self) -> usize {
        self.rows * self.columns
    }
}

/// Image tokens of a `height x width` image.
pub fn image_tokens(height: usize, width: usize, cell: usize) -> usize {
    height.div_ceil(cell) * width.div_ceil(cell)
}

/// Cuts `pixels` (`[height, width, channels]` U8, 3 or 4 channels, 4 being
/// RGBA) into cells.
pub fn preprocess_image(
    pixels: &[u8],
    height: usize,
    width: usize,
    channels: usize,
    cell: usize,
) -> Result<Inkling2Image> {
    if height == 0 || width == 0 || cell == 0 {
        return Err(vision_error("an image needs at least one pixel"));
    }
    if channels != 3 && channels != 4 {
        return Err(vision_error(format!(
            "an image has 3 or 4 channels, not {channels}"
        )));
    }
    if pixels.len() != height * width * channels {
        return Err(vision_error(format!(
            "{} bytes for a {height}x{width} image of {channels} channels",
            pixels.len()
        )));
    }
    let (rows, columns) = (height.div_ceil(cell), width.div_ceil(cell));
    let pad = (PAD_VALUE * (1.0 / 255.0) - 0.5) / 0.5;
    let mut out = vec![pad; rows * columns * cell * cell * 3];
    for y in 0..height {
        for x in 0..width {
            let source = &pixels[(y * width + x) * channels..][..channels];
            let at = ((((y / cell) * columns + x / cell) * cell + y % cell) * cell + x % cell) * 3;
            for channel in 0..3 {
                let value = if channels == 4 {
                    let alpha = source[3] as i32;
                    let background = if (y / CHECKER + x / CHECKER).is_multiple_of(2) {
                        CHECKER_LIGHT
                    } else {
                        CHECKER_DARK
                    };
                    (source[channel] as i32 * alpha + background * (255 - alpha) + 127) / 255
                } else {
                    source[channel] as i32
                };
                out[at + channel] = (value as f32 * (1.0 / 255.0) - 0.5) / 0.5;
            }
        }
    }
    Ok(Inkling2Image {
        rows,
        columns,
        pixels: out,
    })
}

/// The two source rows (or columns) and their weights for target `index` of
/// `size` when a `side`-long table is resampled bilinearly: half-pixel
/// centres, edges clamped.
fn bilinear_taps(index: usize, size: usize, side: usize) -> ([usize; 2], [f32; 2]) {
    let source = (index as f32 + 0.5) * side as f32 / size as f32 - 0.5;
    let floor = source.floor();
    let clamp = |tap: f32| (tap.max(0.0) as usize).min(side - 1);
    let fraction = source - floor;
    (
        [clamp(floor), clamp(floor + 1.0)],
        [
            (1.0 - fraction).max(0.0),
            (1.0 - (fraction - 1.0).abs()).max(0.0),
        ],
    )
}

struct Linear {
    /// Transposed: `[in, out]`.
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = x.matmul(&self.weight)?;
        Ok(match &self.bias {
            Some(bias) => out.broadcast_add(bias)?,
            None => out,
        })
    }
}

struct Norm {
    weight: Tensor,
    bias: Tensor,
}

impl Norm {
    fn forward(&self, x: &Tensor, eps: f64) -> Result<Tensor> {
        let centred = x.broadcast_sub(&x.mean_keepdim(D::Minus1)?)?;
        let variance = centred.sqr()?.mean_keepdim(D::Minus1)?;
        Ok(centred
            .broadcast_div(&(variance + eps)?.sqrt()?)?
            .broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)?)
    }
}

struct Layer {
    norm_attention: Norm,
    norm_mlp: Norm,
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    mlp_in: Linear,
    mlp_out: Linear,
    scale_attention: Tensor,
    scale_mlp: Tensor,
}

/// The vision tower and its projector, on one device in F32.
pub struct Inkling2Vision {
    config: Inkling2VisionConfig,
    text_hidden: usize,
    device: Device,
    patch_embedding: Linear,
    /// `[grid * grid, hidden]`.
    positions: Tensor,
    /// The class token with its position embedding, `[1, hidden]`.
    class_token: Tensor,
    norm_in: Norm,
    layers: Vec<Layer>,
    project_in: Linear,
    project_out: Linear,
}

impl Inkling2Vision {
    pub fn load(
        checkpoint: &Inkling2Checkpoint,
        config: &Inkling2Config,
        device: &Device,
    ) -> Result<Self> {
        let v = config
            .vision
            .clone()
            .ok_or_else(|| vision_error("the checkpoint's config has no vision tower"))?;
        if v.hidden % v.heads != 0 || (v.hidden / v.heads) % 4 != 0 {
            return Err(vision_error(
                "the head width must be a multiple of four for the two-axis rotation",
            ));
        }
        let tensor = |name: String| checkpoint.tensor(&name, device, DType::F32);
        let linear = |name: String, bias: bool| -> Result<Linear> {
            Ok(Linear {
                weight: tensor(format!("{name}.weight"))?.t()?.contiguous()?,
                bias: bias.then(|| tensor(format!("{name}.bias"))).transpose()?,
            })
        };
        let norm = |name: String| -> Result<Norm> {
            Ok(Norm {
                weight: tensor(format!("{name}.weight"))?,
                bias: tensor(format!("{name}.bias"))?,
            })
        };
        let backbone = format!("{VISION_PREFIX}.backbone");
        let class_token = (tensor(format!("{backbone}.embeddings.class_embedding"))?
            + tensor(format!("{backbone}.embeddings.cls_position_embedding"))?)?
        .unsqueeze(0)?;
        let layers = (0..v.layers)
            .map(|layer| {
                let l = format!("{backbone}.layers.{layer}");
                Ok(Layer {
                    norm_attention: norm(format!("{l}.ln_1"))?,
                    norm_mlp: norm(format!("{l}.ln_2"))?,
                    query: linear(format!("{l}.self_attn.q_proj"), true)?,
                    key: linear(format!("{l}.self_attn.k_proj"), true)?,
                    value: linear(format!("{l}.self_attn.v_proj"), true)?,
                    output: linear(format!("{l}.self_attn.out_proj"), true)?,
                    mlp_in: linear(format!("{l}.mlp.c_fc"), true)?,
                    mlp_out: linear(format!("{l}.mlp.c_proj"), true)?,
                    scale_attention: tensor(format!("{l}.ls_1.gamma"))?,
                    scale_mlp: tensor(format!("{l}.ls_2.gamma"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            patch_embedding: Linear {
                weight: tensor(format!("{backbone}.embeddings.patch_embedding.weight"))?
                    .t()?
                    .contiguous()?,
                bias: None,
            },
            positions: tensor(format!("{backbone}.embeddings.position_embedding"))?,
            class_token,
            norm_in: norm(format!("{backbone}.ln_pre"))?,
            layers,
            project_in: linear(format!("{VISION_PREFIX}.proj_in"), false)?,
            project_out: linear(format!("{VISION_PREFIX}.proj_out"), false)?,
            text_hidden: config.hidden,
            device: device.clone(),
            config: v,
        })
    }

    pub fn config(&self) -> &Inkling2VisionConfig {
        &self.config
    }

    /// Position embeddings of the image's patches in block-major order,
    /// `[patches, hidden]`, and each patch's (row, column).
    fn patch_positions(&self, image: &Inkling2Image) -> Result<(Tensor, Vec<(usize, usize)>)> {
        let (merge, side) = (self.config.merge, self.config.grid);
        let (height, width) = (image.rows * merge, image.columns * merge);
        let patches = height * width;
        let mut coordinates = Vec::with_capacity(patches);
        let mut indices = Vec::with_capacity(patches * 4);
        let mut weights = Vec::with_capacity(patches * 4);
        for cell_row in 0..image.rows {
            for cell_column in 0..image.columns {
                for in_row in 0..merge {
                    for in_column in 0..merge {
                        let (row, column) =
                            (cell_row * merge + in_row, cell_column * merge + in_column);
                        coordinates.push((row, column));
                        let (row_taps, row_weights) = bilinear_taps(row, height, side);
                        let (column_taps, column_weights) = bilinear_taps(column, width, side);
                        for r in 0..2 {
                            for c in 0..2 {
                                indices.push((row_taps[r] * side + column_taps[c]) as u32);
                                weights.push(row_weights[r] * column_weights[c]);
                            }
                        }
                    }
                }
            }
        }
        let indices = Tensor::from_vec(indices, patches * 4, &self.device)?;
        let weights = Tensor::from_vec(weights, (patches, 4, 1), &self.device)?;
        let taps =
            self.positions
                .index_select(&indices, 0)?
                .reshape((patches, 4, self.config.hidden))?;
        Ok((taps.broadcast_mul(&weights)?.sum(1)?, coordinates))
    }

    /// Cosines and sines that rotate a head's interleaved channel pairs by a
    /// token's column (first half of the head) and row (second half), each
    /// `[tokens, 1, head_dim]`. The class token leads at (0, 0).
    fn rotation(&self, coordinates: &[(usize, usize)]) -> Result<(Tensor, Tensor)> {
        let head_dim = self.config.hidden / self.config.heads;
        let spatial = head_dim / 2;
        let frequencies = (0..spatial)
            .step_by(2)
            .map(|i| 1.0 / ROPE_THETA.powf(i as f32 / spatial as f32))
            .collect::<Vec<_>>();
        let tokens = coordinates.len() + 1;
        let mut cos = Vec::with_capacity(tokens * head_dim);
        let mut sin = Vec::with_capacity(tokens * head_dim);
        let positions = std::iter::once((0usize, 0usize)).chain(
            coordinates
                .iter()
                .map(|(row, column)| (row + 1, column + 1)),
        );
        for (row, column) in positions {
            for position in [column, row] {
                for frequency in &frequencies {
                    let angle = position as f32 * frequency;
                    cos.extend([angle.cos(); 2]);
                    sin.extend([angle.sin(); 2]);
                }
            }
        }
        Ok((
            Tensor::from_vec(cos, (tokens, 1, head_dim), &self.device)?,
            Tensor::from_vec(sin, (tokens, 1, head_dim), &self.device)?,
        ))
    }

    /// One language-model embedding per cell of `image`, `[cells,
    /// text_hidden]` in F32, cells in raster order.
    pub fn encode(&self, image: &Inkling2Image) -> Result<Tensor> {
        let v = &self.config;
        let (merge, patch, hidden, heads) = (v.merge, v.patch, v.hidden, v.heads);
        let cell = v.cell();
        let cells = image.tokens();
        if cells == 0 || image.pixels.len() != cells * cell * cell * 3 {
            return Err(vision_error("the image's pixels do not fill its cells"));
        }
        let head_dim = hidden / heads;

        let pixels = Tensor::from_slice(
            &image.pixels,
            (cells, merge, patch, merge, patch, 3),
            &self.device,
        )?;
        let patches = pixels
            .transpose(2, 3)?
            .reshape((cells * merge * merge, patch * patch * 3))?;
        let (positions, coordinates) = self.patch_positions(image)?;
        let states = (self.patch_embedding.forward(&patches)? + positions)?;
        let states = Tensor::cat(&[&self.class_token, &states], 0)?;
        let mut states = self.norm_in.forward(&states, v.layer_norm_eps)?;
        let tokens = states.dim(0)?;

        let (cos, sin) = self.rotation(&coordinates)?;
        let rotate = |x: Tensor| -> Result<Tensor> {
            let pairs = x.reshape((tokens, heads, head_dim / 2, 2))?;
            let turned = Tensor::cat(
                &[&pairs.narrow(3, 1, 1)?.neg()?, &pairs.narrow(3, 0, 1)?],
                3,
            )?
            .reshape((tokens, heads, head_dim))?;
            Ok((x.broadcast_mul(&cos)? + turned.broadcast_mul(&sin)?)?)
        };
        let heads_first = |x: Tensor| -> Result<Tensor> { Ok(x.transpose(0, 1)?.contiguous()?) };
        let scale = (head_dim as f64).powf(-0.5);

        for layer in &self.layers {
            let x = layer.norm_attention.forward(&states, v.layer_norm_eps)?;
            let split = |x: Tensor| x.reshape((tokens, heads, head_dim));
            let query = heads_first(rotate(split(layer.query.forward(&x)?)?)?)?;
            let key = heads_first(rotate(split(layer.key.forward(&x)?)?)?)?;
            let value = heads_first(split(layer.value.forward(&x)?)?)?;
            let per_chunk = (ATTENTION_CHUNK_BYTES / (tokens * tokens * 4)).clamp(1, heads);
            let mut attended = Vec::with_capacity(heads.div_ceil(per_chunk));
            for first in (0..heads).step_by(per_chunk) {
                let count = per_chunk.min(heads - first);
                let keys = key.narrow(0, first, count)?.transpose(1, 2)?.contiguous()?;
                let scores = (query.narrow(0, first, count)?.matmul(&keys)? * scale)?;
                attended
                    .push(softmax(&scores, D::Minus1)?.matmul(&value.narrow(0, first, count)?)?);
            }
            let attended = Tensor::cat(&attended, 0)?
                .transpose(0, 1)?
                .reshape((tokens, hidden))?;
            let attended = layer.output.forward(&attended)?;
            states = (states + attended.broadcast_mul(&layer.scale_attention)?)?;
            let x = layer.norm_mlp.forward(&states, v.layer_norm_eps)?;
            let x = layer
                .mlp_out
                .forward(&layer.mlp_in.forward(&x)?.gelu_erf()?)?;
            states = (states + x.broadcast_mul(&layer.scale_mlp)?)?;
        }

        let merged = states
            .narrow(0, 1, tokens - 1)?
            .reshape((cells, merge * merge * hidden))?;
        let out = self
            .project_out
            .forward(&self.project_in.forward(&merged)?.gelu_erf()?)?;
        if out.dims() != [cells, self.text_hidden] {
            return Err(vision_error("the projector's output has the wrong shape"));
        }
        Ok(out)
    }
}
