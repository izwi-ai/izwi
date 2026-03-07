use std::fs;
use std::io::{Cursor, Read, Seek};
use std::path::{Path, PathBuf};
use std::process::Command;

use base64::Engine;
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{ops, Linear, Module};
use image::codecs::gif::GifDecoder;
use image::{imageops, AnimationDecoder, DynamicImage, RgbImage};

use crate::backends::DeviceProfile;
use crate::backends::{open_gguf_reader, BackendKind};
use crate::error::{Error, Result};
use crate::models::shared::chat::{Qwen35MultimodalInput, Qwen35MultimodalKind};

const DEFAULT_TARGET_GRID: usize = 24;
const MAX_MEDIA_BYTES: usize = 64 * 1024 * 1024;
const ROTARY_THETA: f32 = 10_000.0;
const DEFAULT_VIDEO_MAX_FRAMES: usize = 8;

#[derive(Debug, Clone)]
struct VisionConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_heads: usize,
    num_layers: usize,
    patch_size: usize,
    image_size: usize,
    spatial_merge_size: usize,
    projection_dim: usize,
    layer_norm_eps: f64,
    mean: [f32; 3],
    std: [f32; 3],
    target_grid: usize,
}

impl VisionConfig {
    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads.max(1)
    }

    fn target_image_size(&self) -> usize {
        self.target_grid * self.patch_size
    }

    fn seq_len(&self) -> usize {
        self.target_grid * self.target_grid
    }

    fn merged_seq_len(&self) -> usize {
        self.seq_len() / self.spatial_merge_size.pow(2).max(1)
    }
}

struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    fn new(weight: Tensor, bias: Tensor, eps: f64) -> Self {
        Self { weight, bias, eps }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let mean = x_f32.mean_keepdim(D::Minus1)?;
        let centered = x_f32.broadcast_sub(&mean)?;
        let variance = centered.sqr()?.mean_keepdim(D::Minus1)?;
        let eps = Tensor::new(self.eps as f32, x.device())?;
        let denom = variance.broadcast_add(&eps)?.sqrt()?;
        let normalized = centered.broadcast_div(&denom)?;
        let scaled = normalized.broadcast_mul(&self.weight.to_dtype(DType::F32)?)?;
        let shifted = scaled.broadcast_add(&self.bias.to_dtype(DType::F32)?)?;
        shifted.to_dtype(x_dtype).map_err(Error::from)
    }
}

struct VisionAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl VisionAttention {
    fn new(qkv: Linear, proj: Linear, num_heads: usize, head_dim: usize) -> Self {
        Self {
            qkv,
            proj,
            num_heads,
            head_dim,
        }
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let seq_len = x.dim(0)?;
        let hidden = self.qkv.forward(x)?;
        let qkv = hidden.reshape((seq_len, 3, self.num_heads, self.head_dim))?;
        let mut q = qkv.i((.., 0, .., ..))?;
        let mut k = qkv.i((.., 1, .., ..))?;
        let mut v = qkv.i((.., 2, .., ..))?;

        // Rotary terms must match the explicit F32 attention path below.
        let cos = cos.unsqueeze(1)?.to_dtype(DType::F32)?; // [seq, 1, head_dim]
        let sin = sin.unsqueeze(1)?.to_dtype(DType::F32)?;
        q = apply_rotary(&q.to_dtype(DType::F32)?, &cos, &sin)?;
        k = apply_rotary(&k.to_dtype(DType::F32)?, &cos, &sin)?;
        v = v.to_dtype(DType::F32)?;

        let q = q.transpose(0, 1)?.contiguous()?; // [heads, seq, dim]
        let k = k.transpose(0, 1)?.contiguous()?;
        let v = v.transpose(0, 1)?.contiguous()?;

        let scale = Tensor::new((self.head_dim as f32).sqrt(), x.device())?;
        let mut attn = q.matmul(&k.transpose(1, 2)?)?;
        attn = attn.broadcast_div(&scale)?;
        let attn = ops::softmax(&attn, D::Minus1)?;
        let out = attn.matmul(&v)?;
        let out = out
            .transpose(0, 1)?
            .contiguous()?
            .reshape((seq_len, self.num_heads * self.head_dim))?;
        let out = out.to_dtype(x.dtype())?;
        self.proj.forward(&out).map_err(Error::from)
    }
}

struct VisionMlp {
    up: Linear,
    down: Linear,
}

impl VisionMlp {
    fn new(up: Linear, down: Linear) -> Self {
        Self { up, down }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = self.up.forward(x)?;
        let hidden = hidden.gelu()?;
        self.down.forward(&hidden).map_err(Error::from)
    }
}

struct VisionBlock {
    ln1: LayerNorm,
    ln2: LayerNorm,
    attn: VisionAttention,
    mlp: VisionMlp,
}

impl VisionBlock {
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let attn_in = self.ln1.forward(x)?;
        let attn = self.attn.forward(&attn_in, cos, sin)?;
        let x = x.broadcast_add(&attn)?;
        let mlp_in = self.ln2.forward(&x)?;
        let mlp = self.mlp.forward(&mlp_in)?;
        x.broadcast_add(&mlp).map_err(Error::from)
    }
}

pub(crate) struct Qwen35VisionRuntime {
    config: VisionConfig,
    device: DeviceProfile,
    patch_weight_0: Tensor,
    patch_weight_1: Tensor,
    patch_bias: Tensor,
    position_embeddings: Tensor, // [seq, hidden], reordered to merge-friendly token order.
    rotary_cos: Tensor,          // [seq, head_dim]
    rotary_sin: Tensor,          // [seq, head_dim]
    blocks: Vec<VisionBlock>,
    post_ln: LayerNorm,
    projector_fc1: Linear,
    projector_fc2: Linear,
    source_path: PathBuf,
}

impl Qwen35VisionRuntime {
    pub(crate) fn load(
        mmproj_path: &Path,
        device: &DeviceProfile,
        dtype: DType,
        expected_projection_dim: usize,
    ) -> Result<Self> {
        let mut reader = open_gguf_reader(mmproj_path, BackendKind::from(device.kind))?;
        let content = gguf_file::Content::read(&mut reader).map_err(|e| {
            Error::ModelLoadError(format!(
                "Failed to parse Qwen3.5 mmproj GGUF {}: {e}",
                mmproj_path.display()
            ))
        })?;

        let arch = gguf_md_string(&content, "general.architecture")?;
        if arch != "clip" {
            return Err(Error::ModelLoadError(format!(
                "Unsupported mmproj architecture `{arch}` in {} (expected `clip`)",
                mmproj_path.display()
            )));
        }

        let hidden_size = gguf_md_u32(&content, "clip.vision.embedding_length")?;
        let intermediate_size = gguf_md_u32(&content, "clip.vision.feed_forward_length")?;
        let num_heads = gguf_md_u32(&content, "clip.vision.attention.head_count")?;
        let num_layers = gguf_md_u32(&content, "clip.vision.block_count")?;
        let patch_size = gguf_md_u32(&content, "clip.vision.patch_size")?;
        let image_size = gguf_md_u32(&content, "clip.vision.image_size")?;
        let spatial_merge_size = gguf_md_u32(&content, "clip.vision.spatial_merge_size")?.max(1);
        let projection_dim = gguf_md_u32(&content, "clip.vision.projection_dim")?;
        let layer_norm_eps =
            gguf_md_f32_opt(&content, "clip.vision.attention.layer_norm_rms_epsilon")
                .unwrap_or(1e-6) as f64;

        if projection_dim != expected_projection_dim {
            return Err(Error::ModelLoadError(format!(
                "mmproj projection dim mismatch: expected {expected_projection_dim}, got {projection_dim}"
            )));
        }

        let base_grid = position_grid_side(&content)?;
        let target_grid = resolve_target_grid(base_grid, spatial_merge_size)?;
        let mean = gguf_md_rgb_triplet(&content, "clip.vision.image_mean").unwrap_or([0.5; 3]);
        let std = gguf_md_rgb_triplet(&content, "clip.vision.image_std").unwrap_or([0.5; 3]);

        let config = VisionConfig {
            hidden_size,
            intermediate_size,
            num_heads,
            num_layers,
            patch_size,
            image_size,
            spatial_merge_size,
            projection_dim,
            layer_norm_eps,
            mean,
            std,
            target_grid,
        };

        let patch_w0 =
            load_gguf_tensor(&content, &mut reader, device, dtype, "v.patch_embd.weight")?;
        let patch_w1 = if content.tensor_infos.contains_key("v.patch_embd.weight.1") {
            load_gguf_tensor(
                &content,
                &mut reader,
                device,
                dtype,
                "v.patch_embd.weight.1",
            )?
        } else {
            patch_w0.clone()
        };
        let patch_bias =
            load_gguf_tensor(&content, &mut reader, device, dtype, "v.patch_embd.bias")?;

        let pos = load_gguf_tensor(
            &content,
            &mut reader,
            device,
            DType::F32,
            "v.position_embd.weight",
        )?;
        let position_embeddings = interpolate_and_reorder_position_embeddings(
            &pos,
            base_grid,
            config.target_grid,
            config.spatial_merge_size,
            &device.device,
            dtype,
        )?;

        let (rotary_cos, rotary_sin) = build_rotary_cache(
            config.target_grid,
            config.spatial_merge_size,
            config.head_dim(),
            &device.device,
            dtype,
        )?;

        let mut blocks = Vec::with_capacity(config.num_layers);
        for idx in 0..config.num_layers {
            let prefix = format!("v.blk.{idx}");
            let ln1 = LayerNorm::new(
                load_gguf_tensor(
                    &content,
                    &mut reader,
                    device,
                    DType::F32,
                    &format!("{prefix}.ln1.weight"),
                )?,
                load_gguf_tensor(
                    &content,
                    &mut reader,
                    device,
                    DType::F32,
                    &format!("{prefix}.ln1.bias"),
                )?,
                config.layer_norm_eps,
            );
            let ln2 = LayerNorm::new(
                load_gguf_tensor(
                    &content,
                    &mut reader,
                    device,
                    DType::F32,
                    &format!("{prefix}.ln2.weight"),
                )?,
                load_gguf_tensor(
                    &content,
                    &mut reader,
                    device,
                    DType::F32,
                    &format!("{prefix}.ln2.bias"),
                )?,
                config.layer_norm_eps,
            );
            let attn = VisionAttention::new(
                load_linear(
                    &content,
                    &mut reader,
                    device,
                    dtype,
                    &format!("{prefix}.attn_qkv.weight"),
                    &format!("{prefix}.attn_qkv.bias"),
                )?,
                load_linear(
                    &content,
                    &mut reader,
                    device,
                    dtype,
                    &format!("{prefix}.attn_out.weight"),
                    &format!("{prefix}.attn_out.bias"),
                )?,
                config.num_heads,
                config.head_dim(),
            );
            let mlp = VisionMlp::new(
                load_linear(
                    &content,
                    &mut reader,
                    device,
                    dtype,
                    &format!("{prefix}.ffn_up.weight"),
                    &format!("{prefix}.ffn_up.bias"),
                )?,
                load_linear(
                    &content,
                    &mut reader,
                    device,
                    dtype,
                    &format!("{prefix}.ffn_down.weight"),
                    &format!("{prefix}.ffn_down.bias"),
                )?,
            );
            blocks.push(VisionBlock {
                ln1,
                ln2,
                attn,
                mlp,
            });
        }

        let post_ln = LayerNorm::new(
            load_gguf_tensor(
                &content,
                &mut reader,
                device,
                DType::F32,
                "v.post_ln.weight",
            )?,
            load_gguf_tensor(&content, &mut reader, device, DType::F32, "v.post_ln.bias")?,
            config.layer_norm_eps,
        );

        let projector_fc1 = load_linear(
            &content,
            &mut reader,
            device,
            dtype,
            "mm.0.weight",
            "mm.0.bias",
        )?;
        let projector_fc2 = load_linear(
            &content,
            &mut reader,
            device,
            dtype,
            "mm.2.weight",
            "mm.2.bias",
        )?;

        Ok(Self {
            config,
            device: device.clone(),
            patch_weight_0: patch_w0.reshape((hidden_size, 3 * patch_size * patch_size))?,
            patch_weight_1: patch_w1.reshape((hidden_size, 3 * patch_size * patch_size))?,
            patch_bias,
            position_embeddings,
            rotary_cos,
            rotary_sin,
            blocks,
            post_ln,
            projector_fc1,
            projector_fc2,
            source_path: mmproj_path.to_path_buf(),
        })
    }

    pub(crate) fn source_path(&self) -> &Path {
        &self.source_path
    }

    pub(crate) fn llm_grid_hw(&self) -> (usize, usize) {
        (
            self.config.target_grid / self.config.spatial_merge_size,
            self.config.target_grid / self.config.spatial_merge_size,
        )
    }

    pub(crate) fn encode_media(&self, media: &Qwen35MultimodalInput) -> Result<(Tensor, usize)> {
        match media.kind {
            Qwen35MultimodalKind::Image => {
                let bytes = load_media_source_bytes(&media.source)?;
                let image = decode_image(&bytes, &media.source)?;
                let resized = resize_image(&image, self.config.target_image_size() as u32);
                Ok((self.encode_frame_pair(&resized, &resized)?, 1))
            }
            Qwen35MultimodalKind::Video => {
                let bytes = load_media_source_bytes(&media.source)?;
                let frames = decode_video_frames(&bytes, &media.source)?;
                let sampled = sample_video_frames(&frames, qwen35_video_max_frames());
                let temporal_steps = sampled_video_temporal_steps(sampled.len());
                let mut embeddings = Vec::with_capacity(temporal_steps);

                for pair in sampled.chunks(2) {
                    let frame0 = resize_image(&pair[0], self.config.target_image_size() as u32);
                    let frame1 = resize_image(
                        pair.get(1).unwrap_or(&pair[0]),
                        self.config.target_image_size() as u32,
                    );
                    embeddings.push(self.encode_frame_pair(&frame0, &frame1)?);
                }

                let embeddings = if embeddings.len() == 1 {
                    embeddings.pop().expect("single embedding present")
                } else {
                    let refs: Vec<&Tensor> = embeddings.iter().collect();
                    Tensor::cat(&refs, 0)?
                };
                Ok((embeddings, temporal_steps))
            }
        }
    }

    fn encode_frame_pair(&self, frame0: &RgbImage, frame1: &RgbImage) -> Result<Tensor> {
        let patches = self.patchify_pair(&frame0, &frame1)?;
        let hidden = self.forward_vision(&patches)?;
        let merged = hidden.reshape((
            self.config.merged_seq_len(),
            self.config.hidden_size * self.config.spatial_merge_size.pow(2),
        ))?;
        let proj = self.projector_fc1.forward(&merged)?;
        let proj = proj.gelu()?;
        self.projector_fc2.forward(&proj).map_err(Error::from)
    }

    fn forward_vision(&self, patches: &Tensor) -> Result<Tensor> {
        let patch_dtype = self.patch_weight_0.dtype();
        let aligned_patches = if patches.dtype() == patch_dtype {
            None
        } else {
            Some(patches.to_dtype(patch_dtype)?)
        };
        let patches = aligned_patches.as_ref().unwrap_or(patches);

        let in_per_frame = 3 * self.config.patch_size * self.config.patch_size;
        // `narrow` on the last dim of `[seq, in*2]` yields strided views.
        // Metal matmul requires contiguous inputs in this path.
        let p0 = patches.narrow(1, 0, in_per_frame)?.contiguous()?;
        let p1 = patches
            .narrow(1, in_per_frame, in_per_frame)?
            .contiguous()?;
        let w0_t = self.patch_weight_0.transpose(0, 1)?.contiguous()?;
        let w1_t = self.patch_weight_1.transpose(0, 1)?.contiguous()?;

        let mut x = p0.matmul(&w0_t)?;
        x = x.broadcast_add(&p1.matmul(&w1_t)?)?;
        x = x.broadcast_add(&self.patch_bias.unsqueeze(0)?)?;
        x = x.broadcast_add(&self.position_embeddings)?;

        for block in &self.blocks {
            x = block.forward(&x, &self.rotary_cos, &self.rotary_sin)?;
        }
        self.post_ln.forward(&x)
    }

    fn patchify_pair(&self, frame0: &RgbImage, frame1: &RgbImage) -> Result<Tensor> {
        let grid = self.config.target_grid;
        let patch = self.config.patch_size;
        let merge = self.config.spatial_merge_size;
        let seq_len = self.config.seq_len();
        let in_per_frame = 3 * patch * patch;
        let mut data = Vec::with_capacity(seq_len * in_per_frame * 2);

        for br in 0..(grid / merge) {
            for bc in 0..(grid / merge) {
                for ir in 0..merge {
                    for ic in 0..merge {
                        let pr = br * merge + ir;
                        let pc = bc * merge + ic;
                        let y0 = pr * patch;
                        let x0 = pc * patch;

                        write_patch_values(
                            &mut data,
                            frame0,
                            x0,
                            y0,
                            patch,
                            &self.config.mean,
                            &self.config.std,
                        );
                        write_patch_values(
                            &mut data,
                            frame1,
                            x0,
                            y0,
                            patch,
                            &self.config.mean,
                            &self.config.std,
                        );
                    }
                }
            }
        }

        Tensor::from_vec(data, (seq_len, in_per_frame * 2), &self.device.device)
            .map_err(Error::from)
    }
}

fn write_patch_values(
    out: &mut Vec<f32>,
    frame: &RgbImage,
    x0: usize,
    y0: usize,
    patch_size: usize,
    mean: &[f32; 3],
    std: &[f32; 3],
) {
    for c in 0..3 {
        for dy in 0..patch_size {
            for dx in 0..patch_size {
                let px = frame.get_pixel((x0 + dx) as u32, (y0 + dy) as u32).0[c];
                let value = (px as f32 / 255.0 - mean[c]) / std[c].max(1e-6);
                out.push(value);
            }
        }
    }
}

fn decode_image(bytes: &[u8], source: &str) -> Result<RgbImage> {
    let image = image::load_from_memory(bytes).map_err(|e| {
        Error::InferenceError(format!("Failed to decode image from `{source}`: {e}"))
    })?;
    Ok(image.to_rgb8())
}

fn decode_video_frames(bytes: &[u8], source: &str) -> Result<Vec<RgbImage>> {
    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        let decoder = GifDecoder::new(Cursor::new(bytes)).map_err(|e| {
            Error::InferenceError(format!("Failed to decode GIF video `{source}`: {e}"))
        })?;
        let frames = decoder
            .into_frames()
            .collect_frames()
            .map_err(|e| Error::InferenceError(format!("Failed to read GIF frames: {e}")))?;
        let out = frames
            .into_iter()
            .map(|f| DynamicImage::ImageRgba8(f.into_buffer()).to_rgb8())
            .collect::<Vec<_>>();
        if out.is_empty() {
            return Err(Error::InferenceError(
                "GIF video input contains no frames".to_string(),
            ));
        }
        return Ok(out);
    }

    extract_video_frames_with_ffmpeg(bytes, source, qwen35_video_max_frames())
}

fn extract_video_frames_with_ffmpeg(
    bytes: &[u8],
    source: &str,
    max_frames: usize,
) -> Result<Vec<RgbImage>> {
    let temp_root = std::env::temp_dir().join(format!("izwi-qwen35-mm-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&temp_root)?;

    let extension = source_extension(source).unwrap_or("mp4");
    let input_path = temp_root.join(format!("input.{extension}"));
    fs::write(&input_path, bytes)?;
    let pattern = temp_root.join("frame_%02d.png");
    let ffmpeg = resolve_ffmpeg_binary().ok_or_else(|| {
        Error::InferenceError(
            "Video decoding requires `ffmpeg`, but it was not found in PATH. Install ffmpeg or set `IZWI_QWEN35_FFMPEG_PATH` (or `IZWI_FFMPEG_PATH`) to the ffmpeg binary."
                .to_string(),
        )
    })?;
    let sample_fps = resolve_ffprobe_binary()
        .as_deref()
        .and_then(|ffprobe| probe_video_duration(ffprobe, &input_path))
        .and_then(|duration_secs| sample_video_fps(duration_secs, max_frames));

    let mut command = Command::new(&ffmpeg);
    command.arg("-hide_banner").arg("-loglevel").arg("error");
    command.arg("-i").arg(&input_path);
    if let Some(sample_fps) = sample_fps {
        command.arg("-vf").arg(format!("fps={sample_fps:.6}"));
    }
    let output = command
        .arg("-frames:v")
        .arg(max_frames.max(1).to_string())
        .arg(&pattern)
        .output()
        .map_err(|e| {
            Error::InferenceError(format!(
                "Failed to launch ffmpeg (`{}`) for `{source}`: {e}",
                ffmpeg.display()
            ))
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let _ = fs::remove_dir_all(&temp_root);
        return Err(Error::InferenceError(format!(
            "ffmpeg failed to decode video `{source}`: {stderr}"
        )));
    }

    let mut frame_paths = fs::read_dir(&temp_root)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| {
            path.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|name| name.starts_with("frame_") && name.ends_with(".png"))
        })
        .collect::<Vec<_>>();
    frame_paths.sort();

    let mut frames = Vec::new();
    for path in frame_paths {
        if let Ok(bytes) = fs::read(&path) {
            if let Ok(image) = image::load_from_memory(&bytes) {
                frames.push(image.to_rgb8());
            }
        }
    }
    let _ = fs::remove_dir_all(&temp_root);

    if frames.is_empty() {
        return Err(Error::InferenceError(format!(
            "ffmpeg produced no decodable frames for video `{source}`"
        )));
    }
    Ok(frames)
}

fn qwen35_video_max_frames() -> usize {
    std::env::var("IZWI_QWEN35_VIDEO_MAX_FRAMES")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_VIDEO_MAX_FRAMES)
}

fn sampled_video_temporal_steps(frame_count: usize) -> usize {
    frame_count.div_ceil(2)
}

fn sample_video_frames(frames: &[RgbImage], max_frames: usize) -> Vec<RgbImage> {
    if frames.len() <= max_frames {
        return frames.to_vec();
    }
    sample_video_frame_indices(frames.len(), max_frames)
        .into_iter()
        .map(|idx| frames[idx].clone())
        .collect()
}

fn sample_video_frame_indices(total_frames: usize, max_frames: usize) -> Vec<usize> {
    if total_frames == 0 || max_frames == 0 {
        return Vec::new();
    }
    if max_frames == 1 {
        return vec![0];
    }
    if total_frames <= max_frames {
        return (0..total_frames).collect();
    }

    let last_index = total_frames - 1;
    let mut indices = Vec::with_capacity(max_frames);
    for sample_idx in 0..max_frames {
        let idx = ((sample_idx * last_index) + (max_frames - 1) / 2) / (max_frames - 1);
        if indices.last().copied() != Some(idx) {
            indices.push(idx);
        }
    }
    if indices.last().copied() != Some(last_index) {
        if indices.len() == max_frames {
            indices.pop();
        }
        indices.push(last_index);
    }
    indices
}

fn sample_video_fps(duration_secs: f64, max_frames: usize) -> Option<f64> {
    if !duration_secs.is_finite() || duration_secs <= 0.0 || max_frames == 0 {
        return None;
    }
    Some((max_frames as f64 / duration_secs).max(0.01))
}

fn probe_video_duration(ffprobe: &Path, input_path: &Path) -> Option<f64> {
    let output = Command::new(ffprobe)
        .arg("-v")
        .arg("error")
        .arg("-show_entries")
        .arg("format=duration")
        .arg("-of")
        .arg("default=noprint_wrappers=1:nokey=1")
        .arg(input_path)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    output
        .stdout
        .split(|byte| *byte == b'\n' || *byte == b'\r')
        .find_map(|line| std::str::from_utf8(line).ok())
        .and_then(|line| line.trim().parse::<f64>().ok())
        .filter(|duration| *duration > 0.0)
}

fn resolve_ffmpeg_binary() -> Option<PathBuf> {
    resolve_media_binary(["IZWI_QWEN35_FFMPEG_PATH", "IZWI_FFMPEG_PATH"], "ffmpeg")
}

fn resolve_ffprobe_binary() -> Option<PathBuf> {
    resolve_media_binary(["IZWI_QWEN35_FFPROBE_PATH", "IZWI_FFPROBE_PATH"], "ffprobe")
}

fn resolve_media_binary<const N: usize>(env_vars: [&str; N], binary_name: &str) -> Option<PathBuf> {
    for var in env_vars {
        if let Ok(raw) = std::env::var(var) {
            let trimmed = raw.trim();
            if !trimmed.is_empty() {
                return Some(PathBuf::from(trimmed));
            }
        }
    }

    if let Some(path_env) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path_env) {
            let candidate = dir.join(binary_name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }

    for candidate in [
        format!("/opt/homebrew/bin/{binary_name}"),
        format!("/usr/local/bin/{binary_name}"),
        format!("/usr/bin/{binary_name}"),
        format!("/bin/{binary_name}"),
    ] {
        let path = PathBuf::from(candidate);
        if path.is_file() {
            return Some(path);
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            for candidate in [
                parent.join(binary_name),
                parent.join(format!("../Resources/{binary_name}")),
                parent.join(format!("../Resources/bin/{binary_name}")),
            ] {
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
    }

    None
}

fn source_extension(source: &str) -> Option<&str> {
    let mut raw = source;
    if let Some(stripped) = raw.strip_prefix("file://") {
        raw = stripped;
    }
    let raw = raw.split('?').next().unwrap_or(raw);
    let raw = raw.split('#').next().unwrap_or(raw);
    raw.rsplit('.').next().filter(|ext| !ext.is_empty())
}

fn resize_image(image: &RgbImage, size: u32) -> RgbImage {
    imageops::resize(image, size, size, imageops::FilterType::CatmullRom)
}

fn load_media_source_bytes(source: &str) -> Result<Vec<u8>> {
    let source = source.trim();
    if source.is_empty() {
        return Err(Error::InferenceError(
            "Multimodal source cannot be empty".to_string(),
        ));
    }

    if source.starts_with("data:") {
        return decode_data_url(source);
    }
    if source.starts_with("http://") || source.starts_with("https://") {
        return fetch_remote_media(source);
    }
    if let Some(path) = source.strip_prefix("file://") {
        let path = percent_decode_path(path)?;
        return fs::read(&path).map_err(|e| {
            Error::InferenceError(format!("Failed reading media file {}: {e}", path.display()))
        });
    }

    let path = PathBuf::from(source);
    fs::read(&path).map_err(|e| {
        Error::InferenceError(format!("Failed reading media file {}: {e}", path.display()))
    })
}

fn decode_data_url(url: &str) -> Result<Vec<u8>> {
    let Some((header, payload)) = url.split_once(',') else {
        return Err(Error::InferenceError(
            "Invalid data URL for multimodal source".to_string(),
        ));
    };
    if header.ends_with(";base64") {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(payload.trim())
            .map_err(|e| Error::InferenceError(format!("Invalid data URL base64 payload: {e}")))?;
        if bytes.len() > MAX_MEDIA_BYTES {
            return Err(Error::InferenceError(format!(
                "Multimodal payload exceeds {} MiB limit",
                MAX_MEDIA_BYTES / (1024 * 1024)
            )));
        }
        return Ok(bytes);
    }

    let bytes = percent_decode_bytes(payload)?;
    if bytes.len() > MAX_MEDIA_BYTES {
        return Err(Error::InferenceError(format!(
            "Multimodal payload exceeds {} MiB limit",
            MAX_MEDIA_BYTES / (1024 * 1024)
        )));
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use image::Rgb;

    use super::{
        sample_video_fps, sample_video_frame_indices, sample_video_frames,
        sampled_video_temporal_steps,
    };

    #[test]
    fn sample_video_frame_indices_cover_start_and_end_when_bounded() {
        let indices = sample_video_frame_indices(12, 5);
        assert_eq!(indices.len(), 5);
        assert_eq!(indices[0], 0);
        assert_eq!(indices.last().copied(), Some(11));
        assert!(indices.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn sample_video_frame_indices_support_single_frame_budget_without_panicking() {
        let indices = sample_video_frame_indices(12, 1);
        assert_eq!(indices, vec![0]);
    }

    #[test]
    fn sample_video_frames_preserve_budget_and_clip_coverage() {
        let frames = (0..10)
            .map(|idx| image::RgbImage::from_pixel(1, 1, Rgb([idx as u8, 0, 0])))
            .collect::<Vec<_>>();
        let sampled = sample_video_frames(&frames, 4);
        let sampled_values = sampled
            .iter()
            .map(|frame| frame.get_pixel(0, 0).0[0])
            .collect::<Vec<_>>();
        assert_eq!(sampled.len(), 4);
        assert_eq!(sampled_values.first().copied(), Some(0));
        assert_eq!(sampled_values.last().copied(), Some(9));
    }

    #[test]
    fn sampled_video_temporal_steps_round_up_odd_frame_counts() {
        assert_eq!(sampled_video_temporal_steps(1), 1);
        assert_eq!(sampled_video_temporal_steps(2), 1);
        assert_eq!(sampled_video_temporal_steps(3), 2);
        assert_eq!(sampled_video_temporal_steps(8), 4);
    }

    #[test]
    fn sample_video_fps_scales_requested_budget_across_duration() {
        let fps = sample_video_fps(4.0, 8).expect("sample fps");
        assert!((fps - 2.0).abs() < f64::EPSILON);
        assert!(sample_video_fps(0.0, 8).is_none());
    }
}

fn fetch_remote_media(url: &str) -> Result<Vec<u8>> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(20))
        .build()
        .map_err(|e| Error::InferenceError(format!("Failed creating HTTP client: {e}")))?;

    let response = client
        .get(url)
        .send()
        .map_err(|e| Error::InferenceError(format!("Failed downloading media URL `{url}`: {e}")))?;
    let status = response.status();
    if !status.is_success() {
        return Err(Error::InferenceError(format!(
            "Media URL `{url}` returned HTTP {status}"
        )));
    }
    let bytes = response.bytes().map_err(|e| {
        Error::InferenceError(format!("Failed reading media body from `{url}`: {e}"))
    })?;
    if bytes.len() > MAX_MEDIA_BYTES {
        return Err(Error::InferenceError(format!(
            "Downloaded media from `{url}` exceeds {} MiB limit",
            MAX_MEDIA_BYTES / (1024 * 1024)
        )));
    }
    Ok(bytes.to_vec())
}

fn percent_decode_path(path: &str) -> Result<PathBuf> {
    let bytes = percent_decode_bytes(path)?;
    let decoded = String::from_utf8(bytes)
        .map_err(|e| Error::InferenceError(format!("Invalid UTF-8 in file URL path: {e}")))?;
    Ok(PathBuf::from(decoded))
}

fn percent_decode_bytes(input: &str) -> Result<Vec<u8>> {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut idx = 0usize;
    while idx < bytes.len() {
        if bytes[idx] == b'%' {
            if idx + 2 >= bytes.len() {
                return Err(Error::InferenceError(
                    "Invalid percent-encoding in URL payload".to_string(),
                ));
            }
            let hi = from_hex(bytes[idx + 1]).ok_or_else(|| {
                Error::InferenceError("Invalid percent-encoding in URL payload".to_string())
            })?;
            let lo = from_hex(bytes[idx + 2]).ok_or_else(|| {
                Error::InferenceError("Invalid percent-encoding in URL payload".to_string())
            })?;
            out.push((hi << 4) | lo);
            idx += 3;
            continue;
        }
        out.push(bytes[idx]);
        idx += 1;
    }
    Ok(out)
}

fn from_hex(ch: u8) -> Option<u8> {
    match ch {
        b'0'..=b'9' => Some(ch - b'0'),
        b'a'..=b'f' => Some(ch - b'a' + 10),
        b'A'..=b'F' => Some(ch - b'A' + 10),
        _ => None,
    }
}

fn load_linear(
    content: &gguf_file::Content,
    reader: &mut (impl Read + Seek),
    device: &DeviceProfile,
    dtype: DType,
    weight_name: &str,
    bias_name: &str,
) -> Result<Linear> {
    let weight = load_gguf_tensor(content, reader, device, dtype, weight_name)?;
    let bias = load_gguf_tensor(content, reader, device, dtype, bias_name)?;
    Ok(Linear::new(weight, Some(bias)))
}

fn load_gguf_tensor(
    content: &gguf_file::Content,
    reader: &mut (impl Read + Seek),
    device: &DeviceProfile,
    dtype: DType,
    name: &str,
) -> Result<Tensor> {
    let qtensor = content.tensor(reader, name, &device.device).map_err(|e| {
        Error::ModelLoadError(format!("Missing or invalid mmproj tensor `{name}`: {e}"))
    })?;
    let mut tensor = qtensor.dequantize(&device.device).map_err(|e| {
        Error::ModelLoadError(format!("Failed to dequantize mmproj tensor `{name}`: {e}"))
    })?;
    if tensor.dtype() != dtype {
        tensor = tensor.to_dtype(dtype)?;
    }
    Ok(tensor)
}

fn resolve_target_grid(base_grid: usize, spatial_merge: usize) -> Result<usize> {
    let requested = std::env::var("IZWI_QWEN35_MM_TARGET_GRID")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_TARGET_GRID);
    let mut grid = requested.clamp(spatial_merge, base_grid.max(spatial_merge));
    grid -= grid % spatial_merge;
    if grid == 0 {
        grid = spatial_merge;
    }
    Ok(grid)
}

fn position_grid_side(content: &gguf_file::Content) -> Result<usize> {
    let info = content
        .tensor_infos
        .get("v.position_embd.weight")
        .ok_or_else(|| {
            Error::ModelLoadError("Missing v.position_embd.weight tensor in mmproj".to_string())
        })?;
    let dims = info.shape.dims();
    let positions = match dims {
        [first, _hidden] => *first,
        other => {
            return Err(Error::ModelLoadError(format!(
                "Unexpected v.position_embd.weight shape: {other:?}"
            )))
        }
    };
    let side = (positions as f64).sqrt().round() as usize;
    if side * side != positions {
        return Err(Error::ModelLoadError(format!(
            "v.position_embd.weight positions {positions} are not a perfect square"
        )));
    }
    Ok(side)
}

fn interpolate_and_reorder_position_embeddings(
    pos: &Tensor,
    base_grid: usize,
    target_grid: usize,
    spatial_merge: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let (_, hidden_size) = pos.dims2()?;
    let pos = pos.to_dtype(DType::F32)?.to_vec2::<f32>()?;

    let mut raster = vec![0f32; target_grid * target_grid * hidden_size];
    for y in 0..target_grid {
        let fy = if target_grid == 1 {
            0.0
        } else {
            y as f32 * (base_grid - 1) as f32 / (target_grid - 1) as f32
        };
        let y0 = fy.floor() as usize;
        let y1 = fy.ceil().min((base_grid - 1) as f32) as usize;
        let wy = fy - y0 as f32;
        for x in 0..target_grid {
            let fx = if target_grid == 1 {
                0.0
            } else {
                x as f32 * (base_grid - 1) as f32 / (target_grid - 1) as f32
            };
            let x0 = fx.floor() as usize;
            let x1 = fx.ceil().min((base_grid - 1) as f32) as usize;
            let wx = fx - x0 as f32;

            let idx00 = y0 * base_grid + x0;
            let idx01 = y0 * base_grid + x1;
            let idx10 = y1 * base_grid + x0;
            let idx11 = y1 * base_grid + x1;

            let out_offset = (y * target_grid + x) * hidden_size;
            for h in 0..hidden_size {
                let v00 = pos[idx00][h];
                let v01 = pos[idx01][h];
                let v10 = pos[idx10][h];
                let v11 = pos[idx11][h];
                let top = v00 * (1.0 - wx) + v01 * wx;
                let bottom = v10 * (1.0 - wx) + v11 * wx;
                raster[out_offset + h] = top * (1.0 - wy) + bottom * wy;
            }
        }
    }

    let mut reordered = Vec::with_capacity(raster.len());
    for br in 0..(target_grid / spatial_merge) {
        for bc in 0..(target_grid / spatial_merge) {
            for ir in 0..spatial_merge {
                for ic in 0..spatial_merge {
                    let r = br * spatial_merge + ir;
                    let c = bc * spatial_merge + ic;
                    let src = (r * target_grid + c) * hidden_size;
                    reordered.extend_from_slice(&raster[src..src + hidden_size]);
                }
            }
        }
    }

    let seq_len = target_grid * target_grid;
    Tensor::from_vec(reordered, (seq_len, hidden_size), device)?
        .to_dtype(dtype)
        .map_err(Error::from)
}

fn build_rotary_cache(
    target_grid: usize,
    spatial_merge: usize,
    head_dim: usize,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    if head_dim % 2 != 0 {
        return Err(Error::ModelLoadError(format!(
            "Vision head_dim must be even, got {head_dim}"
        )));
    }
    let rotary_half = head_dim / 2;
    let inv_freq = (0..rotary_half)
        .step_by(2)
        .map(|i| 1f32 / ROTARY_THETA.powf(i as f32 / rotary_half as f32))
        .collect::<Vec<_>>();

    let seq_len = target_grid * target_grid;
    let mut angles = Vec::with_capacity(seq_len * rotary_half);
    for br in 0..(target_grid / spatial_merge) {
        for bc in 0..(target_grid / spatial_merge) {
            for ir in 0..spatial_merge {
                for ic in 0..spatial_merge {
                    let row = (br * spatial_merge + ir) as f32;
                    let col = (bc * spatial_merge + ic) as f32;
                    for &inv in &inv_freq {
                        angles.push(row * inv);
                    }
                    for &inv in &inv_freq {
                        angles.push(col * inv);
                    }
                }
            }
        }
    }

    let angles = Tensor::from_vec(angles, (seq_len, rotary_half), device)?;
    let emb = Tensor::cat(&[angles.clone(), angles], 1)?;
    let cos = emb.cos()?.to_dtype(dtype)?;
    let sin = emb.sin()?.to_dtype(dtype)?;
    Ok((cos, sin))
}

fn apply_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let rotated = rotate_half(x)?;
    let out = x.broadcast_mul(cos)?;
    out.broadcast_add(&rotated.broadcast_mul(sin)?)
        .map_err(Error::from)
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let dim = x.dim(D::Minus1)?;
    let half = dim / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, dim - half)?;
    Tensor::cat(&[x2.neg()?, x1], D::Minus1).map_err(Error::from)
}

fn gguf_md_u32(content: &gguf_file::Content, key: &str) -> Result<usize> {
    content
        .metadata
        .get(key)
        .ok_or_else(|| Error::ModelLoadError(format!("Missing mmproj metadata key `{key}`")))?
        .to_u32()
        .map(|v| v as usize)
        .map_err(|e| Error::ModelLoadError(format!("Invalid mmproj metadata `{key}`: {e}")))
}

fn gguf_md_f32_opt(content: &gguf_file::Content, key: &str) -> Option<f32> {
    content.metadata.get(key).and_then(|v| v.to_f32().ok())
}

fn gguf_md_string(content: &gguf_file::Content, key: &str) -> Result<String> {
    content
        .metadata
        .get(key)
        .ok_or_else(|| Error::ModelLoadError(format!("Missing mmproj metadata key `{key}`")))?
        .to_string()
        .cloned()
        .map_err(|e| Error::ModelLoadError(format!("Invalid mmproj metadata `{key}`: {e}")))
}

fn gguf_md_rgb_triplet(content: &gguf_file::Content, key: &str) -> Option<[f32; 3]> {
    let values = content.metadata.get(key)?.to_vec().ok()?;
    if values.len() < 3 {
        return None;
    }
    let mut out = [0.5f32; 3];
    for (idx, slot) in out.iter_mut().enumerate() {
        *slot = values[idx].to_f32().ok()?;
    }
    Some(out)
}
