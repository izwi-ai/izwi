use std::f32::consts::PI;
use std::fmt;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use candle_core::{DType, Tensor};
use candle_nn::{
    ops, Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Module, VarBuilder,
};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::Distribution;
use rand_distr::StandardNormal;
use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

use crate::error::{Error, Result};

use super::config::{KokoroConfig, KokoroIstftNetConfig};
use super::prosody::{
    load_plain_conv1d, load_weight_norm_conv1d, load_weight_norm_conv_transpose1d, AdaIN1d,
    AdainResBlk1d,
};

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

fn kokoro_cpu_resblocks_parallel_enabled() -> bool {
    match std::env::var("IZWI_KOKORO_CPU_RESBLOCKS") {
        Ok(value) => !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "serial" | "sequential"
        ),
        Err(_) => true,
    }
}

fn kokoro_cpu_stage_branches_parallel_enabled() -> bool {
    match std::env::var("IZWI_KOKORO_CPU_STAGE_BRANCHES") {
        Ok(value) => !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "serial" | "sequential"
        ),
        Err(_) => true,
    }
}

#[derive(Debug)]
pub struct KokoroDecoder {
    f0_conv: Conv1d,
    n_conv: Conv1d,
    asr_res: Conv1d,
    encode: AdainResBlk1d,
    decode: Vec<AdainResBlk1d>,
    generator: KokoroIstftGenerator,
}

impl KokoroDecoder {
    pub fn load(cfg: &KokoroConfig, vb: VarBuilder) -> Result<Self> {
        let root = vb.pp("module");
        let style_dim = cfg.style_dim;
        let encode = AdainResBlk1d::load(
            cfg.hidden_dim + 2,
            1024,
            style_dim,
            false,
            root.pp("encode"),
        )?;

        let mut decode = Vec::with_capacity(4);
        decode.push(AdainResBlk1d::load(
            1024 + 2 + 64,
            1024,
            style_dim,
            false,
            root.pp("decode.0"),
        )?);
        decode.push(AdainResBlk1d::load(
            1024 + 2 + 64,
            1024,
            style_dim,
            false,
            root.pp("decode.1"),
        )?);
        decode.push(AdainResBlk1d::load(
            1024 + 2 + 64,
            1024,
            style_dim,
            false,
            root.pp("decode.2"),
        )?);
        decode.push(AdainResBlk1d::load(
            1024 + 2 + 64,
            512,
            style_dim,
            true,
            root.pp("decode.3"),
        )?);

        let conv_stride2_cfg = Conv1dConfig {
            padding: 1,
            stride: 2,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let f0_conv = load_weight_norm_conv1d(root.pp("F0_conv"), conv_stride2_cfg)?;
        let n_conv = load_weight_norm_conv1d(root.pp("N_conv"), conv_stride2_cfg)?;
        let asr_res = load_weight_norm_conv1d(root.pp("asr_res.0"), Conv1dConfig::default())?;
        let generator = KokoroIstftGenerator::load(&cfg.istftnet, style_dim, root.pp("generator"))?;

        Ok(Self {
            f0_conv,
            n_conv,
            asr_res,
            encode,
            decode,
            generator,
        })
    }

    pub fn forward(
        &self,
        asr: &Tensor,      // [B, 512, T]
        f0_curve: &Tensor, // [B, 2T]
        n_curve: &Tensor,  // [B, 2T]
        style: &Tensor,    // [B, 128]
    ) -> Result<Vec<f32>> {
        self.forward_with_seed(asr, f0_curve, n_curve, style, None)
    }

    pub(crate) fn forward_with_seed(
        &self,
        asr: &Tensor,      // [B, 512, T]
        f0_curve: &Tensor, // [B, 2T]
        n_curve: &Tensor,  // [B, 2T]
        style: &Tensor,    // [B, 128]
        rng_seed: Option<u64>,
    ) -> Result<Vec<f32>> {
        let profile = kokoro_profile_enabled();
        let t0 = Instant::now();
        let f0 = self
            .f0_conv
            .forward(&f0_curve.unsqueeze(1).map_err(Error::from)?)
            .map_err(Error::from)?;
        let n = self
            .n_conv
            .forward(&n_curve.unsqueeze(1).map_err(Error::from)?)
            .map_err(Error::from)?;
        if profile {
            log_kokoro_profile("decoder.f0n_conv", t0.elapsed());
        }

        let t1 = Instant::now();
        let x = Tensor::cat(&[asr.clone(), f0.clone(), n.clone()], 1).map_err(Error::from)?;
        let mut x = self.encode.forward(&x, style)?;
        let asr_res = self.asr_res.forward(asr).map_err(Error::from)?;

        let mut still_concat_res = true;
        for block in &self.decode {
            if still_concat_res {
                x = Tensor::cat(&[x, asr_res.clone(), f0.clone(), n.clone()], 1)
                    .map_err(Error::from)?;
            }
            x = block.forward(&x, style)?;
            let (_b, _c, t) = x.dims3().map_err(Error::from)?;
            let (_, _, asr_t) = asr_res.dims3().map_err(Error::from)?;
            if t > asr_t {
                still_concat_res = false;
            }
        }
        if profile {
            log_kokoro_profile("decoder.encode_decode_blocks", t1.elapsed());
        }

        let t2 = Instant::now();
        let out = self.generator.forward(&x, style, f0_curve, rng_seed)?;
        if profile {
            log_kokoro_profile("decoder.generator_total", t2.elapsed());
            log_kokoro_profile("decoder.total", t0.elapsed());
        }
        Ok(out)
    }
}

#[derive(Debug)]
struct KokoroIstftGenerator {
    cfg: KokoroIstftNetConfig,
    num_kernels: usize,
    num_upsamples: usize,
    total_scale: usize,
    harmonic_num: usize,
    sine_amp: f32,
    noise_std: f32,
    voiced_threshold: f32,
    ups: Vec<ConvTranspose1d>,
    resblocks: Vec<AdaInResBlock1>,
    noise_convs: Vec<Conv1d>,
    noise_res: Vec<AdaInResBlock1>,
    conv_post: Conv1d,
    source_linear_w: [f32; 9],
    source_linear_b: f32,
    stft: KokoroStft,
}

impl KokoroIstftGenerator {
    fn load(cfg: &KokoroIstftNetConfig, style_dim: usize, vb: VarBuilder) -> Result<Self> {
        if cfg.resblock_kernel_sizes.is_empty() || cfg.upsample_rates.is_empty() {
            return Err(Error::ModelLoadError(
                "Kokoro ISTFTNet config missing kernels/upsample rates".to_string(),
            ));
        }
        let num_kernels = cfg.resblock_kernel_sizes.len();
        let num_upsamples = cfg.upsample_rates.len();
        let total_scale = cfg
            .upsample_rates
            .iter()
            .copied()
            .product::<usize>()
            .saturating_mul(cfg.gen_istft_hop_size);

        let mut ups = Vec::with_capacity(num_upsamples);
        for (i, (&u, &k)) in cfg
            .upsample_rates
            .iter()
            .zip(cfg.upsample_kernel_sizes.iter())
            .enumerate()
        {
            let padding = (k.saturating_sub(u)) / 2;
            let ct = load_weight_norm_conv_transpose1d(
                vb.pp(format!("ups.{i}")),
                ConvTranspose1dConfig {
                    padding,
                    output_padding: 0,
                    stride: u,
                    dilation: 1,
                    groups: 1,
                },
            )?;
            ups.push(ct);
        }

        let mut resblocks = Vec::with_capacity(num_upsamples * num_kernels);
        for i in 0..num_upsamples {
            let ch = cfg.upsample_initial_channel / (1usize << (i + 1));
            for (j, (&kernel, dils)) in cfg
                .resblock_kernel_sizes
                .iter()
                .zip(cfg.resblock_dilation_sizes.iter())
                .enumerate()
            {
                resblocks.push(AdaInResBlock1::load(
                    ch,
                    kernel,
                    dils,
                    style_dim,
                    vb.pp(format!("resblocks.{}", i * num_kernels + j)),
                )?);
            }
        }

        let mut noise_convs = Vec::with_capacity(num_upsamples);
        let mut noise_res = Vec::with_capacity(num_upsamples);
        for i in 0..num_upsamples {
            let c_cur = cfg.upsample_initial_channel / (1usize << (i + 1));
            if i + 1 < num_upsamples {
                let stride_f0 = cfg.upsample_rates[i + 1..]
                    .iter()
                    .copied()
                    .product::<usize>();
                let padding = (stride_f0 + 1) / 2;
                noise_convs.push(load_plain_conv1d(
                    vb.pp(format!("noise_convs.{i}")),
                    Conv1dConfig {
                        padding,
                        stride: stride_f0,
                        dilation: 1,
                        groups: 1,
                        cudnn_fwd_algo: None,
                    },
                )?);
                noise_res.push(AdaInResBlock1::load(
                    c_cur,
                    7,
                    &[1, 3, 5],
                    style_dim,
                    vb.pp(format!("noise_res.{i}")),
                )?);
            } else {
                noise_convs.push(load_plain_conv1d(
                    vb.pp(format!("noise_convs.{i}")),
                    Conv1dConfig::default(),
                )?);
                noise_res.push(AdaInResBlock1::load(
                    c_cur,
                    11,
                    &[1, 3, 5],
                    style_dim,
                    vb.pp(format!("noise_res.{i}")),
                )?);
            }
        }

        let conv_post = load_weight_norm_conv1d(
            vb.pp("conv_post"),
            Conv1dConfig {
                padding: 3,
                stride: 1,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            },
        )?;

        let source_linear_w_t = vb
            .pp("m_source.l_linear")
            .get_unchecked_dtype("weight", DType::F32)
            .map_err(Error::from)?;
        let source_linear_b_t = vb
            .pp("m_source.l_linear")
            .get_unchecked_dtype("bias", DType::F32)
            .map_err(Error::from)?;
        let source_linear_w_v = source_linear_w_t.to_vec2::<f32>().map_err(Error::from)?;
        let source_linear_b_v = source_linear_b_t.to_vec1::<f32>().map_err(Error::from)?;
        if source_linear_w_v.len() != 1
            || source_linear_w_v[0].len() != 9
            || source_linear_b_v.len() != 1
        {
            return Err(Error::ModelLoadError(format!(
                "Unexpected Kokoro source linear shapes: weight={:?}, bias={:?}",
                source_linear_w_t.shape().dims(),
                source_linear_b_t.shape().dims(),
            )));
        }
        let mut source_linear_w = [0.0f32; 9];
        source_linear_w.copy_from_slice(&source_linear_w_v[0]);
        let source_linear_b = source_linear_b_v[0];
        let stft = KokoroStft::new(cfg.gen_istft_n_fft, cfg.gen_istft_hop_size);

        Ok(Self {
            cfg: cfg.clone(),
            num_kernels,
            num_upsamples,
            total_scale,
            harmonic_num: 8,
            sine_amp: 0.1,
            noise_std: 0.003,
            voiced_threshold: 10.0,
            ups,
            resblocks,
            noise_convs,
            noise_res,
            conv_post,
            source_linear_w,
            source_linear_b,
            stft,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        style: &Tensor,
        f0_curve: &Tensor,
        rng_seed: Option<u64>,
    ) -> Result<Vec<f32>> {
        let profile = kokoro_profile_enabled();
        let t0 = Instant::now();
        let har = self.harmonic_features(f0_curve, x.device(), rng_seed)?; // [B, n_fft+2, T_har]
        if profile {
            log_kokoro_profile("generator.harmonic_features", t0.elapsed());
        }

        let t1 = Instant::now();
        let mut x = x.clone();
        for i in 0..self.num_upsamples {
            let stage_t0 = if profile { Some(Instant::now()) } else { None };
            let t_stage_leaky = if profile { Some(Instant::now()) } else { None };
            x = ops::leaky_relu(&x, 0.1).map_err(Error::from)?;
            if let Some(t) = t_stage_leaky {
                log_kokoro_profile(&format!("generator.stage.{i}.leaky_relu"), t.elapsed());
            }
            let t_stage_branches = if profile { Some(Instant::now()) } else { None };
            let (x_up, x_source) = self.run_stage_branches(i, &x, &har, style)?;
            if let Some(t) = t_stage_branches {
                log_kokoro_profile(&format!("generator.stage.{i}.branches"), t.elapsed());
            }
            x = x_up;
            let t_stage_add = if profile { Some(Instant::now()) } else { None };
            if i + 1 == self.num_upsamples {
                x = reflection_pad_left1(&x)?;
            }
            x = match_time_add(&x, &x_source)?;
            if let Some(t) = t_stage_add {
                log_kokoro_profile(&format!("generator.stage.{i}.pad_add"), t.elapsed());
            }

            let base = i * self.num_kernels;
            let t_stage_resblocks = if profile { Some(Instant::now()) } else { None };
            let xs = self.run_stage_resblocks(base, &x, style)?;
            if let Some(t) = t_stage_resblocks {
                log_kokoro_profile(&format!("generator.stage.{i}.resblocks"), t.elapsed());
            }
            let t_stage_avg = if profile { Some(Instant::now()) } else { None };
            x = (xs / self.num_kernels as f64).map_err(Error::from)?;
            if let Some(t) = t_stage_avg {
                log_kokoro_profile(&format!("generator.stage.{i}.average"), t.elapsed());
            }
            if let Some(t) = stage_t0 {
                log_kokoro_profile(&format!("generator.stage.{i}.total"), t.elapsed());
            }
        }
        if profile {
            log_kokoro_profile("generator.neural_upsample", t1.elapsed());
        }

        let t2 = Instant::now();
        x = ops::leaky_relu(&x, 0.01).map_err(Error::from)?;
        x = self.conv_post.forward(&x).map_err(Error::from)?;
        let (_b, c, _t) = x.dims3().map_err(Error::from)?;
        let n_bins = self.cfg.gen_istft_n_fft / 2 + 1;
        if c < n_bins * 2 {
            return Err(Error::InferenceError(format!(
                "Kokoro generator conv_post output channels {} < required {}",
                c,
                n_bins * 2
            )));
        }
        let spec = x
            .narrow(1, 0, n_bins)
            .map_err(Error::from)?
            .exp()
            .map_err(Error::from)?;
        let phase = x
            .narrow(1, n_bins, n_bins)
            .map_err(Error::from)?
            .sin()
            .map_err(Error::from)?;
        if profile {
            log_kokoro_profile("generator.conv_post_spec_phase", t2.elapsed());
        }
        let t3 = Instant::now();
        self.stft.inverse(&spec, &phase).inspect(|_| {
            if profile {
                log_kokoro_profile("generator.istft_inverse", t3.elapsed());
                log_kokoro_profile("generator.total", t0.elapsed());
            }
        })
    }

    fn harmonic_features(
        &self,
        f0_curve: &Tensor,
        device: &candle_core::Device,
        rng_seed: Option<u64>,
    ) -> Result<Tensor> {
        let profile = kokoro_profile_enabled();
        let t0 = Instant::now();
        let f0_curve = f0_curve.squeeze(0).map_err(Error::from)?;
        let f0 = f0_curve.to_vec1::<f32>().map_err(Error::from)?;
        if profile {
            log_kokoro_profile("generator.harmonic.f0_download", t0.elapsed());
        }
        if f0_curve.rank() != 1 {
            return Err(Error::InferenceError(
                "Kokoro generator currently supports batch size 1 for harmonic source".to_string(),
            ));
        }
        if f0.is_empty() {
            return Err(Error::InferenceError(
                "Kokoro generator received empty F0 curve".to_string(),
            ));
        }
        let t1 = Instant::now();
        let upsampled_f0 = repeat_nearest(&f0, self.total_scale);
        let seed = rng_seed.unwrap_or_else(rand::random::<u64>);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let har_source = synth_harmonic_source_kokoro(
            &upsampled_f0,
            self.harmonic_num,
            self.total_scale,
            self.sine_amp,
            self.noise_std,
            self.voiced_threshold,
            &self.source_linear_w,
            self.source_linear_b,
            KokoroConfig::TARGET_SAMPLE_RATE as f32,
            &mut rng,
        );
        if profile {
            log_kokoro_profile("generator.harmonic.source", t1.elapsed());
        }
        let t2 = Instant::now();
        let (mag, phase) = self.stft.transform(&har_source)?;
        if profile {
            log_kokoro_profile("generator.harmonic.stft", t2.elapsed());
        }
        let n_bins = self.cfg.gen_istft_n_fft / 2 + 1;
        let frames = if n_bins == 0 { 0 } else { mag.len() / n_bins };
        let mut har = vec![0.0f32; n_bins * 2 * frames];
        for k in 0..n_bins {
            for t in 0..frames {
                har[k * frames + t] = mag[t * n_bins + k];
                har[(n_bins + k) * frames + t] = phase[t * n_bins + k];
            }
        }
        let t3 = Instant::now();
        let har_t = Tensor::from_vec(har, (1, n_bins * 2, frames), device).map_err(Error::from)?;
        if profile {
            log_kokoro_profile("generator.harmonic.tensor_upload", t3.elapsed());
            log_kokoro_profile("generator.harmonic.total", t0.elapsed());
        }
        Ok(har_t)
    }

    fn run_stage_branches(
        &self,
        i: usize,
        x: &Tensor,
        har: &Tensor,
        style: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        if x.device().is_cpu() && kokoro_cpu_stage_branches_parallel_enabled() {
            return self.run_stage_branches_parallel_cpu(i, x, har, style);
        }

        let mut x_source = self.noise_convs[i].forward(har).map_err(Error::from)?;
        x_source = self.noise_res[i].forward(&x_source, style)?;
        let x_up = self.ups[i].forward(x).map_err(Error::from)?;
        Ok((x_up, x_source))
    }

    fn run_stage_branches_parallel_cpu(
        &self,
        i: usize,
        x: &Tensor,
        har: &Tensor,
        style: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        thread::scope(|scope| {
            let noise_conv = &self.noise_convs[i];
            let noise_res = &self.noise_res[i];
            let up = &self.ups[i];
            let har = har.clone();
            let style = style.clone();
            let x = x.clone();

            let source_handle = scope.spawn(move || {
                let mut x_source = noise_conv.forward(&har).map_err(|e| e.to_string())?;
                x_source = noise_res
                    .forward(&x_source, &style)
                    .map_err(|e| e.to_string())?;
                Ok::<Tensor, String>(x_source)
            });
            let up_handle = scope.spawn(move || {
                up.forward(&x)
                    .map_err(Error::from)
                    .map_err(|e| e.to_string())
            });

            let x_source = match source_handle.join() {
                Ok(Ok(t)) => t,
                Ok(Err(msg)) => return Err(Error::InferenceError(msg)),
                Err(_) => {
                    return Err(Error::InferenceError(
                        "Kokoro generator source branch worker thread panicked".to_string(),
                    ))
                }
            };
            let x_up = match up_handle.join() {
                Ok(Ok(t)) => t,
                Ok(Err(msg)) => return Err(Error::InferenceError(msg)),
                Err(_) => {
                    return Err(Error::InferenceError(
                        "Kokoro generator upsample branch worker thread panicked".to_string(),
                    ))
                }
            };

            Ok((x_up, x_source))
        })
    }

    fn run_stage_resblocks(&self, base: usize, x: &Tensor, style: &Tensor) -> Result<Tensor> {
        if x.device().is_cpu() && self.num_kernels > 1 && kokoro_cpu_resblocks_parallel_enabled() {
            return self.run_stage_resblocks_parallel_cpu(base, x, style);
        }
        let mut xs = self.resblocks[base].forward(x, style)?;
        for j in 1..self.num_kernels {
            let y = self.resblocks[base + j].forward(x, style)?;
            xs = (xs + y).map_err(Error::from)?;
        }
        Ok(xs)
    }

    fn run_stage_resblocks_parallel_cpu(
        &self,
        base: usize,
        x: &Tensor,
        style: &Tensor,
    ) -> Result<Tensor> {
        let mut outputs = thread::scope(|scope| {
            let mut handles = Vec::with_capacity(self.num_kernels);
            for j in 0..self.num_kernels {
                let block = &self.resblocks[base + j];
                let xj = x.clone();
                let sj = style.clone();
                handles
                    .push(scope.spawn(move || block.forward(&xj, &sj).map_err(|e| e.to_string())));
            }
            let mut outs = Vec::with_capacity(handles.len());
            for h in handles {
                match h.join() {
                    Ok(Ok(t)) => outs.push(t),
                    Ok(Err(msg)) => return Err(Error::InferenceError(msg)),
                    Err(_) => {
                        return Err(Error::InferenceError(
                            "Kokoro generator resblock worker thread panicked".to_string(),
                        ))
                    }
                }
            }
            Ok::<Vec<Tensor>, Error>(outs)
        })?;
        let mut acc = outputs.remove(0);
        for y in outputs {
            acc = (acc + y).map_err(Error::from)?;
        }
        Ok(acc)
    }
}

#[derive(Debug)]
struct AdaInResBlock1 {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
    adain1: Vec<AdaIN1d>,
    adain2: Vec<AdaIN1d>,
    alpha1: Vec<Tensor>,
    alpha2: Vec<Tensor>,
}

impl AdaInResBlock1 {
    fn load(
        channels: usize,
        kernel_size: usize,
        dilations: &[usize],
        style_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        if dilations.len() != 3 {
            return Err(Error::ModelLoadError(format!(
                "Kokoro AdaInResBlock1 expects 3 dilations, got {}",
                dilations.len()
            )));
        }
        let mut convs1 = Vec::with_capacity(3);
        let mut convs2 = Vec::with_capacity(3);
        let mut adain1 = Vec::with_capacity(3);
        let mut adain2 = Vec::with_capacity(3);
        let mut alpha1 = Vec::with_capacity(3);
        let mut alpha2 = Vec::with_capacity(3);
        for j in 0..3 {
            let d1 = dilations[j];
            convs1.push(load_weight_norm_conv1d(
                vb.pp(format!("convs1.{j}")),
                Conv1dConfig {
                    padding: get_padding(kernel_size, d1),
                    stride: 1,
                    dilation: d1,
                    groups: 1,
                    cudnn_fwd_algo: None,
                },
            )?);
            convs2.push(load_weight_norm_conv1d(
                vb.pp(format!("convs2.{j}")),
                Conv1dConfig {
                    padding: get_padding(kernel_size, 1),
                    stride: 1,
                    dilation: 1,
                    groups: 1,
                    cudnn_fwd_algo: None,
                },
            )?);
            adain1.push(AdaIN1d::load(
                style_dim,
                channels,
                vb.pp(format!("adain1.{j}")),
            )?);
            adain2.push(AdaIN1d::load(
                style_dim,
                channels,
                vb.pp(format!("adain2.{j}")),
            )?);
            alpha1.push(
                vb.get_unchecked_dtype(&format!("alpha1.{j}"), DType::F32)
                    .map_err(Error::from)?,
            );
            alpha2.push(
                vb.get_unchecked_dtype(&format!("alpha2.{j}"), DType::F32)
                    .map_err(Error::from)?,
            );
        }
        Ok(Self {
            convs1,
            convs2,
            adain1,
            adain2,
            alpha1,
            alpha2,
        })
    }

    fn forward(&self, x: &Tensor, style: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for j in 0..3 {
            let mut xt = self.adain1[j].forward_snake(&x, style, &self.alpha1[j])?;
            xt = self.convs1[j].forward(&xt).map_err(Error::from)?;
            xt = self.adain2[j].forward_snake(&xt, style, &self.alpha2[j])?;
            xt = self.convs2[j].forward(&xt).map_err(Error::from)?;
            x = (xt + x).map_err(Error::from)?;
        }
        Ok(x)
    }
}

struct KokoroStft {
    n_fft: usize,
    hop: usize,
    window: Vec<f32>,
    fft_fwd: Arc<dyn Fft<f32>>,
    fft_inv: Arc<dyn Fft<f32>>,
}

impl fmt::Debug for KokoroStft {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KokoroStft")
            .field("n_fft", &self.n_fft)
            .field("hop", &self.hop)
            .field("window_len", &self.window.len())
            .finish()
    }
}

impl Clone for KokoroStft {
    fn clone(&self) -> Self {
        Self {
            n_fft: self.n_fft,
            hop: self.hop,
            window: self.window.clone(),
            fft_fwd: self.fft_fwd.clone(),
            fft_inv: self.fft_inv.clone(),
        }
    }
}

impl KokoroStft {
    fn new(n_fft: usize, hop: usize) -> Self {
        let mut planner = FftPlanner::<f32>::new();
        Self {
            n_fft,
            hop,
            window: hann_window_periodic(n_fft),
            fft_fwd: planner.plan_fft_forward(n_fft.max(1)),
            fft_inv: planner.plan_fft_inverse(n_fft.max(1)),
        }
    }

    fn transform(&self, input: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        if input.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        if self.n_fft == 0 || self.hop == 0 {
            return Err(Error::InferenceError(
                "Invalid Kokoro STFT n_fft/hop".to_string(),
            ));
        }
        let pad = self.n_fft / 2;
        let padded = reflect_pad_1d_center(input, pad)?;
        if padded.len() < self.n_fft {
            return Ok((Vec::new(), Vec::new()));
        }
        let n_bins = self.n_fft / 2 + 1;
        let frames = (padded.len() - self.n_fft) / self.hop + 1;
        let mut mag = vec![0.0f32; frames * n_bins];
        let mut phase = vec![0.0f32; frames * n_bins];
        let mut buf = vec![Complex32::new(0.0, 0.0); self.n_fft];

        for frame_idx in 0..frames {
            let start = frame_idx * self.hop;
            for i in 0..self.n_fft {
                buf[i] = Complex32::new(padded[start + i] * self.window[i], 0.0);
            }
            self.fft_fwd.process(&mut buf);
            for k in 0..n_bins {
                let c = buf[k];
                mag[frame_idx * n_bins + k] = c.norm();
                phase[frame_idx * n_bins + k] = c.arg();
            }
        }
        Ok((mag, phase))
    }

    fn inverse(&self, magnitude: &Tensor, phase: &Tensor) -> Result<Vec<f32>> {
        let (b, n_bins, frames) = magnitude.dims3().map_err(Error::from)?;
        let (b2, n_bins2, frames2) = phase.dims3().map_err(Error::from)?;
        if b != 1 || b2 != 1 || n_bins != n_bins2 || frames != frames2 {
            return Err(Error::InferenceError(format!(
                "Kokoro iSTFT expects matching [1,n_bins,frames] tensors, got mag={:?}, phase={:?}",
                magnitude.shape().dims(),
                phase.shape().dims(),
            )));
        }
        let mag = magnitude
            .flatten_all()
            .map_err(Error::from)?
            .to_vec1::<f32>()
            .map_err(Error::from)?;
        let ph = phase
            .flatten_all()
            .map_err(Error::from)?
            .to_vec1::<f32>()
            .map_err(Error::from)?;

        if frames == 0 {
            return Ok(Vec::new());
        }
        let output_len = (frames - 1) * self.hop + self.n_fft;
        let mut output = vec![0.0f32; output_len];
        let mut envelope = vec![0.0f32; output_len];
        let mut spectrum = vec![Complex32::new(0.0, 0.0); self.n_fft];

        for frame_idx in 0..frames {
            spectrum.fill(Complex32::new(0.0, 0.0));
            for k in 0..n_bins {
                let idx = k * frames + frame_idx;
                let m = mag[idx].max(0.0);
                let p = ph[idx];
                spectrum[k] = Complex32::from_polar(m, p);
            }
            for k in 1..(n_bins.saturating_sub(1)) {
                spectrum[self.n_fft - k] = spectrum[k].conj();
            }
            self.fft_inv.process(&mut spectrum);
            let start = frame_idx * self.hop;
            for n in 0..self.n_fft {
                let sample = (spectrum[n].re / self.n_fft as f32) * self.window[n];
                let idx = start + n;
                output[idx] += sample;
                envelope[idx] += self.window[n] * self.window[n];
            }
        }

        for (y, env) in output.iter_mut().zip(envelope.iter()) {
            if *env > 1e-8 {
                *y /= *env;
            }
            if !y.is_finite() {
                *y = 0.0;
            }
        }

        let pad = self.n_fft / 2;
        let mut trimmed = if output.len() > pad * 2 {
            output[pad..output.len() - pad].to_vec()
        } else {
            output
        };
        for s in &mut trimmed {
            *s = s.clamp(-1.0, 1.0);
        }
        Ok(trimmed)
    }
}

fn get_padding(kernel_size: usize, dilation: usize) -> usize {
    (kernel_size
        .saturating_mul(dilation)
        .saturating_sub(dilation))
        / 2
}

fn upsample_nearest_1d(x: &Tensor, factor: usize) -> Result<Tensor> {
    if factor <= 1 {
        return Ok(x.clone());
    }
    let (b, c, t) = x.dims3().map_err(Error::from)?;
    x.unsqueeze(3)
        .map_err(Error::from)?
        .broadcast_as((b, c, t, factor))
        .map_err(Error::from)?
        .reshape((b, c, t * factor))
        .map_err(Error::from)
}

fn reflection_pad_left1(x: &Tensor) -> Result<Tensor> {
    let (_b, _c, t) = x.dims3().map_err(Error::from)?;
    if t < 2 {
        return Err(Error::InferenceError(
            "Kokoro reflection pad requires time length >= 2".to_string(),
        ));
    }
    let left = x.narrow(2, 1, 1).map_err(Error::from)?;
    Tensor::cat(&[left, x.clone()], 2).map_err(Error::from)
}

fn match_time_add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let (ba, ca, ta) = a.dims3().map_err(Error::from)?;
    let (bb, cb, tb) = b.dims3().map_err(Error::from)?;
    if ba != bb || ca != cb {
        return Err(Error::InferenceError(format!(
            "Kokoro generator add shape mismatch {:?} vs {:?}",
            a.shape().dims(),
            b.shape().dims()
        )));
    }
    if ta == tb {
        return (a + b).map_err(Error::from);
    }
    let t = ta.min(tb);
    let a2 = a.narrow(2, 0, t).map_err(Error::from)?;
    let b2 = b.narrow(2, 0, t).map_err(Error::from)?;
    (a2 + b2).map_err(Error::from)
}

fn repeat_nearest(input: &[f32], factor: usize) -> Vec<f32> {
    if factor <= 1 {
        return input.to_vec();
    }
    let mut out = Vec::with_capacity(input.len().saturating_mul(factor));
    for &v in input {
        for _ in 0..factor {
            out.push(v);
        }
    }
    out
}

fn synth_harmonic_source_kokoro(
    upsampled_f0: &[f32],
    harmonic_num: usize,
    upsample_scale: usize,
    sine_amp: f32,
    noise_std: f32,
    voiced_threshold: f32,
    linear_w: &[f32; 9],
    linear_b: f32,
    sample_rate: f32,
    rng: &mut ChaCha8Rng,
) -> Vec<f32> {
    let t = upsampled_f0.len();
    if t == 0 {
        return Vec::new();
    }
    let dim = harmonic_num + 1;
    debug_assert_eq!(dim, 9);

    let mut f0_values = vec![0.0f32; t * dim];
    let mut uv = vec![0.0f32; t];
    for (ti, &f0) in upsampled_f0.iter().enumerate() {
        let voiced = f0.is_finite() && f0 > voiced_threshold;
        uv[ti] = if voiced { 1.0 } else { 0.0 };
        let base = if f0.is_finite() { f0.max(0.0) } else { 0.0 };
        for h in 0..dim {
            f0_values[ti * dim + h] = base * (h + 1) as f32;
        }
    }

    let mut sine_waves = sinegen_f02sine(&f0_values, t, dim, sample_rate, upsample_scale, rng);
    for ti in 0..t {
        let noise_amp = uv[ti] * noise_std + (1.0 - uv[ti]) * (sine_amp / 3.0);
        for h in 0..dim {
            let idx = ti * dim + h;
            let z: f32 = StandardNormal.sample(rng);
            let noise = noise_amp * z;
            sine_waves[idx] = sine_waves[idx] * sine_amp * uv[ti] + noise;
        }
    }

    let mut out = vec![0.0f32; t];
    for ti in 0..t {
        let mut acc = linear_b;
        for h in 0..dim {
            acc += linear_w[h] * sine_waves[ti * dim + h];
        }
        out[ti] = acc.tanh();
    }

    // Mirror SourceModuleHnNSF's unused noise branch RNG draw for behavior parity.
    for _ in 0..t {
        let _z: f32 = StandardNormal.sample(rng);
        let _unused_noise = _z * (sine_amp / 3.0);
    }
    out
}

fn sinegen_f02sine(
    f0_values: &[f32], // [T, D]
    t: usize,
    dim: usize,
    sample_rate: f32,
    upsample_scale: usize,
    rng: &mut ChaCha8Rng,
) -> Vec<f32> {
    if t == 0 || dim == 0 {
        return Vec::new();
    }
    let mut rad_values = vec![0.0f32; f0_values.len()];
    let inv_sr = 1.0f32 / sample_rate.max(1.0);
    for i in 0..f0_values.len() {
        let v = f0_values[i] * inv_sr;
        rad_values[i] = v.rem_euclid(1.0);
    }
    for h in 0..dim {
        let init = if h == 0 { 0.0 } else { rng.gen::<f32>() };
        rad_values[h] += init;
    }

    let down_t = (t / upsample_scale.max(1)).max(1);
    let rad_down = linear_resample_time_time_major(&rad_values, t, down_t, dim);
    let mut phase_down = vec![0.0f32; rad_down.len()];
    for h in 0..dim {
        let mut acc = 0.0f32;
        for ti in 0..down_t {
            let idx = ti * dim + h;
            acc += rad_down[idx];
            phase_down[idx] = acc * (2.0 * PI);
        }
    }
    let scale = upsample_scale.max(1) as f32;
    for v in &mut phase_down {
        *v *= scale;
    }
    let phase = linear_resample_time_time_major(&phase_down, down_t, t, dim);
    phase.into_iter().map(f32::sin).collect()
}

fn linear_resample_time_time_major(
    input: &[f32],
    in_t: usize,
    out_t: usize,
    channels: usize,
) -> Vec<f32> {
    if in_t == 0 || out_t == 0 || channels == 0 {
        return Vec::new();
    }
    if in_t == out_t {
        return input.to_vec();
    }
    let mut out = vec![0.0f32; out_t * channels];
    let scale = in_t as f32 / out_t as f32;
    let max_x = (in_t - 1) as f32;
    for ot in 0..out_t {
        let mut x = (ot as f32 + 0.5) * scale - 0.5;
        if !x.is_finite() {
            x = 0.0;
        }
        x = x.clamp(0.0, max_x);
        let x0 = x.floor() as usize;
        let x1 = (x0 + 1).min(in_t - 1);
        let w1 = (x - x0 as f32).clamp(0.0, 1.0);
        let w0 = 1.0 - w1;
        let in0 = x0 * channels;
        let in1 = x1 * channels;
        let out_base = ot * channels;
        for c in 0..channels {
            out[out_base + c] = input[in0 + c] * w0 + input[in1 + c] * w1;
        }
    }
    out
}

fn reflect_pad_1d_center(input: &[f32], pad: usize) -> Result<Vec<f32>> {
    if pad == 0 {
        return Ok(input.to_vec());
    }
    if input.len() <= 1 || pad >= input.len() {
        return Err(Error::InferenceError(format!(
            "Kokoro STFT reflect pad invalid for len={} pad={}",
            input.len(),
            pad
        )));
    }
    let mut out = Vec::with_capacity(input.len() + pad * 2);
    for i in 0..pad {
        out.push(input[pad - i]);
    }
    out.extend_from_slice(input);
    for i in 0..pad {
        out.push(input[input.len() - 2 - i]);
    }
    Ok(out)
}

fn hann_window_periodic(n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / n as f32).cos())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kokoro_hann_window_periodic_smoke() {
        let w = hann_window_periodic(20);
        assert_eq!(w.len(), 20);
        assert!(w.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn kokoro_repeat_nearest_scales_len() {
        let x = vec![1.0f32, 2.0, 3.0];
        let y = repeat_nearest(&x, 4);
        assert_eq!(y.len(), 12);
        assert_eq!(y[0], 1.0);
        assert_eq!(y[4], 2.0);
        assert_eq!(y[8], 3.0);
    }

    #[test]
    fn kokoro_stft_roundtrip_smoke() {
        let stft = KokoroStft::new(20, 5);
        let mut x = Vec::with_capacity(600);
        for i in 0..600 {
            x.push((i as f32 * 0.01).sin() * 0.2);
        }
        let (mag, phase) = stft.transform(&x).expect("stft transform");
        let n_bins = 20 / 2 + 1;
        let frames = mag.len() / n_bins;
        let device = candle_core::Device::Cpu;
        let mut mag_ch = vec![0.0f32; n_bins * frames];
        let mut phase_ch = vec![0.0f32; n_bins * frames];
        for t in 0..frames {
            for k in 0..n_bins {
                mag_ch[k * frames + t] = mag[t * n_bins + k];
                phase_ch[k * frames + t] = phase[t * n_bins + k];
            }
        }
        let mag_t = Tensor::from_vec(mag_ch, (1, n_bins, frames), &device).expect("mag tensor");
        let phase_t =
            Tensor::from_vec(phase_ch, (1, n_bins, frames), &device).expect("phase tensor");
        let y = stft.inverse(&mag_t, &phase_t).expect("istft inverse");
        assert!(!y.is_empty());
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn kokoro_linear_resample_time_major_preserves_constant() {
        let input = vec![1.25f32; 8 * 3];
        let down = linear_resample_time_time_major(&input, 8, 2, 3);
        let up = linear_resample_time_time_major(&down, 2, 8, 3);
        assert!(up.iter().all(|v| (*v - 1.25).abs() < 1e-5));
    }

    #[test]
    fn kokoro_seeded_harmonic_source_is_repeatable() {
        let f0 = vec![110.0f32; 32];
        let weights = [0.1f32; 9];
        let mut rng_a1 = ChaCha8Rng::seed_from_u64(12345);
        let mut rng_a2 = ChaCha8Rng::seed_from_u64(12345);
        let mut rng_b = ChaCha8Rng::seed_from_u64(54321);
        let a1 = synth_harmonic_source_kokoro(
            &f0,
            8,
            5,
            0.1,
            0.003,
            10.0,
            &weights,
            0.0,
            24_000.0,
            &mut rng_a1,
        );
        let a2 = synth_harmonic_source_kokoro(
            &f0,
            8,
            5,
            0.1,
            0.003,
            10.0,
            &weights,
            0.0,
            24_000.0,
            &mut rng_a2,
        );
        let b = synth_harmonic_source_kokoro(
            &f0, 8, 5, 0.1, 0.003, 10.0, &weights, 0.0, 24_000.0, &mut rng_b,
        );
        assert_eq!(a1.len(), f0.len());
        assert_eq!(a1, a2);
        assert_ne!(a1, b);
    }
}
