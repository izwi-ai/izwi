//! The Mamba2 mixer: `in_proj` → `[z | xBC | dt]` → causal conv with bias
//! and SiLU → selective scan → gated grouped RMSNorm `norm(y * silu(z))`
//! → `out_proj`.

use candle_core::{Device, Tensor};

use super::super::checkpoint::NemotronCheckpoint;
use super::super::config::{MambaConfig, NemotronTranscribeConfig};
use super::super::linear::{NtLinear, Pass};
use super::super::precision::Fp8Mode;
use crate::error::Result;
use crate::kernels::mamba2;

pub struct MambaMixer {
    config: MambaConfig,
    in_proj: NtLinear,
    out_proj: NtLinear,
    /// `[conv_channels, kernel]` F32.
    conv_weight: Tensor,
    conv_bias: Tensor,
    /// `[3, heads]`: `dt_bias`, `A = -exp(A_log)`, `D`.
    params: Tensor,
    norm: Tensor,
    eps: f64,
}

impl MambaMixer {
    pub fn load(
        checkpoint: &NemotronCheckpoint,
        config: &NemotronTranscribeConfig,
        layer: usize,
        device: &Device,
        mode: Fp8Mode,
    ) -> Result<Self> {
        let d = &config.decoder;
        let m = d.mamba;
        let x = format!("backbone.layers.{layer}.mixer");
        let channels = m.conv_channels();
        let conv =
            checkpoint.f32_values(&format!("{x}.conv1d.weight"), &[channels, 1, m.conv_kernel])?;
        let a_log = checkpoint.f32_values(&format!("{x}.A_log"), &[m.heads])?;
        let mut params = checkpoint.f32_values(&format!("{x}.dt_bias"), &[m.heads])?;
        // A = -exp(A_log) in F32 on the host (precise exp).
        params.extend(a_log.iter().map(|v| -v.exp()));
        params.extend(checkpoint.f32_values(&format!("{x}.D"), &[m.heads])?);
        Ok(Self {
            config: m,
            in_proj: NtLinear::load_fp8(
                checkpoint,
                &format!("{x}.in_proj"),
                m.in_proj_rows(),
                d.hidden,
                false,
                device,
                mode,
            )?,
            out_proj: NtLinear::load_fp8(
                checkpoint,
                &format!("{x}.out_proj"),
                d.hidden,
                m.inner(),
                false,
                device,
                mode,
            )?,
            conv_weight: Tensor::from_vec(conv, (channels, m.conv_kernel), device)?,
            conv_bias: Tensor::from_vec(
                checkpoint.f32_values(&format!("{x}.conv1d.bias"), &[channels])?,
                channels,
                device,
            )?,
            params: Tensor::from_vec(params, (3, m.heads), device)?,
            norm: Tensor::from_vec(
                checkpoint.f32_values(&format!("{x}.norm.weight"), &[m.inner()])?,
                m.inner(),
                device,
            )?,
            eps: d.norm_eps,
        })
    }

    pub fn resident_bytes(&self) -> usize {
        self.in_proj.resident_bytes() + self.out_proj.resident_bytes()
    }

    pub fn in_proj(&self) -> &NtLinear {
        &self.in_proj
    }

    pub fn out_proj(&self) -> &NtLinear {
        &self.out_proj
    }

    /// `h` `[rows * tokens, hidden]` (row-major: each row's tokens
    /// contiguous) with each row's convolution history `[rows, K - 1, C]`
    /// and SSM state `[rows, heads, head_dim, state]`. Returns the mixer
    /// output and the next history and state.
    pub fn forward(
        &self,
        h: &Tensor,
        rows: usize,
        tokens: usize,
        conv: &Tensor,
        ssm: &Tensor,
        pass: Pass,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let m = self.config;
        let projected = self
            .in_proj
            .forward(h, pass)?
            .reshape((rows, tokens, m.in_proj_rows()))?;
        let z = projected.narrow(2, 0, m.inner())?;
        let xbc = projected
            .narrow(2, m.inner(), m.conv_channels())?
            .contiguous()?;
        let dt = projected
            .narrow(2, m.inner() + m.conv_channels(), m.heads)?
            .contiguous()?;
        let (u, next_conv) =
            mamba2::causal_conv_silu(&xbc, conv, &self.conv_weight, &self.conv_bias)?;
        let (y, next_ssm) =
            mamba2::selective_scan(m.geometry(), &u.contiguous()?, &dt, &self.params, ssm)?;
        let gated = mamba2::gated_group_rms_norm(
            &y,
            &z.contiguous()?,
            &self.norm,
            m.norm_group(),
            self.eps,
        )?;
        let out = self
            .out_proj
            .forward(&gated.reshape((rows * tokens, m.inner()))?, pass)?;
        Ok((out, next_conv, next_ssm))
    }
}
