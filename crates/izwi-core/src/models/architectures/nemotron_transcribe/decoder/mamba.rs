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

    /// Several rows' prefill spans packed into `h` `[sum(lengths), hidden]`
    /// (row `r`'s `lengths[r]` tokens contiguous, rows in order), each row
    /// with its own convolution history `[1, K - 1, C]` and SSM state
    /// `[1, heads, head_dim, state]`. The projections (each row through the
    /// kernel its span alone takes) and the gated norm run once over every
    /// token; the convolution and the scan run per row over that row's
    /// tokens (one call over the stacked rows when every row has the same
    /// length), so each row's recurrence is its own whatever shares the
    /// call. Returns the output and each row's next history and state.
    pub fn forward_packed(
        &self,
        h: &Tensor,
        lengths: &[usize],
        conv: &[&Tensor],
        ssm: &[&Tensor],
    ) -> Result<(Tensor, Vec<Tensor>, Vec<Tensor>)> {
        let m = self.config;
        let rows = lengths.len();
        if rows == 0 || conv.len() != rows || ssm.len() != rows || lengths.contains(&0) {
            return Err(crate::error::Error::InferenceError(
                "a packed Mamba2 call needs one non-empty span and one state per row".into(),
            ));
        }
        let projected = self.in_proj.forward_rows(h, lengths, false)?;
        let z = projected.narrow(1, 0, m.inner())?;
        let xbc = projected.narrow(1, m.inner(), m.conv_channels())?;
        let dt = projected.narrow(1, m.inner() + m.conv_channels(), m.heads)?;
        let (y, next_conv, next_ssm) = if lengths.iter().all(|&len| len == lengths[0]) {
            let tokens = lengths[0];
            let xbc = xbc
                .contiguous()?
                .reshape((rows, tokens, m.conv_channels()))?;
            let dt = dt.contiguous()?.reshape((rows, tokens, m.heads))?;
            let (u, next_conv) = mamba2::causal_conv_silu(
                &xbc,
                &Tensor::cat(conv, 0)?,
                &self.conv_weight,
                &self.conv_bias,
            )?;
            let (y, next_ssm) = mamba2::selective_scan(
                m.geometry(),
                &u.contiguous()?,
                &dt,
                &self.params,
                &Tensor::cat(ssm, 0)?,
            )?;
            let split = |t: &Tensor| -> Result<Vec<Tensor>> {
                (0..rows).map(|r| Ok(t.narrow(0, r, 1)?)).collect()
            };
            (
                y.reshape((rows * tokens, m.inner()))?,
                split(&next_conv)?,
                split(&next_ssm)?,
            )
        } else {
            let mut ys = Vec::with_capacity(rows);
            let mut next_conv = Vec::with_capacity(rows);
            let mut next_ssm = Vec::with_capacity(rows);
            let mut offset = 0;
            for (r, &tokens) in lengths.iter().enumerate() {
                let xbc = xbc.narrow(0, offset, tokens)?.contiguous()?.reshape((
                    1,
                    tokens,
                    m.conv_channels(),
                ))?;
                let dt = dt
                    .narrow(0, offset, tokens)?
                    .contiguous()?
                    .reshape((1, tokens, m.heads))?;
                let (u, conv) =
                    mamba2::causal_conv_silu(&xbc, conv[r], &self.conv_weight, &self.conv_bias)?;
                let (y, ssm) = mamba2::selective_scan(
                    m.geometry(),
                    &u.contiguous()?,
                    &dt,
                    &self.params,
                    ssm[r],
                )?;
                ys.push(y.reshape((tokens, m.inner()))?);
                next_conv.push(conv);
                next_ssm.push(ssm);
                offset += tokens;
            }
            (Tensor::cat(&ys, 0)?, next_conv, next_ssm)
        };
        let gated = mamba2::gated_group_rms_norm(
            &y,
            &z.contiguous()?,
            &self.norm,
            m.norm_group(),
            self.eps,
        )?;
        let out = self.out_proj.forward_rows(&gated, lengths, false)?;
        Ok((out, next_conv, next_ssm))
    }

    /// Where a step's operands sit in the in_proj row `[z | xBC | dt]`.
    fn step_layout(&self) -> mamba2::Mamba2StepLayout {
        let m = self.config;
        mamba2::Mamba2StepLayout {
            width: m.in_proj_rows(),
            z_offset: 0,
            xbc_offset: m.inner(),
            dt_offset: m.inner() + m.conv_channels(),
            taps: m.conv_kernel,
        }
    }

    /// One decode token of every row through the step kernels: in_proj, the
    /// convolution step (read from the projection, writing dt for the
    /// scan), the scan, the gated norm (z read from the projection) and
    /// out_proj. `h` is `[rows, hidden]`; the states as in [`Self::forward`].
    /// Matches it to rounding.
    pub fn forward_step_fused(
        &self,
        h: &Tensor,
        conv: &Tensor,
        ssm: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let m = self.config;
        let rows = h.dim(0)?;
        let layout = self.step_layout();
        let projected = self.in_proj.forward(h, Pass::Step)?;
        let (u, dt, next_conv) = mamba2::conv_step_silu(
            m.geometry(),
            layout,
            &projected,
            conv,
            &self.conv_weight,
            &self.conv_bias,
        )?;
        let (y, next_ssm) = mamba2::selective_scan(
            m.geometry(),
            &u.reshape((rows, 1, m.conv_channels()))?,
            &dt.reshape((rows, 1, m.heads))?,
            &self.params,
            ssm,
        )?;
        let gated = mamba2::gated_group_rms_norm_step(
            &y.reshape((rows, m.inner()))?,
            &projected,
            layout,
            &self.norm,
            m.norm_group(),
            self.eps,
        )?;
        let out = self.out_proj.forward(&gated, Pass::Step)?;
        Ok((out, next_conv, next_ssm))
    }
}
