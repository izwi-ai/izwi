//! The codebook-MTP tower on a device: the CPU reference's layers
//! (`tower.rs`) in the device forms of `device/mod.rs`.
//!
//! ```text
//! m   = eh_proj([enorm(e) ; hnorm(h)])               # 16-bit, into the residual (F32 on Metal, BF16 on CUDA)
//! r   = m + wo(Attn(attn_ln(m)) * sigmoid(gate))    # contiguous wqkv repacked per head, interleaved RoPE
//! out = final_ln(r + w2(silu(w1 x) * w3 x))          # w2's output added into the residual
//! ```
//!
//! Injections gather the layer's 16-bit code table on the device; layer 0 adds
//! the input frame's `extra` (`FishDrama3Embedding::extra_device`). In a
//! decode step each head's code comes from the device race
//! (`metal::sampling`) and indexes the next layer's table without leaving
//! the device; the step reads the 9 codes back once.

use candle_core::{DType, Device, Tensor};

use super::super::checkpoint::{
    tower_head, tower_residual_layer, tower_shared_layer, FishDrama3Checkpoint,
};
use super::super::config::FishDrama3Config;
use super::super::embed::FishDrama3Embedding;
use super::super::linear::{FishDrama3KernelPhase, FishDrama3Linear};
use super::super::residency::FishDrama3Residency;
use super::super::sampling::FishDrama3DrawPolicy;
use super::super::trunk::{norm_gain, NormGain};
use super::sampling::{race, race_rows, rank};
use super::trunk::{gain, half_linear, half_linear_packed};
use super::{
    add_residual, record, rms_norm, swiglu, DeviceAttention, DeviceForms, DeviceRow, Probe,
};
use crate::error::{Error, Result};
use crate::kernels::residual_norm;

/// One tower layer on the device.
pub(crate) struct DeviceTowerLayer {
    /// `[V, dim]` F16.
    embedding: Tensor,
    enorm: Tensor,
    hnorm: Tensor,
    eh_proj: FishDrama3Linear,
    attention_norm: Tensor,
    attention: DeviceAttention,
    ffn_norm: Tensor,
    /// `[2 intermediate, dim]`: w1's rows, then w3's (gate, then up).
    w13: FishDrama3Linear,
    intermediate: usize,
    w2: FishDrama3Linear,
    final_norm: Tensor,
    /// The 16-bit dtype, or F32 once Metal's range probe breached the layer.
    pub(crate) act: DType,
}

impl DeviceTowerLayer {
    fn load(
        checkpoint: &FishDrama3Checkpoint,
        prefix: &str,
        embedding: &str,
        config: &FishDrama3Config,
        forms: DeviceForms,
        device: &Device,
    ) -> Result<Self> {
        let tower = &config.tower;
        let d = tower.dim;
        let a = tower.attention;
        let (qw, kvw) = (a.q_width(), a.kv_width());
        let eps = tower.norm_eps as f32;
        let norm = |name: &str, width: usize| {
            gain(
                checkpoint,
                &format!("{prefix}.{name}.weight"),
                width,
                NormGain::ZeroCentred,
                device,
            )
        };
        let x = format!("{prefix}.transformer_layer.attention");
        // Contiguous [Q | Gate | K | V] rows, repacked into the trunk's
        // per-head [q_h | gate_h] so both stacks share one kernel path.
        let wqkv = checkpoint.bf16_to_f32(&format!("{x}.wqkv.weight"), &[tower.wqkv_rows(), d])?;
        let (query, rest) = wqkv.split_at(qw * d);
        let (gate, rest) = rest.split_at(qw * d);
        let (keys, values) = rest.split_at(kvw * d);
        let gain_values = |name: &str| {
            norm_gain(
                checkpoint,
                &format!("{x}.{name}.weight"),
                a.head_dim,
                NormGain::ZeroCentred,
            )
        };
        let ffn = format!("{prefix}.transformer_layer.feed_forward");
        Ok(Self {
            embedding: checkpoint.bf16_tensor(
                &format!("{prefix}.{embedding}.weight"),
                &[tower.codebook_vocab, d],
                forms.half,
                device,
            )?,
            enorm: norm("enorm", d)?,
            hnorm: norm("hnorm", d)?,
            eh_proj: half_linear(
                checkpoint,
                &format!("{prefix}.eh_proj.weight"),
                d,
                2 * d,
                forms,
                device,
            )?,
            attention_norm: norm("transformer_layer.attention_norm", d)?,
            attention: DeviceAttention::new(
                DeviceAttention::interleave(query, gate, a.head_dim, d),
                keys,
                values,
                checkpoint.bf16_to_f32(&format!("{x}.wo.weight"), &[d, qw])?,
                &gain_values("q_norm")?,
                &gain_values("k_norm")?,
                a,
                eps,
                config
                    .trunk
                    .max_positions
                    .min(super::super::model::MAX_REQUEST_POSITIONS),
                forms,
                device,
            )?,
            ffn_norm: norm("transformer_layer.ffn_norm", d)?,
            w13: half_linear_packed(
                checkpoint,
                &[&format!("{ffn}.w1.weight"), &format!("{ffn}.w3.weight")],
                tower.intermediate,
                d,
                forms,
                device,
            )?,
            intermediate: tower.intermediate,
            w2: half_linear(
                checkpoint,
                &format!("{ffn}.w2.weight"),
                d,
                tower.intermediate,
                forms,
                device,
            )?,
            final_norm: norm("final_layernorm", d)?,
            act: forms.half,
        })
    }

    fn resident_bytes(&self) -> usize {
        [&self.eh_proj, &self.w13, &self.w2]
            .iter()
            .map(|l| l.resident_bytes())
            .sum::<usize>()
            + self.attention.resident_bytes()
            + self.embedding.elem_count() * 2
            + [
                &self.enorm,
                &self.hnorm,
                &self.attention_norm,
                &self.ffn_norm,
                &self.final_norm,
            ]
            .iter()
            .map(|t| t.elem_count() * 4)
            .sum::<usize>()
    }
}

/// Where a step's codes after `c0` come from.
pub(crate) enum DeviceStepCodes<'a> {
    /// The device race: each row's heads policy and its exponentials
    /// `[heads, rows, V]` F32 (head-major; each row's drawn on the host in
    /// head order).
    Device {
        policies: Vec<FishDrama3DrawPolicy>,
        noise: Tensor,
    },
    /// The host: head `j`'s logits `[rows, V]` (F32 on the device) to each
    /// row's code (host sampling, or a forced frame's).
    Host(&'a mut dyn FnMut(usize, &Tensor) -> Result<Vec<u32>>),
}

/// The tower's weights on the device.
pub(crate) struct DeviceTower {
    pub(crate) layers: Vec<DeviceTowerLayer>,
    /// `[V, dim]` 16-bit each; head `j` follows residual layer `j`.
    heads: Vec<FishDrama3Linear>,
    /// The heads' dtype (F32 once Metal's range probe breached one).
    pub(crate) head_act: DType,
    shared_layers: usize,
    vocab: usize,
    eps: f32,
    forms: DeviceForms,
    /// Span and step layers run [`Self::layer_fused`]; set after the load's
    /// fused kernel check.
    pub(crate) fused: bool,
}

impl DeviceTower {
    pub(crate) fn load(
        checkpoint: &FishDrama3Checkpoint,
        config: &FishDrama3Config,
        device: &Device,
    ) -> Result<Self> {
        let forms = DeviceForms::of(device)?;
        let t = &config.tower;
        let shared = (0..t.shared_layers).map(|index| {
            DeviceTowerLayer::load(
                checkpoint,
                &tower_shared_layer(index),
                "semantic_embedding",
                config,
                forms,
                device,
            )
        });
        let residual = (0..t.residual_layers()).map(|index| {
            DeviceTowerLayer::load(
                checkpoint,
                &tower_residual_layer(index),
                "codebook_embedding",
                config,
                forms,
                device,
            )
        });
        let layers = shared.chain(residual).collect::<Result<Vec<_>>>()?;
        let heads = (0..t.residual_layers())
            .map(|index| {
                half_linear(
                    checkpoint,
                    &tower_head(index),
                    t.codebook_vocab,
                    t.dim,
                    forms,
                    device,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            layers,
            heads,
            head_act: forms.half,
            shared_layers: t.shared_layers,
            vocab: t.codebook_vocab,
            eps: t.norm_eps as f32,
            forms,
            fused: false,
        })
    }

    pub(crate) fn resident_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(DeviceTowerLayer::resident_bytes)
            .sum::<usize>()
            + self.heads.iter().map(|h| h.resident_bytes()).sum::<usize>()
    }

    pub(crate) fn heads(&self) -> usize {
        self.heads.len()
    }

    /// Layer `index` over injections `e` (`[N, dim]` F32) and hidden `h`
    /// (`[N, dim]`), its K/V in slot `index` of each request's view: the
    /// layer's final-norm output `[N, dim]` in its dtype (`act` overrides
    /// it).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn layer(
        &self,
        index: usize,
        e: &Tensor,
        h: &Tensor,
        rows: &mut [DeviceRow<'_>],
        phase: FishDrama3KernelPhase,
        act: Option<DType>,
        probe: &mut Probe<'_>,
    ) -> Result<Tensor> {
        let layer = &self.layers[index];
        let act = act.unwrap_or(layer.act);
        let e = rms_norm(e, &layer.enorm, self.eps, act)?;
        let h = rms_norm(h, &layer.hnorm, self.eps, act)?;
        let merged = Tensor::cat(&[&e, &h], 1)?;
        let x = layer.eh_proj.forward_device(&merged, phase, act)?;
        record(probe, || "merged".into(), &x);
        let x = x.to_dtype(self.forms.residual)?;
        let normed = rms_norm(&x, &layer.attention_norm, self.eps, act)?;
        let mixed = layer
            .attention
            .forward(&normed, rows, index, phase, act, probe)?;
        let x = add_residual(&x, &mixed)?;
        let normed = rms_norm(&x, &layer.ffn_norm, self.eps, act)?;
        let gate_up = layer.w13.forward_device(&normed, phase, act)?;
        let width = layer.intermediate;
        let activations = swiglu(
            &gate_up.narrow(1, 0, width)?,
            &gate_up.narrow(1, width, width)?,
            act,
        )?;
        record(probe, || "ffn_activations".into(), &activations);
        let down = layer.w2.forward_device(&activations, phase, act)?;
        record(probe, || "ffn_out".into(), &down);
        let x = add_residual(&x, &down)?;
        record(probe, || "residual".into(), &x);
        rms_norm(&x, &layer.final_norm, self.eps, act)
    }

    /// [`Self::layer`] in its own dtype through the fused kernels: the
    /// norms read the residual once with the add before them
    /// (`kernels::residual_norm`), attention's q/k norm and RoPE take
    /// `positions` (`[N]` U32, uploaded once per forward), and the SwiGLU
    /// reads the packed w1/w3 projection in one launch. Matches it to
    /// rounding.
    fn layer_fused(
        &self,
        index: usize,
        e: &Tensor,
        h: &Tensor,
        rows: &mut [DeviceRow<'_>],
        phase: FishDrama3KernelPhase,
        positions: &Tensor,
    ) -> Result<Tensor> {
        let layer = &self.layers[index];
        let act = layer.act;
        let e = residual_norm::rms_norm(e, &layer.enorm, self.eps, act)?;
        let h = residual_norm::rms_norm(h, &layer.hnorm, self.eps, act)?;
        let merged = Tensor::cat(&[&e, &h], 1)?;
        let x = layer
            .eh_proj
            .forward_device(&merged, phase, act)?
            .to_dtype(self.forms.residual)?;
        let normed = residual_norm::rms_norm(&x, &layer.attention_norm, self.eps, act)?;
        let mixed = layer
            .attention
            .forward_fused(&normed, rows, index, phase, act, positions)?;
        let (x, normed) = residual_norm::add_rms_norm(&x, &mixed, &layer.ffn_norm, self.eps, act)?;
        let gate_up = layer.w13.forward_device(&normed, phase, act)?;
        let activations = crate::kernels::swiglu::swiglu(&gate_up, act)?;
        let down = layer.w2.forward_device(&activations, phase, act)?;
        Ok(residual_norm::add_rms_norm(&x, &down, &layer.final_norm, self.eps, act)?.1)
    }

    /// Layer `index` of a span or step forward: fused when the load enabled
    /// it, `positions` uploaded on first use.
    fn forward_layer(
        &self,
        index: usize,
        e: &Tensor,
        h: &Tensor,
        rows: &mut [DeviceRow<'_>],
        phase: FishDrama3KernelPhase,
        positions: &mut Option<Tensor>,
    ) -> Result<Tensor> {
        if !self.fused {
            return self.layer(index, e, h, rows, phase, None, &mut None);
        }
        let positions = match positions {
            Some(positions) => positions,
            None => positions.insert(super::positions_tensor(rows, h.device())?),
        };
        self.layer_fused(index, e, h, rows, phase, positions)
    }

    /// Head `index`'s F32 logits `[N, V]` of layer outputs `h`: an F16
    /// projection widened on Metal, a BF16 projection summed into F32 on
    /// CUDA (BF16 logits would lose two decimal digits).
    pub(crate) fn head_logits(
        &self,
        index: usize,
        h: &Tensor,
        phase: FishDrama3KernelPhase,
        act: Option<DType>,
    ) -> Result<Tensor> {
        let act = act.unwrap_or(self.head_act);
        if self.forms.residency == FishDrama3Residency::Cuda && act == self.forms.half {
            return self.heads[index].forward_device_f32(h, phase);
        }
        Ok(self.heads[index]
            .forward_device(h, phase, act)?
            .to_dtype(DType::F32)?)
    }

    /// Layer `slot`'s injections: rows `codes` (`[N]` U32, clamped) of its
    /// table where `labelled`, else zero, added to `base` (layer 0's
    /// `extra`).
    pub(crate) fn inject(
        &self,
        slot: usize,
        codes: &Tensor,
        labelled: &Tensor,
        base: Option<&Tensor>,
    ) -> Result<Tensor> {
        let rows = FishDrama3Embedding::gather(&self.layers[slot].embedding, codes)?;
        let rows = match base {
            Some(base) => (base + &rows)?,
            None => rows,
        };
        let fallback = match base {
            Some(base) => base.clone(),
            None => rows.zeros_like()?,
        };
        let mask = labelled.broadcast_as(rows.shape())?;
        Ok(mask.where_cond(&rows, &fallback)?)
    }

    /// `[N, 1]` U8: whether each row's label (or sampled) main is semantic.
    pub(crate) fn labelled(flags: &[bool], device: &Device) -> Result<Tensor> {
        super::count_uploads(1);
        Ok(Tensor::from_vec(
            flags.iter().map(|&flag| u8::from(flag)).collect::<Vec<_>>(),
            (flags.len(), 1),
            device,
        )?)
    }

    /// `values` clamped to the table, `[N]` U32 on `device`.
    pub(crate) fn codes(
        &self,
        values: impl Iterator<Item = u32>,
        device: &Device,
    ) -> Result<Tensor> {
        let values = values
            .map(|code| code.min(self.vocab as u32 - 1))
            .collect::<Vec<_>>();
        let n = values.len();
        super::count_uploads(1);
        Ok(Tensor::from_vec(values, n, device)?)
    }

    /// Teacher-forced prefill: `hidden` `[n, dim]` F32, `inputs` and
    /// `labels` `n` frames each (`width` values a frame). Returns every
    /// layer's output.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn span(
        &self,
        embedding: &FishDrama3Embedding,
        hidden: &Tensor,
        inputs: &[u32],
        labels: &[u32],
        width: usize,
        semantic: impl Fn(u32) -> bool,
        rows: &mut [DeviceRow<'_>],
    ) -> Result<Vec<Tensor>> {
        let device = hidden.device().clone();
        let extra = embedding.extra_device(inputs)?;
        let labelled = Self::labelled(
            &labels
                .chunks_exact(width)
                .map(|label| semantic(label[0]))
                .collect::<Vec<_>>(),
            &device,
        )?;
        let mut x = hidden.clone();
        let mut outputs = Vec::with_capacity(self.layers.len());
        let mut positions = None;
        for slot in 0..self.layers.len() {
            let column = 1 + slot.saturating_sub(self.shared_layers);
            let codes = self.codes(labels.chunks_exact(width).map(|l| l[column]), &device)?;
            let e = self.inject(slot, &codes, &labelled, (slot == 0).then_some(&extra))?;
            x = self.forward_layer(
                slot,
                &e,
                &x,
                rows,
                FishDrama3KernelPhase::Span,
                &mut positions,
            )?;
            outputs.push(x.clone());
        }
        Ok(outputs)
    }

    /// A decode step over one row per request: `hidden` `[m, dim]` F32,
    /// `inputs` the rows' input frames (row-major), `first` each row's code 0
    /// (`[m]` U32, clamped) and `labelled` whether it injects (`[m, 1]` U8).
    /// Nothing is read back: device codes stay on the device for the caller
    /// to read with whatever else its frame reads.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn step(
        &self,
        embedding: &FishDrama3Embedding,
        hidden: &Tensor,
        inputs: &[u32],
        first: &Tensor,
        labelled: &Tensor,
        rows: &mut [DeviceRow<'_>],
        mut codes: DeviceStepCodes<'_>,
    ) -> Result<DeviceTowerStep> {
        let m = first.dim(0)?;
        let extra = embedding.extra_device(inputs)?;
        let mut x = hidden.clone();
        let mut outputs = Vec::with_capacity(self.layers.len());
        // The code each residual layer injects: c0, then each head's pick.
        let mut injected = first.clone();
        let mut picks = Vec::with_capacity(self.heads.len());
        let mut checks = Vec::new();
        let mut host_codes = vec![Vec::with_capacity(self.heads.len()); m];
        let mut positions = None;
        for slot in 0..self.layers.len() {
            let source = if slot < self.shared_layers {
                first
            } else {
                &injected
            };
            let e = self.inject(slot, source, labelled, (slot == 0).then_some(&extra))?;
            x = self.forward_layer(
                slot,
                &e,
                &x,
                rows,
                FishDrama3KernelPhase::Step,
                &mut positions,
            )?;
            outputs.push(x.clone());
            if slot < self.shared_layers {
                continue;
            }
            let head = slot - self.shared_layers;
            let logits = self.head_logits(head, &x, FishDrama3KernelPhase::Step, None)?;
            injected = match &mut codes {
                DeviceStepCodes::Device { policies, noise } if self.fused => {
                    let draws = race_rows(&logits, &noise.get(head)?, 0, policies)?;
                    checks.push(draws.narrow(1, 1, 1)?.flatten_all()?);
                    let pick = draws.narrow(1, 0, 1)?.flatten_all()?;
                    picks.push(pick.clone());
                    pick
                }
                DeviceStepCodes::Device { policies, noise } => {
                    let mut row_picks = Vec::with_capacity(m);
                    for (r, policy) in policies.iter().enumerate() {
                        let cap = policy.top_k.unwrap_or(self.vocab).min(self.vocab);
                        let ranked = rank(&logits.get(r)?, 0, cap)?;
                        let draw = race(&ranked, policy, &noise.get(head)?.get(r)?)?;
                        checks.push(draw.non_finite);
                        row_picks.push(draw.pick);
                    }
                    let pick = Tensor::cat(&row_picks, 0)?;
                    picks.push(pick.clone());
                    pick
                }
                DeviceStepCodes::Host(sample) => {
                    let values = sample(head, &logits)?;
                    if values.len() != m {
                        return Err(Error::InferenceError(format!(
                            "{} codes for {m} tower rows",
                            values.len()
                        )));
                    }
                    for (row, &code) in host_codes.iter_mut().zip(&values) {
                        row.push(code);
                    }
                    self.codes(values.into_iter(), hidden.device())?
                }
            };
        }
        let device = match codes {
            DeviceStepCodes::Device { .. } => Some(DeviceCodes {
                picks: Tensor::stack(&picks, 0)?,
                checks: Tensor::cat(&checks, 0)?,
            }),
            DeviceStepCodes::Host(_) => None,
        };
        Ok(DeviceTowerStep {
            host_codes,
            device,
            outputs,
        })
    }
}

/// A device decode step's result.
pub(crate) struct DeviceTowerStep {
    /// Each row's codes 1.. under [`DeviceStepCodes::Host`].
    pub(crate) host_codes: Vec<Vec<u32>>,
    /// The codes under [`DeviceStepCodes::Device`], not yet read back.
    pub(crate) device: Option<DeviceCodes>,
    /// Every layer's output.
    pub(crate) outputs: Vec<Tensor>,
}

/// A step's device-drawn codes.
pub(crate) struct DeviceCodes {
    /// `[heads, m]` U32: head `j`'s pick for each row (codebook `j + 1`).
    pub(crate) picks: Tensor,
    /// `[heads * m]` U32: each draw's non-finite logits.
    pub(crate) checks: Tensor,
}

impl DeviceCodes {
    /// Every row's codes 1.. from the host values of `[picks ; checks]`;
    /// a non-finite head is an error.
    pub(crate) fn rows(values: &[u32], m: usize) -> Result<Vec<Vec<u32>>> {
        let (codes, non_finite) = values.split_at(values.len() / 2);
        if let Some(count) = non_finite.iter().find(|count| **count > 0) {
            return Err(Error::InferenceError(format!(
                "Fish Audio Drama 3 codebook head logits are non-finite: {count} values"
            )));
        }
        let mut rows = vec![Vec::with_capacity(codes.len() / m.max(1)); m];
        for head_codes in codes.chunks_exact(m) {
            for (row, &code) in rows.iter_mut().zip(head_codes) {
                row.push(code);
            }
        }
        Ok(rows)
    }

    /// `[picks ; checks]` flattened, for one readback.
    pub(crate) fn packed(&self) -> Result<Tensor> {
        Ok(Tensor::cat(&[&self.picks.flatten_all()?, &self.checks], 0)?)
    }
}
