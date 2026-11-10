//! Parakeet's retained TDT decode, many joint steps per readback.
//!
//! The eager retained step reads every joint step's picks back to the host
//! (blank steps included), applies the transitions there and steps the
//! predictor for the rows that emitted. On CUDA, once the transducer select
//! kernel passed its self-check, a dispatch instead runs
//! [`DEVICE_DECODE_STEPS`] joint steps with no host sync: each step gathers
//! every row's encoder frame from the rows' stacked encoder windows by a
//! device index, runs the joint for every row, applies the picks and the
//! TDT transition in the kernel (`kernels::transducer_greedy`), and steps
//! the predictor for every row, keeping the state of the rows that did not
//! emit. One readback per dispatch returns the cursors and the emissions.
//!
//! A row stops stepping once it finishes, emits its quota or leaves its
//! window; the engine step keeps dispatching until a row emits or finishes,
//! as the eager step keeps evaluating the joint. A row's transitions are the
//! eager ones (`tdt_step`, the same picks), but its predictor runs in a GEMM
//! over every row rather than the emitting rows only, which can round the
//! last bits differently on CUDA. The host form of a dispatch
//! ([`HostSelect`]) runs the same tensor ops with the select step on the host;
//! the CPU tests run it.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, PoisonError};

use candle_core::{DType, Device, DeviceLocation, Tensor};

use super::{
    EncoderWindowing, ParakeetAsrModel, ParakeetPredictorBatchState,
    ParakeetPreparedEncoderArtifact, ParakeetRetainedDecodeBatchRow, ParakeetRetainedDecodeStep,
    TdtTokenFrame,
};
use crate::error::{Error, Result};
use crate::kernels::transducer_greedy::{
    self, limit_field, state_field, SelectRules, SelectTensors, TdtRules, LIMIT_FIELDS,
    NO_DEVICE_FRAME, STATE_FIELDS,
};

/// Joint steps per device dispatch.
pub(super) const DEVICE_DECODE_STEPS: usize = transducer_greedy::DEVICE_DISPATCH_STEPS;

/// Whether dispatches run the select step on the host instead of the
/// device: the CPU tests of the dispatch's tensor work.
#[cfg(test)]
fn host_select() -> bool {
    super::tests::DEVICE_DECODE_ON_THE_HOST.with(std::cell::Cell::get)
}

#[cfg(not(test))]
fn host_select() -> bool {
    false
}

/// Whether dispatches pad their rows to a graph width without graphs: the
/// CPU tests of the spare rows a graphed dispatch carries.
#[cfg(test)]
fn pad_on_the_host() -> bool {
    super::tests::PAD_DISPATCH_ON_THE_HOST.with(std::cell::Cell::get)
}

#[cfg(not(test))]
fn pad_on_the_host() -> bool {
    false
}

/// Devices past this ordinal keep the readback per joint step.
const MAX_DEVICES: usize = 16;
const UNCHECKED: u8 = 0;
const ON: u8 = 1;
const OFF: u8 = 2;
/// Each device's verdict of the whole dispatch against its host form
/// ([`ParakeetAsrModel::check_device_decode`]), and why it is off.
static DISPATCH_VERDICT: [AtomicU8; MAX_DEVICES] =
    [const { AtomicU8::new(UNCHECKED) }; MAX_DEVICES];
static DISPATCH_REASON: Mutex<[Option<String>; MAX_DEVICES]> =
    Mutex::new([const { None }; MAX_DEVICES]);

fn dispatch_verdict(device: &Device) -> Option<(usize, &'static AtomicU8)> {
    let DeviceLocation::Cuda { gpu_id } = device.location() else {
        return None;
    };
    DISPATCH_VERDICT
        .get(gpu_id)
        .map(|verdict| (gpu_id, verdict))
}

/// Whether two tensors hold the same values bit for bit.
fn same_bits(a: &Tensor, b: &Tensor) -> Result<bool> {
    if a.dims() != b.dims() || a.dtype() != b.dtype() {
        return Ok(false);
    }
    let bits = |t: &Tensor| -> Result<Vec<u32>> {
        Ok(t.to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    };
    Ok(bits(a)? == bits(b)?)
}

impl ParakeetAsrModel {
    /// Whether the retained decode runs its dispatches of many joint steps:
    /// on CUDA once the select kernel passed its self-check there.
    pub(super) fn device_decode_on(&self) -> bool {
        let device = self.network.predictor.embed.device();
        host_select()
            || (device.is_cuda()
                && transducer_greedy::eager_reason(device).is_none()
                && dispatch_verdict(device)
                    .is_some_and(|(_, verdict)| verdict.load(Ordering::Acquire) == ON))
    }

    /// At load, on CUDA with the select kernel on: one dispatch of two
    /// synthetic rows on the device against the same dispatch with the
    /// select step on the host, which must agree bit for bit (tokens,
    /// frames, cursors, predictor state and output); on any difference the
    /// decode keeps one readback per joint step on this device, with the
    /// reason in the diagnostics. Checks each device once.
    pub(super) fn check_device_decode(&self) {
        let device = self.network.predictor.embed.device();
        if !device.is_cuda() || transducer_greedy::eager_reason(device).is_some() {
            return;
        }
        let Some((gpu_id, verdict)) = dispatch_verdict(device) else {
            return;
        };
        if verdict.load(Ordering::Acquire) != UNCHECKED {
            return;
        }
        let reason = match self.dispatch_self_check() {
            Ok(None) => {
                tracing::info!(
                    "Parakeet device decode matched its host form on cuda:{gpu_id}; \
                     {DEVICE_DECODE_STEPS} joint steps per readback"
                );
                None
            }
            Ok(Some(failure)) => Some(format!("dispatch self-check failed: {failure}")),
            Err(error) => Some(format!("dispatch self-check could not run: {error}")),
        };
        if let Some(reason) = &reason {
            tracing::warn!("Parakeet device decode is off on cuda:{gpu_id}: {reason}");
        }
        let passed = reason.is_none();
        DISPATCH_REASON
            .lock()
            .unwrap_or_else(PoisonError::into_inner)[gpu_id] = reason;
        verdict.store(if passed { ON } else { OFF }, Ordering::Release);
    }

    /// One dispatch of two random rows (24 and 9 frames, a quota of four)
    /// on the device and in its host form; `None` when they agree.
    pub(super) fn dispatch_self_check(&self) -> Result<Option<String>> {
        let device = self.network.predictor.embed.device();
        let dim = self.network.encoder.geometry.dim;
        let artifact = |len: usize, seed: u64| -> Result<ParakeetPreparedEncoderArtifact> {
            Ok(ParakeetPreparedEncoderArtifact {
                encoded: crate::kernels::numeric_check::xorshift_tensor(
                    &[1, len, dim],
                    seed,
                    2.0,
                    device,
                )?,
                encoded_len: len,
                input_samples: len * 1280,
                input_sample_rate: 16_000,
                resampled_samples: len * 1280,
                feature_frames: len * 8,
                language: None,
                windowing: EncoderWindowing::V3,
                encoder_windows: 1,
                log_mel_ms: 0.0,
                encode_ms: 0.0,
            })
        };
        let artifacts = [artifact(24, 0x7D7_0001)?, artifact(9, 0x7D7_0002)?];
        let mut on_device = [self.start_retained_decode()?, self.start_retained_decode()?];
        let mut on_host = on_device.clone();
        for (states, host) in [(&mut on_device, false), (&mut on_host, true)] {
            let mut rows = states
                .iter_mut()
                .zip(&artifacts)
                .map(|(state, artifact)| ParakeetRetainedDecodeBatchRow {
                    artifact,
                    state,
                    max_tokens: usize::MAX,
                    quota: 4,
                })
                .collect::<Vec<_>>();
            self.retained_decode_dispatch(&mut rows, DEVICE_DECODE_STEPS, host)?;
        }
        for (row, (device_state, host_state)) in on_device.iter().zip(&on_host).enumerate() {
            if device_state.token_ids != host_state.token_ids
                || device_state.token_frames != host_state.token_frames
                || device_state.cursor != host_state.cursor
            {
                return Ok(Some(format!("row {row}: tokens or cursor differ")));
            }
            let (d, h) = (&device_state.predictor, &host_state.predictor);
            for (name, a, b) in [
                ("h0", &d.h0, &h.h0),
                ("c0", &d.c0, &h.c0),
                ("h1", &d.h1, &h.h1),
                ("c1", &d.c1, &h.c1),
                (
                    "output",
                    &device_state.predictor_out,
                    &host_state.predictor_out,
                ),
            ] {
                if !same_bits(a, b)? {
                    return Ok(Some(format!("row {row}: predictor {name} differs")));
                }
            }
        }
        Ok(None)
    }

    /// Tokens per row and engine step the retained decode asks the
    /// scheduler for: the device decode quantum while the dispatches run,
    /// else one (the eager step emits at most one).
    pub(crate) fn decode_quantum(&self) -> usize {
        if self.device_decode_on() {
            transducer_greedy::device_decode_quantum()
        } else {
            1
        }
    }

    /// [`ParakeetAsrModel::retained_decode_step_batch`] through dispatches of
    /// [`DEVICE_DECODE_STEPS`] joint steps, until a row emits or finishes.
    pub(super) fn retained_decode_step_batch_device(
        &self,
        rows: &mut [ParakeetRetainedDecodeBatchRow<'_>],
    ) -> Result<Vec<ParakeetRetainedDecodeStep>> {
        loop {
            let emitted =
                self.retained_decode_dispatch(rows, DEVICE_DECODE_STEPS, host_select())?;
            let steps = self.retained_decode_steps(rows)?;
            if emitted || steps.iter().any(|step| step.finished) {
                return Ok(steps);
            }
        }
    }

    /// One dispatch of up to `steps` joint steps for every row; whether any
    /// row emitted. Updates each row's cursor, tokens, counters and
    /// predictor.
    fn retained_decode_dispatch(
        &self,
        rows: &mut [ParakeetRetainedDecodeBatchRow<'_>],
        steps: usize,
        on_host: bool,
    ) -> Result<bool> {
        let predictor = &self.network.predictor;
        let device = predictor.embed.device().clone();
        let batch = rows.len();
        let window = steps * self.num_durations.saturating_sub(1).max(1);
        let rules = SelectRules::Tdt(TdtRules {
            blank: self.blank_idx,
            num_durations: self.num_durations,
            max_symbols: self.max_symbols,
        });

        // Each row's next frames, stacked: the gather index of row b's frame
        // t is b * window + t - first.
        let mut windows = Vec::with_capacity(batch);
        let mut state = vec![0u32; batch * STATE_FIELDS];
        let mut limits = vec![0u32; batch * LIMIT_FIELDS];
        let mut gather = Vec::with_capacity(batch);
        let fits = |value: usize| {
            u32::try_from(value).map_err(|_| {
                Error::InferenceError("Parakeet device decode sizes exceed u32".into())
            })
        };
        for (b, row) in rows.iter().enumerate() {
            let cursor = row.state.cursor;
            let len = row.artifact.encoded_len;
            if cursor.t >= len {
                return Err(Error::InferenceError(
                    "Parakeet retained cursor exceeded encoder artifact".into(),
                ));
            }
            let frames = window.min(len - cursor.t);
            let rows_window = row.artifact.encoded.narrow(1, cursor.t, frames)?;
            windows.push(rows_window.pad_with_zeros(1, 0, window - frames)?);
            let s = &mut state[b * STATE_FIELDS..(b + 1) * STATE_FIELDS];
            s[state_field::T] = fits(cursor.t)?;
            s[state_field::LAST] = if cursor.last_emit_t == transducer_greedy::NO_FRAME {
                NO_DEVICE_FRAME
            } else {
                fits(cursor.last_emit_t)?
            };
            s[state_field::COUNT] = fits(cursor.emit_count_at_t)?;
            s[state_field::GUARD] = fits(cursor.guard_steps)?;
            s[state_field::TOKENS] = fits(row.state.token_ids.len())?;
            s[state_field::LIVE] = 1;
            let guard_limit = len
                .saturating_mul(self.max_symbols)
                .saturating_mul(8)
                .max(512);
            let l = &mut limits[b * LIMIT_FIELDS..(b + 1) * LIMIT_FIELDS];
            l[limit_field::LEN] = fits(len)?;
            l[limit_field::GUARD] = fits(guard_limit)?;
            // A budget past u32 is unbounded for any row the decode can
            // emit (the dispatch self-check's rows take `usize::MAX`).
            l[limit_field::BUDGET] = fits(row.max_tokens.clamp(1, u32::MAX as usize))?;
            l[limit_field::QUOTA] = fits(row.quota.max(1))?;
            l[limit_field::FIRST] = fits(cursor.t)?;
            l[limit_field::WINDOW] = fits(window)?;
            gather.push(fits(b * window)?);
        }
        let refs = windows.iter().collect::<Vec<_>>();
        let dim = windows[0].dim(2)?;
        let frames = Tensor::cat(&refs, 0)?.reshape((batch * window, dim))?;
        drop(windows);
        let predictors = rows
            .iter()
            .map(|row| &row.state.predictor)
            .collect::<Vec<_>>();
        let cohort = ParakeetPredictorBatchState::gather_rows(&predictors)?;
        let outputs = rows
            .iter()
            .map(|row| &row.state.predictor_out)
            .collect::<Vec<_>>();
        let predictor_out = Tensor::cat(&outputs, 0)?;

        // On CUDA through the graph of the dispatch's padded width, while the
        // graphs are on: spare rows never step.
        let graphs = self
            .decode_graphs
            .as_ref()
            .ok()
            .filter(|graphs| !on_host && device.is_cuda() && graphs.is_on());
        let width = match graphs.and(transducer_greedy::graph_width(batch)) {
            Some(width) => width,
            None if pad_on_the_host() => transducer_greedy::graph_width(batch).unwrap_or(batch),
            None => batch,
        };
        let graphs = graphs.filter(|_| transducer_greedy::graph_width(batch).is_some());
        let spare = width - batch;
        state.resize(width * STATE_FIELDS, 0);
        limits.resize(width * LIMIT_FIELDS, 0);
        gather.resize(width, 0);
        let pad = |tensor: &Tensor, rows: usize| -> Result<Tensor> {
            Ok(tensor.pad_with_zeros(0, 0, rows)?)
        };
        let upload = |values: &[u32]| Tensor::new(values, &device);
        let inputs = [
            pad(&frames, spare * window)?,
            upload(&gather)?,
            upload(&state)?,
            upload(&limits)?,
            pad(&cohort.h0, spare)?,
            pad(&cohort.c0, spare)?,
            pad(&cohort.h1, spare)?,
            pad(&cohort.c1, spare)?,
            pad(&predictor_out, spare)?,
        ];
        drop(frames);
        let body = |inputs: &[Tensor]| self.dispatch_body(inputs, steps, rules, on_host);
        let outputs = match graphs {
            Some(graphs) => {
                let bytes = |t: &Tensor| (t.elem_count() * t.dtype().size_in_bytes()) as u64;
                let retained = 2 * inputs.iter().map(bytes).sum::<u64>()
                    + (width * steps * 3 * std::mem::size_of::<u32>()) as u64;
                graphs.run((width, steps), batch, retained, &inputs, &body)?
            }
            None => body(&inputs)?,
        };
        let [state, emits, h0, c0, h1, c1, out] = outputs.as_slice() else {
            return Err(Error::InferenceError(
                "Parakeet device decode dispatch returned the wrong outputs".into(),
            ));
        };
        let host_state = state.to_vec1::<u32>()?;
        let host_emits = emits.to_vec1::<u32>()?;
        if !on_host {
            crate::models::shared::telemetry::record_host_read(
                DType::U32,
                host_state.len() + host_emits.len(),
            );
        }
        let cohort = ParakeetPredictorBatchState {
            h0: h0.narrow(0, 0, batch)?,
            c0: c0.narrow(0, 0, batch)?,
            h1: h1.narrow(0, 0, batch)?,
            c1: c1.narrow(0, 0, batch)?,
        };
        let predictor_out = out.narrow(0, 0, batch)?;
        let host = (host_state, host_emits);

        let mut scattered = rows
            .iter()
            .map(|row| row.state.predictor.clone())
            .collect::<Vec<_>>();
        let mut targets = scattered.iter_mut().collect::<Vec<_>>();
        cohort.scatter_rows(&mut targets)?;
        drop(targets);
        let mut any_emitted = false;
        for (b, (row, predictor)) in rows.iter_mut().zip(scattered).enumerate() {
            let s = &host.0[b * STATE_FIELDS..(b + 1) * STATE_FIELDS];
            let cursor = &mut row.state.cursor;
            cursor.t = s[state_field::T] as usize;
            cursor.last_emit_t = if s[state_field::LAST] == NO_DEVICE_FRAME {
                transducer_greedy::NO_FRAME
            } else {
                s[state_field::LAST] as usize
            };
            cursor.emit_count_at_t = s[state_field::COUNT] as usize;
            cursor.guard_steps = s[state_field::GUARD] as usize;
            let emitted = s[state_field::EMITTED] as usize;
            for slot in 0..emitted.min(steps) {
                let e = &host.1[(b * steps + slot) * 3..][..3];
                row.state.token_ids.push(e[0] as usize);
                row.state.token_frames.push(TdtTokenFrame {
                    frame: e[1] as usize,
                    duration: e[2] as usize,
                });
            }
            any_emitted |= emitted > 0;
            let counters = &mut row.state.counters;
            let joint = s[state_field::JOINT] as usize;
            counters.tdt_joint_steps = counters.tdt_joint_steps.saturating_add(joint);
            counters.tdt_blank_steps = counters
                .tdt_blank_steps
                .saturating_add(s[state_field::BLANKS] as usize);
            // One readback per dispatch on the device; one per step on the
            // host.
            let reads = if on_host {
                joint
            } else {
                usize::from(joint > 0)
            };
            counters.device_argmax_scalar_reads =
                counters.device_argmax_scalar_reads.saturating_add(reads);
            row.state.predictor = predictor;
            row.state.predictor_out = predictor_out.narrow(0, b, 1)?.contiguous()?;
        }
        Ok(any_emitted)
    }

    /// A dispatch's tensor work, `steps` joint steps with no host sync on
    /// the device (a CUDA graph's body): `inputs` are the stacked encoder
    /// windows `[B * W, D]`, the gather indices, state and limits, the
    /// predictor's `h0, c0, h1, c1` and output; the outputs are the state,
    /// the emissions and the predictor's state and output. The select step
    /// writes fresh copies of the buffers, so the inputs stay as given.
    fn dispatch_body(
        &self,
        inputs: &[Tensor],
        steps: usize,
        rules: SelectRules,
        on_host: bool,
    ) -> Result<Vec<Tensor>> {
        let [frames, gather, state, limits, h0, c0, h1, c1, predictor_out] = inputs else {
            return Err(Error::InferenceError(format!(
                "Parakeet device decode dispatch received {} inputs, expected 9",
                inputs.len()
            )));
        };
        let rows = gather.dim(0)?;
        let device = frames.device();
        let mut buffers = SelectTensors {
            state: state.copy()?,
            emits: Tensor::zeros(rows * steps * 3, DType::U32, device)?,
            labels: Tensor::zeros(rows, DType::U32, device)?,
            mask: Tensor::zeros(rows, DType::U32, device)?,
            gather: gather.copy()?,
        };
        let mut cohort = ParakeetPredictorBatchState {
            h0: h0.clone(),
            c0: c0.clone(),
            h1: h1.clone(),
            c1: c1.clone(),
        };
        let mut predictor_out = predictor_out.clone();
        for _ in 0..steps {
            let encoded = frames.index_select(&buffers.gather, 0)?;
            let logits = self.joint_batch_rows(&encoded, &predictor_out)?;
            buffers = transducer_greedy::select_step_tensors(
                &logits, limits, buffers, steps, rules, on_host,
            )?;
            predictor_out =
                self.predictor_step_masked(&buffers.labels, &buffers.mask, &mut cohort)?;
        }
        Ok(vec![
            buffers.state,
            buffers.emits,
            cohort.h0,
            cohort.c0,
            cohort.h1,
            cohort.c1,
            predictor_out,
        ])
    }

    /// One predictor step for every row of `cohort`: the rows whose `mask`
    /// (`[B]` U32) is set take their `labels` (`[B]` U32), the others keep
    /// their state. Returns the predictor output `[B, 1, H]`.
    fn predictor_step_masked(
        &self,
        labels: &Tensor,
        mask: &Tensor,
        cohort: &mut ParakeetPredictorBatchState,
    ) -> Result<Tensor> {
        let predictor = &self.network.predictor;
        let x = predictor.embed.index_select(labels, 0)?;
        let (h0, c0) = predictor.lstm_l0.step(&x, &cohort.h0, &cohort.c0)?;
        let (h1, c1) = predictor.lstm_l1.step(&h0, &cohort.h1, &cohort.c1)?;
        let keep = |new: &Tensor, old: &Tensor| -> Result<Tensor> {
            let predicate = mask.reshape((mask.dim(0)?, 1))?.broadcast_as(new.shape())?;
            Ok(predicate.where_cond(new, old)?)
        };
        cohort.h0 = keep(&h0, &cohort.h0)?;
        cohort.c0 = keep(&c0, &cohort.c0)?;
        cohort.h1 = keep(&h1, &cohort.h1)?;
        cohort.c1 = keep(&c1, &cohort.c1)?;
        Ok(cohort.h1.unsqueeze(1)?)
    }
}

/// Whether `device` runs the device decode and why not, for diagnostics.
pub(super) fn device_decode_diagnostics(device: &Device) -> serde_json::Value {
    let dispatch_reason = dispatch_verdict(device).and_then(|(gpu_id, verdict)| {
        match verdict.load(Ordering::Acquire) {
            ON => None,
            OFF => DISPATCH_REASON
                .lock()
                .unwrap_or_else(PoisonError::into_inner)[gpu_id]
                .clone(),
            _ => Some("dispatch self-check not run yet".to_string()),
        }
    });
    let reason = transducer_greedy::eager_reason(device).or(dispatch_reason);
    serde_json::json!({
        "path": if reason.is_none() { "device" } else { "readback per joint step" },
        "reason": reason,
        "steps_per_dispatch": DEVICE_DECODE_STEPS,
        "decode_quantum": if reason.is_none() {
            transducer_greedy::device_decode_quantum()
        } else {
            1
        },
    })
}
