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

use candle_core::{DType, Device, Tensor};

use super::{
    ParakeetAsrModel, ParakeetPredictorBatchState, ParakeetRetainedDecodeBatchRow,
    ParakeetRetainedDecodeStep, TdtTokenFrame,
};
use crate::error::{Error, Result};
use crate::kernels::transducer_greedy::{
    self, limit_field, select_step_reference, state_field, DeviceSelect, SelectBuffers,
    SelectRules, TdtRules, LIMIT_FIELDS, NO_DEVICE_FRAME, STATE_FIELDS,
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

/// The select step's buffers on the host, for [`host_select`].
struct HostSelect {
    state: Vec<u32>,
    limits: Vec<u32>,
    emits: Vec<u32>,
    labels: Vec<u32>,
    mask: Vec<u32>,
    gather: Vec<u32>,
}

impl ParakeetAsrModel {
    /// Whether the retained decode runs its dispatches of many joint steps:
    /// on CUDA once the select kernel passed its self-check there.
    pub(super) fn device_decode_on(&self) -> bool {
        let device = self.network.predictor.embed.device();
        host_select() || (device.is_cuda() && transducer_greedy::eager_reason(device).is_none())
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
            let emitted = self.retained_decode_dispatch(rows, DEVICE_DECODE_STEPS)?;
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
            l[limit_field::BUDGET] = fits(row.max_tokens.max(1))?;
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
        let mut cohort = ParakeetPredictorBatchState::gather_rows(&predictors)?;
        let outputs = rows
            .iter()
            .map(|row| &row.state.predictor_out)
            .collect::<Vec<_>>();
        let mut predictor_out = Tensor::cat(&outputs, 0)?;

        let on_host = host_select();
        let mut host = HostSelect {
            emits: vec![0u32; batch * steps * 3],
            labels: vec![0u32; batch],
            mask: vec![0u32; batch],
            state,
            limits,
            gather,
        };
        let upload = |values: &[u32]| Tensor::new(values, &device);
        let buffers = if on_host {
            None
        } else {
            Some([
                upload(&host.state)?,
                upload(&host.limits)?,
                upload(&host.emits)?,
                upload(&host.labels)?,
                upload(&host.mask)?,
                upload(&host.gather)?,
            ])
        };
        let mut gather_index = upload(&host.gather)?;
        for _ in 0..steps {
            let encoded = frames.index_select(&gather_index, 0)?;
            let logits = self.joint_batch_rows(&encoded, &predictor_out)?;
            let (labels, mask) = match &buffers {
                Some([state, limits, emits, labels, mask, gather]) => {
                    transducer_greedy::select_step(
                        &logits.to_dtype(DType::F32)?,
                        &DeviceSelect {
                            state,
                            limits,
                            emits,
                            labels,
                            mask,
                            gather,
                        },
                        steps,
                        rules,
                    )?;
                    gather_index = gather.clone();
                    (labels.clone(), mask.clone())
                }
                None => {
                    let logits = logits
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let width = logits.len() / batch;
                    select_step_reference(
                        SelectBuffers {
                            logits: &logits,
                            width,
                            state: &mut host.state,
                            limits: &host.limits,
                            emits: &mut host.emits,
                            max_emits: steps,
                            labels: &mut host.labels,
                            mask: &mut host.mask,
                            gather: &mut host.gather,
                        },
                        rules,
                    );
                    gather_index = upload(&host.gather)?;
                    (upload(&host.labels)?, upload(&host.mask)?)
                }
            };
            predictor_out = self.predictor_step_masked(&labels, &mask, &mut cohort)?;
        }
        if let Some([state, _, emits, ..]) = &buffers {
            host.state = state.to_vec1::<u32>()?;
            host.emits = emits.flatten_all()?.to_vec1::<u32>()?;
            crate::models::shared::telemetry::record_host_read(
                DType::U32,
                host.state.len() + host.emits.len(),
            );
        }

        let mut scattered = rows
            .iter()
            .map(|row| row.state.predictor.clone())
            .collect::<Vec<_>>();
        let mut targets = scattered.iter_mut().collect::<Vec<_>>();
        cohort.scatter_rows(&mut targets)?;
        drop(targets);
        let mut any_emitted = false;
        for (b, (row, predictor)) in rows.iter_mut().zip(scattered).enumerate() {
            let s = &host.state[b * STATE_FIELDS..(b + 1) * STATE_FIELDS];
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
                let e = &host.emits[(b * steps + slot) * 3..][..3];
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

/// Whether `device` runs the device decode, for diagnostics.
pub(super) fn device_decode_diagnostics(device: &Device) -> serde_json::Value {
    let reason = transducer_greedy::eager_reason(device);
    serde_json::json!({
        "path": if reason.is_none() { "device" } else { "readback per joint step" },
        "reason": reason,
        "steps_per_dispatch": DEVICE_DECODE_STEPS,
    })
}
