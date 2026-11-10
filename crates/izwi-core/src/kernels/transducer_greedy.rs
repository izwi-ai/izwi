//! Greedy transducer decoding, one joint step per row: the TDT rule of
//! Parakeet's retained decode and the RNNT rule of Nemotron ASR's.
//!
//! A joint step reads one encoder frame for each row, picks a label (and for
//! TDT a duration) from the joint's logits, and moves the row's cursor. The
//! eager decode loops apply these transitions on the host after reading the
//! picks back; the device decode applies the same transitions in a kernel,
//! so both are one definition: [`tdt_step`] and [`rnnt_step`].
//!
//! Picks follow [`crate::kernels::greedy`]: the largest logit, ties to the
//! lowest id, NaN never picked.

use candle_core::{Device, Result, Tensor};

/// Sentinel for "no frame yet" in [`TdtRow::last_emit_t`].
pub const NO_FRAME: usize = usize::MAX;

/// The fixed rules of a TDT decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TdtRules {
    pub blank: usize,
    /// Duration logits after the token logits; durations are `0..n`.
    pub num_durations: usize,
    /// Labels a frame may emit before the cursor is pushed past it.
    pub max_symbols: usize,
}

/// One row's TDT cursor between joint steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TdtRow {
    /// The encoder frame the next joint step reads.
    pub t: usize,
    /// The frame of the row's last emission ([`NO_FRAME`] before any).
    pub last_emit_t: usize,
    /// Labels emitted at `last_emit_t` so far.
    pub emit_count_at_t: usize,
    /// Joint steps taken.
    pub guard_steps: usize,
}

impl TdtRow {
    /// A row before its first joint step.
    pub const START: Self = Self {
        t: 0,
        last_emit_t: NO_FRAME,
        emit_count_at_t: 0,
        guard_steps: 0,
    };
}

/// What one joint step did to a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TdtStep {
    /// A blank: the cursor moved by the duration (at least one frame).
    Blank,
    /// A label emitted at `frame` with its (clamped) duration index; the
    /// predictor consumes it.
    Emit {
        label: usize,
        frame: usize,
        duration: usize,
    },
    /// A label past the frame's `max_symbols`: dropped, and the cursor moved
    /// one frame past the frame the step read.
    Suppressed,
}

/// Parakeet's retained TDT transition for one joint step that picked
/// `label` and duration index `duration`: the cursor moves by the duration
/// (a blank of duration 0 by one frame), a label counts against its frame's
/// `max_symbols`, and one past them is dropped with the cursor pushed one
/// frame on.
pub fn tdt_step(row: &mut TdtRow, label: usize, duration: usize, rules: TdtRules) -> TdtStep {
    let duration = duration.min(rules.num_durations.saturating_sub(1));
    let mut jump = duration;
    if label == rules.blank && jump == 0 {
        jump = 1;
    }
    let frame = row.t;
    row.t = row.t.saturating_add(jump);
    row.guard_steps = row.guard_steps.saturating_add(1);
    if label == rules.blank {
        return TdtStep::Blank;
    }
    if frame == row.last_emit_t {
        row.emit_count_at_t = row.emit_count_at_t.saturating_add(1);
    } else {
        row.last_emit_t = frame;
        row.emit_count_at_t = 1;
    }
    if row.emit_count_at_t > rules.max_symbols {
        row.t = frame.saturating_add(1);
        return TdtStep::Suppressed;
    }
    TdtStep::Emit {
        label,
        frame,
        duration,
    }
}

/// The fixed rules of an RNNT decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RnntRules {
    pub blank: usize,
    /// Labels a frame may emit before the cursor moves on.
    pub max_symbols: usize,
}

/// One row's RNNT cursor between joint steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RnntRow {
    /// The encoder frame the next joint step reads.
    pub t: usize,
    /// Labels emitted on frame `t` so far.
    pub symbols: usize,
}

/// What one RNNT joint step did to a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RnntStep {
    /// A blank: the cursor moved to the next frame.
    Blank,
    /// A label emitted; `frame_full` when it was the frame's last allowed
    /// label and the cursor moved on.
    Emit { label: usize, frame_full: bool },
    /// A label past the blank: the joint's output is malformed.
    Invalid { label: usize },
}

/// Nemotron's RNNT transition for one joint step that picked `label`: a
/// blank moves to the next frame; a label is emitted, and the frame's
/// `max_symbols`-th moves the cursor on.
pub fn rnnt_step(row: &mut RnntRow, label: usize, rules: RnntRules) -> RnntStep {
    if label > rules.blank {
        return RnntStep::Invalid { label };
    }
    if label == rules.blank {
        row.t = row.t.saturating_add(1);
        row.symbols = 0;
        return RnntStep::Blank;
    }
    row.symbols = row.symbols.saturating_add(1);
    let frame_full = row.symbols >= rules.max_symbols;
    if frame_full {
        row.t = row.t.saturating_add(1);
        row.symbols = 0;
    }
    RnntStep::Emit { label, frame_full }
}

/// Fields of a row's device decode state (`[B, STATE_FIELDS]` U32): the
/// cursor, the tokens so far, whether it still steps, and this dispatch's
/// emissions and counters.
pub mod state_field {
    pub const T: usize = 0;
    /// The last emission's frame ([`super::NO_DEVICE_FRAME`] before any).
    pub const LAST: usize = 1;
    pub const COUNT: usize = 2;
    pub const GUARD: usize = 3;
    pub const TOKENS: usize = 4;
    pub const LIVE: usize = 5;
    pub const EMITTED: usize = 6;
    pub const JOINT: usize = 7;
    pub const BLANKS: usize = 8;
    pub const FULL: usize = 9;
    pub const SYMBOLS: usize = 10;
    pub const ERROR: usize = 11;
}
pub const STATE_FIELDS: usize = 12;

/// Fields of a row's limits (`[B, LIMIT_FIELDS]` U32).
pub mod limit_field {
    /// Encoded frames.
    pub const LEN: usize = 0;
    /// Joint steps before the row stops (`u32::MAX`: none).
    pub const GUARD: usize = 1;
    /// The request's token budget.
    pub const BUDGET: usize = 2;
    /// Emissions this dispatch may make.
    pub const QUOTA: usize = 3;
    /// The first frame of the row's stacked encoder window.
    pub const FIRST: usize = 4;
    /// Frames in the stacked window.
    pub const WINDOW: usize = 5;
}
pub const LIMIT_FIELDS: usize = 6;

/// [`NO_FRAME`] in the device state.
pub const NO_DEVICE_FRAME: u32 = u32::MAX;

/// Which transition a select step applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectRules {
    /// Labels in `[0, blank]`, durations after them.
    Tdt(TdtRules),
    /// Labels over the whole row.
    Rnnt(RnntRules),
}

/// The device select step's buffers on the host: `logits` `[B, width]`,
/// `state` `[B, STATE_FIELDS]`, `limits` `[B, LIMIT_FIELDS]`, `emits`
/// `[B, max_emits, 3]`, and the predictor's `labels`, `mask` and the next
/// `gather` index, `[B]` each.
pub struct SelectBuffers<'a> {
    pub logits: &'a [f32],
    pub width: usize,
    pub state: &'a mut [u32],
    pub limits: &'a [u32],
    pub emits: &'a mut [u32],
    pub max_emits: usize,
    pub labels: &'a mut [u32],
    pub mask: &'a mut [u32],
    pub gather: &'a mut [u32],
}

/// One select step on the host, as `kernels/cuda/transducer_greedy.cu` runs
/// it on the device: each live row's picks ([`crate::kernels::greedy::host_pick`]),
/// its transition ([`tdt_step`] or [`rnnt_step`]), its emission, and whether
/// it keeps stepping (not finished, under its quota, inside its window).
pub fn select_step_reference(buffers: SelectBuffers<'_>, rules: SelectRules) {
    use crate::kernels::greedy::host_pick;
    use limit_field as lim;
    use state_field as f;
    let rows = buffers.labels.len();
    for b in 0..rows {
        let s = &mut buffers.state[b * STATE_FIELDS..(b + 1) * STATE_FIELDS];
        let l = &buffers.limits[b * LIMIT_FIELDS..(b + 1) * LIMIT_FIELDS];
        buffers.labels[b] = 0;
        buffers.mask[b] = 0;
        if s[f::LIVE] == 0 {
            continue;
        }
        let row = &buffers.logits[b * buffers.width..(b + 1) * buffers.width];
        let frame = s[f::T];
        s[f::JOINT] += 1;
        let (emitted, label, duration) = match rules {
            SelectRules::Tdt(rules) => {
                let token_end = rules.blank + 1;
                let label = host_pick(&row[..token_end]).0 as usize;
                let duration = host_pick(&row[token_end..token_end + rules.num_durations]).0;
                let mut cursor = TdtRow {
                    t: s[f::T] as usize,
                    last_emit_t: if s[f::LAST] == NO_DEVICE_FRAME {
                        NO_FRAME
                    } else {
                        s[f::LAST] as usize
                    },
                    emit_count_at_t: s[f::COUNT] as usize,
                    guard_steps: s[f::GUARD] as usize,
                };
                let step = tdt_step(&mut cursor, label, duration as usize, rules);
                s[f::T] = cursor.t as u32;
                s[f::LAST] = if cursor.last_emit_t == NO_FRAME {
                    NO_DEVICE_FRAME
                } else {
                    cursor.last_emit_t as u32
                };
                s[f::COUNT] = cursor.emit_count_at_t as u32;
                s[f::GUARD] = cursor.guard_steps as u32;
                match step {
                    TdtStep::Blank => {
                        s[f::BLANKS] += 1;
                        (false, 0, 0)
                    }
                    TdtStep::Suppressed => (false, 0, 0),
                    TdtStep::Emit {
                        label, duration, ..
                    } => (true, label as u32, duration as u32),
                }
            }
            SelectRules::Rnnt(rules) => {
                let label = host_pick(row).0 as usize;
                let mut cursor = RnntRow {
                    t: s[f::T] as usize,
                    symbols: s[f::SYMBOLS] as usize,
                };
                let step = rnnt_step(&mut cursor, label, rules);
                s[f::T] = cursor.t as u32;
                s[f::SYMBOLS] = cursor.symbols as u32;
                match step {
                    RnntStep::Blank => {
                        s[f::BLANKS] += 1;
                        (false, 0, 0)
                    }
                    RnntStep::Emit { label, frame_full } => {
                        if frame_full {
                            s[f::FULL] += 1;
                        }
                        (true, label as u32, 0)
                    }
                    RnntStep::Invalid { .. } => {
                        s[f::ERROR] = 1;
                        s[f::LIVE] = 0;
                        continue;
                    }
                }
            }
        };
        if emitted {
            let slot = s[f::EMITTED] as usize;
            if slot < buffers.max_emits {
                let e = &mut buffers.emits[(b * buffers.max_emits + slot) * 3..][..3];
                e.copy_from_slice(&[label, frame, duration]);
            }
            s[f::EMITTED] += 1;
            s[f::TOKENS] += 1;
            buffers.labels[b] = label;
            buffers.mask[b] = 1;
        }
        let max_symbols = match rules {
            SelectRules::Tdt(rules) => rules.max_symbols,
            SelectRules::Rnnt(rules) => rules.max_symbols,
        } as u64;
        let t = s[f::T];
        let tokens = u64::from(s[f::TOKENS]);
        let exhausted = t >= l[lim::LEN] || tokens >= u64::from(l[lim::LEN]) * max_symbols;
        let budget = tokens >= u64::from(l[lim::BUDGET]);
        let guard = s[f::GUARD] >= l[lim::GUARD];
        let quota = s[f::EMITTED] >= l[lim::QUOTA];
        let outside = t < l[lim::FIRST] || t - l[lim::FIRST] >= l[lim::WINDOW];
        if exhausted || budget || guard || quota || outside {
            s[f::LIVE] = 0;
        } else {
            buffers.gather[b] = b as u32 * l[lim::WINDOW] + (t - l[lim::FIRST]);
        }
    }
}

/// Kill switch for the device decode loops (any false boolean, e.g. `0`),
/// read once per process.
pub const CUDA_TRANSDUCER_DECODE_ENV: &str = "IZWI_CUDA_TRANSDUCER_DECODE";

/// The decode quantum (tokens per row and engine step) of the device
/// decode loops: any positive integer, capped at 16 (a dispatch's joint
/// steps); 8 unset.
pub const CUDA_TRANSDUCER_QUANTUM_ENV: &str = "IZWI_CUDA_TRANSDUCER_QUANTUM";

/// Joint steps per device dispatch, and the largest decode quantum.
pub const DEVICE_DISPATCH_STEPS: usize = 16;

/// The decode quantum a device decode loop asks the scheduler for
/// ([`CUDA_TRANSDUCER_QUANTUM_ENV`]), read once.
pub fn device_decode_quantum() -> usize {
    static QUANTUM: crate::env::ReadOnce<usize> = crate::env::ReadOnce::new(|| {
        crate::env::value(CUDA_TRANSDUCER_QUANTUM_ENV)
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|quantum| *quantum > 0)
            .unwrap_or(8)
            .min(DEVICE_DISPATCH_STEPS)
    });
    QUANTUM.get()
}

/// The select step's device buffers: `state` `[B, STATE_FIELDS]`, `limits`
/// `[B, LIMIT_FIELDS]`, `emits` `[B, max_emits, 3]`, `labels`, `mask` and
/// `gather` `[B]`, all U32, contiguous, owned by the caller's dispatch: the
/// step updates them in place.
pub struct DeviceSelect<'a> {
    pub state: &'a Tensor,
    pub limits: &'a Tensor,
    pub emits: &'a Tensor,
    pub labels: &'a Tensor,
    pub mask: &'a Tensor,
    pub gather: &'a Tensor,
}

/// Whether the device decode loops run on `device`. The first call for a
/// CUDA device runs the select kernel's self-check against
/// [`select_step_reference`] (bit for bit) and logs the verdict; later calls
/// read it. False off CUDA, without the `cuda` feature, or with
/// `IZWI_CUDA_TRANSDUCER_DECODE=0`.
pub fn prepare_cuda_kernel(device: &Device) -> bool {
    #[cfg(feature = "cuda")]
    {
        cuda_impl::enabled(device)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = device;
        false
    }
}

/// Why `device` decodes with a readback per joint step, or `None` when the
/// device loops run (no check runs here).
pub fn eager_reason(device: &Device) -> Option<String> {
    #[cfg(feature = "cuda")]
    {
        cuda_impl::eager_reason(device)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = device;
        Some("not a CUDA build".to_string())
    }
}

/// One select step over F32 `logits` (`[B, width]`) on the device, in
/// place on `buffers`. Errors off CUDA.
pub fn select_step(
    logits: &Tensor,
    buffers: &DeviceSelect<'_>,
    max_emits: usize,
    rules: SelectRules,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        cuda_impl::run(logits, buffers, max_emits, rules)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (logits, buffers, max_emits, rules);
        candle_core::bail!("the device transducer select step needs the cuda feature")
    }
}

/// A dispatch's select-step buffers as tensors: `state` `[B * STATE_FIELDS]`,
/// `emits` `[B * max_emits * 3]`, `labels`, `mask` and `gather` `[B]`, U32.
#[derive(Debug, Clone)]
pub struct SelectTensors {
    pub state: Tensor,
    pub emits: Tensor,
    pub labels: Tensor,
    pub mask: Tensor,
    pub gather: Tensor,
}

/// One select step over `logits` with `limits` (`[B * LIMIT_FIELDS]` U32):
/// on CUDA, unless `on_host`, the kernel updates `buffers` in place (no host
/// sync, so a CUDA graph can capture it) and returns them; otherwise the
/// host reference reads them back and returns new tensors on their device
/// (the dispatch's host form, and every non-CUDA device).
pub fn select_step_tensors(
    logits: &Tensor,
    limits: &Tensor,
    buffers: SelectTensors,
    max_emits: usize,
    rules: SelectRules,
    on_host: bool,
) -> Result<SelectTensors> {
    let logits = logits.to_dtype(candle_core::DType::F32)?;
    if logits.device().is_cuda() && !on_host {
        select_step(
            &logits,
            &DeviceSelect {
                state: &buffers.state,
                limits,
                emits: &buffers.emits,
                labels: &buffers.labels,
                mask: &buffers.mask,
                gather: &buffers.gather,
            },
            max_emits,
            rules,
        )?;
        return Ok(buffers);
    }
    let read = |t: &Tensor| t.flatten_all()?.to_vec1::<u32>();
    let values = logits.flatten_all()?.to_vec1::<f32>()?;
    let (rows, width) = logits.dims2()?;
    let (mut state, mut emits, mut labels, mut mask, mut gather) = (
        read(&buffers.state)?,
        read(&buffers.emits)?,
        vec![0; rows],
        vec![0; rows],
        read(&buffers.gather)?,
    );
    select_step_reference(
        SelectBuffers {
            logits: &values,
            width,
            state: &mut state,
            limits: &read(limits)?,
            emits: &mut emits,
            max_emits,
            labels: &mut labels,
            mask: &mut mask,
            gather: &mut gather,
        },
        rules,
    );
    let device = logits.device();
    Ok(SelectTensors {
        state: Tensor::new(state.as_slice(), device)?,
        emits: Tensor::new(emits.as_slice(), device)?,
        labels: Tensor::new(labels.as_slice(), device)?,
        mask: Tensor::new(mask.as_slice(), device)?,
        gather: Tensor::new(gather.as_slice(), device)?,
    })
}

/// Kill switch for the transducer dispatches' CUDA graphs (a boolean; on
/// when unset), also off with `cuda.decode_graphs=off`.
pub const CUDA_TRANSDUCER_GRAPHS_ENV: &str = "IZWI_CUDA_TRANSDUCER_GRAPHS";

/// Batch widths a dispatch's graph is captured at: a dispatch pads its rows
/// up to the next with rows that never step.
pub const GRAPH_WIDTHS: [usize; 6] = [1, 2, 4, 8, 16, 32];

/// The graph width a dispatch of `rows` rows runs at, `None` past 32.
pub fn graph_width(rows: usize) -> Option<usize> {
    GRAPH_WIDTHS.into_iter().find(|width| *width >= rows.max(1))
}

const MIB: u64 = 1024 * 1024;

/// What a model's dispatch graphs may hold: one graph per (width, steps),
/// each retaining its stable inputs (the rows' stacked encoder windows, up
/// to 8 MiB at 32 rows) and outputs; the pool holds a dispatch's working
/// set (per step the joint's logits and the predictor's activations, about
/// 1.6 MiB at 32 rows of Parakeet's 8,198 classes, 16 steps).
pub const TRANSDUCER_GRAPH_LIMITS: crate::kernels::cuda::shape_graphs::ShapeGraphLimits =
    crate::kernels::cuda::shape_graphs::ShapeGraphLimits {
        max_graphs: 12,
        retained_bytes: 64 * MIB,
        pool_bytes: 192 * MIB,
    };

/// Whether the dispatch graphs' switches leave them on.
pub fn transducer_graphs_requested() -> bool {
    crate::env::flag_or(CUDA_TRANSDUCER_DECODE_ENV, true)
        && crate::env::flag_or(CUDA_TRANSDUCER_GRAPHS_ENV, true)
        && !crate::kernels::cuda::stretch_graphs::decode_graphs_switched_off()
}

/// Device bytes a transducer load on `backend` prices for its dispatch
/// graphs, which allocate on first use: zero off CUDA or with a switch off.
pub fn transducer_graph_device_bytes(backend: crate::backends::BackendKind) -> u64 {
    if backend == crate::backends::BackendKind::Cuda && transducer_graphs_requested() {
        TRANSDUCER_GRAPH_LIMITS.device_bytes()
    } else {
        0
    }
}

/// A model's dispatch graphs, keyed by (graph width, joint steps).
pub struct TransducerGraphs {
    graphs: crate::kernels::cuda::shape_graphs::ShapeGraphs<(usize, usize)>,
    dispatches: std::sync::atomic::AtomicU64,
    spare_rows: std::sync::atomic::AtomicU64,
}

impl TransducerGraphs {
    /// The graphs on `device`, or why there are none: CUDA only, with the
    /// switches on.
    pub fn resolve(label: &'static str, device: &Device) -> std::result::Result<Self, String> {
        if !crate::env::flag_or(CUDA_TRANSDUCER_DECODE_ENV, true) {
            return Err(format!("{CUDA_TRANSDUCER_DECODE_ENV} is off"));
        }
        if !crate::env::flag_or(CUDA_TRANSDUCER_GRAPHS_ENV, true) {
            return Err(format!("{CUDA_TRANSDUCER_GRAPHS_ENV} is off"));
        }
        if crate::kernels::cuda::stretch_graphs::decode_graphs_switched_off() {
            return Err("cuda.decode_graphs=off".into());
        }
        Ok(Self {
            graphs: crate::kernels::cuda::shape_graphs::ShapeGraphs::resolve(
                label,
                device,
                TRANSDUCER_GRAPH_LIMITS,
            )?,
            dispatches: std::sync::atomic::AtomicU64::new(0),
            spare_rows: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Whether dispatches still go through the graphs (no capture or
    /// replay check has failed).
    pub fn is_on(&self) -> bool {
        self.graphs.is_on()
    }

    /// `body` over `inputs` through the graph of `(width, steps)`, for a
    /// dispatch of `rows` real rows; `retained` as [`ShapeGraphs::run`]
    /// takes it. A capture is checked bit for bit against the eager body on
    /// its first two dispatches; a mismatch or CUDA error turns the graphs
    /// off for the process and runs the body eagerly.
    ///
    /// [`ShapeGraphs::run`]: crate::kernels::cuda::shape_graphs::ShapeGraphs::run
    pub fn run(
        &self,
        (width, steps): (usize, usize),
        rows: usize,
        retained: u64,
        inputs: &[Tensor],
        body: &dyn Fn(&[Tensor]) -> crate::error::Result<Vec<Tensor>>,
    ) -> crate::error::Result<Vec<Tensor>> {
        use std::sync::atomic::Ordering::Relaxed;
        self.dispatches.fetch_add(1, Relaxed);
        self.spare_rows
            .fetch_add(width.saturating_sub(rows) as u64, Relaxed);
        self.graphs.run((width, steps), retained, inputs, body)
    }

    pub fn diagnostics(&self) -> serde_json::Value {
        use std::sync::atomic::Ordering::Relaxed;
        serde_json::json!({
            "dispatches": self.dispatches.load(Relaxed),
            "spare_rows": self.spare_rows.load(Relaxed),
            "graphs": self.graphs.diagnostics(),
        })
    }
}

/// A model's dispatch graphs for its diagnostics: their counters, or why
/// there are none.
pub fn transducer_graphs_diagnostics(
    graphs: &std::result::Result<TransducerGraphs, String>,
) -> serde_json::Value {
    match graphs {
        Ok(graphs) => graphs.diagnostics(),
        Err(reason) => serde_json::json!({ "off": reason }),
    }
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use std::sync::atomic::{AtomicU8, Ordering};
    use std::sync::{Mutex, PoisonError};

    use candle_core::cuda_backend::cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
    use candle_core::cuda_backend::{CudaDType, WrapErr};
    use candle_core::{DType, Device, DeviceLocation, Result, Storage, Tensor};

    use super::{
        DeviceSelect, RnntRules, SelectRules, TdtRules, CUDA_TRANSDUCER_DECODE_ENV, LIMIT_FIELDS,
        STATE_FIELDS,
    };

    const MODULE: &str = "izwi_transducer_greedy";
    const MAX_DEVICES: usize = 16;
    const UNCHECKED: u8 = 0;
    const ON: u8 = 1;
    const OFF: u8 = 2;
    static VERDICT: [AtomicU8; MAX_DEVICES] = [const { AtomicU8::new(UNCHECKED) }; MAX_DEVICES];
    static REASON: Mutex<[Option<String>; MAX_DEVICES]> = Mutex::new([const { None }; MAX_DEVICES]);
    static CHECKING: Mutex<()> = Mutex::new(());

    fn switched_on() -> bool {
        static SWITCH: crate::env::ReadOnce<bool> =
            crate::env::ReadOnce::new(|| crate::env::flag_or(CUDA_TRANSDUCER_DECODE_ENV, true));
        SWITCH.get()
    }

    fn verdict(device: &Device) -> Option<(usize, &'static AtomicU8)> {
        let DeviceLocation::Cuda { gpu_id } = device.location() else {
            return None;
        };
        VERDICT.get(gpu_id).map(|verdict| (gpu_id, verdict))
    }

    pub(super) fn enabled(device: &Device) -> bool {
        if !switched_on() {
            return false;
        }
        let Some((gpu_id, verdict)) = verdict(device) else {
            return false;
        };
        match verdict.load(Ordering::Acquire) {
            ON => true,
            OFF => false,
            _ => decide(device, gpu_id, verdict),
        }
    }

    pub(super) fn eager_reason(device: &Device) -> Option<String> {
        if !switched_on() {
            return Some(format!("{CUDA_TRANSDUCER_DECODE_ENV} is off"));
        }
        let Some((gpu_id, verdict)) = verdict(device) else {
            return Some("not a checked CUDA device".to_string());
        };
        match verdict.load(Ordering::Acquire) {
            ON => None,
            OFF => Some(
                REASON.lock().unwrap_or_else(PoisonError::into_inner)[gpu_id]
                    .clone()
                    .unwrap_or_else(|| "self-check failed".to_string()),
            ),
            _ => Some("self-check not run yet".to_string()),
        }
    }

    fn decide(device: &Device, gpu_id: usize, verdict: &AtomicU8) -> bool {
        let _checking = CHECKING.lock().unwrap_or_else(PoisonError::into_inner);
        match verdict.load(Ordering::Acquire) {
            ON => return true,
            OFF => return false,
            _ => {}
        }
        let reason = match self_check(device) {
            Ok(None) => {
                tracing::info!(
                    "transducer select CUDA kernel matched its host reference on cuda:{gpu_id}; \
                     the TDT and RNNT decodes run many joint steps per readback"
                );
                None
            }
            Ok(Some(failure)) => {
                tracing::warn!(
                    "transducer select CUDA kernel failed its self-check on cuda:{gpu_id} \
                     ({failure}); keeping one readback per joint step"
                );
                Some(format!("self-check failed: {failure}"))
            }
            Err(error) => {
                tracing::warn!(
                    %error,
                    "transducer select CUDA kernel self-check could not run on cuda:{gpu_id}; \
                     keeping one readback per joint step"
                );
                Some(format!("self-check could not run: {error}"))
            }
        };
        let passed = reason.is_none();
        REASON.lock().unwrap_or_else(PoisonError::into_inner)[gpu_id] = reason;
        verdict.store(if passed { ON } else { OFF }, Ordering::Release);
        passed
    }

    /// A contiguous buffer's storage and start offset.
    fn slice(tensor: &Tensor) -> Result<(impl std::ops::Deref<Target = Storage> + '_, usize)> {
        let (storage, layout) = tensor.storage_and_layout();
        if !layout.is_contiguous() {
            candle_core::bail!("the transducer select step needs contiguous buffers")
        }
        let start = layout.start_offset();
        Ok((storage, start))
    }

    fn cuda_slice<T: CudaDType>(storage: &Storage) -> Result<&CudaSlice<T>> {
        let Storage::Cuda(cuda) = storage else {
            candle_core::bail!("the transducer select step needs CUDA tensors")
        };
        cuda.as_cuda_slice::<T>()
    }

    pub(super) fn run(
        logits: &Tensor,
        buffers: &DeviceSelect<'_>,
        max_emits: usize,
        rules: SelectRules,
    ) -> Result<()> {
        let (rows, width) = logits.dims2()?;
        let u32_buffers = [
            (buffers.state, rows * STATE_FIELDS),
            (buffers.limits, rows * LIMIT_FIELDS),
            (buffers.emits, rows * max_emits * 3),
            (buffers.labels, rows),
            (buffers.mask, rows),
            (buffers.gather, rows),
        ];
        if logits.dtype() != DType::F32
            || u32_buffers
                .iter()
                .any(|(tensor, len)| tensor.dtype() != DType::U32 || tensor.elem_count() != *len)
        {
            candle_core::bail!("the transducer select step got buffers of the wrong dtype or size")
        }
        let (rnnt, blank, token_end, durations, max_symbols) = match rules {
            SelectRules::Tdt(TdtRules {
                blank,
                num_durations,
                max_symbols,
            }) => {
                if width < blank + 1 + num_durations {
                    candle_core::bail!("TDT logits of width {width} lack the duration logits")
                }
                (0i32, blank, blank + 1, num_durations, max_symbols)
            }
            SelectRules::Rnnt(RnntRules { blank, max_symbols }) => {
                (1i32, blank, width, 0, max_symbols)
            }
        };
        let ints = [width, max_emits, blank, token_end, durations, max_symbols]
            .map(|value| i32::try_from(value).ok());
        let (Some(rows_u32), true) = (u32::try_from(rows).ok(), ints.iter().all(Option::is_some))
        else {
            candle_core::bail!("the transducer select step's sizes exceed i32")
        };
        let [width, max_emits, blank, token_end, durations, max_symbols] =
            ints.map(|value| value.unwrap_or_default());
        let logits = logits.contiguous()?;
        let device = logits.device().as_cuda_device()?;
        let function = device.get_or_load_custom_func(
            "izwi_transducer_select_f32",
            MODULE,
            crate::kernels::cuda::cuda_ptx_transducer_greedy(),
        )?;
        let (l_storage, l_start) = slice(&logits)?;
        let guards = [
            slice(buffers.state)?,
            slice(buffers.limits)?,
            slice(buffers.emits)?,
            slice(buffers.labels)?,
            slice(buffers.mask)?,
            slice(buffers.gather)?,
        ];
        let ls = cuda_slice::<f32>(&l_storage)?.slice(l_start..);
        let views = guards
            .iter()
            .map(|(storage, start)| Ok(cuda_slice::<u32>(storage)?.slice(*start..)))
            .collect::<Result<Vec<_>>>()?;
        let mut builder = function.builder();
        builder.arg(&ls);
        candle_core::builder_arg!(builder, width);
        builder.arg(&views[0]);
        builder.arg(&views[1]);
        builder.arg(&views[2]);
        candle_core::builder_arg!(builder, max_emits);
        builder.arg(&views[3]);
        builder.arg(&views[4]);
        builder.arg(&views[5]);
        candle_core::builder_arg!(builder, rnnt, blank, token_end, durations, max_symbols);
        let config = LaunchConfig {
            grid_dim: (rows_u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: argument order and types match the kernel; each warp reads
        // its row's `width` logits and state and writes its own state,
        // emission slots (at most `max_emits`), label, mask and gather index.
        unsafe { builder.launch(config) }.w()?;
        Ok(())
    }

    /// The kernel against [`select_step_reference`] over
    /// [`super::check_cases`], two steps each. `None` when every output
    /// buffer matches bit for bit.
    fn self_check(device: &Device) -> Result<Option<String>> {
        for case in super::check_cases() {
            let rows = case.limits.len() / LIMIT_FIELDS;
            let mut want = case.expected_inputs();
            let upload = |values: &[u32]| Tensor::new(values, device);
            let dev = [
                upload(&case.state)?,
                upload(&case.limits)?,
                upload(&want.emits)?,
                upload(&want.labels)?,
                upload(&want.mask)?,
                upload(&want.gather)?,
            ];
            let logits = Tensor::from_vec(case.logits.clone(), (rows, case.width), device)?;
            for _ in 0..2 {
                case.reference_step(&mut want);
                run(
                    &logits,
                    &DeviceSelect {
                        state: &dev[0],
                        limits: &dev[1],
                        emits: &dev[2],
                        labels: &dev[3],
                        mask: &dev[4],
                        gather: &dev[5],
                    },
                    case.max_emits,
                    case.rules,
                )?;
            }
            let got = |t: &Tensor| t.flatten_all()?.to_vec1::<u32>();
            for (name, device_values, host_values) in [
                ("state", got(&dev[0])?, &want.state),
                ("emits", got(&dev[2])?, &want.emits),
                ("labels", got(&dev[3])?, &want.labels),
                ("mask", got(&dev[4])?, &want.mask),
                ("gather", got(&dev[5])?, &want.gather),
            ] {
                if &device_values != host_values {
                    return Ok(Some(format!(
                        "{:?}: {name} {device_values:?} != {host_values:?}",
                        case.rules
                    )));
                }
            }
        }
        Ok(None)
    }
}

/// One select-step check: its rules, logits and starting buffers.
#[cfg(any(test, feature = "cuda"))]
struct CheckCase {
    rules: SelectRules,
    width: usize,
    max_emits: usize,
    logits: Vec<f32>,
    state: Vec<u32>,
    limits: Vec<u32>,
}

/// A check's evolving output buffers.
#[cfg(any(test, feature = "cuda"))]
struct CheckBuffers {
    state: Vec<u32>,
    emits: Vec<u32>,
    labels: Vec<u32>,
    mask: Vec<u32>,
    gather: Vec<u32>,
}

#[cfg(any(test, feature = "cuda"))]
impl CheckCase {
    /// The buffers before the first step: zero emissions, and labels, mask
    /// and gather indices at 9 so the steps must write them.
    fn expected_inputs(&self) -> CheckBuffers {
        let rows = self.limits.len() / LIMIT_FIELDS;
        CheckBuffers {
            state: self.state.clone(),
            emits: vec![0; rows * self.max_emits * 3],
            labels: vec![9; rows],
            mask: vec![9; rows],
            gather: vec![9; rows],
        }
    }

    fn reference_step(&self, buffers: &mut CheckBuffers) {
        select_step_reference(
            SelectBuffers {
                logits: &self.logits,
                width: self.width,
                state: &mut buffers.state,
                limits: &self.limits,
                emits: &mut buffers.emits,
                max_emits: self.max_emits,
                labels: &mut buffers.labels,
                mask: &mut buffers.mask,
                gather: &mut buffers.gather,
            },
            self.rules,
        );
    }
}

/// Select-step checks built to hit every branch: ties (lowest id), NaN,
/// +inf and all -inf logits, blanks, blanks of duration 0, max-symbols
/// suppression, budget, quota, window and joint-step stops, a stopped row,
/// and an RNNT label past the blank (width 9).
#[cfg(any(test, feature = "cuda"))]
fn check_cases() -> Vec<CheckCase> {
    use crate::kernels::numeric_check::xorshift_values;
    let tdt = SelectRules::Tdt(TdtRules {
        blank: 6,
        num_durations: 5,
        max_symbols: 2,
    });
    let rnnt = SelectRules::Rnnt(RnntRules {
        blank: 6,
        max_symbols: 2,
    });
    let rows = 13;
    [(1u64, tdt, 12), (2, rnnt, 7), (3, rnnt, 9), (4, tdt, 12)]
        .into_iter()
        .map(|(seed, rules, width)| {
            let mut logits = xorshift_values(rows * width, seed, 4.0);
            logits[2] = logits[5];
            logits[width + 1] = f32::NAN;
            logits[2 * width + 3] = f32::INFINITY;
            logits[2 * width + 4] = f32::INFINITY;
            for value in &mut logits[3 * width..4 * width] {
                *value = f32::NEG_INFINITY;
            }
            for (b, row) in logits.chunks_exact_mut(width).enumerate().skip(4) {
                if b % 3 == 0 {
                    row[6] = 100.0; // the blank
                }
                if seed == 4 && b % 2 == 0 {
                    row[7] = 50.0; // duration 0
                }
            }
            let mut state = vec![0u32; rows * STATE_FIELDS];
            let mut limits = vec![0u32; rows * LIMIT_FIELDS];
            for b in 0..rows {
                let s = &mut state[b * STATE_FIELDS..(b + 1) * STATE_FIELDS];
                s[state_field::T] = (b % 4) as u32 + 2;
                s[state_field::LAST] = if b % 2 == 0 {
                    NO_DEVICE_FRAME
                } else {
                    s[state_field::T]
                };
                s[state_field::COUNT] = (b % 3) as u32;
                s[state_field::GUARD] = b as u32;
                s[state_field::TOKENS] = (b % 5) as u32;
                s[state_field::LIVE] = u32::from(b != 7);
                s[state_field::SYMBOLS] = (b % 2) as u32;
                let l = &mut limits[b * LIMIT_FIELDS..(b + 1) * LIMIT_FIELDS];
                l[limit_field::LEN] = 4 + (b % 6) as u32;
                l[limit_field::GUARD] = if b == 9 { b as u32 + 1 } else { u32::MAX };
                l[limit_field::BUDGET] = if b == 10 { 5 } else { 100 };
                l[limit_field::QUOTA] = if b == 11 { 1 } else { 3 };
                l[limit_field::FIRST] = 1;
                l[limit_field::WINDOW] = 2 + (b % 4) as u32;
            }
            CheckCase {
                rules,
                width,
                max_emits: 3,
                logits,
                state,
                limits,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TDT: TdtRules = TdtRules {
        blank: 5,
        num_durations: 3,
        max_symbols: 2,
    };

    #[test]
    fn tdt_blank_of_duration_zero_moves_one_frame() {
        let mut row = TdtRow::START;
        assert_eq!(tdt_step(&mut row, 5, 0, TDT), TdtStep::Blank);
        assert_eq!((row.t, row.guard_steps), (1, 1));
        assert_eq!(tdt_step(&mut row, 5, 2, TDT), TdtStep::Blank);
        assert_eq!(row.t, 3);
        // Durations past the last clamp to it.
        assert_eq!(tdt_step(&mut row, 5, 9, TDT), TdtStep::Blank);
        assert_eq!(row.t, 5);
    }

    #[test]
    fn tdt_labels_count_per_frame_and_the_extra_one_is_dropped() {
        let mut row = TdtRow::START;
        let emit = |frame, duration| TdtStep::Emit {
            label: 1,
            frame,
            duration,
        };
        assert_eq!(tdt_step(&mut row, 1, 0, TDT), emit(0, 0));
        assert_eq!(tdt_step(&mut row, 1, 0, TDT), emit(0, 0));
        assert_eq!((row.t, row.emit_count_at_t), (0, 2));
        assert_eq!(tdt_step(&mut row, 1, 0, TDT), TdtStep::Suppressed);
        assert_eq!(row.t, 1);
        // A new frame restarts the count; the duration moves the cursor.
        assert_eq!(tdt_step(&mut row, 1, 2, TDT), emit(1, 2));
        assert_eq!((row.t, row.last_emit_t, row.emit_count_at_t), (3, 1, 1));
        assert_eq!(row.guard_steps, 4);
    }

    /// The check cases reach every branch on the host: emissions, blanks,
    /// suppressions or full frames, stops of each kind and an RNNT error.
    #[test]
    fn check_cases_reach_every_branch() {
        let mut stopped = 0;
        let (mut emitted, mut blanks, mut errors) = (0, 0, 0);
        for case in check_cases() {
            let mut buffers = case.expected_inputs();
            case.reference_step(&mut buffers);
            case.reference_step(&mut buffers);
            for s in buffers.state.as_chunks::<STATE_FIELDS>().0 {
                emitted += s[state_field::EMITTED];
                blanks += s[state_field::BLANKS];
                errors += s[state_field::ERROR];
                stopped += u32::from(s[state_field::LIVE] == 0);
            }
            assert!(buffers.labels.iter().all(|label| *label != 9));
        }
        assert!(emitted > 10 && blanks > 5 && errors > 0 && stopped > 20);
    }

    /// Writes the check cases for the CUDA kernel's emulator, a C++ harness
    /// that runs `kernels/cuda/transducer_greedy.cu` thread by thread and
    /// compares two steps with the host reference bit for bit. One file per
    /// case: 8 i64 (rows, width, max emits, RNNT, blank, token end,
    /// durations, max symbols), the logits as f32, the state and limits,
    /// then the reference's state, emissions, labels, mask and gather after
    /// two steps, as u32, little-endian.
    #[test]
    #[ignore = "writes the CUDA emulator's inputs; set IZWI_TRANSDUCER_EMULATOR_DIR"]
    fn writes_the_cuda_emulator_cases() {
        let Some(dir) = std::env::var_os("IZWI_TRANSDUCER_EMULATOR_DIR") else {
            return;
        };
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir).unwrap();
        for (index, case) in check_cases().into_iter().enumerate() {
            let (rnnt, blank, token_end, durations, max_symbols) = match case.rules {
                SelectRules::Tdt(rules) => (
                    0,
                    rules.blank,
                    rules.blank + 1,
                    rules.num_durations,
                    rules.max_symbols,
                ),
                SelectRules::Rnnt(rules) => (1, rules.blank, case.width, 0, rules.max_symbols),
            };
            let rows = case.limits.len() / LIMIT_FIELDS;
            let mut bytes = Vec::new();
            for value in [
                rows,
                case.width,
                case.max_emits,
                rnnt,
                blank,
                token_end,
                durations,
                max_symbols,
            ] {
                bytes.extend_from_slice(&(value as i64).to_le_bytes());
            }
            for value in &case.logits {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            let mut buffers = case.expected_inputs();
            case.reference_step(&mut buffers);
            case.reference_step(&mut buffers);
            for values in [
                &case.state,
                &case.limits,
                &buffers.state,
                &buffers.emits,
                &buffers.labels,
                &buffers.mask,
                &buffers.gather,
            ] {
                for value in values {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
            }
            std::fs::write(dir.join(format!("case{index:02}.bin")), bytes).unwrap();
        }
    }

    #[test]
    fn rnnt_moves_on_after_a_blank_or_a_full_frame() {
        let rules = RnntRules {
            blank: 4,
            max_symbols: 2,
        };
        let mut row = RnntRow::default();
        assert_eq!(
            rnnt_step(&mut row, 1, rules),
            RnntStep::Emit {
                label: 1,
                frame_full: false
            }
        );
        assert_eq!(
            rnnt_step(&mut row, 2, rules),
            RnntStep::Emit {
                label: 2,
                frame_full: true
            }
        );
        assert_eq!(row, RnntRow { t: 1, symbols: 0 });
        assert_eq!(rnnt_step(&mut row, 4, rules), RnntStep::Blank);
        assert_eq!(row, RnntRow { t: 2, symbols: 0 });
        assert_eq!(
            rnnt_step(&mut row, 5, rules),
            RnntStep::Invalid { label: 5 }
        );
        assert_eq!(row, RnntRow { t: 2, symbols: 0 });
    }

    /// A dispatch pads its rows to the next graph width; past 32 it runs
    /// without a graph.
    #[test]
    fn graph_widths_pad_rows_up_to_32() {
        let widths = [0, 1, 2, 3, 5, 8, 9, 17, 32].map(|rows| graph_width(rows).unwrap());
        assert_eq!(widths, [1, 1, 2, 4, 8, 8, 16, 32, 32]);
        assert_eq!(graph_width(33), None);
    }

    /// The graphs price their memory on CUDA loads with their switches on,
    /// never resolve off CUDA, and say which switch turned them off.
    #[test]
    fn transducer_graphs_price_and_resolve_on_cuda_only() {
        use crate::backends::BackendKind;
        let _env = crate::env_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(transducer_graph_device_bytes(BackendKind::Cpu), 0);
        assert_eq!(transducer_graph_device_bytes(BackendKind::Metal), 0);
        let reason = TransducerGraphs::resolve("test", &Device::Cpu)
            .err()
            .expect("no graphs on CPU");
        if transducer_graphs_requested() {
            assert!(reason.contains("CUDA only"), "{reason}");
            assert_eq!(transducer_graph_device_bytes(BackendKind::Cuda), 256 << 20);
        }
        for (name, value, expected) in [
            (CUDA_TRANSDUCER_GRAPHS_ENV, "0", CUDA_TRANSDUCER_GRAPHS_ENV),
            (CUDA_TRANSDUCER_DECODE_ENV, "0", CUDA_TRANSDUCER_DECODE_ENV),
            ("IZWI_CUDA_DECODE_GRAPHS", "off", "cuda.decode_graphs=off"),
        ] {
            let previous = std::env::var_os(name);
            std::env::set_var(name, value);
            let bytes = transducer_graph_device_bytes(BackendKind::Cuda);
            let reason = TransducerGraphs::resolve("test", &Device::Cpu).err();
            std::env::remove_var(name);
            if let Some(previous) = previous {
                std::env::set_var(name, previous);
            }
            assert_eq!(bytes, 0, "{name}");
            assert!(reason.expect("off").contains(expected), "{name}");
        }
        assert_eq!(
            transducer_graphs_diagnostics(&Err("test".into()))["off"],
            "test"
        );
    }

    /// The functional select step on the host returns the reference's
    /// buffers as new tensors.
    #[test]
    fn select_step_tensors_on_the_host_is_the_reference() {
        let case = check_cases().remove(0);
        let rows = case.limits.len() / LIMIT_FIELDS;
        let mut want = case.expected_inputs();
        case.reference_step(&mut want);
        let cpu = &Device::Cpu;
        let tensors = SelectTensors {
            state: Tensor::new(case.state.as_slice(), cpu).unwrap(),
            emits: Tensor::zeros(rows * case.max_emits * 3, candle_core::DType::U32, cpu).unwrap(),
            labels: Tensor::zeros(rows, candle_core::DType::U32, cpu).unwrap(),
            mask: Tensor::zeros(rows, candle_core::DType::U32, cpu).unwrap(),
            gather: Tensor::new(vec![9u32; rows].as_slice(), cpu).unwrap(),
        };
        let logits = Tensor::from_vec(case.logits.clone(), (rows, case.width), cpu).unwrap();
        let limits = Tensor::new(case.limits.as_slice(), cpu).unwrap();
        let got = select_step_tensors(&logits, &limits, tensors, case.max_emits, case.rules, false)
            .unwrap();
        let read = |t: &Tensor| t.to_vec1::<u32>().unwrap();
        assert_eq!(read(&got.state), want.state);
        assert_eq!(read(&got.emits), want.emits);
        assert_eq!(read(&got.labels), want.labels);
        assert_eq!(read(&got.mask), want.mask);
        assert_eq!(read(&got.gather), want.gather);
    }

    /// On a CUDA device the graphs resolve with the switches on, and a
    /// graphed select step matches the eager one bit for bit.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_transducer_graphs_replay_the_select_step() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        if !transducer_graphs_requested() || !prepare_cuda_kernel(&device) {
            return;
        }
        let graphs = TransducerGraphs::resolve("test", &device).unwrap();
        let case = check_cases().remove(0);
        let rows = case.limits.len() / LIMIT_FIELDS;
        let upload = |values: &[u32]| Tensor::new(values, &device).unwrap();
        let inputs = [
            Tensor::from_vec(case.logits.clone(), (rows, case.width), &device).unwrap(),
            upload(&case.state),
            upload(&case.limits),
        ];
        let body = |inputs: &[Tensor]| -> crate::error::Result<Vec<Tensor>> {
            let [logits, state, limits] = inputs else {
                unreachable!()
            };
            let mut buffers = SelectTensors {
                state: state.copy()?,
                emits: Tensor::zeros(
                    rows * case.max_emits * 3,
                    candle_core::DType::U32,
                    logits.device(),
                )?,
                labels: Tensor::zeros(rows, candle_core::DType::U32, logits.device())?,
                mask: Tensor::zeros(rows, candle_core::DType::U32, logits.device())?,
                gather: Tensor::zeros(rows, candle_core::DType::U32, logits.device())?,
            };
            for _ in 0..2 {
                buffers = select_step_tensors(
                    logits,
                    limits,
                    buffers,
                    case.max_emits,
                    case.rules,
                    false,
                )?;
            }
            Ok(vec![buffers.state, buffers.emits])
        };
        let eager = body(&inputs).unwrap();
        for _ in 0..3 {
            let graphed = graphs
                .run((rows, 2), rows, 1 << 20, &inputs, &body)
                .unwrap();
            for (a, b) in graphed.iter().zip(&eager) {
                assert_eq!(a.to_vec1::<u32>().unwrap(), b.to_vec1::<u32>().unwrap());
            }
        }
        assert!(graphs.is_on());
    }
}
