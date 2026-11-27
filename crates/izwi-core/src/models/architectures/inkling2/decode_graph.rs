//! Whole-step CUDA graphs of an Inkling 2 decode step: one token of each
//! row, from the token embeddings to the logits, captured once per padded
//! width (`decode_step::WIDTHS`) and replayed.
//!
//! The graph body is the batched decode of [`Inkling2Trunk::forward_rows`]
//! with every per-step value on the device:
//! - each cache (the main one and, in the split layout, the sliding-window
//!   one) has a U32 step buffer (`decode_step::DecodeStepLayout`): every
//!   row's flat KV slot and the block table it attends; the rows' keys and
//!   values are written by device slot (`decode_step::write_kv_rows`) and
//!   the fused attention (`kernels::paged_bias_sink`) reads its tables from
//!   the buffer;
//! - the token ids, the log scale of the rows' positions (made on the host
//!   in F64 as the eager step makes it) and every application's convolution
//!   histories, stacked over the rows, are inputs; the next histories are
//!   outputs beside the logits;
//! - spare rows decode token 0 into no KV slot and attend one token of page
//!   0; every op is row-independent, so they change no real row.
//!
//! Per width, `ShapeGraphs` checks a capture against the eager body bit for
//! bit on its first two steps, and the model compares a width's first
//! graphed step with its standard batched decode
//! (`decode_step::logits_agree`). Any failure turns the graphs off for the
//! process. The graphs need the experts on the device (the host expert path
//! reads activations back) and the drafter off; they are a request
//! ([`STEP_GRAPHS_ENV`]) until a GPU run has checked them.
//!
//! Off CUDA the same body runs eagerly (no capture) for tests: on Metal it
//! is checked against the standard batched decode.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;

use candle_core::{Device, Tensor};
use serde_json::json;

use super::trunk::Inkling2RowsContext;
use crate::backends::BackendKind;
use crate::error::{Error, Result};
use crate::kernels::cuda::decode_step::{write_kv_rows, DecodeStepLayout};
use crate::kernels::cuda::shape_graphs::{ShapeGraphLimits, ShapeGraphs};
use crate::kernels::paged_bias_sink::{self, PagedBiasSink};
use crate::kv::KvArenaId;

/// Asks for the whole-step graphs on CUDA: `on`; off when unset.
pub const STEP_GRAPHS_ENV: &str = "IZWI_INKLING2_STEP_GRAPHS";

const MIB: u64 = 1024 * 1024;

/// What the graphs may hold. A kept graph retains its stable inputs (the
/// stacked convolution histories, about 3.8 MB a row at the published
/// geometry, and the step buffers) and its outputs (the next histories and
/// `width x vocab` F32 logits, 0.8 MB a row); the pool holds the widest
/// step's working set, with the experts' scratch.
pub(crate) const STEP_GRAPH_LIMITS: ShapeGraphLimits = ShapeGraphLimits {
    max_graphs: 12,
    retained_bytes: 384 * MIB,
    pool_bytes: 256 * MIB,
};

/// Whether [`STEP_GRAPHS_ENV`] asks for the graphs.
pub fn step_graphs_asked() -> bool {
    crate::env::value(STEP_GRAPHS_ENV)
        .map(|value| value.trim().to_ascii_lowercase())
        .is_some_and(|value| matches!(value.as_str(), "on" | "1" | "true"))
}

/// Device bytes a load on `backend` prices for the step graphs: zero off
/// CUDA or unless asked for.
pub fn step_graph_device_bytes(backend: BackendKind) -> u64 {
    if backend == BackendKind::Cuda
        && step_graphs_asked()
        && !crate::kernels::cuda::stretch_graphs::decode_graphs_switched_off()
    {
        STEP_GRAPH_LIMITS.device_bytes()
    } else {
        0
    }
}

/// What a cache's part of a graph depends on beside its inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CacheKey {
    pub(crate) arena: KvArenaId,
    pub(crate) generation: u64,
    pub(crate) max_blocks: usize,
}

/// What a graph's captured addresses depend on beside its inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GraphKey {
    pub(crate) width: usize,
    pub(crate) main: CacheKey,
    pub(crate) windowed: Option<CacheKey>,
}

/// One model's step graphs, or (off CUDA, for tests) the eager body.
pub(crate) struct Inkling2StepGraph {
    graphs: Option<ShapeGraphs<GraphKey>>,
    checked: Mutex<Vec<usize>>,
    off: Mutex<Option<String>>,
    graphed_steps: AtomicU64,
    spare_rows: AtomicU64,
    warm_checks: AtomicU64,
}

impl Inkling2StepGraph {
    /// The graphs on `device`, or why there are none.
    pub(crate) fn resolve(device: &Device) -> std::result::Result<Self, String> {
        if !step_graphs_asked() {
            return Err(format!("{STEP_GRAPHS_ENV} is not on"));
        }
        if crate::kernels::cuda::stretch_graphs::decode_graphs_switched_off() {
            return Err("cuda.decode_graphs=off".into());
        }
        Ok(Self::with(Some(ShapeGraphs::resolve(
            "Inkling 2 step",
            device,
            STEP_GRAPH_LIMITS,
        )?)))
    }

    /// The step's body run eagerly every step, through the same inputs and
    /// writes: what the graphs replay, without a capture.
    #[cfg(test)]
    pub(crate) fn eager() -> Self {
        Self::with(None)
    }

    fn with(graphs: Option<ShapeGraphs<GraphKey>>) -> Self {
        Self {
            graphs,
            checked: Mutex::new(Vec::new()),
            off: Mutex::new(None),
            graphed_steps: AtomicU64::new(0),
            spare_rows: AtomicU64::new(0),
            warm_checks: AtomicU64::new(0),
        }
    }

    pub(crate) fn is_on(&self) -> bool {
        self.graphs.as_ref().is_none_or(ShapeGraphs::is_on)
            && self.off.lock().is_ok_and(|off| off.is_none())
    }

    pub(crate) fn disable(&self, reason: String) {
        tracing::warn!("Inkling 2 step graphs are off: {reason}");
        if let Ok(mut off) = self.off.lock() {
            off.get_or_insert(reason);
        }
    }

    pub(crate) fn needs_check(&self, width: usize) -> bool {
        !self
            .checked
            .lock()
            .is_ok_and(|checked| checked.contains(&width))
    }

    pub(crate) fn mark_checked(&self, width: usize) {
        self.warm_checks.fetch_add(1, Relaxed);
        if let Ok(mut checked) = self.checked.lock() {
            if !checked.contains(&width) {
                checked.push(width);
            }
        }
    }

    pub(crate) fn record_step(&self, rows: usize, width: usize) {
        self.graphed_steps.fetch_add(1, Relaxed);
        self.spare_rows
            .fetch_add(width.saturating_sub(rows) as u64, Relaxed);
    }

    /// Runs `body` over `inputs`: replayed from its graph under `key`, or
    /// eagerly where there are no graphs.
    pub(crate) fn run(
        &self,
        key: GraphKey,
        retained: u64,
        inputs: &[Tensor],
        body: &dyn Fn(&[Tensor]) -> Result<Vec<Tensor>>,
    ) -> Result<Vec<Tensor>> {
        match &self.graphs {
            Some(graphs) => {
                // A grown arena replaced buffers older graphs captured.
                graphs.retain(|kept| {
                    kept.main.arena != key.main.arena || kept.main.generation >= key.main.generation
                });
                graphs.run(key, retained, inputs, body)
            }
            None => body(inputs),
        }
    }

    pub(crate) fn diagnostics(&self) -> serde_json::Value {
        json!({
            "off": self.off.lock().ok().and_then(|off| off.clone()),
            "graphed_steps": self.graphed_steps.load(Relaxed),
            "spare_rows": self.spare_rows.load(Relaxed),
            "warm_checks": self.warm_checks.load(Relaxed),
            "checked_widths": self.checked.lock().map(|c| c.clone()).unwrap_or_default(),
            "graphs": self.graphs.as_ref().map(ShapeGraphs::diagnostics),
        })
    }
}

/// One cache's part of a graphed step: its layers' storage, the step
/// buffer's slot and metadata views, and the metadata's host copy (the
/// fused kernel checks it when the step is captured or run eagerly).
pub(crate) struct GraphCache<'a> {
    pub(crate) layers: &'a [(Tensor, Tensor)],
    pub(crate) slots: Tensor,
    pub(crate) metadata: Tensor,
    pub(crate) host_metadata: &'a [u32],
    pub(crate) layout: DecodeStepLayout,
}

/// The rows context of a graphed step: device-slot writes and the fused
/// attention over the step buffers' tables.
pub(crate) struct GraphRowsContext<'a> {
    pub(crate) main: GraphCache<'a>,
    pub(crate) windowed: Option<GraphCache<'a>>,
    /// Per application: whether it lives in the windowed cache, and at
    /// which of that cache's layers.
    pub(crate) routes: &'a [(bool, usize)],
}

impl Inkling2RowsContext for GraphRowsContext<'_> {
    fn attend_rows(
        &mut self,
        application: usize,
        keys: &Tensor,
        values: &Tensor,
        queries: &Tensor,
        profiles: &Tensor,
        sinks: &Tensor,
        scale: f32,
        window: Option<usize>,
    ) -> Result<Tensor> {
        let (windowed, ordinal) = *self
            .routes
            .get(application)
            .ok_or_else(|| Error::InferenceError(format!("no layer application {application}")))?;
        let cache = match (windowed, &self.windowed) {
            (true, Some(cache)) => cache,
            _ => &self.main,
        };
        let (cache_keys, cache_values) = cache
            .layers
            .get(ordinal)
            .ok_or_else(|| Error::InferenceError("a graphed step lost a cache layer".into()))?;
        let storage = cache_keys.dtype();
        write_kv_rows(
            cache_keys,
            cache_values,
            &keys.to_dtype(storage)?,
            &values.to_dtype(storage)?,
            &cache.slots,
        )?;
        let by_row =
            |tensor: &Tensor| -> Result<Tensor> { Ok(tensor.transpose(0, 1)?.contiguous()?) };
        paged_bias_sink::attend(&PagedBiasSink {
            queries: &by_row(queries)?,
            profiles: &by_row(profiles)?,
            sinks: &by_row(&sinks.squeeze(2)?)?,
            keys: cache_keys,
            values: cache_values,
            metadata: cache.host_metadata,
            device_metadata: Some(&cache.metadata),
            max_blocks: cache.layout.max_blocks,
            queries_per_table: 1,
            scale,
            window: window
                .map(|window| {
                    u32::try_from(window)
                        .map_err(|_| Error::InferenceError("a window exceeds u32".into()))
                })
                .transpose()?,
        })
    }
}
