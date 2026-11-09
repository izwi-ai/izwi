//! The per-step inputs of a whole-step decode graph over a paged KV arena,
//! for families whose decode step is pure attention layers (the Qwen3 trunk
//! of Qwen3-ASR and VibeVoice-ASR). A step pads its rows up to the next
//! graphed width; every per-step value lives in one U32 **step buffer** the
//! graph reads on the device:
//! `[position[w] | flat KV slot[w] | context[w] | first-page offset[w] |
//! block table[w x max_blocks]]`, the last three being the metadata
//! [`super::split_decode::paged_decode_adaptive`] reads. A spare row writes
//! no KV ([`SPARE_SLOT`]) and attends one token of page 0; every op of the
//! step is row-independent, so a spare row changes no real row's values.
//!
//! The family's graph takes the buffer as an input (copied into the graph's
//! stable buffer each replay), so building it is a host vector and one
//! upload per step. Chat's Qwen3.5 graphs keep their own buffer
//! (`qwen35::decode_graph`); this one is shared by the ASR families.

use candle_core::{DType, Device, Tensor};

use crate::error::{Error, Result};
use crate::kv::{KvSequenceBlockTable, KvSlotRef};

/// The graphed batch widths: a step of `n` rows replays the graph of the
/// smallest width that holds it.
pub const WIDTHS: [usize; 10] = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32];

/// A spare row's flat KV slot: the KV write skips it (Candle's scatter
/// skips the index type's maximum).
pub const SPARE_SLOT: u32 = u32::MAX;

/// Partitions a row's attention may take before its partitions grow past
/// `split_decode::ROW_PARTITION_TOKENS` (8k tokens of context), as the
/// Qwen3.5 graphs cap them.
const ROW_INVARIANT_PARTITIONS: usize = 64;

/// Logit agreement a family's first graphed step of a width demands
/// against its standard decode. They differ in attention partitioning, so
/// this catches composition bugs, not rounding.
pub const WARM_CHECK_REL_L2: f64 = 0.05;

/// The width whose graph serves a step of `rows` rows.
pub fn graph_width(rows: usize) -> Option<usize> {
    WIDTHS.iter().copied().find(|width| *width >= rows.max(1))
}

/// One real row of a step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeStepRow {
    /// The position its RoPE rotates by.
    pub position: u32,
    /// Its flat KV slot: page index × page tokens + offset.
    pub slot: u32,
    /// Its visible context after this token, as the split decode reads it.
    pub context_len: u32,
    pub first_page_offset: u32,
    /// Its block table's page indices.
    pub blocks: Vec<u32>,
}

impl DecodeStepRow {
    /// The row appending at `slot` whose context after the token is
    /// `sequence`, rotating at `position`.
    pub fn new(
        position: u32,
        slot: &KvSlotRef,
        sequence: &KvSequenceBlockTable,
        page_tokens: u32,
    ) -> Result<Self> {
        let slot = slot
            .block
            .index
            .checked_mul(page_tokens)
            .and_then(|base| base.checked_add(slot.offset))
            .filter(|flat| *flat != SPARE_SLOT)
            .ok_or_else(|| Error::InferenceError("decode graph KV slot overflows u32".into()))?;
        Ok(Self {
            position,
            slot,
            context_len: sequence.context_len,
            first_page_offset: sequence.first_page_offset,
            blocks: sequence.blocks.iter().map(|block| block.index).collect(),
        })
    }
}

/// The step buffer's layout for `rows` rows of `max_blocks` block-table
/// entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeStepLayout {
    pub rows: usize,
    pub max_blocks: usize,
}

impl DecodeStepLayout {
    /// The layout of a width-`rows` step whose longest block table has
    /// `blocks` pages: the table width rounds up to a power of two (at
    /// least 8), so a graph serves a run of steps as rows grow.
    pub fn for_blocks(rows: usize, blocks: usize) -> Self {
        Self {
            rows,
            max_blocks: blocks.max(8).next_power_of_two(),
        }
    }

    pub fn positions(&self) -> usize {
        0
    }

    pub fn slots(&self) -> usize {
        self.rows
    }

    pub fn metadata(&self) -> usize {
        2 * self.rows
    }

    pub fn metadata_len(&self) -> usize {
        self.rows * (2 + self.max_blocks)
    }

    pub fn len(&self) -> usize {
        self.metadata() + self.metadata_len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    /// The longest context the layout's block tables hold.
    pub fn max_context(&self, page_tokens: usize) -> usize {
        self.max_blocks.saturating_mul(page_tokens)
    }

    /// The buffer of `rows`, the rows past them spare.
    pub fn values(&self, rows: &[DecodeStepRow]) -> Result<Vec<u32>> {
        if rows.len() > self.rows {
            return Err(Error::InferenceError(format!(
                "a decode step of {} rows exceeds its width {}",
                rows.len(),
                self.rows
            )));
        }
        let mut values = vec![0u32; self.len()];
        let table = self.metadata() + 2 * self.rows;
        for index in 0..self.rows {
            let Some(row) = rows.get(index) else {
                // A spare row writes nothing and attends one token of page 0:
                // a zero context would leave its partitions empty.
                values[self.slots() + index] = SPARE_SLOT;
                values[self.metadata() + index] = 1;
                continue;
            };
            if row.blocks.len() > self.max_blocks || row.blocks.is_empty() {
                return Err(Error::InferenceError(format!(
                    "a decode row's {} blocks do not fit the step's {}",
                    row.blocks.len(),
                    self.max_blocks
                )));
            }
            values[self.positions() + index] = row.position;
            values[self.slots() + index] = row.slot;
            values[self.metadata() + index] = row.context_len;
            values[self.metadata() + self.rows + index] = row.first_page_offset;
            let start = table + index * self.max_blocks;
            values[start..start + row.blocks.len()].copy_from_slice(&row.blocks);
        }
        Ok(values)
    }

    /// The buffer of `rows` on `device`: one upload.
    pub fn upload(&self, rows: &[DecodeStepRow], device: &Device) -> Result<Tensor> {
        let values = self.values(rows)?;
        Ok(Tensor::from_vec(values, self.len(), device)?)
    }
}

/// The split decode's partition cap for a `width`-row step whose rows hold
/// at most `max_context` tokens: the occupancy count, but at least enough
/// for each row to split its own context into 128-token partitions up to an
/// 8k-token context, so a row's attention does not depend on its batch.
pub fn max_partitions(width: usize, kv_heads: usize, max_context: usize) -> usize {
    super::split_decode::partitions(width, kv_heads, max_context).max(
        max_context
            .div_ceil(super::split_decode::ROW_PARTITION_TOKENS)
            .min(ROW_INVARIANT_PARTITIONS),
    )
}

/// Writes each row's `keys` and `values` (`[rows, kv_heads, head_dim]`, the
/// cache dtype) into the paged caches (`[pages, page_tokens, kv_heads,
/// head_dim]`) at its flat slot in `slots` (U32 `[rows]`, a view of the step
/// buffer); a [`SPARE_SLOT`] writes nothing. Device indices only, so a
/// capture records it.
pub fn write_kv_rows(
    cache_keys: &Tensor,
    cache_values: &Tensor,
    keys: &Tensor,
    values: &Tensor,
    slots: &Tensor,
) -> candle_core::Result<()> {
    for (cache, rows) in [(cache_keys, keys), (cache_values, values)] {
        let (heads, dim) = (rows.dim(1)?, rows.dim(2)?);
        let flat = cache.elem_count() / (heads * dim).max(1);
        let cache = cache.reshape((flat, heads, dim))?;
        let indices = slots
            .reshape((slots.elem_count(), 1, 1))?
            .broadcast_as(rows.shape())?
            .contiguous()?;
        cache.scatter_set(&indices, &rows.contiguous()?, 0)?;
    }
    Ok(())
}

/// Whether a graphed step's `graph` logits agree with the standard
/// decode's `standard` (`[rows, .., vocab]`, the step's real rows): relative
/// L2 within [`WARM_CHECK_REL_L2`] and the same greedy token on every row.
/// Reads both to the host (the warm step only).
pub fn logits_agree(graph: &Tensor, standard: &Tensor) -> std::result::Result<f64, String> {
    let read = |t: &Tensor| -> candle_core::Result<Vec<Vec<f32>>> {
        let rows = t.dim(0)?;
        t.to_dtype(DType::F32)?
            .reshape((rows, ()))?
            .to_vec2::<f32>()
    };
    if graph.dims() != standard.dims() {
        return Err(format!(
            "graph logits {:?} differ in shape from the standard decode's {:?}",
            graph.dims(),
            standard.dims()
        ));
    }
    let (graph, standard) = match (read(graph), read(standard)) {
        (Ok(graph), Ok(standard)) => (graph, standard),
        (Err(error), _) | (_, Err(error)) => return Err(format!("logit readback: {error}")),
    };
    let (mut difference, mut norm) = (0f64, 0f64);
    for (row, (graph, standard)) in graph.iter().zip(&standard).enumerate() {
        if argmax(graph) != argmax(standard) {
            return Err(format!(
                "row {row} picks token {:?} against the standard decode's {:?}",
                argmax(graph),
                argmax(standard)
            ));
        }
        for (g, s) in graph.iter().zip(standard) {
            let delta = f64::from(*g) - f64::from(*s);
            // A non-finite difference never passes.
            difference += if delta.is_finite() {
                delta * delta
            } else {
                f64::INFINITY
            };
            norm += f64::from(*s) * f64::from(*s);
        }
    }
    let relative = (difference / norm.max(f64::MIN_POSITIVE)).sqrt();
    if relative.is_finite() && relative <= WARM_CHECK_REL_L2 {
        Ok(relative)
    } else {
        Err(format!(
            "graph logits differ from the standard decode by relative L2 {relative:e}"
        ))
    }
}

/// The first greedy index of `values`, `None` when one is not finite.
fn argmax(values: &[f32]) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return None;
        }
        if best.is_none_or(|(_, top)| value > top) {
            best = Some((index, value));
        }
    }
    best.map(|(index, _)| index)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(position: u32, slot: u32, context: u32, blocks: &[u32]) -> DecodeStepRow {
        DecodeStepRow {
            position,
            slot,
            context_len: context,
            first_page_offset: 0,
            blocks: blocks.to_vec(),
        }
    }

    #[test]
    fn steps_pad_up_to_the_next_graphed_width() {
        assert_eq!(graph_width(0), Some(1));
        assert_eq!(graph_width(5), Some(6));
        assert_eq!(graph_width(17), Some(24));
        assert_eq!(graph_width(32), Some(32));
        assert_eq!(graph_width(33), None);
    }

    #[test]
    fn the_step_buffer_packs_rows_then_spare_rows_then_metadata() {
        let layout = DecodeStepLayout::for_blocks(3, 2);
        assert_eq!(layout.max_blocks, 8);
        assert_eq!(layout.len(), 3 + 3 + 3 * (2 + 8));
        let values = layout
            .values(&[row(7, 40, 8, &[5]), row(19, 61, 20, &[2, 3])])
            .unwrap();
        assert_eq!(&values[..3], [7, 19, 0], "positions");
        assert_eq!(&values[3..6], [40, 61, SPARE_SLOT], "slots");
        assert_eq!(
            &values[6..9],
            [8, 20, 1],
            "contexts: a spare row attends one token"
        );
        assert_eq!(&values[9..12], [0, 0, 0], "first-page offsets");
        assert_eq!(&values[12..14], [5, 0], "row 0's table");
        assert_eq!(&values[20..22], [2, 3], "row 1's table");
        assert!(
            values[28..36].iter().all(|block| *block == 0),
            "spare row: page 0"
        );
        assert!(layout.values(&[row(0, 0, 1, &[0; 9])]).is_err());
        assert!(layout.values(&[row(0, 0, 1, &[])]).is_err());
        let four = [
            row(0, 0, 1, &[0]),
            row(0, 1, 1, &[0]),
            row(0, 2, 1, &[0]),
            row(0, 3, 1, &[0]),
        ];
        assert!(layout.values(&four).is_err(), "more rows than the width");
        assert_eq!(DecodeStepLayout::for_blocks(1, 9).max_blocks, 16);
    }

    #[test]
    fn partitions_cover_each_rows_own_context_up_to_8k_tokens() {
        let tokens = super::super::split_decode::ROW_PARTITION_TOKENS;
        assert!(max_partitions(32, 8, 8 * 1024) >= (8 * 1024usize).div_ceil(tokens));
        assert!(max_partitions(1, 8, 300) >= 300usize.div_ceil(tokens));
        assert_eq!(
            max_partitions(32, 8, 1 << 20),
            max_partitions(32, 8, 1 << 21),
            "capped past 8k tokens"
        );
    }

    #[test]
    fn kv_rows_land_at_their_flat_slots_and_spare_rows_write_nothing() {
        let device = Device::Cpu;
        // Two pages of two tokens, one head of two dims.
        let keys = Tensor::zeros((2, 2, 1, 2), DType::F32, &device).unwrap();
        let values = Tensor::zeros((2, 2, 1, 2), DType::F32, &device).unwrap();
        let k = Tensor::new(&[[[1f32, 2.]], [[3., 4.]]], &device).unwrap();
        let v = Tensor::new(&[[[5f32, 6.]], [[7., 8.]]], &device).unwrap();
        let slots = Tensor::new(&[2u32, SPARE_SLOT], &device).unwrap();
        write_kv_rows(&keys, &values, &k, &v, &slots).unwrap();
        let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(flat(&keys), [0., 0., 0., 0., 1., 2., 0., 0.]);
        assert_eq!(flat(&values), [0., 0., 0., 0., 5., 6., 0., 0.]);
    }

    /// On CUDA the KV write lands by device slot and skips a spare row, as
    /// on CPU.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_kv_rows_land_at_their_flat_slots_and_spare_rows_write_nothing() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        let keys = Tensor::zeros((2, 2, 1, 2), DType::BF16, &device).unwrap();
        let values = Tensor::zeros((2, 2, 1, 2), DType::BF16, &device).unwrap();
        let k = Tensor::new(&[[[1f32, 2.]], [[3., 4.]]], &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let v = (&k + 4.0).unwrap();
        let slots = Tensor::new(&[2u32, SPARE_SLOT], &device).unwrap();
        write_kv_rows(&keys, &values, &k, &v, &slots).unwrap();
        let flat = |t: &Tensor| {
            t.to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        assert_eq!(flat(&keys), [0., 0., 0., 0., 1., 2., 0., 0.]);
        assert_eq!(flat(&values), [0., 0., 0., 0., 5., 6., 0., 0.]);
    }

    #[test]
    fn logits_agree_within_rounding_and_refuse_divergence_and_nan() {
        let device = Device::Cpu;
        let standard = Tensor::new(&[[1f32, 4., 2.], [3., 0., 1.]], &device).unwrap();
        let close = Tensor::new(&[[1.01f32, 4., 2.], [3., 0.01, 1.]], &device).unwrap();
        assert!(logits_agree(&close, &standard).unwrap() < 0.01);
        let flipped = Tensor::new(&[[1f32, 4., 4.5], [3., 0., 1.]], &device).unwrap();
        assert!(logits_agree(&flipped, &standard)
            .unwrap_err()
            .contains("row 0"));
        let far = Tensor::new(&[[1f32, 4., 2.], [9., 0., 1.]], &device).unwrap();
        assert!(logits_agree(&far, &standard)
            .unwrap_err()
            .contains("relative L2"));
        let nan = Tensor::new(&[[1f32, 4., f32::NAN], [3., 0., 1.]], &device).unwrap();
        assert!(logits_agree(&nan, &standard).is_err());
        let shape = Tensor::new(&[[1f32, 4., 2.]], &device).unwrap();
        assert!(logits_agree(&shape, &standard).is_err());
    }
}
