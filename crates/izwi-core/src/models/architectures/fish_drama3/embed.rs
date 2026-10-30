//! The trunk's row embedding (§3.3).
//!
//! Every position is a row `[main, c0, …, c9]`:
//!
//! ```text
//! text row     : x = E[main]
//! semantic row : x = (E[main] + CB[1 * V + c1]) / sqrt(2)     # add, then scale
//! ```
//!
//! `E` is `embed_tokens` and `CB` is `audio_decoder.codebook_embeddings`,
//! indexed `k * V + code` with `V` codes per codebook. A row is semantic when
//! its main id lies in the semantic band. The trunk reads codebook 1 only;
//! the tower adds codebooks 2..9 itself ([`FishDrama3Embedding::add_codebook_row`]),
//! and codebook 0 is never read, so it is not loaded. Codes are clamped to
//! `[0, V - 1]` before a lookup.
//!
//! Both tables stay BF16 on the host (D4) and are gathered to F32 per row,
//! so they cost no device memory.

use std::ops::Range;

use candle_core::{Device, Tensor};
use half::bf16;
use safetensors::tensor::Dtype;

use super::checkpoint::{FishDrama3Checkpoint, CODEBOOK_EMBEDDINGS, TRUNK_PREFIX};
use super::config::{FishDrama3Config, FishDrama3TokenIds};
use super::prompt::FishDrama3Prompt;
use crate::error::{Error, Result};

/// The codebook the trunk's semantic rows add.
pub const TRUNK_CODEBOOK: usize = 1;

pub struct FishDrama3Embedding {
    hidden: usize,
    vocab: usize,
    /// `embed_tokens`, `[vocab, hidden]`.
    tokens: Vec<bf16>,
    /// Codes per codebook (`V`).
    codebook_vocab: usize,
    codebooks: usize,
    /// `CB` rows `V..codebooks * V` (codebooks 1 and up).
    codebook_table: Vec<bf16>,
    ids: FishDrama3TokenIds,
}

impl std::fmt::Debug for FishDrama3Embedding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FishDrama3Embedding")
            .field("hidden", &self.hidden)
            .field("vocab", &self.vocab)
            .field("codebooks", &self.codebooks)
            .finish()
    }
}

fn load_error(message: impl Into<String>) -> Error {
    Error::ModelLoadError(format!("Fish Audio Drama 3 embedding: {}", message.into()))
}

/// Rows `rows` of the BF16 matrix `name` (`[.., cols]`), copied out.
fn bf16_table(
    checkpoint: &FishDrama3Checkpoint,
    name: &str,
    shape: [usize; 2],
    rows: Range<usize>,
) -> Result<Vec<bf16>> {
    let (dtype, found) = checkpoint.info(name)?;
    if dtype != Dtype::BF16 || found != shape {
        return Err(load_error(format!(
            "{name} is {dtype:?} {found:?}; expected BF16 {shape:?}"
        )));
    }
    let row_bytes = shape[1] * 2;
    let bytes = checkpoint
        .bytes(name)?
        .get(rows.start * row_bytes..rows.end * row_bytes)
        .ok_or_else(|| load_error(format!("rows {rows:?} of {name} lie outside it")))?;
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| bf16::from_le_bytes(*b))
        .collect())
}

impl FishDrama3Embedding {
    pub fn load(checkpoint: &FishDrama3Checkpoint, config: &FishDrama3Config) -> Result<Self> {
        let hidden = config.trunk.hidden;
        let vocab = config.trunk.vocab;
        let tower = &config.tower;
        let (v, k) = (tower.codebook_vocab, tower.codebooks);
        let tokens = bf16_table(
            checkpoint,
            &format!("{TRUNK_PREFIX}.embed_tokens.weight"),
            [vocab, hidden],
            0..vocab,
        )?;
        let codebook_table =
            bf16_table(checkpoint, CODEBOOK_EMBEDDINGS, [k * v, hidden], v..k * v)?;
        Ok(Self {
            hidden,
            vocab,
            tokens,
            codebook_vocab: v,
            codebooks: k,
            codebook_table,
            ids: config.ids,
        })
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// Values per row: the main id and every code.
    pub fn frame_width(&self) -> usize {
        self.codebooks + 1
    }

    /// Host bytes of both tables.
    pub fn host_bytes(&self) -> usize {
        (self.tokens.len() + self.codebook_table.len()) * 2
    }

    /// `CB[codebook * V + clamp(code)]`, the row's BF16 values.
    fn codebook_row(&self, codebook: usize, code: u32) -> Result<&[bf16]> {
        if codebook == 0 || codebook >= self.codebooks {
            return Err(Error::InferenceError(format!(
                "codebook {codebook} is not one of the loaded 1..{}",
                self.codebooks
            )));
        }
        let code = (code as usize).min(self.codebook_vocab - 1);
        let row = (codebook - 1) * self.codebook_vocab + code;
        Ok(&self.codebook_table[row * self.hidden..(row + 1) * self.hidden])
    }

    /// Adds `CB[codebook * V + clamp(code)]` to `out` (the tower's sum over
    /// codebooks 2..9 of a frame).
    pub fn add_codebook_row(&self, codebook: usize, code: u32, out: &mut [f32]) -> Result<()> {
        if out.len() != self.hidden {
            return Err(Error::InferenceError(format!(
                "a codebook row is {} wide, not {}",
                self.hidden,
                out.len()
            )));
        }
        for (o, v) in out.iter_mut().zip(self.codebook_row(codebook, code)?) {
            *o += v.to_f32();
        }
        Ok(())
    }

    /// The F32 embedding `[n, hidden]` of `rows`, row-major `[n, width]`.
    pub fn embed(&self, rows: &[u32]) -> Result<Vec<f32>> {
        let width = self.frame_width();
        if rows.is_empty() || !rows.len().is_multiple_of(width) {
            return Err(Error::InvalidInput(format!(
                "{} row values are not whole rows of {width}",
                rows.len()
            )));
        }
        let mut out = Vec::with_capacity(rows.len() / width * self.hidden);
        for row in rows.chunks_exact(width) {
            let main = row[0] as usize;
            if main >= self.vocab {
                return Err(Error::InvalidInput(format!(
                    "token {main} is outside the {}-token vocabulary",
                    self.vocab
                )));
            }
            let e = &self.tokens[main * self.hidden..(main + 1) * self.hidden];
            if self.ids.is_semantic(row[0]) {
                let cb = self.codebook_row(TRUNK_CODEBOOK, row[1 + TRUNK_CODEBOOK])?;
                out.extend(
                    e.iter()
                        .zip(cb)
                        .map(|(e, c)| (e.to_f32() + c.to_f32()) / std::f32::consts::SQRT_2),
                );
            } else {
                out.extend(e.iter().map(|e| e.to_f32()));
            }
        }
        Ok(out)
    }

    /// [`Self::embed`] as an F32 `[n, hidden]` tensor on `device`.
    pub fn embed_tensor(&self, rows: &[u32], device: &Device) -> Result<Tensor> {
        let values = self.embed(rows)?;
        let n = values.len() / self.hidden;
        Ok(Tensor::from_vec(values, (n, self.hidden), device)?)
    }
}

/// Rows `range` of `prompt`, row-major `[main, c0, …]` as the embedding
/// takes them.
pub fn prompt_rows(prompt: &FishDrama3Prompt, range: Range<usize>) -> Result<Vec<u32>> {
    if range.start >= range.end || range.end > prompt.rows {
        return Err(Error::InvalidInput(format!(
            "rows {range:?} of a {}-row prompt",
            prompt.rows
        )));
    }
    let mut out = Vec::with_capacity(range.len() * prompt.values.len());
    for t in range {
        for values in &prompt.values {
            out.push(*values.get(t).ok_or_else(|| {
                Error::InvalidInput(format!("the prompt has no value at row {t}"))
            })?);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::super::synthetic::SyntheticCheckpoint;
    use super::*;

    #[test]
    fn text_rows_read_e_and_semantic_rows_add_codebook_1_then_scale() {
        let synthetic = SyntheticCheckpoint::write(1);
        let config = &synthetic.config;
        let checkpoint = synthetic.open();
        let embedding = FishDrama3Embedding::load(&checkpoint, config).unwrap();
        let hidden = config.trunk.hidden;
        let width = embedding.frame_width();
        assert_eq!(width, 11);
        let e = checkpoint
            .bf16_to_f32(
                &format!("{TRUNK_PREFIX}.embed_tokens.weight"),
                &[config.trunk.vocab, hidden],
            )
            .unwrap();
        let v = config.tower.codebook_vocab;
        let cb = checkpoint
            .bf16_to_f32(CODEBOOK_EMBEDDINGS, &[10 * v, hidden])
            .unwrap();
        let ids = config.ids;
        let semantic = ids.semantic(5).unwrap();
        // A text row (its codes are ignored), a semantic row, a semantic row
        // whose code 1 is past the codebook (clamped), and the band's ends.
        let mut rows = Vec::new();
        let row = |main: u32, c1: u32| {
            let mut r = vec![0u32; width];
            r[0] = main;
            r[1] = main.saturating_sub(ids.semantic_start);
            r[2] = c1;
            r[3] = 7; // codebook 2: never read by the trunk
            r
        };
        rows.extend(row(ids.eos, 9));
        rows.extend(row(semantic, 3));
        rows.extend(row(semantic, 9_999));
        rows.extend(row(ids.semantic_start, 0));
        rows.extend(row(ids.semantic_end, v as u32 - 1));
        rows.extend(row(ids.semantic_start - 1, 3)); // a speaker tag: text
        let out = embedding.embed(&rows).unwrap();
        let at = |i: usize| &out[i * hidden..(i + 1) * hidden];
        let e_row = |id: u32| &e[id as usize * hidden..(id as usize + 1) * hidden];
        let cb_row =
            |k: usize, code: usize| &cb[(k * v + code) * hidden..(k * v + code + 1) * hidden];
        let mixed = |id: u32, code: usize| {
            e_row(id)
                .iter()
                .zip(cb_row(1, code))
                .map(|(a, b)| (a + b) / std::f32::consts::SQRT_2)
                .collect::<Vec<_>>()
        };
        assert_eq!(at(0), e_row(ids.eos));
        assert_eq!(at(1), mixed(semantic, 3).as_slice());
        assert_eq!(at(2), mixed(semantic, v - 1).as_slice());
        assert_eq!(at(3), mixed(ids.semantic_start, 0).as_slice());
        assert_eq!(at(4), mixed(ids.semantic_end, v - 1).as_slice());
        assert_eq!(at(5), e_row(ids.semantic_start - 1));

        // RED: CB1 unscaled is off by far more than the exact match above.
        let unscaled = e_row(semantic)
            .iter()
            .zip(cb_row(1, 3))
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>();
        let error = crate::kernels::numeric_check::relative_l2(at(1), &unscaled);
        assert!(error > 0.25, "{error}");

        // The tower's codebook rows, clamped the same way.
        let mut sum = vec![0f32; hidden];
        embedding.add_codebook_row(2, 4, &mut sum).unwrap();
        embedding.add_codebook_row(9, 70_000, &mut sum).unwrap();
        let expected = cb_row(2, 4)
            .iter()
            .zip(cb_row(9, v - 1))
            .map(|(a, b)| 0.0 + a + b)
            .collect::<Vec<_>>();
        assert_eq!(sum, expected);
        assert!(embedding.add_codebook_row(0, 1, &mut sum).is_err());
        assert!(embedding.add_codebook_row(10, 1, &mut sum).is_err());

        // Malformed input.
        assert!(embedding.embed(&rows[..width - 1]).is_err());
        let mut bad = row(config.trunk.vocab as u32, 0);
        assert!(embedding.embed(&bad).is_err());
        bad[0] = ids.eos;
        let tensor = embedding.embed_tensor(&bad, &Device::Cpu).unwrap();
        assert_eq!(tensor.dims(), [1, hidden]);
    }

    #[test]
    fn prompt_rows_lay_codebook_major_values_out_row_major() {
        let prompt = FishDrama3Prompt {
            values: vec![vec![10, 11, 12], vec![0, 1, 2], vec![0, 5, 6]],
            rows: 3,
        };
        assert_eq!(
            prompt_rows(&prompt, 1..3).unwrap(),
            vec![11, 1, 5, 12, 2, 6]
        );
        assert!(prompt_rows(&prompt, 2..2).is_err());
        assert!(prompt_rows(&prompt, 0..4).is_err());
    }
}
