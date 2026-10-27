//! Mimi's split residual vector quantizer: one semantic codebook and the
//! acoustic ones, each group with its own 1×1 input and output projections.
//! Codebooks are EMA centroids, `embed_sum / max(cluster_usage, 1e-5)`.

use candle_core::{DType, Tensor, D};
use candle_nn::VarBuilder;

use super::config::MimiConfig;
use crate::error::{Error, Result};

const CLUSTER_USAGE_EPSILON: f64 = 1e-5;

struct RvqGroup {
    /// `[codebooks · codebook_size, codebook_dim]`.
    table: Tensor,
    /// `|e|²` per entry, `[codebooks, codebook_size]` (encode only).
    norms: Tensor,
    codebooks: usize,
    /// `[codebook_dim, hidden]` and `[hidden, codebook_dim]`.
    input_proj: Tensor,
    output_proj: Tensor,
}

pub(crate) struct MimiQuantizer {
    semantic: RvqGroup,
    acoustic: RvqGroup,
    codebook_size: usize,
    codebook_dim: usize,
}

impl RvqGroup {
    fn load(cfg: &MimiConfig, codebooks: usize, vb: VarBuilder) -> Result<Self> {
        let (size, dim, hidden) = (cfg.codebook_size, cfg.codebook_dim, cfg.hidden_size);
        let tables = (0..codebooks)
            .map(|index| {
                let codebook = vb.pp(format!("layers.{index}.codebook"));
                let sum = codebook.get((size, dim), "embed_sum")?;
                let usage = codebook
                    .get(size, "cluster_usage")?
                    .clamp(CLUSTER_USAGE_EPSILON, f64::INFINITY)?;
                Ok(sum.broadcast_div(&usage.unsqueeze(1)?)?)
            })
            .collect::<Result<Vec<_>>>()?;
        let table = Tensor::cat(&tables, 0)?;
        let norms = table.sqr()?.sum(1)?.reshape((codebooks, size))?;
        Ok(Self {
            table,
            norms,
            codebooks,
            input_proj: vb.get((dim, hidden, 1), "input_proj.weight")?.squeeze(2)?,
            output_proj: vb.get((hidden, dim, 1), "output_proj.weight")?.squeeze(2)?,
        })
    }

    /// Σ of the selected entries, `[N, codebook_dim]`, for `codes`
    /// `[N, codebooks]` (U32, already offset by codebook).
    fn lookup(&self, offset_codes: &Tensor, n: usize, dim: usize) -> Result<Tensor> {
        let picked = self
            .table
            .index_select(&offset_codes.flatten_all()?, 0)?
            .reshape((n, self.codebooks, dim))?;
        Ok(picked.sum(1)?)
    }

    /// Residual quantization of `x` `[N, hidden]`: codes `[N, codebooks]`.
    fn encode(&self, x: &Tensor, size: usize) -> Result<Vec<Vec<u32>>> {
        let mut residual = x.matmul(&self.input_proj.t()?)?;
        let mut codes = Vec::with_capacity(self.codebooks);
        for book in 0..self.codebooks {
            let table = self.table.narrow(0, book * size, size)?;
            // argmin |r − e|² = argmin (|e|² − 2 r·e).
            let scores = self
                .norms
                .get(book)?
                .unsqueeze(0)?
                .broadcast_sub(&(residual.matmul(&table.t()?)? * 2.0)?)?;
            let picked = scores.argmin(D::Minus1)?;
            residual = (residual - table.index_select(&picked, 0)?)?;
            codes.push(picked.to_dtype(DType::U32)?.to_vec1::<u32>()?);
        }
        Ok(codes)
    }
}

impl MimiQuantizer {
    pub(crate) fn load(cfg: &MimiConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            semantic: RvqGroup::load(
                cfg,
                cfg.num_semantic_quantizers,
                vb.pp("semantic_residual_vector_quantizer"),
            )?,
            acoustic: RvqGroup::load(
                cfg,
                cfg.acoustic_quantizers(),
                vb.pp("acoustic_residual_vector_quantizer"),
            )?,
            codebook_size: cfg.codebook_size,
            codebook_dim: cfg.codebook_dim,
        })
    }

    pub(crate) fn codebooks(&self) -> usize {
        self.semantic.codebooks + self.acoustic.codebooks
    }

    /// Frames of codes, frame-major `[N · codebooks]`, to the quantized
    /// latent `[N, hidden]`.
    pub(crate) fn decode(&self, codes: &[u32], device: &candle_core::Device) -> Result<Tensor> {
        let books = self.codebooks();
        if codes.is_empty() || !codes.len().is_multiple_of(books) {
            return Err(Error::InvalidInput(format!(
                "Mimi decodes whole frames of {books} codes, got {}",
                codes.len()
            )));
        }
        if let Some(&code) = codes.iter().find(|&&c| c as usize >= self.codebook_size) {
            return Err(Error::InvalidInput(format!(
                "Mimi code {code} is outside the {}-entry codebook",
                self.codebook_size
            )));
        }
        let n = codes.len() / books;
        let size = self.codebook_size as u32;
        let semantic_books = self.semantic.codebooks;
        let mut semantic = Vec::with_capacity(n * semantic_books);
        let mut acoustic = Vec::with_capacity(n * self.acoustic.codebooks);
        for frame in codes.chunks_exact(books) {
            for (book, &code) in frame.iter().enumerate() {
                if book < semantic_books {
                    semantic.push(book as u32 * size + code);
                } else {
                    acoustic.push((book - semantic_books) as u32 * size + code);
                }
            }
        }
        let semantic = Tensor::from_vec(semantic, n * semantic_books, device)?;
        let acoustic = Tensor::from_vec(acoustic, n * self.acoustic.codebooks, device)?;
        let dim = self.codebook_dim;
        let s = self.semantic.lookup(&semantic, n, dim)?;
        let a = self.acoustic.lookup(&acoustic, n, dim)?;
        Ok((s.matmul(&self.semantic.output_proj.t()?)?
            + a.matmul(&self.acoustic.output_proj.t()?)?)?)
    }

    /// Codes for the latent `x` `[N, hidden]`, frame-major `[N · codebooks]`.
    /// The semantic and acoustic groups both quantize `x` itself.
    pub(crate) fn encode(&self, x: &Tensor) -> Result<Vec<u32>> {
        let size = self.codebook_size;
        let mut books = self.semantic.encode(x, size)?;
        books.extend(self.acoustic.encode(x, size)?);
        let n = x.dim(0)?;
        let mut codes = Vec::with_capacity(n * books.len());
        for frame in 0..n {
            codes.extend(books.iter().map(|book| book[frame]));
        }
        Ok(codes)
    }
}
