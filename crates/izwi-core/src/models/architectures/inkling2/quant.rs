//! The two expert weight formats of the published checkpoint, read as stored.
//!
//! * **NVFP4** (most layers): 4-bit E2M1 codes, two per byte with the even
//!   element in the low nibble; one FP8 E4M3 scale per 16 values along a row;
//!   one factor `amax / outer_scale_denominator` per expert.
//!   `value = code * scale * factor`.
//! * **MXFP8** (the last layers): FP8 E4M3 values; one scale byte `s` per 32
//!   values along a row. `value = e4m3 * 2^(s - 127)`.
//!
//! The arithmetic is the same as transformers'
//! `Inkling2LoopDequantizeExperts`; the tests decode rows of the published
//! checkpoint and compare them with values that script-side arithmetic
//! produced (`tests/fixtures/inkling2/real_slices.safetensors`).
//!
//! Weights stay packed in memory. [`PackedExperts::matvec`] and
//! [`PackedExperts::gate_up`] multiply straight from the packed bytes.

use std::sync::Arc;
use std::sync::OnceLock;

use memmap2::Mmap;

use crate::error::{Error, Result};

/// The value of each 4-bit E2M1 code.
pub const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];
/// Values one NVFP4 scale covers.
pub const NVFP4_BLOCK: usize = 16;
/// Values one MXFP8 scale covers.
pub const MXFP8_BLOCK: usize = 32;

/// The value of an FP8 E4M3 (`float8_e4m3fn`) byte. 0x7F and 0xFF are NaN.
pub fn e4m3(byte: u8) -> f32 {
    e4m3_table()[byte as usize]
}

pub fn e4m3_table() -> &'static [f32; 256] {
    static TABLE: OnceLock<[f32; 256]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [0.0f32; 256];
        for (byte, slot) in table.iter_mut().enumerate() {
            let sign = if byte & 0x80 != 0 { -1.0f32 } else { 1.0 };
            let exponent = ((byte >> 3) & 0xF) as i32;
            let mantissa = (byte & 0x7) as f32;
            let magnitude = if exponent == 0xF && byte & 0x7 == 0x7 {
                f32::NAN
            } else if exponent == 0 {
                mantissa / 8.0 * 2.0f32.powi(-6)
            } else {
                (1.0 + mantissa / 8.0) * 2.0f32.powi(exponent - 7)
            };
            *slot = sign * magnitude;
        }
        table
    })
}

/// Both values of an NVFP4 byte: `[low nibble, high nibble]`.
fn nvfp4_pairs() -> &'static [[f32; 2]; 256] {
    static TABLE: OnceLock<[[f32; 2]; 256]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [[0.0f32; 2]; 256];
        for (byte, slot) in table.iter_mut().enumerate() {
            *slot = [E2M1[byte & 0xF], E2M1[byte >> 4]];
        }
        table
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertFormat {
    Nvfp4,
    Mxfp8,
}

impl ExpertFormat {
    pub fn block(self) -> usize {
        match self {
            Self::Nvfp4 => NVFP4_BLOCK,
            Self::Mxfp8 => MXFP8_BLOCK,
        }
    }

    /// Stored bytes of a row of `columns` values.
    pub fn code_bytes(self, columns: usize) -> usize {
        match self {
            Self::Nvfp4 => columns / 2,
            Self::Mxfp8 => columns,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Nvfp4 => "nvfp4",
            Self::Mxfp8 => "mxfp8",
        }
    }
}

/// Bytes of a stored tensor: a range of the mapped checkpoint, or owned.
#[derive(Clone)]
pub enum PackedBytes {
    Mapped {
        map: Arc<Mmap>,
        start: usize,
        len: usize,
    },
    Owned(Arc<Vec<u8>>),
}

impl PackedBytes {
    pub fn as_slice(&self) -> &[u8] {
        match self {
            Self::Mapped { map, start, len } => &map[*start..*start + *len],
            Self::Owned(bytes) => bytes,
        }
    }

    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl From<Vec<u8>> for PackedBytes {
    fn from(bytes: Vec<u8>) -> Self {
        Self::Owned(Arc::new(bytes))
    }
}

/// One projection of every expert of a layer, `[experts, rows, columns]`,
/// in its stored form.
#[derive(Clone)]
pub struct PackedExperts {
    format: ExpertFormat,
    experts: usize,
    rows: usize,
    columns: usize,
    codes: PackedBytes,
    scales: PackedBytes,
    /// NVFP4: `amax / outer_scale_denominator` per expert. MXFP8: empty.
    factors: Vec<f32>,
}

impl std::fmt::Debug for PackedExperts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackedExperts")
            .field("format", &self.format)
            .field("experts", &self.experts)
            .field("rows", &self.rows)
            .field("columns", &self.columns)
            .finish()
    }
}

fn quant_error(message: impl std::fmt::Display) -> Error {
    Error::ModelLoadError(format!("Inkling 2 experts: {message}"))
}

impl PackedExperts {
    pub fn nvfp4(
        experts: usize,
        rows: usize,
        columns: usize,
        codes: PackedBytes,
        scales: PackedBytes,
        amax: &[f32],
        outer_scale_denominator: f32,
    ) -> Result<Self> {
        if columns % NVFP4_BLOCK != 0 {
            return Err(quant_error(format!(
                "{columns} columns are not a multiple of {NVFP4_BLOCK}"
            )));
        }
        if codes.len() != experts * rows * columns / 2
            || scales.len() != experts * rows * columns / NVFP4_BLOCK
            || amax.len() != experts
        {
            return Err(quant_error("NVFP4 tensor sizes disagree with the geometry"));
        }
        if !(outer_scale_denominator.is_finite() && outer_scale_denominator > 0.0)
            || amax.iter().any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(quant_error(
                "NVFP4 expert factors are not finite and positive",
            ));
        }
        Ok(Self {
            format: ExpertFormat::Nvfp4,
            experts,
            rows,
            columns,
            codes,
            scales,
            factors: amax
                .iter()
                .map(|value| value / outer_scale_denominator)
                .collect(),
        })
    }

    pub fn mxfp8(
        experts: usize,
        rows: usize,
        columns: usize,
        values: PackedBytes,
        scales: PackedBytes,
    ) -> Result<Self> {
        if columns % MXFP8_BLOCK != 0 {
            return Err(quant_error(format!(
                "{columns} columns are not a multiple of {MXFP8_BLOCK}"
            )));
        }
        if values.len() != experts * rows * columns
            || scales.len() != experts * rows * columns / MXFP8_BLOCK
        {
            return Err(quant_error("MXFP8 tensor sizes disagree with the geometry"));
        }
        Ok(Self {
            format: ExpertFormat::Mxfp8,
            experts,
            rows,
            columns,
            codes: values,
            scales,
            factors: Vec::new(),
        })
    }

    pub fn format(&self) -> ExpertFormat {
        self.format
    }

    pub fn experts(&self) -> usize {
        self.experts
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn codes(&self) -> &PackedBytes {
        &self.codes
    }

    pub fn scales(&self) -> &PackedBytes {
        &self.scales
    }

    /// NVFP4 per-expert factors (empty for MXFP8).
    pub fn factors(&self) -> &[f32] {
        &self.factors
    }

    /// Bytes this projection keeps resident.
    pub fn resident_bytes(&self) -> u64 {
        (self.codes.len() + self.scales.len() + self.factors.len() * 4) as u64
    }

    fn row_codes(&self, expert: usize, row: usize) -> &[u8] {
        let width = self.format.code_bytes(self.columns);
        let start = (expert * self.rows + row) * width;
        &self.codes.as_slice()[start..start + width]
    }

    fn row_scales(&self, expert: usize, row: usize) -> &[u8] {
        let width = self.columns / self.format.block();
        let start = (expert * self.rows + row) * width;
        &self.scales.as_slice()[start..start + width]
    }

    /// The values of one row, in the reference's operation order.
    pub fn decode_row(&self, expert: usize, row: usize, out: &mut [f32]) {
        assert_eq!(out.len(), self.columns);
        let codes = self.row_codes(expert, row);
        let scales = self.row_scales(expert, row);
        match self.format {
            ExpertFormat::Nvfp4 => {
                let pairs = nvfp4_pairs();
                let factor = self.factors[expert];
                for (block, scale) in scales.iter().enumerate() {
                    let scale = e4m3(*scale);
                    for (index, byte) in codes[block * 8..block * 8 + 8].iter().enumerate() {
                        let pair = pairs[*byte as usize];
                        out[block * 16 + 2 * index] = pair[0] * scale * factor;
                        out[block * 16 + 2 * index + 1] = pair[1] * scale * factor;
                    }
                }
            }
            ExpertFormat::Mxfp8 => {
                let table = e4m3_table();
                for (block, scale) in scales.iter().enumerate() {
                    let scale = (*scale as f32 - 127.0).exp2();
                    for index in 0..MXFP8_BLOCK {
                        let at = block * MXFP8_BLOCK + index;
                        out[at] = table[codes[at] as usize] * scale;
                    }
                }
            }
        }
    }

    /// `row . x`, straight from the packed bytes.
    #[inline]
    fn row_dot(&self, expert: usize, row: usize, x: &[f32]) -> f32 {
        let codes = self.row_codes(expert, row);
        let scales = self.row_scales(expert, row);
        match self.format {
            ExpertFormat::Nvfp4 => {
                let pairs = nvfp4_pairs();
                let mut total = 0.0f32;
                for ((scale, codes), x) in scales
                    .iter()
                    .zip(codes.chunks_exact(8))
                    .zip(x.chunks_exact(NVFP4_BLOCK))
                {
                    let mut values = [0.0f32; NVFP4_BLOCK];
                    for (pair, byte) in values.chunks_exact_mut(2).zip(codes) {
                        pair.copy_from_slice(&pairs[*byte as usize]);
                    }
                    let mut dot = 0.0f32;
                    for (value, x) in values.iter().zip(x) {
                        dot += value * x;
                    }
                    total += dot * e4m3(*scale);
                }
                total * self.factors[expert]
            }
            ExpertFormat::Mxfp8 => {
                let table = e4m3_table();
                let mut total = 0.0f32;
                for ((scale, codes), x) in scales
                    .iter()
                    .zip(codes.chunks_exact(MXFP8_BLOCK))
                    .zip(x.chunks_exact(MXFP8_BLOCK))
                {
                    let mut dot = 0.0f32;
                    for (byte, x) in codes.iter().zip(x) {
                        dot += table[*byte as usize] * x;
                    }
                    total += dot * (*scale as f32 - 127.0).exp2();
                }
                total
            }
        }
    }

    /// `out = W[expert] x`.
    pub fn matvec(&self, expert: usize, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.columns);
        assert_eq!(out.len(), self.rows);
        for (row, out) in out.iter_mut().enumerate() {
            *out = self.row_dot(expert, row, x);
        }
    }

    /// The gated first projection. Stored rows interleave gate and up
    /// (`gate0, up0, gate1, up1, ...`), so each output reads two adjacent
    /// rows: `out[i] = softcap(gate) * sigmoid(gate) * softcap(up)`.
    pub fn gate_up(&self, expert: usize, x: &[f32], gate_cap: f32, up_cap: f32, out: &mut [f32]) {
        assert_eq!(x.len(), self.columns);
        assert_eq!(out.len() * 2, self.rows);
        for (index, out) in out.iter_mut().enumerate() {
            let gate = self.row_dot(expert, 2 * index, x);
            let up = self.row_dot(expert, 2 * index + 1, x);
            *out = softcapped_swiglu(gate, up, gate_cap, up_cap);
        }
    }

    /// One expert as dense F32, row major (tests and self-checks).
    pub fn dequantize_expert(&self, expert: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.rows * self.columns];
        for (row, out) in out.chunks_exact_mut(self.columns).enumerate() {
            self.decode_row(expert, row, out);
        }
        out
    }
}

/// `cap * tanh(x / cap)`.
#[inline]
pub fn softcap(x: f32, cap: f32) -> f32 {
    (x / cap).tanh() * cap
}

/// The expert activation: the sigmoid sees the raw gate; the softcaps bound
/// the linear gate factor and `up`.
#[inline]
pub fn softcapped_swiglu(gate: f32, up: f32, gate_cap: f32, up_cap: f32) -> f32 {
    softcap(gate, gate_cap) * (1.0 / (1.0 + (-gate).exp())) * softcap(up, up_cap)
}

#[cfg(test)]
mod tests {
    use safetensors::SafeTensors;

    use super::super::config::tests::fixture;
    use super::*;

    fn f32s(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|quad| f32::from_le_bytes(quad.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn e4m3_matches_the_format() {
        assert_eq!(e4m3(0x00), 0.0);
        assert_eq!(e4m3(0x38), 1.0);
        assert_eq!(e4m3(0x7E), 448.0);
        assert_eq!(e4m3(0xB8), -1.0);
        assert_eq!(e4m3(0x01), 2.0f32.powi(-9));
        assert!(e4m3(0x7F).is_nan());
        assert!(e4m3(0xFF).is_nan());
        // Ascending over the positive codes.
        for byte in 1..0x7Fu8 {
            assert!(e4m3(byte) > e4m3(byte - 1));
        }
    }

    /// Rows of the published checkpoint decode to the values the reference
    /// arithmetic gives, bit for bit, and the packed products agree with a
    /// dense product over those values.
    #[test]
    fn published_expert_rows_decode_exactly() {
        let bytes = std::fs::read(fixture("inkling2").join("real_slices.safetensors")).unwrap();
        let file = SafeTensors::deserialize(&bytes).unwrap();
        let rows = 4usize;
        let mut checked = 0;
        for (key, format) in [
            ("nvfp4.layer0.expert3.w13_weight", ExpertFormat::Nvfp4),
            ("nvfp4.layer0.expert3.w2_weight", ExpertFormat::Nvfp4),
            ("mxfp8.layer20.expert5.w13_weight", ExpertFormat::Mxfp8),
            ("mxfp8.layer20.expert5.w2_weight", ExpertFormat::Mxfp8),
        ] {
            let expected = file.tensor(&format!("{key}.expected")).unwrap();
            let columns = expected.shape()[1];
            let expected = f32s(expected.data());
            let scales = file
                .tensor(&format!("{key}.scale"))
                .unwrap()
                .data()
                .to_vec();
            let packed = match format {
                ExpertFormat::Nvfp4 => PackedExperts::nvfp4(
                    1,
                    rows,
                    columns,
                    file.tensor(&format!("{key}.codes"))
                        .unwrap()
                        .data()
                        .to_vec()
                        .into(),
                    scales.into(),
                    &f32s(file.tensor(&format!("{key}.amax")).unwrap().data()),
                    f32s(
                        file.tensor(&format!("{key}.outer_scale_denominator"))
                            .unwrap()
                            .data(),
                    )[0],
                )
                .unwrap(),
                ExpertFormat::Mxfp8 => PackedExperts::mxfp8(
                    1,
                    rows,
                    columns,
                    file.tensor(&format!("{key}.values"))
                        .unwrap()
                        .data()
                        .to_vec()
                        .into(),
                    scales.into(),
                )
                .unwrap(),
            };
            let decoded = packed.dequantize_expert(0);
            assert_eq!(decoded.len(), expected.len());
            for (index, (got, want)) in decoded.iter().zip(&expected).enumerate() {
                assert_eq!(got.to_bits(), want.to_bits(), "{key} value {index}");
            }
            assert!(
                decoded.iter().any(|value| *value != 0.0),
                "{key} is all zero"
            );

            // Unstructured input (a sinusoid can hide a layout error).
            let mut state = 0x9E37_79B9u32;
            let x = (0..columns)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    (state as f32 / u32::MAX as f32) * 2.0 - 1.0
                })
                .collect::<Vec<_>>();
            let mut out = vec![0.0f32; rows];
            packed.matvec(0, &x, &mut out);
            for (row, got) in out.iter().enumerate() {
                let want = decoded[row * columns..(row + 1) * columns]
                    .iter()
                    .zip(&x)
                    .map(|(w, x)| *w as f64 * *x as f64)
                    .sum::<f64>();
                assert!(
                    (*got as f64 - want).abs() <= 1e-5 * want.abs().max(1.0),
                    "{key} row {row}: {got} vs {want}"
                );
            }
            checked += 1;
        }
        assert_eq!(checked, 4);
    }

    #[test]
    fn gate_up_reads_interleaved_rows() {
        // One expert, 2 outputs (4 rows), 32 columns, MXFP8 with unit scale.
        let columns = 32;
        let mut values = vec![0u8; 4 * columns];
        // gate0 = 1.0 * x[0], up0 = 2.0 * x[1], gate1 = -1.0 * x[2], up1 = 0.5 * x[3]
        values[0] = 0x38;
        values[columns + 1] = 0x40;
        values[2 * columns + 2] = 0xB8;
        values[3 * columns + 3] = 0x30;
        let packed =
            PackedExperts::mxfp8(1, 4, columns, values.into(), vec![127u8; 4].into()).unwrap();
        let mut x = vec![0.0f32; columns];
        x[..4].copy_from_slice(&[0.7, -0.4, 1.3, 2.0]);
        let mut out = [0.0f32; 2];
        packed.gate_up(0, &x, 4.0, 25.0, &mut out);
        let want0 = softcapped_swiglu(0.7, -0.8, 4.0, 25.0);
        let want1 = softcapped_swiglu(-1.3, 1.0, 4.0, 25.0);
        assert!((out[0] - want0).abs() < 1e-6 && (out[1] - want1).abs() < 1e-6);
        assert!(want0 != want1);
    }

    #[test]
    fn sizes_that_disagree_with_the_geometry_are_refused() {
        assert!(PackedExperts::mxfp8(1, 2, 48, vec![0; 96].into(), vec![0; 3].into()).is_err());
        assert!(PackedExperts::mxfp8(1, 2, 32, vec![0; 63].into(), vec![0; 2].into()).is_err());
        assert!(
            PackedExperts::nvfp4(1, 2, 16, vec![0; 16].into(), vec![0; 2].into(), &[1.0], 0.0)
                .is_err()
        );
    }
}
