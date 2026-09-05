//! Fish's adjacent-pair rotary embeddings, backed by Candle's native kernels.
//!
//! The released Dual-AR and DAC implementations round their frequency tables to
//! BF16, then rotate in F32 before restoring the activation dtype. Keep the table
//! precision explicit: it is a checkpoint contract, not the activation dtype.

use candle_core::{DType, Device, Tensor};

use crate::error::{Error, Result};

/// Load-owned tables shared by every layer in one transformer stack.
#[derive(Clone)]
pub(crate) struct FishS2RotaryCache {
    cos: Tensor,
    sin: Tensor,
    max_seq_len: usize,
    head_dim: usize,
    storage_bytes: u64,
}

impl FishS2RotaryCache {
    pub(crate) fn new(
        max_seq_len: usize,
        head_dim: usize,
        rope_theta: f64,
        frequency_dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        if max_seq_len == 0 || head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(Error::ConfigError(
                "Fish S2 rotary cache requires nonzero positions and an even nonzero head dimension"
                    .into(),
            ));
        }
        let rope_theta_f32 = rope_theta as f32;
        if !rope_theta_f32.is_finite() || rope_theta_f32 <= 0.0 {
            return Err(Error::ConfigError(
                "Fish S2 rotary base must be positive and finite in F32".into(),
            ));
        }
        if !matches!(frequency_dtype, DType::BF16 | DType::F16 | DType::F32) {
            return Err(Error::ConfigError(format!(
                "Fish S2 rotary frequency dtype must be BF16, F16, or F32, got {frequency_dtype:?}"
            )));
        }
        let half_dim = head_dim / 2;
        let elements = max_seq_len.checked_mul(half_dim).ok_or_else(|| {
            Error::ConfigError("Fish S2 rotary cache element count overflowed".into())
        })?;
        let storage_bytes = u64::try_from(elements)
            .ok()
            .and_then(|elements| elements.checked_mul(2 * DType::F32.size_in_bytes() as u64))
            .ok_or_else(|| {
                Error::ConfigError("Fish S2 rotary cache byte count overflowed".into())
            })?;

        // Match the reference's F32 frequency construction before its explicit
        // table rounding. Round on the host so a Metal F16/F32 model never needs
        // BF16 device storage or casts just to reproduce the checkpoint table.
        let inv_freqs = (0..half_dim)
            .map(|pair| {
                rope_theta_f32
                    .powf((2 * pair) as f32 / head_dim as f32)
                    .recip()
            })
            .collect::<Vec<_>>();
        let round_frequency = |value| match frequency_dtype {
            DType::BF16 => half::bf16::from_f32(value).to_f32(),
            DType::F16 => half::f16::from_f32(value).to_f32(),
            _ => value,
        };
        let mut cos = Vec::with_capacity(elements);
        let mut sin = Vec::with_capacity(elements);
        for position in 0..max_seq_len {
            for &inv_freq in &inv_freqs {
                let angle = position as f32 * inv_freq;
                cos.push(round_frequency(angle.cos()));
                sin.push(round_frequency(angle.sin()));
            }
        }
        Ok(Self {
            cos: Tensor::from_vec(cos, (max_seq_len, half_dim), device)?,
            sin: Tensor::from_vec(sin, (max_seq_len, half_dim), device)?,
            max_seq_len,
            head_dim,
            storage_bytes,
        })
    }

    pub(crate) fn storage_bytes(&self) -> u64 {
        self.storage_bytes
    }

    /// Rotate `[batch, sequence, heads, head_dim]` using absolute positions.
    pub(crate) fn apply(&self, x: &Tensor, start_pos: usize) -> Result<Tensor> {
        let (batch_size, seq_len, num_heads, head_dim) = x.dims4()?;
        if batch_size == 0 || seq_len == 0 || num_heads == 0 || head_dim != self.head_dim {
            return Err(Error::InvalidInput(format!(
                "Fish S2 rotary input requires nonempty [batch,sequence,heads,{}], got {:?}",
                self.head_dim,
                x.dims()
            )));
        }
        let end_pos = start_pos.checked_add(seq_len).ok_or_else(|| {
            Error::InvalidInput("Fish S2 rotary position range overflowed".into())
        })?;
        if end_pos > self.max_seq_len {
            return Err(Error::InvalidInput(format!(
                "Fish S2 rotary position range {start_pos}..{end_pos} exceeds cache capacity {}",
                self.max_seq_len
            )));
        }
        if !matches!(x.dtype(), DType::BF16 | DType::F16 | DType::F32) {
            return Err(Error::InvalidInput(format!(
                "Fish S2 rotary input must be BF16, F16, or F32, got {:?}",
                x.dtype()
            )));
        }
        let cos = self.cos.narrow(0, start_pos, seq_len)?;
        let sin = self.sin.narrow(0, start_pos, seq_len)?;
        let input = x.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()?;
        let rotated = candle_nn::rotary_emb::rope_i(&input, &cos, &sin)?;
        rotated
            .transpose(1, 2)?
            .to_dtype(x.dtype())?
            .contiguous()
            .map_err(Error::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Frozen with the published PyTorch precompute_freqs_cis/apply_rotary_emb,
    // fishaudio/fish-speech befe4001745417f8c42131739d862b8a6fdbd15a,
    // fish_speech/models/text2semantic/llama.py:1004-1038. Input [1,2,3,4],
    // base 1e6, BF16 frequency table, F32 rotation. These fixtures distinguish
    // adjacent pairs from split halves and table rounding from full precision.
    #[allow(clippy::excessive_precision)]
    const HEAD4_ORACLES: [(usize, [f32; 4]); 4] = [
        (0, [1.0, 2.0, 3.0, 4.0]),
        (
            1,
            [-1.140625, 1.91796875, 2.996002197265625, 4.002998352050781],
        ),
        (
            17,
            [1.646484375, -1.51171875, 2.93212890625, 4.0509033203125],
        ),
        (1024, [1.3046875, 1.818359375, -1.86328125, 4.64453125]),
    ];

    // Same independent oracle at the model's real head_dim=128, position 17,
    // with nonuniform input x[i] = i/32 - 1.75.
    #[allow(clippy::excessive_precision)]
    const HEAD128_POSITION17: [f32; 128] = [
        -1.169677734375,
        2.15496826171875,
        0.7857666015625,
        -2.23126220703125,
        -1.6651611328125,
        1.554962158203125,
        2.1204833984375,
        0.5345458984375,
        0.186767578125,
        -2.089599609375,
        -1.93896484375,
        -0.5341796875,
        -1.26519775390625,
        1.4517669677734375,
        0.3409423828125,
        1.8046875,
        1.384246826171875,
        1.0615234375,
        1.654296875,
        0.1107177734375,
        1.4432373046875,
        -0.622802734375,
        1.0431175231933594,
        -1.0509815216064453,
        0.6361083984375,
        -1.23895263671875,
        0.2918701171875,
        -1.269287109375,
        0.0316162109375,
        -1.2161865234375,
        -0.15576171875,
        -1.1148681640625,
        -0.2767333984375,
        -1.00146484375,
        -0.34747314453125,
        -0.8834228515625,
        -0.38543701171875,
        -0.7725830078125,
        -0.3931884765625,
        -0.66650390625,
        -0.38104248046875,
        -0.5682373046875,
        -0.35687255859375,
        -0.4793701171875,
        -0.32025146484375,
        -0.3946533203125,
        -0.276824951171875,
        -0.31597900390625,
        -0.2281951904296875,
        -0.24169921875,
        -0.174713134765625,
        -0.17010498046875,
        -0.1191864013671875,
        -0.10150146484375,
        -0.06093597412109375,
        -0.0343780517578125,
        -0.00125885009765625,
        0.03125,
        0.05945587158203125,
        0.0957794189453125,
        0.12091827392578125,
        0.159515380859375,
        0.18288040161132812,
        0.22270965576171875,
        0.24522781372070312,
        0.285491943359375,
        0.30780029296875,
        0.3480224609375,
        0.3705120086669922,
        0.41039276123046875,
        0.4333229064941406,
        0.47264862060546875,
        0.49619007110595703,
        0.5348358154296875,
        0.5590753555297852,
        0.5969944000244141,
        0.6219358444213867,
        0.6591682434082031,
        0.6848020553588867,
        0.7213306427001953,
        0.7476396560668945,
        0.7835159301757812,
        0.8104400634765625,
        0.845733642578125,
        0.8732161521911621,
        0.9079723358154297,
        0.9359700679779053,
        0.9702305793762207,
        0.9986860752105713,
        1.0325241088867188,
        1.0613734722137451,
        1.0948443412780762,
        1.1240428686141968,
        1.1571812629699707,
        1.186686396598816,
        1.2195427417755127,
        1.249310851097107,
        1.2819223403930664,
        1.3119182586669922,
        1.3443182706832886,
        1.3745090961456299,
        1.4067299365997314,
        1.4370882511138916,
        1.4691530466079712,
        1.49965238571167,
        1.531590461730957,
        1.5622081756591797,
        1.5940361022949219,
        1.6247568130493164,
        1.6564886569976807,
        1.687295913696289,
        1.7189503908157349,
        1.7498301267623901,
        1.781416893005371,
        1.8123575448989868,
        1.8438899517059326,
        1.8748818635940552,
        1.906366229057312,
        1.9374014139175415,
        1.9688470363616943,
        1.9999181032180786,
        2.0313305854797363,
        2.062432050704956,
        2.0938167572021484,
        2.124943494796753,
        2.1563057899475098,
        2.187453269958496,
        2.2187962532043457,
    ];

    fn check_oracles(device: &Device) {
        let cache = FishS2RotaryCache::new(1025, 4, 1e6, DType::BF16, device).unwrap();
        let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 1, 1, 4), device).unwrap();
        for (position, expected) in HEAD4_ORACLES {
            let output = cache
                .apply(&input, position)
                .unwrap()
                .flatten_all()
                .unwrap();
            let values = output.to_vec1::<f32>().unwrap();
            for (actual, expected) in values.iter().zip(expected) {
                assert!(
                    (actual - expected).abs() < 2e-6,
                    "position {position}: {actual} != {expected}"
                );
            }
        }
        let cache = FishS2RotaryCache::new(32, 128, 1e6, DType::BF16, device).unwrap();
        let values = (0..128).map(|i| i as f32 / 32.0 - 1.75).collect::<Vec<_>>();
        let input = Tensor::from_vec(values, (1, 1, 1, 128), device).unwrap();
        let output = cache.apply(&input, 17).unwrap().flatten_all().unwrap();
        for (index, (actual, expected)) in output
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .zip(HEAD128_POSITION17)
            .enumerate()
        {
            assert!(
                (actual - expected).abs() < 2e-6,
                "channel {index}: {actual} != {expected}"
            );
        }
    }

    #[test]
    fn adjacent_rotation_matches_frozen_upstream_oracles() {
        check_oracles(&Device::Cpu);
    }

    #[test]
    fn cached_positions_match_chunked_noncontiguous_inputs() {
        let device = Device::Cpu;
        let cache = FishS2RotaryCache::new(64, 128, 1e6, DType::BF16, &device).unwrap();
        let values = (0..2 * 5 * 3 * 128)
            .map(|i| (i % 113) as f32 / 37.0 - 1.0)
            .collect::<Vec<_>>();
        let input = Tensor::from_vec(values, (2, 5, 3, 128), &device).unwrap();
        let whole = cache.apply(&input, 17).unwrap();
        assert!(
            whole.is_contiguous(),
            "paged attention requires contiguous THD inputs"
        );
        let first = cache.apply(&input.narrow(1, 0, 2).unwrap(), 17).unwrap();
        let second = cache.apply(&input.narrow(1, 2, 3).unwrap(), 19).unwrap();
        let chunked = Tensor::cat(&[first, second], 1).unwrap();
        assert_eq!(
            whole.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            chunked.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
        assert_eq!(cache.storage_bytes(), 64 * 128 * 4);
    }

    #[test]
    fn rotation_rounds_only_after_f32_compute_for_half_inputs() {
        let device = Device::Cpu;
        let cache = FishS2RotaryCache::new(18, 4, 1e6, DType::BF16, &device).unwrap();
        for dtype in [DType::F16, DType::BF16] {
            let input =
                Tensor::from_vec(vec![1.101f32, -2.303, 3.777, 0.009], (1, 1, 1, 4), &device)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap();
            let expected = cache
                .apply(&input.to_dtype(DType::F32).unwrap(), 17)
                .unwrap()
                .to_dtype(dtype)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();
            let output = cache.apply(&input, 17).unwrap();
            assert_eq!(output.dtype(), dtype);
            assert_eq!(
                output
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
                expected.flatten_all().unwrap().to_vec1::<f32>().unwrap()
            );
        }
    }

    #[test]
    fn rotary_rejects_invalid_geometry_and_out_of_range_positions() {
        let device = Device::Cpu;
        assert!(FishS2RotaryCache::new(4, 3, 1e6, DType::BF16, &device).is_err());
        assert!(FishS2RotaryCache::new(0, 4, 1e6, DType::BF16, &device).is_err());
        assert!(FishS2RotaryCache::new(4, 4, f64::NAN, DType::BF16, &device).is_err());
        let cache = FishS2RotaryCache::new(4, 4, 1e6, DType::BF16, &device).unwrap();
        let input = Tensor::zeros((1, 2, 1, 4), DType::F32, &device).unwrap();
        assert!(cache.apply(&input, 3).is_err());
        assert!(cache.apply(&input, usize::MAX).is_err());
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_rotation_matches_frozen_upstream_oracles_if_available() {
        if let Some(device) = crate::backends::metal_device_if_available(0) {
            check_oracles(&device);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_rotation_matches_frozen_upstream_oracles_if_available() {
        if let Ok(device) = Device::new_cuda(0) {
            check_oracles(&device);
        }
    }
}
