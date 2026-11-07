//! ITU-T G.711 μ-law and A-law companding (8 kHz telephony audio, one byte a
//! sample).

/// G.711's sample rate.
pub const SAMPLE_RATE: u32 = 8_000;

/// μ-law code of one 16-bit linear sample.
pub fn linear_to_ulaw(sample: i16) -> u8 {
    const BIAS: i32 = 0x84;
    const CLIP: i32 = 32_635;
    let mut pcm = i32::from(sample);
    let sign = if pcm < 0 {
        pcm = -pcm;
        0x80
    } else {
        0
    };
    let pcm = pcm.min(CLIP) + BIAS;
    let exponent = (15 - (pcm as u16).leading_zeros() as i32 - 7).max(0);
    let mantissa = (pcm >> (exponent + 3)) & 0x0F;
    !((sign | (exponent << 4) | mantissa) as u8)
}

/// A-law code of one 16-bit linear sample.
pub fn linear_to_alaw(sample: i16) -> u8 {
    const SEGMENT_ENDS: [i32; 8] = [0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF];
    let mut pcm = i32::from(sample) >> 3;
    let mask = if pcm >= 0 {
        0xD5
    } else {
        pcm = -pcm - 1;
        0x55
    };
    let Some(segment) = SEGMENT_ENDS.iter().position(|end| pcm <= *end) else {
        return 0x7F ^ mask;
    };
    let shift = if segment < 2 { 1 } else { segment };
    let value = ((segment as i32) << 4) | ((pcm >> shift) & 0x0F);
    (value as u8) ^ mask
}

/// The 16-bit linear sample of one μ-law code.
pub fn ulaw_to_linear(code: u8) -> i16 {
    let code = !code;
    let exponent = (code >> 4) & 0x07;
    let magnitude = ((((code & 0x0F) as i32) << 3) + 0x84) << exponent;
    let value = magnitude - 0x84;
    (if code & 0x80 != 0 { -value } else { value }) as i16
}

/// The 16-bit linear sample of one A-law code.
pub fn alaw_to_linear(code: u8) -> i16 {
    let code = code ^ 0x55;
    let segment = (code & 0x70) >> 4;
    let mut value = ((code & 0x0F) as i32) << 4;
    value = match segment {
        0 => value + 8,
        1 => value + 0x108,
        _ => (value + 0x108) << (segment - 1),
    };
    (if code & 0x80 != 0 { value } else { -value }) as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_match_the_standard_anchors() {
        assert_eq!(linear_to_ulaw(0), 0xFF);
        assert_eq!(linear_to_ulaw(i16::MAX), 0x80);
        assert_eq!(linear_to_ulaw(i16::MIN), 0x00);
        assert_eq!(linear_to_alaw(0), 0xD5);
        assert_eq!(linear_to_alaw(i16::MAX), 0xAA);
        assert_eq!(linear_to_alaw(i16::MIN), 0x2A);
    }

    /// Companding keeps every sample within its segment's quantization step
    /// (relative error under 1/16 above the smallest steps).
    #[test]
    fn round_trips_within_one_quantization_step() {
        for sample in (i16::MIN..=i16::MAX).step_by(7) {
            for (name, decoded) in [
                ("ulaw", ulaw_to_linear(linear_to_ulaw(sample))),
                ("alaw", alaw_to_linear(linear_to_alaw(sample))),
            ] {
                let error = (i32::from(decoded) - i32::from(sample)).abs();
                let allowed = (i32::from(sample).abs() / 16).max(140);
                assert!(error <= allowed, "{name} {sample} -> {decoded}");
            }
        }
    }
}
