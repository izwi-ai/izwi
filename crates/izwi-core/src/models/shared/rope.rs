//! Rotary position embedding frequencies, including the scaled variants a
//! checkpoint declares in its `rope_scaling` (transformers v4) or
//! `rope_parameters` (v5) config object.
//!
//! Everything here is host-side F64. Candle compiles its Metal kernels with
//! fast math, so device `cos`/`sin` drift by about 1e-4 at position 4K; a
//! family builds its cos/sin table here and uploads it once.

use serde_json::Value;

use crate::error::{Error, Result};

/// The RoPE variant a checkpoint's language model was trained with.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RopeScaling {
    /// Unscaled RoPE: `inv_freq[i] = theta^(-2i/d)`.
    Default,
    /// YaRN, as transformers' `_compute_yarn_parameters` computes it.
    Yarn(YarnRopeScaling),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct YarnRopeScaling {
    /// Position interpolation factor applied to the low-frequency pairs.
    pub factor: f64,
    /// Context length the base model was pretrained at.
    pub original_max_position_embeddings: usize,
    /// Rotation counts bounding the extrapolated and interpolated pairs.
    pub beta_fast: f64,
    pub beta_slow: f64,
    /// Explicit cos/sin scale. Inferred from `factor` (and the mscale pair,
    /// when both are set) when absent.
    pub attention_factor: Option<f64>,
    pub mscale: Option<f64>,
    pub mscale_all_dim: Option<f64>,
    /// Round the correction range outward to whole pairs.
    pub truncate: bool,
}

/// Per-pair inverse frequencies, and the factor RoPE multiplies both cos
/// and sin by (1 for default RoPE).
#[derive(Debug, Clone, PartialEq)]
pub struct RopeFrequencies {
    pub inv_freq: Vec<f64>,
    pub attention_scale: f64,
}

impl RopeScaling {
    /// Parse a `rope_scaling` or `rope_parameters` object. Absent, `null`, or
    /// `rope_type: "default"` is unscaled RoPE; a type this module does not
    /// implement is an error, never silently unscaled. Older configs name
    /// the type `type`. `max_position_embeddings` stands in for an absent
    /// `original_max_position_embeddings`, as in transformers.
    pub fn from_config_value(
        value: Option<&Value>,
        max_position_embeddings: usize,
    ) -> Result<Self> {
        let Some(params) = value.filter(|value| !value.is_null()) else {
            return Ok(Self::Default);
        };
        let Some(object) = params.as_object() else {
            return Err(Error::ModelLoadError(format!(
                "RoPE scaling must be an object, got {params}"
            )));
        };
        let rope_type = object
            .get("rope_type")
            .or_else(|| object.get("type"))
            .and_then(Value::as_str)
            .unwrap_or("default");
        match rope_type {
            "default" => Ok(Self::Default),
            "yarn" => {
                let number = |key: &str| object.get(key).and_then(Value::as_f64);
                // transformers reads these with `or`: zero means absent.
                let nonzero = |key: &str| number(key).filter(|value| *value != 0.0);
                let factor = number("factor").ok_or_else(|| {
                    Error::ModelLoadError("YaRN RoPE scaling is missing its factor".into())
                })?;
                let original = match object.get("original_max_position_embeddings") {
                    Some(value) => value
                        .as_u64()
                        .and_then(|value| usize::try_from(value).ok())
                        .ok_or_else(|| {
                            Error::ModelLoadError(format!(
                                "YaRN original_max_position_embeddings must be a positive integer, got {value}"
                            ))
                        })?,
                    None => max_position_embeddings,
                };
                let yarn = YarnRopeScaling {
                    factor,
                    original_max_position_embeddings: original,
                    beta_fast: nonzero("beta_fast").unwrap_or(32.0),
                    beta_slow: nonzero("beta_slow").unwrap_or(1.0),
                    attention_factor: number("attention_factor"),
                    mscale: nonzero("mscale"),
                    mscale_all_dim: nonzero("mscale_all_dim"),
                    truncate: object
                        .get("truncate")
                        .and_then(Value::as_bool)
                        .unwrap_or(true),
                };
                yarn.validate()?;
                Ok(Self::Yarn(yarn))
            }
            other => Err(Error::ModelLoadError(format!(
                "RoPE scaling type {other:?} is not implemented"
            ))),
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Yarn(_) => "yarn",
        }
    }
}

impl YarnRopeScaling {
    fn validate(&self) -> Result<()> {
        let positive = |name: &str, value: f64| {
            if value.is_finite() && value > 0.0 {
                Ok(())
            } else {
                Err(Error::ModelLoadError(format!(
                    "YaRN {name} must be finite and positive, got {value}"
                )))
            }
        };
        positive("factor", self.factor)?;
        positive("beta_fast", self.beta_fast)?;
        positive("beta_slow", self.beta_slow)?;
        if let Some(attention_factor) = self.attention_factor {
            positive("attention_factor", attention_factor)?;
        }
        if self.original_max_position_embeddings == 0 {
            return Err(Error::ModelLoadError(
                "YaRN original_max_position_embeddings must be positive".into(),
            ));
        }
        Ok(())
    }

    /// transformers' `get_mscale`.
    fn mscale(scale: f64, mscale: f64) -> f64 {
        if scale <= 1.0 {
            1.0
        } else {
            0.1 * mscale * scale.ln() + 1.0
        }
    }

    fn attention_scale(&self) -> f64 {
        if let Some(attention_factor) = self.attention_factor {
            return attention_factor;
        }
        match (self.mscale, self.mscale_all_dim) {
            (Some(mscale), Some(all_dim)) => {
                Self::mscale(self.factor, mscale) / Self::mscale(self.factor, all_dim)
            }
            _ => Self::mscale(self.factor, 1.0),
        }
    }

    /// The pair index `num_rotations` full turns span at the pretraining
    /// context.
    fn correction_dim(&self, num_rotations: f64, rotary_dim: usize, theta: f64) -> f64 {
        rotary_dim as f64
            * (self.original_max_position_embeddings as f64
                / (num_rotations * 2.0 * std::f64::consts::PI))
                .ln()
            / (2.0 * theta.ln())
    }
}

/// Inverse frequencies for a rotary dimension of `rotary_dim` (even) with
/// base `theta` under `scaling`.
pub fn rope_frequencies(
    rotary_dim: usize,
    theta: f64,
    scaling: &RopeScaling,
) -> Result<RopeFrequencies> {
    if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) {
        return Err(Error::ModelLoadError(format!(
            "RoPE rotary dimension must be even and positive, got {rotary_dim}"
        )));
    }
    if !theta.is_finite() || theta <= 1.0 {
        return Err(Error::ModelLoadError(format!(
            "RoPE theta must be finite and greater than 1, got {theta}"
        )));
    }
    let pairs = rotary_dim / 2;
    let base = (0..pairs)
        .map(|pair| theta.powf(-((2 * pair) as f64) / rotary_dim as f64))
        .collect::<Vec<_>>();
    let RopeScaling::Yarn(yarn) = scaling else {
        return Ok(RopeFrequencies {
            inv_freq: base,
            attention_scale: 1.0,
        });
    };

    let mut low = yarn.correction_dim(yarn.beta_fast, rotary_dim, theta);
    let mut high = yarn.correction_dim(yarn.beta_slow, rotary_dim, theta);
    if yarn.truncate {
        low = low.floor();
        high = high.ceil();
    }
    let low = low.max(0.0);
    let mut high = high.min((rotary_dim - 1) as f64);
    if low == high {
        // transformers' guard against a zero-width ramp.
        high += 0.001;
    }
    let inv_freq = base
        .iter()
        .enumerate()
        .map(|(pair, extrapolated)| {
            let interpolated = extrapolated / yarn.factor;
            let ramp = ((pair as f64 - low) / (high - low)).clamp(0.0, 1.0);
            // Below `low` a pair keeps its trained frequency; above `high`
            // it is interpolated by the full factor.
            let extrapolation = 1.0 - ramp;
            interpolated * (1.0 - extrapolation) + extrapolated * extrapolation
        })
        .collect();
    Ok(RopeFrequencies {
        inv_freq,
        attention_scale: yarn.attention_scale(),
    })
}

impl RopeFrequencies {
    /// `(cos, sin)` of `position * inv_freq` for positions `start..start +
    /// len`, each times the attention scale: row-major `[len, pairs]`,
    /// computed in F64 and rounded to F32 once.
    pub fn cos_sin_table(&self, start: usize, len: usize) -> (Vec<f32>, Vec<f32>) {
        let pairs = self.inv_freq.len();
        let mut cos = Vec::with_capacity(len * pairs);
        let mut sin = Vec::with_capacity(len * pairs);
        for position in start..start + len {
            for inv in &self.inv_freq {
                let angle = position as f64 * inv;
                cos.push((angle.cos() * self.attention_scale) as f32);
                sin.push((angle.sin() * self.attention_scale) as f32);
            }
        }
        (cos, sin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Granite Speech 5.0's language model: head_dim 128, theta 1e4, YaRN
    /// factor 4 from a 4096-token pretraining context to 16384.
    fn granite_speech_5_yarn() -> RopeScaling {
        RopeScaling::from_config_value(
            Some(&json!({
                "factor": 4.0,
                "original_max_position_embeddings": 4096,
                "rope_type": "yarn"
            })),
            16_384,
        )
        .unwrap()
    }

    fn assert_close(actual: f64, expected: f64) {
        let tolerance = 1e-12 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} != {expected}"
        );
    }

    #[test]
    fn default_rope_is_the_unscaled_power_series() {
        let freqs = rope_frequencies(8, 10_000.0, &RopeScaling::Default).unwrap();
        assert_eq!(freqs.attention_scale, 1.0);
        for (pair, inv) in freqs.inv_freq.iter().enumerate() {
            assert_close(*inv, 1.0 / 10_000f64.powf((2 * pair) as f64 / 8.0));
        }
    }

    /// Values from transformers 4.57's `_compute_yarn_parameters` equations
    /// in F64: the correction range is pairs (20, 46), so pairs 0-20 keep
    /// their frequency, 46-63 are divided by the factor and 21-45 ramp.
    #[test]
    fn yarn_matches_the_reference_equations_for_granite_speech_5() {
        let freqs = rope_frequencies(128, 10_000.0, &granite_speech_5_yarn()).unwrap();
        let default = rope_frequencies(128, 10_000.0, &RopeScaling::Default).unwrap();
        assert_close(freqs.attention_scale, 1.138_629_436_111_989);
        for (pair, expected) in [
            (0, 1.0),
            (20, 0.056_234_132_519_034_905),
            (21, 0.047_292_038_501_684_786),
            (30, 0.009_488_517_882_700_576),
            (45, 0.000_429_402_588_997_358_2),
            (46, 0.000_333_380_358_040_831),
            (63, 2.886_954_961_723_645_5e-5),
        ] {
            assert_close(freqs.inv_freq[pair], expected);
        }
        let changed = freqs
            .inv_freq
            .iter()
            .zip(&default.inv_freq)
            .filter(|(yarn, base)| yarn != base)
            .count();
        assert_eq!(changed, 43);
        for pair in 46..64 {
            assert_close(freqs.inv_freq[pair], default.inv_freq[pair] / 4.0);
        }
    }

    #[test]
    fn cos_sin_table_scales_both_components() {
        let freqs = rope_frequencies(128, 10_000.0, &granite_speech_5_yarn()).unwrap();
        let (cos, sin) = freqs.cos_sin_table(4_999, 2);
        assert_eq!(cos.len(), 128);
        assert_eq!(sin.len(), 128);
        // Position 5000, pair 30.
        assert!((f64::from(cos[64 + 30]) - -1.081_292_241_830_471_8).abs() < 1e-6);
        assert!((f64::from(sin[64 + 30]) - -0.356_768_945_590_754_3).abs() < 1e-6);
        let unscaled = rope_frequencies(8, 10_000.0, &RopeScaling::Default).unwrap();
        let (cos, sin) = unscaled.cos_sin_table(0, 1);
        assert!(cos.iter().all(|value| *value == 1.0));
        assert!(sin.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn mscale_pair_and_explicit_attention_factor_follow_transformers() {
        let deepseek = RopeScaling::from_config_value(
            Some(&json!({
                "type": "yarn",
                "factor": 40,
                "original_max_position_embeddings": 4096,
                "mscale": 1.0,
                "mscale_all_dim": 0.707
            })),
            163_840,
        )
        .unwrap();
        let freqs = rope_frequencies(64, 10_000.0, &deepseek).unwrap();
        assert_close(freqs.attention_scale, 1.085_726_399_256_135_5);

        let explicit = RopeScaling::from_config_value(
            Some(&json!({"rope_type": "yarn", "factor": 4.0, "attention_factor": 1.0})),
            4096,
        )
        .unwrap();
        let RopeScaling::Yarn(yarn) = explicit else {
            panic!("expected YaRN");
        };
        // An absent original context falls back to max_position_embeddings.
        assert_eq!(yarn.original_max_position_embeddings, 4096);
        assert_eq!(
            rope_frequencies(64, 10_000.0, &explicit)
                .unwrap()
                .attention_scale,
            1.0
        );
    }

    #[test]
    fn parser_accepts_default_forms_and_rejects_what_it_cannot_run() {
        for value in [
            None,
            Some(Value::Null),
            Some(json!({"rope_type": "default", "rope_theta": 10000})),
            Some(json!({"rope_theta": 10000})),
        ] {
            assert_eq!(
                RopeScaling::from_config_value(value.as_ref(), 4096).unwrap(),
                RopeScaling::Default
            );
        }
        let error = |value: Value| {
            RopeScaling::from_config_value(Some(&value), 4096)
                .unwrap_err()
                .to_string()
        };
        assert!(error(json!({"rope_type": "linear", "factor": 2.0})).contains("\"linear\""));
        assert!(error(json!({"type": "dynamic", "factor": 2.0})).contains("\"dynamic\""));
        assert!(error(json!({"rope_type": "yarn"})).contains("missing its factor"));
        assert!(error(json!({"rope_type": "yarn", "factor": -1.0})).contains("factor"));
        assert!(error(json!({
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": "4096"
        }))
        .contains("original_max_position_embeddings"));
        assert!(error(json!("yarn")).contains("must be an object"));
    }

    #[test]
    fn frequencies_reject_degenerate_geometry() {
        assert!(rope_frequencies(0, 10_000.0, &RopeScaling::Default).is_err());
        assert!(rope_frequencies(7, 10_000.0, &RopeScaling::Default).is_err());
        assert!(rope_frequencies(8, f64::NAN, &RopeScaling::Default).is_err());
        assert!(rope_frequencies(8, 1.0, &RopeScaling::Default).is_err());
    }
}
