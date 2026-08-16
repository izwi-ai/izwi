//! Checked model attention geometry shared by transformer loaders.

use crate::error::{Error, Result};

/// Validated query/KV head geometry with checked projection widths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AttentionGeometry {
    query_heads: usize,
    kv_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    query_width: usize,
    key_width: usize,
    value_width: usize,
}

impl AttentionGeometry {
    pub(crate) fn new(
        label: &str,
        query_heads: usize,
        kv_heads: usize,
        key_head_dim: usize,
        value_head_dim: usize,
    ) -> Result<Self> {
        if query_heads == 0 || kv_heads == 0 {
            return Err(invalid(
                label,
                format!("head counts must be non-zero, got {query_heads}Q/{kv_heads}KV"),
            ));
        }
        if !query_heads.is_multiple_of(kv_heads) {
            return Err(invalid(
                label,
                format!(
                    "query heads must be divisible by KV heads, got {query_heads}Q/{kv_heads}KV"
                ),
            ));
        }
        if key_head_dim == 0 || value_head_dim == 0 {
            return Err(invalid(
                label,
                format!(
                    "key/value head dimensions must be non-zero, got {key_head_dim}/{value_head_dim}"
                ),
            ));
        }

        let query_width = checked_width(label, "query", query_heads, key_head_dim)?;
        let key_width = checked_width(label, "key", kv_heads, key_head_dim)?;
        let value_width = checked_width(label, "value", kv_heads, value_head_dim)?;
        Ok(Self {
            query_heads,
            kv_heads,
            key_head_dim,
            value_head_dim,
            query_width,
            key_width,
            value_width,
        })
    }

    /// Resolve a shared Q/K/V head dimension, inferring it from hidden size
    /// only when the checkpoint does not provide an explicit value.
    pub(crate) fn from_hidden_size(
        label: &str,
        hidden_size: usize,
        query_heads: usize,
        kv_heads: usize,
        explicit_head_dim: Option<usize>,
    ) -> Result<Self> {
        if hidden_size == 0 {
            return Err(invalid(label, "hidden size must be non-zero"));
        }
        if query_heads == 0 || kv_heads == 0 {
            return Self::new(label, query_heads, kv_heads, 1, 1);
        }
        let head_dim = match explicit_head_dim {
            Some(head_dim) => head_dim,
            None => {
                if !hidden_size.is_multiple_of(query_heads) {
                    return Err(invalid(
                        label,
                        format!(
                            "hidden size {hidden_size} is not divisible by {query_heads} query heads"
                        ),
                    ));
                }
                hidden_size / query_heads
            }
        };
        Self::new(label, query_heads, kv_heads, head_dim, head_dim)
    }

    pub(crate) fn validate_rotary_dim(&self, label: &str, rotary_dim: usize) -> Result<()> {
        if rotary_dim == 0 || rotary_dim > self.key_head_dim || !rotary_dim.is_multiple_of(2) {
            return Err(invalid(
                label,
                format!(
                    "rotary dimension must be non-zero, even, and no larger than key head dimension {}; got {rotary_dim}",
                    self.key_head_dim
                ),
            ));
        }
        Ok(())
    }

    pub(crate) const fn query_heads(self) -> usize {
        self.query_heads
    }

    pub(crate) const fn kv_heads(self) -> usize {
        self.kv_heads
    }

    pub(crate) const fn key_head_dim(self) -> usize {
        self.key_head_dim
    }

    pub(crate) const fn value_head_dim(self) -> usize {
        self.value_head_dim
    }

    pub(crate) const fn query_width(self) -> usize {
        self.query_width
    }

    pub(crate) const fn key_width(self) -> usize {
        self.key_width
    }

    pub(crate) const fn value_width(self) -> usize {
        self.value_width
    }

    pub(crate) const fn kv_groups(self) -> usize {
        self.query_heads / self.kv_heads
    }
}

fn checked_width(label: &str, projection: &str, heads: usize, head_dim: usize) -> Result<usize> {
    heads.checked_mul(head_dim).ok_or_else(|| {
        invalid(
            label,
            format!("{projection} projection width overflows usize ({heads} * {head_dim})"),
        )
    })
}

fn invalid(label: &str, detail: impl AsRef<str>) -> Error {
    Error::ModelLoadError(format!(
        "{label} attention geometry is invalid: {}",
        detail.as_ref()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_realistic_grouped_query_geometry() {
        let geometry = AttentionGeometry::new("test", 32, 8, 128, 128).unwrap();
        assert_eq!(geometry.query_heads(), 32);
        assert_eq!(geometry.kv_heads(), 8);
        assert_eq!(geometry.kv_groups(), 4);
        assert_eq!(geometry.query_width(), 4096);
        assert_eq!(geometry.key_width(), 1024);
        assert_eq!(geometry.value_width(), 1024);
        geometry.validate_rotary_dim("test", 128).unwrap();
    }

    #[test]
    fn rejects_zero_and_non_divisible_head_counts() {
        let zero = AttentionGeometry::new("test", 0, 0, 128, 128).unwrap_err();
        assert!(zero.to_string().contains("head counts must be non-zero"));

        let non_divisible = AttentionGeometry::new("test", 12, 5, 128, 128).unwrap_err();
        assert!(non_divisible
            .to_string()
            .contains("query heads must be divisible by KV heads"));
    }

    #[test]
    fn inference_and_rotary_validation_are_fallible() {
        let inference = AttentionGeometry::from_hidden_size("test", 10, 4, 2, None).unwrap_err();
        assert!(inference.to_string().contains("is not divisible"));

        let geometry = AttentionGeometry::from_hidden_size("test", 16, 4, 2, None).unwrap();
        let rotary = geometry.validate_rotary_dim("test", 3).unwrap_err();
        assert!(rotary.to_string().contains("rotary dimension"));
    }

    #[test]
    fn rejects_projection_width_overflow() {
        let error = AttentionGeometry::new("test", usize::MAX, 1, 2, 2).unwrap_err();
        assert!(error.to_string().contains("projection width overflows"));
    }
}
