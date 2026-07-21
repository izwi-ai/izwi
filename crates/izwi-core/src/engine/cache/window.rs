//! Logical sliding-window page rotation for managed cache tables.
//!
//! Advancing a window drops only pages that are wholly before the new logical
//! start. A partially visible first page remains in place; no K/V rerotation or
//! large physical copy is required.

use thiserror::Error;

use crate::kv::CacheBlockRef;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvWindowAdvance {
    pub window_start: u32,
    /// Absolute logical page represented by `visible_blocks[0]`.
    pub first_logical_page: u32,
    /// Token offset within the first visible page. Direct attention metadata
    /// must carry this value when it is non-zero.
    pub first_page_offset: u32,
    pub visible_blocks: Vec<CacheBlockRef>,
    pub released_blocks: Vec<CacheBlockRef>,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum KvWindowError {
    #[error("KV page size must be non-zero")]
    ZeroPageSize,
    #[error("sliding-window positions must satisfy old_start <= new_start <= committed_tokens")]
    InvalidRange,
    #[error("block table has {actual} pages, expected {expected} for its logical range")]
    InvalidTable { expected: usize, actual: usize },
    #[error("the selected model/backend table ABI cannot represent a non-zero first-page offset")]
    OffsetMetadataUnsupported,
}

pub fn pages_for_logical_range(
    window_start: u32,
    committed_tokens: u32,
    page_tokens: u32,
) -> Result<usize, KvWindowError> {
    if page_tokens == 0 {
        return Err(KvWindowError::ZeroPageSize);
    }
    if window_start > committed_tokens {
        return Err(KvWindowError::InvalidRange);
    }
    if window_start == committed_tokens {
        return Ok(0);
    }
    let first_page = window_start / page_tokens;
    let end_page = committed_tokens.div_ceil(page_tokens);
    Ok((end_page - first_page) as usize)
}

pub fn advance_window(
    blocks: &[CacheBlockRef],
    old_window_start: u32,
    new_window_start: u32,
    committed_tokens: u32,
    page_tokens: u32,
) -> Result<KvWindowAdvance, KvWindowError> {
    if old_window_start > new_window_start || new_window_start > committed_tokens {
        return Err(KvWindowError::InvalidRange);
    }
    let expected = pages_for_logical_range(old_window_start, committed_tokens, page_tokens)?;
    if blocks.len() != expected {
        return Err(KvWindowError::InvalidTable {
            expected,
            actual: blocks.len(),
        });
    }

    let expected_visible =
        pages_for_logical_range(new_window_start, committed_tokens, page_tokens)?;
    let release_count = blocks.len().saturating_sub(expected_visible);
    let (released, visible) = blocks.split_at(release_count);
    if visible.len() != expected_visible {
        return Err(KvWindowError::InvalidTable {
            expected: expected_visible,
            actual: visible.len(),
        });
    }

    Ok(KvWindowAdvance {
        window_start: new_window_start,
        first_logical_page: new_window_start / page_tokens,
        first_page_offset: new_window_start % page_tokens,
        visible_blocks: visible.to_vec(),
        released_blocks: released.to_vec(),
    })
}

/// Fail closed until a model adapter's direct-attention metadata can carry the
/// offset into a retained partial first page. Page-aligned window starts do not
/// require the extra field.
pub fn validate_window_offset_support(
    advance: &KvWindowAdvance,
    supports_first_page_offset: bool,
) -> Result<(), KvWindowError> {
    if advance.first_page_offset != 0 && !supports_first_page_offset {
        return Err(KvWindowError::OffsetMetadataUnsupported);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::backends::BackendKind;
    use crate::engine::ModelInstanceId;
    use crate::kv::{KvArenaId, KvGroupId};

    use super::*;

    fn block(index: u32) -> CacheBlockRef {
        CacheBlockRef {
            arena: KvArenaId {
                model_instance: ModelInstanceId::new(1),
                backend: BackendKind::Cpu,
                device_ordinal: None,
                generation: 1,
            },
            group: KvGroupId::new(0),
            index,
            slot_generation: 1,
        }
    }

    #[test]
    fn partial_first_page_is_retained_without_physical_rotation() {
        let blocks = vec![block(2), block(7), block(3)];
        let advanced = advance_window(&blocks, 0, 5, 12, 4).unwrap();
        assert_eq!(advanced.released_blocks, vec![block(2)]);
        assert_eq!(advanced.visible_blocks, vec![block(7), block(3)]);
        assert_eq!(advanced.first_logical_page, 1);
        assert_eq!(advanced.first_page_offset, 1);
        assert_eq!(
            validate_window_offset_support(&advanced, false).unwrap_err(),
            KvWindowError::OffsetMetadataUnsupported
        );
        validate_window_offset_support(&advanced, true).unwrap();
    }

    #[test]
    fn memory_plateaus_at_window_pages_plus_partial_boundaries() {
        let page_tokens = 16;
        let window_tokens = 64;
        for committed in 1..=4096_u32 {
            let start = committed.saturating_sub(window_tokens);
            let pages = pages_for_logical_range(start, committed, page_tokens).unwrap();
            assert!(pages <= window_tokens.div_ceil(page_tokens) as usize + 1);
        }
    }

    #[test]
    fn malformed_tables_and_window_rewinds_fail_closed() {
        assert_eq!(
            advance_window(&[block(0)], 8, 4, 16, 4).unwrap_err(),
            KvWindowError::InvalidRange
        );
        assert_eq!(
            advance_window(&[block(0)], 0, 0, 8, 4).unwrap_err(),
            KvWindowError::InvalidTable {
                expected: 2,
                actual: 1
            }
        );
    }

    #[test]
    fn advancing_to_a_non_aligned_sequence_end_releases_every_page() {
        let advanced = advance_window(&[block(0), block(1), block(2)], 0, 10, 10, 4).unwrap();
        assert_eq!(advanced.released_blocks, vec![block(0), block(1), block(2)]);
        assert!(advanced.visible_blocks.is_empty());
        assert_eq!(advanced.first_logical_page, 2);
        assert_eq!(advanced.first_page_offset, 2);
    }
}
