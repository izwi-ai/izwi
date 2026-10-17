//! Logical sliding-window page rotation for managed cache tables.
//!
//! Advancing a window drops only pages that are wholly before the new logical
//! start. A partially visible first page remains in place; no K/V rerotation or
//! large physical copy is required.

use thiserror::Error;

/// Physical page operations required to move one committed table forward.
///
/// The retained pages are always a suffix of the old table. `released_pages`
/// therefore names only pages whose complete logical ranges end at or before
/// the new window start. One retained tail may need to become writable when
/// newly committed tokens extend a partially-filled page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvWindowPagePlan {
    pub released_pages: usize,
    pub retained_pages: usize,
    pub fresh_pages: usize,
    pub writable_retained_page: Option<usize>,
    pub first_logical_page: u32,
    pub first_page_offset: u32,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum KvWindowError {
    #[error("KV page size must be non-zero")]
    ZeroPageSize,
    #[error("sliding-window positions must satisfy old_start <= new_start <= committed_tokens")]
    InvalidRange,
    #[error("sliding-window committed tokens cannot move backwards")]
    CommittedTokenRegression,
    #[error("block table has {actual} pages, expected {expected} for its logical range")]
    InvalidTable { expected: usize, actual: usize },
}

/// Derive the exact suffix retention/allocation plan for an atomic append and
/// window advance. This function deals only in logical positions; generation,
/// ownership, and snapshot-version checks remain the coordinator's authority.
pub fn plan_window_step(
    old_window_start: u32,
    old_committed_tokens: u32,
    new_window_start: u32,
    new_committed_tokens: u32,
    page_tokens: u32,
) -> Result<KvWindowPagePlan, KvWindowError> {
    if page_tokens == 0 {
        return Err(KvWindowError::ZeroPageSize);
    }
    if new_committed_tokens < old_committed_tokens {
        return Err(KvWindowError::CommittedTokenRegression);
    }
    if old_window_start > old_committed_tokens
        || old_window_start > new_window_start
        || new_window_start > new_committed_tokens
    {
        return Err(KvWindowError::InvalidRange);
    }

    let old_first = old_window_start / page_tokens;
    let old_end = if old_window_start == old_committed_tokens {
        old_first
    } else {
        old_committed_tokens.div_ceil(page_tokens)
    };
    let new_first = new_window_start / page_tokens;
    let new_end = if new_window_start == new_committed_tokens {
        new_first
    } else {
        new_committed_tokens.div_ceil(page_tokens)
    };

    let retained_start = old_first.max(new_first).min(old_end);
    let released_pages = retained_start.saturating_sub(old_first) as usize;
    let retained_pages = old_end.saturating_sub(retained_start) as usize;
    let fresh_start = old_end.max(new_first);
    let fresh_pages = new_end.saturating_sub(fresh_start) as usize;

    let old_tail_is_retained = retained_pages != 0 && old_end > new_first;
    let writable_retained_page = (new_committed_tokens > old_committed_tokens
        && !old_committed_tokens.is_multiple_of(page_tokens)
        && old_tail_is_retained)
        .then_some(retained_pages - 1);

    debug_assert_eq!(
        retained_pages + fresh_pages,
        pages_for_logical_range(new_window_start, new_committed_tokens, page_tokens)?
    );

    Ok(KvWindowPagePlan {
        released_pages,
        retained_pages,
        fresh_pages,
        writable_retained_page,
        first_logical_page: new_first,
        first_page_offset: new_window_start % page_tokens,
    })
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn append_plan_retains_only_the_intersecting_suffix() {
        let plan = plan_window_step(1, 10, 5, 13, 4).unwrap();
        assert_eq!(plan.released_pages, 1);
        assert_eq!(plan.retained_pages, 2);
        assert_eq!(plan.fresh_pages, 1);
        assert_eq!(plan.writable_retained_page, Some(1));
        assert_eq!(plan.first_logical_page, 1);
        assert_eq!(plan.first_page_offset, 1);
    }

    #[test]
    fn window_step_rejects_position_regressions() {
        assert_eq!(
            plan_window_step(0, 8, 0, 7, 4).unwrap_err(),
            KvWindowError::CommittedTokenRegression
        );
        assert_eq!(
            plan_window_step(4, 8, 3, 9, 4).unwrap_err(),
            KvWindowError::InvalidRange
        );
    }
}
