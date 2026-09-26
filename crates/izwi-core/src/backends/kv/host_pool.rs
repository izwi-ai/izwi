//! Budget-bounded host-side page store for hierarchical KV offload (DS4).
//!
//! The pool holds whole pages in the arena's page geometry — per layer, one
//! key block and one value block of `[page_tokens, kv_heads, head_dim]` — as
//! plain host bytes. It is deliberately independent of any backend device
//! allocator: capturing and restoring page bytes is the arena's job
//! (`KvArena::capture_page` / `restore_page`), the pool only owns storage and
//! the byte budget. Budget accounting folds into the engine's resource
//! authority by the manager reserving the pool's bytes, so on unified-memory
//! hosts (Metal) the pool is charged to the shared host/unified ledger
//! (DINV-05) rather than pretending to add capacity.

use crate::{Error, Result};

/// Byte-backed storage for offloaded KV pages, bounded by an explicit budget.
#[derive(Debug)]
pub struct KvHostPool {
    page_bytes: u64,
    budget_bytes: u64,
    resident_bytes: u64,
    slots: Vec<Option<Box<[u8]>>>,
    free: Vec<usize>,
}

impl KvHostPool {
    /// Creates a pool holding whole pages of `page_bytes` each, refusing to
    /// hold more than `budget_bytes` at once.
    pub fn new(page_bytes: u64, budget_bytes: u64) -> Result<Self> {
        if page_bytes == 0 {
            return Err(Error::InvalidInput(
                "KV host pool page size must be positive".to_string(),
            ));
        }
        Ok(Self {
            page_bytes,
            budget_bytes,
            resident_bytes: 0,
            slots: Vec::new(),
            free: Vec::new(),
        })
    }

    pub fn page_bytes(&self) -> u64 {
        self.page_bytes
    }

    pub fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }

    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub fn resident_pages(&self) -> usize {
        self.slots.len() - self.free.len()
    }

    /// Whether `count` additional pages fit inside the budget.
    pub fn can_admit_pages(&self, count: usize) -> bool {
        let wanted = self
            .page_bytes
            .saturating_mul(count as u64)
            .saturating_add(self.resident_bytes);
        self.budget_bytes >= wanted
    }

    /// Remaining page capacity under the budget.
    pub fn free_page_capacity(&self) -> usize {
        if self.page_bytes == 0 {
            return 0;
        }
        ((self.budget_bytes - self.resident_bytes) / self.page_bytes) as usize
    }

    /// Allocates a zeroed page slot, or `None` when the budget is exhausted.
    pub fn allocate_slot(&mut self) -> Option<usize> {
        if !self.can_admit_pages(1) {
            return None;
        }
        let slot = match self.free.pop() {
            Some(slot) => slot,
            None => {
                let slot = self.slots.len();
                self.slots.push(None);
                slot
            }
        };
        self.slots[slot] = Some(vec![0_u8; self.page_bytes as usize].into_boxed_slice());
        self.resident_bytes += self.page_bytes;
        Some(slot)
    }

    pub fn page(&self, slot: usize) -> Result<&[u8]> {
        self.slots
            .get(slot)
            .and_then(|entry| entry.as_deref())
            .ok_or_else(|| Error::InvalidInput(format!("KV host pool slot {slot} is not live")))
    }

    pub fn page_mut(&mut self, slot: usize) -> Result<&mut [u8]> {
        self.slots
            .get_mut(slot)
            .and_then(|entry| entry.as_deref_mut())
            .ok_or_else(|| Error::InvalidInput(format!("KV host pool slot {slot} is not live")))
    }

    /// Releases a slot back to the pool, freeing its budget charge.
    pub fn release_slot(&mut self, slot: usize) -> Result<()> {
        match self.slots.get_mut(slot) {
            Some(entry @ Some(_)) => {
                *entry = None;
                self.resident_bytes = self
                    .resident_bytes
                    .checked_sub(self.page_bytes)
                    .ok_or_else(|| {
                        Error::InvalidInput("KV host pool byte accounting underflow".to_string())
                    })?;
                self.free.push(slot);
                Ok(())
            }
            _ => Err(Error::InvalidInput(format!(
                "KV host pool slot {slot} is not live"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocation_is_bounded_by_the_budget() {
        let mut pool = KvHostPool::new(128, 3 * 128).unwrap();
        let first = pool.allocate_slot().expect("first slot fits");
        let second = pool.allocate_slot().expect("second slot fits");
        let third = pool.allocate_slot().expect("third slot fits");
        assert_eq!(pool.resident_pages(), 3);
        assert_eq!(pool.resident_bytes(), 3 * 128);
        assert!(pool.allocate_slot().is_none(), "budget is exhausted");
        assert!(!pool.can_admit_pages(1));

        pool.release_slot(second).unwrap();
        assert!(pool.can_admit_pages(1));
        let reused = pool.allocate_slot().expect("freed slot is reused");
        assert_eq!(reused, second, "free slots are recycled");
        assert_ne!(first, third);
        assert_eq!(pool.resident_bytes(), 3 * 128);
    }

    #[test]
    fn released_slots_are_not_readable() {
        let mut pool = KvHostPool::new(64, 64).unwrap();
        let slot = pool.allocate_slot().expect("slot fits");
        pool.page_mut(slot).unwrap()[0] = 7;
        assert_eq!(pool.page(slot).unwrap()[0], 7);
        pool.release_slot(slot).unwrap();
        assert!(pool.page(slot).is_err());
        assert!(pool.page_mut(slot).is_err());
        assert!(pool.release_slot(slot).is_err(), "double release rejected");
    }

    #[test]
    fn zero_budget_admits_nothing_but_stays_constructible() {
        let mut pool = KvHostPool::new(64, 0).unwrap();
        assert!(pool.allocate_slot().is_none());
        assert_eq!(pool.free_page_capacity(), 0);
    }

    #[test]
    fn free_page_capacity_counts_whole_pages_only() {
        let pool = KvHostPool::new(128, 300).unwrap();
        assert_eq!(pool.free_page_capacity(), 2);
    }

    #[test]
    fn zero_page_size_is_rejected() {
        assert!(KvHostPool::new(0, 128).is_err());
    }
}
