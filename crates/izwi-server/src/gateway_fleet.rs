//! Fleet configuration for multi-gateway shared-state profiles.
//!
//! When multiple gateways serve the same fleet, each gateway must own a
//! strict, non-overlapping partition of the tenant rate and concurrency
//! budgets. This module provides conservatively partitioned quota
//! configuration so that N gateways each get 1/N of the configured limit
//! without requiring a shared atomic counter or coordination service.
//!
//! Partitioning is the simplest safe multi-gateway strategy: no shared
//! state is required, and the sum of all partitions cannot exceed the
//! total configured budget. A gateway that crashes releases its
//! partition immediately (its process-local state is lost), which is
//! safe because the other gateways' partitions are unaffected.

use std::fmt;

const FLEET_PARTITION_ENV: &str = "IZWI_GATEWAY_FLEET_PARTITION";
const FLEET_SIZE_ENV: &str = "IZWI_GATEWAY_FLEET_SIZE";
const MAX_ENV_VALUE_BYTES: usize = 20;
const MAX_FLEET_SIZE: u32 = 256;

/// A bounded fleet partition index and total size.
///
/// Format: `IZWI_GATEWAY_FLEET_PARTITION=0` and
/// `IZWI_GATEWAY_FLEET_SIZE=2` means this gateway is partition 0 of 2.
/// Partition indices are 0-based and must be less than the fleet size.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct FleetPartition {
    index: u32,
    size: u32,
}

impl FleetPartition {
    pub fn new(index: u32, size: u32) -> Result<Self, FleetPartitionError> {
        if size == 0 {
            return Err(FleetPartitionError::InvalidSize);
        }
        if index >= size {
            return Err(FleetPartitionError::IndexOutOfRange);
        }
        if size > MAX_FLEET_SIZE {
            return Err(FleetPartitionError::SizeTooLarge);
        }
        Ok(Self { index, size })
    }

    pub fn from_env() -> Result<Option<Self>, FleetPartitionError> {
        let Some(size_raw) = std::env::var_os(FLEET_SIZE_ENV) else {
            if std::env::var_os(FLEET_PARTITION_ENV).is_some() {
                return Err(FleetPartitionError::PartitionWithoutSize);
            }
            return Ok(None);
        };
        let size =
            parse_bounded_u32(&size_raw, FLEET_SIZE_ENV).ok_or(FleetPartitionError::InvalidSize)?;
        let index_raw =
            std::env::var_os(FLEET_PARTITION_ENV).ok_or(FleetPartitionError::MissingPartition)?;
        let index = parse_bounded_u32(&index_raw, FLEET_PARTITION_ENV)
            .ok_or(FleetPartitionError::InvalidIndex)?;
        Self::new(index, size).map(Some)
    }

    pub fn index(self) -> u32 {
        self.index
    }

    pub fn size(self) -> u32 {
        self.size
    }

    /// Divide a per-tenant limit by the fleet size, ensuring at least 1.
    pub fn partition_limit(self, total: u32) -> u32 {
        if self.size == 0 {
            return total;
        }
        (total / self.size).max(1)
    }

    /// Divide a concurrency limit by the fleet size, ensuring at least 1.
    pub fn partition_concurrency(self, total: u32) -> u32 {
        if self.size == 0 {
            return total;
        }
        (total / self.size).max(1)
    }
}

impl fmt::Debug for FleetPartition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FleetPartition({}/{})", self.index, self.size)
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum FleetPartitionError {
    #[error("IZWI_GATEWAY_FLEET_SIZE must be a bounded non-zero integer")]
    InvalidSize,
    #[error("IZWI_GATEWAY_FLEET_SIZE must not exceed {MAX_FLEET_SIZE}")]
    SizeTooLarge,
    #[error("IZWI_GATEWAY_FLEET_PARTITION must be a bounded non-negative integer")]
    InvalidIndex,
    #[error("IZWI_GATEWAY_FLEET_PARTITION must be less than IZWI_GATEWAY_FLEET_SIZE")]
    IndexOutOfRange,
    #[error("IZWI_GATEWAY_FLEET_PARTITION is set but IZWI_GATEWAY_FLEET_SIZE is missing")]
    PartitionWithoutSize,
    #[error("IZWI_GATEWAY_FLEET_SIZE is set but IZWI_GATEWAY_FLEET_PARTITION is missing")]
    MissingPartition,
}

fn parse_bounded_u32(os_value: &std::ffi::OsStr, _name: &str) -> Option<u32> {
    let s = os_value.to_str()?;
    if s.is_empty() || s.len() > MAX_ENV_VALUE_BYTES || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_env_returns_none() {
        std::env::remove_var(FLEET_PARTITION_ENV);
        std::env::remove_var(FLEET_SIZE_ENV);
        assert!(FleetPartition::from_env().unwrap().is_none());
    }

    #[test]
    fn valid_partition_parses() {
        std::env::set_var(FLEET_SIZE_ENV, "2");
        std::env::set_var(FLEET_PARTITION_ENV, "1");
        let p = FleetPartition::from_env().unwrap().unwrap();
        assert_eq!(p.index(), 1);
        assert_eq!(p.size(), 2);
        std::env::remove_var(FLEET_SIZE_ENV);
        std::env::remove_var(FLEET_PARTITION_ENV);
    }

    #[test]
    fn partition_without_size_is_error() {
        std::env::remove_var(FLEET_SIZE_ENV);
        std::env::set_var(FLEET_PARTITION_ENV, "0");
        assert_eq!(
            FleetPartition::from_env(),
            Err(FleetPartitionError::PartitionWithoutSize)
        );
        std::env::remove_var(FLEET_PARTITION_ENV);
    }

    #[test]
    fn size_without_partition_is_error() {
        std::env::set_var(FLEET_SIZE_ENV, "2");
        std::env::remove_var(FLEET_PARTITION_ENV);
        assert_eq!(
            FleetPartition::from_env(),
            Err(FleetPartitionError::MissingPartition)
        );
        std::env::remove_var(FLEET_SIZE_ENV);
    }

    #[test]
    fn index_out_of_range_is_rejected() {
        assert_eq!(
            FleetPartition::new(2, 2),
            Err(FleetPartitionError::IndexOutOfRange)
        );
        assert_eq!(
            FleetPartition::new(0, 0),
            Err(FleetPartitionError::InvalidSize)
        );
    }

    #[test]
    fn partition_limit_divides_and_floors_to_one() {
        let p = FleetPartition::new(0, 3).unwrap();
        assert_eq!(p.partition_limit(600), 200);
        assert_eq!(p.partition_limit(1), 1);
        assert_eq!(p.partition_limit(5), 1);
    }

    #[test]
    fn partition_concurrency_divides_and_floors_to_one() {
        let p = FleetPartition::new(1, 4).unwrap();
        assert_eq!(p.partition_concurrency(8), 2);
        assert_eq!(p.partition_concurrency(1), 1);
    }
}
