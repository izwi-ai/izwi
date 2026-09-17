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
const FLEET_DB_PATH_ENV: &str = "IZWI_GATEWAY_FLEET_DB_PATH";
const GATEWAY_ID_ENV: &str = "IZWI_GATEWAY_ID";
const MAX_ENV_VALUE_BYTES: usize = 20;
const MAX_FLEET_SIZE: u32 = 256;
const MAX_DB_PATH_BYTES: usize = 4096;

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
    #[error("IZWI_GATEWAY_FLEET_DB_PATH must be a bounded absolute path")]
    InvalidDbPath,
    #[error("IZWI_GATEWAY_ID exceeds its encoded size limit")]
    InvalidGatewayId,
}

/// Resolve the shared fleet coordination database path, if configured.
/// All gateways pointing at the same file share worker observations and
/// capacity claims. Unset means single-gateway operation with no shared
/// state at all.
pub fn fleet_db_path_from_env() -> Result<Option<std::path::PathBuf>, FleetPartitionError> {
    let Some(raw) = std::env::var_os(FLEET_DB_PATH_ENV) else {
        return Ok(None);
    };
    let raw = raw
        .into_string()
        .map_err(|_| FleetPartitionError::InvalidDbPath)?;
    if raw.is_empty() || raw.len() > MAX_DB_PATH_BYTES {
        return Err(FleetPartitionError::InvalidDbPath);
    }
    let path = std::path::PathBuf::from(raw);
    if !path.is_absolute() {
        return Err(FleetPartitionError::InvalidDbPath);
    }
    Ok(Some(path))
}

/// Stable gateway identity for fleet claim ownership. Operator-set via
/// `IZWI_GATEWAY_ID`; otherwise unique per process boot. A restarted
/// gateway never inherits its predecessor's in-flight claims — TTL expiry
/// and worker-authoritative teardown own that recovery.
pub fn gateway_identity() -> String {
    match std::env::var(GATEWAY_ID_ENV) {
        Ok(id)
            if !id.is_empty()
                && id.len() <= 128
                && id
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_')) =>
        {
            id
        }
        _ => format!("gateway-{}", std::process::id()),
    }
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

    #[test]
    fn fleet_db_path_requires_absolute_bounded_paths() {
        std::env::remove_var("IZWI_GATEWAY_FLEET_DB_PATH");
        assert!(fleet_db_path_from_env().unwrap().is_none());
        std::env::set_var("IZWI_GATEWAY_FLEET_DB_PATH", "relative/fleet.sqlite3");
        assert_eq!(
            fleet_db_path_from_env(),
            Err(FleetPartitionError::InvalidDbPath)
        );
        std::env::set_var("IZWI_GATEWAY_FLEET_DB_PATH", "/var/lib/izwi/fleet.sqlite3");
        assert_eq!(
            fleet_db_path_from_env().unwrap().unwrap().to_str(),
            Some("/var/lib/izwi/fleet.sqlite3")
        );
        std::env::remove_var("IZWI_GATEWAY_FLEET_DB_PATH");
    }

    #[test]
    fn gateway_identity_prefers_bounded_operator_value() {
        std::env::set_var("IZWI_GATEWAY_ID", "gateway-east-1");
        assert_eq!(gateway_identity(), "gateway-east-1");
        std::env::set_var("IZWI_GATEWAY_ID", "not valid!!");
        assert!(gateway_identity().starts_with("gateway-"));
        std::env::remove_var("IZWI_GATEWAY_ID");
    }
}
