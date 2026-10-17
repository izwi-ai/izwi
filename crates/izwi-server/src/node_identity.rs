//! The identity of this server node on a database other nodes may share.
//!
//! Diarization jobs, streamed transcriptions and streamed speech run as tasks
//! of the process that accepted them, so a restart must fail the ones it left
//! running. On a database several nodes share, "left running" means "left
//! running by this node": failing every pending record would fail jobs other
//! nodes are still running. Records carry the node that runs them
//! (`owner_node_id`), and startup recovery fails only its own.

use sea_orm::DbBackend;

/// Names this node on a shared database. Each node must use its own, stable
/// across restarts (for example the host or pod name).
pub const NODE_ID_ENV: &str = "IZWI_NODE_ID";
/// The node id when `IZWI_NODE_ID` is unset: a single local node.
const DEFAULT_NODE_ID: &str = "local";
const MAX_NODE_ID_BYTES: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeIdentity {
    id: String,
    configured: bool,
}

/// Which interrupted in-process records a node may fail at startup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveryScope {
    /// Records this node owns.
    pub owner: String,
    /// Also records with no owner, written before owners were recorded.
    pub include_unowned: bool,
}

impl NodeIdentity {
    /// `IZWI_NODE_ID` when it is set and valid, else the single local node.
    pub fn from_env() -> Self {
        match std::env::var(NODE_ID_ENV) {
            Ok(raw) if !raw.is_empty() => Self::configured(&raw).unwrap_or_else(|| {
                tracing::warn!(
                    "{NODE_ID_ENV} must be 1-{MAX_NODE_ID_BYTES} ASCII letters, digits, '-' \
                     or '_'; ignoring it"
                );
                Self::local()
            }),
            _ => Self::local(),
        }
    }

    /// The single local node (`IZWI_NODE_ID` unset).
    pub fn local() -> Self {
        Self {
            id: DEFAULT_NODE_ID.to_string(),
            configured: false,
        }
    }

    /// A node an operator named; `None` when `id` is not a valid node id.
    pub fn configured(id: &str) -> Option<Self> {
        let valid = !id.is_empty()
            && id.len() <= MAX_NODE_ID_BYTES
            && id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_'));
        valid.then(|| Self {
            id: id.to_string(),
            configured: true,
        })
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    /// The records this node may fail at startup on `backend`, or `None`
    /// when it cannot tell its records from another node's.
    ///
    /// A SQLite database belongs to one node, so it also recovers records
    /// written before owners were recorded. Any other database may be shared:
    /// there a node recovers only the records it owns, and only once an
    /// operator has named it, since every unnamed node is "local".
    pub fn recovery_scope(&self, backend: DbBackend) -> Option<RecoveryScope> {
        let single_node = backend == DbBackend::Sqlite;
        (single_node || self.configured).then(|| RecoveryScope {
            owner: self.id.clone(),
            include_unowned: single_node,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_ids_are_bounded_and_plain() {
        assert_eq!(
            NodeIdentity::configured("node-a_1").map(|node| node.id),
            Some("node-a_1".to_string())
        );
        for invalid in ["", "has space", "slash/", &"x".repeat(129)] {
            assert_eq!(NodeIdentity::configured(invalid), None, "{invalid:?}");
        }
    }

    #[test]
    fn env_names_the_node_and_falls_back_to_local() {
        let _guard = crate::test_support::env_lock();
        std::env::set_var(NODE_ID_ENV, "gpu-node-2");
        assert_eq!(
            NodeIdentity::from_env(),
            NodeIdentity::configured("gpu-node-2").unwrap()
        );
        std::env::set_var(NODE_ID_ENV, "not valid!");
        assert_eq!(NodeIdentity::from_env(), NodeIdentity::local());
        std::env::remove_var(NODE_ID_ENV);
        assert_eq!(NodeIdentity::from_env(), NodeIdentity::local());
    }

    /// SQLite recovers its own and legacy unowned records. A shared-capable
    /// database recovers only a named node's own records, and nothing when
    /// the node is unnamed (every unnamed node would be "local").
    #[test]
    fn recovery_scope_never_reaches_another_nodes_records() {
        let local = NodeIdentity::local();
        let named = NodeIdentity::configured("node-a").unwrap();
        assert_eq!(
            local.recovery_scope(DbBackend::Sqlite),
            Some(RecoveryScope {
                owner: "local".to_string(),
                include_unowned: true,
            })
        );
        assert_eq!(
            named.recovery_scope(DbBackend::Sqlite),
            Some(RecoveryScope {
                owner: "node-a".to_string(),
                include_unowned: true,
            })
        );
        for shared in [DbBackend::Postgres, DbBackend::MySql] {
            assert_eq!(local.recovery_scope(shared), None);
            assert_eq!(
                named.recovery_scope(shared),
                Some(RecoveryScope {
                    owner: "node-a".to_string(),
                    include_unowned: false,
                })
            );
        }
    }
}
