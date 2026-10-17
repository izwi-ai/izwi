//! Per-tenant and global record quotas kept in `runtime_quota_usage`.

use super::*;

/// Bounded records counted in `runtime_quota_usage`, per tenant and in total.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum QuotaKind {
    DurableIdempotency,
    ProviderWrite,
    ArtifactCleanup,
}

impl QuotaKind {
    pub(super) fn as_db_value(self) -> &'static str {
        match self {
            Self::DurableIdempotency => "durable_idempotency",
            Self::ProviderWrite => "provider_write",
            Self::ArtifactCleanup => "artifact_cleanup",
        }
    }
}

/// Caps on one quota kind; each is at least one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct QuotaLimits {
    pub(crate) global: u64,
    pub(crate) per_tenant: u64,
}

impl QuotaLimits {
    pub(super) fn from_env(global_key: &str, global: u64, tenant_key: &str, per_tenant: u64) -> Self {
        let read = |key: &str, default: u64| {
            std::env::var(key)
                .ok()
                .and_then(|value| value.trim().parse::<u64>().ok())
                .filter(|value| *value > 0)
                .unwrap_or(default)
        };
        Self {
            global: read(global_key, global),
            per_tenant: read(tenant_key, per_tenant),
        }
    }
}

/// The scope of the row that counts every tenant; tenant scopes are never
/// empty.
pub(super) const GLOBAL_QUOTA_SCOPE: &str = "";

pub(super) async fn ensure_quota_row_with<C: ConnectionTrait>(
    db: &C,
    kind: QuotaKind,
    scope: &str,
) -> anyhow::Result<()> {
    let sql = match db.get_database_backend() {
        DbBackend::Sqlite | DbBackend::Postgres => {
            "INSERT INTO runtime_quota_usage (quota_kind, scope, used) VALUES (?1, ?2, 0) ON CONFLICT (quota_kind, scope) DO NOTHING"
        }
        backend => bail!("Unsupported quota database backend: {backend:?}"),
    };
    db.execute_raw(raw::statement(
        db,
        sql,
        vec![kind.as_db_value().into(), scope.into()],
    )?)
    .await
    .context("Failed to initialize quota usage")?;
    Ok(())
}

/// Take one unit of `kind` for `tenant` when both its tenant cap and the
/// global cap allow it. Every transaction locks quota rows in one order,
/// tenants before the global row, so PostgreSQL cannot deadlock on them.
pub(super) async fn acquire_quota_with<C: ConnectionTrait>(
    db: &C,
    kind: QuotaKind,
    tenant: &str,
    limits: QuotaLimits,
) -> anyhow::Result<bool> {
    for (scope, cap) in [
        (tenant, limits.per_tenant),
        (GLOBAL_QUOTA_SCOPE, limits.global),
    ] {
        ensure_quota_row_with(db, kind, scope).await?;
        let taken = db
            .execute_raw(raw::statement(
                db,
                "UPDATE runtime_quota_usage SET used = used + 1 WHERE quota_kind = ?1 AND scope = ?2 AND used < ?3",
                vec![
                    kind.as_db_value().into(),
                    scope.into(),
                    i64::try_from(cap)?.into(),
                ],
            )?)
            .await
            .context("Failed to take quota")?
            .rows_affected()
            == 1;
        if !taken {
            if scope == GLOBAL_QUOTA_SCOPE {
                adjust_quota_with(db, kind, tenant, 1).await?;
            }
            return Ok(false);
        }
    }
    Ok(true)
}

/// Give back `count` units of `tenant`'s quota and of the global quota.
pub(super) async fn release_quota_with<C: ConnectionTrait>(
    db: &C,
    kind: QuotaKind,
    tenant: &str,
    count: u64,
) -> anyhow::Result<()> {
    release_quotas_with(
        db,
        kind,
        &std::collections::BTreeMap::from([(tenant.to_string(), count)]),
    )
    .await
}

/// Give back quota for several tenants: tenant rows in sorted order, then
/// the global row once.
pub(super) async fn release_quotas_with<C: ConnectionTrait>(
    db: &C,
    kind: QuotaKind,
    released: &std::collections::BTreeMap<String, u64>,
) -> anyhow::Result<()> {
    let mut total = 0_u64;
    for (tenant, count) in released {
        adjust_quota_with(db, kind, tenant, *count).await?;
        total = total.saturating_add(*count);
    }
    adjust_quota_with(db, kind, GLOBAL_QUOTA_SCOPE, total).await
}

pub(super) async fn adjust_quota_with<C: ConnectionTrait>(
    db: &C,
    kind: QuotaKind,
    scope: &str,
    released: u64,
) -> anyhow::Result<()> {
    if released == 0 {
        return Ok(());
    }
    db.execute_raw(raw::statement(
        db,
        "UPDATE runtime_quota_usage SET used = CASE WHEN used > ?3 THEN used - ?3 ELSE 0 END WHERE quota_kind = ?1 AND scope = ?2",
        vec![
            kind.as_db_value().into(),
            scope.into(),
            i64::try_from(released)?.into(),
        ],
    )?)
    .await
    .context("Failed to release quota")?;
    Ok(())
}
