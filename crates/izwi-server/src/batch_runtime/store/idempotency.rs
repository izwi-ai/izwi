//! Durable idempotency v2: reservation, commit, replay and pruning.

use super::*;

/// Versioned, bounded identity for one durable create-operation request.
///
/// This contract is intentionally separate from worker-attempt duplicate
/// suppression and does not imply replay for synchronous streaming requests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DurableIdempotencyRequest {
    pub tenant_scope: String,
    pub operation: String,
    pub idempotency_key: String,
    pub digest_version: u16,
    pub request_digest: String,
    pub reservation_ttl_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DurableIdempotencyReservation {
    pub tenant_scope: String,
    pub operation: String,
    pub idempotency_key: String,
    pub digest_version: u16,
    pub request_digest: String,
    pub reservation_token: String,
    pub expires_at: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DurableIdempotencyReplay {
    pub runtime_job_id: String,
    pub response_json: serde_json::Value,
    pub expires_at: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DurableIdempotencyBegin {
    Acquired(DurableIdempotencyReservation),
    Replay(DurableIdempotencyReplay),
    Conflict,
    InProgress { expires_at: u64 },
    CapacityExceeded,
}

pub const DURABLE_IDEMPOTENCY_DIGEST_VERSION: u16 = 1;

pub const MAX_DURABLE_IDEMPOTENCY_TENANT_BYTES: usize = 256;

pub const MAX_DURABLE_IDEMPOTENCY_OPERATION_BYTES: usize = 128;

pub const MAX_DURABLE_IDEMPOTENCY_KEY_BYTES: usize = 256;

pub const MAX_DURABLE_IDEMPOTENCY_CANONICAL_REQUEST_BYTES: usize = 1024 * 1024;

pub const MAX_DURABLE_IDEMPOTENCY_RESULT_BYTES: usize = 64 * 1024;

pub const MAX_DURABLE_IDEMPOTENCY_RESERVATION_TTL_MS: u64 = 10 * 60 * 1_000;

pub const MAX_DURABLE_IDEMPOTENCY_RETENTION_MS: u64 = 7 * 24 * 60 * 60 * 1_000;

pub(super) const DEFAULT_DURABLE_IDEMPOTENCY_PRUNE_LIMIT: usize = 64;

pub(super) const MAX_DURABLE_IDEMPOTENCY_PRUNE_LIMIT: usize = 512;

pub(super) const MAX_DURABLE_IDEMPOTENCY_JSON_DEPTH: usize = 128;

pub(super) const MAX_DURABLE_IDEMPOTENCY_JSON_OBJECT_KEYS: usize = 16_384;

/// Hash a semantically complete request envelope using the version-one
/// canonical JSON encoding. Object keys are sorted recursively by Rust string
/// order (Unicode scalar value order); array order, scalar types, and values
/// remain significant. Uploads must be represented in the envelope by their
/// verified content digest rather than a temporary path.
pub fn canonical_request_digest(request: &serde_json::Value) -> anyhow::Result<String> {
    let mut canonical = BoundedJsonBytes::new(MAX_DURABLE_IDEMPOTENCY_CANONICAL_REQUEST_BYTES);
    write_canonical_json(request, &mut canonical, 0)?;
    Ok(sha256_hex(&canonical.bytes))
}

#[derive(Debug)]
pub(super) struct StoredDurableIdempotency {
    expires_at: u64,
    digest_version: u16,
    request_digest: String,
    state: String,
    reservation_token: String,
    runtime_job_id: Option<String>,
    response_json: Option<serde_json::Value>,
}

impl StoredDurableIdempotency {
    fn begin_outcome(
        self,
        request: &DurableIdempotencyRequest,
    ) -> anyhow::Result<DurableIdempotencyBegin> {
        if self.digest_version != request.digest_version
            || self.request_digest != request.request_digest
        {
            return Ok(DurableIdempotencyBegin::Conflict);
        }
        match self.state.as_str() {
            "reserved" => Ok(DurableIdempotencyBegin::InProgress {
                expires_at: self.expires_at,
            }),
            "committed" => Ok(DurableIdempotencyBegin::Replay(DurableIdempotencyReplay {
                runtime_job_id: self.runtime_job_id.ok_or_else(|| {
                    anyhow!("Committed durable idempotency record is missing its runtime job")
                })?,
                response_json: self.response_json.ok_or_else(|| {
                    anyhow!("Committed durable idempotency record is missing its response")
                })?,
                expires_at: self.expires_at,
            })),
            state => bail!("Unknown durable idempotency state: {state}"),
        }
    }
}

pub(super) fn validate_durable_idempotency_request(request: &DurableIdempotencyRequest) -> anyhow::Result<()> {
    validate_durable_idempotency_identity(
        &request.tenant_scope,
        &request.operation,
        &request.idempotency_key,
        request.digest_version,
        &request.request_digest,
    )?;
    anyhow::ensure!(
        (1..=MAX_DURABLE_IDEMPOTENCY_RESERVATION_TTL_MS)
            .contains(&request.reservation_ttl_ms),
        "Durable idempotency reservation TTL must be between 1 and {MAX_DURABLE_IDEMPOTENCY_RESERVATION_TTL_MS} milliseconds"
    );
    Ok(())
}

pub(super) fn validate_durable_idempotency_identity(
    tenant_scope: &str,
    operation: &str,
    idempotency_key: &str,
    digest_version: u16,
    request_digest: &str,
) -> anyhow::Result<()> {
    validate_bounded_field(
        "tenant scope",
        tenant_scope,
        MAX_DURABLE_IDEMPOTENCY_TENANT_BYTES,
    )?;
    validate_bounded_field(
        "operation",
        operation,
        MAX_DURABLE_IDEMPOTENCY_OPERATION_BYTES,
    )?;
    validate_bounded_field(
        "idempotency key",
        idempotency_key,
        MAX_DURABLE_IDEMPOTENCY_KEY_BYTES,
    )?;
    validate_request_digest(digest_version, request_digest)
}

pub(super) fn validate_request_digest(digest_version: u16, request_digest: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        digest_version == DURABLE_IDEMPOTENCY_DIGEST_VERSION,
        "Unsupported durable idempotency digest version: {digest_version}"
    );
    anyhow::ensure!(
        request_digest.len() == 64
            && request_digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "Durable idempotency request digest must be lowercase SHA-256 hex"
    );
    Ok(())
}

pub(super) struct BoundedJsonBytes {
    bytes: Vec<u8>,
    limit: usize,
}

impl BoundedJsonBytes {
    fn new(limit: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(limit.min(8192)),
            limit,
        }
    }
}

impl Write for BoundedJsonBytes {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        if self.bytes.len().saturating_add(buffer.len()) > self.limit {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "bounded JSON encoding exceeded its byte limit",
            ));
        }
        self.bytes.extend_from_slice(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub(super) fn bounded_json_string(
    value: &serde_json::Value,
    limit: usize,
    label: &str,
) -> anyhow::Result<String> {
    let mut encoded = BoundedJsonBytes::new(limit);
    write_canonical_json(value, &mut encoded, 0)
        .with_context(|| format!("{label} exceeds {limit} bytes or could not be encoded"))?;
    String::from_utf8(encoded.bytes).context("JSON serialization produced invalid UTF-8")
}

pub(super) fn write_canonical_json<W: Write>(
    value: &serde_json::Value,
    output: &mut W,
    depth: usize,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        depth <= MAX_DURABLE_IDEMPOTENCY_JSON_DEPTH,
        "Durable idempotency JSON exceeds maximum nesting depth of {MAX_DURABLE_IDEMPOTENCY_JSON_DEPTH}"
    );
    match value {
        serde_json::Value::Null => output.write_all(b"null")?,
        serde_json::Value::Bool(true) => output.write_all(b"true")?,
        serde_json::Value::Bool(false) => output.write_all(b"false")?,
        serde_json::Value::Number(value) => serde_json::to_writer(&mut *output, value)?,
        serde_json::Value::String(value) => serde_json::to_writer(&mut *output, value)?,
        serde_json::Value::Array(values) => {
            output.write_all(b"[")?;
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.write_all(b",")?;
                }
                write_canonical_json(value, output, depth + 1)?;
            }
            output.write_all(b"]")?;
        }
        serde_json::Value::Object(values) => {
            anyhow::ensure!(
                values.len() <= MAX_DURABLE_IDEMPOTENCY_JSON_OBJECT_KEYS,
                "Durable idempotency JSON object exceeds {MAX_DURABLE_IDEMPOTENCY_JSON_OBJECT_KEYS} keys"
            );
            output.write_all(b"{")?;
            let mut keys = values.keys().collect::<Vec<_>>();
            keys.sort_unstable();
            for (index, key) in keys.into_iter().enumerate() {
                if index != 0 {
                    output.write_all(b",")?;
                }
                serde_json::to_writer(&mut *output, key)?;
                output.write_all(b":")?;
                write_canonical_json(&values[key], output, depth + 1)?;
            }
            output.write_all(b"}")?;
        }
    }
    Ok(())
}

pub(super) async fn load_durable_idempotency_with<C: ConnectionTrait>(
    db: &C,
    tenant_scope: &str,
    operation: &str,
    idempotency_key: &str,
) -> anyhow::Result<Option<StoredDurableIdempotency>> {
    let row = db
        .query_one_raw(raw::statement(
            db,
            r#"
            SELECT expires_at, digest_version, request_digest, state,
                   reservation_token, runtime_job_id, response_json
            FROM durable_idempotency_keys_v2
            WHERE tenant_scope = ?1 AND operation = ?2 AND idempotency_key = ?3
            "#,
            vec![
                tenant_scope.into(),
                operation.into(),
                idempotency_key.into(),
            ],
        )?)
        .await
        .context("Failed to load durable idempotency key")?;
    row.map(|row| -> anyhow::Result<StoredDurableIdempotency> {
        let digest_version = u16::try_from(row.try_get_by_index::<i64>(1)?)?;
        let request_digest: String = row.try_get_by_index(2)?;
        validate_request_digest(digest_version, &request_digest)?;
        let state: String = row.try_get_by_index(3)?;
        anyhow::ensure!(
            matches!(state.as_str(), "reserved" | "committed"),
            "Unknown durable idempotency state: {state}"
        );
        let reservation_token: String = row.try_get_by_index(4)?;
        validate_bounded_field("reservation token", &reservation_token, 64)?;
        let runtime_job_id: Option<String> = row.try_get_by_index(5)?;
        if let Some(runtime_job_id) = runtime_job_id.as_deref() {
            validate_bounded_field("runtime job ID", runtime_job_id, 128)?;
        }
        let response_json = match row.try_get_by_index::<Option<String>>(6)? {
            Some(raw) => {
                anyhow::ensure!(
                    raw.len() <= MAX_DURABLE_IDEMPOTENCY_RESULT_BYTES,
                    "Stored durable idempotency result exceeds {MAX_DURABLE_IDEMPOTENCY_RESULT_BYTES} bytes"
                );
                Some(
                    serde_json::from_str(&raw)
                        .context("Failed to parse durable idempotency response")?,
                )
            }
            None => None,
        };
        Ok(StoredDurableIdempotency {
            expires_at: u64::try_from(row.try_get_by_index::<i64>(0)?)?,
            digest_version,
            request_digest,
            state,
            reservation_token,
            runtime_job_id,
            response_json,
        })
    })
    .transpose()
}

/// Whether `reservation` still owns its unexpired, uncommitted row.
pub(super) async fn durable_reservation_is_live_with<C: ConnectionTrait>(
    db: &C,
    reservation: &DurableIdempotencyReservation,
    now: u64,
) -> anyhow::Result<bool> {
    let Some(stored) = load_durable_idempotency_with(
        db,
        &reservation.tenant_scope,
        &reservation.operation,
        &reservation.idempotency_key,
    )
    .await?
    else {
        return Ok(false);
    };
    Ok(stored.state == "reserved"
        && stored.expires_at > now
        && stored.digest_version == reservation.digest_version
        && stored.request_digest == reservation.request_digest
        && stored.reservation_token == reservation.reservation_token)
}

/// Commit a reservation's replay result inside the acceptance transaction
/// that creates its job. False means the reservation was lost or expired.
pub(super) async fn commit_durable_reservation_with<C: ConnectionTrait>(
    db: &C,
    reservation: &DurableIdempotencyReservation,
    now: u64,
    retention_ms: u64,
    runtime_job_id: &str,
    response_json: &str,
) -> anyhow::Result<bool> {
    let expires_at = now
        .checked_add(retention_ms)
        .context("Durable idempotency retention expiry overflow")?;
    let result = db
        .execute_raw(raw::statement(
            db,
            r#"
            UPDATE durable_idempotency_keys_v2
            SET state = 'committed', updated_at = ?1, expires_at = ?2,
                runtime_job_id = ?3, response_json = ?4
            WHERE tenant_scope = ?5 AND operation = ?6
              AND idempotency_key = ?7 AND state = 'reserved'
              AND reservation_token = ?8 AND digest_version = ?9
              AND request_digest = ?10 AND expires_at > ?1
            "#,
            vec![
                i64::try_from(now)?.into(),
                i64::try_from(expires_at)?.into(),
                runtime_job_id.into(),
                response_json.into(),
                reservation.tenant_scope.clone().into(),
                reservation.operation.clone().into(),
                reservation.idempotency_key.clone().into(),
                reservation.reservation_token.clone().into(),
                i64::from(reservation.digest_version).into(),
                reservation.request_digest.clone().into(),
            ],
        )?)
        .await?;
    Ok(result.rows_affected() == 1)
}

pub(super) async fn delete_durable_idempotency_with<C: ConnectionTrait>(
    db: &C,
    tenant_scope: &str,
    operation: &str,
    idempotency_key: &str,
    now: u64,
) -> anyhow::Result<bool> {
    let result = db
        .execute_raw(raw::statement(
            db,
            r#"
            DELETE FROM durable_idempotency_keys_v2
            WHERE tenant_scope = ?1 AND operation = ?2 AND idempotency_key = ?3
              AND expires_at <= ?4
            "#,
            vec![
                tenant_scope.into(),
                operation.into(),
                idempotency_key.into(),
                i64::try_from(now)?.into(),
            ],
        )?)
        .await
        .context("Failed to delete expired durable idempotency key")?;
    Ok(result.rows_affected() == 1)
}

/// Delete a deterministic bounded page of expired records, of one tenant or
/// of all, and give their quota back.
pub(super) async fn prune_expired_durable_idempotency_with<C: ConnectionTrait>(
    db: &C,
    tenant_scope: Option<&str>,
    now: u64,
    limit: usize,
) -> anyhow::Result<u64> {
    let mut values = vec![i64::try_from(now)?.into(), i64::try_from(limit)?.into()];
    let tenant_predicate = match tenant_scope {
        Some(tenant_scope) => {
            values.push(tenant_scope.into());
            "AND tenant_scope = ?3"
        }
        None => "",
    };
    let rows = db
        .query_all_raw(raw::statement(
            db,
            format!(
                r#"
                SELECT tenant_scope, operation, idempotency_key
                FROM durable_idempotency_keys_v2
                WHERE expires_at <= ?1 {tenant_predicate}
                ORDER BY expires_at ASC, created_at ASC, tenant_scope ASC,
                         operation ASC, idempotency_key ASC
                LIMIT ?2
                "#
            ),
            values,
        )?)
        .await
        .context("Failed to select expired durable idempotency keys")?;
    let mut removed = std::collections::BTreeMap::<String, u64>::new();
    for row in rows {
        let tenant_scope: String = row.try_get_by_index(0)?;
        let operation: String = row.try_get_by_index(1)?;
        let idempotency_key: String = row.try_get_by_index(2)?;
        if delete_durable_idempotency_with(db, &tenant_scope, &operation, &idempotency_key, now)
            .await?
        {
            *removed.entry(tenant_scope).or_default() += 1;
        }
    }
    release_quotas_with(db, QuotaKind::DurableIdempotency, &removed).await?;
    Ok(removed.values().sum())
}

impl BatchRuntimeStore {
    pub(super) fn durable_idempotency_limits(&self) -> QuotaLimits {
        crate::env_config::tuning().idempotency_limits
    }

    /// Reserve one tenant-scoped durable create operation.
    ///
    /// Callers must not acknowledge work until `commit_durable_idempotency`
    /// succeeds. An expired reservation is eligible for a new owner; an active
    /// reservation never permits a second caller to create work concurrently.
    pub async fn reserve_durable_idempotency(
        &self,
        request: DurableIdempotencyRequest,
    ) -> anyhow::Result<DurableIdempotencyBegin> {
        self.reserve_durable_idempotency_with_limits(request, self.durable_idempotency_limits())
            .await
    }

    #[cfg(test)]
    pub(super) async fn reserve_durable_idempotency_with_capacity(
        &self,
        request: DurableIdempotencyRequest,
        max_records: u64,
    ) -> anyhow::Result<DurableIdempotencyBegin> {
        self.reserve_durable_idempotency_with_limits(
            request,
            QuotaLimits {
                global: max_records,
                per_tenant: max_records,
            },
        )
        .await
    }

    /// Records count against per-tenant and global quota rows kept in the
    /// same transactions that insert and delete them, so a reservation costs
    /// two keyed row updates instead of a table count under a global lock.
    pub(super) async fn reserve_durable_idempotency_with_limits(
        &self,
        request: DurableIdempotencyRequest,
        limits: QuotaLimits,
    ) -> anyhow::Result<DurableIdempotencyBegin> {
        validate_durable_idempotency_request(&request)?;
        anyhow::ensure!(
            limits.global > 0 && limits.per_tenant > 0,
            "Durable idempotency capacity must be positive"
        );
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(write_transaction_options())
            .await
            .context("Failed to start durable idempotency reservation transaction")?;
        let now = nonnegative_timestamp(self.now_millis())?;
        let kind = QuotaKind::DurableIdempotency;

        if let Some(existing) = load_durable_idempotency_with(
            &tx,
            &request.tenant_scope,
            &request.operation,
            &request.idempotency_key,
        )
        .await?
        {
            if existing.expires_at > now {
                let outcome = existing.begin_outcome(&request)?;
                tx.commit().await?;
                return Ok(outcome);
            }
            if delete_durable_idempotency_with(
                &tx,
                &request.tenant_scope,
                &request.operation,
                &request.idempotency_key,
                now,
            )
            .await?
            {
                release_quota_with(&tx, kind, &request.tenant_scope, 1).await?;
            }
        }

        let mut acquired = acquire_quota_with(&tx, kind, &request.tenant_scope, limits).await?;
        // A tenant at its cap first gives back its own expired records; other
        // tenants' are left to the retention sweeper.
        if !acquired
            && prune_expired_durable_idempotency_with(
                &tx,
                Some(&request.tenant_scope),
                now,
                DEFAULT_DURABLE_IDEMPOTENCY_PRUNE_LIMIT,
            )
            .await?
                > 0
        {
            acquired = acquire_quota_with(&tx, kind, &request.tenant_scope, limits).await?;
        }
        if !acquired {
            tx.commit().await?;
            return Ok(DurableIdempotencyBegin::CapacityExceeded);
        }

        let reservation_token = new_uuid();
        let expires_at = now
            .checked_add(request.reservation_ttl_ms)
            .context("Durable idempotency reservation expiry overflow")?;
        let insert_sql = match tx.get_database_backend() {
            DbBackend::Sqlite | DbBackend::Postgres => {
                r#"
                INSERT INTO durable_idempotency_keys_v2 (
                    tenant_scope, operation, idempotency_key, created_at,
                    updated_at, expires_at, digest_version, request_digest,
                    state, reservation_token, runtime_job_id, response_json
                )
                VALUES (?1, ?2, ?3, ?4, ?4, ?5, ?6, ?7, 'reserved', ?8, NULL, NULL)
                ON CONFLICT(tenant_scope, operation, idempotency_key) DO NOTHING
            "#
            }
            backend => bail!("Unsupported durable idempotency database backend: {backend:?}"),
        };
        let inserted = tx
            .execute_raw(raw::statement(
                &tx,
                insert_sql,
                vec![
                    request.tenant_scope.clone().into(),
                    request.operation.clone().into(),
                    request.idempotency_key.clone().into(),
                    i64::try_from(now)?.into(),
                    i64::try_from(expires_at)?.into(),
                    i64::from(request.digest_version).into(),
                    request.request_digest.clone().into(),
                    reservation_token.clone().into(),
                ],
            )?)
            .await
            .context("Failed to reserve durable idempotency key")?;

        if inserted.rows_affected() == 0 {
            release_quota_with(&tx, kind, &request.tenant_scope, 1).await?;
            let existing = load_durable_idempotency_with(
                &tx,
                &request.tenant_scope,
                &request.operation,
                &request.idempotency_key,
            )
            .await?
            .ok_or_else(|| anyhow!("Conflicting durable idempotency key disappeared"))?;
            let outcome = existing.begin_outcome(&request)?;
            tx.commit().await?;
            return Ok(outcome);
        }

        tx.commit()
            .await
            .context("Failed to commit durable idempotency reservation")?;
        Ok(DurableIdempotencyBegin::Acquired(
            DurableIdempotencyReservation {
                tenant_scope: request.tenant_scope,
                operation: request.operation,
                idempotency_key: request.idempotency_key,
                digest_version: request.digest_version,
                request_digest: request.request_digest,
                reservation_token,
                expires_at,
            },
        ))
    }

    /// Commit the replayable response for an acknowledged durable job.
    ///
    /// The conditional update verifies reservation ownership, request digest,
    /// unexpired state, and existence of the referenced job in one transaction.
    pub async fn commit_durable_idempotency(
        &self,
        reservation: &DurableIdempotencyReservation,
        runtime_job_id: &str,
        response_json: serde_json::Value,
        retention_ms: u64,
    ) -> anyhow::Result<Option<DurableIdempotencyReplay>> {
        validate_durable_idempotency_identity(
            &reservation.tenant_scope,
            &reservation.operation,
            &reservation.idempotency_key,
            reservation.digest_version,
            &reservation.request_digest,
        )?;
        validate_bounded_field("reservation token", &reservation.reservation_token, 64)?;
        validate_bounded_field("runtime job ID", runtime_job_id, 128)?;
        anyhow::ensure!(
            (1..=MAX_DURABLE_IDEMPOTENCY_RETENTION_MS).contains(&retention_ms),
            "Durable idempotency retention must be between 1 and {MAX_DURABLE_IDEMPOTENCY_RETENTION_MS} milliseconds"
        );
        let response_json_string = bounded_json_string(
            &response_json,
            MAX_DURABLE_IDEMPOTENCY_RESULT_BYTES,
            "Durable idempotency result",
        )?;

        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(write_transaction_options())
            .await
            .context("Failed to start durable idempotency commit transaction")?;
        let now = nonnegative_timestamp(self.now_millis())?;
        let expires_at = now
            .checked_add(retention_ms)
            .context("Durable idempotency retention expiry overflow")?;
        let result = tx
            .execute_raw(raw::statement(
                &tx,
                r#"
                UPDATE durable_idempotency_keys_v2
                SET state = 'committed', updated_at = ?1, expires_at = ?2,
                    runtime_job_id = ?3, response_json = ?4
                WHERE tenant_scope = ?5
                  AND operation = ?6
                  AND idempotency_key = ?7
                  AND state = 'reserved'
                  AND reservation_token = ?8
                  AND digest_version = ?9
                  AND request_digest = ?10
                  AND expires_at > ?1
                  AND EXISTS (SELECT 1 FROM runtime_jobs WHERE id = ?3)
                "#,
                vec![
                    i64::try_from(now)?.into(),
                    i64::try_from(expires_at)?.into(),
                    runtime_job_id.into(),
                    response_json_string.into(),
                    reservation.tenant_scope.clone().into(),
                    reservation.operation.clone().into(),
                    reservation.idempotency_key.clone().into(),
                    reservation.reservation_token.clone().into(),
                    i64::from(reservation.digest_version).into(),
                    reservation.request_digest.clone().into(),
                ],
            )?)
            .await
            .context("Failed to commit durable idempotency result")?;
        if result.rows_affected() == 0 {
            let existing = load_durable_idempotency_with(
                &tx,
                &reservation.tenant_scope,
                &reservation.operation,
                &reservation.idempotency_key,
            )
            .await?;
            let replay = existing.and_then(|existing| {
                (existing.state == "committed"
                    && existing.expires_at > now
                    && existing.digest_version == reservation.digest_version
                    && existing.request_digest == reservation.request_digest
                    && existing.reservation_token == reservation.reservation_token
                    && existing.runtime_job_id.as_deref() == Some(runtime_job_id)
                    && existing.response_json.as_ref() == Some(&response_json))
                .then(|| DurableIdempotencyReplay {
                    runtime_job_id: runtime_job_id.to_string(),
                    response_json: response_json.clone(),
                    expires_at: existing.expires_at,
                })
            });
            if replay.is_some() {
                tx.commit().await?;
            } else {
                tx.rollback().await?;
            }
            return Ok(replay);
        }
        tx.commit()
            .await
            .context("Failed to commit durable idempotency result transaction")?;
        Ok(Some(DurableIdempotencyReplay {
            runtime_job_id: runtime_job_id.to_string(),
            response_json,
            expires_at,
        }))
    }

    /// Release an uncommitted reservation after a request is rejected locally.
    /// The opaque token prevents an old owner from deleting a replacement.
    pub async fn release_durable_idempotency(
        &self,
        reservation: &DurableIdempotencyReservation,
    ) -> anyhow::Result<bool> {
        validate_durable_idempotency_identity(
            &reservation.tenant_scope,
            &reservation.operation,
            &reservation.idempotency_key,
            reservation.digest_version,
            &reservation.request_digest,
        )?;
        validate_bounded_field("reservation token", &reservation.reservation_token, 64)?;
        let db = self.db.connection().await?;
        let tx = db.begin_with_options(write_transaction_options()).await?;
        let result = tx
            .execute_raw(raw::statement(
                &tx,
                r#"
                DELETE FROM durable_idempotency_keys_v2
                WHERE tenant_scope = ?1 AND operation = ?2 AND idempotency_key = ?3
                  AND state = 'reserved' AND reservation_token = ?4
                  AND digest_version = ?5 AND request_digest = ?6
                "#,
                vec![
                    reservation.tenant_scope.clone().into(),
                    reservation.operation.clone().into(),
                    reservation.idempotency_key.clone().into(),
                    reservation.reservation_token.clone().into(),
                    i64::from(reservation.digest_version).into(),
                    reservation.request_digest.clone().into(),
                ],
            )?)
            .await
            .context("Failed to release durable idempotency reservation")?;
        let released = result.rows_affected() == 1;
        if released {
            release_quota_with(
                &tx,
                QuotaKind::DurableIdempotency,
                &reservation.tenant_scope,
                1,
            )
            .await?;
        }
        tx.commit().await?;
        Ok(released)
    }

    /// Delete a deterministic bounded page of expired reservations/results.
    pub async fn prune_expired_durable_idempotency(&self, limit: usize) -> anyhow::Result<u64> {
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(write_transaction_options())
            .await
            .context("Failed to start durable idempotency prune transaction")?;
        let removed = prune_expired_durable_idempotency_with(
            &tx,
            None,
            nonnegative_timestamp(self.now_millis())?,
            limit.clamp(1, MAX_DURABLE_IDEMPOTENCY_PRUNE_LIMIT),
        )
        .await?;
        tx.commit()
            .await
            .context("Failed to commit durable idempotency pruning")?;
        Ok(removed)
    }
}
