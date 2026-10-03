//! Scoped per-principal API keys for the public gateway (DS0.5).
//!
//! The single shared perimeter key stays the bootstrap root principal. This
//! module adds bounded, individually scoped credentials: every key carries a
//! principal id, a role set (`inference`/`admin`/`metrics`), and an optional
//! tenant scope. Only salted HMAC-SHA256 digests are persisted in the durable
//! store; key material is resolved at boot from bounded `env:`/`file:`
//! references so secrets never appear in argv, manifests, or `Debug` output.
//!
//! The digest construction is deliberately a fast MAC rather than an
//! argon2-class password KDF: provisioned keys are uniform-random bearer
//! tokens (validated by the same policy as the perimeter key), so there is no
//! low-entropy secret to brute force and per-request verification must stay
//! in the microseconds.

use crate::db::sqlite::StoreDatabase;
use crate::gateway_security::{
    bearer_token, constant_time_eq, valid_env_name, valid_identity, validate_secret,
};
use hmac::{Hmac, Mac};
use izwi_hooks::Principal;
use sea_orm::{ConnectionTrait, DatabaseConnection, DbBackend};
use serde::Deserialize;
use sha2::Sha256;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::http::HeaderMap;

pub const PRINCIPAL_KEYS_MANIFEST_ENV: &str = "IZWI_GATEWAY_PRINCIPAL_KEYS_MANIFEST";

pub const ROLE_INFERENCE: &str = "inference";
pub const ROLE_ADMIN: &str = "admin";
pub const ROLE_METRICS: &str = "metrics";

const KNOWN_ROLES: [&str; 3] = [ROLE_INFERENCE, ROLE_ADMIN, ROLE_METRICS];

const MAX_GATEWAY_PRINCIPAL_KEYS: usize = 256;
const SALT_BYTES: usize = 16;
const DIGEST_BYTES: usize = 32;
const MAX_MANIFEST_PATH_BYTES: usize = 1024;
const MAX_MANIFEST_BYTES: usize = 64 * 1024;
const MAX_KEY_FILE_BYTES: usize = 64 * 1024;
const MAX_ENV_REF_BYTES: usize = 128;
const CREDENTIAL_TYPE_ATTRIBUTE: &str = "api_key";
const SOURCE_ATTRIBUTE: &str = "scoped_key_store";

type HmacSha256 = Hmac<Sha256>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum GatewayPrincipalKeysError {
    #[error("gateway principal key manifest path must be a bounded filesystem path")]
    InvalidManifestPath,
    #[error("gateway principal key manifest could not be read")]
    ManifestUnreadable,
    #[error("gateway principal key manifest exceeds the bounded size limit")]
    ManifestTooLarge,
    #[error("gateway principal key manifest is not valid JSON with the expected shape")]
    InvalidManifest,
    #[error("gateway principal key manifest version is not supported")]
    UnsupportedManifestVersion,
    #[error("gateway principal key manifest must list between 1 and 256 principals")]
    InvalidPrincipalCount,
    #[error("gateway principal key manifest contains an invalid principal or tenant identity")]
    InvalidIdentity,
    #[error("gateway principal key manifest contains an invalid role set")]
    InvalidRoles,
    #[error("gateway principal key manifest contains a duplicate principal id")]
    DuplicatePrincipal,
    #[error(
        "gateway principal key references must be bounded env:VARIABLE or file:PATH references"
    )]
    InvalidKeyReference,
    #[error("gateway principal key reference could not be resolved")]
    MissingKeyMaterial,
    #[error("gateway principal key material does not satisfy the bounded credential policy")]
    InvalidKeyMaterial,
    #[error("gateway principal key material must not reuse a perimeter credential")]
    ReusedPerimeterSecret,
    #[error("gateway principal key material must be unique per principal")]
    DuplicateKeyMaterial,
    #[error("durable principal key store contains an invalid record")]
    InvalidStoredRecord,
}

/// A manifest principal with its key material resolved at boot.
pub struct ProvisionedPrincipal {
    pub principal_id: String,
    pub roles: Vec<String>,
    pub tenant_id: Option<String>,
    key: Arc<[u8]>,
}

impl fmt::Debug for ProvisionedPrincipal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProvisionedPrincipal")
            .field("principal_id", &self.principal_id)
            .field("roles", &self.roles)
            .field("tenant_id", &self.tenant_id)
            .field("key", &"[REDACTED]")
            .finish()
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestDocument {
    version: u32,
    principals: Vec<ManifestPrincipal>,
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct ManifestPrincipal {
    principal_id: String,
    roles: Vec<String>,
    #[serde(default)]
    tenant_id: Option<String>,
    key_ref: String,
}

/// Resolve the optional bounded manifest path from the environment. `None`
/// keeps the gateway scoped-key-free and never opens the durable store.
pub fn manifest_path_from_env() -> Result<Option<PathBuf>, GatewayPrincipalKeysError> {
    let Some(value) = std::env::var_os(PRINCIPAL_KEYS_MANIFEST_ENV) else {
        return Ok(None);
    };
    let value = value
        .into_string()
        .map_err(|_| GatewayPrincipalKeysError::InvalidManifestPath)?;
    let trimmed = value.trim();
    if trimmed.is_empty()
        || trimmed.len() > MAX_MANIFEST_PATH_BYTES
        || trimmed.bytes().any(|byte| byte.is_ascii_control())
    {
        return Err(GatewayPrincipalKeysError::InvalidManifestPath);
    }
    Ok(Some(PathBuf::from(trimmed)))
}

/// Parse and structurally validate the manifest without resolving key
/// material. Every rejection is fail-closed.
fn load_manifest(path: &Path) -> Result<Vec<ManifestPrincipal>, GatewayPrincipalKeysError> {
    let bytes = std::fs::read(path).map_err(|_| GatewayPrincipalKeysError::ManifestUnreadable)?;
    if bytes.len() > MAX_MANIFEST_BYTES {
        return Err(GatewayPrincipalKeysError::ManifestTooLarge);
    }
    let document: ManifestDocument =
        serde_json::from_slice(&bytes).map_err(|_| GatewayPrincipalKeysError::InvalidManifest)?;
    if document.version != 1 {
        return Err(GatewayPrincipalKeysError::UnsupportedManifestVersion);
    }
    if document.principals.is_empty() || document.principals.len() > MAX_GATEWAY_PRINCIPAL_KEYS {
        return Err(GatewayPrincipalKeysError::InvalidPrincipalCount);
    }
    let mut seen_ids = std::collections::BTreeSet::new();
    for principal in &document.principals {
        if !valid_identity(&principal.principal_id) {
            return Err(GatewayPrincipalKeysError::InvalidIdentity);
        }
        if let Some(tenant_id) = &principal.tenant_id {
            if !valid_identity(tenant_id) {
                return Err(GatewayPrincipalKeysError::InvalidIdentity);
            }
        }
        if !valid_role_set(&principal.roles) {
            return Err(GatewayPrincipalKeysError::InvalidRoles);
        }
        if !valid_key_ref(&principal.key_ref) {
            return Err(GatewayPrincipalKeysError::InvalidKeyReference);
        }
        if !seen_ids.insert(principal.principal_id.as_str()) {
            return Err(GatewayPrincipalKeysError::DuplicatePrincipal);
        }
    }
    Ok(document.principals)
}

/// Roles must be a duplicate-free non-empty subset of the known roles; the
/// returned vector is in canonical order `[inference, admin, metrics]`.
fn canonical_roles(roles: &[String]) -> Option<Vec<String>> {
    if roles.is_empty() || roles.len() > KNOWN_ROLES.len() {
        return None;
    }
    let mut canonical = Vec::with_capacity(roles.len());
    for known in KNOWN_ROLES {
        if roles.iter().any(|role| role == known) {
            canonical.push(known.to_string());
        }
    }
    if canonical.len() != roles.len() {
        return None;
    }
    Some(canonical)
}

fn valid_role_set(roles: &[String]) -> bool {
    canonical_roles(roles).is_some()
}

fn valid_key_ref(key_ref: &str) -> bool {
    if let Some(variable) = key_ref.strip_prefix("env:") {
        return key_ref.len() <= MAX_ENV_REF_BYTES && valid_env_name(variable);
    }
    if let Some(path) = key_ref.strip_prefix("file:") {
        return !path.is_empty()
            && key_ref.len() <= MAX_MANIFEST_PATH_BYTES
            && !path.bytes().any(|byte| byte.is_ascii_control());
    }
    false
}

/// Resolve one bounded key reference to key material. `env:` values are used
/// exactly as provided; `file:` contents are whitespace-trimmed because key
/// files conventionally end in a newline.
fn resolve_key_ref(key_ref: &str) -> Result<Arc<[u8]>, GatewayPrincipalKeysError> {
    let material = if let Some(variable) = key_ref.strip_prefix("env:") {
        std::env::var_os(variable)
            .ok_or(GatewayPrincipalKeysError::MissingKeyMaterial)?
            .into_string()
            .map_err(|_| GatewayPrincipalKeysError::InvalidKeyMaterial)?
            .into_bytes()
    } else if let Some(path) = key_ref.strip_prefix("file:") {
        let contents = std::fs::read_to_string(path)
            .map_err(|_| GatewayPrincipalKeysError::MissingKeyMaterial)?;
        if contents.len() > MAX_KEY_FILE_BYTES {
            return Err(GatewayPrincipalKeysError::InvalidKeyMaterial);
        }
        contents.trim().as_bytes().to_vec()
    } else {
        return Err(GatewayPrincipalKeysError::InvalidKeyReference);
    };
    let material =
        String::from_utf8(material).map_err(|_| GatewayPrincipalKeysError::InvalidKeyMaterial)?;
    validate_secret(&material).map_err(|_| GatewayPrincipalKeysError::InvalidKeyMaterial)?;
    Ok(Arc::from(material.into_bytes()))
}

/// Resolve every manifest entry into key material, rejecting perimeter and
/// pairwise reuse. Provisioning is all-or-nothing: one invalid entry aborts
/// the whole bootstrap before the store is touched.
fn resolve_principals(
    manifest: &[ManifestPrincipal],
    perimeter: &crate::gateway_security::GatewayPerimeterConfig,
) -> Result<Vec<ProvisionedPrincipal>, GatewayPrincipalKeysError> {
    let mut provisioned = Vec::with_capacity(manifest.len());
    for entry in manifest {
        let key = resolve_key_ref(&entry.key_ref)?;
        if perimeter.contains_credential(&key) {
            return Err(GatewayPrincipalKeysError::ReusedPerimeterSecret);
        }
        if provisioned
            .iter()
            .any(|existing: &ProvisionedPrincipal| existing.has_key(&key))
        {
            return Err(GatewayPrincipalKeysError::DuplicateKeyMaterial);
        }
        provisioned.push(ProvisionedPrincipal {
            principal_id: entry.principal_id.clone(),
            roles: canonical_roles(&entry.roles).unwrap_or_default(),
            tenant_id: entry.tenant_id.clone(),
            key,
        });
    }
    Ok(provisioned)
}

impl ProvisionedPrincipal {
    fn has_key(&self, candidate: &[u8]) -> bool {
        constant_time_eq(candidate, &self.key)
    }
}

/// A durable salted digest record. `key_salt` and `key_hash` are hex-encoded
/// so the record stays inspectable with plain SQL. Only digests live here —
/// never key material — so the derived `Debug` is safe.
#[derive(Debug)]
struct StoredPrincipalKey {
    principal_id: String,
    roles: Vec<String>,
    tenant_id: Option<String>,
    salt: [u8; SALT_BYTES],
    hash: [u8; DIGEST_BYTES],
}

impl StoredPrincipalKey {
    fn provision(provisioned: &ProvisionedPrincipal) -> Self {
        let salt = fresh_salt();
        Self {
            principal_id: provisioned.principal_id.clone(),
            roles: provisioned.roles.clone(),
            tenant_id: provisioned.tenant_id.clone(),
            salt,
            hash: key_digest(&salt, &provisioned.key),
        }
    }

    fn from_row(
        principal_id: String,
        roles_json: String,
        tenant_id: Option<String>,
        key_salt: String,
        key_hash: String,
    ) -> Result<Self, GatewayPrincipalKeysError> {
        let roles: Vec<String> = serde_json::from_str(&roles_json)
            .map_err(|_| GatewayPrincipalKeysError::InvalidStoredRecord)?;
        if !valid_role_set(&roles) || !valid_identity(&principal_id) {
            return Err(GatewayPrincipalKeysError::InvalidStoredRecord);
        }
        let salt: [u8; SALT_BYTES] = from_hex(&key_salt)
            .and_then(|bytes| bytes.try_into().ok())
            .ok_or(GatewayPrincipalKeysError::InvalidStoredRecord)?;
        let hash: [u8; DIGEST_BYTES] = from_hex(&key_hash)
            .and_then(|bytes| bytes.try_into().ok())
            .ok_or(GatewayPrincipalKeysError::InvalidStoredRecord)?;
        Ok(Self {
            principal_id,
            roles,
            tenant_id,
            salt,
            hash,
        })
    }
}

async fn upsert_principal_key(
    db: &DatabaseConnection,
    record: &StoredPrincipalKey,
) -> anyhow::Result<()> {
    let now = current_timestamp_millis();
    let assignments = match db.get_database_backend() {
        DbBackend::MySql => {
            r#"
        INSERT INTO gateway_principal_keys (
            principal_id, roles_json, tenant_id, key_salt, key_hash, created_at, updated_at
        )
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6)
        AS new
        ON DUPLICATE KEY UPDATE
            roles_json = new.roles_json,
            tenant_id = new.tenant_id,
            key_salt = new.key_salt,
            key_hash = new.key_hash,
            updated_at = new.updated_at
        "#
        }
        _ => {
            r#"
        INSERT INTO gateway_principal_keys (
            principal_id, roles_json, tenant_id, key_salt, key_hash, created_at, updated_at
        )
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6)
        ON CONFLICT(principal_id) DO UPDATE SET
            roles_json = excluded.roles_json,
            tenant_id = excluded.tenant_id,
            key_salt = excluded.key_salt,
            key_hash = excluded.key_hash,
            updated_at = excluded.updated_at
        "#
        }
    };
    db.execute_raw(crate::db::raw::statement(
        db,
        assignments,
        vec![
            record.principal_id.clone().into(),
            serde_json::to_string(&record.roles)?.into(),
            record.tenant_id.clone().into(),
            to_hex(&record.salt).into(),
            to_hex(&record.hash).into(),
            now.into(),
        ],
    )?)
    .await?;
    Ok(())
}

async fn load_principal_keys(
    db: &DatabaseConnection,
) -> Result<Vec<StoredPrincipalKey>, GatewayPrincipalKeysError> {
    let rows = db
        .query_all_raw(
            crate::db::raw::statement_without_values(
                db,
                "SELECT principal_id, roles_json, tenant_id, key_salt, key_hash FROM gateway_principal_keys ORDER BY principal_id",
            ),
        )
        .await
        .map_err(|_| GatewayPrincipalKeysError::InvalidStoredRecord)?;
    if rows.len() > MAX_GATEWAY_PRINCIPAL_KEYS {
        return Err(GatewayPrincipalKeysError::InvalidPrincipalCount);
    }
    rows.into_iter()
        .map(|row| {
            let principal_id: String = row
                .try_get_by_index(0)
                .map_err(|_| GatewayPrincipalKeysError::InvalidStoredRecord)?;
            let roles_json: String = row
                .try_get_by_index(1)
                .map_err(|_| GatewayPrincipalKeysError::InvalidStoredRecord)?;
            let tenant_id: Option<String> = row
                .try_get_by_index(2)
                .map_err(|_| GatewayPrincipalKeysError::InvalidStoredRecord)?;
            let key_salt: String = row
                .try_get_by_index(3)
                .map_err(|_| GatewayPrincipalKeysError::InvalidStoredRecord)?;
            let key_hash: String = row
                .try_get_by_index(4)
                .map_err(|_| GatewayPrincipalKeysError::InvalidStoredRecord)?;
            StoredPrincipalKey::from_row(principal_id, roles_json, tenant_id, key_salt, key_hash)
        })
        .collect()
}

/// Immutable in-memory view of the scoped principals. Authentication is a
/// bounded scan: every entry's digest is evaluated so total cost and memory
/// access pattern are independent of which, or whether, an entry matches.
#[derive(Clone, Default)]
pub struct GatewayPrincipalDirectory {
    entries: Arc<[ScopedEntry]>,
}

struct ScopedEntry {
    principal: Principal,
    salt: [u8; SALT_BYTES],
    hash: [u8; DIGEST_BYTES],
}

impl GatewayPrincipalDirectory {
    pub fn empty() -> Self {
        Self {
            entries: Arc::from([]),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Provision manifest principals into the durable store, then load the
    /// full directory from the store. The store is the system of record; the
    /// manifest only bootstraps entries at boot.
    pub async fn bootstrap(
        store: &StoreDatabase,
        manifest_path: &Path,
        perimeter: &crate::gateway_security::GatewayPerimeterConfig,
    ) -> anyhow::Result<Self> {
        let manifest = load_manifest(manifest_path)?;
        let provisioned = resolve_principals(&manifest, perimeter)?;
        let db = store.connection().await?;
        for principal in &provisioned {
            upsert_principal_key(db, &StoredPrincipalKey::provision(principal)).await?;
        }
        let stored = load_principal_keys(db).await?;
        let entries: Vec<ScopedEntry> = stored
            .iter()
            .map(|record| ScopedEntry {
                principal: build_principal(record),
                salt: record.salt,
                hash: record.hash,
            })
            .collect();
        Ok(Self {
            entries: entries.into(),
        })
    }

    /// Authenticate a request bearer against the scoped principals.
    pub fn authenticate(&self, headers: &HeaderMap) -> Option<Principal> {
        let supplied = bearer_token(headers)?;
        let mut matched = None;
        for entry in self.entries.iter() {
            let candidate = key_digest(&entry.salt, supplied.as_bytes());
            if constant_time_eq(&candidate, &entry.hash) {
                matched = Some(entry.principal.clone());
            }
        }
        matched
    }

    /// Whether any scoped principal carries `role`; gates endpoint existence
    /// for the drain/metrics side-channels exactly like the optional
    /// perimeter credentials do.
    pub fn has_role(&self, role: &str) -> bool {
        self.entries.iter().any(|entry| {
            entry
                .principal
                .roles
                .iter()
                .any(|candidate| candidate == role)
        })
    }

    pub fn authorize(&self, headers: &HeaderMap, role: &str) -> bool {
        self.authenticate(headers)
            .is_some_and(|principal| principal.roles.iter().any(|candidate| candidate == role))
    }
}

impl fmt::Debug for GatewayPrincipalDirectory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GatewayPrincipalDirectory")
            .field("principals", &self.entries.len())
            .finish()
    }
}

fn build_principal(record: &StoredPrincipalKey) -> Principal {
    let mut attributes = izwi_hooks::HookMetadata::new();
    attributes.insert(
        "credential_type".to_string(),
        CREDENTIAL_TYPE_ATTRIBUTE.to_string(),
    );
    attributes.insert("scope".to_string(), record.roles.join(","));
    attributes.insert("source".to_string(), SOURCE_ATTRIBUTE.to_string());
    Principal {
        id: record.principal_id.clone(),
        display_name: None,
        tenant_id: record.tenant_id.clone(),
        roles: record.roles.clone(),
        attributes,
    }
}

fn key_digest(salt: &[u8; SALT_BYTES], key: &[u8]) -> [u8; DIGEST_BYTES] {
    let mut mac = HmacSha256::new_from_slice(salt).expect("HMAC accepts any key length");
    mac.update(key);
    mac.finalize().into_bytes().into()
}

fn fresh_salt() -> [u8; SALT_BYTES] {
    rand::random()
}

fn to_hex(bytes: &[u8]) -> String {
    let mut hex = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        hex.push_str(&format!("{byte:02x}"));
    }
    hex
}

fn from_hex(value: &str) -> Option<Vec<u8>> {
    if value.len() % 2 != 0 {
        return None;
    }
    (0..value.len() / 2)
        .map(|index| u8::from_str_radix(&value[index * 2..index * 2 + 2], 16).ok())
        .collect()
}

fn current_timestamp_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gateway_security::GatewayPerimeterConfig;
    use axum::http::HeaderValue;

    const ROOT_KEY: &str = "test-public-api-key-123456";
    const SCOPED_KEY: &str = "scoped-principal-key-abc123";
    const OTHER_SCOPED_KEY: &str = "other-principal-key-xyz789";

    fn test_perimeter() -> GatewayPerimeterConfig {
        GatewayPerimeterConfig::new_for_test(ROOT_KEY, 1024 * 1024)
            .expect("test perimeter should be valid")
    }

    fn bearer(headers: &HeaderMap, token: &str) -> HeaderMap {
        let mut headers = headers.clone();
        headers.insert(
            axum::http::header::AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {token}")).expect("bearer header"),
        );
        headers
    }

    fn write_manifest(dir: &Path, body: &str) -> PathBuf {
        let path = dir.join("principals.json");
        std::fs::write(&path, body).expect("manifest write");
        path
    }

    fn manifest_body(key_ref: &str) -> String {
        format!(
            r#"{{"version":1,"principals":[{{"principal_id":"svc-alpha","roles":["inference"],"tenant_id":"tenant-alpha","key_ref":"{key_ref}"}}]}}"#
        )
    }

    #[test]
    fn manifest_path_is_optional_and_bounded() {
        let _guard = crate::test_support::env_lock();
        std::env::remove_var(PRINCIPAL_KEYS_MANIFEST_ENV);
        assert_eq!(manifest_path_from_env().unwrap(), None);

        std::env::set_var(PRINCIPAL_KEYS_MANIFEST_ENV, "  /tmp/principals.json ");
        assert_eq!(
            manifest_path_from_env().unwrap(),
            Some(PathBuf::from("/tmp/principals.json"))
        );

        std::env::set_var(PRINCIPAL_KEYS_MANIFEST_ENV, "path\nwith\nnewline");
        assert_eq!(
            manifest_path_from_env().unwrap_err(),
            GatewayPrincipalKeysError::InvalidManifestPath
        );
        let oversized = "a".repeat(MAX_MANIFEST_PATH_BYTES + 1);
        std::env::set_var(PRINCIPAL_KEYS_MANIFEST_ENV, oversized);
        assert_eq!(
            manifest_path_from_env().unwrap_err(),
            GatewayPrincipalKeysError::InvalidManifestPath
        );
        std::env::remove_var(PRINCIPAL_KEYS_MANIFEST_ENV);
    }

    #[test]
    fn manifest_rejects_inline_secrets_and_unknown_fields() {
        let dir = tempfile::tempdir().expect("temp dir");
        let inline = write_manifest(
            dir.path(),
            r#"{"version":1,"principals":[{"principal_id":"svc","roles":["inference"],"key":"inline-secret-123456"}]}"#,
        );
        assert_eq!(
            load_manifest(&inline).unwrap_err(),
            GatewayPrincipalKeysError::InvalidManifest
        );

        let unref = write_manifest(dir.path(), &manifest_body("scoped-principal-key-abc123"));
        assert_eq!(
            load_manifest(&unref).unwrap_err(),
            GatewayPrincipalKeysError::InvalidKeyReference
        );
    }

    #[test]
    fn manifest_rejects_invalid_structures() {
        let dir = tempfile::tempdir().expect("temp dir");
        let cases: [(GatewayPrincipalKeysError, &str); 6] = [
            (
                GatewayPrincipalKeysError::UnsupportedManifestVersion,
                r#"{"version":2,"principals":[{"principal_id":"svc","roles":["inference"],"key_ref":"env:KEY"}]}"#,
            ),
            (
                GatewayPrincipalKeysError::InvalidPrincipalCount,
                r#"{"version":1,"principals":[]}"#,
            ),
            (
                GatewayPrincipalKeysError::InvalidIdentity,
                r#"{"version":1,"principals":[{"principal_id":"bad id!","roles":["inference"],"key_ref":"env:KEY"}]}"#,
            ),
            (
                GatewayPrincipalKeysError::InvalidRoles,
                r#"{"version":1,"principals":[{"principal_id":"svc","roles":["root"],"key_ref":"env:KEY"}]}"#,
            ),
            (
                GatewayPrincipalKeysError::InvalidRoles,
                r#"{"version":1,"principals":[{"principal_id":"svc","roles":[],"key_ref":"env:KEY"}]}"#,
            ),
            (
                GatewayPrincipalKeysError::DuplicatePrincipal,
                r#"{"version":1,"principals":[{"principal_id":"svc","roles":["inference"],"key_ref":"env:KEY"},{"principal_id":"svc","roles":["metrics"],"key_ref":"env:KEY2"}]}"#,
            ),
        ];
        for (expected, body) in cases {
            let path = write_manifest(dir.path(), body);
            assert_eq!(load_manifest(&path).unwrap_err(), expected);
        }
    }

    #[test]
    fn key_refs_resolve_from_env_and_file() {
        let _guard = crate::test_support::env_lock();
        let dir = tempfile::tempdir().expect("temp dir");
        let key_file = dir.path().join("alpha.key");
        std::fs::write(&key_file, format!("{SCOPED_KEY}\n")).expect("key file");

        std::env::set_var("IZWI_TEST_SCOPED_KEY", SCOPED_KEY);
        let from_env = resolve_key_ref("env:IZWI_TEST_SCOPED_KEY").unwrap();
        let from_file = resolve_key_ref(&format!("file:{}", key_file.display())).unwrap();
        assert_eq!(from_env.as_ref(), SCOPED_KEY.as_bytes());
        assert_eq!(from_file.as_ref(), SCOPED_KEY.as_bytes());

        std::env::remove_var("IZWI_TEST_SCOPED_KEY");
        assert_eq!(
            resolve_key_ref("env:IZWI_TEST_SCOPED_KEY").unwrap_err(),
            GatewayPrincipalKeysError::MissingKeyMaterial
        );
        assert_eq!(
            resolve_key_ref(&format!(
                "file:{}",
                dir.path().join("missing.key").display()
            ))
            .unwrap_err(),
            GatewayPrincipalKeysError::MissingKeyMaterial
        );
        assert_eq!(
            resolve_key_ref("plain-secret").unwrap_err(),
            GatewayPrincipalKeysError::InvalidKeyReference
        );
    }

    #[test]
    fn provisioning_rejects_perimeter_and_pairwise_reuse() {
        let _guard = crate::test_support::env_lock();
        let perimeter = test_perimeter();
        let dir = tempfile::tempdir().expect("temp dir");
        std::env::set_var("IZWI_TEST_SCOPED_KEY", ROOT_KEY);
        let root_reuse = write_manifest(dir.path(), &manifest_body("env:IZWI_TEST_SCOPED_KEY"));
        let manifest = load_manifest(&root_reuse).unwrap();
        assert_eq!(
            resolve_principals(&manifest, &perimeter).unwrap_err(),
            GatewayPrincipalKeysError::ReusedPerimeterSecret
        );

        std::env::set_var("IZWI_TEST_SCOPED_KEY", SCOPED_KEY);
        std::env::set_var("IZWI_TEST_SCOPED_KEY_2", SCOPED_KEY);
        let duplicated = write_manifest(
            dir.path(),
            r#"{"version":1,"principals":[
                {"principal_id":"svc-a","roles":["inference"],"tenant_id":"tenant-alpha","key_ref":"env:IZWI_TEST_SCOPED_KEY"},
                {"principal_id":"svc-b","roles":["metrics"],"key_ref":"env:IZWI_TEST_SCOPED_KEY_2"}]}"#,
        );
        let manifest = load_manifest(&duplicated).unwrap();
        assert_eq!(
            resolve_principals(&manifest, &perimeter).unwrap_err(),
            GatewayPrincipalKeysError::DuplicateKeyMaterial
        );

        std::env::set_var("IZWI_TEST_SCOPED_KEY_2", OTHER_SCOPED_KEY);
        let manifest = load_manifest(&duplicated).unwrap();
        let provisioned = resolve_principals(&manifest, &perimeter).unwrap();
        assert_eq!(provisioned.len(), 2);
        assert_eq!(provisioned[0].roles, vec![ROLE_INFERENCE.to_string()]);
        assert_eq!(provisioned[0].tenant_id.as_deref(), Some("tenant-alpha"));
        std::env::remove_var("IZWI_TEST_SCOPED_KEY");
        std::env::remove_var("IZWI_TEST_SCOPED_KEY_2");
    }

    #[test]
    fn digests_are_salted_and_verification_is_exact() {
        let salt = fresh_salt();
        let other_salt = fresh_salt();
        let first = key_digest(&salt, SCOPED_KEY.as_bytes());
        let second = key_digest(&other_salt, SCOPED_KEY.as_bytes());
        assert_ne!(salt, other_salt, "salts are independently drawn");
        assert_ne!(first, second, "same key under different salts differs");
        assert_eq!(
            key_digest(&salt, SCOPED_KEY.as_bytes()),
            first,
            "digest is deterministic per salt"
        );

        let entry_hash = key_digest(&salt, SCOPED_KEY.as_bytes());
        assert!(constant_time_eq(
            &key_digest(&salt, SCOPED_KEY.as_bytes()),
            &entry_hash
        ));
        assert!(!constant_time_eq(
            &key_digest(&salt, OTHER_SCOPED_KEY.as_bytes()),
            &entry_hash
        ));
    }

    #[tokio::test]
    async fn store_round_trips_upsert_and_rotation() {
        let dir = tempfile::tempdir().expect("temp dir");
        let store = StoreDatabase::new(dir.path().join("store.sqlite3"));
        let db = store.connection().await.expect("store connection");

        let provisioned = ProvisionedPrincipal {
            principal_id: "svc-alpha".to_string(),
            roles: vec![ROLE_INFERENCE.to_string()],
            tenant_id: Some("tenant-alpha".to_string()),
            key: Arc::from(SCOPED_KEY.as_bytes()),
        };
        let original = StoredPrincipalKey::provision(&provisioned);
        upsert_principal_key(db, &original).await.expect("insert");
        let loaded = load_principal_keys(db).await.expect("load");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].principal_id, "svc-alpha");
        assert_eq!(loaded[0].salt, original.salt);
        assert_eq!(loaded[0].hash, original.hash);

        let rotated = StoredPrincipalKey::provision(&provisioned);
        upsert_principal_key(db, &rotated).await.expect("rotate");
        let loaded = load_principal_keys(db).await.expect("reload");
        assert_eq!(loaded.len(), 1, "rotation replaces in place");
        assert_eq!(loaded[0].hash, rotated.hash);
        assert_ne!(loaded[0].hash, original.hash);

        let malformed = StoredPrincipalKey::from_row(
            "svc".to_string(),
            "not-json".to_string(),
            None,
            to_hex(&[0u8; SALT_BYTES]),
            to_hex(&[0u8; DIGEST_BYTES]),
        )
        .unwrap_err();
        assert_eq!(malformed, GatewayPrincipalKeysError::InvalidStoredRecord);
    }

    #[tokio::test]
    async fn bootstrap_authenticates_scoped_principals_and_gates_roles() {
        let _guard = crate::test_support::env_lock();
        let dir = tempfile::tempdir().expect("temp dir");
        std::env::set_var("IZWI_TEST_SCOPED_KEY", SCOPED_KEY);
        std::env::set_var("IZWI_TEST_SCOPED_KEY_2", OTHER_SCOPED_KEY);
        let manifest = write_manifest(
            dir.path(),
            r#"{"version":1,"principals":[
                {"principal_id":"svc-alpha","roles":["inference"],"tenant_id":"tenant-alpha","key_ref":"env:IZWI_TEST_SCOPED_KEY"},
                {"principal_id":"ops","roles":["metrics","admin"],"key_ref":"env:IZWI_TEST_SCOPED_KEY_2"}]}"#,
        );
        let store = StoreDatabase::new(dir.path().join("store.sqlite3"));
        let directory = GatewayPrincipalDirectory::bootstrap(&store, &manifest, &test_perimeter())
            .await
            .expect("bootstrap");
        std::env::remove_var("IZWI_TEST_SCOPED_KEY");
        std::env::remove_var("IZWI_TEST_SCOPED_KEY_2");

        assert_eq!(directory.len(), 2);
        let debug = format!("{directory:?}");
        assert!(!debug.contains(SCOPED_KEY));
        assert!(!debug.contains(OTHER_SCOPED_KEY));

        let principal = directory
            .authenticate(&bearer(&HeaderMap::new(), SCOPED_KEY))
            .expect("scoped key authenticates");
        assert_eq!(principal.id, "svc-alpha");
        assert_eq!(principal.tenant_id.as_deref(), Some("tenant-alpha"));
        assert_eq!(principal.roles, vec![ROLE_INFERENCE.to_string()]);
        assert_eq!(
            principal
                .attributes
                .get("credential_type")
                .map(String::as_str),
            Some("api_key")
        );

        assert!(directory
            .authenticate(&bearer(&HeaderMap::new(), ROOT_KEY))
            .is_none());
        assert!(directory
            .authenticate(&bearer(&HeaderMap::new(), "wrong-principal-key-000000"))
            .is_none());
        assert!(directory.authenticate(&HeaderMap::new()).is_none());

        assert!(directory.has_role(ROLE_METRICS));
        assert!(directory.has_role(ROLE_ADMIN));
        assert!(directory.has_role(ROLE_INFERENCE));
        assert!(!directory.has_role("root"));
        assert!(directory.authorize(&bearer(&HeaderMap::new(), OTHER_SCOPED_KEY), ROLE_METRICS));
        assert!(!directory.authorize(&bearer(&HeaderMap::new(), OTHER_SCOPED_KEY), ROLE_INFERENCE));
    }

    #[tokio::test]
    async fn bootstrap_survives_restart_from_store_without_reprovoked_material() {
        let _guard = crate::test_support::env_lock();
        let dir = tempfile::tempdir().expect("temp dir");
        std::env::set_var("IZWI_TEST_SCOPED_KEY", SCOPED_KEY);
        let manifest = write_manifest(dir.path(), &manifest_body("env:IZWI_TEST_SCOPED_KEY"));
        let store = StoreDatabase::new(dir.path().join("store.sqlite3"));
        let directory = GatewayPrincipalDirectory::bootstrap(&store, &manifest, &test_perimeter())
            .await
            .expect("bootstrap");

        // Restart against the same store with a *new* manifest that provisions
        // a second principal: the first entry must keep authenticating and the
        // second must be added.
        std::env::set_var("IZWI_TEST_SCOPED_KEY_2", OTHER_SCOPED_KEY);
        let second_manifest = write_manifest(
            dir.path(),
            r#"{"version":1,"principals":[{"principal_id":"ops","roles":["metrics"],"key_ref":"env:IZWI_TEST_SCOPED_KEY_2"}]}"#,
        );
        let restarted =
            GatewayPrincipalDirectory::bootstrap(&store, &second_manifest, &test_perimeter())
                .await
                .expect("restart bootstrap");
        std::env::remove_var("IZWI_TEST_SCOPED_KEY");
        std::env::remove_var("IZWI_TEST_SCOPED_KEY_2");

        assert_eq!(restarted.len(), 2);
        assert!(restarted
            .authenticate(&bearer(&HeaderMap::new(), SCOPED_KEY))
            .is_some_and(|principal| principal.id == "svc-alpha"));
        assert!(restarted
            .authenticate(&bearer(&HeaderMap::new(), OTHER_SCOPED_KEY))
            .is_some_and(|principal| principal.id == "ops"));
        let _ = directory;
    }

    #[test]
    fn canonical_roles_are_ordered_and_strict() {
        assert_eq!(
            canonical_roles(&["metrics".into(), "inference".into()]),
            Some(vec![ROLE_INFERENCE.to_string(), ROLE_METRICS.to_string()])
        );
        assert_eq!(canonical_roles(&["admin".into(), "admin".into()]), None);
        assert_eq!(canonical_roles(&["unknown".into()]), None);
        assert_eq!(canonical_roles(&[]), None);
    }
}
