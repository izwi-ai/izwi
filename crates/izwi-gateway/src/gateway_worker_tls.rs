//! Bounded, reference-only TLS material loading for gateway worker clients.

use izwi_serving_client::{WorkerClientTlsConfig, MAX_PRIVATE_CA_ROOTS, MAX_TLS_PEM_BYTES};
use izwi_serving_protocol::secret_file::{
    file_reference_path, read_secret_file, SecretFile, SecretFileError,
};
use tracing::warn;

const PRIVATE_CA_REFS_ENV: &str = "IZWI_GATEWAY_WORKER_TLS_CA_REFS";
const CLIENT_CERT_REF_ENV: &str = "IZWI_GATEWAY_WORKER_TLS_CLIENT_CERT_REF";
const CLIENT_KEY_REF_ENV: &str = "IZWI_GATEWAY_WORKER_TLS_CLIENT_KEY_REF";
const MAX_REFERENCE_BYTES: usize = 4096;
const MAX_CA_REFS_BYTES: usize = MAX_REFERENCE_BYTES * MAX_PRIVATE_CA_ROOTS;

pub fn worker_client_tls_from_env() -> Result<WorkerClientTlsConfig, GatewayWorkerTlsConfigError> {
    let ca_refs = optional_env(PRIVATE_CA_REFS_ENV)?;
    let certificate_ref = optional_env(CLIENT_CERT_REF_ENV)?;
    let private_key_ref = optional_env(CLIENT_KEY_REF_ENV)?;

    if certificate_ref.is_some() != private_key_ref.is_some() {
        return Err(GatewayWorkerTlsConfigError::PartialClientIdentity);
    }

    let mut roots = Vec::new();
    if let Some(ca_refs) = ca_refs {
        if ca_refs.is_empty() || ca_refs.len() > MAX_CA_REFS_BYTES {
            return Err(GatewayWorkerTlsConfigError::InvalidReference);
        }
        for reference in ca_refs.split(',') {
            if roots.len() == MAX_PRIVATE_CA_ROOTS {
                return Err(GatewayWorkerTlsConfigError::TooManyPrivateCaRoots);
            }
            roots.push(read_bounded_pem(reference.trim())?.bytes);
        }
    }

    let certificate = certificate_ref
        .as_deref()
        .map(|reference| read_bounded_pem(reference).map(|file| file.bytes))
        .transpose()?;
    let private_key = private_key_ref
        .as_deref()
        .map(|reference| {
            let file = read_bounded_pem(reference)?;
            if let Some(mode) = file.shared_mode {
                warn!(
                    mode = format_args!("{mode:o}"),
                    "the gateway worker TLS private key file is readable by other users"
                );
            }
            Ok(file.bytes)
        })
        .transpose()?;

    WorkerClientTlsConfig::from_pem(roots, certificate, private_key)
        .map_err(|_| GatewayWorkerTlsConfigError::InvalidPemConfiguration)
}

fn optional_env(name: &str) -> Result<Option<String>, GatewayWorkerTlsConfigError> {
    std::env::var_os(name)
        .map(|value| {
            value
                .into_string()
                .map_err(|_| GatewayWorkerTlsConfigError::InvalidReference)
        })
        .transpose()
}

fn read_bounded_pem(reference: &str) -> Result<SecretFile, GatewayWorkerTlsConfigError> {
    if reference.is_empty() || reference.len() > MAX_REFERENCE_BYTES {
        return Err(GatewayWorkerTlsConfigError::InvalidReference);
    }
    let path = file_reference_path(reference)
        .map_err(|_| GatewayWorkerTlsConfigError::InvalidReference)?;
    read_secret_file(path, MAX_TLS_PEM_BYTES as u64).map_err(|error| match error {
        SecretFileError::NotFileReference
        | SecretFileError::NotAbsolute
        | SecretFileError::NotRegularFile => GatewayWorkerTlsConfigError::InvalidReference,
        SecretFileError::Unavailable => GatewayWorkerTlsConfigError::MissingPemFile,
        SecretFileError::TooLarge(_) => GatewayWorkerTlsConfigError::PemFileTooLarge,
        SecretFileError::Empty => GatewayWorkerTlsConfigError::EmptyPemFile,
        SecretFileError::ReadFailed => GatewayWorkerTlsConfigError::PemReadFailed,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum GatewayWorkerTlsConfigError {
    #[error("gateway worker TLS references must be bounded absolute file: references")]
    InvalidReference,
    #[error("gateway worker TLS configuration has too many private CA roots")]
    TooManyPrivateCaRoots,
    #[error("gateway worker mTLS requires both client certificate and private key references")]
    PartialClientIdentity,
    #[error("gateway worker TLS PEM file does not exist or cannot be opened")]
    MissingPemFile,
    #[error("gateway worker TLS PEM file could not be read")]
    PemReadFailed,
    #[error("gateway worker TLS PEM file is empty")]
    EmptyPemFile,
    #[error("gateway worker TLS PEM file exceeds the size limit")]
    PemFileTooLarge,
    #[error("gateway worker TLS PEM configuration is invalid")]
    InvalidPemConfiguration,
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_CERT_PEM: &[u8] =
        b"-----BEGIN CERTIFICATE-----\nMAECAQ==\n-----END CERTIFICATE-----\n";
    const TEST_KEY_PEM: &[u8] =
        b"-----BEGIN PRIVATE KEY-----\nMAECAQ==\n-----END PRIVATE KEY-----\n";

    fn clear_env() {
        for name in [PRIVATE_CA_REFS_ENV, CLIENT_CERT_REF_ENV, CLIENT_KEY_REF_ENV] {
            std::env::remove_var(name);
        }
    }

    #[test]
    fn tls_files_are_loaded_through_bounded_references_and_redacted() {
        let _guard = crate::test_support::env_lock();
        clear_env();
        let root = tempfile::tempdir().expect("temp dir");
        let ca = root.path().join("ca.pem");
        let certificate = root.path().join("client.pem");
        let key = root.path().join("client-key.pem");
        std::fs::write(&ca, TEST_CERT_PEM).expect("CA fixture");
        std::fs::write(&certificate, TEST_CERT_PEM).expect("certificate fixture");
        std::fs::write(&key, TEST_KEY_PEM).expect("key fixture");
        std::env::set_var(PRIVATE_CA_REFS_ENV, format!("file:{}", ca.display()));
        std::env::set_var(
            CLIENT_CERT_REF_ENV,
            format!("file:{}", certificate.display()),
        );
        std::env::set_var(CLIENT_KEY_REF_ENV, format!("file:{}", key.display()));

        let tls = worker_client_tls_from_env().expect("bounded TLS configuration");
        let debug = format!("{tls:?}");
        assert!(debug.contains("private_ca_root_count: 1"));
        for marker in ["BEGIN CERTIFICATE", "BEGIN PRIVATE KEY"] {
            assert!(!debug.contains(marker));
        }
        clear_env();
    }

    #[test]
    fn tls_environment_rejects_partial_missing_and_oversized_configuration() {
        let _guard = crate::test_support::env_lock();
        clear_env();
        std::env::set_var(CLIENT_CERT_REF_ENV, "file:/missing-client.pem");
        assert_eq!(
            worker_client_tls_from_env().unwrap_err(),
            GatewayWorkerTlsConfigError::PartialClientIdentity
        );

        clear_env();
        std::env::set_var(PRIVATE_CA_REFS_ENV, "file:relative-ca.pem");
        assert_eq!(
            worker_client_tls_from_env().unwrap_err(),
            GatewayWorkerTlsConfigError::InvalidReference
        );

        clear_env();
        std::env::set_var(PRIVATE_CA_REFS_ENV, "file:/missing-ca.pem");
        assert_eq!(
            worker_client_tls_from_env().unwrap_err(),
            GatewayWorkerTlsConfigError::MissingPemFile
        );

        clear_env();
        let root = tempfile::tempdir().expect("temp dir");
        let oversized = root.path().join("oversized.pem");
        std::fs::write(&oversized, vec![b'x'; MAX_TLS_PEM_BYTES + 1]).expect("oversized fixture");
        std::env::set_var(PRIVATE_CA_REFS_ENV, format!("file:{}", oversized.display()));
        assert_eq!(
            worker_client_tls_from_env().unwrap_err(),
            GatewayWorkerTlsConfigError::PemFileTooLarge
        );
        clear_env();
    }
}
