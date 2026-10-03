use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};
use std::{fmt, str::FromStr};
use subtle::ConstantTimeEq;

pub const MAX_ID_BYTES: usize = 128;
pub const MAX_SERVICE_TOKEN_BYTES: usize = 4096;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum IdentifierError {
    #[error("identifier must not be empty")]
    Empty,
    #[error("identifier is {actual} bytes; maximum is {maximum}")]
    TooLong { actual: usize, maximum: usize },
    #[error("identifier contains an invalid byte at offset {offset}")]
    InvalidByte { offset: usize },
}

fn validate_identifier(value: &str) -> Result<(), IdentifierError> {
    if value.is_empty() {
        return Err(IdentifierError::Empty);
    }
    if value.len() > MAX_ID_BYTES {
        return Err(IdentifierError::TooLong {
            actual: value.len(),
            maximum: MAX_ID_BYTES,
        });
    }
    if let Some((offset, _)) = value.bytes().enumerate().find(|(_, byte)| {
        !byte.is_ascii_alphanumeric() && !matches!(byte, b'-' | b'_' | b'.' | b':' | b'/' | b'@')
    }) {
        return Err(IdentifierError::InvalidByte { offset });
    }
    Ok(())
}

macro_rules! bounded_id {
    ($name:ident, $description:literal) => {
        #[doc = $description]
        #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Result<Self, IdentifierError> {
                let value = value.into();
                validate_identifier(&value)?;
                Ok(Self(value))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }

            pub fn into_inner(self) -> String {
                self.0
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter
                    .debug_tuple(stringify!($name))
                    .field(&self.0)
                    .finish()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(&self.0)
            }
        }

        impl FromStr for $name {
            type Err = IdentifierError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                Self::new(value)
            }
        }

        impl TryFrom<String> for $name {
            type Error = IdentifierError;

            fn try_from(value: String) -> Result<Self, Self::Error> {
                Self::new(value)
            }
        }

        impl TryFrom<&str> for $name {
            type Error = IdentifierError;

            fn try_from(value: &str) -> Result<Self, Self::Error> {
                Self::new(value)
            }
        }

        impl Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                serializer.serialize_str(&self.0)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let value = String::deserialize(deserializer)?;
                Self::new(value).map_err(de::Error::custom)
            }
        }
    };
}

bounded_id!(NodeId, "Stable identity of a supervised node.");
bounded_id!(WorkerId, "Stable logical worker identity.");
bounded_id!(
    IncarnationId,
    "Identity regenerated whenever a worker process starts."
);
bounded_id!(
    DeploymentId,
    "Immutable resolved model deployment identity."
);
bounded_id!(
    RequestId,
    "Logical request identity retained across an attempt."
);
bounded_id!(AttemptId, "Unique execution-attempt identity.");
bounded_id!(TenantId, "Trusted caller namespace identity.");
bounded_id!(CallerId, "Trusted caller identity.");
bounded_id!(SessionId, "Tenant-scoped stateful session identity.");
bounded_id!(ServiceId, "Authenticated internal service identity.");
bounded_id!(
    CredentialId,
    "Identifier for a rotatable internal credential."
);
bounded_id!(
    PolicyRevision,
    "Revision of the policy attested by a gateway."
);
bounded_id!(DeviceId, "Stable accelerator or resource-group identity.");
bounded_id!(ModelAlias, "Public model alias resolved by a gateway.");
bounded_id!(ArtifactRevision, "Pinned model artifact revision.");
bounded_id!(RequestDigest, "Tenant-scoped canonical invocation digest.");

/// Monotonic model-instance generation within a worker incarnation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct ModelGeneration(u64);

impl ModelGeneration {
    pub fn new(value: u64) -> Result<Self, IdentifierError> {
        if value == 0 {
            return Err(IdentifierError::Empty);
        }
        Ok(Self(value))
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

impl<'de> Deserialize<'de> for ModelGeneration {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = u64::deserialize(deserializer)?;
        Self::new(value).map_err(|_| de::Error::custom("model generation must be non-zero"))
    }
}

/// Secret bearer material for an internal service request.
///
/// It intentionally has no serialization implementation so it cannot be placed in protocol JSON
/// accidentally. HTTP clients should put it only in the authorization header.
#[derive(Clone, PartialEq, Eq)]
pub struct ServiceBearerToken(String);

impl ServiceBearerToken {
    pub fn new(value: impl Into<String>) -> Result<Self, IdentifierError> {
        let value = value.into();
        if value.is_empty() {
            return Err(IdentifierError::Empty);
        }
        if value.len() > MAX_SERVICE_TOKEN_BYTES {
            return Err(IdentifierError::TooLong {
                actual: value.len(),
                maximum: MAX_SERVICE_TOKEN_BYTES,
            });
        }
        if let Some((offset, _)) = value
            .bytes()
            .enumerate()
            .find(|(_, byte)| byte.is_ascii_control() || byte.is_ascii_whitespace())
        {
            return Err(IdentifierError::InvalidByte { offset });
        }
        Ok(Self(value))
    }

    pub fn expose_secret(&self) -> &str {
        &self.0
    }

    /// Compares presented bearer material through fixed-size digests so the
    /// comparison does not short-circuit on a secret byte or token length.
    pub fn matches_presented(&self, presented: &str) -> bool {
        let expected: [u8; 32] = Sha256::digest(self.0.as_bytes()).into();
        let actual: [u8; 32] = Sha256::digest(presented.as_bytes()).into();
        bool::from(expected.ct_eq(&actual))
    }
}

impl fmt::Debug for ServiceBearerToken {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ServiceBearerToken([REDACTED])")
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct ServiceCredentials {
    pub credential_id: CredentialId,
    pub bearer_token: ServiceBearerToken,
}

impl fmt::Debug for ServiceCredentials {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ServiceCredentials")
            .field("credential_id", &self.credential_id)
            .field("bearer_token", &self.bearer_token)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identities_validate_construction_and_deserialization() {
        let worker = WorkerId::new("node-a/worker:1").unwrap();
        assert_eq!(worker.as_str(), "node-a/worker:1");
        assert_eq!(
            serde_json::to_string(&worker).unwrap(),
            "\"node-a/worker:1\""
        );
        assert_eq!(
            serde_json::from_str::<WorkerId>("\"node-a/worker:1\"").unwrap(),
            worker
        );
        assert!(WorkerId::new("").is_err());
        assert!(WorkerId::new("contains space").is_err());
        let oversized = "a".repeat(MAX_ID_BYTES + 1);
        assert!(WorkerId::new(oversized.clone()).is_err());
        assert!(serde_json::from_value::<WorkerId>(serde_json::json!(oversized)).is_err());
    }

    #[test]
    fn model_generation_rejects_zero_on_both_paths() {
        assert!(ModelGeneration::new(0).is_err());
        assert!(serde_json::from_str::<ModelGeneration>("0").is_err());
        assert_eq!(
            serde_json::from_str::<ModelGeneration>("7").unwrap().get(),
            7
        );
    }

    #[test]
    fn service_token_debug_is_redacted_and_rejects_header_injection() {
        let token = ServiceBearerToken::new("top-secret").unwrap();
        assert_eq!(format!("{token:?}"), "ServiceBearerToken([REDACTED])");
        assert!(!format!("{token:?}").contains("top-secret"));
        assert!(token.matches_presented("top-secret"));
        assert!(!token.matches_presented("top-secreu"));
        assert!(!token.matches_presented("short"));
        assert!(ServiceBearerToken::new("bad\r\ntoken").is_err());
    }
}
