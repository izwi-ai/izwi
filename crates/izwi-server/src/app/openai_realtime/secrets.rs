//! Ephemeral client secrets (`POST /v1/realtime/client_secrets`): short-lived
//! `ek_…` keys a server mints with its real API key and hands to a browser,
//! which cannot set an `Authorization` header on a WebSocket. The browser
//! presents the key as the `openai-insecure-api-key.<key>` subprotocol (or a
//! bearer token); the key admits `/v1/realtime` and its media-stream framing
//! only, until it expires, and
//! the session config minted with it applies to every session it opens.
//!
//! Keys live in memory, hashed, for at most two hours; a restart forgets them.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;
use sha2::{Digest, Sha256};

/// OpenAI's limits on a secret's lifetime, and its default.
pub(crate) const MIN_TTL_SECS: u64 = 10;
pub(crate) const MAX_TTL_SECS: u64 = 7_200;
pub(crate) const DEFAULT_TTL_SECS: u64 = 600;
/// The most live keys the process keeps.
const MAX_KEYS: usize = 10_000;
/// The subprotocol prefix browsers present a key with.
pub(crate) const SUBPROTOCOL_KEY_PREFIX: &str = "openai-insecure-api-key.";

struct Entry {
    expires_at: u64,
    session: Value,
}

fn keys() -> &'static Mutex<HashMap<[u8; 32], Entry>> {
    static KEYS: OnceLock<Mutex<HashMap<[u8; 32], Entry>>> = OnceLock::new();
    KEYS.get_or_init(Mutex::default)
}

fn digest(key: &str) -> [u8; 32] {
    Sha256::digest(key.as_bytes()).into()
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Mint a key for `session` (the validated session config it binds) that
/// expires `ttl_secs` from now. Returns the key and its expiry (unix secs).
pub(crate) fn mint(session: Value, ttl_secs: u64) -> Result<(String, u64), String> {
    let key = format!(
        "ek_{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    );
    let expires_at = now() + ttl_secs.clamp(MIN_TTL_SECS, MAX_TTL_SECS);
    let mut keys = keys().lock().unwrap_or_else(|poison| poison.into_inner());
    let now = now();
    keys.retain(|_, entry| entry.expires_at > now);
    if keys.len() >= MAX_KEYS {
        return Err("Too many live client secrets; let some expire first.".to_string());
    }
    keys.insert(
        digest(&key),
        Entry {
            expires_at,
            session,
        },
    );
    Ok((key, expires_at))
}

/// The session config a live key binds; `None` for an unknown or expired key.
pub(crate) fn lookup(key: &str) -> Option<Value> {
    if !key.starts_with("ek_") {
        return None;
    }
    let keys = keys().lock().unwrap_or_else(|poison| poison.into_inner());
    keys.get(&digest(key))
        .filter(|entry| entry.expires_at > now())
        .map(|entry| entry.session.clone())
}

/// The ephemeral key a request presents: an `openai-insecure-api-key.ek_…`
/// subprotocol, or an `ek_…` bearer token. `headers` are (lowercase name,
/// value) pairs.
pub(crate) fn presented_key<'a>(
    headers: impl Iterator<Item = (&'a str, &'a str)>,
) -> Option<String> {
    let mut bearer = None;
    for (name, value) in headers {
        match name {
            "sec-websocket-protocol" => {
                if let Some(key) = value
                    .split(',')
                    .map(str::trim)
                    .find_map(|protocol| protocol.strip_prefix(SUBPROTOCOL_KEY_PREFIX))
                    .filter(|key| key.starts_with("ek_"))
                {
                    return Some(key.to_string());
                }
            }
            "authorization" => {
                bearer = value
                    .split_once(' ')
                    .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"))
                    .map(|(_, token)| token.trim())
                    .filter(|token| token.starts_with("ek_"))
                    .map(str::to_string)
                    .or(bearer);
            }
            _ => {}
        }
    }
    bearer
}

/// Whether a request to `path` presents a live ephemeral key, which admits
/// the realtime socket and its media-stream framing only.
pub(crate) fn admits<'a>(
    method: &str,
    path: &str,
    headers: impl Iterator<Item = (&'a str, &'a str)>,
) -> bool {
    method == "GET"
        && matches!(path, "/v1/realtime" | "/v1/realtime/media-stream")
        && presented_key(headers).is_some_and(|key| lookup(&key).is_some())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn minted_keys_admit_the_realtime_socket_only_until_they_expire() {
        let (key, expires_at) =
            mint(json!({"type": "realtime", "instructions": "x"}), 60).expect("mint");
        assert!(key.starts_with("ek_"));
        assert!(expires_at > now());
        assert_eq!(lookup(&key).expect("live")["instructions"], "x");
        let protocol = format!("realtime, {SUBPROTOCOL_KEY_PREFIX}{key}");
        let headers = [("sec-websocket-protocol", protocol.as_str())];
        assert!(admits("GET", "/v1/realtime", headers.iter().copied()));
        assert!(admits(
            "GET",
            "/v1/realtime/media-stream",
            headers.iter().copied()
        ));
        assert!(!admits("GET", "/v1/models", headers.iter().copied()));
        assert!(!admits("POST", "/v1/realtime", headers.iter().copied()));
        let bearer = format!("Bearer {key}");
        assert!(admits(
            "GET",
            "/v1/realtime",
            [("authorization", bearer.as_str())].into_iter()
        ));
        assert!(lookup("ek_not_a_key").is_none());
        assert!(lookup("sk-not-ephemeral").is_none());

        // An expired key admits nothing.
        keys()
            .lock()
            .expect("keys")
            .get_mut(&digest(&key))
            .expect("entry")
            .expires_at = 0;
        assert!(lookup(&key).is_none());
    }
}
