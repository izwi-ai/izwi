//! Authenticated speech connection quotas. Permits follow response body lifetime,
//! including streaming disconnects; request headers never supply tenant identity.
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

#[derive(Default)]
struct Counts {
    tenants: HashMap<String, usize>,
    total: usize,
}

#[derive(Default)]
struct Admission(Mutex<Counts>);

pub(super) struct Permit {
    admission: Arc<Admission>,
    tenant: String,
}

impl Drop for Permit {
    fn drop(&mut self) {
        let mut counts = self.admission.0.lock().unwrap_or_else(|e| e.into_inner());
        counts.total -= 1;
        if let Some(active) = counts.tenants.get_mut(&self.tenant) {
            *active -= 1;
            if *active == 0 {
                counts.tenants.remove(&self.tenant);
            }
        }
    }
}

impl Admission {
    fn acquire(
        self: &Arc<Self>,
        tenant: String,
        global: usize,
        per_tenant: usize,
    ) -> Result<Permit, axum::http::StatusCode> {
        let mut counts = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if counts.tenants.get(&tenant).copied().unwrap_or(0) >= per_tenant {
            return Err(axum::http::StatusCode::TOO_MANY_REQUESTS);
        }
        if counts.total >= global {
            return Err(axum::http::StatusCode::SERVICE_UNAVAILABLE);
        }
        *counts.tenants.entry(tenant.clone()).or_default() += 1;
        counts.total += 1;
        Ok(Permit {
            admission: self.clone(),
            tenant,
        })
    }
}

pub(super) fn acquire(
    principal: &izwi_hooks::Principal,
    global: usize,
) -> Result<Permit, axum::http::StatusCode> {
    static ADMISSION: OnceLock<Arc<Admission>> = OnceLock::new();
    let global = global.max(1);
    let per_tenant = crate::env_config::tuning()
        .max_speech_requests_per_tenant
        .map_or(global, |limit| limit.min(global));
    // Distinguish namespaces so a principal ID cannot collide with a tenant ID.
    let tenant = super::principal_namespace(principal);
    ADMISSION
        .get_or_init(|| Arc::new(Admission::default()))
        .acquire(tenant, global, per_tenant)
}

pub(super) fn is_speech_generation(method: &axum::http::Method, path: &str) -> bool {
    (method == axum::http::Method::POST
        && (path.ends_with("/audio/speech")
            || path.contains("/text-to-speech")
            || path.contains("/voice-clone")
            || path.contains("/voice-design")))
        || (method == axum::http::Method::GET
            && path.contains("/text-to-speech/")
            && path.ends_with("/events"))
}

pub(super) fn guard_response(
    response: axum::response::Response,
    permit: Permit,
) -> axum::response::Response {
    use futures::StreamExt;
    let (parts, body) = response.into_parts();
    let stream = async_stream::stream! {
        let _permit = permit;
        let mut stream = body.into_data_stream();
        while let Some(frame) = stream.next().await { yield frame; }
    };
    axum::response::Response::from_parts(parts, axum::body::Body::from_stream(stream))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn replay_connections_share_speech_admission() {
        use axum::http::Method;
        assert!(is_speech_generation(
            &Method::GET,
            "/v1/text-to-speech/record/events"
        ));
        assert!(!is_speech_generation(
            &Method::GET,
            "/v1/text-to-speech/record"
        ));
        assert!(!is_speech_generation(
            &Method::GET,
            "/v1/text-to-speech/record/audio"
        ));
    }

    /// TPIPE-20: every synthesizing route shares the speech quota, voice
    /// design included.
    #[test]
    fn every_synthesis_route_shares_speech_admission() {
        use axum::http::Method;
        for path in [
            "/v1/audio/speech",
            "/v1/text-to-speech",
            "/v1/voice-clones",
            "/v1/voice-designs",
        ] {
            assert!(is_speech_generation(&Method::POST, path), "{path}");
        }
        assert!(!is_speech_generation(&Method::GET, "/v1/voice-designs"));
    }

    #[test]
    fn tenant_quota_preserves_peer_capacity_and_releases_all_metadata() {
        let admission = Arc::new(Admission::default());
        let a = admission.acquire("a".into(), 3, 1).unwrap();
        assert!(matches!(
            admission.acquire("a".into(), 3, 1),
            Err(axum::http::StatusCode::TOO_MANY_REQUESTS)
        ));
        let b = admission.acquire("b".into(), 3, 1).unwrap();
        let c = admission.acquire("c".into(), 3, 1).unwrap();
        assert!(matches!(
            admission.acquire("d".into(), 3, 1),
            Err(axum::http::StatusCode::SERVICE_UNAVAILABLE)
        ));
        drop(a);
        let d = admission.acquire("d".into(), 3, 1).unwrap();
        drop((b, c, d));
        let counts = admission.0.lock().unwrap();
        assert_eq!(counts.total, 0);
        assert!(counts.tenants.is_empty());
    }
    #[tokio::test]
    async fn permit_follows_stream_until_disconnect_and_unpolled_body_drop() {
        use futures::StreamExt;
        let admission = Arc::new(Admission::default());
        for poll in [false, true] {
            let permit = admission.acquire("a".into(), 1, 1).unwrap();
            let response = axum::response::Response::new(axum::body::Body::from("pcm"));
            let response = guard_response(response, permit);
            let mut body = response.into_body().into_data_stream();
            if poll {
                assert!(body.next().await.unwrap().is_ok());
            }
            assert_eq!(admission.0.lock().unwrap().total, 1);
            drop(body);
            assert_eq!(admission.0.lock().unwrap().total, 0);
        }
    }
}
