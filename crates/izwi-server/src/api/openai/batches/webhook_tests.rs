//! Completion webhooks against a local listener: the signature, the URL
//! policy, and the retries.

use std::sync::{Arc, Mutex};

use axum::http::{HeaderMap, StatusCode};
use serde_json::json;

use super::*;

/// One delivery as the listener received it.
#[derive(Debug, Clone)]
pub(crate) struct Delivery {
    pub(crate) id: String,
    pub(crate) timestamp: String,
    pub(crate) signature: String,
    pub(crate) body: String,
}

/// A webhook endpoint on loopback that answers its deliveries with
/// `answers` in turn (the last one from then on) and keeps them.
pub(crate) struct Listener {
    pub(crate) url: String,
    deliveries: Arc<Mutex<Vec<Delivery>>>,
}

impl Listener {
    pub(crate) async fn start(answers: &'static [StatusCode]) -> Self {
        let deliveries = Arc::new(Mutex::new(Vec::new()));
        let received = deliveries.clone();
        let app = axum::Router::new().route(
            "/hook",
            axum::routing::post(move |headers: HeaderMap, body: String| async move {
                let header = |name: &str| {
                    headers
                        .get(name)
                        .and_then(|value| value.to_str().ok())
                        .unwrap_or_default()
                        .to_string()
                };
                let mut received = received.lock().unwrap();
                received.push(Delivery {
                    id: header("webhook-id"),
                    timestamp: header("webhook-timestamp"),
                    signature: header("webhook-signature"),
                    body,
                });
                answers[(received.len() - 1).min(answers.len() - 1)]
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let url = format!("http://{}/hook", listener.local_addr().expect("address"));
        tokio::spawn(async move { axum::serve(listener, app).await });
        Self { url, deliveries }
    }

    pub(crate) fn deliveries(&self) -> Vec<Delivery> {
        self.deliveries.lock().unwrap().clone()
    }
}

/// A policy with `secret` that may post to the loopback listener.
pub(crate) fn local_policy(secret: &str) -> WebhookPolicy {
    WebhookPolicy {
        secret: secret.to_string(),
        allow_http: true,
        allow_private: true,
        retry_backoff: Duration::from_millis(5),
    }
}

/// Check a delivery's signature the way a receiver does: HMAC-SHA256 of
/// `id.timestamp.body` under `key`, base64, after `v1,`.
pub(crate) fn assert_signed(delivery: &Delivery, key: &[u8]) {
    let mut mac = Hmac::<Sha256>::new_from_slice(key).unwrap();
    mac.update(format!("{}.{}.{}", delivery.id, delivery.timestamp, delivery.body).as_bytes());
    let signature = delivery
        .signature
        .strip_prefix("v1,")
        .expect("a v1 signature");
    let signature = base64::engine::general_purpose::STANDARD
        .decode(signature)
        .expect("base64");
    mac.verify_slice(&signature)
        .expect("the signature verifies");
}

#[tokio::test]
async fn a_delivery_is_signed_and_retried_until_it_is_accepted() {
    static ANSWERS: [StatusCode; 3] = [
        StatusCode::INTERNAL_SERVER_ERROR,
        StatusCode::FOUND,
        StatusCode::NO_CONTENT,
    ];
    let listener = Listener::start(&ANSWERS).await;
    // A `whsec_` secret is base64 of the key.
    let policy = local_policy("whsec_aXp3aS10ZXN0LWtleQ==");
    let body = json!({"id": "evt_1", "type": "batch.completed"}).to_string();

    let attempt = deliver(&policy, &listener.url, "evt_1", &body).await;

    assert_eq!(attempt, Ok(3));
    let deliveries = listener.deliveries();
    assert_eq!(deliveries.len(), 3);
    for delivery in &deliveries {
        assert_eq!(delivery.id, "evt_1");
        assert_eq!(delivery.body, body);
        assert!(delivery.timestamp.parse::<i64>().unwrap() > 1_700_000_000);
        assert_signed(delivery, b"izwi-test-key");
    }
    // Any other secret is the key as it is.
    assert_eq!(
        local_policy("plain").key(),
        b"plain",
        "a secret without the prefix signs as its bytes"
    );
}

#[tokio::test]
async fn a_delivery_is_dropped_after_three_attempts() {
    static ANSWERS: [StatusCode; 1] = [StatusCode::SERVICE_UNAVAILABLE];
    let listener = Listener::start(&ANSWERS).await;

    let result = deliver(&local_policy("secret"), &listener.url, "evt_2", "{}").await;

    assert_eq!(result, Err("the endpoint answered HTTP 503".to_string()));
    assert_eq!(listener.deliveries().len(), 3);
}

#[tokio::test]
async fn a_host_on_a_private_address_is_never_contacted() {
    static ANSWERS: [StatusCode; 1] = [StatusCode::OK];
    let listener = Listener::start(&ANSWERS).await;
    let policy = WebhookPolicy {
        allow_private: false,
        ..local_policy("secret")
    };

    // An address is refused before it is resolved, a name once it is.
    let by_address = deliver(&policy, &listener.url, "evt_3", "{}").await;
    let by_name = deliver(
        &policy,
        &listener.url.replace("127.0.0.1", "localhost"),
        "evt_3",
        "{}",
    )
    .await;

    for result in [by_address, by_name] {
        assert!(
            result
                .as_ref()
                .is_err_and(|reason| reason.contains("not publicly routable")),
            "{result:?}"
        );
    }
    assert!(listener.deliveries().is_empty());
}

#[test]
fn webhook_urls_are_checked_when_a_batch_is_created() {
    let metadata = |url: &str| json!({"webhook_url": url});
    let check = |policy: Option<&WebhookPolicy>, url: &str| {
        validate_metadata(policy, Some(&metadata(url))).map_err(|error| error.message)
    };
    let strict = WebhookPolicy {
        allow_http: false,
        allow_private: false,
        ..local_policy("secret")
    };

    // No webhook named: nothing to check, on or off.
    assert!(validate_metadata(None, Some(&json!({"team": "a"}))).is_ok());
    assert!(validate_metadata(None, None).is_ok());

    let off = check(None, "https://hooks.example.com/izwi").unwrap_err();
    assert!(off.contains("IZWI_BATCH_WEBHOOK_SECRET"), "{off}");

    assert!(check(Some(&strict), "https://hooks.example.com/izwi").is_ok());
    for (url, reason) in [
        ("http://hooks.example.com/izwi", "must use https"),
        ("ftp://hooks.example.com/izwi", "must use https"),
        ("https://user:pass@hooks.example.com/", "credentials"),
        ("https://127.0.0.1/hook", "not publicly routable"),
        ("https://10.1.2.3/hook", "not publicly routable"),
        ("https://169.254.169.254/latest", "not publicly routable"),
        ("https://[::1]/hook", "not publicly routable"),
        ("https://0.0.0.0/hook", "not publicly routable"),
        ("not a url", "invalid URL"),
    ] {
        let refused = check(Some(&strict), url).unwrap_err();
        assert!(refused.contains(reason), "{url}: {refused}");
    }

    let local = local_policy("secret");
    assert!(check(Some(&local), "http://127.0.0.1:9000/hook").is_ok());
    assert!(format!("{local:?}").contains("<redacted>"));
}
