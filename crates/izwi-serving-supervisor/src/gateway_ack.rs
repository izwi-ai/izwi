//! Gateway-acknowledged rollout steps (CP-5).
//!
//! A rollout that knows its gateways' admin endpoints does not guess when a
//! view was adopted: after writing one it polls every gateway's
//! `GET /internal/admin/approvals` until each reports the view adopted, and
//! only then moves on. A gateway that cannot be reached, or answers without
//! the view, counts as not yet acknowledged until the timeout.

use std::time::Duration;

use izwi_serving_protocol::{GatewayApprovalsStatus, GATEWAY_APPROVALS_STATUS_PATH};
use tokio::time::Instant;

use crate::rollout::GatewayAckConfig;

const POLL_INTERVAL: Duration = Duration::from_millis(250);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(2);
/// A status report is a few KiB; refuse anything far larger.
const MAX_STATUS_BYTES: usize = 1024 * 1024;

pub struct GatewayAck {
    urls: Vec<String>,
    admin_key: String,
    timeout: Duration,
    client: reqwest::Client,
}

impl std::fmt::Debug for GatewayAck {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GatewayAck")
            .field("urls", &self.urls)
            .field("admin_key", &"[REDACTED]")
            .field("timeout", &self.timeout)
            .finish()
    }
}

impl GatewayAck {
    pub fn new(config: &GatewayAckConfig, admin_key: String) -> Result<Self, reqwest::Error> {
        Ok(Self {
            urls: config
                .admin_urls
                .iter()
                .map(|url| url.trim_end_matches('/').to_string())
                .collect(),
            admin_key,
            timeout: Duration::from_secs(config.timeout_secs),
            client: reqwest::Client::builder()
                .timeout(REQUEST_TIMEOUT)
                .build()?,
        })
    }

    pub fn gateways(&self) -> usize {
        self.urls.len()
    }

    /// Polls every gateway until each reports a status that satisfies
    /// `adopted`. On timeout returns the gateways that never did.
    pub async fn wait_for(
        &self,
        adopted: impl Fn(&GatewayApprovalsStatus) -> bool,
    ) -> Result<(), Vec<String>> {
        let deadline = Instant::now() + self.timeout;
        let mut pending: Vec<&String> = self.urls.iter().collect();
        loop {
            let mut still_pending = Vec::new();
            for url in pending {
                match self.status(url).await {
                    Some(status) if adopted(&status) => {}
                    _ => still_pending.push(url),
                }
            }
            pending = still_pending;
            if pending.is_empty() {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(pending.into_iter().cloned().collect());
            }
            tokio::time::sleep(POLL_INTERVAL).await;
        }
    }

    async fn status(&self, url: &str) -> Option<GatewayApprovalsStatus> {
        let response = self
            .client
            .get(format!("{url}{GATEWAY_APPROVALS_STATUS_PATH}"))
            .bearer_auth(&self.admin_key)
            .send()
            .await
            .ok()?;
        if !response.status().is_success() {
            return None;
        }
        let body = response.bytes().await.ok()?;
        if body.len() > MAX_STATUS_BYTES {
            return None;
        }
        serde_json::from_slice(&body).ok()
    }
}
