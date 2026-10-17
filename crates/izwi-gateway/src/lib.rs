//! The gateway's accelerator-free components: the worker registry and its
//! routing policy, the approved-deployment table and shared approvals, tenant
//! rate quotas and concurrency admission, and worker TLS configuration.
//!
//! This crate must never depend on izwi-core (check-serving-boundary.sh
//! enforces it). The HTTP gateway role still lives in izwi-server, because its
//! chat path renders requests with izwi-core's model catalog; it moves here
//! once that catalog is split out of izwi-core.

// The lint policy these modules had in izwi-server.
#![allow(clippy::type_complexity)]

pub mod api;
pub mod error;
pub mod gateway_deployments;
pub mod gateway_rate_quota;
pub mod gateway_shared_approvals;
pub mod gateway_tenant_concurrency;
pub mod gateway_worker_tls;
pub mod ids;
pub mod lifecycle;
pub mod logging;
pub mod shutdown;
#[cfg(test)]
mod test_support;
pub mod worker_registry;
