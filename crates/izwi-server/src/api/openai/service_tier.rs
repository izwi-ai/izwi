//! `service_tier` on the speech routes.
//!
//! OpenAI defines `service_tier` for Responses and Chat Completions; Izwi
//! takes the same field and values on `/v1/audio/*`. `flex` marks a request
//! as batch work, which yields to interactive and streaming requests;
//! `auto` and `default` keep the route's own class; `priority` is accepted
//! and served as `default`. `/v1/audio/speech/batch` is batch work unless
//! the request names another tier. A multipart client that cannot add a form field
//! sends the `X-Izwi-Service-Tier` header instead; the body field wins when
//! both are present.
//!
//! Every response names the tier that served it in `X-Izwi-Service-Tier`.
//! JSON bodies and the final SSE event carry `service_tier` only when the
//! request named a tier, so a client that does not use tiers reads the
//! bodies it always has.

use axum::http::{HeaderName, HeaderValue};
use axum::response::Response;
use izwi_core::WorkloadClass;

use crate::error::ApiError;

/// The request and response header carrying a service tier.
pub(crate) const SERVICE_TIER_HEADER: HeaderName = HeaderName::from_static("x-izwi-service-tier");

/// The values `service_tier` takes, for error messages.
const SUPPORTED_TIERS: &str = "auto, default, flex, priority";

/// The tier a request is served at, and whether the request named one.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ServiceTier {
    flex: bool,
    requested: bool,
}

impl ServiceTier {
    /// The tier of a request whose body field is `field` and whose
    /// `X-Izwi-Service-Tier` header is `header`. A blank value is no value.
    pub(crate) fn parse(
        field: Option<&str>,
        header: Option<&HeaderValue>,
    ) -> Result<Self, ApiError> {
        let named = |value: &&str| !value.trim().is_empty();
        let value = match field.filter(named) {
            Some(field) => Some(field),
            None => header
                .map(|header| header.to_str().unwrap_or("(not text)"))
                .filter(named),
        };
        let Some(value) = value else {
            return Ok(Self::default());
        };
        let flex = match value.trim().to_ascii_lowercase().as_str() {
            "flex" => true,
            "auto" | "default" | "priority" => false,
            other => {
                return Err(ApiError::invalid_param(
                    "service_tier",
                    format!("Unsupported service_tier: {other}. Supported: {SUPPORTED_TIERS}"),
                ))
            }
        };
        Ok(Self {
            flex,
            requested: true,
        })
    }

    /// The tier of a batch surface: `flex` unless the request named one.
    pub(crate) fn or_flex(self) -> Self {
        Self {
            flex: self.flex || !self.requested,
            ..self
        }
    }

    /// The class a request of this tier runs as on a route whose own class
    /// is `route_class`: admission and the engine both use it.
    pub(crate) fn workload_class(self, route_class: WorkloadClass) -> WorkloadClass {
        if self.flex {
            WorkloadClass::Batch
        } else {
            route_class
        }
    }

    /// The tier the request is served at.
    pub(crate) fn as_str(self) -> &'static str {
        if self.flex {
            "flex"
        } else {
            "default"
        }
    }

    /// The `service_tier` of a JSON body or final SSE event: the tier
    /// served, when the request named one.
    pub(crate) fn reported(self) -> Option<&'static str> {
        self.requested.then(|| self.as_str())
    }

    /// Names the tier served in the response's `X-Izwi-Service-Tier`.
    pub(crate) fn stamp<B>(self, response: &mut Response<B>) {
        response
            .headers_mut()
            .insert(SERVICE_TIER_HEADER, HeaderValue::from_static(self.as_str()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(value: &'static str) -> HeaderValue {
        HeaderValue::from_static(value)
    }

    #[test]
    fn flex_runs_as_batch_and_every_other_tier_keeps_the_route_class() {
        for (value, served, class) in [
            ("flex", "flex", WorkloadClass::Batch),
            ("FLEX ", "flex", WorkloadClass::Batch),
            ("auto", "default", WorkloadClass::Streaming),
            ("default", "default", WorkloadClass::Streaming),
            ("priority", "default", WorkloadClass::Streaming),
        ] {
            let tier = ServiceTier::parse(Some(value), None).expect(value);
            assert_eq!(tier.as_str(), served, "{value}");
            assert_eq!(tier.reported(), Some(served), "{value}");
            assert_eq!(
                tier.workload_class(WorkloadClass::Streaming),
                class,
                "{value}"
            );
        }
    }

    #[test]
    fn a_request_without_a_tier_is_served_as_default_and_reports_none() {
        for tier in [
            ServiceTier::parse(None, None).unwrap(),
            ServiceTier::parse(Some(" "), Some(&header(""))).unwrap(),
        ] {
            assert_eq!(tier, ServiceTier::default());
            assert_eq!(tier.as_str(), "default");
            assert_eq!(tier.reported(), None);
            assert_eq!(
                tier.workload_class(WorkloadClass::Interactive),
                WorkloadClass::Interactive
            );
        }
    }

    #[test]
    fn a_batch_surface_is_flex_until_the_request_names_a_tier() {
        let unnamed = ServiceTier::default().or_flex();
        assert_eq!(unnamed.as_str(), "flex");
        assert_eq!(unnamed.reported(), None);
        assert_eq!(
            unnamed.workload_class(WorkloadClass::Interactive),
            WorkloadClass::Batch
        );
        for (value, served, class) in [
            ("flex", "flex", WorkloadClass::Batch),
            ("default", "default", WorkloadClass::Interactive),
            ("auto", "default", WorkloadClass::Interactive),
            ("priority", "default", WorkloadClass::Interactive),
        ] {
            let tier = ServiceTier::parse(Some(value), None).unwrap().or_flex();
            assert_eq!(tier.reported(), Some(served), "{value}");
            assert_eq!(
                tier.workload_class(WorkloadClass::Interactive),
                class,
                "{value}"
            );
        }
    }

    #[test]
    fn the_header_names_the_tier_unless_the_body_does() {
        let from_header = ServiceTier::parse(None, Some(&header("flex"))).unwrap();
        assert_eq!(from_header.reported(), Some("flex"));
        let body_wins = ServiceTier::parse(Some("default"), Some(&header("flex"))).unwrap();
        assert_eq!(body_wins.reported(), Some("default"));
        // The header is not read when the body names the tier.
        assert!(ServiceTier::parse(Some("flex"), Some(&header("fastest"))).is_ok());
    }

    #[test]
    fn an_unknown_tier_is_an_invalid_service_tier_parameter() {
        for (field, header_value) in [(Some("fastest"), None), (None, Some(header("scale")))] {
            let error = ServiceTier::parse(field, header_value.as_ref()).unwrap_err();
            assert_eq!(error.status, axum::http::StatusCode::BAD_REQUEST);
            let detail = error.detail.expect("detail");
            assert_eq!(detail.param, Some("service_tier"));
            assert_eq!(detail.code, Some("invalid_value"));
            assert!(error.message.contains("auto, default, flex, priority"));
        }
    }

    #[test]
    fn the_response_header_names_the_tier_served() {
        let mut response = Response::new(());
        ServiceTier::parse(Some("priority"), None)
            .unwrap()
            .stamp(&mut response);
        assert_eq!(response.headers()[SERVICE_TIER_HEADER], "default");
        ServiceTier::parse(None, Some(&header("flex")))
            .unwrap()
            .stamp(&mut response);
        assert_eq!(response.headers()[SERVICE_TIER_HEADER], "flex");
    }
}
