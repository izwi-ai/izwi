use axum::extract::Request;
use axum::http::HeaderValue;
use axum::middleware::Next;
use axum::response::Response;
use uuid::Uuid;

const REQUEST_ID_HEADER: &str = "x-request-id";

#[derive(Clone, Debug)]
pub struct RequestContext {
    pub correlation_id: String,
}

pub async fn attach_request_context(mut req: Request, next: Next) -> Response {
    let correlation_id = req
        .headers()
        .get(REQUEST_ID_HEADER)
        .and_then(|h| h.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| Uuid::new_v4().to_string());

    req.extensions_mut().insert(RequestContext {
        correlation_id: correlation_id.clone(),
    });

    if let Ok(value) = HeaderValue::from_str(&correlation_id) {
        req.headers_mut().insert(REQUEST_ID_HEADER, value);
    }

    let mut response = next.run(req).await;
    if let Ok(value) = correlation_id.parse() {
        response.headers_mut().insert(REQUEST_ID_HEADER, value);
    }
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        extract::Extension,
        http::{header::HeaderName, HeaderMap, Request},
        middleware,
        routing::get,
        Router,
    };
    use tower::Service;

    async fn echo_context(Extension(context): Extension<RequestContext>) -> String {
        context.correlation_id
    }

    async fn echo_context_and_header(
        headers: HeaderMap,
        Extension(context): Extension<RequestContext>,
    ) -> String {
        let header = headers
            .get(REQUEST_ID_HEADER)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default();
        format!("{}|{}", context.correlation_id, header)
    }

    #[tokio::test]
    async fn generated_request_id_is_inserted_and_returned() {
        let response = send_request(
            Router::new()
                .route("/", get(echo_context_and_header))
                .layer(middleware::from_fn(attach_request_context)),
            Request::builder()
                .uri("/")
                .body(Body::empty())
                .expect("request should build"),
        )
        .await;

        let request_id = response
            .headers()
            .get(REQUEST_ID_HEADER)
            .expect("request id header should exist")
            .to_str()
            .expect("request id should be ascii")
            .to_string();
        Uuid::parse_str(&request_id).expect("generated request id should be a UUID");

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should read");
        assert_eq!(body, format!("{request_id}|{request_id}").as_bytes());
    }

    #[tokio::test]
    async fn supplied_request_id_is_preserved() {
        let supplied = "client-request-123";
        let response = send_request(
            Router::new()
                .route("/", get(echo_context))
                .layer(middleware::from_fn(attach_request_context)),
            Request::builder()
                .uri("/")
                .header(HeaderName::from_static(REQUEST_ID_HEADER), supplied)
                .body(Body::empty())
                .expect("request should build"),
        )
        .await;

        assert_eq!(
            response
                .headers()
                .get(REQUEST_ID_HEADER)
                .and_then(|value| value.to_str().ok()),
            Some(supplied)
        );

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should read");
        assert_eq!(body, supplied.as_bytes());
    }

    async fn send_request(mut app: Router, request: Request<Body>) -> Response {
        app.as_service::<Body>()
            .call(request)
            .await
            .expect("request should succeed")
    }
}
