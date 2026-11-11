//! `/v1/files` through the router: every route and error shape, the size
//! cap, expiry and tenant isolation.

use async_trait::async_trait;
use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use axum::response::Response;
use axum::Router;
use izwi_core::{RuntimeService, ServeRuntimeConfig};
use izwi_hooks::{AuthProvider, EnterpriseHooks, HookResult, Principal, RequestEnvelope};
use serde_json::Value;
use std::sync::Arc;
use tower::Service;

use crate::state::AppState;
use crate::test_support::env_lock;

/// Authenticates each request as the tenant its `x-test-tenant` names.
struct HeaderTenantAuth;

#[async_trait]
impl AuthProvider for HeaderTenantAuth {
    async fn authenticate(&self, request: &RequestEnvelope) -> HookResult<Principal> {
        let tenant = request
            .headers
            .iter()
            .find(|header| header.name.eq_ignore_ascii_case("x-test-tenant"))
            .map(|header| header.value.clone())
            .unwrap_or_else(|| "default".to_string());
        let mut principal = Principal::local_anonymous();
        principal.id = format!("{tenant}-user");
        principal.tenant_id = Some(tenant);
        Ok(principal)
    }
}

/// The API key of a [`TestServer::behind_gateway`].
const GATEWAY_UPSTREAM_KEY: &str = "files-test-upstream-key-123456";

/// A server on a temporary database and media directory whose requests
/// belong to the tenant `x-test-tenant` names.
pub(crate) struct TestServer {
    pub(crate) app: Router,
    pub(crate) state: AppState,
    pub(crate) root: std::path::PathBuf,
    /// The header that carries a request's `tenant`.
    tenant_header: &'static str,
    /// Sent with every request that brings no credential of its own.
    api_key: Option<&'static str>,
}

impl Drop for TestServer {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

impl TestServer {
    pub(crate) fn new(tag: &str) -> Self {
        let root = std::env::temp_dir().join(format!("izwi-{tag}-{}", uuid::Uuid::new_v4()));
        let models_dir = root.join("models");
        std::fs::create_dir_all(&models_dir).expect("models dir");
        Self::with_models_dir(root, models_dir)
    }

    pub(crate) fn with_models_dir(
        root: std::path::PathBuf,
        models_dir: std::path::PathBuf,
    ) -> Self {
        let mut hooks = EnterpriseHooks::noop();
        hooks.auth = Arc::new(HeaderTenantAuth);
        Self::with_hooks(root, models_dir, hooks, "x-test-tenant", None)
    }

    /// A server as a gateway's upstream: every request carries the server's
    /// API key, and `tenant` is the gateway's `X-Izwi-Gateway-Tenant` (none
    /// for an empty `tenant`, a caller that reaches the server directly),
    /// which the server trusts when `trusts_tenant_header`.
    pub(crate) fn behind_gateway(tag: &str, trusts_tenant_header: bool) -> Self {
        let root = std::env::temp_dir().join(format!("izwi-{tag}-{}", uuid::Uuid::new_v4()));
        let models_dir = root.join("models");
        std::fs::create_dir_all(&models_dir).expect("models dir");
        let hooks = crate::local_security::with_api_key(
            EnterpriseHooks::noop(),
            Some(crate::local_security::LocalApiKey::new(GATEWAY_UPSTREAM_KEY).expect("key")),
            trusts_tenant_header,
        );
        Self::with_hooks(
            root,
            models_dir,
            hooks,
            crate::api::request_context::GATEWAY_TENANT_HEADER,
            Some(GATEWAY_UPSTREAM_KEY),
        )
    }

    fn with_hooks(
        root: std::path::PathBuf,
        models_dir: std::path::PathBuf,
        hooks: EnterpriseHooks,
        tenant_header: &'static str,
        api_key: Option<&'static str>,
    ) -> Self {
        let serve_config = ServeRuntimeConfig {
            backend: izwi_core::backends::BackendPreference::Cpu,
            ui_enabled: false,
            models_dir,
            ..ServeRuntimeConfig::default()
        };
        let state = {
            let _guard = env_lock();
            std::env::set_var("IZWI_DB_PATH", root.join("izwi.sqlite3"));
            std::env::set_var("IZWI_MEDIA_DIR", root.join("media"));
            let runtime = RuntimeService::new(serve_config.engine_config()).expect("runtime");
            let state =
                AppState::with_enterprise_hooks(runtime, &serve_config, hooks).expect("app state");
            std::env::remove_var("IZWI_DB_PATH");
            std::env::remove_var("IZWI_MEDIA_DIR");
            state
        };
        state.lifecycle.mark_ready();
        Self {
            app: crate::api::create_router(state.clone(), &serve_config),
            state,
            root,
            tenant_header,
            api_key,
        }
    }

    pub(crate) async fn send(&self, mut request: Request<Body>) -> Response {
        if let Some(key) = self.api_key {
            request
                .headers_mut()
                .entry(header::AUTHORIZATION)
                .or_insert_with(|| format!("Bearer {key}").parse().expect("a header value"));
        }
        self.app
            .clone()
            .as_service::<Body>()
            .call(request)
            .await
            .expect("a response")
    }

    /// A request builder for `tenant`.
    fn request(&self, tenant: &str) -> axum::http::request::Builder {
        let builder = Request::builder();
        if tenant.is_empty() {
            return builder;
        }
        builder.header(self.tenant_header, tenant)
    }

    /// A request without a body from `tenant`.
    pub(crate) async fn call(&self, method: Method, path: &str, tenant: &str) -> Response {
        self.send(
            self.request(tenant)
                .method(method)
                .uri(path)
                .body(Body::empty())
                .unwrap(),
        )
        .await
    }

    /// A JSON request from `tenant`.
    pub(crate) async fn post_json(&self, path: &str, tenant: &str, body: Value) -> Response {
        self.send(
            self.request(tenant)
                .method(Method::POST)
                .uri(path)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
    }

    /// `POST /v1/files` from `tenant`, as the SDK sends it.
    pub(crate) async fn upload(
        &self,
        tenant: &str,
        purpose: Option<&str>,
        filename: &str,
        content: &[u8],
    ) -> Response {
        let boundary = "izwi-test-boundary";
        let mut body = Vec::new();
        if let Some(purpose) = purpose {
            body.extend_from_slice(
                format!(
                    "--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\n{purpose}\r\n"
                )
                .as_bytes(),
            );
        }
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(content);
        body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
        self.send(
            self.request(tenant)
                .method(Method::POST)
                .uri("/v1/files")
                .header(
                    header::CONTENT_TYPE,
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
    }

    /// Upload a file that must be accepted; returns its id.
    pub(crate) async fn upload_ok(
        &self,
        tenant: &str,
        purpose: &str,
        filename: &str,
        content: &[u8],
    ) -> String {
        let response = self.upload(tenant, Some(purpose), filename, content).await;
        assert_eq!(response.status(), StatusCode::OK);
        json(response).await["id"].as_str().unwrap().to_string()
    }

    /// The content of `tenant`'s file `id`.
    pub(crate) async fn content(&self, tenant: &str, id: &str) -> Vec<u8> {
        let response = self
            .call(Method::GET, &format!("/v1/files/{id}/content"), tenant)
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        bytes(response).await
    }
}

pub(crate) async fn bytes(response: Response) -> Vec<u8> {
    axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("a body")
        .to_vec()
}

pub(crate) async fn json(response: Response) -> Value {
    serde_json::from_slice(&bytes(response).await).expect("a JSON body")
}

#[tokio::test]
async fn a_file_is_uploaded_read_listed_and_deleted() {
    let server = TestServer::new("files-lifecycle");
    let content = b"{\"custom_id\":\"a\"}\n";
    let response = server
        .upload("a", Some("batch"), "../dir/input.jsonl", content)
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    let file = json(response).await;
    let id = file["id"].as_str().unwrap().to_string();
    assert!(id.starts_with("file-"));
    assert_eq!(file["object"], "file");
    assert_eq!(file["bytes"], content.len());
    assert_eq!(file["filename"], "input.jsonl");
    assert_eq!(file["purpose"], "batch");
    assert!(file["created_at"].as_i64().unwrap() > 1_700_000_000);
    // Uploads are kept until deleted.
    assert!(file.get("expires_at").is_none());

    let fetched = json(
        server
            .call(Method::GET, &format!("/v1/files/{id}"), "a")
            .await,
    )
    .await;
    assert_eq!(fetched, file);
    assert_eq!(server.content("a", &id).await, content);

    let deleted = json(
        server
            .call(Method::DELETE, &format!("/v1/files/{id}"), "a")
            .await,
    )
    .await;
    assert_eq!(
        deleted,
        serde_json::json!({"id": id, "object": "file", "deleted": true})
    );
    for path in [format!("/v1/files/{id}"), format!("/v1/files/{id}/content")] {
        let response = server.call(Method::GET, &path, "a").await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{path}");
        let error = json(response).await;
        assert_eq!(error["error"]["type"], "not_found_error");
        assert!(error["error"]["message"].as_str().unwrap().contains(&id));
    }
    let again = server
        .call(Method::DELETE, &format!("/v1/files/{id}"), "a")
        .await;
    assert_eq!(again.status(), StatusCode::NOT_FOUND);
    // Nothing is left in the media provider.
    assert_eq!(
        server
            .state
            .artifact_store
            .cleanup_due(16)
            .await
            .unwrap()
            .deferred,
        0
    );
}

#[tokio::test]
async fn uploads_are_validated() {
    let server = TestServer::new("files-validation");
    for (purpose, content, param) in [
        (None, &b"x"[..], "purpose"),
        (Some("assistants"), &b"x"[..], "purpose"),
        (Some("batch_output"), &b"x"[..], "purpose"),
        (Some("batch"), &b""[..], "file"),
    ] {
        let response = server.upload("a", purpose, "f.jsonl", content).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{purpose:?}");
        let error = json(response).await;
        assert_eq!(error["error"]["param"], param, "{purpose:?}");
    }
    let unsupported = json(server.upload("a", Some("fine-tune"), "f.jsonl", b"x").await).await;
    assert!(unsupported["error"]["message"]
        .as_str()
        .unwrap()
        .contains("batch, user_data"));

    // A file past the cap is refused, and nothing is stored.
    server.state.file_store.set_max_bytes_for_test(8);
    let response = server
        .upload("a", Some("batch"), "f.jsonl", b"123456789")
        .await;
    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    assert!(json(response).await["error"]["message"]
        .as_str()
        .unwrap()
        .contains("IZWI_FILES_MAX_BYTES"));
    let at_the_cap = server
        .upload("a", Some("batch"), "f.jsonl", b"12345678")
        .await;
    assert_eq!(at_the_cap.status(), StatusCode::OK);
    let list = json(server.call(Method::GET, "/v1/files", "a").await).await;
    assert_eq!(list["data"].as_array().unwrap().len(), 1);

    for path in ["/v1/files/not-a-file", "/v1/files/file-00"] {
        let response = server.call(Method::GET, path, "a").await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{path}");
    }
}

#[tokio::test]
async fn files_are_listed_by_purpose_in_pages_and_either_order() {
    let server = TestServer::new("files-list");
    let mut ids = Vec::new();
    for (index, purpose) in ["batch", "user_data", "batch"].into_iter().enumerate() {
        ids.push(
            server
                .upload_ok("a", purpose, &format!("f{index}"), b"x")
                .await,
        );
    }
    let listed = |list: &Value| -> Vec<String> {
        list["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|file| file["id"].as_str().unwrap().to_string())
            .collect()
    };

    let all = json(server.call(Method::GET, "/v1/files", "a").await).await;
    assert_eq!(all["object"], "list");
    assert_eq!(all["has_more"], false);
    let newest_first = listed(&all);
    assert_eq!(newest_first.len(), 3);
    assert_eq!(all["first_id"], newest_first[0].as_str());
    assert_eq!(all["last_id"], newest_first[2].as_str());

    let oldest_first =
        listed(&json(server.call(Method::GET, "/v1/files?order=asc", "a").await).await);
    assert_eq!(
        oldest_first,
        newest_first.iter().rev().cloned().collect::<Vec<_>>()
    );

    let first_page = json(server.call(Method::GET, "/v1/files?limit=2", "a").await).await;
    assert_eq!(first_page["has_more"], true);
    assert_eq!(listed(&first_page), newest_first[..2]);
    let second_page = json(
        server
            .call(
                Method::GET,
                &format!("/v1/files?limit=2&after={}", newest_first[1]),
                "a",
            )
            .await,
    )
    .await;
    assert_eq!(second_page["has_more"], false);
    assert_eq!(listed(&second_page), newest_first[2..]);

    let audio = json(
        server
            .call(Method::GET, "/v1/files?purpose=user_data", "a")
            .await,
    )
    .await;
    assert_eq!(listed(&audio), vec![ids[1].clone()]);

    for query in ["order=sideways", "purpose=vision", "after=file-unknown"] {
        let response = server
            .call(Method::GET, &format!("/v1/files?{query}"), "a")
            .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{query}");
    }
}

#[tokio::test]
async fn a_file_is_visible_to_its_tenant_only() {
    let server = TestServer::new("files-tenants");
    let id = server.upload_ok("a", "batch", "f.jsonl", b"secret").await;
    for (method, path) in [
        (Method::GET, format!("/v1/files/{id}")),
        (Method::GET, format!("/v1/files/{id}/content")),
        (Method::DELETE, format!("/v1/files/{id}")),
    ] {
        let response = server.call(method.clone(), &path, "b").await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{method} {path}");
    }
    let other = json(server.call(Method::GET, "/v1/files", "b").await).await;
    assert_eq!(other["data"].as_array().unwrap().len(), 0);
    assert_eq!(other["first_id"], Value::Null);
    // The other tenant's delete removed nothing.
    assert_eq!(server.content("a", &id).await, b"secret");
}

/// The ids `tenant` lists.
async fn listed_ids(server: &TestServer, tenant: &str) -> Vec<String> {
    json(server.call(Method::GET, "/v1/files", tenant).await).await["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|file| file["id"].as_str().unwrap().to_string())
        .collect()
}

#[tokio::test]
async fn a_trusted_gateway_tenant_header_keeps_each_tenants_files_apart() {
    let server = TestServer::behind_gateway("files-gateway-tenants", true);
    // A caller that reaches the server directly names no tenant.
    let direct = server.upload_ok("", "batch", "f.jsonl", b"direct").await;
    let of_a = server.upload_ok("tenant-a", "batch", "f.jsonl", b"a").await;
    let of_b = server.upload_ok("tenant-b", "batch", "f.jsonl", b"b").await;

    for (tenant, own) in [("", &direct), ("tenant-a", &of_a), ("tenant-b", &of_b)] {
        assert_eq!(listed_ids(&server, tenant).await, std::slice::from_ref(own));
    }
    for (tenant, id) in [("tenant-b", &of_a), ("", &of_a), ("tenant-a", &direct)] {
        for (method, path) in [
            (Method::GET, format!("/v1/files/{id}")),
            (Method::GET, format!("/v1/files/{id}/content")),
            (Method::DELETE, format!("/v1/files/{id}")),
        ] {
            let response = server.call(method.clone(), &path, tenant).await;
            assert_eq!(
                response.status(),
                StatusCode::NOT_FOUND,
                "{tenant:?} {method} {path}"
            );
        }
    }
    assert_eq!(server.content("tenant-a", &of_a).await, b"a");
    assert_eq!(server.content("", &direct).await, b"direct");

    // A tenant the gateway would never send is refused, not guessed at.
    let too_long = "t".repeat(129);
    for tenant in ["tenant a", "tenant/a", "tenant,b", too_long.as_str()] {
        let response = server.call(Method::GET, "/v1/files", tenant).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{tenant:?}");
    }

    // The header counts only with the server's API key.
    let keyless = server
        .send(
            server
                .request("tenant-a")
                .uri("/v1/files")
                .header(header::AUTHORIZATION, "Bearer not-the-upstream-key-123")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
    assert_eq!(keyless.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn an_untrusted_gateway_tenant_header_is_ignored() {
    let server = TestServer::behind_gateway("files-gateway-untrusted", false);
    let id = server.upload_ok("tenant-a", "batch", "f.jsonl", b"a").await;
    for tenant in ["tenant-b", "", "not a tenant"] {
        assert_eq!(
            listed_ids(&server, tenant).await,
            std::slice::from_ref(&id),
            "{tenant:?}"
        );
        assert_eq!(server.content(tenant, &id).await, b"a");
    }
}

#[tokio::test]
async fn server_written_output_expires_and_is_swept() {
    let server = TestServer::new("files-expiry");
    let tenant = crate::artifact_store::ArtifactTenant::parse("anonymous").unwrap();
    let mut spool = crate::file_store::FileSpool::new().unwrap();
    spool.write(b"{}\n").await.unwrap();
    let file = server
        .state
        .file_store
        .create(
            &tenant,
            crate::file_store::FilePurpose::BatchOutput,
            "output.jsonl".to_string(),
            spool,
        )
        .await
        .unwrap();
    assert_eq!(file.size_bytes, 3);
    // Thirty days, as OpenAI keeps batch output.
    assert_eq!(
        file.expires_at.unwrap() - file.created_at,
        30 * 24 * 60 * 60 * 1000
    );
    let store = &server.state.file_store;
    assert_eq!(store.delete_expired(16).await.unwrap(), 0);
    assert!(store.get(&tenant, &file.id).await.unwrap().is_some());

    // An expired file is gone at once, and the sweep removes its bytes.
    store.set_expires_at_for_test(&file.id, 1).await.unwrap();
    assert!(store.get(&tenant, &file.id).await.unwrap().is_none());
    assert!(store.open(&tenant, &file.id).await.unwrap().is_none());
    assert_eq!(store.delete_expired(16).await.unwrap(), 1);
    assert_eq!(store.delete_expired(16).await.unwrap(), 0);
}
