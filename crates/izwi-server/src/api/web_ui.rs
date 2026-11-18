//! Where the web UI next to the API comes from.
//!
//! A release install (the terminal archives, the .deb, the CLI inside the
//! desktop app) has no `ui/dist` beside it, so the UI built before the
//! server is compiled into the binary. A configured UI directory that holds
//! an `index.html` still wins: a source checkout, Docker's `IZWI_UI_DIR` and
//! a user's own `ui.dir` keep serving from disk.

use std::borrow::Cow;
use std::path::{Path, PathBuf};

use axum::{
    body::Body,
    http::{header, HeaderValue, StatusCode, Uri},
    response::{IntoResponse, Response},
};
use rust_embed::RustEmbed;

/// The `ui/dist` build present when this binary was compiled, or nothing
/// when there was none (CI lanes and Docker's Rust stage build without it).
/// Debug builds read the folder at run time instead of embedding it.
#[derive(RustEmbed)]
#[folder = "../../ui/dist"]
#[allow_missing = true]
struct BuiltInUi;

const INDEX: &str = "index.html";

/// Vite names every file under `assets/` by its content hash, so a changed
/// file is a new URL and the old one can be cached for good.
const HASHED_ASSET_CACHE_CONTROL: &str = "public, max-age=31536000, immutable";

/// Everything else, the index above all, is revalidated so an updated
/// server never keeps a stale UI on screen.
const REVALIDATE_CACHE_CONTROL: &str = "no-cache";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum UiSource {
    /// Serve this folder from disk.
    Directory(PathBuf),
    /// Serve the UI compiled into this binary.
    BuiltIn,
    /// No UI anywhere: serve the API only.
    Missing,
}

pub(crate) fn select_ui_source(ui_dir: &Path, has_built_in: bool) -> UiSource {
    if ui_dir.join(INDEX).is_file() {
        UiSource::Directory(ui_dir.to_path_buf())
    } else if has_built_in {
        UiSource::BuiltIn
    } else {
        UiSource::Missing
    }
}

pub(crate) fn has_built_in_ui() -> bool {
    BuiltInUi::get(INDEX).is_some()
}

pub(crate) async fn serve_built_in_ui(uri: Uri) -> Response {
    built_in_ui_response(uri.path(), |path| {
        BuiltInUi::get(path).map(|file| file.data)
    })
}

/// Answers a UI request the way the folder service does: a file is served
/// as itself, and any other path, including one with a trailing slash, gets
/// the index so the UI's router can handle deep links.
pub(crate) fn built_in_ui_response(
    request_path: &str,
    lookup: impl Fn(&str) -> Option<Cow<'static, [u8]>>,
) -> Response {
    let path = request_path.trim_start_matches('/');
    if !path.is_empty() && !path.ends_with('/') {
        if let Some(body) = lookup(path) {
            return asset_response(path, body);
        }
    }
    match lookup(INDEX) {
        Some(body) => asset_response(INDEX, body),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

fn asset_response(path: &str, body: Cow<'static, [u8]>) -> Response {
    let content_type = mime_guess::from_path(path)
        .first_raw()
        .unwrap_or("application/octet-stream");
    let cache_control = if path.starts_with("assets/") {
        HASHED_ASSET_CACHE_CONTROL
    } else {
        REVALIDATE_CACHE_CONTROL
    };
    (
        [
            (header::CONTENT_TYPE, HeaderValue::from_static(content_type)),
            (
                header::CACHE_CONTROL,
                HeaderValue::from_static(cache_control),
            ),
        ],
        Body::from(body),
    )
        .into_response()
}
