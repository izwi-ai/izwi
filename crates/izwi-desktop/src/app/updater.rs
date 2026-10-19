use std::sync::Mutex;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tauri::{ipc::Channel, AppHandle, State};
use tauri_plugin_updater::{Update, UpdaterExt};
use tokio::time::sleep;
use url::Url;

use super::updater_contract::{
    github_manifest_url, github_releases_api_url, install_behavior_for_platform, is_newer_release,
    parse_beta_sequence, updater_target_for_platform, PlatformInstallBehavior, UpdaterContract,
};

const DEFAULT_REQUEST_TIMEOUT_SECONDS: u64 = 20;
const DEFAULT_MAX_CHECK_ATTEMPTS: usize = 3;
const DEFAULT_RETRY_BACKOFF_MS: u64 = 1500;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateMetadata {
    pub version: String,
    pub current_version: String,
    pub notes: Option<String>,
    pub published_at: Option<String>,
    pub release_tag: String,
    pub manifest_url: String,
    pub platform_behavior: PlatformInstallBehavior,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "event", content = "data")]
pub enum DownloadEvent {
    #[serde(rename_all = "camelCase")]
    Started {
        content_length: Option<u64>,
    },
    #[serde(rename_all = "camelCase")]
    Progress {
        chunk_length: usize,
        content_length: Option<u64>,
    },
    Finished,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InstallResult {
    pub app_exits_during_install: bool,
    pub supports_restart_later: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdaterHealthStatus {
    pub enabled: bool,
    pub disable_reason: Option<String>,
    pub request_timeout_ms: u64,
    pub max_check_attempts: usize,
    pub retry_backoff_ms: u64,
    pub forced_manifest_url: Option<String>,
}

#[derive(Debug, Clone)]
struct UpdaterRuntimeConfig {
    enabled: bool,
    disable_reason: Option<String>,
    request_timeout: Duration,
    max_check_attempts: usize,
    retry_backoff: Duration,
    forced_manifest_url: Option<Url>,
}

impl UpdaterRuntimeConfig {
    fn from_env() -> Self {
        let disable_flag = env_bool_true("IZWI_DISABLE_APP_UPDATES");
        // Without a pubkey the updater plugin is never registered, and asking
        // it for a builder would panic, so treat that build as update-less.
        let has_pubkey = configured_pubkey().is_some();
        let request_timeout_ms = env_u64(
            "IZWI_UPDATER_REQUEST_TIMEOUT_MS",
            DEFAULT_REQUEST_TIMEOUT_SECONDS * 1000,
        );
        let max_check_attempts = env_usize(
            "IZWI_UPDATER_MAX_CHECK_ATTEMPTS",
            DEFAULT_MAX_CHECK_ATTEMPTS,
        );
        let retry_backoff_ms = env_u64("IZWI_UPDATER_RETRY_BACKOFF_MS", DEFAULT_RETRY_BACKOFF_MS);
        let forced_manifest_url = std::env::var("IZWI_UPDATER_FORCE_MANIFEST_URL")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .and_then(|value| Url::parse(&value).ok());

        let disable_reason = if disable_flag {
            Some("IZWI_DISABLE_APP_UPDATES is set".to_string())
        } else if !has_pubkey {
            Some("this build has no updater public key".to_string())
        } else {
            None
        };

        Self {
            enabled: disable_reason.is_none(),
            disable_reason,
            request_timeout: Duration::from_millis(request_timeout_ms),
            max_check_attempts: max_check_attempts.max(1),
            retry_backoff: Duration::from_millis(retry_backoff_ms.max(1)),
            forced_manifest_url,
        }
    }

    fn health(&self) -> UpdaterHealthStatus {
        UpdaterHealthStatus {
            enabled: self.enabled,
            disable_reason: self.disable_reason.clone(),
            request_timeout_ms: self.request_timeout.as_millis() as u64,
            max_check_attempts: self.max_check_attempts,
            retry_backoff_ms: self.retry_backoff.as_millis() as u64,
            forced_manifest_url: self.forced_manifest_url.as_ref().map(Url::to_string),
        }
    }
}

#[derive(Default)]
pub struct UpdaterState {
    pending_update: Mutex<Option<Update>>,
}

impl UpdaterState {
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum UpdaterError {
    #[error(transparent)]
    Http(#[from] reqwest::Error),
    #[error(transparent)]
    Updater(#[from] tauri_plugin_updater::Error),
    #[error(transparent)]
    Url(#[from] url::ParseError),
    #[error("no matching beta release found for channel")]
    NoMatchingBetaRelease,
    #[error("there is no pending update")]
    NoPendingUpdate,
    #[error("in-app updates are disabled by runtime policy")]
    UpdatesDisabled,
}

impl Serialize for UpdaterError {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

type Result<T> = std::result::Result<T, UpdaterError>;

#[derive(Debug, Clone, Deserialize)]
struct GitHubRelease {
    tag_name: String,
    draft: bool,
    assets: Vec<GitHubReleaseAsset>,
}

#[derive(Debug, Clone, Deserialize)]
struct GitHubReleaseAsset {
    name: String,
}

#[derive(Debug, Clone)]
struct ResolvedRelease {
    tag_name: String,
    manifest_url: Url,
}

#[tauri::command]
pub async fn check_for_beta_update(
    app: AppHandle,
    updater_state: State<'_, UpdaterState>,
) -> Result<Option<UpdateMetadata>> {
    let runtime = UpdaterRuntimeConfig::from_env();
    if !runtime.enabled {
        return Err(UpdaterError::UpdatesDisabled);
    }

    let contract = UpdaterContract::default();
    let release = if let Some(forced_manifest_url) = runtime.forced_manifest_url.clone() {
        ResolvedRelease {
            tag_name: "forced-manifest".to_string(),
            manifest_url: forced_manifest_url,
        }
    } else {
        retry_with_backoff(runtime.max_check_attempts, runtime.retry_backoff, || {
            resolve_latest_release(&contract, runtime.request_timeout)
        })
        .await?
    };

    let updater_target = updater_target_for_platform(std::env::consts::OS);
    let app_handle = app.clone();
    let release_manifest_url = release.manifest_url.clone();
    let update = retry_with_backoff(runtime.max_check_attempts, runtime.retry_backoff, || {
        let app_handle = app_handle.clone();
        let release_manifest_url = release_manifest_url.clone();
        async move {
            let updater = app_handle
                .updater_builder()
                .timeout(runtime.request_timeout)
                .target(updater_target)
                .version_comparator(|current, release| is_newer_release(&current, &release.version))
                .endpoints(vec![release_manifest_url])?
                .build()?;
            updater.check().await.map_err(UpdaterError::from)
        }
    })
    .await?;

    let update_metadata = update.as_ref().map(|update| {
        let platform_behavior = install_behavior_for_platform(std::env::consts::OS);
        UpdateMetadata {
            version: update.version.clone(),
            current_version: update.current_version.clone(),
            notes: update.body.clone(),
            published_at: update.date.map(|date| date.to_string()),
            release_tag: release.tag_name.clone(),
            manifest_url: release.manifest_url.to_string(),
            platform_behavior,
        }
    });

    let mut pending_slot = updater_state
        .pending_update
        .lock()
        .expect("poisoned updater state mutex");
    *pending_slot = update;

    Ok(update_metadata)
}

#[tauri::command]
pub async fn install_beta_update(
    updater_state: State<'_, UpdaterState>,
    on_event: Channel<DownloadEvent>,
) -> Result<InstallResult> {
    let runtime = UpdaterRuntimeConfig::from_env();
    if !runtime.enabled {
        return Err(UpdaterError::UpdatesDisabled);
    }

    // Keep the pending update until it installs: taking it before the
    // download meant a failed download could not be retried ("there is no
    // pending update") until the next check.
    let update = {
        let pending_slot = updater_state
            .pending_update
            .lock()
            .expect("poisoned updater state mutex");
        pending_slot.clone().ok_or(UpdaterError::NoPendingUpdate)?
    };

    let _ = on_event.send(DownloadEvent::Started {
        content_length: None,
    });

    update
        .download_and_install(
            |chunk_length, content_length| {
                let _ = on_event.send(DownloadEvent::Progress {
                    chunk_length,
                    content_length,
                });
            },
            || {
                let _ = on_event.send(DownloadEvent::Finished);
            },
        )
        .await?;

    updater_state
        .pending_update
        .lock()
        .expect("poisoned updater state mutex")
        .take();

    let behavior = install_behavior_for_platform(std::env::consts::OS);

    Ok(InstallResult {
        app_exits_during_install: behavior.app_exits_during_install,
        supports_restart_later: behavior.supports_restart_later,
    })
}

#[tauri::command]
pub fn relaunch_after_update(app: AppHandle) {
    app.restart();
}

#[tauri::command]
pub fn updater_health_snapshot() -> UpdaterHealthStatus {
    UpdaterRuntimeConfig::from_env().health()
}

async fn resolve_latest_release(
    contract: &UpdaterContract,
    request_timeout: Duration,
) -> Result<ResolvedRelease> {
    let client = reqwest::Client::builder()
        .timeout(request_timeout)
        .build()?;
    let endpoint = github_releases_api_url(contract);

    let releases = client
        .get(endpoint)
        .header(reqwest::header::ACCEPT, "application/vnd.github+json")
        .header(
            reqwest::header::USER_AGENT,
            format!("izwi-desktop/{}", env!("CARGO_PKG_VERSION")),
        )
        .send()
        .await?
        .error_for_status()?
        .json::<Vec<GitHubRelease>>()
        .await?;

    let best_release =
        select_best_release(&releases, contract).ok_or(UpdaterError::NoMatchingBetaRelease)?;

    let manifest_url = Url::parse(&github_manifest_url(contract, &best_release.tag_name))?;
    Ok(ResolvedRelease {
        tag_name: best_release.tag_name.clone(),
        manifest_url,
    })
}

async fn retry_with_backoff<T, Fut, F>(
    max_attempts: usize,
    retry_backoff: Duration,
    mut operation: F,
) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    let attempts = max_attempts.max(1);
    let mut last_error: Option<UpdaterError> = None;

    for attempt in 1..=attempts {
        match operation().await {
            Ok(value) => return Ok(value),
            Err(error) => {
                last_error = Some(error);
                if attempt == attempts {
                    break;
                }
                sleep(retry_backoff * attempt as u32).await;
            }
        }
    }

    Err(last_error.unwrap_or(UpdaterError::NoMatchingBetaRelease))
}

/// The minisign public key update signatures are checked against: baked in
/// at compile time for release builds, or read from the environment.
pub fn configured_pubkey() -> Option<String> {
    option_env!("IZWI_UPDATER_PUBKEY")
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| {
            std::env::var("IZWI_UPDATER_PUBKEY")
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
        })
}

fn env_bool_true(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_ascii_lowercase())
        .map(|value| matches!(value.as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

fn env_u64(name: &str, fallback: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(fallback)
}

fn env_usize(name: &str, fallback: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(fallback)
}

fn release_has_manifest_asset(release: &GitHubRelease, contract: &UpdaterContract) -> bool {
    release
        .assets
        .iter()
        .any(|asset| asset.name == contract.manifest_asset_name)
}

fn select_best_release<'a>(
    releases: &'a [GitHubRelease],
    contract: &UpdaterContract,
) -> Option<&'a GitHubRelease> {
    releases
        .iter()
        .filter(|release| !release.draft)
        .filter(|release| release_has_manifest_asset(release, contract))
        .filter_map(|release| {
            parse_beta_sequence(contract.channel, &release.tag_name)
                .map(|sequence| (sequence, release))
        })
        .max_by_key(|(sequence, _)| *sequence)
        .map(|(_, release)| release)
}

#[cfg(test)]
mod tests {
    use super::{
        env_bool_true, env_u64, env_usize, release_has_manifest_asset, select_best_release,
        GitHubRelease, GitHubReleaseAsset, UpdaterRuntimeConfig, DEFAULT_MAX_CHECK_ATTEMPTS,
        DEFAULT_RETRY_BACKOFF_MS,
    };
    use crate::app::updater_contract::UpdaterContract;

    #[test]
    fn detects_manifest_asset_on_release() {
        let contract = UpdaterContract::default();
        let release = GitHubRelease {
            tag_name: "v0.1.0-beta-12".to_string(),
            draft: false,
            assets: vec![
                GitHubReleaseAsset {
                    name: "izwi_linux_x86_64.AppImage".to_string(),
                },
                GitHubReleaseAsset {
                    name: "latest-beta.json".to_string(),
                },
            ],
        };

        assert!(release_has_manifest_asset(&release, &contract));
    }

    #[test]
    fn selects_highest_beta_release_with_manifest() {
        let contract = UpdaterContract::default();
        let releases = vec![
            GitHubRelease {
                tag_name: "v0.1.0-beta-2".to_string(),
                draft: false,
                assets: vec![GitHubReleaseAsset {
                    name: "latest-beta.json".to_string(),
                }],
            },
            GitHubRelease {
                tag_name: "v0.1.0-beta-12".to_string(),
                draft: false,
                assets: vec![GitHubReleaseAsset {
                    name: "latest-beta.json".to_string(),
                }],
            },
            GitHubRelease {
                tag_name: "v0.1.0-beta-12".to_string(),
                draft: false,
                assets: vec![GitHubReleaseAsset {
                    name: "something-else.json".to_string(),
                }],
            },
        ];
        let selected = select_best_release(&releases, &contract);

        assert_eq!(
            selected.map(|release| release.tag_name.as_str()),
            Some("v0.1.0-beta-12")
        );
    }

    #[test]
    fn parses_runtime_env_helpers() {
        assert!(!env_bool_true("IZWI_DOES_NOT_EXIST"));
        assert_eq!(env_u64("IZWI_DOES_NOT_EXIST", 42), 42);
        assert_eq!(env_usize("IZWI_DOES_NOT_EXIST", 5), 5);
    }

    #[test]
    fn runtime_config_defaults_are_safe() {
        let runtime = UpdaterRuntimeConfig::from_env();
        assert!(runtime.max_check_attempts >= DEFAULT_MAX_CHECK_ATTEMPTS.min(1));
        assert!(runtime.retry_backoff.as_millis() >= DEFAULT_RETRY_BACKOFF_MS.min(1) as u128);
    }
}
