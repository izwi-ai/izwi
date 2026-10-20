//! "Show in Finder" for downloaded models: shows a path inside the Izwi
//! server's models folder in the OS file manager. Any other path is refused,
//! so the webview can't use it to open arbitrary folders.

use anyhow::{anyhow, Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;
use tauri::{AppHandle, Manager};
use url::Url;

use super::server::{is_local_server_host, ServerEndpoint};

const STORAGE_TIMEOUT: Duration = Duration::from_secs(5);

#[tauri::command]
pub async fn reveal_path(app: AppHandle, path: String) -> Result<(), String> {
    reveal(&app, Path::new(&path))
        .await
        .map_err(|error| format!("{error:#}"))
}

/// Show the app's logs in the file manager, for when the engine won't
/// start ("Open the log folder").
#[tauri::command]
pub async fn reveal_logs(app: AppHandle) -> Result<(), String> {
    let dir = app
        .path()
        .app_log_dir()
        .map_err(|error| format!("Izwi's log folder is unknown: {error}"))?;
    let log = dir.join(super::desktop_log::DESKTOP_LOG_FILE);
    let target = if log.exists() { log } else { dir };
    tauri::async_runtime::spawn_blocking(move || show_in_file_manager(&target))
        .await
        .map_err(|error| format!("reveal task failed: {error}"))?
        .map_err(|error| format!("{error:#}"))
}

async fn reveal(app: &AppHandle, path: &Path) -> Result<()> {
    let server_url = app
        .try_state::<ServerEndpoint>()
        .map(|endpoint| endpoint.url().clone())
        .context("Izwi's server isn't running")?;
    let models_dir = models_dir(&server_url).await?;
    let target = path_inside(&models_dir, path)?;
    tauri::async_runtime::spawn_blocking(move || show_in_file_manager(&target))
        .await
        .map_err(|error| anyhow!("reveal task failed: {error}"))?
}

/// The models folder the server uses, as it reports it.
async fn models_dir(server_url: &Url) -> Result<PathBuf> {
    // A remote server's folders aren't on this machine.
    if !server_url.host_str().is_some_and(is_local_server_host) {
        anyhow::bail!("models on a remote Izwi server can't be shown here");
    }
    let storage_url = server_url
        .join("/v1/admin/storage")
        .context("failed to build the storage URL")?;
    let storage: serde_json::Value = reqwest::Client::builder()
        .timeout(STORAGE_TIMEOUT)
        .build()
        .context("failed to build an HTTP client")?
        .get(storage_url)
        .send()
        .await
        .context("couldn't ask Izwi's server where models are stored")?
        .error_for_status()
        .context("couldn't ask Izwi's server where models are stored")?
        .json()
        .await
        .context("unexpected answer about where models are stored")?;
    storage
        .get("models_dir")
        .and_then(|value| value.as_str())
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .context("Izwi's server didn't say where models are stored")
}

/// `path` with links and `..` resolved, when it lies inside `root`.
fn path_inside(root: &Path, path: &Path) -> Result<PathBuf> {
    if !path.is_absolute() {
        anyhow::bail!("{} is not an absolute path", path.display());
    }
    let root = root
        .canonicalize()
        .with_context(|| format!("the models folder {} doesn't exist", root.display()))?;
    let resolved = path
        .canonicalize()
        .with_context(|| format!("{} doesn't exist", path.display()))?;
    if !resolved.starts_with(&root) {
        anyhow::bail!("only files in the models folder can be shown");
    }
    Ok(resolved)
}

fn show_in_file_manager(path: &Path) -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        let status = Command::new("open")
            .arg("-R")
            .arg(path)
            .status()
            .context("failed to run open")?;
        if !status.success() {
            anyhow::bail!("open -R {} failed ({status})", path.display());
        }
    }

    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        // Explorer wants `/select,"path"` as one raw argument, and exits 1
        // even when it worked, so its status says nothing.
        let path = without_verbatim_prefix(path);
        super::console::hide_console_window(&mut Command::new("explorer"))
            .raw_arg(format!("/select,\"{}\"", path.display()))
            .spawn()
            .context("failed to run explorer")?;
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let folder = path.parent().unwrap_or(path);
        let status = Command::new("xdg-open")
            .arg(folder)
            .status()
            .context("failed to run xdg-open")?;
        if !status.success() {
            anyhow::bail!("xdg-open {} failed ({status})", folder.display());
        }
    }

    Ok(())
}

/// `canonicalize` returns `\\?\C:\…` on Windows, which Explorer can't read.
#[cfg(any(target_os = "windows", test))]
fn without_verbatim_prefix(path: &Path) -> PathBuf {
    let text = path.to_string_lossy();
    if let Some(rest) = text.strip_prefix(r"\\?\UNC\") {
        PathBuf::from(format!(r"\\{rest}"))
    } else if let Some(rest) = text.strip_prefix(r"\\?\") {
        PathBuf::from(rest)
    } else {
        path.to_path_buf()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_models_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("izwi-reveal-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("models").join("Kokoro-82M")).unwrap();
        std::fs::create_dir_all(dir.join("elsewhere")).unwrap();
        dir
    }

    #[test]
    fn accepts_paths_inside_the_models_folder() {
        let dir = temp_models_dir("inside");
        let models = dir.join("models");
        let model = models.join("Kokoro-82M");

        assert_eq!(
            path_inside(&models, &model).unwrap(),
            model.canonicalize().unwrap()
        );
        assert!(path_inside(&models, &models).is_ok());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn refuses_paths_outside_the_models_folder() {
        let dir = temp_models_dir("outside");
        let models = dir.join("models");

        assert!(path_inside(&models, &dir.join("elsewhere")).is_err());
        assert!(path_inside(&models, &models.join("..").join("elsewhere")).is_err());
        assert!(path_inside(&models, Path::new("models/Kokoro-82M")).is_err());
        assert!(path_inside(&models, &models.join("missing")).is_err());
        // A folder whose name only starts like the models folder's.
        std::fs::create_dir_all(dir.join("models-old")).unwrap();
        assert!(path_inside(&models, &dir.join("models-old")).is_err());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(unix)]
    #[test]
    fn refuses_links_out_of_the_models_folder() {
        let dir = temp_models_dir("link");
        let models = dir.join("models");
        let link = models.join("escape");
        std::os::unix::fs::symlink(dir.join("elsewhere"), &link).unwrap();

        assert!(path_inside(&models, &link).is_err());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn strips_windows_verbatim_prefixes() {
        assert_eq!(
            without_verbatim_prefix(Path::new(r"\\?\C:\Users\me\models")),
            PathBuf::from(r"C:\Users\me\models")
        );
        assert_eq!(
            without_verbatim_prefix(Path::new(r"\\?\UNC\server\share\models")),
            PathBuf::from(r"\\server\share\models")
        );
        assert_eq!(
            without_verbatim_prefix(Path::new("/Users/me/models")),
            PathBuf::from("/Users/me/models")
        );
    }

    #[test]
    fn remote_servers_have_no_local_models_folder() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let error = runtime
            .block_on(models_dir(&Url::parse("http://10.0.0.5:8080").unwrap()))
            .unwrap_err();
        assert!(error.to_string().contains("remote"));
    }
}
