use anyhow::{Context, Result};
use clap::Parser;
use std::path::{Path, PathBuf};
use tauri::Manager;
use url::Url;

mod app;

use app::server::{is_local_server_host, platform_binary_name};

#[tauri::command]
async fn download_audio_file(
    url: String,
    suggested_filename: Option<String>,
) -> Result<String, String> {
    let task = tauri::async_runtime::spawn_blocking(move || {
        save_audio_from_url(url.as_str(), suggested_filename.as_deref())
    });

    let saved_path = task
        .await
        .map_err(|err| format!("audio download task failed: {err}"))?
        .map_err(|err| err.to_string())?;

    Ok(saved_path.to_string_lossy().to_string())
}

fn main() -> Result<()> {
    app::run(app::DesktopArgs::parse())
}

fn save_audio_from_url(url: &str, suggested_filename: Option<&str>) -> Result<PathBuf> {
    let parsed_url = Url::parse(url).with_context(|| format!("invalid audio URL: {url}"))?;
    if !matches!(parsed_url.scheme(), "http" | "https") {
        anyhow::bail!("unsupported audio URL scheme: {}", parsed_url.scheme());
    }

    let host = parsed_url
        .host_str()
        .context("audio URL is missing host")?
        .to_string();
    if !is_local_server_host(host.as_str()) {
        anyhow::bail!("audio download is allowed only from local Izwi server URLs");
    }

    let response = reqwest::blocking::get(parsed_url.clone())
        .with_context(|| format!("failed downloading audio from {parsed_url}"))?
        .error_for_status()
        .with_context(|| format!("audio download failed for {parsed_url}"))?;
    let bytes = response
        .bytes()
        .context("failed reading downloaded audio bytes")?;

    let filename = sanitize_download_filename(suggested_filename.unwrap_or("speech.wav"));
    let downloads_dir = dirs::download_dir()
        .or_else(|| std::env::current_dir().ok())
        .context("could not determine a downloads directory")?;
    std::fs::create_dir_all(&downloads_dir)
        .with_context(|| format!("failed creating {}", downloads_dir.display()))?;

    let destination = unique_download_path(&downloads_dir, filename.as_str());
    std::fs::write(&destination, bytes.as_ref())
        .with_context(|| format!("failed writing {}", destination.display()))?;

    Ok(destination)
}

fn sanitize_download_filename(raw: &str) -> String {
    let trimmed = raw.trim();
    let source = if trimmed.is_empty() {
        "speech.wav"
    } else {
        trimmed
    };

    let mut sanitized = String::with_capacity(source.len());
    for ch in source.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_' | ' ') {
            sanitized.push(ch);
        } else {
            sanitized.push('_');
        }
    }

    let sanitized = sanitized.trim().trim_matches('.').to_string();
    if sanitized.is_empty() {
        "speech.wav".to_string()
    } else {
        sanitized
    }
}

fn unique_download_path(downloads_dir: &Path, filename: &str) -> PathBuf {
    let first_path = downloads_dir.join(filename);
    if !first_path.exists() {
        return first_path;
    }

    let path = Path::new(filename);
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .unwrap_or("speech");
    let ext = path.extension().and_then(|value| value.to_str());

    for index in 1..=10_000usize {
        let candidate_name = match ext {
            Some(ext) if !ext.is_empty() => format!("{stem}-{index}.{ext}"),
            _ => format!("{stem}-{index}"),
        };
        let candidate_path = downloads_dir.join(candidate_name);
        if !candidate_path.exists() {
            return candidate_path;
        }
    }

    downloads_dir.join(format!("{stem}-{}.wav", std::process::id()))
}

fn ensure_cli_setup<R: tauri::Runtime>(app: &tauri::AppHandle<R>) -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        if is_running_from_macos_app_bundle() {
            ensure_macos_cli_links(app)?;
        }
        return Ok(());
    }

    #[cfg(target_os = "linux")]
    {
        ensure_linux_cli_links(app)?;
        return Ok(());
    }

    #[cfg(target_os = "windows")]
    {
        ensure_windows_cli_links(app)?;
        return Ok(());
    }

    #[allow(unreachable_code)]
    Ok(())
}

fn resolve_bundled_cli_targets<R: tauri::Runtime>(
    app: &tauri::AppHandle<R>,
) -> Result<Option<(PathBuf, Option<PathBuf>)>> {
    let resource_dir = match app.path().resource_dir() {
        Ok(path) => path,
        Err(err) => {
            if err.to_string().to_lowercase().contains("unknown path") {
                return Ok(None);
            }
            return Err(err.into());
        }
    };

    let cli_target = resource_dir.join("bin").join(platform_binary_name("izwi"));
    if !cli_target.exists() {
        return Ok(None);
    }

    let server_target = resource_dir
        .join("bin")
        .join(platform_binary_name("izwi-server"));
    let server_target = if server_target.exists() {
        Some(server_target)
    } else {
        None
    };

    Ok(Some((cli_target, server_target)))
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn path_contains_dir(dir: &std::path::Path) -> bool {
    let Some(path) = std::env::var_os("PATH") else {
        return false;
    };

    std::env::split_paths(&path).any(|entry| entry == dir)
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn install_binary_copy(src: &std::path::Path, dest: &std::path::Path) -> Result<()> {
    let existing = std::fs::symlink_metadata(dest).ok();
    if let Some(metadata) = existing {
        if metadata.file_type().is_dir() {
            anyhow::bail!("{} exists and is a directory", dest.display());
        }
        std::fs::remove_file(dest)
            .with_context(|| format!("failed removing existing {}", dest.display()))?;
    }

    std::fs::copy(src, dest)
        .with_context(|| format!("failed copying {} -> {}", src.display(), dest.display()))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let mut perms = std::fs::metadata(dest)
            .with_context(|| format!("failed reading metadata for {}", dest.display()))?
            .permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(dest, perms)
            .with_context(|| format!("failed setting executable mode for {}", dest.display()))?;
    }

    Ok(())
}

#[cfg(target_os = "linux")]
fn ensure_linux_cli_links<R: tauri::Runtime>(app: &tauri::AppHandle<R>) -> Result<()> {
    use std::io::Write;

    let Some((cli_target, server_target)) = resolve_bundled_cli_targets(app)? else {
        return Ok(());
    };

    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .context("HOME is not set")?;
    let bin_dir = home.join(".local").join("bin");

    std::fs::create_dir_all(&bin_dir)
        .with_context(|| format!("failed creating {}", bin_dir.display()))?;

    install_binary_copy(&cli_target, &bin_dir.join("izwi"))?;
    if let Some(server_target) = server_target.as_ref() {
        install_binary_copy(server_target, &bin_dir.join("izwi-server"))?;
    }

    if !path_contains_dir(&bin_dir) {
        let profile_path = home.join(".profile");
        let export_line = format!("export PATH=\"{}:$PATH\"", bin_dir.display());
        let existing = std::fs::read_to_string(&profile_path).unwrap_or_default();
        if !existing.lines().any(|line| line.trim() == export_line) {
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&profile_path)
                .with_context(|| format!("failed opening {}", profile_path.display()))?;
            if !existing.is_empty() && !existing.ends_with('\n') {
                writeln!(file)?;
            }
            writeln!(file, "{}", export_line)?;
            eprintln!(
                "info: appended {} to {} (restart shell to use `izwi`)",
                bin_dir.display(),
                profile_path.display()
            );
        }
    }

    Ok(())
}

#[cfg(target_os = "windows")]
fn ensure_windows_cli_links<R: tauri::Runtime>(app: &tauri::AppHandle<R>) -> Result<()> {
    let Some((cli_target, server_target)) = resolve_bundled_cli_targets(app)? else {
        return Ok(());
    };

    let bin_dir = std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("USERPROFILE").map(PathBuf::from))
        .map(|p| p.join("Izwi").join("bin"))
        .context("LOCALAPPDATA/USERPROFILE is not set")?;

    std::fs::create_dir_all(&bin_dir)
        .with_context(|| format!("failed creating {}", bin_dir.display()))?;

    install_binary_copy(&cli_target, &bin_dir.join("izwi.exe"))?;
    if let Some(server_target) = server_target.as_ref() {
        install_binary_copy(server_target, &bin_dir.join("izwi-server.exe"))?;
    }

    if !path_contains_dir(&bin_dir) {
        add_windows_user_path(&bin_dir)?;
        eprintln!(
            "info: added {} to user PATH (restart terminal to use `izwi`)",
            bin_dir.display()
        );
    }

    Ok(())
}

#[cfg(target_os = "windows")]
fn add_windows_user_path(path: &std::path::Path) -> Result<()> {
    let escaped = path.to_string_lossy().replace('\'', "''");
    let script = format!(
        "$target='{}';$current=[Environment]::GetEnvironmentVariable('Path','User');if(-not $current){{$current=''}};$parts=$current.Split(';')|Where-Object{{$_ -and $_.Trim() -ne ''}};if($parts -notcontains $target){{$new=if($current -and -not $current.EndsWith(';')){{$current+';'+$target}}elseif($current){{$current+$target}}else{{$target}};[Environment]::SetEnvironmentVariable('Path',$new,'User')}}",
        escaped
    );

    let status = std::process::Command::new("powershell")
        .arg("-NoProfile")
        .arg("-NonInteractive")
        .arg("-Command")
        .arg(script)
        .status()
        .context("failed running powershell to update PATH")?;

    if !status.success() {
        anyhow::bail!("powershell failed to update user PATH");
    }

    Ok(())
}

#[cfg(target_os = "macos")]
fn is_running_from_macos_app_bundle() -> bool {
    let exe = match std::env::current_exe() {
        Ok(path) => path,
        Err(_) => return false,
    };

    exe.components()
        .any(|component| component.as_os_str() == "Contents")
}

#[cfg(target_os = "macos")]
fn ensure_macos_cli_links<R: tauri::Runtime>(app: &tauri::AppHandle<R>) -> Result<()> {
    use std::process::Command;

    let Some((cli_target, server_target)) = resolve_bundled_cli_targets(app)? else {
        return Ok(());
    };

    let link_dir = preferred_path_bin_dir();
    let cli_link = link_dir.join("izwi");
    let server_link = link_dir.join("izwi-server");

    if has_non_symlink_collision(&cli_link)? {
        eprintln!(
            "warning: {} exists and is not a symlink; not overwriting",
            cli_link.display()
        );
        return Ok(());
    }
    if server_target.is_some() && has_non_symlink_collision(&server_link)? {
        eprintln!(
            "warning: {} exists and is not a symlink; not overwriting",
            server_link.display()
        );
        return Ok(());
    }

    let mut needs_privileged_install = false;

    if let Err(err) = std::fs::create_dir_all(&link_dir) {
        needs_privileged_install = true;
        eprintln!("warning: {}", err);
    }

    if let Err(err) = ensure_symlink(&cli_target, &cli_link) {
        needs_privileged_install = true;
        eprintln!("warning: {}", err);
    }

    if let Some(server_target) = server_target.as_ref() {
        if let Err(err) = ensure_symlink(&server_target, &server_link) {
            needs_privileged_install = true;
            eprintln!("warning: {}", err);
        }
    }

    if needs_privileged_install {
        let mut shell_cmd = vec![
            format!("mkdir -p '{}'", escape_single_quotes(&link_dir)),
            format!(
                "ln -sf '{}' '{}'",
                escape_single_quotes(&cli_target),
                escape_single_quotes(&cli_link)
            ),
        ];

        if let Some(server_target) = server_target.as_ref() {
            shell_cmd.push(format!(
                "ln -sf '{}' '{}'",
                escape_single_quotes(&server_target),
                escape_single_quotes(&server_link)
            ));
        }

        let shell_cmd = shell_cmd.join(" && ");
        let apple_script = format!(
            "do shell script \"{}\" with administrator privileges",
            escape_applescript(&shell_cmd)
        );

        match Command::new("osascript")
            .arg("-e")
            .arg(apple_script)
            .status()
        {
            Ok(status) if status.success() => return Ok(()),
            Ok(_) | Err(_) => {
                eprintln!("warning: automatic privileged setup was not completed");
                eprintln!(
                    "run manually: {}",
                    manual_link_command(&cli_target, &cli_link)
                );
                if let Some(server_target) = server_target.as_ref() {
                    eprintln!(
                        "run manually: {}",
                        manual_link_command(&server_target, &server_link)
                    );
                }
            }
        }
    }

    Ok(())
}

#[cfg(target_os = "macos")]
fn ensure_symlink(target: &std::path::Path, link: &std::path::Path) -> Result<()> {
    use std::os::unix::fs::symlink;

    let existing = std::fs::symlink_metadata(link).ok();
    if let Some(metadata) = existing {
        if metadata.file_type().is_symlink() {
            let current_target = std::fs::read_link(link)
                .with_context(|| format!("failed reading existing link {}", link.display()))?;
            if current_target == target {
                return Ok(());
            }

            std::fs::remove_file(link)
                .with_context(|| format!("failed removing stale link {}", link.display()))?;
        } else {
            anyhow::bail!("{} exists and is not a symlink", link.display());
        }
    }

    symlink(target, link).with_context(|| {
        format!(
            "failed to create symlink {} -> {}",
            link.display(),
            target.display()
        )
    })?;

    Ok(())
}

#[cfg(target_os = "macos")]
fn has_non_symlink_collision(path: &std::path::Path) -> Result<bool> {
    let existing = std::fs::symlink_metadata(path).ok();
    Ok(matches!(existing, Some(metadata) if !metadata.file_type().is_symlink()))
}

#[cfg(target_os = "macos")]
fn preferred_path_bin_dir() -> std::path::PathBuf {
    use std::path::PathBuf;

    let preferred = [
        PathBuf::from("/opt/homebrew/bin"),
        PathBuf::from("/usr/local/bin"),
    ];

    if let Some(path) = std::env::var_os("PATH") {
        for entry in std::env::split_paths(&path) {
            if preferred.iter().any(|candidate| candidate == &entry) {
                return entry;
            }
        }
    }

    if cfg!(target_arch = "aarch64") {
        PathBuf::from("/opt/homebrew/bin")
    } else {
        PathBuf::from("/usr/local/bin")
    }
}

#[cfg(target_os = "macos")]
fn manual_link_command(target: &std::path::Path, link: &std::path::Path) -> String {
    format!(
        "sudo ln -sf '{}' '{}'",
        escape_single_quotes(target),
        escape_single_quotes(link)
    )
}

#[cfg(target_os = "macos")]
fn escape_single_quotes(path: &std::path::Path) -> String {
    path.to_string_lossy().replace('\'', r"'\''")
}

#[cfg(target_os = "macos")]
fn escape_applescript(input: &str) -> String {
    input.replace('\\', "\\\\").replace('\"', "\\\"")
}
