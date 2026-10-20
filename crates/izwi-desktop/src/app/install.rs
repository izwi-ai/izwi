use anyhow::{Context, Result};
use std::path::PathBuf;
use tauri::Manager;

use super::server::platform_binary_name;

struct BundledCliTargets {
    cli: PathBuf,
    server: Option<PathBuf>,
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    runtime_dir: Option<PathBuf>,
}

/// What a CLI setup pass may do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CliSetupMode {
    /// The user asked to install: create links or copies and add the
    /// directory to PATH, prompting for an administrator password on macOS
    /// when the link directory needs it.
    Install,
    /// App launch: only refresh copies a previous install created (Linux and
    /// Windows copy binaries, which must follow app updates). Never installs
    /// for the first time, edits PATH or prompts. macOS links point into the
    /// app bundle, so there is nothing to refresh.
    RefreshExisting,
}

/// Whether the `izwi` command can be installed from this app and whether a
/// shell can already find it.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CliInstallStatus {
    /// This build bundles the CLI (packaged app, not a dev build).
    pub available: bool,
    /// `izwi` is on PATH or in this app's install location.
    pub installed: bool,
    /// Where the command was found.
    pub location: Option<String>,
}

#[tauri::command]
pub fn cli_install_status<R: tauri::Runtime>(app: tauri::AppHandle<R>) -> CliInstallStatus {
    cli_status(&app)
}

/// Install the `izwi` command line tool on request. Runs on a blocking
/// thread: on macOS it may wait on an administrator password prompt.
#[tauri::command]
pub async fn install_cli<R: tauri::Runtime>(
    app: tauri::AppHandle<R>,
) -> std::result::Result<CliInstallStatus, String> {
    let handle = app.clone();
    tauri::async_runtime::spawn_blocking(move || ensure_cli_setup(&handle, CliSetupMode::Install))
        .await
        .map_err(|error| error.to_string())?
        .map_err(|error| format!("{error:#}"))?;
    Ok(cli_status(&app))
}

fn cli_status<R: tauri::Runtime>(app: &tauri::AppHandle<R>) -> CliInstallStatus {
    let available = matches!(resolve_bundled_cli_targets(app), Ok(Some(_)));
    let binary = platform_binary_name("izwi");
    let on_path = std::env::var_os("PATH").and_then(|path| {
        std::env::split_paths(&path)
            .map(|dir| dir.join(&binary))
            .find(|candidate| candidate.is_file())
    });
    let location = on_path.or_else(|| {
        cli_install_dir()
            .map(|dir| dir.join(&binary))
            .filter(|candidate| candidate.exists())
    });
    CliInstallStatus {
        available,
        installed: location.is_some(),
        location: location.map(|path| path.display().to_string()),
    }
}

/// Directory the installer puts `izwi` in on this platform.
fn cli_install_dir() -> Option<PathBuf> {
    #[cfg(target_os = "macos")]
    {
        return Some(preferred_path_bin_dir());
    }
    #[cfg(target_os = "linux")]
    {
        return std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local").join("bin"));
    }
    #[cfg(target_os = "windows")]
    {
        return windows_cli_bin_dir();
    }
    #[allow(unreachable_code)]
    None
}

pub fn ensure_cli_setup<R: tauri::Runtime>(
    app: &tauri::AppHandle<R>,
    mode: CliSetupMode,
) -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        if mode == CliSetupMode::Install && is_running_from_macos_app_bundle() {
            ensure_macos_cli_links(app)?;
        }
        return Ok(());
    }

    #[cfg(target_os = "linux")]
    {
        ensure_linux_cli_links(app, mode)?;
        return Ok(());
    }

    #[cfg(target_os = "windows")]
    {
        ensure_windows_cli_links(app, mode)?;
        return Ok(());
    }

    #[allow(unreachable_code)]
    {
        let _ = (app, mode);
        Ok(())
    }
}

fn resolve_bundled_cli_targets<R: tauri::Runtime>(
    app: &tauri::AppHandle<R>,
) -> Result<Option<BundledCliTargets>> {
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

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    let runtime_dir = {
        let runtime_dir = resource_dir.join("bin").join("runtime");
        if runtime_dir.exists() {
            Some(runtime_dir)
        } else {
            None
        }
    };

    Ok(Some(BundledCliTargets {
        cli: cli_target,
        server: server_target,
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        runtime_dir,
    }))
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn path_contains_dir(dir: &std::path::Path) -> bool {
    let Some(path) = std::env::var_os("PATH") else {
        return false;
    };

    std::env::split_paths(&path).any(|entry| entry == dir)
}

/// Whether `src` and `dest` are one file. A per-user Windows install puts
/// the bundled binaries where the command line tool is installed: there is
/// nothing to copy, and removing the destination first would delete the
/// source (or fail on the running server).
#[cfg(any(target_os = "linux", target_os = "windows"))]
fn is_same_path(src: &std::path::Path, dest: &std::path::Path) -> bool {
    match (std::fs::canonicalize(src), std::fs::canonicalize(dest)) {
        (Ok(src), Ok(dest)) => src == dest,
        _ => false,
    }
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn install_binary_copy(src: &std::path::Path, dest: &std::path::Path) -> Result<()> {
    if is_same_path(src, dest) {
        return Ok(());
    }
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

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn install_dir_copy(src: &std::path::Path, dest: &std::path::Path) -> Result<()> {
    if is_same_path(src, dest) {
        return Ok(());
    }
    let existing = std::fs::symlink_metadata(dest).ok();
    if let Some(metadata) = existing {
        if metadata.file_type().is_dir() {
            std::fs::remove_dir_all(dest)
                .with_context(|| format!("failed removing existing {}", dest.display()))?;
        } else {
            std::fs::remove_file(dest)
                .with_context(|| format!("failed removing existing {}", dest.display()))?;
        }
    }

    copy_dir_recursive(src, dest)
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn copy_dir_recursive(src: &std::path::Path, dest: &std::path::Path) -> Result<()> {
    std::fs::create_dir_all(dest).with_context(|| format!("failed creating {}", dest.display()))?;

    for entry in
        std::fs::read_dir(src).with_context(|| format!("failed reading {}", src.display()))?
    {
        let entry = entry.with_context(|| format!("failed reading entry in {}", src.display()))?;
        let src_path = entry.path();
        let dest_path = dest.join(entry.file_name());
        let file_type = entry
            .file_type()
            .with_context(|| format!("failed reading metadata for {}", src_path.display()))?;

        if file_type.is_dir() {
            copy_dir_recursive(&src_path, &dest_path)?;
        } else {
            std::fs::copy(&src_path, &dest_path).with_context(|| {
                format!(
                    "failed copying {} -> {}",
                    src_path.display(),
                    dest_path.display()
                )
            })?;

            let perms = std::fs::metadata(&src_path)
                .with_context(|| format!("failed reading metadata for {}", src_path.display()))?
                .permissions();
            std::fs::set_permissions(&dest_path, perms).with_context(|| {
                format!("failed setting permissions for {}", dest_path.display())
            })?;
        }
    }

    Ok(())
}

#[cfg(target_os = "linux")]
fn ensure_linux_cli_links<R: tauri::Runtime>(
    app: &tauri::AppHandle<R>,
    mode: CliSetupMode,
) -> Result<()> {
    use std::io::Write;

    let Some(targets) = resolve_bundled_cli_targets(app)? else {
        return Ok(());
    };

    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .context("HOME is not set")?;
    let bin_dir = home.join(".local").join("bin");
    if mode == CliSetupMode::RefreshExisting && !bin_dir.join("izwi").exists() {
        return Ok(());
    }

    std::fs::create_dir_all(&bin_dir)
        .with_context(|| format!("failed creating {}", bin_dir.display()))?;

    install_binary_copy(&targets.cli, &bin_dir.join("izwi"))?;
    if let Some(server_target) = targets.server.as_ref() {
        install_binary_copy(server_target, &bin_dir.join("izwi-server"))?;
    }
    if let Some(runtime_dir) = targets.runtime_dir.as_ref() {
        install_dir_copy(runtime_dir, &bin_dir.join("runtime"))?;
    }

    if mode == CliSetupMode::Install && !path_contains_dir(&bin_dir) {
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
fn windows_cli_bin_dir() -> Option<PathBuf> {
    std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("USERPROFILE").map(PathBuf::from))
        .map(|p| p.join("Izwi").join("bin"))
}

#[cfg(target_os = "windows")]
fn ensure_windows_cli_links<R: tauri::Runtime>(
    app: &tauri::AppHandle<R>,
    mode: CliSetupMode,
) -> Result<()> {
    let Some(targets) = resolve_bundled_cli_targets(app)? else {
        return Ok(());
    };

    let bin_dir = windows_cli_bin_dir().context("LOCALAPPDATA/USERPROFILE is not set")?;
    if mode == CliSetupMode::RefreshExisting && !bin_dir.join("izwi.exe").exists() {
        return Ok(());
    }

    std::fs::create_dir_all(&bin_dir)
        .with_context(|| format!("failed creating {}", bin_dir.display()))?;

    install_binary_copy(&targets.cli, &bin_dir.join("izwi.exe"))?;
    if let Some(server_target) = targets.server.as_ref() {
        install_binary_copy(server_target, &bin_dir.join("izwi-server.exe"))?;
    }
    if let Some(runtime_dir) = targets.runtime_dir.as_ref() {
        install_dir_copy(runtime_dir, &bin_dir.join("runtime"))?;
    }

    if mode == CliSetupMode::Install && !path_contains_dir(&bin_dir) {
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

    let mut command = std::process::Command::new("powershell");
    let status = super::console::hide_console_window(&mut command)
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

    let Some(targets) = resolve_bundled_cli_targets(app)? else {
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
    if targets.server.is_some() && has_non_symlink_collision(&server_link)? {
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

    if let Err(err) = ensure_symlink(&targets.cli, &cli_link) {
        needs_privileged_install = true;
        eprintln!("warning: {}", err);
    }

    if let Some(server_target) = targets.server.as_ref() {
        if let Err(err) = ensure_symlink(server_target, &server_link) {
            needs_privileged_install = true;
            eprintln!("warning: {}", err);
        }
    }

    if needs_privileged_install {
        let mut shell_cmd = vec![
            format!("mkdir -p '{}'", escape_single_quotes(&link_dir)),
            format!(
                "ln -sf '{}' '{}'",
                escape_single_quotes(&targets.cli),
                escape_single_quotes(&cli_link)
            ),
        ];

        if let Some(server_target) = targets.server.as_ref() {
            shell_cmd.push(format!(
                "ln -sf '{}' '{}'",
                escape_single_quotes(server_target),
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
                    manual_link_command(&targets.cli, &cli_link)
                );
                if let Some(server_target) = targets.server.as_ref() {
                    eprintln!(
                        "run manually: {}",
                        manual_link_command(server_target, &server_link)
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

#[cfg(test)]
mod tests {
    #[cfg(target_os = "macos")]
    use super::manual_link_command;

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    #[test]
    fn installing_a_binary_onto_itself_keeps_it() {
        let dir = std::env::temp_dir().join(format!("izwi-install-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let binary = dir.join("izwi");
        std::fs::write(&binary, b"binary").unwrap();
        super::install_binary_copy(&binary, &binary).unwrap();
        assert_eq!(std::fs::read(&binary).unwrap(), b"binary");
        std::fs::remove_dir_all(&dir).unwrap();
    }
    #[cfg(target_os = "macos")]
    use std::path::Path;

    #[cfg(target_os = "macos")]
    #[test]
    fn manual_link_command_quotes_paths() {
        let command = manual_link_command(Path::new("/tmp/izwi"), Path::new("/usr/local/bin/izwi"));
        assert!(command.contains("sudo ln -sf"));
        assert!(command.contains("/tmp/izwi"));
    }
}
