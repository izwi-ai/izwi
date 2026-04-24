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

pub fn ensure_cli_setup<R: tauri::Runtime>(app: &tauri::AppHandle<R>) -> Result<()> {
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

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn install_dir_copy(src: &std::path::Path, dest: &std::path::Path) -> Result<()> {
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
fn ensure_linux_cli_links<R: tauri::Runtime>(app: &tauri::AppHandle<R>) -> Result<()> {
    use std::io::Write;

    let Some(targets) = resolve_bundled_cli_targets(app)? else {
        return Ok(());
    };

    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .context("HOME is not set")?;
    let bin_dir = home.join(".local").join("bin");

    std::fs::create_dir_all(&bin_dir)
        .with_context(|| format!("failed creating {}", bin_dir.display()))?;

    install_binary_copy(&targets.cli, &bin_dir.join("izwi"))?;
    if let Some(server_target) = targets.server.as_ref() {
        install_binary_copy(server_target, &bin_dir.join("izwi-server"))?;
    }
    if let Some(runtime_dir) = targets.runtime_dir.as_ref() {
        install_dir_copy(runtime_dir, &bin_dir.join("runtime"))?;
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
    let Some(targets) = resolve_bundled_cli_targets(app)? else {
        return Ok(());
    };

    let bin_dir = std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("USERPROFILE").map(PathBuf::from))
        .map(|p| p.join("Izwi").join("bin"))
        .context("LOCALAPPDATA/USERPROFILE is not set")?;

    std::fs::create_dir_all(&bin_dir)
        .with_context(|| format!("failed creating {}", bin_dir.display()))?;

    install_binary_copy(&targets.cli, &bin_dir.join("izwi.exe"))?;
    if let Some(server_target) = targets.server.as_ref() {
        install_binary_copy(server_target, &bin_dir.join("izwi-server.exe"))?;
    }
    if let Some(runtime_dir) = targets.runtime_dir.as_ref() {
        install_dir_copy(runtime_dir, &bin_dir.join("runtime"))?;
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
