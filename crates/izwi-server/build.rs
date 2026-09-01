use std::env;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=IZWI_BUILD_GIT_SHA");

    let explicit = env::var("IZWI_BUILD_GIT_SHA")
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty());
    if explicit.is_none() {
        emit_git_rerun_paths();
    }
    let git = explicit.or_else(git_head);
    if let Some(git) = git {
        println!("cargo:rustc-env=IZWI_BUILD_GIT_SHA={git}");
    } else {
        println!(
            "cargo:warning=izwi-server build is not bound to a Git SHA; CUDA runtime evidence will fail closed"
        );
    }
}

fn emit_git_rerun_paths() {
    let Some(manifest_dir) = env::var("CARGO_MANIFEST_DIR").ok() else {
        return;
    };
    for git_path in ["HEAD", "packed-refs"] {
        if let Some(path) = resolve_git_path(&manifest_dir, git_path) {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    let symbolic = Command::new("git")
        .args(["-C", &manifest_dir, "symbolic-ref", "-q", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty());
    if let Some(symbolic) = symbolic {
        if let Some(path) = resolve_git_path(&manifest_dir, &symbolic) {
            println!("cargo:rerun-if-changed={path}");
        }
    }
}

fn resolve_git_path(manifest_dir: &str, name: &str) -> Option<String> {
    let output = Command::new("git")
        .args(["-C", manifest_dir, "rev-parse", "--git-path", name])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8(output.stdout).ok())
        .flatten()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn git_head() -> Option<String> {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").ok()?;
    let output = Command::new("git")
        .args(["-C", &manifest_dir, "rev-parse", "--verify", "HEAD"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}
