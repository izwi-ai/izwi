use std::env;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=IZWI_BUILD_GIT_SHA");

    let explicit = env::var("IZWI_BUILD_GIT_SHA")
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty());
    let git = explicit.or_else(git_head);
    if let Some(git) = git {
        println!("cargo:rustc-env=IZWI_BUILD_GIT_SHA={git}");
    } else {
        println!(
            "cargo:warning=izwi-server build is not bound to a Git SHA; CUDA runtime evidence will fail closed"
        );
    }
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
