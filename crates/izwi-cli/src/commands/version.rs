use crate::style::Theme;
use clap::crate_version;
use console::style;

fn compiled_backends() -> Vec<&'static str> {
    let mut backends = vec!["CPU"];

    if cfg!(feature = "metal") {
        backends.push("Metal");
    }
    if cfg!(feature = "cuda") {
        backends.push("CUDA");
    }

    backends
}

pub fn execute(full: bool, theme: &Theme) {
    theme.print_banner();

    println!("{}", style(format!("Version: {}", crate_version!())).bold());

    if full {
        println!("\n{}", style("Build Info:").bold());
        println!(
            "  Target:    {}-{}",
            std::env::consts::OS,
            std::env::consts::ARCH
        );
        println!(
            "  Rust:      {}.{}.{}",
            rustc_version_runtime::version().major,
            rustc_version_runtime::version().minor,
            rustc_version_runtime::version().patch
        );

        println!("\n{}", style("Compiled Backends:").bold());
        for backend in compiled_backends() {
            println!("  ✓ {}", backend);
        }

        println!("\n{}", style("Features:").bold());
        if cfg!(feature = "playback") {
            println!("  ✓ Audio playback");
        } else {
            println!("  (none)");
        }

        println!("\n{}", style("License:").bold());
        println!("  Apache 2.0");
    }

    println!();
    println!(
        "For more information, visit: {}",
        style("https://github.com/izwi-ai/izwi").cyan()
    );
}
