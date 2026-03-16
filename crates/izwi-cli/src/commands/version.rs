use crate::style::Theme;
use clap::crate_version;
use console::style;

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

        println!("\n{}", style("Features:").bold());
        if cfg!(target_os = "macos") {
            println!("  ✓ Metal GPU acceleration");
        }
        if std::env::var("CUDA_VISIBLE_DEVICES").is_ok() {
            println!("  ✓ CUDA support");
        }
        if cfg!(feature = "playback") {
            println!("  ✓ Audio playback");
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
