// Release builds run under the Windows GUI subsystem so launching the app does
// not open a console window. Debug builds keep the console for logs.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use anyhow::Result;
use clap::Parser;

mod app;

#[tokio::main]
async fn main() -> Result<()> {
    // The app's and the Aptabase and updater plugins' reqwest clients use rustls
    // without a built-in crypto provider; install ring before any is built.
    let _ = rustls::crypto::ring::default_provider().install_default();
    app::run(app::DesktopArgs::parse())
}
