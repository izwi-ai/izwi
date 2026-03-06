use anyhow::Result;
use clap::Parser;

mod app;

fn main() -> Result<()> {
    app::run(app::DesktopArgs::parse())
}
