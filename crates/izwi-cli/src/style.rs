use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::borrow::Cow;
use std::time::Duration;

pub const HELP_TEMPLATE: &str = r#"
{before-help}{name} {version}
{author-with-newline}{about-with-newline}
{usage-heading}
{tab}{usage}

{all-args}{after-help}
"#;

pub const BANNER: &str = r#"
 ___ ____ ____ 
|_ _|_  /_  /  High-performance audio inference
 | | / / / /   Text-to-Speech & Speech-to-Text
|___/___/___|  Optimized for Apple Silicon & CUDA
"#;

/// Theme for styled terminal output
#[derive(Clone)]
pub struct Theme {
    pub accent: fn(&str) -> console::StyledObject<&str>,
    pub success: fn(&str) -> console::StyledObject<&str>,
    pub warning: fn(&str) -> console::StyledObject<&str>,
    pub info: fn(&str) -> console::StyledObject<&str>,
    pub no_color: bool,
}

impl Default for Theme {
    fn default() -> Self {
        Self {
            accent: |s| style(s).cyan().bold(),
            success: |s| style(s).green().bold(),
            warning: |s| style(s).yellow(),
            info: |s| style(s).blue(),
            no_color: false,
        }
    }
}

impl Theme {
    pub fn no_color() -> Self {
        Self {
            accent: |s| style(s),
            success: |s| style(s),
            warning: |s| style(s),
            info: |s| style(s),
            no_color: true,
        }
    }

    pub fn print_banner(&self) {
        if self.no_color {
            println!("{}", BANNER.trim_start_matches('\n'));
        } else {
            println!("{}", (self.accent)(BANNER.trim_start_matches('\n')));
        }
    }

    pub fn success(&self, msg: &str) {
        println!("{} {}", (self.success)("✓"), msg);
    }

    pub fn warning(&self, msg: &str) {
        println!("{} {}", (self.warning)("⚠"), msg);
    }

    pub fn info(&self, msg: &str) {
        println!("{} {}", (self.info)("ℹ"), msg);
    }

    pub fn step(&self, n: usize, total: usize, msg: &str) {
        println!("{} {}", (self.accent)(&format!("[{}/{}]", n, total)), msg);
    }
}

/// Prints a failed command's error to stderr, on one line.
pub fn print_error(error: &dyn std::fmt::Display) {
    eprintln!("{} {}", style("error:").for_stderr().red().bold(), error);
}

/// A spinner with the elapsed time on stderr; hidden when stderr is not a
/// terminal, so it never mixes with output on stdout.
pub fn spinner(message: impl Into<Cow<'static, str>>) -> ProgressBar {
    let spinner = ProgressBar::new_spinner();
    if let Ok(template) =
        ProgressStyle::default_spinner().template("{spinner:.green} [{elapsed}] {msg}")
    {
        spinner.set_style(template);
    }
    spinner.set_message(message);
    spinner.enable_steady_tick(Duration::from_millis(120));
    spinner
}
