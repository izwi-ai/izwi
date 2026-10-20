//! A server an older Izwi left behind. Betas before 0.1.0 started their
//! server without an owner pipe, so after an update relaunches the app the
//! old server keeps the port with nothing to stop it. When the process on
//! the port runs this app's own server binary, it is ours to stop.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn output(command: &mut Command) -> Option<String> {
    let output = command.stderr(Stdio::null()).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).into_owned())
}

/// The first pid in `lsof -t` output.
fn parse_lsof_pid(text: &str) -> Option<u32> {
    text.lines().find_map(|line| line.trim().parse().ok())
}

/// The pid in `ss -ltnp` output: `users:(("izwi-server",pid=1234,fd=9))`.
fn parse_ss_pid(text: &str) -> Option<u32> {
    let start = text.find("pid=")? + "pid=".len();
    let digits: String = text[start..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().ok()
}

/// The pid listening on `port` in `netstat -ano -p TCP` output.
fn parse_netstat_pid(text: &str, port: u16) -> Option<u32> {
    let suffix = format!(":{port}");
    text.lines().find_map(|line| {
        let fields: Vec<&str> = line.split_whitespace().collect();
        match fields.as_slice() {
            [proto, local, _, state, pid]
                if proto.eq_ignore_ascii_case("TCP")
                    && state.eq_ignore_ascii_case("LISTENING")
                    && local.ends_with(&suffix) =>
            {
                pid.parse().ok()
            }
            _ => None,
        }
    })
}

/// The process listening on `port`, if the OS says.
pub(super) fn listening_pid(port: u16) -> Option<u32> {
    if cfg!(windows) {
        return output(Command::new("netstat").args(["-ano", "-p", "TCP"]))
            .and_then(|text| parse_netstat_pid(&text, port));
    }
    output(Command::new("lsof").args(["-nP", &format!("-iTCP:{port}"), "-sTCP:LISTEN", "-t"]))
        .and_then(|text| parse_lsof_pid(&text))
        .or_else(|| {
            output(Command::new("ss").args(["-ltnpH", &format!("sport = :{port}")]))
                .and_then(|text| parse_ss_pid(&text))
        })
}

/// The executable `pid` runs.
pub(super) fn process_executable(pid: u32) -> Option<PathBuf> {
    if cfg!(target_os = "linux") {
        return std::fs::read_link(format!("/proc/{pid}/exe")).ok();
    }
    let text = if cfg!(windows) {
        output(Command::new("powershell").args([
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            &format!("(Get-Process -Id {pid}).Path"),
        ]))
    } else {
        // macOS reports the full path a process was started with.
        output(Command::new("ps").args(["-o", "comm=", "-p", &pid.to_string()]))
    }?;
    let path = text.trim();
    (!path.is_empty()).then(|| PathBuf::from(path))
}

/// Whether `running` is the binary at `ours`. An update replaces the file
/// in place, so the old process still reports the same path.
pub(super) fn is_our_server(running: &Path, ours: &Path) -> bool {
    let canonical =
        |path: &Path| std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    canonical(running) == canonical(ours)
}

/// Ask `pid` to stop, or with `force` end it.
pub(super) fn terminate(pid: u32, force: bool) {
    let mut command = if cfg!(windows) {
        let mut command = Command::new("taskkill");
        command.args(["/PID", &pid.to_string(), "/T"]);
        if force {
            command.arg("/F");
        }
        command
    } else {
        let mut command = Command::new("kill");
        command.args([if force { "-KILL" } else { "-TERM" }, &pid.to_string()]);
        command
    };
    let _ = super::console::hide_console_window(&mut command)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_listening_pid_from_each_tool() {
        assert_eq!(parse_lsof_pid("4321\n"), Some(4321));
        assert_eq!(parse_lsof_pid(""), None);
        assert_eq!(
            parse_ss_pid(
                r#"LISTEN 0 1024 127.0.0.1:8080 0.0.0.0:* users:(("izwi-server",pid=2718,fd=9))"#
            ),
            Some(2718)
        );
        let netstat = "\
  Proto  Local Address          Foreign Address        State           PID
  TCP    0.0.0.0:135            0.0.0.0:0              LISTENING       988
  TCP    127.0.0.1:8080         0.0.0.0:0              LISTENING       5150
  TCP    127.0.0.1:18080        0.0.0.0:0              LISTENING       77
  TCP    127.0.0.1:52011        127.0.0.1:8080         ESTABLISHED     5150
";
        assert_eq!(parse_netstat_pid(netstat, 8080), Some(5150));
        assert_eq!(parse_netstat_pid(netstat, 9999), None);
    }

    #[test]
    fn only_this_apps_binary_counts_as_ours() {
        let dir = std::env::temp_dir().join(format!("izwi-stale-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ours = dir.join("izwi-server");
        std::fs::write(&ours, b"server").unwrap();
        assert!(is_our_server(&dir.join(".").join("izwi-server"), &ours));
        assert!(!is_our_server(
            Path::new("/usr/local/bin/izwi-server"),
            &ours
        ));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
