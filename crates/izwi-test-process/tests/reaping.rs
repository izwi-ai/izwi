//! Every process a guard spawned, its own children included, is gone after a
//! panicking test and after the test binary itself is killed.
#![cfg(unix)]

use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use izwi_test_process::ProcessGuard;

const HELPER_ENV: &str = "IZWI_TEST_PROCESS_HELPER";

/// A shell (the group leader) with a background `sleep` child; prints
/// `pids <leader> <child>` once both run.
fn spawn_tree() -> (ProcessGuard, u32, u32) {
    let mut guard = ProcessGuard::spawn(
        Command::new("/bin/sh")
            .arg("-c")
            .arg("sleep 600 & echo \"$!\"; wait")
            .stdout(Stdio::piped()),
    )
    .expect("spawn tree");
    let stdout = guard.stdout.take().expect("piped stdout");
    let mut line = String::new();
    BufReader::new(stdout)
        .read_line(&mut line)
        .expect("read child pid");
    let child = line.trim().parse().expect("child pid");
    let leader = guard.id();
    (guard, leader, child)
}

fn alive(pid: u32) -> bool {
    // SAFETY: signal 0 only checks that the process exists.
    unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
}

/// A killed process is a zombie until its new parent reaps it, so give
/// init a moment before calling it a leak.
fn assert_gone(pids: &[u32]) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while pids.iter().any(|pid| alive(*pid)) {
        assert!(
            Instant::now() < deadline,
            "processes {pids:?} outlived their guard"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn a_panicking_test_body_leaves_no_process_behind() {
    let (guard, leader, child) = spawn_tree();
    assert!(alive(leader) && alive(child));

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let _guard = guard;
        panic!("test body failed");
    }));

    assert!(result.is_err());
    assert_gone(&[leader, child]);
}

#[test]
fn a_killed_test_binary_leaves_no_process_behind() {
    let mut helper = Command::new(std::env::current_exe().expect("test binary path"))
        .args([
            "--exact",
            "helper_spawns_a_tree_and_waits",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(HELPER_ENV, "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn helper test binary");
    let stdout = helper.stdout.take().expect("piped stdout");
    let pids: Vec<u32> = BufReader::new(stdout)
        .lines()
        .map_while(Result::ok)
        .find_map(|line| {
            // libtest prints `test <name> ... ` on the same line first.
            let pids = line.split("pids ").nth(1)?;
            Some(pids.split(' ').map(|pid| pid.parse().unwrap()).collect())
        })
        .expect("helper reports its tree");
    assert!(pids.iter().all(|pid| alive(*pid)));

    // SIGKILL: the helper unwinds nothing and runs no Drop.
    helper.kill().expect("kill helper");
    helper.wait().expect("reap helper");

    assert_gone(&pids);
}

/// Run by `a_killed_test_binary_leaves_no_process_behind` in a child test
/// binary; a no-op otherwise.
#[test]
fn helper_spawns_a_tree_and_waits() {
    if std::env::var_os(HELPER_ENV).is_none() {
        return;
    }
    let (_guard, leader, child) = spawn_tree();
    println!("pids {leader} {child}");
    std::thread::sleep(Duration::from_secs(600));
}
