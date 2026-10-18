//! Spawns the processes integration tests run against (gateways, supervisors
//! that start their own workers, workers) so that none outlives its test.
//!
//! A bare [`std::process::Child`] is never killed on drop, so a panicking
//! test orphaned it, and a supervisor's workers with it. A test binary that
//! dies without unwinding (Ctrl+C, a timeout's SIGKILL, an abort) skips every
//! `Drop` as well. On unix each process therefore leads its own process group:
//! [`ProcessGuard`] kills the whole group when dropped, taking the process's
//! own children with it, and a watcher process kills every group still
//! registered once the test binary is gone, however it exited. Off unix the
//! guard kills only the direct child.

use std::io;
use std::ops::{Deref, DerefMut};
use std::process::{Child, Command, ExitStatus, Output};

/// A spawned process (and, on unix, its process group) that is killed and
/// reaped when the guard drops. Derefs to the [`Child`].
#[derive(Debug)]
pub struct ProcessGuard {
    /// `None` only after [`ProcessGuard::wait_with_output`] consumed it.
    child: Option<Child>,
}

impl ProcessGuard {
    /// Spawns `command` as the leader of a new process group.
    pub fn spawn(command: &mut Command) -> io::Result<Self> {
        #[cfg(unix)]
        std::os::unix::process::CommandExt::process_group(command, 0);
        let child = command.spawn()?;
        #[cfg(unix)]
        watcher::send("add", child.id());
        Ok(Self { child: Some(child) })
    }

    /// Kills the process group, then reaps the leader.
    pub fn kill(&mut self) -> io::Result<ExitStatus> {
        let child = self.child.as_mut().expect("child present until consumed");
        terminate(child)
    }

    /// Like [`Child::wait_with_output`]; whatever the leader left running in
    /// its group is still killed.
    pub fn wait_with_output(mut self) -> io::Result<Output> {
        let child = self.child.take().expect("child present until consumed");
        let id = child.id();
        let output = child.wait_with_output();
        #[cfg(unix)]
        {
            kill_group(id);
            watcher::send("del", id);
        }
        #[cfg(not(unix))]
        let _ = id;
        output
    }
}

impl Deref for ProcessGuard {
    type Target = Child;

    fn deref(&self) -> &Child {
        self.child.as_ref().expect("child present until consumed")
    }
}

impl DerefMut for ProcessGuard {
    fn deref_mut(&mut self) -> &mut Child {
        self.child.as_mut().expect("child present until consumed")
    }
}

impl Drop for ProcessGuard {
    fn drop(&mut self) {
        if let Some(child) = self.child.as_mut() {
            let _ = terminate(child);
        }
    }
}

fn terminate(child: &mut Child) -> io::Result<ExitStatus> {
    // Signal the group before reaping the leader: while the leader is
    // unreaped its id cannot be reused, so the signal cannot reach a
    // stranger's group.
    #[cfg(unix)]
    {
        kill_group(child.id());
        watcher::send("del", child.id());
    }
    let _ = child.kill();
    child.wait()
}

#[cfg(unix)]
fn kill_group(leader: u32) {
    let Ok(group) = libc::pid_t::try_from(leader) else {
        return;
    };
    // SAFETY: killpg only sends a signal; it dereferences no memory. An
    // already empty group fails with ESRCH, which is the goal anyway.
    unsafe {
        libc::killpg(group, libc::SIGKILL);
    }
}

/// A `sh` process that outlives an abnormal exit of the test binary just long
/// enough to kill the groups it was told about. It reads `add <group>` and
/// `del <group>` lines from a pipe whose only write end the test binary holds
/// (std opens pipes close-on-exec, so spawned processes do not inherit it), and
/// kills every group still listed when the pipe closes. Its own group and the
/// ignored INT/HUP/TERM keep a terminal's Ctrl+C from killing it before it has
/// done so. The group is named as `kill -KILL -<group>` without `--`: dash
/// (Debian and Ubuntu's `/bin/sh`) reads the word after a signal option as a
/// pid, so `--` aborts its `kill` with "Illegal number" and kills nothing.
#[cfg(unix)]
mod watcher {
    use std::io::Write;
    use std::os::unix::process::CommandExt;
    use std::process::{ChildStdin, Command, Stdio};
    use std::sync::{Mutex, OnceLock};

    const SCRIPT: &str = r#"trap '' INT HUP TERM
groups=" "
while read -r op group; do
  case "$op" in
    add) groups="$groups$group " ;;
    del) kept=" "
         for g in $groups; do [ "$g" = "$group" ] || kept="$kept$g "; done
         groups=$kept ;;
  esac
done
for g in $groups; do kill -KILL "-$g" 2>/dev/null; done
"#;

    fn pipe() -> Option<&'static Mutex<ChildStdin>> {
        static PIPE: OnceLock<Option<Mutex<ChildStdin>>> = OnceLock::new();
        PIPE.get_or_init(|| {
            // The handle is dropped unwaited: the watcher exits when the test
            // binary does, and init reaps it.
            let mut watcher = Command::new("/bin/sh")
                .arg("-c")
                .arg(SCRIPT)
                .process_group(0)
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .ok()?;
            watcher.stdin.take().map(Mutex::new)
        })
        .as_ref()
    }

    pub(super) fn send(op: &str, group: u32) {
        if let Some(pipe) = pipe() {
            if let Ok(mut pipe) = pipe.lock() {
                let _ = writeln!(pipe, "{op} {group}");
            }
        }
    }
}
