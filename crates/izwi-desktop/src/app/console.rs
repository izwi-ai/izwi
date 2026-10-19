use std::process::Command;

/// Keep helper processes from opening a console window.
///
/// The release desktop binary runs under the Windows GUI subsystem, so it has
/// no console of its own; without `CREATE_NO_WINDOW` every console child (the
/// managed `izwi-server`, PowerShell) would flash up a new terminal window.
/// Pipes and file redirection are unaffected. No-op on other platforms.
pub fn hide_console_window(cmd: &mut Command) -> &mut Command {
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}
