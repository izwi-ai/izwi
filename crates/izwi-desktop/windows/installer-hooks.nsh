; Stop Izwi's server before the installer replaces or removes its files.
; Windows can't overwrite bin\izwi-server.exe while it runs, and a server can
; outlive the app: for a few seconds after an update closes it, or for good
; if the app was killed. A server whose parent is the running app is left
; alone: the installer closes the app next, and stopping the server first
; would only make the app start another one.
!macro IZWI_STOP_SERVER
  ; Passed through the environment so no path can break the quoting.
  System::Call 'Kernel32::SetEnvironmentVariable(t "IZWI_INSTALL_DIR", t "$INSTDIR") i'
  System::Call 'Kernel32::SetEnvironmentVariable(t "IZWI_APP_EXE", t "${MAINBINARYNAME}.exe") i'
  nsExec::Exec `powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "$$dir = $$env:IZWI_INSTALL_DIR.TrimEnd('\') + '\'; $$app = $$dir + $$env:IZWI_APP_EXE; Get-CimInstance Win32_Process | Where-Object { $$_.Name -eq 'izwi-server.exe' -and $$_.ExecutablePath -and $$_.ExecutablePath.StartsWith($$dir, [StringComparison]::OrdinalIgnoreCase) } | Where-Object { $$parent = Get-Process -Id $$_.ParentProcessId -ErrorAction SilentlyContinue; -not ($$parent -and $$parent.Path -eq $$app) } | ForEach-Object { Wait-Process -Id $$_.ProcessId -Timeout 5 -ErrorAction SilentlyContinue; Stop-Process -Id $$_.ProcessId -Force -ErrorAction SilentlyContinue }"`
  Pop $0
!macroend

!macro NSIS_HOOK_PREINSTALL
  !insertmacro IZWI_STOP_SERVER
!macroend

!macro NSIS_HOOK_PREUNINSTALL
  !insertmacro IZWI_STOP_SERVER
!macroend
