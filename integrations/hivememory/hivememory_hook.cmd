@echo off
rem Codex hook entry. Codex hands the hook command to the session shell (PowerShell or cmd),
rem which spell environment variables differently, so the interpreter is chosen here:
rem HIVEMEMORY_PYTHON (the Python bundled with HiveMemory, written when the plugin is
rem installed) when it is set, otherwise python on PATH. Kept ASCII: cmd reads batch files
rem in the OEM code page.
if defined HIVEMEMORY_PYTHON (
  "%HIVEMEMORY_PYTHON%" "%~dp0hivememory_hook.py" %*
) else (
  python "%~dp0hivememory_hook.py" %*
)
exit /b %ERRORLEVEL%
