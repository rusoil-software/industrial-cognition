@echo off
rem Copyright (c), The Rusoil Software Development Team. All rights reserved.
rem Licensed under the Apache License, Version 2.0.
rem
rem Windows wrapper for scripts\console.ps1.
rem
rem It exists because a fresh Windows install defaults to a Restricted execution policy,
rem which refuses to run an unsigned .ps1 with "cannot be loaded ... is not digitally
rem signed". A .cmd can be run from any prompt (or double-clicked) and passes the bypass
rem for this invocation only - no machine-wide policy is changed.

setlocal
set "SCRIPT=%~dp0scripts\console.ps1"
if not exist "%SCRIPT%" (
  echo error: %SCRIPT% not found >&2
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%" %*
exit /b %ERRORLEVEL%
