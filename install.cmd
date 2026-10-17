@echo off
rem Copyright (c), The Rusoil Software Development Team. All rights reserved.
rem Licensed under the Apache License, Version 2.0.
rem
rem Windows wrapper for install.ps1: build, deploy, migrate, then open the console.
rem
rem Passes -ExecutionPolicy Bypass for this invocation only, because a fresh Windows
rem install refuses to run unsigned .ps1 files ("is not digitally signed").

setlocal
set "SCRIPT=%~dp0install.ps1"
if not exist "%SCRIPT%" (
  echo error: %SCRIPT% not found >&2
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%" %*
exit /b %ERRORLEVEL%
