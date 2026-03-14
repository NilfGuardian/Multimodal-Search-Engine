@echo off
setlocal
cd /d "%~dp0"
if not exist "MultiSearchLauncher.exe" (
  echo ERROR: MultiSearchLauncher.exe not found in %~dp0
  pause
  exit /b 1
)
start "" "%~dp0MultiSearchLauncher.exe"
exit /b 0
