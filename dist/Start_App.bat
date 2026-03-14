@echo off
setlocal
cd /d "%~dp0"
if not exist "MultiSearchLauncher.exe" (
  echo ERROR: MultiSearchLauncher.exe not found in %~dp0
  pause
  exit /b 1
)

set "LOG_FILE=%~dp0launcher_last_run.log"
echo [%date% %time%] Launching Multimodal Search Engine... > "%LOG_FILE%"

"%~dp0MultiSearchLauncher.exe" %* >> "%LOG_FILE%" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
  echo.
  echo Launcher failed with exit code %EXIT_CODE%.
  echo Showing log:
  type "%LOG_FILE%"
  echo.
  pause
)

exit /b %EXIT_CODE%
