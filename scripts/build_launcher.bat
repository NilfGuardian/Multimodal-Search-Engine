@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: .venv\Scripts\python.exe not found.
  exit /b 1
)
.venv\Scripts\python.exe -m pip install pyinstaller
.venv\Scripts\python.exe -m PyInstaller --onefile --name MultiSearchLauncher scripts\app_launcher.py
if errorlevel 1 exit /b 1
echo.
echo Built: dist\MultiSearchLauncher.exe
endlocal
