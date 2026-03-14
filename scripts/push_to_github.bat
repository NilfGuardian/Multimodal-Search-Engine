@echo off
setlocal
cd /d "%~dp0.."

if "%~1"=="" (
  echo Usage: scripts\push_to_github.bat ^<github_repo_url^>
  echo Example: scripts\push_to_github.bat https://github.com/yourname/multimodal-search-engine.git
  exit /b 1
)

set REPO_URL=%~1

git rev-parse --is-inside-work-tree >nul 2>nul
if errorlevel 1 (
  echo ERROR: Not a git repository.
  exit /b 1
)

git remote remove origin >nul 2>nul
git remote add origin %REPO_URL%
git push -u origin main

if errorlevel 1 (
  echo Push failed. Check repo URL and authentication.
  exit /b 1
)

echo Successfully pushed to %REPO_URL%
endlocal
