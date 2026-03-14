"""Desktop launcher for Multimodal Search Engine.

Build with PyInstaller:
    pyinstaller --onefile --name MultiSearchLauncher scripts/app_launcher.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.request import urlopen


BACKEND_URL = "http://127.0.0.1:8000/stats"
FRONTEND_URL = "http://127.0.0.1:8501"


def _project_root() -> Path:
    if getattr(sys, "frozen", False):
        # Exe is expected at <project>/dist/MultiSearchLauncher.exe.
        return Path(sys.executable).resolve().parent.parent
    return Path(__file__).resolve().parents[1]


def _python_executable(root: Path) -> str:
    venv_python = root / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)

    fallback = os.environ.get("PYTHON") or "python"
    return fallback


def _wait_http(url: str, timeout_seconds: int = 90) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=4) as resp:  # noqa: S310
                if 200 <= int(resp.status) < 500:
                    return True
        except URLError:
            pass
        except Exception:
            pass
        time.sleep(0.6)
    return False


def _free_port_windows(port: int) -> None:
    """Kill process listening on a port (Windows-only best effort)."""
    if os.name != "nt":
        return

    try:
        result = subprocess.run(
            ["netstat", "-ano", "-p", "tcp"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return

        target = f":{port}"
        pids: set[str] = set()
        for line in result.stdout.splitlines():
            line = line.strip()
            if "LISTENING" not in line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            local_addr = parts[1]
            pid = parts[-1]
            if target in local_addr:
                pids.add(pid)

        for pid in pids:
            subprocess.run(["taskkill", "/PID", pid, "/F"], capture_output=True, text=True, check=False)
    except Exception:
        # Best-effort cleanup; startup will still fail clearly if port cannot be used.
        pass


def _terminate(proc: Optional[subprocess.Popen[bytes]]) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch Multimodal Search Engine desktop app")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser")
    parser.add_argument("--auto-stop", type=int, default=0, help="Auto stop after N seconds")
    args = parser.parse_args()

    root = _project_root()
    python_exe = _python_executable(root)

    print("Multimodal Search Engine Launcher")
    print(f"Project root: {root}")
    print(f"Python runtime: {python_exe}")

    env = os.environ.copy()
    env["CLIP_DOWNLOAD_ALLOWED"] = "1"
    env.pop("FORCE_FAKE_EMBEDDINGS", None)

    _free_port_windows(8000)
    _free_port_windows(8501)

    backend: Optional[subprocess.Popen[bytes]] = None
    frontend: Optional[subprocess.Popen[bytes]] = None

    try:
        backend = subprocess.Popen(
            [python_exe, "-m", "uvicorn", "src.api:app", "--host", "127.0.0.1", "--port", "8000"],
            cwd=str(root),
            env=env,
        )
        print("Starting backend...")

        if not _wait_http(BACKEND_URL, timeout_seconds=120):
            print("Backend did not become ready in time.")
            return 1

        frontend = subprocess.Popen(
            [python_exe, "-m", "streamlit", "run", "src/ui.py", "--server.port", "8501"],
            cwd=str(root),
            env=env,
        )
        print("Starting frontend...")

        if not _wait_http(FRONTEND_URL, timeout_seconds=60):
            print("Frontend did not become ready in time.")
            return 1

        print("Application is running.")
        print(f"Backend: {BACKEND_URL}")
        print(f"Frontend: {FRONTEND_URL}")

        if not args.no_browser:
            webbrowser.open(FRONTEND_URL)

        if args.auto_stop > 0:
            print(f"Auto-stop enabled: {args.auto_stop}s")
            time.sleep(args.auto_stop)
        else:
            input("Press Enter to stop application...\n")

        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        _terminate(frontend)
        _terminate(backend)
        print("Application stopped.")


if __name__ == "__main__":
    raise SystemExit(main())
