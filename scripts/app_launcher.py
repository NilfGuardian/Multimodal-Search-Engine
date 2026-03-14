"""Desktop launcher for Multimodal Search Engine.

Build with PyInstaller:
    pyinstaller --onefile --name MultiSearchLauncher scripts/app_launcher.py
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.request import Request, urlopen


BACKEND_URL = "http://127.0.0.1:8000/stats"
FRONTEND_URL = "http://127.0.0.1:8501"
INDEX_LOCAL_URL = "http://127.0.0.1:8000/index/local"


def _project_root() -> Path:
    if getattr(sys, "frozen", False):
        # Exe is expected at <project>/dist/MultiSearchLauncher.exe.
        return Path(sys.executable).resolve().parent.parent
    return Path(__file__).resolve().parents[1]


def _python_executable(root: Path) -> str:
    venv_python = root / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)

    runtime_venv_python = root / ".runtime_venv" / "Scripts" / "python.exe"
    if runtime_venv_python.exists():
        return str(runtime_venv_python)

    fallback = os.environ.get("PYTHON")
    if fallback:
        return fallback

    return ""


def _find_system_python() -> str:
    candidates = [
        ["py", "-3", "-c", "import sys; print(sys.executable)"],
        ["python", "-c", "import sys; print(sys.executable)"],
    ]
    for cmd in candidates:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                exe = result.stdout.strip().splitlines()[-1].strip()
                if exe:
                    return exe
        except Exception:
            pass
    return ""


def _requirements_hash(req_path: Path) -> str:
    data = req_path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def _ensure_runtime_python(root: Path) -> str:
    """Return a usable python runtime, bootstrapping .runtime_venv when needed."""
    detected = _python_executable(root)
    if detected:
        return detected

    system_python = _find_system_python()
    if not system_python:
        raise RuntimeError("Python not found. Install Python 3.10+ and re-run launcher.")

    runtime_venv = root / ".runtime_venv"
    runtime_python = runtime_venv / "Scripts" / "python.exe"
    if not runtime_python.exists():
        print("First run setup: creating runtime virtual environment...")
        result = subprocess.run([system_python, "-m", "venv", str(runtime_venv)], cwd=str(root), check=False)
        if result.returncode != 0 or not runtime_python.exists():
            raise RuntimeError("Failed to create runtime virtual environment")

    return str(runtime_python)


def _ensure_dependencies(root: Path, python_exe: str) -> None:
    """Install requirements on first run or when requirements file changes."""
    req_path = root / "requirements.txt"
    marker_dir = root / ".runtime_venv"
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker_file = marker_dir / ".deps_hash"
    wanted = _requirements_hash(req_path)
    current = marker_file.read_text(encoding="utf-8").strip() if marker_file.exists() else ""

    if current == wanted:
        return

    print("First run setup: installing dependencies (this may take several minutes)...")
    subprocess.run([python_exe, "-m", "pip", "install", "--upgrade", "pip"], cwd=str(root), check=False)
    install = subprocess.run([python_exe, "-m", "pip", "install", "-r", "requirements.txt"], cwd=str(root), check=False)
    if install.returncode != 0:
        raise RuntimeError("Dependency installation failed")

    marker_file.write_text(wanted, encoding="utf-8")


def _ensure_seed_dataset(root: Path, python_exe: str) -> None:
    """Create starter topic-labeled dataset when images folder is nearly empty."""
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    count = len([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if count >= 120:
        return

    print("First run setup: generating starter image dataset...")
    generate = subprocess.run([python_exe, "scripts/generate_topic_seed_dataset.py"], cwd=str(root), check=False)
    if generate.returncode != 0:
        raise RuntimeError("Failed to generate starter dataset")


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


def _post_index_local(timeout_seconds: int = 300) -> None:
    """Trigger backend local indexing endpoint."""
    try:
        req = Request(INDEX_LOCAL_URL, method="POST")  # noqa: S310
        with urlopen(req, timeout=timeout_seconds):
            return
    except Exception:
        # Search endpoints also auto-index on demand; this is best effort.
        return


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

    try:
        python_exe = _ensure_runtime_python(root)
        _ensure_dependencies(root, python_exe)
        _ensure_seed_dataset(root, python_exe)
    except Exception as exc:  # noqa: BLE001
        print(f"Setup failed: {exc}")
        input("Press Enter to exit...\n")
        return 1

    print("Multimodal Search Engine Launcher")
    print(f"Project root: {root}")
    print(f"Python runtime: {python_exe}")

    env = os.environ.copy()
    env["CLIP_DOWNLOAD_ALLOWED"] = "0"
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

        _post_index_local(timeout_seconds=1800)

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
