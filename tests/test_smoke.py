"""Basic smoke checks for project modules.

Run with: python tests/test_smoke.py
"""

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    required = [
        root / "src" / "embedding.py",
        root / "src" / "milvus_client.py",
        root / "src" / "api.py",
        root / "src" / "ui.py",
        root / "requirements.txt",
        root / "README.md",
    ]

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")

    print("Smoke check passed: required files exist.")


if __name__ == "__main__":
    main()
