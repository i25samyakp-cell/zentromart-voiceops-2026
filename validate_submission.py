#!/usr/bin/env python3
"""Offline pre-submission checks. This script never prints secret values."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from src.config import PROJECT_ROOT, get_settings


def check(label: str, condition: bool, detail: str = "") -> bool:
    mark = "PASS" if condition else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"[{mark}] {label}{suffix}")
    return condition


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path)
    args = parser.parse_args()
    settings = get_settings(args.dataset)
    failures = 0

    try:
        settings.validate_dataset()
        check("Dataset files found", True, str(settings.dataset_dir))
    except Exception as exc:
        check("Dataset files found", False, str(exc))
        return 1

    validator = settings.dataset_dir / "validate_dataset.py"
    result = subprocess.run(
        [sys.executable, str(validator)],
        cwd=settings.dataset_dir,
        capture_output=True,
        text=True,
    )
    if not check("Dataset validator", result.returncode == 0, result.stdout.strip() or result.stderr.strip()):
        failures += 1

    test_env = dict(os.environ)
    test_env["DATASET_DIR"] = str(settings.dataset_dir)
    tests = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=PROJECT_ROOT,
        env=test_env,
        capture_output=True,
        text=True,
    )
    if not check("Core unit tests", tests.returncode == 0):
        print(tests.stdout)
        print(tests.stderr)
        failures += 1

    required = [
        "README.md",
        "requirements.txt",
        ".env.example",
        "run.py",
        "src/agent.py",
        "src/voice.py",
        "report.pdf",
    ]
    missing = [name for name in required if not (PROJECT_ROOT / name).exists()]
    if not check("Required submission files", not missing, ", ".join(missing) if missing else "complete"):
        failures += 1

    gitignore_path = PROJECT_ROOT / ".gitignore"
    ignored_lines = {
        line.strip() for line in gitignore_path.read_text(encoding="utf-8").splitlines()
    } if gitignore_path.exists() else set()
    if not check(".env excluded from Git", ".env" in ignored_lines):
        failures += 1

    key_state = (
        f"OpenAI={'set' if settings.openai_api_key else 'missing'}, "
        f"Gnani={'set' if settings.gnani_api_key else 'missing'}"
    )
    check("Runtime credentials", bool(settings.openai_api_key and settings.gnani_api_key), key_state)
    if not settings.openai_api_key or not settings.gnani_api_key:
        print("[ACTION] Add both keys to a local .env before end-to-end testing.")

    print(f"\nValidation completed with {failures} blocking failure(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

