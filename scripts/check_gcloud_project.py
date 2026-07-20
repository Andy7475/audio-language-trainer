"""Preflight check that gcloud is authenticated against the right GCP project.

Run this before working in this repo if you use the gcloud CLI for other
projects too, since `gcloud config set project` is machine-wide state that's
easy to leave pointed at the wrong project:

    uv run python scripts/check_gcloud_project.py
"""

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

load_dotenv(PROJECT_ROOT / ".env")

EXPECTED_PROJECT = os.environ.get("GOOGLE_PROJECT_ID")


def run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip()


def main() -> None:
    if not EXPECTED_PROJECT:
        print("(x) GOOGLE_PROJECT_ID is not set in .env - cannot verify")
        sys.exit(1)

    ok = True

    account = run(["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"])
    if account:
        print(f"(y) Active gcloud account: {account}")
    else:
        print("(x) No active gcloud account. Fix with: gcloud auth login")
        ok = False

    active_project = run(["gcloud", "config", "get-value", "project"])
    if active_project == EXPECTED_PROJECT:
        print(f"(y) gcloud CLI default project matches: {active_project}")
    else:
        print(
            f"(x) gcloud CLI default project is '{active_project}', expected "
            f"'{EXPECTED_PROJECT}'.\n"
            f"    Fix with: gcloud config set project {EXPECTED_PROJECT}"
        )
        ok = False

    adc_check = subprocess.run(
        ["gcloud", "auth", "application-default", "print-access-token"],
        capture_output=True,
        text=True,
    )
    if adc_check.returncode == 0:
        print("(y) Application Default Credentials are valid")
    else:
        print(
            "(x) Application Default Credentials missing/invalid.\n"
            "    Fix with: gcloud auth application-default login"
        )
        ok = False

    # Confirm the Python client library resolves to the right project too -
    # this is what actually matters for notebooks/scripts, independent of
    # whatever the gcloud CLI's ambient config says.
    from connections.gcloud_auth import setup_authentication

    try:
        _, resolved_project = setup_authentication()
        if resolved_project == EXPECTED_PROJECT:
            print(f"(y) Python client library resolves to: {resolved_project}")
        else:
            print(
                f"(x) Python client library resolves to '{resolved_project}', "
                f"expected '{EXPECTED_PROJECT}'"
            )
            ok = False
    except SystemExit:
        ok = False

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
