#!/usr/bin/env python3
"""Standalone, explicit module verification; no ambient workspace or duplicate suites."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from process_runner import run_process

ROOT = Path(__file__).resolve().parent.parent


def modules(root=ROOT):
    listed = (root / "scripts/check-modules.txt").read_text().splitlines()
    if not listed or listed[0] != "." or len(set(listed)) != len(listed):
        raise ValueError("Check manifest must start with root and contain unique modules")
    found = set()
    for directory, children, files in os.walk(root):
        children[:] = sorted(name for name in children if not name.startswith(".") and name != "vendor")
        if "go.mod" in files:
            found.add(Path(directory).relative_to(root).as_posix())
    if set(listed) != found:
        raise ValueError(f"Module inventory drift: omitted={sorted(found-set(listed))}, stale={sorted(set(listed)-found)}")
    release = (root / "scripts/release-modules.txt").read_text().splitlines()
    if not set(release) <= found or any(name.startswith("examples/") for name in release):
        raise ValueError("Publishable manifest must be separate from development examples")
    return listed


def run(command, directory, timeout=600, capture=False):
    env = {**os.environ, "GOWORK": "off", "PYTHONDONTWRITEBYTECODE": "1"}
    started = time.monotonic()
    print(json.dumps({"command": command, "module": str(directory.relative_to(ROOT)) if directory.is_relative_to(ROOT) else str(directory),
                      "GOWORK": "off", "timeout_seconds": timeout}), flush=True)
    process = run_process(command, directory, env, timeout, capture=capture)
    output = process.stdout
    if capture and process.stderr:
        print(process.stderr, end="", file=sys.stderr, flush=True)
    print(json.dumps({"exit_code": process.returncode, "elapsed_seconds": round(time.monotonic()-started, 3)}), flush=True)
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)
    return output or ""


def fuzz_names(output):
    return [line for line in output.splitlines() if line.startswith("Fuzz") and not any(character.isspace() for character in line)]


def fuzz_module(directory, go, seconds, run_command=run):
    # Listing and each function execution are bounded separately; errors propagate.
    packages = run_command([go, "list", "-tags=fuzz", "./..."], directory, capture=True).splitlines()
    for package in packages:
        output = run_command([go, "test", "-tags=fuzz", "-list=^Fuzz", package], directory, capture=True)
        for name in fuzz_names(output):
            run_command([go, "test", "-tags=fuzz", "-run=^$", f"-fuzz=^{re.escape(name)}$",
                         f"-fuzztime={seconds}s", "-parallel=2", package], directory, timeout=seconds+60)


def versions(go, lint, enforce=False):
    expected = json.loads((ROOT / "scripts/toolchain.json").read_text())
    actual_go = run([go, "version"], ROOT, capture=True).strip()
    actual_lint = run([lint, "version"], ROOT, capture=True).strip()
    print(json.dumps({"go": actual_go, "golangci_lint": actual_lint, "python": sys.version, "validated": expected}), flush=True)
    if enforce and (not re.search(rf"(?:^|\s)go{re.escape(expected['go'])}(?:\s|$)", actual_go) or
                    not re.search(rf"\bversion {re.escape(expected['golangci_lint'])}(?:\s|$)", actual_lint)):
        raise ValueError("Fresh acceptance requires the recorded validated toolchain")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("modules", "versions", "test", "acceptance", "lint", "examples", "fuzz", "bench", "cover", "fix"))
    parser.add_argument("--module", action="append", default=[])
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--fuzz-seconds", type=int, default=30)
    args = parser.parse_args()
    inventory = modules()
    if args.mode == "modules":
        print(json.dumps(inventory) if args.json else "\n".join(inventory))
        return 0
    selected = args.module or inventory
    if len(set(selected)) != len(selected) or not set(selected) <= set(inventory):
        raise ValueError("Selected modules must be unique inventory entries")
    go, lint = os.environ.get("GO", "go"), os.environ.get("GOLANGCI_LINT", "golangci-lint")
    if args.mode in ("versions", "acceptance"):
        versions(go, lint, enforce=args.mode == "acceptance")
    if args.mode == "versions":
        return 0
    if not 1 <= args.fuzz_seconds <= 300:
        raise ValueError("Fuzz budget must be between 1 and 300 seconds per function")
    for module in selected:
        directory = ROOT / module
        if args.mode in ("lint", "acceptance"):
            run([lint, "run", "--allow-serial-runners", "./..."], directory)
        if args.mode in ("test", "acceptance"):
            fresh = ["-count=1"] if args.fresh or args.mode == "acceptance" else []
            run([go, "test", *fresh, "-race", "./..."], directory)
        if args.mode in ("examples", "acceptance"):
            if module.startswith("examples/"):
                run([go, "build", "./..."], directory)
            elif module == ".":
                run([go, "build", "-o", os.devnull, "./examples/local-bm25"], directory)
        if args.mode == "acceptance" and module == ".":
            run([sys.executable, "scripts/verify_test.py", "-v"], ROOT)
            run([sys.executable, "scripts/check_release_consumer_test.py", "-v"], ROOT)
            run([sys.executable, "scripts/process_runner_test.py", "-v"], ROOT)
        if args.mode == "fuzz":
            fuzz_module(directory, go, args.fuzz_seconds)
        if args.mode == "fix":
            run([go, "fix", "./..."], directory)
            run([go, "mod", "tidy"], directory)
            run([lint, "run", "--fix", "./..."], directory)
        if args.mode == "bench":
            run([go, "test", "-run=^$", "-bench=.", "./..."], directory)
        if args.mode == "cover":
            run([go, "test", "-count=1", "-coverprofile=coverage.out", "./..."], directory)
            run([go, "tool", "cover", "-func=coverage.out"], directory)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"Verification failed: {error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
