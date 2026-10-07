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
    registry = json.loads((root / "scripts/check-registry.json").read_text())
    listed = (root / registry["inventories"]["test"]).read_text().splitlines()
    if not listed or listed[0] != "." or len(set(listed)) != len(listed):
        raise ValueError("Check manifest must start with root and contain unique modules")
    found = set()
    for directory, children, files in os.walk(root):
        children[:] = sorted(name for name in children if not name.startswith(".") and name != "vendor")
        if "go.mod" in files:
            found.add(Path(directory).relative_to(root).as_posix())
    if set(listed) != found:
        raise ValueError(f"Module inventory drift: omitted={sorted(found-set(listed))}, stale={sorted(set(listed)-found)}")
    release = (root / registry["inventories"]["release"]).read_text().splitlines()
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
        raise subprocess.CalledProcessError(process.returncode, command, output=output, stderr=process.stderr)
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


def versions(go, lint, enforce=False, run_command=None):
    run_command = run_command or run
    registry = json.loads((ROOT / "scripts/check-registry.json").read_text())
    expected = json.loads((ROOT / registry["toolchain"]).read_text())
    actual_go = run_command([go, "version"], ROOT, capture=True).strip()
    actual_lint = run_command([lint, "version"], ROOT, capture=True).strip()
    print(json.dumps({"go": actual_go, "golangci_lint": actual_lint, "python": sys.version, "validated": expected}), flush=True)
    if enforce and (not re.search(rf"(?:^|\s)go{re.escape(expected['go'])}(?:\s|$)", actual_go) or
                    not re.search(rf"\bversion {re.escape(expected['golangci_lint'])}(?:\s|$)", actual_lint)):
        raise ValueError("Fresh acceptance requires the recorded validated toolchain")


def registry_plan(profile, inventory):
    registry = json.loads((ROOT / "scripts/check-registry.json").read_text())
    plan = []
    for lane in registry["lanes"]:
        if profile not in lane["profiles"]:
            continue
        for module in inventory if lane.get("modules") else ["."]:
            if lane.get("module_filter") == "examples" and module != "." and not module.startswith("examples/"):
                continue
            row = dict(lane, module=lane.get("module", module), id=lane["id"].replace("{module}", module))
            plan.append(row)
    if len({row["id"] for row in plan}) != len(plan):
        raise ValueError("Duplicate registry lane IDs")
    known = {row["id"] for row in plan}
    for row in plan:
        if not set(row.get("needs", [])) <= known:
            raise ValueError("Unknown registry dependency: " + row["id"])
    ordered = []
    pending = list(plan)
    while pending:
        ready = [row for row in pending if set(row.get("needs", [])) <= {r["id"] for r in ordered}]
        if not ready:
            raise ValueError("Registry dependency cycle")
        ordered.extend(ready)
        pending = [row for row in pending if row not in ready]
    return ordered


def execute_plan(plan, execute):
    """Independent failures accumulate; dependent lanes cannot claim execution."""
    results = []
    for row in plan:
        started = time.monotonic()
        result = {"id": row["id"], "module": row.get("module", "."), "required": not row.get("optional", False), "command": row.get("command", ["registry-action:" + row.get("action", "unclassified")])}
        failed_needs = [name for name in row.get("needs", [])
                        if not any(r["id"] == name and r["status"] == "PASS" for r in results)]
        try:
            if row.get("optional"):
                result.update(status="SKIP", reason=row["reason"])
            elif failed_needs:
                result.update(status="BLOCKED", reason="Dependencies did not pass: " + ", ".join(failed_needs))
            else:
                evidence = execute(row) or {}
                result.update(evidence, status="PASS")
        except FileNotFoundError as error:
            result.update(status="BLOCKED", reason=str(error))
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            result.update(status="FAIL", reason=str(error))
        result["commands"] = row.get("_commands", result.get("commands", []))
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        results.append(result)
    success = all(r["status"] == "PASS" for r in results if r["required"])
    return results, success


def check_profile(args):
    import hashlib
    import platform
    import shutil
    import tempfile
    inventory = modules()
    profile = "check" if args.mode == "acceptance" else args.mode
    plan = registry_plan(profile, inventory)
    if args.mode == "plan":
        plan = registry_plan("check", inventory)
    if args.mode == "plan" or args.list:
        print(json.dumps(plan, indent=2))
        return 0
    output = Path(args.output or tempfile.mkdtemp(prefix="ragy-check-")).resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("Check artifacts must be outside the repository to preserve candidate bytes")
    output.mkdir(parents=True, exist_ok=True)
    registry = json.loads((ROOT / "scripts/check-registry.json").read_text())
    pins = json.loads((ROOT / registry["toolchain"]).read_text())
    go = os.environ.get("GO", "go")
    # The validated compiler is selected explicitly, even on a fresh checkout.
    os.environ["GOTOOLCHAIN"] = "go" + pins["go"]
    os.environ["GOWORK"] = "off"
    os.environ["GOENV"] = "off"
    os.environ["GOFLAGS"] = ""
    for key in ("GOPRIVATE", "GONOPROXY", "GONOSUMDB"):
        os.environ[key] = ""
    for key in ("RAGY_LIVE_GEMINI", "RAGY_LIVE_PROVIDERS", "RAGY_PROVIDER_SMOKE"):
        os.environ.pop(key, None)
    source = run(["git", "rev-parse", "HEAD"], ROOT, capture=True).strip()
    if args.source and args.source != source:
        raise ValueError("Selected source does not match checkout HEAD")
    version = args.version or os.environ.get("RAGY_CANDIDATE_VERSION", "")
    if not version:
        version = run(["git", "describe", "--tags", "--match=v*", "--abbrev=0"], ROOT, capture=True).strip()
    lint = os.environ.get("GOLANGCI_LINT")
    commands = []
    active_row = None
    def command(argv, directory=ROOT, **kwargs):
        entry = {"command": argv, "cwd": str(directory)}
        commands.append(entry)
        if active_row is not None:
            active_row.setdefault("_commands", []).append(entry)
        kwargs.setdefault("capture", True)
        log = output / ("command-" + str(len(commands)) + ".log")
        entry["log"] = str(log)
        try:
            text = run(argv, directory, **kwargs)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            stdout = error.output or ""
            stderr = error.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            log.write_text(stdout + stderr)
            raise
        log.write_text(text)
        return text
    def execute(row):
        nonlocal lint, active_row
        active_row = row
        begin = len(commands)
        action = row.get("action")
        directory = ROOT / row["module"]
        if action == "source-integrity":
            if command(["git", "diff", "HEAD", "--name-only"], capture=True).strip():
                raise ValueError("Full check requires committed tracked bytes; use test-fast for working edits")
            untracked = command(["git", "ls-files", "--others", "--exclude-standard"], capture=True).splitlines()
            inputs = [name for name in untracked if not (name.startswith("docs/") and name.endswith(".md"))]
            if inputs:
                raise ValueError("Untracked inputs would differ from committed consumers: " + ", ".join(inputs))
            if args.candidate:
                from check_release_consumer import proxy_module
                files = command(["git", "ls-tree", "-r", "--name-only", "HEAD"], capture=True).splitlines()
                publishable = (ROOT / registry["inventories"]["release"]).read_text().splitlines()
                proxy = Path(tempfile.mkdtemp(prefix="candidate-proxy-", dir=output))
                paths = [proxy_module(ROOT, module, version, files, proxy, dict(os.environ)) for module in publishable]
                os.environ.update(GOPROXY=proxy.as_uri()+",https://proxy.golang.org", GOSUMDB="sum.golang.org",
                                  GOMODCACHE=tempfile.mkdtemp(prefix="candidate-modcache-", dir=output), GONOPROXY="none",
                                  GONOSUMDB=paths[0]+","+paths[0]+"/*")
                return {"candidate_proxy": str(proxy), "modules": paths, "commands": commands[begin:]}
        elif action == "toolchain":
            if not lint:
                cache = Path(tempfile.gettempdir()) / ("ragy-lint-v" + pins["golangci_lint"] + "-" + platform.machine())
                cache.mkdir(exist_ok=True)
                lint = str(cache / "golangci-lint")
                if not Path(lint).is_file():
                    previous = os.environ.get("GOBIN")
                    os.environ["GOBIN"] = str(cache)
                    try:
                        command([go, "install", "github.com/golangci/golangci-lint/v2/cmd/golangci-lint@v" + pins["golangci_lint"]], timeout=900)
                    finally:
                        if previous is None:
                            os.environ.pop("GOBIN", None)
                        else:
                            os.environ["GOBIN"] = previous
            versions(go, lint, enforce=True, run_command=command)
        elif action == "baseline":
            if (ROOT / ".golangci.yml").read_bytes() != (ROOT / "scripts/lint-baseline.yml").read_bytes():
                raise ValueError("Linter baseline drift: update reviewed versioned baseline with justified exclusions")
        elif action == "example-build":
            if row["module"] == ".":
                command([go, "build", "-o", os.devnull, "./examples/local-bm25"])
            else:
                command([go, "build", "./..."], directory)
        elif action == "generation":
            # No directives is a verified inventory result, not a silently skipped generation job.
            result = run_process(["git", "grep", "-n", "^//go:generate", "--", "*.go"], ROOT, dict(os.environ), 30, capture=True)
            if result.returncode not in (0, 1):
                raise ValueError("Cannot establish generation inventory")
            generated = result.stdout or ""
            if generated:
                with tempfile.TemporaryDirectory(prefix="ragy-generate-") as temporary:
                    frozen = Path(temporary) / "source"
                    shutil.copytree(ROOT, frozen, ignore=shutil.ignore_patterns(".git", "__pycache__"))
                    before = {str(p.relative_to(frozen)): hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen.rglob("*") if p.is_file()}
                    for module in inventory:
                        command([go, "generate", "./..."], frozen / module)
                    after = {str(p.relative_to(frozen)): hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen.rglob("*") if p.is_file()}
                    if before != after:
                        raise ValueError("Generated outputs differ from committed inputs")
            return {"directives": generated.splitlines(), "commands": commands[begin:]}
        elif action == "pdf":
            python = os.environ.get("RAGY_PDF_PYTHON")
            if not python:
                raise FileNotFoundError("RAGY_PDF_PYTHON is required for actual parser integration")
            versions_code = "import pdfplumber,pypdf,json; print(json.dumps({'pdfplumber':pdfplumber.__version__,'pypdf':pypdf.__version__}))"
            actual = json.loads(command([python, "-c", versions_code], capture=True))
            if actual != registry["pdf"]:
                raise ValueError("PDF runtime drift: " + str(actual))
            events = command([go, "test", "-json", "-count=1", "-race", "./..."], ROOT / "adapters/pdf", capture=True)
            validate_test_events(events, reject_skips=True, require_tests=True)
        elif action == "linux":
            if platform.system() == "Linux":
                return {"platform": platform.platform(), "reason": "All registry lanes execute on Linux in this profile"}
            if not shutil.which("docker"):
                raise FileNotFoundError("Docker is required to reproduce the mandatory Linux profile")
            argv = [sys.executable, "scripts/check_linux.py", profile, "--version", version, "--source", source, "--output", str(output / "linux")]
            if args.candidate:
                argv.append("--candidate")
            command(argv, timeout=7200)
        else:
            if row["id"] == "postgres" and not os.environ.get("RAGY_PG_TEST_CONTAINER"):
                raise FileNotFoundError("RAGY_PG_TEST_CONTAINER is required; isolated Docker pgvector prerequisite")
            values = dict(go=go, lint=lint, python=sys.executable, source=source, version=version, output=str(output))
            argv = [part.format(**values) for part in row["command"]]
            if row["id"] == "release-consumer" and not version:
                raise ValueError("Consumer requires explicit candidate version")
            if row["id"] == "release-consumer" and args.candidate:
                argv.append("--exact-candidate")
            capture = row.get("test_events") or row.get("no_output")
            result = command(argv, directory, capture=capture)
            if row.get("no_output") and result.strip():
                raise ValueError("Format drift: " + result[:4000])
            if row.get("test_events"):
                skips = validate_test_events(result, reject_skips=row.get("reject_skips", False), require_tests=row["id"] == "postgres")
                return {"commands": commands[begin:], "test_skips": skips}
        return {"commands": commands[begin:]}
    tracked = run(["git", "ls-files", "-z"], ROOT, capture=True).split("\0")
    def fingerprint():
        return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in tracked if name and (ROOT / name).is_file()}
    before = fingerprint()
    results, success = execute_plan(plan, execute)
    if fingerprint() != before:
        success = False
        results.append({"id": "source-stability", "required": True, "status": "FAIL", "elapsed_seconds": 0, "reason": "Tracked bytes changed during check"})
    report = {"schema": "ragy.check-report/v1", "profile": profile, "source": source, "candidate": args.candidate,
              "version": version, "platform": platform.platform(), "toolchain": pins, "peer_refs": {name: peer["ref"] for name, peer in registry["peers"].items()},
              "modules": inventory, "results": results, "status": "PASS" if success else "FAIL"}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print("\nLANE                         STATUS    SECONDS   REASON")
    for result in results:
        print(f"{result['id']:<28} {result['status']:<9} {result['elapsed_seconds']:>8}   {result.get('reason', '')}")
    print("Full profile " + report["status"] + ": " + str(output / "summary.json"))
    return 0 if success else 1


def validate_test_events(output, reject_skips=False, require_tests=False):
    events = []
    for line in output.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    if not any(e.get("Action") in ("pass", "fail") and not e.get("Test") for e in events):
        raise ValueError("Missing Go package terminal events")
    if any(e.get("Action") in ("fail", "build-fail") for e in events):
        raise ValueError("Go test failure in recorded events")
    skips = [e for e in events if e.get("Action") == "skip"]
    if reject_skips and skips:
        raise ValueError("Required integration test skipped: " + json.dumps(skips))
    if require_tests and not any(e.get("Action") == "pass" and e.get("Test") for e in events):
        raise ValueError("Required integration executed no tests")
    return skips

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "check", "test-fast", "modules", "versions", "test", "acceptance", "lint", "examples", "fuzz", "bench", "cover", "fix"))
    parser.add_argument("--module", action="append", default=[])
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--fuzz-seconds", type=int, default=30)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--source")
    parser.add_argument("--version")
    parser.add_argument("--candidate", action="store_true")
    args = parser.parse_args()
    if args.mode in ("plan", "check", "test", "acceptance", "lint"):
        if args.module:
            raise ValueError("Full profiles cannot select partial modules; use test-fast")
        return check_profile(args)
    fast = args.mode == "test-fast"
    if fast:
        args.mode = "test"
        print(json.dumps({"profile": "test-fast", "status": "PARTIAL", "full_acceptance": False,
                          "reason": "Shortened developer cycle; required integration, consumer, script, lint and Linux lanes excluded"}))
        for lane in registry_plan("check", modules()):
            if not lane["id"].startswith("tests:"):
                print(json.dumps({"id": lane["id"], "status": "SKIP", "reason": "Not selected by shortened test-fast profile"}))
    inventory = modules()
    if args.mode == "modules":
        print(json.dumps(inventory) if args.json else "\n".join(inventory))
        return 0
    selected = args.module or inventory
    if len(set(selected)) != len(selected) or not set(selected) <= set(inventory):
        raise ValueError("Selected modules must be unique inventory entries")
    go, lint = os.environ.get("GO", "go"), os.environ.get("GOLANGCI_LINT", "golangci-lint")
    if args.mode == "versions":
        versions(go, lint)
    if args.mode == "versions":
        return 0
    if not 1 <= args.fuzz_seconds <= 300:
        raise ValueError("Fuzz budget must be between 1 and 300 seconds per function")
    for module in selected:
        directory = ROOT / module
        if args.mode == "test":
            fresh = ["-count=1"] if args.fresh else []
            run([go, "test", *fresh, "-race", "./..."], directory)
        if args.mode == "examples":
            if module.startswith("examples/"):
                run([go, "build", "./..."], directory)
            elif module == ".":
                run([go, "build", "-o", os.devnull, "./examples/local-bm25"], directory)
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
