#!/usr/bin/env python3
"""Run semantic context-bridge checks in disposable, explicit dependency modes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent.parent


def run(command, cwd, env, log):
    print(json.dumps({"command": command, "cwd": str(cwd), "GOWORK": "off"}), flush=True)
    result = subprocess.run(command, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=900, check=False)
    print(result.stdout, end="", flush=True)
    log.append({"command": command, "exit_code": result.returncode, "output": result.stdout})
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("checkout", "published"))
    parser.add_argument("--dependency-root", type=Path, default=ROOT.parent)
    parser.add_argument("--ragy-ref")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    env = {**os.environ, "GOWORK": "off"}
    record = {"mode": args.mode, "dependencies": {}, "commands": [], "status": "failed"}
    try:
        with tempfile.TemporaryDirectory(prefix="ragy-context-bridge-") as temporary:
            module = Path(temporary) / "consumer"
            shutil.copytree(ROOT / "examples/context-bridge", module)
            record["source_files"] = {path.relative_to(module).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in sorted(module.rglob("*")) if path.is_file()}
            go = os.environ.get("GO", "go")
            for name in ("ragy", "memy", "contexty"):
                run([go, "mod", "edit", f"-dropreplace=github.com/skosovsky/{name}"], module, env, record["commands"])
            if args.mode == "checkout":
                for name in ("ragy", "memy", "contexty"):
                    checkout = ROOT if name == "ragy" else args.dependency_root.resolve() / name
                    if not (checkout / "go.mod").is_file():
                        raise ValueError(f"Missing dependency checkout: {name}")
                    identity = run(["git", "rev-parse", "HEAD"], checkout, env, record["commands"]).strip()
                    dirty = run(["git", "status", "--porcelain"], checkout, env, record["commands"]).strip()
                    frozen = Path(temporary) / "dependencies" / name
                    shutil.copytree(checkout, frozen, ignore=shutil.ignore_patterns(".git", ".cursor", ".codex", ".agents", "__pycache__"))
                    fingerprints = {p.relative_to(frozen).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(frozen.rglob("*")) if p.is_file() and (p.suffix in (".go", ".json") or p.name in ("go.mod", "go.sum"))}
                    record["dependencies"][name] = {"source": identity, "dirty": dirty, "input_sha256": fingerprints}
                    run([go, "mod", "edit", f"-replace=github.com/skosovsky/{name}={frozen}"], module, env, record["commands"])
            elif args.ragy_ref:
                run([go, "mod", "edit", f"-require=github.com/skosovsky/ragy@{args.ragy_ref}"], module, env, record["commands"])
            run([go, "mod", "tidy"], module, env, record["commands"])
            manifests = json.loads(run([go, "mod", "edit", "-json"], module, env, record["commands"]))
            if args.mode == "published" and manifests.get("Replace"):
                raise ValueError("Published mode must not contain any replace directives")
            record["resolved_modules"] = run([go, "list", "-m", "all"], module, env, record["commands"])
            if args.mode == "published":
                requirements = {item["Path"]: item["Version"] for item in manifests["Require"]}
                for name in ("ragy", "memy", "contexty"):
                    path = f"github.com/skosovsky/{name}"
                    downloaded = json.loads(run([go, "mod", "download", "-json", f"{path}@{requirements[path]}"],
                                                module, env, record["commands"]))
                    record["dependencies"][name] = {key: downloaded[key] for key in
                                                   ("Path", "Version", "Sum", "GoModSum", "Origin") if key in downloaded}
            run([go, "test", "-race", "-count=1", "./..."], module, env, record["commands"])
            run([go, "run", "./cmd/demo"], module, env, record["commands"])
            record["status"] = "passed"
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
