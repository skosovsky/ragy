#!/usr/bin/env python3
"""Build all publishable modules from an actual isolated reviewed release candidate."""
import argparse
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import re
import subprocess
import shutil
import tempfile
import zipfile

from process_runner import run_process

ROOT = Path(__file__).resolve().parent.parent
PUBLIC_PACKAGE_TEMPLATE = '{{if and (ne .Name "main") (or .GoFiles .CgoFiles)}}{{.ImportPath}}{{end}}'


def command(directory, *argv, env=None, input_text=None, timeout=300):
    print(json.dumps({"command": list(argv), "cwd": str(directory), "timeout_seconds": timeout}), flush=True)
    result = run_process(argv, directory, env, timeout, capture=True, input_text=input_text)
    print(result.stdout, end="", flush=True)
    print(result.stderr, end="", flush=True)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, argv, result.stdout+result.stderr)
    return result.stdout.strip()


def proxy_module(candidate, module, version, files, proxy, env, revision="HEAD", overrides=None):
    modfile = "go.mod" if module == "." else module + "/go.mod"
    overrides = overrides or {}
    manifest = overrides.get(modfile) or subprocess.check_output(["git", "show", revision+":"+modfile], cwd=candidate, env=env)
    path = re.search(rb"^module\s+(\S+)", manifest, re.MULTILINE).group(1).decode()
    if any(character.isupper() for character in path):
        raise ValueError("This repository profile expects lowercase module paths")
    base = proxy / path / "@v"
    base.mkdir(parents=True)
    (base / (version+".mod")).write_bytes(manifest)
    (base / (version+".info")).write_text(json.dumps({"Version": version, "Time": datetime.now(timezone.utc).isoformat()}))
    (base / "list").write_text(version+"\n")
    prefix = "" if module == "." else module+"/"
    nested = [name[:-len("go.mod")] for name in files if name.endswith("/go.mod") and name.startswith(prefix) and name != modfile]
    with zipfile.ZipFile(base / (version+".zip"), "w", zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            if not name.startswith(prefix) or any(name.startswith(child) for child in nested):
                continue
            relative = name[len(prefix):]
            if "/vendor/" in "/"+relative or relative.startswith("vendor/"):
                continue
            payload = overrides.get(name) or subprocess.check_output(["git", "show", revision+":"+name], cwd=candidate, env=env)
            archive.writestr(path+"@"+version+"/"+relative, payload)
    return path


def verify(source, version=None, published=False, exact_candidate=False):
    version = version or os.environ.get("RAGY_CANDIDATE_VERSION", "v0.0.1")
    if not re.fullmatch(r"v[01]\.[0-9]+\.[0-9]+", version):
        raise ValueError("Exact patch version required")
    if not re.fullmatch(r"[0-9a-f]{40}", source):
        raise ValueError("Reviewed source must be an exact full commit SHA")
    resolved = command(ROOT, "git", "rev-parse", "--verify", source+"^{commit}")
    if resolved != source:
        raise ValueError("Source is not an exact commit object")
    pins = json.loads(subprocess.check_output(["git", "show", source+":scripts/toolchain.json"], cwd=ROOT, text=True))
    # Only readonly original Git commands; every mutation/publication is disposable.
    with tempfile.TemporaryDirectory(prefix="ragy-clean-consumer-") as temporary:
        root = Path(temporary)
        repo = root / "reviewed"
        repo.mkdir()
        env = {**os.environ, "GOWORK": "off", "PYTHONDONTWRITEBYTECODE": "1", "GOENV": "off", "GOFLAGS": "", "GOTOOLCHAIN": "go"+pins["go"]}
        for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR", "GIT_OBJECT_DIRECTORY",
                    "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_NAMESPACE", "GIT_CONFIG"):
            env.pop(key, None)
        go = os.environ.get("GO", "go")
        executable = shutil.which(go)
        if not executable:
            raise ValueError("Selected Go executable is unavailable")
        env["PATH"] = str(Path(executable).resolve().parent)+os.pathsep+env.get("PATH", "")
        command(repo, go, "version", env=env)
        command(repo, "git", "init", "--quiet", env=env)
        command(repo, "git", "fetch", "--quiet", "--no-tags", str(ROOT), source, env=env)
        command(repo, "git", "checkout", "--quiet", "--detach", source, env=env)
        for key, value in (("user.name", "Clean consumer fixture"), ("user.email", "fixture@example.invalid"),
                           ("commit.gpgsign", "false"), ("tag.gpgsign", "false")):
            command(repo, "git", "config", key, value, env=env)
        # Artifact preparation never invokes the release entrypoint: a check cannot publish.
        import release
        modules, manifests, module_root = release.release_scope(repo, source)
        release.rewrite_manifests(repo, manifests, module_root, version)
        if exact_candidate and command(repo, "git", "diff", "--name-only", env=env):
            raise ValueError("Immutable candidate manifests differ from intended version")
        command(repo, "git", "add", "--", *manifests, env=env)
        if command(repo, "git", "diff", "--cached", "--name-only", env=env):
            command(repo, "git", "commit", "--quiet", "-m", "isolated artifact manifests", env=env)
        candidate = repo
        record = {"source": source, "candidate": command(repo, "git", "rev-parse", "HEAD", env=env),
                  "version": version, "modules": modules, "files": manifests, "root_module": module_root}
        expected = {("refs/tags/" + (version if item == "." else item + "/" + version)) + " " + record["candidate"] for item in modules}
        files = command(candidate, "git", "ls-tree", "-r", "--name-only", "HEAD", env=env).splitlines()
        changed = command(candidate, "git", "diff", "--name-only", source, "HEAD", env=env).splitlines()
        if not set(changed) <= set(record["files"]):
            raise ValueError("Candidate changed non-manifest source")
        proxy = root / "proxy"
        paths = [proxy_module(candidate, module, record["version"], files, proxy, env) for module in record["modules"]]
        consumer = root / "consumer"
        consumer.mkdir()
        env.update(GOPROXY="https://proxy.golang.org" if published else proxy.as_uri()+",https://proxy.golang.org",
                   GOMODCACHE=str(root / "modcache"), GOPATH=str(root / "gopath"), GOPRIVATE="",
                   GONOSUMDB="" if published else record["root_module"]+","+record["root_module"]+"/*",
                   GONOPROXY="none", GOSUMDB="sum.golang.org")
        command(consumer, go, "mod", "init", "example.invalid/ragy-clean-consumer", env=env)
        for path in paths:
            command(consumer, go, "mod", "edit", "-require="+path+"@"+record["version"], env=env)
        # Let the Go tool select actual buildable non-main packages, not archived
        # .go probes or platform/build-tag files guessed from filenames.
        packages = set()
        for module, path in zip(record["modules"], paths):
            listing = command(consumer, go, "list", "-mod=mod", "-f", PUBLIC_PACKAGE_TEMPLATE, path+"/...", env=env)
            selected = [package for package in listing.splitlines() if package and "internal" not in package.split("/")]
            if not selected or any(not (package == path or package.startswith(path+"/")) for package in selected):
                raise ValueError("Invalid public package inventory for "+path)
            packages.update(selected)
        packages = sorted(packages)
        # Compile every actual adapter, and execute the exact canonical onboarding.
        main_source = subprocess.check_output(["git", "show", "HEAD:examples/local-bm25/main.go"], cwd=candidate, env=env).decode()
        existing_imports = set(re.findall(r'"(github.com/skosovsky/ragy[^" ]*)"', main_source))
        imports = "\n".join('\t_ "'+package+'"' for package in packages if package not in existing_imports)+"\n"
        (consumer / "main.go").write_text(main_source.replace("import (\n", "import (\n"+imports, 1))
        (consumer / "consumer_test.go").write_text('package main\nimport("context";"testing")\nfunc TestLocalConsumer(t *testing.T){ if err:=run(context.Background());err!=nil{t.Fatal(err)} }\n')
        command(consumer, go, "mod", "tidy", env=env)
        listed = command(consumer, go, "list", "-m", "-json", "all", env=env)
        decoder, cursor, modules = json.JSONDecoder(), 0, []
        while cursor < len(listed):
            while cursor < len(listed) and listed[cursor].isspace():
                cursor += 1
            if cursor == len(listed):
                break
            item, cursor = decoder.raw_decode(listed, cursor)
            modules.append(item)
        local = [item for item in modules if item["Path"] in paths]
        if len(local) != len(paths) or any(item.get("Replace") or item["Version"] != record["version"] for item in local):
            raise ValueError("Consumer used a replacement or wrong candidate version")
        # Verify downloaded artifacts came from our exact candidate proxy, even
        # though public dependency resolution has a fallback proxy.
        for path in paths:
            downloaded = Path(env["GOMODCACHE"]) / "cache/download" / path / "@v" / (record["version"]+".zip")
            authored = proxy / path / "@v" / (record["version"]+".zip")
            if published:
                with zipfile.ZipFile(downloaded) as actual, zipfile.ZipFile(authored) as intended:
                    expected_bytes = {name: intended.read(name) for name in intended.namelist()}
                    actual_bytes = {name: actual.read(name) for name in actual.namelist()}
                prefix = path+"@"+version+"/"
                # Go includes a repository root LICENSE in a nested module archive.
                license_path = prefix+"LICENSE"
                if license_path in actual_bytes and license_path not in expected_bytes and "LICENSE" in files:
                    expected_bytes[license_path] = subprocess.check_output(["git", "show", "HEAD:LICENSE"], cwd=candidate, env=env)
                if actual_bytes != expected_bytes:
                    raise ValueError("Public module ZIP contents differ from exact candidate: "+path)
            elif hashlib.sha256(downloaded.read_bytes()).digest() != hashlib.sha256(authored.read_bytes()).digest():
                raise ValueError("Consumer downloaded a different candidate archive for "+path)
        downloads = []
        for path in paths:
            downloaded = json.loads(command(consumer, go, "mod", "download", "-json", path+"@"+version, env=env))
            if downloaded.get("Error") or not downloaded.get("Sum") or not downloaded.get("GoModSum"):
                raise ValueError("Missing checksum evidence for "+path)
            if published:
                authored_mod = proxy / path / "@v" / (version+".mod")
                if Path(downloaded["GoMod"]).read_bytes() != authored_mod.read_bytes():
                    raise ValueError("Public .mod differs from exact candidate: "+path)
            downloads.append(downloaded)
        command(consumer, go, "mod", "verify", env=env)
        if command(candidate, "git", "status", "--porcelain", env=env):
            raise ValueError("Consumer verification mutated the isolated candidate")
        command(consumer, go, "test", "-count=1", "-race", "./...", env=env)
        command(consumer, go, "build", "./...", env=env)
        print(json.dumps({"PASS": True, "source": source, "candidate": record["candidate"], "version": record["version"],
                          "modules": paths, "packages": packages, "tags": sorted(expected), "GOWORK": "off", "ragy_replacements": False,
                          "resolved_modules": modules, "downloads": downloads,
                          "publication": "public proxy with checksum verification" if published else "isolated artifact proxy, no refs"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reviewed_source")
    parser.add_argument("--version")
    parser.add_argument("--published", action="store_true")
    parser.add_argument("--exact-candidate", action="store_true")
    args = parser.parse_args()
    verify(args.reviewed_source, args.version, args.published, args.exact_candidate or args.published)
