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


def proxy_module(candidate, module, version, files, proxy, env):
    modfile = "go.mod" if module == "." else module + "/go.mod"
    manifest = subprocess.check_output(["git", "show", "HEAD:"+modfile], cwd=candidate, env=env)
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
            payload = subprocess.check_output(["git", "show", "HEAD:"+name], cwd=candidate, env=env)
            archive.writestr(path+"@"+version+"/"+relative, payload)
    return path


def verify(source):
    if not re.fullmatch(r"[0-9a-f]{40}", source):
        raise ValueError("Reviewed source must be an exact full commit SHA")
    resolved = command(ROOT, "git", "rev-parse", "--verify", source+"^{commit}")
    if resolved != source:
        raise ValueError("Source is not an exact commit object")
    # Only readonly original Git commands; every mutation/publication is disposable.
    with tempfile.TemporaryDirectory(prefix="ragy-clean-consumer-") as temporary:
        root = Path(temporary)
        repo, remote = root / "reviewed", root / "remote.git"
        repo.mkdir()
        env = {**os.environ, "GOWORK": "off", "PYTHONDONTWRITEBYTECODE": "1", "GOENV": "off", "GOFLAGS": "", "GOTOOLCHAIN": "local"}
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
        command(root, "git", "init", "--bare", "--quiet", str(remote), env=env)
        command(repo, "git", "remote", "add", "origin", str(remote), env=env)
        before = command(repo, "git", "rev-parse", "HEAD", env=env)
        command(repo, "bash", "scripts/release.sh", "patch", source, env=env, input_text="y\n")
        record = json.loads((repo / ".git/ragy-releases/active/state.json").read_text())
        candidate = repo / ".git/ragy-releases/active/checkout"
        if record["source"] != source or record["status"] != "complete":
            raise ValueError("Candidate does not retain complete exact reviewed source")
        if command(repo, "git", "rev-parse", "HEAD", env=env) != before or command(repo, "git", "status", "--porcelain", env=env):
            raise ValueError("Release mutated fixture caller")
        parents = command(candidate, "git", "rev-list", "--parents", "-n", "1", "HEAD", env=env).split()
        if record["candidate"] != source and parents != [record["candidate"], source]:
            raise ValueError("Manifest-rewritten candidate must derive directly from source")
        tags = command(root, "git", "--git-dir="+str(remote), "for-each-ref", "--format=%(refname) %(objectname)", "refs/tags", env=env)
        expected = {ref+" "+oid for ref, oid in record["expected_refs"].items()}
        if set(tags.splitlines()) != expected or any("examples/" in line for line in tags.splitlines()):
            raise ValueError("Candidate remote tags differ from exact publishable scope")
        files = command(candidate, "git", "ls-tree", "-r", "--name-only", "HEAD", env=env).splitlines()
        changed = command(candidate, "git", "diff", "--name-only", source, "HEAD", env=env).splitlines()
        if not set(changed) <= set(record["files"]):
            raise ValueError("Candidate changed non-manifest source")
        proxy = root / "proxy"
        paths = [proxy_module(candidate, module, record["version"], files, proxy, env) for module in record["modules"]]
        consumer = root / "consumer"
        consumer.mkdir()
        env.update(GOPROXY=proxy.as_uri()+",https://proxy.golang.org", GOMODCACHE=str(root / "modcache"), GOPATH=str(root / "gopath"),
                   GONOSUMDB=record["root_module"]+","+record["root_module"]+"/*", GONOPROXY="none", GOSUMDB="sum.golang.org")
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
            if hashlib.sha256(downloaded.read_bytes()).digest() != hashlib.sha256(authored.read_bytes()).digest():
                raise ValueError("Consumer downloaded a different candidate archive for "+path)
        if command(candidate, "git", "status", "--porcelain", env=env):
            raise ValueError("Consumer verification mutated the isolated candidate")
        command(consumer, go, "test", "-count=1", "-race", "./...", env=env)
        command(consumer, go, "build", "./...", env=env)
        print(json.dumps({"PASS": True, "source": source, "candidate": record["candidate"], "version": record["version"],
                          "modules": paths, "packages": packages, "tags": sorted(expected), "GOWORK": "off", "ragy_replacements": False,
                          "publication": "disposable local bare remote only"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reviewed_source")
    args = parser.parse_args()
    verify(args.reviewed_source)
