#!/usr/bin/env python3
"""Publish only reviewed source and allowlisted module manifests/refs.

Persistent candidate identity and remote observations govern every retry.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

import release_state as state


class ReleaseError(Exception):
    """A failed release precondition, without changing the caller checkout."""


def run(repo, *args):
    return subprocess.check_output(
        args, cwd=repo, text=True, env={**os.environ, "GOWORK": "off", "GIT_OPTIONAL_LOCKS": "0"}
    ).strip()


def git(repo, *args):
    return run(repo, "git", *args)


def tracked_source(repo, source, path):
    entry = git(repo, "ls-tree", source, "--", path).split()
    if len(entry) != 4 or entry[0] not in ("100644", "100755") or entry[3] != path:
        raise ReleaseError(f"Not a tracked regular source file: {path}")
    return git(repo, "show", f"{source}:{path}")


def module_path(text):
    match = re.search(r'^module\s+([^\s]+)\s*$', text, re.MULTILINE)
    if not match:
        raise ReleaseError("Missing module path")
    return match[1].strip('"')


def release_scope(repo, source):
    modules = tracked_source(repo, source, "scripts/release-modules.txt").splitlines()
    if not modules or modules[0] != "." or len(set(modules)) != len(modules):
        raise ReleaseError("Publishable manifest must start with root and contain unique modules")
    root = module_path(tracked_source(repo, source, "go.mod"))
    files = []
    for module in modules:
        if module != "." and not re.fullmatch(r"adapters/[a-z0-9_-]+(?:/[a-z0-9_-]+)*", module):
            raise ReleaseError(f"Invalid publishable module directory: {module}")
        modfile = "go.mod" if module == "." else f"{module}/go.mod"
        expected = root if module == "." else f"{root}/{module}"
        if module_path(tracked_source(repo, source, modfile)) != expected:
            raise ReleaseError(f"Unexpected module path: {modfile}")
        files.append(modfile)
        sumfile = modfile.removesuffix("go.mod") + "go.sum"
        if module != "." or git(repo, "ls-tree", source, "--", sumfile):
            files.append(sumfile)
    return modules, files, root


def destination(repo):
    urls = git(repo, "remote", "get-url", "--push", "--all", "origin").splitlines()
    if len(urls) != 1:
        raise ReleaseError("Exactly one origin push destination is required")
    url = urls[0]
    # Git resolves filesystem remotes relative to cwd; bind them before isolation.
    colon, slash = url.find(":"), url.find("/")
    local_path = colon < 0 or (slash >= 0 and slash < colon)
    if local_path and not Path(url).is_absolute():
        url = str((repo / url).resolve())
    return url


def remote_refs(repo, remote):
    lines = git(repo, "ls-remote", "--refs", "--", remote, "refs/tags/*").splitlines()
    return dict((ref, oid) for oid, ref in (line.split() for line in lines))


def next_version(refs, kind):
    versions = [tuple(map(int, m.groups())) for ref in refs
                if (m := re.fullmatch(r"refs/tags/v([0-9]+)\.([0-9]+)\.([0-9]+)", ref))]
    major, minor, patch = max(versions, default=(0, 0, 0))
    if kind == "break":
        if major == 0:
            minor, patch = minor + 1, 0
        else:
            major, minor, patch = major + 1, 0, 0
    else:
        patch += 1
    if major >= 2:
        raise ReleaseError("v2+ requires a reviewed semantic import-version migration")
    return f"v{major}.{minor}.{patch}"


def commit_config(repo):
    result = []
    for key in ("user.name", "user.email", "user.signingkey", "commit.gpgsign",
                "gpg.format", "gpg.program", "gpg.ssh.program"):
        value = subprocess.run(["git", "config", "--get", key], cwd=repo,
                               text=True, capture_output=True, check=False)
        if value.returncode == 0:
            result += ["-c", f"{key}={value.stdout.strip()}"]
    return result


def rewrite_manifests(checkout, files, root, version, source_repo=None, source="HEAD"):
    for modfile in files:
        if not modfile.endswith("go.mod"):
            continue
        data = json.loads(run(checkout, "go", "mod", "edit", "-json", modfile))
        options = []
        for requirement in data.get("Require") or []:
            path = requirement["Path"]
            if path == root or path.startswith(root + "/"):
                options.append(f"-require={path}@{version}")
        for replace in data.get("Replace") or []:
            old = replace["Old"]
            if old["Path"] == root or old["Path"].startswith(root + "/"):
                target = old["Path"] + ("@" + old["Version"] if old.get("Version") else "")
                options.append(f"-dropreplace={target}")
        run(checkout, "go", "mod", "edit", *options, "-fmt", modfile)

    # Candidate adapter go.sum must carry the exact future root artifact checksums.
    # Root has no own-module dependencies; nested-module manifests/sums are excluded
    # from its archive, so this does not introduce a checksum cycle.
    if source_repo is None:
        source_repo = checkout
    sums = [path for path in files if path != "go.sum" and path.endswith("go.sum")]
    if not sums:
        return
    root_requirements = json.loads(run(checkout, "go", "mod", "edit", "-json", "go.mod")).get("Require") or []
    if any(item["Path"] == root or item["Path"].startswith(root+"/") for item in root_requirements):
        raise ReleaseError("Root own-module dependencies need separately reviewed checksum preparation")
    from check_release_consumer import proxy_module
    with tempfile.TemporaryDirectory(prefix="ragy-candidate-sums-") as temporary:
        directory = Path(temporary)
        proxy = directory / "proxy"
        listing = git(source_repo, "ls-tree", "-r", "--name-only", source).splitlines()
        env = {**os.environ, "GOWORK": "off", "GOENV": "off", "GOFLAGS": "", "GOPRIVATE": "",
               "GOPROXY": proxy.as_uri(), "GONOSUMDB": root, "GONOPROXY": "none", "GOSUMDB": "sum.golang.org",
               "GOMODCACHE": str(directory / "modcache"), "GOPATH": str(directory / "gopath")}
        proxy_module(source_repo, ".", version, listing, proxy, env, revision=source,
                     overrides={"go.mod": (checkout / "go.mod").read_bytes()})
        (directory / "go.mod").write_text("module example.invalid/checksum-preparation\n")
        downloaded = json.loads(subprocess.check_output([os.environ.get("GO", "go"), "mod", "download", "-json", root+"@"+version], cwd=directory, env=env, text=True))
        if downloaded.get("Error") or not downloaded.get("Sum") or not downloaded.get("GoModSum"):
            raise ReleaseError("Cannot determine exact candidate module checksums")
        additions = [f"{root} {version} {downloaded['Sum']}", f"{root} {version}/go.mod {downloaded['GoModSum']}"]
        for path in sums:
            lines = (checkout / path).read_text().splitlines() if (checkout / path).exists() else []
            # Replace only this intended version, keeping historical/public dependency sums.
            lines = [line for line in lines if not line.startswith(root+" "+version+" ") and not line.startswith(root+" "+version+"/go.mod ")]
            (checkout / path).write_text("\n".join(sorted(set(lines+additions)))+"\n")


def caller_repo(repo, clean=True):
    selectors = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR",
                 "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_NAMESPACE", "GIT_CONFIG")
    if any(key in os.environ for key in selectors):
        raise ReleaseError("Repository-selection Git environment is unsupported")
    repo = Path(git(repo, "rev-parse", "--show-toplevel"))
    if clean and git(repo, "status", "--porcelain", "--untracked-files=no"):
        raise ReleaseError("Tracked files and index must be clean")
    return repo


def intended_manifests(checkout, record):
    original = {}
    with tempfile.TemporaryDirectory(prefix="ragy-manifests-") as temporary:
        tree = Path(temporary)
        for path in record["files"]:
            entry = git(checkout, "ls-tree", record["source"], "--", path).split()
            if not entry and path.endswith("/go.sum"):
                original[path] = None
            else:
                if len(entry) != 4 or entry[0] not in ("100644", "100755") or entry[3] != path:
                    raise ReleaseError(f"Not a tracked regular manifest file: {path}")
                original[path] = subprocess.check_output(
                    ["git", "show", f"{record['source']}:{path}"], cwd=checkout,
                    env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
                )
            target = tree / path
            target.parent.mkdir(parents=True, exist_ok=True)
            if original[path] is not None:
                target.write_bytes(original[path])
        rewrite_manifests(tree, record["files"], record["root_module"], record["version"], checkout, record["source"])
        expected = {path: (tree / path).read_bytes() for path in record["files"]}
    return original, expected


def validate_preparing_files(checkout, original, expected):
    for path, content in expected.items():
        target = checkout / path
        cursor = checkout
        for part in Path(path).parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ReleaseError("Stored manifest paths must not be symlinks")
        actual = target.read_bytes() if target.exists() else None
        if actual not in (original[path], content):
            raise ReleaseError("Incomplete preparation contains unreviewed manifest content")


def verify_candidate(checkout, record, expected):
    source, candidate = record["source"], record["candidate"]
    if git(checkout, "rev-parse", "HEAD") != candidate:
        raise ReleaseError("Stored checkout HEAD does not match candidate identity")
    if candidate != source and git(checkout, "rev-list", "--parents", "-n", "1", candidate).split() != [candidate, source]:
        raise ReleaseError("Candidate is not derived directly from reviewed source")
    changed = git(checkout, "diff", "--name-only", source, candidate).splitlines()
    if not set(changed) <= set(record["files"]):
        raise ReleaseError("Candidate commit changed files outside manifest allowlist")
    for path, content in expected.items():
        committed = subprocess.check_output(["git", "show", f"{candidate}:{path}"], cwd=checkout)
        if committed != content:
            raise ReleaseError("Candidate contains unreviewed manifest content")
    if git(checkout, "status", "--porcelain", "--untracked-files=no"):
        raise ReleaseError("Stored candidate checkout is dirty")


def prepare(repo, active, record):
    checkout = active / "checkout"
    checkout.mkdir(exist_ok=True)
    if not (checkout / ".git").exists():
        git(checkout, "init", "--quiet")
    hooks = checkout / ".git/disabled-hooks"
    hooks.mkdir(exist_ok=True)
    git(checkout, "config", "core.hooksPath", str(hooks))
    source = record["source"]
    present = subprocess.run(["git", "cat-file", "-e", source + "^{commit}"], cwd=checkout,
                             capture_output=True, check=False)
    if present.returncode != 0:
        git(checkout, "fetch", "--quiet", "--no-tags", "--", str(repo), source)
    current = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=checkout,
                             text=True, capture_output=True, check=False)
    if current.returncode != 0 and not record["candidate"]:
        git(checkout, "checkout", "--quiet", "--detach", source)
        current = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=checkout,
                                 text=True, capture_output=True, check=False)
    original, expected = intended_manifests(checkout, record)
    if not record["candidate"]:
        if current.stdout.strip() != source:
            # Recover a committed candidate after interruption before the record write.
            recovered = {**record, "candidate": current.stdout.strip()}
            verify_candidate(checkout, recovered, expected)
            record["candidate"] = recovered["candidate"]
        if not record["candidate"]:
            validate_preparing_files(checkout, original, expected)
            rewrite_manifests(checkout, record["files"], record["root_module"], record["version"], checkout, record["source"])
            changed = git(checkout, "diff", "--name-only").splitlines()
            if not set(changed) <= set(record["files"]):
                raise ReleaseError("Candidate changed files outside manifest allowlist")
            git(checkout, "add", "--", *record["files"])
            staged = git(checkout, "diff", "--cached", "--name-only").splitlines()
            if not set(staged) <= set(record["files"]):
                raise ReleaseError("Candidate staged files outside manifest allowlist")
            if staged:
                git(checkout, *commit_config(repo), "commit", "--quiet", "-m", f"chore: release {record['version']}")
            record["candidate"] = git(checkout, "rev-parse", "HEAD")
        record["expected_refs"] = dict.fromkeys(record["expected_refs"], record["candidate"])
        state.save(active, record)
    verify_candidate(checkout, record, expected)
    # Full gate precedes even isolated release tag creation.
    validate_gates(repo, active, checkout, record, expected)
    caller_refs = git(repo, "for-each-ref", "--format=%(refname) %(objectname)", "refs/tags").splitlines()
    for entry in caller_refs:
        ref, oid = entry.split()
        if ref in record["expected_refs"] and record["expected_refs"][ref] != oid:
            raise ReleaseError("Caller tag collision; no refs will be overwritten")
    for ref, oid in record["expected_refs"].items():
        existing = subprocess.run(["git", "show-ref", "--verify", "--hash", ref], cwd=checkout,
                                  text=True, capture_output=True, check=False)
        if existing.returncode == 0:
            if existing.stdout.strip() != oid:
                raise ReleaseError("Isolated tag collision; preserve all refs")
        else:
            git(checkout, "tag", "--no-sign", ref.removeprefix("refs/tags/"), oid)
        record["created_refs"][ref] = oid
        state.save(active, record)
    record["phase"] = "prepared"
    state.save(active, record)
    return checkout


def checkout_fingerprint(checkout):
    if git(checkout, "status", "--porcelain", "--untracked-files=all"):
        raise ReleaseError("Check mutated source/candidate checkout")
    files = git(checkout, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
    return hashlib.sha256(b"".join(path.encode()+b"\0"+(checkout/path).read_bytes() for path in files)).hexdigest()


def full_gate(checkout, active, label, record, candidate=False):
    identity = git(checkout, "rev-parse", "HEAD")
    fingerprint = checkout_fingerprint(checkout)
    output = Path(tempfile.mkdtemp(prefix=label+"-check-", dir=active))
    report = output / "summary.json"
    argv = [sys.executable, "scripts/verify.py", "check", "--output", str(output),
            "--source", identity, "--version", record["version"]]
    if candidate:
        argv.append("--candidate")
    env = {**os.environ, "GOWORK": "off", "PYTHONDONTWRITEBYTECODE": "1", "RAGY_CANDIDATE_VERSION": record["version"]}
    plan = json.loads(subprocess.check_output([sys.executable, "scripts/verify.py", "check", "--list"], cwd=checkout, env=env, text=True))
    report.unlink(missing_ok=True)
    subprocess.run(argv, cwd=checkout, env=env, check=True)
    if git(checkout, "rev-parse", "HEAD") != identity or checkout_fingerprint(checkout) != fingerprint:
        raise ReleaseError("Source/candidate changed after check; gate invalidated")
    evidence = json.loads(report.read_text())
    # Runner exit status is mandatory; required lane evidence also cannot be omitted.
    lanes = evidence["results"]
    required = {row["id"] for row in plan if not row.get("optional", False)}
    observed = {row["id"]: row for row in lanes}
    if (not required or len(observed) != len(lanes) or set(observed) != {row["id"] for row in plan}
            or evidence.get("source") != identity or evidence.get("status") != "PASS"
            or evidence.get("version") != record["version"] or evidence.get("candidate") is not candidate
            or evidence.get("profile") != "check"
            or any(not observed[name].get("required") or observed[name]["status"] != "PASS" for name in required)):
        raise ReleaseError("Full gate has missing/non-PASS required evidence")
    return {"commit": identity, "fingerprint": fingerprint, "report": str(report),
            "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest()}


def validate_gates(repo, active, checkout, record, expected):
    # Always rerun on recovery: saved PASS is evidence, never authority to push later bytes.
    source_tree = active / "source"
    if not source_tree.exists():
        source_tree.mkdir()
        git(source_tree, "init", "--quiet")
        hooks = source_tree / ".git/disabled-hooks"
        hooks.mkdir()
        git(source_tree, "config", "core.hooksPath", str(hooks))
        git(source_tree, "fetch", "--quiet", "--no-tags", "--", str(repo), record["source"])
        git(source_tree, "checkout", "--quiet", "--detach", record["source"])
    if git(source_tree, "rev-parse", "HEAD") != record["source"]:
        raise ReleaseError("Selected source checkout identity changed")
    record["source_gate"] = full_gate(source_tree, active, "source", record)
    record["candidate_gate"] = full_gate(checkout, active, "candidate", record, candidate=True)
    verify_candidate(checkout, record, expected)
    state.save(active, record)


def verify_publication(active, checkout, record):
    record["publication_verified"] = False
    state.save(active, record)
    log = active / "published-consumer.log"
    env = {**os.environ, "GOWORK": "off", "PYTHONDONTWRITEBYTECODE": "1", "GOPRIVATE": "", "GONOSUMDB": "", "GONOPROXY": "none", "GOSUMDB": "sum.golang.org"}
    # Fresh module caches per attempt inside consumer; bounded proxy propagation retries.
    for attempt in range(6):
        with log.open("a") as output:
            result = subprocess.run([sys.executable, "scripts/check_release_consumer.py", record["candidate"],
                                     "--published", "--version", record["version"]], cwd=checkout,
                                    env=env, stdout=output, stderr=subprocess.STDOUT, check=False)
        if result.returncode == 0:
            record.update(publication_verified=True, published_evidence=str(log), verified_at=state.timestamp())
            state.save(active, record)
            return
        if attempt < 5:
            time.sleep(10)
    record["last_error"] = "Public proxy/checksum/consumer verification failed; resume retries verification without rewriting tags"
    state.save(active, record)
    raise ReleaseError(record["last_error"])


def publish(repo, active, record):
    if record["status"] == "unknown":
        raise ReleaseError("Publication is unknown; run inspect before resume")
    if record["status"] == "collision":
        raise ReleaseError("Candidate collision requires explicit host resolution and inspect")
    try:
        checkout = prepare(repo, active, record)
    except Exception as cause:
        record["last_error"] = str(cause)
        state.observe(repo, active, record, remote_refs)
        raise
    status = state.observe(repo, active, record, remote_refs)
    if status == "complete":
        verify_publication(active, checkout, record)
        print(f"Already published {record['version']} at {record['candidate']}")
        return
    if status not in ("none", "partial"):
        raise ReleaseError(f"Publication {status}; inspect before any further push")
    missing = [ref for ref in record["expected_refs"] if ref not in record["observed_refs"]]
    # Persist uncertainty before dispatch; killed/lost transport requires explicit inspect.
    record["status"] = "unknown"
    state.save(active, record)
    failure = None
    try:
        git(checkout, "push", "--atomic", "--", record["remote"], *[f"{ref}:{ref}" for ref in missing])
    except subprocess.CalledProcessError as cause:
        failure = str(cause)
    finally:
        record["last_error"] = failure
        status = state.observe(repo, active, record, remote_refs)
    if status != "complete":
        raise ReleaseError(f"Publication {status}; preserved {record['version']} candidate. {failure or ''}")
    verify_publication(active, checkout, record)
    print(f"Published {record['version']} at {record['candidate']}")


def release(repo, kind, reviewed):
    if kind not in ("patch", "break") or not re.fullmatch(r"[0-9a-f]{40}", reviewed):
        raise ReleaseError("Usage: scripts/release.sh patch|break REVIEWED_FULL_SHA")
    repo = caller_repo(repo)
    source = git(repo, "rev-parse", "--verify", reviewed + "^{commit}")
    if source != reviewed:
        raise ReleaseError("Reviewed source must be an exact commit object")
    root = state.state_root(repo, git, ReleaseError)
    with state.locked(root, ReleaseError):
        active = root / "active"
        if active.exists():
            record = state.load(active, repo, git, release_scope, destination, ReleaseError)
            if (source, kind) != (record["source"], record["kind"]):
                raise ReleaseError("An existing candidate must be resumed/inspected/finished before choosing another")
            publish(repo, active, record)
            return
        modules, files, module_root = release_scope(repo, source)
        remote = destination(repo)
        observed = remote_refs(repo, remote)
        version = next_version(observed, kind)
        tags = [version if module == "." else f"{module}/{version}" for module in modules]
        refs = [f"refs/tags/{tag}" for tag in tags]
        local = set(git(repo, "for-each-ref", "--format=%(refname)", "refs/tags").splitlines())
        if any(ref in local or ref in observed for ref in refs):
            raise ReleaseError("Candidate tag collision; no refs will be overwritten")
        print(f"Reviewed source: {source}\nDestination: {remote}\nVersion: {version}")
        print("Modules: " + ", ".join(modules))
        print("Permitted changes: " + ", ".join(files))
        print("Exact refs: " + ", ".join(refs))
        if input("Publish this reviewed scope? [y/N] ").lower() != "y":
            raise ReleaseError("Aborted")
        active.mkdir(mode=0o700)
        record = dict(format=1, kind=kind, version=version, source=source, candidate=None,
                      modules=modules, files=files, root_module=module_root, remote=remote,
                      expected_refs=dict.fromkeys(refs), created_refs={}, phase="preparing",
                      status="none", observed_refs={}, observed_at=state.timestamp(), last_error=None)
        state.save(active, record)
        publish(repo, active, record)


def recovery(repo, operation):
    repo = caller_repo(repo, clean=operation != "inspect")
    root = state.state_root(repo, git, ReleaseError)
    with state.locked(root, ReleaseError):
        active = root / "active"
        record = state.load(active, repo, git, release_scope, destination, ReleaseError)
        if operation == "resume":
            publish(repo, active, record)
        else:
            status = state.observe(repo, active, record, remote_refs)
            print(json.dumps(record, indent=2, sort_keys=True))
            if operation == "finish":
                if status != "complete" or not record.get("publication_verified"):
                    raise ReleaseError("Only complete tags plus verified public checksums/consumer can be finished; preserve outstanding state")
                history = root / "history"
                history.mkdir(exist_ok=True)
                active.rename(history / (record["version"] + "-" + record["candidate"]))
                state.sync_directory(history)
                state.sync_directory(root)
            elif status in ("unknown", "collision"):
                raise ReleaseError(f"Inspection {status}; candidate preserved")


def main():
    try:
        if len(sys.argv) == 2 and sys.argv[1] in ("inspect", "resume", "finish"):
            recovery(Path.cwd(), sys.argv[1])
        elif len(sys.argv) == 3:
            release(Path.cwd(), sys.argv[1], sys.argv[2])
        else:
            raise ReleaseError("Usage: scripts/release.sh patch|break REVIEWED_FULL_SHA | inspect|resume|finish")
    except (ReleaseError, subprocess.CalledProcessError, EOFError, OSError, KeyboardInterrupt) as error:
        print(f"Release failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
