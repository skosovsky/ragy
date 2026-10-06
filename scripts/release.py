#!/usr/bin/env python3
"""Publish only reviewed source and allowlisted module manifests/refs.

Persistent candidate recovery is implemented separately from release isolation.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


class ReleaseError(Exception):
    """A failed release precondition, without changing the caller checkout."""


def run(repo, *args):
    return subprocess.check_output(
        args, cwd=repo, text=True, env={**os.environ, "GOWORK": "off"}
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


def rewrite_manifests(checkout, files, root, version):
    for modfile in files:
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


def release(repo, kind, reviewed):
    if kind not in ("patch", "break") or not re.fullmatch(r"[0-9a-f]{40}", reviewed):
        raise ReleaseError("Usage: scripts/release.sh patch|break REVIEWED_FULL_SHA")
    selectors = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR",
                 "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_NAMESPACE", "GIT_CONFIG")
    if any(key in os.environ for key in selectors):
        raise ReleaseError("Repository-selection Git environment is unsupported")
    repo = Path(git(repo, "rev-parse", "--show-toplevel"))
    source = git(repo, "rev-parse", "--verify", reviewed + "^{commit}")
    if source != reviewed:
        raise ReleaseError("Reviewed source must be an exact commit object")
    if git(repo, "status", "--porcelain", "--untracked-files=no"):
        raise ReleaseError("Tracked files and index must be clean")
    modules, files, root = release_scope(repo, source)
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
    config = commit_config(repo)
    with tempfile.TemporaryDirectory(prefix="ragy-release-") as work:
        checkout = Path(work)
        git(checkout, "init", "--quiet")
        hooks = checkout / ".git/disabled-hooks"
        hooks.mkdir()
        git(checkout, "config", "core.hooksPath", str(hooks))
        git(checkout, "fetch", "--quiet", "--no-tags", "--", str(repo), source)
        git(checkout, "checkout", "--quiet", "--detach", source)
        rewrite_manifests(checkout, files, root, version)
        changed = git(checkout, "diff", "--name-only").splitlines()
        if not set(changed) <= set(files):
            raise ReleaseError("Candidate changed files outside manifest allowlist")
        git(checkout, "add", "--", *files)
        staged = git(checkout, "diff", "--cached", "--name-only").splitlines()
        if not set(staged) <= set(files):
            raise ReleaseError("Candidate staged files outside manifest allowlist")
        if staged:
            git(checkout, *config, "commit", "--quiet", "-m", f"chore: release {version}")
        candidate = git(checkout, "rev-parse", "HEAD")
        committed = git(checkout, "diff", "--name-only", source, candidate).splitlines()
        if not set(committed) <= set(files):
            raise ReleaseError("Candidate commit changed files outside manifest allowlist")
        for tag in tags:
            git(checkout, "tag", "--no-sign", tag, candidate)
        # Exact refs and atomic transaction; never force or fall back to --tags.
        git(checkout, "push", "--atomic", "--", remote, *[f"{ref}:{ref}" for ref in refs])
        print(f"Published {version} at {candidate}")


def main():
    try:
        if len(sys.argv) != 3:
            raise ReleaseError("Usage: scripts/release.sh patch|break REVIEWED_FULL_SHA")
        release(Path.cwd(), sys.argv[1], sys.argv[2])
    except (ReleaseError, subprocess.CalledProcessError, EOFError) as error:
        print(f"Release failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
