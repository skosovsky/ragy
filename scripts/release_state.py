"""Durable release state; no caller checkout/ref mutation."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
import re
from pathlib import Path


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def sync_directory(path):
    directory = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def state_root(repo, git, error):
    common = Path(git(repo, "rev-parse", "--git-common-dir"))
    if not common.is_absolute():
        common = repo / common
    root = common.resolve() / "ragy-releases"
    if root.is_symlink():
        raise error("Release state directory must not be a symlink")
    root.mkdir(mode=0o700, exist_ok=True)
    sync_directory(root.parent)
    return root


@contextmanager
def locked(root, error):
    path = root / "lock"
    if path.is_symlink():
        raise error("Release lock must not be a symlink")
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as cause:
            raise error("Another release operation is running") from cause
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def save(active, record):
    record["updated_at"] = timestamp()
    temporary = active / "state.json.tmp"
    if temporary.is_symlink() or (active / "state.json").is_symlink():
        raise ValueError("Candidate record must not be a symlink")
    with temporary.open("w") as output:
        json.dump(record, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, active / "state.json")
    sync_directory(active)
    sync_directory(active.parent)


def observe(repo, active, record, remote_refs):
    try:
        remote = remote_refs(repo, record["remote"])
    except Exception as cause:
        record.update(status="unknown", observed_refs={}, observed_at=timestamp(),
                      observation_error=str(cause))
        save(active, record)
        return "unknown"
    expected = record["expected_refs"]
    selected = {ref: remote[ref] for ref in expected if ref in remote}
    if any(oid != expected[ref] for ref, oid in selected.items()):
        status = "collision"
    elif not selected:
        status = "none"
    elif len(selected) == len(expected):
        status = "complete"
    else:
        status = "partial"
    record.update(status=status, observed_refs=selected, observed_at=timestamp(), observation_error=None)
    save(active, record)
    return status


def load(active, repo, git, release_scope, destination, error):
    if active.is_symlink() or (active / "state.json").is_symlink() or (active / "checkout").is_symlink() or (active / "checkout/.git").is_symlink():
        raise error("Release candidate paths must not be symlinks")
    try:
        record = json.loads((active / "state.json").read_text())
        if record["format"] != 1 or record["kind"] not in ("patch", "break"):
            raise ValueError("unsupported record")
        source = record["source"]
        if not re.fullmatch(r"[0-9a-f]{40}", source):
            raise ValueError("source SHA")
        candidate = record["candidate"]
        if candidate is not None and not re.fullmatch(r"[0-9a-f]{40}", candidate):
            raise ValueError("candidate SHA")
        source_repo = active / "checkout" if candidate is not None else repo
        if git(source_repo, "rev-parse", "--verify", source + "^{commit}") != source:
            raise ValueError("source identity")
        modules, files, root = release_scope(source_repo, source)
        if (modules, files, root) != (record["modules"], record["files"], record["root_module"]):
            raise ValueError("release scope")
        version = record["version"]
        # Validate persisted version/tag spelling, not a new version calculation.
        if not re.fullmatch(r"v[01]\.[0-9]+\.[0-9]+", version):
            raise ValueError("candidate version")
        refs = ["refs/tags/" + (version if module == "." else module + "/" + version) for module in modules]
        if record["expected_refs"] != dict.fromkeys(refs, record["candidate"]):
            raise ValueError("exact ref identity")
        if not isinstance(record["created_refs"], dict) or any(
                ref not in record["expected_refs"] or oid != record["expected_refs"][ref]
                for ref, oid in record["created_refs"].items()):
            raise ValueError("owned local refs")
        if record["phase"] == "prepared" and candidate is None:
            raise ValueError("prepared candidate identity")
        if record["remote"] != destination(repo):
            raise ValueError("origin push destination changed")
        if record["status"] not in ("none", "partial", "complete", "unknown", "collision"):
            raise ValueError("publication status")
        if record["phase"] not in ("preparing", "prepared"):
            raise ValueError("candidate phase")
    except (KeyError, ValueError, TypeError, OSError) as cause:
        raise error("Invalid or missing candidate record; preserve state for explicit investigation") from cause
    return record
