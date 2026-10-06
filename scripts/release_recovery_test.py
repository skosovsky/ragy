#!/usr/bin/env python3
"""AAA durable release recovery with actual local Git transactions."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from release_test import Fixture


def active(fixture):
    return fixture.repo / ".git/ragy-releases/active"


def record(fixture):
    return json.loads((active(fixture) / "state.json").read_text())


def command(fixture, operation, env=None):
    return subprocess.run(["bash", str(fixture.repo / "scripts/release.sh"), operation],
                          cwd=fixture.repo, text=True, capture_output=True, timeout=60,
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **(env or {})})


def invoke(fixture, env=None):
    return subprocess.run(["bash", str(fixture.repo / "scripts/release.sh"), "patch", fixture.source],
                          cwd=fixture.repo, input="y\n", text=True, capture_output=True, timeout=60,
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **(env or {})})


def rejecting_hook(fixture):
    hook = fixture.remote / "hooks/pre-receive"
    hook.write_text("#!/bin/sh\nexit 1\n")
    hook.chmod(0o755)
    return hook


def transport(directory, fail_observe=0, fail_push=False, fail_tag=0):
    """Wrap the actual Git executable; only inject transport exit/observation loss."""
    bin_dir = Path(directory) / "bin"
    bin_dir.mkdir()
    count = Path(directory) / "observations"
    executable = bin_dir / "git"
    executable.write_text(f'''#!/usr/bin/env python3
from pathlib import Path
import subprocess,sys
args=sys.argv[1:]
if args and args[0]=='tag':
 p=Path({str(count)+'-tags'!r})
 n=int(p.read_text())+1 if p.exists() else 1
 p.write_text(str(n))
 if {fail_tag} and n>={fail_tag}:
  sys.exit(1)
if args and args[0]=='ls-remote':
 p=Path({str(count)!r})
 n=int(p.read_text())+1 if p.exists() else 1
 p.write_text(str(n))
 if {fail_observe} and n>={fail_observe}:
  sys.exit(1)
r=subprocess.run([{shutil.which('git')!r},*args])
if args and args[0]=='push' and {fail_push!r}:
 sys.exit(1)
sys.exit(r.returncode)
''')
    executable.chmod(0o755)
    return {"PATH": str(bin_dir) + os.pathsep + os.environ["PATH"]}


class Recovery(unittest.TestCase):
    def test_incomplete_and_recovered_commit_reject_unreviewed_manifest_content(self):
        for committed in (False, True):
            for mutation in ("module", "dependency"):
                with self.subTest(committed=committed, mutation=mutation), tempfile.TemporaryDirectory() as directory:
                    # Arrange: fail before Go editing, then alter only an allowlisted manifest.
                    fixture = Fixture(directory)
                    bin_dir = Path(directory) / "bin"
                    bin_dir.mkdir()
                    executable = bin_dir / "go"
                    executable.write_text("#!/bin/sh\nexit 1\n")
                    executable.chmod(0o755)
                    failed = invoke(fixture, {"PATH": str(bin_dir) + os.pathsep + os.environ["PATH"]})
                    self.assertNotEqual(failed.returncode, 0)
                    pending = record(fixture)
                    checkout = active(fixture) / "checkout"
                    path = checkout / "go.mod"
                    text = path.read_text()
                    if mutation == "module":
                        text = text.replace("example.invalid/ragy", "example.invalid/unreviewed")
                    else:
                        text += "\nrequire example.invalid/private v1.0.0\n"
                    path.write_text(text)
                    if committed:
                        subprocess.run(["git", "add", "go.mod"], cwd=checkout, check=True)
                        subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                                        "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "interrupted commit"],
                                       cwd=checkout, check=True)
                    before = fixture.snapshot()
                    # Act.
                    resumed = command(fixture, "resume")
                    # Assert: filename allowlists cannot authorize new module identity/external dependencies.
                    self.assertNotEqual(resumed.returncode, 0)
                    self.assertIn("unreviewed manifest content", resumed.stderr)
                    self.assertEqual(record(fixture)["source"], pending["source"])
                    self.assertEqual(record(fixture)["version"], pending["version"])
                    self.assertIsNone(record(fixture)["candidate"])
                    self.assertEqual(path.read_text(), text)
                    self.assertEqual(fixture.remote_git("tag", "--list"), "")
                    self.assertEqual(fixture.snapshot(), before)

    def test_read_checks_preserve_caller_index_bytes(self):
        # Arrange: same tracked bytes with changed stat metadata would normally refresh the index on git status.
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            path = fixture.repo / "reviewed.txt"
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 3000000000))
            index = fixture.repo / ".git/index"
            before = index.read_bytes()
            hook = rejecting_hook(fixture)
            # Act.
            failed = fixture.invoke()
            after_failure = index.read_bytes()
            hook.unlink()
            resumed = command(fixture, "resume")
            # Assert: even optional status-refresh writes are disabled in caller Git reads.
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(after_failure, before)
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertEqual(index.read_bytes(), before)

    def test_reject_then_retry_preserves_exact_candidate_and_version(self):
        # Arrange.
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            fixture.git("tag", "scratch-local")
            fixture.write("private.txt", "private synthetic payload\n")
            hook = rejecting_hook(fixture)
            before = fixture.snapshot()
            # Act: a real rejecting atomic push, then retry through the original invocation.
            first = fixture.invoke()
            failed = record(fixture)
            hook.unlink()
            retry = fixture.invoke()
            completed = record(fixture)
            # Assert.
            self.assertNotEqual(first.returncode, 0)
            self.assertEqual(failed["status"], "none")
            self.assertEqual(failed["phase"], "prepared")
            self.assertEqual(failed["created_refs"], failed["expected_refs"])
            self.assertEqual(retry.returncode, 0, retry.stderr)
            self.assertEqual(completed["status"], "complete")
            self.assertEqual(completed["candidate"], failed["candidate"])
            self.assertEqual(completed["version"], "v0.0.1")
            self.assertEqual(fixture.snapshot(), before)
            self.assertEqual(fixture.git("tag", "--list"), "scratch-local")
            self.assertEqual(fixture.remote_git("rev-parse", "v0.0.1"), failed["candidate"])

    def test_failure_before_tags_preserves_version_then_resumes(self):
        # Arrange: genuine subprocess failure from Go before any tags/commit exist.
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            bin_dir = Path(directory) / "bin"
            bin_dir.mkdir()
            executable = bin_dir / "go"
            executable.write_text("#!/bin/sh\nexit 1\n")
            executable.chmod(0o755)
            before = fixture.snapshot()
            # Act.
            failed = invoke(fixture, {"PATH": str(bin_dir) + os.pathsep + os.environ["PATH"]})
            pending = record(fixture)
            retry = command(fixture, "resume")
            # Assert.
            self.assertNotEqual(failed.returncode, 0)
            self.assertIsNone(pending["candidate"])
            self.assertEqual(pending["created_refs"], {})
            self.assertEqual(pending["status"], "none")
            self.assertEqual(pending["version"], "v0.0.1")
            self.assertEqual(retry.returncode, 0, retry.stderr)
            self.assertEqual(record(fixture)["version"], pending["version"])
            self.assertEqual(fixture.snapshot(), before)

    def test_partial_external_publication_completes_only_missing_refs(self):
        # Arrange: failed atomic push leaves owned tags; an external actor publishes one.
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            hook = rejecting_hook(fixture)
            fixture.invoke()
            saved = record(fixture)
            hook.unlink()
            checkout = active(fixture) / "checkout"
            subprocess.run(["git", "push", "--quiet", str(fixture.remote), "refs/tags/v0.0.1"],
                           cwd=checkout, check=True)
            log = Path(directory) / "updates"
            hook.write_text(f'#!/bin/sh\ncat > "{log}"\n')
            hook.chmod(0o755)
            before = fixture.snapshot()
            # Act.
            inspected = command(fixture, "inspect")
            partial = record(fixture)
            retry = command(fixture, "resume")
            # Assert.
            self.assertEqual(inspected.returncode, 0, inspected.stderr)
            self.assertEqual(partial["status"], "partial")
            self.assertEqual(retry.returncode, 0, retry.stderr)
            self.assertEqual(record(fixture)["candidate"], saved["candidate"])
            self.assertEqual(record(fixture)["status"], "complete")
            self.assertEqual(len(log.read_text().splitlines()), 1)
            self.assertIn("refs/tags/adapters/test/v0.0.1", log.read_text())
            self.assertEqual(fixture.snapshot(), before)

    def test_unknown_after_push_requires_explicit_inspect_then_idempotent_resume(self):
        # Arrange: actual successful publication, but final remote observation fails.
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            env = transport(directory, fail_observe=3)
            before = fixture.snapshot()
            # Act.
            failed = invoke(fixture, env)
            uncertain = record(fixture)
            refused = command(fixture, "resume")
            inspected = command(fixture, "inspect")
            resumed = command(fixture, "resume")
            # Assert: exit code is not evidence of none; unknown cannot dispatch another push.
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(uncertain["status"], "unknown")
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("run inspect", refused.stderr)
            self.assertEqual(inspected.returncode, 0, inspected.stderr)
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertIn("Already published", resumed.stdout)
            self.assertEqual(record(fixture)["candidate"], uncertain["candidate"])
            self.assertEqual(record(fixture)["status"], "complete")
            self.assertEqual(fixture.snapshot(), before)

    def test_error_exit_after_actual_push_is_reconciled_as_complete(self):
        # Arrange: successful actual Git push followed by synthetic transport exit failure.
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            env = transport(directory, fail_push=True)
            # Act.
            completed = invoke(fixture, env)
            # Assert.
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(record(fixture)["status"], "complete")
            self.assertIsNotNone(record(fixture)["last_error"])

    def test_collision_inspection_preserves_different_remote_object(self):
        # Arrange.
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            hook = rejecting_hook(fixture)
            fixture.invoke()
            saved = record(fixture)
            hook.unlink()
            fixture.git("tag", "adapters/test/v0.0.1", fixture.source)
            fixture.git("push", "--quiet", "origin", "refs/tags/adapters/test/v0.0.1")
            before = fixture.snapshot()
            # Act.
            inspected = command(fixture, "inspect")
            resumed = command(fixture, "resume")
            # Assert: no force/deletion, even if a tag points to the source ancestor.
            self.assertNotEqual(inspected.returncode, 0)
            self.assertEqual(record(fixture)["status"], "collision")
            self.assertNotEqual(resumed.returncode, 0)
            self.assertEqual(record(fixture)["candidate"], saved["candidate"])
            self.assertEqual(fixture.remote_git("rev-parse", "adapters/test/v0.0.1"), fixture.source)
            self.assertEqual(fixture.snapshot(), before)

    def test_failure_between_tags_preserves_candidate_and_resumes(self):
        # Arrange: fail the second actual tag creation after the first has been recorded.
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            env = transport(directory, fail_tag=2)
            before = fixture.snapshot()
            # Act.
            failed = invoke(fixture, env)
            pending = record(fixture)
            resumed = command(fixture, "resume")
            # Assert.
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(pending["phase"], "preparing")
            self.assertEqual(len(pending["created_refs"]), 1)
            self.assertEqual(pending["status"], "none")
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertEqual(record(fixture)["candidate"], pending["candidate"])
            self.assertEqual(record(fixture)["version"], pending["version"])
            self.assertEqual(record(fixture)["status"], "complete")
            self.assertEqual(fixture.snapshot(), before)

    def test_atomic_unsupported_never_falls_back_to_partial_push(self):
        # Arrange: actual Git server disables atomic advertisement.
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            fixture.remote_git("config", "receive.advertiseAtomic", "false")
            before = fixture.snapshot()
            # Act.
            failed = fixture.invoke()
            pending = record(fixture)
            unpublished = fixture.remote_git("tag", "--list")
            fixture.remote_git("config", "receive.advertiseAtomic", "true")
            resumed = command(fixture, "resume")
            # Assert: unsupported capability left zero refs; resume kept the candidate.
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(pending["status"], "none")
            self.assertEqual(unpublished, "")
            self.assertIn("does not support --atomic", failed.stderr)
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertEqual(record(fixture)["candidate"], pending["candidate"])
            self.assertEqual(record(fixture)["version"], pending["version"])
            self.assertEqual(fixture.snapshot(), before)

    def test_annotated_ref_to_same_commit_is_exact_object_collision(self):
        # Arrange: expected refs are lightweight, not merely a matching peeled commit.
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            hook = rejecting_hook(fixture)
            fixture.invoke()
            saved = record(fixture)
            hook.unlink()
            checkout = active(fixture) / "checkout"
            subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                            "-c", "tag.gpgsign=false", "tag", "-a", "foreign", saved["candidate"], "-m", "foreign"],
                           cwd=checkout, check=True)
            subprocess.run(["git", "push", "--quiet", str(fixture.remote),
                            "refs/tags/foreign:refs/tags/v0.0.1"], cwd=checkout, check=True)
            before = fixture.snapshot()
            oid = fixture.remote_git("rev-parse", "v0.0.1")
            # Act.
            inspected = command(fixture, "inspect")
            resumed = command(fixture, "resume")
            # Assert.
            self.assertNotEqual(inspected.returncode, 0)
            self.assertEqual(record(fixture)["status"], "collision")
            self.assertNotEqual(resumed.returncode, 0)
            self.assertEqual(fixture.remote_git("rev-parse", "v0.0.1"), oid)
            self.assertEqual(fixture.remote_git("rev-parse", "v0.0.1^{}"), saved["candidate"])
            self.assertNotEqual(oid, saved["candidate"])
            self.assertEqual(fixture.snapshot(), before)

    def test_finish_archives_complete_record_before_new_version(self):
        # Arrange.
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            fixture.invoke()
            saved = record(fixture)
            fixture.write("reviewed.txt", "next reviewed source\n")
            fixture.git("add", "reviewed.txt")
            fixture.git("commit", "--quiet", "-m", "next reviewed source")
            fixture.source = fixture.git("rev-parse", "HEAD")
            before = fixture.snapshot()
            # Act: a different source is forbidden until explicit inspected finish.
            rejected = fixture.invoke()
            finished = command(fixture, "finish")
            completed = fixture.invoke()
            # Assert.
            self.assertNotEqual(rejected.returncode, 0)
            self.assertEqual(finished.returncode, 0, finished.stderr)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(record(fixture)["version"], "v0.0.2")
            history = fixture.repo / ".git/ragy-releases/history" / (saved["version"] + "-" + saved["candidate"])
            self.assertTrue((history / "state.json").is_file())
            self.assertEqual(json.loads((history / "state.json").read_text())["candidate"], saved["candidate"])
            self.assertEqual(fixture.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
