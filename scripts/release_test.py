#!/usr/bin/env python3
"""AAA release acceptance against disposable local repositories only."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parent


class Fixture:
    def __init__(self, directory):
        self.repo = Path(directory) / "caller"
        self.remote = Path(directory) / "remote.git"
        self.repo.mkdir()
        subprocess.run(["git", "init", "--bare", "--quiet", str(self.remote)], check=True)
        self.git("init", "--quiet")
        for key, value in (("user.name", "Release fixture"), ("user.email", "fixture@example.invalid"),
                           ("commit.gpgsign", "false"), ("tag.gpgsign", "false")):
            self.git("config", key, value)
        self.git("remote", "add", "origin", str(self.remote))
        for name in ("release.sh", "release.py", "release_state.py", "check_release_consumer.py", "process_runner.py"):
            self.write("scripts/" + name, (SCRIPTS / name).read_text().replace("time.sleep(10)", "time.sleep(0)"))
        # Fixture-owned registry runner validates immutable input and emits real required evidence.
        self.write("scripts/verify.py", "import json, pathlib, sys\n"
                   "args=sys.argv; root=pathlib.Path.cwd()\n"
                   "if '--list' in args: print(json.dumps([{'id':'fixture-required'}])); sys.exit(0)\n"
                   "failure=(root/'gate-failure').exists()\n"
                   "if (root/'gate-mutation').exists() and '--candidate' in args: (root/'reviewed.txt').write_text('mutated')\n"
                   "output=pathlib.Path(args[args.index('--output')+1]); output.mkdir(exist_ok=True)\n"
                   "(output/'summary.json').write_text(json.dumps({'source':args[args.index('--source')+1],'profile':'check','candidate':'--candidate' in args,'version':args[args.index('--version')+1],'status':'FAIL' if failure else 'PASS','results':[{'id':'fixture-required','required':True,'status':'FAIL' if failure else 'PASS'}]}))\n"
                   "sys.exit(1 if failure else 0)\n")
        self.write("scripts/check_release_consumer.py", (SCRIPTS / "check_release_consumer.py").read_text().split('if __name__ == "__main__":')[0] + 'if __name__ == "__main__":\n import os,sys\n sys.exit(1 if os.environ.get("FIXTURE_PUBLIC_FAILURE") else 0)\n')
        self.write("scripts/release-modules.txt", ".\nadapters/test\n")
        self.write("go.mod", "module example.invalid/ragy\n\ngo 1.26.1\n")
        self.write("adapters/test/go.mod", "module example.invalid/ragy/adapters/test\n\ngo 1.26.1\n"
                   "\nrequire example.invalid/ragy v0.0.0\n\nreplace example.invalid/ragy => ../..\n")
        self.write("examples/demo/go.mod", "module example.invalid/ragy/examples/demo\n\ngo 1.26.1\n")
        self.write("reviewed.txt", "reviewed source\n")
        self.git("add", ".")
        self.git("commit", "--quiet", "-m", "reviewed source")
        self.source = self.git("rev-parse", "HEAD")

    def write(self, path, text):
        destination = self.repo / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.repo, text=True).strip()

    def remote_git(self, *args):
        return subprocess.check_output(["git", "--git-dir=" + str(self.remote), *args], text=True).strip()

    def invoke(self, source=None):
        return subprocess.run(["bash", str(self.repo / "scripts/release.sh"), "patch", source or self.source],
                              cwd=self.repo, input="y\n", text=True, capture_output=True, timeout=180,
                              env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})

    def snapshot(self):
        return (self.git("rev-parse", "HEAD"), self.git("symbolic-ref", "HEAD"),
                self.git("status", "--porcelain"), self.git("ls-files", "--stage"),
                {str(p.relative_to(self.repo)): p.read_bytes() for p in self.repo.rglob("*")
                 if p.is_file() and ".git" not in p.relative_to(self.repo).parts})


class ReleaseIsolation(unittest.TestCase):
    def test_exact_source_files_and_refs_leave_caller_intact(self):
        # Arrange: reviewed source predates a clean unrelated commit/tag; private files are untracked.
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            fixture.write("reviewed.txt", "unreviewed later source\n")
            fixture.write("later.txt", "unreviewed tracked payload\n")
            fixture.git("add", "reviewed.txt", "later.txt")
            fixture.git("commit", "--quiet", "-m", "unreviewed later commit")
            fixture.git("tag", "scratch-local")
            fixture.write("private-untracked.txt", "private synthetic payload\n")
            fixture.write("adapters/test/private.txt", "nested private synthetic payload\n")
            before = fixture.snapshot()
            # Act: execute the actual release implementation, including real Go manifest editing and atomic Git push.
            completed = fixture.invoke()
            # Assert: exact refs/ancestry/tree/manifests, unchanged caller branch/index/files/tags.
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertEqual(fixture.snapshot(), before)
            self.assertEqual(fixture.git("tag", "--list"), "scratch-local")
            self.assertEqual(fixture.remote_git("tag", "--list").splitlines(),
                             ["adapters/test/v0.0.1", "v0.0.1"])
            candidate = fixture.remote_git("rev-parse", "v0.0.1")
            self.assertEqual(fixture.remote_git("rev-parse", "adapters/test/v0.0.1"), candidate)
            self.assertEqual(fixture.remote_git("rev-parse", candidate + "^"), fixture.source)
            self.assertEqual(fixture.remote_git("diff", "--name-only", fixture.source, candidate),
                             "adapters/test/go.mod")
            files = fixture.remote_git("ls-tree", "-r", "--name-only", candidate).splitlines()
            self.assertNotIn("private-untracked.txt", files)
            self.assertNotIn("adapters/test/private.txt", files)
            self.assertNotIn("later.txt", files)
            self.assertEqual(fixture.remote_git("show", candidate + ":reviewed.txt"), "reviewed source")
            manifest = fixture.remote_git("show", candidate + ":adapters/test/go.mod")
            self.assertIn("require example.invalid/ragy v0.0.1", manifest)
            self.assertNotIn("replace", manifest)
            self.assertEqual(fixture.remote_git("show", candidate + ":examples/demo/go.mod"),
                             (fixture.repo / "examples/demo/go.mod").read_text().strip())

    def test_required_gate_failure_creates_no_refs(self):
        # Arrange: committed required-lane failure on selected source, newer caller HEAD is green.
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            fixture.write("gate-failure", "ordinary required lane failure")
            fixture.git("add", "gate-failure")
            fixture.git("commit", "--quiet", "-m", "failing selected source")
            fixture.source = fixture.git("rev-parse", "HEAD")
            fixture.git("rm", "gate-failure")
            fixture.git("commit", "--quiet", "-m", "green caller is irrelevant")
            before = fixture.snapshot()
            # Act: invoke actual isolated release with failing older source.
            result = fixture.invoke()
            # Assert: caller, isolated tags and public bare remote all remain without new refs.
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(fixture.snapshot(), before)
            self.assertEqual(fixture.remote_git("tag", "--list"), "")
            isolated = fixture.repo / ".git/ragy-releases/active/checkout"
            self.assertEqual(subprocess.check_output(["git", "tag", "--list"], cwd=isolated, text=True), "")

    def test_post_gate_candidate_mutation_invalidates_pass(self):
        # Arrange: fixture lane reports PASS but mutates the checked candidate afterwards.
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            fixture.write("gate-mutation", "mutate candidate")
            fixture.git("add", "gate-mutation")
            fixture.git("commit", "--quiet", "-m", "mutation fixture")
            fixture.source = fixture.git("rev-parse", "HEAD")
            before = fixture.snapshot()
            # Act.
            result = fixture.invoke()
            # Assert: false PASS cannot authorize even isolated tags, much less publication.
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("mutated source/candidate", result.stderr)
            self.assertEqual(fixture.remote_git("tag", "--list"), "")
            self.assertEqual(fixture.snapshot(), before)
            isolated = fixture.repo / ".git/ragy-releases/active/checkout"
            self.assertEqual(subprocess.check_output(["git", "tag", "--list"], cwd=isolated, text=True), "")

    def test_dirty_tracked_and_staged_reject_before_mutation(self):
        for staged in (False, True):
            with self.subTest(staged=staged), tempfile.TemporaryDirectory() as directory:
                # Arrange.
                fixture = Fixture(directory)
                fixture.write("reviewed.txt", "dirty tracked source\n")
                if staged:
                    fixture.git("add", "reviewed.txt")
                before = fixture.snapshot()
                # Act.
                completed = fixture.invoke()
                # Assert.
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("Tracked files and index must be clean", completed.stderr)
                self.assertEqual(fixture.snapshot(), before)
                self.assertEqual(fixture.remote_git("tag", "--list"), "")

    def test_local_and_remote_module_tag_collisions_reject_before_mutation(self):
        for remote in (False, True):
            with self.subTest(remote=remote), tempfile.TemporaryDirectory() as directory:
                # Arrange.
                fixture = Fixture(directory)
                fixture.git("tag", "adapters/test/v0.0.1")
                if remote:
                    fixture.git("push", "--quiet", "origin", "refs/tags/adapters/test/v0.0.1")
                    fixture.git("tag", "-d", "adapters/test/v0.0.1")
                before = fixture.snapshot()
                remote_before = fixture.remote_git("show-ref") if remote else ""
                # Act.
                completed = fixture.invoke()
                # Assert.
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("Candidate tag collision", completed.stderr)
                self.assertEqual(fixture.snapshot(), before)
                self.assertEqual(fixture.remote_git("show-ref") if remote else fixture.remote_git("tag", "--list"),
                                 remote_before)

    def test_relative_origin_remains_bound_to_caller_repository(self):
        for name in ("remote.git", "remote:fixture.git"):
            # Arrange: a slash before a colon denotes a local path in Git, not SCP/URL syntax.
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                fixture = Fixture(directory)
                destination = Path(directory) / name
                if fixture.remote != destination:
                    fixture.remote.rename(destination)
                    fixture.remote = destination
                fixture.git("remote", "set-url", "origin", "../" + name)
                before = fixture.snapshot()
                # Act.
                completed = fixture.invoke()
                # Assert: isolated cwd must not redirect a valid caller-relative destination.
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                self.assertEqual(fixture.snapshot(), before)
                self.assertEqual(fixture.remote_git("tag", "--list").splitlines(),
                                 ["adapters/test/v0.0.1", "v0.0.1"])

    def test_repository_environment_cannot_redirect_isolated_writes(self):
        for variable in ("GIT_DIR", "GIT_INDEX_FILE", "GIT_CONFIG"):
            with self.subTest(variable=variable), tempfile.TemporaryDirectory() as directory:
                # Arrange: inherited repository selectors must not reach isolated init/config/staging.
                fixture = Fixture(directory)
                before = fixture.snapshot()
                target = str(fixture.repo / ".git") if variable == "GIT_DIR" else str(Path(directory) / "private")
                # Act.
                completed = subprocess.run(
                    ["bash", str(fixture.repo / "scripts/release.sh"), "patch", fixture.source],
                    cwd=fixture.repo, input="y\n", text=True, capture_output=True, timeout=180,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", variable: target},
                )
                # Assert.
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("Repository-selection Git environment", completed.stderr)
                self.assertEqual(fixture.snapshot(), before)
                self.assertEqual(fixture.remote_git("tag", "--list"), "")
                if variable != "GIT_DIR":
                    self.assertFalse(Path(target).exists())

    def test_global_hook_cannot_expand_candidate_index(self):
        # Arrange: a global checkout hook would stage synthetic private data into any clone.
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            hooks = Path(directory) / "external-hooks"
            hooks.mkdir()
            hook = hooks / "post-checkout"
            hook.write_text("#!/bin/sh\nprintf private > hook-private.txt\ngit add hook-private.txt\n")
            hook.chmod(0o755)
            config = Path(directory) / "global-config"
            config.write_text('[core]\n\thooksPath = ' + str(hooks) + '\n')
            before = fixture.snapshot()
            # Act.
            completed = subprocess.run(
                ["bash", str(fixture.repo / "scripts/release.sh"), "patch", fixture.source],
                cwd=fixture.repo, input="y\n", text=True, capture_output=True, timeout=180,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "GIT_CONFIG_GLOBAL": str(config)},
            )
            # Assert.
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertEqual(fixture.snapshot(), before)
            files = fixture.remote_git("ls-tree", "-r", "--name-only", "v0.0.1").splitlines()
            self.assertNotIn("hook-private.txt", files)
            self.assertEqual(fixture.remote_git("diff", "--name-only", fixture.source, "v0.0.1"),
                             "adapters/test/go.mod")

    def test_rejected_push_preserves_caller_and_unrelated_refs(self):
        # Arrange: an actual bare-remote hook rejects the complete atomic transaction.
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            fixture.git("tag", "scratch-local")
            fixture.write("private-untracked.txt", "private synthetic payload\n")
            hook = fixture.remote / "hooks/pre-receive"
            hook.write_text("#!/bin/sh\nexit 1\n")
            hook.chmod(0o755)
            before = fixture.snapshot()
            # Act.
            completed = fixture.invoke()
            # Assert: the caller remains on its branch with identical files and tags.
            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(fixture.snapshot(), before)
            self.assertEqual(fixture.git("tag", "--list"), "scratch-local")
            self.assertEqual(fixture.remote_git("tag", "--list"), "")

    def test_untracked_staged_file_rejects_before_mutation(self):
        # Arrange: staging turns an unrelated new file into index dirtiness.
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            fixture.write("private-staged.txt", "private synthetic payload\n")
            fixture.git("add", "private-staged.txt")
            before = fixture.snapshot()
            # Act.
            completed = fixture.invoke()
            # Assert.
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Tracked files and index must be clean", completed.stderr)
            self.assertEqual(fixture.snapshot(), before)
            self.assertEqual(fixture.remote_git("tag", "--list"), "")

    def test_publishable_manifest_cannot_include_examples(self):
        # Arrange.
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            fixture.write("scripts/release-modules.txt", ".\nexamples/demo\n")
            fixture.git("add", "scripts/release-modules.txt")
            fixture.git("commit", "--quiet", "-m", "invalid release scope")
            source = fixture.git("rev-parse", "HEAD")
            before = fixture.snapshot()
            # Act.
            completed = fixture.invoke(source)
            # Assert.
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Invalid publishable module", completed.stderr)
            self.assertEqual(fixture.snapshot(), before)
            self.assertEqual(fixture.remote_git("tag", "--list"), "")


if __name__ == "__main__":
    unittest.main()
