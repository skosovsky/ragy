#!/usr/bin/env python3
"""AAA clean-consumer fixture regressions, including actual Go selection."""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

import check_release_consumer as consumer


class ConsumerProfile(unittest.TestCase):
    def test_diagnostic_stderr_is_not_inventory_stdout(self):
        # Arrange: Go diagnostics can accompany a valid package inventory.
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            # Act: use an actual subprocess with independent streams.
            output = consumer.command(Path(directory), sys.executable, "-c",
                                      "import sys; print('example.invalid/pkg'); print('go: downloading dependency', file=sys.stderr)")
        # Assert: only machine stdout is parsed; diagnostics stay in the command log.
        self.assertEqual(output, "example.invalid/pkg")

    def test_actual_go_selection_excludes_main_and_test_only_packages(self):
        # Arrange: production library, integration-test-only dir and command dir.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "go.mod").write_text("module example.invalid/selection\n\ngo 1.26.1\n")
            (root / "library.go").write_text("package selection\n")
            (root / "integration").mkdir()
            (root / "integration/only_test.go").write_text("package integration\n")
            (root / "command").mkdir()
            (root / "command/main.go").write_text("package main\nfunc main(){}\n")
            env = {**os.environ, "GOWORK": "off", "GOENV": "off"}
            # Act: the same actual Go template used by the clean consumer gate.
            output = subprocess.check_output([os.environ.get("GO", "go"), "list", "-f",
                                              consumer.PUBLIC_PACKAGE_TEMPLATE, "./..."], cwd=root, env=env, text=True)
        # Assert: importer must not invent imports for test-only/main packages.
        self.assertEqual(output.split(), ["example.invalid/selection"])

    def test_proxy_root_excludes_nested_modules_and_untracked_payload(self):
        # Arrange: synthetic tracked root/nested module and untracked secret.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, proxy = root / "repo", root / "proxy"
            repo.mkdir()
            subprocess.run(["git", "init", "--quiet"], cwd=repo, check=True)
            for key, value in (("user.name", "Fixture"), ("user.email", "fixture@example.invalid"), ("commit.gpgsign", "false")):
                subprocess.run(["git", "config", key, value], cwd=repo, check=True)
            (repo / "go.mod").write_text("module example.invalid/root\n\ngo 1.26.1\n")
            (repo / "root.go").write_text("package root\n")
            (repo / "nested").mkdir()
            (repo / "nested/go.mod").write_text("module example.invalid/root/nested\n\ngo 1.26.1\n")
            (repo / "nested/nested.go").write_text("package nested\n")
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(["git", "commit", "--quiet", "-m", "fixture"], cwd=repo, check=True)
            (repo / "untracked-secret.txt").write_text("SECRET")
            files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", "HEAD"], cwd=repo, text=True).splitlines()
            # Act: emit proxy directly from committed source, not caller contents.
            path = consumer.proxy_module(repo, ".", "v0.0.1", files, proxy, os.environ.copy())
            with zipfile.ZipFile(proxy / path / "@v/v0.0.1.zip") as archive:
                names = archive.namelist()
            # Assert: nested module is separate, untracked payload never enters zip.
            self.assertEqual(set(names), {"example.invalid/root@v0.0.1/go.mod", "example.invalid/root@v0.0.1/root.go"})

    def test_irregular_git_entries_reject_artifact_without_ref_mutation(self):
        for mode in ("120000", "160000"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                # Arrange: an actual committed symlink or gitlink in a disposable repository.
                root = Path(directory)
                repo = root / "repo"
                repo.mkdir()
                subprocess.run(["git", "init", "--quiet"], cwd=repo, check=True)
                for key, value in (("user.name", "Fixture"), ("user.email", "fixture@example.invalid"), ("commit.gpgsign", "false")):
                    subprocess.run(["git", "config", key, value], cwd=repo, check=True)
                (repo / "go.mod").write_text("module example.invalid/irregular\n\ngo 1.26.1\n")
                (repo / "regular.go").write_text("package irregular\n")
                subprocess.run(["git", "add", "."], cwd=repo, check=True)
                subprocess.run(["git", "commit", "--quiet", "-m", "regular source"], cwd=repo, check=True)
                name = "linked.go" if mode == "120000" else "embedded-repository"
                if mode == "120000":
                    (repo / name).symlink_to("regular.go")
                    subprocess.run(["git", "add", name], cwd=repo, check=True)
                else:
                    oid = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
                    subprocess.run(["git", "update-index", "--add", "--cacheinfo", "160000,"+oid+","+name], cwd=repo, check=True)
                subprocess.run(["git", "commit", "--quiet", "-m", "irregular source"], cwd=repo, check=True)
                files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", "HEAD"], cwd=repo, text=True).splitlines()
                before = subprocess.check_output(["git", "show-ref"], cwd=repo)
                # Act: same committed-artifact builder used by candidate checksum preparation and gates.
                with self.assertRaisesRegex(ValueError, "Unsupported Git mode "+mode+".*"+name):
                    consumer.proxy_module(repo, ".", "v0.0.1", files, root / "proxy", os.environ.copy())
                # Assert: invalid artifacts and release tags were never created, refs untouched.
                self.assertFalse((root / "proxy").exists())
                self.assertEqual(subprocess.check_output(["git", "show-ref"], cwd=repo), before)
                self.assertEqual(subprocess.check_output(["git", "tag", "--list"], cwd=repo, text=True), "")


if __name__ == "__main__":
    unittest.main()
