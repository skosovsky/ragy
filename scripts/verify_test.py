#!/usr/bin/env python3
"""AAA command-dispatch and inventory regression fixtures."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import verify


class VerificationContract(unittest.TestCase):
    def test_inventory_rejects_omitted_and_duplicate_modules(self):
        # Arrange: two actual modules, only one inventoried.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "scripts").mkdir()
            (root / "go.mod").write_text("module example.invalid/root\n")
            (root / "nested").mkdir()
            (root / "nested/go.mod").write_text("module example.invalid/nested\n")
            (root / "scripts/release-modules.txt").write_text(".\n")
            manifest = root / "scripts/check-modules.txt"
            manifest.write_text(".\n")
            # Act / Assert: drift and duplicate selections never dispatch.
            with self.assertRaisesRegex(ValueError, "inventory drift"):
                verify.modules(root)
            manifest.write_text(".\nnested\n")
            self.assertEqual(verify.modules(root), [".", "nested"])
            manifest.write_text(".\nnested\nnested\n")
            with self.assertRaisesRegex(ValueError, "unique"):
                verify.modules(root)

    def test_multiple_fuzz_functions_are_individual_and_anchored(self):
        # Arrange: two fuzz names in one package, noisy non-fuzz listing lines.
        calls = []
        def command(argv, directory, **options):
            calls.append((argv, options))
            if argv[1] == "list":
                return "example.invalid/pkg\n"
            if "-list=^Fuzz" in argv:
                return "FuzzOne\nFuzzTwo\nTestIgnored\nok example.invalid/pkg\n"
            return ""
        # Act.
        verify.fuzz_module(Path("/tmp/fixture"), "go", 2, command)
        fuzz = [(argv, options) for argv, options in calls if any(a.startswith("-fuzz=") for a in argv)]
        # Assert: no broad selector, bounded execution per distinct name.
        self.assertEqual([next(a for a in argv if a.startswith("-fuzz=")) for argv, _ in fuzz],
                         ["-fuzz=^FuzzOne$", "-fuzz=^FuzzTwo$"])
        self.assertEqual(len(fuzz), 2)
        for argv, options in fuzz:
            self.assertIn("-run=^$", argv)
            self.assertIn("-fuzztime=2s", argv)
            self.assertEqual(options["timeout"], 62)

    def test_fuzz_names_cover_unicode_and_bare_prefix(self):
        # Arrange: Go identifiers may be Unicode; bare Fuzz is also a valid prefix name.
        output = "Fuzz\nFuzzКоординаты\nFuzzASCII\nTestIgnored\nok example.invalid/pkg\n"
        # Act / Assert: preserve every listed fuzz identifier, no status noise.
        self.assertEqual(verify.fuzz_names(output), ["Fuzz", "FuzzКоординаты", "FuzzASCII"])

    def test_actual_go_unicode_letter_is_not_filtered_by_python_identifiers(self):
        # Arrange: U+037A is a valid Go Unicode letter but not a Python identifier suffix.
        import subprocess
        import os
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "go.mod").write_text("module example.invalid/fuzzletters\n\ngo 1.26.1\n")
            (root / "fuzz_test.go").write_text('package fuzzletters\nimport "testing"\nfunc Fuzz\u037a(f *testing.F){f.Add("seed");f.Fuzz(func(t *testing.T,s string){})}\n')
            env = {**os.environ, "GOWORK": "off"}
            # Act: obtain an actual Go-generated list, then dispatch the same campaign.
            output = subprocess.check_output([os.environ.get("GO", "go"), "test", "-list=^Fuzz", "./..."], cwd=root, env=env, text=True)
            self.assertIn("Fuzz\u037a", verify.fuzz_names(output))
            with contextlib.redirect_stdout(io.StringIO()):
                verify.fuzz_module(root, os.environ.get("GO", "go"), 1)
            # Assert: no silent successful omission of the Go-listed fuzz identifier.
            self.assertEqual(verify.fuzz_names(output), ["Fuzz\u037a"])

    def test_fuzz_listing_failure_propagates(self):
        # Arrange: listing cannot establish which functions exist.
        command = mock.Mock(side_effect=ValueError("listing failed"))
        # Act / Assert: no silent successful fuzz run.
        with self.assertRaisesRegex(ValueError, "listing failed"):
            verify.fuzz_module(Path("/tmp/fixture"), "go", 1, command)
        self.assertEqual(command.call_count, 1)

    def test_fresh_and_cached_dispatch_once_per_module(self):
        # Arrange: a root plus development example, no release behavior.
        inventory = [".", "examples/demo"]
        for fresh in (False, True):
            argv = ["verify.py", "test"] + (["--fresh"] if fresh else [])
            calls = []
            with mock.patch.object(verify, "modules", return_value=inventory), \
                 mock.patch.object(verify, "run", side_effect=lambda command, directory, **kw: calls.append((command, directory))), \
                 mock.patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                # Act.
                self.assertEqual(verify.main(), 0)
            # Assert: both modules once, explicit race/fresh distinction, no repeated examples test.
            self.assertEqual(len(calls), 2)
            self.assertEqual([d for _, d in calls], [verify.ROOT, verify.ROOT / "examples/demo"])
            for command, _ in calls:
                self.assertIn("-race", command)
                self.assertEqual("-count=1" in command, fresh)

    def test_acceptance_rejects_toolchain_suffix_or_mismatch(self):
        # Arrange: same-prefix compiler/linter is not the exact validated release.
        import json
        pins = json.loads((verify.ROOT / "scripts/toolchain.json").read_text())
        for go, lint in (("go version go"+pins["go"]+".custom darwin/arm64", "golangci-lint has version "+pins["golangci_lint"]),
                         ("go version go"+pins["go"]+" darwin/arm64", "golangci-lint has version "+pins["golangci_lint"]+"-custom")):
            with mock.patch.object(verify, "run", side_effect=[go, lint]), contextlib.redirect_stdout(io.StringIO()):
                # Act / Assert: no fresh acceptance on unvalidated tool identity.
                with self.assertRaisesRegex(ValueError, "recorded validated toolchain"):
                    verify.versions("go", "lint", enforce=True)

    def test_zero_and_excessive_fuzz_budget_rejected(self):
        for value in ("0", "301"):
            # Arrange / Act / Assert: invalid budgets reject before command dispatch.
            with mock.patch.object(verify, "modules", return_value=["."]), \
                 mock.patch.object(verify, "run") as command, \
                 mock.patch("sys.argv", ["verify.py", "fuzz", "--fuzz-seconds", value]):
                with self.assertRaisesRegex(ValueError, "between 1 and 300"):
                    verify.main()
                command.assert_not_called()

    def test_actual_subprocess_timeout_is_bounded(self):
        # Arrange: a cooperative process exceeds the runner's hard process budget.
        import subprocess
        import sys
        import time
        started = time.monotonic()
        with contextlib.redirect_stdout(io.StringIO()):
            # Act / Assert: timed-out child is terminated and failure propagates.
            with self.assertRaises(subprocess.TimeoutExpired):
                verify.run([sys.executable, "-c", "import time; time.sleep(30)"], verify.ROOT, timeout=0.05)
        self.assertLess(time.monotonic() - started, 5)

    def test_subprocess_forces_off_workspace_and_preserves_nonzero(self):
        # Arrange: actual child checks environment; host supplied ambient workspace is hostile.
        import subprocess
        import sys
        with mock.patch.dict("os.environ", {"GOWORK": "/tmp/ambient.go.work"}), contextlib.redirect_stdout(io.StringIO()):
            # Act.
            output = verify.run([sys.executable, "-c", "import os; print(os.environ['GOWORK'])"], verify.ROOT, capture=True)
            # Assert.
            self.assertEqual(output.strip(), "off")
            with self.assertRaises(subprocess.CalledProcessError):
                verify.run([sys.executable, "-c", "raise SystemExit(7)"], verify.ROOT)


if __name__ == "__main__":
    unittest.main()
