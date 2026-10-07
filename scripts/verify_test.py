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
            (root / "scripts/check-registry.json").write_text('{"inventories":{"test":"scripts/check-modules.txt","release":"scripts/release-modules.txt"}}')
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
            argv = ["verify.py", "test-fast"] + (["--fresh"] if fresh else [])
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

    def test_gate_accumulates_independent_failures_and_blocks_dependents(self):
        # Arrange: all four mandatory failure classes plus a dependent lane.
        plan = [{"id": name} for name in ("ordinary", "linter", "script", "consumer")]
        plan.append({"id": "dependent", "needs": ["consumer"]})
        calls = []
        def execute(row):
            calls.append(row["id"])
            raise ValueError(row["id"] + " injected failure")
        # Act.
        results, success = verify.execute_plan(plan, execute)
        # Assert: every independent failure is visible; no successful aggregate.
        self.assertFalse(success)
        self.assertEqual(calls, ["ordinary", "linter", "script", "consumer"])
        self.assertEqual([row["status"] for row in results], ["FAIL"] * 4 + ["BLOCKED"])

    def test_missing_required_prerequisites_never_pass(self):
        # Arrange: mandatory missing Docker, peer, toolchain cases.
        for prerequisite in ("Docker", "exact peer ref", "pinned toolchain"):
            def execute(row):
                raise FileNotFoundError(prerequisite)
            # Act.
            results, success = verify.execute_plan([{"id": prerequisite}], execute)
            # Assert.
            self.assertFalse(success)
            self.assertEqual(results[0]["status"], "BLOCKED")
            self.assertIn(prerequisite, results[0]["reason"])

    def test_required_skipped_and_empty_integration_fail(self):
        # Arrange: Go can exit zero for no selected tests or skipped prerequisites.
        import json
        package = json.dumps({"Action": "pass", "Package": "fixture"})
        skip = json.dumps({"Action": "skip", "Package": "fixture", "Test": "TestRequired"})
        # Act / Assert: neither zero-exit shape establishes required execution.
        with self.assertRaisesRegex(ValueError, "skipped"):
            verify.validate_test_events(skip + "\n" + package, reject_skips=True, require_tests=True)
        with self.assertRaisesRegex(ValueError, "no tests"):
            verify.validate_test_events(package, reject_skips=True, require_tests=True)

    def test_optional_skips_classified_without_waiving_required_skips(self):
        # Arrange: an explicit optional profile next to real required evidence.
        plan = [{"id": "required"}, {"id": "paid-live", "optional": True, "reason": "paid opt-in"}]
        # Act.
        rows, success = verify.execute_plan(plan, lambda row: {})
        # Assert: optional absence has a reason; required execution controls success.
        self.assertTrue(success)
        self.assertEqual(rows[1]["status"], "SKIP")
        self.assertFalse(rows[1]["required"])
        self.assertEqual(rows[1]["reason"], "paid opt-in")

    def test_registry_covers_all_script_tests_and_linux(self):
        # Arrange: read real registry and actual script inventory, not an implementation mock.
        plan = verify.registry_plan("check", verify.modules())
        # Act.
        scripts = {row["command"][1] for row in plan if row["id"].startswith("script:")}
        # Assert: adding a runner test requires registration; Linux is mandatory.
        self.assertEqual(scripts, {str(p.relative_to(verify.ROOT)) for p in (verify.ROOT / "scripts").glob("*_test.py")})
        self.assertTrue(any(row["id"] == "linux" and not row.get("optional") for row in plan))
        self.assertEqual([row["id"] for row in verify.registry_plan("test", verify.modules()) if row["id"].startswith("tests:")],
                         ["tests:" + module for module in verify.modules()])


    def test_real_full_runner_collects_command_failures_and_exits_nonzero(self):
        # Arrange: isolated committed repository, actual failing child processes.
        import json
        import os
        import shutil
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            root.mkdir()
            (root / "scripts").mkdir()
            for name in ("verify.py", "process_runner.py", "toolchain.json"):
                shutil.copy2(verify.ROOT / "scripts" / name, root / "scripts" / name)
            (root / "go.mod").write_text("module example.invalid/fixture\n")
            for name in ("check-modules.txt", "release-modules.txt"):
                (root / "scripts" / name).write_text(".\n")
            (root / "script.py").write_text("raise SystemExit(5)\n")
            (root / "consumer.py").write_text("raise SystemExit(6)\n")
            registry = {"inventories": {"test": "scripts/check-modules.txt", "release": "scripts/release-modules.txt"},
                        "toolchain": "scripts/toolchain.json", "peers": {}, "lanes": [
                {"id": "toolchain", "profiles": ["check"], "action": "toolchain"},
                {"id": "ordinary", "profiles": ["check"], "command": ["{go}", "test", "./..."]},
                {"id": "linter", "profiles": ["check"], "command": ["{lint}", "run", "./..."]},
                {"id": "script", "profiles": ["check"], "command": ["{python}", "script.py"]},
                {"id": "consumer", "profiles": ["check"], "command": ["{python}", "consumer.py"]},
                {"id": "dependent", "profiles": ["check"], "command": ["{python}", "consumer.py"], "needs": ["consumer"]}]}
            (root / "scripts/check-registry.json").write_text(json.dumps(registry))
            pins = json.loads((root / "scripts/toolchain.json").read_text())
            binaries = Path(temporary) / "bin"
            binaries.mkdir()
            go = binaries / "go"
            lint = binaries / "lint"
            go.write_text("#!" + sys.executable + "\nimport sys\nprint('go version go" + pins["go"] + " linux/amd64')\nraise SystemExit(0 if sys.argv[1]=='version' else 3)\n")
            lint.write_text("#!" + sys.executable + "\nimport sys\nprint('golangci-lint has version " + pins["golangci_lint"] + "')\nraise SystemExit(0 if sys.argv[1]=='version' else 4)\n")
            go.chmod(0o755)
            lint.chmod(0o755)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "-c", "user.name=fixture", "-c", "user.email=f@example.invalid", "-c", "commit.gpgsign=false", "commit", "-qm", "fixture"], check=True)
            output = Path(temporary) / "report"
            env = dict(os.environ, GO=str(go), GOLANGCI_LINT=str(lint), PYTHONDONTWRITEBYTECODE="1")
            # Act: run the real CLI and actual executable/script child failures.
            completed = subprocess.run([sys.executable, str(root / "scripts/verify.py"), "check", "--version", "v0.0.1", "--output", str(output)], cwd=root, env=env, capture_output=True, text=True)
            report = json.loads((output / "summary.json").read_text())
            # Assert: nonzero exit, all four commands attempted, dependent blocked.
            self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            rows = {row["id"]: row for row in report["results"]}
            for name in ("ordinary", "linter", "script", "consumer"):
                self.assertEqual(rows[name]["status"], "FAIL")
                self.assertEqual(len(rows[name]["commands"]), 1)
            self.assertEqual(rows["dependent"]["status"], "BLOCKED")
            self.assertEqual(report["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
