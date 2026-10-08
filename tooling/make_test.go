//go:build !integration

package tooling_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func makeFixture(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	for _, path := range []string{"Makefile", "scripts/toolchain.mk"} {
		write(t, filepath.Join(root, path), read(t, filepath.Join(repoRoot(t), path)))
	}
	write(t, filepath.Join(root, "go.mod"), []byte("module example.invalid/root\n"))
	return root
}

func TestMakeDiscoversModules(t *testing.T) {
	// Arrange: new modules need no registry or Git staging; hidden/vendor trees are excluded.
	root := makeFixture(t)
	for _, dir := range []string{"adapters/new", "examples/new", ".cache/ignored", "vendor/ignored", "adapters/new/vendor/ignored"} {
		write(t, filepath.Join(root, dir, "go.mod"), []byte("module example.invalid/fixture\n"))
	}
	// Act.
	actual := command(t, root, nil, "make", "--no-print-directory", "-s", "modules")
	// Assert.
	if actual != ".\nadapters/new\nexamples/new" {
		t.Fatalf("discovered modules: %q", actual)
	}
}

func TestMakePropagatesToolFailures(t *testing.T) {
	for _, target := range []string{"test", "lint"} {
		t.Run(target, func(t *testing.T) {
			// Arrange: use the actual Make recipes with a failing tool.
			root := makeFixture(t)
			cmd := exec.CommandContext(t.Context(), "make", target, "GO=false", "GOLANGCI_LINT=false")
			cmd.Dir = root
			// Act.
			out, err := cmd.CombinedOutput()
			// Assert.
			if err == nil {
				t.Fatalf("%s swallowed failure: %s", target, out)
			}
		})
	}
}

func TestMakeLintRejectsFormattingDiff(t *testing.T) {
	// Arrange: the pinned formatter returns a nonzero status when formatting differs.
	root := makeFixture(t)
	tool := filepath.Join(root, "lint")
	write(t, tool, []byte("#!/bin/sh\nif [ \"$1\" = fmt ]; then echo 'formatting differs'; exit 1; fi\n"))
	if err := os.Chmod(tool, 0700); err != nil {
		t.Fatal(err)
	}
	cmd := exec.CommandContext(t.Context(), "make", "lint", "GOLANGCI_LINT="+tool)
	cmd.Dir = root
	// Act.
	out, err := cmd.CombinedOutput()
	// Assert.
	if err == nil || !strings.Contains(string(out), "formatting differs") {
		t.Fatalf("format gate: %v %s", err, out)
	}
}

func TestFuzzSelectsIndividualGoTargets(t *testing.T) {
	// Arrange: Go list output includes two fuzz targets, Unicode and tool diagnostics.
	root := t.TempDir()
	write(t, filepath.Join(root, "scripts/fuzz.sh"), read(t, filepath.Join(repoRoot(t), "scripts/fuzz.sh")))
	tool := filepath.Join(root, "go")
	log := filepath.Join(root, "calls")
	write(t, tool, []byte(`#!/bin/sh
case "$1" in
list) echo example.invalid/fuzz ;;
test) case " $* " in
*' -list='*) printf 'FuzzOne\nFuzzТекст\nok example.invalid/fuzz\n' ;;
*) printf '%s\n' "$*" >> "$CALLS" ;;
esac ;;
esac
`))
	if err := os.Chmod(tool, 0700); err != nil {
		t.Fatal(err)
	}
	// Act: durations above five minutes are valid; the fake tool performs no campaign.
	command(t, root, []string{"GO=" + tool, "CALLS=" + log}, "bash", "scripts/fuzz.sh", "600", ".")
	// Assert: one anchored target per invocation, no broad pattern or diagnostic line.
	calls := string(read(t, log))
	if strings.Count(calls, "-fuzz=") != 2 || !strings.Contains(calls, "-fuzz=^FuzzOne$") ||
		!strings.Contains(calls, "-fuzz=^FuzzТекст$") || strings.Count(calls, "-fuzztime=600s") != 2 {
		t.Fatalf("incorrect fuzz selection: %s", calls)
	}
}

func TestCommonMakeCheckOrder(t *testing.T) {
	for _, multi := range []bool{false, true} {
		t.Run(map[bool]string{false: "single", true: "multiple"}[multi], func(t *testing.T) {
			// Arrange: a foreign library uses only the common Makefile and project hooks.
			root := makeFixture(t)
			if multi {
				write(t, filepath.Join(root, "packages/extra/go.mod"), []byte("module example.invalid/extra\n"))
			}
			log := filepath.Join(root, "calls")
			tool := filepath.Join(root, "tool")
			write(t, tool, []byte(`#!/bin/bash
case "$1" in
version) if [[ "$0" == *lint ]]; then echo 'golangci-lint version 2.14.0 '; else echo 'go version go1.27.1 test/arch'; fi ;;
*) printf '%s\n' "$1" >> "$CALLS" ;;
esac
`))
			if err := os.Chmod(tool, 0700); err != nil {
				t.Fatal(err)
			}
			lint := filepath.Join(root, "lint")
			if err := os.Symlink(tool, lint); err != nil {
				t.Fatal(err)
			}
			write(t, filepath.Join(root, "project.mk"), []byte(`prerequisites-project:
	@echo prerequisites >> "$$CALLS"
examples-project:
	@echo examples >> "$$CALLS"
check-project:
	@echo integration >> "$$CALLS"
`))
			// Act: parallel Make must still run check stages sequentially.
			command(t, root, []string{"CALLS=" + log}, "make", "-j8", "check", "GO="+tool, "GOLANGCI_LINT="+lint)
			// Assert.
			expected := "prerequisites\nconfig\nfmt\nrun\ntest\nexamples\nintegration\n"
			if multi {
				expected = "prerequisites\nconfig\nfmt\nrun\nfmt\nrun\ntest\ntest\nexamples\nintegration\n"
			}
			if got := string(read(t, log)); got != expected {
				t.Fatalf("stage order: %q, want %q", got, expected)
			}
		})
	}
}

func TestCommonMakeOptionalAndFailingProjectHooks(t *testing.T) {
	// Arrange: common targets work without a project file.
	root := makeFixture(t)
	// Act / Assert.
	command(t, root, nil, "make", "examples", "test-integration")
	for _, target := range []string{"examples", "test-integration"} {
		t.Run(target, func(t *testing.T) {
			// Arrange: failure in a project hook must propagate.
			write(t, filepath.Join(root, "project.mk"), []byte("examples-project check-project:\n\t@false\n"))
			cmd := exec.CommandContext(t.Context(), "make", target)
			cmd.Dir = root
			// Act.
			out, err := cmd.CombinedOutput()
			// Assert.
			if err == nil {
				t.Fatalf("project failure swallowed: %s", out)
			}
		})
	}
}
