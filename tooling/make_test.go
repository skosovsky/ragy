package tooling_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestMakePropagatesToolFailures(t *testing.T) {
	for _, target := range []string{"test", "examples", "lint"} {
		t.Run(target, func(t *testing.T) {
			// Arrange: use the actual Make recipes with a failing tool.
			root := t.TempDir()
			write(t, filepath.Join(root, "Makefile"), read(t, filepath.Join(repoRoot(t), "Makefile")))
			write(
				t,
				filepath.Join(root, "scripts/toolchain.mk"),
				read(t, filepath.Join(repoRoot(t), "scripts/toolchain.mk")),
			)
			write(t, filepath.Join(root, "scripts/check-modules.txt"), []byte(".\n"))
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
	// Arrange: formatting command exits successfully but reports required changes.
	root := t.TempDir()
	write(t, filepath.Join(root, "Makefile"), read(t, filepath.Join(repoRoot(t), "Makefile")))
	write(t, filepath.Join(root, "scripts/toolchain.mk"), read(t, filepath.Join(repoRoot(t), "scripts/toolchain.mk")))
	write(t, filepath.Join(root, "scripts/check-modules.txt"), []byte(".\n"))
	tool := filepath.Join(root, "lint")
	write(t, tool, []byte("#!/bin/sh\nif [ \"$1\" = fmt ]; then echo 'formatting differs'; fi\n"))
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
	write(t, filepath.Join(root, "scripts/check-modules.txt"), []byte(".\n"))
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
	// Act.
	command(t, root, []string{"GO=" + tool, "CALLS=" + log}, "bash", "scripts/fuzz.sh", "1")
	// Assert: one anchored target per invocation, no broad pattern or diagnostic line.
	calls := string(read(t, log))
	if strings.Count(calls, "-fuzz=") != 2 || !strings.Contains(calls, "-fuzz=^FuzzOne$") ||
		!strings.Contains(calls, "-fuzz=^FuzzТекст$") {
		t.Fatalf("incorrect fuzz selection: %s", calls)
	}
}
