package tooling_test

import (
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestModuleInventory(t *testing.T) {
	// Arrange: tracked source is the inventory authority, not ignored build artifacts.
	root := repoRoot(t)
	listed := strings.Fields(string(read(t, filepath.Join(root, "scripts/check-modules.txt"))))
	publish := strings.Fields(string(read(t, filepath.Join(root, "scripts/release-modules.txt"))))
	paths := strings.Split(
		command(
			t,
			root,
			nil,
			"git",
			"ls-files",
			"--cached",
			"--",
			"go.mod",
			"**/go.mod",
		),
		"\n",
	)
	// Act.
	found := make([]string, 0, len(paths))
	for _, path := range paths {
		if path != "" {
			found = append(found, filepath.ToSlash(filepath.Dir(path)))
		}
	}
	sorted := slices.Clone(listed)
	slices.Sort(sorted)
	slices.Sort(found)
	// Assert.
	if len(listed) == 0 || listed[0] != "." || !slices.Equal(sorted, found) {
		t.Fatalf("module inventory: listed %v; actual %v", listed, found)
	}
	for i, module := range sorted {
		if i > 0 && sorted[i-1] == module {
			t.Fatalf("duplicate %s", module)
		}
	}
	for _, module := range publish {
		if !slices.Contains(listed, module) || strings.HasPrefix(module, "examples/") || module == "tooling" {
			t.Fatalf("invalid publishable module %s", module)
		}
	}
	// Every root/adapter module is publishable; examples and tooling are development-only.
	wantPublish := make([]string, 0, len(listed))
	for _, module := range listed {
		if module == "." || strings.HasPrefix(module, "adapters/") {
			wantPublish = append(wantPublish, module)
		}
	}
	slices.Sort(wantPublish)
	slices.Sort(publish)
	if !slices.Equal(publish, wantPublish) {
		t.Fatalf("publishable inventory: listed %v; expected %v", publish, wantPublish)
	}
}

func TestGoWorkspaceIsolated(t *testing.T) {
	// Arrange: an ambient workspace must not decide module resolution for tooling.
	t.Setenv("GOWORK", "/nonexistent/ambient.go.work")
	// Act.
	actual := command(t, t.TempDir(), []string{"GOENV=off"}, "go", "env", "GOWORK")
	// Assert.
	if actual != "off" {
		t.Fatalf("ambient workspace leaked: %s", actual)
	}
}
