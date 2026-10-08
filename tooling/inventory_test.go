//go:build !integration && !e2e

package tooling_test

import (
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestPublishableModuleInventory(t *testing.T) {
	// Arrange: tracked source is the inventory authority, not ignored build artifacts.
	root := repoRoot(t)
	publish := strings.Fields(command(t, root, nil, "make", "--no-print-directory", "-s", "modules"))
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
	wantPublish := found
	// Assert.
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
