//go:build integration || e2e

package tooling_test

import (
	"encoding/json"
	"io/fs"
	"os"
	"path/filepath"
	"testing"
)

func copyTree(t *testing.T, from, to string) {
	t.Helper()
	if err := filepath.WalkDir(from, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			if entry.Name() == ".git" {
				return filepath.SkipDir
			}
			return nil
		}
		if entry.Type()&os.ModeSymlink != 0 {
			t.Fatalf("unexpected symlink %s", path)
		}
		relative, err := filepath.Rel(from, path)
		if err != nil {
			return err
		}
		write(t, filepath.Join(to, relative), read(t, path))
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

func contextConsumer(t *testing.T, root, mode, version string, env []string) {
	t.Helper()
	dir := filepath.Join(t.TempDir(), "consumer")
	copyTree(t, filepath.Join(root, "examples/context-bridge"), dir)
	for _, name := range []string{"ragy", "memy", "contexty"} {
		command(t, dir, env, "go", "mod", "edit", "-dropreplace=github.com/skosovsky/"+name)
	}
	if mode == "checkout" {
		command(t, dir, env, "go", "mod", "edit", "-replace=github.com/skosovsky/ragy="+root)
		for _, peer := range []struct{ name, sha string }{
			{"memy", "9719bc7967031c56daba9ef302edc4994ec8554a"},
			{"contexty", "912c0413994b2b3a1a7d3849a0aa09c71350f915"},
		} {
			checkout := filepath.Join(t.TempDir(), peer.name)
			command(t, filepath.Dir(checkout), nil, "git", "init", "--quiet", checkout)
			command(
				t,
				checkout,
				nil,
				"git",
				"fetch",
				"--quiet",
				"--depth=1",
				"https://github.com/skosovsky/"+peer.name+".git",
				peer.sha,
			)
			command(t, checkout, nil, "git", "checkout", "--quiet", "--detach", peer.sha)
			command(t, dir, env, "go", "mod", "edit", "-replace=github.com/skosovsky/"+peer.name+"="+checkout)
		}
	} else if version != "" {
		command(t, dir, env, "go", "mod", "edit", "-require=github.com/skosovsky/ragy@"+version)
	}
	command(t, dir, env, "go", "mod", "tidy")
	var manifest struct {
		Replace []any `json:"Replace"`
	}
	if err := json.Unmarshal([]byte(command(t, dir, env, "go", "mod", "edit", "-json")), &manifest); err != nil {
		t.Fatal(err)
	}
	if mode == "published" && len(manifest.Replace) > 0 {
		t.Fatal("published consumer has replacements")
	}
	command(t, dir, env, "go", "test", "-race", "-count=1", "./...")
	command(t, dir, env, "go", "run", "./cmd/demo")
}

func TestE2EContextBridgeCheckout(t *testing.T) {
	// Arrange: explicit peer revisions, independent of ambient sibling checkouts.
	root := repoRoot(t)
	// Act / Assert: semantic fixtures and real executable demo against the current source.
	contextConsumer(t, root, "checkout", "", nil)
}
