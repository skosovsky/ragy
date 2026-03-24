package ragy_test

import (
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestProductionDocsAvoidMigrationLanguage(t *testing.T) {
	t.Parallel()

	paths, err := productionDocPaths(".")
	if err != nil {
		t.Fatalf("productionDocPaths(): %v", err)
	}
	if len(paths) == 0 {
		t.Fatal("productionDocPaths() returned no files")
	}

	banned := []string{
		"legacy",
		"deprecated",
		"roadmap",
		"future release",
		"instead of",
		"migration (clean break)",
	}

	for _, path := range paths {
		content, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("ReadFile(%s): %v", path, err)
		}
		if isGeneratedDoc(content) {
			continue
		}

		lower := strings.ToLower(string(content))
		for _, token := range banned {
			if strings.Contains(lower, token) {
				t.Fatalf("%s contains banned migration wording %q", path, token)
			}
		}
	}
}

func productionDocPaths(root string) ([]string, error) {
	paths := make([]string, 0)
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if shouldSkipDocDir(d.Name()) && path != root {
				return filepath.SkipDir
			}
			return nil
		}
		if !shouldCheckDocFile(d.Name()) {
			return nil
		}
		paths = append(paths, path)
		return nil
	})
	return paths, err
}

func shouldCheckDocFile(name string) bool {
	if name == "README.md" {
		return true
	}

	return strings.HasSuffix(name, ".go") && !strings.HasSuffix(name, "_test.go")
}

func shouldSkipDocDir(name string) bool {
	switch name {
	case ".git", ".cursor", "vendor", "testdata", "internal":
		return true
	default:
		return strings.HasPrefix(name, ".")
	}
}

func isGeneratedDoc(content []byte) bool {
	return strings.Contains(string(content), "Code generated") && strings.Contains(string(content), "DO NOT EDIT")
}
