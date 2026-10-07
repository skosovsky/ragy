package ragy_test

import (
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

func TestQuickstartMatchesExecutableSource(t *testing.T) {
	t.Parallel()

	// Arrange: the published onboarding and canonical compiled package source.
	readme := readDocument(t, "README.md")
	source := readDocument(t, "examples/local-bm25/main.go")

	// Act: select the complete Go fence and parse it as a Go source file.
	_, rest, ok := strings.Cut(readme, "```go\n")
	if !ok {
		t.Fatal("README has no complete Go onboarding fence")
	}
	snippet, _, closed := strings.Cut(rest, "```")
	_, err := parser.ParseFile(token.NewFileSet(), "README.go", snippet, parser.AllErrors)

	// Assert: documentation cannot drift independently from the compiled example.
	if !closed || err != nil {
		t.Fatalf("README source fence: closed=%v, parse error=%v", closed, err)
	}
	if snippet != source {
		t.Fatal("README onboarding differs from examples/local-bm25/main.go")
	}
}

func TestCurrentDocumentationLinks(t *testing.T) {
	t.Parallel()

	// Arrange: stable current authority, excluding deliberately preserved history.
	paths := []string{
		"README.md",
		"docs/architecture.md",
		"docs/capabilities.md",
		"docs/errors-and-recovery.md",
		"docs/integration.md",
		"docs/limits.md",
		"docs/ownership.md",
		"docs/project-policies.md",
		"docs/release/runbook.md",
		"docs/verification.md",
		"lifecycle/README.md",
		"source/README.md",
		"layout/README.md",
		"chunking/README.md",
		"graphingest/README.md",
		"graphingest/resolution/README.md",
	}
	pattern := regexp.MustCompile(`\]\(([^)]+)\)`)

	for _, path := range paths {
		// Act: inspect actual Markdown links, excluding source fences.
		text := withoutCodeFences(readDocument(t, path))
		for _, match := range pattern.FindAllStringSubmatch(text, -1) {
			target := strings.Trim(match[1], "<>")
			if strings.Contains(target, "://") || strings.HasPrefix(target, "#") {
				continue
			}
			target, _, _ = strings.Cut(target, "#")
			_, err := os.Stat(filepath.Join(filepath.Dir(path), target))

			// Assert: every local current guide/example/evidence target exists.
			if err != nil {
				t.Errorf("%s links to missing %s: %v", path, target, err)
			}
		}
	}
}

func TestCurrentDocsDoNotReferenceRemovedRetrievalSymbols(t *testing.T) {
	t.Parallel()

	// Arrange: exact removed API selectors, not natural-language blacklists.
	removed := []string{"NewPipelineBuilder", "NewRequestPipelineBuilder", "RetrieverNode", "RequestRetrieverNode"}
	paths := []string{"README.md", "docs/integration.md"}

	for _, path := range paths {
		content := readDocument(t, path)
		for _, symbol := range removed {
			// Act: match the whole retrieval selector, excluding newer longer names.
			pattern := regexp.MustCompile(`\bretrieval\.` + regexp.QuoteMeta(symbol) + `\b`)

			// Assert: old selectors must not be presented as current APIs.
			if pattern.MatchString(content) {
				t.Errorf("%s references removed retrieval.%s", path, symbol)
			}
		}
	}
}

func readDocument(t *testing.T, path string) string {
	t.Helper()

	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile(%s): %v", path, err)
	}

	return string(content)
}

func withoutCodeFences(text string) string {
	var out strings.Builder
	inside := false
	for line := range strings.SplitSeq(text, "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), "```") {
			inside = !inside
			continue
		}
		if !inside {
			out.WriteString(line)
			out.WriteByte('\n')
		}
	}

	return out.String()
}
