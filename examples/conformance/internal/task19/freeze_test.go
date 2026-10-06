package task19

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFreezeAcceptsPinnedConsumerAndRejectsChangedCodeDigest(t *testing.T) {
	// Arrange: independently hash current public consumer files; no holdout input.
	root, err := repositoryRoot()
	if err != nil {
		t.Fatal(err)
	}
	files := map[string]string{}
	for _, dir := range []string{"examples/conformance/recipe_comparison", "examples/conformance/graph_comparison", "examples/conformance/tensor_comparison", "examples/conformance/internal/task19"} {
		paths, globErr := filepath.Glob(filepath.Join(root, dir, "*.go"))
		if globErr != nil {
			t.Fatal(globErr)
		}
		for _, path := range paths {
			b, readErr := os.ReadFile(path)
			if readErr != nil {
				t.Fatal(readErr)
			}
			relative, _ := filepath.Rel(root, path)
			files[relative] = Digest(b)
		}
	}
	corpusPath := "examples/conformance/datasets/task19/corpus.json"
	devPath := "examples/conformance/datasets/task19/dev.json"
	cb, err := os.ReadFile(filepath.Join(root, corpusPath))
	if err != nil {
		t.Fatal(err)
	}
	sb, err := os.ReadFile(filepath.Join(root, devPath))
	if err != nil {
		t.Fatal(err)
	}
	files[corpusPath] = Digest(cb)
	files[devPath] = Digest(sb)
	policy, _ := json.Marshal(Policy())
	f := Freeze{
		Schema:        "task19-freeze/v1",
		CodeRevision:  "test",
		Files:         files,
		Configuration: policy,
		CorpusSHA256:  Digest(cb),
		DevSHA256:     Digest(sb),
		HoldoutSHA256: strings.Repeat("0", 64),
	}
	valid, _ := json.Marshal(f)
	// Act.
	validErr := VerifyFreeze(valid, cb, sb, "dev")
	f.Files["examples/conformance/internal/task19/data.go"] = strings.Repeat("f", 64)
	changed, _ := json.Marshal(f)
	changedErr := VerifyFreeze(changed, cb, sb, "dev")
	// Assert.
	if validErr != nil || changedErr == nil {
		t.Fatalf("valid=%v changed=%v", validErr, changedErr)
	}
}
