package task19

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
)

// Freeze pins the reviewed consumer files and common policy before holdout publication.
type Freeze struct {
	Schema        string            `json:"schema"`
	CodeRevision  string            `json:"code_revision"`
	Files         map[string]string `json:"files"`
	Configuration json.RawMessage   `json:"configuration"`
	CorpusSHA256  string            `json:"corpus_sha256"`
	DevSHA256     string            `json:"dev_sha256"`
	HoldoutSHA256 string            `json:"holdout_sha256"`
	Created       string            `json:"created"`
}

//nolint:gocognit // Independent manifest validation checks code, policy and each dataset identity before execution.
func VerifyFreeze(data []byte, cBytes, sBytes []byte, split string) error {
	var f Freeze
	if err := json.Unmarshal(data, &f); err != nil {
		return err
	}
	if f.Schema != "task19-freeze/v1" || f.CodeRevision == "" || len(f.Files) == 0 ||
		f.CorpusSHA256 != Digest(cBytes) ||
		f.DevSHA256 == "" ||
		f.HoldoutSHA256 == "" {
		return errors.New("incomplete or mismatched freeze manifest")
	}
	expected, _ := json.Marshal(Policy())
	var declared any
	var policy any
	if json.Unmarshal(f.Configuration, &declared) != nil || json.Unmarshal(expected, &policy) != nil {
		return errors.New("invalid frozen policy")
	}
	a, _ := json.Marshal(declared)
	b, _ := json.Marshal(policy)
	if !bytes.Equal(a, b) {
		return errors.New("current policy differs from frozen policy")
	}
	root, err := repositoryRoot()
	if err != nil {
		return err
	}
	for _, dir := range []string{"examples/conformance/recipe_comparison", "examples/conformance/graph_comparison", "examples/conformance/tensor_comparison", "examples/conformance/internal/task19"} {
		paths, globErr := filepath.Glob(filepath.Join(root, dir, "*.go"))
		if globErr != nil {
			return globErr
		}
		for _, file := range paths {
			relative, _ := filepath.Rel(root, file)
			if f.Files[relative] == "" {
				return errors.New("consumer file missing from freeze: " + relative)
			}
		}
	}
	corpusPresent, devPresent := false, false
	for path, digest := range f.Files {
		if filepath.IsAbs(path) || filepath.Clean(path) != path || path == ".." ||
			strings.HasPrefix(path, ".."+string(filepath.Separator)) ||
			len(digest) != 64 {
			return errors.New("invalid frozen file identity")
		}
		content, err := os.ReadFile(filepath.Join(root, path))
		if err != nil {
			return err
		}
		if Digest(content) != digest {
			return errors.New("frozen file changed: " + path)
		}
		if digest == f.CorpusSHA256 {
			corpusPresent = true
		}
		if digest == f.DevSHA256 {
			devPresent = true
		}
	}
	if !corpusPresent || !devPresent {
		return errors.New("corpus/dev absent from frozen files")
	}
	if split == "holdout" && Digest(sBytes) != f.HoldoutSHA256 {
		return errors.New("holdout differs from independently sealed digest")
	}
	return nil
}

func repositoryRoot() (string, error) {
	root, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for {
		if _, err = os.Stat(filepath.Join(root, "go.work")); err == nil {
			return root, nil
		}
		parent := filepath.Dir(root)
		if parent == root {
			return "", errors.New("checkout root not found")
		}
		root = parent
	}
}
