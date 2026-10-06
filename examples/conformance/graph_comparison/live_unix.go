//go:build darwin || linux

package main

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"

	"example.com/ragyconsumer/internal/modelcounter"

	"github.com/skosovsky/ragy/adapters/openai/structured"
)

var errLiveConfiguration = errors.New(
	"live graph capture requires OPENAI_API_KEY, model and qualified host tokenizer configuration",
)

func liveCaptureFile(ctx context.Context, path, model, counterPath, tokenizerIdentity string) error {
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		return errLiveConfiguration
	}
	if path == "" || strings.TrimSpace(model) == "" || strings.TrimSpace(tokenizerIdentity) == "" ||
		!filepath.IsAbs(counterPath) {
		return modelcounter.ErrUnavailable
	}
	//nolint:gosec // This is the explicit absolute trusted host executable path; stat checks the selected program before launching it, without accepting source/query data as a program.
	program, err := os.Stat(counterPath)
	if err != nil || !program.Mode().IsRegular() || program.Mode().Perm()&0o111 == 0 {
		return modelcounter.ErrUnavailable
	}
	root, err := os.MkdirTemp("", "ragy-graph-capture-")
	if err != nil {
		return err
	}
	defer func() { _ = os.RemoveAll(root) }()
	counter := modelcounter.Counter{Program: counterPath, Model: model, Identity: tokenizerIdentity}
	cfg := structured.Config{APIKey: key, Model: model}
	factories := captureFactories{
		extraction: func(attempt context.Context) (extractionPorts, error) {
			return newProviderExtractionPorts(attempt, cfg, counter.Count)
		},
		summary: func(attempt context.Context) (summaryModelPorts, error) {
			return newProviderSummaryPorts(attempt, cfg, counter.Count)
		},
	}
	raw, runErr := executeGraphCapture(
		ctx,
		root,
		captureIdentity{
			execution: liveExecution,
			adapter:   "persistent-dense+owned-bm25+managed-graph+structured-http",
			model:     model,
			tokenizer: tokenizerIdentity,
		},
		factories,
	)
	// Retain observed partial preparation on failure without making it an evaluable complete capture.
	if raw.FixtureIdentity != "" {
		if err = saveCapture(path, raw); err != nil {
			return errors.Join(runErr, err)
		}
	}
	return runErr
}
func saveCapture(path string, raw capture) error {
	data, err := json.MarshalIndent(raw, "", "  ")
	if err != nil {
		return err
	}
	if len(data)+1 > maxArtifactBytes {
		return errInvalid
	}
	//nolint:gosec // G703: path is the explicit host CLI/test output selection; source text, model output and captured references cannot select or alter it. The caller's filesystem permissions govern that destination.
	return os.WriteFile(path, append(data, '\n'), 0o600)
}
