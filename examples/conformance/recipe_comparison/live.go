package main

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
)

var errLiveConfiguration = errors.New(
	"live experiment requires OPENAI_API_KEY, model and qualified host tokenizer configuration",
)

func liveCaptureFile(ctx context.Context, path, model, counterPath, tokenizerIdentity string) error {
	if os.Getenv("OPENAI_API_KEY") == "" {
		return errLiveConfiguration
	}
	if path == "" || strings.TrimSpace(model) == "" || strings.TrimSpace(tokenizerIdentity) == "" ||
		!filepath.IsAbs(counterPath) {
		return errCounter
	}
	//nolint:gosec // Explicit absolute trusted host executable selection, not query/source input; stat validates only that selected program before execution.
	program, err := os.Stat(counterPath)
	if err != nil || !program.Mode().IsRegular() || program.Mode().Perm()&0o111 == 0 {
		return errCounter
	}
	cfg := liveModelConfig{
		apiKey:  os.Getenv("OPENAI_API_KEY"),
		counter: hostCounter{program: counterPath, model: model, identity: tokenizerIdentity},
	}
	raw, err := captureExperiment(ctx, cfg, liveExecution)
	if err != nil {
		return err
	}
	data, err := json.MarshalIndent(raw, "", "  ")
	if err != nil {
		return err
	}
	if len(data) > maxInputBytes {
		return errInvalid
	}
	return os.WriteFile(path, append(data, '\n'), 0o600)
}
