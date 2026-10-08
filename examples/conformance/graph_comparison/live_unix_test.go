//go:build darwin || linux

package main

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/skosovsky/ragy/examples/conformance/internal/modelcounter"
)

func graphCounterFixture(t *testing.T) modelcounter.Counter {
	t.Helper()
	path := filepath.Join(t.TempDir(), "counter-contract-fixture")
	// Explicit trusted test executable: fixed receipt, no actual token qualification.
	program := `#!/bin/sh
[ -z "$OPENAI_API_KEY" ] || exit 2
cat >/dev/null
printf '%s\n' '{"model":"contract-model","tokenizer_identity":"scripted-counter-only","input_tokens":20}'
`
	if err := os.WriteFile(path, []byte(program), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(path, 0o700); err != nil {
		t.Fatal(err)
	}
	return modelcounter.Counter{Program: path, Model: "contract-model", Identity: "scripted-counter-only"}
}
func TestLiveCapturePreflightRequiresCredentialsBeforeArtifactCreation(t *testing.T) {
	// Arrange.
	t.Setenv("OPENAI_API_KEY", "")
	output := filepath.Join(t.TempDir(), "capture.json")
	// Act.
	err := liveCaptureFile(t.Context(), output, "", "", "")
	// Assert: missing environment cannot manufacture a successful artifact.
	if !errors.Is(err, errLiveConfiguration) {
		t.Fatal(err)
	}
	if _, statErr := os.Stat(output); !errors.Is(statErr, os.ErrNotExist) {
		t.Fatal(statErr)
	}
}
func TestLiveCapturePreflightRejectsUnqualifiedHostConfiguration(t *testing.T) {
	// Arrange: non-network tests validate inputs before any temporary publication.
	t.Setenv("OPENAI_API_KEY", "fixture-key")
	for _, scenario := range []string{"model", "tokenizer-id", "relative", "missing", "not-executable"} {
		t.Run(scenario, func(t *testing.T) {
			counter := graphCounterFixture(t)
			model, identity := counter.Model, counter.Identity
			switch scenario {
			case "model":
				model = ""
			case "tokenizer-id":
				identity = ""
			case "relative":
				counter.Program = "relative-program"
			case "missing":
				counter.Program = filepath.Join(t.TempDir(), "missing")
			case "not-executable":
				if err := os.Chmod(counter.Program, 0o600); err != nil {
					t.Fatal(err)
				}
			}
			output := filepath.Join(t.TempDir(), "capture.json")
			// Act.
			err := liveCaptureFile(context.Background(), output, model, counter.Program, identity)
			// Assert.
			if !errors.Is(err, modelcounter.ErrUnavailable) {
				t.Fatal(err)
			}
			if _, statErr := os.Stat(output); !errors.Is(statErr, os.ErrNotExist) {
				t.Fatal(statErr)
			}
		})
	}
}
func TestCaptureFileRetainsPartialObservedFailureWithoutEvaluatingIt(t *testing.T) {
	// Arrange.
	raw := fixtureCapture(t)
	raw.Samples = nil
	raw.Preparation.Extractions = raw.Preparation.Extractions[:1]
	raw.Preparation.Extractions[0].Failed = true
	path := filepath.Join(t.TempDir(), "partial.json")
	// Act.
	err := saveCapture(path, raw)
	encoded, readErr := os.ReadFile(path)
	var retained capture
	decodeErr := decodeStrict(encoded, &retained)
	// Assert.
	if err != nil || readErr != nil || decodeErr != nil || len(retained.Preparation.Extractions) != 1 ||
		!retained.Preparation.Extractions[0].Failed {
		t.Fatal(err, readErr, decodeErr, retained)
	}
	if err = evaluateFile(path, ""); err == nil {
		t.Fatal("partial capture certified")
	}
}
