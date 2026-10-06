package main

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"strings"
	"testing"
	"time"
)

const fixtureCounterModel = "counter-fixture-model"
const fixtureCounterIdentity = "counter-fixture-only"

// TestCounterProcess is only invoked as a child process by counter protocol tests.
// Its fixed count is a contract fixture, never a qualified model tokenizer.
func TestCounterProcess(t *testing.T) {
	t.Helper()
	if len(os.Args) < 3 || os.Args[len(os.Args)-2] != "counter-child" {
		return
	}
	mode := os.Args[len(os.Args)-1]
	input, err := io.ReadAll(os.Stdin)
	if err != nil {
		os.Exit(2)
	}
	var wire struct {
		Model    string            `json:"model"`
		Messages []json.RawMessage `json:"messages"`
	}
	if json.Unmarshal(input, &wire) != nil || wire.Model != fixtureCounterModel || len(wire.Messages) == 0 {
		os.Exit(3)
	}
	if os.Getenv("OPENAI_API_KEY") != "" {
		os.Exit(4)
	}
	switch mode {
	case "sleep":
		time.Sleep(time.Minute)
	case "oversize":
		_, _ = os.Stdout.WriteString(strings.Repeat("x", maxCounterResponse+1))
	case "failure":
		_, _ = os.Stderr.WriteString("private counter error")
		os.Exit(5)
	case "malformed":
		_, _ = os.Stdout.WriteString("invalid-json")
	case "unknown":
		_, _ = os.Stdout.WriteString(
			`{"model":"counter-fixture-model","tokenizer_identity":"counter-fixture-only","input_tokens":20,"secret":"raw-error"}`,
		)
	default:
		receipt := counterReceipt{Model: fixtureCounterModel, Identity: fixtureCounterIdentity, Tokens: 20}
		if mode == "foreign-model" {
			receipt.Model = "other"
		}
		if mode == "foreign-tokenizer" {
			receipt.Identity = "other"
		}
		if mode == "zero" {
			receipt.Tokens = 0
		}
		if mode == "negative" {
			_, _ = os.Stdout.WriteString(
				`{"model":"counter-fixture-model","tokenizer_identity":"counter-fixture-only","input_tokens":-1}`,
			)
			os.Exit(0)
		}
		if err = json.NewEncoder(os.Stdout).Encode(receipt); err != nil {
			os.Exit(6)
		}
	}
	os.Exit(0)
}
func counterFixture(t *testing.T, mode string) hostCounter {
	t.Helper()
	program, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	return hostCounter{
		program:  program,
		args:     []string{"-test.run=^TestCounterProcess$", "--", "counter-child", mode},
		model:    fixtureCounterModel,
		identity: fixtureCounterIdentity,
	}
}
func counterRequest() []byte {
	return []byte(`{"model":"counter-fixture-model","messages":["complete request framing/schema fixture"]}`)
}
func TestHostCounterProtocolBoundaries(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "counter-test-secret")
	for _, mode := range []string{"valid", "foreign-model", "foreign-tokenizer", "zero", "negative", "malformed", "unknown", "oversize", "failure"} {
		t.Run(mode, func(t *testing.T) {
			// Arrange: actual child executable with a bounded fixture-only receipt.
			counter := counterFixture(t, mode)
			// Act.
			tokens, err := counter.count(t.Context(), counterRequest())
			// Assert: malformed/foreign/unbounded host output cannot qualify accounting.
			if mode == "valid" {
				if err != nil || tokens != 20 {
					t.Fatal(tokens, err)
				}
				return
			}
			if !errors.Is(err, errCounter) || tokens != 0 || strings.Contains(err.Error(), "private") {
				t.Fatal(tokens, err)
			}
		})
	}
}
func TestHostCounterCanceledDeadlineAndPreflight(t *testing.T) {
	// Arrange: child which would exceed host computation budget.
	counter := counterFixture(t, "sleep")
	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Millisecond)
	defer cancel()
	// Act.
	tokens, err := counter.count(ctx, counterRequest())
	// Assert: bounded parent cancellation kills child and preserves context classification.
	if tokens != 0 || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal(tokens, err)
	}
	canceled, stop := context.WithCancel(t.Context())
	stop()
	if _, err = counter.count(canceled, counterRequest()); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	for _, request := range [][]byte{nil, []byte(`{"model":"foreign"}`), []byte(strings.Repeat("x", maxCounterRequest+1))} {
		if _, err = counter.count(t.Context(), request); !errors.Is(err, errCounter) {
			t.Fatal(err)
		}
	}
	counter.program = "relative-path"
	if _, err = counter.count(t.Context(), counterRequest()); !errors.Is(err, errCounter) {
		t.Fatal(err)
	}
}
