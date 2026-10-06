package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

func TestActualBM25RecipesToCompleteContractCapture(t *testing.T) {
	// Arrange: actual BM25 and actual transport; model/tokenizer outputs are protocol fixtures.
	var calls atomic.Uint64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(io.LimitReader(r.Body, maxCounterRequest+1))
		if err != nil || strings.Contains(string(body), "private payload") ||
			strings.Contains(string(body), "foreign") ||
			strings.Contains(string(body), "fixture-key") {
			t.Error("private corpus or credentials reached model")
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		calls.Add(1)
		content := `{"selected":[0],"sufficient":true}`
		if strings.Contains(string(body), "retrieval_planning") {
			content = `{"queries":["возврат оплаты срок"]}`
		}
		encoded, err := json.Marshal(content)
		if err != nil {
			t.Error(err)
			return
		}
		_, _ = w.Write(
			[]byte(
				`{"choices":[{"index":0,"message":{"role":"assistant","content":` + string(
					encoded,
				) + `},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":5,"total_tokens":25}}`,
			),
		)
	}))
	defer server.Close()
	cfg := liveModelConfig{
		apiKey:  "fixture-key",
		baseURL: server.URL,
		client:  server.Client(),
		counter: counterFixture(t, "valid"),
	}
	// Act: complete 5-query × 4-profile orchestration and offline evaluator.
	raw, err := captureExperiment(t.Context(), cfg, contractExecution)
	if err != nil {
		t.Fatal(err)
	}
	result, err := evaluate(raw)
	// Assert: no scripted success is relabeled as live; actual counts and source refs survive.
	if err != nil || len(raw.Samples) != 20 || calls.Load() != 30 ||
		result.Capture.ExecutionKind != contractExecution ||
		result.DefaultProfile != baselineProfile {
		t.Fatal(err, len(raw.Samples), calls.Load())
	}
	for _, sample := range raw.Samples {
		assertCapturedSample(t, sample)
	}
	encoded, err := json.Marshal(raw)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), "fixture-key") || strings.Contains(string(encoded), "foreign") ||
		strings.Contains(string(encoded), "private payload") {
		t.Fatal("capture exported forbidden payload/credentials")
	}
}
func assertCapturedSample(t *testing.T, sample observation) {
	t.Helper()
	if sample.Failed || !sample.UsageKnown || !withinBudget(sample) {
		t.Fatal("bounded recipe capture failed", sample)
	}
	if sample.Strategy == baselineProfile {
		if sample.ModelCalls != 0 || len(sample.ModelUsage) != 0 || sample.RetrievalCalls != 1 {
			t.Fatal(sample)
		}
	} else if sample.ModelCalls != 2 || len(sample.ModelUsage) != 2 || sample.InputTokens != 40 || sample.OutputTokens != 10 || sample.Cost != 60 {
		t.Fatal(sample)
	}
	for _, ref := range sample.SourceRefs {
		if ref != corpusReference(ref.Artifact) || ref.Artifact == "foreign" {
			t.Fatal("incorrect captured revision/representation", ref)
		}
	}
}
func TestCaptureCanceledOrUnconfiguredNeverMakesModelCalls(t *testing.T) {
	// Arrange.
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	cfg := liveModelConfig{apiKey: "fixture-key", counter: counterFixture(t, "valid")}
	// Act and Assert: no incomplete or synthetic success artifact is written.
	if _, err := captureExperiment(ctx, cfg, contractExecution); err == nil {
		t.Fatal("canceled capture accepted")
	}
	if _, err := captureExperiment(t.Context(), liveModelConfig{}, liveExecution); err == nil {
		t.Fatal("unconfigured live capture accepted")
	}
	t.Setenv("OPENAI_API_KEY", "")
	path := filepath.Join(t.TempDir(), "capture.json")
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	if err = liveCaptureFile(t.Context(), path, fixtureCounterModel, executable, fixtureCounterIdentity); err == nil {
		t.Fatal("missing credentials accepted")
	}
	if _, err = os.Stat(path); !os.IsNotExist(err) {
		t.Fatal("failed live preflight wrote an artifact", err)
	}
}

func TestFailedHTTPAttemptRetainsActualDispatchAndUnknownUsage(t *testing.T) {
	// Arrange: actual scoped corpus and failing single-dispatch provider fixture.
	var calls atomic.Uint64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer server.Close()
	corpus, err := newCaptureCorpus(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	cfg := liveModelConfig{
		apiKey:  "fixture-key",
		baseURL: server.URL,
		client:  server.Client(),
		counter: counterFixture(t, "valid"),
	}
	// Act: failure is recorded as an experiment observation, never successful retrieval.
	sample, err := captureSample(t.Context(), cfg, corpus, rewriteProfile, corpus.fixture.Queries[0])
	// Assert: one actual dispatch, no retry, no selected payload and explicitly unknown usage.
	if err != nil || !sample.Failed || sample.Outcome != failedOutcome || sample.ModelCalls != 1 || calls.Load() != 1 ||
		sample.UsageKnown ||
		len(sample.IDs) != 0 ||
		len(sample.SourceRefs) != 0 ||
		len(sample.ModelUsage) != 1 ||
		sample.ModelUsage[0].Known {
		t.Fatal(sample, err, calls.Load())
	}
}
