//go:build darwin || linux

package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/skosovsky/ragy/adapters/openai/structured"
	"github.com/skosovsky/ragy/graphingest/extraction"
	"github.com/skosovsky/ragy/recipe/graphsummary"
)

func graphCaptureContractServer(t *testing.T, f fixture, calls *atomic.Uint64) *httptest.Server {
	t.Helper()
	outputs := extractionOutputs(t, f)
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(io.LimitReader(r.Body, summaryInputBytes+1))
		if err != nil || len(body) > summaryInputBytes {
			t.Error("unbounded request")
			return
		}
		input, err := providerTestInput(body)
		if err != nil || len(input.Snippets) == 0 {
			t.Error("missing snippets")
			return
		}
		var output any
		if strings.Contains(string(body), "graph_extraction") {
			value, ok := outputs[input.Snippets[0].Text]
			if !ok {
				t.Error("foreign extraction")
				return
			}
			output = value
		} else {
			selected := make([]int, len(input.Snippets))
			for i, snippet := range input.Snippets {
				selected[i] = snippet.Index
			}
			output = graphsummary.ModelOutput{Text: "explicit contract summary", Selected: selected}
		}
		content, err := json.Marshal(output)
		if err != nil {
			t.Error(err)
			return
		}
		encoded, _ := json.Marshal(string(content))
		calls.Add(1)
		_, _ = w.Write(
			[]byte(
				`{"choices":[{"index":0,"message":{"role":"assistant","content":` + string(
					encoded,
				) + `},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":5,"total_tokens":25}}`,
			),
		)
	}))
}
func TestCompleteGraphCaptureExecutesAllProducersAndSeparatePreparation(t *testing.T) {
	// Arrange: fixed HTTP/counter outputs explicitly label this contract-only capture.
	var f fixture
	if err := decodeStrict(fixtureJSON, &f); err != nil {
		t.Fatal(err)
	}
	var calls atomic.Uint64
	server := graphCaptureContractServer(t, f, &calls)
	defer server.Close()
	cfg := structured.Config{
		APIKey:     "fixture-key",
		Model:      "contract-model",
		BaseURL:    server.URL,
		HTTPClient: server.Client(),
	}
	qualifiedFixture := graphCounterFixture(t)
	counter := qualifiedFixture.Count
	factories := captureFactories{
		extraction: func(ctx context.Context) (extractionPorts, error) {
			return newProviderExtractionPorts(ctx, cfg, counter)
		},
		summary: func(ctx context.Context) (summaryModelPorts, error) {
			return newProviderSummaryPorts(ctx, cfg, counter)
		},
	}
	// Act: all four extraction calls, real publication, hybrid/local and both summary paths.
	raw, err := executeGraphCapture(
		t.Context(),
		t.TempDir(),
		captureIdentity{
			execution: contractExecution,
			adapter:   "actual-local-profile",
			model:     "contract-model",
			tokenizer: "scripted-counter-only",
		},
		factories,
	)
	if err != nil {
		t.Fatal(raw, err)
	}
	report, err := evaluate(raw)
	// Assert: six actual observations; four preparation receipts; eight actual HTTP calls.
	if err != nil || len(raw.Samples) != 6 || len(raw.Preparation.Extractions) != 4 ||
		raw.Preparation.MembershipGraphCalls != 2 ||
		calls.Load() != 8 ||
		!report.PreparationBudgetsHonored ||
		report.DefaultProfile != baselineProfile {
		t.Fatal(report, err, calls.Load())
	}
	for _, q := range f.Queries {
		measurement := report.Measurements[q.ID]
		if !measurement.BaselineBudgetsHonored || !measurement.RecipeBudgetsHonored || measurement.RecipeRecall != 1 {
			t.Fatal(q, measurement)
		}
	}
	for _, sample := range raw.Samples {
		if sample.ModelCalls > 0 && (!sample.TransportCallsKnown || sample.TransportCalls != sample.ModelCalls) {
			t.Fatal(sample)
		}
	}
	retainContractCapture(t, raw)
}
func TestCompleteCaptureStopsOnExtractionFailurePreservingPartialPreparation(t *testing.T) {
	// Arrange: provider failure must not produce a successful graph/capture artifact.
	factories := captureFactories{extraction: func(context.Context) (extractionPorts, error) {
		return extractionPorts{
			count: func(extraction.ModelInput) (uint64, error) { return 20, nil },
			model: func(context.Context, extraction.ModelInput) (extraction.ModelOutput[string, string, graphAttributes], extraction.Usage, error) {
				return extraction.ModelOutput[string, string, graphAttributes]{}, extraction.Usage{}, errInvalid
			},
		}, nil
	}, summary: func(context.Context) (summaryModelPorts, error) { return contractSummaryPorts(), nil }}
	// Act.
	raw, err := executeGraphCapture(
		t.Context(),
		t.TempDir(),
		captureIdentity{
			execution: contractExecution,
			adapter:   "actual-local-profile",
			model:     "scripted-error",
			tokenizer: "fixed-counter",
		},
		factories,
	)
	// Assert: exact one unknown failed source receipt survives; no published comparative rows.
	if err == nil || len(raw.Preparation.Extractions) != 1 || !raw.Preparation.Extractions[0].Failed ||
		raw.Preparation.Extractions[0].UsageKnown ||
		len(raw.Samples) != 0 {
		t.Fatal(raw, err)
	}
	if _, evalErr := evaluate(raw); evalErr == nil {
		t.Fatal("partial preparation promoted to complete capture")
	}
}

func providerTestInput(body []byte) (graphsummary.ModelInput, error) {
	var envelope struct {
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil {
		return graphsummary.ModelInput{}, err
	}
	for _, msg := range envelope.Messages {
		if msg.Role == "user" {
			var input graphsummary.ModelInput
			err := json.Unmarshal([]byte(msg.Content), &input)
			return input, err
		}
	}
	return graphsummary.ModelInput{}, errInvalid
}

func retainContractCapture(t *testing.T, raw capture) {
	t.Helper()
	if path := os.Getenv("RAGY_GRAPH_CONTRACT_CAPTURE"); path != "" {
		if err := saveCapture(path, raw); err != nil {
			t.Fatal(err)
		}
	}
}
