//go:build darwin || linux

package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/skosovsky/ragy/adapters/openai/structured"
	"github.com/skosovsky/ragy/recipe/graphsummary"
)

func TestSummaryProviderTransportUsesScopedOriginalsAndActualUsage(t *testing.T) {
	// Arrange: scripted HTTP replies check the entire graph recipe transport chain.
	f, corpus, baseline, read := publishedLocalFixture(t)
	prepared, err := corpus.summarySources(t.Context(), read, f, baseline.lexical.Schema())
	if err != nil {
		t.Fatal(err)
	}
	var dispatches, counters atomic.Uint64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, bodyErr := io.ReadAll(io.LimitReader(r.Body, summaryInputBytes+1))
		if bodyErr != nil || len(body) > summaryInputBytes || strings.Contains(string(body), "fixture-key") ||
			strings.Contains(string(body), "staging") || r.Header.Get("Authorization") != "Bearer fixture-key" {
			t.Error("invalid or out-of-scope provider request")
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		dispatches.Add(1)
		content, _ := json.Marshal(`{"text":"contract summary","selected":[0,1]}`)
		_, _ = w.Write(
			[]byte(
				`{"choices":[{"index":0,"message":{"role":"assistant","content":` + string(
					content,
				) + `},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":5,"total_tokens":25}}`,
			),
		)
	}))
	defer server.Close()
	ctx, cancel := context.WithTimeout(t.Context(), attemptDuration)
	defer cancel()
	ports, err := newProviderSummaryPorts(
		ctx,
		structured.Config{
			APIKey:     "fixture-key",
			Model:      "contract-model",
			BaseURL:    server.URL,
			HTTPClient: server.Client(),
		},
		func(counterContext context.Context, request []byte) (uint64, error) {
			if _, ok := counterContext.Deadline(); !ok || counterContext.Err() != nil ||
				!strings.Contains(string(request), "graph_summary") {
				return 0, errInvalid
			}
			counters.Add(1)
			return 20, nil
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	// Act: actual core summary, HTTP adapter, scoped original reader and fresh Resolve.
	sample, err := prepared.summary(ctx, read, f.Queries[1], ports)
	// Assert: exact provider receipt and fixed experiment cost; no real model claim.
	if err != nil || sample.Failed || sample.ModelCalls != 1 || !sample.TransportCallsKnown ||
		sample.TransportCalls != 1 ||
		dispatches.Load() != 1 ||
		counters.Load() < 2 ||
		sample.InputTokens != 20 ||
		sample.OutputTokens != 5 ||
		sample.Cost != summaryCallCost ||
		!budgetsHonored(sample) ||
		recall(sample, f.Queries[1]) != 1 {
		t.Fatal(sample, err, dispatches.Load(), counters.Load())
	}
}
func TestSummaryProviderSchemaAndStageQuotes(t *testing.T) {
	// Arrange/Act/Assert: host schema rejects missing, null, unknown and trailing fields.
	for _, raw := range []string{`{}`, `{"text":null,"selected":[]}`, `{"text":"ok","selected":null}`, `{"text":"ok","selected":[],"extra":true}`, `{"text":"ok","selected":[]} {}`} {
		if validateSummaryOutput([]byte(raw)) == nil {
			t.Fatal("invalid schema", raw)
		}
	}
	if validateSummaryOutput([]byte(`{"text":"ok","selected":[0]}`)) != nil {
		t.Fatal("valid schema")
	}
	for _, stage := range []graphsummary.Stage{graphsummary.Map, graphsummary.Reduce} {
		quote, err := summaryQuote(t.Context(), stage)
		if err != nil || !quote.CostKnown || quote.Usage.Cost != summaryCallCost {
			t.Fatal(quote, err)
		}
	}
	if _, err := summaryQuote(t.Context(), "invalid"); err == nil {
		t.Fatal("invalid stage")
	}
	if _, err := newProviderSummaryPorts(context.Background(), structured.Config{}, nil); err == nil {
		t.Fatal("unbounded attempt")
	}
}
