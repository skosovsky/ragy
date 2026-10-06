//go:build darwin || linux

package main

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/skosovsky/ragy/adapters/openai/structured"
	"github.com/skosovsky/ragy/graph"
	"github.com/skosovsky/ragy/graph/managed"
	"github.com/skosovsky/ragy/graphingest/extraction"
)

func contractExtractionOutput(batch sourceExtraction) extraction.ModelOutput[string, string, graphAttributes] {
	out := extraction.ModelOutput[string, string, graphAttributes]{
		Entities:  []extraction.Entity[string, graphAttributes]{},
		Relations: []extraction.Relation[string, graphAttributes]{},
	}
	for _, e := range batch.Value.Entities {
		out.Entities = append(
			out.Entities,
			extraction.Entity[string, graphAttributes]{
				ID:         e.ID,
				Name:       e.Name,
				Kind:       e.Kind,
				Attributes: e.Attributes,
				Snippets:   []int{0},
			},
		)
	}
	for _, e := range batch.Value.Relations {
		out.Relations = append(
			out.Relations,
			extraction.Relation[string, graphAttributes]{
				ID:         e.ID,
				From:       e.From,
				To:         e.To,
				Kind:       e.Kind,
				Attributes: e.Attributes,
				Snippets:   []int{0},
			},
		)
	}
	return out
}
func extractionOutputs(t *testing.T, f fixture) map[string]extraction.ModelOutput[string, string, graphAttributes] {
	t.Helper()
	byText := make(map[string]extraction.ModelOutput[string, string, graphAttributes])
	for _, batch := range deterministicSourceExtractions(t, f) {
		for _, row := range f.Sources {
			if row.ID == batch.SourceID {
				byText[row.Text] = contractExtractionOutput(batch)
			}
		}
	}
	return byText
}

func extractionContractServer(
	t *testing.T,
	outputs map[string]extraction.ModelOutput[string, string, graphAttributes],
	calls *atomic.Uint64,
) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(io.LimitReader(r.Body, summaryInputBytes+1))
		if err != nil || len(body) > summaryInputBytes || !strings.Contains(string(body), "graph_extraction") ||
			strings.Contains(string(body), "fixture-key") {
			t.Error("invalid extraction request")
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		// Locate the model evidence inside the actual provider envelope, never gold entities.
		var envelope struct {
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		if err = json.Unmarshal(body, &envelope); err != nil {
			t.Error(err)
			return
		}
		var input extraction.ModelInput
		for _, msg := range envelope.Messages {
			if msg.Role == "user" {
				err = json.Unmarshal([]byte(msg.Content), &input)
			}
		}
		if err != nil || len(input.Snippets) != 1 {
			t.Error("invalid evidence")
			return
		}
		output, ok := outputs[input.Snippets[0].Text]
		if !ok {
			t.Error("foreign source evidence")
			return
		}
		calls.Add(1)
		content, err := json.Marshal(output)
		if err != nil {
			t.Error(err)
			return
		}
		encoded, _ := json.Marshal(string(content))
		_, _ = w.Write(
			[]byte(
				`{"choices":[{"index":0,"message":{"role":"assistant","content":` + string(
					encoded,
				) + `},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":5,"total_tokens":25}}`,
			),
		)
	}))
}
func TestActualProviderExtractionPublishesFullReferenceGraph(t *testing.T) {
	// Arrange: actual HTTP/core extraction with independent contract replies per source.
	var f fixture
	if err := decodeStrict(fixtureJSON, &f); err != nil {
		t.Fatal(err)
	}
	outputs := extractionOutputs(t, f)
	var calls atomic.Uint64
	server := extractionContractServer(t, outputs, &calls)
	defer server.Close()
	dense, err := buildDenseCorpus(t.Context(), t.TempDir(), f)
	if err != nil {
		t.Fatal(err)
	}
	read, err := dense.bind(t.Context(), nil)
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := dense.baseline(t.Context(), read)
	if err != nil {
		t.Fatal(err)
	}
	var batches []sourceExtraction
	// Act: each source owns a bounded request/counter/reservation, then real publication.
	for _, row := range f.Sources {
		ctx, cancel := context.WithTimeout(t.Context(), attemptDuration)
		ports, portErr := newProviderExtractionPorts(
			ctx,
			structured.Config{
				APIKey:     "fixture-key",
				Model:      "contract-model",
				BaseURL:    server.URL,
				HTTPClient: server.Client(),
			},
			func(context.Context, []byte) (uint64, error) { return 20, nil },
		)
		if portErr != nil {
			cancel()
			t.Fatal(portErr)
		}
		batch, observed, runErr := extractSource(ctx, read, baseline.lexical.Schema(), row, f, ports)
		cancel()
		if runErr != nil || observed.Failed || observed.ModelCalls != 1 || !observed.TransportCallsKnown ||
			observed.TransportCalls != 1 ||
			!observed.UsageKnown ||
			observed.InputTokens != 20 ||
			observed.OutputTokens != 5 ||
			observed.Cost != summaryCallCost {
			t.Fatal(observed, runErr)
		}
		assertExtractionBindings(t, row, batch)
		batches = append(batches, batch)
	}
	corpus, err := buildGraphCorpus(t.Context(), t.TempDir(), dense, read, batches)
	// Assert: actual provider extraction produced the expected resolved cardinality/supports.
	if err != nil || calls.Load() != uint64(len(f.Sources)) || len(corpus.resolved.Entities) != len(f.Entities) ||
		len(corpus.resolved.Relations) != len(f.Edges) {
		t.Fatal(err, calls.Load(), corpus.resolved)
	}
	targets, err := corpus.targets(t.Context())
	if err != nil || len(targets) != len(f.Sources) {
		t.Fatal(targets, err)
	}
	assertProviderGraphGold(t, dense, corpus, f)
}
func TestActualExtractionFailurePreservesUnknownUsageWithoutRetry(t *testing.T) {
	// Arrange.
	f, _, baseline, read := publishedLocalFixture(t)
	ports := extractionPorts{
		count: func(extraction.ModelInput) (uint64, error) { return 20, nil },
		model: func(context.Context, extraction.ModelInput) (extraction.ModelOutput[string, string, graphAttributes], extraction.Usage, error) {
			return extraction.ModelOutput[string, string, graphAttributes]{}, extraction.Usage{}, errors.New(
				"lost model response",
			)
		},
	}
	// Act.
	batch, observed, err := extractSource(t.Context(), read, baseline.lexical.Schema(), f.Sources[0], f, ports)
	// Assert.
	if err == nil || !observed.Failed || observed.ModelCalls != 1 || observed.UsageKnown ||
		len(batch.Value.Entities) != 0 {
		t.Fatal(batch, observed, err)
	}
}
func TestExtractionSchemaRejectsMissingAttributesAndModelAuthority(t *testing.T) {
	// Arrange/Act/Assert.
	for _, raw := range []string{`{}`, `{"entities":null,"relations":[]}`, `{"entities":[{"id":"s","name":"Billing","kind":"Service","snippets":[0]}],"relations":[]}`, `{"entities":[{"id":"s","name":"Billing","kind":"Service","attributes":{},"snippets":[0]}],"relations":[]}`, `{"entities":[],"relations":[],"namespace":"model-authority"}`} {
		if validateExtractionOutput([]byte(raw)) == nil {
			t.Fatal("invalid schema", raw)
		}
	}
	if validateExtractionOutput([]byte(`{"entities":[],"relations":[]}`)) != nil {
		t.Fatal("valid empty extraction")
	}
}

func assertExtractionBindings(t *testing.T, row sourceRow, batch sourceExtraction) {
	t.Helper()
	for _, e := range batch.Value.Entities {
		if e.Namespace != row.Namespace || len(e.Supports) != 1 ||
			e.Supports[0].Reference != originalReference(row.ID) {
			t.Fatal(e)
		}
	}
}

func assertProviderGraphGold(t *testing.T, dense denseCorpus, corpus graphCorpus, f fixture) {
	t.Helper()
	targets, err := corpus.targets(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	read, err := dense.bind(t.Context(), targets)
	if err != nil {
		t.Fatal(err)
	}
	var seeds []string
	for _, e := range corpus.resolved.Entities {
		seeds = append(seeds, e.ID)
	}
	actual, err := corpus.adapter.Traverse(
		t.Context(),
		managed.Request{
			Read:      read,
			Traversal: graph.TraversalRequest{Seeds: seeds, Direction: graph.DirectionUndirected, Depth: 1},
			MaxNodes:  localNodeCap,
			MaxEdges:  localEdgeCap,
		},
	)
	if err != nil || len(actual.Conflicts) != 0 || len(corpus.resolved.Unresolved) != 0 {
		t.Fatal(actual, err)
	}
	assertGoldGraph(t, f, actual)
}
