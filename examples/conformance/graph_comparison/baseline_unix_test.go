//go:build darwin || linux

package main

import (
	"context"
	"testing"
)

func TestActualPersistentDenseBM25HybridBaseline(t *testing.T) {
	// Arrange: actual staged durable dense records, fresh adapter and owned BM25.
	var f fixture
	if err := decodeStrict(fixtureJSON, &f); err != nil {
		t.Fatal(err)
	}
	corpus, err := buildDenseCorpus(t.Context(), t.TempDir(), f)
	if err != nil {
		t.Fatal(err)
	}
	read, err := corpus.bind(t.Context(), nil)
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := corpus.baseline(t.Context(), read)
	if err != nil {
		t.Fatal(err)
	}
	for _, q := range f.Queries {
		// Act: execute actual dense+lexical retrieval and the shipped RRF.
		result, runErr := baseline.retrieve(t.Context(), q)
		// Assert: no model, bounded calls, original source references and current binding.
		if runErr != nil || result.Failed || result.RetrievalCalls != 2 || result.ModelCalls != 0 ||
			result.Scope != read.Snapshot().Identity || result.Publication != read.Publication().Reference() ||
			len(result.Supports) == 0 || len(result.Supports) > supportTopK || !budgetsHonored(result) {
			t.Fatal(result, runErr)
		}
		if err = validateObservation(result, f); err != nil {
			t.Fatal(result, err)
		}
		for _, ref := range result.Supports {
			if ref.Source == foreignSource {
				t.Fatal("foreign payload selected", result)
			}
		}
		t.Log(q.ID, result.Supports, recall(result, q), result.Nanos)
	}
}
func TestHybridBaselineCanceledParentDeliversNoObservation(t *testing.T) {
	// Arrange.
	var f fixture
	if err := decodeStrict(fixtureJSON, &f); err != nil {
		t.Fatal(err)
	}
	corpus, err := buildDenseCorpus(t.Context(), t.TempDir(), f)
	if err != nil {
		t.Fatal(err)
	}
	read, err := corpus.bind(t.Context(), nil)
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := corpus.baseline(t.Context(), read)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	// Act.
	result, err := baseline.retrieve(ctx, f.Queries[0])
	// Assert: admission failure cannot be a successful empty retrieval sample.
	if err == nil || len(result.Supports) != 0 || result.RetrievalCalls != 0 {
		t.Fatal(result, err)
	}
}
