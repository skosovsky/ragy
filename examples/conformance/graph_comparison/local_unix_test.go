//go:build darwin || linux

package main

import (
	"context"
	"testing"

	"github.com/skosovsky/ragy/access"
)

func publishedLocalFixture(t *testing.T) (fixture, graphCorpus, hybridBaseline, access.Binding) {
	t.Helper()
	var f fixture
	if err := decodeStrict(fixtureJSON, &f); err != nil {
		t.Fatal(err)
	}
	dense, err := buildDenseCorpus(t.Context(), t.TempDir(), f)
	if err != nil {
		t.Fatal(err)
	}
	read, err := dense.bind(t.Context(), nil)
	if err != nil {
		t.Fatal(err)
	}
	corpus, err := buildGraphCorpus(t.Context(), t.TempDir(), dense, read, deterministicSourceExtractions(t, f))
	if err != nil {
		t.Fatal(err)
	}
	targets, err := corpus.targets(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	read, err = dense.bind(t.Context(), targets)
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := dense.baseline(t.Context(), read)
	if err != nil {
		t.Fatal(err)
	}
	return f, corpus, baseline, read
}
func TestActualLocalExpansionSharesHybridBindingAndRetainsGoldSupports(t *testing.T) {
	// Arrange: actual durable source graph, dense and immutable scoped lexical snapshot.
	f, corpus, baseline, read := publishedLocalFixture(t)
	q := f.Queries[0]
	// Act.
	base, baseErr := baseline.retrieve(t.Context(), q)
	local, localErr := corpus.local(t.Context(), read, q)
	// Assert: exact original s1/s2 supports and one bounded model-free graph attempt.
	if baseErr != nil || localErr != nil || local.Failed || local.GraphCalls != 1 || local.ModelCalls != 0 ||
		local.RetrievalCalls != 0 || local.InputTokens != 0 || local.OutputTokens != 0 || local.Cost != 0 ||
		!local.UsageKnown || !local.CallsKnown || !budgetsHonored(local) ||
		local.Scope != base.Scope || local.Publication != base.Publication || len(local.Supports) != 2 || recall(local, q) != 1 {
		t.Fatal(base, baseErr, local, localErr)
	}
	if err := validateObservation(local, f); err != nil {
		t.Fatal(local, err)
	}
	t.Log(local.Supports, local.Nanos)
}
func TestActualLocalExpansionRejectsWrongProfileAndCanceledParent(t *testing.T) {
	for _, scenario := range []string{"profile", "canceled"} {
		t.Run(scenario, func(t *testing.T) {
			// Arrange.
			f, corpus, _, read := publishedLocalFixture(t)
			q := f.Queries[0]
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			if scenario == "profile" {
				q.Recipe = globalProfile
			} else {
				cancel()
			}
			// Act.
			result, err := corpus.local(ctx, read, q)
			// Assert: no manufactured empty successful observation on admission failure.
			if err == nil || len(result.Supports) != 0 || result.GraphCalls != 0 {
				t.Fatal(result, err)
			}
		})
	}
}
