//go:build darwin || linux

package main

import (
	"context"
	"errors"
	"testing"

	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/recipe/graphsummary"
)

// Contract model only: this cannot establish live model quality.
func contractSummaryPorts() summaryModelPorts {
	return summaryModelPorts{
		quote: func(context.Context, graphsummary.Stage) (budget.Reservation, error) {
			return budget.Reservation{
				Kind:      budget.Model,
				Usage:     budget.Usage{InputTokens: 1024, OutputTokens: 256, Cost: 30},
				CostKnown: true,
			}, nil
		},
		count: func(graphsummary.ModelInput) (uint64, error) { return 20, nil },
		model: func(_ context.Context, input graphsummary.ModelInput) (graphsummary.ModelOutput, graphsummary.Usage, error) {
			var selected []int
			for _, snippet := range input.Snippets {
				selected = append(selected, snippet.Index)
			}
			return graphsummary.ModelOutput{
				Text:     "contract summary",
				Selected: selected,
			}, graphsummary.Usage{
				Value: budget.Usage{InputTokens: 20, OutputTokens: 10, Cost: 30},
				Known: true,
			}, nil
		},
	}
}
func TestActualCommunityGlobalSummaryCaptureUsesOriginalSupportsAndUsage(t *testing.T) {
	for _, profile := range []string{communityProfile, globalProfile} {
		t.Run(profile, func(t *testing.T) {
			// Arrange: actual durable graph, scoped original source loader; model contract fixture.
			f, corpus, baseline, read := publishedLocalFixture(t)
			prepared, err := corpus.summarySources(t.Context(), read, f, baseline.lexical.Schema())
			if err != nil {
				t.Fatal(err)
			}
			q := summaryQuery(t, f, profile)
			// Act.
			sample, err := prepared.summary(t.Context(), read, q, contractSummaryPorts())
			base, baseErr := baseline.retrieve(t.Context(), q)
			// Assert: actual core dispatch/usage, shared binding, original supports and source I/O.
			calls := uint64(1)
			if profile == globalProfile {
				calls = 3
			}
			if err != nil || baseErr != nil || sample.Failed || sample.ModelCalls != calls ||
				sample.InputTokens != 20*calls || sample.OutputTokens != 10*calls || sample.Cost != 30*calls ||
				!budgetsHonored(
					sample,
				) || recall(sample, q) != 1 || sample.Scope != base.Scope || sample.Publication != base.Publication ||
				sample.SourceMetadataCalls == 0 || sample.SourcePayloadCalls == 0 || prepared.host.payloadCalls.Load() != 0 {
				t.Fatal(sample, err, baseErr)
			}
			if err = validateObservation(sample, f); err != nil {
				t.Fatal(err)
			}
			t.Log(sample)
		})
	}
}
func TestActualSummaryFailureRetainsUnknownUsageAndKnownSingleDispatch(t *testing.T) {
	// Arrange.
	f, corpus, baseline, read := publishedLocalFixture(t)
	prepared, err := corpus.summarySources(t.Context(), read, f, baseline.lexical.Schema())
	if err != nil {
		t.Fatal(err)
	}
	ports := contractSummaryPorts()
	ports.model = func(context.Context, graphsummary.ModelInput) (graphsummary.ModelOutput, graphsummary.Usage, error) {
		return graphsummary.ModelOutput{}, graphsummary.Usage{}, errors.New("lost provider response")
	}
	// Act.
	sample, err := prepared.summary(t.Context(), read, f.Queries[1], ports)
	// Assert: no retry, no output, no fabricated zero usage acceptance.
	if err != nil || !sample.Failed || sample.ModelCalls != 1 || !sample.CallsKnown || sample.UsageKnown ||
		len(sample.Supports) != 0 || budgetsHonored(sample) {
		t.Fatal(sample, err)
	}
}
func TestActualSummaryRetirementDuringModelSettlesUsageAndExportsNoPayload(t *testing.T) {
	// Arrange.
	f, corpus, baseline, read := publishedLocalFixture(t)
	prepared, err := corpus.summarySources(t.Context(), read, f, baseline.lexical.Schema())
	if err != nil {
		t.Fatal(err)
	}
	ports := contractSummaryPorts()
	original := ports.model
	ports.model = func(ctx context.Context, input graphsummary.ModelInput) (graphsummary.ModelOutput, graphsummary.Usage, error) {
		retireSummaryGraphSource(t, corpus, "s1")
		return original(ctx, input)
	}
	// Act.
	sample, err := prepared.summary(t.Context(), read, f.Queries[1], ports)
	// Assert: current host tombstone prevents cached summary delivery after real dispatch.
	if err != nil || !sample.Failed || sample.ModelCalls != 1 || !sample.UsageKnown || sample.Cost != 30 ||
		len(sample.Supports) != 0 {
		t.Fatal(sample, err)
	}
}
func TestActualSummaryCanceledAdmissionMakesNoModelCall(t *testing.T) {
	// Arrange.
	f, corpus, baseline, read := publishedLocalFixture(t)
	prepared, err := corpus.summarySources(t.Context(), read, f, baseline.lexical.Schema())
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	calls := 0
	ports := contractSummaryPorts()
	ports.model = func(context.Context, graphsummary.ModelInput) (graphsummary.ModelOutput, graphsummary.Usage, error) {
		calls++
		return graphsummary.ModelOutput{}, graphsummary.Usage{}, nil
	}
	// Act.
	sample, err := prepared.summary(ctx, read, f.Queries[1], ports)
	// Assert.
	if !access.IsProtectionFailure(err) || calls != 0 || sample.ModelCalls != 0 {
		t.Fatal(sample, err, calls)
	}
}

func summaryQuery(t *testing.T, f fixture, profile string) query {
	t.Helper()
	for _, row := range f.Queries {
		if row.Recipe == profile {
			return row
		}
	}
	t.Fatal("missing summary question")
	return query{}
}

func TestActualSummarySourceRetirementDuringTokenCountingPreventsDispatch(t *testing.T) {
	// Arrange: tokenizer is host code and may observe a concurrent source retirement.
	f, corpus, baseline, read := publishedLocalFixture(t)
	prepared, err := corpus.summarySources(t.Context(), read, f, baseline.lexical.Schema())
	if err != nil {
		t.Fatal(err)
	}
	ports := contractSummaryPorts()
	ports.count = func(graphsummary.ModelInput) (uint64, error) {
		retireSummaryGraphSource(t, corpus, "s1")
		return 20, nil
	}
	// Act.
	sample, err := prepared.summary(t.Context(), read, f.Queries[1], ports)
	// Assert: fresh original admission after counting denies before model dispatch.
	if err != nil || !sample.Failed || sample.ModelCalls != 0 || !sample.CallsKnown || !sample.UsageKnown ||
		sample.InputTokens != 0 || sample.OutputTokens != 0 || sample.Cost != 0 || len(sample.Supports) != 0 {
		t.Fatal(sample, err)
	}
}

func TestActualSummaryConcurrentAttemptsKeepIndependentSourceCounters(t *testing.T) {
	// Arrange: shared pinned membership; fresh per-attempt source reader and ledger.
	f, corpus, baseline, read := publishedLocalFixture(t)
	prepared, err := corpus.summarySources(t.Context(), read, f, baseline.lexical.Schema())
	if err != nil {
		t.Fatal(err)
	}
	expected, err := prepared.summary(t.Context(), read, f.Queries[1], contractSummaryPorts())
	if err != nil || expected.Failed {
		t.Fatal(expected, err)
	}
	type execution struct {
		sample observation
		err    error
	}
	results := make(chan execution, 4)
	// Act: four concurrent requests against the same prepared membership.
	for range 4 {
		go func() {
			sample, runErr := prepared.summary(t.Context(), read, f.Queries[1], contractSummaryPorts())
			results <- execution{sample, runErr}
		}()
	}
	// Assert: each receipt owns its source I/O and one settled model invocation.
	for range 4 {
		result := <-results
		if result.err != nil || result.sample.Failed || result.sample.ModelCalls != 1 ||
			result.sample.SourceMetadataCalls != expected.SourceMetadataCalls ||
			result.sample.SourcePayloadCalls != expected.SourcePayloadCalls || result.sample.Cost != summaryCallCost {
			t.Fatal(result, expected)
		}
	}
	if prepared.host.metadataCalls.Load() != 0 || prepared.host.payloadCalls.Load() != 0 {
		t.Fatal("shared attempt counters")
	}
}
