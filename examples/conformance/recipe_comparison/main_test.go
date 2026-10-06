package main

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/skosovsky/ragy/source"
)

func comparisonCapture(t *testing.T) capture {
	t.Helper()
	var f fixture
	if err := decodeStrict(fixtureJSON, &f); err != nil {
		t.Fatal(err)
	}
	raw := capture{
		FixtureIdentity:   f.Identity,
		AdapterIdentity:   "actual-bm25",
		ModelIdentity:     "scripted-for-evaluator-test",
		TokenizerIdentity: "test-only",
		Configuration:     referenceConfiguration(),
		ConfigIdentity:    configurationIdentity(referenceConfiguration()),
		ExecutionKind:     "contract-fixture",
	}
	for _, strategy := range profiles() {
		for _, q := range f.Queries {
			sample := observation{
				Query:          q.ID,
				Strategy:       strategy,
				Outcome:        "complete",
				Stop:           "test-fixture",
				RetrievalCalls: 1,
				UsageKnown:     true,
				Nanos:          100,
				Scope:          "policy",
				Publication:    "pub1",
			}
			if strategy != "baseline" {
				sample.IDs = q.Relevant
				sample.ModelCalls = 2
				sample.InputTokens = 200
				sample.OutputTokens = 40
				sample.Cost = 60
			}
			for _, id := range sample.IDs {
				sample.SourceRefs = append(sample.SourceRefs, corpusReference(id))
			}
			raw.Samples = append(raw.Samples, sample)
		}
	}
	return raw
}
func TestRankMetricsAndNoAnswerDenominator(t *testing.T) {
	// Arrange: d4 is the first relevant hit, d1 falls outside TopK.
	ids := []string{"d3", "d4", "d2", "d1"}
	// Act.
	recall, mrr := rankMetrics(ids, []string{"d1", "d4"})
	// Assert: Recall is document-based, MRR is the first relevant rank, no truncation inflation.
	if recall != 0.5 || mrr != 0.5 {
		t.Fatal(recall, mrr)
	}
	raw := comparisonCapture(t)
	result, err := evaluate(raw)
	if err != nil {
		t.Fatal(err)
	}
	q := result.Quality["single-rewrite"]
	if q.Recall != 1 || q.MRR != 1 || q.NoAnswerErrors != 0 || !q.NumericThresholdsMet ||
		result.DefaultProfile != "baseline" {
		t.Fatal(q, result.DefaultProfile)
	}
}
func TestRecommendationRefusedForUnknownUsageFailuresAndNoAnswerRegression(t *testing.T) {
	for _, scenario := range []string{"unknown-usage", "failure", "no-answer", "budget", "deadline", "mrr-regression"} {
		t.Run(scenario, func(t *testing.T) {
			// Arrange.
			raw := comparisonCapture(t)
			corruptComparison(raw.Samples, scenario)
			// Act.
			result, err := evaluate(raw)
			// Assert: incomplete accounting and measured regressions never meet the recommendation gate.
			if err != nil {
				t.Fatal(err)
			}
			if result.Quality["single-rewrite"].NumericThresholdsMet || result.DefaultProfile != "baseline" {
				t.Fatal(result.Quality)
			}
		})
	}
}
func corruptComparison(samples []observation, scenario string) {
	for i := range samples {
		sample := &samples[i]
		if sample.Strategy == "single-rewrite" && sample.Query == "initial-miss" {
			switch scenario {
			case "unknown-usage":
				sample.UsageKnown = false
			case "failure":
				sample.Failed = true
				sample.Outcome = failedOutcome
				sample.IDs = nil
			case "budget":
				sample.ModelCalls = 3
			case "deadline":
				sample.Nanos = 5_000_000_001
			case "mrr-regression":
				sample.IDs = []string{"d3", "d1"}
			}
		}
		if scenario == "no-answer" && sample.Strategy == "single-rewrite" && sample.Query == "no-answer" {
			sample.IDs = []string{"d1"}
		}
		if scenario == "mrr-regression" && sample.Strategy == "baseline" {
			sample.IDs = []string{"d1"}
			if sample.Query == "multi-query" {
				sample.IDs = []string{"d2"}
			}
			if sample.Query == "no-answer" {
				sample.IDs = nil
			}
		}
		refreshComparisonSources(sample)
	}
}
func TestIncompleteUnknownForeignAndDuplicateCaptureRejected(t *testing.T) {
	for _, scenario := range []string{"missing", "duplicate-row", "duplicate-hit", "foreign-hit", "query", "strategy", "identity", "negative-time", "failed-hits", "foreign-ref"} {
		t.Run(scenario, func(t *testing.T) {
			// Arrange.
			raw := comparisonCapture(t)
			invalidComparison(&raw, scenario)
			// Act.
			_, err := evaluate(raw)
			// Assert.
			if !errors.Is(err, errInvalid) {
				t.Fatal("invalid experiment accepted", err)
			}
		})
	}
}
func invalidComparison(raw *capture, scenario string) {
	switch scenario {
	case "missing":
		raw.Samples = raw.Samples[1:]
	case "duplicate-row":
		raw.Samples[0] = raw.Samples[1]
	case "duplicate-hit":
		raw.Samples[0].IDs = []string{"d1", "d1"}
	case "foreign-hit":
		raw.Samples[0].IDs = []string{"secret"}
	case "query":
		raw.Samples[0].Query = "absent"
	case "strategy":
		raw.Samples[0].Strategy = "absent"
	case "identity":
		raw.FixtureIdentity = "another-fixture"
	case "negative-time":
		raw.Samples[0].Nanos = -1
	case "foreign-ref":
		raw.Samples[0].SourceRefs = []source.Reference{corpusReference("d1")}
		raw.Samples[0].SourceRefs[0].Source = "another-source"
	case "failed-hits":
		raw.Samples[0].Failed = true
		raw.Samples[0].IDs = []string{"d1"}
	}
}
func TestEvaluatorExecutableStrictInputAndRawUsageRoundtrip(t *testing.T) {
	// Arrange: evaluator contract fixture, explicitly not a quality experiment.
	raw := comparisonCapture(t)
	raw.Samples[0].InputTokens = math.MaxUint64
	data, err := json.Marshal(raw)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	input, output := filepath.Join(dir, "capture.json"), filepath.Join(dir, "report.json")
	if err = os.WriteFile(input, data, 0o600); err != nil {
		t.Fatal(err)
	}
	// Act.
	if err = evaluateFile(input, output); err != nil {
		t.Fatal(err)
	}
	saved, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	var result report
	if err = decodeStrict(saved, &result); err != nil {
		t.Fatal(err)
	}
	// Assert: no lossy uint64 roundtrip or success from an out-of-budget baseline.
	if result.Capture.Samples[0].InputTokens != math.MaxUint64 || result.Quality["baseline"].BudgetsHonored {
		t.Fatal(result)
	}
	var parsed capture
	for _, invalid := range []string{`{"unknown":true}`, string(data) + ` {}`} {
		if decodeStrict([]byte(invalid), &parsed) == nil {
			t.Fatal("invalid wire capture accepted")
		}
	}
}

func refreshComparisonSources(sample *observation) {
	sample.SourceRefs = nil
	for _, id := range sample.IDs {
		sample.SourceRefs = append(sample.SourceRefs, corpusReference(id))
	}
}
