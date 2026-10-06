package main

import (
	"encoding/json"
	"errors"
	"math"
	"testing"

	"github.com/skosovsky/ragy/source"
)

func fixtureCapture(t *testing.T) capture {
	t.Helper()
	var f fixture
	if err := decodeStrict(fixtureJSON, &f); err != nil {
		t.Fatal(err)
	}
	controls, identity, err := configurationBytes()
	if err != nil {
		t.Fatal(err)
	}
	raw := capture{
		FixtureIdentity:   f.Identity,
		ExecutionKind:     contractExecution,
		AdapterIdentity:   "test-only",
		ModelIdentity:     "scripted-test-only",
		TokenizerIdentity: "fixed-test-only",
		Configuration:     controls,
		ConfigIdentity:    identity,
	}
	raw.Preparation.ResolutionHistoryID = digest([]byte("contract-fixture-history"))
	raw.Preparation.MembershipGraphCalls = uint64(len(f.Communities))
	for _, src := range f.Sources {
		raw.Preparation.Extractions = append(
			raw.Preparation.Extractions,
			extractionObservation{
				Configuration: extractionConfigurationIdentity(),
				Reference:     originalReference(src.ID),
				ModelCalls:    1,
				InputTokens:   20,
				OutputTokens:  5,
				Cost:          summaryCallCost,
				UsageKnown:    true,
				Nanos:         1,
			},
		)
	}
	for _, q := range f.Queries {
		for _, profile := range []string{baselineProfile, q.Recipe} {
			row := observation{
				Query:   q.ID,
				Profile: profile,
				Outcome: completeOutcome, Stop: "test-fixture",
				UsageKnown:  true,
				CallsKnown:  true,
				Nanos:       1,
				Scope:       "scope",
				Publication: "publication",
			}
			for _, id := range q.Relevant {
				row.Supports = append(row.Supports, originalReference(id))
			}
			switch profile {
			case baselineProfile:
				row.RetrievalCalls = 2
			case localProfile:
				row.GraphCalls = 1
			case communityProfile:
				row.ModelCalls = 1
				row.InputTokens = 32
				row.OutputTokens = 16
				row.Cost = 30
			case globalProfile:
				row.ModelCalls = 3
				row.InputTokens = 96
				row.OutputTokens = 48
				row.Cost = 90
			}
			raw.Samples = append(raw.Samples, row)
		}
	}
	return raw
}
func TestSupportRecallIncludesMissingGoldAndRankCutoff(t *testing.T) {
	// Arrange: one relevant source outside K and two visible irrelevant sources.
	q := query{Relevant: []string{"s1", "s2"}}
	sample := observation{
		Supports: []source.Reference{
			originalReference("s3"),
			originalReference("s1"),
			originalReference("s4"),
			originalReference("s2"),
		},
	}
	// Act.
	value := recall(sample, q)
	// Assert: Recall counts unique gold sources in the bounded returned prefix.
	if value != 0.5 {
		t.Fatal(value)
	}
}
func TestEvaluatorPreservesMeasuredLossAndUnknownBudgets(t *testing.T) {
	// Arrange: a complete contract grid with missing local support and unknown cost.
	raw := fixtureCapture(t)
	raw.Samples[1].Supports = raw.Samples[1].Supports[:1]
	raw.Samples[1].UsageKnown = false
	// Act.
	result, err := evaluate(raw)
	// Assert: no fabricated improvement or promotion; unknown accounting fails budget acceptance.
	if err != nil || result.Measurements["local"].RecipeRecall != 0.5 || result.Measurements["local"].Gain != -0.5 ||
		result.Measurements["local"].RecipeBudgetsHonored ||
		result.DefaultProfile != baselineProfile {
		t.Fatal(result, err)
	}
}
func TestEvaluatorRejectsIncompleteForeignAndIncompatibleCaptures(t *testing.T) {
	for _, scenario := range []string{"missing-row", "duplicate-row", "foreign-reference", "wrong-revision", "duplicate-support", "failed-payload", "unknown-profile", "configuration", "scope-mismatch", "publication-mismatch", "missing-outcome"} {
		t.Run(scenario, func(t *testing.T) {
			// Arrange.
			raw := fixtureCapture(t)
			corruptCapture(&raw, scenario)
			// Act.
			_, err := evaluate(raw)
			// Assert.
			if !errors.Is(err, errInvalid) {
				t.Fatal(err)
			}
		})
	}
}
func corruptCapture(raw *capture, scenario string) {
	switch scenario {
	case "missing-row":
		raw.Samples = raw.Samples[:len(raw.Samples)-1]
	case "duplicate-row":
		raw.Samples[1] = raw.Samples[0]
	case "foreign-reference":
		raw.Samples[1].Supports[0] = originalReference("private-source")
	case "wrong-revision":
		raw.Samples[1].Supports[0].Revision = "v2"
	case "duplicate-support":
		raw.Samples[1].Supports[1] = raw.Samples[1].Supports[0]
	case "failed-payload":
		raw.Samples[1].Failed = true
	case "unknown-profile":
		raw.Samples[1].Profile = "unbounded-agent"
	case "scope-mismatch":
		raw.Samples[1].Scope = "other-policy"
	case "publication-mismatch":
		raw.Samples[1].Publication = "other-publication"
	case "missing-outcome":
		raw.Samples[1].Outcome = ""
	case "configuration":
		raw.Configuration = json.RawMessage(`{}`)
	}
}
func TestExactUsageRoundtripCannotHideBudgetOverflow(t *testing.T) {
	// Arrange.
	raw := fixtureCapture(t)
	raw.Samples[1].Cost = math.MaxUint64
	encoded, err := json.Marshal(raw)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	var decoded capture
	if err = decodeStrict(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	result, err := evaluate(decoded)
	// Assert.
	if err != nil || decoded.Samples[1].Cost != math.MaxUint64 || result.Measurements["local"].RecipeBudgetsHonored {
		t.Fatal(result, err)
	}
}

func TestUnknownDispatchCountersCannotQualifyBudgets(t *testing.T) {
	// Arrange: actual references may exist while dispatch accounting is unavailable.
	raw := fixtureCapture(t)
	raw.Samples[1].CallsKnown = false
	// Act.
	result, err := evaluate(raw)
	// Assert: retain retrieval metrics but do not certify unknown calls as zero.
	if err != nil || result.Measurements["local"].RecipeRecall != 1 ||
		result.Measurements["local"].RecipeBudgetsHonored {
		t.Fatal(result, err)
	}
}
