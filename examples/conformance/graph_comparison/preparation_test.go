package main

import "testing"

func TestPreparationRejectsMissingDuplicateAndForeignOriginalSources(t *testing.T) {
	for _, scenario := range []string{"missing", "duplicate", "foreign", "failed", "membership"} {
		t.Run(scenario, func(t *testing.T) {
			// Arrange.
			raw := fixtureCapture(t)
			switch scenario {
			case "missing":
				raw.Preparation.Extractions = raw.Preparation.Extractions[:3]
			case "duplicate":
				raw.Preparation.Extractions[1] = raw.Preparation.Extractions[0]
			case "foreign":
				raw.Preparation.Extractions[0].Reference = originalReference("foreign")
			case "failed":
				raw.Preparation.Extractions[0].Failed = true
			case "membership":
				raw.Preparation.MembershipGraphCalls = 0
			}
			// Act/Assert.
			if _, err := evaluate(raw); err == nil {
				t.Fatal("invalid preparation accepted")
			}
		})
	}
}
func TestPreparationUnknownAndExceededAccountingCannotCertifyBudget(t *testing.T) {
	for _, scenario := range []string{"unknown", "input", "output", "cost", "calls", "latency"} {
		t.Run(scenario, func(t *testing.T) {
			// Arrange.
			raw := fixtureCapture(t)
			row := &raw.Preparation.Extractions[0]
			switch scenario {
			case "unknown":
				row.UsageKnown = false
			case "input":
				row.InputTokens = extractionInputTokens + 1
			case "output":
				row.OutputTokens = extractionOutputTokens + 1
			case "cost":
				row.Cost = referenceCostCap + 1
			case "calls":
				row.ModelCalls = 2
			case "latency":
				row.Nanos = int64(attemptDuration) + 1
			}
			// Act.
			report, err := evaluate(raw)
			// Assert: preserve raw observations and metrics, never certify unavailable usage.
			if err != nil || report.PreparationBudgetsHonored {
				t.Fatal(report, err)
			}
		})
	}
}
func TestLiveCaptureRequiresObservedPreparationTransport(t *testing.T) {
	// Arrange: changing a metadata label cannot convert a contract fixture to live evidence.
	raw := fixtureCapture(t)
	raw.ExecutionKind = liveExecution
	// Act/Assert.
	if _, err := evaluate(raw); err == nil {
		t.Fatal("unobserved live transport accepted")
	}
}

func TestObservedRecipeTransportCannotExceedModelInvocationBudget(t *testing.T) {
	// Arrange: the raw HTTP count contradicts a lower logical invocation count.
	raw := fixtureCapture(t)
	raw.Samples[3].TransportCallsKnown = true
	raw.Samples[3].TransportCalls = 3
	// Act.
	report, err := evaluate(raw)
	// Assert: retain the observation but reject budget certification.
	if err != nil || report.Measurements["community"].RecipeBudgetsHonored {
		t.Fatal(report, err)
	}
}
