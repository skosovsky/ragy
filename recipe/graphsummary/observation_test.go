package graphsummary_test

import (
	"context"
	"errors"
	"testing"

	"github.com/skosovsky/ragy/observation"
	"github.com/skosovsky/ragy/recipe"
	"github.com/skosovsky/ragy/recipe/graphsummary"
)

//nolint:gocognit // Matrix asserts actual dispatch counts, terminal outcomes and exporter failure together.
func TestActualSummaryDiagnosticsAndExporterFailureNoReplay(t *testing.T) {
	for _, exhausted := range []bool{false, true} {
		t.Run(map[bool]string{false: "complete", true: "budget"}[exhausted], func(t *testing.T) {
			// Arrange.
			f := newFixture(t)
			if exhausted {
				f.limits.Limits.ModelCalls = 1
			}
			var events []observation.Event
			session, err := observation.New(
				observation.Config{
					MaxEvents: 64,
					Observer: observation.ObserverFunc(func(_ context.Context, event observation.Event) error {
						events = append(events, event)
						return errors.New("private-exporter-error")
					}),
				},
			)
			if err != nil {
				t.Fatal(err)
			}
			ctx := observation.WithSession(context.Background(), session)
			// Act.
			result, _, err := run(ctx, t, f, true)
			// Assert.
			if err != nil {
				t.Fatal(err)
			}
			expectedCalls := 3
			expectedOutcome := observation.OutcomeSuccess
			if exhausted {
				expectedCalls = 1
				expectedOutcome = observation.OutcomeExhausted
				if result.Outcome != recipe.Partial || result.Stop != graphsummary.BudgetExhausted {
					t.Fatal(result)
				}
			}
			actualCalls := 0
			ended := false
			for _, event := range events {
				if event.Kind != observation.KindEnd {
					continue
				}
				if event.Stage == observation.StageModel {
					actualCalls++
					if !event.Completion.Usage.InputTokens.Known || event.Completion.Usage.BilledUnits.Known {
						t.Fatal(event)
					}
				}
				if event.Stage == observation.StagePipeline {
					ended = true
					if event.Completion.Outcome != expectedOutcome {
						t.Fatal(event)
					}
				}
			}
			if actualCalls != expectedCalls || f.calls != expectedCalls || !ended {
				t.Fatal(actualCalls, f.calls, events)
			}
		})
	}
}
