package graphsummary_test

import (
	"context"
	"errors"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/recipe/graphsummary"
)

func TestModelCausesSurviveSummaryExpiry(t *testing.T) {
	for _, cause := range []error{ragy.ErrProtocol, errors.New("model failed"), access.NonSkippable(ragy.ErrUnavailable), errors.Join(access.NonSkippable(ragy.ErrUnavailable), context.DeadlineExceeded), errors.Join(errors.New("ordinary"), budget.ErrExhausted)} {
		// Arrange: failed model also overruns and crosses the local clock deadline.
		f := newFixture(t)
		f.config.Model = func(context.Context, graphsummary.ModelInput) (graphsummary.ModelOutput, graphsummary.Usage, error) {
			f.calls++
			f.now = f.now.Add(6 * time.Second)
			return graphsummary.ModelOutput{}, graphsummary.Usage{
				Known: true,
				Value: budget.Usage{InputTokens: 1025, OutputTokens: 20, Cost: 30},
			}, cause
		}
		// Act.
		result, ledger, err := run(context.Background(), t, f, false)
		// Assert.
		if !errors.Is(err, cause) || !errors.Is(err, context.DeadlineExceeded) ||
			!errors.Is(err, budget.ErrUsageExceeded) ||
			len(result.Communities) != 0 ||
			f.calls != 1 ||
			ledger.Snapshot().Outstanding != 0 {
			t.Fatal("summary causes lost", err)
		}
		if access.IsProtectionFailure(cause) && !access.IsProtectionFailure(err) {
			t.Fatal("protection lost", err)
		}
	}
}
