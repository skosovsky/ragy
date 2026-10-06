package extraction_test

import (
	"context"
	"errors"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/graphingest/extraction"
	"github.com/skosovsky/ragy/recipe/budget"
)

func TestModelCausesSurviveExtractionExpiry(t *testing.T) {
	for _, cause := range []error{ragy.ErrProtocol, errors.New("model failed"), access.NonSkippable(ragy.ErrUnavailable), errors.Join(access.NonSkippable(ragy.ErrUnavailable), context.DeadlineExceeded)} {
		// Arrange: callback completes after local duration with known usage overrun.
		f := newFixture(t)
		f.config.Model = func(context.Context, extraction.ModelInput) (extraction.ModelOutput[string, string, attributes], extraction.Usage, error) {
			f.calls++
			f.now = f.now.Add(6 * time.Second)
			return f.output, extraction.Usage{
				Known: true,
				Value: budget.Usage{InputTokens: 101, OutputTokens: 10, Cost: 30},
			}, cause
		}
		adapter, err := extraction.New(f.config)
		if err != nil {
			t.Fatal(err)
		}
		// Act.
		result, err := adapter.Extract(context.Background(), f.read, f.ledger, f.input)
		// Assert: every independent cause and conservative settlement survives zero delivery.
		if !errors.Is(err, cause) || !errors.Is(err, context.DeadlineExceeded) ||
			!errors.Is(err, budget.ErrUsageExceeded) ||
			len(result.Extraction.Entities) != 0 ||
			f.calls != 1 ||
			f.ledger.Snapshot().Outstanding != 0 {
			t.Fatal("extraction cause/accounting lost", err)
		}
		if access.IsProtectionFailure(cause) && !access.IsProtectionFailure(err) {
			t.Fatal("protection classification lost", err)
		}
	}
}
