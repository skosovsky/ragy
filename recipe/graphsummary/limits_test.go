package graphsummary_test

import (
	"context"
	"errors"
	"strconv"
	"strings"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/recipe"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/recipe/graphsummary"
)

func TestGlobalExplicitCommunityAndCallBounds(t *testing.T) {
	for _, count := range []int{1, 2, 4} {
		t.Run(strconv.Itoa(count), func(t *testing.T) {
			// Arrange.
			f := newFixture(t)
			f.config.MaxCommunities = 4
			f.config.MaxModelCalls = 5
			f.config.Membership = func(context.Context, access.Binding, string, []string) error { return nil }
			base := f.request.Communities[0]
			f.request.Communities = nil
			for i := range count {
				c := base
				c.ID = strconv.Itoa(i)
				f.request.Communities = append(f.request.Communities, c)
			}
			f.limits.Limits.ModelCalls = 5
			f.limits.Limits.Usage = budget.Usage{InputTokens: 10000, OutputTokens: 10000, Cost: 1000}
			f.config.Model = func(_ context.Context, in graphsummary.ModelInput) (graphsummary.ModelOutput, graphsummary.Usage, error) {
				f.calls++
				ids := make([]int, len(in.Snippets))
				for i := range ids {
					ids[i] = i
				}
				return graphsummary.ModelOutput{Text: "summary", Selected: ids}, graphsummary.Usage{}, nil
			}
			// Act.
			result, ledger, err := run(context.Background(), t, f, true)
			// Assert.
			if err != nil || result.Global == nil || result.Outcome != recipe.Complete || f.calls != count+1 ||
				ledger.Snapshot().Occupied.ModelCalls != uint64(count+1) {
				t.Fatal(result, err, f.calls, ledger.Snapshot())
			}
		})
	}
}

func TestGlobalRejectsCommunityOrCallOverflowBeforeDispatch(t *testing.T) {
	for _, mode := range []string{"communities", "calls"} {
		t.Run(mode, func(t *testing.T) {
			// Arrange.
			f := newFixture(t)
			if mode == "communities" {
				f.config.MaxCommunities = 1
			} else {
				f.config.MaxModelCalls = 2
			}
			// Act.
			_, ledger, err := run(context.Background(), t, f, true)
			// Assert.
			if !errors.Is(err, ragy.ErrInvalidArgument) || f.calls != 0 || ledger.Snapshot().Occupied.ModelCalls != 0 {
				t.Fatal(err, f.calls, ledger.Snapshot())
			}
		})
	}
}

func TestReducerCountsCompleteSerializedEnvelopeBeforeDispatch(t *testing.T) {
	// Arrange: raw summaries plus question fit, but their JSON representation does not.
	f := newFixture(t)
	f.config.MaxInputBytes = 512
	f.config.MaxSummaryBytes = 220
	f.config.Model = func(_ context.Context, _ graphsummary.ModelInput) (graphsummary.ModelOutput, graphsummary.Usage, error) {
		f.calls++
		return graphsummary.ModelOutput{Text: strings.Repeat("\"", 200), Selected: []int{0}}, graphsummary.Usage{}, nil
	}
	// Act.
	result, ledger, err := run(context.Background(), t, f, true)
	// Assert: both maps fit; quoted reducer text is doubled by JSON escaping.
	if err != nil || f.calls != 2 || result.Global != nil || result.Stop != graphsummary.BudgetExhausted ||
		result.Outcome != recipe.Partial ||
		ledger.Snapshot().Occupied.ModelCalls != 2 {
		t.Fatal(result, err, f.calls, ledger.Snapshot())
	}
}

func TestSharedDeadlineCancelsGraphModelCooperatively(t *testing.T) {
	// Arrange.
	f := newFixture(t)
	f.limits.Deadline = f.now.Add(20 * time.Millisecond)
	var observed time.Time
	f.config.Model = func(ctx context.Context, _ graphsummary.ModelInput) (graphsummary.ModelOutput, graphsummary.Usage, error) {
		observed, _ = ctx.Deadline()
		<-ctx.Done()
		return graphsummary.ModelOutput{}, graphsummary.Usage{}, ctx.Err()
	}
	// Act.
	_, ledger, err := run(context.Background(), t, f, false)
	// Assert.
	if !errors.Is(err, context.DeadlineExceeded) || observed.IsZero() || time.Until(observed) > 100*time.Millisecond ||
		ledger.Snapshot().Occupied.ModelCalls != 1 ||
		ledger.Snapshot().UnknownUsage != 1 {
		t.Fatal(err, observed, ledger.Snapshot())
	}
}

func TestHostMayAdmitMoreThanTwentySnippets(t *testing.T) {
	// Arrange.
	f := newFixture(t)
	f.config.MaxSnippets = 21
	snippet := f.request.Communities[0].Snippets[0]
	f.request.Communities[0].Snippets = make([]graphsummary.Snippet[acl], 21)
	for i := range f.request.Communities[0].Snippets {
		f.request.Communities[0].Snippets[i] = snippet
	}
	// Act.
	result, _, err := run(context.Background(), t, f, false)
	// Assert.
	if err != nil || result.Outcome != recipe.Complete || f.calls != 1 {
		t.Fatal(result, err, f.calls)
	}
}
