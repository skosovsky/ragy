package graphsummary_test

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/recipe"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/recipe/graphsummary"
	"github.com/skosovsky/ragy/source"
)

type concurrentSummaryResult struct {
	value graphsummary.Result
	err   error
}

func TestConcurrentCommunityVariantsCannotOverrunLastSharedModelCall(t *testing.T) {
	// Arrange: both actual recipe instances share one attempt ledger and immutable read.
	f := newFixture(t)
	var hostMu sync.Mutex
	membership, admitSource := f.config.Membership, f.config.AdmitSource
	f.config.Membership = func(ctx context.Context, read access.Binding, id string, members []string) error {
		hostMu.Lock()
		defer hostMu.Unlock()
		return membership(ctx, read, id, members)
	}
	f.config.AdmitSource = func(ctx context.Context, read access.Binding, loc source.Locator) error {
		hostMu.Lock()
		defer hostMu.Unlock()
		return admitSource(ctx, read, loc)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 3*time.Second)
	defer cancel()
	release := make(chan struct{})
	var releaseOnce sync.Once
	unblock := func() { releaseOnce.Do(func() { close(release) }) }
	defer unblock()
	calls, quotes := atomic.Int64{}, atomic.Int64{}
	quotesReady := make(chan struct{})
	quote := f.config.Quote
	f.config.Quote = func(callCtx context.Context, stage graphsummary.Stage) (budget.Reservation, error) {
		reservation, err := quote(callCtx, stage)
		if err != nil {
			return reservation, err
		}
		if quotes.Add(1) == 2 {
			close(quotesReady)
		}
		select {
		case <-quotesReady:
			return reservation, nil
		case <-callCtx.Done():
			return budget.Reservation{}, callCtx.Err()
		}
	}
	f.config.Model = func(callCtx context.Context, input graphsummary.ModelInput) (graphsummary.ModelOutput, graphsummary.Usage, error) {
		calls.Add(1)
		select {
		case <-release:
		case <-callCtx.Done():
			return graphsummary.ModelOutput{}, graphsummary.Usage{}, callCtx.Err()
		}
		return graphsummary.ModelOutput{
			Text:     input.Snippets[0].Text,
			Selected: []int{0},
		}, graphsummary.Usage{
			Known: true,
			Value: budget.Usage{InputTokens: 32, OutputTokens: 16, Cost: 30},
		}, nil
	}
	f.limits.Limits.ModelCalls = 1
	f.limits.Limits.Usage.Cost = 30
	ledger, err := budget.New(f.limits)
	if err != nil {
		t.Fatal(err)
	}
	instance, err := graphsummary.New(f.config)
	if err != nil {
		t.Fatal(err)
	}
	request := f.request
	request.Communities = request.Communities[:1]
	results := make(chan concurrentSummaryResult, 2)
	// Act: a quote barrier makes both variants race for the one remaining reservation.
	for range 2 {
		go func() {
			value, runErr := instance.Community(ctx, request, ledger)
			results <- concurrentSummaryResult{value: value, err: runErr}
		}()
	}
	denied := awaitConcurrentSummary(ctx, t, results)
	// Assert: the losing variant completes while the winning model is still in flight.
	if denied.err != nil || denied.value.Stop != graphsummary.BudgetExhausted ||
		denied.value.Outcome != recipe.Insufficient ||
		denied.value.ModelCalls != 0 ||
		len(denied.value.Communities) != 0 {
		t.Fatal("budget loser dispatched or claimed success", denied.err, denied.value)
	}
	unblock()
	completed := awaitConcurrentSummary(ctx, t, results)
	assertConcurrentSummaryBudget(t, completed, calls.Load(), quotes.Load(), ledger.Snapshot())
}

func awaitConcurrentSummary(
	ctx context.Context,
	t *testing.T,
	results <-chan concurrentSummaryResult,
) concurrentSummaryResult {
	t.Helper()
	select {
	case result := <-results:
		return result
	case <-ctx.Done():
		t.Fatal("concurrent summary did not finish", ctx.Err())
		return concurrentSummaryResult{}
	}
}

func assertConcurrentSummaryBudget(
	t *testing.T,
	completed concurrentSummaryResult,
	calls, quotes int64,
	snapshot budget.Snapshot,
) {
	t.Helper()
	if completed.err != nil || completed.value.Outcome != recipe.Complete ||
		completed.value.Stop != graphsummary.Summarized ||
		completed.value.ModelCalls != 1 ||
		len(completed.value.Communities) != 1 {
		t.Fatal("admitted summary failed", completed.err, completed.value)
	}
	if calls != 1 || quotes != 2 || snapshot.Occupied.ModelCalls != 1 || snapshot.Occupied.RetrievalCalls != 0 ||
		snapshot.Occupied.Usage.Cost != 30 ||
		snapshot.Actual != (budget.Usage{InputTokens: 32, OutputTokens: 16, Cost: 30}) ||
		snapshot.Outstanding != 0 ||
		snapshot.UnknownUsage != 0 {
		t.Fatal("shared ledger overrun, leaked lease or hidden retry", calls, quotes, snapshot)
	}
}
