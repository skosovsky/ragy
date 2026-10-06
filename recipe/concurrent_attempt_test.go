package recipe_test

import (
	"context"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/recipe"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/retrieval"
)

type synchronizedRecipeBackend struct {
	mu   *sync.Mutex
	next backend
}

func (b synchronizedRecipeBackend) Schema() filter.Schema { return b.next.Schema() }
func (b synchronizedRecipeBackend) ReadCapabilities() access.Capabilities {
	return b.next.ReadCapabilities()
}
func (b synchronizedRecipeBackend) Retrieve(ctx context.Context, req request) (retrieval.ResultSet[meta], error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.next.Retrieve(ctx, req)
}

type parallelRecipeResult struct {
	value recipe.Result[meta]
	err   error
}

func TestParallelRunsOfOneRecipeOwnIndependentAttempts(t *testing.T) {
	// Arrange: one immutable recipe instance, concurrency-safe host ports, separate attempts.
	f := newFixture(t, recipe.SingleRewrite)
	f.planned, f.selected = []string{"refund"}, []int{1}
	f.results["refund"] = []retrieval.Document[meta]{document("d1")}
	f.config.Limits.ModelCalls = 2
	f.config.Limits.RetrievalCalls = 2
	var mu sync.Mutex
	base, ok := f.config.Backend.(backend)
	if !ok {
		t.Fatal("backend fixture type")
	}
	f.config.Backend = synchronizedRecipeBackend{mu: &mu, next: base}
	plan, assess := f.config.Planner, f.config.Assessor
	f.config.Planner = func(ctx context.Context, req request, limits recipe.ModelLimits) (recipe.Planning, error) {
		mu.Lock()
		defer mu.Unlock()
		return plan(ctx, req, limits)
	}
	f.config.Assessor = func(ctx context.Context, input recipe.AssessmentInput[[]string, []string, meta], limits recipe.ModelLimits) (recipe.Assessment, error) {
		mu.Lock()
		defer mu.Unlock()
		return assess(ctx, input, limits)
	}
	pricing := f.config.Pricing
	var quotes atomic.Int64
	ready := make(chan struct{})
	f.config.Pricing = func(ctx context.Context, operation recipe.Operation) (recipe.Quote, error) {
		quote, err := pricing(ctx, operation)
		if err != nil || operation != recipe.Plan {
			return quote, err
		}
		if quotes.Add(1) == 2 {
			close(ready)
		}
		select {
		case <-ready:
			return quote, nil
		case <-ctx.Done():
			return recipe.Quote{}, ctx.Err()
		}
	}
	instance, err := recipe.New(f.config)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 3*time.Second)
	defer cancel()
	results := make(chan parallelRecipeResult, 2)
	requests := []request{
		{
			Read:    f.read,
			Text:    "original",
			Intent:  []string{"a"},
			Meta:    []string{"request-a"},
			Options: retrieval.RetrieveOptions{TopK: 3},
		},
		{
			Read:    f.read,
			Text:    "original",
			Intent:  []string{"b"},
			Meta:    []string{"request-b"},
			Options: retrieval.RetrieveOptions{TopK: 3},
		},
	}
	// Act: both planning quotes rendezvous before their independently reserved dispatch.
	for _, req := range requests {
		go func() {
			value, runErr := instance.Run(ctx, req)
			results <- parallelRecipeResult{value: value, err: runErr}
		}()
	}
	first := awaitParallelRecipe(ctx, t, results)
	second := awaitParallelRecipe(ctx, t, results)
	// Assert: each attempt gets its own limit, without aliasing results or caller metadata.
	assertParallelRecipeBudget(t, first)
	assertParallelRecipeBudget(t, second)
	if quotes.Load() != 2 || f.modelCalls != 4 || len(f.retrieved) != 4 {
		t.Fatal("attempt calls lost, overrun or secretly retried")
	}
	first.value.Selected[0].Document.Meta.Tags[0] = "mutated-result"
	if second.value.Selected[0].Document.Meta.Tags[0] != "owned" || f.results["refund"][0].Meta.Tags[0] != "owned" ||
		!slices.Equal(requests[0].Meta, []string{"request-a"}) ||
		!slices.Equal(requests[1].Meta, []string{"request-b"}) {
		t.Fatal("parallel attempt ownership leaked")
	}
}
func awaitParallelRecipe(ctx context.Context, t *testing.T, results <-chan parallelRecipeResult) parallelRecipeResult {
	t.Helper()
	select {
	case result := <-results:
		return result
	case <-ctx.Done():
		t.Fatal("parallel recipe stalled", ctx.Err())
		return parallelRecipeResult{}
	}
}
func assertParallelRecipeBudget(t *testing.T, result parallelRecipeResult) {
	t.Helper()
	if result.err != nil || result.value.Outcome != recipe.Complete || result.value.Stop != recipe.Assessed ||
		len(result.value.Selected) != 1 {
		t.Fatal("independent attempt failed", result.err, result.value)
	}
	snapshot := result.value.Budget
	if snapshot.Occupied.ModelCalls != 2 || snapshot.Occupied.RetrievalCalls != 2 ||
		snapshot.Actual != (budget.Usage{InputTokens: 200, OutputTokens: 40, Cost: 60}) ||
		snapshot.Outstanding != 0 ||
		snapshot.UnknownUsage != 0 {
		t.Fatal("attempt ledger mixed with sibling", snapshot)
	}
}
