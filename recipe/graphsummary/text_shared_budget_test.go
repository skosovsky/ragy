package graphsummary_test

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/recipe"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/recipe/graphsummary"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

type emptyTextBackend struct{}

func (emptyTextBackend) Retrieve(
	ctx context.Context,
	request retrieval.Request[struct{}, struct{}],
) (retrieval.ResultSet[struct{}], error) {
	return retrieval.NewResultSet[struct{}](nil, nil), request.Read.Check(ctx)
}

func TestActualTextAndGraphRecipesShareLastSlotWithoutRefund(t *testing.T) {
	// Arrange: actual text planner and graph mapper race under one ledger.
	f := newFixture(t)
	f.limits.Limits.ModelCalls = 1
	f.limits.Limits.RetrievalCalls = 1
	ledger, err := budget.New(f.limits)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()
	ready := make(chan struct{})
	release := make(chan struct{})
	defer close(release)
	var quotes, calls atomic.Int64
	fail := errors.New("model failed with unknown accounting")
	barrier := func(callCtx context.Context) error {
		if quotes.Add(1) == 2 {
			close(ready)
		}
		select {
		case <-ready:
			return nil
		case <-callCtx.Done():
			return callCtx.Err()
		}
	}
	model := func(callCtx context.Context) error {
		calls.Add(1)
		select {
		case <-release:
			return fail
		case <-callCtx.Done():
			return callCtx.Err()
		}
	}
	quote := f.config.Quote
	f.config.Quote = func(callCtx context.Context, stage graphsummary.Stage) (budget.Reservation, error) {
		if barrierErr := barrier(callCtx); barrierErr != nil {
			return budget.Reservation{}, barrierErr
		}
		return quote(callCtx, stage)
	}
	f.config.Model = func(callCtx context.Context, _ graphsummary.ModelInput) (graphsummary.ModelOutput, graphsummary.Usage, error) {
		return graphsummary.ModelOutput{}, graphsummary.Usage{}, model(callCtx)
	}
	graph, err := graphsummary.New(f.config)
	if err != nil {
		t.Fatal(err)
	}
	text := newSharedTextRecipe(t, f, barrier, model)
	completed := make(chan error, 2)
	// Act.
	go func() {
		req := f.request
		req.Communities = req.Communities[:1]
		_, runErr := graph.Community(ctx, req, ledger)
		completed <- runErr
	}()
	go func() {
		_, runErr := text.Run(
			ctx,
			retrieval.Request[struct{}, struct{}]{
				Read:    f.request.Read,
				Text:    "question",
				Options: retrieval.RetrieveOptions{TopK: 1},
			},
			ledger,
		)
		completed <- runErr
	}()
	// Assert: loser returns while the admitted model is held; cancellation then settles unknown.
	select {
	case runErr := <-completed:
		if runErr != nil {
			t.Fatal(runErr)
		}
	case <-ctx.Done():
		t.Fatal("loser did not finish")
	}
	cancel()
	select {
	case <-completed:
	case <-time.After(time.Second):
		t.Fatal("winner did not join")
	}
	snapshot := ledger.Snapshot()
	if calls.Load() != 1 || quotes.Load() != 2 || snapshot.Occupied.ModelCalls != 1 || snapshot.Outstanding != 0 ||
		snapshot.UnknownUsage != 1 {
		t.Fatal(calls.Load(), quotes.Load(), snapshot)
	}
	_, reserveErr := ledger.Reserve(context.Background(), budget.Reservation{Kind: budget.Model, CostKnown: true})
	if !errors.Is(reserveErr, budget.ErrExhausted) {
		t.Fatal("failed call refunded its slot", reserveErr)
	}
}

func newSharedTextRecipe(
	t *testing.T,
	f *fixture,
	barrier, model func(context.Context) error,
) *recipe.Recipe[struct{}, struct{}, struct{}] {
	t.Helper()
	clone := func(v struct{}) (struct{}, error) { return v, nil }
	cfg := recipe.Config[struct{}, struct{}, struct{}]{
		Strategy:         recipe.SingleRewrite,
		BackendModelFree: true,
		Revision:         "shared-budget",
		Backend:          emptyTextBackend{},
		Identity:         retrieval.DocumentIDResolver[struct{}]{},
		Admission: func(ctx context.Context, req retrieval.Request[struct{}, struct{}]) (retrieval.ReadCoverage, error) {
			return retrieval.CompleteReadCoverage(), req.Read.Check(ctx)
		},
		Planner: func(ctx context.Context, _ retrieval.Request[struct{}, struct{}], _ recipe.ModelLimits) (recipe.Planning, error) {
			return recipe.Planning{}, model(ctx)
		},
		Assessor: func(context.Context, recipe.AssessmentInput[struct{}, struct{}, struct{}], recipe.ModelLimits) (recipe.Assessment, error) {
			return recipe.Assessment{}, nil
		},
		Pricing: func(ctx context.Context, op recipe.Operation) (recipe.Quote, error) {
			if op == recipe.Retrieve {
				return recipe.Quote{CostKnown: true}, nil
			}
			if err := barrier(ctx); err != nil {
				return recipe.Quote{}, err
			}
			return recipe.Quote{
				Usage:     budget.Usage{InputTokens: 1024, OutputTokens: 256, Cost: 30},
				CostKnown: true,
			}, nil
		},
		CloneIntent:      clone,
		CloneRequestMeta: clone,
		CloneMeta:        clone,
		Supports: func(context.Context, access.Binding, retrieval.Document[struct{}]) ([]source.Locator, error) {
			return nil, nil
		},
		Duration:     5 * time.Second,
		Now:          func() time.Time { return f.now },
		MaxQueries:   1,
		MaxDocuments: 2,
		FusionK:      60,
	}
	instance, err := recipe.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return instance
}
