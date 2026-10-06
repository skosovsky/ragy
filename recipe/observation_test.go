package recipe_test

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/observation"
	"github.com/skosovsky/ragy/recipe"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/retrieval"
)

func observedContext(t *testing.T, fail bool) (context.Context, *[]observation.Event) {
	t.Helper()
	events := new([]observation.Event)
	session, err := observation.New(
		observation.Config{
			MaxEvents: 128,
			Observer: observation.ObserverFunc(func(_ context.Context, event observation.Event) error {
				*events = append(*events, event)
				if fail {
					return errors.New("private-exporter-body")
				}
				return nil
			}),
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	return observation.WithSession(context.Background(), session), events
}
func completedStages(events []observation.Event, stage observation.Stage) []observation.Event {
	var matches []observation.Event
	for _, event := range events {
		if event.Stage == stage && event.Kind == observation.KindEnd {
			matches = append(matches, event)
		}
	}
	return matches
}
func TestRecipeDiagnosticFailureDoesNotReplayDispatch(t *testing.T) {
	// Arrange.
	f := newFixture(t, recipe.SingleRewrite)
	f.results["original"] = []retrieval.Document[meta]{document("private-document")}
	f.selected = []int{0}
	ctx, events := observedContext(t, true)
	// Act.
	result, err := f.run(ctx, t)
	// Assert.
	if err != nil || result.Outcome != recipe.Complete || len(f.retrieved) != 1 || f.modelCalls != 2 {
		t.Fatal(result, err, f.retrieved, f.modelCalls)
	}
	models := completedStages(*events, observation.StageModel)
	if len(models) != 2 {
		t.Fatalf("model observations: %+v", models)
	}
	for _, event := range models {
		if event.Completion.Outcome != observation.OutcomeSuccess || !event.Completion.Usage.InputTokens.Known ||
			event.Completion.Usage.InputTokens.Value != 100 ||
			event.Completion.Usage.BilledUnits.Known {
			t.Fatalf("fabricated accounting: %+v", event)
		}
	}
	if len(completedStages(*events, observation.StageFusion)) != 1 ||
		len(completedStages(*events, observation.StageRetrieval)) != 1 {
		t.Fatal(*events)
	}
}
func TestRecipeEncodingUnknownUsageAndQueryOrdinal(t *testing.T) {
	// Arrange.
	f := newFixture(t, recipe.SingleRewrite)
	f.planned = []string{"private-rewrite"}
	f.selected = []int{1}
	f.results["private-rewrite"] = []retrieval.Document[meta]{document("one")}
	f.config.QueryEncoder = &boundedEncoder{}
	f.config.Limits.ModelCalls = 4
	f.config.Limits.Usage = budget.Usage{InputTokens: 4096, OutputTokens: 1024, Cost: 200}
	ctx, events := observedContext(t, false)
	// Act.
	result, err := f.run(ctx, t)
	// Assert.
	if err != nil || len(result.Encoding) != 2 {
		t.Fatal(result, err)
	}
	encodings := completedStages(*events, observation.StageEncoding)
	if len(encodings) != 2 {
		t.Fatal(encodings)
	}
	for index, event := range encodings {
		if !event.Query.Known || event.Query.Value != uint64(index) {
			t.Fatalf("query correlation: %+v", event)
		}
		var child *observation.Event
		for _, candidate := range *events {
			if candidate.Kind == observation.KindEnd && candidate.Stage == observation.StageModel &&
				candidate.Parent == event.Operation {
				value := candidate
				child = &value
			}
		}
		if child == nil || child.Completion.Usage.InputTokens.Known || child.Completion.Usage.OutputTokens.Known {
			t.Fatalf("unknown encoder usage fabricated: %+v", child)
		}
	}
}

//nolint:gocognit // Matrix asserts actual dispatch counts, terminal outcomes and exporter failure together.
func TestRecipeBudgetAndUnsupportedDispatchObservation(t *testing.T) {
	for _, unsupported := range []bool{false, true} {
		t.Run(map[bool]string{false: "budget", true: "unsupported"}[unsupported], func(t *testing.T) {
			// Arrange.
			f := newFixture(t, recipe.SingleRewrite)
			f.config.Limits.ModelCalls = 1
			if unsupported {
				f.config.QueryEncoder = recipe.UnsupportedQueryEncoderBridge{}
			}
			ctx, events := observedContext(t, false)
			// Act.
			result, err := f.run(ctx, t)
			// Assert.
			completed := completedStages(*events, observation.StagePipeline)
			if len(completed) != 1 {
				t.Fatal(completed)
			}
			if unsupported {
				if !errors.Is(err, ragy.ErrUnsupported) || f.modelCalls != 0 || len(f.retrieved) != 0 ||
					completed[0].Completion.Outcome != observation.OutcomeUnsupported {
					t.Fatal(err, *events)
				}
				if len(completedStages(*events, observation.StageModel)) != 0 {
					t.Fatal("unsupported admission dispatched model")
				}
			} else if err != nil || result.Stop != recipe.BudgetExhausted || f.modelCalls != 1 || completed[0].Completion.Outcome != observation.OutcomeExhausted {
				t.Fatal(result, err, *events)
			}
		})
	}
}
func TestRecipeFailedModelDoesNotInventUsageOrRetry(t *testing.T) {
	// Arrange.
	f := newFixture(t, recipe.SingleRewrite)
	calls := 0
	f.config.Planner = func(context.Context, request, recipe.ModelLimits) (recipe.Planning, error) {
		calls++
		return recipe.Planning{}, errors.New("secret-provider-error")
	}
	ctx, events := observedContext(t, false)
	// Act.
	_, err := f.run(ctx, t)
	// Assert.
	models := completedStages(*events, observation.StageModel)
	if err == nil || calls != 1 || len(models) != 1 || models[0].Completion.Outcome != observation.OutcomeFailed ||
		models[0].Completion.Usage.InputTokens.Known {
		t.Fatal(err, calls, models)
	}
}
func TestRecipeCancellationObservationSuppressesPayload(t *testing.T) {
	// Arrange.
	f := newFixture(t, recipe.SingleRewrite)
	ctx, events := observedContext(t, false)
	ctx, cancel := context.WithCancel(ctx)
	f.retrieveHook = func(context.Context) { cancel() }
	// Act.
	result, err := f.run(ctx, t)
	// Assert.
	pipelines := completedStages(*events, observation.StagePipeline)
	if !errors.Is(err, context.Canceled) || len(result.Queries) != 0 || f.modelCalls != 0 || len(pipelines) != 1 ||
		pipelines[0].Completion.Outcome != observation.OutcomeCanceled {
		t.Fatal(result, err, *events)
	}
}

var _ recipe.QueryEncoder = (*boundedEncoder)(nil)

func TestFailedDispatchStageIsNotSuccessfulEmptyObservation(t *testing.T) {
	// Arrange.
	f := newFixture(t, recipe.SingleRewrite)
	f.config.Planner = func(context.Context, request, recipe.ModelLimits) (recipe.Planning, error) {
		return recipe.Planning{}, errors.New("private-provider")
	}
	r, err := recipe.New(f.config)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	result, err := r.RunOwnObserved(context.Background(), recordedRequest(f))
	// Assert.
	if err == nil || len(result.Stages) != 2 || !result.Stages[0].Completed || result.Stages[1].Completed ||
		result.Stages[1].Usage.Known {
		t.Fatal(result, err)
	}
}
