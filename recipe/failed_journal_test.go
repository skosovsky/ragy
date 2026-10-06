package recipe_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/evidence"
	"github.com/skosovsky/ragy/recipe"
	"github.com/skosovsky/ragy/recipe/recording"
	"github.com/skosovsky/ragy/retrieval"
)

func failingModel(f *fixture, operation recipe.Operation) {
	f.planned, f.selected = []string{"refund"}, []int{1}
	f.results["original"], f.results["refund"] = []retrieval.Document[meta]{
		document("d0"),
	}, []retrieval.Document[meta]{
		document("d1"),
	}
	if operation == recipe.Plan {
		f.config.Planner = func(context.Context, request, recipe.ModelLimits) (recipe.Planning, error) {
			f.modelCalls++
			return recipe.Planning{Usage: observed()}, errors.Join(ragy.ErrProtocol, errors.New("private-model-error"))
		}
	} else {
		f.config.Assessor = func(context.Context, recipe.AssessmentInput[[]string, []string, meta], recipe.ModelLimits) (recipe.Assessment, error) {
			f.modelCalls++
			return recipe.Assessment{Usage: observed()}, ragy.ErrProtocol
		}
	}
}

func TestFailedRecipeRecordingRetainsActualObservations(t *testing.T) {
	for _, operation := range []recipe.Operation{recipe.Plan, recipe.Assess} {
		t.Run(string(operation), func(t *testing.T) {
			// Arrange.
			f := newFixture(t, recipe.SingleRewrite)
			failingModel(f, operation)
			sink := &recordSink{}
			cfg := recordingConfig(t, f, evidence.Required, sink)
			// Act.
			execution, err := recording.Run(context.Background(), recordedRequest(f), cfg)
			// Assert: failed call is recorded once, prior hits retained only as observations.
			expected := 1
			if operation == recipe.Assess {
				expected = 2
			}
			if !errors.Is(err, ragy.ErrProtocol) || sink.calls != 1 || f.modelCalls != expected ||
				execution.Result.Outcome != recipe.Failure ||
				len(execution.Result.Selected) != 0 ||
				len(execution.Result.Queries) != expected {
				t.Fatal("failed attempt journal lost", err)
			}
			if execution.Result.Budget.Actual.Cost != uint64(expected*30) ||
				execution.Result.Fusion != recipe.FusionNotRun {
				t.Fatal("failed-call accounting/fusion fabricated")
			}
			assertFailedJournal(t, sink.record, operation, expected)
			encoded, marshalErr := sink.record.MarshalJSON()
			if marshalErr != nil || strings.Contains(string(encoded), "private-model-error") {
				t.Fatal("error diagnostic leaked", marshalErr)
			}
			execution.Result.Queries[0].Documents[0].Meta.Tags[0] = "changed"
			after, marshalErr := sink.record.MarshalJSON()
			if marshalErr != nil || string(after) != string(encoded) ||
				f.results["original"][0].Meta.Tags[0] != "owned" {
				t.Fatal("journal ownership failed", marshalErr)
			}
		})
	}
}

func assertFailedJournal(t *testing.T, record evidence.Record, operation recipe.Operation, expected int) {
	t.Helper()
	snapshot, err := record.Snapshot()
	if err != nil || snapshot.Outcome != evidence.Failed || snapshot.Reason != evidence.TargetFailure {
		t.Fatal("failure claimed success", err)
	}
	stages := map[string]evidence.WireStage{}
	for _, stage := range snapshot.Stages {
		if stage.Name.Value != nil {
			stages[*stage.Name.Value] = stage
		}
	}
	if stages["retrieve/0"].Status != evidence.StageObserved || len(stages["retrieve/0"].Hits) != 1 ||
		stages["plan"].Status != evidence.StageObserved ||
		stages["fusion"].Status != evidence.NotRun {
		t.Fatal("actual stage association lost")
	}
	if operation == recipe.Plan && stages["assess"].Status != evidence.NotRun {
		t.Fatal("unstarted assessor marked observed")
	}
	if operation == recipe.Assess &&
		(stages["assess"].Status != evidence.StageObserved || len(stages["retrieve/1"].Hits) != 1) {
		t.Fatal("failed assessor lost previous retrieval")
	}
	for _, diag := range snapshot.Diagnostics {
		if diag.Kind == evidence.ModelCalls && (diag.Number.Value == nil || *diag.Number.Value != float64(expected)) {
			t.Fatal("failed model omitted from diagnostics")
		}
	}
}

func TestRunObservedProtectionSuppressesJournal(t *testing.T) {
	// Arrange.
	f := newFixture(t, recipe.SingleRewrite)
	f.results["original"] = []retrieval.Document[meta]{document("d0")}
	f.config.Planner = func(context.Context, request, recipe.ModelLimits) (recipe.Planning, error) {
		f.modelCalls++
		f.epoch++
		return recipe.Planning{Usage: observed()}, ragy.ErrProtocol
	}
	r, err := recipe.New(f.config)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	result, err := r.RunOwnObserved(context.Background(), recordedRequest(f))
	// Assert.
	if !access.IsProtectionFailure(err) || len(result.Queries) != 0 || len(result.Stages) != 0 ||
		result.Publication != "" {
		t.Fatal("revocation retained journal", err)
	}
}

func TestFailedRetrievalObservationIsNotObservedEmpty(t *testing.T) {
	// Arrange: the second retrieval executes but cannot retain valid documents.
	f := newFixture(t, recipe.SingleRewrite)
	f.planned = []string{"refund"}
	f.results["original"] = []retrieval.Document[meta]{document("d0")}
	malformed := document("d1")
	malformed.ID = ""
	f.results["refund"] = []retrieval.Document[meta]{malformed}
	sink := &recordSink{}
	cfg := recordingConfig(t, f, evidence.Required, sink)
	// Act.
	execution, err := recording.Run(context.Background(), recordedRequest(f), cfg)
	// Assert.
	if err == nil || sink.calls != 1 || len(execution.Result.Queries) != 1 || len(execution.Result.Stages) != 3 ||
		f.modelCalls != 1 {
		t.Fatal("failed retrieval journal fabricated output", err)
	}
	snapshot, decodeErr := sink.record.Snapshot()
	if decodeErr != nil {
		t.Fatal(decodeErr)
	}
	stages := map[string]evidence.WireStage{}
	for _, stage := range snapshot.Stages {
		if stage.Name.Value != nil {
			stages[*stage.Name.Value] = stage
		}
	}
	if stages["retrieve/0"].Status != evidence.StageObserved || len(stages["retrieve/0"].Hits) != 1 ||
		stages["retrieve/1"].Status != evidence.MissingObservation ||
		stages["retrieve/1"].HitsState != evidence.Unavailable ||
		stages["assess"].Status != evidence.NotRun {
		t.Fatal("missing retrieval was represented as success")
	}
}

func TestRunObservedParentCancellationSuppressesJournal(t *testing.T) {
	// Arrange.
	f := newFixture(t, recipe.SingleRewrite)
	f.results["original"] = []retrieval.Document[meta]{document("d0")}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	f.planHook = func(request) { cancel() }
	r, err := recipe.New(f.config)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	result, err := r.RunOwnObserved(ctx, recordedRequest(f))
	// Assert.
	if !errors.Is(err, context.Canceled) || len(result.Queries) != 0 || len(result.Stages) != 0 ||
		result.Publication != "" {
		t.Fatal("parent cancellation retained journal", err)
	}
}
