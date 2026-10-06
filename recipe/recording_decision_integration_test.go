package recipe_test

import (
	"context"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/evidence"
	"github.com/skosovsky/ragy/recipe"
	"github.com/skosovsky/ragy/recipe/recording"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

func TestRecordingDecisionDistinguishesDirectAndMissingPackedDelivery(t *testing.T) {
	for _, mode := range []string{"direct", "deadline", "partial"} {
		t.Run(mode, func(t *testing.T) {
			// Arrange: all modes select the same actually retrieved document.
			f := newFixture(t, recipe.Decomposition)
			f.planned = []string{"one"}
			f.selected = []int{0}
			f.sufficient = true
			f.results["one"] = []retrieval.Document[meta]{document("d1")}
			switch mode {
			case "deadline":
				f.config.Duration = 35 * time.Millisecond
				resource := retrieval.RuneResource(100)
				resource.Measure = func(ctx context.Context, _ string) (int64, error) { <-ctx.Done(); return 0, ctx.Err() }
				f.config.Artifact = &retrieval.ArtifactRenderOptions[meta]{
					Resource:  resource,
					CloneMeta: f.config.CloneMeta,
				}
			case "partial":
				f.config.Artifact = &retrieval.ArtifactRenderOptions[meta]{
					Resource:  retrieval.RuneResource(100),
					CloneMeta: f.config.CloneMeta,
					Snippet:   func(retrieval.Document[meta]) string { return "partial" },
				}
			}
			sink := &recordSink{}
			config := recordingConfig(t, f, evidence.Required, sink)
			config.Policy.AllowDecisions = true
			// Act: required immutable export runs after the one recipe attempt.
			output, err := recording.Run(context.Background(), recordedRequest(f), config)
			// Assert: delivery follows actual renderer intent/output, not IDs or nil alone.
			if err != nil || sink.calls != 1 {
				t.Fatal(err, sink.calls)
			}
			snapshot, err := sink.record.Snapshot()
			if err != nil {
				t.Fatal(err)
			}
			if len(snapshot.Decision.Selected) != 1 || len(snapshot.Decision.Queries) != 1 {
				t.Fatal(snapshot.Decision)
			}
			selected := snapshot.Decision.Selected[0]
			query := snapshot.Decision.Queries[0]
			assertDecisionDelivery(t, mode, output.Result, selected, query)
		})
	}
}

func assertDecisionDelivery(
	t *testing.T,
	mode string,
	result recipe.Result[meta],
	selected evidence.DecisionSelection,
	query evidence.DecisionQuery,
) {
	t.Helper()
	switch mode {
	case "direct":
		if result.ArtifactRequested || result.Artifact != nil ||
			result.Outcome != recipe.Complete ||
			!selected.Delivered ||
			selected.Uncertain ||
			!query.Delivered ||
			query.Uncertain {
			t.Fatal(result, selected, query)
		}
	case "deadline":
		if !result.ArtifactRequested || result.Artifact != nil ||
			result.Stop != recipe.DeadlineReached ||
			selected.Delivered ||
			!selected.Uncertain ||
			query.Delivered ||
			!query.Uncertain {
			t.Fatal(result, selected, query)
		}
	case "partial":
		if !result.ArtifactRequested || result.Artifact == nil ||
			result.Outcome != recipe.Partial ||
			!selected.Delivered ||
			!selected.Uncertain ||
			query.Delivered ||
			!query.Uncertain {
			t.Fatal(result, selected, query)
		}
	}
}

func TestDerivedDeliveryFreshSourceDenialSuppressesRecording(t *testing.T) {
	// Arrange: permit retrieval/fusion source admission, deny the same original
	// contributor at the delivery boundary after derived snippet packing.
	f := newFixture(t, recipe.Decomposition)
	f.planned = []string{"one"}
	f.selected = []int{0}
	f.sufficient = true
	f.results["one"] = []retrieval.Document[meta]{document("d1")}
	f.config.Artifact = &retrieval.ArtifactRenderOptions[meta]{
		Resource:  retrieval.RuneResource(100),
		CloneMeta: f.config.CloneMeta,
		Snippet:   func(retrieval.Document[meta]) string { return "partial" },
	}
	sink := &recordSink{}
	config := recordingConfig(t, f, evidence.Required, sink)
	config.Policy.AllowDecisions = true
	original := config.SourceAdmission
	calls := 0
	config.SourceAdmission = func(ctx context.Context, read access.Binding, ref source.Reference) error {
		calls++
		if calls == 3 {
			return access.NonSkippable(ragy.ErrUnavailable)
		}
		return original(ctx, read, ref)
	}
	// Act.
	output, err := recording.Run(context.Background(), recordedRequest(f), config)
	// Assert: delivery lineage is freshly admitted; denial suppresses all payload.
	if !access.IsProtectionFailure(err) || calls != 3 || sink.calls != 0 || len(output.Result.Queries) != 0 {
		t.Fatal(err, calls, sink.calls, output.Result)
	}
}
