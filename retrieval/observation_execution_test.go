package retrieval

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/observation"
)

type diagnosticBackend struct {
	calls     *atomic.Uint64
	err       error
	documents []Document[struct{}]
}

func (b diagnosticBackend) Retrieve(_ context.Context, _ Query[struct{}]) (ResultSet[struct{}], error) {
	b.calls.Add(1)
	return NewResultSet(b.documents, nil), b.err
}
func diagnosticContext(t *testing.T, callback observation.ObserverFunc) context.Context {
	t.Helper()
	session, err := observation.New(observation.Config{MaxEvents: 1024, Observer: callback})
	if err != nil {
		t.Fatal(err)
	}
	return observation.WithQuery(observation.WithSession(context.Background(), session), 1)
}

//nolint:gocognit // Integration matrix checks branch dispatch, correlation and exporter failure together.
func TestObservationExecutionFallbackAndRescue(t *testing.T) {
	t.Parallel()
	for _, rescue := range []bool{false, true} {
		t.Run(map[bool]string{false: "fallback", true: "rescue"}[rescue], func(t *testing.T) {
			t.Parallel()
			// Arrange.
			var primaryCalls, secondaryCalls atomic.Uint64
			primaryError := error(nil)
			if rescue {
				primaryError = ragy.ErrUnavailable
			}
			primary := BackendNode[struct{}, struct{}, NoExecutionMeta]{
				Backend: diagnosticBackend{calls: &primaryCalls, err: primaryError},
			}
			secondary := BackendNode[struct{}, struct{}, NoExecutionMeta]{
				Backend: diagnosticBackend{
					calls:     &secondaryCalls,
					documents: []Document[struct{}]{{ID: "private", Content: "secret"}},
				},
			}
			var events []observation.Event
			ctx := diagnosticContext(t, func(_ context.Context, event observation.Event) error {
				events = append(events, event)
				return errors.New("private exporter error")
			})
			var node ExecutionNode[struct{}, struct{}, NoExecutionMeta] = FallbackNode[struct{}, struct{}, NoExecutionMeta]{Primary: primary, Secondary: secondary}
			stage := observation.StageFallback
			if rescue {
				node = RescueNode[struct{}, struct{}, NoExecutionMeta]{Primary: primary, Secondary: secondary}
				stage = observation.StageRescue
			}
			// Act.
			result, err := node.Execute(
				ctx,
				Query[struct{}]{Read: UnrestrictedRead(), Options: RetrieveOptions{TopK: 1}},
				NoExecutionMeta{},
			)
			// Assert.
			if err != nil || result.Len() != 1 || primaryCalls.Load() != 1 || secondaryCalls.Load() != 1 {
				t.Fatalf("unexpected dispatch/result: %v %#v", err, result)
			}
			var ended bool
			for _, event := range events {
				if !event.Query.Known || event.Query.Value != 1 {
					t.Fatal("query correlation lost")
				}
				if event.Kind == observation.KindEnd && event.Stage == stage {
					ended = true
					if event.Completion.Outcome != observation.OutcomeSuccess {
						t.Fatal(event)
					}
				}
				if event.Stage == observation.StageRetrieval && (!event.Branch.Known ||
					(event.Branch.Value != 1 && event.Branch.Value != 2)) {
					t.Fatal("branch ordinal missing")
				}
			}
			if !ended {
				t.Fatal("missing branch completion")
			}
		})
	}
}
func TestObservationProtectionSuppressesRetrievedCount(t *testing.T) {
	t.Parallel()
	// Arrange.
	var calls atomic.Uint64
	var end observation.Event
	ctx := diagnosticContext(t, func(_ context.Context, event observation.Event) error {
		if event.Kind == observation.KindEnd {
			end = event
		}
		panic("observer panic")
	})
	node := BackendNode[struct{}, struct{}, NoExecutionMeta]{
		Backend: diagnosticBackend{
			calls:     &calls,
			documents: []Document[struct{}]{{ID: "private", Content: "secret"}},
			err:       access.Protect(ragy.ErrUnavailable),
		},
	}
	// Act.
	result, err := node.Execute(
		ctx,
		Query[struct{}]{Read: UnrestrictedRead(), Options: RetrieveOptions{TopK: 1}},
		NoExecutionMeta{},
	)
	// Assert.
	if !access.IsProtectionFailure(err) || result.Len() != 0 || calls.Load() != 1 {
		t.Fatal("protection or dispatch changed")
	}
	if end.Completion.Error != observation.ErrorProtection || !end.Completion.Count.Known ||
		end.Completion.Count.Value != 0 {
		t.Fatal(end)
	}
}
func TestObservationAggregateParallelBranchesAndPartial(t *testing.T) {
	t.Parallel()
	// Arrange.
	var calls atomic.Uint64
	var events []observation.Event
	ctx := diagnosticContext(
		t,
		func(_ context.Context, event observation.Event) error { events = append(events, event); return nil },
	)
	success := BackendNode[struct{}, struct{}, NoExecutionMeta]{
		Backend: diagnosticBackend{calls: &calls, documents: []Document[struct{}]{{ID: "one", Content: "one"}}},
	}
	failed := BackendNode[struct{}, struct{}, NoExecutionMeta]{
		Backend: diagnosticBackend{calls: &calls, err: ragy.ErrUnavailable},
	}
	node := AggregateNode[struct{}, struct{}, NoExecutionMeta]{
		Nodes:       []ExecutionNode[struct{}, struct{}, NoExecutionMeta]{success, failed},
		Concurrency: 2,
		Merger:      NewScoreMerger[struct{}](nil),
	}
	// Act.
	result, err := node.Execute(
		ctx,
		Query[struct{}]{Read: UnrestrictedRead(), Options: RetrieveOptions{TopK: 1}},
		NoExecutionMeta{},
	)
	// Assert.
	if err == nil || result.Len() != 1 || calls.Load() != 2 {
		t.Fatalf("unexpected partial: %v %d", err, result.Len())
	}
	var branchMask uint64
	for _, event := range events {
		if event.Kind != observation.KindEnd {
			continue
		}
		if event.Stage == observation.StageRetrieval {
			branchMask |= 1 << event.Branch.Value
			if event.Parent == 0 {
				t.Fatal("missing aggregate parent")
			}
		}
		if event.Stage == observation.StageAggregate && event.Completion.Outcome != observation.OutcomePartial {
			t.Fatal(event)
		}
	}
	if branchMask != 6 {
		t.Fatalf("branch mask %d", branchMask)
	}
}

type diagnosticInvalidCount struct{ ResultSet[struct{}] }

func (diagnosticInvalidCount) Len() int { return -1 }
func TestObservationNegativeCardinalityStaysUnknown(t *testing.T) {
	t.Parallel()
	// Arrange.
	result := diagnosticInvalidCount{ResultSet: NewResultSet[struct{}](nil, nil)}
	// Act.
	completion := observationCompletion[struct{}](nil, result)
	// Assert.
	if completion.Count.Known || completion.Count.Value != 0 {
		t.Fatal(completion)
	}
}

func TestObservationConditionalSelectedEmptyChild(t *testing.T) {
	t.Parallel()
	// Arrange.
	var completions []observation.Outcome
	ctx := diagnosticContext(t, func(_ context.Context, event observation.Event) error {
		if event.Kind == observation.KindEnd && event.Stage == observation.StageConditional {
			completions = append(completions, event.Completion.Outcome)
		}
		return nil
	})
	child := ConditionalNode[struct{}, struct{}, NoExecutionMeta]{
		Predicate: func(Query[struct{}]) bool { return false },
	}
	parent := ConditionalNode[struct{}, struct{}, NoExecutionMeta]{Child: child}
	// Act.
	_, err := parent.Execute(ctx, Query[struct{}]{Read: UnrestrictedRead()}, NoExecutionMeta{})
	// Assert.
	if err != nil || len(completions) != 2 || completions[0] != observation.OutcomeSkipped ||
		completions[1] != observation.OutcomeEmpty {
		t.Fatalf("unexpected completions: %v %v", completions, err)
	}
}
