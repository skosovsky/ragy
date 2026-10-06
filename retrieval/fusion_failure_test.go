package retrieval

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
)

type compositionMergerFunc func(context.Context, ...ResultSet[struct{}]) (ResultSet[struct{}], error)

func (f compositionMergerFunc) Merge(
	ctx context.Context,
	sets ...ResultSet[struct{}],
) (ResultSet[struct{}], error) {
	return f(ctx, sets...)
}

func TestFusionFailureKeepsObservationsWithoutImplicitScoring(t *testing.T) {
	// Arrange: incomparable scales cannot be fused by an implicit score fallback.
	left := NewResultSet(
		[]Document[struct{}]{
			{
				ID:             "left",
				Content:        "left",
				ScoreState:     ScorePresent,
				ScoreSemantics: "distance",
				Score:          -10,
			},
		},
		nil,
	)
	right := NewResultSet(
		[]Document[struct{}]{
			{
				ID:             "right",
				Content:        "right",
				ScoreState:     ScorePresent,
				ScoreSemantics: "similarity",
				Score:          99,
			},
		},
		nil,
	)
	failure := errors.New("fusion failure")
	sibling := errors.New("branch failure")
	merger := compositionMergerFunc(
		func(context.Context, ...ResultSet[struct{}]) (ResultSet[struct{}], error) { return nil, failure },
	)
	// Act.
	out, err := finalizeAggregateRetrieve(t.Context(), DefaultResolver[struct{}](nil), merger,
		[]aggregateChildResult[struct{}]{{rs: left}, {rs: right, err: sibling}})
	// Assert.
	fusion, ok := errors.AsType[*FusionFailureError[struct{}]](err)
	if !ok || !out.IsEmpty() || !errors.Is(err, failure) || !errors.Is(err, sibling) {
		t.Fatal(out, err)
	}
	observed := fusion.Observations()
	if len(observed) != 2 || observed[0].Documents()[0].ID != "left" ||
		observed[1].Documents()[0].ID != "right" {
		t.Fatal(observed)
	}
	observed[0] = right
	docs := fusion.Observations()[0].Documents()
	docs[0].Content = "mutated"
	if fusion.Observations()[0].Documents()[0].Content != "left" ||
		left.Documents()[0].Content != "left" {
		t.Fatal("observations or input containers mutated")
	}
}

func TestDegradationSuppressesProtectionAndCanceledOutput(t *testing.T) {
	for _, mode := range []string{"protection", "primary_cancel", "fallback_cancel", "fallback_protection"} {
		t.Run(mode, func(t *testing.T) {
			// Arrange.
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			calls := 0
			input := NewResultSet([]Document[struct{}]{{ID: "a", Content: "a"}}, nil)
			primaryCause := errors.New("ordinary fusion failure")
			primary := degradationPrimary(mode, input, primaryCause, cancel)
			fallback := degradationFallback(mode, input, &calls, cancel)
			// Act.
			out, err := (DegradingMerger[struct{}]{Primary: primary, Fallback: fallback}).Merge(
				ctx,
				input,
			)
			// Assert.
			wantCalls := 1
			if mode == "protection" || mode == "primary_cancel" {
				wantCalls = 0
			}
			if out == nil || !out.IsEmpty() || err == nil || calls != wantCalls {
				t.Fatal(out, err, calls)
			}
			if mode != "protection" && !errors.Is(err, primaryCause) {
				t.Fatal("lost primary cause", err)
			}
		})
	}
}

func TestAggregateProtectionDoesNotExposeFusionObservations(t *testing.T) {
	// Arrange.
	calls := 0
	merger := compositionMergerFunc(
		func(context.Context, ...ResultSet[struct{}]) (ResultSet[struct{}], error) {
			calls++
			return nil, errors.New("must not dispatch")
		},
	)
	input := NewResultSet([]Document[struct{}]{{ID: "a", Content: "a"}}, nil)
	// Act.
	out, err := finalizeAggregateRetrieve(
		t.Context(),
		DefaultResolver[struct{}](nil),
		merger,
		[]aggregateChildResult[struct{}]{
			{rs: input, err: access.NonSkippable(ragy.ErrUnavailable)},
		},
	)
	// Assert.
	_, exposes := errors.AsType[*FusionFailureError[struct{}]](err)
	if !out.IsEmpty() || !access.IsProtectionFailure(err) || exposes || calls != 0 {
		t.Fatal(out, err, calls)
	}
}

func degradationPrimary(
	mode string,
	input ResultSet[struct{}],
	cause error,
	cancel context.CancelFunc,
) compositionMergerFunc {
	return func(context.Context, ...ResultSet[struct{}]) (ResultSet[struct{}], error) {
		if mode == "protection" {
			return input, access.NonSkippable(ragy.ErrUnavailable)
		}
		if mode == "primary_cancel" {
			cancel()
		}
		return input, cause
	}
}

func degradationFallback(
	mode string,
	input ResultSet[struct{}],
	calls *int,
	cancel context.CancelFunc,
) compositionMergerFunc {
	return func(context.Context, ...ResultSet[struct{}]) (ResultSet[struct{}], error) {
		*calls++
		if mode == "fallback_cancel" {
			cancel()
		}
		if mode == "fallback_protection" {
			return input, access.NonSkippable(ragy.ErrProtocol)
		}
		return input, nil
	}
}

func TestAggregateWithoutObservationsHasOrdinaryFailure(t *testing.T) {
	for _, observed := range []bool{false, true} {
		// Arrange: empty merger output does not alone determine whether evidence was partial.
		cause := errors.New("branch outage")
		input := NewResultSet[struct{}](nil, nil)
		if observed {
			input = NewResultSet([]Document[struct{}]{{ID: "a", Content: "a"}}, nil)
		}
		merger := compositionMergerFunc(func(context.Context, ...ResultSet[struct{}]) (ResultSet[struct{}], error) {
			return NewResultSet[struct{}](nil, nil), nil
		})
		// Act.
		out, err := finalizeAggregateRetrieve(
			t.Context(),
			DefaultResolver[struct{}](nil),
			merger,
			[]aggregateChildResult[struct{}]{{rs: input, err: cause}},
		)
		// Assert: no admitted observation creates no partial marker; real evidence retains partial.
		_, partial := AsPartialFailure[struct{}](err)
		if !out.IsEmpty() || !errors.Is(err, cause) || partial != observed {
			t.Fatal(observed, out, err)
		}
	}
}
