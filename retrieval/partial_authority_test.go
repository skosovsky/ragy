package retrieval

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
)

type authorityOuterError struct{ cause error }

func (*authorityOuterError) Error() string   { return "outer authority failure" }
func (e *authorityOuterError) Unwrap() error { return e.cause }

func TestPartialResultAuthorityPreservesOuterCauses(t *testing.T) {
	for _, nonempty := range []bool{false, true} {
		t.Run(map[bool]string{false: "empty", true: "new_payload"}[nonempty], func(t *testing.T) {
			checkPartialAuthority(t, nonempty)
		})
	}
}

func checkPartialAuthority(t *testing.T, nonempty bool) {
	t.Helper()
	// Arrange.
	stale := NewResultSet(
		[]Document[struct{}]{{ID: "old", Content: "old", ScoreState: ScoreAbsent}},
		nil,
	)
	original := &PartialFailureError[struct{}]{Errors: []error{ragy.ErrProtocol}, Result: stale}
	outer := &authorityOuterError{cause: original}
	sibling := errors.New("independent sibling")
	joined := errors.Join(outer, sibling)
	current := NewResultSet[struct{}](nil, nil)
	if nonempty {
		current = NewResultSet(
			[]Document[struct{}]{{ID: "current", Content: "current", ScoreState: ScoreAbsent}},
			nil,
		)
	}
	// Act.
	result, err := PreserveResultOnError(current, joined, nil)
	// Assert.
	if result.Len() != current.Len() || !errors.Is(err, outer) || !errors.Is(err, sibling) ||
		!errors.Is(err, original) ||
		!errors.Is(err, ragy.ErrProtocol) {
		t.Fatal(result, err)
	}
	var foundOuter *authorityOuterError
	if !errors.As(err, &foundOuter) || foundOuter != outer {
		t.Fatal("outer typed cause lost", err)
	}
	view, ok := AsPartialFailure[struct{}](err)
	if !ok || view == nil || view.Result.Len() != current.Len() {
		t.Fatal("diagnostic view mismatch", view)
	}
	if nonempty &&
		(result.Documents()[0].ID != "current" || view.Result.Documents()[0].ID != "current") {
		t.Fatal("stale result resurrected", result, view)
	}
	if original.Result.Documents()[0].ID != "old" {
		t.Fatal("host error mutated")
	}
}

func TestPartialProtectionSuppressesReturnedAndDiagnosticPayload(t *testing.T) {
	// Arrange.
	current := NewResultSet(
		[]Document[struct{}]{{ID: "current", Content: "private", ScoreState: ScoreAbsent}},
		nil,
	)
	original := &PartialFailureError[struct{}]{Errors: []error{ragy.ErrProtocol}, Result: current}
	joined := errors.Join(original, access.Protect(context.Canceled))
	// Act.
	result, err := PreserveResultOnError(current, joined, nil)
	// Assert.
	if result.Len() != 0 || !access.IsProtectionFailure(err) || !errors.Is(err, context.Canceled) {
		t.Fatal(result, err)
	}
}

func TestFinalDeliverySuppressesDiagnosticsAndRetainsJoinedCauses(t *testing.T) {
	// Arrange: a binding expires after ordinary fusion admitted observations.
	input := NewResultSet([]Document[struct{}]{{ID: "a", Content: "private"}}, nil)
	ordinary := errors.New("ordinary failure")
	fusion := &FusionFailureError[struct{}]{
		Cause:        ordinary,
		observations: []ResultSet[struct{}]{input},
	}
	partial := &PartialFailureError[struct{}]{Errors: []error{fusion}, Result: input}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	// Act.
	out, err := DeliverRead(ctx, UnrestrictedRead(), input, partial, nil)
	// Assert.
	partialView, hasPartial := AsPartialFailure[struct{}](err)
	fusionView, hasFusion := errors.AsType[*FusionFailureError[struct{}]](err)
	if !out.IsEmpty() || !errors.Is(err, context.Canceled) || !errors.Is(err, ordinary) ||
		!hasPartial || !partialView.Result.IsEmpty() || !hasFusion || len(fusionView.Observations()) != 0 {
		t.Fatal(out, err, partialView, fusionView)
	}
	if partial.Result.IsEmpty() || len(fusion.Observations()) != 1 {
		t.Fatal("mutated host error")
	}
}
