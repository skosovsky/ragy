package retrieval

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
)

type reviewNode struct {
	calls *int
	err   error
}

func (n *reviewNode) Execute(
	context.Context,
	Query[struct{}],
	NoExecutionMeta,
) (RetrievalResult[struct{}, NoExecutionMeta], error) {
	if n == nil {
		panic("typed nil dispatched")
	}
	*n.calls++
	return emptyRetrievalResult[struct{}](nil, NoExecutionMeta{}), n.err
}
func TestReviewConditionalInvalidChildBeforePredicate(t *testing.T) {
	// Arrange: independent acceptance counterexample retained as regression.
	calls := 0
	n := ConditionalNode[struct{}, struct{}, NoExecutionMeta]{
		Predicate: func(Query[struct{}]) bool { calls++; return false },
	}
	// Act.
	_, err := n.Execute(
		t.Context(),
		Query[struct{}]{Read: UnrestrictedRead(), Options: RetrieveOptions{TopK: 1}},
		NoExecutionMeta{},
	)
	// Assert.
	if !errors.Is(err, ragy.ErrInvalidArgument) || calls != 0 {
		t.Fatalf("err=%v calls=%d", err, calls)
	}
}
func TestReviewFallbackTypedNilSecondaryBeforePrimary(t *testing.T) {
	// Arrange: independent acceptance counterexample retained as regression.
	calls := 0
	var nilNode *reviewNode
	n := FallbackNode[struct{}, struct{}, NoExecutionMeta]{Primary: &reviewNode{calls: &calls}, Secondary: nilNode}
	defer func() {
		if p := recover(); p != nil {
			t.Fatalf("panic=%v primary calls=%d", p, calls)
		}
	}()
	// Act.
	_, err := n.Execute(
		t.Context(),
		Query[struct{}]{Read: UnrestrictedRead(), Options: RetrieveOptions{TopK: 1}},
		NoExecutionMeta{},
	)
	// Assert.
	if !errors.Is(err, ragy.ErrInvalidArgument) || calls != 0 {
		t.Fatalf("err=%v calls=%d", err, calls)
	}
}
func TestReviewDegradingInvalidPortsRemainInvalidAtBuild(t *testing.T) {
	// Arrange: independent acceptance counterexample retained as regression.
	calls := 0
	root := AggregateNode[struct{}, struct{}, NoExecutionMeta]{
		Nodes:  []ExecutionNode[struct{}, struct{}, NoExecutionMeta]{&reviewNode{calls: &calls}},
		Merger: DegradingMerger[struct{}]{},
	}
	// Act.
	_, err := NewExecutionPipelineBuilder[struct{}, struct{}, NoExecutionMeta]().WithRoot(root).Build()
	// Assert.
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("invalid degrading config build err=%v", err)
	}
}
func TestReviewProtectionPreservesJoinedCauses(t *testing.T) {
	// Arrange: independent acceptance counterexample retained as regression.
	sibling := errors.New("sibling")
	input := NewResultSet([]Document[struct{}]{{ID: "a", Content: "a"}}, nil)
	merger := compositionMergerFunc(
		func(context.Context, ...ResultSet[struct{}]) (ResultSet[struct{}], error) {
			return NewResultSet[struct{}](nil, nil), nil
		},
	)
	// Act.
	_, err := finalizeAggregateRetrieve(
		t.Context(),
		DefaultResolver[struct{}](nil),
		merger,
		[]aggregateChildResult[struct{}]{{rs: input, err: errors.Join(access.Protect(ragy.ErrUnavailable), sibling)}},
	)
	// Assert.
	if !errors.Is(err, sibling) {
		t.Fatalf("lost joined sibling: %v", err)
	}
}

func TestReviewDegradePrimarySuccessCanceledSuppresses(t *testing.T) {
	// Arrange: independent acceptance counterexample retained as regression.
	ctx, cancel := context.WithCancel(t.Context())
	input := NewResultSet([]Document[struct{}]{{ID: "a", Content: "a"}}, nil)
	m := DegradingMerger[struct{}]{
		Primary: compositionMergerFunc(func(context.Context, ...ResultSet[struct{}]) (ResultSet[struct{}], error) {
			cancel()
			return input, nil
		}),
		Fallback: compositionMergerFunc(
			func(context.Context, ...ResultSet[struct{}]) (ResultSet[struct{}], error) { panic("fallback") },
		),
	}
	// Act.
	out, err := m.Merge(ctx, input)
	// Assert.
	if !out.IsEmpty() || !errors.Is(err, context.Canceled) {
		t.Fatalf("payload len=%d err=%v", out.Len(), err)
	}
}
func TestReviewPartialEmptyNeverRescued(t *testing.T) {
	// Arrange: independent acceptance counterexample retained as regression.
	calls := 0
	secondaryCalls := 0
	partial := &PartialFailureError[struct{}]{
		Errors: []error{ragy.ErrUnavailable},
		Result: NewResultSet[struct{}](nil, nil),
	}
	n := RescueNode[struct{}, struct{}, NoExecutionMeta]{
		Primary:   &reviewNode{calls: &calls, err: partial},
		Secondary: &reviewNode{calls: &secondaryCalls},
	}
	// Act.
	_, err := n.Execute(
		t.Context(),
		Query[struct{}]{Read: UnrestrictedRead(), Options: RetrieveOptions{TopK: 1}},
		NoExecutionMeta{},
	)
	// Assert.
	if secondaryCalls != 0 || !errors.Is(err, partial) {
		t.Fatalf("secondary calls=%d err=%v", secondaryCalls, err)
	}
}

func TestReviewTypedNilRoutePlanner(t *testing.T) {
	// Arrange: independent acceptance counterexample retained as regression.
	var planner RoutePlannerFunc[struct{}, string, struct{}]
	n := RouteSwitchNode[struct{}, string, struct{}, struct{}, NoExecutionMeta]{Planner: planner}
	// Act: builder admission, followed by direct node admission.
	_, buildErr := NewExecutionPipelineBuilder[struct{}, struct{}, NoExecutionMeta]().WithRoot(n).Build()
	// Assert.
	if !errors.Is(buildErr, ragy.ErrInvalidArgument) {
		t.Errorf("build err=%v", buildErr)
	}
	defer func() {
		if p := recover(); p != nil {
			t.Errorf("typed nil route planner panic: %v", p)
		}
	}()
	// Act.
	_, err := n.Execute(
		t.Context(),
		Query[struct{}]{Read: UnrestrictedRead(), Options: RetrieveOptions{TopK: 1}},
		NoExecutionMeta{},
	)
	// Assert.
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Errorf("execute err=%v", err)
	}
}
