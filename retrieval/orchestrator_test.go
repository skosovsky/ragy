package retrieval

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
)

type stubIntent struct{}

const pipelineTestTopK = 10

func pipelineTestQuery(text string) Query[stubIntent] {
	return Query[stubIntent]{
		Text:    text,
		Options: RetrieveOptions{TopK: pipelineTestTopK},
	}
}

type stubNode[TMeta any] struct {
	docs []Document[TMeta]
}

func (n stubNode[TMeta]) Retrieve(_ context.Context, _ Query[stubIntent]) (ResultSet[TMeta], error) {
	return NewResultSet(n.docs, DocumentIDResolver[TMeta]{}), nil
}

type errorNode[TIntent, TMeta any] struct {
	err error
}

func (n errorNode[TIntent, TMeta]) Retrieve(context.Context, Query[TIntent]) (ResultSet[TMeta], error) {
	return NewResultSet[TMeta](nil, DocumentIDResolver[TMeta]{}), n.err
}

type orchestratorStubBackend[TIntent, TMeta any] struct {
	docs []Document[TMeta]
}

func (b orchestratorStubBackend[TIntent, TMeta]) Schema() filter.Schema { return filter.EmptySchema() }

func (b orchestratorStubBackend[TIntent, TMeta]) Retrieve(
	_ context.Context,
	_ Query[TIntent],
) (ResultSet[TMeta], error) {
	return NewResultSet(b.docs, DocumentIDResolver[TMeta]{}), nil
}

type orchestratorFailingBackend[TIntent, TMeta any] struct{}

func (orchestratorFailingBackend[TIntent, TMeta]) Schema() filter.Schema { return filter.EmptySchema() }

func (orchestratorFailingBackend[TIntent, TMeta]) Retrieve(
	context.Context,
	Query[TIntent],
) (ResultSet[TMeta], error) {
	return NewResultSet[TMeta](nil, DocumentIDResolver[TMeta]{}), ragy.ErrUnavailable
}

func TestFallbackNodeUsesSecondaryWhenPrimaryEmpty(t *testing.T) {
	t.Parallel()

	primary := stubNode[struct{}]{docs: nil}
	secondary := stubNode[struct{}]{docs: []Document[struct{}]{{ID: "fb", Content: "hit", Score: 1}}}

	node := resultFallbackNodeNoMeta[stubIntent, struct{}]{Primary: primary, Secondary: secondary}
	rs, err := node.Retrieve(context.Background(), Query[stubIntent]{Text: "q"})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.IsEmpty() || rs.Documents()[0].ID != "fb" {
		t.Fatalf("Documents() = %#v, want fallback doc", rs.Documents())
	}
}

func TestFallbackNodePropagatesPrimaryError(t *testing.T) {
	t.Parallel()

	node := resultFallbackNodeNoMeta[stubIntent, struct{}]{
		Primary: errorNode[stubIntent, struct{}]{err: ragy.ErrUnavailable},
		Secondary: stubNode[struct{}]{
			docs: []Document[struct{}]{{ID: "fb", Content: "hit", Score: 1}},
		},
	}
	out, err := node.Retrieve(context.Background(), Query[stubIntent]{Text: "q"})
	requireNonNilResultSetOnError(t, out, err)
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("Retrieve() error = %v, want unavailable", err)
	}
}

func TestRescueNodeUsesSecondaryOnPrimaryError(t *testing.T) {
	t.Parallel()

	node := resultRescueNodeNoMeta[stubIntent, struct{}]{
		Primary: errorNode[stubIntent, struct{}]{err: ragy.ErrUnavailable},
		Secondary: stubNode[struct{}]{
			docs: []Document[struct{}]{{ID: "fb", Content: "hit", Score: 1}},
		},
	}
	rs, err := node.Retrieve(context.Background(), Query[stubIntent]{Text: "q"})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Documents()[0].ID != "fb" {
		t.Fatalf("Documents() = %#v, want rescue doc", rs.Documents())
	}
}

func TestRescueNodeDoesNotUseSecondaryOnEmpty(t *testing.T) {
	t.Parallel()

	node := resultRescueNodeNoMeta[stubIntent, struct{}]{
		Primary: stubNode[struct{}]{docs: nil},
		Secondary: stubNode[struct{}]{
			docs: []Document[struct{}]{{ID: "fb", Content: "hit", Score: 1}},
		},
	}
	rs, err := node.Retrieve(context.Background(), Query[stubIntent]{Text: "q"})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if !rs.IsEmpty() {
		t.Fatalf("Documents() = %#v, want empty without rescue", rs.Documents())
	}
}

func TestPartialSuccessRS(t *testing.T) {
	t.Parallel()

	doc := Document[struct{}]{ID: "a", Content: "hit", Score: 1}
	rsWithDoc := NewResultSet([]Document[struct{}]{doc}, DocumentIDResolver[struct{}]{})
	emptyRS := NewResultSet[struct{}](nil, DocumentIDResolver[struct{}]{})

	cases := []struct {
		name string
		rs   ResultSet[struct{}]
		err  error
		want bool
	}{
		{name: "nil err", rs: emptyRS, err: nil, want: false},
		{
			name: "partial empty",
			rs:   emptyRS,
			err:  &PartialFailureError[struct{}]{Errors: []error{ragy.ErrUnavailable}, Result: emptyRS},
			want: false,
		},
		{
			name: "partial non-empty",
			rs:   emptyRS,
			err:  &PartialFailureError[struct{}]{Errors: []error{ragy.ErrUnavailable}, Result: rsWithDoc},
			want: true,
		},
		{name: "plain err empty rs", rs: emptyRS, err: ragy.ErrUnavailable, want: false},
		{name: "plain err with docs", rs: rsWithDoc, err: ragy.ErrUnavailable, want: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if got := partialSuccessRS(tc.rs, tc.err); got != tc.want {
				t.Fatalf("partialSuccessRS() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestAggregateNodeToleratesChildError(t *testing.T) {
	t.Parallel()

	node := resultAggregateNodeNoMeta[stubIntent, struct{}]{
		Nodes: []resultNodeNoMeta[stubIntent, struct{}]{
			errorNode[stubIntent, struct{}]{err: ragy.ErrEmptyVector},
			stubNode[struct{}]{docs: []Document[struct{}]{{ID: "a", Content: "hit", Score: 1}}},
		},
	}
	rs, err := node.Retrieve(context.Background(), Query[stubIntent]{Text: "q"})
	if err == nil {
		t.Fatal("Retrieve() error = nil, want partial failure")
	}
	var partial *PartialFailureError[struct{}]
	if !errors.As(err, &partial) {
		t.Fatalf("Retrieve() error = %v, want PartialFailureError", err)
	}
	if partial.Result.Len() != 1 || partial.Result.Documents()[0].ID != "a" {
		t.Fatalf("partial.Result = %#v, want doc a", partial.Result.Documents())
	}
	if len(partial.Errors) != 1 || !errors.Is(partial.Errors[0], ragy.ErrEmptyVector) {
		t.Fatalf("partial.Errors = %#v, want empty vector", partial.Errors)
	}
	if rs.Len() != 1 || rs.Documents()[0].ID != "a" {
		t.Fatalf("Documents() = %#v, want doc a", rs.Documents())
	}
}

func TestAggregateNodeFailsWhenAllChildrenError(t *testing.T) {
	t.Parallel()

	node := resultAggregateNodeNoMeta[stubIntent, struct{}]{
		Nodes: []resultNodeNoMeta[stubIntent, struct{}]{
			errorNode[stubIntent, struct{}]{err: ragy.ErrEmptyVector},
			errorNode[stubIntent, struct{}]{err: ragy.ErrUnavailable},
		},
	}
	out, err := node.Retrieve(context.Background(), Query[stubIntent]{Text: "q"})
	requireNonNilResultSetOnError(t, out, err)
	if !errors.Is(err, ragy.ErrEmptyVector) || !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("Retrieve() error = %v, want joined child errors", err)
	}
}

func TestRescueNodeReturnsPrimaryErrorWhenSecondaryFails(t *testing.T) {
	t.Parallel()

	node := resultRescueNodeNoMeta[stubIntent, struct{}]{
		Primary:   errorNode[stubIntent, struct{}]{err: ragy.ErrUnavailable},
		Secondary: errorNode[stubIntent, struct{}]{err: ragy.ErrProtocol},
	}
	out, err := node.Retrieve(context.Background(), Query[stubIntent]{Text: "q"})
	requireNonNilResultSetOnError(t, out, err)
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("Retrieve() error = %v, want unavailable", err)
	}
	if !errors.Is(err, ragy.ErrProtocol) {
		t.Fatalf("Retrieve() error = %v, want wrapped protocol error", err)
	}
}

func TestAggregateNodeUsesRRFByDefault(t *testing.T) {
	t.Parallel()

	node1 := stubNode[struct{}]{docs: []Document[struct{}]{
		{ID: "solo", Content: "solo", Score: 0.99},
	}}
	node2 := stubNode[struct{}]{docs: []Document[struct{}]{
		{ID: "a", Content: "a", Score: 0.9},
		{ID: "b", Content: "b", Score: 0.8},
		{ID: "c", Content: "c", Score: 0.7},
		{ID: "d", Content: "d", Score: 0.01},
	}}

	aggregate := resultAggregateNodeNoMeta[stubIntent, struct{}]{
		Nodes: []resultNodeNoMeta[stubIntent, struct{}]{node1, node2},
	}
	rs, err := aggregate.Retrieve(context.Background(), Query[stubIntent]{Text: "q"})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}

	rrf, err := NewReciprocalRankFusion[struct{}](defaultAggregateRRFK, DocumentIDResolver[struct{}]{})
	if err != nil {
		t.Fatalf("NewReciprocalRankFusion(): %v", err)
	}
	expected, err := rrf.Merge(
		context.Background(),
		NewResultSet(node1.docs, DocumentIDResolver[struct{}]{}),
		NewResultSet(node2.docs, DocumentIDResolver[struct{}]{}),
	)
	if err != nil {
		t.Fatalf("Merge(): %v", err)
	}
	if rs.Len() != expected.Len() {
		t.Fatalf("Len() = %d, want %d from RRF", rs.Len(), expected.Len())
	}
	gotScores := scoresByID(rs.Documents())
	wantScores := scoresByID(expected.Documents())
	for id, score := range wantScores {
		if gotScores[id] != score {
			t.Fatalf("doc %q RRF score = %v, want %v", id, gotScores[id], score)
		}
	}

	scoreMerger := NewScoreMerger(DocumentIDResolver[struct{}]{})
	scoreMerged, err := scoreMerger.Merge(
		context.Background(),
		NewResultSet(node1.docs, DocumentIDResolver[struct{}]{}),
		NewResultSet(node2.docs, DocumentIDResolver[struct{}]{}),
	)
	if err != nil {
		t.Fatalf("ScoreMerger.Merge(): %v", err)
	}
	if scoreMerged.Documents()[0].ID == rs.Documents()[0].ID && rs.Len() == scoreMerged.Len() {
		// Both may rank solo first; ensure ordering differs when scales diverge.
		if rs.Documents()[0].Score == scoreMerged.Documents()[0].Score {
			t.Fatal("default aggregate should use RRF scores, not raw score-merge scores")
		}
	}
}

func TestAggregateNodeReportsPartialFailure(t *testing.T) {
	t.Parallel()

	node := resultAggregateNodeNoMeta[stubIntent, struct{}]{
		Nodes: []resultNodeNoMeta[stubIntent, struct{}]{
			errorNode[stubIntent, struct{}]{err: ragy.ErrUnavailable},
			stubNode[struct{}]{docs: []Document[struct{}]{{ID: "hit", Content: "ok", Score: 1}}},
		},
	}
	rs, err := node.Retrieve(context.Background(), Query[stubIntent]{Text: "q"})
	if err == nil {
		t.Fatal("Retrieve() error = nil, want partial failure")
	}
	var partial *PartialFailureError[struct{}]
	if !errors.As(err, &partial) {
		t.Fatalf("Retrieve() error = %v, want PartialFailureError", err)
	}
	if rs.Len() != 1 || rs.Documents()[0].ID != "hit" {
		t.Fatalf("Documents() = %#v, want hit", rs.Documents())
	}
}

func TestInjectNodeResolverRecursive(t *testing.T) {
	t.Parallel()

	resolver := mergeKeyResolver[struct{}]{key: func(doc Document[struct{}]) string { return doc.Content }}
	root := resultFallbackNodeNoMeta[stubIntent, struct{}]{
		Primary: resultAggregateNodeNoMeta[stubIntent, struct{}]{
			Merger: NewScoreMerger[struct{}](nil),
			Nodes: []resultNodeNoMeta[stubIntent, struct{}]{
				stubNode[struct{}]{docs: []Document[struct{}]{
					{ID: "a1", Content: "grp", Score: 0.2},
					{ID: "a2", Content: "grp", Score: 0.9},
				}},
				stubNode[struct{}]{docs: []Document[struct{}]{
					{ID: "b1", Content: "grp", Score: 0.5},
				}},
			},
		},
		Secondary: stubNode[struct{}]{docs: nil},
	}

	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(root).
		WithResolver(resolver).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), pipelineTestQuery("q"))
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Len() != 1 {
		t.Fatalf("Len() = %d, want 1 merged doc by Content MergeKey", rs.Len())
	}
	if rs.Documents()[0].ID != "a2" {
		t.Fatalf("Documents()[0].ID = %q, want a2 (highest score in merge group)", rs.Documents()[0].ID)
	}
}

func TestPipelineRetrieveRewrapsResolver(t *testing.T) {
	t.Parallel()

	resolver := mergeKeyResolver[struct{}]{key: func(doc Document[struct{}]) string { return doc.Content }}
	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(stubNode[struct{}]{docs: []Document[struct{}]{
			{ID: "left", Content: "key", Score: 0.2},
		}}).
		WithResolver(resolver).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), pipelineTestQuery("q"))
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}

	other := NewResultSet([]Document[struct{}]{
		{ID: "right", Content: "key", Score: 0.9},
	}, DocumentIDResolver[struct{}]{})
	merged, err := rs.ResultSet.Merge(other)
	if err != nil {
		t.Fatalf("Merge(): %v", err)
	}
	if merged.Len() != 1 || merged.Documents()[0].ID != "right" {
		t.Fatalf("Merge() = %#v, want winner by pipeline MergeKey", merged.Documents())
	}
}

func TestFallbackNodeRejectsNilPrimary(t *testing.T) {
	t.Parallel()

	node := resultFallbackNodeNoMeta[stubIntent, struct{}]{Primary: nil}
	out, err := node.Retrieve(context.Background(), Query[stubIntent]{Text: "q"})
	requireNonNilResultSetOnError(t, out, err)
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Retrieve() error = %v, want invalid argument", err)
	}
}

func TestAggregateNodeReturnsErrorOnNilChild(t *testing.T) {
	t.Parallel()

	node := resultAggregateNodeNoMeta[stubIntent, struct{}]{
		Nodes: []resultNodeNoMeta[stubIntent, struct{}]{
			nil,
			stubNode[struct{}]{docs: []Document[struct{}]{{ID: "a", Content: "A", Score: 1}}},
		},
	}
	_, err := node.Retrieve(context.Background(), Query[stubIntent]{
		Text:    "q",
		Options: RetrieveOptions{TopK: 1},
	})
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Retrieve() error = %v, want invalid argument", err)
	}
}

func TestAggregateNodeReturnsEmptyWhenNoChildren(t *testing.T) {
	t.Parallel()

	node := resultAggregateNodeNoMeta[stubIntent, struct{}]{
		Nodes: []resultNodeNoMeta[stubIntent, struct{}]{},
	}
	rs, err := node.Retrieve(context.Background(), pipelineTestQuery("q"))
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs == nil || !rs.IsEmpty() {
		t.Fatalf("ResultSet = %#v, want non-nil empty", rs)
	}
}

func TestAggregateNodeReturnsEmptyWhenAllChildrenNil(t *testing.T) {
	t.Parallel()

	_, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(resultAggregateNodeNoMeta[stubIntent, struct{}]{
			Nodes: []resultNodeNoMeta[stubIntent, struct{}]{nil, nil},
		}).
		Build()
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Build() error = %v, want invalid argument for nil aggregate child", err)
	}
}

func TestPipelineBuildRejectsNilFallbackPrimary(t *testing.T) {
	t.Parallel()

	_, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(resultFallbackNodeNoMeta[stubIntent, struct{}]{Primary: nil}).
		Build()
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Build() error = %v, want invalid argument", err)
	}
}

func TestPipelineBuildRejectsNilRescuePrimary(t *testing.T) {
	t.Parallel()

	_, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(resultRescueNodeNoMeta[stubIntent, struct{}]{Primary: nil}).
		Build()
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Build() error = %v, want invalid argument", err)
	}
}

func TestPipelineBuildRejectsNilConditionalChild(t *testing.T) {
	t.Parallel()

	_, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(resultConditionalNodeNoMeta[stubIntent, struct{}]{
			Predicate: func(Query[stubIntent]) bool { return true },
			Child:     nil,
		}).
		Build()
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Build() error = %v, want invalid argument", err)
	}
}

func TestConditionalNodeReturnsErrorOnNilChild(t *testing.T) {
	t.Parallel()

	node := resultConditionalNodeNoMeta[stubIntent, struct{}]{
		Predicate: func(Query[stubIntent]) bool { return true },
		Child:     nil,
	}
	_, err := node.Retrieve(context.Background(), pipelineTestQuery("q"))
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Retrieve() error = %v, want invalid argument", err)
	}
}

func TestPipelineBuildRejectsNilRetrieverBackend(t *testing.T) {
	t.Parallel()

	_, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(resultRetrieverNodeNoMeta[stubIntent, struct{}]{Backend: nil}).
		Build()
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Build() error = %v, want invalid argument", err)
	}
}

func TestPipelineBuildRejectsNilAggregateChild(t *testing.T) {
	t.Parallel()

	_, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(resultAggregateNodeNoMeta[stubIntent, struct{}]{
			Nodes: []resultNodeNoMeta[stubIntent, struct{}]{stubNode[struct{}]{}, nil},
		}).
		Build()
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Build() error = %v, want invalid argument", err)
	}
}

func TestFallbackNodeSkipsSecondaryWhenPrimaryHasResults(t *testing.T) {
	t.Parallel()

	secondary := stubNode[struct{}]{
		docs: []Document[struct{}]{{ID: "secondary", Score: 1}},
	}
	node := resultFallbackNodeNoMeta[stubIntent, struct{}]{
		Primary: stubNode[struct{}]{
			docs: []Document[struct{}]{{ID: "primary", Score: 1}},
		},
		Secondary: secondary,
	}
	rs, err := node.Retrieve(context.Background(), Query[stubIntent]{Text: "q"})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.IsEmpty() || rs.Documents()[0].ID != "primary" {
		t.Fatalf("Documents() = %#v, want primary without secondary", rs.Documents())
	}
}

func TestPipelineBuilderSecondWithResolverOverwritesFirst(t *testing.T) {
	t.Parallel()

	first := DocumentIDResolver[struct{}]{}
	second := mergeKeyResolver[struct{}]{key: func(doc Document[struct{}]) string { return doc.Content }}
	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(stubNode[struct{}]{docs: []Document[struct{}]{
			{ID: "a", Content: "same", Score: 0.9},
			{ID: "b", Content: "same", Score: 0.5},
		}}).
		WithResolver(first).
		WithResolver(second).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), pipelineTestQuery("q"))
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Len() != 2 {
		t.Fatalf("Len() = %d, want 2 docs before merge", rs.Len())
	}
	merged, mergeErr := rs.ResultSet.Merge(NewResultSet([]Document[struct{}]{
		{ID: "c", Content: "same", Score: 0.1},
	}, second))
	if mergeErr != nil {
		t.Fatalf("Merge(): %v", mergeErr)
	}
	if merged.Len() != 1 {
		t.Fatalf("merged Len() = %d, want 1 (second WithResolver wins for merge key)", merged.Len())
	}
}

func TestConditionalNodePropagatesChildError(t *testing.T) {
	t.Parallel()

	node := resultConditionalNodeNoMeta[stubIntent, struct{}]{
		Predicate: func(Query[stubIntent]) bool { return true },
		Child:     errorNode[stubIntent, struct{}]{err: ragy.ErrUnavailable},
	}
	out, err := node.Retrieve(context.Background(), Query[stubIntent]{Text: "q"})
	requireNonNilResultSetOnError(t, out, err)
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("Retrieve() error = %v, want unavailable", err)
	}
}

func TestConditionalNodeReturnsEmptyWhenPredicateFalse(t *testing.T) {
	t.Parallel()

	node := resultConditionalNodeNoMeta[stubIntent, struct{}]{
		Predicate: func(Query[stubIntent]) bool { return false },
		Child: stubNode[struct{}]{
			docs: []Document[struct{}]{{ID: "x", Content: "y", Score: 1}},
		},
	}
	rs, err := node.Retrieve(context.Background(), Query[stubIntent]{Text: "q"})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs == nil || !rs.IsEmpty() {
		t.Fatalf("Documents() = %#v, want empty", rs.Documents())
	}
}

func TestPipelineBuilderRetrieve(t *testing.T) {
	t.Parallel()

	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(stubNode[struct{}]{docs: []Document[struct{}]{{ID: "x", Content: "y", Score: 0.5}}}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), pipelineTestQuery("hello"))
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Len() != 1 {
		t.Fatalf("Len() = %d, want 1", rs.Len())
	}
}

func TestPipelineBuilderWithRootThenWithFallbackOverwrites(t *testing.T) {
	t.Parallel()

	custom := stubNode[struct{}]{docs: []Document[struct{}]{{ID: "custom", Content: "x", Score: 1}}}
	fallbackPrimary := stubNode[struct{}]{docs: []Document[struct{}]{{ID: "fb", Content: "y", Score: 1}}}

	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(custom).
		WithFallback(fallbackPrimary, stubNode[struct{}]{docs: nil}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), pipelineTestQuery("q"))
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Documents()[0].ID != "fb" {
		t.Fatalf("Documents()[0].ID = %q, want fb (WithFallback overwrote WithRoot)", rs.Documents()[0].ID)
	}
}

func TestPipelineBuilderWithRootThenWithRescueOverwrites(t *testing.T) {
	t.Parallel()

	custom := stubNode[struct{}]{docs: []Document[struct{}]{{ID: "custom", Content: "x", Score: 1}}}

	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(custom).
		WithRescue(
			errorNode[stubIntent, struct{}]{err: ragy.ErrUnavailable},
			stubNode[struct{}]{docs: []Document[struct{}]{{ID: "rescue", Content: "y", Score: 1}}},
		).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), pipelineTestQuery("q"))
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Documents()[0].ID != "rescue" {
		t.Fatalf("Documents()[0].ID = %q, want rescue (WithRescue overwrote WithRoot)", rs.Documents()[0].ID)
	}
}

func TestPipelineBuilderWithRootThenWithAggregateOverwrites(t *testing.T) {
	t.Parallel()

	custom := stubNode[struct{}]{docs: []Document[struct{}]{{ID: "custom", Content: "x", Score: 1}}}

	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(custom).
		WithAggregate(
			[]resultNodeNoMeta[stubIntent, struct{}]{
				stubNode[struct{}]{docs: []Document[struct{}]{{ID: "a", Content: "A", Score: 0.9}}},
				stubNode[struct{}]{docs: []Document[struct{}]{{ID: "b", Content: "B", Score: 0.1}}},
			},
			0,
			nil,
		).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), pipelineTestQuery("q"))
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Len() != 2 {
		t.Fatalf("Len() = %d, want 2 (WithAggregate overwrote WithRoot)", rs.Len())
	}
	for _, doc := range rs.Documents() {
		if doc.ID == "custom" {
			t.Fatalf("Documents() = %#v, must not contain custom root", rs.Documents())
		}
	}
}

func TestPipelineBuilderWithRootThenWithConditionalOverwrites(t *testing.T) {
	t.Parallel()

	custom := stubNode[struct{}]{docs: []Document[struct{}]{{ID: "custom", Content: "x", Score: 1}}}
	child := stubNode[struct{}]{docs: []Document[struct{}]{{ID: "x", Content: "y", Score: 1}}}

	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(custom).
		WithConditional(func(Query[stubIntent]) bool { return true }, child).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), pipelineTestQuery("q"))
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Documents()[0].ID != "x" {
		t.Fatalf("Documents()[0].ID = %q, want x (WithConditional overwrote WithRoot)", rs.Documents()[0].ID)
	}
}

func TestPipelineBuilderWithFallbackThenWithRootOverwrites(t *testing.T) {
	t.Parallel()

	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithFallback(
			stubNode[struct{}]{},
			stubNode[struct{}]{docs: []Document[struct{}]{{ID: "fb", Score: 1}}},
		).
		WithRoot(stubNode[struct{}]{docs: []Document[struct{}]{{ID: "custom", Score: 1}}}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), pipelineTestQuery("q"))
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Documents()[0].ID != "custom" {
		t.Fatalf("Documents()[0].ID = %q, want custom (WithRoot overwrote WithFallback)", rs.Documents()[0].ID)
	}
}

func TestPipelineBuilderWithRescueThenWithRootOverwrites(t *testing.T) {
	t.Parallel()

	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRescue(
			errorNode[stubIntent, struct{}]{err: ragy.ErrUnavailable},
			stubNode[struct{}]{docs: []Document[struct{}]{{ID: "rescue", Score: 1}}},
		).
		WithRoot(stubNode[struct{}]{docs: []Document[struct{}]{{ID: "custom", Score: 1}}}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), pipelineTestQuery("q"))
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Documents()[0].ID != "custom" {
		t.Fatalf("Documents()[0].ID = %q, want custom (WithRoot overwrote WithRescue)", rs.Documents()[0].ID)
	}
}

func TestPipelineBuilderWithAggregateThenWithRootOverwrites(t *testing.T) {
	t.Parallel()

	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithAggregate([]resultNodeNoMeta[stubIntent, struct{}]{
			stubNode[struct{}]{docs: []Document[struct{}]{{ID: "a", Score: 0.9}}},
			stubNode[struct{}]{docs: []Document[struct{}]{{ID: "b", Score: 0.1}}},
		}, 0, nil).
		WithRoot(stubNode[struct{}]{docs: []Document[struct{}]{{ID: "custom", Score: 1}}}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), pipelineTestQuery("q"))
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Len() != 1 || rs.Documents()[0].ID != "custom" {
		t.Fatalf("Documents() = %#v, want custom only (WithRoot overwrote WithAggregate)", rs.Documents())
	}
}

func TestPipelineBuilderWithConditionalThenWithRootOverwrites(t *testing.T) {
	t.Parallel()

	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithConditional(
			func(Query[stubIntent]) bool { return true },
			stubNode[struct{}]{docs: []Document[struct{}]{{ID: "x", Score: 1}}},
		).
		WithRoot(stubNode[struct{}]{docs: []Document[struct{}]{{ID: "custom", Score: 1}}}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), pipelineTestQuery("q"))
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Documents()[0].ID != "custom" {
		t.Fatalf("Documents()[0].ID = %q, want custom (WithRoot overwrote WithConditional)", rs.Documents()[0].ID)
	}
}

type suffixPostProcessor[TMeta any] struct {
	suffix string
}

func (p suffixPostProcessor[TMeta]) Process(rs ResultSet[TMeta]) (ResultSet[TMeta], error) {
	docs := rs.Documents()
	if len(docs) == 0 {
		return rs, nil
	}
	doc := docs[0]
	doc.Content += p.suffix
	return NewResultSet([]Document[TMeta]{doc}, DocumentIDResolver[TMeta]{}), nil
}

func TestPipelineBuilderWithPostProcessorsOverwrites(t *testing.T) {
	t.Parallel()

	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(stubNode[struct{}]{docs: []Document[struct{}]{{ID: "a", Content: "base", Score: 1}}}).
		WithPostProcessors(suffixPostProcessor[struct{}]{suffix: "-first"}).
		WithPostProcessors(suffixPostProcessor[struct{}]{suffix: "-second"}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), pipelineTestQuery("q"))
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	got := rs.Documents()[0].Content
	if got != "base-second" {
		t.Fatalf("Content = %q, want base-second (second WithPostProcessors overwrote first)", got)
	}
	if strings.Contains(got, "-first") {
		t.Fatalf("Content = %q, first processor must not run", got)
	}
}

func TestPipelineBuilderShorthandPreservesPostProcessors(t *testing.T) {
	t.Parallel()

	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithPostProcessors(suffixPostProcessor[struct{}]{suffix: "-pp"}).
		WithFallback(
			stubNode[struct{}]{docs: []Document[struct{}]{{ID: "a", Content: "base", Score: 1}}},
			stubNode[struct{}]{},
		).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), pipelineTestQuery("q"))
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Documents()[0].Content != "base-pp" {
		t.Fatalf("Content = %q, want base-pp (postChain survives WithFallback)", rs.Documents()[0].Content)
	}
}

func TestPipelineBuilderWithRescue(t *testing.T) {
	t.Parallel()

	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRescue(
			errorNode[stubIntent, struct{}]{err: ragy.ErrUnavailable},
			stubNode[struct{}]{docs: []Document[struct{}]{{ID: "fb", Content: "hit", Score: 1}}},
		).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), pipelineTestQuery("q"))
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Documents()[0].ID != "fb" {
		t.Fatalf("Documents() = %#v, want rescue doc", rs.Documents())
	}
}

func TestPipelineBuilderWithConditional(t *testing.T) {
	t.Parallel()

	child := stubNode[struct{}]{docs: []Document[struct{}]{{ID: "x", Content: "y", Score: 1}}}

	t.Run("predicate false", func(t *testing.T) {
		t.Parallel()

		pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
			WithConditional(func(Query[stubIntent]) bool { return false }, child).
			Build()
		if err != nil {
			t.Fatalf("Build(): %v", err)
		}

		rs, err := pipeline.Execute(context.Background(), pipelineTestQuery("q"))
		if err != nil {
			t.Fatalf("Retrieve(): %v", err)
		}
		if !rs.IsEmpty() {
			t.Fatalf("Documents() = %#v, want empty", rs.Documents())
		}
	})

	t.Run("predicate true", func(t *testing.T) {
		t.Parallel()

		pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
			WithConditional(func(Query[stubIntent]) bool { return true }, child).
			Build()
		if err != nil {
			t.Fatalf("Build(): %v", err)
		}

		rs, err := pipeline.Execute(context.Background(), pipelineTestQuery("q"))
		if err != nil {
			t.Fatalf("Retrieve(): %v", err)
		}
		if rs.Len() != 1 || rs.Documents()[0].ID != "x" {
			t.Fatalf("Documents() = %#v, want child doc", rs.Documents())
		}
	})
}

func TestPipelineBuilderRejectsNilRoot(t *testing.T) {
	t.Parallel()

	_, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().Build()
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Build() error = %v, want invalid argument", err)
	}
}

type mergeKeyResolver[TMeta any] struct {
	key func(Document[TMeta]) string
}

func (r mergeKeyResolver[TMeta]) Resolve(doc Document[TMeta]) Identity {
	return Identity{MergeKey: r.key(doc), DocumentID: doc.ID}
}

func TestPipelineBuilderWithResolverOrdering(t *testing.T) {
	t.Parallel()

	resolver := mergeKeyResolver[struct{}]{key: func(doc Document[struct{}]) string { return doc.Content }}
	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithFallback(
			stubNode[struct{}]{docs: []Document[struct{}]{{ID: "a", Content: "grp", Score: 1}}},
			stubNode[struct{}]{docs: nil},
		).
		WithResolver(resolver).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), pipelineTestQuery("q"))
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.ResultSet == nil {
		t.Fatal("Execute() ResultSet = nil")
	}
	if rs.Len() != 1 || rs.Documents()[0].ID != "a" {
		t.Fatalf("Documents() = %#v, want primary doc", rs.Documents())
	}
}

func TestPipelineBuilderDoesNotInjectResolverIntoCustomNode(t *testing.T) {
	t.Parallel()

	resolver := mergeKeyResolver[struct{}]{key: func(doc Document[struct{}]) string { return doc.Content }}
	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(stubNode[struct{}]{docs: []Document[struct{}]{
			{ID: "a", Content: "merge-key", Score: 0.9},
			{ID: "b", Content: "merge-key", Score: 0.5},
		}}).
		WithResolver(resolver).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), pipelineTestQuery("q"))
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Len() != 2 {
		t.Fatalf("Len() = %d, want 2 docs (custom stubNode not merged by injected resolver)", rs.Len())
	}
}

func TestPipelineBuilderDoesNotInjectResolverIntoCustomNodeInAggregate(t *testing.T) {
	t.Parallel()

	resolver := mergeKeyResolver[struct{}]{key: func(doc Document[struct{}]) string { return doc.Content }}
	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(resultAggregateNodeNoMeta[stubIntent, struct{}]{
			Nodes: []resultNodeNoMeta[stubIntent, struct{}]{
				stubNode[struct{}]{docs: []Document[struct{}]{{ID: "a", Content: "grp-a", Score: 0.9}}},
				stubNode[struct{}]{docs: []Document[struct{}]{{ID: "b", Content: "grp-b", Score: 0.5}}},
			},
		}).
		WithResolver(resolver).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), pipelineTestQuery("q"))
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Len() != 2 {
		t.Fatalf("Len() = %d, want 2 (custom nodes inside Aggregate keep distinct merge keys)", rs.Len())
	}
}

func TestPipelineBuilderDoesNotInjectResolverIntoCustomNodeInConditional(t *testing.T) {
	t.Parallel()

	resolver := mergeKeyResolver[struct{}]{key: func(doc Document[struct{}]) string { return doc.Content }}
	custom := stubNode[struct{}]{docs: []Document[struct{}]{
		{ID: "a", Content: "merge-key", Score: 0.9},
		{ID: "b", Content: "merge-key", Score: 0.5},
	}}
	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(resultConditionalNodeNoMeta[stubIntent, struct{}]{
			Predicate: func(Query[stubIntent]) bool { return true },
			Child:     custom,
		}).
		WithResolver(resolver).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), pipelineTestQuery("q"))
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Len() != 2 {
		t.Fatalf("Len() = %d, want 2 (custom node inside Conditional not merged by injected resolver)", rs.Len())
	}
}

func TestPipelineBuilderDoesNotInjectResolverIntoCustomNodeInFallback(t *testing.T) {
	t.Parallel()

	resolver := mergeKeyResolver[struct{}]{key: func(doc Document[struct{}]) string { return doc.Content }}
	custom := stubNode[struct{}]{docs: []Document[struct{}]{
		{ID: "a", Content: "merge-key", Score: 0.9},
		{ID: "b", Content: "merge-key", Score: 0.5},
	}}
	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(resultFallbackNodeNoMeta[stubIntent, struct{}]{
			Primary:   stubNode[struct{}]{},
			Secondary: custom,
		}).
		WithResolver(resolver).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), pipelineTestQuery("q"))
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Len() != 2 {
		t.Fatalf("Len() = %d, want 2 (custom node inside Fallback not merged by injected resolver)", rs.Len())
	}
}

func TestPipelineBuilderDoesNotInjectResolverIntoCustomNodeInRescue(t *testing.T) {
	t.Parallel()

	resolver := mergeKeyResolver[struct{}]{key: func(doc Document[struct{}]) string { return doc.Content }}
	custom := stubNode[struct{}]{docs: []Document[struct{}]{
		{ID: "a", Content: "merge-key", Score: 0.9},
		{ID: "b", Content: "merge-key", Score: 0.5},
	}}
	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(resultRescueNodeNoMeta[stubIntent, struct{}]{
			Primary:   errorNode[stubIntent, struct{}]{err: ragy.ErrUnavailable},
			Secondary: custom,
		}).
		WithResolver(resolver).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), pipelineTestQuery("q"))
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Len() != 2 {
		t.Fatalf("Len() = %d, want 2 (custom node inside Rescue not merged by injected resolver)", rs.Len())
	}
}

func TestPostProcessorChainUsesCustomResolver(t *testing.T) {
	t.Parallel()

	resolver := mergeKeyResolver[struct{}]{key: func(doc Document[struct{}]) string { return doc.Content }}
	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(resultRetrieverNodeNoMeta[stubIntent, struct{}]{
			Backend: orchestratorStubBackend[stubIntent, struct{}]{
				docs: []Document[struct{}]{{ID: "a", Content: "grp", Score: 1}},
			},
		}).
		WithPostProcessors(GroupBy(func(struct{}) string { return "g" }, DefaultMergeStrategy[struct{}]())).
		WithResolver(resolver).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}
	rs, err := pipeline.Execute(context.Background(), Query[stubIntent]{
		Text:    "q",
		Options: RetrieveOptions{TopK: 1},
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.ResultSet == nil || rs.IsEmpty() {
		t.Fatalf("Documents() = %#v, want hit", rs.Documents())
	}
}

type partialFailureNode[TIntent, TMeta any] struct {
	docs   []Document[TMeta]
	errors []error
}

func (n partialFailureNode[TIntent, TMeta]) Retrieve(
	_ context.Context,
	_ Query[TIntent],
) (ResultSet[TMeta], error) {
	rs := NewResultSet(n.docs, DocumentIDResolver[TMeta]{})
	return rs, &PartialFailureError[TMeta]{Errors: n.errors, Result: rs}
}

func TestPipelineRetrievePreservesPartialFailureResult(t *testing.T) {
	t.Parallel()

	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(resultAggregateNodeNoMeta[stubIntent, struct{}]{
			Nodes: []resultNodeNoMeta[stubIntent, struct{}]{
				errorNode[stubIntent, struct{}]{err: ragy.ErrEmptyVector},
				stubNode[struct{}]{docs: []Document[struct{}]{{ID: "a", Content: "hit", Score: 1}}},
			},
		}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), pipelineTestQuery("q"))
	if err == nil {
		t.Fatal("Retrieve() error = nil, want partial failure")
	}
	if rs.Len() != 1 || rs.Documents()[0].ID != "a" {
		t.Fatalf("Documents() = %#v, want preserved partial result", rs.Documents())
	}
	var partial *PartialFailureError[struct{}]
	if !errors.As(err, &partial) {
		t.Fatalf("Retrieve() error = %v, want PartialFailureError", err)
	}
}

func TestFallbackNodePreservesPartialFailureWithoutSecondary(t *testing.T) {
	t.Parallel()

	secondary := stubNode[struct{}]{docs: []Document[struct{}]{{ID: "fb", Content: "fallback", Score: 1}}}
	node := resultFallbackNodeNoMeta[stubIntent, struct{}]{
		Primary: partialFailureNode[stubIntent, struct{}]{
			docs:   []Document[struct{}]{{ID: "partial", Content: "hit", Score: 1}},
			errors: []error{ragy.ErrUnavailable},
		},
		Secondary: secondary,
	}
	rs, err := node.Retrieve(context.Background(), Query[stubIntent]{Text: "q"})
	var partial *PartialFailureError[struct{}]
	if !errors.As(err, &partial) {
		t.Fatalf("Retrieve() error = %v, want PartialFailureError", err)
	}
	if rs.Len() != 1 || rs.Documents()[0].ID != "partial" {
		t.Fatalf("Documents() = %#v, want partial without secondary", rs.Documents())
	}
}

func TestRescueNodeDoesNotRescueOnPartialFailure(t *testing.T) {
	t.Parallel()

	node := resultRescueNodeNoMeta[stubIntent, struct{}]{
		Primary: partialFailureNode[stubIntent, struct{}]{
			docs:   []Document[struct{}]{{ID: "partial", Content: "hit", Score: 1}},
			errors: []error{ragy.ErrUnavailable},
		},
		Secondary: stubNode[struct{}]{docs: []Document[struct{}]{{ID: "rescue", Content: "fb", Score: 1}}},
	}
	rs, err := node.Retrieve(context.Background(), Query[stubIntent]{Text: "q"})
	var partial *PartialFailureError[struct{}]
	if !errors.As(err, &partial) {
		t.Fatalf("Retrieve() error = %v, want PartialFailureError", err)
	}
	if rs.Len() != 1 || rs.Documents()[0].ID != "partial" {
		t.Fatalf("Documents() = %#v, want partial without rescue", rs.Documents())
	}
}

func TestRescueNodePropagatesPrimaryErrorWhenSecondaryEmpty(t *testing.T) {
	t.Parallel()

	node := resultRescueNodeNoMeta[stubIntent, struct{}]{
		Primary:   errorNode[stubIntent, struct{}]{err: ragy.ErrUnavailable},
		Secondary: stubNode[struct{}]{},
	}
	rs, err := node.Retrieve(context.Background(), Query[stubIntent]{Text: "q"})
	requireNonNilResultSetOnError(t, rs, err)
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("Retrieve() error = %v, want unavailable when secondary empty", err)
	}
	if !rs.IsEmpty() {
		t.Fatalf("Documents() = %#v, want empty", rs.Documents())
	}
}

func TestConditionalNodePreservesPartialFailureResult(t *testing.T) {
	t.Parallel()

	node := resultConditionalNodeNoMeta[stubIntent, struct{}]{
		Predicate: func(Query[stubIntent]) bool { return true },
		Child: partialFailureNode[stubIntent, struct{}]{
			docs:   []Document[struct{}]{{ID: "partial", Content: "hit", Score: 1}},
			errors: []error{ragy.ErrUnavailable},
		},
	}
	rs, err := node.Retrieve(context.Background(), Query[stubIntent]{Text: "q"})
	var partial *PartialFailureError[struct{}]
	if !errors.As(err, &partial) {
		t.Fatalf("Retrieve() error = %v, want PartialFailureError", err)
	}
	if rs.Len() != 1 || rs.Documents()[0].ID != "partial" {
		t.Fatalf("Documents() = %#v, want preserved partial result", rs.Documents())
	}
}

func TestAggregatePreservesPartialFailureFromNestedChild(t *testing.T) {
	t.Parallel()

	node := resultAggregateNodeNoMeta[stubIntent, struct{}]{
		Nodes: []resultNodeNoMeta[stubIntent, struct{}]{
			partialFailureNode[stubIntent, struct{}]{
				docs:   []Document[struct{}]{{ID: "nested", Content: "hit", Score: 1}},
				errors: []error{ragy.ErrUnavailable},
			},
		},
	}
	rs, err := node.Retrieve(context.Background(), Query[stubIntent]{Text: "q"})
	var partial *PartialFailureError[struct{}]
	if !errors.As(err, &partial) {
		t.Fatalf("Retrieve() error = %v, want PartialFailureError", err)
	}
	if rs.Len() != 1 || rs.Documents()[0].ID != "nested" {
		t.Fatalf("Documents() = %#v, want nested partial docs", rs.Documents())
	}
}

func TestAggregateNestedPartialFailureMergesWithSibling(t *testing.T) {
	t.Parallel()

	node := resultAggregateNodeNoMeta[stubIntent, struct{}]{
		Nodes: []resultNodeNoMeta[stubIntent, struct{}]{
			partialFailureNode[stubIntent, struct{}]{
				docs:   []Document[struct{}]{{ID: "nested", Content: "hit", Score: 1}},
				errors: []error{ragy.ErrUnavailable},
			},
			stubNode[struct{}]{docs: []Document[struct{}]{{ID: "sibling", Content: "ok", Score: 0.5}}},
		},
	}
	rs, err := node.Retrieve(context.Background(), Query[stubIntent]{Text: "q"})
	var partial *PartialFailureError[struct{}]
	if !errors.As(err, &partial) {
		t.Fatalf("Retrieve() error = %v, want PartialFailureError", err)
	}
	if rs.Len() != 2 {
		t.Fatalf("Len() = %d, want merged nested + sibling", rs.Len())
	}
}

func TestBuildFailsWhenRRFRebindFails(t *testing.T) {
	t.Parallel()

	invalidRRF := &ReciprocalRankFusion[struct{}]{k: 0, resolver: DocumentIDResolver[struct{}]{}}
	_, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(resultAggregateNodeNoMeta[stubIntent, struct{}]{
			Nodes:  []resultNodeNoMeta[stubIntent, struct{}]{stubNode[struct{}]{docs: nil}},
			Merger: invalidRRF,
		}).
		Build()
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Build() error = %v, want invalid argument", err)
	}
}

func TestPipelineBuilderWithAggregateCustomMerger(t *testing.T) {
	t.Parallel()

	scoreMerger := NewScoreMerger(DocumentIDResolver[struct{}]{})
	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithAggregate(
			[]resultNodeNoMeta[stubIntent, struct{}]{
				stubNode[struct{}]{docs: []Document[struct{}]{{ID: "a", Content: "A", Score: 0.9}}},
				stubNode[struct{}]{docs: []Document[struct{}]{{ID: "b", Content: "B", Score: 0.1}}},
			},
			2,
			scoreMerger,
		).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), pipelineTestQuery("q"))
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Len() != 2 {
		t.Fatalf("Len() = %d, want 2 docs from score merger aggregate", rs.Len())
	}
}

func TestPipelineBuilderWithAggregateUsesDefaultRRF(t *testing.T) {
	t.Parallel()

	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithAggregate(
			[]resultNodeNoMeta[stubIntent, struct{}]{
				stubNode[struct{}]{docs: []Document[struct{}]{{ID: "a", Content: "A", Score: 0.9}}},
				stubNode[struct{}]{docs: []Document[struct{}]{{ID: "b", Content: "B", Score: 0.1}}},
			},
			2,
			nil,
		).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), pipelineTestQuery("q"))
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Len() != 2 {
		t.Fatalf("Len() = %d, want 2 docs from default RRF aggregate", rs.Len())
	}
	ids := make(map[string]struct{}, 2)
	for _, doc := range rs.Documents() {
		ids[doc.ID] = struct{}{}
	}
	if _, ok := ids["a"]; !ok {
		t.Fatalf("Documents() = %#v, want doc a", rs.Documents())
	}
	if _, ok := ids["b"]; !ok {
		t.Fatalf("Documents() = %#v, want doc b", rs.Documents())
	}
}

func TestPartialFailureErrorUnwrap(t *testing.T) {
	t.Parallel()

	err := &PartialFailureError[struct{}]{
		Errors: []error{ragy.ErrUnavailable, ragy.ErrProtocol},
	}
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatal("errors.Is unavailable = false, want true")
	}
	if !errors.Is(err, ragy.ErrProtocol) {
		t.Fatal("errors.Is protocol = false, want true")
	}
}

func requireNonNilResultSetOnError[TMeta any](t *testing.T, out ResultSet[TMeta], err error) {
	// Partial-preserve contract: on error ResultSet is non-nil (may contain docs).
	// Differs from contracttest.RequireErrorResultSet which requires empty RS.
	t.Helper()
	if err == nil {
		return
	}
	if out == nil {
		t.Fatal("ResultSet = nil, want non-nil set on error")
	}
}

func TestRetrieverNodePreservesBackendResultOnError(t *testing.T) {
	t.Parallel()

	node := resultRetrieverNodeNoMeta[stubIntent, struct{}]{
		Backend: partialBackend[stubIntent, struct{}]{
			docs: []Document[struct{}]{{ID: "a", Content: "hit", Score: 1}},
			err:  ragy.ErrUnavailable,
		},
	}
	rs, err := node.Retrieve(context.Background(), pipelineTestQuery("q"))
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("Retrieve() error = %v, want unavailable", err)
	}
	if rs.Len() != 1 || rs.Documents()[0].ID != "a" {
		t.Fatalf("Documents() = %#v, want preserved backend docs", rs.Documents())
	}
}

func TestRescueNodePreservesSecondaryPartialFailure(t *testing.T) {
	t.Parallel()

	node := resultRescueNodeNoMeta[stubIntent, struct{}]{
		Primary: errorNode[stubIntent, struct{}]{err: ragy.ErrUnavailable},
		Secondary: partialFailureNode[stubIntent, struct{}]{
			docs:   []Document[struct{}]{{ID: "sec", Content: "hit", Score: 1}},
			errors: []error{ragy.ErrProtocol},
		},
	}
	rs, err := node.Retrieve(context.Background(), Query[stubIntent]{Text: "q"})
	var partial *PartialFailureError[struct{}]
	if !errors.As(err, &partial) {
		t.Fatalf("Retrieve() error = %v, want PartialFailureError", err)
	}
	if rs.Len() != 1 || rs.Documents()[0].ID != "sec" {
		t.Fatalf("Documents() = %#v, want secondary partial docs", rs.Documents())
	}
}

type errorProcessor[TMeta any] struct {
	err error
}

func (p errorProcessor[TMeta]) Process(rs ResultSet[TMeta]) (ResultSet[TMeta], error) {
	return rs, p.err
}

func TestPipelinePostChainErrorPreservesRootResult(t *testing.T) {
	t.Parallel()

	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(stubNode[struct{}]{docs: []Document[struct{}]{{ID: "root", Content: "hit", Score: 1}}}).
		WithPostProcessors(errorProcessor[struct{}]{err: ragy.ErrUnavailable}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), pipelineTestQuery("q"))
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("Retrieve() error = %v, want unavailable", err)
	}
	if rs.Len() != 1 || rs.Documents()[0].ID != "root" {
		t.Fatalf("Documents() = %#v, want preserved root docs", rs.Documents())
	}
}

type topKMarkerProcessor[TMeta any] struct {
	resolver IdentityResolver[TMeta]
}

func (p topKMarkerProcessor[TMeta]) Process(rs ResultSet[TMeta]) (ResultSet[TMeta], error) {
	docs := rs.Documents()
	if len(docs) == 0 {
		return rs, nil
	}
	docs[0].Content = "processed"
	return NewResultSet(docs, p.resolver), nil
}

func TestPipelinePartialFailureRunsPostProcessors(t *testing.T) {
	t.Parallel()

	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(partialFailureNode[stubIntent, struct{}]{
			docs:   []Document[struct{}]{{ID: "partial", Content: "raw", Score: 1}},
			errors: []error{ragy.ErrUnavailable},
		}).
		WithPostProcessors(topKMarkerProcessor[struct{}]{resolver: DocumentIDResolver[struct{}]{}}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), pipelineTestQuery("q"))
	var partial *PartialFailureError[struct{}]
	if !errors.As(err, &partial) {
		t.Fatalf("Retrieve() error = %v, want PartialFailureError", err)
	}
	if rs.Len() != 1 || rs.Documents()[0].Content != "processed" {
		t.Fatalf("Documents() = %#v, want post-processed partial docs", rs.Documents())
	}
}

type aggregateBadMergeKeyResolver[TMeta any] struct {
	emptyID string
}

func (r aggregateBadMergeKeyResolver[TMeta]) Resolve(doc Document[TMeta]) Identity {
	if doc.ID == r.emptyID {
		return Identity{MergeKey: "", DocumentID: doc.ID}
	}
	return Identity{MergeKey: doc.ID, DocumentID: doc.ID}
}

func TestAggregateMergeFailurePreservesChildResults(t *testing.T) {
	t.Parallel()

	resolver := aggregateBadMergeKeyResolver[struct{}]{emptyID: "bad"}
	node := resultAggregateNodeNoMeta[stubIntent, struct{}]{
		Resolver: resolver,
		Nodes: []resultNodeNoMeta[stubIntent, struct{}]{
			stubNode[struct{}]{docs: []Document[struct{}]{{ID: "good", Content: "ok", Score: 1}}},
			stubNode[struct{}]{docs: []Document[struct{}]{{ID: "bad", Content: "x", Score: 0.5}}},
		},
		Merger: NewScoreMerger(resolver),
	}
	rs, err := node.Retrieve(context.Background(), Query[stubIntent]{Text: "q"})
	var partial *PartialFailureError[struct{}]
	if !errors.As(err, &partial) {
		t.Fatalf("Retrieve() error = %v, want PartialFailureError", err)
	}
	if rs.IsEmpty() {
		t.Fatal("Documents() empty, want fallback unmerged child results")
	}
}

func TestPipelineBuilderWithPostProcessorsPreservesResultOnError(t *testing.T) {
	t.Parallel()

	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(stubNode[struct{}]{docs: []Document[struct{}]{{ID: "root", Content: "hit", Score: 1}}}).
		WithPostProcessors(errorProcessor[struct{}]{err: ragy.ErrProtocol}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), pipelineTestQuery("q"))
	requireNonNilResultSetOnError(t, rs.ResultSet, err)
	if !errors.Is(err, ragy.ErrProtocol) {
		t.Fatalf("Retrieve() error = %v, want protocol", err)
	}
	if rs.Len() != 1 || rs.Documents()[0].ID != "root" {
		t.Fatalf("Documents() = %#v, want preserved root docs", rs.Documents())
	}
}

func TestAggregateNodePreservesResultsOnContextCancel(t *testing.T) {
	t.Parallel()

	// Concurrency=1 exercises sequential dispatch; high-concurrency cancel is covered below.
	gate := make(chan struct{})
	ctx, cancel := context.WithCancel(context.Background())

	node := resultAggregateNodeNoMeta[stubIntent, struct{}]{
		Nodes: []resultNodeNoMeta[stubIntent, struct{}]{
			stubNode[struct{}]{docs: []Document[struct{}]{{ID: "a", Content: "A", Score: 1}}},
			gateNode[struct{}]{
				docs: []Document[struct{}]{{ID: "b", Content: "B", Score: 0.5}},
				gate: gate,
			},
		},
		Concurrency: 1,
	}

	go func() {
		<-gate
		cancel()
	}()

	rs, err := node.Retrieve(ctx, Query[stubIntent]{Text: "q"})
	if err == nil {
		t.Fatal("Retrieve() error = nil, want context error")
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Retrieve() error = %v, want context canceled", err)
	}
	if rs.Len() != 1 || rs.Documents()[0].ID != "a" {
		t.Fatalf("Documents() = %#v, want partial results before cancel", rs.Documents())
	}
}

func TestAggregateNodeFailsOnContextCancelWithHighConcurrency(t *testing.T) {
	t.Parallel()

	gate := make(chan struct{})
	ctx, cancel := context.WithCancel(context.Background())

	node := resultAggregateNodeNoMeta[stubIntent, struct{}]{
		Nodes: []resultNodeNoMeta[stubIntent, struct{}]{
			stubNode[struct{}]{docs: []Document[struct{}]{{ID: "a", Content: "A", Score: 1}}},
			gateNode[struct{}]{
				docs: []Document[struct{}]{{ID: "b", Content: "B", Score: 0.5}},
				gate: gate,
			},
		},
		Concurrency: 2,
	}

	go func() {
		<-gate
		cancel()
	}()

	rs, err := node.Retrieve(ctx, Query[stubIntent]{Text: "q"})
	if err == nil {
		t.Fatal("Retrieve() error = nil, want context or partial failure error")
	}
	if !errors.Is(err, context.Canceled) {
		var partial *PartialFailureError[struct{}]
		if !errors.As(err, &partial) {
			t.Fatalf("Retrieve() error = %v, want canceled or partial failure", err)
		}
	}
	if rs == nil {
		t.Fatal("Retrieve() rs = nil")
	}
	// Must not silently merge as if the blocked branch returned empty success.
	if rs.Len() == 2 {
		t.Fatalf("Documents() = %#v, want incomplete merge when branch canceled", rs.Documents())
	}
}

type gateNode[TMeta any] struct {
	docs []Document[TMeta]
	gate chan struct{}
}

func (n gateNode[TMeta]) Retrieve(ctx context.Context, _ Query[stubIntent]) (ResultSet[TMeta], error) {
	if n.gate != nil {
		n.gate <- struct{}{}
	}
	select {
	case <-ctx.Done():
		return NewResultSet[TMeta](nil, DocumentIDResolver[TMeta]{}), ctx.Err()
	case <-time.After(time.Second):
		return NewResultSet(n.docs, DocumentIDResolver[TMeta]{}), nil
	}
}

func TestPipelinePartialFailureAndPostChainErrorJoinsErrors(t *testing.T) {
	t.Parallel()

	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(partialFailureNode[stubIntent, struct{}]{
			docs:   []Document[struct{}]{{ID: "partial", Content: "hit", Score: 1}},
			errors: []error{ragy.ErrProtocol},
		}).
		WithPostProcessors(errorProcessor[struct{}]{err: ragy.ErrUnavailable}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), pipelineTestQuery("q"))
	var partial *PartialFailureError[struct{}]
	if !errors.As(err, &partial) {
		t.Fatalf("Retrieve() error = %v, want PartialFailureError in join", err)
	}
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("Retrieve() error = %v, want joined post-chain error", err)
	}
	if rs.Len() != 1 || rs.Documents()[0].ID != "partial" {
		t.Fatalf("Documents() = %#v, want preserved partial docs", rs.Documents())
	}
}

func TestPipelineRetrieveRewrapsResolverOnPartialFailure(t *testing.T) {
	t.Parallel()

	resolver := mergeKeyResolver[struct{}]{key: func(doc Document[struct{}]) string { return doc.Content }}
	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(partialFailureNode[stubIntent, struct{}]{
			docs:   []Document[struct{}]{{ID: "left", Content: "key", Score: 0.2}},
			errors: []error{ragy.ErrUnavailable},
		}).
		WithResolver(resolver).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), pipelineTestQuery("q"))
	var partial *PartialFailureError[struct{}]
	if !errors.As(err, &partial) {
		t.Fatalf("Retrieve() error = %v, want PartialFailureError", err)
	}

	other := NewResultSet([]Document[struct{}]{
		{ID: "right", Content: "key", Score: 0.9},
	}, DocumentIDResolver[struct{}]{})
	merged, err := rs.ResultSet.Merge(other)
	if err != nil {
		t.Fatalf("Merge(): %v", err)
	}
	if merged.Len() != 1 || merged.Documents()[0].ID != "right" {
		t.Fatalf("Merge() = %#v, want winner by pipeline MergeKey", merged.Documents())
	}
}

func TestFallbackNodePreservesPrimaryDocsOnPlainError(t *testing.T) {
	t.Parallel()

	node := resultFallbackNodeNoMeta[stubIntent, struct{}]{
		Primary: resultRetrieverNodeNoMeta[stubIntent, struct{}]{
			Backend: partialBackend[stubIntent, struct{}]{
				docs: []Document[struct{}]{{ID: "primary", Content: "hit", Score: 1}},
				err:  ragy.ErrUnavailable,
			},
		},
		Secondary: stubNode[struct{}]{
			docs: []Document[struct{}]{{ID: "secondary", Content: "fb", Score: 1}},
		},
	}
	rs, err := node.Retrieve(context.Background(), pipelineTestQuery("q"))
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("Retrieve() error = %v, want unavailable", err)
	}
	if rs.Len() != 1 || rs.Documents()[0].ID != "primary" {
		t.Fatalf("Documents() = %#v, want primary docs without fallback", rs.Documents())
	}
}

func TestRescueNodePreservesPrimaryDocsOnPlainError(t *testing.T) {
	t.Parallel()

	node := resultRescueNodeNoMeta[stubIntent, struct{}]{
		Primary: resultRetrieverNodeNoMeta[stubIntent, struct{}]{
			Backend: partialBackend[stubIntent, struct{}]{
				docs: []Document[struct{}]{{ID: "primary", Content: "hit", Score: 1}},
				err:  ragy.ErrUnavailable,
			},
		},
		Secondary: stubNode[struct{}]{
			docs: []Document[struct{}]{{ID: "rescue", Content: "fb", Score: 1}},
		},
	}
	rs, err := node.Retrieve(context.Background(), pipelineTestQuery("q"))
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("Retrieve() error = %v, want unavailable", err)
	}
	if rs.Len() != 1 || rs.Documents()[0].ID != "primary" {
		t.Fatalf("Documents() = %#v, want primary docs without rescue", rs.Documents())
	}
}

func TestAggregateReportsChildErrorWhenMergeEmpty(t *testing.T) {
	t.Parallel()

	node := resultAggregateNodeNoMeta[stubIntent, struct{}]{
		Nodes: []resultNodeNoMeta[stubIntent, struct{}]{
			errorNode[stubIntent, struct{}]{err: ragy.ErrUnavailable},
			stubNode[struct{}]{docs: nil},
		},
	}
	rs, err := node.Retrieve(context.Background(), Query[stubIntent]{Text: "q"})
	requireNonNilResultSetOnError(t, rs, err)
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("Retrieve() error = %v, want unavailable", err)
	}
}

type partialErrorNode[TIntent, TMeta any] struct {
	docs []Document[TMeta]
	err  error
}

func (n partialErrorNode[TIntent, TMeta]) Retrieve(context.Context, Query[TIntent]) (ResultSet[TMeta], error) {
	return NewResultSet(n.docs, DocumentIDResolver[TMeta]{}), n.err
}

type stubEmptyMerger[TMeta any] struct {
	resolver IdentityResolver[TMeta]
}

func (m stubEmptyMerger[TMeta]) Merge(_ context.Context, _ ...ResultSet[TMeta]) (ResultSet[TMeta], error) {
	resolver := m.resolver
	if resolver == nil {
		resolver = DocumentIDResolver[TMeta]{}
	}
	return NewResultSet[TMeta](nil, resolver), nil
}

func TestAggregateReportsPartialWhenMergeEmptyButChildHadDocs(t *testing.T) {
	t.Parallel()

	node := resultAggregateNodeNoMeta[stubIntent, struct{}]{
		Nodes: []resultNodeNoMeta[stubIntent, struct{}]{
			partialErrorNode[stubIntent, struct{}]{
				docs: []Document[struct{}]{{ID: "a", Content: "hit", Score: 0.9}},
				err:  ragy.ErrUnavailable,
			},
			errorNode[stubIntent, struct{}]{err: ragy.ErrUnavailable},
		},
		Merger: stubEmptyMerger[struct{}]{},
	}
	rs, err := node.Retrieve(context.Background(), Query[stubIntent]{Text: "q"})
	partial, ok := AsPartialFailure[struct{}](err)
	if !ok {
		t.Fatalf("Retrieve() error = %v, want partial failure", err)
	}
	if rs.Len() != 1 || rs.Documents()[0].ID != "a" {
		t.Fatalf("Documents() = %#v, want child partial doc", rs.Documents())
	}
	if partial.Result.Len() != 1 {
		t.Fatalf("partial.Result.Len() = %d, want 1", partial.Result.Len())
	}
}

func TestRetrieverNodeUsesInjectedResolverOnPreserve(t *testing.T) {
	t.Parallel()

	resolver := mergeKeyResolver[struct{}]{key: func(doc Document[struct{}]) string { return doc.Content }}
	node := resultRetrieverNodeNoMeta[stubIntent, struct{}]{
		Backend: partialFailureBackend[stubIntent, struct{}]{
			docs: []Document[struct{}]{{ID: "a", Content: "key-a", Score: 0.9}},
		},
		Resolver: resolver,
	}
	rs, err := node.Retrieve(context.Background(), pipelineTestQuery("q"))
	if err == nil {
		t.Fatal("Retrieve() error = nil, want partial failure")
	}
	var partial *PartialFailureError[struct{}]
	if !errors.As(err, &partial) {
		t.Fatalf("Retrieve() error = %v, want PartialFailureError", err)
	}
	if rs.Len() != 1 {
		t.Fatalf("Documents() = %#v, want preserved partial doc", rs.Documents())
	}
	merged, mergeErr := NewResultSet([]Document[struct{}]{
		{ID: "b", Content: "key-a", Score: 0.2},
	}, resolver).Merge(rs)
	if mergeErr != nil {
		t.Fatalf("Merge(): %v", mergeErr)
	}
	if merged.Len() != 1 {
		t.Fatalf("merged Len() = %d, want 1 doc under injected merge key", merged.Len())
	}
}

func TestRetrieverNodeRewrapsResolverOnSuccess(t *testing.T) {
	t.Parallel()

	resolver := mergeKeyResolver[struct{}]{key: func(doc Document[struct{}]) string { return doc.Content }}
	node := resultRetrieverNodeNoMeta[stubIntent, struct{}]{
		Backend: orchestratorStubBackend[stubIntent, struct{}]{
			docs: []Document[struct{}]{{ID: "a", Content: "merge-key", Score: 0.9}},
		},
		Resolver: resolver,
	}
	rs, err := node.Retrieve(context.Background(), pipelineTestQuery("q"))
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	merged, mergeErr := NewResultSet([]Document[struct{}]{
		{ID: "b", Content: "merge-key", Score: 0.2},
	}, resolver).Merge(rs)
	if mergeErr != nil {
		t.Fatalf("Merge(): %v", mergeErr)
	}
	if merged.Len() != 1 {
		t.Fatalf("merged Len() = %d, want 1 doc under pipeline merge key", merged.Len())
	}
}

func TestPipelineBuildOverwritesRetrieverNodeResolver(t *testing.T) {
	t.Parallel()

	nodeResolver := mergeKeyResolver[struct{}]{key: func(doc Document[struct{}]) string { return doc.Content }}
	pipelineResolver := mergeKeyResolver[struct{}]{key: func(doc Document[struct{}]) string { return doc.ID }}
	backend := orchestratorStubBackend[stubIntent, struct{}]{docs: []Document[struct{}]{
		{ID: "a", Content: "same", Score: 0.9},
		{ID: "b", Content: "same", Score: 0.5},
	}}
	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(resultRetrieverNodeNoMeta[stubIntent, struct{}]{
			Backend:  backend,
			Resolver: nodeResolver,
		}).
		WithResolver(pipelineResolver).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), Query[stubIntent]{Text: "q", Options: RetrieveOptions{TopK: 5}})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Len() != 2 {
		t.Fatalf("Len() = %d, want 2 (pipeline WithResolver wins over node resolver)", rs.Len())
	}
}

func TestPipelineBuildOverwritesFallbackNodeResolver(t *testing.T) {
	t.Parallel()

	nodeResolver := mergeKeyResolver[struct{}]{key: func(doc Document[struct{}]) string { return doc.Content }}
	pipelineResolver := mergeKeyResolver[struct{}]{key: func(doc Document[struct{}]) string { return doc.ID }}
	backend := orchestratorStubBackend[stubIntent, struct{}]{docs: []Document[struct{}]{
		{ID: "a", Content: "same", Score: 0.9},
		{ID: "b", Content: "same", Score: 0.5},
	}}
	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(resultFallbackNodeNoMeta[stubIntent, struct{}]{
			Primary: resultRetrieverNodeNoMeta[stubIntent, struct{}]{
				Backend: backend,
			},
			Secondary: stubNode[struct{}]{},
			Resolver:  nodeResolver,
		}).
		WithResolver(pipelineResolver).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), Query[stubIntent]{Text: "q", Options: RetrieveOptions{TopK: 5}})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Len() != 2 {
		t.Fatalf("Len() = %d, want 2 (pipeline WithResolver wins over Fallback.Resolver)", rs.Len())
	}
}

func TestPipelineBuildOverwritesRescueNodeResolver(t *testing.T) {
	t.Parallel()

	nodeResolver := mergeKeyResolver[struct{}]{key: func(doc Document[struct{}]) string { return doc.Content }}
	pipelineResolver := mergeKeyResolver[struct{}]{key: func(doc Document[struct{}]) string { return doc.ID }}
	backend := orchestratorStubBackend[stubIntent, struct{}]{docs: []Document[struct{}]{
		{ID: "a", Content: "same", Score: 0.9},
		{ID: "b", Content: "same", Score: 0.5},
	}}
	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(resultRescueNodeNoMeta[stubIntent, struct{}]{
			Primary: errorNode[stubIntent, struct{}]{err: ragy.ErrUnavailable},
			Secondary: resultRetrieverNodeNoMeta[stubIntent, struct{}]{
				Backend:  backend,
				Resolver: nodeResolver,
			},
			Resolver: nodeResolver,
		}).
		WithResolver(pipelineResolver).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), Query[stubIntent]{Text: "q", Options: RetrieveOptions{TopK: 5}})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Len() != 2 {
		t.Fatalf("Len() = %d, want 2 (pipeline WithResolver wins over Rescue.Resolver)", rs.Len())
	}
}

func TestPipelineBuildOverwritesAggregateNodeResolver(t *testing.T) {
	t.Parallel()

	nodeResolver := mergeKeyResolver[struct{}]{key: func(doc Document[struct{}]) string { return doc.Content }}
	pipelineResolver := mergeKeyResolver[struct{}]{key: func(doc Document[struct{}]) string { return doc.ID }}
	backend := orchestratorStubBackend[stubIntent, struct{}]{docs: []Document[struct{}]{
		{ID: "a", Content: "same", Score: 0.9},
		{ID: "b", Content: "same", Score: 0.5},
	}}
	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(resultAggregateNodeNoMeta[stubIntent, struct{}]{
			Nodes: []resultNodeNoMeta[stubIntent, struct{}]{
				resultRetrieverNodeNoMeta[stubIntent, struct{}]{Backend: backend},
			},
			Resolver: nodeResolver,
		}).
		WithResolver(pipelineResolver).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), Query[stubIntent]{Text: "q", Options: RetrieveOptions{TopK: 5}})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Len() != 2 {
		t.Fatalf("Len() = %d, want 2 (pipeline WithResolver wins over Aggregate.Resolver)", rs.Len())
	}
}

func TestPipelineBuildOverwritesConditionalNodeResolver(t *testing.T) {
	t.Parallel()

	nodeResolver := mergeKeyResolver[struct{}]{key: func(doc Document[struct{}]) string { return doc.Content }}
	pipelineResolver := mergeKeyResolver[struct{}]{key: func(doc Document[struct{}]) string { return doc.ID }}
	backend := orchestratorStubBackend[stubIntent, struct{}]{docs: []Document[struct{}]{
		{ID: "a", Content: "same", Score: 0.9},
		{ID: "b", Content: "same", Score: 0.5},
	}}
	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(resultConditionalNodeNoMeta[stubIntent, struct{}]{
			Predicate: func(Query[stubIntent]) bool { return true },
			Child: resultRetrieverNodeNoMeta[stubIntent, struct{}]{
				Backend:  backend,
				Resolver: nodeResolver,
			},
			Resolver: nodeResolver,
		}).
		WithResolver(pipelineResolver).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), Query[stubIntent]{Text: "q", Options: RetrieveOptions{TopK: 5}})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Len() != 2 {
		t.Fatalf("Len() = %d, want 2 (pipeline WithResolver wins over Conditional.Resolver)", rs.Len())
	}
}

type capturedResolverMerger[TMeta any] struct {
	bound IdentityResolver[TMeta]
}

func (m capturedResolverMerger[TMeta]) Merge(
	ctx context.Context,
	sets ...ResultSet[TMeta],
) (ResultSet[TMeta], error) {
	merger := NewScoreMerger(m.bound)
	return merger.Merge(ctx, sets...)
}

func TestAggregateCustomMergerNotReboundByPipelineResolver(t *testing.T) {
	t.Parallel()

	nodeResolver := mergeKeyResolver[struct{}]{key: func(doc Document[struct{}]) string { return doc.Content }}
	pipelineResolver := mergeKeyResolver[struct{}]{key: func(doc Document[struct{}]) string { return doc.ID }}
	customMerger := capturedResolverMerger[struct{}]{bound: nodeResolver}

	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(resultAggregateNodeNoMeta[stubIntent, struct{}]{
			Merger: customMerger,
			Nodes: []resultNodeNoMeta[stubIntent, struct{}]{
				stubNode[struct{}]{docs: []Document[struct{}]{{ID: "a", Content: "key", Score: 0.9}}},
				stubNode[struct{}]{docs: []Document[struct{}]{{ID: "b", Content: "key", Score: 0.5}}},
			},
		}).
		WithResolver(pipelineResolver).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), pipelineTestQuery("q"))
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Len() != 1 {
		t.Fatalf("Len() = %d, want 1 (custom merger kept constructor resolver, not pipeline)", rs.Len())
	}
}

func TestPipelineBuildRebindsAggregateScoreMergerResolver(t *testing.T) {
	t.Parallel()

	idResolver := DocumentIDResolver[struct{}]{}
	contentResolver := mergeKeyResolver[struct{}]{key: func(doc Document[struct{}]) string { return doc.Content }}

	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(resultAggregateNodeNoMeta[stubIntent, struct{}]{
			Merger: NewScoreMerger(idResolver),
			Nodes: []resultNodeNoMeta[stubIntent, struct{}]{
				stubNode[struct{}]{docs: []Document[struct{}]{
					{ID: "a", Content: "same", Score: 0.9},
				}},
				stubNode[struct{}]{docs: []Document[struct{}]{
					{ID: "b", Content: "same", Score: 0.5},
				}},
			},
		}).
		WithResolver(contentResolver).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), Query[stubIntent]{Text: "q", Options: RetrieveOptions{TopK: 5}})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Len() != 1 {
		t.Fatalf("Len() = %d, want 1 (ScoreMerger rebound to pipeline resolver)", rs.Len())
	}
}

func TestPipelineBuildRebindsAggregateRRFResolver(t *testing.T) {
	t.Parallel()

	idResolver := DocumentIDResolver[struct{}]{}
	contentResolver := mergeKeyResolver[struct{}]{key: func(doc Document[struct{}]) string { return doc.Content }}
	rrf, err := NewReciprocalRankFusion(60, idResolver)
	if err != nil {
		t.Fatalf("NewReciprocalRankFusion(): %v", err)
	}

	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(resultAggregateNodeNoMeta[stubIntent, struct{}]{
			Merger: rrf,
			Nodes: []resultNodeNoMeta[stubIntent, struct{}]{
				stubNode[struct{}]{docs: []Document[struct{}]{
					{ID: "a", Content: "key", Score: 0.9},
					{ID: "b", Content: "other", Score: 0.5},
				}},
				stubNode[struct{}]{docs: []Document[struct{}]{
					{ID: "b", Content: "other", Score: 0.99},
					{ID: "a", Content: "key", Score: 0.1},
				}},
			},
		}).
		WithResolver(contentResolver).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), pipelineTestQuery("q"))
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.IsEmpty() {
		t.Fatal("Documents() empty, want RRF merge with rebound resolver")
	}
}

type stubFailingMerger[TMeta any] struct{}

func (stubFailingMerger[TMeta]) Merge(context.Context, ...ResultSet[TMeta]) (ResultSet[TMeta], error) {
	return NewResultSet[TMeta](nil, DocumentIDResolver[TMeta]{}), ragy.ErrInvalidArgument
}

func TestAggregateFallbackUnmergedUsesPipelineResolver(t *testing.T) {
	t.Parallel()

	resolver := mergeKeyResolver[struct{}]{key: func(doc Document[struct{}]) string { return doc.Content }}
	node := resultAggregateNodeNoMeta[stubIntent, struct{}]{
		Nodes: []resultNodeNoMeta[stubIntent, struct{}]{
			stubNode[struct{}]{docs: []Document[struct{}]{{ID: "a", Content: "key", Score: 0.9}}},
			stubNode[struct{}]{docs: []Document[struct{}]{{ID: "b", Content: "key", Score: 0.5}}},
		},
		Merger:   stubFailingMerger[struct{}]{},
		Resolver: resolver,
	}
	rs, err := node.Retrieve(context.Background(), Query[stubIntent]{Text: "q"})
	if err == nil {
		t.Fatal("Retrieve() error = nil, want merge failure")
	}
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Retrieve() error = %v, want invalid argument", err)
	}
	if rs.Len() != 1 {
		t.Fatalf("Documents() = %#v, want one doc merged by content key", rs.Documents())
	}
}

func TestAggregateEmptyMergeFallbackIncludesFbErr(t *testing.T) {
	t.Parallel()

	resolver := aggregateBadMergeKeyResolver[struct{}]{emptyID: "bad"}
	node := resultAggregateNodeNoMeta[stubIntent, struct{}]{
		Resolver: resolver,
		Merger: stubEmptyMerger[struct{}]{
			resolver: resolver,
		},
		Nodes: []resultNodeNoMeta[stubIntent, struct{}]{
			partialFailureNode[stubIntent, struct{}]{
				docs:   []Document[struct{}]{{ID: "good", Content: "ok", Score: 0.9}},
				errors: []error{ragy.ErrUnavailable},
			},
			stubNode[struct{}]{docs: []Document[struct{}]{{ID: "bad", Content: "x", Score: 0.5}}},
		},
	}
	rs, err := node.Retrieve(context.Background(), Query[stubIntent]{Text: "q"})
	var partial *PartialFailureError[struct{}]
	if !errors.As(err, &partial) {
		t.Fatalf("Retrieve() error = %v, want PartialFailureError", err)
	}
	if rs.IsEmpty() || rs.Documents()[0].ID != "good" {
		t.Fatalf("Documents() = %#v, want fallback good doc", rs.Documents())
	}
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("error = %v, want unavailable in chain", err)
	}
	hasInvalidArgument := false
	for _, childErr := range partial.Errors {
		if errors.Is(childErr, ragy.ErrInvalidArgument) {
			hasInvalidArgument = true
		}
	}
	if !hasInvalidArgument {
		t.Fatalf("partial.Errors = %v, want fallback merge error", partial.Errors)
	}
}

func TestAggregateMergeFailurePreservesChildErrors(t *testing.T) {
	t.Parallel()

	node := resultAggregateNodeNoMeta[stubIntent, struct{}]{
		Merger: stubFailingMerger[struct{}]{},
		Nodes: []resultNodeNoMeta[stubIntent, struct{}]{
			partialFailureNode[stubIntent, struct{}]{
				docs:   nil,
				errors: []error{ragy.ErrUnavailable},
			},
			errorNode[stubIntent, struct{}]{err: ragy.ErrProtocol},
		},
	}
	_, err := node.Retrieve(context.Background(), Query[stubIntent]{Text: "q"})
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("error = %v, want unavailable", err)
	}
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("error = %v, want merge failure", err)
	}
	if !errors.Is(err, ragy.ErrProtocol) {
		t.Fatalf("error = %v, want protocol child error", err)
	}
}

func TestPipelinePartialFailureResultMatchesReturnedSetAfterPostProcess(t *testing.T) {
	t.Parallel()

	resolver := mergeKeyResolver[struct{}]{key: func(doc Document[struct{}]) string { return doc.Content }}
	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithResolver(resolver).
		WithRoot(resultAggregateNodeNoMeta[stubIntent, struct{}]{
			Nodes: []resultNodeNoMeta[stubIntent, struct{}]{
				partialFailureNode[stubIntent, struct{}]{
					docs: []Document[struct{}]{
						{ID: "a", Content: "alpha", Score: 0.9, Meta: struct{}{}},
						{ID: "b", Content: "beta", Score: 0.5, Meta: struct{}{}},
					},
					errors: []error{ragy.ErrUnavailable},
				},
				errorNode[stubIntent, struct{}]{err: ragy.ErrProtocol},
			},
		}).
		WithPostProcessors(TopPerGroup(func(struct{}) string { return "g" }, 1)).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), pipelineTestQuery("q"))
	partial, ok := AsPartialFailure[struct{}](err)
	if !ok {
		t.Fatalf("Retrieve() error = %v, want partial failure", err)
	}
	if partial.Result.Len() != rs.Len() {
		t.Fatalf("partial.Result.Len() = %d, returned Len() = %d, want equal", partial.Result.Len(), rs.Len())
	}
	if partial.Result.Documents()[0].ID != rs.Documents()[0].ID {
		t.Fatalf("partial.Result doc ID = %q, returned = %q", partial.Result.Documents()[0].ID, rs.Documents()[0].ID)
	}
	other, mergeErr := partial.Result.Merge(NewResultSet([]Document[struct{}]{
		{ID: "c", Content: "alpha", Score: 0.1},
	}, resolver))
	if mergeErr != nil {
		t.Fatalf("partial.Result.Merge() error = %v", mergeErr)
	}
	if other.Len() != 1 {
		t.Fatalf("merged Len() = %d, want 1 doc by content merge key", other.Len())
	}
}

func TestPipelineRetrieveAppliesTopKWithoutPostChain(t *testing.T) {
	t.Parallel()

	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(stubNode[struct{}]{docs: []Document[struct{}]{
			{ID: "a", Content: "one", Score: 0.9},
			{ID: "b", Content: "two", Score: 0.5},
			{ID: "c", Content: "three", Score: 0.1},
		}}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), Query[stubIntent]{
		Text:    "q",
		Options: RetrieveOptions{TopK: 1},
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Len() != 1 || rs.Documents()[0].ID != "a" {
		t.Fatalf("Documents() = %#v, want top scored doc after terminal top_k", rs.Documents())
	}
}

func TestPipelinePartialFailureResultAfterTerminalOptions(t *testing.T) {
	t.Parallel()

	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(partialFailureNode[stubIntent, struct{}]{
			docs: []Document[struct{}]{
				{ID: "a", Content: "alpha", Score: 0.2},
				{ID: "b", Content: "beta", Score: 0.1},
			},
			errors: []error{ragy.ErrUnavailable},
		}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), Query[stubIntent]{
		Text:    "q",
		Options: RetrieveOptions{TopK: 5, MinSimilarity: 0.5},
	})
	partial, ok := AsPartialFailure[struct{}](err)
	if !ok {
		t.Fatalf("Retrieve() error = %v, want partial failure", err)
	}
	if !rs.IsEmpty() {
		t.Fatalf("Documents() = %#v, want empty after min similarity", rs.Documents())
	}
	if partial.Result.Len() != rs.Len() {
		t.Fatalf("partial.Result.Len() = %d, returned Len() = %d, want equal", partial.Result.Len(), rs.Len())
	}
}

func TestPipelinePartialFailureResultMatchesReturnedSetWithoutPostChain(t *testing.T) {
	t.Parallel()

	pipeline, err := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(partialFailureNode[stubIntent, struct{}]{
			docs: []Document[struct{}]{
				{ID: "a", Content: "alpha", Score: 0.9},
				{ID: "b", Content: "beta", Score: 0.5},
			},
			errors: []error{ragy.ErrUnavailable},
		}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), Query[stubIntent]{
		Text:    "q",
		Options: RetrieveOptions{TopK: 1},
	})
	partial, ok := AsPartialFailure[struct{}](err)
	if !ok {
		t.Fatalf("Retrieve() error = %v, want partial failure", err)
	}
	if partial.Result.Len() != rs.Len() {
		t.Fatalf("partial.Result.Len() = %d, returned Len() = %d, want equal", partial.Result.Len(), rs.Len())
	}
	if partial.Result.Documents()[0].ID != rs.Documents()[0].ID {
		t.Fatalf("partial.Result doc ID = %q, returned = %q", partial.Result.Documents()[0].ID, rs.Documents()[0].ID)
	}
}

func TestFallbackNodePropagatesSecondaryError(t *testing.T) {
	t.Parallel()

	node := resultFallbackNodeNoMeta[stubIntent, struct{}]{
		Primary:   stubNode[struct{}]{},
		Secondary: errorNode[stubIntent, struct{}]{err: ragy.ErrUnavailable},
	}
	rs, err := node.Retrieve(context.Background(), Query[stubIntent]{Text: "q"})
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("Retrieve() error = %v, want unavailable", err)
	}
	if rs == nil {
		t.Fatal("ResultSet = nil, want non-nil empty set")
	}
}

func TestRetrieverNodeRejectsNilBackend(t *testing.T) {
	t.Parallel()

	node := resultRetrieverNodeNoMeta[stubIntent, struct{}]{Backend: nil}
	rs, err := node.Retrieve(context.Background(), Query[stubIntent]{Text: "q"})
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Retrieve() error = %v, want invalid argument", err)
	}
	if rs == nil || !rs.IsEmpty() {
		t.Fatalf("ResultSet = %#v, want empty", rs)
	}
}

func TestRetrieverNodeRejectsInvalidRetrieveOptions(t *testing.T) {
	t.Parallel()

	node := resultRetrieverNodeNoMeta[stubIntent, struct{}]{
		Backend: orchestratorStubBackend[stubIntent, struct{}]{},
	}
	_, err := node.Retrieve(context.Background(), Query[stubIntent]{Text: "q"})
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Retrieve() error = %v, want invalid argument", err)
	}
}

func TestRescueNodePropagatesPrimaryWhenSecondaryNil(t *testing.T) {
	t.Parallel()

	node := resultRescueNodeNoMeta[stubIntent, struct{}]{
		Primary: errorNode[stubIntent, struct{}]{err: ragy.ErrUnavailable},
	}
	rs, err := node.Retrieve(context.Background(), Query[stubIntent]{Text: "q"})
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("Retrieve() error = %v, want unavailable", err)
	}
	if rs == nil || !rs.IsEmpty() {
		t.Fatalf("ResultSet = %#v, want empty", rs)
	}
}

type intentWithMode struct {
	Mode string
}

type intentStubNode[TMeta any] struct {
	docs []Document[TMeta]
}

func (n intentStubNode[TMeta]) Retrieve(_ context.Context, _ Query[intentWithMode]) (ResultSet[TMeta], error) {
	return NewResultSet(n.docs, DocumentIDResolver[TMeta]{}), nil
}

func TestConditionalNodeUsesQueryIntent(t *testing.T) {
	t.Parallel()

	child := intentStubNode[struct{}]{docs: []Document[struct{}]{{ID: "hit", Content: "ok", Score: 1}}}
	node := resultConditionalNodeNoMeta[intentWithMode, struct{}]{
		Predicate: func(query Query[intentWithMode]) bool {
			return query.Intent.Mode == "run"
		},
		Child: child,
	}

	runRS, err := node.Retrieve(context.Background(), Query[intentWithMode]{
		Text:   "q",
		Intent: intentWithMode{Mode: "run"},
	})
	if err != nil {
		t.Fatalf("Retrieve(run): %v", err)
	}
	if runRS.Len() != 1 {
		t.Fatalf("Retrieve(run) Len() = %d, want 1", runRS.Len())
	}

	skipRS, err := node.Retrieve(context.Background(), Query[intentWithMode]{
		Text:   "q",
		Intent: intentWithMode{Mode: "skip"},
	})
	if err != nil {
		t.Fatalf("Retrieve(skip): %v", err)
	}
	if !skipRS.IsEmpty() {
		t.Fatalf("Retrieve(skip) = %#v, want empty", skipRS.Documents())
	}
}

func TestResultPipelinePlanBinderCanBindMissingOptions(t *testing.T) {
	t.Parallel()

	spy := &querySpyBackend[intentWithMode, struct{}]{
		orchestratorStubBackend: orchestratorStubBackend[intentWithMode, struct{}]{
			docs: []Document[struct{}]{{ID: "hit", Content: "ok", Score: 1}},
		},
	}
	pipeline, err := newResultPipelineBuilderNoMeta[intentWithMode, struct{}]().
		WithPlanBinder(RequestPlanBinderFunc[intentWithMode, NoRequestMeta, NoExecutionMeta](
			func(
				_ context.Context,
				req Query[intentWithMode],
				_ *PlannedQuery[intentWithMode],
				exec NoExecutionMeta,
			) (BoundRequest[intentWithMode, NoRequestMeta, NoExecutionMeta], error) {
				req.Options.TopK = 1
				return BoundRequest[intentWithMode, NoRequestMeta, NoExecutionMeta]{
					Request:  req,
					Executed: exec,
				}, nil
			},
		)).
		WithRoot(resultRetrieverNodeNoMeta[intentWithMode, struct{}]{Backend: spy}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	result, err := pipeline.Execute(context.Background(), Query[intentWithMode]{Text: "q"})
	if err != nil {
		t.Fatalf("Execute(): %v", err)
	}
	if spy.lastRequest.Options.TopK != 1 {
		t.Fatalf("backend TopK = %d, want binder-bound TopK", spy.lastRequest.Options.TopK)
	}
	if result.Len() != 1 {
		t.Fatalf("Len() = %d, want backend hit", result.Len())
	}
}

type querySpyBackend[TIntent, TMeta any] struct {
	orchestratorStubBackend[TIntent, TMeta]

	lastRequest Query[TIntent]
}

func (b *querySpyBackend[TIntent, TMeta]) Retrieve(
	_ context.Context,
	req Query[TIntent],
) (ResultSet[TMeta], error) {
	b.lastRequest = req
	return b.orchestratorStubBackend.Retrieve(context.Background(), req)
}

func TestRetrieverNodePassesRequestEnvelopeToBackend(t *testing.T) {
	t.Parallel()

	spy := &querySpyBackend[intentWithMode, struct{}]{
		orchestratorStubBackend: orchestratorStubBackend[intentWithMode, struct{}]{
			docs: []Document[struct{}]{{ID: "hit", Content: "ok", Score: 1}},
		},
	}
	node := resultRetrieverNodeNoMeta[intentWithMode, struct{}]{Backend: spy}

	_, err := node.Retrieve(context.Background(), Query[intentWithMode]{
		Text:    "hello",
		Intent:  intentWithMode{Mode: "secret-mode"},
		Options: RetrieveOptions{TopK: 1},
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if spy.lastRequest.Text != "hello" || spy.lastRequest.Intent.Mode != "secret-mode" {
		t.Fatalf("backend request = %#v, want full request envelope", spy.lastRequest)
	}
}

func TestPipelinePlannerAttachesPlanBeforeBackend(t *testing.T) {
	t.Parallel()

	spy := &querySpyBackend[intentWithMode, struct{}]{
		orchestratorStubBackend: orchestratorStubBackend[intentWithMode, struct{}]{
			docs: []Document[struct{}]{{ID: "hit", Content: "ok", Score: 1}},
		},
	}
	pipeline, err := newResultPipelineBuilderNoMeta[intentWithMode, struct{}]().
		WithPlanner(QueryPlannerFunc[intentWithMode, NoRequestMeta](
			func(_ context.Context, req Query[intentWithMode]) (PlannedQuery[intentWithMode], error) {
				return PlannedQuery[intentWithMode]{
					Text:         strings.TrimSpace(req.Text),
					ExpandedText: "expanded query",
					Intent:       req.Intent,
					CacheKey:     "mode:" + req.Intent.Mode,
					Diagnostics:  []PlannerDiagnostic{{Key: "source", Value: "test"}},
				}, nil
			},
		)).
		WithRoot(resultRetrieverNodeNoMeta[intentWithMode, struct{}]{Backend: spy}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	_, err = pipeline.Execute(context.Background(), Query[intentWithMode]{
		Text:    "  raw query  ",
		Intent:  intentWithMode{Mode: "run"},
		Options: RetrieveOptions{TopK: 1},
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if spy.lastRequest.Plan == nil {
		t.Fatal("backend request Plan = nil, want planned query")
	}
	if got := spy.lastRequest.EffectiveText(); got != "expanded query" {
		t.Fatalf("EffectiveText() = %q, want expanded query", got)
	}
	if spy.lastRequest.Plan.CacheKey != "mode:run" {
		t.Fatalf("Plan.CacheKey = %q, want mode:run", spy.lastRequest.Plan.CacheKey)
	}
}

type requestMetaFixture struct {
	TraceID string
}

type requestMetaSpyBackend[TIntent, TRequestMeta, TMeta any] struct {
	docs        []Document[TMeta]
	lastRequest Request[TIntent, TRequestMeta]
}

func (b *requestMetaSpyBackend[TIntent, TRequestMeta, TMeta]) Retrieve(
	_ context.Context,
	req Request[TIntent, TRequestMeta],
) (ResultSet[TMeta], error) {
	b.lastRequest = req
	return NewResultSet(b.docs, DocumentIDResolver[TMeta]{}), nil
}

func TestRequestPipelinePassesTypedRequestMetaToPlannerAndBackend(t *testing.T) {
	t.Parallel()

	spy := &requestMetaSpyBackend[intentWithMode, requestMetaFixture, struct{}]{
		docs: []Document[struct{}]{{ID: "hit", Content: "ok", Score: 1}},
	}
	var plannerMeta requestMetaFixture
	pipeline, err := newResultPipelineBuilder[intentWithMode, requestMetaFixture, struct{}]().
		WithPlanner(QueryPlannerFunc[intentWithMode, requestMetaFixture](
			func(_ context.Context, req Request[intentWithMode, requestMetaFixture]) (PlannedQuery[intentWithMode], error) {
				plannerMeta = req.Meta
				return PlannedQuery[intentWithMode]{
					Text:     strings.TrimSpace(req.Text),
					Intent:   req.Intent,
					CacheKey: req.Meta.TraceID,
				}, nil
			},
		)).
		WithRoot(resultRetrieverNode[intentWithMode, requestMetaFixture, struct{}]{Backend: spy}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	_, err = pipeline.Execute(context.Background(), Request[intentWithMode, requestMetaFixture]{
		Text:    "  raw  ",
		Intent:  intentWithMode{Mode: "run"},
		Meta:    requestMetaFixture{TraceID: "trace-1"},
		Options: RetrieveOptions{TopK: 1},
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if plannerMeta.TraceID != "trace-1" {
		t.Fatalf("planner meta = %#v, want trace-1", plannerMeta)
	}
	if spy.lastRequest.Meta.TraceID != "trace-1" {
		t.Fatalf("backend meta = %#v, want trace-1", spy.lastRequest.Meta)
	}
	if spy.lastRequest.Plan == nil || spy.lastRequest.Plan.CacheKey != "trace-1" {
		t.Fatalf("backend plan = %#v, want cache key trace-1", spy.lastRequest.Plan)
	}
}

func TestPipelineUsesPreplannedQueryWithoutCallingPlanner(t *testing.T) {
	t.Parallel()

	spy := &querySpyBackend[intentWithMode, struct{}]{
		orchestratorStubBackend: orchestratorStubBackend[intentWithMode, struct{}]{
			docs: []Document[struct{}]{{ID: "hit", Content: "ok", Score: 1}},
		},
	}
	plannerCalls := 0
	pipeline, err := newResultPipelineBuilderNoMeta[intentWithMode, struct{}]().
		WithPlanner(QueryPlannerFunc[intentWithMode, NoRequestMeta](
			func(_ context.Context, _ Query[intentWithMode]) (PlannedQuery[intentWithMode], error) {
				plannerCalls++
				return PlannedQuery[intentWithMode]{Text: "unexpected"}, nil
			},
		)).
		WithRoot(resultRetrieverNodeNoMeta[intentWithMode, struct{}]{Backend: spy}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	_, err = pipeline.Execute(context.Background(), Query[intentWithMode]{
		Text: "raw",
		Plan: &PlannedQuery[intentWithMode]{
			ExpandedText: "cached expanded",
			CacheKey:     "cached",
		},
		Options: RetrieveOptions{TopK: 1},
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if plannerCalls != 0 {
		t.Fatalf("planner calls = %d, want 0 for preplanned request", plannerCalls)
	}
	if got := spy.lastRequest.EffectiveText(); got != "cached expanded" {
		t.Fatalf("EffectiveText() = %q, want cached expanded", got)
	}
}

func TestConditionalNodeRunsChildWhenPredicateNil(t *testing.T) {
	t.Parallel()

	node := resultConditionalNodeNoMeta[stubIntent, struct{}]{
		Predicate: nil,
		Child:     stubNode[struct{}]{docs: []Document[struct{}]{{ID: "hit", Score: 1}}},
	}

	rs, err := node.Retrieve(context.Background(), Query[stubIntent]{Text: "q"})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.IsEmpty() || rs.Documents()[0].ID != "hit" {
		t.Fatalf("Documents() = %#v, want hit when Predicate nil", rs.Documents())
	}
}

func TestAggregateFallbackOrderingDiffersFromRRF(t *testing.T) {
	t.Parallel()

	left := NewResultSet([]Document[struct{}]{
		{ID: "rank-first", Content: "a", Score: 0.2},
	}, DocumentIDResolver[struct{}]{})
	right := NewResultSet([]Document[struct{}]{
		{ID: "score-first", Content: "b", Score: 0.99},
		{ID: "tail", Content: "c", Score: 0.98},
	}, DocumentIDResolver[struct{}]{})

	rrf, err := NewReciprocalRankFusion[struct{}](60, DocumentIDResolver[struct{}]{})
	if err != nil {
		t.Fatalf("NewReciprocalRankFusion(): %v", err)
	}
	rrfOut, err := rrf.Merge(context.Background(), left, right)
	if err != nil {
		t.Fatalf("RRF Merge(): %v", err)
	}

	node := resultAggregateNodeNoMeta[stubIntent, struct{}]{
		Nodes: []resultNodeNoMeta[stubIntent, struct{}]{
			stubNode[struct{}]{docs: left.Documents()},
			stubNode[struct{}]{docs: right.Documents()},
		},
		Merger: stubFailingMerger[struct{}]{},
	}
	fallbackOut, err := node.Retrieve(context.Background(), Query[stubIntent]{Text: "q"})
	if err == nil {
		t.Fatal("Retrieve() error = nil, want merge failure")
	}
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Retrieve() error = %v, want invalid argument from failing merger", err)
	}
	if fallbackOut.Documents()[0].ID != "score-first" {
		t.Fatalf("fallback top = %q, want score-first (raw score merge)", fallbackOut.Documents()[0].ID)
	}
	if rrfOut.Documents()[0].ID == fallbackOut.Documents()[0].ID {
		t.Fatalf(
			"RRF top = %q, fallback top = %q, want different ordering",
			rrfOut.Documents()[0].ID,
			fallbackOut.Documents()[0].ID,
		)
	}
}

func scoresByID(docs []Document[struct{}]) map[string]float64 {
	out := make(map[string]float64, len(docs))
	for _, doc := range docs {
		out[doc.ID] = doc.Score
	}
	return out
}

type bareCustomNode[TMeta any] struct {
	docs []Document[TMeta]
}

func (n bareCustomNode[TMeta]) Retrieve(_ context.Context, _ Query[stubIntent]) (ResultSet[TMeta], error) {
	return NewResultSet(n.docs, DocumentIDResolver[TMeta]{}), nil
}

func TestCustomNodeWithoutResolverKeepsDocumentIDMergeKey(t *testing.T) {
	t.Parallel()

	node := bareCustomNode[struct{}]{
		docs: []Document[struct{}]{{ID: "a", Content: "merge-key", Score: 0.9}},
	}
	out, err := node.Retrieve(context.Background(), Query[stubIntent]{Text: "q"})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	merged, mergeErr := out.Merge(NewResultSet(
		[]Document[struct{}]{{ID: "b", Content: "merge-key", Score: 0.1}},
		DocumentIDResolver[struct{}]{},
	))
	if mergeErr != nil {
		t.Fatalf("Merge(): %v", mergeErr)
	}
	if merged.Len() != 2 {
		t.Fatalf("merged Len() = %d, want 2 docs (custom node kept DocumentID resolver)", merged.Len())
	}
}

type catalogVectorIntent struct {
	AllowWeb bool
}

func TestPipelineCatalogVectorFallbackGraph(t *testing.T) {
	t.Parallel()

	catalog := orchestratorStubBackend[catalogVectorIntent, struct{}]{}
	vector := orchestratorStubBackend[catalogVectorIntent, struct{}]{}
	web := orchestratorStubBackend[catalogVectorIntent, struct{}]{
		docs: []Document[struct{}]{{ID: "web-1", Content: "web", Score: 0.8}},
	}

	pipeline, err := newResultPipelineBuilderNoMeta[catalogVectorIntent, struct{}]().
		WithRoot(resultFallbackNodeNoMeta[catalogVectorIntent, struct{}]{
			Primary: resultAggregateNodeNoMeta[catalogVectorIntent, struct{}]{
				Nodes: []resultNodeNoMeta[catalogVectorIntent, struct{}]{
					resultRetrieverNodeNoMeta[catalogVectorIntent, struct{}]{Backend: catalog},
					resultConditionalNodeNoMeta[catalogVectorIntent, struct{}]{
						Predicate: func(query Query[catalogVectorIntent]) bool {
							return len(query.Options.Vector) > 0
						},
						Child: resultRetrieverNodeNoMeta[catalogVectorIntent, struct{}]{Backend: vector},
					},
				},
			},
			Secondary: resultConditionalNodeNoMeta[catalogVectorIntent, struct{}]{
				Predicate: func(query Query[catalogVectorIntent]) bool {
					return query.Intent.AllowWeb
				},
				Child: resultRetrieverNodeNoMeta[catalogVectorIntent, struct{}]{Backend: web},
			},
		}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), Query[catalogVectorIntent]{
		Text:    "q",
		Intent:  catalogVectorIntent{AllowWeb: true},
		Options: RetrieveOptions{TopK: 5},
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Len() != 1 || rs.Documents()[0].ID != "web-1" {
		t.Fatalf("Documents() = %#v, want web fallback doc", rs.Documents())
	}
}

func TestRescueNestedFallbackRespectsIntentGate(t *testing.T) {
	t.Parallel()

	type rescueSearchIntent struct {
		AllowWeb bool
	}

	catalog := orchestratorStubBackend[rescueSearchIntent, struct{}]{}
	vector := orchestratorFailingBackend[rescueSearchIntent, struct{}]{}
	web := orchestratorStubBackend[rescueSearchIntent, struct{}]{
		docs: []Document[struct{}]{{ID: "web-1", Content: "web", Score: 0.8}},
	}
	webIfAllowed := func(query Query[rescueSearchIntent]) bool {
		return query.Intent.AllowWeb
	}
	webBranch := func() resultNodeNoMeta[rescueSearchIntent, struct{}] {
		return resultConditionalNodeNoMeta[rescueSearchIntent, struct{}]{
			Predicate: webIfAllowed,
			Child:     resultRetrieverNodeNoMeta[rescueSearchIntent, struct{}]{Backend: web},
		}
	}

	pipeline, err := newResultPipelineBuilderNoMeta[rescueSearchIntent, struct{}]().
		WithRoot(resultRescueNodeNoMeta[rescueSearchIntent, struct{}]{
			Primary: resultFallbackNodeNoMeta[rescueSearchIntent, struct{}]{
				Primary: resultAggregateNodeNoMeta[rescueSearchIntent, struct{}]{
					Nodes: []resultNodeNoMeta[rescueSearchIntent, struct{}]{
						resultRetrieverNodeNoMeta[rescueSearchIntent, struct{}]{Backend: catalog},
						resultRetrieverNodeNoMeta[rescueSearchIntent, struct{}]{Backend: vector},
					},
				},
				Secondary: webBranch(),
			},
			Secondary: webBranch(),
		}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), Query[rescueSearchIntent]{
		Text:    "query",
		Intent:  rescueSearchIntent{AllowWeb: false},
		Options: RetrieveOptions{TopK: 5},
	})
	if !rs.IsEmpty() {
		t.Fatalf("Documents() = %#v, want empty when AllowWeb false", rs.Documents())
	}
	for _, doc := range rs.Documents() {
		if doc.ID == "web-1" {
			t.Fatal("web branch ran with AllowWeb=false")
		}
	}
	if err != nil && !errors.Is(err, ragy.ErrUnavailable) {
		var partial *PartialFailureError[struct{}]
		if !errors.As(err, &partial) {
			t.Fatalf("Retrieve() error = %v, want unavailable or partial failure", err)
		}
	}
}
