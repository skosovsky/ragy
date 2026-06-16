package contracttest_test

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/internal/contracttest"
	"github.com/skosovsky/ragy/retrieval"
)

type stubIntent struct{}

type stubNode[TIntent any] struct {
	docs []retrieval.Document[struct{}]
}

func (n stubNode[TIntent]) Retrieve(
	_ context.Context,
	_ retrieval.Query[TIntent],
) (retrieval.ResultSet[struct{}], error) {
	return retrieval.NewResultSet(n.docs, retrieval.DocumentIDResolver[struct{}]{}), nil
}

type errorNode[TIntent any] struct {
	err error
}

func (n errorNode[TIntent]) Retrieve(
	context.Context,
	retrieval.Query[TIntent],
) (retrieval.ResultSet[struct{}], error) {
	return retrieval.NewResultSet[struct{}](nil, retrieval.DocumentIDResolver[struct{}]{}), n.err
}

type partialFailureNode[TIntent any] struct {
	docs   []retrieval.Document[struct{}]
	errors []error
}

func (n partialFailureNode[TIntent]) Retrieve(
	_ context.Context,
	_ retrieval.Query[TIntent],
) (retrieval.ResultSet[struct{}], error) {
	rs := retrieval.NewResultSet(n.docs, retrieval.DocumentIDResolver[struct{}]{})
	return rs, &retrieval.PartialFailureError[struct{}]{Errors: n.errors, Result: rs}
}

type stubFailingMerger struct{}

func (stubFailingMerger) Merge(
	context.Context,
	...retrieval.ResultSet[struct{}],
) (retrieval.ResultSet[struct{}], error) {
	return retrieval.NewResultSet[struct{}](nil, retrieval.DocumentIDResolver[struct{}]{}), ragy.ErrInvalidArgument
}

type pipelineIntent struct {
	Mode string
}

func TestPipelineContractConformance(t *testing.T) {
	contracttest.RunPipelineRetrieveOptionsInvalidSuite(
		t,
		func(t *testing.T) *retrieval.Pipeline[stubIntent, struct{}] {
			t.Helper()
			p, err := retrieval.NewPipelineBuilder[stubIntent, struct{}]().
				WithRoot(stubNode[stubIntent]{
					docs: []retrieval.Document[struct{}]{{ID: "x", Score: 1}},
				}).
				Build()
			if err != nil {
				t.Fatalf("Build(): %v", err)
			}
			return p
		},
	)

	contracttest.RunPipelineNodeSemanticsSuite(t, contracttest.PipelineNodeSemanticsConfig{
		FallbackPrimary: stubNode[struct{}]{},
		FallbackSecondary: stubNode[struct{}]{
			docs: []retrieval.Document[struct{}]{{ID: "fb", Score: 1}},
		},
		FallbackWantID: "fb",
		FallbackPrimaryHasResults: stubNode[struct{}]{
			docs: []retrieval.Document[struct{}]{{ID: "primary", Score: 1}},
		},
		FallbackSkipsSecondary: stubNode[struct{}]{
			docs: []retrieval.Document[struct{}]{{ID: "secondary", Score: 1}},
		},
		FallbackSkipsSecondaryWantID: "primary",
		FallbackPrimaryFail: errorNode[struct{}]{
			err: ragy.ErrUnavailable,
		},
		FallbackPrimaryErr: ragy.ErrUnavailable,
		FallbackSecondaryFail: errorNode[struct{}]{
			err: ragy.ErrUnavailable,
		},
		FallbackSecondaryErr: ragy.ErrUnavailable,
		FallbackPartialPrimary: partialFailureNode[struct{}]{
			docs:   []retrieval.Document[struct{}]{{ID: "partial", Score: 1}},
			errors: []error{ragy.ErrUnavailable},
		},
		FallbackPartialWantID: "partial",
		RescuePrimaryFail:     errorNode[struct{}]{err: ragy.ErrUnavailable},
		RescuePrimaryOK: stubNode[struct{}]{
			docs: []retrieval.Document[struct{}]{{ID: "ok", Score: 1}},
		},
		RescueSecondary: stubNode[struct{}]{
			docs: []retrieval.Document[struct{}]{{ID: "rescue", Score: 1}},
		},
		RescueWantID:    "rescue",
		RescuePrimaryID: "ok",
		RescuePartialPrimary: partialFailureNode[struct{}]{
			docs:   []retrieval.Document[struct{}]{{ID: "partial", Score: 1}},
			errors: []error{ragy.ErrUnavailable},
		},
		RescuePartialWantID: "partial",
		RescueEmptyPrimaryFail: errorNode[struct{}]{
			err: ragy.ErrUnavailable,
		},
		RescueEmptySecondary:      stubNode[struct{}]{},
		RescueSecondaryFail:       errorNode[struct{}]{err: ragy.ErrProtocol},
		RescueSecondaryWrappedErr: ragy.ErrProtocol,
		ConditionalChild: stubNode[struct{}]{
			docs: []retrieval.Document[struct{}]{{ID: "skip", Score: 1}},
		},
		ConditionalPredicate: func(retrieval.Query[struct{}]) bool {
			return false
		},
		ConditionalPredicateTrue: func(retrieval.Query[struct{}]) bool {
			return true
		},
		ConditionalTrueWantID: "skip",
		ConditionalNilPredicateChild: stubNode[struct{}]{
			docs: []retrieval.Document[struct{}]{{ID: "hit", Score: 1}},
		},
		ConditionalNilPredicateWantID: "hit",
		AggregatePartialNodes: []retrieval.Node[struct{}, struct{}]{
			errorNode[struct{}]{err: ragy.ErrUnavailable},
			stubNode[struct{}]{docs: []retrieval.Document[struct{}]{{ID: "hit", Score: 1}}},
		},
		AggregatePartialWantID: "hit",
		AggregateMergeFallbackNodes: []retrieval.Node[struct{}, struct{}]{
			stubNode[struct{}]{docs: []retrieval.Document[struct{}]{
				{ID: "a", Content: "key", Score: 0.9},
			}},
			stubNode[struct{}]{docs: []retrieval.Document[struct{}]{
				{ID: "b", Content: "key", Score: 0.5},
			}},
		},
		AggregateMergeFallbackMerger:  stubFailingMerger{},
		AggregateMergeFallbackWantLen: 1,
	})
}

func TestPipelineRetrieveRejectsInvalidOptions(t *testing.T) {
	t.Parallel()

	pipeline, err := retrieval.NewPipelineBuilder[stubIntent, struct{}]().
		WithRoot(stubNode[stubIntent]{
			docs: []retrieval.Document[struct{}]{{ID: "a", Content: "hit", Score: 1}},
		}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	out, err := pipeline.Retrieve(context.Background(), retrieval.Query[stubIntent]{
		Text:    "q",
		Options: retrieval.RetrieveOptions{FetchLimit: 1, TopK: 3},
	})
	contracttest.RequireErrorResultSet(t, out, err)
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Retrieve() error = %v, want invalid argument", err)
	}
}

func TestPipelineConditionalUsesQueryIntent(t *testing.T) {
	t.Parallel()

	child := stubNode[pipelineIntent]{
		docs: []retrieval.Document[struct{}]{{ID: "hit", Content: "ok", Score: 1}},
	}
	pipeline, err := retrieval.NewPipelineBuilder[pipelineIntent, struct{}]().
		WithRoot(retrieval.ConditionalNode[pipelineIntent, struct{}]{
			Predicate: func(query retrieval.Query[pipelineIntent]) bool {
				return query.Intent.Mode == "run"
			},
			Child: child,
		}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	runRS, err := pipeline.Retrieve(context.Background(), retrieval.Query[pipelineIntent]{
		Text:    "q",
		Intent:  pipelineIntent{Mode: "run"},
		Options: contracttest.DefaultRetrieveOptions(),
	})
	if err != nil {
		t.Fatalf("Retrieve(run): %v", err)
	}
	if runRS.Len() != 1 || runRS.Documents()[0].ID != "hit" {
		t.Fatalf("Retrieve(run) = %#v, want hit", runRS.Documents())
	}

	skipRS, err := pipeline.Retrieve(context.Background(), retrieval.Query[pipelineIntent]{
		Text:    "q",
		Intent:  pipelineIntent{Mode: "skip"},
		Options: contracttest.DefaultRetrieveOptions(),
	})
	if err != nil {
		t.Fatalf("Retrieve(skip): %v", err)
	}
	if !skipRS.IsEmpty() {
		t.Fatalf("Retrieve(skip) = %#v, want empty", skipRS.Documents())
	}
}
