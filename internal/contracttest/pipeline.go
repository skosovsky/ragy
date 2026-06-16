package contracttest

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/retrieval"
)

// PipelineFactory builds a pipeline for contract tests.
type PipelineFactory[TIntent, TMeta any] func(t *testing.T) *retrieval.Pipeline[TIntent, TMeta]

// PipelineNodeSemanticsConfig supplies stub nodes for RunPipelineNodeSemanticsSuite.
type PipelineNodeSemanticsConfig struct {
	FallbackPrimary, FallbackSecondary retrieval.Node[struct{}, struct{}]
	FallbackWantID                     string

	FallbackPrimaryHasResults    retrieval.Node[struct{}, struct{}]
	FallbackSkipsSecondary       retrieval.Node[struct{}, struct{}]
	FallbackSkipsSecondaryWantID string

	FallbackPrimaryFail retrieval.Node[struct{}, struct{}]
	FallbackPrimaryErr  error

	FallbackSecondaryFail retrieval.Node[struct{}, struct{}]
	FallbackSecondaryErr  error

	FallbackPartialPrimary retrieval.Node[struct{}, struct{}]
	FallbackPartialWantID  string

	RescuePrimaryFail retrieval.Node[struct{}, struct{}]
	RescuePrimaryOK   retrieval.Node[struct{}, struct{}]
	RescueSecondary   retrieval.Node[struct{}, struct{}]
	RescueWantID      string
	RescuePrimaryID   string

	RescuePartialPrimary retrieval.Node[struct{}, struct{}]
	RescuePartialWantID  string

	RescueEmptyPrimaryFail retrieval.Node[struct{}, struct{}]
	RescueEmptySecondary   retrieval.Node[struct{}, struct{}]

	RescueSecondaryFail       retrieval.Node[struct{}, struct{}]
	RescueSecondaryWrappedErr error

	ConditionalChild         retrieval.Node[struct{}, struct{}]
	ConditionalPredicate     func(retrieval.Query[struct{}]) bool
	ConditionalPredicateTrue func(retrieval.Query[struct{}]) bool
	ConditionalTrueWantID    string

	ConditionalNilPredicateChild  retrieval.Node[struct{}, struct{}]
	ConditionalNilPredicateWantID string

	AggregatePartialNodes  []retrieval.Node[struct{}, struct{}]
	AggregatePartialWantID string

	AggregateMergeFallbackNodes   []retrieval.Node[struct{}, struct{}]
	AggregateMergeFallbackMerger  retrieval.ResultMerger[struct{}]
	AggregateMergeFallbackWantLen int
}

func pipelineSemanticsQuery[TIntent any]() retrieval.Query[TIntent] {
	return retrieval.Query[TIntent]{
		Text:    "q",
		Options: DefaultRetrieveOptions(),
	}
}
func RunPipelineRetrieveOptionsInvalidSuite[TIntent, TMeta any](
	t *testing.T,
	factory PipelineFactory[TIntent, TMeta],
) {
	t.Helper()

	t.Run("pipeline retrieve rejects fetch limit less than top k", func(t *testing.T) {
		t.Parallel()

		pipeline := factory(t)
		out, err := pipeline.Retrieve(context.Background(), retrieval.Query[TIntent]{
			Text:    "q",
			Options: retrieval.RetrieveOptions{FetchLimit: 1, TopK: retrieveOptionsInvalidTopK},
		})
		RequireErrorResultSet(t, out, err)
		if !errors.Is(err, ragy.ErrInvalidArgument) {
			t.Fatalf("Retrieve() error = %v, want invalid argument", err)
		}
	})

	t.Run("pipeline retrieve rejects negative top k", func(t *testing.T) {
		t.Parallel()

		pipeline := factory(t)
		out, err := pipeline.Retrieve(context.Background(), retrieval.Query[TIntent]{
			Text:    "q",
			Options: retrieval.RetrieveOptions{TopK: -1},
		})
		RequireErrorResultSet(t, out, err)
		if !errors.Is(err, ragy.ErrInvalidArgument) {
			t.Fatalf("Retrieve() error = %v, want invalid argument", err)
		}
	})

	t.Run("pipeline retrieve rejects negative fetch limit", func(t *testing.T) {
		t.Parallel()

		pipeline := factory(t)
		out, err := pipeline.Retrieve(context.Background(), retrieval.Query[TIntent]{
			Text: "q",
			Options: retrieval.RetrieveOptions{
				FetchLimit: -1,
				TopK:       retrieveOptionsInvalidTopK,
			},
		})
		RequireErrorResultSet(t, out, err)
		if !errors.Is(err, ragy.ErrInvalidArgument) {
			t.Fatalf("Retrieve() error = %v, want invalid argument", err)
		}
	})

	t.Run("pipeline retrieve rejects min similarity out of range", func(t *testing.T) {
		t.Parallel()

		pipeline := factory(t)
		out, err := pipeline.Retrieve(context.Background(), retrieval.Query[TIntent]{
			Text: "q",
			Options: retrieval.RetrieveOptions{
				TopK:          retrieveOptionsInvalidTopK,
				MinSimilarity: retrieveOptionsInvalidMinSimilarity,
			},
		})
		RequireErrorResultSet(t, out, err)
		if !errors.Is(err, ragy.ErrInvalidArgument) {
			t.Fatalf("Retrieve() error = %v, want invalid argument", err)
		}
	})

	t.Run("pipeline retrieve rejects zero top k and zero fetch limit", func(t *testing.T) {
		t.Parallel()

		pipeline := factory(t)
		out, err := pipeline.Retrieve(context.Background(), retrieval.Query[TIntent]{Text: "q"})
		RequireErrorResultSet(t, out, err)
		if !errors.Is(err, ragy.ErrInvalidArgument) {
			t.Fatalf("Retrieve() error = %v, want invalid argument", err)
		}
	})
}

// RunPipelineNodeSemanticsSuite checks cross-cutting pipeline node invariants.
func RunPipelineNodeSemanticsSuite(t *testing.T, cfg PipelineNodeSemanticsConfig) {
	t.Helper()

	runPipelineFallbackSemantics(t, cfg)
	runPipelineRescueSemantics(t, cfg)
	runPipelineConditionalSemantics(t, cfg)
	runPipelineAggregatePartialSemantics(t, cfg)
	runPipelineAggregateMergeFallbackSemantics(t, cfg)
}

func runPipelineFallbackSemantics(t *testing.T, cfg PipelineNodeSemanticsConfig) {
	t.Helper()

	t.Run("fallback uses secondary when primary empty", func(t *testing.T) {
		t.Parallel()
		testPipelineFallbackUsesSecondary(t, cfg)
	})
	t.Run("fallback propagates primary error without secondary", func(t *testing.T) {
		t.Parallel()
		testPipelineFallbackPropagatesPrimaryError(t, cfg)
	})
	t.Run("fallback preserves partial failure without secondary", func(t *testing.T) {
		t.Parallel()
		testPipelineFallbackPreservesPartialFailure(t, cfg)
	})
	t.Run("fallback propagates secondary error", func(t *testing.T) {
		t.Parallel()
		testPipelineFallbackPropagatesSecondaryError(t, cfg)
	})
	t.Run("fallback skips secondary when primary has results", func(t *testing.T) {
		t.Parallel()
		testPipelineFallbackSkipsSecondaryWhenPrimaryHasResults(t, cfg)
	})
}

func testPipelineFallbackUsesSecondary(t *testing.T, cfg PipelineNodeSemanticsConfig) {
	t.Helper()

	pipeline, err := retrieval.NewPipelineBuilder[struct{}, struct{}]().
		WithRoot(retrieval.FallbackNode[struct{}, struct{}]{
			Primary:   cfg.FallbackPrimary,
			Secondary: cfg.FallbackSecondary,
		}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Retrieve(context.Background(), pipelineSemanticsQuery[struct{}]())
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.IsEmpty() || rs.Documents()[0].ID != cfg.FallbackWantID {
		t.Fatalf("Documents() = %#v, want id %q", rs.Documents(), cfg.FallbackWantID)
	}
}

func testPipelineFallbackSkipsSecondaryWhenPrimaryHasResults(t *testing.T, cfg PipelineNodeSemanticsConfig) {
	t.Helper()

	if cfg.FallbackPrimaryHasResults == nil {
		t.Skip("no FallbackPrimaryHasResults configured")
	}

	pipeline, err := retrieval.NewPipelineBuilder[struct{}, struct{}]().
		WithRoot(retrieval.FallbackNode[struct{}, struct{}]{
			Primary:   cfg.FallbackPrimaryHasResults,
			Secondary: cfg.FallbackSkipsSecondary,
		}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Retrieve(context.Background(), pipelineSemanticsQuery[struct{}]())
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	wantID := cfg.FallbackSkipsSecondaryWantID
	if rs.IsEmpty() || rs.Documents()[0].ID != wantID {
		t.Fatalf("Documents() = %#v, want id %q without secondary", rs.Documents(), wantID)
	}
}

func testPipelineFallbackPropagatesPrimaryError(t *testing.T, cfg PipelineNodeSemanticsConfig) {
	t.Helper()

	if cfg.FallbackPrimaryFail == nil {
		t.Skip("no FallbackPrimaryFail configured")
	}

	pipeline, err := retrieval.NewPipelineBuilder[struct{}, struct{}]().
		WithRoot(retrieval.FallbackNode[struct{}, struct{}]{
			Primary:   cfg.FallbackPrimaryFail,
			Secondary: cfg.FallbackSecondary,
		}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	out, err := pipeline.Retrieve(context.Background(), pipelineSemanticsQuery[struct{}]())
	RequireErrorResultSet(t, out, err)
	if !errors.Is(err, cfg.FallbackPrimaryErr) {
		t.Fatalf("Retrieve() error = %v, want %v", err, cfg.FallbackPrimaryErr)
	}
}

func testPipelineFallbackPreservesPartialFailure(t *testing.T, cfg PipelineNodeSemanticsConfig) {
	t.Helper()

	if cfg.FallbackPartialPrimary == nil {
		t.Skip("no FallbackPartialPrimary configured")
	}

	pipeline, err := retrieval.NewPipelineBuilder[struct{}, struct{}]().
		WithRoot(retrieval.FallbackNode[struct{}, struct{}]{
			Primary:   cfg.FallbackPartialPrimary,
			Secondary: cfg.FallbackSecondary,
		}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Retrieve(context.Background(), pipelineSemanticsQuery[struct{}]())
	var partial *retrieval.PartialFailureError[struct{}]
	if !errors.As(err, &partial) {
		t.Fatalf("Retrieve() error = %v, want PartialFailureError", err)
	}
	wantID := cfg.FallbackPartialWantID
	if rs.IsEmpty() || rs.Documents()[0].ID != wantID {
		t.Fatalf("Documents() = %#v, want partial id %q without secondary", rs.Documents(), wantID)
	}
}

func runPipelineRescueSemantics(t *testing.T, cfg PipelineNodeSemanticsConfig) {
	t.Helper()

	t.Run("rescue uses secondary when primary errors empty", func(t *testing.T) {
		t.Parallel()
		testPipelineRescueUsesSecondaryOnPrimaryError(t, cfg)
	})
	t.Run("rescue skips secondary when primary succeeds", func(t *testing.T) {
		t.Parallel()
		testPipelineRescueSkipsSecondaryOnPrimarySuccess(t, cfg)
	})
	t.Run("rescue does not run secondary on partial failure", func(t *testing.T) {
		t.Parallel()
		testPipelineRescueDoesNotRunSecondaryOnPartialFailure(t, cfg)
	})
	t.Run("rescue propagates primary error when secondary empty", func(t *testing.T) {
		t.Parallel()
		testPipelineRescuePropagatesPrimaryWhenSecondaryEmpty(t, cfg)
	})
	t.Run("rescue wraps primary and secondary errors", func(t *testing.T) {
		t.Parallel()
		testPipelineRescueWrapsPrimaryAndSecondaryErrors(t, cfg)
	})
}

func testPipelineRescueUsesSecondaryOnPrimaryError(t *testing.T, cfg PipelineNodeSemanticsConfig) {
	t.Helper()

	pipeline, err := retrieval.NewPipelineBuilder[struct{}, struct{}]().
		WithRoot(retrieval.RescueNode[struct{}, struct{}]{
			Primary:   cfg.RescuePrimaryFail,
			Secondary: cfg.RescueSecondary,
		}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Retrieve(context.Background(), pipelineSemanticsQuery[struct{}]())
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.IsEmpty() || rs.Documents()[0].ID != cfg.RescueWantID {
		t.Fatalf("Documents() = %#v, want id %q", rs.Documents(), cfg.RescueWantID)
	}
}

func testPipelineRescueSkipsSecondaryOnPrimarySuccess(t *testing.T, cfg PipelineNodeSemanticsConfig) {
	t.Helper()

	pipeline, err := retrieval.NewPipelineBuilder[struct{}, struct{}]().
		WithRoot(retrieval.RescueNode[struct{}, struct{}]{
			Primary:   cfg.RescuePrimaryOK,
			Secondary: cfg.RescueSecondary,
		}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Retrieve(context.Background(), pipelineSemanticsQuery[struct{}]())
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	wantID := cfg.RescuePrimaryID
	if wantID == "" {
		wantID = "ok"
	}
	if rs.IsEmpty() || rs.Documents()[0].ID != wantID {
		t.Fatalf("Documents() = %#v, want primary id %q", rs.Documents(), wantID)
	}
}

func testPipelineRescueDoesNotRunSecondaryOnPartialFailure(t *testing.T, cfg PipelineNodeSemanticsConfig) {
	t.Helper()

	if cfg.RescuePartialPrimary == nil {
		t.Skip("no RescuePartialPrimary configured")
	}

	pipeline, err := retrieval.NewPipelineBuilder[struct{}, struct{}]().
		WithRoot(retrieval.RescueNode[struct{}, struct{}]{
			Primary:   cfg.RescuePartialPrimary,
			Secondary: cfg.RescueSecondary,
		}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Retrieve(context.Background(), pipelineSemanticsQuery[struct{}]())
	var partial *retrieval.PartialFailureError[struct{}]
	if !errors.As(err, &partial) {
		t.Fatalf("Retrieve() error = %v, want PartialFailureError", err)
	}
	wantID := cfg.RescuePartialWantID
	if rs.IsEmpty() || rs.Documents()[0].ID != wantID {
		t.Fatalf("Documents() = %#v, want partial id %q without rescue", rs.Documents(), wantID)
	}
	if rs.Documents()[0].ID == cfg.RescueWantID {
		t.Fatalf("Documents() = %#v, secondary rescue must not run", rs.Documents())
	}
}

func testPipelineRescuePropagatesPrimaryWhenSecondaryEmpty(t *testing.T, cfg PipelineNodeSemanticsConfig) {
	t.Helper()

	if cfg.RescueEmptyPrimaryFail == nil {
		t.Skip("no RescueEmptyPrimaryFail configured")
	}

	pipeline, err := retrieval.NewPipelineBuilder[struct{}, struct{}]().
		WithRoot(retrieval.RescueNode[struct{}, struct{}]{
			Primary:   cfg.RescueEmptyPrimaryFail,
			Secondary: cfg.RescueEmptySecondary,
		}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Retrieve(context.Background(), pipelineSemanticsQuery[struct{}]())
	RequireErrorResultSet(t, rs, err)
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("Retrieve() error = %v, want unavailable on empty secondary rescue", err)
	}
}

func runPipelineConditionalSemantics(t *testing.T, cfg PipelineNodeSemanticsConfig) {
	t.Helper()

	t.Run("conditional skips child when predicate false", func(t *testing.T) {
		t.Parallel()

		pipeline, err := retrieval.NewPipelineBuilder[struct{}, struct{}]().
			WithRoot(retrieval.ConditionalNode[struct{}, struct{}]{
				Predicate: cfg.ConditionalPredicate,
				Child:     cfg.ConditionalChild,
			}).
			Build()
		if err != nil {
			t.Fatalf("Build(): %v", err)
		}

		rs, err := pipeline.Retrieve(context.Background(), pipelineSemanticsQuery[struct{}]())
		if err != nil {
			t.Fatalf("Retrieve(): %v", err)
		}
		if !rs.IsEmpty() {
			t.Fatalf("Documents() = %#v, want empty when predicate false", rs.Documents())
		}
	})

	t.Run("conditional runs child when predicate true", func(t *testing.T) {
		t.Parallel()

		pred := cfg.ConditionalPredicateTrue
		if pred == nil {
			pred = func(retrieval.Query[struct{}]) bool { return true }
		}
		wantID := cfg.ConditionalTrueWantID
		if wantID == "" {
			wantID = "skip"
		}

		pipeline, err := retrieval.NewPipelineBuilder[struct{}, struct{}]().
			WithRoot(retrieval.ConditionalNode[struct{}, struct{}]{
				Predicate: pred,
				Child:     cfg.ConditionalChild,
			}).
			Build()
		if err != nil {
			t.Fatalf("Build(): %v", err)
		}

		rs, err := pipeline.Retrieve(context.Background(), pipelineSemanticsQuery[struct{}]())
		if err != nil {
			t.Fatalf("Retrieve(): %v", err)
		}
		if rs.IsEmpty() || rs.Documents()[0].ID != wantID {
			t.Fatalf("Documents() = %#v, want id %q when predicate true", rs.Documents(), wantID)
		}
	})

	t.Run("conditional runs child when predicate nil", func(t *testing.T) {
		t.Parallel()
		testPipelineConditionalRunsChildWhenPredicateNil(t, cfg)
	})
}

func testPipelineConditionalRunsChildWhenPredicateNil(t *testing.T, cfg PipelineNodeSemanticsConfig) {
	t.Helper()

	if cfg.ConditionalNilPredicateChild == nil {
		t.Skip("no ConditionalNilPredicateChild configured")
	}
	wantID := cfg.ConditionalNilPredicateWantID
	if wantID == "" {
		wantID = "hit"
	}

	pipeline, err := retrieval.NewPipelineBuilder[struct{}, struct{}]().
		WithRoot(retrieval.ConditionalNode[struct{}, struct{}]{
			Predicate: nil,
			Child:     cfg.ConditionalNilPredicateChild,
		}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Retrieve(context.Background(), pipelineSemanticsQuery[struct{}]())
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.IsEmpty() || rs.Documents()[0].ID != wantID {
		t.Fatalf("Documents() = %#v, want id %q when predicate nil", rs.Documents(), wantID)
	}
}

func runPipelineAggregatePartialSemantics(t *testing.T, cfg PipelineNodeSemanticsConfig) {
	t.Helper()

	if len(cfg.AggregatePartialNodes) == 0 {
		return
	}

	t.Run("aggregate reports partial failure with sibling hit", func(t *testing.T) {
		t.Parallel()

		pipeline, err := retrieval.NewPipelineBuilder[struct{}, struct{}]().
			WithRoot(retrieval.AggregateNode[struct{}, struct{}]{
				Nodes: cfg.AggregatePartialNodes,
			}).
			Build()
		if err != nil {
			t.Fatalf("Build(): %v", err)
		}

		rs, err := pipeline.Retrieve(context.Background(), pipelineSemanticsQuery[struct{}]())
		var partial *retrieval.PartialFailureError[struct{}]
		if !errors.As(err, &partial) {
			t.Fatalf("Retrieve() error = %v, want PartialFailureError", err)
		}
		wantID := cfg.AggregatePartialWantID
		if rs.IsEmpty() || rs.Documents()[0].ID != wantID {
			t.Fatalf("Documents() = %#v, want id %q", rs.Documents(), wantID)
		}
	})
}

func runPipelineAggregateMergeFallbackSemantics(t *testing.T, cfg PipelineNodeSemanticsConfig) {
	t.Helper()

	if len(cfg.AggregateMergeFallbackNodes) == 0 {
		return
	}

	t.Run("aggregate uses score merge fallback when merger fails", func(t *testing.T) {
		t.Parallel()

		merger := cfg.AggregateMergeFallbackMerger
		if merger == nil {
			t.Fatal("AggregateMergeFallbackMerger must be configured")
		}

		pipeline, err := retrieval.NewPipelineBuilder[struct{}, struct{}]().
			WithRoot(retrieval.AggregateNode[struct{}, struct{}]{
				Nodes:  cfg.AggregateMergeFallbackNodes,
				Merger: merger,
			}).
			WithResolver(ContentMergeResolver[struct{}]{}).
			Build()
		if err != nil {
			t.Fatalf("Build(): %v", err)
		}

		rs, err := pipeline.Retrieve(context.Background(), pipelineSemanticsQuery[struct{}]())
		var partial *retrieval.PartialFailureError[struct{}]
		if !errors.As(err, &partial) {
			t.Fatalf("Retrieve() error = %v, want PartialFailureError", err)
		}
		if !errors.Is(err, ragy.ErrInvalidArgument) {
			t.Fatalf("Retrieve() error = %v, want invalid argument from failing merger", err)
		}
		wantLen := cfg.AggregateMergeFallbackWantLen
		if wantLen == 0 {
			wantLen = 1
		}
		if rs.Len() != wantLen {
			t.Fatalf("Len() = %d, want %d (degraded merge preserved docs)", rs.Len(), wantLen)
		}
		if rs.IsEmpty() {
			t.Fatalf("Documents() empty, want score-merge fallback result")
		}
	})
}

func testPipelineFallbackPropagatesSecondaryError(t *testing.T, cfg PipelineNodeSemanticsConfig) {
	t.Helper()

	if cfg.FallbackSecondaryFail == nil {
		t.Skip("no FallbackSecondaryFail configured")
	}

	pipeline, err := retrieval.NewPipelineBuilder[struct{}, struct{}]().
		WithRoot(retrieval.FallbackNode[struct{}, struct{}]{
			Primary:   cfg.FallbackPrimary,
			Secondary: cfg.FallbackSecondaryFail,
		}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	out, err := pipeline.Retrieve(context.Background(), pipelineSemanticsQuery[struct{}]())
	RequireErrorResultSet(t, out, err)
	if !errors.Is(err, cfg.FallbackSecondaryErr) {
		t.Fatalf("Retrieve() error = %v, want %v", err, cfg.FallbackSecondaryErr)
	}
}

func testPipelineRescueWrapsPrimaryAndSecondaryErrors(t *testing.T, cfg PipelineNodeSemanticsConfig) {
	t.Helper()

	if cfg.RescueSecondaryFail == nil {
		t.Skip("no RescueSecondaryFail configured")
	}

	pipeline, err := retrieval.NewPipelineBuilder[struct{}, struct{}]().
		WithRoot(retrieval.RescueNode[struct{}, struct{}]{
			Primary:   cfg.RescuePrimaryFail,
			Secondary: cfg.RescueSecondaryFail,
		}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	out, err := pipeline.Retrieve(context.Background(), pipelineSemanticsQuery[struct{}]())
	RequireErrorResultSet(t, out, err)
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("Retrieve() error = %v, want unavailable primary", err)
	}
	if !errors.Is(err, cfg.RescueSecondaryWrappedErr) {
		t.Fatalf("Retrieve() error = %v, want wrapped %v", err, cfg.RescueSecondaryWrappedErr)
	}
}
