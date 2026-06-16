package otel

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/documents"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/graph"
	"github.com/skosovsky/ragy/internal/contracttest"
	"github.com/skosovsky/ragy/multimodal"
	"github.com/skosovsky/ragy/ranking"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/tensor"

	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
)

type captureDenseIndex struct{ valid bool }

func (i *captureDenseIndex) Upsert(ctx context.Context, _ []dense.Record[contracttest.StructMeta]) error {
	i.valid = trace.SpanFromContext(ctx).SpanContext().IsValid()
	return nil
}

func (i *captureDenseIndex) Schema() filter.Schema { return filter.EmptySchema() }

type captureDenseEmbedder struct{ valid bool }

func (e *captureDenseEmbedder) Embed(ctx context.Context, _ []string) ([][]float32, error) {
	e.valid = trace.SpanFromContext(ctx).SpanContext().IsValid()
	return nil, nil
}

type captureTensorEmbedder struct{ valid bool }

func (e *captureTensorEmbedder) Embed(ctx context.Context, _ []string) ([]tensor.Tensor, error) {
	e.valid = trace.SpanFromContext(ctx).SpanContext().IsValid()
	return nil, nil
}

type captureGraphStore struct{ valid bool }

func (s *captureGraphStore) Traverse(
	ctx context.Context,
	_ graph.TraversalRequest,
) (graph.Snapshot[contracttest.StructMeta], error) {
	s.valid = trace.SpanFromContext(ctx).SpanContext().IsValid()
	return graph.Snapshot[contracttest.StructMeta]{}, nil
}

func (s *captureGraphStore) Upsert(ctx context.Context, _ graph.Snapshot[contracttest.StructMeta]) error {
	s.valid = trace.SpanFromContext(ctx).SpanContext().IsValid()
	return nil
}

func (s *captureGraphStore) Schema() graph.Schema { return graph.EmptySchema() }

type captureTensorIndex struct{ valid bool }

func (i *captureTensorIndex) Upsert(ctx context.Context, _ []tensor.Record[contracttest.StructMeta]) error {
	i.valid = trace.SpanFromContext(ctx).SpanContext().IsValid()
	return nil
}

func (i *captureTensorIndex) Schema() filter.Schema { return filter.EmptySchema() }

type captureMultimodalEmbedder struct{ valid bool }

func (e *captureMultimodalEmbedder) Embed(ctx context.Context, _ []multimodal.Input) ([][]float32, error) {
	e.valid = trace.SpanFromContext(ctx).SpanContext().IsValid()
	return nil, nil
}

type captureDocumentStore struct{ valid bool }

func (s *captureDocumentStore) FindByIDs(
	ctx context.Context,
	_ []string,
) ([]retrieval.Document[contracttest.StructMeta], error) {
	s.valid = trace.SpanFromContext(ctx).SpanContext().IsValid()
	return nil, nil
}

func (s *captureDocumentStore) DeleteByIDs(ctx context.Context, _ []string) (documents.DeleteResult, error) {
	s.valid = trace.SpanFromContext(ctx).SpanContext().IsValid()
	return documents.DeleteResult{}, nil
}

func (s *captureDocumentStore) DeleteByFilter(ctx context.Context, _ filter.Condition) (documents.DeleteResult, error) {
	s.valid = trace.SpanFromContext(ctx).SpanContext().IsValid()
	return documents.DeleteResult{}, nil
}

func (s *captureDocumentStore) Schema() filter.Schema { return filter.EmptySchema() }

type captureQueryReranker struct{ valid bool }

func (r *captureQueryReranker) Rerank(
	ctx context.Context,
	_ string,
	_ retrieval.ResultSet[contracttest.StructMeta],
) (retrieval.ResultSet[contracttest.StructMeta], error) {
	r.valid = trace.SpanFromContext(ctx).SpanContext().IsValid()
	return retrieval.NewResultSet[contracttest.StructMeta](
		nil,
		retrieval.DocumentIDResolver[contracttest.StructMeta]{},
	), nil
}

type captureMerger struct{ valid bool }

func (m *captureMerger) Merge(
	ctx context.Context,
	_ ...retrieval.ResultSet[contracttest.StructMeta],
) (retrieval.ResultSet[contracttest.StructMeta], error) {
	m.valid = trace.SpanFromContext(ctx).SpanContext().IsValid()
	return retrieval.NewResultSet[contracttest.StructMeta](
		nil,
		retrieval.DocumentIDResolver[contracttest.StructMeta]{},
	), nil
}

func TestWrapDenseIndexPassesDerivedContext(t *testing.T) {
	runSpanTest(t, "ragy.dense.upsert", func(ctx context.Context, tracer trace.Tracer) (bool, error) {
		next := &captureDenseIndex{}
		wrapped, err := WrapDenseIndex(next, tracer)
		if err != nil {
			return false, err
		}
		err = wrapped.Upsert(ctx, []dense.Record[contracttest.StructMeta]{{ID: "doc-1", Vector: []float32{1}}})
		return next.valid, err
	})
}

func TestWrapDenseEmbedderPassesDerivedContext(t *testing.T) {
	runSpanTest(t, "ragy.dense.embed", func(ctx context.Context, tracer trace.Tracer) (bool, error) {
		next := &captureDenseEmbedder{}
		wrapped, err := WrapDenseEmbedder(next, tracer)
		if err != nil {
			return false, err
		}
		_, err = wrapped.Embed(ctx, []string{"hello"})
		return next.valid, err
	})
}

func TestWrapTensorEmbedderPassesDerivedContext(t *testing.T) {
	runSpanTest(t, "ragy.tensor.embed", func(ctx context.Context, tracer trace.Tracer) (bool, error) {
		next := &captureTensorEmbedder{}
		wrapped, err := WrapTensorEmbedder(next, tracer)
		if err != nil {
			return false, err
		}
		_, err = wrapped.Embed(ctx, []string{"hello"})
		return next.valid, err
	})
}

func TestWrapTensorIndexPassesDerivedContext(t *testing.T) {
	runSpanTest(t, "ragy.tensor.upsert", func(ctx context.Context, tracer trace.Tracer) (bool, error) {
		next := &captureTensorIndex{}
		wrapped, err := WrapTensorIndex(next, tracer)
		if err != nil {
			return false, err
		}
		err = wrapped.Upsert(ctx, []tensor.Record[contracttest.StructMeta]{{ID: "doc-1", Tensor: tensor.Tensor{{1}}}})
		return next.valid, err
	})
}

func TestWrapMultimodalEmbedderPassesDerivedContext(t *testing.T) {
	runSpanTest(t, "ragy.multimodal.embed", func(ctx context.Context, tracer trace.Tracer) (bool, error) {
		next := &captureMultimodalEmbedder{}
		wrapped, err := WrapMultimodalEmbedder(next, tracer)
		if err != nil {
			return false, err
		}
		_, err = wrapped.Embed(ctx, []multimodal.Input{{
			Parts: []multimodal.Part{{Kind: multimodal.PartText, Text: "hello"}},
		}})
		return next.valid, err
	})
}

func TestWrapDocumentStoreFindByIDsPassesDerivedContext(t *testing.T) {
	runSpanTest(t, "ragy.documents.find", func(ctx context.Context, tracer trace.Tracer) (bool, error) {
		next := &captureDocumentStore{}
		wrapped, err := WrapDocumentStore(next, tracer)
		if err != nil {
			return false, err
		}
		_, err = wrapped.FindByIDs(ctx, []string{"doc-1"})
		return next.valid, err
	})
}

func TestWrapDocumentStoreDeleteByIDsPassesDerivedContext(t *testing.T) {
	runSpanTest(t, "ragy.documents.delete_ids", func(ctx context.Context, tracer trace.Tracer) (bool, error) {
		next := &captureDocumentStore{}
		wrapped, err := WrapDocumentStore(next, tracer)
		if err != nil {
			return false, err
		}
		_, err = wrapped.DeleteByIDs(ctx, []string{"doc-1"})
		return next.valid, err
	})
}

func TestWrapDocumentStoreDeleteByFilterPassesDerivedContext(t *testing.T) {
	runSpanTest(t, "ragy.documents.delete_filter", func(ctx context.Context, tracer trace.Tracer) (bool, error) {
		next := &captureDocumentStore{}
		wrapped, err := WrapDocumentStore(next, tracer)
		if err != nil {
			return false, err
		}
		_, err = wrapped.DeleteByFilter(ctx, filter.Condition{})
		return next.valid, err
	})
}

func TestWrapQueryRerankerPropagatesErrorResultSet(t *testing.T) {
	t.Parallel()

	next := &errorQueryReranker{}
	provider := sdktrace.NewTracerProvider()
	wrapped, err := WrapQueryReranker(next, provider.Tracer("test"))
	if err != nil {
		t.Fatalf("WrapQueryReranker(): %v", err)
	}

	rs := retrieval.NewResultSet(
		[]retrieval.Document[contracttest.StructMeta]{{ID: "doc-1"}},
		retrieval.DocumentIDResolver[contracttest.StructMeta]{},
	)
	out, err := wrapped.Rerank(context.Background(), "q", rs)
	contracttest.RequireErrorResultSet(t, out, err)
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("Rerank() error = %v, want unavailable", err)
	}
}

func TestWrapMergerPropagatesErrorResultSet(t *testing.T) {
	t.Parallel()

	next := &errorMerger{}
	provider := sdktrace.NewTracerProvider()
	wrapped, err := WrapMerger(next, provider.Tracer("test"))
	if err != nil {
		t.Fatalf("WrapMerger(): %v", err)
	}

	input := retrieval.NewResultSet(
		[]retrieval.Document[contracttest.StructMeta]{{ID: "doc-1"}},
		retrieval.DocumentIDResolver[contracttest.StructMeta]{},
	)
	out, err := wrapped.Merge(context.Background(), input)
	contracttest.RequireErrorResultSet(t, out, err)
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("Merge() error = %v, want unavailable", err)
	}
}

type errorQueryReranker struct{}

func (errorQueryReranker) Rerank(
	_ context.Context,
	_ string,
	_ retrieval.ResultSet[contracttest.StructMeta],
) (retrieval.ResultSet[contracttest.StructMeta], error) {
	return retrieval.NewResultSet[contracttest.StructMeta](
		nil,
		retrieval.DocumentIDResolver[contracttest.StructMeta]{},
	), ragy.ErrUnavailable
}

type errorMerger struct{}

func (errorMerger) Merge(
	_ context.Context,
	_ ...retrieval.ResultSet[contracttest.StructMeta],
) (retrieval.ResultSet[contracttest.StructMeta], error) {
	return retrieval.NewResultSet[contracttest.StructMeta](
		nil,
		retrieval.DocumentIDResolver[contracttest.StructMeta]{},
	), ragy.ErrUnavailable
}

func TestWrapQueryRerankerPassesDerivedContext(t *testing.T) {
	runSpanTest(t, "ragy.ranking.rerank", func(ctx context.Context, tracer trace.Tracer) (bool, error) {
		next := &captureQueryReranker{}
		wrapped, err := WrapQueryReranker(next, tracer)
		if err != nil {
			return false, err
		}
		_, err = wrapped.Rerank(
			ctx,
			"hello",
			retrieval.NewResultSet(
				[]retrieval.Document[contracttest.StructMeta]{{ID: "doc-1"}},
				retrieval.DocumentIDResolver[contracttest.StructMeta]{},
			),
		)
		return next.valid, err
	})
}

func TestWrapMergerPassesDerivedContext(t *testing.T) {
	runSpanTest(t, "ragy.ranking.merge", func(ctx context.Context, tracer trace.Tracer) (bool, error) {
		next := &captureMerger{}
		wrapped, err := WrapMerger(next, tracer)
		if err != nil {
			return false, err
		}
		input := retrieval.NewResultSet(
			[]retrieval.Document[contracttest.StructMeta]{{ID: "doc-1"}},
			retrieval.DocumentIDResolver[contracttest.StructMeta]{},
		)
		_, err = wrapped.Merge(ctx, input)
		return next.valid, err
	})
}

func TestWrapGraphStorePassesDerivedContextAndDelegatesSchema(t *testing.T) {
	runSpanTest(t, "ragy.graph.traverse", func(ctx context.Context, tracer trace.Tracer) (bool, error) {
		next := &captureGraphStore{}
		wrapped, err := WrapGraphStore(next, tracer)
		if err != nil {
			return false, err
		}
		if schemaErr := wrapped.Schema().Validate(); schemaErr != nil {
			return false, schemaErr
		}
		_, err = wrapped.Traverse(ctx, graph.TraversalRequest{
			Seeds:     []string{"node-1"},
			Direction: graph.DirectionOutbound,
			Depth:     1,
		})
		return next.valid, err
	})
}

func TestWrapGraphStoreUpsertPassesDerivedContext(t *testing.T) {
	runSpanTest(t, "ragy.graph.upsert", func(ctx context.Context, tracer trace.Tracer) (bool, error) {
		next := &captureGraphStore{}
		wrapped, err := WrapGraphStore(next, tracer)
		if err != nil {
			return false, err
		}
		err = wrapped.Upsert(ctx, graph.Snapshot[contracttest.StructMeta]{})
		return next.valid, err
	})
}

func runSpanTest(
	t *testing.T,
	wantSpan string,
	run func(context.Context, trace.Tracer) (bool, error),
) {
	t.Helper()

	recorder := tracetest.NewSpanRecorder()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	t.Cleanup(func() { _ = provider.Shutdown(context.Background()) })

	valid, err := run(context.Background(), provider.Tracer("test"))
	if err != nil {
		t.Fatalf("run(): %v", err)
	}
	if !valid {
		t.Fatal("downstream context does not carry a valid span")
	}

	spans := recorder.Ended()
	if len(spans) != 1 {
		t.Fatalf("len(spans) = %d, want 1", len(spans))
	}
	if spans[0].Name() != wantSpan {
		t.Fatalf("span name = %q, want %q", spans[0].Name(), wantSpan)
	}
}

type captureBackend struct{ valid bool }

func (b *captureBackend) Retrieve(
	ctx context.Context,
	_ string,
	_ retrieval.RetrieveOptions,
) (retrieval.ResultSet[contracttest.StructMeta], error) {
	b.valid = trace.SpanFromContext(ctx).SpanContext().IsValid()
	return retrieval.NewResultSet[contracttest.StructMeta](
		nil,
		retrieval.DocumentIDResolver[contracttest.StructMeta]{},
	), nil
}

func TestWrapBackendPropagatesErrorResultSet(t *testing.T) {
	t.Parallel()

	next := &errorStructBackend{}
	provider := sdktrace.NewTracerProvider()
	wrapped, err := WrapBackend(next, provider.Tracer("test"))
	if err != nil {
		t.Fatalf("WrapBackend(): %v", err)
	}

	out, err := wrapped.Retrieve(context.Background(), "q", retrieval.RetrieveOptions{TopK: 1})
	contracttest.RequireErrorResultSet(t, out, err)
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("Retrieve() error = %v, want unavailable", err)
	}
}

func TestWrapBackendPreservesPartialResult(t *testing.T) {
	t.Parallel()

	backend := partialFailureBackend{}
	provider := sdktrace.NewTracerProvider()
	wrapped, err := WrapBackend(backend, provider.Tracer("test"))
	if err != nil {
		t.Fatalf("WrapBackend(): %v", err)
	}

	out, err := wrapped.Retrieve(context.Background(), "q", retrieval.RetrieveOptions{TopK: 1})
	if err == nil {
		t.Fatal("Retrieve() error = nil, want partial failure")
	}
	var partial *retrieval.PartialFailureError[struct{}]
	if !errors.As(err, &partial) {
		t.Fatalf("Retrieve() error = %v, want PartialFailureError", err)
	}
	if out.Len() != 1 || out.Documents()[0].ID != "a" {
		t.Fatalf("Documents() = %#v, want preserved partial backend docs", out.Documents())
	}
}

type errorStructBackend struct{}

func (errorStructBackend) Retrieve(
	_ context.Context,
	_ string,
	_ retrieval.RetrieveOptions,
) (retrieval.ResultSet[contracttest.StructMeta], error) {
	return retrieval.NewResultSet[contracttest.StructMeta](
		nil,
		retrieval.DocumentIDResolver[contracttest.StructMeta]{},
	), ragy.ErrUnavailable
}

func TestWrapBackendPassesDerivedContext(t *testing.T) {
	runSpanTest(t, "ragy.retrieval.backend", func(ctx context.Context, tracer trace.Tracer) (bool, error) {
		next := &captureBackend{}
		wrapped, err := WrapBackend(next, tracer)
		if err != nil {
			return false, err
		}
		_, err = wrapped.Retrieve(ctx, "hello", retrieval.RetrieveOptions{TopK: 10})
		return next.valid, err
	})
}

type errorBackend struct{}

func (errorBackend) Retrieve(
	_ context.Context,
	_ string,
	_ retrieval.RetrieveOptions,
) (retrieval.ResultSet[struct{}], error) {
	return retrieval.NewResultSet[struct{}](nil, retrieval.DocumentIDResolver[struct{}]{}), ragy.ErrUnavailable
}

type partialFailureBackend struct{}

func (partialFailureBackend) Retrieve(
	_ context.Context,
	_ string,
	_ retrieval.RetrieveOptions,
) (retrieval.ResultSet[struct{}], error) {
	rs := retrieval.NewResultSet([]retrieval.Document[struct{}]{
		{ID: "a", Content: "hit", Score: 1},
	}, retrieval.DocumentIDResolver[struct{}]{})
	return rs, &retrieval.PartialFailureError[struct{}]{
		Errors: []error{ragy.ErrUnavailable},
		Result: rs,
	}
}

type contentMergeKeyResolver[TMeta any] = contracttest.ContentMergeResolver[TMeta]

func TestWrapPipelinePreservesPartialFailureResult(t *testing.T) {
	t.Parallel()

	next, err := retrieval.NewPipelineBuilder[struct{}, struct{}]().
		WithRoot(retrieval.AggregateNode[struct{}, struct{}]{
			Nodes: []retrieval.Node[struct{}, struct{}]{
				retrieval.RetrieverNode[struct{}, struct{}]{Backend: errorBackend{}},
				retrieval.RetrieverNode[struct{}, struct{}]{
					Backend: partialFailureBackend{},
				},
			},
		}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	provider := sdktrace.NewTracerProvider()
	wrapped, err := WrapPipeline(next, provider.Tracer("test"))
	if err != nil {
		t.Fatalf("WrapPipeline(): %v", err)
	}

	rs, err := wrapped.Retrieve(context.Background(), retrieval.Query[struct{}]{
		Text:    "q",
		Options: retrieval.RetrieveOptions{TopK: 10},
	})
	if err == nil {
		t.Fatal("Retrieve() error = nil, want partial failure")
	}
	var partial *retrieval.PartialFailureError[struct{}]
	if !errors.As(err, &partial) {
		t.Fatalf("Retrieve() error = %v, want PartialFailureError", err)
	}
	if rs.Len() != 1 || rs.Documents()[0].ID != "a" {
		t.Fatalf("Documents() = %#v, want preserved partial result", rs.Documents())
	}
}

func TestWrapPipelinePartialFailureResolverParity(t *testing.T) {
	t.Parallel()

	resolver := contentMergeKeyResolver[struct{}]{}
	next, err := retrieval.NewPipelineBuilder[struct{}, struct{}]().
		WithResolver(resolver).
		WithRoot(retrieval.AggregateNode[struct{}, struct{}]{
			Nodes: []retrieval.Node[struct{}, struct{}]{
				retrieval.RetrieverNode[struct{}, struct{}]{Backend: errorBackend{}},
				retrieval.RetrieverNode[struct{}, struct{}]{
					Backend: partialFailureBackend{},
				},
			},
		}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	provider := sdktrace.NewTracerProvider()
	wrapped, err := WrapPipeline(next, provider.Tracer("test"))
	if err != nil {
		t.Fatalf("WrapPipeline(): %v", err)
	}

	rs, err := wrapped.Retrieve(context.Background(), retrieval.Query[struct{}]{
		Text:    "q",
		Options: retrieval.RetrieveOptions{TopK: 10},
	})
	if err == nil {
		t.Fatal("Retrieve() error = nil, want partial failure")
	}
	partial, ok := retrieval.AsPartialFailure[struct{}](err)
	if !ok {
		t.Fatalf("Retrieve() error = %v, want partial failure", err)
	}
	other, mergeErr := partial.Result.Merge(retrieval.NewResultSet([]retrieval.Document[struct{}]{
		{ID: "b", Content: "hit", Score: 0.5},
	}, resolver))
	if mergeErr != nil {
		t.Fatalf("partial.Result.Merge() error = %v", mergeErr)
	}
	if other.Len() != 1 {
		t.Fatalf("merged Len() = %d, want 1 by content merge key", other.Len())
	}
	if rs.Len() != partial.Result.Len() {
		t.Fatalf("rs.Len() = %d, partial.Result.Len() = %d, want equal", rs.Len(), partial.Result.Len())
	}
}

func TestWrapPipelinePropagatesErrorResultSet(t *testing.T) {
	t.Parallel()

	next, err := retrieval.NewPipelineBuilder[struct{}, struct{}]().
		WithRoot(retrieval.RetrieverNode[struct{}, struct{}]{Backend: errorBackend{}}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	provider := sdktrace.NewTracerProvider()
	wrapped, err := WrapPipeline(next, provider.Tracer("test"))
	if err != nil {
		t.Fatalf("WrapPipeline(): %v", err)
	}

	out, err := wrapped.Retrieve(context.Background(), retrieval.Query[struct{}]{
		Text:    "q",
		Options: retrieval.RetrieveOptions{TopK: 10},
	})
	contracttest.RequireErrorResultSet(t, out, err)
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("Retrieve() error = %v, want unavailable", err)
	}
}

func TestWrapPipelinePassesDerivedContext(t *testing.T) {
	runSpanTest(t, "ragy.retrieval.pipeline", func(ctx context.Context, tracer trace.Tracer) (bool, error) {
		backend := &captureBackend{}
		next, err := retrieval.NewPipelineBuilder[struct{}, contracttest.StructMeta]().
			WithRoot(retrieval.RetrieverNode[struct{}, contracttest.StructMeta]{Backend: backend}).
			Build()
		if err != nil {
			return false, err
		}
		wrapped, err := WrapPipeline(next, tracer)
		if err != nil {
			return false, err
		}
		_, err = wrapped.Retrieve(
			ctx,
			retrieval.Query[struct{}]{Text: "hello", Options: retrieval.RetrieveOptions{TopK: 10}},
		)
		return backend.valid, err
	})
}

var (
	_ dense.Index[contracttest.StructMeta]           = (*captureDenseIndex)(nil)
	_ retrieval.Backend[contracttest.StructMeta]     = (*captureBackend)(nil)
	_ ranking.QueryReranker[contracttest.StructMeta] = (*captureQueryReranker)(nil)
	_ ranking.Merger[contracttest.StructMeta]        = (*captureMerger)(nil)
	_ documents.Store[contracttest.StructMeta]       = (*captureDocumentStore)(nil)
)
