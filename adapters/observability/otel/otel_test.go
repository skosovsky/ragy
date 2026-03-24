package otel

import (
	"context"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/documents"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/graph"
	"github.com/skosovsky/ragy/lexical"
	"github.com/skosovsky/ragy/multimodal"
	"github.com/skosovsky/ragy/ranking"
	"github.com/skosovsky/ragy/tensor"

	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
)

type captureDenseSearcher struct{ valid bool }

func (s *captureDenseSearcher) Search(ctx context.Context, _ dense.Request) ([]ragy.Document, error) {
	s.valid = trace.SpanFromContext(ctx).SpanContext().IsValid()
	return nil, nil
}

func (s *captureDenseSearcher) Schema() filter.Schema { return filter.EmptySchema() }

type captureDenseIndex struct{ valid bool }

func (i *captureDenseIndex) Upsert(ctx context.Context, _ []dense.Record) error {
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

type captureTensorSearcher struct{ valid bool }

func (s *captureTensorSearcher) Search(ctx context.Context, _ tensor.Request) ([]ragy.Document, error) {
	s.valid = trace.SpanFromContext(ctx).SpanContext().IsValid()
	return nil, nil
}

func (s *captureTensorSearcher) Schema() filter.Schema { return filter.EmptySchema() }

type captureLexicalSearcher struct{ valid bool }

func (s *captureLexicalSearcher) Search(ctx context.Context, _ lexical.Request) ([]ragy.Document, error) {
	s.valid = trace.SpanFromContext(ctx).SpanContext().IsValid()
	return nil, nil
}

func (s *captureLexicalSearcher) Schema() filter.Schema { return filter.EmptySchema() }

type captureGraphStore struct{ valid bool }

func (s *captureGraphStore) Traverse(ctx context.Context, _ graph.TraversalRequest) (graph.Snapshot, error) {
	s.valid = trace.SpanFromContext(ctx).SpanContext().IsValid()
	return graph.Snapshot{}, nil
}

func (s *captureGraphStore) Upsert(ctx context.Context, _ graph.Snapshot) error {
	s.valid = trace.SpanFromContext(ctx).SpanContext().IsValid()
	return nil
}

func (s *captureGraphStore) Schema() graph.Schema { return graph.EmptySchema() }

type captureTensorIndex struct{ valid bool }

func (i *captureTensorIndex) Upsert(ctx context.Context, _ []tensor.Record) error {
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

func (s *captureDocumentStore) FindByIDs(ctx context.Context, _ []string) ([]ragy.Document, error) {
	s.valid = trace.SpanFromContext(ctx).SpanContext().IsValid()
	return nil, nil
}

func (s *captureDocumentStore) DeleteByIDs(ctx context.Context, _ []string) (documents.DeleteResult, error) {
	s.valid = trace.SpanFromContext(ctx).SpanContext().IsValid()
	return documents.DeleteResult{}, nil
}

func (s *captureDocumentStore) DeleteByFilter(ctx context.Context, _ filter.IR) (documents.DeleteResult, error) {
	s.valid = trace.SpanFromContext(ctx).SpanContext().IsValid()
	return documents.DeleteResult{}, nil
}

func (s *captureDocumentStore) Schema() filter.Schema { return filter.EmptySchema() }

type captureQueryReranker struct{ valid bool }

func (r *captureQueryReranker) Rerank(ctx context.Context, _ string, _ []ragy.Document) ([]ragy.Document, error) {
	r.valid = trace.SpanFromContext(ctx).SpanContext().IsValid()
	return nil, nil
}

type captureMerger struct{ valid bool }

func (m *captureMerger) Merge(ctx context.Context, _ ...[]ragy.Document) ([]ragy.Document, error) {
	m.valid = trace.SpanFromContext(ctx).SpanContext().IsValid()
	return nil, nil
}

func TestWrapDenseSearcherPassesDerivedContext(t *testing.T) {
	runSpanTest(t, "ragy.dense.search", func(ctx context.Context, tracer trace.Tracer) (bool, error) {
		next := &captureDenseSearcher{}
		wrapped, err := WrapDenseSearcher(next, tracer)
		if err != nil {
			return false, err
		}
		_, err = wrapped.Search(ctx, dense.Request{Vector: []float32{1}})
		return next.valid, err
	})
}

func TestWrapDenseIndexPassesDerivedContext(t *testing.T) {
	runSpanTest(t, "ragy.dense.upsert", func(ctx context.Context, tracer trace.Tracer) (bool, error) {
		next := &captureDenseIndex{}
		wrapped, err := WrapDenseIndex(next, tracer)
		if err != nil {
			return false, err
		}
		err = wrapped.Upsert(ctx, []dense.Record{{ID: "doc-1", Vector: []float32{1}}})
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

func TestWrapTensorSearcherPassesDerivedContext(t *testing.T) {
	runSpanTest(t, "ragy.tensor.search", func(ctx context.Context, tracer trace.Tracer) (bool, error) {
		next := &captureTensorSearcher{}
		wrapped, err := WrapTensorSearcher(next, tracer)
		if err != nil {
			return false, err
		}
		_, err = wrapped.Search(ctx, tensor.Request{Query: tensor.Tensor{{1}}})
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
		err = wrapped.Upsert(ctx, []tensor.Record{{ID: "doc-1", Tensor: tensor.Tensor{{1}}}})
		return next.valid, err
	})
}

func TestWrapLexicalSearcherPassesDerivedContext(t *testing.T) {
	runSpanTest(t, "ragy.lexical.search", func(ctx context.Context, tracer trace.Tracer) (bool, error) {
		next := &captureLexicalSearcher{}
		wrapped, err := WrapLexicalSearcher(next, tracer)
		if err != nil {
			return false, err
		}
		_, err = wrapped.Search(ctx, lexical.Request{Text: "hello"})
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
		_, err = wrapped.DeleteByFilter(ctx, nil)
		return next.valid, err
	})
}

func TestWrapQueryRerankerPassesDerivedContext(t *testing.T) {
	runSpanTest(t, "ragy.ranking.rerank", func(ctx context.Context, tracer trace.Tracer) (bool, error) {
		next := &captureQueryReranker{}
		wrapped, err := WrapQueryReranker(next, tracer)
		if err != nil {
			return false, err
		}
		_, err = wrapped.Rerank(ctx, "hello", []ragy.Document{{ID: "doc-1"}})
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
		_, err = wrapped.Merge(ctx, []ragy.Document{{ID: "doc-1"}})
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
		err = wrapped.Upsert(ctx, graph.Snapshot{})
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

var (
	_ dense.Index           = (*captureDenseIndex)(nil)
	_ ranking.QueryReranker = (*captureQueryReranker)(nil)
	_ ranking.Merger        = (*captureMerger)(nil)
	_ lexical.Searcher      = (*captureLexicalSearcher)(nil)
	_ documents.Store       = (*captureDocumentStore)(nil)
)
