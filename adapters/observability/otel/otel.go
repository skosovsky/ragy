package otel

import (
	"context"
	"fmt"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/documents"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/graph"
	"github.com/skosovsky/ragy/multimodal"
	"github.com/skosovsky/ragy/ranking"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/tensor"

	"go.opentelemetry.io/otel/trace"
)

// DenseEmbedder wraps a dense embedder with tracing.
type DenseEmbedder struct {
	next   dense.Embedder
	tracer trace.Tracer
}

// WrapDenseEmbedder constructs a traced dense embedder.
func WrapDenseEmbedder(next dense.Embedder, tracer trace.Tracer) (*DenseEmbedder, error) {
	if next == nil {
		return nil, fmt.Errorf("%w: dense embedder", ragy.ErrInvalidArgument)
	}

	if tracer == nil {
		return nil, fmt.Errorf("%w: tracer", ragy.ErrInvalidArgument)
	}

	return &DenseEmbedder{next: next, tracer: tracer}, nil
}

// Embed implements dense.Embedder.
func (w *DenseEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	ctx, span := w.tracer.Start(ctx, "ragy.dense.embed")
	defer span.End()
	return w.next.Embed(ctx, texts)
}

// RequestBackend wraps a typed retrieval backend with tracing.
type RequestBackend[TIntent, TRequestMeta, TMeta any] struct {
	next   retrieval.RequestBackend[TIntent, TRequestMeta, TMeta]
	tracer trace.Tracer
}

// Backend is the no-request-metadata traced retrieval backend.
type Backend[TIntent, TMeta any] = RequestBackend[TIntent, retrieval.NoRequestMeta, TMeta]

// WrapRequestBackend constructs a traced request-metadata-aware retrieval backend.
func WrapRequestBackend[TIntent, TRequestMeta, TMeta any](
	next retrieval.RequestBackend[TIntent, TRequestMeta, TMeta],
	tracer trace.Tracer,
) (*RequestBackend[TIntent, TRequestMeta, TMeta], error) {
	if next == nil {
		return nil, fmt.Errorf("%w: retrieval backend", ragy.ErrInvalidArgument)
	}
	if tracer == nil {
		return nil, fmt.Errorf("%w: tracer", ragy.ErrInvalidArgument)
	}
	return &RequestBackend[TIntent, TRequestMeta, TMeta]{next: next, tracer: tracer}, nil
}

// WrapBackend constructs a traced no-request-metadata retrieval backend.
func WrapBackend[TIntent, TMeta any](
	next retrieval.Backend[TIntent, TMeta],
	tracer trace.Tracer,
) (*Backend[TIntent, TMeta], error) {
	return WrapRequestBackend[TIntent, retrieval.NoRequestMeta, TMeta](next, tracer)
}

// Retrieve implements retrieval.RequestBackend.
func (w *RequestBackend[TIntent, TRequestMeta, TMeta]) Retrieve(
	ctx context.Context,
	req retrieval.Request[TIntent, TRequestMeta],
) (retrieval.ResultSet[TMeta], error) {
	ctx, span := w.tracer.Start(ctx, "ragy.retrieval.backend")
	defer span.End()
	return w.next.Retrieve(ctx, req)
}

var _ retrieval.RequestBackend[struct{}, struct{}, any] = (*RequestBackend[struct{}, struct{}, any])(nil)

// DenseIndex wraps a dense index with tracing.
type DenseIndex[TMeta any] struct {
	next   dense.Index[TMeta]
	tracer trace.Tracer
}

// WrapDenseIndex constructs a traced dense index.
func WrapDenseIndex[TMeta any](next dense.Index[TMeta], tracer trace.Tracer) (*DenseIndex[TMeta], error) {
	if next == nil {
		return nil, fmt.Errorf("%w: dense index", ragy.ErrInvalidArgument)
	}

	if tracer == nil {
		return nil, fmt.Errorf("%w: tracer", ragy.ErrInvalidArgument)
	}

	return &DenseIndex[TMeta]{next: next, tracer: tracer}, nil
}

// Upsert implements dense.Index.
func (w *DenseIndex[TMeta]) Upsert(ctx context.Context, records []dense.Record[TMeta]) error {
	ctx, span := w.tracer.Start(ctx, "ragy.dense.upsert")
	defer span.End()
	return w.next.Upsert(ctx, records)
}

// Schema returns the wrapped dense index schema.
func (w *DenseIndex[TMeta]) Schema() filter.Schema {
	return w.next.Schema()
}

// TensorEmbedder wraps a tensor embedder with tracing.
type TensorEmbedder struct {
	next   tensor.Embedder
	tracer trace.Tracer
}

// WrapTensorEmbedder constructs a traced tensor embedder.
func WrapTensorEmbedder(next tensor.Embedder, tracer trace.Tracer) (*TensorEmbedder, error) {
	if next == nil {
		return nil, fmt.Errorf("%w: tensor embedder", ragy.ErrInvalidArgument)
	}

	if tracer == nil {
		return nil, fmt.Errorf("%w: tracer", ragy.ErrInvalidArgument)
	}

	return &TensorEmbedder{next: next, tracer: tracer}, nil
}

// Embed implements tensor.Embedder.
func (w *TensorEmbedder) Embed(ctx context.Context, texts []string) ([]tensor.Tensor, error) {
	ctx, span := w.tracer.Start(ctx, "ragy.tensor.embed")
	defer span.End()
	return w.next.Embed(ctx, texts)
}

// TensorIndex wraps a tensor index with tracing.
type TensorIndex[TMeta any] struct {
	next   tensor.Index[TMeta]
	tracer trace.Tracer
}

// WrapTensorIndex constructs a traced tensor index.
func WrapTensorIndex[TMeta any](next tensor.Index[TMeta], tracer trace.Tracer) (*TensorIndex[TMeta], error) {
	if next == nil {
		return nil, fmt.Errorf("%w: tensor index", ragy.ErrInvalidArgument)
	}

	if tracer == nil {
		return nil, fmt.Errorf("%w: tracer", ragy.ErrInvalidArgument)
	}

	return &TensorIndex[TMeta]{next: next, tracer: tracer}, nil
}

// Upsert implements tensor.Index.
func (w *TensorIndex[TMeta]) Upsert(ctx context.Context, records []tensor.Record[TMeta]) error {
	ctx, span := w.tracer.Start(ctx, "ragy.tensor.upsert")
	defer span.End()
	return w.next.Upsert(ctx, records)
}

// Schema returns the wrapped tensor index schema.
func (w *TensorIndex[TMeta]) Schema() filter.Schema {
	return w.next.Schema()
}

// MultimodalEmbedder wraps a multimodal embedder with tracing.
type MultimodalEmbedder struct {
	next   multimodal.Embedder
	tracer trace.Tracer
}

// WrapMultimodalEmbedder constructs a traced multimodal embedder.
func WrapMultimodalEmbedder(next multimodal.Embedder, tracer trace.Tracer) (*MultimodalEmbedder, error) {
	if next == nil {
		return nil, fmt.Errorf("%w: multimodal embedder", ragy.ErrInvalidArgument)
	}

	if tracer == nil {
		return nil, fmt.Errorf("%w: tracer", ragy.ErrInvalidArgument)
	}

	return &MultimodalEmbedder{next: next, tracer: tracer}, nil
}

// Embed implements multimodal.Embedder.
func (w *MultimodalEmbedder) Embed(ctx context.Context, inputs []multimodal.Input) ([][]float32, error) {
	ctx, span := w.tracer.Start(ctx, "ragy.multimodal.embed")
	defer span.End()
	return w.next.Embed(ctx, inputs)
}

// GraphStore wraps a graph store with tracing.
type GraphStore[TMeta any] struct {
	next   graph.Store[TMeta]
	tracer trace.Tracer
}

// WrapGraphStore constructs a traced graph store.
func WrapGraphStore[TMeta any](next graph.Store[TMeta], tracer trace.Tracer) (*GraphStore[TMeta], error) {
	if next == nil {
		return nil, fmt.Errorf("%w: graph store", ragy.ErrInvalidArgument)
	}

	if tracer == nil {
		return nil, fmt.Errorf("%w: tracer", ragy.ErrInvalidArgument)
	}
	return &GraphStore[TMeta]{next: next, tracer: tracer}, nil
}

// Traverse implements graph.Store.
func (w *GraphStore[TMeta]) Traverse(ctx context.Context, req graph.TraversalRequest) (graph.Snapshot[TMeta], error) {
	ctx, span := w.tracer.Start(ctx, "ragy.graph.traverse")
	defer span.End()
	return w.next.Traverse(ctx, req)
}

// Upsert implements graph.Store.
func (w *GraphStore[TMeta]) Upsert(ctx context.Context, snapshot graph.Snapshot[TMeta]) error {
	ctx, span := w.tracer.Start(ctx, "ragy.graph.upsert")
	defer span.End()
	return w.next.Upsert(ctx, snapshot)
}

// Schema returns the wrapped graph schema.
func (w *GraphStore[TMeta]) Schema() graph.Schema {
	return w.next.Schema()
}

// DocumentStore wraps a document store with tracing.
type DocumentStore[TMeta any] struct {
	next   documents.Store[TMeta]
	tracer trace.Tracer
}

// WrapDocumentStore constructs a traced document store.
func WrapDocumentStore[TMeta any](next documents.Store[TMeta], tracer trace.Tracer) (*DocumentStore[TMeta], error) {
	if next == nil {
		return nil, fmt.Errorf("%w: document store", ragy.ErrInvalidArgument)
	}

	if tracer == nil {
		return nil, fmt.Errorf("%w: tracer", ragy.ErrInvalidArgument)
	}

	return &DocumentStore[TMeta]{next: next, tracer: tracer}, nil
}

// FindByIDs implements documents.Store.
func (w *DocumentStore[TMeta]) FindByIDs(ctx context.Context, ids []string) ([]retrieval.Document[TMeta], error) {
	ctx, span := w.tracer.Start(ctx, "ragy.documents.find")
	defer span.End()
	return w.next.FindByIDs(ctx, ids)
}

// DeleteByIDs implements documents.Store.
func (w *DocumentStore[TMeta]) DeleteByIDs(ctx context.Context, ids []string) (documents.DeleteResult, error) {
	ctx, span := w.tracer.Start(ctx, "ragy.documents.delete_ids")
	defer span.End()
	return w.next.DeleteByIDs(ctx, ids)
}

// DeleteByFilter implements documents.Store.
func (w *DocumentStore[TMeta]) DeleteByFilter(
	ctx context.Context,
	cond filter.Condition,
) (documents.DeleteResult, error) {
	ctx, span := w.tracer.Start(ctx, "ragy.documents.delete_filter")
	defer span.End()
	return w.next.DeleteByFilter(ctx, cond)
}

// Schema returns the wrapped document-store schema.
func (w *DocumentStore[TMeta]) Schema() filter.Schema {
	return w.next.Schema()
}

// QueryReranker wraps a query-aware reranker with tracing.
type QueryReranker[TMeta any] struct {
	next   ranking.QueryReranker[TMeta]
	tracer trace.Tracer
}

// WrapQueryReranker constructs a traced query-aware reranker.
func WrapQueryReranker[TMeta any](
	next ranking.QueryReranker[TMeta],
	tracer trace.Tracer,
) (*QueryReranker[TMeta], error) {
	if next == nil {
		return nil, fmt.Errorf("%w: query reranker", ragy.ErrInvalidArgument)
	}

	if tracer == nil {
		return nil, fmt.Errorf("%w: tracer", ragy.ErrInvalidArgument)
	}

	return &QueryReranker[TMeta]{next: next, tracer: tracer}, nil
}

// Rerank implements ranking.QueryReranker.
func (w *QueryReranker[TMeta]) Rerank(
	ctx context.Context,
	query string,
	rs retrieval.ResultSet[TMeta],
) (retrieval.ResultSet[TMeta], error) {
	ctx, span := w.tracer.Start(ctx, "ragy.ranking.rerank")
	defer span.End()
	return w.next.Rerank(ctx, query, rs)
}

// RequestPipeline wraps a declarative retrieval orchestrator with tracing.
type RequestPipeline[TIntent, TRequestMeta, TMeta any] struct {
	next   *retrieval.RequestPipeline[TIntent, TRequestMeta, TMeta]
	tracer trace.Tracer
}

// Pipeline is the no-request-metadata traced retrieval orchestrator.
type Pipeline[TIntent, TMeta any] = RequestPipeline[TIntent, retrieval.NoRequestMeta, TMeta]

// WrapRequestPipeline constructs a traced request-metadata-aware retrieval orchestrator.
func WrapRequestPipeline[TIntent, TRequestMeta, TMeta any](
	next *retrieval.RequestPipeline[TIntent, TRequestMeta, TMeta],
	tracer trace.Tracer,
) (*RequestPipeline[TIntent, TRequestMeta, TMeta], error) {
	if next == nil {
		return nil, fmt.Errorf("%w: retrieval pipeline", ragy.ErrInvalidArgument)
	}
	if tracer == nil {
		return nil, fmt.Errorf("%w: tracer", ragy.ErrInvalidArgument)
	}
	return &RequestPipeline[TIntent, TRequestMeta, TMeta]{next: next, tracer: tracer}, nil
}

// WrapPipeline constructs a traced no-request-metadata retrieval orchestrator.
func WrapPipeline[TIntent, TMeta any](
	next *retrieval.Pipeline[TIntent, TMeta],
	tracer trace.Tracer,
) (*Pipeline[TIntent, TMeta], error) {
	return WrapRequestPipeline[TIntent, retrieval.NoRequestMeta, TMeta](next, tracer)
}

// Retrieve implements orchestrator retrieval with pipeline span.
func (w *RequestPipeline[TIntent, TRequestMeta, TMeta]) Retrieve(
	ctx context.Context,
	query retrieval.Request[TIntent, TRequestMeta],
) (retrieval.ResultSet[TMeta], error) {
	ctx, span := w.tracer.Start(ctx, "ragy.retrieval.pipeline")
	defer span.End()
	return w.next.Retrieve(ctx, query)
}

// Merger wraps a ranked-list merger with tracing.
type Merger[TMeta any] struct {
	next   ranking.Merger[TMeta]
	tracer trace.Tracer
}

// WrapMerger constructs a traced ranked-list merger.
func WrapMerger[TMeta any](next ranking.Merger[TMeta], tracer trace.Tracer) (*Merger[TMeta], error) {
	if next == nil {
		return nil, fmt.Errorf("%w: ranking merger", ragy.ErrInvalidArgument)
	}

	if tracer == nil {
		return nil, fmt.Errorf("%w: tracer", ragy.ErrInvalidArgument)
	}

	return &Merger[TMeta]{next: next, tracer: tracer}, nil
}

// Merge implements ranking.Merger.
func (w *Merger[TMeta]) Merge(
	ctx context.Context,
	sets ...retrieval.ResultSet[TMeta],
) (retrieval.ResultSet[TMeta], error) {
	ctx, span := w.tracer.Start(ctx, "ragy.ranking.merge")
	defer span.End()
	return w.next.Merge(ctx, sets...)
}

var (
	_ dense.Embedder                   = (*DenseEmbedder)(nil)
	_ retrieval.Backend[struct{}, any] = (*Backend[struct{}, any])(nil)
	_ dense.Index[any]                 = (*DenseIndex[any])(nil)
	_ tensor.Embedder                  = (*TensorEmbedder)(nil)
	_ tensor.Index[any]                = (*TensorIndex[any])(nil)
	_ multimodal.Embedder              = (*MultimodalEmbedder)(nil)
	_ graph.Store[any]                 = (*GraphStore[any])(nil)
	_ documents.Store[any]             = (*DocumentStore[any])(nil)
	_ ranking.QueryReranker[any]       = (*QueryReranker[any])(nil)
	_ ranking.Merger[any]              = (*Merger[any])(nil)
)
