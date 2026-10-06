package otel

import (
	"github.com/skosovsky/ragy/access"

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

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/skosovsky/ragy/observation"
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

// Space forwards the configured encoding profile.
func (w *DenseEmbedder) Space() dense.Space { return w.next.Space() }

func (w *DenseEmbedder) Embed(ctx context.Context, texts dense.Request) (dense.Result, error) {
	ctx, span := w.tracer.Start(ctx, "ragy.dense.embed")
	defer span.End()
	result, err := w.next.Embed(ctx, texts)
	recordEncoding(span, len(result.Embeddings), result.Usage, err)
	return result, err
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

// Schema forwards the wrapped target's admission schema.
func (w *RequestBackend[TIntent, TRequestMeta, TMeta]) Schema() filter.Schema {
	if provider, ok := w.next.(retrieval.ReadCapabilityProvider); ok {
		return provider.Schema()
	}
	return filter.Schema{}
}

// ReadCapabilities forwards only the wrapped target's declared guarantees.
func (w *RequestBackend[TIntent, TRequestMeta, TMeta]) ReadCapabilities() access.Capabilities {
	if provider, ok := w.next.(retrieval.ReadCapabilityProvider); ok {
		return provider.ReadCapabilities()
	}
	return access.Capabilities{}
}

// AdmitRead preserves request-aware target negotiation without target payload I/O.
func (w *RequestBackend[TIntent, TRequestMeta, TMeta]) AdmitRead(
	ctx context.Context,
	req retrieval.Request[TIntent, TRequestMeta],
) (retrieval.ReadCoverage, error) {
	return retrieval.InspectRead(
		ctx,
		req,
		retrieval.RequestBackendNode[TIntent, TRequestMeta, TMeta, retrieval.NoExecutionMeta]{
			Backend:  w.next,
			Resolver: nil,
			Name:     "",
		},
	)
}

// AdmitPublication forwards partial publication admission.
func (w *RequestBackend[TIntent, TRequestMeta, TMeta]) AdmitPublication(publication access.Publication) error {
	if admission, ok := w.next.(retrieval.PublicationAdmission); ok {
		return admission.AdmitPublication(publication)
	}
	return access.UnsupportedCapability(ragy.ErrUnsupported)
}

// Retrieve implements retrieval.RequestBackend.
func (w *RequestBackend[TIntent, TRequestMeta, TMeta]) Retrieve(
	ctx context.Context,
	req retrieval.Request[TIntent, TRequestMeta],
) (retrieval.ResultSet[TMeta], error) {
	ctx, span := w.tracer.Start(ctx, "ragy.retrieval.backend")
	defer span.End()
	if err := req.Read.Check(ctx); err != nil {
		recordOutcome(span, err, 0)
		return retrieval.NewResultSet[TMeta](nil, nil), err
	}
	coverage, err := w.AdmitRead(ctx, req)
	if err != nil {
		recordOutcome(span, err, 0)
		return retrieval.NewResultSet[TMeta](nil, nil), err
	}
	rs, err := w.next.Retrieve(ctx, req)
	result, err := retrieval.DeliverRead(ctx, req.Read, rs, err, nil)
	recordOutcome(span, err, resultCount(result))
	if err == nil && coverage.IsPartial() {
		span.SetAttributes(attribute.Int("ragy.outcome", int(observation.OutcomePartial)))
	}
	return result, err
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
	err := w.next.Upsert(ctx, records)
	span.SetAttributes(attribute.Int("ragy.input.count", len(records)))
	recordOutcome(span, err, -1)
	return err
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

// Space forwards the configured encoding profile.
func (w *TensorEmbedder) Space() tensor.Space { return w.next.Space() }

func (w *TensorEmbedder) Embed(ctx context.Context, texts tensor.Request) (tensor.Result, error) {
	ctx, span := w.tracer.Start(ctx, "ragy.tensor.embed")
	defer span.End()
	result, err := w.next.Embed(ctx, texts)
	recordEncoding(span, len(result.Embeddings), result.Usage, err)
	return result, err
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
	err := w.next.Upsert(ctx, records)
	span.SetAttributes(attribute.Int("ragy.input.count", len(records)))
	recordOutcome(span, err, -1)
	return err
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

// Space forwards the configured encoding profile.
func (w *MultimodalEmbedder) Space() dense.Space { return w.next.Space() }

func (w *MultimodalEmbedder) Embed(ctx context.Context, inputs multimodal.Request) (multimodal.Result, error) {
	ctx, span := w.tracer.Start(ctx, "ragy.multimodal.embed")
	defer span.End()
	result, err := w.next.Embed(ctx, inputs)
	recordEncoding(span, len(result.Embeddings), result.Usage, err)
	return result, err
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
	result, err := w.next.Traverse(ctx, req)
	recordOutcome(span, err, len(result.Nodes)+len(result.Edges))
	return result, err
}

// Upsert implements graph.Store.
func (w *GraphStore[TMeta]) Upsert(ctx context.Context, snapshot graph.Snapshot[TMeta]) error {
	ctx, span := w.tracer.Start(ctx, "ragy.graph.upsert")
	defer span.End()
	err := w.next.Upsert(ctx, snapshot)
	span.SetAttributes(attribute.Int("ragy.input.count", len(snapshot.Nodes)+len(snapshot.Edges)))
	recordOutcome(span, err, -1)
	return err
}

// Schema returns the wrapped graph schema.
func (w *GraphStore[TMeta]) Schema() graph.Schema {
	return w.next.Schema()
}

// DocumentStore wraps a document store with tracing.
type DocumentStore[TMeta any] struct {
	next   documents.RawStore[TMeta]
	tracer trace.Tracer
}

// WrapDocumentStore constructs a traced document store.
func WrapDocumentStore[TMeta any](next documents.RawStore[TMeta], tracer trace.Tracer) (*DocumentStore[TMeta], error) {
	if next == nil {
		return nil, fmt.Errorf("%w: document store", ragy.ErrInvalidArgument)
	}

	if tracer == nil {
		return nil, fmt.Errorf("%w: tracer", ragy.ErrInvalidArgument)
	}

	return &DocumentStore[TMeta]{next: next, tracer: tracer}, nil
}

// FindByIDs implements documents.RawStore.
func (w *DocumentStore[TMeta]) FindByIDs(ctx context.Context, ids []string) ([]retrieval.Document[TMeta], error) {
	ctx, span := w.tracer.Start(ctx, "ragy.documents.find")
	defer span.End()
	result, err := w.next.FindByIDs(ctx, ids)
	recordOutcome(span, err, len(result))
	return result, err
}

// DeleteByIDs implements documents.RawStore.
func (w *DocumentStore[TMeta]) DeleteByIDs(ctx context.Context, ids []string) (documents.DeleteResult, error) {
	ctx, span := w.tracer.Start(ctx, "ragy.documents.delete_ids")
	defer span.End()
	result, err := w.next.DeleteByIDs(ctx, ids)
	recordOutcome(span, err, result.Deleted)
	return result, err
}

// DeleteByFilter implements documents.RawStore.
func (w *DocumentStore[TMeta]) DeleteByFilter(
	ctx context.Context,
	cond filter.Condition,
) (documents.DeleteResult, error) {
	ctx, span := w.tracer.Start(ctx, "ragy.documents.delete_filter")
	defer span.End()
	result, err := w.next.DeleteByFilter(ctx, cond)
	recordOutcome(span, err, result.Deleted)
	return result, err
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
	read access.Binding, query string,
	rs retrieval.ResultSet[TMeta],
) (retrieval.ResultSet[TMeta], error) {
	ctx, span := w.tracer.Start(ctx, "ragy.ranking.rerank")
	defer span.End()
	if err := read.Check(ctx); err != nil {
		recordOutcome(span, err, 0)
		return retrieval.NewResultSet[TMeta](nil, retrieval.ResolverFor(rs)), err
	}
	result, err := w.next.Rerank(ctx, read, query, rs)
	result, err = retrieval.DeliverRead(ctx, read, result, err, retrieval.ResolverFor(rs))
	recordOutcome(span, err, resultCount(result))
	return result, err
}

// RequestExecutionPipeline wraps an execution-aware retrieval orchestrator with tracing.
type RequestExecutionPipeline[TIntent, TRequestMeta, TMeta, TExecMeta any] struct {
	next   *retrieval.RequestExecutionPipeline[TIntent, TRequestMeta, TMeta, TExecMeta]
	tracer trace.Tracer
}

// ExecutionPipeline is the no-request-metadata traced execution orchestrator.
type ExecutionPipeline[TIntent, TMeta, TExecMeta any] = RequestExecutionPipeline[
	TIntent,
	retrieval.NoRequestMeta,
	TMeta,
	TExecMeta,
]

// WrapRequestExecutionPipeline constructs a traced execution-aware retrieval orchestrator.
func WrapRequestExecutionPipeline[TIntent, TRequestMeta, TMeta, TExecMeta any](
	next *retrieval.RequestExecutionPipeline[TIntent, TRequestMeta, TMeta, TExecMeta],
	tracer trace.Tracer,
) (*RequestExecutionPipeline[TIntent, TRequestMeta, TMeta, TExecMeta], error) {
	if next == nil {
		return nil, fmt.Errorf("%w: retrieval execution pipeline", ragy.ErrInvalidArgument)
	}
	if tracer == nil {
		return nil, fmt.Errorf("%w: tracer", ragy.ErrInvalidArgument)
	}
	return &RequestExecutionPipeline[TIntent, TRequestMeta, TMeta, TExecMeta]{
		next:   next,
		tracer: tracer,
	}, nil
}

// WrapExecutionPipeline constructs a traced no-request-metadata execution orchestrator.
func WrapExecutionPipeline[TIntent, TMeta, TExecMeta any](
	next *retrieval.ExecutionPipeline[TIntent, TMeta, TExecMeta],
	tracer trace.Tracer,
) (*ExecutionPipeline[TIntent, TMeta, TExecMeta], error) {
	return WrapRequestExecutionPipeline[TIntent, retrieval.NoRequestMeta, TMeta, TExecMeta](next, tracer)
}

// Execute implements execution-aware retrieval with pipeline span.
func (w *RequestExecutionPipeline[TIntent, TRequestMeta, TMeta, TExecMeta]) Execute(
	ctx context.Context,
	query retrieval.Request[TIntent, TRequestMeta],
) (retrieval.RetrievalResult[TMeta, TExecMeta], error) {
	ctx, span := w.tracer.Start(ctx, "ragy.retrieval.pipeline")
	defer span.End()
	result, err := w.next.Execute(ctx, query)
	recordOutcome(span, err, resultCount(result.ResultSet))
	if err == nil && result.Coverage.IsPartial() {
		span.SetAttributes(attribute.Int("ragy.outcome", int(observation.OutcomePartial)))
	}
	return result, err
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
	result, err := w.next.Merge(ctx, sets...)
	recordOutcome(span, err, resultCount(result))
	return result, err
}

var (
	_ dense.Embedder                   = (*DenseEmbedder)(nil)
	_ retrieval.Backend[struct{}, any] = (*Backend[struct{}, any])(nil)
	_ dense.Index[any]                 = (*DenseIndex[any])(nil)
	_ tensor.Embedder                  = (*TensorEmbedder)(nil)
	_ tensor.Index[any]                = (*TensorIndex[any])(nil)
	_ multimodal.Embedder              = (*MultimodalEmbedder)(nil)
	_ graph.Store[any]                 = (*GraphStore[any])(nil)
	_ documents.RawStore[any]          = (*DocumentStore[any])(nil)
	_ ranking.QueryReranker[any]       = (*QueryReranker[any])(nil)
	_ ranking.Merger[any]              = (*Merger[any])(nil)
)
