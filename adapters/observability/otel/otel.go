package otel

import (
	"context"
	"fmt"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/documents"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/graph"
	"github.com/skosovsky/ragy/lexical"
	"github.com/skosovsky/ragy/multimodal"
	"github.com/skosovsky/ragy/ranking"
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

// DenseSearcher wraps a dense searcher with tracing.
type DenseSearcher struct {
	next   dense.Searcher
	tracer trace.Tracer
}

// WrapDenseSearcher constructs a traced dense searcher.
func WrapDenseSearcher(next dense.Searcher, tracer trace.Tracer) (*DenseSearcher, error) {
	if next == nil {
		return nil, fmt.Errorf("%w: dense searcher", ragy.ErrInvalidArgument)
	}

	if tracer == nil {
		return nil, fmt.Errorf("%w: tracer", ragy.ErrInvalidArgument)
	}

	return &DenseSearcher{next: next, tracer: tracer}, nil
}

// Search implements dense.Searcher.
func (w *DenseSearcher) Search(ctx context.Context, req dense.Request) ([]ragy.Document, error) {
	ctx, span := w.tracer.Start(ctx, "ragy.dense.search")
	defer span.End()
	return w.next.Search(ctx, req)
}

// Schema returns the wrapped dense search schema.
func (w *DenseSearcher) Schema() filter.Schema {
	return w.next.Schema()
}

// DenseIndex wraps a dense index with tracing.
type DenseIndex struct {
	next   dense.Index
	tracer trace.Tracer
}

// WrapDenseIndex constructs a traced dense index.
func WrapDenseIndex(next dense.Index, tracer trace.Tracer) (*DenseIndex, error) {
	if next == nil {
		return nil, fmt.Errorf("%w: dense index", ragy.ErrInvalidArgument)
	}

	if tracer == nil {
		return nil, fmt.Errorf("%w: tracer", ragy.ErrInvalidArgument)
	}

	return &DenseIndex{next: next, tracer: tracer}, nil
}

// Upsert implements dense.Index.
func (w *DenseIndex) Upsert(ctx context.Context, records []dense.Record) error {
	ctx, span := w.tracer.Start(ctx, "ragy.dense.upsert")
	defer span.End()
	return w.next.Upsert(ctx, records)
}

// Schema returns the wrapped dense index schema.
func (w *DenseIndex) Schema() filter.Schema {
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

// TensorSearcher wraps a tensor searcher with tracing.
type TensorSearcher struct {
	next   tensor.Searcher
	tracer trace.Tracer
}

// WrapTensorSearcher constructs a traced tensor searcher.
func WrapTensorSearcher(next tensor.Searcher, tracer trace.Tracer) (*TensorSearcher, error) {
	if next == nil {
		return nil, fmt.Errorf("%w: tensor searcher", ragy.ErrInvalidArgument)
	}

	if tracer == nil {
		return nil, fmt.Errorf("%w: tracer", ragy.ErrInvalidArgument)
	}

	return &TensorSearcher{next: next, tracer: tracer}, nil
}

// Search implements tensor.Searcher.
func (w *TensorSearcher) Search(ctx context.Context, req tensor.Request) ([]ragy.Document, error) {
	ctx, span := w.tracer.Start(ctx, "ragy.tensor.search")
	defer span.End()
	return w.next.Search(ctx, req)
}

// Schema returns the wrapped tensor search schema.
func (w *TensorSearcher) Schema() filter.Schema {
	return w.next.Schema()
}

// TensorIndex wraps a tensor index with tracing.
type TensorIndex struct {
	next   tensor.Index
	tracer trace.Tracer
}

// WrapTensorIndex constructs a traced tensor index.
func WrapTensorIndex(next tensor.Index, tracer trace.Tracer) (*TensorIndex, error) {
	if next == nil {
		return nil, fmt.Errorf("%w: tensor index", ragy.ErrInvalidArgument)
	}

	if tracer == nil {
		return nil, fmt.Errorf("%w: tracer", ragy.ErrInvalidArgument)
	}

	return &TensorIndex{next: next, tracer: tracer}, nil
}

// Upsert implements tensor.Index.
func (w *TensorIndex) Upsert(ctx context.Context, records []tensor.Record) error {
	ctx, span := w.tracer.Start(ctx, "ragy.tensor.upsert")
	defer span.End()
	return w.next.Upsert(ctx, records)
}

// Schema returns the wrapped tensor index schema.
func (w *TensorIndex) Schema() filter.Schema {
	return w.next.Schema()
}

// LexicalSearcher wraps a lexical searcher with tracing.
type LexicalSearcher struct {
	next   lexical.Searcher
	tracer trace.Tracer
}

// WrapLexicalSearcher constructs a traced lexical searcher.
func WrapLexicalSearcher(next lexical.Searcher, tracer trace.Tracer) (*LexicalSearcher, error) {
	if next == nil {
		return nil, fmt.Errorf("%w: lexical searcher", ragy.ErrInvalidArgument)
	}

	if tracer == nil {
		return nil, fmt.Errorf("%w: tracer", ragy.ErrInvalidArgument)
	}

	return &LexicalSearcher{next: next, tracer: tracer}, nil
}

// Search implements lexical.Searcher.
func (w *LexicalSearcher) Search(ctx context.Context, req lexical.Request) ([]ragy.Document, error) {
	ctx, span := w.tracer.Start(ctx, "ragy.lexical.search")
	defer span.End()
	return w.next.Search(ctx, req)
}

// Schema returns the wrapped lexical search schema.
func (w *LexicalSearcher) Schema() filter.Schema {
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
type GraphStore struct {
	next   graph.Store
	tracer trace.Tracer
}

// WrapGraphStore constructs a traced graph store.
func WrapGraphStore(next graph.Store, tracer trace.Tracer) (*GraphStore, error) {
	if next == nil {
		return nil, fmt.Errorf("%w: graph store", ragy.ErrInvalidArgument)
	}

	if tracer == nil {
		return nil, fmt.Errorf("%w: tracer", ragy.ErrInvalidArgument)
	}
	return &GraphStore{next: next, tracer: tracer}, nil
}

// Traverse implements graph.Store.
func (w *GraphStore) Traverse(ctx context.Context, req graph.TraversalRequest) (graph.Snapshot, error) {
	ctx, span := w.tracer.Start(ctx, "ragy.graph.traverse")
	defer span.End()
	return w.next.Traverse(ctx, req)
}

// Upsert implements graph.Store.
func (w *GraphStore) Upsert(ctx context.Context, snapshot graph.Snapshot) error {
	ctx, span := w.tracer.Start(ctx, "ragy.graph.upsert")
	defer span.End()
	return w.next.Upsert(ctx, snapshot)
}

// Schema returns the wrapped graph schema.
func (w *GraphStore) Schema() graph.Schema {
	return w.next.Schema()
}

// DocumentStore wraps a document store with tracing.
type DocumentStore struct {
	next   documents.Store
	tracer trace.Tracer
}

// WrapDocumentStore constructs a traced document store.
func WrapDocumentStore(next documents.Store, tracer trace.Tracer) (*DocumentStore, error) {
	if next == nil {
		return nil, fmt.Errorf("%w: document store", ragy.ErrInvalidArgument)
	}

	if tracer == nil {
		return nil, fmt.Errorf("%w: tracer", ragy.ErrInvalidArgument)
	}

	return &DocumentStore{next: next, tracer: tracer}, nil
}

// FindByIDs implements documents.Store.
func (w *DocumentStore) FindByIDs(ctx context.Context, ids []string) ([]ragy.Document, error) {
	ctx, span := w.tracer.Start(ctx, "ragy.documents.find")
	defer span.End()
	return w.next.FindByIDs(ctx, ids)
}

// DeleteByIDs implements documents.Store.
func (w *DocumentStore) DeleteByIDs(ctx context.Context, ids []string) (documents.DeleteResult, error) {
	ctx, span := w.tracer.Start(ctx, "ragy.documents.delete_ids")
	defer span.End()
	return w.next.DeleteByIDs(ctx, ids)
}

// DeleteByFilter implements documents.Store.
func (w *DocumentStore) DeleteByFilter(ctx context.Context, expr filter.IR) (documents.DeleteResult, error) {
	ctx, span := w.tracer.Start(ctx, "ragy.documents.delete_filter")
	defer span.End()
	return w.next.DeleteByFilter(ctx, expr)
}

// Schema returns the wrapped document-store schema.
func (w *DocumentStore) Schema() filter.Schema {
	return w.next.Schema()
}

// QueryReranker wraps a query-aware reranker with tracing.
type QueryReranker struct {
	next   ranking.QueryReranker
	tracer trace.Tracer
}

// WrapQueryReranker constructs a traced query-aware reranker.
func WrapQueryReranker(next ranking.QueryReranker, tracer trace.Tracer) (*QueryReranker, error) {
	if next == nil {
		return nil, fmt.Errorf("%w: query reranker", ragy.ErrInvalidArgument)
	}

	if tracer == nil {
		return nil, fmt.Errorf("%w: tracer", ragy.ErrInvalidArgument)
	}

	return &QueryReranker{next: next, tracer: tracer}, nil
}

// Rerank implements ranking.QueryReranker.
func (w *QueryReranker) Rerank(ctx context.Context, query string, docs []ragy.Document) ([]ragy.Document, error) {
	ctx, span := w.tracer.Start(ctx, "ragy.ranking.rerank")
	defer span.End()
	return w.next.Rerank(ctx, query, docs)
}

// Merger wraps a ranked-list merger with tracing.
type Merger struct {
	next   ranking.Merger
	tracer trace.Tracer
}

// WrapMerger constructs a traced ranked-list merger.
func WrapMerger(next ranking.Merger, tracer trace.Tracer) (*Merger, error) {
	if next == nil {
		return nil, fmt.Errorf("%w: ranking merger", ragy.ErrInvalidArgument)
	}

	if tracer == nil {
		return nil, fmt.Errorf("%w: tracer", ragy.ErrInvalidArgument)
	}

	return &Merger{next: next, tracer: tracer}, nil
}

// Merge implements ranking.Merger.
func (w *Merger) Merge(ctx context.Context, lists ...[]ragy.Document) ([]ragy.Document, error) {
	ctx, span := w.tracer.Start(ctx, "ragy.ranking.merge")
	defer span.End()
	return w.next.Merge(ctx, lists...)
}

var (
	_ dense.Embedder        = (*DenseEmbedder)(nil)
	_ dense.Searcher        = (*DenseSearcher)(nil)
	_ dense.Index           = (*DenseIndex)(nil)
	_ tensor.Embedder       = (*TensorEmbedder)(nil)
	_ tensor.Searcher       = (*TensorSearcher)(nil)
	_ tensor.Index          = (*TensorIndex)(nil)
	_ lexical.Searcher      = (*LexicalSearcher)(nil)
	_ multimodal.Embedder   = (*MultimodalEmbedder)(nil)
	_ graph.Store           = (*GraphStore)(nil)
	_ documents.Store       = (*DocumentStore)(nil)
	_ ranking.QueryReranker = (*QueryReranker)(nil)
	_ ranking.Merger        = (*Merger)(nil)
)
