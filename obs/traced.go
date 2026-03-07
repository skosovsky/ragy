// Package obs provides OpenTelemetry tracing decorators for ragy Retriever, VectorStore, GraphStore, and Reranker.
package obs

import (
	"context"
	"encoding/json"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
)

// TracedRetriever wraps a Retriever and creates a span for each Retrieve call.
func TracedRetriever(retriever ragy.Retriever, tracer trace.Tracer) ragy.Retriever {
	return &tracedRetriever{inner: retriever, tracer: tracer}
}

// NewTracedRetriever is an alias for TracedRetriever (TD naming).
func NewTracedRetriever(retriever ragy.Retriever, tracer trace.Tracer) ragy.Retriever {
	return TracedRetriever(retriever, tracer)
}

type tracedRetriever struct {
	inner  ragy.Retriever
	tracer trace.Tracer
}

// Retrieve runs the inner Retriever and records span attributes. On success, EvalData is
// written to span as ragy.eval_data (JSON). When err != nil, EvalData and results_count
// are not set so traces clearly distinguish error from success-with-zero-results.
func (t *tracedRetriever) Retrieve(ctx context.Context, req ragy.SearchRequest) (ragy.RetrievalResult, error) {
	ctx, span := t.tracer.Start(ctx, "ragy.retriever.retrieve")
	defer span.End()
	start := time.Now()
	res, err := t.inner.Retrieve(ctx, req)
	dur := time.Since(start)

	span.SetAttributes(
		attribute.String("ragy.query", req.Query),
		attribute.Int("ragy.limit", req.Limit),
		attribute.Int64("ragy.duration_ms", dur.Milliseconds()),
	)
	if req.Filter != nil {
		span.SetAttributes(attribute.String("ragy.filters", "set"))
	}
	if err != nil {
		span.RecordError(err)
		return res, err
	}
	span.SetAttributes(attribute.Int("ragy.results_count", len(res.Documents)))
	if len(res.EvalData) > 0 {
		if b, marshalErr := json.Marshal(res.EvalData); marshalErr == nil {
			span.SetAttributes(attribute.String("ragy.eval_data", string(b)))
		}
	}
	return res, nil
}

// TracedVectorStore wraps a VectorStore with tracing.
func TracedVectorStore(store ragy.VectorStore, tracer trace.Tracer) ragy.VectorStore {
	return &tracedVectorStore{inner: store, tracer: tracer}
}

// NewTracedVectorStore is an alias for TracedVectorStore (TD naming).
func NewTracedVectorStore(store ragy.VectorStore, tracer trace.Tracer) ragy.VectorStore {
	return TracedVectorStore(store, tracer)
}

type tracedVectorStore struct {
	inner  ragy.VectorStore
	tracer trace.Tracer
}

func (t *tracedVectorStore) Search(ctx context.Context, req ragy.SearchRequest) ([]ragy.Document, error) {
	ctx, span := t.tracer.Start(ctx, "ragy.vectorstore.search")
	defer span.End()
	start := time.Now()
	docs, err := t.inner.Search(ctx, req)
	dur := time.Since(start)
	span.SetAttributes(
		attribute.String("ragy.store.operation", "search"),
		attribute.Int("ragy.store.results_count", len(docs)),
		attribute.Int64("ragy.duration_ms", dur.Milliseconds()),
	)
	if err != nil {
		span.RecordError(err)
	}
	return docs, err
}

func (t *tracedVectorStore) Upsert(ctx context.Context, docs []ragy.Document) error {
	ctx, span := t.tracer.Start(ctx, "ragy.vectorstore.upsert")
	defer span.End()
	span.SetAttributes(
		attribute.String("ragy.store.operation", "upsert"),
		attribute.Int("ragy.store.docs_count", len(docs)),
	)
	err := t.inner.Upsert(ctx, docs)
	if err != nil {
		span.RecordError(err)
	}
	return err
}

func (t *tracedVectorStore) DeleteByFilter(ctx context.Context, f filter.Expr) error {
	ctx, span := t.tracer.Start(ctx, "ragy.vectorstore.delete_by_filter")
	defer span.End()
	span.SetAttributes(attribute.String("ragy.store.operation", "delete"))
	err := t.inner.DeleteByFilter(ctx, f)
	if err != nil {
		span.RecordError(err)
	}
	return err
}

// TracedGraphStore wraps a GraphStore with tracing.
func TracedGraphStore(store ragy.GraphStore, tracer trace.Tracer) ragy.GraphStore {
	return &tracedGraphStore{inner: store, tracer: tracer}
}

// NewTracedGraphStore is an alias for TracedGraphStore (TD naming).
func NewTracedGraphStore(store ragy.GraphStore, tracer trace.Tracer) ragy.GraphStore {
	return TracedGraphStore(store, tracer)
}

type tracedGraphStore struct {
	inner  ragy.GraphStore
	tracer trace.Tracer
}

func (t *tracedGraphStore) SearchGraph(ctx context.Context, entities []string, depth int, req ragy.SearchRequest) ([]ragy.Node, []ragy.Edge, error) {
	ctx, span := t.tracer.Start(ctx, "ragy.graphstore.search_graph")
	defer span.End()
	nodes, edges, err := t.inner.SearchGraph(ctx, entities, depth, req)
	span.SetAttributes(
		attribute.Int("ragy.graph.entities", len(entities)),
		attribute.Int("ragy.graph.depth", depth),
		attribute.Int("ragy.graph.nodes_count", len(nodes)),
		attribute.Int("ragy.graph.edges_count", len(edges)),
	)
	if err != nil {
		span.RecordError(err)
	}
	return nodes, edges, err
}

func (t *tracedGraphStore) UpsertGraph(ctx context.Context, nodes []ragy.Node, edges []ragy.Edge) error {
	ctx, span := t.tracer.Start(ctx, "ragy.graphstore.upsert_graph")
	defer span.End()
	err := t.inner.UpsertGraph(ctx, nodes, edges)
	if err != nil {
		span.RecordError(err)
	}
	return err
}

// TracedReranker wraps a Reranker with tracing.
func TracedReranker(reranker ragy.Reranker, tracer trace.Tracer) ragy.Reranker {
	return &tracedReranker{inner: reranker, tracer: tracer}
}

// NewTracedReranker is an alias for TracedReranker (TD naming).
func NewTracedReranker(reranker ragy.Reranker, tracer trace.Tracer) ragy.Reranker {
	return TracedReranker(reranker, tracer)
}

type tracedReranker struct {
	inner  ragy.Reranker
	tracer trace.Tracer
}

func (t *tracedReranker) Rerank(ctx context.Context, query string, docs []ragy.Document, topK int) ([]ragy.Document, error) {
	ctx, span := t.tracer.Start(ctx, "ragy.reranker.rerank")
	defer span.End()
	start := time.Now()
	out, err := t.inner.Rerank(ctx, query, docs, topK)
	dur := time.Since(start)
	span.SetAttributes(
		attribute.Int("ragy.rerank.input_count", len(docs)),
		attribute.Int("ragy.rerank.output_count", len(out)),
		attribute.Int64("ragy.rerank.duration_ms", dur.Milliseconds()),
	)
	if err != nil {
		span.RecordError(err)
	}
	return out, err
}
