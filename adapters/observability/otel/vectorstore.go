package otel

import (
	"context"
	"iter"

	"github.com/skosovsky/ragy/filter"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/skosovsky/ragy"
)

// WrapVectorStore returns a ragy.VectorStore that records spans for Search, Stream, Upsert, DeleteByFilter.
func WrapVectorStore(inner ragy.VectorStore, tracer trace.Tracer, name string) ragy.VectorStore {
	if name == "" {
		name = "ragy.VectorStore"
	}
	return &tracedVectorStore{inner: inner, tracer: tracer, name: name}
}

type tracedVectorStore struct {
	inner  ragy.VectorStore
	tracer trace.Tracer
	name   string
}

func (t *tracedVectorStore) Search(ctx context.Context, req ragy.SearchRequest) ([]ragy.Document, error) {
	ctx, span := t.tracer.Start(ctx, t.name+".Search")
	defer span.End()
	docs, err := t.inner.Search(ctx, req)
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, err.Error())
		return nil, err
	}
	return docs, nil
}

func (t *tracedVectorStore) Stream(ctx context.Context, req ragy.SearchRequest) iter.Seq2[ragy.Document, error] {
	_, span := t.tracer.Start(ctx, t.name+".Stream")
	return func(yield func(ragy.Document, error) bool) {
		defer span.End()
		for doc, err := range t.inner.Stream(ctx, req) {
			if err != nil {
				span.RecordError(err)
				span.SetStatus(codes.Error, err.Error())
				yield(doc, err)
				return
			}
			if !yield(doc, nil) {
				return
			}
		}
	}
}

func (t *tracedVectorStore) Upsert(ctx context.Context, docs []ragy.Document) error {
	ctx, span := t.tracer.Start(ctx, t.name+".Upsert")
	defer span.End()
	err := t.inner.Upsert(ctx, docs)
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, err.Error())
	}
	return err
}

func (t *tracedVectorStore) DeleteByFilter(ctx context.Context, f filter.Expr) error {
	ctx, span := t.tracer.Start(ctx, t.name+".DeleteByFilter")
	defer span.End()
	err := t.inner.DeleteByFilter(ctx, f)
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, err.Error())
	}
	return err
}
