package otel

import (
	"context"
	"iter"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/skosovsky/ragy"
)

// WrapRetriever returns a ragy.Retriever that records spans for Retrieve and Stream.
func WrapRetriever(inner ragy.Retriever, tracer trace.Tracer, name string) ragy.Retriever {
	if name == "" {
		name = "ragy.Retriever"
	}
	return &tracedRetriever{inner: inner, tracer: tracer, name: name}
}

type tracedRetriever struct {
	inner  ragy.Retriever
	tracer trace.Tracer
	name   string
}

func (t *tracedRetriever) Retrieve(ctx context.Context, req ragy.SearchRequest) ([]ragy.Document, error) {
	ctx, span := t.tracer.Start(ctx, t.name+".Retrieve")
	defer span.End()
	docs, err := t.inner.Retrieve(ctx, req)
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, err.Error())
		return nil, err
	}
	span.SetAttributes(attribute.Int("ragy.retrieve.doc_count", len(docs)))
	return docs, nil
}

func (t *tracedRetriever) Stream(ctx context.Context, req ragy.SearchRequest) iter.Seq2[ragy.Document, error] {
	ctx, span := t.tracer.Start(ctx, t.name+".Stream")
	return func(yield func(ragy.Document, error) bool) {
		defer span.End()
		n := 0
		for doc, err := range t.inner.Stream(ctx, req) {
			if err != nil {
				span.RecordError(err)
				span.SetStatus(codes.Error, err.Error())
				yield(doc, err)
				return
			}
			n++
			if !yield(doc, nil) {
				return
			}
		}
		span.SetAttributes(attribute.Int("ragy.stream.doc_count", n))
	}
}
