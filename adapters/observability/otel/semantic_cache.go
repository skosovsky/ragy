package otel

import (
	"context"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/skosovsky/ragy/cache"
)

// WrapSemanticCache returns a cache.SemanticCache that records spans for Get and Set.
func WrapSemanticCache(inner cache.SemanticCache, tracer trace.Tracer, name string) cache.SemanticCache {
	if name == "" {
		name = "ragy.SemanticCache"
	}
	return &tracedSemanticCache{inner: inner, tracer: tracer, name: name}
}

type tracedSemanticCache struct {
	inner  cache.SemanticCache
	tracer trace.Tracer
	name   string
}

func (t *tracedSemanticCache) Get(ctx context.Context, query string, threshold float64) (string, bool, error) {
	ctx, span := t.tracer.Start(ctx, t.name+".Get")
	defer span.End()
	resp, hit, err := t.inner.Get(ctx, query, threshold)
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, err.Error())
		return resp, hit, err
	}
	span.SetAttributes(attribute.Bool("ragy.cache.hit", hit))
	return resp, hit, nil
}

func (t *tracedSemanticCache) Set(ctx context.Context, query string, response string) error {
	ctx, span := t.tracer.Start(ctx, t.name+".Set")
	defer span.End()
	err := t.inner.Set(ctx, query, response)
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, err.Error())
	}
	return err
}
