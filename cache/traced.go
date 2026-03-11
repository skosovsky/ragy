package cache

import (
	"context"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

// tracedSemanticCache wraps a SemanticCache with OpenTelemetry tracing.
type tracedSemanticCache struct {
	inner  SemanticCache
	tracer trace.Tracer
}

// NewTracedSemanticCache returns a SemanticCache that records spans for Get and Set.
func NewTracedSemanticCache(c SemanticCache, tracer trace.Tracer) SemanticCache {
	return &tracedSemanticCache{inner: c, tracer: tracer}
}

// Get implements SemanticCache and records span "ragy.cache.get".
func (t *tracedSemanticCache) Get(ctx context.Context, query string, threshold float64) (string, bool, error) {
	ctx, span := t.tracer.Start(ctx, "ragy.cache.get")
	defer span.End()
	start := time.Now()
	response, hit, err := t.inner.Get(ctx, query, threshold)
	dur := time.Since(start)

	span.SetAttributes(
		attribute.String("ragy.cache.query", query),
		attribute.Float64("ragy.cache.threshold", threshold),
		attribute.Bool("ragy.cache.hit", hit),
		attribute.Int64("ragy.duration_ms", dur.Milliseconds()),
	)
	if err != nil {
		span.RecordError(err)
	}
	return response, hit, err
}

// Set implements SemanticCache and records span "ragy.cache.set".
func (t *tracedSemanticCache) Set(ctx context.Context, query string, response string) error {
	ctx, span := t.tracer.Start(ctx, "ragy.cache.set")
	defer span.End()
	start := time.Now()
	err := t.inner.Set(ctx, query, response)
	dur := time.Since(start)

	span.SetAttributes(
		attribute.String("ragy.cache.query", query),
		attribute.Int("ragy.cache.response_length", len(response)),
		attribute.Int64("ragy.duration_ms", dur.Milliseconds()),
	)
	if err != nil {
		span.RecordError(err)
	}
	return err
}
