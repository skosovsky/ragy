package otel

import (
	"context"
	"math"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/observation"
)

// Observer exports completed core diagnostics as spans without retaining starts.
// Each session can reuse this concurrency-safe exporter: numeric ordinals remain
// session-local attributes, never a global lookup key or a metric label.
// Start events are not exported independently. Completion uses the locally
// observed elapsed duration; missing completion is not fabricated on shutdown.
type Observer struct{ tracer trace.Tracer }

// NewObserver constructs a payload-free diagnostic exporter. Use observation.New
// to explicitly bound callback count and observation.WithSession to enable it.
func NewObserver(tracer trace.Tracer) (*Observer, error) {
	if tracer == nil {
		return nil, ragy.ErrInvalidArgument
	}
	return &Observer{tracer: tracer}, nil
}

// Observe implements observation.Observer. Exporter failures and panics are
// isolated by the core session; this method never dispatches an operation.
func (o *Observer) Observe(ctx context.Context, event observation.Event) error {
	if event.Kind != observation.KindEnd {
		return nil
	}
	ended := time.Now()
	elapsed := max(event.Elapsed, 0)
	_, span := o.tracer.Start(ctx, stageName(event.Stage), trace.WithTimestamp(ended.Add(-elapsed)))
	defer span.End(trace.WithTimestamp(ended))
	span.SetAttributes(
		attribute.Int("ragy.stage", int(event.Stage)),
		boundedInteger("ragy.operation.ordinal", event.Operation),
		boundedInteger("ragy.parent.ordinal", event.Parent),

		attribute.Int("ragy.outcome", int(event.Completion.Outcome)),
		attribute.Int("ragy.error.class", int(event.Completion.Error)),
	)
	putCount(span, "ragy.query.ordinal", event.Query)
	putCount(span, "ragy.branch.ordinal", event.Branch)
	putCount(span, "ragy.result.count", event.Completion.Count)
	putCount(span, "ragy.usage.input_tokens", event.Completion.Usage.InputTokens)
	putCount(span, "ragy.usage.output_tokens", event.Completion.Usage.OutputTokens)
	putCount(span, "ragy.usage.billed_units", event.Completion.Usage.BilledUnits)
	// Generic model stages do not reveal whether a port is chat or another
	// operation, so no GenAI inference operation or provider identity is guessed.
	if event.Stage == observation.StageEncoding {
		span.SetAttributes(attribute.String("gen_ai.operation.name", "embeddings"))
		if event.Completion.Usage.InputTokens.Known {
			span.SetAttributes(boundedInteger("gen_ai.usage.input_tokens", event.Completion.Usage.InputTokens.Value))
		}
	}
	if event.Completion.Error != observation.ErrorNone ||
		event.Completion.Outcome == observation.OutcomeFailed ||
		event.Completion.Outcome == observation.OutcomeCanceled ||
		event.Completion.Outcome == observation.OutcomeUnsupported ||
		event.Completion.Outcome == observation.OutcomeExhausted {
		span.SetStatus(codes.Error, errorName(event.Completion.Error))
		span.SetAttributes(attribute.String("error.type", errorName(event.Completion.Error)))
	}
	return nil
}

func boundedInteger(name string, value uint64) attribute.KeyValue {
	if value > math.MaxInt64 {
		value = math.MaxInt64
	}
	return attribute.Int64(name, int64(value))
}
func putCount(span trace.Span, name string, count observation.Count) {
	span.SetAttributes(attribute.Bool(name+".known", count.Known))
	if count.Known {
		span.SetAttributes(
			boundedInteger(name, count.Value),
			attribute.Bool(name+".saturated", count.Value > math.MaxInt64),
		)
	}
}
func stageName(stage observation.Stage) string {
	names := [...]string{
		"ragy.unknown", "ragy.pipeline", "ragy.retrieval", "ragy.fallback", "ragy.rescue",
		"ragy.route", "ragy.conditional", "ragy.aggregate", "ragy.cache", "ragy.cache.hit",
		"ragy.cache.miss", "ragy.plan", "ragy.assess", "ragy.encoding", "ragy.model", "ragy.fusion",
		"ragy.delivery", "ragy.summary.map", "ragy.summary.reduce", "ragy.lifecycle",
		"ragy.lifecycle.stage", "ragy.lifecycle.publish", "ragy.lifecycle.cleanup",
		"ragy.lifecycle.prepare", "ragy.lifecycle.reconcile", "ragy.lifecycle.inspect",
		"ragy.lifecycle.cleanup.begin", "ragy.lifecycle.cleanup.inspect", "ragy.lifecycle.cleanup.reconcile",
		"ragy.lifecycle.reuse", "ragy.lifecycle.bootstrap",
	}
	if int(stage) >= len(names) {
		return "ragy.unknown"
	}
	return names[stage]
}
