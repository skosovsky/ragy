package otel

import (
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/skosovsky/ragy/embedding"
	"github.com/skosovsky/ragy/observation"
	"github.com/skosovsky/ragy/retrieval"
)

func resultCount[T any](result retrieval.ResultSet[T]) int {
	if result == nil {
		return -1
	}
	return result.Len()
}

func recordOutcome(span trace.Span, err error, count int) {
	observed := observation.Count{Known: false, Value: 0}
	if count >= 0 && (err == nil || count > 0) {
		observed = observation.Count{Known: true, Value: uint64(count)}
	}
	completion := observation.Finish(err, observed)
	putCount(span, "ragy.result.count", observed)
	span.SetAttributes(
		attribute.Int("ragy.outcome", int(completion.Outcome)),
		attribute.Int("ragy.error.class", int(completion.Error)),
	)
	if err != nil {
		class := errorName(completion.Error)
		span.SetAttributes(attribute.String("error.type", class))
		span.SetStatus(codes.Error, class)
	}
}
func errorName(class observation.ErrorClass) string {
	switch class {
	case observation.ErrorNone, observation.ErrorUnknown:
		return "operation_failed"
	case observation.ErrorCanceled:
		return "canceled"
	case observation.ErrorDeadline:
		return "deadline"
	case observation.ErrorUnsupported:
		return "unsupported"
	case observation.ErrorInvalid:
		return "invalid_argument"
	case observation.ErrorProtocol:
		return "protocol"
	case observation.ErrorUnavailable:
		return "unavailable"
	case observation.ErrorProtection:
		return "protection"
	case observation.ErrorResource:
		return "resource"
	default:
		return "operation_failed"
	}
}

func recordEncoding(span trace.Span, count int, usage embedding.Usage, err error) {
	recordOutcome(span, err, count)
	span.SetAttributes(attribute.String("gen_ai.operation.name", "embeddings"))
	if usage.Validate() != nil {
		span.SetAttributes(
			attribute.Bool("ragy.usage.input_tokens.known", false),
			attribute.Bool("ragy.usage.billed_units.known", false),
		)
		return
	}
	span.SetAttributes(attribute.Bool("ragy.usage.input_tokens.known", usage.InputTokensKnown),
		attribute.Bool("ragy.usage.billed_units.known", usage.BilledUnitsKnown))
	if usage.InputTokensKnown {
		span.SetAttributes(attribute.Int64("gen_ai.usage.input_tokens", usage.InputTokens))
	}
	if usage.BilledUnitsKnown {
		span.SetAttributes(attribute.Int64("ragy.usage.billed_units", usage.BilledUnits))
	}
}
