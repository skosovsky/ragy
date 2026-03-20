module github.com/skosovsky/ragy/adapters/observability/otel

go 1.26.0

require (
	github.com/skosovsky/ragy v0.0.0
	go.opentelemetry.io/otel v1.32.0
	go.opentelemetry.io/otel/trace v1.32.0
)

replace github.com/skosovsky/ragy => ../../..
