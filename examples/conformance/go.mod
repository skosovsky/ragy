module example.com/ragyconsumer

go 1.27.1

require (
	github.com/skosovsky/ragy v0.0.0
	github.com/skosovsky/ragy/adapters/observability/otel v0.0.0
	github.com/skosovsky/ragy/adapters/openai v0.0.0
	go.opentelemetry.io/otel/trace v1.47.0
)

require (
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	go.opentelemetry.io/otel v1.47.0 // indirect
)

replace github.com/skosovsky/ragy => ../..

replace github.com/skosovsky/ragy/adapters/openai => ../../adapters/openai

replace github.com/skosovsky/ragy/adapters/observability/otel => ../../adapters/observability/otel
