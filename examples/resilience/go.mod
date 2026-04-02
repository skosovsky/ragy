module github.com/skosovsky/ragy/examples/resilience

go 1.26.1

replace github.com/skosovsky/ragy => ../..

replace github.com/skosovsky/ragy/adapters/openai => ../../adapters/openai

require (
	github.com/skosovsky/ragy v0.0.0
	github.com/skosovsky/ragy/adapters/openai v0.0.0
)
