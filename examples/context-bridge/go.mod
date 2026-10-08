module github.com/skosovsky/ragy/examples/context-bridge

go 1.27.1

require (
	github.com/santhosh-tekuri/jsonschema/v6 v6.0.3
	github.com/skosovsky/contexty v0.13.1
	github.com/skosovsky/memy v0.3.1
	github.com/skosovsky/ragy v0.8.0
)

require golang.org/x/text v0.42.0 // indirect

replace github.com/skosovsky/ragy => ../..
