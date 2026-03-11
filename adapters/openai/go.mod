module github.com/skosovsky/ragy/adapters/openai

go 1.26.0

require (
	github.com/sashabaranov/go-openai v1.41.2
	github.com/skosovsky/ragy v0.0.0
	github.com/stretchr/testify v1.11.1
)

require (
	github.com/davecgh/go-spew v1.1.1 // indirect
	github.com/pmezard/go-difflib v1.0.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)

replace github.com/skosovsky/ragy => ../..
