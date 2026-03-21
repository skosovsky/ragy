module github.com/skosovsky/ragy/adapters/neo4j

go 1.26.1

require (
	github.com/neo4j/neo4j-go-driver/v5 v5.26.0
	github.com/skosovsky/ragy v0.0.0
	github.com/stretchr/testify v1.11.1
)

require (
	github.com/davecgh/go-spew v1.1.2-0.20180830191138-d8f796af33cc // indirect
	github.com/kr/text v0.2.0 // indirect
	github.com/pmezard/go-difflib v1.0.1-0.20181226105442-5d4384ee4fb2 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)

replace github.com/skosovsky/ragy => ../..
