ADAPTERS := adapters/observability/otel adapters/openai adapters/gemini adapters/pgvector adapters/jina adapters/qdrant adapters/cohere adapters/elasticsearch adapters/neo4j

.PHONY: test lint bench fuzz cover tidy-all test-all lint-all

# tidy-all runs go mod tidy in root and all adapter modules.
tidy-all:
	@go mod tidy
	@for d in $(ADAPTERS); do (cd "$$d" && go mod tidy); done

# test-all runs tests in root and all adapter modules.
test-all:
	@go test -race -count=1 ./...
	@for d in $(ADAPTERS); do (cd "$$d" && go test -race -count=1 ./...); done

# lint-all runs golangci-lint in root and all adapter modules.
lint-all:
	@golangci-lint run ./...
	@for d in $(ADAPTERS); do (cd "$$d" && golangci-lint run ./...); done

test:
	@go test -race -count=1 ./...

lint:
	@golangci-lint run ./...

bench:
	@go test -bench=. -benchmem ./...

fuzz:
	@go test -fuzz=. -fuzztime=30s .

cover:
	@go test -coverprofile=coverage.out -covermode=atomic ./...
	@go tool cover -func=coverage.out
