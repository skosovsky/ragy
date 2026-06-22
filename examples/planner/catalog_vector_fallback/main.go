// Command catalog_vector_fallback demonstrates aggregate + conditional web routing.
// For PartialFailureError handling see examples/planner/partial_failure_aggregate.
package main

import (
	"context"
	"fmt"

	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
)

type searchIntent struct {
	AllowWeb bool
}

type memoryBackend struct {
	name filter.Schema
	hits []retrieval.Document[struct{}]
}

func (m memoryBackend) Schema() filter.Schema { return m.name }

func (m memoryBackend) Retrieve(
	_ context.Context,
	_ retrieval.Query[searchIntent],
) (retrieval.ResultSet[struct{}], error) {
	return retrieval.NewResultSet(m.hits, retrieval.DocumentIDResolver[struct{}]{}), nil
}

const (
	webFallbackScore = 0.8
	exampleTopK      = 5
)

func buildPipeline(
	catalog,
	vector,
	web memoryBackend,
) (*retrieval.ExecutionPipeline[searchIntent, struct{}, retrieval.NoExecutionMeta], error) {
	return retrieval.NewExecutionPipelineBuilder[searchIntent, struct{}, retrieval.NoExecutionMeta]().
		WithRoot(retrieval.FallbackNode[searchIntent, struct{}, retrieval.NoExecutionMeta]{
			Primary: retrieval.AggregateNode[searchIntent, struct{}, retrieval.NoExecutionMeta]{
				Nodes: []retrieval.ExecutionNode[searchIntent, struct{}, retrieval.NoExecutionMeta]{
					retrieval.BackendNode[searchIntent, struct{}, retrieval.NoExecutionMeta]{Backend: catalog},
					retrieval.ConditionalNode[searchIntent, struct{}, retrieval.NoExecutionMeta]{
						Predicate: func(query retrieval.Query[searchIntent]) bool {
							return len(query.Options.Vector) > 0
						},
						Child: retrieval.BackendNode[searchIntent, struct{}, retrieval.NoExecutionMeta]{
							Backend: vector,
						},
					},
				},
			},
			Secondary: retrieval.ConditionalNode[searchIntent, struct{}, retrieval.NoExecutionMeta]{
				Predicate: func(query retrieval.Query[searchIntent]) bool {
					return query.Intent.AllowWeb
				},
				Child: retrieval.BackendNode[searchIntent, struct{}, retrieval.NoExecutionMeta]{Backend: web},
			},
		}).
		Build()
}

func main() {
	schema, err := filter.NewSchema().Build()
	if err != nil {
		panic(err)
	}

	catalog := memoryBackend{name: schema}
	vector := memoryBackend{name: schema}
	web := memoryBackend{
		name: schema,
		hits: []retrieval.Document[struct{}]{
			{ID: "web-1", Content: "web", Score: webFallbackScore},
		},
	}

	pipeline, err := buildPipeline(catalog, vector, web)
	if err != nil {
		panic(err)
	}

	result, err := pipeline.Execute(context.Background(), retrieval.Query[searchIntent]{
		Text:    "query",
		Intent:  searchIntent{AllowWeb: true},
		Options: retrieval.RetrieveOptions{TopK: exampleTopK},
	})
	if _, ok := retrieval.AsPartialFailure[struct{}](err); ok {
		fmt.Printf("partial failure with %d docs\n", result.Len())
	} else if err != nil {
		panic(err)
	}
	docs := result.Documents()
	if len(docs) != 1 || docs[0].ID != "web-1" {
		panic(fmt.Sprintf("expected web fallback, got %#v", docs))
	}
	fmt.Printf("hit: %s\n", docs[0].ID)
}
