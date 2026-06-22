// Command rescue_fallback_aggregate demonstrates README topology:
// Rescue(Fallback(Aggregate(catalog, vector), Conditional(AllowWeb, web)),
//
//	Conditional(AllowWeb, web)).
//
// Sparse empty recall triggers web fallback; vector outage triggers rescue to web.
package main

import (
	"context"
	"fmt"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
)

type searchIntent struct {
	AllowWeb bool
}

type memoryBackend struct {
	name filter.Schema
	hits []retrieval.Document[struct{}]
	fail bool
}

func (m memoryBackend) Schema() filter.Schema { return m.name }

func (m memoryBackend) Retrieve(
	_ context.Context,
	_ retrieval.Query[searchIntent],
) (retrieval.ResultSet[struct{}], error) {
	if m.fail {
		return retrieval.NewResultSet[struct{}](nil, retrieval.DocumentIDResolver[struct{}]{}),
			fmt.Errorf("%w: backend unavailable", ragy.ErrUnavailable)
	}
	return retrieval.NewResultSet(m.hits, retrieval.DocumentIDResolver[struct{}]{}), nil
}

const (
	webFallbackScore = 0.8
	exampleTopK      = 5
)

func webIfAllowed(web memoryBackend) retrieval.ExecutionNode[searchIntent, struct{}, retrieval.NoExecutionMeta] {
	return retrieval.ConditionalNode[searchIntent, struct{}, retrieval.NoExecutionMeta]{
		Predicate: func(query retrieval.Query[searchIntent]) bool {
			return query.Intent.AllowWeb
		},
		Child: retrieval.BackendNode[searchIntent, struct{}, retrieval.NoExecutionMeta]{Backend: web},
	}
}

func buildPipeline(
	catalog,
	vector,
	web memoryBackend,
) (*retrieval.ExecutionPipeline[searchIntent, struct{}, retrieval.NoExecutionMeta], error) {
	return retrieval.NewExecutionPipelineBuilder[searchIntent, struct{}, retrieval.NoExecutionMeta]().
		WithRoot(retrieval.RescueNode[searchIntent, struct{}, retrieval.NoExecutionMeta]{
			Primary: retrieval.FallbackNode[searchIntent, struct{}, retrieval.NoExecutionMeta]{
				Primary: retrieval.AggregateNode[searchIntent, struct{}, retrieval.NoExecutionMeta]{
					Nodes: []retrieval.ExecutionNode[searchIntent, struct{}, retrieval.NoExecutionMeta]{
						retrieval.BackendNode[searchIntent, struct{}, retrieval.NoExecutionMeta]{Backend: catalog},
						retrieval.BackendNode[searchIntent, struct{}, retrieval.NoExecutionMeta]{Backend: vector},
					},
				},
				Secondary: webIfAllowed(web),
			},
			Secondary: webIfAllowed(web),
		}).
		Build()
}

func main() {
	schema, err := filter.NewSchema().Build()
	if err != nil {
		panic(err)
	}

	catalog := memoryBackend{name: schema}
	vector := memoryBackend{name: schema, fail: true}
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
	if err != nil {
		panic(err)
	}
	docs := result.Documents()
	if len(docs) != 1 || docs[0].ID != "web-1" {
		panic(fmt.Sprintf("expected web rescue hit, got %#v", docs))
	}
	fmt.Printf("hit: %s\n", docs[0].ID)
}
