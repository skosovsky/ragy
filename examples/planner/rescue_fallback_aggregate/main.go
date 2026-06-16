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
	_ string,
	_ retrieval.RetrieveOptions,
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

func webIfAllowed(web memoryBackend) retrieval.Node[searchIntent, struct{}] {
	return retrieval.ConditionalNode[searchIntent, struct{}]{
		Predicate: func(query retrieval.Query[searchIntent]) bool {
			return query.Intent.AllowWeb
		},
		Child: retrieval.RetrieverNode[searchIntent, struct{}]{Backend: web},
	}
}

func buildPipeline(catalog, vector, web memoryBackend) (*retrieval.Pipeline[searchIntent, struct{}], error) {
	return retrieval.NewPipelineBuilder[searchIntent, struct{}]().
		WithRoot(retrieval.RescueNode[searchIntent, struct{}]{
			Primary: retrieval.FallbackNode[searchIntent, struct{}]{
				Primary: retrieval.AggregateNode[searchIntent, struct{}]{
					Nodes: []retrieval.Node[searchIntent, struct{}]{
						retrieval.RetrieverNode[searchIntent, struct{}]{Backend: catalog},
						retrieval.RetrieverNode[searchIntent, struct{}]{Backend: vector},
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

	rs, err := pipeline.Retrieve(context.Background(), retrieval.Query[searchIntent]{
		Text:    "query",
		Intent:  searchIntent{AllowWeb: true},
		Options: retrieval.RetrieveOptions{TopK: exampleTopK},
	})
	if err != nil {
		panic(err)
	}
	docs := rs.Documents()
	if len(docs) != 1 || docs[0].ID != "web-1" {
		panic(fmt.Sprintf("expected web rescue hit, got %#v", docs))
	}
	fmt.Printf("hit: %s\n", docs[0].ID)
}
