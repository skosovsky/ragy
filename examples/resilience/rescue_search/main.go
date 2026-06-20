// Command rescue_search demonstrates orchestrator RescueNode when primary returns an error.
// For empty secondary success, RescueNode propagates the primary error (see task9 §3).
package main

import (
	"context"
	"fmt"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
)

type intent struct{}

const (
	rescueScore = 0.9
	defaultTopK = 10
)

type stubBackend struct {
	schema filter.Schema
	fail   bool
	docs   []retrieval.Document[struct{}]
}

func (s *stubBackend) Schema() filter.Schema { return s.schema }

func (s *stubBackend) Retrieve(
	_ context.Context,
	_ retrieval.Query[intent],
) (retrieval.ResultSet[struct{}], error) {
	if s.fail {
		return retrieval.NewResultSet[struct{}](nil, retrieval.DocumentIDResolver[struct{}]{}),
			fmt.Errorf("%w: primary vector store down", ragy.ErrUnavailable)
	}
	return retrieval.NewResultSet(s.docs, retrieval.DocumentIDResolver[struct{}]{}), nil
}

func main() {
	schema, err := filter.NewSchema().Build()
	if err != nil {
		panic(err)
	}

	primary := &stubBackend{schema: schema, fail: true}
	secondary := &stubBackend{
		schema: schema,
		docs:   []retrieval.Document[struct{}]{{ID: "fb-1", Content: "rescue hit", Score: rescueScore}},
	}

	pipeline, err := retrieval.NewPipelineBuilder[intent, struct{}]().
		WithRescue(
			retrieval.RetrieverNode[intent, struct{}]{Backend: primary},
			retrieval.RetrieverNode[intent, struct{}]{Backend: secondary},
		).
		Build()
	if err != nil {
		panic(err)
	}

	rs, err := pipeline.Retrieve(context.Background(), retrieval.Query[intent]{
		Text: "hello",
		Options: retrieval.RetrieveOptions{
			TopK:   defaultTopK,
			Vector: []float32{1, 0, 0},
		},
	})
	if err != nil {
		panic(err)
	}
	docs := rs.Documents()
	if len(docs) != 1 || docs[0].ID != "fb-1" {
		panic("expected rescue document")
	}
	fmt.Printf("ok: %q score=%.2f\n", docs[0].Content, docs[0].Score)
}
