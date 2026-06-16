// Command partial_failure_aggregate demonstrates PartialFailureError handling.
package main

import (
	"context"
	"errors"
	"fmt"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/retrieval"
)

const exampleTopK = 5

type stubIntent struct{}

type errorNode struct{}

func (errorNode) Retrieve(
	_ context.Context,
	_ retrieval.Query[stubIntent],
) (retrieval.ResultSet[struct{}], error) {
	return retrieval.NewResultSet[struct{}](nil, retrieval.DocumentIDResolver[struct{}]{}), ragy.ErrUnavailable
}

type hitNode struct{}

func (hitNode) Retrieve(
	_ context.Context,
	_ retrieval.Query[stubIntent],
) (retrieval.ResultSet[struct{}], error) {
	return retrieval.NewResultSet([]retrieval.Document[struct{}]{
		{ID: "ok", Content: "hit", Score: 1},
	}, retrieval.DocumentIDResolver[struct{}]{}), nil
}

func buildPipeline() (*retrieval.Pipeline[stubIntent, struct{}], error) {
	return retrieval.NewPipelineBuilder[stubIntent, struct{}]().
		WithRoot(retrieval.AggregateNode[stubIntent, struct{}]{
			Nodes: []retrieval.Node[stubIntent, struct{}]{
				errorNode{},
				hitNode{},
			},
		}).
		Build()
}

func main() {
	pipeline, err := buildPipeline()
	if err != nil {
		panic(err)
	}

	rs, err := pipeline.Retrieve(context.Background(), retrieval.Query[stubIntent]{
		Text:    "q",
		Options: retrieval.RetrieveOptions{TopK: exampleTopK},
	})
	if err == nil {
		panic("expected partial failure")
	}

	if partial, ok := retrieval.AsPartialFailure[struct{}](err); ok {
		fmt.Printf("partial errors: %d, returned result len: %d\n", len(partial.Errors), rs.Len())
	}

	if rs.IsEmpty() {
		panic("expected non-empty ResultSet despite partial failure")
	}
	fmt.Printf("returned docs: %d (id=%s)\n", rs.Len(), rs.Documents()[0].ID)

	if !errors.Is(err, ragy.ErrUnavailable) {
		panic(fmt.Sprintf("expected unavailable in partial error chain, got %v", err))
	}
}
