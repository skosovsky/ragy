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

func buildPipeline() (*retrieval.ExecutionPipeline[stubIntent, struct{}, retrieval.NoExecutionMeta], error) {
	return retrieval.NewExecutionPipelineBuilder[stubIntent, struct{}, retrieval.NoExecutionMeta]().
		WithRoot(retrieval.AggregateNode[stubIntent, struct{}, retrieval.NoExecutionMeta]{
			Nodes: []retrieval.ExecutionNode[stubIntent, struct{}, retrieval.NoExecutionMeta]{
				retrieval.BackendNode[stubIntent, struct{}, retrieval.NoExecutionMeta]{Backend: errorNode{}},
				retrieval.BackendNode[stubIntent, struct{}, retrieval.NoExecutionMeta]{Backend: hitNode{}},
			},
		}).
		Build()
}

func main() {
	pipeline, err := buildPipeline()
	if err != nil {
		panic(err)
	}

	result, err := pipeline.Execute(context.Background(), retrieval.Query[stubIntent]{
		Text:    "q",
		Options: retrieval.RetrieveOptions{TopK: exampleTopK},
	})
	if err == nil {
		panic("expected partial failure")
	}

	if partial, ok := retrieval.AsPartialFailure[struct{}](err); ok {
		fmt.Printf("partial errors: %d, returned result len: %d\n", len(partial.Errors), result.Len())
	}

	if result.IsEmpty() {
		panic("expected non-empty ResultSet despite partial failure")
	}
	fmt.Printf("returned docs: %d (id=%s)\n", result.Len(), result.Documents()[0].ID)

	if !errors.Is(err, ragy.ErrUnavailable) {
		panic(fmt.Sprintf("expected unavailable in partial error chain, got %v", err))
	}
}
