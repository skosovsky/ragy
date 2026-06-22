package main

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
)

func TestRescueSearch_EmptySecondaryPropagatesPrimaryError(t *testing.T) {
	schema, err := filter.NewSchema().Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	primary := &stubBackend{schema: schema, fail: true}
	secondary := &stubBackend{schema: schema}

	pipeline, err := retrieval.NewExecutionPipelineBuilder[intent, struct{}, retrieval.NoExecutionMeta]().
		WithRoot(retrieval.RescueNode[intent, struct{}, retrieval.NoExecutionMeta]{
			Primary:   retrieval.BackendNode[intent, struct{}, retrieval.NoExecutionMeta]{Backend: primary},
			Secondary: retrieval.BackendNode[intent, struct{}, retrieval.NoExecutionMeta]{Backend: secondary},
		}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	result, err := pipeline.Execute(context.Background(), retrieval.Query[intent]{
		Text: "hello",
		Options: retrieval.RetrieveOptions{
			TopK:   defaultTopK,
			Vector: []float32{1, 0, 0},
		},
	})
	if err == nil {
		t.Fatal("Retrieve() error = nil, want primary error")
	}
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("Retrieve() error = %v, want unavailable", err)
	}
	if !result.IsEmpty() {
		t.Fatalf("Documents() = %#v, want empty", result.Documents())
	}
}
