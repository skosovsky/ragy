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

	pipeline, err := retrieval.NewPipelineBuilder[intent, struct{}]().
		WithRescue(
			retrieval.RetrieverNode[intent, struct{}]{Backend: primary},
			retrieval.RetrieverNode[intent, struct{}]{Backend: secondary},
		).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Retrieve(context.Background(), retrieval.Query[intent]{
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
	if !rs.IsEmpty() {
		t.Fatalf("Documents() = %#v, want empty", rs.Documents())
	}
}
