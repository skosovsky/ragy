package main

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/retrieval"
)

func TestPartialFailureAggregate_ReturnsPartialWithSiblingHit(t *testing.T) {
	pipeline, err := buildPipeline()
	if err != nil {
		t.Fatalf("buildPipeline(): %v", err)
	}

	rs, err := pipeline.Execute(context.Background(), retrieval.Query[stubIntent]{
		Text:    "q",
		Options: retrieval.RetrieveOptions{TopK: exampleTopK},
	})
	if err == nil {
		t.Fatal("Retrieve() error = nil, want partial failure")
	}
	partial, ok := retrieval.AsPartialFailure[struct{}](err)
	if !ok {
		t.Fatalf("Retrieve() error = %v, want PartialFailureError", err)
	}
	if len(partial.Errors) == 0 {
		t.Fatal("partial.Errors empty, want at least one branch error")
	}
	if rs.Len() != 1 || rs.Documents()[0].ID != "ok" {
		t.Fatalf("Documents() = %#v, want single doc id=ok", rs.Documents())
	}
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("error chain missing ErrUnavailable: %v", err)
	}
}
