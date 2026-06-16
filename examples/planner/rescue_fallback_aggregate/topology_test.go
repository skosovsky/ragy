package main

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
)

func testSchema(t *testing.T) filter.Schema {
	t.Helper()

	schema, err := filter.NewSchema().Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}
	return schema
}

func TestRescueFallbackAggregate_AllowWebTrue_VectorOutage(t *testing.T) {
	schema := testSchema(t)
	web := memoryBackend{
		name: schema,
		hits: []retrieval.Document[struct{}]{{ID: "web-1", Content: "web", Score: webFallbackScore}},
	}
	pipeline, err := buildPipeline(
		memoryBackend{name: schema},
		memoryBackend{name: schema, fail: true},
		web,
	)
	if err != nil {
		t.Fatalf("buildPipeline(): %v", err)
	}

	rs, err := pipeline.Retrieve(context.Background(), retrieval.Query[searchIntent]{
		Text:    "query",
		Intent:  searchIntent{AllowWeb: true},
		Options: retrieval.RetrieveOptions{TopK: exampleTopK},
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.IsEmpty() || rs.Documents()[0].ID != "web-1" {
		t.Fatalf("Documents() = %#v, want web-1", rs.Documents())
	}
}

func TestRescueFallbackAggregate_AllowWebFalse_VectorOutage(t *testing.T) {
	schema := testSchema(t)
	web := memoryBackend{
		name: schema,
		hits: []retrieval.Document[struct{}]{{ID: "web-1", Content: "web", Score: webFallbackScore}},
	}
	pipeline, err := buildPipeline(
		memoryBackend{name: schema},
		memoryBackend{name: schema, fail: true},
		web,
	)
	if err != nil {
		t.Fatalf("buildPipeline(): %v", err)
	}

	rs, err := pipeline.Retrieve(context.Background(), retrieval.Query[searchIntent]{
		Text:    "query",
		Intent:  searchIntent{AllowWeb: false},
		Options: retrieval.RetrieveOptions{TopK: exampleTopK},
	})
	if !rs.IsEmpty() {
		t.Fatalf("Documents() = %#v, want empty when AllowWeb false", rs.Documents())
	}
	for _, doc := range rs.Documents() {
		if doc.ID == "web-1" {
			t.Fatalf("web branch ran with AllowWeb=false")
		}
	}
	if err != nil && !errors.Is(err, ragy.ErrUnavailable) {
		var partial *retrieval.PartialFailureError[struct{}]
		if !errors.As(err, &partial) {
			t.Fatalf("Retrieve() error = %v, want unavailable or partial failure", err)
		}
	}
}

func TestRescueFallbackAggregate_AllowWebFalse_SparseEmpty(t *testing.T) {
	schema := testSchema(t)
	web := memoryBackend{
		name: schema,
		hits: []retrieval.Document[struct{}]{{ID: "web-1", Content: "web", Score: webFallbackScore}},
	}
	pipeline, err := buildPipeline(
		memoryBackend{name: schema},
		memoryBackend{name: schema},
		web,
	)
	if err != nil {
		t.Fatalf("buildPipeline(): %v", err)
	}

	rs, err := pipeline.Retrieve(context.Background(), retrieval.Query[searchIntent]{
		Text:    "query",
		Intent:  searchIntent{AllowWeb: false},
		Options: retrieval.RetrieveOptions{TopK: exampleTopK},
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if !rs.IsEmpty() {
		t.Fatalf("Documents() = %#v, want empty sparse recall without web", rs.Documents())
	}
}
