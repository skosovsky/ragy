package main

import (
	"context"
	"testing"

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

func TestCatalogVectorFallback_AllowWebTrue_SparseEmpty(t *testing.T) {
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

func TestCatalogVectorFallback_AllowWebFalse_SparseEmpty(t *testing.T) {
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
		t.Fatalf("Documents() = %#v, want empty when AllowWeb false", rs.Documents())
	}
}

func TestCatalogVectorFallback_VectorBranchReturnsVectorHit(t *testing.T) {
	schema := testSchema(t)
	vector := memoryBackend{
		name: schema,
		hits: []retrieval.Document[struct{}]{
			{ID: "vec-1", Content: "dense", Score: 0.9},
		},
	}
	web := memoryBackend{
		name: schema,
		hits: []retrieval.Document[struct{}]{
			{ID: "web-1", Content: "web", Score: webFallbackScore},
		},
	}
	pipeline, err := buildPipeline(
		memoryBackend{name: schema},
		vector,
		web,
	)
	if err != nil {
		t.Fatalf("buildPipeline(): %v", err)
	}

	rs, err := pipeline.Retrieve(context.Background(), retrieval.Query[searchIntent]{
		Text:   "query",
		Intent: searchIntent{AllowWeb: true},
		Options: retrieval.RetrieveOptions{
			TopK:   exampleTopK,
			Vector: []float32{0.1},
		},
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.IsEmpty() || rs.Documents()[0].ID != "vec-1" {
		t.Fatalf("Documents() = %#v, want vec-1 (web fallback must not run)", rs.Documents())
	}
}
