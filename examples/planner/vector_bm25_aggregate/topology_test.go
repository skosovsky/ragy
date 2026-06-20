package main

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/lexical"
	"github.com/skosovsky/ragy/retrieval"
)

func testFixtures(t *testing.T) (vectorBackend, retrieval.Backend[struct{}, struct{}]) {
	t.Helper()

	schema, err := filter.NewSchema().Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	bm25, err := lexical.NewBM25Index[struct{}](
		schema,
		lexical.Config[struct{}]{SearchFields: []string{"content"}},
		lexical.DefaultTokenizer{},
		nil,
	)
	if err != nil {
		t.Fatalf("NewBM25Index(): %v", err)
	}
	if upsertErr := bm25.Upsert(retrieval.Document[struct{}]{
		ID: "lex-1", Content: "bm25 keyword match", Score: 0,
	}); upsertErr != nil {
		t.Fatalf("Upsert(): %v", upsertErr)
	}

	vector := vectorBackend{
		schema: schema,
		hits: []retrieval.Document[struct{}]{
			{ID: "vec-1", Content: "dense neighbor", Score: vectorHitScore},
		},
	}
	return vector, bm25
}

func TestVectorBM25Aggregate_FusesBothBranches(t *testing.T) {
	vector, bm25 := testFixtures(t)
	pipeline, err := buildPipeline(vector, bm25)
	if err != nil {
		t.Fatalf("buildPipeline(): %v", err)
	}

	rs, err := pipeline.Retrieve(context.Background(), retrieval.Query[searchIntent]{
		Text:    "keyword",
		Options: retrieval.RetrieveOptions{TopK: exampleTopK, Vector: []float32{0.1, 0.2}},
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Len() < 2 {
		t.Fatalf("Len() = %d, want at least 2 fused docs", rs.Len())
	}
	ids := make(map[string]struct{}, rs.Len())
	for _, doc := range rs.Documents() {
		ids[doc.ID] = struct{}{}
	}
	if _, ok := ids["vec-1"]; !ok {
		t.Fatalf("Documents() = %#v, want vec-1", rs.Documents())
	}
	if _, ok := ids["lex-1"]; !ok {
		t.Fatalf("Documents() = %#v, want lex-1", rs.Documents())
	}
}

func TestVectorBM25Aggregate_EmptyVectorStillReturnsLexical(t *testing.T) {
	vector, bm25 := testFixtures(t)
	pipeline, err := buildPipeline(vector, bm25)
	if err != nil {
		t.Fatalf("buildPipeline(): %v", err)
	}

	rs, err := pipeline.Retrieve(context.Background(), retrieval.Query[searchIntent]{
		Text:    "keyword",
		Options: retrieval.RetrieveOptions{TopK: exampleTopK},
	})
	if err == nil {
		t.Fatal("Retrieve() error = nil, want partial failure")
	}
	if _, ok := retrieval.AsPartialFailure[struct{}](err); !ok {
		t.Fatalf("Retrieve() error = %v, want PartialFailureError", err)
	}
	if rs.Len() != 1 || rs.Documents()[0].ID != "lex-1" {
		t.Fatalf("Documents() = %#v, want lex-1", rs.Documents())
	}
	if !errors.Is(err, ragy.ErrEmptyVector) {
		t.Fatalf("error chain missing ErrEmptyVector: %v", err)
	}
}
