// Command fallback_search demonstrates a retrieval retriever decorator that falls back on ErrUnavailable.
// Primary is a stub (not a live Qdrant client) so the example builds without external services;
// the same pattern applies when Primary wraps a real vector backend.
package main

import (
	"context"
	"errors"
	"fmt"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
)

const fallbackDocScore = 0.9

func main() {
	schema, err := filter.NewSchema().Build()
	if err != nil {
		panic(err)
	}

	primary := &stubBackend{
		schema: schema,
		fail:   true,
		docs:   nil,
	}
	fallback := &stubBackend{
		schema: schema,
		fail:   false,
		docs: []retrieval.Document[struct{}]{
			{ID: "fb-1", Content: "fallback hit", Score: fallbackDocScore},
		},
	}

	combo := fallbackRetriever{Primary: primary, Fallback: fallback}

	ctx := context.Background()
	const topK = 10
	docs, err := combo.Retrieve(ctx, "hello", retrieval.RetrieveOptions{
		TopK:   topK,
		Vector: []float32{1, 0, 0},
	})
	if err != nil {
		panic(err)
	}
	if len(docs) != 1 || docs[0].ID != "fb-1" {
		panic("expected fallback document")
	}
	fmt.Printf("ok: %q score=%.2f\n", docs[0].Content, docs[0].Score)
}

type stubBackend struct {
	schema filter.Schema
	fail   bool
	docs   []retrieval.Document[struct{}]
}

func (s *stubBackend) Schema() filter.Schema { return s.schema }

func (s *stubBackend) Retrieve(
	_ context.Context,
	_ string,
	_ retrieval.RetrieveOptions,
) ([]retrieval.Document[struct{}], error) {
	if s.fail {
		return nil, fmt.Errorf("%w: primary vector store down", ragy.ErrUnavailable)
	}
	out := make([]retrieval.Document[struct{}], len(s.docs))
	copy(out, s.docs)
	return out, nil
}

type fallbackRetriever struct {
	Primary  retrieval.Backend[struct{}]
	Fallback retrieval.Backend[struct{}]
}

func (f fallbackRetriever) Retrieve(
	ctx context.Context,
	query string,
	opts retrieval.RetrieveOptions,
) ([]retrieval.Document[struct{}], error) {
	docs, err := f.Primary.Retrieve(ctx, query, opts)
	if err == nil {
		return docs, nil
	}
	// Only degrade on transient failures; see README "Resilience & execution control".
	if errors.Is(err, ragy.ErrUnavailable) {
		return f.Fallback.Retrieve(ctx, query, opts)
	}
	return nil, err
}
