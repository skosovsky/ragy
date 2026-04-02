// Command fallback_search demonstrates a dense.Searcher decorator that falls back on ErrUnavailable.
// Primary is a stub (not a live Qdrant client) so the example builds without external services;
// the same pattern applies when Primary wraps a real vector backend.
package main

import (
	"context"
	"errors"
	"fmt"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/filter"
)

const (
	fallbackDocRelevance = 0.9
	searchPageLimit      = 10
)

func main() {
	schema, err := filter.NewSchema().Build()
	if err != nil {
		panic(err)
	}

	primary := &stubSearcher{
		schema: schema,
		fail:   true,
		docs:   nil,
	}
	fallback := &stubSearcher{
		schema: schema,
		fail:   false,
		docs: []ragy.Document{
			{ID: "fb-1", Content: "fallback hit", Relevance: fallbackDocRelevance},
		},
	}

	combo := fallbackSearcher{Primary: primary, Fallback: fallback}
	page, err := ragy.NewPage(searchPageLimit, 0)
	if err != nil {
		panic(err)
	}

	ctx := context.Background()
	docs, err := combo.Search(ctx, dense.Request{
		Vector: []float32{1, 0, 0},
		Filter: nil,
		Page:   page,
	})
	if err != nil {
		panic(err)
	}
	if len(docs) != 1 || docs[0].ID != "fb-1" {
		panic("expected fallback document")
	}
	fmt.Printf("ok: %q relevance=%.2f\n", docs[0].Content, docs[0].Relevance)
}

type stubSearcher struct {
	schema filter.Schema
	fail   bool
	docs   []ragy.Document
}

func (s *stubSearcher) Schema() filter.Schema { return s.schema }

func (s *stubSearcher) Search(_ context.Context, _ dense.Request) ([]ragy.Document, error) {
	if s.fail {
		return nil, fmt.Errorf("%w: primary vector store down", ragy.ErrUnavailable)
	}
	out := make([]ragy.Document, len(s.docs))
	copy(out, s.docs)
	return out, nil
}

type fallbackSearcher struct {
	Primary  dense.Searcher
	Fallback dense.Searcher
}

func (f fallbackSearcher) Schema() filter.Schema {
	return f.Primary.Schema()
}

func (f fallbackSearcher) Search(ctx context.Context, req dense.Request) ([]ragy.Document, error) {
	docs, err := f.Primary.Search(ctx, req)
	if err == nil {
		return docs, nil
	}
	// Only degrade on transient failures; see README "Resilience & execution control".
	if errors.Is(err, ragy.ErrUnavailable) {
		return f.Fallback.Search(ctx, req)
	}
	return nil, err
}
