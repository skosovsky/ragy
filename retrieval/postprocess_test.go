package retrieval

import (
	"context"
	"fmt"
	"strconv"
	"testing"
)

func TestDefaultMergeStrategyUsesBestScoreMeta(t *testing.T) {
	t.Parallel()

	type meta struct {
		Source string
	}

	merge := DefaultMergeStrategy[meta]()
	out, err := merge([]Document[meta]{
		{ID: "a", Content: "first", Score: 0.2, Meta: meta{Source: "s1"}},
		{ID: "b", Content: "second", Score: 0.9, Meta: meta{Source: "s2"}},
	})
	if err != nil {
		t.Fatalf("merge: %v", err)
	}
	if out.ID != "b" || out.Meta.Source != "s2" {
		t.Fatalf("out = %#v, want id/meta from highest score", out)
	}
	if out.Content != "first\n\nsecond" {
		t.Fatalf("content = %q", out.Content)
	}
}

func TestPipelineAppliesProcessors(t *testing.T) {
	t.Parallel()

	type meta struct {
		Group string
	}

	backend := stubBackend[meta]{
		docs: []Document[meta]{
			{ID: "1", Content: "a", Score: 0.9, Meta: meta{Group: "g1"}},
			{ID: "2", Content: "b", Score: 0.8, Meta: meta{Group: "g1"}},
			{ID: "3", Content: "c", Score: 0.7, Meta: meta{Group: "g2"}},
		},
	}

	pipeline := NewPipeline[meta](
		backend,
		GroupBy(func(m meta) string { return m.Group }, DefaultMergeStrategy[meta]()),
	)

	out, err := pipeline.Retrieve(context.Background(), "query", RetrieveOptions{TopK: 10})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(out) != 2 {
		t.Fatalf("len(out) = %d, want 2", len(out))
	}
}

func TestPipelineAppliesTopKAfterGroupBy(t *testing.T) {
	t.Parallel()

	type meta struct {
		Group string
	}

	docs := make([]Document[meta], 0, 10)
	for i := range 10 {
		group := string(rune('a' + i/2))
		docs = append(docs, Document[meta]{
			ID:      strconv.Itoa(i + 1),
			Content: fmt.Sprintf("chunk-%d", i+1),
			Score:   1.0 - float64(i)*0.05,
			Meta:    meta{Group: group},
		})
	}

	pipeline := NewPipeline[meta](
		stubBackend[meta]{docs: docs},
		GroupBy(func(m meta) string { return m.Group }, DefaultMergeStrategy[meta]()),
	)

	out, err := pipeline.Retrieve(context.Background(), "query", RetrieveOptions{TopK: 3})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(out) != 3 {
		t.Fatalf("len(out) = %d, want 3", len(out))
	}
}

func TestPipelineNormalizesFetchLimitFromTopK(t *testing.T) {
	t.Parallel()

	backend := &capturingBackend[struct{}]{
		docs: []Document[struct{}]{{ID: "doc-1", Content: "ok", Score: 0.5}},
	}
	pipeline := NewPipeline[struct{}](backend)

	_, err := pipeline.Retrieve(context.Background(), "query", RetrieveOptions{TopK: 7})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if backend.lastOpts.FetchLimit != 7 {
		t.Fatalf("FetchLimit = %d, want 7", backend.lastOpts.FetchLimit)
	}
}

func TestPipelinePreservesExplicitFetchLimit(t *testing.T) {
	t.Parallel()

	backend := &capturingBackend[struct{}]{
		docs: []Document[struct{}]{{ID: "doc-1", Content: "ok", Score: 0.5}},
	}
	pipeline := NewPipeline[struct{}](backend)

	_, err := pipeline.Retrieve(context.Background(), "query", RetrieveOptions{
		FetchLimit: 50,
		TopK:       10,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if backend.lastOpts.FetchLimit != 50 {
		t.Fatalf("FetchLimit = %d, want 50", backend.lastOpts.FetchLimit)
	}
}

type capturingBackend[TMeta any] struct {
	docs     []Document[TMeta]
	lastOpts RetrieveOptions
}

func (s *capturingBackend[TMeta]) Retrieve(
	_ context.Context,
	_ string,
	opts RetrieveOptions,
) ([]Document[TMeta], error) {
	s.lastOpts = opts
	out := make([]Document[TMeta], len(s.docs))
	copy(out, s.docs)
	return out, nil
}

type stubBackend[TMeta any] struct {
	docs []Document[TMeta]
}

func (s stubBackend[TMeta]) Retrieve(_ context.Context, _ string, _ RetrieveOptions) ([]Document[TMeta], error) {
	out := make([]Document[TMeta], len(s.docs))
	copy(out, s.docs)
	return out, nil
}
