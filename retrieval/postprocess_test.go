package retrieval

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"testing"

	ragy "github.com/skosovsky/ragy"
)

type ppMeta struct {
	Group string
}

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

	pipeline, err := NewPipelineBuilder[struct{}, meta]().
		WithRoot(RetrieverNode[struct{}, meta]{Backend: backend}).
		WithPostProcessors(GroupBy(func(m meta) string { return m.Group }, DefaultMergeStrategy[meta]())).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	out, err := pipeline.Retrieve(context.Background(), Query[struct{}]{
		Text:    "query",
		Options: RetrieveOptions{TopK: 10},
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if out.Len() != 2 {
		t.Fatalf("Len() = %d, want 2", out.Len())
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

	pipeline, err := NewPipelineBuilder[struct{}, meta]().
		WithRoot(RetrieverNode[struct{}, meta]{Backend: stubBackend[meta]{docs: docs}}).
		WithPostProcessors(GroupBy(func(m meta) string { return m.Group }, DefaultMergeStrategy[meta]())).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	out, err := pipeline.Retrieve(context.Background(), Query[struct{}]{
		Text:    "query",
		Options: RetrieveOptions{TopK: 3},
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if out.Len() != 3 {
		t.Fatalf("Len() = %d, want 3", out.Len())
	}
}

type stubBackend[TMeta any] struct {
	docs []Document[TMeta]
}

func (s stubBackend[TMeta]) Retrieve(_ context.Context, _ string, _ RetrieveOptions) (ResultSet[TMeta], error) {
	out := make([]Document[TMeta], len(s.docs))
	copy(out, s.docs)
	return NewResultSet(out, DocumentIDResolver[TMeta]{}), nil
}

func TestGroupByPreservesPartialResultOnMergeStrategyError(t *testing.T) {
	t.Parallel()

	type meta struct {
		Group string
	}

	calls := 0
	merge := func(docs []Document[meta]) (Document[meta], error) {
		calls++
		if calls == 2 {
			return Document[meta]{}, ragy.ErrInvalidArgument
		}
		return DefaultMergeStrategy[meta]()(docs)
	}

	processor := GroupBy(func(m meta) string { return m.Group }, merge)
	rs := NewResultSet([]Document[meta]{
		{ID: "1", Content: "a", Score: 0.9, Meta: meta{Group: "g1"}},
		{ID: "2", Content: "b", Score: 0.8, Meta: meta{Group: "g2"}},
		{ID: "3", Content: "c", Score: 0.7, Meta: meta{Group: "g3"}},
	}, DocumentIDResolver[meta]{})

	out, err := processor.Process(rs)
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Process() error = %v, want invalid argument", err)
	}
	if out.Len() != 1 {
		t.Fatalf("Len() = %d, want one merged group preserved", out.Len())
	}
}

func TestGroupByRejectsNilKeySelector(t *testing.T) {
	t.Parallel()

	processor := GroupBy[ppMeta](nil, DefaultMergeStrategy[ppMeta]())
	rs := NewResultSet([]Document[ppMeta]{{ID: "1", Meta: ppMeta{Group: "g1"}}}, DocumentIDResolver[ppMeta]{})
	out, err := processor.Process(rs)
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Process() error = %v, want invalid argument", err)
	}
	if out.Len() != 1 {
		t.Fatalf("Len() = %d, want preserved input doc", out.Len())
	}
}

func TestTopPerGroupRejectsNilKeySelector(t *testing.T) {
	t.Parallel()

	processor := TopPerGroup[ppMeta](nil, 1)
	rs := NewResultSet([]Document[ppMeta]{{ID: "1", Meta: ppMeta{Group: "g1"}}}, DocumentIDResolver[ppMeta]{})
	out, err := processor.Process(rs)
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Process() error = %v, want invalid argument", err)
	}
	if out.Len() != 1 {
		t.Fatalf("Len() = %d, want preserved input doc", out.Len())
	}
}

func TestTopPerGroupRejectsNonPositiveLimit(t *testing.T) {
	t.Parallel()

	processor := TopPerGroup(func(m ppMeta) string { return m.Group }, 0)
	rs := NewResultSet([]Document[ppMeta]{{ID: "1", Meta: ppMeta{Group: "g1"}}}, DocumentIDResolver[ppMeta]{})
	out, err := processor.Process(rs)
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Process() error = %v, want invalid argument", err)
	}
	if out.Len() != 1 {
		t.Fatalf("Len() = %d, want preserved input doc", out.Len())
	}
}

func TestRerankRejectsNilLess(t *testing.T) {
	t.Parallel()

	processor := Rerank[ppMeta](nil)
	rs := NewResultSet([]Document[ppMeta]{{ID: "1", Meta: ppMeta{Group: "g1"}}}, DocumentIDResolver[ppMeta]{})
	out, err := processor.Process(rs)
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Process() error = %v, want invalid argument", err)
	}
	if out.Len() != 1 {
		t.Fatalf("Len() = %d, want preserved input doc", out.Len())
	}
}

func TestGroupByRejectsInvalidDocument(t *testing.T) {
	t.Parallel()

	processor := GroupBy(func(m ppMeta) string { return m.Group }, DefaultMergeStrategy[ppMeta]())
	rs := NewResultSet([]Document[ppMeta]{
		{ID: "ok", Content: "good", Score: 0.5, Meta: ppMeta{Group: "g1"}},
		{Content: "broken", Score: 0.5, Meta: ppMeta{Group: "g2"}},
	}, DocumentIDResolver[ppMeta]{})
	out, err := processor.Process(rs)
	if !errors.Is(err, ragy.ErrMissingID) {
		t.Fatalf("Process() error = %v, want missing id", err)
	}
	if out.Len() != 2 {
		t.Fatalf("Len() = %d, want preserved input docs", out.Len())
	}
	if out.Documents()[0].ID != "ok" {
		t.Fatalf("Documents()[0].ID = %q, want ok", out.Documents()[0].ID)
	}
}

func TestTopPerGroupRejectsInvalidDocument(t *testing.T) {
	t.Parallel()

	processor := TopPerGroup(func(m ppMeta) string { return m.Group }, 1)
	rs := NewResultSet([]Document[ppMeta]{
		{ID: "ok", Content: "good", Score: 0.5, Meta: ppMeta{Group: "g1"}},
		{Content: "broken", Score: 0.5, Meta: ppMeta{Group: "g2"}},
	}, DocumentIDResolver[ppMeta]{})
	out, err := processor.Process(rs)
	if !errors.Is(err, ragy.ErrMissingID) {
		t.Fatalf("Process() error = %v, want missing id", err)
	}
	if out.Len() != 2 {
		t.Fatalf("Len() = %d, want preserved input docs", out.Len())
	}
	if out.Documents()[0].ID != "ok" {
		t.Fatalf("Documents()[0].ID = %q, want ok", out.Documents()[0].ID)
	}
}

func TestRerankRejectsInvalidDocument(t *testing.T) {
	t.Parallel()

	processor := Rerank(func(a, b Document[ppMeta]) bool { return a.Score > b.Score })
	rs := NewResultSet([]Document[ppMeta]{
		{ID: "ok", Content: "good", Score: 0.5, Meta: ppMeta{Group: "g1"}},
		{Content: "broken", Score: 0.5, Meta: ppMeta{Group: "g2"}},
	}, DocumentIDResolver[ppMeta]{})
	out, err := processor.Process(rs)
	if !errors.Is(err, ragy.ErrMissingID) {
		t.Fatalf("Process() error = %v, want missing id", err)
	}
	if out.Len() != 2 {
		t.Fatalf("Len() = %d, want preserved input docs", out.Len())
	}
	if out.Documents()[0].ID != "ok" {
		t.Fatalf("Documents()[0].ID = %q, want ok", out.Documents()[0].ID)
	}
}

func TestGroupByRejectsEmptyGroupKey(t *testing.T) {
	t.Parallel()

	processor := GroupBy(func(m ppMeta) string { return m.Group }, DefaultMergeStrategy[ppMeta]())
	rs := NewResultSet([]Document[ppMeta]{
		{ID: "ok", Content: "good", Score: 0.5, Meta: ppMeta{Group: "g1"}},
		{ID: "bad", Content: "other", Score: 0.4, Meta: ppMeta{Group: ""}},
	}, DocumentIDResolver[ppMeta]{})
	out, err := processor.Process(rs)
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Process() error = %v, want invalid argument", err)
	}
	if out.Len() != 2 {
		t.Fatalf("Len() = %d, want preserved input docs", out.Len())
	}
}

func TestTopPerGroupRejectsEmptyGroupKey(t *testing.T) {
	t.Parallel()

	processor := TopPerGroup(func(m ppMeta) string { return m.Group }, 1)
	rs := NewResultSet([]Document[ppMeta]{
		{ID: "ok", Content: "good", Score: 0.5, Meta: ppMeta{Group: "g1"}},
		{ID: "bad", Content: "other", Score: 0.4, Meta: ppMeta{Group: ""}},
	}, DocumentIDResolver[ppMeta]{})
	out, err := processor.Process(rs)
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Process() error = %v, want invalid argument", err)
	}
	if out.Len() != 2 {
		t.Fatalf("Len() = %d, want preserved input docs", out.Len())
	}
}
