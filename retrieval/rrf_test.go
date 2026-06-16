package retrieval

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
)

type cancelOnErrCallContext struct {
	context.Context

	cancelOnCall int
	calls        int
}

func newCancelOnErrCallContext(cancelOnCall int) *cancelOnErrCallContext {
	return &cancelOnErrCallContext{
		Context:      context.Background(),
		cancelOnCall: cancelOnCall,
	}
}

func (c *cancelOnErrCallContext) Err() error {
	c.calls++
	if c.calls >= c.cancelOnCall {
		return context.Canceled
	}

	return nil
}

func TestRRFRejectsEmptyMergeKey(t *testing.T) {
	t.Parallel()

	rrf, err := NewReciprocalRankFusion[struct{}](60, emptyMergeKeyResolver{})
	if err != nil {
		t.Fatalf("NewReciprocalRankFusion(): %v", err)
	}

	left := NewResultSet([]Document[struct{}]{{ID: "a", Content: "A", Score: 0.2}}, emptyMergeKeyResolver{})
	_, err = rrf.Merge(context.Background(), left)
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Merge() error = %v, want invalid argument", err)
	}
}

func TestRRFPreservesPartialFuseOnLateInvalidDoc(t *testing.T) {
	t.Parallel()

	rrf, err := NewReciprocalRankFusion[struct{}](60, mixedMergeKeyResolver{invalid: map[string]struct{}{"b": {}}})
	if err != nil {
		t.Fatalf("NewReciprocalRankFusion(): %v", err)
	}

	resolver := mixedMergeKeyResolver{invalid: map[string]struct{}{"b": {}}}
	left := NewResultSet([]Document[struct{}]{
		{ID: "a", Content: "A", Score: 0.9},
	}, resolver)
	right := NewResultSet([]Document[struct{}]{
		{ID: "b", Content: "B", Score: 0.5},
	}, resolver)

	out, err := rrf.Merge(context.Background(), left, right)
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Merge() error = %v, want invalid argument", err)
	}
	if out.Len() != 1 || out.Documents()[0].ID != "a" {
		t.Fatalf("Documents() = %#v, want partial fused doc from first list", out.Documents())
	}
}

func TestRRFSortPreservesTieOrder(t *testing.T) {
	t.Parallel()

	rrf, err := NewReciprocalRankFusion[struct{}](60, DocumentIDResolver[struct{}]{})
	if err != nil {
		t.Fatalf("NewReciprocalRankFusion(): %v", err)
	}

	left := NewResultSet([]Document[struct{}]{
		{ID: "b", Content: "B", Score: 0.5},
		{ID: "a", Content: "A", Score: 0.5},
	}, DocumentIDResolver[struct{}]{})
	right := NewResultSet([]Document[struct{}]{
		{ID: "c", Content: "C", Score: 0.1},
	}, DocumentIDResolver[struct{}]{})

	out, err := rrf.Merge(context.Background(), left, right)
	if err != nil {
		t.Fatalf("Merge(): %v", err)
	}
	docs := out.Documents()
	if len(docs) != 3 {
		t.Fatalf("Documents() len = %d, want 3", len(docs))
	}
	// Sorted merge keys (a, b, c) materialized before score-only stable sort.
	if docs[0].ID != "b" || docs[1].ID != "c" || docs[2].ID != "a" {
		t.Fatalf("Documents() = %#v, want b then c then a on equal top scores", docs)
	}
	for i := 1; i < len(docs); i++ {
		if docs[i].Score > docs[i-1].Score {
			t.Fatalf("Documents() = %#v, want non-increasing scores", docs)
		}
	}
}

func TestRRFPreservesFusedDocsOnContextCancelBetweenLists(t *testing.T) {
	t.Parallel()

	rrf, err := NewReciprocalRankFusion[struct{}](60, DocumentIDResolver[struct{}]{})
	if err != nil {
		t.Fatalf("NewReciprocalRankFusion(): %v", err)
	}

	left := NewResultSet([]Document[struct{}]{
		{ID: "a", Content: "A", Score: 0.9},
	}, DocumentIDResolver[struct{}]{})
	right := NewResultSet([]Document[struct{}]{
		{ID: "b", Content: "B", Score: 0.5},
	}, DocumentIDResolver[struct{}]{})

	ctx := newCancelOnErrCallContext(2)

	out, err := rrf.Merge(ctx, left, right)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Merge() error = %v, want context canceled", err)
	}
	if out.Len() != 1 || out.Documents()[0].ID != "a" {
		t.Fatalf("Documents() = %#v, want partial fused doc a", out.Documents())
	}
}

func TestBuildMergedDocumentsPreservesPartialOnContextCancel(t *testing.T) {
	t.Parallel()

	seen := map[string]fusedState[struct{}]{
		"a": {doc: Document[struct{}]{ID: "a", Content: "A", Score: 0}, score: 1},
		"b": {doc: Document[struct{}]{ID: "b", Content: "B", Score: 0}, score: 0.5},
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	out, err := buildMergedDocuments(ctx, seen, 1)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("buildMergedDocuments() error = %v, want context canceled", err)
	}
	if len(out) != 1 {
		t.Fatalf("buildMergedDocuments() = %#v, want one partial doc before cancel", out)
	}
}
