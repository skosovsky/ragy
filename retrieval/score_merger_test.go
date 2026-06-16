package retrieval

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
)

type sharedKeyResolver struct{}

func (sharedKeyResolver) Resolve(doc Document[struct{}]) Identity {
	return Identity{MergeKey: "shared", DocumentID: doc.ID}
}

func TestScoreMergerMergeKeepsMaxScorePerMergeKey(t *testing.T) {
	t.Parallel()

	resolver := sharedKeyResolver{}
	merger := NewScoreMerger[struct{}](resolver)
	left := NewResultSet([]Document[struct{}]{
		{ID: "a", Content: "key", Score: 0.4},
	}, resolver)
	right := NewResultSet([]Document[struct{}]{
		{ID: "b", Content: "key", Score: 0.9},
	}, resolver)

	out, err := merger.Merge(context.Background(), left, right)
	if err != nil {
		t.Fatalf("Merge() error = %v", err)
	}
	if out.Len() != 1 {
		t.Fatalf("Len() = %d, want 1 merged doc", out.Len())
	}
	if out.Documents()[0].ID != "b" {
		t.Fatalf("Documents()[0].ID = %q, want b (max score)", out.Documents()[0].ID)
	}
}

func TestScoreMergerPreservesPartialMergeOnInvalidKey(t *testing.T) {
	t.Parallel()

	merger := NewScoreMerger[struct{}](mixedMergeKeyResolver{invalid: map[string]struct{}{"b": {}}})
	left := NewResultSet([]Document[struct{}]{
		{ID: "a", Content: "A", Score: 0.9},
	}, mixedMergeKeyResolver{invalid: map[string]struct{}{"b": {}}})
	right := NewResultSet([]Document[struct{}]{
		{ID: "b", Content: "B", Score: 0.5},
	}, mixedMergeKeyResolver{invalid: map[string]struct{}{"b": {}}})

	out, err := merger.Merge(context.Background(), left, right)
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Merge() error = %v, want invalid argument", err)
	}
	if out.Len() != 1 || out.Documents()[0].ID != "a" {
		t.Fatalf("Documents() = %#v, want preserved left doc", out.Documents())
	}
}

func TestScoreMergerRejectsInvalidDocument(t *testing.T) {
	t.Parallel()

	merger := NewScoreMerger[struct{}](DocumentIDResolver[struct{}]{})
	left := NewResultSet([]Document[struct{}]{
		{ID: "a", Content: "A", Score: 0.9},
	}, DocumentIDResolver[struct{}]{})
	right := NewResultSet([]Document[struct{}]{
		{ID: "b", Content: "B", Score: 1.5},
	}, DocumentIDResolver[struct{}]{})

	out, err := merger.Merge(context.Background(), left, right)
	if !errors.Is(err, ragy.ErrProtocol) {
		t.Fatalf("Merge() error = %v, want protocol", err)
	}
	if out.Len() != 1 || out.Documents()[0].ID != "a" {
		t.Fatalf("Documents() = %#v, want preserved left doc", out.Documents())
	}
}
