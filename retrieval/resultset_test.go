package retrieval

import (
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
)

func TestResultSetDedupSortsByScore(t *testing.T) {
	t.Parallel()

	rs := NewResultSet([]Document[struct{}]{
		{ID: "low", Content: "A", Score: 0.1},
		{ID: "high", Content: "B", Score: 0.9},
		{ID: "mid", Content: "C", Score: 0.5},
	}, DocumentIDResolver[struct{}]{})
	deduped, err := rs.Dedup()
	if err != nil {
		t.Fatalf("Dedup(): %v", err)
	}
	docs := deduped.Documents()
	if len(docs) != 3 || docs[0].ID != "high" || docs[1].ID != "mid" || docs[2].ID != "low" {
		t.Fatalf("Documents() = %#v, want score-desc order", docs)
	}
}

type emptyMergeKeyResolver struct{}

func (emptyMergeKeyResolver) Resolve(doc Document[struct{}]) Identity {
	return Identity{MergeKey: "", DocumentID: doc.ID}
}

type mixedMergeKeyResolver struct {
	invalid map[string]struct{}
}

func (m mixedMergeKeyResolver) Resolve(doc Document[struct{}]) Identity {
	if _, ok := m.invalid[doc.ID]; ok {
		return Identity{MergeKey: "", DocumentID: doc.ID}
	}
	return Identity{MergeKey: doc.ID, DocumentID: doc.ID}
}

func TestResultSetMergeRejectsEmptyMergeKey(t *testing.T) {
	t.Parallel()

	left := NewResultSet([]Document[struct{}]{{ID: "a", Content: "A", Score: 0.2}}, emptyMergeKeyResolver{})
	right := NewResultSet([]Document[struct{}]{{ID: "b", Content: "B", Score: 0.9}}, emptyMergeKeyResolver{})

	_, err := left.Merge(right)
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Merge() error = %v, want invalid argument", err)
	}
}

func TestResultSetMergePreservesPartialOnInvalidKey(t *testing.T) {
	t.Parallel()

	resolver := mixedMergeKeyResolver{invalid: map[string]struct{}{"b": {}}}
	left := NewResultSet([]Document[struct{}]{
		{ID: "a", Content: "A", Score: 0.9},
	}, resolver)
	right := NewResultSet([]Document[struct{}]{
		{ID: "b", Content: "B", Score: 0.5},
	}, resolver)

	out, err := left.Merge(right)
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Merge() error = %v, want invalid argument", err)
	}
	if out.Len() != 1 || out.Documents()[0].ID != "a" {
		t.Fatalf("Documents() = %#v, want preserved left doc", out.Documents())
	}
}

func TestResultSetMergeRejectsInvalidDocument(t *testing.T) {
	t.Parallel()

	left := NewResultSet([]Document[struct{}]{
		{ID: "a", Content: "A", Score: 0.9},
	}, DocumentIDResolver[struct{}]{})
	right := NewResultSet([]Document[struct{}]{
		{ID: "b", Content: "B", Score: 1.5},
	}, DocumentIDResolver[struct{}]{})

	out, err := left.Merge(right)
	if !errors.Is(err, ragy.ErrProtocol) {
		t.Fatalf("Merge() error = %v, want protocol", err)
	}
	if out.Len() != 1 || out.Documents()[0].ID != "a" {
		t.Fatalf("Documents() = %#v, want preserved left doc", out.Documents())
	}
}

func TestResultSetMergePreservesFirstSeenOnScoreTie(t *testing.T) {
	t.Parallel()

	resolver := sharedKeyResolver{}
	left := NewResultSet([]Document[struct{}]{
		{ID: "first", Content: "key", Score: 0.5},
	}, resolver)
	right := NewResultSet([]Document[struct{}]{
		{ID: "second", Content: "key", Score: 0.5},
	}, resolver)

	merged, err := left.Merge(right)
	if err != nil {
		t.Fatalf("Merge(): %v", err)
	}
	if merged.Len() != 1 || merged.Documents()[0].ID != "first" {
		t.Fatalf("Documents() = %#v, want first seen doc on score tie", merged.Documents())
	}
}

func TestResultSetDedupPreservesFirstSeenOnScoreTie(t *testing.T) {
	t.Parallel()

	resolver := sharedKeyResolver{}
	rs := NewResultSet([]Document[struct{}]{
		{ID: "first", Content: "key", Score: 0.5},
		{ID: "second", Content: "key", Score: 0.5},
	}, resolver)

	deduped, err := rs.Dedup()
	if err != nil {
		t.Fatalf("Dedup(): %v", err)
	}
	if deduped.Len() != 1 || deduped.Documents()[0].ID != "first" {
		t.Fatalf("Documents() = %#v, want first seen doc on score tie", deduped.Documents())
	}
}

func TestResultSetDedupRejectsEmptyMergeKey(t *testing.T) {
	t.Parallel()

	rs := NewResultSet([]Document[struct{}]{{ID: "a", Content: "A", Score: 0.2}}, emptyMergeKeyResolver{})
	_, err := rs.Dedup()
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Dedup() error = %v, want invalid argument", err)
	}
}

func TestResultSetMergeKeepsHighestScore(t *testing.T) {
	t.Parallel()

	left := NewResultSet([]Document[struct{}]{
		{ID: "a", Content: "A", Score: 0.2},
	}, DocumentIDResolver[struct{}]{})
	right := NewResultSet([]Document[struct{}]{
		{ID: "a", Content: "B", Score: 0.9},
	}, DocumentIDResolver[struct{}]{})

	merged, err := left.Merge(right)
	if err != nil {
		t.Fatalf("Merge(): %v", err)
	}
	docs := merged.Documents()
	if len(docs) != 1 || docs[0].Score != 0.9 {
		t.Fatalf("Documents() = %#v, want score 0.9", docs)
	}
}

func TestResultSetIsEmpty(t *testing.T) {
	t.Parallel()

	rs := NewResultSet[struct{}](nil, DocumentIDResolver[struct{}]{})
	if rs == nil || !rs.IsEmpty() {
		t.Fatalf("IsEmpty() = false, want true for nil docs")
	}
}

type tenantMergeResolver struct{}

func (tenantMergeResolver) Resolve(doc Document[struct{ Tenant string }]) Identity {
	return Identity{MergeKey: doc.Meta.Tenant, DocumentID: doc.ID}
}

func TestResultSetDedupUsesMergeKey(t *testing.T) {
	t.Parallel()

	rs := NewResultSet([]Document[struct{ Tenant string }]{
		{ID: "a", Content: "A", Score: 0.2, Meta: struct{ Tenant string }{Tenant: "acme"}},
		{ID: "b", Content: "B", Score: 0.9, Meta: struct{ Tenant string }{Tenant: "acme"}},
	}, tenantMergeResolver{})
	deduped, err := rs.Dedup()
	if err != nil {
		t.Fatalf("Dedup(): %v", err)
	}
	if deduped.Len() != 1 || deduped.Documents()[0].Score != 0.9 {
		t.Fatalf("Documents() = %#v, want one highest score", deduped.Documents())
	}
}

func TestResultSetDocumentsDefensiveCopy(t *testing.T) {
	t.Parallel()

	rs := NewResultSet([]Document[struct{}]{{ID: "a", Content: "A", Score: 1}}, DocumentIDResolver[struct{}]{})
	docs := rs.Documents()
	docs[0].Content = "mutated"
	if rs.Documents()[0].Content != "A" {
		t.Fatalf("internal content mutated: %#v", rs.Documents())
	}
}

func TestResultSetMergeImmutability(t *testing.T) {
	t.Parallel()

	left := NewResultSet([]Document[struct{}]{{ID: "a", Score: 0.2}}, DocumentIDResolver[struct{}]{})
	right := NewResultSet([]Document[struct{}]{{ID: "b", Score: 0.9}}, DocumentIDResolver[struct{}]{})
	leftDocs := left.Documents()
	rightDocs := right.Documents()
	_, err := left.Merge(right)
	if err != nil {
		t.Fatalf("Merge(): %v", err)
	}
	if leftDocs[0].Score != 0.2 || rightDocs[0].Score != 0.9 {
		t.Fatalf("merge mutated inputs: left=%#v right=%#v", leftDocs, rightDocs)
	}
}
