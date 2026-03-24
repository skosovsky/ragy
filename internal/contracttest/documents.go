package contracttest

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/documents"
	"github.com/skosovsky/ragy/filter"
)

type DocumentsStoreFactory func(t *testing.T, docs []ragy.Document) documents.Store

// RunDocumentsStoreSuite checks common documents.Store semantics.
func RunDocumentsStoreSuite(t *testing.T, factory DocumentsStoreFactory) {
	t.Helper()

	t.Run("find missing returns nil", func(t *testing.T) {
		testFindMissingReturnsNil(t, factory)
	})

	t.Run("delete empty filter rejects", func(t *testing.T) {
		testDeleteEmptyFilterRejects(t, factory)
	})

	t.Run("delete nil filter rejects without mutation", func(t *testing.T) {
		testDeleteNilFilterRejectsWithoutMutation(t, factory)
	})

	t.Run("delete by filter reports count and mutates state", func(t *testing.T) {
		testDeleteByFilterMutatesState(t, factory)
	})

	t.Run("delete undeclared filter rejects", func(t *testing.T) {
		testDeleteUndeclaredFilterRejects(t, factory)
	})
}

func testFindMissingReturnsNil(t *testing.T, factory DocumentsStoreFactory) {
	t.Helper()

	store := factory(t, []ragy.Document{{
		ID:      "doc-1",
		Content: "hello",
	}})

	docs, err := store.FindByIDs(context.Background(), []string{"missing"})
	if err != nil {
		t.Fatalf("FindByIDs(): %v", err)
	}
	if docs != nil {
		t.Fatalf("FindByIDs() = %#v, want nil", docs)
	}
}

func testDeleteEmptyFilterRejects(t *testing.T, factory DocumentsStoreFactory) {
	t.Helper()

	store := factory(t, []ragy.Document{{
		ID:      "doc-1",
		Content: "hello",
	}})

	expr, err := filter.Normalize(nil)
	if err != nil {
		t.Fatalf("Normalize(nil): %v", err)
	}

	if _, err := store.DeleteByFilter(context.Background(), expr); err == nil {
		t.Fatal("DeleteByFilter(empty) error = nil, want error")
	}
}

func testDeleteNilFilterRejectsWithoutMutation(t *testing.T, factory DocumentsStoreFactory) {
	t.Helper()

	store := factory(t, []ragy.Document{{
		ID:      "doc-1",
		Content: "hello",
		Attributes: ragy.Attributes{
			"tenant": "acme",
		},
	}})

	if _, err := store.DeleteByFilter(context.Background(), nil); err == nil {
		t.Fatal("DeleteByFilter(nil) error = nil, want error")
	}

	remaining, err := store.FindByIDs(context.Background(), []string{"doc-1"})
	if err != nil {
		t.Fatalf("FindByIDs(remaining): %v", err)
	}
	if len(remaining) != 1 || remaining[0].ID != "doc-1" {
		t.Fatalf("FindByIDs(remaining) = %#v, want doc-1", remaining)
	}
}

func testDeleteByFilterMutatesState(t *testing.T, factory DocumentsStoreFactory) {
	t.Helper()

	store := factory(t, []ragy.Document{
		{
			ID:      "doc-1",
			Content: "hello",
			Attributes: ragy.Attributes{
				"tenant": "acme",
			},
		},
		{
			ID:      "doc-2",
			Content: "world",
			Attributes: ragy.Attributes{
				"tenant": "globex",
			},
		},
	})

	tenant, err := store.Schema().StringField("tenant")
	if err != nil {
		t.Fatalf("Schema().StringField(tenant): %v", err)
	}
	expr, err := filter.Normalize(filter.Equal(tenant, "acme"))
	if err != nil {
		t.Fatalf("Normalize(): %v", err)
	}

	result, err := store.DeleteByFilter(context.Background(), expr)
	if err != nil {
		t.Fatalf("DeleteByFilter(): %v", err)
	}
	if result.Deleted != 1 {
		t.Fatalf("DeleteResult.Deleted = %d, want 1", result.Deleted)
	}

	deleted, err := store.FindByIDs(context.Background(), []string{"doc-1"})
	if err != nil {
		t.Fatalf("FindByIDs(deleted): %v", err)
	}
	if deleted != nil {
		t.Fatalf("FindByIDs(deleted) = %#v, want nil", deleted)
	}

	remaining, err := store.FindByIDs(context.Background(), []string{"doc-2"})
	if err != nil {
		t.Fatalf("FindByIDs(remaining): %v", err)
	}
	if len(remaining) != 1 || remaining[0].ID != "doc-2" {
		t.Fatalf("FindByIDs(remaining) = %#v, want doc-2", remaining)
	}
}

func testDeleteUndeclaredFilterRejects(t *testing.T, factory DocumentsStoreFactory) {
	t.Helper()

	store := factory(t, []ragy.Document{{
		ID:      "doc-1",
		Content: "hello",
	}})

	_, err := store.Schema().StringField("missing")
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Schema().StringField(missing) error = %v, want invalid argument", err)
	}
}
