package contracttest

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/documents"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
)

// StructMeta is a typed metadata fixture for struct-based contract tests.
type StructMeta struct {
	Tenant string `json:"tenant"`
}

type DenseStructBackendFactory func(t *testing.T, docs []retrieval.Document[StructMeta]) retrieval.Backend[StructMeta]
type LexicalStructBackendFactory func(t *testing.T, docs []retrieval.Document[StructMeta]) retrieval.Backend[StructMeta]
type DocumentsStructStoreFactory func(t *testing.T, docs []retrieval.Document[StructMeta]) documents.Store[StructMeta]

func tenantStructCondition(t *testing.T, schema filter.Schema, value string) filter.Condition {
	t.Helper()

	tenant, err := schema.StringField("tenant")
	if err != nil {
		t.Fatalf("Schema().StringField(tenant): %v", err)
	}

	builder, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatalf("NewBuilder(): %v", err)
	}

	cond, err := filter.Eq(builder, tenant, value).Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	return cond
}

// RunDenseStructBackendSuite checks typed struct metadata for dense backends.
func RunDenseStructBackendSuite(t *testing.T, factory DenseStructBackendFactory) {
	t.Helper()

	t.Run("valid docs pass through", func(t *testing.T) {
		backend := factory(t, []retrieval.Document[StructMeta]{{ID: wantedDocID, Content: "hello"}})
		out, err := backend.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
			Vector: []float32{1},
		})
		if err != nil {
			t.Fatalf("Retrieve(): %v", err)
		}
		if len(out) != 1 || out[0].ID != wantedDocID {
			t.Fatalf("Retrieve() = %#v, want doc-1", out)
		}
	})

	t.Run("declared filter built from builder passes", func(t *testing.T) {
		backend := factory(t, []retrieval.Document[StructMeta]{{
			ID:      wantedDocID,
			Content: "hello",
			Meta:    StructMeta{Tenant: tenantAcme},
		}})
		schema := schemaFromTypedBackend(t, backend)
		out, err := backend.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
			Vector:  []float32{1},
			Filters: tenantStructCondition(t, schema, tenantAcme),
		})
		if err != nil {
			t.Fatalf("Retrieve(): %v", err)
		}
		if len(out) != 1 || out[0].Meta.Tenant != tenantAcme {
			t.Fatalf("Retrieve() = %#v, want tenant acme", out)
		}
	})

	t.Run("invalid docs reject", func(t *testing.T) {
		backend := factory(t, []retrieval.Document[StructMeta]{{Content: "broken"}})
		_, err := backend.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
			Vector: []float32{1},
		})
		if err == nil {
			t.Fatal("Retrieve() error = nil, want error")
		}
	})
}

// RunLexicalStructBackendSuite checks typed struct metadata for lexical backends.
func RunLexicalStructBackendSuite(t *testing.T, factory LexicalStructBackendFactory) {
	t.Helper()

	t.Run("valid docs pass through", func(t *testing.T) {
		backend := factory(t, []retrieval.Document[StructMeta]{{ID: wantedDocID, Content: "hello"}})
		out, err := backend.Retrieve(context.Background(), "hello", retrieval.RetrieveOptions{})
		if err != nil {
			t.Fatalf("Retrieve(): %v", err)
		}
		if len(out) != 1 || out[0].ID != wantedDocID {
			t.Fatalf("Retrieve() = %#v, want doc-1", out)
		}
	})

	t.Run("declared filter built from builder passes", func(t *testing.T) {
		backend := factory(t, []retrieval.Document[StructMeta]{{
			ID:      wantedDocID,
			Content: "hello",
			Meta:    StructMeta{Tenant: tenantAcme},
		}})
		schema := schemaFromTypedBackend(t, backend)
		out, err := backend.Retrieve(context.Background(), "hello", retrieval.RetrieveOptions{
			Filters: tenantStructCondition(t, schema, tenantAcme),
		})
		if err != nil {
			t.Fatalf("Retrieve(): %v", err)
		}
		if len(out) != 1 || out[0].Meta.Tenant != tenantAcme {
			t.Fatalf("Retrieve() = %#v, want tenant acme", out)
		}
	})
}

// RunDocumentsStructStoreSuite checks typed struct metadata for document stores.
func RunDocumentsStructStoreSuite(t *testing.T, factory DocumentsStructStoreFactory) {
	t.Helper()

	t.Run("find returns typed meta", func(t *testing.T) {
		store := factory(t, []retrieval.Document[StructMeta]{{
			ID:      "doc-1",
			Content: "hello",
			Meta:    StructMeta{Tenant: tenantAcme},
		}})

		docs, err := store.FindByIDs(context.Background(), []string{"doc-1"})
		if err != nil {
			t.Fatalf("FindByIDs(): %v", err)
		}
		if len(docs) != 1 || docs[0].Meta.Tenant != tenantAcme {
			t.Fatalf("FindByIDs() = %#v, want tenant acme", docs)
		}
	})

	t.Run("delete by filter mutates state", func(t *testing.T) {
		store := factory(t, []retrieval.Document[StructMeta]{
			{ID: "doc-1", Content: "hello", Meta: StructMeta{Tenant: tenantAcme}},
			{ID: "doc-2", Content: "world", Meta: StructMeta{Tenant: "globex"}},
		})

		tenant, err := store.Schema().StringField("tenant")
		if err != nil {
			t.Fatalf("Schema().StringField(tenant): %v", err)
		}
		builder, err := filter.NewBuilder(store.Schema())
		if err != nil {
			t.Fatalf("NewBuilder(): %v", err)
		}
		cond, err := filter.Eq(builder, tenant, tenantAcme).Build()
		if err != nil {
			t.Fatalf("Build(): %v", err)
		}

		result, err := store.DeleteByFilter(context.Background(), cond)
		if err != nil {
			t.Fatalf("DeleteByFilter(): %v", err)
		}
		if result.Deleted != 1 {
			t.Fatalf("DeleteResult.Deleted = %d, want 1", result.Deleted)
		}
	})

	t.Run("undeclared filter rejects", func(t *testing.T) {
		store := factory(t, []retrieval.Document[StructMeta]{{ID: "doc-1", Content: "hello"}})
		_, err := store.Schema().StringField("missing")
		if !errors.Is(err, ragy.ErrInvalidArgument) {
			t.Fatalf("Schema().StringField(missing) error = %v, want invalid argument", err)
		}
	})
}
