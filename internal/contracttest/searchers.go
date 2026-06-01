package contracttest

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
)

// Meta is a flexible metadata type for contract tests.
type Meta map[string]any

const wantedDocID = "doc-1"
const tenantAcme = "acme"

type DenseBackendFactory func(t *testing.T, docs []retrieval.Document[Meta]) retrieval.Backend[Meta]
type LexicalBackendFactory func(t *testing.T, docs []retrieval.Document[Meta]) retrieval.Backend[Meta]

func tenantCondition(t *testing.T, schema filter.Schema, value string) filter.Condition {
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

func RunDenseBackendSuite(t *testing.T, factory DenseBackendFactory) {
	t.Helper()

	t.Run("valid docs pass through", func(t *testing.T) {
		backend := factory(t, []retrieval.Document[Meta]{{ID: wantedDocID, Content: "hello"}})
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
		backend := factory(t, []retrieval.Document[Meta]{{
			ID:      wantedDocID,
			Content: "hello",
			Meta:    Meta{"tenant": tenantAcme},
		}})
		schema := schemaFromBackend(t, backend)
		out, err := backend.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
			Vector:  []float32{1},
			Filters: tenantCondition(t, schema, tenantAcme),
		})
		if err != nil {
			t.Fatalf("Retrieve(): %v", err)
		}
		if len(out) != 1 || out[0].ID != wantedDocID {
			t.Fatalf("Retrieve() = %#v, want doc-1", out)
		}
	})

	t.Run("invalid docs reject", func(t *testing.T) {
		backend := factory(t, []retrieval.Document[Meta]{{Content: "broken"}})
		_, err := backend.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
			Vector: []float32{1},
		})
		if err == nil {
			t.Fatal("Retrieve() error = nil, want error")
		}
	})

	t.Run("no results returns nil", func(t *testing.T) {
		backend := factory(t, nil)
		out, err := backend.Retrieve(context.Background(), "", retrieval.RetrieveOptions{
			Vector: []float32{1},
		})
		if err != nil {
			t.Fatalf("Retrieve(): %v", err)
		}
		if out != nil {
			t.Fatalf("Retrieve() = %#v, want nil", out)
		}
	})

	t.Run("undeclared filter rejects", func(t *testing.T) {
		backend := factory(t, []retrieval.Document[Meta]{{ID: wantedDocID}})
		schema := schemaFromBackend(t, backend)
		_, err := schema.StringField("missing")
		if !errors.Is(err, ragy.ErrInvalidArgument) {
			t.Fatalf("Schema().StringField(missing) error = %v, want invalid argument", err)
		}
	})
}

func RunLexicalBackendSuite(t *testing.T, factory LexicalBackendFactory) {
	t.Helper()

	t.Run("valid docs pass through", func(t *testing.T) {
		backend := factory(t, []retrieval.Document[Meta]{{ID: wantedDocID, Content: "hello"}})
		out, err := backend.Retrieve(context.Background(), "hello", retrieval.RetrieveOptions{})
		if err != nil {
			t.Fatalf("Retrieve(): %v", err)
		}
		if len(out) != 1 || out[0].ID != wantedDocID {
			t.Fatalf("Retrieve() = %#v, want doc-1", out)
		}
	})

	t.Run("declared filter built from builder passes", func(t *testing.T) {
		backend := factory(t, []retrieval.Document[Meta]{{
			ID:      wantedDocID,
			Content: "hello",
			Meta:    Meta{"tenant": tenantAcme},
		}})
		schema := schemaFromBackend(t, backend)
		out, err := backend.Retrieve(context.Background(), "hello", retrieval.RetrieveOptions{
			Filters: tenantCondition(t, schema, tenantAcme),
		})
		if err != nil {
			t.Fatalf("Retrieve(): %v", err)
		}
		if len(out) != 1 || out[0].ID != wantedDocID {
			t.Fatalf("Retrieve() = %#v, want doc-1", out)
		}
	})

	t.Run("invalid docs reject", func(t *testing.T) {
		backend := factory(t, []retrieval.Document[Meta]{{Content: "broken"}})
		_, err := backend.Retrieve(context.Background(), "hello", retrieval.RetrieveOptions{})
		if err == nil {
			t.Fatal("Retrieve() error = nil, want error")
		}
	})

	t.Run("no results returns nil", func(t *testing.T) {
		backend := factory(t, nil)
		out, err := backend.Retrieve(context.Background(), "hello", retrieval.RetrieveOptions{})
		if err != nil {
			t.Fatalf("Retrieve(): %v", err)
		}
		if out != nil {
			t.Fatalf("Retrieve() = %#v, want nil", out)
		}
	})

	t.Run("undeclared filter rejects", func(t *testing.T) {
		backend := factory(t, []retrieval.Document[Meta]{{ID: wantedDocID}})
		schema := schemaFromBackend(t, backend)
		_, err := schema.StringField("missing")
		if !errors.Is(err, ragy.ErrInvalidArgument) {
			t.Fatalf("Schema().StringField(missing) error = %v, want invalid argument", err)
		}
	})
}

type schemaProvider interface {
	Schema() filter.Schema
}

func schemaFromTypedBackend[TMeta any](t *testing.T, backend retrieval.Backend[TMeta]) filter.Schema {
	t.Helper()

	provider, ok := backend.(schemaProvider)
	if !ok {
		t.Fatal("backend does not expose Schema()")
	}
	return provider.Schema()
}

func schemaFromBackend(t *testing.T, backend retrieval.Backend[Meta]) filter.Schema {
	return schemaFromTypedBackend(t, backend)
}
