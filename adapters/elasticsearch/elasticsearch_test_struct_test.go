package elasticsearch

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/internal/contracttest"
	"github.com/skosovsky/ragy/retrieval"
)

func TestRetrieveUnmarshalsStructMeta(t *testing.T) {
	t.Parallel()

	client := &fakeClient{
		hits: []Hit{{
			ID:    "doc-1",
			Score: 0.9,
			Source: map[string]any{
				"content": "hello",
				"tenant":  "acme",
			},
		}},
	}

	schema := schemaWithContentAndTenant(t)
	store, err := New[contracttest.TenantOnlyMeta](client, Config[contracttest.TenantOnlyMeta]{
		Index:        "docs",
		SearchFields: []string{"content"},
		Schema:       schema,
	}, contracttest.JSONCodec[contracttest.TenantOnlyMeta](t, schema))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	out, err := retrieveStore(context.Background(), store, "hello", retrieval.RetrieveOptions{TopK: 10})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	docs := out.Documents()
	if len(docs) != 1 || docs[0].Meta.Tenant != "acme" {
		t.Fatalf("Retrieve() = %#v, want tenant acme", docs)
	}
}

func TestRetrieveRejectsIncompatibleStructMeta(t *testing.T) {
	t.Parallel()

	client := &fakeClient{
		hits: []Hit{{
			ID:    "doc-1",
			Score: 0.9,
			Source: map[string]any{
				"content": "hello",
				"tenant":  123,
			},
		}},
	}

	schema := schemaWithContentAndTenant(t)
	store, err := New[contracttest.TenantOnlyMeta](client, Config[contracttest.TenantOnlyMeta]{
		Index:        "docs",
		SearchFields: []string{"content"},
		Schema:       schema,
	}, contracttest.JSONCodec[contracttest.TenantOnlyMeta](t, schema))
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	out, err := retrieveStore(context.Background(), store, "hello", retrieval.RetrieveOptions{TopK: 10})
	contracttest.RequireErrorResultSet(t, out, err)
	if !errors.Is(err, ragy.ErrProtocol) {
		t.Fatalf("Retrieve() error = %v, want protocol", err)
	}
}
