package elasticsearch

import (
	"context"
	"testing"

	"github.com/skosovsky/ragy/retrieval"
)

type tenantMeta struct {
	Tenant string `json:"tenant"`
}

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

	store, err := New[tenantMeta](client, Config{
		Index:        "docs",
		SearchFields: []string{"content"},
		Schema:       schemaWithContentAndTenant(t),
	})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	out, err := store.Retrieve(context.Background(), "hello", retrieval.RetrieveOptions{})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if len(out) != 1 || out[0].Meta.Tenant != "acme" {
		t.Fatalf("Retrieve() = %#v, want tenant acme", out)
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

	store, err := New[tenantMeta](client, Config{
		Index:        "docs",
		SearchFields: []string{"content"},
		Schema:       schemaWithContentAndTenant(t),
	})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	_, err = store.Retrieve(context.Background(), "hello", retrieval.RetrieveOptions{})
	if err == nil {
		t.Fatal("Retrieve() error = nil, want decode error")
	}
}
