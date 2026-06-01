package elasticsearch

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/internal/contracttest"
	"github.com/skosovsky/ragy/retrieval"
)

type fakeClient struct {
	body map[string]any
	hits []Hit
}

func (f *fakeClient) Search(_ context.Context, _ string, body map[string]any) ([]Hit, error) {
	f.body = body
	return f.hits, nil
}

func schemaWithContent(t *testing.T) filter.Schema {
	t.Helper()
	return filter.EmptySchema()
}

func schemaWithContentAndTenant(t *testing.T) filter.Schema {
	t.Helper()

	builder := filter.NewSchema()
	if _, err := builder.String("tenant"); err != nil {
		t.Fatalf("builder.String(tenant): %v", err)
	}

	schema, err := builder.Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	return schema
}

func TestRetrieveProjectsCanonicalDocumentShape(t *testing.T) {
	client := &fakeClient{
		hits: []Hit{{
			ID:    "doc-1",
			Score: 4,
			Source: map[string]any{
				"content": "hello",
				"tenant":  "acme",
			},
		}},
	}

	store, err := New[contracttest.Meta](client, Config{
		Index:        "docs",
		SearchFields: []string{"content"},
		Schema:       schemaWithContentAndTenant(t),
	})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	schema := schemaWithContentAndTenant(t)
	tenant, err := schema.StringField("tenant")
	if err != nil {
		t.Fatalf("schema.StringField(tenant): %v", err)
	}
	filterBuilder, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatalf("NewBuilder(): %v", err)
	}
	cond, err := filter.Eq(filterBuilder, tenant, "acme").Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	out, err := store.Retrieve(context.Background(), "hello", retrieval.RetrieveOptions{
		Filters: cond,
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}

	if len(out) != 1 {
		t.Fatalf("len(out) = %d, want 1", len(out))
	}

	if _, ok := out[0].Meta["content"]; ok {
		t.Fatal("document meta unexpectedly contains content")
	}
	if got := out[0].Meta["tenant"]; got != "acme" {
		t.Fatalf("document tenant = %#v, want acme", got)
	}
}

func TestRetrieveReturnsNilMetaWhenOnlyContentIsPresent(t *testing.T) {
	client := &fakeClient{
		hits: []Hit{{
			ID:    "doc-1",
			Score: 4,
			Source: map[string]any{
				"content": "hello",
			},
		}},
	}

	store, err := New[contracttest.Meta](client, Config{
		Index:        "docs",
		SearchFields: []string{"content"},
		Schema:       schemaWithContent(t),
	})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	out, err := store.Retrieve(context.Background(), "hello", retrieval.RetrieveOptions{})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}

	if len(out) != 1 {
		t.Fatalf("len(out) = %d, want 1", len(out))
	}

	if len(out[0].Meta) != 0 {
		t.Fatalf("document meta = %#v, want empty", out[0].Meta)
	}
}

func TestRetrieveRejectsMissingContent(t *testing.T) {
	client := &fakeClient{
		hits: []Hit{{
			ID:     "doc-1",
			Score:  4,
			Source: map[string]any{"tenant": "acme"},
		}},
	}

	store, err := New[contracttest.Meta](client, Config{
		Index:        "docs",
		SearchFields: []string{"content"},
		Schema:       schemaWithContentAndTenant(t),
	})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	_, err = store.Retrieve(context.Background(), "hello", retrieval.RetrieveOptions{})
	if !errors.Is(err, ragy.ErrProtocol) {
		t.Fatalf("Retrieve() error = %v, want protocol error", err)
	}
}

func TestRetrieveRejectsNonStringContent(t *testing.T) {
	client := &fakeClient{
		hits: []Hit{{
			ID:    "doc-1",
			Score: 4,
			Source: map[string]any{
				"content": 7,
			},
		}},
	}

	store, err := New[contracttest.Meta](client, Config{
		Index:        "docs",
		SearchFields: []string{"content"},
		Schema:       schemaWithContent(t),
	})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	_, err = store.Retrieve(context.Background(), "hello", retrieval.RetrieveOptions{})
	if !errors.Is(err, ragy.ErrProtocol) {
		t.Fatalf("Retrieve() error = %v, want protocol error", err)
	}
}

func TestRetrieveRejectsUndeclaredFilterField(t *testing.T) {
	client := &fakeClient{}
	store, err := New[contracttest.Meta](client, Config{
		Index:        "docs",
		SearchFields: []string{"content"},
		Schema:       schemaWithContent(t),
	})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	foreign := filter.NewSchema()
	tenant, err := foreign.String("tenant")
	if err != nil {
		t.Fatalf("foreign.String(tenant): %v", err)
	}
	foreignSchema, err := foreign.Build()
	if err != nil {
		t.Fatalf("foreign.Build(): %v", err)
	}
	filterBuilder, err := filter.NewBuilder(foreignSchema)
	if err != nil {
		t.Fatalf("NewBuilder(): %v", err)
	}
	cond, err := filter.Eq(filterBuilder, tenant, "acme").Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	if _, err := store.Retrieve(context.Background(), "hello", retrieval.RetrieveOptions{
		Filters: cond,
	}); err == nil {
		t.Fatal("Retrieve() error = nil, want error")
	}
	if client.body != nil {
		t.Fatalf("body = %#v, want no backend call", client.body)
	}
}

func TestNewRejectsInvalidIndexName(t *testing.T) {
	if _, err := New[contracttest.Meta](&fakeClient{}, Config{
		Index:        "1Bad",
		SearchFields: []string{"content"},
		Schema:       schemaWithContent(t),
	}); err == nil {
		t.Fatal("New() error = nil, want error")
	}
}

func TestNewRejectsInvalidSearchField(t *testing.T) {
	if _, err := New[contracttest.Meta](&fakeClient{}, Config{
		Index:        "docs",
		SearchFields: []string{"1bad"},
		Schema:       schemaWithContent(t),
	}); err == nil {
		t.Fatal("New() error = nil, want error")
	}
}
