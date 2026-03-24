package elasticsearch

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/lexical"
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

func TestSearchProjectsCanonicalDocumentShape(t *testing.T) {
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

	searcher, err := New(client, Config{
		Index:        "docs",
		SearchFields: []string{"content"},
		Schema:       schemaWithContentAndTenant(t),
	})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	filterSchema := filter.NewSchema()
	tenant, err := filterSchema.String("tenant")
	if err != nil {
		t.Fatalf("schema.String(tenant): %v", err)
	}
	expr, err := filter.Normalize(filter.Equal(tenant, "acme"))
	if err != nil {
		t.Fatalf("Normalize(): %v", err)
	}

	out, err := searcher.Search(context.Background(), lexical.Request{Text: "hello", Filter: expr})
	if err != nil {
		t.Fatalf("Search(): %v", err)
	}

	if len(out) != 1 {
		t.Fatalf("len(out) = %d, want 1", len(out))
	}

	if _, ok := out[0].Attributes["content"]; ok {
		t.Fatal("document attributes unexpectedly contain content")
	}
	if got := out[0].Attributes["tenant"]; got != "acme" {
		t.Fatalf("document tenant = %#v, want acme", got)
	}
}

func TestSearchReturnsNilAttributesWhenOnlyContentIsPresent(t *testing.T) {
	client := &fakeClient{
		hits: []Hit{{
			ID:    "doc-1",
			Score: 4,
			Source: map[string]any{
				"content": "hello",
			},
		}},
	}

	searcher, err := New(client, Config{
		Index:        "docs",
		SearchFields: []string{"content"},
		Schema:       schemaWithContent(t),
	})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	out, err := searcher.Search(context.Background(), lexical.Request{Text: "hello"})
	if err != nil {
		t.Fatalf("Search(): %v", err)
	}

	if len(out) != 1 {
		t.Fatalf("len(out) = %d, want 1", len(out))
	}

	if out[0].Attributes != nil {
		t.Fatalf("document attributes = %#v, want nil", out[0].Attributes)
	}
}

func TestSearchRejectsMissingContent(t *testing.T) {
	client := &fakeClient{
		hits: []Hit{{
			ID:     "doc-1",
			Score:  4,
			Source: map[string]any{"tenant": "acme"},
		}},
	}

	searcher, err := New(client, Config{
		Index:        "docs",
		SearchFields: []string{"content"},
		Schema:       schemaWithContentAndTenant(t),
	})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	_, err = searcher.Search(context.Background(), lexical.Request{Text: "hello"})
	if !errors.Is(err, ragy.ErrProtocol) {
		t.Fatalf("Search() error = %v, want protocol error", err)
	}
}

func TestSearchRejectsNonStringContent(t *testing.T) {
	client := &fakeClient{
		hits: []Hit{{
			ID:    "doc-1",
			Score: 4,
			Source: map[string]any{
				"content": 7,
			},
		}},
	}

	searcher, err := New(client, Config{
		Index:        "docs",
		SearchFields: []string{"content"},
		Schema:       schemaWithContent(t),
	})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	_, err = searcher.Search(context.Background(), lexical.Request{Text: "hello"})
	if !errors.Is(err, ragy.ErrProtocol) {
		t.Fatalf("Search() error = %v, want protocol error", err)
	}
}

func TestSearchRejectsUndeclaredFilterField(t *testing.T) {
	client := &fakeClient{}
	searcher, err := New(client, Config{
		Index:        "docs",
		SearchFields: []string{"content"},
		Schema:       schemaWithContent(t),
	})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	filterSchema := filter.NewSchema()
	tenant, err := filterSchema.String("tenant")
	if err != nil {
		t.Fatalf("schema.String(tenant): %v", err)
	}
	expr, err := filter.Normalize(filter.Equal(tenant, "acme"))
	if err != nil {
		t.Fatalf("Normalize(): %v", err)
	}

	if _, err := searcher.Search(context.Background(), lexical.Request{Text: "hello", Filter: expr}); err == nil {
		t.Fatal("Search() error = nil, want error")
	}
	if client.body != nil {
		t.Fatalf("body = %#v, want no backend call", client.body)
	}
}

func TestNewRejectsInvalidIndexName(t *testing.T) {
	if _, err := New(&fakeClient{}, Config{
		Index:        "1Bad",
		SearchFields: []string{"content"},
		Schema:       schemaWithContent(t),
	}); err == nil {
		t.Fatal("New() error = nil, want error")
	}
}

func TestNewRejectsInvalidSearchField(t *testing.T) {
	if _, err := New(&fakeClient{}, Config{
		Index:        "docs",
		SearchFields: []string{"1bad"},
		Schema:       schemaWithContent(t),
	}); err == nil {
		t.Fatal("New() error = nil, want error")
	}
}
