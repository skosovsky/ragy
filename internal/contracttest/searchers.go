package contracttest

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/lexical"
	"github.com/skosovsky/ragy/tensor"
)

type DenseSearcherFactory func(t *testing.T, docs []ragy.Document) dense.Searcher
type LexicalSearcherFactory func(t *testing.T, docs []ragy.Document) lexical.Searcher
type TensorSearcherFactory func(t *testing.T, docs []ragy.Document) tensor.Searcher

const wantedDocID = "doc-1"

func tenantFilter(t *testing.T, schema filter.Schema, value string) filter.IR {
	t.Helper()

	tenant, err := schema.StringField("tenant")
	if err != nil {
		t.Fatalf("Schema().StringField(tenant): %v", err)
	}

	expr, err := filter.Normalize(filter.Equal(tenant, value))
	if err != nil {
		t.Fatalf("Normalize(): %v", err)
	}

	return expr
}

func RunDenseSearcherSuite(t *testing.T, factory DenseSearcherFactory) {
	t.Helper()

	t.Run("valid docs pass through", func(t *testing.T) {
		searcher := factory(t, []ragy.Document{{ID: wantedDocID, Content: "hello"}})
		out, err := searcher.Search(context.Background(), dense.Request{
			Vector: []float32{1},
			Filter: nil,
			Page:   nil,
		})
		if err != nil {
			t.Fatalf("Search(): %v", err)
		}
		if len(out) != 1 || out[0].ID != wantedDocID {
			t.Fatalf("Search() = %#v, want doc-1", out)
		}
	})

	t.Run("declared filter built from schema passes", func(t *testing.T) {
		searcher := factory(t, []ragy.Document{{
			ID:         wantedDocID,
			Content:    "hello",
			Attributes: ragy.Attributes{"tenant": "acme"},
		}})
		out, err := searcher.Search(context.Background(), dense.Request{
			Vector: []float32{1},
			Filter: tenantFilter(t, searcher.Schema(), "acme"),
			Page:   nil,
		})
		if err != nil {
			t.Fatalf("Search(): %v", err)
		}
		if len(out) != 1 || out[0].ID != wantedDocID {
			t.Fatalf("Search() = %#v, want doc-1", out)
		}
	})

	t.Run("invalid docs reject", func(t *testing.T) {
		searcher := factory(t, []ragy.Document{{Content: "broken"}})
		_, err := searcher.Search(context.Background(), dense.Request{
			Vector: []float32{1},
			Filter: nil,
			Page:   nil,
		})
		if err == nil {
			t.Fatal("Search() error = nil, want error")
		}
	})

	t.Run("no results returns nil", func(t *testing.T) {
		searcher := factory(t, nil)
		out, err := searcher.Search(context.Background(), dense.Request{
			Vector: []float32{1},
			Filter: nil,
			Page:   nil,
		})
		if err != nil {
			t.Fatalf("Search(): %v", err)
		}
		if out != nil {
			t.Fatalf("Search() = %#v, want nil", out)
		}
	})

	t.Run("undeclared filter rejects", func(t *testing.T) {
		searcher := factory(t, []ragy.Document{{ID: wantedDocID}})
		_, err := searcher.Schema().StringField("missing")
		if !errors.Is(err, ragy.ErrInvalidArgument) {
			t.Fatalf("Schema().StringField(missing) error = %v, want invalid argument", err)
		}
	})
}

func RunLexicalSearcherSuite(t *testing.T, factory LexicalSearcherFactory) {
	t.Helper()

	t.Run("valid docs pass through", func(t *testing.T) {
		searcher := factory(t, []ragy.Document{{ID: wantedDocID, Content: "hello"}})
		out, err := searcher.Search(context.Background(), lexical.Request{
			Text:   "hello",
			Filter: nil,
			Page:   nil,
		})
		if err != nil {
			t.Fatalf("Search(): %v", err)
		}
		if len(out) != 1 || out[0].ID != wantedDocID {
			t.Fatalf("Search() = %#v, want doc-1", out)
		}
	})

	t.Run("declared filter built from schema passes", func(t *testing.T) {
		searcher := factory(t, []ragy.Document{{
			ID:         wantedDocID,
			Content:    "hello",
			Attributes: ragy.Attributes{"tenant": "acme"},
		}})
		out, err := searcher.Search(context.Background(), lexical.Request{
			Text:   "hello",
			Filter: tenantFilter(t, searcher.Schema(), "acme"),
			Page:   nil,
		})
		if err != nil {
			t.Fatalf("Search(): %v", err)
		}
		if len(out) != 1 || out[0].ID != wantedDocID {
			t.Fatalf("Search() = %#v, want doc-1", out)
		}
	})

	t.Run("invalid docs reject", func(t *testing.T) {
		searcher := factory(t, []ragy.Document{{Content: "broken"}})
		_, err := searcher.Search(context.Background(), lexical.Request{
			Text:   "hello",
			Filter: nil,
			Page:   nil,
		})
		if err == nil {
			t.Fatal("Search() error = nil, want error")
		}
	})

	t.Run("no results returns nil", func(t *testing.T) {
		searcher := factory(t, nil)
		out, err := searcher.Search(context.Background(), lexical.Request{
			Text:   "hello",
			Filter: nil,
			Page:   nil,
		})
		if err != nil {
			t.Fatalf("Search(): %v", err)
		}
		if out != nil {
			t.Fatalf("Search() = %#v, want nil", out)
		}
	})

	t.Run("undeclared filter rejects", func(t *testing.T) {
		searcher := factory(t, []ragy.Document{{ID: wantedDocID}})
		_, err := searcher.Schema().StringField("missing")
		if !errors.Is(err, ragy.ErrInvalidArgument) {
			t.Fatalf("Schema().StringField(missing) error = %v, want invalid argument", err)
		}
	})
}

func RunTensorSearcherSuite(t *testing.T, factory TensorSearcherFactory) {
	t.Helper()

	t.Run("valid docs pass through", func(t *testing.T) {
		searcher := factory(t, []ragy.Document{{ID: wantedDocID, Content: "hello"}})
		out, err := searcher.Search(context.Background(), tensor.Request{
			Query:  tensor.Tensor{{1}},
			Filter: nil,
			Page:   nil,
		})
		if err != nil {
			t.Fatalf("Search(): %v", err)
		}
		if len(out) != 1 || out[0].ID != wantedDocID {
			t.Fatalf("Search() = %#v, want doc-1", out)
		}
	})

	t.Run("declared filter built from schema passes", func(t *testing.T) {
		searcher := factory(t, []ragy.Document{{
			ID:         wantedDocID,
			Content:    "hello",
			Attributes: ragy.Attributes{"tenant": "acme"},
		}})
		out, err := searcher.Search(context.Background(), tensor.Request{
			Query:  tensor.Tensor{{1}},
			Filter: tenantFilter(t, searcher.Schema(), "acme"),
			Page:   nil,
		})
		if err != nil {
			t.Fatalf("Search(): %v", err)
		}
		if len(out) != 1 || out[0].ID != wantedDocID {
			t.Fatalf("Search() = %#v, want doc-1", out)
		}
	})

	t.Run("invalid docs reject", func(t *testing.T) {
		searcher := factory(t, []ragy.Document{{Content: "broken"}})
		_, err := searcher.Search(context.Background(), tensor.Request{
			Query:  tensor.Tensor{{1}},
			Filter: nil,
			Page:   nil,
		})
		if err == nil {
			t.Fatal("Search() error = nil, want error")
		}
	})

	t.Run("no results returns nil", func(t *testing.T) {
		searcher := factory(t, nil)
		out, err := searcher.Search(context.Background(), tensor.Request{
			Query:  tensor.Tensor{{1}},
			Filter: nil,
			Page:   nil,
		})
		if err != nil {
			t.Fatalf("Search(): %v", err)
		}
		if out != nil {
			t.Fatalf("Search() = %#v, want nil", out)
		}
	})

	t.Run("undeclared filter rejects", func(t *testing.T) {
		searcher := factory(t, []ragy.Document{{ID: wantedDocID}})
		_, err := searcher.Schema().StringField("missing")
		if !errors.Is(err, ragy.ErrInvalidArgument) {
			t.Fatalf("Schema().StringField(missing) error = %v, want invalid argument", err)
		}
	})
}
