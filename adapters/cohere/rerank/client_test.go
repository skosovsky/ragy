package rerank

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/internal/contracttest"
	"github.com/skosovsky/ragy/retrieval"
)

func TestNewRejectsEmptyAPIKey(t *testing.T) {
	if _, err := New[contracttest.StructMeta](Config{Model: "rerank-v3.5"}); err == nil {
		t.Fatal("New() error = nil, want error")
	} else if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("New() error = %v, want invalid argument", err)
	}
}

func TestNewRejectsEmptyModel(t *testing.T) {
	t.Parallel()
	if _, err := New[contracttest.StructMeta](Config{APIKey: "key"}); err == nil {
		t.Fatal("New() error = nil, want error")
	} else if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("New() error = %v, want invalid argument", err)
	}
}

func TestRerankUsesProviderIndexes(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"results":[{"index":1,"relevance_score":0.9},{"index":0,"relevance_score":0.1}]}`))
	}))
	defer server.Close()

	client, err := New[contracttest.StructMeta](Config{APIKey: "key", Model: "rerank-v3.5", BaseURL: server.URL})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	rs := retrieval.NewResultSet(
		[]retrieval.Document[contracttest.StructMeta]{
			{ID: "a", Content: "alpha"},
			{ID: "b", Content: "beta"},
		},
		retrieval.DocumentIDResolver[contracttest.StructMeta]{},
	)
	out, err := client.Rerank(context.Background(), "q", rs)
	if err != nil {
		t.Fatalf("Rerank(): %v", err)
	}

	docs := out.Documents()
	if len(docs) != 2 || docs[0].ID != "b" {
		t.Fatalf("Rerank() order = %#v", out)
	}
}

func TestRerankPreservesMeta(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"results":[{"index":0,"relevance_score":0.9}]}`))
	}))
	defer server.Close()

	client, err := New[contracttest.StructMeta](Config{APIKey: "key", Model: "rerank-v3.5", BaseURL: server.URL})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	rs := retrieval.NewResultSet(
		[]retrieval.Document[contracttest.StructMeta]{{
			ID:      "a",
			Content: "alpha",
			Meta:    contracttest.StructMeta{Age: 7},
		}},
		retrieval.DocumentIDResolver[contracttest.StructMeta]{},
	)
	out, err := client.Rerank(context.Background(), "q", rs)
	if err != nil {
		t.Fatalf("Rerank(): %v", err)
	}

	if out.Documents()[0].Meta.Age != 7 {
		t.Fatalf("Rerank() age = %d, want 7", out.Documents()[0].Meta.Age)
	}
}

func TestRerankPreservesResolverOnEmptyQuery(t *testing.T) {
	t.Parallel()

	client, err := New[contracttest.StructMeta](Config{APIKey: "key", Model: "rerank-v3.5", BaseURL: "http://example"})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	resolver := contracttest.ContentMergeResolver[contracttest.StructMeta]{}
	rs := retrieval.NewResultSet(
		[]retrieval.Document[contracttest.StructMeta]{{ID: "a", Content: "merge-key"}},
		resolver,
	)
	out, err := client.Rerank(context.Background(), "", rs)
	contracttest.RequireErrorResultSet(t, out, err)
	if !errors.Is(err, ragy.ErrEmptyText) {
		t.Fatalf("Rerank() error = %v, want empty text", err)
	}

	merged, mergeErr := retrieval.NewResultSet(
		[]retrieval.Document[contracttest.StructMeta]{{ID: "b", Content: "merge-key", Score: 0.1}},
		resolver,
	).Merge(out)
	if mergeErr != nil {
		t.Fatalf("Merge(): %v", mergeErr)
	}
	if merged.Len() != 1 {
		t.Fatalf("merged Len() = %d, want 1 doc under input resolver merge key", merged.Len())
	}
}

func TestRerankNilResultSetReturnsEmpty(t *testing.T) {
	t.Parallel()

	client, err := New[contracttest.StructMeta](Config{
		APIKey:  "key",
		Model:   "rerank-v3.5",
		BaseURL: "http://127.0.0.1:1",
	})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	out, err := client.Rerank(context.Background(), "q", nil)
	if err != nil {
		t.Fatalf("Rerank(nil): %v", err)
	}
	if out == nil || !out.IsEmpty() {
		t.Fatalf("Rerank(nil) = %#v, want empty non-nil ResultSet", out)
	}
}

func TestRerankReturnsErrorOnHTTP4xx(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"message":"bad request"}`))
	}))
	defer server.Close()

	client, err := New[contracttest.StructMeta](Config{APIKey: "key", Model: "rerank-v3.5", BaseURL: server.URL})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	rs := retrieval.NewResultSet(
		[]retrieval.Document[contracttest.StructMeta]{{ID: "a", Content: "alpha"}},
		retrieval.DocumentIDResolver[contracttest.StructMeta]{},
	)
	out, err := client.Rerank(context.Background(), "q", rs)
	if err == nil {
		t.Fatal("Rerank() error = nil, want HTTP error")
	}
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Rerank() error = %v, want invalid argument", err)
	}
	if out.Len() != 1 || out.Documents()[0].ID != "a" {
		t.Fatalf("Rerank() docs = %#v, want preserved input", out.Documents())
	}
}

func TestRerankRejectsCardinalityMismatch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"results":[{"index":0,"relevance_score":0.9}]}`))
	}))
	defer server.Close()

	client, err := New[contracttest.StructMeta](Config{APIKey: "key", Model: "rerank-v3.5", BaseURL: server.URL})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	rs := retrieval.NewResultSet(
		[]retrieval.Document[contracttest.StructMeta]{
			{ID: "a", Content: "alpha"},
			{ID: "b", Content: "beta"},
		},
		retrieval.DocumentIDResolver[contracttest.StructMeta]{},
	)
	out, err := client.Rerank(context.Background(), "q", rs)
	if !errors.Is(err, ragy.ErrProtocol) {
		t.Fatalf("Rerank() error = %v, want protocol error", err)
	}
	if out.Len() != 2 {
		t.Fatalf("Rerank() Len() = %d, want preserved input docs", out.Len())
	}
}

func TestRerankPreservesInputOnTransportError(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name   string
		status int
	}{
		{"503", http.StatusServiceUnavailable},
		{"429", http.StatusTooManyRequests},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(tc.status)
			}))
			defer server.Close()

			client, err := New[contracttest.StructMeta](Config{
				APIKey:  "key",
				Model:   "rerank-v3.5",
				BaseURL: server.URL,
			})
			if err != nil {
				t.Fatalf("New(): %v", err)
			}

			rs := retrieval.NewResultSet(
				[]retrieval.Document[contracttest.StructMeta]{{ID: "a", Content: "alpha"}},
				retrieval.DocumentIDResolver[contracttest.StructMeta]{},
			)
			out, err := client.Rerank(context.Background(), "q", rs)
			if err == nil {
				t.Fatal("Rerank() error = nil, want transport error")
			}
			if !errors.Is(err, ragy.ErrUnavailable) {
				t.Fatalf("Rerank() error = %v, want unavailable", err)
			}
			if out.Len() != 1 || out.Documents()[0].ID != "a" {
				t.Fatalf("Rerank() docs = %#v, want preserved input", out.Documents())
			}
		})
	}
}

func TestRerankReturnsErrorOnTransportFailure(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {}))
	server.Close()

	client, err := New[contracttest.StructMeta](Config{
		APIKey:  "key",
		Model:   "rerank-v3.5",
		BaseURL: server.URL,
	})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	rs := retrieval.NewResultSet(
		[]retrieval.Document[contracttest.StructMeta]{{ID: "a", Content: "alpha"}},
		retrieval.DocumentIDResolver[contracttest.StructMeta]{},
	)
	out, err := client.Rerank(context.Background(), "q", rs)
	if err == nil {
		t.Fatal("Rerank() error = nil, want transport error")
	}
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("Rerank() error = %v, want unavailable", err)
	}
	if out.Len() != 1 || out.Documents()[0].ID != "a" {
		t.Fatalf("Rerank() docs = %#v, want preserved input on transport error", out.Documents())
	}
}

type contentMergeResolver[TMeta any] = contracttest.ContentMergeResolver[TMeta]

func TestRerankPreservesInputResolver(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	client, err := New[contracttest.StructMeta](Config{APIKey: "key", Model: "rerank-v3.5", BaseURL: server.URL})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	resolver := contentMergeResolver[contracttest.StructMeta]{}
	rs := retrieval.NewResultSet(
		[]retrieval.Document[contracttest.StructMeta]{{ID: "a", Content: "merge-key", Score: 0.9}},
		resolver,
	)
	out, err := client.Rerank(context.Background(), "q", rs)
	if err == nil {
		t.Fatal("Rerank() error = nil, want transport error")
	}
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("Rerank() error = %v, want unavailable", err)
	}
	if out.Len() != 1 {
		t.Fatalf("Rerank() docs = %#v, want preserved input", out.Documents())
	}

	merged, mergeErr := retrieval.NewResultSet(
		[]retrieval.Document[contracttest.StructMeta]{{ID: "b", Content: "merge-key", Score: 0.2}},
		resolver,
	).Merge(out)
	if mergeErr != nil {
		t.Fatalf("Merge(): %v", mergeErr)
	}
	if merged.Len() != 1 {
		t.Fatalf("merged Len() = %d, want 1 doc under input merge key", merged.Len())
	}
}

func TestRerankPartialConformance(t *testing.T) {
	contracttest.RunRerankPartialSuite(t, func(t *testing.T) contracttest.QueryRerankerLike[contracttest.StructMeta] {
		t.Helper()
		client, err := New[contracttest.StructMeta](Config{
			APIKey:  "key",
			Model:   "rerank-v3.5",
			BaseURL: "http://example",
		})
		if err != nil {
			t.Fatalf("New(): %v", err)
		}
		return client
	})
}
