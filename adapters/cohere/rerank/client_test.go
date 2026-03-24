package rerank

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	ragy "github.com/skosovsky/ragy"
)

func TestRerankUsesProviderIndexes(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"results":[{"index":1,"relevance_score":0.9},{"index":0,"relevance_score":0.1}]}`))
	}))
	defer server.Close()

	client, err := New(Config{APIKey: "key", Model: "rerank-v3.5", BaseURL: server.URL})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	out, err := client.Rerank(context.Background(), "q", []ragy.Document{
		{ID: "a", Content: "alpha"},
		{ID: "b", Content: "beta"},
	})
	if err != nil {
		t.Fatalf("Rerank(): %v", err)
	}

	if len(out) != 2 || out[0].ID != "b" {
		t.Fatalf("Rerank() order = %#v", out)
	}
}

func TestRerankReturnsCanonicalizedAttributes(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"results":[{"index":0,"relevance_score":0.9}]}`))
	}))
	defer server.Close()

	client, err := New(Config{APIKey: "key", Model: "rerank-v3.5", BaseURL: server.URL})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	out, err := client.Rerank(context.Background(), "q", []ragy.Document{{
		ID:         "a",
		Content:    "alpha",
		Attributes: ragy.Attributes{"age": json.Number("7")},
	}})
	if err != nil {
		t.Fatalf("Rerank(): %v", err)
	}

	value, ok := out[0].Attributes["age"].(int64)
	if !ok || value != 7 {
		t.Fatalf("Rerank() age = %#v, want int64(7)", out[0].Attributes["age"])
	}
}
