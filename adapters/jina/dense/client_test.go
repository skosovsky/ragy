package dense

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	ragy "github.com/skosovsky/ragy"
)

func TestNewRejectsEmptyAPIKey(t *testing.T) {
	if _, err := New(Config{Model: "jina-embeddings-v3"}); err == nil {
		t.Fatal("New() error = nil, want error")
	} else if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("New() error = %v, want invalid argument", err)
	}
}

func TestNewRejectsEmptyModel(t *testing.T) {
	t.Parallel()
	if _, err := New(Config{APIKey: "key"}); err == nil {
		t.Fatal("New() error = nil, want error")
	} else if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("New() error = %v, want invalid argument", err)
	}
}

func TestEmbedRejectsDuplicateIndexes(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"data":[{"index":0,"embedding":[0.1]},{"index":0,"embedding":[0.2]}]}`))
	}))
	defer server.Close()

	client, err := New(Config{APIKey: "key", Model: "jina-embeddings-v3", BaseURL: server.URL})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	if _, err := client.Embed(context.Background(), []string{"hello", "world"}); err == nil {
		t.Fatal("Embed() error = nil, want error")
	} else if !errors.Is(err, ragy.ErrProtocol) {
		t.Fatalf("Embed() error = %v, want protocol", err)
	}
}

func TestEmbedReturnsErrorOnHTTP4xx(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"detail":"bad request"}`))
	}))
	defer server.Close()

	client, err := New(Config{APIKey: "key", Model: "jina-embeddings-v3", BaseURL: server.URL})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	if _, err := client.Embed(context.Background(), []string{"hello"}); err == nil {
		t.Fatal("Embed() error = nil, want HTTP error")
	} else if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Embed() error = %v, want invalid argument", err)
	}
}

func TestEmbedReturnsErrorOnHTTP503And429(t *testing.T) {
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

			client, err := New(Config{APIKey: "key", Model: "jina-embeddings-v3", BaseURL: server.URL})
			if err != nil {
				t.Fatalf("New(): %v", err)
			}

			if _, err := client.Embed(context.Background(), []string{"hello"}); err == nil {
				t.Fatal("Embed() error = nil, want HTTP error")
			} else if !errors.Is(err, ragy.ErrUnavailable) {
				t.Fatalf("Embed() error = %v, want unavailable", err)
			}
		})
	}
}

func TestEmbedReturnsErrorOnTransportFailure(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {}))
	server.Close()

	client, err := New(Config{APIKey: "key", Model: "jina-embeddings-v3", BaseURL: server.URL})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	if _, err := client.Embed(context.Background(), []string{"hello"}); err == nil {
		t.Fatal("Embed() error = nil, want transport error")
	} else if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("Embed() error = %v, want unavailable", err)
	}
}
