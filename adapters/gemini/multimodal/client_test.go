package multimodal

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	ragy "github.com/skosovsky/ragy"
	rootmultimodal "github.com/skosovsky/ragy/multimodal"
)

func TestNewRejectsEmptyAPIKey(t *testing.T) {
	if _, err := New(Config{Model: "gemini-embedding-001"}); err == nil {
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

func TestNewAcceptsMinimalConfig(t *testing.T) {
	if _, err := New(Config{APIKey: "key", Model: "gemini-embedding-001"}); err != nil {
		t.Fatalf("New() error = %v", err)
	}
}

func TestEmbedRejectsProtocolIndexMismatch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"embeddings":[{"index":1,"vector":[0.1,0.2]}]}`))
	}))
	defer server.Close()

	client, err := New(Config{
		APIKey:  "key",
		Model:   "gemini-embedding-001",
		BaseURL: server.URL,
	})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	inputs := []rootmultimodal.Input{{
		Parts: []rootmultimodal.Part{{
			Kind: rootmultimodal.PartText,
			Text: "hello",
		}},
	}}

	if _, err := client.Embed(context.Background(), inputs); err == nil {
		t.Fatal("Embed() error = nil, want error")
	} else if !errors.Is(err, ragy.ErrProtocol) {
		t.Fatalf("Embed() error = %v, want protocol", err)
	}
}

type panicDoer struct {
	called bool
}

func (d *panicDoer) Do(_ *http.Request) (*http.Response, error) {
	d.called = true
	return nil, errors.New("unexpected request")
}

func TestEmbedRejectsInvalidInputBeforeHTTP(t *testing.T) {
	doer := &panicDoer{}
	client, err := New(Config{
		APIKey:     "key",
		Model:      "gemini-embedding-001",
		HTTPClient: doer,
	})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	_, err = client.Embed(context.Background(), []rootmultimodal.Input{{
		Parts: []rootmultimodal.Part{{
			Kind: rootmultimodal.PartBytes,
			Text: "broken",
		}},
	}})
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Embed() error = %v, want invalid argument", err)
	}
	if doer.called {
		t.Fatal("Embed() called HTTP client for invalid input")
	}
}

func TestEmbedReturnsErrorOnHTTP4xx(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"bad request"}}`))
	}))
	defer server.Close()

	client, err := New(Config{
		APIKey:  "key",
		Model:   "gemini-embedding-001",
		BaseURL: server.URL,
	})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	inputs := []rootmultimodal.Input{{
		Parts: []rootmultimodal.Part{{
			Kind: rootmultimodal.PartText,
			Text: "hello",
		}},
	}}

	if _, err := client.Embed(context.Background(), inputs); err == nil {
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

			client, err := New(Config{
				APIKey:  "key",
				Model:   "gemini-embedding-001",
				BaseURL: server.URL,
			})
			if err != nil {
				t.Fatalf("New(): %v", err)
			}

			inputs := []rootmultimodal.Input{{
				Parts: []rootmultimodal.Part{{
					Kind: rootmultimodal.PartText,
					Text: "hello",
				}},
			}}

			if _, err := client.Embed(context.Background(), inputs); err == nil {
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

	client, err := New(Config{
		APIKey:  "key",
		Model:   "gemini-embedding-001",
		BaseURL: server.URL,
	})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	inputs := []rootmultimodal.Input{{
		Parts: []rootmultimodal.Part{{
			Kind: rootmultimodal.PartText,
			Text: "hello",
		}},
	}}

	if _, err := client.Embed(context.Background(), inputs); err == nil {
		t.Fatal("Embed() error = nil, want transport error")
	} else if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("Embed() error = %v, want unavailable", err)
	}
}
