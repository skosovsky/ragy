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
