package dense

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestNewRejectsEmptyAPIKey(t *testing.T) {
	if _, err := New(Config{Model: "text-embedding-3-small"}); err == nil {
		t.Fatal("New() error = nil, want error")
	}
}

func TestEmbedRejectsProtocolIndexMismatch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"data":[{"index":1,"embedding":[0.1,0.2]}]}`))
	}))
	defer server.Close()

	client, err := New(Config{
		APIKey:  "key",
		Model:   "text-embedding-3-small",
		BaseURL: server.URL,
	})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	if _, err := client.Embed(context.Background(), []string{"hello"}); err == nil {
		t.Fatal("Embed() error = nil, want error")
	}
}
