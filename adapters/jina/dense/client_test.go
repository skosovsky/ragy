package dense

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

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
	}
}
