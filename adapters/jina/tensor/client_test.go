package tensor

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestEmbedRejectsProtocolIndexMismatch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"data":[{"index":1,"tensor":[[0.1,0.2]]}]}`))
	}))
	defer server.Close()

	client, err := New(Config{APIKey: "key", Model: "jina-colbert-v2", BaseURL: server.URL})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}

	if _, err := client.Embed(context.Background(), []string{"hello"}); err == nil {
		t.Fatal("Embed() error = nil, want error")
	}
}
