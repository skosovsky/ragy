//go:build live

package rerank

import (
	"context"
	"os"
	"testing"

	"github.com/skosovsky/ragy/retrieval"
)

func TestLiveProviderSmoke(t *testing.T) {
	// Arrange. A paid call requires explicit opt-in and host-selected model.
	key, model := os.Getenv("COHERE_API_KEY"), os.Getenv("COHERE_RERANK_MODEL")
	if os.Getenv("RAGY_PROVIDER_SMOKE") != "1" || key == "" || model == "" {
		t.Fatal("set RAGY_PROVIDER_SMOKE=1, COHERE_API_KEY and COHERE_RERANK_MODEL for a paid live call")
	}
	client, err := New[struct{}](Config{APIKey: key, Model: model})
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	out, err := client.RerankWithUsage(context.Background(), retrieval.UnrestrictedRead(), "alpha", fixtureDocs())
	// Assert.
	if err != nil || out.Documents.Len() != 2 {
		t.Fatalf("live rerank failed: %v", err)
	}
}
