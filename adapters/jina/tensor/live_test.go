//go:build live

package tensor

import (
	"context"
	"os"
	"testing"

	"github.com/skosovsky/ragy/embedding"
	roottensor "github.com/skosovsky/ragy/tensor"
)

func TestLiveProviderSmoke(t *testing.T) {
	if os.Getenv("RAGY_LIVE_PROVIDERS") != "1" {
		t.Fatal("opt-in live provider smoke disabled")
	}
	key := os.Getenv("JINA_API_KEY")
	if key == "" {
		t.Fatal("provider credentials absent")
	}
	c, err := New(Config{APIKey: key, Space: testSpace()})
	if err != nil {
		t.Fatal(err)
	}
	result, err := c.Embed(
		context.Background(),
		roottensor.Request{Inputs: []string{"retrieval smoke"}, Purpose: embedding.Document},
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Embeddings) != 1 {
		t.Fatal("cardinality")
	}
}
