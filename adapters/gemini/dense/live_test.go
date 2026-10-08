//go:build live

package dense

import (
	"os"
	"testing"

	root "github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/embedding"
)

func TestLiveGemini(t *testing.T) {
	if os.Getenv("RAGY_LIVE_GEMINI") != "1" || os.Getenv("GEMINI_API_KEY") == "" {
		t.Fatal("opt-in live request requires RAGY_LIVE_GEMINI=1 and GEMINI_API_KEY")
	}
	space := profile("gemini-embedding-001")
	space.Dimension = 768
	client, err := New(Config{APIKey: os.Getenv("GEMINI_API_KEY"), Space: space})
	if err != nil {
		t.Fatal(err)
	}
	result, err := client.Embed(t.Context(), root.Request{Inputs: []string{"test retrieval"}, Purpose: embedding.Query})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Embeddings) != 1 {
		t.Fatal("missing embedding")
	}
}
