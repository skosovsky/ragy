//go:build live

package multimodal

import (
	"os"
	"testing"

	"github.com/skosovsky/ragy/embedding"
	root "github.com/skosovsky/ragy/multimodal"
)

func TestLiveGeminiMultimodal(t *testing.T) {
	if os.Getenv("RAGY_LIVE_GEMINI") != "1" || os.Getenv("GEMINI_API_KEY") == "" ||
		os.Getenv("RAGY_GEMINI_IMAGE") == "" {
		t.Fatal("opt-in requires RAGY_LIVE_GEMINI=1, GEMINI_API_KEY and RAGY_GEMINI_IMAGE PNG path")
	}
	image, err := os.ReadFile(os.Getenv("RAGY_GEMINI_IMAGE"))
	if err != nil {
		t.Fatal(err)
	}
	space := profile()
	space.Dimension = 768
	client, err := New(Config{APIKey: os.Getenv("GEMINI_API_KEY"), Space: space})
	if err != nil {
		t.Fatal(err)
	}
	result, err := client.Embed(
		t.Context(),
		root.Request{
			Purpose: embedding.Document,
			Inputs:  []root.Input{{Parts: []root.Part{{Kind: root.PartBytes, MIME: "image/png", Bytes: image}}}},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Embeddings) != 1 {
		t.Fatal("missing embedding")
	}
}
