package dense

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	root "github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/embedding"
)

// TestSDKL03DiscardedFields documents genai v1.73.0, not corrected upstream behavior.
func TestSDKL03DiscardedFields(t *testing.T) {
	for _, extra := range []string{`,"usageMetadata":{"promptTokenCount":7}`, `,"model":"contradictory"`, `,"shape":[2]`} {
		t.Run(extra, func(t *testing.T) {
			// Arrange.
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				_, _ = io.WriteString(w, `{"embeddings":[{"values":[1,2]}]`+extra+`}`)
			}))
			t.Cleanup(server.Close)
			client, err := New(Config{SDK: newSDK("fixture", server.URL, nil), Space: profile("gemini-embedding-001")})
			if err != nil {
				t.Fatal(err)
			}
			// Act.
			result, err := client.Embed(
				context.Background(),
				root.Request{Inputs: []string{"text"}, Purpose: embedding.Query},
			)
			// Assert: absent public SDK counters remain unknown; echo is not an attestation.
			if err != nil || len(result.Embeddings) != 1 || result.Usage.InputTokensKnown {
				t.Fatalf("SDK-L03 changed: %+v %v", result, err)
			}
		})
	}
}

func TestSDKL01ResponseRepresentationBound(t *testing.T) {
	// Arrange: wire bytes exceed cap but the public SDK DTO fits.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, strings.Repeat(" ", 4096)+`{"embeddings":[{"values":[1,2],"shape":[2]}]}`)
	}))
	t.Cleanup(server.Close)
	client, err := New(
		Config{
			SDK:    newSDK("fixture", server.URL, nil),
			Space:  profile("gemini-embedding-001"),
			Limits: embedding.Limits{MaxResponseBytes: 2048},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	result, err := client.Embed(context.Background(), root.Request{Inputs: []string{"text"}, Purpose: embedding.Query})
	// Assert: SDK-L01/L03 accepted, not a strict retained-wire guarantee.
	if err != nil || len(result.Embeddings) != 1 {
		t.Fatalf("representation/shape characterization changed: %+v %v", result, err)
	}
}
