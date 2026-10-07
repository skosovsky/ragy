package dense

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	ragy "github.com/skosovsky/ragy"
	root "github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/embedding"
)

func TestRejectedPayloadRetainsObservedUsage(t *testing.T) {
	for _, body := range []string{
		`{"embeddings":[],"usageMetadata":{"promptTokenCount":7}}`,
		`{"embeddings":[{"values":[]}],"usageMetadata":{"promptTokenCount":7}}`,
		`{"embeddings":[{"values":"malformed"}],"usageMetadata":{"promptTokenCount":7}}`,
		`{"embeddings":[{"values":[1e100]}],"usageMetadata":{"promptTokenCount":7}}`,
		`{"embeddings":"malformed","usageMetadata":{"promptTokenCount":7}}`,
		`{"model":123,"embeddings":[],"usageMetadata":{"promptTokenCount":7}}`,
		`{"model":"contradictory","embeddings":[],"usageMetadata":{"promptTokenCount":7}}`,
	} {
		t.Run("rejected", func(t *testing.T) {
			// Arrange.
			calls := 0
			server := httptest.NewServer(
				http.HandlerFunc(
					func(w http.ResponseWriter, _ *http.Request) { calls++; _, _ = io.WriteString(w, body) },
				),
			)
			t.Cleanup(server.Close)
			client, err := New(Config{APIKey: "secret", Space: profile("gemini-embedding-001"), BaseURL: server.URL})
			if err != nil {
				t.Fatal(err)
			}
			// Act.
			out, err := client.Embed(
				context.Background(),
				root.Request{Inputs: []string{"text"}, Purpose: embedding.Query},
			)
			// Assert.
			if !errors.Is(err, ragy.ErrProtocol) || len(out.Embeddings) != 0 || !out.Usage.InputTokensKnown ||
				out.Usage.InputTokens != 7 ||
				calls != 1 {
				t.Fatalf("usage: %#v %v calls=%d", out, err, calls)
			}
		})
	}
}

func TestRejectedEnvelopeDoesNotInventUsage(t *testing.T) {
	for _, body := range []string{
		`{"embeddings":[]}`,
		`{"embeddings":[],"usageMetadata":{"promptTokenCount":-1}}`,
		`{"embeddings":[],"usageMetadata":{"promptTokenCount":7}} trailing`,
		`{"embeddings":[],"usageMetadata":{"promptTokenCount":7}`,
	} {
		t.Run("unknown", func(t *testing.T) {
			// Arrange: valid counter bytes in incomplete/invalid wire are not observed accounting.
			calls := 0
			server := httptest.NewServer(
				http.HandlerFunc(
					func(w http.ResponseWriter, _ *http.Request) { calls++; _, _ = io.WriteString(w, body) },
				),
			)
			t.Cleanup(server.Close)
			client, err := New(Config{APIKey: "secret", Space: profile("gemini-embedding-001"), BaseURL: server.URL})
			if err != nil {
				t.Fatal(err)
			}
			// Act.
			out, err := client.Embed(
				context.Background(),
				root.Request{Inputs: []string{"text"}, Purpose: embedding.Query},
			)
			// Assert.
			if !errors.Is(err, ragy.ErrProtocol) || len(out.Embeddings) != 0 || out.Usage.InputTokensKnown ||
				out.Usage.InputTokens != 0 ||
				calls != 1 {
				t.Fatalf("unknown usage: %#v %v calls=%d", out, err, calls)
			}
		})
	}
}
