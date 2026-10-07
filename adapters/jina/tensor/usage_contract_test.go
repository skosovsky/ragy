package tensor

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/embedding"
	root "github.com/skosovsky/ragy/tensor"
)

func TestRejectedPayloadRetainsObservedUsage(t *testing.T) {
	for _, body := range []string{
		`{"data":[],"usage":{"total_tokens":7}}`,
		`{"data":[{"index":0,"embeddings":[]}],"usage":{"total_tokens":7}}`,
		`{"data":[{"index":0,"embeddings":"malformed"}],"usage":{"total_tokens":7}}`,
		`{"data":[{"index":"malformed","embeddings":[]}],"usage":{"total_tokens":7}}`,
		`{"data":"malformed","usage":{"total_tokens":7}}`,
		`{"data":[{"index":0,"embeddings":[[1e100]]}],"usage":{"total_tokens":7}}`,
		`{"model":123,"data":[],"usage":{"total_tokens":7}}`,
		`{"model":"contradictory","data":[],"usage":{"total_tokens":7}}`,
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
			client := testClient(t, server.URL, embedding.Limits{})
			// Act.
			out, err := client.Embed(
				context.Background(),
				root.Request{Inputs: []string{"text"}, Purpose: embedding.Query},
			)
			// Assert.
			if !errors.Is(err, ragy.ErrProtocol) || len(out.Embeddings) != 0 || !out.Usage.InputTokensKnown ||
				out.Usage.InputTokens != 7 ||
				calls != 1 {
				t.Fatalf("usage or payload contract: %#v %v calls=%d", out, err, calls)
			}
		})
	}
}

func TestCredentialRejectedBeforeDispatch(t *testing.T) {
	// Arrange.
	cfg := Config{APIKey: "secret\t", Space: testSpace()}
	// Act.
	client, err := New(cfg)
	// Assert.
	if client != nil || !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatal("credential admitted", err)
	}
}

func TestMaterializationCancellationPreservesAccounting(t *testing.T) {
	// Arrange: complete decoded envelope, then cancellation before materialization.
	client := testClient(t, "https://provider.example/v1", embedding.Limits{})
	var decoded embedResponse
	if err := json.Unmarshal([]byte(`{"data":[],"usage":{"total_tokens":7}}`), &decoded); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	// Act.
	out, err := client.materialize(ctx, 1, decoded)
	// Assert: context wins over rejected cardinality, payload suppressed, actual usage retained.
	if !errors.Is(err, context.Canceled) || len(out.Embeddings) != 0 || !out.Usage.InputTokensKnown ||
		out.Usage.InputTokens != 7 {
		t.Fatalf("late gate: %#v %v", out, err)
	}
}

func TestRejectedEnvelopeDoesNotInventUsage(t *testing.T) {
	for _, body := range []string{
		`{"data":[]}`,
		`{"data":[],"usage":{"total_tokens":-1}}`,
		`{"data":[],"usage":{"total_tokens":7}} trailing`,
		`{"data":[],"usage":{"total_tokens":7}`,
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
			client := testClient(t, server.URL, embedding.Limits{})
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
