package wire

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/skosovsky/ragy/embedding"
)

func TestMaterializationCancellationPreservesAccounting(t *testing.T) {
	// Arrange: completed envelope, canceled before materialization.
	client, err := New(
		Config{
			APIKey: "secret",
			Space: embedding.Space{
				Model:         "gemini-embedding-001",
				ModelRevision: "host-pinned",
				Configuration: "test",
				VectorSpace:   "pair",
				Dimension:     2,
				Metric:        embedding.Dot,
			},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	var wire Response
	if err = json.Unmarshal([]byte(`{"embeddings":[],"usageMetadata":{"promptTokenCount":7}}`), &wire); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	// Act.
	out, err := client.materialize(ctx, 1, wire)
	// Assert: context beats cardinality without losing independently observed accounting.
	if !errors.Is(err, context.Canceled) || len(out.Embeddings) != 0 || !out.Usage.InputTokensKnown ||
		out.Usage.InputTokens != 7 {
		t.Fatalf("late gate: %#v %v", out, err)
	}
}
