package rerankers

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/skosovsky/ragy"
)

func TestCrossEncoderReranker_Rerank(t *testing.T) {
	ctx := context.Background()
	score := func(_ context.Context, _ string, doc ragy.Document) (float32, error) {
		if doc.ID == "best" {
			return 1.0, nil
		}
		return 0.5, nil
	}
	r := NewCrossEncoderReranker(score)
	docs := []ragy.Document{
		{ID: "a", Content: "a"},
		{ID: "best", Content: "best"},
		{ID: "c", Content: "c"},
	}
	out, err := r.Rerank(ctx, "q", docs, 2)
	require.NoError(t, err)
	require.Len(t, out, 2)
	assert.Equal(t, "best", out[0].ID)
	assert.InDelta(t, 1.0, float64(out[0].Score), 1e-5)
}
