package splitters

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/testutil"
)

func TestSemanticSplitter_Split(t *testing.T) {
	ctx := context.Background()
	emb := testutil.NewMockDenseEmbedder(4)
	s := NewSemanticSplitter(emb, WithSemanticThreshold(0.3))
	doc := ragy.Document{
		ID:      "s1",
		Content: "First sentence. Second sentence. Third sentence.",
	}
	var chunks []ragy.Document
	for c, err := range s.Split(ctx, doc) {
		require.NoError(t, err)
		chunks = append(chunks, c)
	}
	require.GreaterOrEqual(t, len(chunks), 1)
}

func TestSemanticSplitter_ContextCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	emb := testutil.NewMockDenseEmbedder(4)
	s := NewSemanticSplitter(emb)
	doc := ragy.Document{ID: "s2", Content: "A. B. C."}
	cancel()
	var gotErr error
	for _, err := range s.Split(ctx, doc) {
		gotErr = err
		break
	}
	require.Error(t, gotErr)
}

func TestSemanticSplitter_YieldSafety(t *testing.T) {
	ctx := context.Background()
	emb := testutil.NewMockDenseEmbedder(4)
	s := NewSemanticSplitter(emb)
	doc := ragy.Document{ID: "s3", Content: "One. Two. Three."}
	count := 0
	for c, err := range s.Split(ctx, doc) {
		require.NoError(t, err)
		count++
		_ = c
		if count >= 1 {
			break
		}
	}
	assert.Equal(t, 1, count)
}
