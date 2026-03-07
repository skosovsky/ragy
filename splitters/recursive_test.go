package splitters

import (
	"context"
	"testing"

	"github.com/skosovsky/ragy"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRecursiveSplitter_Split(t *testing.T) {
	ctx := context.Background()
	r := NewRecursiveSplitter(WithChunkSize(50), WithChunkOverlap(10))
	doc := ragy.Document{
		ID:       "doc1",
		Content:  "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
		Metadata: map[string]any{"author": "test"},
	}
	var chunks []ragy.Document
	for c, err := range r.Split(ctx, doc) {
		require.NoError(t, err)
		chunks = append(chunks, c)
	}
	require.GreaterOrEqual(t, len(chunks), 1)
	assert.Equal(t, "doc1", chunks[0].Metadata["ParentID"])
	assert.Equal(t, "test", chunks[0].Metadata["author"])
}

func TestRecursiveSplitter_YieldSafety(t *testing.T) {
	ctx := context.Background()
	r := NewRecursiveSplitter(WithChunkSize(10))
	doc := ragy.Document{ID: "d", Content: "a b c d e f g h i j k"}
	count := 0
	for c, err := range r.Split(ctx, doc) {
		require.NoError(t, err)
		count++
		if count >= 2 {
			break
		}
		_ = c
	}
	assert.Equal(t, 2, count)
}

func TestRecursiveSplitter_ContextCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	r := NewRecursiveSplitter(WithChunkSize(5))
	doc := ragy.Document{ID: "d", Content: "one two three four five six seven"}
	cancel()
	var gotErr error
	for _, err := range r.Split(ctx, doc) {
		gotErr = err
		break
	}
	require.Error(t, gotErr)
	assert.Equal(t, context.Canceled, gotErr)
}
