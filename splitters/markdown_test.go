package splitters

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/skosovsky/ragy"
)

func TestMarkdownSplitter_Split(t *testing.T) {
	ctx := context.Background()
	m := NewMarkdownSplitter()
	doc := ragy.Document{
		ID:      "md1",
		Content: "# Title\n\nBody one.\n\n## Section 2\n\nBody two.",
	}
	var chunks []ragy.Document
	for c, err := range m.Split(ctx, doc) {
		require.NoError(t, err)
		chunks = append(chunks, c)
	}
	require.GreaterOrEqual(t, len(chunks), 1)
	assert.Contains(t, chunks[0].Content, "Title")
}

func TestMarkdownSplitter_Empty(t *testing.T) {
	ctx := context.Background()
	m := NewMarkdownSplitter()
	doc := ragy.Document{ID: "e", Content: ""}
	var n int
	for range m.Split(ctx, doc) {
		n++
	}
	assert.Equal(t, 0, n)
}
