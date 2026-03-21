package rerankers

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/skosovsky/ragy"
)

func TestRRFReranker_MergeRankedLists(t *testing.T) {
	ctx := context.Background()
	r := NewRRFReranker(WithRRFK(60))
	lists := [][]ragy.Document{
		{{ID: "a", Content: "a"}, {ID: "b", Content: "b"}, {ID: "c", Content: "c"}},
		{{ID: "b", Content: "b"}, {ID: "a", Content: "a"}, {ID: "d", Content: "d"}},
	}
	merged := r.MergeRankedLists(ctx, lists, 2)
	require.Len(t, merged, 2)
	// b appears in both lists with high rank -> high RRF score; a as well.
	ids := make([]string, len(merged))
	for i := range merged {
		ids[i] = merged[i].ID
	}
	assert.Contains(t, ids, "a")
	assert.Contains(t, ids, "b")
}

func TestRRFReranker_Rerank(t *testing.T) {
	ctx := context.Background()
	r := NewRRFReranker()
	docs := []ragy.Document{
		{ID: "1", Content: "x"},
		{ID: "2", Content: "y"},
		{ID: "3", Content: "z"},
	}
	out, err := r.Rerank(ctx, "q", docs, 2)
	require.NoError(t, err)
	require.Len(t, out, 2)
	assert.Equal(t, "1", out[0].ID)
	assert.Equal(t, "2", out[1].ID)
}
