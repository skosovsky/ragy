package cohere

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/skosovsky/ragy"
)

func TestCohereRelevanceToDocScores(t *testing.T) {
	const eps = 1e-9

	s, c := cohereRelevanceToDocScores(0.5)
	assert.InDelta(t, float32(0.5), s, 1e-6)
	assert.InDelta(t, 0.5, c, eps)
	s, c = cohereRelevanceToDocScores(1.0)
	assert.InDelta(t, float32(1), s, 1e-6)
	assert.InDelta(t, 1.0, c, eps)
	s, c = cohereRelevanceToDocScores(-0.1)
	assert.InDelta(t, float32(0), s, 1e-6)
	assert.InDelta(t, 0.0, c, eps)
	s, c = cohereRelevanceToDocScores(1.5)
	assert.InDelta(t, float32(1), s, 1e-6)
	assert.InDelta(t, 1.0, c, eps)
}

func TestRerank_EmptyDocs(t *testing.T) {
	r := New("test-token")
	out, err := r.Rerank(context.Background(), "query", nil, 5)
	require.NoError(t, err)
	assert.Nil(t, out)
	out, err = r.Rerank(context.Background(), "q", []ragy.Document{}, 5)
	require.NoError(t, err)
	assert.Empty(t, out)
}
