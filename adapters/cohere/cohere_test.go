package cohere

import (
	"context"
	"testing"

	"github.com/skosovsky/ragy"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCohereRelevanceToDocScores(t *testing.T) {
	s, c := cohereRelevanceToDocScores(0.5)
	assert.Equal(t, float32(0.5), s)
	assert.Equal(t, 0.5, c)
	s, c = cohereRelevanceToDocScores(1.0)
	assert.Equal(t, float32(1), s)
	assert.Equal(t, 1.0, c)
	s, c = cohereRelevanceToDocScores(-0.1)
	assert.Equal(t, float32(0), s)
	assert.Equal(t, 0.0, c)
	s, c = cohereRelevanceToDocScores(1.5)
	assert.Equal(t, float32(1), s)
	assert.Equal(t, 1.0, c)
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
