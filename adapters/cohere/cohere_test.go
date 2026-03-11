package cohere

import (
	"context"
	"testing"

	"github.com/skosovsky/ragy"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRerank_EmptyDocs(t *testing.T) {
	r := New("test-token")
	out, err := r.Rerank(context.Background(), "query", nil, 5)
	require.NoError(t, err)
	assert.Nil(t, out)
	out, err = r.Rerank(context.Background(), "q", []ragy.Document{}, 5)
	require.NoError(t, err)
	assert.Empty(t, out)
}
