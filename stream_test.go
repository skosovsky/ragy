package ragy

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestYieldDocuments_ErrorOnly(t *testing.T) {
	ctx := context.Background()
	seq := YieldDocuments(ctx, nil, errors.New("boom"))
	n := 0
	for _, err := range seq {
		n++
		require.Error(t, err)
		assert.Equal(t, "boom", err.Error())
	}
	assert.Equal(t, 1, n)
}

func TestYieldDocuments_CancelBeforeYield(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	docs := []Document{{ID: "1"}, {ID: "2"}}
	seq := YieldDocuments(ctx, docs, nil)
	n := 0
	for d, err := range seq {
		require.ErrorIs(t, err, context.Canceled)
		assert.Empty(t, d.ID)
		n++
		break
	}
	assert.Equal(t, 1, n)
}

func TestYieldDocuments_EarlyStopYield(t *testing.T) {
	ctx := context.Background()
	docs := []Document{{ID: "a"}, {ID: "b"}}
	seq := YieldDocuments(ctx, docs, nil)
	n := 0
	seq(func(d Document, err error) bool {
		require.NoError(t, err)
		assert.Equal(t, "a", d.ID)
		n++
		return false
	})
	assert.Equal(t, 1, n)
}
