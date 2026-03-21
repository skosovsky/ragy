package ragy

import (
	"errors"
	"iter"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCollect_Empty(t *testing.T) {
	seq := func(_ func(Document, error) bool) {}
	docs, err := Collect(seq)
	require.NoError(t, err)
	assert.Empty(t, docs)
}

func TestCollect_Values(t *testing.T) {
	seq := func(yield func(Document, error) bool) {
		yield(Document{ID: "1", Content: "a"}, nil)
		yield(Document{ID: "2", Content: "b"}, nil)
	}
	docs, err := Collect(seq)
	require.NoError(t, err)
	require.Len(t, docs, 2)
	assert.Equal(t, "1", docs[0].ID)
	assert.Equal(t, "2", docs[1].ID)
}

func TestCollect_StopsOnError(t *testing.T) {
	sentinel := errors.New("stop")
	seq := func(yield func(Document, error) bool) {
		if !yield(Document{ID: "1", Content: "a"}, nil) {
			return
		}
		if !yield(Document{}, sentinel) {
			return
		}
		yield(Document{ID: "2", Content: "b"}, nil)
	}
	docs, err := Collect(seq)
	require.Error(t, err)
	require.ErrorIs(t, err, sentinel)
	require.Len(t, docs, 1)
}

// Ensure [iter.Seq2] type works (compile-time check).
var _ iter.Seq2[Document, error] = (func(func(Document, error) bool))(nil)
