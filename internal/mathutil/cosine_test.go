package mathutil

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/skosovsky/ragy"
)

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name     string
		a        []float32
		b        []float32
		expected float32
	}{
		{"same vector", []float32{1, 0, 0}, []float32{1, 0, 0}, 1.0},
		{"opposite", []float32{1, 0, 0}, []float32{-1, 0, 0}, -1.0},
		{"orthogonal", []float32{1, 0, 0}, []float32{0, 1, 0}, 0},
		{"empty a", []float32{}, []float32{1, 0}, 0},
		{"empty b", []float32{1, 0}, []float32{}, 0},
		{"length mismatch", []float32{1, 0}, []float32{1, 0, 0}, 0},
		{"zero vector", []float32{0, 0}, []float32{1, 1}, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := CosineSimilarity(tt.a, tt.b)
			assert.InDelta(t, tt.expected, got, 1e-5)
		})
	}
}

func TestCosineSimilarity_positive(t *testing.T) {
	a := []float32{1.0, 2.0, 3.0}
	b := []float32{2.0, 4.0, 6.0} // same direction
	got := CosineSimilarity(a, b)
	require.InDelta(t, 1.0, got, 1e-5)
}

func TestDeduplicateDocuments(t *testing.T) {
	docs := []ragy.Document{
		{ID: "1", Content: "a"},
		{ID: "2", Content: "b"},
		{ID: "1", Content: "a again"},
		{ID: "3", Content: "c"},
		{ID: "2", Content: "b again"},
	}
	out := DeduplicateDocuments(docs)
	require.Len(t, out, 3)
	assert.Equal(t, "1", out[0].ID)
	assert.Equal(t, "2", out[1].ID)
	assert.Equal(t, "3", out[2].ID)
	assert.Equal(t, "a", out[0].Content)
	assert.Equal(t, "b", out[1].Content)
	assert.Equal(t, "c", out[2].Content)
}

func TestDeduplicateDocuments_empty(t *testing.T) {
	out := DeduplicateDocuments(nil)
	assert.Nil(t, out)
	out = DeduplicateDocuments([]ragy.Document{})
	assert.Empty(t, out)
}

func TestDeduplicateDocuments_no_dupes(t *testing.T) {
	docs := []ragy.Document{
		{ID: "1", Content: "a"},
		{ID: "2", Content: "b"},
	}
	out := DeduplicateDocuments(docs)
	require.Len(t, out, 2)
	assert.Equal(t, "1", out[0].ID)
	assert.Equal(t, "2", out[1].ID)
}
