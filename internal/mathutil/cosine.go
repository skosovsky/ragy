// Package mathutil provides math utilities for ragy (cosine similarity).
package mathutil

import (
	"math"
)

// CosineSimilarity returns the cosine similarity between two vectors:
// dot(a,b) / (||a|| * ||b||). Result is in [-1, 1]. Returns 0 if either vector is empty or zero-norm.
func CosineSimilarity(a, b []float32) float32 {
	if len(a) == 0 || len(b) == 0 || len(a) != len(b) {
		return 0
	}
	var dot, normA, normB float32
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	denom := float32(math.Sqrt(float64(normA) * float64(normB)))
	if denom == 0 {
		return 0
	}
	return dot / denom
}
