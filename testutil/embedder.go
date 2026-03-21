// Package testutil provides in-memory stores and mock embedders for testing ragy.
package testutil

import (
	"context"
	"hash/fnv"
	"sync"

	"github.com/skosovsky/ragy"
)

// MockDenseEmbedder returns deterministic dense vectors from text (hash-based).
// Implements ragy.DenseEmbedder for tests. Dimension is fixed (e.g. 8).
type MockDenseEmbedder struct {
	Dimension int
	mu        sync.Mutex
}

// NewMockDenseEmbedder returns a mock dense embedder with the given dimension (default 8).
func NewMockDenseEmbedder(dim int) *MockDenseEmbedder {
	if dim <= 0 {
		dim = 8
	}
	return &MockDenseEmbedder{Dimension: dim}
}

// Embed implements ragy.DenseEmbedder.
func (m *MockDenseEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([][]float32, len(texts))
	for i, t := range texts {
		out[i] = m.embedOne(t)
	}
	return out, nil
}

func (m *MockDenseEmbedder) embedOne(text string) []float32 {
	h := fnv.New32a()
	_, _ = h.Write([]byte(text))
	u := h.Sum32()
	vec := make([]float32, m.Dimension)
	for i := range vec {
		u = u*lcgMultiplier + lcgIncrement
		vec[i] = float32(int32(u%lcgModulus)) / lcgHalfScale
	}
	return vec
}

var _ ragy.DenseEmbedder = (*MockDenseEmbedder)(nil)

const (
	lcgMultiplier = 1103515245
	lcgIncrement  = 12345
	lcgModulus    = 0x10000
	lcgHalfScale  = 0x8000
)

// MockTensorEmbedder returns deterministic per-token tensors (hash-based).
// Implements ragy.TensorEmbedder for tests. Each text yields a fixed number of token vectors.
type MockTensorEmbedder struct {
	TokensPerText int
	DimPerToken   int
	mu            sync.Mutex
}

// NewMockTensorEmbedder returns a mock tensor embedder.
func NewMockTensorEmbedder(tokensPerText, dimPerToken int) *MockTensorEmbedder {
	if tokensPerText <= 0 {
		tokensPerText = 4
	}
	if dimPerToken <= 0 {
		dimPerToken = 8
	}
	return &MockTensorEmbedder{TokensPerText: tokensPerText, DimPerToken: dimPerToken}
}

// EmbedTensors implements ragy.TensorEmbedder.
func (m *MockTensorEmbedder) EmbedTensors(_ context.Context, texts []string) ([][][]float32, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([][][]float32, len(texts))
	for i, t := range texts {
		out[i] = m.embedOneTensor(t)
	}
	return out, nil
}

func (m *MockTensorEmbedder) embedOneTensor(text string) [][]float32 {
	h := fnv.New32a()
	_, _ = h.Write([]byte(text))
	u := h.Sum32()
	tensors := make([][]float32, m.TokensPerText)
	for j := range tensors {
		vec := make([]float32, m.DimPerToken)
		for k := range vec {
			u = u*lcgMultiplier + lcgIncrement
			vec[k] = float32(int32(u%lcgModulus)) / lcgHalfScale
		}
		tensors[j] = vec
	}
	return tensors
}

var _ ragy.TensorEmbedder = (*MockTensorEmbedder)(nil)
