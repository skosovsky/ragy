package splitters

import (
	"context"
	"errors"
	"math/rand"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/skosovsky/ragy"
)

func TestContextualSplitter_Split(t *testing.T) {
	ctx := context.Background()
	inner := NewRecursiveSplitter(WithChunkSize(50), WithChunkOverlap(10))
	contextualizer := &mockContextualizer{
		generate: func(_ context.Context, _, chunkContent string) (string, error) {
			prefix := chunkContent
			if len(prefix) > 10 {
				prefix = prefix[:10]
			}
			return "Context: " + prefix, nil
		},
	}
	cs := NewContextualSplitter(inner, contextualizer, WithContextualConcurrency(2))
	doc := ragy.Document{
		ID:      "doc1",
		Content: "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
	}
	var chunks []ragy.Document
	for c, err := range cs.Split(ctx, doc) {
		require.NoError(t, err)
		chunks = append(chunks, c)
	}
	require.GreaterOrEqual(t, len(chunks), 1)
	for _, ch := range chunks {
		assert.True(
			t,
			strings.HasPrefix(ch.Content, "Context: "),
			"chunk content should start with Context: ; got %q",
			ch.Content,
		)
	}
}

func TestContextualSplitter_YieldSafety(t *testing.T) {
	ctx := context.Background()
	inner := NewRecursiveSplitter(WithChunkSize(5))
	contextualizer := &mockContextualizer{
		generate: func(_ context.Context, _, chunkContent string) (string, error) {
			return "ctx:" + chunkContent, nil
		},
	}
	cs := NewContextualSplitter(inner, contextualizer, WithContextualConcurrency(1))
	doc := ragy.Document{ID: "d", Content: "a b c d e f g h i j k"}
	count := 0
	for c, err := range cs.Split(ctx, doc) {
		require.NoError(t, err)
		count++
		_ = c
		if count >= 1 {
			break
		}
	}
	assert.Equal(t, 1, count)
}

func TestContextualSplitter_ErrorPropagation(t *testing.T) {
	ctx := context.Background()
	inner := NewRecursiveSplitter(WithChunkSize(50))
	errMock := errors.New("contextualizer error")
	contextualizer := &mockContextualizer{
		generate: func(_ context.Context, _, _ string) (string, error) {
			return "", errMock
		},
	}
	cs := NewContextualSplitter(inner, contextualizer, WithContextualConcurrency(2))
	doc := ragy.Document{ID: "d", Content: "some text here"}
	var gotErr error
	for _, err := range cs.Split(ctx, doc) {
		gotErr = err
		break
	}
	require.Error(t, gotErr)
	assert.Equal(t, errMock, gotErr)
}

func TestContextualSplitter_OrderPreservation(t *testing.T) {
	ctx := context.Background()
	inner := NewRecursiveSplitter(WithChunkSize(100))

	rng := rand.New(rand.NewSource(12345))
	contextualizer := &mockContextualizer{
		generate: func(_ context.Context, _, _ string) (string, error) {
			time.Sleep(time.Duration(rng.Intn(11)) * time.Millisecond)
			return "ctx:", nil
		},
	}
	cs := NewContextualSplitter(inner, contextualizer, WithContextualConcurrency(3))
	parts := []string{"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"}
	doc := ragy.Document{ID: "order", Content: strings.Join(parts, "\n\n")}
	var chunks []ragy.Document
	for c, err := range cs.Split(ctx, doc) {
		require.NoError(t, err)
		chunks = append(chunks, c)
	}
	require.GreaterOrEqual(t, len(chunks), 1)
	for i := range chunks {
		idx, ok := chunks[i].Metadata["ChunkIndex"].(int)
		require.True(t, ok, "ChunkIndex should be int at position %d", i)
		assert.Equal(t, i, idx, "chunk at position %d should have ChunkIndex %d", i, idx)
	}
}

type mockContextualizer struct {
	generate func(ctx context.Context, fullContent, chunkContent string) (string, error)
}

func (m *mockContextualizer) GenerateContext(ctx context.Context, fullContent, chunkContent string) (string, error) {
	if m.generate != nil {
		return m.generate(ctx, fullContent, chunkContent)
	}
	return "", nil
}
