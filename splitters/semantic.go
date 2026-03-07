package splitters

import (
	"context"
	"iter"
	"maps"
	"strings"
	"unicode"

	"github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/internal/mathutil"
)

// SemanticSplitter splits text at sentence boundaries where cosine similarity between
// consecutive sentences drops below Threshold. Uses DenseEmbedder for embeddings.
type SemanticSplitter struct {
	Embedder     ragy.DenseEmbedder
	Threshold    float32
	MinChunkSize int
}

// SemanticOption configures SemanticSplitter.
type SemanticOption func(*SemanticSplitter)

// WithSemanticThreshold sets the minimum similarity to keep sentences in the same chunk.
func WithSemanticThreshold(t float32) SemanticOption {
	return func(s *SemanticSplitter) {
		s.Threshold = t
	}
}

// WithMinChunkSize sets the minimum number of characters per chunk.
func WithMinChunkSize(n int) SemanticOption {
	return func(s *SemanticSplitter) {
		s.MinChunkSize = n
	}
}

// NewSemanticSplitter returns a SemanticSplitter.
func NewSemanticSplitter(embedder ragy.DenseEmbedder, opts ...SemanticOption) *SemanticSplitter {
	s := &SemanticSplitter{
		Embedder:     embedder,
		Threshold:    0.5,
		MinChunkSize: 100,
	}
	for _, o := range opts {
		o(s)
	}
	return s
}

// Split implements Splitter.
func (s *SemanticSplitter) Split(ctx context.Context, doc ragy.Document) iter.Seq2[ragy.Document, error] {
	return func(yield func(ragy.Document, error) bool) {
		text := strings.TrimSpace(doc.Content)
		if text == "" {
			return
		}
		sentences := splitSentences(text)
		if len(sentences) == 0 {
			chunk := ragy.Document{
				ID:       doc.ID + "_0",
				Content:  text,
				Metadata: cloneMeta(doc.Metadata, doc.ID, 0),
			}
			_ = yield(chunk, nil)
			return
		}
		if len(sentences) == 1 {
			chunk := ragy.Document{
				ID:       doc.ID + "_0",
				Content:  sentences[0],
				Metadata: cloneMeta(doc.Metadata, doc.ID, 0),
			}
			_ = yield(chunk, nil)
			return
		}

		// Embed sentences in batches to respect ctx and yield-safety.
		embeddings, err := s.Embedder.Embed(ctx, sentences)
		if err != nil {
			_ = yield(ragy.Document{}, err)
			return
		}
		if len(embeddings) != len(sentences) {
			chunk := ragy.Document{ID: doc.ID + "_0", Content: text, Metadata: cloneMeta(doc.Metadata, doc.ID, 0)}
			_ = yield(chunk, nil)
			return
		}

		// Find split points where similarity drops below threshold.
		splits := []int{0}
		for i := 1; i < len(sentences); i++ {
			if ctx.Err() != nil {
				_ = yield(ragy.Document{}, ctx.Err())
				return
			}
			sim := mathutil.CosineSimilarity(embeddings[i-1], embeddings[i])
			if sim < s.Threshold {
				splits = append(splits, i)
			}
		}
		splits = append(splits, len(sentences))

		for i := 0; i < len(splits)-1; i++ {
			if ctx.Err() != nil {
				_ = yield(ragy.Document{}, ctx.Err())
				return
			}
			start, end := splits[i], splits[i+1]
			chunkContent := strings.TrimSpace(strings.Join(sentences[start:end], " "))
			if len(chunkContent) < s.MinChunkSize && i < len(splits)-2 {
				continue
			}
			if chunkContent == "" {
				continue
			}
			chunkID := doc.ID + "_" + itoa(i)
			if doc.ID == "" {
				chunkID = itoa(i)
			}
			chunk := ragy.Document{
				ID:       chunkID,
				Content:  chunkContent,
				Metadata: cloneMeta(doc.Metadata, doc.ID, i),
			}
			if !yield(chunk, nil) {
				return
			}
		}
	}
}

func cloneMeta(meta map[string]any, parentID string, chunkIndex int) map[string]any {
	out := make(map[string]any)
	maps.Copy(out, meta)
	out["ParentID"] = parentID
	out["ChunkIndex"] = chunkIndex
	return out
}

func splitSentences(text string) []string {
	var sentences []string
	var buf strings.Builder
	for i, r := range text {
		buf.WriteRune(r)
		if (r == '.' || r == '!' || r == '?') && (i+1 >= len(text) || unicode.IsSpace(rune(text[i+1])) || text[i+1] == '\n') {
			s := strings.TrimSpace(buf.String())
			if s != "" {
				sentences = append(sentences, s)
			}
			buf.Reset()
		}
	}
	if buf.Len() > 0 {
		s := strings.TrimSpace(buf.String())
		if s != "" {
			sentences = append(sentences, s)
		}
	}
	return sentences
}
