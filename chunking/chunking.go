// Package chunking provides chunking contracts and implementations.
package chunking

import (
	"context"
	"fmt"
	"math"
	"slices"
	"sort"
	"strings"
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/internal/parallel"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

// Splitter splits a source document into typed chunks.
type Splitter[TMeta any] interface {
	Split(ctx context.Context, doc retrieval.Document[TMeta]) ([]Chunk[TMeta], error)
}

// ContextGenerator derives context from immutable host metadata/content. It must
// honor cancellation: Split cancels and joins all started cooperative callbacks.
type ContextGenerator[TMeta any] interface {
	Context(ctx context.Context, source retrieval.Document[TMeta], chunk Chunk[TMeta]) (string, error)
}

// SentenceSegmenter returns ordered non-overlapping original UTF-8 byte spans.
// Implementations must honor context and must not replace or normalize source text.
type SentenceSegmenter interface {
	Split(context.Context, string) ([]source.ByteSpan, error)
}

func validateSource[TMeta any](ctx context.Context, doc retrieval.Document[TMeta]) (retrieval.Document[TMeta], error) {
	if err := ctx.Err(); err != nil {
		return retrieval.Document[TMeta]{}, err
	}
	if doc.ID == "" {
		return retrieval.Document[TMeta]{}, fmt.Errorf("%w: source document id", ragy.ErrMissingSourceID)
	}
	if strings.TrimSpace(doc.Content) == "" {
		return retrieval.Document[TMeta]{}, ragy.ErrEmptyText
	}
	if !utf8.ValidString(doc.Content) {
		return retrieval.Document[TMeta]{}, ragy.ErrInvalidArgument
	}
	if doc.SourceMapping.Text() != "" &&
		(doc.SourceMapping.Text() != doc.Content || doc.SourceMapping.Validate() != nil) {
		return retrieval.Document[TMeta]{}, ragy.ErrInvalidArgument
	}
	for _, support := range doc.SourceSupports {
		if err := support.Validate(); err != nil {
			return retrieval.Document[TMeta]{}, err
		}
	}
	return doc, nil
}

func buildChunks[TMeta any](
	ctx context.Context,
	doc retrieval.Document[TMeta],
	spans []source.ByteSpan,
) ([]Chunk[TMeta], error) {
	chunks := make([]Chunk[TMeta], 0, len(spans))
	for index, span := range spans {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err := span.ValidateText(doc.Content); err != nil {
			return nil, err
		}
		var mapping source.MappedText
		if doc.SourceMapping.Text() != "" {
			var err error
			mapping, err = doc.SourceMapping.Slice(span)
			if err != nil {
				return nil, err
			}
		}
		supports := append(slices.Clone(doc.SourceSupports), mapping.Supports()...)
		chunk := Chunk[TMeta]{
			ID:             fmt.Sprintf("%s_%d", doc.ID, index),
			SourceID:       doc.ID,
			Index:          index,
			Total:          len(spans),
			Content:        doc.Content[span.Start:span.End],
			Context:        "",
			Meta:           doc.Meta,
			InputSpan:      span,
			SourceMapping:  mapping,
			SourceSupports: supports,
		}
		if err := ValidateChunk(chunk); err != nil {
			return nil, err
		}
		chunks = append(chunks, chunk)
	}
	return chunks, nil
}

// Recursive is an iterative chunk splitter.
type Recursive[TMeta any] struct {
	chunkSize  int
	overlap    int
	separators []string
}

// NewRecursive constructs a recursive splitter.
func NewRecursive[TMeta any](chunkSize, overlap int, separators []string) (*Recursive[TMeta], error) {
	if chunkSize <= 0 {
		return nil, fmt.Errorf("%w: chunk size must be > 0", ragy.ErrInvalidArgument)
	}

	if overlap < 0 || overlap >= chunkSize {
		return nil, fmt.Errorf("%w: overlap must be >= 0 and < chunk size", ragy.ErrInvalidArgument)
	}

	if len(separators) == 0 {
		separators = []string{"\n\n", "\n", " "}
	}

	for _, separator := range separators {
		if separator == "" || !utf8.ValidString(separator) {
			return nil, ragy.ErrInvalidArgument
		}
	}
	return &Recursive[TMeta]{
		chunkSize:  chunkSize,
		overlap:    overlap,
		separators: slices.Clone(separators),
	}, nil
}

// Split splits a source document.
func (r *Recursive[TMeta]) Split(ctx context.Context, doc retrieval.Document[TMeta]) ([]Chunk[TMeta], error) {
	if r == nil || r.chunkSize <= 0 || r.overlap < 0 || r.overlap >= r.chunkSize {
		return nil, ragy.ErrInvalidArgument
	}
	normalized, err := validateSource(ctx, doc)
	if err != nil {
		return nil, err
	}
	parts, err := splitRanges(
		ctx,
		normalized.Content,
		source.ByteSpan{Start: 0, End: len(normalized.Content)},
		r.chunkSize,
		r.overlap,
		r.separators,
	)
	if err != nil {
		return nil, err
	}
	return buildChunks(ctx, normalized, parts)
}

// Markdown splits markdown documents by headings first and then recursively.
type Markdown[TMeta any] struct {
	base *Recursive[TMeta]
}

// NewMarkdown constructs a markdown splitter.
func NewMarkdown[TMeta any](base *Recursive[TMeta]) (*Markdown[TMeta], error) {
	if base == nil {
		return nil, fmt.Errorf("%w: markdown base splitter", ragy.ErrInvalidArgument)
	}

	return &Markdown[TMeta]{base: base}, nil
}

// Split splits a markdown document.
func (m *Markdown[TMeta]) Split(ctx context.Context, doc retrieval.Document[TMeta]) ([]Chunk[TMeta], error) {
	if m == nil || m.base == nil || m.base.chunkSize <= 0 {
		return nil, ragy.ErrInvalidArgument
	}
	normalized, err := validateSource(ctx, doc)
	if err != nil {
		return nil, err
	}
	sections, err := markdownRanges(ctx, normalized.Content)
	if err != nil {
		return nil, err
	}
	var parts []source.ByteSpan
	for _, section := range sections {
		fragments, splitErr := splitRanges(
			ctx,
			normalized.Content,
			section,
			m.base.chunkSize,
			m.base.overlap,
			m.base.separators,
		)
		if splitErr != nil {
			return nil, splitErr
		}
		parts = append(parts, fragments...)
	}
	return buildChunks(ctx, normalized, parts)
}

// DefaultSentenceSegmenter returns original punctuation-delimited byte spans.
type DefaultSentenceSegmenter struct{}

// Split returns UTF-8-safe ranges without reconstructing source text.
func (DefaultSentenceSegmenter) Split(ctx context.Context, text string) ([]source.ByteSpan, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if !utf8.ValidString(text) {
		return nil, ragy.ErrInvalidArgument
	}
	var out []source.ByteSpan
	start := 0
	for index, r := range text {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if !isSentenceBoundary(r) {
			continue
		}
		end := index + utf8.RuneLen(r)
		span, err := trimRange(ctx, text, source.ByteSpan{Start: start, End: end})
		if err != nil {
			return nil, err
		}
		if span.End > span.Start {
			out = append(out, span)
		}
		start = end
	}
	span, err := trimRange(ctx, text, source.ByteSpan{Start: start, End: len(text)})
	if err != nil {
		return nil, err
	}
	if span.End > span.Start {
		out = append(out, span)
	}
	return out, nil
}

func isSentenceBoundary(r rune) bool {
	switch r {
	case '.', '!', '?', '。', '！', '？':
		return true
	default:
		return false
	}
}

// Semantic groups caller-provided sentence segments by embedding similarity.
type Semantic[TMeta any] struct {
	embedder  dense.Embedder
	segmenter SentenceSegmenter
	threshold float64
	minGroup  int
}

// NewSemantic constructs a semantic splitter with an explicit sentence segmentation strategy.
func NewSemantic[TMeta any](
	embedder dense.Embedder,
	segmenter SentenceSegmenter,
	threshold float64,
	minGroup int,
) (*Semantic[TMeta], error) {
	if embedder == nil {
		return nil, fmt.Errorf("%w: semantic embedder", ragy.ErrInvalidArgument)
	}
	if segmenter == nil {
		return nil, fmt.Errorf("%w: semantic sentence segmenter", ragy.ErrInvalidArgument)
	}

	if math.IsNaN(threshold) || math.IsInf(threshold, 0) || threshold < -1 || threshold > 1 {
		return nil, fmt.Errorf("%w: semantic threshold must be in [-1,1]", ragy.ErrInvalidArgument)
	}

	if minGroup <= 0 {
		return nil, fmt.Errorf("%w: min group must be > 0", ragy.ErrInvalidArgument)
	}

	return &Semantic[TMeta]{
		embedder:  embedder,
		segmenter: segmenter,
		threshold: threshold,
		minGroup:  minGroup,
	}, nil
}

// Split splits a source document by semantic boundaries.
func (s *Semantic[TMeta]) Split(ctx context.Context, doc retrieval.Document[TMeta]) ([]Chunk[TMeta], error) {
	if s == nil || s.segmenter == nil || s.embedder == nil || s.minGroup <= 0 {
		return nil, ragy.ErrInvalidArgument
	}
	normalized, err := validateSource(ctx, doc)
	if err != nil {
		return nil, err
	}

	spans, err := s.segmenter.Split(ctx, normalized.Content)
	if gateErr := ctx.Err(); gateErr != nil {
		return nil, gateErr
	}
	if err != nil {
		return nil, err
	}
	sentences, err := sentenceTexts(ctx, normalized.Content, spans)
	if err != nil {
		return nil, err
	}
	if len(sentences) == 0 {
		return nil, fmt.Errorf("%w: semantic sentence segmentation returned no sentences", ragy.ErrProtocol)
	}

	embeddings, err := s.embedder.Embed(ctx, sentences)
	if gateErr := ctx.Err(); gateErr != nil {
		return nil, gateErr
	}
	if err != nil {
		return nil, err
	}

	if len(embeddings) != len(sentences) {
		return nil, fmt.Errorf(
			"%w: semantic embedding cardinality mismatch: %d sentences, %d embeddings",
			ragy.ErrProtocol,
			len(sentences),
			len(embeddings),
		)
	}

	if err = ctx.Err(); err != nil {
		return nil, err
	}
	if err = validateSemanticEmbeddings(ctx, embeddings); err != nil {
		return nil, err
	}

	parts, err := semanticGroups(ctx, spans, embeddings, s.threshold, s.minGroup)
	if err != nil {
		return nil, err
	}
	return buildChunks(ctx, normalized, parts)
}

func validateSemanticEmbeddings(ctx context.Context, embeddings [][]float32) error {
	if len(embeddings) == 0 {
		return fmt.Errorf("%w: semantic embeddings missing", ragy.ErrProtocol)
	}

	expectedDim := len(embeddings[0])
	if expectedDim == 0 {
		return fmt.Errorf("%w: semantic embedding dimension must be > 0", ragy.ErrProtocol)
	}

	for index, embedding := range embeddings {
		if len(embedding) == 0 {
			return fmt.Errorf("%w: semantic embedding %d is empty", ragy.ErrProtocol, index)
		}
		if len(embedding) != expectedDim {
			return fmt.Errorf(
				"%w: semantic embedding dimension mismatch: expected %d, got %d",
				ragy.ErrProtocol,
				expectedDim,
				len(embedding),
			)
		}
		for _, value := range embedding {
			if err := ctx.Err(); err != nil {
				return err
			}
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return fmt.Errorf("%w: non-finite semantic embedding", ragy.ErrProtocol)
			}
		}
		if isZeroNormVector(embedding) {
			return fmt.Errorf("%w: semantic embedding %d has zero norm", ragy.ErrProtocol, index)
		}
	}

	return nil
}

func isZeroNormVector(embedding []float32) bool {
	for _, value := range embedding {
		if value != 0 {
			return false
		}
	}
	return true
}

func semanticGroups(
	ctx context.Context,
	spans []source.ByteSpan,
	embeddings [][]float32,
	threshold float64,
	minGroup int,
) ([]source.ByteSpan, error) {
	var parts []source.ByteSpan
	groupStart := 0
	for index := 1; index < len(spans); index++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if index-groupStart < minGroup {
			continue
		}
		similarity, err := cosine(ctx, embeddings[index-1], embeddings[index])
		if err != nil {
			return nil, err
		}
		if similarity >= threshold {
			continue
		}
		parts = append(parts, source.ByteSpan{Start: spans[groupStart].Start, End: spans[index-1].End})
		groupStart = index
	}
	parts = append(parts, source.ByteSpan{Start: spans[groupStart].Start, End: spans[len(spans)-1].End})
	return parts, nil
}

func cosine(ctx context.Context, left, right []float32) (float64, error) {
	if len(left) == 0 || len(right) == 0 || len(left) != len(right) {
		return 0, ragy.ErrProtocol
	}

	dot := 0.0
	leftNorm := 0.0
	rightNorm := 0.0
	for index := range left {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		lv := float64(left[index])
		rv := float64(right[index])
		dot += lv * rv
		leftNorm += lv * lv
		rightNorm += rv * rv
	}

	if leftNorm == 0 || rightNorm == 0 {
		return 0, ragy.ErrProtocol
	}

	return math.Max(-1, math.Min(1, dot/(math.Sqrt(leftNorm)*math.Sqrt(rightNorm)))), nil
}

// Contextual augments chunks with derived context.
type Contextual[TMeta any] struct {
	base        Splitter[TMeta]
	generator   ContextGenerator[TMeta]
	concurrency int
}

// NewContextual constructs a contextual splitter.
func NewContextual[TMeta any](
	base Splitter[TMeta],
	generator ContextGenerator[TMeta],
	concurrency int,
) (*Contextual[TMeta], error) {
	if base == nil {
		return nil, fmt.Errorf("%w: contextual base splitter", ragy.ErrInvalidArgument)
	}

	if generator == nil {
		return nil, fmt.Errorf("%w: contextual generator", ragy.ErrInvalidArgument)
	}

	if concurrency <= 0 {
		return nil, fmt.Errorf("%w: contextual concurrency", ragy.ErrInvalidArgument)
	}

	return &Contextual[TMeta]{
		base:        base,
		generator:   generator,
		concurrency: concurrency,
	}, nil
}

// Split splits a source document and enriches chunk context in parallel.
func (c *Contextual[TMeta]) Split(ctx context.Context, doc retrieval.Document[TMeta]) ([]Chunk[TMeta], error) {
	if c == nil || c.base == nil || c.generator == nil || c.concurrency <= 0 {
		return nil, ragy.ErrInvalidArgument
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	chunks, err := c.base.Split(ctx, doc)
	if err != nil {
		return nil, err
	}

	enriched, err := parallel.MapOrdered(
		ctx,
		c.concurrency,
		chunks,
		func(ctx context.Context, chunk Chunk[TMeta]) (Chunk[TMeta], error) {
			if gateErr := ctx.Err(); gateErr != nil {
				return Chunk[TMeta]{}, gateErr
			}
			contextText, contextErr := c.generator.Context(ctx, doc, chunk)
			if contextErr != nil {
				return Chunk[TMeta]{}, contextErr
			}

			if gateErr := ctx.Err(); gateErr != nil {
				return Chunk[TMeta]{}, gateErr
			}
			chunk.Context = contextText
			return chunk, nil
		},
	)
	if err != nil {
		return nil, err
	}

	sort.SliceStable(enriched, func(i, j int) bool {
		return enriched[i].Index < enriched[j].Index
	})

	return enriched, nil
}
