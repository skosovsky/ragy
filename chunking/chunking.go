// Package chunking provides chunking contracts and implementations.
package chunking

import (
	"context"
	"fmt"
	"sort"
	"strings"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/internal/parallel"
)

// Splitter splits a source document into typed chunks.
type Splitter interface {
	Split(ctx context.Context, doc ragy.Document) ([]ragy.Chunk, error)
}

// ContextGenerator derives chunk context without mutating raw chunk content.
type ContextGenerator interface {
	Context(ctx context.Context, source ragy.Document, chunk ragy.Chunk) (string, error)
}

// SentenceSegmenter extracts sentences from text.
type SentenceSegmenter interface {
	Split(text string) []string
}

func validateSource(doc ragy.Document) (ragy.Document, error) {
	normalized, err := ragy.NormalizeDocument(doc)
	if err != nil {
		if doc.ID == "" {
			return ragy.Document{}, fmt.Errorf("%w: source document id", ragy.ErrMissingSourceID)
		}
		return ragy.Document{}, err
	}
	if strings.TrimSpace(doc.Content) == "" {
		return ragy.Document{}, fmt.Errorf("%w: source document content", ragy.ErrEmptyText)
	}

	return normalized, nil
}

func buildChunks(doc ragy.Document, parts []string) []ragy.Chunk {
	if len(parts) == 0 {
		return nil
	}

	chunks := make([]ragy.Chunk, 0, len(parts))
	for index, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}

		chunks = append(chunks, ragy.Chunk{
			ID:         fmt.Sprintf("%s_%d", doc.ID, index),
			SourceID:   doc.ID,
			Index:      index,
			Content:    part,
			Context:    "",
			Attributes: ragy.CloneAttributes(doc.Attributes),
		})
	}

	return chunks
}

// Recursive is an iterative chunk splitter.
type Recursive struct {
	chunkSize  int
	overlap    int
	separators []string
}

// NewRecursive constructs a recursive splitter.
func NewRecursive(chunkSize, overlap int, separators []string) (*Recursive, error) {
	if chunkSize <= 0 {
		return nil, fmt.Errorf("%w: chunk size must be > 0", ragy.ErrInvalidArgument)
	}

	if overlap < 0 || overlap >= chunkSize {
		return nil, fmt.Errorf("%w: overlap must be >= 0 and < chunk size", ragy.ErrInvalidArgument)
	}

	if len(separators) == 0 {
		separators = []string{"\n\n", "\n", " "}
	}

	return &Recursive{
		chunkSize:  chunkSize,
		overlap:    overlap,
		separators: separators,
	}, nil
}

// Split splits a source document.
func (r *Recursive) Split(_ context.Context, doc ragy.Document) ([]ragy.Chunk, error) {
	normalized, err := validateSource(doc)
	if err != nil {
		return nil, err
	}

	parts := splitIterative(normalized.Content, r.chunkSize, r.overlap, r.separators)
	chunks := buildChunks(normalized, parts)
	for i, chunk := range chunks {
		normalizedChunk, err := ragy.NormalizeChunk(chunk)
		if err != nil {
			return nil, err
		}
		chunks[i] = normalizedChunk
	}
	return chunks, nil
}

type splitTask struct {
	text       string
	separators []string
}

func splitIterative(text string, chunkSize, overlap int, separators []string) []string {
	var out []string
	queue := []splitTask{{text: strings.TrimSpace(text), separators: separators}}

	for len(queue) > 0 {
		task := queue[0]
		queue = queue[1:]
		out, queue = processSplitTask(task, queue, out, chunkSize, overlap)
	}

	return out
}

func processSplitTask(
	task splitTask,
	queue []splitTask,
	out []string,
	chunkSize int,
	overlap int,
) ([]string, []splitTask) {
	switch {
	case task.text == "":
		return out, queue
	case runeLen(task.text) <= chunkSize:
		return append(out, task.text), queue
	case len(task.separators) == 0:
		return append(out, splitFixed(task.text, chunkSize, overlap)...), queue
	default:
		return out, append(queue, splitBySeparator(task, chunkSize)...)
	}
}

func splitBySeparator(task splitTask, chunkSize int) []splitTask {
	sep := task.separators[0]
	rest := task.separators[1:]
	pieces := strings.Split(task.text, sep)
	if len(pieces) == 1 {
		return []splitTask{{text: task.text, separators: rest}}
	}

	var (
		out     []splitTask
		current strings.Builder
	)
	for _, piece := range pieces {
		appendSplitPiece(&out, &current, piece, sep, rest, chunkSize)
	}

	if current.Len() > 0 {
		out = append(out, splitTask{text: current.String(), separators: rest})
	}

	return out
}

func appendSplitPiece(
	out *[]splitTask,
	current *strings.Builder,
	piece string,
	sep string,
	rest []string,
	chunkSize int,
) {
	piece = strings.TrimSpace(piece)
	if piece == "" {
		return
	}

	candidate := piece
	if current.Len() > 0 {
		candidate = current.String() + sep + piece
	}

	if runeLen(candidate) <= chunkSize {
		current.Reset()
		current.WriteString(candidate)
		return
	}

	if current.Len() > 0 {
		*out = append(*out, splitTask{text: current.String(), separators: rest})
		current.Reset()
	}

	*out = append(*out, splitTask{text: piece, separators: rest})
}

func splitFixed(text string, chunkSize, overlap int) []string {
	runes := []rune(text)
	if len(runes) == 0 {
		return nil
	}

	step := chunkSize - overlap
	if step <= 0 {
		step = chunkSize
	}

	out := make([]string, 0, 1+(len(runes)/step))
	for start := 0; start < len(runes); start += step {
		end := min(start+chunkSize, len(runes))

		out = append(out, strings.TrimSpace(string(runes[start:end])))
		if end == len(runes) {
			break
		}
	}

	return out
}

func runeLen(text string) int {
	return len([]rune(text))
}

// Markdown splits markdown documents by headings first and then recursively.
type Markdown struct {
	base *Recursive
}

// NewMarkdown constructs a markdown splitter.
func NewMarkdown(base *Recursive) (*Markdown, error) {
	if base == nil {
		return nil, fmt.Errorf("%w: markdown base splitter", ragy.ErrInvalidArgument)
	}

	return &Markdown{base: base}, nil
}

// Split splits a markdown document.
func (m *Markdown) Split(ctx context.Context, doc ragy.Document) ([]ragy.Chunk, error) {
	normalized, err := validateSource(doc)
	if err != nil {
		return nil, err
	}

	sections := splitMarkdownSections(normalized.Content)
	if len(sections) == 0 {
		return m.base.Split(ctx, normalized)
	}

	var parts []string
	for _, section := range sections {
		parts = append(parts, splitIterative(section, m.base.chunkSize, m.base.overlap, m.base.separators)...)
	}

	chunks := buildChunks(normalized, parts)
	for i, chunk := range chunks {
		normalizedChunk, err := ragy.NormalizeChunk(chunk)
		if err != nil {
			return nil, err
		}
		chunks[i] = normalizedChunk
	}
	return chunks, nil
}

func splitMarkdownSections(text string) []string {
	lines := strings.Split(text, "\n")
	var sections []string
	var current []string

	for _, line := range lines {
		if strings.HasPrefix(strings.TrimSpace(line), "#") && len(current) > 0 {
			sections = append(sections, strings.Join(current, "\n"))
			current = current[:0]
		}

		current = append(current, line)
	}

	if len(current) > 0 {
		sections = append(sections, strings.Join(current, "\n"))
	}

	return sections
}

// DefaultSentenceSegmenter is a UTF-8-safe punctuation-based segmenter.
type DefaultSentenceSegmenter struct{}

// Split segments text into sentences.
func (DefaultSentenceSegmenter) Split(text string) []string {
	runes := []rune(text)
	if len(runes) == 0 {
		return nil
	}

	var out []string
	var start int
	for index, r := range runes {
		if !isSentenceBoundary(r) {
			continue
		}

		next := index + 1
		sentence := strings.TrimSpace(string(runes[start : index+1]))
		if sentence != "" {
			out = append(out, sentence)
		}
		start = next
	}

	if start < len(runes) {
		tail := strings.TrimSpace(string(runes[start:]))
		if tail != "" {
			out = append(out, tail)
		}
	}

	return out
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
type Semantic struct {
	embedder  dense.Embedder
	segmenter SentenceSegmenter
	threshold float64
	minGroup  int
}

// NewSemantic constructs a semantic splitter with an explicit sentence segmentation strategy.
func NewSemantic(
	embedder dense.Embedder,
	segmenter SentenceSegmenter,
	threshold float64,
	minGroup int,
) (*Semantic, error) {
	if embedder == nil {
		return nil, fmt.Errorf("%w: semantic embedder", ragy.ErrInvalidArgument)
	}
	if segmenter == nil {
		return nil, fmt.Errorf("%w: semantic sentence segmenter", ragy.ErrInvalidArgument)
	}

	if threshold < -1 || threshold > 1 {
		return nil, fmt.Errorf("%w: semantic threshold must be in [-1,1]", ragy.ErrInvalidArgument)
	}

	if minGroup <= 0 {
		return nil, fmt.Errorf("%w: min group must be > 0", ragy.ErrInvalidArgument)
	}

	return &Semantic{
		embedder:  embedder,
		segmenter: segmenter,
		threshold: threshold,
		minGroup:  minGroup,
	}, nil
}

// Split splits a source document by semantic boundaries.
func (s *Semantic) Split(ctx context.Context, doc ragy.Document) ([]ragy.Chunk, error) {
	normalized, err := validateSource(doc)
	if err != nil {
		return nil, err
	}

	sentences := s.segmenter.Split(normalized.Content)
	if len(sentences) == 0 {
		return nil, fmt.Errorf("%w: semantic sentence segmentation returned no sentences", ragy.ErrProtocol)
	}

	embeddings, err := s.embedder.Embed(ctx, sentences)
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

	if err := validateSemanticEmbeddings(embeddings); err != nil {
		return nil, err
	}

	parts := semanticGroups(sentences, embeddings, s.threshold, s.minGroup)
	chunks := buildChunks(normalized, parts)
	for i, chunk := range chunks {
		normalizedChunk, err := ragy.NormalizeChunk(chunk)
		if err != nil {
			return nil, err
		}
		chunks[i] = normalizedChunk
	}
	return chunks, nil
}

func validateSemanticEmbeddings(embeddings [][]float32) error {
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

func semanticGroups(sentences []string, embeddings [][]float32, threshold float64, minGroup int) []string {
	if len(sentences) == 0 {
		return nil
	}

	var parts []string
	groupStart := 0
	for index := 1; index < len(sentences); index++ {
		if index-groupStart < minGroup {
			continue
		}

		if cosine(embeddings[index-1], embeddings[index]) >= threshold {
			continue
		}

		parts = append(parts, strings.Join(sentences[groupStart:index], " "))
		groupStart = index
	}

	parts = append(parts, strings.Join(sentences[groupStart:], " "))
	return parts
}

func cosine(left, right []float32) float64 {
	if len(left) == 0 || len(right) == 0 || len(left) != len(right) {
		return 0
	}

	dot := 0.0
	leftNorm := 0.0
	rightNorm := 0.0
	for index := range left {
		lv := float64(left[index])
		rv := float64(right[index])
		dot += lv * rv
		leftNorm += lv * lv
		rightNorm += rv * rv
	}

	if leftNorm == 0 || rightNorm == 0 {
		return 0
	}

	return dot / (sqrt(leftNorm) * sqrt(rightNorm))
}

func sqrt(value float64) float64 {
	z := value
	if z == 0 {
		return 0
	}

	for range 8 {
		z -= (z*z - value) / (2 * z)
	}

	return z
}

// Contextual augments chunks with derived context.
type Contextual struct {
	base        Splitter
	generator   ContextGenerator
	concurrency int
}

// NewContextual constructs a contextual splitter.
func NewContextual(base Splitter, generator ContextGenerator, concurrency int) (*Contextual, error) {
	if base == nil {
		return nil, fmt.Errorf("%w: contextual base splitter", ragy.ErrInvalidArgument)
	}

	if generator == nil {
		return nil, fmt.Errorf("%w: contextual generator", ragy.ErrInvalidArgument)
	}

	if concurrency <= 0 {
		return nil, fmt.Errorf("%w: contextual concurrency", ragy.ErrInvalidArgument)
	}

	return &Contextual{
		base:        base,
		generator:   generator,
		concurrency: concurrency,
	}, nil
}

// Split splits a source document and enriches chunk context in parallel.
func (c *Contextual) Split(ctx context.Context, doc ragy.Document) ([]ragy.Chunk, error) {
	chunks, err := c.base.Split(ctx, doc)
	if err != nil {
		return nil, err
	}

	enriched, err := parallel.MapOrdered(
		ctx,
		c.concurrency,
		chunks,
		func(ctx context.Context, chunk ragy.Chunk) (ragy.Chunk, error) {
			contextText, contextErr := c.generator.Context(ctx, doc, chunk)
			if contextErr != nil {
				return ragy.Chunk{}, contextErr
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
