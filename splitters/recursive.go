package splitters

import (
	"context"
	"iter"
	"maps"
	"strings"

	"github.com/skosovsky/ragy"
)

// DefaultRecursiveSeparators is the default list of separators (largest to smallest).
var DefaultRecursiveSeparators = []string{"\n\n", "\n", ". ", " "}

// RecursiveSplitter splits text recursively by a list of separators, respecting ChunkSize and ChunkOverlap.
type RecursiveSplitter struct {
	ChunkSize    int
	ChunkOverlap int
	Separators   []string
}

// RecursiveOption configures RecursiveSplitter.
type RecursiveOption func(*RecursiveSplitter)

// WithChunkSize sets the target chunk size in characters.
func WithChunkSize(n int) RecursiveOption {
	return func(r *RecursiveSplitter) {
		r.ChunkSize = n
	}
}

// WithChunkOverlap sets the overlap between consecutive chunks.
func WithChunkOverlap(n int) RecursiveOption {
	return func(r *RecursiveSplitter) {
		r.ChunkOverlap = n
	}
}

// WithSeparators sets the list of separators (tried in order, largest first).
func WithSeparators(sep []string) RecursiveOption {
	return func(r *RecursiveSplitter) {
		r.Separators = sep
	}
}

// NewRecursiveSplitter returns a RecursiveSplitter with the given options.
func NewRecursiveSplitter(opts ...RecursiveOption) *RecursiveSplitter {
	r := &RecursiveSplitter{
		ChunkSize:    1000,
		ChunkOverlap: 200,
		Separators:   append([]string(nil), DefaultRecursiveSeparators...),
	}
	for _, o := range opts {
		o(r)
	}
	return r
}

// Split implements Splitter. Yields chunks with inherited metadata plus ParentID and ChunkIndex.
func (r *RecursiveSplitter) Split(ctx context.Context, doc ragy.Document) iter.Seq2[ragy.Document, error] {
	return func(yield func(ragy.Document, error) bool) {
		text := strings.TrimSpace(doc.Content)
		if text == "" {
			return
		}
		chunks := r.splitRecursive(text, r.Separators)
		for i, content := range chunks {
			if ctx.Err() != nil {
				_ = yield(ragy.Document{}, ctx.Err())
				return
			}
			meta := make(map[string]any)
			maps.Copy(meta, doc.Metadata)
			meta["ParentID"] = doc.ID
			meta["ChunkIndex"] = i
			chunk := ragy.Document{
				ID:       doc.ID + "_" + itoa(i),
				Content:  content,
				Metadata: meta,
			}
			if doc.ID == "" {
				chunk.ID = itoa(i)
			}
			if !yield(chunk, nil) {
				return
			}
		}
	}
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b [20]byte
	pos := len(b)
	for i > 0 {
		pos--
		b[pos] = byte('0' + i%10)
		i /= 10
	}
	return string(b[pos:])
}

func (r *RecursiveSplitter) splitRecursive(text string, separators []string) []string {
	if r.ChunkSize <= 0 || len(text) <= r.ChunkSize {
		if text != "" {
			return []string{text}
		}
		return nil
	}
	var sep string
	var rest []string
	for _, s := range separators {
		if strings.Contains(text, s) {
			sep = s
			rest = separators
			break
		}
	}
	if sep == "" {
		return r.splitByLength(text)
	}
	parts := strings.Split(text, sep)
	var out []string
	var buf strings.Builder
	for i, p := range parts {
		piece := p
		if i < len(parts)-1 {
			piece += sep
		}
		if buf.Len()+len(piece) <= r.ChunkSize {
			if buf.Len() > 0 {
				buf.WriteString(piece)
			} else {
				buf.WriteString(strings.TrimSpace(piece))
			}
			continue
		}
		if buf.Len() > 0 {
			out = append(out, strings.TrimSpace(buf.String()))
			buf.Reset()
		}
		// Recurse on the part without sep to avoid infinite recursion (e.g. "abc\n\n" -> parts ["abc",""]).
		if len(p) > r.ChunkSize {
			sub := r.splitRecursive(p, rest)
			out = append(out, sub...)
			continue
		}
		buf.WriteString(piece)
	}
	if buf.Len() > 0 {
		out = append(out, strings.TrimSpace(buf.String()))
	}
	return out
}

func (r *RecursiveSplitter) splitByLength(text string) []string {
	var out []string
	overlap := r.ChunkOverlap
	if overlap < 0 || overlap >= r.ChunkSize {
		overlap = 0
	}
	start := 0
	for start < len(text) {
		end := min(start+r.ChunkSize, len(text))
		out = append(out, text[start:end])
		start = end - overlap
		if start >= len(text) {
			break
		}
	}
	return out
}
