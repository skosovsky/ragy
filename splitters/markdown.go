package splitters

import (
	"context"
	"iter"
	"maps"
	"regexp"
	"strings"

	"github.com/skosovsky/ragy"
)

// MarkdownSplitter splits Markdown by headers (#, ##, ###) and preserves fenced code blocks.
type MarkdownSplitter struct {
	// HeaderPattern matches ATX-style headers (e.g. ## Section).
	HeaderPattern *regexp.Regexp
}

// NewMarkdownSplitter returns a new MarkdownSplitter.
func NewMarkdownSplitter() *MarkdownSplitter {
	return &MarkdownSplitter{
		HeaderPattern: regexp.MustCompile(`(?m)^(#{1,6})\s+(.+)$`),
	}
}

// Split implements Splitter.
func (m *MarkdownSplitter) Split(ctx context.Context, doc ragy.Document) iter.Seq2[ragy.Document, error] {
	return func(yield func(ragy.Document, error) bool) {
		text := doc.Content
		if strings.TrimSpace(text) == "" {
			return
		}
		chunks := m.splitMarkdown(text)
		for i, content := range chunks {
			if ctx.Err() != nil {
				_ = yield(ragy.Document{}, ctx.Err())
				return
			}
			meta := make(map[string]any)
			maps.Copy(meta, doc.Metadata)
			meta["ParentID"] = doc.ID
			meta["ChunkIndex"] = i
			chunkID := doc.ID + "_" + itoa(i)
			if doc.ID == "" {
				chunkID = itoa(i)
			}
			chunk := ragy.Document{
				ID:       chunkID,
				Content:  strings.TrimSpace(content),
				Metadata: meta,
			}
			if !yield(chunk, nil) {
				return
			}
		}
	}
}

// splitMarkdown splits by headers but keeps fenced code blocks intact.
func (m *MarkdownSplitter) splitMarkdown(text string) []string {
	// Replace fenced code blocks with placeholders so we don't split inside them.
	placeholders := make([]string, 0)
	fencedPattern := regexp.MustCompile("(?ms)```[^`]*?```")
	cleaned := fencedPattern.ReplaceAllStringFunc(text, func(match string) string {
		idx := len(placeholders)
		placeholders = append(placeholders, match)
		return "\n__FENCED_" + itoa(idx) + "__\n"
	})

	var chunks []string
	indices := m.HeaderPattern.FindAllStringIndex(cleaned, -1)
	if len(indices) == 0 {
		if strings.TrimSpace(cleaned) != "" {
			chunks = append(chunks, m.restoreFenced(cleaned, placeholders))
		}
		return chunks
	}
	for i := range indices {
		start := indices[i][0]
		var end int
		if i+1 < len(indices) {
			end = indices[i+1][0]
		} else {
			end = len(cleaned)
		}
		block := cleaned[start:end]
		block = m.restoreFenced(block, placeholders)
		if strings.TrimSpace(block) != "" {
			chunks = append(chunks, block)
		}
	}
	return chunks
}

func (m *MarkdownSplitter) restoreFenced(s string, placeholders []string) string {
	pat := regexp.MustCompile("__FENCED_([0-9]+)__")
	return pat.ReplaceAllStringFunc(s, func(match string) string {
		// match is "__FENCED_0__" or "__FENCED_12__"
		inner := strings.TrimSuffix(strings.TrimPrefix(match, "__FENCED_"), "__")
		i := 0
		for _, c := range inner {
			if c >= '0' && c <= '9' {
				i = i*10 + int(c-'0')
			}
		}
		if i < len(placeholders) {
			return placeholders[i]
		}
		return match
	})
}
