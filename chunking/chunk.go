package chunking

import (
	"fmt"
	"strings"
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/source"
)

// Chunk is a typed document fragment produced by a splitter.
type Chunk[TMeta any] struct {
	ID       string
	SourceID string
	Index    int
	Total    int
	Content  string
	Context  string
	Meta     TMeta
	// InputSpan addresses original input bytes, never a guessed retained offset.
	InputSpan      source.ByteSpan
	SourceMapping  source.MappedText
	SourceSupports []source.Locator
}

// ValidateChunk checks standalone shape. Total=0 means unknown. InputSpan cannot
// be checked against original source bytes without the source document.
func ValidateChunk[TMeta any](c Chunk[TMeta]) error {
	if c.ID == "" {
		return fmt.Errorf("%w: chunk id", ragy.ErrMissingID)
	}
	if c.SourceID == "" {
		return fmt.Errorf("%w: chunk source id", ragy.ErrMissingSourceID)
	}
	if c.Index < 0 {
		return fmt.Errorf("%w: chunk index must be >= 0", ragy.ErrInvalidArgument)
	}
	if c.Total < 0 {
		return fmt.Errorf("%w: chunk total must be >= 0", ragy.ErrInvalidArgument)
	}
	if c.Total > 0 && c.Index >= c.Total {
		return fmt.Errorf("%w: chunk index must be less than total", ragy.ErrInvalidArgument)
	}
	if !utf8.ValidString(c.Content) || !utf8.ValidString(c.Context) {
		return ragy.ErrInvalidArgument
	}
	if c.InputSpan != (source.ByteSpan{Start: 0, End: 0}) &&
		(c.InputSpan.Start < 0 || c.InputSpan.End <= c.InputSpan.Start) {
		return ragy.ErrInvalidArgument
	}
	if strings.TrimSpace(c.Content) == "" {
		return fmt.Errorf("%w: chunk content", ragy.ErrEmptyText)
	}
	if c.SourceMapping.Text() != "" && (c.SourceMapping.Text() != c.Content || c.SourceMapping.Validate() != nil) {
		return ragy.ErrInvalidArgument
	}
	for _, support := range c.SourceSupports {
		if err := support.Validate(); err != nil {
			return err
		}
	}
	return nil
}
