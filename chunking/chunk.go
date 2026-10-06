package chunking

import (
	"fmt"
	"strings"

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

// ValidateChunk checks chunk invariants.
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
