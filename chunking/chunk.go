package chunking

import (
	"fmt"
	"strings"

	ragy "github.com/skosovsky/ragy"
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
	return nil
}
