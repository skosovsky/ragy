package retrieval

import (
	"fmt"
	"math"

	ragy "github.com/skosovsky/ragy"
)

// Document is a strictly typed retrieval result.
type Document[TMeta any] struct {
	ID      string
	Content string
	Score   float64
	Meta    TMeta
}

// ValidateDocument checks invariants for a document payload.
func ValidateDocument[TMeta any](d Document[TMeta]) error {
	if d.ID == "" {
		return fmt.Errorf("%w: document id", ragy.ErrMissingID)
	}
	if math.IsNaN(d.Score) || math.IsInf(d.Score, 0) || d.Score < 0 || d.Score > 1 {
		return fmt.Errorf("%w: document score must be in [0,1]", ragy.ErrInvalidArgument)
	}
	return nil
}
