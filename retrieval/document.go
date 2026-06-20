package retrieval

import (
	"fmt"
	"math"

	ragy "github.com/skosovsky/ragy"
)

// ScoreState describes whether Document.Score is native, absent, or derived by
// an explicit normalization policy.
type ScoreState int

const (
	// ScorePresent means Score is a native source score.
	ScorePresent ScoreState = iota
	// ScoreAbsent means the source is rank-only and Score must not be treated as
	// relevance evidence.
	ScoreAbsent
	// ScoreNormalized means Score was produced by an explicit ScorePolicy.
	ScoreNormalized
)

// IsScored reports whether the document has a usable numeric score.
func (s ScoreState) IsScored() bool {
	return s == ScorePresent || s == ScoreNormalized
}

// Document is a strictly typed retrieval result.
type Document[TMeta any] struct {
	ID         string
	Content    string
	Score      float64
	ScoreState ScoreState
	Rank       int
	Meta       TMeta
}

// ValidateDocument checks invariants for a document payload.
func ValidateDocument[TMeta any](d Document[TMeta]) error {
	if d.ID == "" {
		return fmt.Errorf("%w: document id", ragy.ErrMissingID)
	}
	if d.ScoreState != ScorePresent && d.ScoreState != ScoreAbsent && d.ScoreState != ScoreNormalized {
		return fmt.Errorf("%w: document score state", ragy.ErrInvalidArgument)
	}
	if d.Rank < 0 {
		return fmt.Errorf("%w: document rank must be >= 0", ragy.ErrInvalidArgument)
	}
	if !d.ScoreState.IsScored() {
		if math.IsNaN(d.Score) || math.IsInf(d.Score, 0) || d.Score != 0 {
			return fmt.Errorf("%w: score-absent document must not carry score", ragy.ErrInvalidArgument)
		}
		return nil
	}
	if math.IsNaN(d.Score) || math.IsInf(d.Score, 0) || d.Score < 0 || d.Score > 1 {
		return fmt.Errorf("%w: document score must be in [0,1]", ragy.ErrInvalidArgument)
	}
	return nil
}
