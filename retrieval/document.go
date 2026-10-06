package retrieval

import (
	"fmt"
	"math"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/source"
)

// ScoreState describes whether Document.Score is native, absent, or derived by
// an explicit normalization policy.
type ScoreState int

const (
	// ScoreAbsent is rank-only evidence; the zero value carries no numeric score.
	ScoreAbsent ScoreState = iota
	// ScorePresent is a finite native score in its declared semantics.
	ScorePresent
	// ScoreNormalized is a score explicitly transformed into [0,1].
	ScoreNormalized
)

// ScoreSemantics identifies a comparable score scale, including relevant configuration.
// Hosts must distinguish different model/configuration scales, even for equal algorithms.
type ScoreSemantics string

// IsScored reports whether the document has a usable numeric score.
func (s ScoreState) IsScored() bool {
	return s == ScorePresent || s == ScoreNormalized
}

// Document is a strictly typed retrieval result.
type Document[TMeta any] struct {
	ID             string
	Content        string
	Score          float64
	ScoreState     ScoreState
	ScoreSemantics ScoreSemantics
	ScoreHistory   []ScoreObservation
	Rank           int
	Meta           TMeta
	// SourceMapping addresses Content only; zero means unobserved precision.
	SourceMapping source.MappedText
	// SourceSupports retains every original contributor, including dedup losers.
	SourceSupports []source.Locator
}

// ValidateDocument checks invariants for a document payload.
func ValidateDocument[TMeta any](d Document[TMeta]) error {
	if err := validateDocumentSources(d); err != nil {
		return err
	}
	if d.ID == "" {
		return fmt.Errorf("%w: document id", ragy.ErrMissingID)
	}
	if d.ScoreState != ScorePresent && d.ScoreState != ScoreAbsent && d.ScoreState != ScoreNormalized {
		return fmt.Errorf("%w: document score state", ragy.ErrInvalidArgument)
	}
	if d.Rank < 0 {
		return fmt.Errorf("%w: document rank must be >= 0", ragy.ErrInvalidArgument)
	}
	for _, observation := range d.ScoreHistory {
		if observation.DocumentID == "" || !observation.State.IsScored() || observation.Semantics == "" ||
			math.IsNaN(observation.Value) ||
			math.IsInf(observation.Value, 0) ||
			(observation.State == ScoreNormalized && (observation.Value < 0 || observation.Value > 1)) {
			return fmt.Errorf("%w: invalid score history", ragy.ErrInvalidArgument)
		}
	}
	if !d.ScoreState.IsScored() {
		if math.IsNaN(d.Score) || math.IsInf(d.Score, 0) || d.Score != 0 || d.ScoreSemantics != "" {
			return fmt.Errorf("%w: score-absent document must not carry score", ragy.ErrInvalidArgument)
		}
		return nil
	}
	if math.IsNaN(d.Score) || math.IsInf(d.Score, 0) || d.ScoreSemantics == "" {
		return fmt.Errorf("%w: scored document requires finite value and semantics", ragy.ErrInvalidArgument)
	}
	if d.ScoreState == ScoreNormalized && (d.Score < 0 || d.Score > 1) {
		return fmt.Errorf("%w: normalized score must be in [0,1]", ragy.ErrInvalidArgument)
	}
	return nil
}

// validateComparable rejects ordering/arithmetic across undeclared or incompatible scales.
func validateComparable[TMeta any](docs []Document[TMeta]) error {
	var reference *Document[TMeta]
	for i := range docs {
		doc := &docs[i]
		if err := ValidateDocument(*doc); err != nil {
			return err
		}
		if reference == nil {
			reference = doc
			continue
		}
		if reference.ScoreState.IsScored() != doc.ScoreState.IsScored() ||
			reference.ScoreSemantics != doc.ScoreSemantics ||
			reference.ScoreState != doc.ScoreState {
			return fmt.Errorf(
				"%w: incompatible score scales; use explicit rank fusion or normalization",
				ragy.ErrInvalidArgument,
			)
		}
	}
	return nil
}

// ScoreObservation is value-only evidence of an input score before transformation.
// DocumentID remains the original contributor ID through merging and fusion.
type ScoreObservation struct {
	DocumentID string
	Value      float64
	State      ScoreState
	Semantics  ScoreSemantics
}

// ObservedScores snapshots previous observations and the current numeric score.
// Rank-only evidence contributes no fabricated numeric zero.
func (d Document[TMeta]) ObservedScores() []ScoreObservation {
	out := append([]ScoreObservation(nil), d.ScoreHistory...)
	if d.ScoreState.IsScored() {
		out = append(
			out,
			ScoreObservation{DocumentID: d.ID, Value: d.Score, State: d.ScoreState, Semantics: d.ScoreSemantics},
		)
	}
	return out
}
