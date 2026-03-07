package ragy

import "errors"

// Sentinel errors for ragy operations.
// Use errors.Is(err, ragy.ErrNotFound) for checks.
var (
	ErrNotFound        = errors.New("ragy: not found")
	ErrInvalidInput    = errors.New("ragy: invalid input")
	ErrEmptyQuery      = errors.New("ragy: empty query")
	ErrNoResults       = errors.New("ragy: no results")
	ErrEmbeddingFailed = errors.New("ragy: embedding failed")
)
