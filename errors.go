package ragy

import "errors"

// Sentinel errors for ragy operations.
// Use errors.Is(err, ragy.ErrNotFound) for checks.
var (
	ErrNotFound           = errors.New("ragy: not found")
	ErrInvalidInput       = errors.New("ragy: invalid input")
	ErrEmptyQuery         = errors.New("ragy: empty query")
	ErrNoResults          = errors.New("ragy: no results")
	ErrEmbeddingFailed    = errors.New("ragy: embedding failed")
	ErrMissingParsedQuery = errors.New("ragy: SearchRequest.ParsedQuery is required for SelfQueryRetriever")
	ErrMissingGraphSeeds  = errors.New("ragy: SearchRequest.GraphSeedEntityIDs is required for GraphRetriever")
	// ErrSparseVectorNotSupported is returned by a VectorStore when SearchRequest.SparseVector is set but the adapter has no sparse index path yet.
	ErrSparseVectorNotSupported = errors.New("ragy: sparse vector search not supported by this store")
)
