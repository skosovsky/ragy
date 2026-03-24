// Package dense provides dense-vector contracts.
package dense

import (
	"context"
	"fmt"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
)

// Request is a typed dense-search request.
type Request struct {
	Vector []float32
	Filter filter.IR
	Page   *ragy.Page
}

// Validate checks request invariants.
func (r Request) Validate() error {
	if len(r.Vector) == 0 {
		return fmt.Errorf("%w: dense request vector", ragy.ErrEmptyVector)
	}

	if err := filter.ValidateIR(r.Filter); err != nil {
		return err
	}

	return r.Page.Validate()
}

// Record is a typed dense-index record.
type Record struct {
	ID         string
	Content    string
	Attributes ragy.Attributes
	Vector     []float32
}

// Validate checks record invariants.
func (r Record) Validate() error {
	if r.ID == "" {
		return fmt.Errorf("%w: dense record id", ragy.ErrMissingID)
	}

	if _, err := ragy.NormalizeAttributes(r.Attributes); err != nil {
		return err
	}

	if len(r.Vector) == 0 {
		return fmt.Errorf("%w: dense record vector", ragy.ErrEmptyVector)
	}

	return nil
}

// Searcher executes dense-vector search.
type Searcher interface {
	Search(ctx context.Context, req Request) ([]ragy.Document, error)
	Schema() filter.Schema
}

// Index writes dense-vector records.
type Index interface {
	Upsert(ctx context.Context, records []Record) error
	Schema() filter.Schema
}

// Embedder produces dense embeddings.
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}
