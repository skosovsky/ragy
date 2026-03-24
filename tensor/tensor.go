// Package tensor provides late-interaction tensor contracts.
package tensor

import (
	"context"
	"fmt"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
)

// Request is a typed tensor-search request.
type Request struct {
	Query  Tensor
	Filter filter.IR
	Page   *ragy.Page
}

// Validate checks request invariants.
func (r Request) Validate() error {
	if len(r.Query) == 0 {
		return fmt.Errorf("%w: tensor request", ragy.ErrEmptyVector)
	}

	if err := filter.ValidateIR(r.Filter); err != nil {
		return err
	}

	return r.Page.Validate()
}

// Tensor is a document or query token matrix.
type Tensor [][]float32

// Record is a typed tensor-index record.
type Record struct {
	ID         string
	Content    string
	Attributes ragy.Attributes
	Tensor     Tensor
}

// Validate checks record invariants.
func (r Record) Validate() error {
	if r.ID == "" {
		return fmt.Errorf("%w: tensor record id", ragy.ErrMissingID)
	}

	if _, err := ragy.NormalizeAttributes(r.Attributes); err != nil {
		return err
	}

	if len(r.Tensor) == 0 {
		return fmt.Errorf("%w: tensor record", ragy.ErrEmptyVector)
	}

	return nil
}

// Searcher executes tensor search.
type Searcher interface {
	Search(ctx context.Context, req Request) ([]ragy.Document, error)
	Schema() filter.Schema
}

// Index writes tensor records.
type Index interface {
	Upsert(ctx context.Context, records []Record) error
	Schema() filter.Schema
}

// Embedder produces tensor embeddings.
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([]Tensor, error)
}
