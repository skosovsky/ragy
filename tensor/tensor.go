// Package tensor provides late-interaction tensor contracts.
package tensor

import (
	"context"
	"encoding/json"
	"fmt"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
)

// Tensor is a document or query token matrix.
type Tensor [][]float32

// Record is a typed tensor-index record.
type Record[TMeta any] struct {
	ID      string
	Content string
	Meta    TMeta
	Tensor  Tensor
}

// Validate checks record invariants.
func (r Record[TMeta]) Validate() error {
	if r.ID == "" {
		return fmt.Errorf("%w: tensor record id", ragy.ErrMissingID)
	}
	if len(r.Tensor) == 0 {
		return fmt.Errorf("%w: tensor record", ragy.ErrEmptyVector)
	}
	return nil
}

// MarshalMeta serializes typed metadata for backend storage.
func (r Record[TMeta]) MarshalMeta() ([]byte, error) {
	return json.Marshal(r.Meta)
}

// Index writes tensor records.
type Index[TMeta any] interface {
	Upsert(ctx context.Context, records []Record[TMeta]) error
	Schema() filter.Schema
}

// Embedder produces tensor embeddings.
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([]Tensor, error)
}
