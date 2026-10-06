package dense

import (
	"context"
	"encoding/json"
	"fmt"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/embedding"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
)

// Record is a typed dense-index record.
type Record[TMeta any] struct {
	ID      string
	Content string
	Meta    TMeta
	Vector  []float32
	Space   Space
}

// Validate checks record invariants.
func (r Record[TMeta]) Validate() error {
	if r.ID == "" {
		return fmt.Errorf("%w: dense record id", ragy.ErrMissingID)
	}
	if len(r.Vector) == 0 {
		return ragy.ErrEmptyVector
	}
	return (Embedding{Space: r.Space, Vector: r.Vector}).Validate()
}

// NormalizeRecordMeta validates and canonicalizes record metadata against a schema.
func NormalizeRecordMeta[TMeta any](schema filter.Schema, meta TMeta) (filter.RawAttributes, error) {
	return retrieval.NewJSONCodec[TMeta](schema).Encode(meta)
}

// MarshalMetaForSchema serializes schema-normalized metadata for backend storage.
func (r Record[TMeta]) MarshalMetaForSchema(schema filter.Schema) ([]byte, error) {
	normalized, err := NormalizeRecordMeta(schema, r.Meta)
	if err != nil {
		return nil, err
	}
	if len(normalized) == 0 {
		return []byte("null"), nil
	}
	data, err := json.Marshal(normalized)
	if err != nil {
		return nil, fmt.Errorf("%w: marshal record meta: %w", ragy.ErrInvalidArgument, err)
	}
	return data, nil
}

// Index writes dense-vector records.
type Index[TMeta any] interface {
	Upsert(ctx context.Context, records []Record[TMeta]) error
	Schema() filter.Schema
}

// Embedder produces dense embeddings.
type Embedder interface {
	Space() Space
	Embed(ctx context.Context, request Request) (Result, error)
}

type Request = embedding.Request[string]
type Result = embedding.Result[Embedding]
