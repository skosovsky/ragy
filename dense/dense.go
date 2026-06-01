package dense

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
)

// Record is a typed dense-index record.
type Record[TMeta any] struct {
	ID      string
	Content string
	Meta    TMeta
	Vector  []float32
}

// Validate checks record invariants.
func (r Record[TMeta]) Validate() error {
	if r.ID == "" {
		return fmt.Errorf("%w: dense record id", ragy.ErrMissingID)
	}
	if len(r.Vector) == 0 {
		return fmt.Errorf("%w: dense record vector", ragy.ErrEmptyVector)
	}
	return nil
}

// MarshalMeta serializes typed metadata for backend storage.
func (r Record[TMeta]) MarshalMeta() ([]byte, error) {
	return json.Marshal(r.Meta)
}

// MetaRawAttributes extracts string-keyed attributes when meta is a map type.
func MetaRawAttributes[TMeta any](meta TMeta) (filter.RawAttributes, bool) {
	value := reflect.ValueOf(meta)
	if !value.IsValid() {
		return nil, false
	}
	if value.Kind() == reflect.Pointer {
		if value.IsNil() {
			return nil, false
		}
		value = value.Elem()
	}
	if value.Kind() != reflect.Map || value.Type().Key().Kind() != reflect.String {
		return nil, false
	}
	if value.IsNil() {
		return nil, false
	}

	attrs := make(filter.RawAttributes, value.Len())
	for _, key := range value.MapKeys() {
		attrs[key.String()] = value.MapIndex(key).Interface()
	}
	return attrs, true
}

// NormalizeRecordMeta validates and canonicalizes record metadata against a schema.
func NormalizeRecordMeta[TMeta any](schema filter.Schema, meta TMeta) (filter.RawAttributes, error) {
	if attrs, ok := MetaRawAttributes(meta); ok {
		return schema.NormalizeAttributes(attrs)
	}

	data, err := json.Marshal(meta)
	if err != nil {
		return nil, err
	}
	if string(data) == "null" {
		return filter.RawAttributes{}, nil
	}

	attrs := make(filter.RawAttributes)
	if err := json.Unmarshal(data, &attrs); err != nil {
		return nil, err
	}
	return schema.NormalizeAttributes(attrs)
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
	return json.Marshal(normalized)
}

// Index writes dense-vector records.
type Index[TMeta any] interface {
	Upsert(ctx context.Context, records []Record[TMeta]) error
	Schema() filter.Schema
}

// Embedder produces dense embeddings.
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}
