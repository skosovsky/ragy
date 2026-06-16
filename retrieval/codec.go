package retrieval

import (
	"encoding/json"
	"fmt"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/internal/metaattrs"
)

// MetadataCodec serializes typed metadata to storage attributes and back.
type MetadataCodec[TMeta any] interface {
	Encode(meta TMeta) (filter.RawAttributes, error)
	Decode(attrs filter.RawAttributes) (TMeta, error)
}

// JSONCodec encodes struct metadata through schema-normalized JSON attributes.
type JSONCodec[TMeta any] struct {
	schema filter.Schema
}

// NewJSONCodec constructs a schema-aware JSON metadata codec.
func NewJSONCodec[TMeta any](schema filter.Schema) JSONCodec[TMeta] {
	return JSONCodec[TMeta]{schema: schema}
}

// Encode validates and canonicalizes metadata for storage.
func (c JSONCodec[TMeta]) Encode(meta TMeta) (filter.RawAttributes, error) {
	if _, ok := metaattrs.FromValue(meta); ok {
		return nil, fmt.Errorf("%w: map metadata is not supported in public API", ragy.ErrInvalidArgument)
	}

	data, err := json.Marshal(meta)
	if err != nil {
		return nil, fmt.Errorf("%w: encode metadata: %w", ragy.ErrInvalidArgument, err)
	}
	if string(data) == "null" {
		return filter.RawAttributes{}, nil
	}

	attrs := make(filter.RawAttributes)
	if err := json.Unmarshal(data, &attrs); err != nil {
		return nil, fmt.Errorf("%w: encode metadata json: %w", ragy.ErrInvalidArgument, err)
	}
	return c.schema.NormalizeAttributes(attrs)
}

// Decode normalizes raw attributes and unmarshals into TMeta without panicking on partial data.
func (c JSONCodec[TMeta]) Decode(attrs filter.RawAttributes) (TMeta, error) {
	var meta TMeta

	normalized, err := c.schema.NormalizeAttributes(attrs)
	if err != nil {
		return meta, err
	}
	if len(normalized) == 0 {
		return meta, nil
	}

	data, err := json.Marshal(normalized)
	if err != nil {
		return meta, ragy.WrapProjectionError(err, "decode metadata marshal")
	}
	if err := json.Unmarshal(data, &meta); err != nil {
		return meta, ragy.WrapProjectionError(err, "decode metadata")
	}
	return meta, nil
}
