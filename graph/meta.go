package graph

import (
	"encoding/json"
	"fmt"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/internal/metaattrs"
)

// NormalizeMeta validates and canonicalizes typed metadata against a schema.
func NormalizeMeta[TMeta any](schema filter.Schema, meta TMeta) (TMeta, error) {
	attrs, err := encodeMeta(schema, meta)
	if err != nil {
		var zero TMeta
		return zero, err
	}
	return decodeMeta[TMeta](schema, attrs)
}

func encodeMeta[TMeta any](schema filter.Schema, meta TMeta) (filter.RawAttributes, error) {
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
	return schema.NormalizeAttributes(attrs)
}

func decodeMeta[TMeta any](schema filter.Schema, attrs filter.RawAttributes) (TMeta, error) {
	var meta TMeta

	normalized, err := schema.NormalizeAttributes(attrs)
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
