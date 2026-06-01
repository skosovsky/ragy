package graph

import (
	"encoding/json"

	"github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/filter"
)

// NormalizeMeta validates and canonicalizes typed metadata against a schema.
func NormalizeMeta[TMeta any](schema filter.Schema, meta TMeta) (TMeta, error) {
	attrs, err := metaToRawAttributes(meta)
	if err != nil {
		var zero TMeta
		return zero, err
	}

	normalized, err := schema.NormalizeAttributes(attrs)
	if err != nil {
		var zero TMeta
		return zero, err
	}

	return rawAttributesToMeta[TMeta](normalized)
}

func metaToRawAttributes[TMeta any](meta TMeta) (filter.RawAttributes, error) {
	if attrs, ok := dense.MetaRawAttributes(meta); ok {
		return attrs, nil
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
	return attrs, nil
}

func rawAttributesToMeta[TMeta any](attrs filter.RawAttributes) (TMeta, error) {
	var out TMeta
	if len(attrs) == 0 {
		return out, nil
	}

	data, err := json.Marshal(attrs)
	if err != nil {
		return out, err
	}
	if err := json.Unmarshal(data, &out); err != nil {
		return out, err
	}
	return out, nil
}
