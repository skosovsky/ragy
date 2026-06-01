package filter

import (
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"reflect"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/internal/ident"
)

// RawAttributes stores scalar metadata at storage adapter boundaries.
type RawAttributes map[string]any

// CloneRawAttributes returns a shallow copy of raw attributes.
func CloneRawAttributes(in RawAttributes) RawAttributes {
	if len(in) == 0 {
		return nil
	}

	out := make(RawAttributes, len(in))
	maps.Copy(out, in)

	return out
}

// NormalizeRawAttributes validates and canonicalizes raw attribute values.
func NormalizeRawAttributes(in RawAttributes) (RawAttributes, error) {
	if len(in) == 0 {
		var normalized RawAttributes
		return normalized, nil
	}

	out := make(RawAttributes, len(in))
	for key, raw := range in {
		if !ident.IsField(key) {
			return nil, fmt.Errorf("%w: invalid identifier %q", ragy.ErrInvalidArgument, key)
		}
		value, err := normalizeRawAttributeValue(raw)
		if err != nil {
			return nil, fmt.Errorf("attribute %q: %w", key, err)
		}
		out[key] = value
	}

	return out, nil
}

func normalizeRawAttributeValue(raw any) (any, error) {
	if raw == nil {
		return nil, fmt.Errorf("%w: attribute value must not be nil", ragy.ErrInvalidArgument)
	}

	if number, ok := raw.(json.Number); ok {
		if integer, err := number.Int64(); err == nil {
			return integer, nil
		}
		floatValue, err := number.Float64()
		if err != nil || math.IsNaN(floatValue) || math.IsInf(floatValue, 0) {
			return nil, fmt.Errorf("%w: unsupported attribute value type %T", ragy.ErrInvalidArgument, raw)
		}
		return floatValue, nil
	}

	value := reflect.ValueOf(raw)
	switch value.Kind() {
	case reflect.String:
		return value.String(), nil
	case reflect.Bool:
		return value.Bool(), nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return value.Int(), nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return nil, fmt.Errorf("%w: unsupported attribute value type %T", ragy.ErrInvalidArgument, raw)
	case reflect.Float32, reflect.Float64:
		floatValue := value.Float()
		if math.IsNaN(floatValue) || math.IsInf(floatValue, 0) {
			return nil, fmt.Errorf("%w: unsupported attribute value type %T", ragy.ErrInvalidArgument, raw)
		}
		return floatValue, nil
	case reflect.Invalid,
		reflect.Uintptr,
		reflect.Complex64,
		reflect.Complex128,
		reflect.Array,
		reflect.Chan,
		reflect.Func,
		reflect.Interface,
		reflect.Map,
		reflect.Pointer,
		reflect.Slice,
		reflect.Struct,
		reflect.UnsafePointer:
		return nil, fmt.Errorf("%w: unsupported attribute value type %T", ragy.ErrInvalidArgument, raw)
	}

	return nil, fmt.Errorf("%w: unsupported attribute value type %T", ragy.ErrInvalidArgument, raw)
}
