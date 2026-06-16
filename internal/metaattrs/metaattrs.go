// Package metaattrs detects map-like metadata values at API boundaries.
package metaattrs

import (
	"reflect"

	"github.com/skosovsky/ragy/filter"
)

// FromValue reports whether meta is a string-keyed map and returns its attributes.
func FromValue[TMeta any](meta TMeta) (filter.RawAttributes, bool) {
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
