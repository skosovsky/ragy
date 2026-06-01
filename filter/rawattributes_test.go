package filter

import (
	"encoding/json"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
)

func TestNormalizeRawAttributesRejectsUnsupportedValues(t *testing.T) {
	t.Parallel()

	type sample struct {
		Name string
	}

	cases := []RawAttributes{
		{"tags": []string{"x"}},
		{"nested": map[string]any{"tenant": "acme"}},
		{"object": sample{Name: "bad"}},
		{"value": (*int)(nil)},
		{"value": nil},
		{"quota": uint16(7)},
		{"bad-field": "x"},
	}

	for _, attrs := range cases {
		if _, err := NormalizeRawAttributes(attrs); !errors.Is(err, ragy.ErrInvalidArgument) {
			t.Fatalf("NormalizeRawAttributes(%#v) error = %v, want invalid argument", attrs, err)
		}
	}
}

func TestNormalizeRawAttributesCanonicalizesValues(t *testing.T) {
	t.Parallel()

	normalized, err := NormalizeRawAttributes(RawAttributes{
		"age":     int(42),
		"score":   float32(1.5),
		"integer": json.Number("7"),
		"ratio":   json.Number("2.75"),
	})
	if err != nil {
		t.Fatalf("NormalizeRawAttributes(): %v", err)
	}

	if value, ok := normalized["age"].(int64); !ok || value != 42 {
		t.Fatalf("normalized[age] = %#v, want int64(42)", normalized["age"])
	}
}
