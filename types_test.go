package ragy

import (
	"encoding/json"
	"errors"
	"testing"
)

func TestDocumentValidateAcceptsScalarAttributes(t *testing.T) {
	doc := Document{
		ID:      "doc-1",
		Content: "hello",
		Attributes: Attributes{
			"tenant": "acme",
			"active": true,
			"age":    int(42),
			"score":  float32(1.5),
			"ratio":  json.Number("2.75"),
		},
	}

	if err := doc.Validate(); err != nil {
		t.Fatalf("Validate(): %v", err)
	}
}

func TestDocumentValidateRejectsUnsupportedAttributeValues(t *testing.T) {
	type sample struct {
		Name string
	}

	cases := []Attributes{
		{"tags": []string{"x"}},
		{"nested": map[string]any{"tenant": "acme"}},
		{"object": sample{Name: "bad"}},
		{"value": (*int)(nil)},
		{"value": nil},
		{"quota": uint16(7)},
		{"bad-field": "x"},
	}

	for _, attrs := range cases {
		doc := Document{ID: "doc-1", Attributes: attrs}
		if err := doc.Validate(); !errors.Is(err, ErrInvalidArgument) {
			t.Fatalf("Validate(%#v) error = %v, want invalid argument", attrs, err)
		}
	}
}

func TestNormalizeAttributesCanonicalizesValues(t *testing.T) {
	normalized, err := NormalizeAttributes(Attributes{
		"age":     int(42),
		"score":   float32(1.5),
		"integer": json.Number("7"),
		"ratio":   json.Number("2.75"),
	})
	if err != nil {
		t.Fatalf("NormalizeAttributes(): %v", err)
	}

	if value, ok := normalized["age"].(int64); !ok || value != 42 {
		t.Fatalf("normalized[age] = %#v, want int64(42)", normalized["age"])
	}
	if value, ok := normalized["score"].(float64); !ok || value != 1.5 {
		t.Fatalf("normalized[score] = %#v, want float64(1.5)", normalized["score"])
	}
	if value, ok := normalized["integer"].(int64); !ok || value != 7 {
		t.Fatalf("normalized[integer] = %#v, want int64(7)", normalized["integer"])
	}
	if value, ok := normalized["ratio"].(float64); !ok || value != 2.75 {
		t.Fatalf("normalized[ratio] = %#v, want float64(2.75)", normalized["ratio"])
	}
}

func TestNormalizeAttributesTreatsNilAndEmptyAsNil(t *testing.T) {
	if out, err := NormalizeAttributes(nil); err != nil || out != nil {
		t.Fatalf("NormalizeAttributes(nil) = (%#v, %v), want nil attrs and nil error", out, err)
	}

	if out, err := NormalizeAttributes(Attributes{}); err != nil || out != nil {
		t.Fatalf("NormalizeAttributes(empty) = (%#v, %v), want nil attrs and nil error", out, err)
	}
}

func TestNormalizeDocumentCanonicalizesAttributes(t *testing.T) {
	doc, err := NormalizeDocument(Document{
		ID:      "doc-1",
		Content: "hello",
		Attributes: Attributes{
			"age":   int(42),
			"ratio": json.Number("2.5"),
		},
	})
	if err != nil {
		t.Fatalf("NormalizeDocument(): %v", err)
	}

	if value, ok := doc.Attributes["age"].(int64); !ok || value != 42 {
		t.Fatalf("normalized age = %#v, want int64(42)", doc.Attributes["age"])
	}
	if value, ok := doc.Attributes["ratio"].(float64); !ok || value != 2.5 {
		t.Fatalf("normalized ratio = %#v, want float64(2.5)", doc.Attributes["ratio"])
	}
}

func TestNormalizeChunkRejectsInvalidAttributes(t *testing.T) {
	_, err := NormalizeChunk(Chunk{
		ID:       "chunk-1",
		SourceID: "doc-1",
		Index:    0,
		Content:  "hello",
		Attributes: Attributes{
			"bad-field": "x",
		},
	})
	if !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("NormalizeChunk() error = %v, want invalid argument", err)
	}
}
