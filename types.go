package ragy

import (
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"reflect"

	"github.com/skosovsky/ragy/internal/ident"
)

// Attributes stores user-defined scalar metadata.
type Attributes map[string]any

// CloneAttributes returns a shallow copy of the attributes map.
func CloneAttributes(in Attributes) Attributes {
	if len(in) == 0 {
		return nil
	}

	out := make(Attributes, len(in))
	maps.Copy(out, in)

	return out
}

// NormalizeAttributes validates and canonicalizes attribute values.
func NormalizeAttributes(in Attributes) (Attributes, error) {
	if len(in) == 0 {
		var normalized Attributes
		return normalized, nil
	}

	out := make(Attributes, len(in))
	for key, raw := range in {
		if !ident.IsField(key) {
			return nil, fmt.Errorf("%w: invalid identifier %q", ErrInvalidArgument, key)
		}
		value, err := normalizeAttributeValue(raw)
		if err != nil {
			return nil, fmt.Errorf("attribute %q: %w", key, err)
		}
		out[key] = value
	}

	return out, nil
}

func normalizeAttributeValue(raw any) (any, error) {
	if raw == nil {
		return nil, fmt.Errorf("%w: attribute value must not be nil", ErrInvalidArgument)
	}

	if number, ok := raw.(json.Number); ok {
		if integer, err := number.Int64(); err == nil {
			return integer, nil
		}
		floatValue, err := number.Float64()
		if err != nil || math.IsNaN(floatValue) || math.IsInf(floatValue, 0) {
			return nil, fmt.Errorf("%w: unsupported attribute value type %T", ErrInvalidArgument, raw)
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
		return nil, fmt.Errorf("%w: unsupported attribute value type %T", ErrInvalidArgument, raw)
	case reflect.Float32, reflect.Float64:
		floatValue := value.Float()
		if math.IsNaN(floatValue) || math.IsInf(floatValue, 0) {
			return nil, fmt.Errorf("%w: unsupported attribute value type %T", ErrInvalidArgument, raw)
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
		return nil, fmt.Errorf("%w: unsupported attribute value type %T", ErrInvalidArgument, raw)
	}

	return nil, fmt.Errorf("%w: unsupported attribute value type %T", ErrInvalidArgument, raw)
}

// ClampRelevance bounds a public relevance score to [0, 1].
func ClampRelevance(v float64) float64 {
	switch {
	case v < 0:
		return 0
	case v > 1:
		return 1
	default:
		return v
	}
}

// Document is the canonical public retrieval result.
type Document struct {
	ID         string
	Content    string
	Attributes Attributes
	Relevance  float64
}

// NormalizeDocument validates and canonicalizes a public document payload.
func NormalizeDocument(d Document) (Document, error) {
	if d.ID == "" {
		return Document{}, fmt.Errorf("%w: document id", ErrMissingID)
	}

	attrs, err := NormalizeAttributes(d.Attributes)
	if err != nil {
		return Document{}, err
	}
	d.Attributes = attrs

	if math.IsNaN(d.Relevance) || math.IsInf(d.Relevance, 0) || d.Relevance < 0 || d.Relevance > 1 {
		return Document{}, fmt.Errorf("%w: document relevance must be in [0,1]", ErrInvalidArgument)
	}

	return d, nil
}

// Validate checks document invariants.
func (d Document) Validate() error {
	_, err := NormalizeDocument(d)
	return err
}

// Chunk is the canonical chunking output.
type Chunk struct {
	ID         string
	SourceID   string
	Index      int
	Content    string
	Context    string
	Attributes Attributes
}

// NormalizeChunk validates and canonicalizes a public chunk payload.
func NormalizeChunk(c Chunk) (Chunk, error) {
	if c.ID == "" {
		return Chunk{}, fmt.Errorf("%w: chunk id", ErrMissingID)
	}

	if c.SourceID == "" {
		return Chunk{}, fmt.Errorf("%w: chunk source id", ErrMissingSourceID)
	}

	if c.Index < 0 {
		return Chunk{}, fmt.Errorf("%w: chunk index must be >= 0", ErrInvalidArgument)
	}

	attrs, err := NormalizeAttributes(c.Attributes)
	if err != nil {
		return Chunk{}, err
	}
	c.Attributes = attrs

	return c, nil
}

// Validate checks chunk invariants.
func (c Chunk) Validate() error {
	_, err := NormalizeChunk(c)
	return err
}

// Page is an explicit pagination contract.
type Page struct {
	Limit  int
	Offset int
}

// NewPage validates and constructs a page.
func NewPage(limit, offset int) (*Page, error) {
	p := &Page{Limit: limit, Offset: offset}
	if err := p.Validate(); err != nil {
		return nil, err
	}

	return p, nil
}

// Validate checks page invariants.
func (p *Page) Validate() error {
	if p == nil {
		return nil
	}

	if p.Limit <= 0 {
		return fmt.Errorf("%w: limit must be > 0", ErrInvalidPage)
	}

	if p.Offset < 0 {
		return fmt.Errorf("%w: offset must be >= 0", ErrInvalidPage)
	}

	return nil
}
