package retrieval

import (
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
)

func TestJSONCodecRejectsMapMeta(t *testing.T) {
	t.Parallel()

	schema, err := filter.NewSchema().Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	codec := NewJSONCodec[map[string]any](schema)
	if _, err := codec.Encode(map[string]any{"tenant": "acme"}); !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Encode(map) error = %v, want invalid argument", err)
	}
}

func TestJSONCodecRejectsUndeclaredAttributesOnDecode(t *testing.T) {
	t.Parallel()

	builder := filter.NewSchema()
	if _, err := builder.String("tenant"); err != nil {
		t.Fatalf("String(tenant): %v", err)
	}
	schema, err := builder.Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	type meta struct {
		Tenant string `json:"tenant"`
	}

	codec := NewJSONCodec[meta](schema)
	_, decodeErr := codec.Decode(filter.RawAttributes{"tenant": "acme", "extra": "x"})
	if !errors.Is(decodeErr, ragy.ErrInvalidArgument) {
		t.Fatalf("Decode(undeclared) error = %v, want invalid argument", decodeErr)
	}
}

func TestJSONCodecRejectsWrongAttributeType(t *testing.T) {
	t.Parallel()

	builder := filter.NewSchema()
	if _, err := builder.String("tenant"); err != nil {
		t.Fatalf("String(tenant): %v", err)
	}
	schema, err := builder.Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	codec := NewJSONCodec[struct{}](schema)
	_, err = codec.Decode(filter.RawAttributes{"tenant": 7})
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Decode(wrong type) error = %v, want invalid argument", err)
	}
}

func TestJSONCodecRoundTrip(t *testing.T) {
	t.Parallel()

	builder := filter.NewSchema()
	if _, err := builder.String("tenant"); err != nil {
		t.Fatalf("String(tenant): %v", err)
	}
	if _, err := builder.Int("age"); err != nil {
		t.Fatalf("Int(age): %v", err)
	}
	schema, err := builder.Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	type meta struct {
		Tenant string `json:"tenant"`
		Age    int64  `json:"age"`
	}

	codec := NewJSONCodec[meta](schema)
	in := meta{Tenant: "acme", Age: 7}
	attrs, err := codec.Encode(in)
	if err != nil {
		t.Fatalf("Encode(): %v", err)
	}
	out, err := codec.Decode(attrs)
	if err != nil {
		t.Fatalf("Decode(): %v", err)
	}
	if out.Tenant != in.Tenant || out.Age != in.Age {
		t.Fatalf("round trip = %#v, want %#v", out, in)
	}
}

type badMarshalMeta struct{}

func (badMarshalMeta) MarshalJSON() ([]byte, error) {
	return nil, errors.New("marshal failed")
}

func TestJSONCodecEncodeWrapsMarshalError(t *testing.T) {
	t.Parallel()

	schema, err := filter.NewSchema().Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	codec := NewJSONCodec[badMarshalMeta](schema)
	if _, err := codec.Encode(badMarshalMeta{}); !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Encode() error = %v, want invalid argument", err)
	}
}

type badEncodeUnmarshalMeta struct{}

func (badEncodeUnmarshalMeta) MarshalJSON() ([]byte, error) {
	return []byte(`"not-an-object"`), nil
}

func TestJSONCodecEncodeWrapsUnmarshalError(t *testing.T) {
	t.Parallel()

	schema, err := filter.NewSchema().Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	codec := NewJSONCodec[badEncodeUnmarshalMeta](schema)
	if _, err := codec.Encode(badEncodeUnmarshalMeta{}); !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Encode() error = %v, want invalid argument", err)
	}
}

type badDecodeMeta struct {
	Tenant string `json:"tenant"`
}

func (badDecodeMeta) UnmarshalJSON([]byte) error {
	return errors.New("unmarshal failed")
}

func TestJSONCodecDecodeWrapsUnmarshalError(t *testing.T) {
	t.Parallel()

	builder := filter.NewSchema()
	if _, err := builder.String("tenant"); err != nil {
		t.Fatalf("String(tenant): %v", err)
	}
	schema, err := builder.Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	codec := NewJSONCodec[badDecodeMeta](schema)
	_, err = codec.Decode(filter.RawAttributes{"tenant": "acme"})
	if !errors.Is(err, ragy.ErrProtocol) {
		t.Fatalf("Decode() error = %v, want protocol", err)
	}
}
