package retrieval

import (
	"encoding/json"
	"errors"
	"math"
	"strings"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
)

func TestJSONCodecPreservesExactIntegers(t *testing.T) {
	t.Parallel()
	// Arrange.
	builder := filter.NewSchema()
	if _, err := builder.Int("tenant"); err != nil {
		t.Fatal(err)
	}
	schema, err := builder.Build()
	if err != nil {
		t.Fatal(err)
	}
	type meta struct {
		Tenant int64 `json:"tenant"`
	}
	codec := NewJSONCodec[meta](schema)
	for _, tenant := range []int64{9007199254740992, 9007199254740993, math.MinInt64, math.MaxInt64} {
		// Act.
		attrs, encodeErr := codec.Encode(meta{Tenant: tenant})
		out, decodeErr := codec.Decode(attrs)
		// Assert.
		if encodeErr != nil || decodeErr != nil || out.Tenant != tenant {
			t.Fatalf("tenant=%d attrs=%v out=%v encode=%v decode=%v", tenant, attrs, out, encodeErr, decodeErr)
		}
	}
}

func TestJSONCodecRejectsInvalidIntegerNumbers(t *testing.T) {
	t.Parallel()
	// Arrange.
	builder := filter.NewSchema()
	if _, err := builder.Int("tenant"); err != nil {
		t.Fatal(err)
	}
	schema, err := builder.Build()
	if err != nil {
		t.Fatal(err)
	}
	type meta struct {
		Tenant int64 `json:"tenant"`
	}
	codec := NewJSONCodec[meta](schema)
	for _, number := range []string{"1.5", "9223372036854775808", "-9223372036854775809", "1e999", "1." + strings.Repeat("0", 100) + "1"} {
		// Act.
		_, decodeErr := codec.Decode(filter.RawAttributes{"tenant": json.Number(number)})
		// Assert.
		if !errors.Is(decodeErr, ragy.ErrInvalidArgument) {
			t.Fatalf("number=%s error=%v", number, decodeErr)
		}
	}
}
