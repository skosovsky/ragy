package pgvector

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
)

type canonicalCodec struct {
	calls      int
	attributes filter.RawAttributes
}

func (*canonicalCodec) Encode(m integerWireMeta) (filter.RawAttributes, error) {
	return filter.RawAttributes{"tenant": m.Tenant}, nil
}
func (c *canonicalCodec) Decode(attrs filter.RawAttributes) (integerWireMeta, error) {
	c.calls++
	c.attributes = attrs
	if len(attrs) == 0 {
		return integerWireMeta{}, nil
	}
	tenant, ok := attrs["tenant"].(int64)
	if !ok {
		return integerWireMeta{}, fmt.Errorf("noncanonical tenant: %T", attrs["tenant"])
	}
	return integerWireMeta{Tenant: tenant}, nil
}

func TestStoredCustomCodecCanonicalBoundary(t *testing.T) {
	for _, wire := range []string{"", "{}", `{"tenant":9007199254740993}`, `{"tenant":null}`} {
		t.Run(wire, func(t *testing.T) {
			// Arrange: custom codec accepts canonical int64 only, not json.Number.
			codec := &canonicalCodec{}
			store := &Store[integerWireMeta]{schema: integerWireSchema(t), codec: codec}
			// Act.
			meta, err := store.decodeStoredMeta([]byte(wire))
			// Assert: null is rejected before callback; empty wire still calls Decode.
			assertCanonicalDecoded(t, wire, codec, meta, err)
		})
	}
}

func TestTableIdentifierPreservesCaseAndKeywords(t *testing.T) {
	for _, table := range []string{"docs", "MixedCase", "select"} {
		t.Run(table, func(t *testing.T) {
			// Arrange.
			db := &fakeDB{queryRows: &fakeRows{}}
			schema := emptySchema(t)
			store, err := New(
				db,
				Config[integerWireMeta]{Space: fixtureSpace(), Table: table, Schema: schema},
				retrieval.NewJSONCodec[integerWireMeta](schema),
			)
			if err != nil {
				t.Fatal(err)
			}
			vector := make([]float32, fixtureSpace().Dimension)
			vector[0] = 1
			// Act.
			_, err = retrieveStore(
				t.Context(),
				store,
				"query",
				retrieval.RetrieveOptions{Vector: vector, Space: fixtureSpace(), TopK: 1},
			)
			// Assert: single validated name is quoted unchanged; values remain parameters.
			if err != nil || !strings.Contains(db.query, `FROM "`+table+`"`) {
				t.Fatal("table identity", db.query, err)
			}
		})
	}
}

func assertCanonicalDecoded(t *testing.T, wire string, codec *canonicalCodec, meta integerWireMeta, err error) {
	t.Helper()
	if wire == `{"tenant":null}` {
		if !errors.Is(err, ragy.ErrInvalidArgument) || codec.calls != 0 {
			t.Fatal("invalid attributes reached codec", err, codec.calls)
		}
		return
	}
	if err != nil || codec.calls != 1 {
		t.Fatal("canonical decode", meta, err, codec.calls)
	}
	if strings.Contains(wire, "9007199254740993") && meta.Tenant != 9007199254740993 {
		t.Fatal("lost integer identity", meta)
	}
	if len(wire) == 0 && codec.attributes != nil {
		t.Fatal("noncanonical empty attributes", codec.attributes)
	}
}
