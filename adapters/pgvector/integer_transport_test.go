package pgvector

import (
	"encoding/json"
	"errors"
	"math"
	"strconv"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
)

type integerWireMeta struct {
	Tenant int64 `json:"tenant"`
}

func integerWireSchema(t *testing.T) filter.Schema {
	t.Helper()
	fields := filter.NewSchema()
	if _, err := fields.Int("tenant"); err != nil {
		t.Fatal(err)
	}
	schema, err := fields.Build()
	if err != nil {
		t.Fatal(err)
	}
	return schema
}
func integerWireValues() []int64 {
	return []int64{9007199254740992, 9007199254740993, math.MinInt64, math.MaxInt64}
}
func invalidWireIntegers() []string {
	return []string{"1.5", "9223372036854775808", "-9223372036854775809", "1e999"}
}

func integerWireStore(t *testing.T, number string) (*Store[integerWireMeta], *fakeDB) {
	t.Helper()
	db := &fakeDB{
		queryRows: &fakeRows{
			rows: []fakeRow{
				{id: "doc", content: "policy", relevance: 1, attrsJSON: []byte(`{"tenant":` + number + `}`)},
			},
		},
	}
	schema := integerWireSchema(t)
	store, err := New[integerWireMeta](
		db,
		Config[integerWireMeta]{Table: "docs", Schema: schema},
		retrieval.NewJSONCodec[integerWireMeta](schema),
	)
	if err != nil {
		t.Fatal(err)
	}
	return store, db
}
func TestIntegerWireProjectionPreservesBoundsAndAdjacentIDs(t *testing.T) {
	for _, value := range integerWireValues() {
		t.Run(strconv.FormatInt(value, 10), func(t *testing.T) {
			// Arrange: DB Rows delivers the stored JSON bytes to actual scan/decode.
			store, _ := integerWireStore(t, strconv.FormatInt(value, 10))
			// Act.
			result, err := retrieveStore(
				t.Context(),
				store,
				"policy",
				retrieval.RetrieveOptions{TopK: 1, Vector: []float32{1, 0}},
			)
			// Assert.
			if err != nil || result.Len() != 1 || result.Documents()[0].Meta.Tenant != value {
				t.Fatal("integer SQL row corrupted", value, err)
			}
		})
	}
}
func TestIntegerWireProjectionRejectsInvalidIntegers(t *testing.T) {
	for _, number := range invalidWireIntegers() {
		t.Run(number, func(t *testing.T) {
			// Arrange.
			store, _ := integerWireStore(t, number)
			// Act.
			result, err := retrieveStore(
				t.Context(),
				store,
				"policy",
				retrieval.RetrieveOptions{TopK: 1, Vector: []float32{1, 0}},
			)
			// Assert.
			if !errors.Is(err, ragy.ErrInvalidArgument) || result.Len() != 0 {
				t.Fatal("invalid integer SQL row projected", err)
			}
		})
	}
}

func TestIntegerWireUpsertPreservesExactStoredNumbers(t *testing.T) {
	for _, value := range integerWireValues() {
		t.Run(strconv.FormatInt(value, 10), func(t *testing.T) {
			// Arrange.
			store, port := integerWireStore(t, strconv.FormatInt(value, 10))
			// Act.
			err := store.Upsert(
				t.Context(),
				[]dense.Record[integerWireMeta]{
					{ID: "doc", Content: "policy", Meta: integerWireMeta{Tenant: value}, Vector: []float32{1, 0}},
				},
			)
			// Assert.
			if err != nil {
				t.Fatal(err)
			}
			if len(port.execArgs) != 4 {
				t.Fatal("row not written")
			}
			data, ok := port.execArgs[2].([]byte)
			if !ok {
				t.Fatal("wrong stored JSON boundary")
			}
			if string(data) != `{"tenant":`+strconv.FormatInt(value, 10)+`}` {
				t.Fatal("stored integer JSON changed", string(data))
			}
			decoded, decodeErr := store.decodeStoredMeta(data)
			if decodeErr != nil || decoded.Tenant != value {
				t.Fatal("upsert number changed", decoded, decodeErr)
			}
		})
	}
}

type malformedIntegerWireCodec struct{ value any }

func (c malformedIntegerWireCodec) Encode(meta integerWireMeta) (filter.RawAttributes, error) {
	if meta.Tenant != 9007199254740993 {
		return filter.RawAttributes{"tenant": meta.Tenant}, nil
	}
	return filter.RawAttributes{"tenant": c.value}, nil
}
func (malformedIntegerWireCodec) Decode(filter.RawAttributes) (integerWireMeta, error) {
	return integerWireMeta{}, ragy.ErrProtocol
}
func TestIntegerWireUpsertRejectsMalformedHostCodecBeforeIO(t *testing.T) {
	values := []any{
		json.Number("1.5"),
		json.Number("9223372036854775808"),
		json.Number("-9223372036854775809"),
		json.Number("1e999"),
		math.NaN(),
		math.Inf(1),
		math.Inf(-1),
	}
	for i, value := range values {
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			// Arrange: host codec violates the declared integer schema; caller metadata is valid.
			store, port := integerWireStore(t, "9007199254740993")
			store.codec = malformedIntegerWireCodec{value: value}
			// Act.
			err := store.Upsert(
				t.Context(),
				[]dense.Record[integerWireMeta]{
					{
						ID:      "doc",
						Content: "policy",
						Meta:    integerWireMeta{Tenant: 9007199254740993},
						Vector:  []float32{1, 0},
					},
				},
			)
			// Assert.
			if !errors.Is(err, ragy.ErrInvalidArgument) || port.execCalls != 0 {
				t.Fatal("malformed integer reached write port", err, port.execCalls)
			}
		})
	}
}

func integerMembership(t *testing.T, schema filter.Schema) filter.Condition {
	t.Helper()
	field, err := schema.IntField("tenant")
	if err != nil {
		t.Fatal(err)
	}
	builder, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatal(err)
	}
	condition, err := filter.In(builder, field, integerWireValues()...).Build()
	if err != nil {
		t.Fatal(err)
	}
	return condition
}
func TestIntegerWireMembershipRetainsExactValues(t *testing.T) {
	// Arrange: membership includes adjacent values and both signed integer bounds.
	store, port := integerWireStore(t, "9007199254740993")
	options := retrieval.RetrieveOptions{TopK: 1, Filters: integerMembership(t, store.Schema())}
	options.Vector = []float32{1, 0}
	// Act.
	_, err := retrieveStore(t.Context(), store, "policy", options)
	// Assert: transport values remain distinct integers; no float64 intermediary.
	if err != nil {
		t.Fatal(err)
	}
	if len(port.args) != 6 {
		t.Fatal("membership SQL arguments changed", port.args)
	}
	for i, expected := range integerWireValues() {
		value, typed := port.args[i+1].(int64)
		if !typed || value != expected {
			t.Fatal("membership number changed", port.args[i+1])
		}
	}
}
