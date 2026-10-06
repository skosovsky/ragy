package elasticsearch

import (
	"encoding/json"
	"errors"
	"math"
	"strconv"
	"strings"
	"testing"

	ragy "github.com/skosovsky/ragy"
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

func integerWireStore(t *testing.T, number string) (*Store[integerWireMeta], *fakeClient) {
	t.Helper()
	var source filter.RawAttributes
	if err := json.Unmarshal([]byte(`{"content":"policy","tenant":`+number+`}`), &source); err != nil {
		t.Fatal(err)
	}
	client := &fakeClient{hits: []Hit{{ID: "doc", Score: 1, Source: source}}}
	schema := integerWireSchema(t)
	store, err := New[integerWireMeta](
		client,
		Config[integerWireMeta]{Index: "docs", SearchFields: []string{"content"}, Schema: schema},
		retrieval.NewJSONCodec[integerWireMeta](schema),
	)
	if err != nil {
		t.Fatal(err)
	}
	return store, client
}
func TestIntegerWireProjectionPreservesBoundsAndAdjacentIDs(t *testing.T) {
	for _, value := range integerWireValues() {
		t.Run(strconv.FormatInt(value, 10), func(t *testing.T) {
			// Arrange: JSON decodes directly into the public Hit.Source boundary.
			store, _ := integerWireStore(t, strconv.FormatInt(value, 10))
			// Act.
			result, err := retrieveStore(t.Context(), store, "policy", retrieval.RetrieveOptions{TopK: 1})
			// Assert.
			if err != nil || result.Len() != 1 || result.Documents()[0].Meta.Tenant != value {
				t.Fatal("integer hit corrupted", value, err)
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
			result, err := retrieveStore(t.Context(), store, "policy", retrieval.RetrieveOptions{TopK: 1})
			// Assert.
			if !errors.Is(err, ragy.ErrInvalidArgument) || result.Len() != 0 {
				t.Fatal("invalid integer hit projected", err)
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

	// Act.
	_, err := retrieveStore(t.Context(), store, "policy", options)
	// Assert: transport values remain distinct integers; no float64 intermediary.
	if err != nil {
		t.Fatal(err)
	}
	data, marshalErr := json.Marshal(port.body)
	if marshalErr != nil {
		t.Fatal(marshalErr)
	}
	if !strings.Contains(
		string(data),
		`"tenant":[9007199254740992,9007199254740993,-9223372036854775808,9223372036854775807]`,
	) {
		t.Fatal("membership numbers changed", string(data))
	}
}
