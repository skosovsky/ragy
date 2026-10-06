package contracttest

import (
	"fmt"
	"math"
	"testing"

	"github.com/skosovsky/ragy/filter"
)

const (
	parityIntegerEqual      int64 = 9007199254740993
	parityIntegerUnequal    int64 = 9007199254740992
	parityFloatEqual              = 1.25
	parityFloatUnequal            = 2.5
	parityFractionalInteger       = 1.5
)

// FilterParityRecord is a validated boundary attribute fixture, not a domain type.
type FilterParityRecord struct {
	ID         string
	Attributes filter.RawAttributes
}

// FilterParityCase holds a portable validated predicate for any backend profile.
type FilterParityCase struct {
	Name      string
	Condition filter.Condition
}

// FilterParityFixture separates admitted corpus from malformed metadata cases.
type FilterParityFixture struct {
	Schema  filter.Schema
	Records []FilterParityRecord
	Cases   []FilterParityCase
	Invalid []filter.RawAttributes
}

// PortableFilterParity includes all four scalar domains, omissions, exact int64
// neighbors above 2^53, Unicode/SQL-shaped strings and nested boolean composition.
func PortableFilterParity(t *testing.T) FilterParityFixture {
	t.Helper()
	fields := filter.NewSchema()
	str, err := fields.String("s")
	if err != nil {
		t.Fatal(err)
	}
	integer, err := fields.Int("i")
	if err != nil {
		t.Fatal(err)
	}
	number, err := fields.Float("f")
	if err != nil {
		t.Fatal(err)
	}
	boolean, err := fields.Bool("b")
	if err != nil {
		t.Fatal(err)
	}
	schema, err := fields.Build()
	if err != nil {
		t.Fatal(err)
	}
	fixture := FilterParityFixture{Schema: schema}
	addScalarFilterCases(t, &fixture, str, "team's\\雪", "x'); DROP TABLE corpus; --")
	addScalarFilterCases(t, &fixture, integer, parityIntegerEqual, parityIntegerUnequal)
	addScalarFilterCases(t, &fixture, number, parityFloatEqual, parityFloatUnequal)
	addScalarFilterCases(t, &fixture, boolean, true, false)
	addOrderedFilterCases(t, &fixture, integer, parityIntegerEqual)
	addOrderedFilterCases(t, &fixture, number, parityFloatEqual)
	fresh := func() *filter.Builder {
		b, e := filter.NewBuilder(schema)
		if e != nil {
			t.Fatal(e)
		}
		return b
	}
	left := func() *filter.Builder { return filter.Eq(fresh(), str, "team's\\雪") }
	right := func() *filter.Builder { return filter.Eq(fresh(), integer, parityIntegerEqual) }
	third := func() *filter.Builder { return filter.Eq(fresh(), boolean, true) }
	both := func() *filter.Builder { return filter.Eq(left(), integer, parityIntegerEqual) }
	addFilterCase(t, &fixture, "empty", fresh())
	addFilterCase(t, &fixture, "and", both())
	addFilterCase(t, &fixture, "or", filter.Or(left(), right()))
	addFilterCase(t, &fixture, "not_and", filter.Not(both()))
	addFilterCase(t, &fixture, "not_or", filter.Not(filter.Or(left(), right())))
	addFilterCase(t, &fixture, "not_or_and", filter.Not(filter.Or(both(), third())))
	addFilterCase(t, &fixture, "or_not_and", filter.Or(filter.Not(both()), third()))
	addFilterCase(t, &fixture, "and_not_or", filter.Eq(filter.Not(filter.Or(left(), right())), boolean, true))
	fixture.Records = filterParityRecords(t, schema)
	fixture.Invalid = []filter.RawAttributes{
		{"s": nil},
		{"i": nil},
		{"f": nil},
		{"b": nil},
		{
			"s": string([]byte{0xff}),
		},
		{"s": true},
		{"i": "oops"},
		{"i": parityFractionalInteger},
		{"f": "oops"},
		{"b": "true"},
		{"f": math.NaN()},
		{"f": math.Inf(1)},
	}
	return fixture
}

func addFilterCase(t *testing.T, fixture *FilterParityFixture, name string, builder *filter.Builder) {
	t.Helper()
	condition, err := builder.Build()
	if err != nil {
		t.Fatal(err)
	}
	fixture.Cases = append(fixture.Cases, FilterParityCase{Name: name, Condition: condition})
}

func addScalarFilterCases[T interface {
	~string | ~int64 | ~float64 | ~bool
}](t *testing.T, fixture *FilterParityFixture, field filter.Field[T], equal, unequal T) {
	t.Helper()
	fresh := func() *filter.Builder {
		b, e := filter.NewBuilder(fixture.Schema)
		if e != nil {
			t.Fatal(e)
		}
		return b
	}
	predicates := []struct {
		name string
		make func() *filter.Builder
	}{
		{"eq", func() *filter.Builder { return filter.Eq(fresh(), field, equal) }},
		{"neq", func() *filter.Builder { return filter.NotEq(fresh(), field, equal) }},
		{"in", func() *filter.Builder { return filter.In(fresh(), field, equal) }},
		{"in_both", func() *filter.Builder { return filter.In(fresh(), field, equal, unequal) }},
	}
	for _, predicate := range predicates {
		addFilterCase(t, fixture, fmt.Sprintf("%s_%s", field.Name(), predicate.name), predicate.make())
		addFilterCase(t, fixture, fmt.Sprintf("%s_not_%s", field.Name(), predicate.name), filter.Not(predicate.make()))
	}
}

func addOrderedFilterCases[T interface{ ~int64 | ~float64 }](
	t *testing.T,
	fixture *FilterParityFixture,
	field filter.Field[T],
	value T,
) {
	t.Helper()
	fresh := func() *filter.Builder {
		b, e := filter.NewBuilder(fixture.Schema)
		if e != nil {
			t.Fatal(e)
		}
		return b
	}
	predicates := []struct {
		name string
		make func() *filter.Builder
	}{
		{"gt", func() *filter.Builder { return filter.Gt(fresh(), field, value) }},
		{"gte", func() *filter.Builder { return filter.Gte(fresh(), field, value) }},
		{"lt", func() *filter.Builder { return filter.Lt(fresh(), field, value) }},
		{"lte", func() *filter.Builder { return filter.Lte(fresh(), field, value) }},
	}
	for _, predicate := range predicates {
		addFilterCase(t, fixture, fmt.Sprintf("%s_%s", field.Name(), predicate.name), predicate.make())
		addFilterCase(t, fixture, fmt.Sprintf("%s_not_%s", field.Name(), predicate.name), filter.Not(predicate.make()))
	}
}

func filterParityRecords(t *testing.T, schema filter.Schema) []FilterParityRecord {
	t.Helper()
	choices := []filter.RawAttributes{{}, {"s": "team's\\雪", "i": parityIntegerEqual, "f": parityFloatEqual, "b": true},
		{"s": "x'); DROP TABLE corpus; --", "i": parityIntegerUnequal, "f": parityFloatUnequal, "b": false}}
	var records []FilterParityRecord
	const combinations = 81 // Four independently absent/equal/unequal scalar fields.
	for ordinal := range combinations {
		raw := filter.RawAttributes{}
		choiceIndex := ordinal
		for _, key := range []string{"s", "i", "f", "b"} {
			if value, present := choices[choiceIndex%len(choices)][key]; present {
				raw[key] = value
			}
			choiceIndex /= len(choices)
		}
		attrs, err := schema.NormalizeAttributes(raw)
		if err != nil {
			t.Fatal(err)
		}
		records = append(records, FilterParityRecord{ID: fmt.Sprintf("r%02d", ordinal), Attributes: attrs})
	}
	return records
}
