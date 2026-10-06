package contracttest_test

import (
	"errors"
	"strings"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/contracttest"
	"github.com/skosovsky/ragy/filter"
)

func TestPortableFilterTruthTableAndComposition(t *testing.T) {
	// Arrange: exhaustive absent/equal/unequal states, independently named truth cases.
	fixture := contracttest.PortableFilterParity(t)
	for _, test := range fixture.Cases {
		t.Run(test.Name, func(t *testing.T) {
			for _, record := range fixture.Records {
				// Act.
				actual, err := filter.MatchCondition(
					test.Condition,
					func(field string) (any, bool) { v, ok := record.Attributes[field]; return v, ok },
				)
				// Assert: fixed two-valued oracle, not another IR walker or SQL renderer.
				expected := portableTruth(t, test.Name, record.Attributes)
				if err != nil || actual != expected {
					t.Fatal(test.Name, record.ID, record.Attributes, actual, expected, err)
				}
			}
		})
	}
}

func portableTruth(t *testing.T, name string, attrs filter.RawAttributes) bool {
	t.Helper()
	s := attrs["s"] == "team's\\雪"
	i := attrs["i"] == int64(9007199254740993)
	b := attrs["b"] == true
	switch name {
	case "empty":
		return true
	case "and":
		return s && i
	case "or":
		return s || i
	case "not_and":
		return !s || !i
	case "not_or":
		return !s && !i
	case "not_or_and":
		return (!s || !i) && !b
	case "or_not_and":
		return !s || !i || b
	case "and_not_or":
		return !s && !i && b
	default:
		return scalarTruth(t, name, attrs)
	}
}

func scalarTruth(t *testing.T, name string, attrs filter.RawAttributes) bool {
	t.Helper()
	field, op, _ := strings.Cut(name, "_")
	not := strings.HasPrefix(op, "not_")
	op = strings.TrimPrefix(op, "not_")
	value, present := attrs[field]
	expected := map[string]any{"s": "team's\\雪", "i": int64(9007199254740993), "f": 1.25, "b": true}[field]
	equal := present && value == expected
	result := false
	switch op {
	case "eq", "in":
		result = equal
	case "neq":
		result = !equal
	case "in_both":
		result = present
	case "gt", "gte", "lt", "lte":
		result = orderedTruth(field, op, value, present)
	default:
		t.Fatal("uncovered truth case", name)
	}
	if not {
		return !result
	}
	return result
}
func orderedTruth(field, op string, value any, present bool) bool {
	if !present {
		return false
	}
	var comparison int
	if field == "i" {
		comparison = scalarComparison(value.(int64), int64(9007199254740993))
	} else {
		comparison = scalarComparison(value.(float64), 1.25)
	}
	switch op {
	case "gt":
		return comparison > 0
	case "gte":
		return comparison >= 0
	case "lt":
		return comparison < 0
	case "lte":
		return comparison <= 0
	default:
		return false
	}
}
func TestPortableMalformedMetadataIsNotAbsence(t *testing.T) {
	// Arrange.
	fixture := contracttest.PortableFilterParity(t)
	// Act/Assert: malformed present values are rejected, whole empty map is lawful.
	for _, raw := range fixture.Invalid {
		if _, err := fixture.Schema.NormalizeAttributes(raw); !errors.Is(err, ragy.ErrInvalidArgument) {
			t.Fatal(raw, err)
		}
	}
	for _, raw := range []filter.RawAttributes{nil, {}} {
		if attrs, err := fixture.Schema.NormalizeAttributes(raw); err != nil || len(attrs) != 0 {
			t.Fatal(attrs, err)
		}
	}
}

func scalarComparison[T interface{ ~int64 | ~float64 }](a, b T) int {
	if a < b {
		return -1
	}
	if a > b {
		return 1
	}
	return 0
}

func TestMalformedStringFilterValueRejectedBeforeExecution(t *testing.T) {
	// Arrange: a declared string field and malformed bytes distinct from valid U+FFFD.
	fixture := contracttest.PortableFilterParity(t)
	field, err := fixture.Schema.StringField("s")
	if err != nil {
		t.Fatal(err)
	}
	for _, value := range []string{string([]byte{0xff}), string([]byte{0xfe})} {
		builder, e := filter.NewBuilder(fixture.Schema)
		if e != nil {
			t.Fatal(e)
		}
		// Act.
		_, e = filter.Eq(builder, field, value).Build()
		// Assert: malformed input is not silently repaired into a portable value.
		if !errors.Is(e, ragy.ErrInvalidArgument) {
			t.Fatal(e)
		}
	}
}
