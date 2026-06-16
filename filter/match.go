package filter

import (
	"fmt"

	ragy "github.com/skosovsky/ragy"
)

// MatchCondition evaluates a validated filter against field values from lookup.
func MatchCondition(cond Condition, lookup func(field string) (any, bool)) (bool, error) {
	return MatchIR(cond.IR(), lookup)
}

// MatchIR evaluates filter IR against field values from lookup.
func MatchIR(expr IR, lookup func(field string) (any, bool)) (bool, error) {
	matcher := &irMatcher{
		lookup: lookup,
		stack:  nil,
		result: false,
	}
	if err := Walk(expr, matcher); err != nil {
		return false, err
	}

	return matcher.result, nil
}

type matchFrame struct {
	op     string
	values []bool
}

type irMatcher struct {
	lookup func(string) (any, bool)
	stack  []matchFrame
	result bool
}

func (m *irMatcher) OnEmpty() error {
	return m.push(true)
}

func (m *irMatcher) OnEq(field string, value Value) error {
	matched, err := compareEqual(m.lookup, field, value)
	if err != nil {
		return err
	}

	return m.push(matched)
}

func (m *irMatcher) OnNeq(field string, value Value) error {
	matched, err := compareEqual(m.lookup, field, value)
	if err != nil {
		return err
	}

	return m.push(!matched)
}

func (m *irMatcher) OnGt(field string, value Value) error {
	return m.pushOrdered(field, value, func(cmp int) bool { return cmp > 0 })
}

func (m *irMatcher) OnGte(field string, value Value) error {
	return m.pushOrdered(field, value, func(cmp int) bool { return cmp >= 0 })
}

func (m *irMatcher) OnLt(field string, value Value) error {
	return m.pushOrdered(field, value, func(cmp int) bool { return cmp < 0 })
}

func (m *irMatcher) OnLte(field string, value Value) error {
	return m.pushOrdered(field, value, func(cmp int) bool { return cmp <= 0 })
}

func (m *irMatcher) OnIn(field string, values []Value) error {
	for _, value := range values {
		matched, err := compareEqual(m.lookup, field, value)
		if err != nil {
			return err
		}
		if matched {
			return m.push(true)
		}
	}

	return m.push(false)
}

func (m *irMatcher) EnterAnd(_ int) error {
	m.stack = append(m.stack, matchFrame{op: "and", values: nil})
	return nil
}

func (m *irMatcher) LeaveAnd() error {
	frame, err := m.popFrame("and")
	if err != nil {
		return err
	}

	result := true
	for _, value := range frame.values {
		if !value {
			result = false
			break
		}
	}

	return m.push(result)
}

func (m *irMatcher) EnterOr(_ int) error {
	m.stack = append(m.stack, matchFrame{op: "or", values: nil})
	return nil
}

func (m *irMatcher) LeaveOr() error {
	frame, err := m.popFrame("or")
	if err != nil {
		return err
	}

	result := false
	for _, value := range frame.values {
		if value {
			result = true
			break
		}
	}

	return m.push(result)
}

func (m *irMatcher) EnterNot() error {
	m.stack = append(m.stack, matchFrame{op: "not", values: nil})
	return nil
}

func (m *irMatcher) LeaveNot() error {
	frame, err := m.popFrame("not")
	if err != nil {
		return err
	}
	if len(frame.values) != 1 {
		return fmt.Errorf("%w: NOT matcher arity", ragy.ErrUnsupported)
	}

	return m.push(!frame.values[0])
}

func (m *irMatcher) pushOrdered(field string, value Value, check func(int) bool) error {
	matched, err := compareOrdered(m.lookup, field, value, check)
	if err != nil {
		return err
	}

	return m.push(matched)
}

func (m *irMatcher) push(value bool) error {
	if len(m.stack) == 0 {
		m.result = value
		return nil
	}

	last := len(m.stack) - 1
	m.stack[last].values = append(m.stack[last].values, value)
	return nil
}

func (m *irMatcher) popFrame(op string) (matchFrame, error) {
	if len(m.stack) == 0 {
		return matchFrame{}, fmt.Errorf("%w: unmatched filter group", ragy.ErrUnsupported)
	}

	last := len(m.stack) - 1
	frame := m.stack[last]
	m.stack = m.stack[:last]
	if frame.op != op {
		return matchFrame{}, fmt.Errorf("%w: unexpected filter group %q", ragy.ErrUnsupported, frame.op)
	}

	return frame, nil
}

func compareEqual(lookup func(string) (any, bool), field string, expected Value) (bool, error) {
	actual, ok := lookup(field)
	if !ok {
		return false, nil
	}

	switch expected.Kind() {
	case KindString:
		value, ok := actual.(string)
		expectedValue, expectedOK := expected.Raw().(string)
		if !expectedOK {
			return false, fmt.Errorf("%w: invalid string filter value", ragy.ErrUnsupported)
		}
		return ok && value == expectedValue, nil
	case KindBool:
		value, ok := actual.(bool)
		expectedValue, expectedOK := expected.Raw().(bool)
		if !expectedOK {
			return false, fmt.Errorf("%w: invalid bool filter value", ragy.ErrUnsupported)
		}
		return ok && value == expectedValue, nil
	case KindInt:
		value, ok := toInt64(actual)
		expectedValue, expectedOK := expected.Raw().(int64)
		if !expectedOK {
			return false, fmt.Errorf("%w: invalid int filter value", ragy.ErrUnsupported)
		}
		return ok && value == expectedValue, nil
	case KindFloat:
		value, ok := toFloat64(actual)
		expectedValue, expectedOK := expected.Raw().(float64)
		if !expectedOK {
			return false, fmt.Errorf("%w: invalid float filter value", ragy.ErrUnsupported)
		}
		return ok && value == expectedValue, nil
	default:
		return false, fmt.Errorf("%w: unsupported filter kind %q", ragy.ErrUnsupported, expected.Kind())
	}
}

func compareOrdered(
	lookup func(string) (any, bool),
	field string,
	expected Value,
	check func(int) bool,
) (bool, error) {
	actual, ok := lookup(field)
	if !ok {
		return false, nil
	}

	switch expected.Kind() {
	case KindString:
		return false, fmt.Errorf("%w: unsupported ordered filter kind %q", ragy.ErrUnsupported, expected.Kind())
	case KindInt:
		value, ok := toInt64(actual)
		if !ok {
			return false, nil
		}
		expectedValue, expectedOK := expected.Raw().(int64)
		if !expectedOK {
			return false, fmt.Errorf("%w: invalid int filter value", ragy.ErrUnsupported)
		}
		return check(compareInts(value, expectedValue)), nil
	case KindFloat:
		value, ok := toFloat64(actual)
		if !ok {
			return false, nil
		}
		expectedValue, expectedOK := expected.Raw().(float64)
		if !expectedOK {
			return false, fmt.Errorf("%w: invalid float filter value", ragy.ErrUnsupported)
		}
		return check(compareFloats(value, expectedValue)), nil
	case KindBool:
		return false, fmt.Errorf("%w: unsupported ordered filter kind %q", ragy.ErrUnsupported, expected.Kind())
	default:
		return false, fmt.Errorf("%w: unsupported ordered filter kind %q", ragy.ErrUnsupported, expected.Kind())
	}
}

func toInt64(value any) (int64, bool) {
	switch v := value.(type) {
	case int:
		return int64(v), true
	case int8:
		return int64(v), true
	case int16:
		return int64(v), true
	case int32:
		return int64(v), true
	case int64:
		return v, true
	default:
		return 0, false
	}
}

func toFloat64(value any) (float64, bool) {
	switch v := value.(type) {
	case int:
		return float64(v), true
	case int8:
		return float64(v), true
	case int16:
		return float64(v), true
	case int32:
		return float64(v), true
	case int64:
		return float64(v), true
	case float32:
		return float64(v), true
	case float64:
		return v, true
	default:
		return 0, false
	}
}

func compareInts(left, right int64) int {
	switch {
	case left < right:
		return -1
	case left > right:
		return 1
	default:
		return 0
	}
}

func compareFloats(left, right float64) int {
	switch {
	case left < right:
		return -1
	case left > right:
		return 1
	default:
		return 0
	}
}
