package filter

import "errors"

// Builder constructs validated filter conditions without exposing IR construction to callers.
type Builder struct {
	schema Schema
	expr   expr
}

// NewBuilder creates a filter builder bound to a finalized schema.
func NewBuilder(schema Schema) (*Builder, error) {
	if !schema.IsFinalized() {
		return nil, errors.New("filter builder requires finalized schema")
	}
	return &Builder{schema: schema}, nil
}

func (b *Builder) with(e expr) *Builder {
	if b == nil {
		return nil
	}
	if e == nil {
		return b
	}
	if b.expr == nil {
		b.expr = e
		return b
	}
	b.expr = all(b.expr, e)
	return b
}

// Build validates the expression against the schema and returns a Condition.
func (b *Builder) Build() (Condition, error) {
	if b == nil {
		return Condition{}, errors.New("filter builder is nil")
	}
	if b.expr == nil {
		return emptyBuiltCondition(), nil
	}
	ir, err := normalize(b.expr)
	if err != nil {
		return Condition{}, err
	}
	if err := b.schema.ValidateSchemaIR(ir); err != nil {
		return Condition{}, err
	}
	return conditionFromIR(ir)
}

// Eq adds an equality predicate for a declared field.
func Eq[T scalar](b *Builder, field Field[T], value T) *Builder {
	return b.with(equal(field, value))
}

// NotEq adds a not-equal predicate for a declared field.
func NotEq[T scalar](b *Builder, field Field[T], value T) *Builder {
	return b.with(notEqual(field, value))
}

// In adds a membership predicate for a declared field.
func In[T scalar](b *Builder, field Field[T], values ...T) *Builder {
	return b.with(oneOf(field, values...))
}

// Gt adds a greater-than predicate for an ordered field.
func Gt[T orderedScalar](b *Builder, field Field[T], value T) *Builder {
	return b.with(greater(field, value))
}

// Gte adds a greater-or-equal predicate for an ordered field.
func Gte[T orderedScalar](b *Builder, field Field[T], value T) *Builder {
	return b.with(greaterOrEqual(field, value))
}

// Lt adds a less-than predicate for an ordered field.
func Lt[T orderedScalar](b *Builder, field Field[T], value T) *Builder {
	return b.with(less(field, value))
}

// Lte adds a less-or-equal predicate for an ordered field.
func Lte[T orderedScalar](b *Builder, field Field[T], value T) *Builder {
	return b.with(lessOrEqual(field, value))
}

// Not negates the current builder expression.
func Not(b *Builder) *Builder {
	if b == nil || b.expr == nil {
		return b
	}
	b.expr = inverse(b.expr)
	return b
}

// Or combines two builder expressions with logical OR.
func Or(b *Builder, other *Builder) *Builder {
	if b == nil || other == nil {
		return b
	}
	switch {
	case b.expr == nil && other.expr == nil:
		return b
	case b.expr == nil:
		b.expr = other.expr
	case other.expr == nil:
		return b
	default:
		b.expr = anyOf(b.expr, other.expr)
	}
	return b
}
