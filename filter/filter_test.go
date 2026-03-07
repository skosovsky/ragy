package filter

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEqual(t *testing.T) {
	e := Equal("status", "active")
	eq, ok := e.(Eq)
	require.True(t, ok)
	assert.Equal(t, "status", eq.Field)
	assert.Equal(t, "active", eq.Value)
}

func TestNotEqual(t *testing.T) {
	e := NotEqual("status", "archived")
	neq, ok := e.(Neq)
	require.True(t, ok)
	assert.Equal(t, "status", neq.Field)
	assert.Equal(t, "archived", neq.Value)
}

func TestInverse(t *testing.T) {
	inner := Equal("x", 1)
	e := Inverse(inner)
	not, ok := e.(Not)
	require.True(t, ok)
	assert.Equal(t, inner, not.Expr)
	// Double negation
	e2 := Inverse(e)
	not2, ok := e2.(Not)
	require.True(t, ok)
	innerNot, ok := not2.Expr.(Not)
	require.True(t, ok)
	assert.Equal(t, inner, innerNot.Expr)
}

func TestAll(t *testing.T) {
	e1 := Equal("tenant_id", "company_1")
	e2 := Greater("created_at", 0)
	e := All(e1, e2)
	and, ok := e.(And)
	require.True(t, ok)
	require.Len(t, and.Exprs, 2)
	assert.Equal(t, e1, and.Exprs[0])
	assert.Equal(t, e2, and.Exprs[1])
}

func TestAny(t *testing.T) {
	e1 := Equal("status", "active")
	e2 := Equal("status", "pending")
	e := Any(e1, e2)
	or, ok := e.(Or)
	require.True(t, ok)
	require.Len(t, or.Exprs, 2)
	assert.Equal(t, e1, or.Exprs[0])
	assert.Equal(t, e2, or.Exprs[1])
}

func TestOneOf(t *testing.T) {
	e := OneOf("tag", "a", "b", "c")
	in, ok := e.(In)
	require.True(t, ok)
	assert.Equal(t, "tag", in.Field)
	assert.Equal(t, []any{"a", "b", "c"}, in.Values)
}

func TestGreater_Less_Gte_Lte(t *testing.T) {
	g := Greater("age", 18)
	gt, ok := g.(Gt)
	require.True(t, ok)
	assert.Equal(t, "age", gt.Field)
	assert.Equal(t, 18, gt.Value)

	l := Less("score", 100.0)
	lt, ok := l.(Lt)
	require.True(t, ok)
	assert.Equal(t, "score", lt.Field)

	gte := GreaterOrEqual("x", 0)
	gteVal, ok := gte.(Gte)
	require.True(t, ok)
	assert.Equal(t, "x", gteVal.Field)

	lte := LessOrEqual("y", 10)
	lteVal, ok := lte.(Lte)
	require.True(t, ok)
	assert.Equal(t, "y", lteVal.Field)
}

func TestEmptyAnd(t *testing.T) {
	e := All()
	and, ok := e.(And)
	require.True(t, ok)
	assert.NotNil(t, and.Exprs)
	assert.Empty(t, and.Exprs)
}

func TestEmptyOr(t *testing.T) {
	e := Any()
	or, ok := e.(Or)
	require.True(t, ok)
	assert.NotNil(t, or.Exprs)
	assert.Empty(t, or.Exprs)
}
