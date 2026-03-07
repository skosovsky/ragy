// Package filter provides an AST-based type-safe filter expression builder
// for SearchRequest and VectorStore/GraphStore adapters.
// Adapters traverse the tree via type switch to build native queries (SQL WHERE, Qdrant filter JSON, etc.).
package filter

// Expr is the marker interface for all filter expression nodes.
// Implementations must implement isExpr() to satisfy the interface.
type Expr interface {
	isExpr()
}

// Eq represents field == value.
type Eq struct {
	Field string
	Value any
}

func (Eq) isExpr() {}

// Neq represents field != value (Not Equal).
type Neq struct {
	Field string
	Value any
}

func (Neq) isExpr() {}

// Gt represents field > value.
type Gt struct {
	Field string
	Value any
}

func (Gt) isExpr() {}

// Lt represents field < value.
type Lt struct {
	Field string
	Value any
}

func (Lt) isExpr() {}

// Gte represents field >= value.
type Gte struct {
	Field string
	Value any
}

func (Gte) isExpr() {}

// Lte represents field <= value.
type Lte struct {
	Field string
	Value any
}

func (Lte) isExpr() {}

// In represents field IN (values).
type In struct {
	Field  string
	Values []any
}

func (In) isExpr() {}

// And represents logical AND of multiple expressions.
type And struct {
	Exprs []Expr
}

func (And) isExpr() {}

// Or represents logical OR of multiple expressions.
type Or struct {
	Exprs []Expr
}

func (Or) isExpr() {}

// Not represents logical NOT of a single expression (inverse of subtree).
type Not struct {
	Expr Expr
}

func (Not) isExpr() {}

// Builder helpers for constructing filter expressions.

// Equal returns an equality expression: field == value.
func Equal(field string, value any) Expr {
	return Eq{Field: field, Value: value}
}

// NotEqual returns a not-equal expression: field != value.
func NotEqual(field string, value any) Expr {
	return Neq{Field: field, Value: value}
}

// Greater returns a greater-than expression: field > value.
func Greater(field string, value any) Expr {
	return Gt{Field: field, Value: value}
}

// Less returns a less-than expression: field < value.
func Less(field string, value any) Expr {
	return Lt{Field: field, Value: value}
}

// GreaterOrEqual returns field >= value.
func GreaterOrEqual(field string, value any) Expr {
	return Gte{Field: field, Value: value}
}

// LessOrEqual returns field <= value.
func LessOrEqual(field string, value any) Expr {
	return Lte{Field: field, Value: value}
}

// OneOf returns an IN expression: field IN (vals...).
func OneOf(field string, vals ...any) Expr {
	return In{Field: field, Values: vals}
}

// All returns a logical AND of all given expressions.
func All(exprs ...Expr) Expr {
	if exprs == nil {
		exprs = []Expr{}
	}
	return And{Exprs: exprs}
}

// Any returns a logical OR of all given expressions.
func Any(exprs ...Expr) Expr {
	if exprs == nil {
		exprs = []Expr{}
	}
	return Or{Exprs: exprs}
}

// Inverse returns the logical NOT of the given expression.
func Inverse(expr Expr) Expr {
	return Not{Expr: expr}
}
