package pgvector

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/skosovsky/ragy/filter"
)

// SQLFilterVisitor translates filter.Expr into a SQL fragment and positional arguments for PostgreSQL JSONB metadata.
// Identifiers for JSON field names are validated; values are always passed as parameters (no string concatenation of values).
type SQLFilterVisitor struct {
	metaCol sanitizedIdent
}

// NewSQLFilterVisitor creates a visitor that builds conditions on the given metadata column (sanitized).
func NewSQLFilterVisitor(metaCol sanitizedIdent) *SQLFilterVisitor {
	return &SQLFilterVisitor{metaCol: metaCol}
}

// ToSQL returns a WHERE fragment (without leading WHERE) and positional args starting at startArg ($startArg, ...).
func (v *SQLFilterVisitor) ToSQL(expr filter.Expr, startArg int) (string, []any, error) {
	if expr == nil {
		return "", nil, nil
	}
	return v.walk(expr, startArg)
}

//nolint:gocognit,funlen // filter.Expr recursive translation is inherently branchy.
func (v *SQLFilterVisitor) walk(expr filter.Expr, startArg int) (string, []any, error) {
	switch e := expr.(type) {
	case filter.Eq:
		if !fieldSanitize.MatchString(e.Field) {
			return "", nil, fmt.Errorf("pgvector: invalid field name %q", e.Field)
		}
		jb, err := json.Marshal(map[string]any{e.Field: e.Value})
		if err != nil {
			return "", nil, err
		}
		var b strings.Builder
		b.WriteString(string(v.metaCol))
		b.WriteString(" @> $")
		b.WriteString(strconv.Itoa(startArg))
		return b.String(), []any{jb}, nil

	case filter.Neq, filter.Gt, filter.Gte, filter.Lt, filter.Lte:
		field, val, op := "", any(nil), ""
		switch x := expr.(type) {
		case filter.Neq:
			field, val, op = x.Field, x.Value, "!="
		case filter.Gt:
			field, val, op = x.Field, x.Value, ">"
		case filter.Gte:
			field, val, op = x.Field, x.Value, ">="
		case filter.Lt:
			field, val, op = x.Field, x.Value, "<"
		case filter.Lte:
			field, val, op = x.Field, x.Value, "<="
		}
		if !fieldSanitize.MatchString(field) {
			return "", nil, fmt.Errorf("pgvector: invalid field name %q", field)
		}
		cast := "::text"
		switch val.(type) {
		case int, int32, int64, float32, float64:
			cast = "::numeric"
		case time.Time:
			cast = "::timestamptz"
		}
		var b strings.Builder
		b.WriteString("(")
		b.WriteString(string(v.metaCol))
		b.WriteString("->>'")
		b.WriteString(field)
		b.WriteString("')")
		b.WriteString(cast)
		b.WriteByte(' ')
		b.WriteString(op)
		b.WriteString(" $")
		b.WriteString(strconv.Itoa(startArg))
		return b.String(), []any{val}, nil

	case filter.In:
		if !fieldSanitize.MatchString(e.Field) {
			return "", nil, fmt.Errorf("pgvector: invalid field name %q", e.Field)
		}
		if len(e.Values) == 0 {
			return "false", nil, nil
		}
		var b strings.Builder
		b.WriteByte('(')
		b.WriteString(string(v.metaCol))
		b.WriteString("->>'")
		b.WriteString(e.Field)
		b.WriteString("') IN (")
		for i := range e.Values {
			if i > 0 {
				b.WriteString(", ")
			}
			b.WriteByte('$')
			b.WriteString(strconv.Itoa(startArg + i))
		}
		b.WriteByte(')')
		return b.String(), e.Values, nil

	case filter.And:
		var parts []string
		var allArgs []any
		argIdx := startArg
		for _, sub := range e.Exprs {
			part, a, err := v.walk(sub, argIdx)
			if err != nil {
				return "", nil, err
			}
			parts = append(parts, "("+part+")")
			allArgs = append(allArgs, a...)
			argIdx += len(a)
		}
		return joinParts(parts, " AND "), allArgs, nil

	case filter.Or:
		var parts []string
		var allArgs []any
		argIdx := startArg
		for _, sub := range e.Exprs {
			part, a, err := v.walk(sub, argIdx)
			if err != nil {
				return "", nil, err
			}
			parts = append(parts, "("+part+")")
			allArgs = append(allArgs, a...)
			argIdx += len(a)
		}
		return joinParts(parts, " OR "), allArgs, nil

	case filter.Not:
		part, a, err := v.walk(e.Expr, startArg)
		if err != nil {
			return "", nil, err
		}
		return "NOT (" + part + ")", a, nil

	default:
		return "", nil, fmt.Errorf("pgvector: unsupported filter type %T", expr)
	}
}

func joinParts(parts []string, sep string) string {
	if len(parts) == 0 {
		return ""
	}
	var s strings.Builder
	s.WriteString(parts[0])
	for i := 1; i < len(parts); i++ {
		s.WriteString(sep + parts[i])
	}
	return s.String()
}
