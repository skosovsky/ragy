package pgvector

import (
	"encoding/json"
	"testing"

	"github.com/skosovsky/ragy/filter"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSQLFilterVisitor_Nested(t *testing.T) {
	meta, err := sanitizeIdent("metadata")
	require.NoError(t, err)
	v := NewSQLFilterVisitor(meta)
	expr := filter.All(
		filter.Any(
			filter.Equal("a", 1),
			filter.Inverse(filter.Equal("b", "x")),
		),
		filter.OneOf("c", "u", "v"),
	)
	sql, args, err := v.ToSQL(expr, 1)
	require.NoError(t, err)
	assert.Contains(t, sql, "AND")
	assert.Contains(t, sql, "OR")
	assert.Contains(t, sql, "NOT")
	assert.GreaterOrEqual(t, len(args), 3)
}

func TestSQLFilterVisitor_In(t *testing.T) {
	meta, err := sanitizeIdent("metadata")
	require.NoError(t, err)
	v := NewSQLFilterVisitor(meta)
	sql, args, err := v.ToSQL(filter.OneOf("k", "a", "b"), 10)
	require.NoError(t, err)
	assert.Contains(t, sql, "IN")
	assert.Contains(t, sql, "$10")
	assert.Contains(t, sql, "$11")
	assert.Len(t, args, 2)
}

// Positional placeholders must stay consistent when merging with main query args (startArg offsets).
func TestSQLFilterVisitor_PositionalOffsets(t *testing.T) {
	meta, err := sanitizeIdent("metadata")
	require.NoError(t, err)
	v := NewSQLFilterVisitor(meta)
	expr := filter.All(
		filter.Equal("tenant", "t1"),
		filter.OneOf("role", "a", "b"),
	)
	sql, args, err := v.ToSQL(expr, 4)
	require.NoError(t, err)
	assert.Contains(t, sql, "$4")
	assert.Contains(t, sql, "$5")
	assert.Contains(t, sql, "$6")
	require.Len(t, args, 3)
	var tenant map[string]any
	require.NoError(t, json.Unmarshal(args[0].([]byte), &tenant))
	assert.Equal(t, "t1", tenant["tenant"])
}

func TestSQLFilterVisitor_DeepAndOr(t *testing.T) {
	meta, err := sanitizeIdent("metadata")
	require.NoError(t, err)
	v := NewSQLFilterVisitor(meta)
	sql, args, err := v.ToSQL(filter.All(
		filter.Any(filter.Equal("x", 1), filter.Equal("y", 2)),
		filter.Inverse(filter.Equal("z", 3)),
	), 1)
	require.NoError(t, err)
	assert.Contains(t, sql, "AND")
	assert.Contains(t, sql, "OR")
	assert.Contains(t, sql, "NOT")
	require.GreaterOrEqual(t, len(args), 3)
}
