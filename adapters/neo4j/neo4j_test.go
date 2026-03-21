package neo4j

import (
	"testing"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/skosovsky/ragy/filter"
)

func TestSanitizeLabel(t *testing.T) {
	assert.Equal(t, "Node", sanitizeLabel("bad-label!"))
	assert.Equal(t, "Person", sanitizeLabel("Person"))
}

func TestCopyProps_FiltersUnsafeKeys(t *testing.T) {
	in := map[string]any{"ok": 1, "bad!": 2, "also_ok": "x"}
	out := copyProps(in)
	assert.Equal(t, 1, out["ok"])
	assert.Equal(t, "x", out["also_ok"])
	assert.NotContains(t, out, "bad!")
}

func TestNeo4jNodeToRagy(t *testing.T) {
	n := neo4j.Node{
		ElementId: "elem-1",
		Labels:    []string{"Person"},
		Props:     map[string]any{"id": "u1", "name": "Alice"},
	}
	node := neo4jNodeToRagy(n)
	assert.Equal(t, "u1", node.ID)
	assert.Equal(t, "Person", node.Label)
	assert.Equal(t, "Alice", node.Properties["name"])
}

func TestNeo4jNodeToRagy_UsesElementIdWhenNoIdProp(t *testing.T) {
	n := neo4j.Node{
		ElementId: "elem-1",
		Labels:    []string{"X"},
		Props:     map[string]any{"name": "Bob"},
	}
	node := neo4jNodeToRagy(n)
	assert.Equal(t, "elem-1", node.ID)
}

func TestNeo4jRelToRagy(t *testing.T) {
	r := neo4j.Relationship{
		ElementId:      "rel-1",
		StartElementId: "start-1",
		EndElementId:   "end-1",
		Type:           "KNOWS",
		Props:          map[string]any{"since": 2020},
	}
	e := neo4jRelToRagy(r)
	assert.Equal(t, "start-1", e.SourceID)
	assert.Equal(t, "end-1", e.TargetID)
	assert.Equal(t, "KNOWS", e.Relation)
	assert.Equal(t, 2020, e.Properties["since"])
}

func TestBuildCypherWhere_EqAndIn(t *testing.T) {
	clause, params := buildCypherWhere(filter.All(
		filter.Equal("tenant", "t1"),
		filter.OneOf("role", "a", "b"),
	))
	require.NotEmpty(t, clause)
	assert.Contains(t, clause, "AND")
	assert.Contains(t, clause, "n.tenant")
	assert.Contains(t, clause, "n.role")
	assert.Len(t, params, 2)
}

func TestBuildCypherWhere_OrNot(t *testing.T) {
	clause, params := buildCypherWhere(filter.Any(
		filter.Equal("a", 1),
		filter.Inverse(filter.Equal("z", "bad")),
	))
	require.NotEmpty(t, clause)
	assert.Contains(t, clause, "OR")
	assert.Contains(t, clause, "NOT")
	assert.NotEmpty(t, params)
}

func TestBuildCypherWhere_Nil(t *testing.T) {
	clause, params := buildCypherWhere(nil)
	assert.Empty(t, clause)
	assert.Empty(t, params)
}

func TestJoinCypher(t *testing.T) {
	assert.Equal(t, "a AND b", joinCypher([]string{"a", "b"}, " AND "))
	assert.Empty(t, joinCypher(nil, " AND "))
}
