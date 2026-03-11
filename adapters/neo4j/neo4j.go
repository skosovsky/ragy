// Package neo4j provides a ragy.GraphStore implementation using Neo4j (Cypher).
//
// UpsertGraph uses MERGE for nodes and edges; SearchGraph uses MATCH (n)-[r*1..depth]-(m) with filter.Expr in WHERE.
// Property keys in Cypher cannot be parameterized; values are parameterized. Use fieldSanitize for key safety.
//
// SearchGraph accepts only entity IDs (start nodes); Vector Index Integration (finding start nodes by
// req.DenseVector via Neo4j vector index) is not implemented yet and is planned for a future release.
//
// SearchGraph returns nodes (with deduplication by ID) and edges. Edge SourceID/TargetID are Neo4j driver
// element IDs (internal identifiers) unless the path has a single edge, in which case business IDs (n.id, m.id)
// are used when returned by the query. To correlate edges with ragy.Node, prefer matching Node.ID (from node
// properties) with the edge endpoints when possible.
package neo4j

import (
	"context"
	"fmt"
	"regexp"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
	"github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
)

var fieldSanitize = regexp.MustCompile(`^[a-zA-Z0-9_]+$`)

const maxDepth = 5

// Store implements ragy.GraphStore using Neo4j.
type Store struct {
	driver neo4j.DriverWithContext
}

// Option configures the Store (reserved for future options).
type Option func(*Store)

// New returns a new Neo4j GraphStore.
func New(driver neo4j.DriverWithContext, _ ...Option) *Store {
	return &Store{driver: driver}
}

// UpsertGraph implements ragy.GraphStore. MERGE nodes by id, then MERGE edges.
func (s *Store) UpsertGraph(ctx context.Context, nodes []ragy.Node, edges []ragy.Edge) error {
	if len(nodes) == 0 && len(edges) == 0 {
		return nil
	}
	sess := s.driver.NewSession(ctx, neo4j.SessionConfig{AccessMode: neo4j.AccessModeWrite})
	defer func() { _ = sess.Close(ctx) }()
	_, err := sess.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		for _, n := range nodes {
			label := sanitizeLabel(n.Label)
			if label == "" {
				label = "Node"
			}
			props := copyProps(n.Properties)
			if props == nil {
				props = make(map[string]any)
			}
			props["id"] = n.ID
			query := fmt.Sprintf("MERGE (n:%s {id: $id}) SET n += $props", label)
			if _, err := tx.Run(ctx, query, map[string]any{"id": n.ID, "props": props}); err != nil {
				return nil, err
			}
		}
		for _, e := range edges {
			rel := sanitizeLabel(e.Relation)
			if rel == "" {
				rel = "RELATES"
			}
			props := copyProps(e.Properties)
			if props == nil {
				props = make(map[string]any)
			}
			query := fmt.Sprintf("MERGE (a {id: $sid}) MERGE (b {id: $tid}) MERGE (a)-[r:%s]->(b) SET r += $props", rel)
			if _, err := tx.Run(ctx, query, map[string]any{"sid": e.SourceID, "tid": e.TargetID, "props": props}); err != nil {
				return nil, err
			}
		}
		return nil, nil
	})
	return err
}

// SearchGraph implements ragy.GraphStore. MATCH (n)-[r*1..depth]-(m) WHERE n.id IN $entities, optional filter.
func (s *Store) SearchGraph(ctx context.Context, entities []string, depth int, req ragy.SearchRequest) ([]ragy.Node, []ragy.Edge, error) {
	if len(entities) == 0 {
		return nil, nil, nil
	}
	if depth <= 0 || depth > maxDepth {
		depth = maxDepth
	}
	sess := s.driver.NewSession(ctx, neo4j.SessionConfig{AccessMode: neo4j.AccessModeRead})
	defer func() { _ = sess.Close(ctx) }()
	whereFilter, params := buildCypherWhere(req.Filter, "n", 1)
	params["entities"] = entities
	query := fmt.Sprintf("MATCH (n)-[r*1..%d]-(m) WHERE n.id IN $entities", depth)
	if whereFilter != "" {
		query += " AND " + whereFilter
	}
	query += " RETURN n, r, m, n.id AS start_id, m.id AS end_id"
	result, err := sess.Run(ctx, query, params)
	if err != nil {
		return nil, nil, err
	}
	seenNodes := make(map[string]struct{})
	seenEdges := make(map[string]struct{})
	var nodes []ragy.Node
	var edges []ragy.Edge
	for result.Next(ctx) {
		rec := result.Record()
		if v, ok := rec.Get("n"); ok {
			if node, ok := v.(neo4j.Node); ok {
				ragyN := neo4jNodeToRagy(node)
				if _, seen := seenNodes[ragyN.ID]; !seen {
					seenNodes[ragyN.ID] = struct{}{}
					nodes = append(nodes, ragyN)
				}
			}
		}
		if v, ok := rec.Get("m"); ok {
			if node, ok := v.(neo4j.Node); ok {
				ragyM := neo4jNodeToRagy(node)
				if _, seen := seenNodes[ragyM.ID]; !seen {
					seenNodes[ragyM.ID] = struct{}{}
					nodes = append(nodes, ragyM)
				}
			}
		}
		startID, _ := recordString(rec, "start_id")
		endID, _ := recordString(rec, "end_id")
		if v, ok := rec.Get("r"); ok {
			if segs, ok := v.([]any); ok {
				for _, seg := range segs {
					if rel, ok := seg.(neo4j.Relationship); ok {
						if _, seen := seenEdges[rel.ElementId]; seen {
							continue
						}
						seenEdges[rel.ElementId] = struct{}{}
						edge := neo4jRelToRagy(rel)
						if len(segs) == 1 && startID != "" && endID != "" {
							edge.SourceID = startID
							edge.TargetID = endID
						}
						edges = append(edges, edge)
					}
				}
			}
		}
	}
	if err := result.Err(); err != nil {
		return nil, nil, err
	}
	return nodes, edges, nil
}

func sanitizeLabel(s string) string {
	if !fieldSanitize.MatchString(s) {
		return "Node"
	}
	return s
}

func copyProps(m map[string]any) map[string]any {
	if m == nil {
		return nil
	}
	out := make(map[string]any, len(m))
	for k, v := range m {
		if fieldSanitize.MatchString(k) {
			out[k] = v
		}
	}
	return out
}

func neo4jNodeToRagy(n neo4j.Node) ragy.Node {
	props := make(map[string]any)
	for k, v := range n.Props {
		props[k] = v
	}
	id, _ := props["id"].(string)
	if id == "" {
		id = n.ElementId
	}
	label := "Unknown"
	if len(n.Labels) > 0 {
		label = n.Labels[0]
	}
	return ragy.Node{ID: id, Label: label, Properties: props}
}

func recordString(rec *neo4j.Record, key string) (string, bool) {
	v, ok := rec.Get(key)
	if !ok || v == nil {
		return "", false
	}
	if s, ok := v.(string); ok {
		return s, true
	}
	return "", false
}

func neo4jRelToRagy(r neo4j.Relationship) ragy.Edge {
	props := make(map[string]any)
	for k, v := range r.Props {
		props[k] = v
	}
	return ragy.Edge{SourceID: r.StartElementId, TargetID: r.EndElementId, Relation: r.Type, Properties: props}
}

// buildCypherWhere returns WHERE clause fragment and params. Keys are sanitized and concatenated; values are params.
func buildCypherWhere(expr filter.Expr, nodeVar string, paramStart int) (clause string, params map[string]any) {
	params = make(map[string]any)
	if expr == nil {
		return "", params
	}
	clause, _ = buildCypherWhereRec(expr, nodeVar, paramStart, params)
	return clause, params
}

func buildCypherWhereRec(expr filter.Expr, nodeVar string, paramIdx int, params map[string]any) (string, int) {
	switch e := expr.(type) {
	case filter.Eq:
		if !fieldSanitize.MatchString(e.Field) {
			return "", paramIdx
		}
		key := fmt.Sprintf("p%d", paramIdx)
		params[key] = e.Value
		return fmt.Sprintf("%s.%s = $%s", nodeVar, e.Field, key), paramIdx + 1
	case filter.Neq:
		if !fieldSanitize.MatchString(e.Field) {
			return "", paramIdx
		}
		key := fmt.Sprintf("p%d", paramIdx)
		params[key] = e.Value
		return fmt.Sprintf("%s.%s <> $%s", nodeVar, e.Field, key), paramIdx + 1
	case filter.In:
		if !fieldSanitize.MatchString(e.Field) || len(e.Values) == 0 {
			return "", paramIdx
		}
		key := fmt.Sprintf("p%d", paramIdx)
		params[key] = e.Values
		return fmt.Sprintf("%s.%s IN $%s", nodeVar, e.Field, key), paramIdx + 1
	case filter.Gt, filter.Gte, filter.Lt, filter.Lte:
		field, val, op := "", any(nil), ""
		switch x := expr.(type) {
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
			return "", paramIdx
		}
		key := fmt.Sprintf("p%d", paramIdx)
		params[key] = val
		return fmt.Sprintf("%s.%s %s $%s", nodeVar, field, op, key), paramIdx + 1
	case filter.And:
		var parts []string
		for _, sub := range e.Exprs {
			part, next := buildCypherWhereRec(sub, nodeVar, paramIdx, params)
			paramIdx = next
			if part != "" {
				parts = append(parts, part)
			}
		}
		if len(parts) == 0 {
			return "", paramIdx
		}
		return "(" + joinCypher(parts, " AND ") + ")", paramIdx
	case filter.Or:
		var parts []string
		for _, sub := range e.Exprs {
			part, next := buildCypherWhereRec(sub, nodeVar, paramIdx, params)
			paramIdx = next
			if part != "" {
				parts = append(parts, part)
			}
		}
		if len(parts) == 0 {
			return "", paramIdx
		}
		return "(" + joinCypher(parts, " OR ") + ")", paramIdx
	case filter.Not:
		part, next := buildCypherWhereRec(e.Expr, nodeVar, paramIdx, params)
		if part == "" {
			return "", paramIdx
		}
		return "NOT (" + part + ")", next
	default:
		return "", paramIdx
	}
}

func joinCypher(parts []string, sep string) string {
	if len(parts) == 0 {
		return ""
	}
	s := parts[0]
	for i := 1; i < len(parts); i++ {
		s += sep + parts[i]
	}
	return s
}

var _ ragy.GraphStore = (*Store)(nil)
