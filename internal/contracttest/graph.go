package contracttest

import (
	"context"
	"errors"
	"slices"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/graph"
)

type GraphStoreFactory func(t *testing.T, snapshot graph.Snapshot, schema graph.Schema) graph.Store

// RunGraphStoreSuite checks common graph.Store traversal semantics.
func RunGraphStoreSuite(t *testing.T, factory GraphStoreFactory) {
	t.Helper()

	base := graph.Snapshot{
		Nodes: []graph.Node{
			{ID: "n1", Labels: []string{"Doc"}, Content: "", Attributes: ragy.Attributes{"tenant": "acme"}},
			{ID: "n2", Labels: []string{"Doc"}, Content: "", Attributes: ragy.Attributes{"tenant": "acme"}},
			{ID: "n3", Labels: []string{"Doc"}, Content: "", Attributes: ragy.Attributes{"tenant": "globex"}},
		},
		Edges: []graph.Edge{
			{ID: "e12", SourceID: "n1", TargetID: "n2", Type: "LINKS", Attributes: ragy.Attributes{"kind": "keep"}},
			{ID: "e23", SourceID: "n2", TargetID: "n3", Type: "LINKS", Attributes: ragy.Attributes{"kind": "drop"}},
			{ID: "e31", SourceID: "n3", TargetID: "n1", Type: "LINKS", Attributes: ragy.Attributes{"kind": "keep"}},
		},
	}

	schema := buildGraphSchema(t)

	t.Run("direction and depth", func(t *testing.T) {
		testDirectionAndDepth(t, factory, base, schema)
	})

	t.Run("node and edge filters", func(t *testing.T) {
		testNodeAndEdgeFilters(t, factory, base, schema)
	})

	t.Run("page trims nodes and dependent edges", func(t *testing.T) {
		testPageTrimsNodesAndEdges(t, factory, base, schema)
	})

	t.Run("invalid traverse output rejects", func(t *testing.T) {
		testInvalidTraverseOutputRejects(t, factory, schema)
	})

	t.Run("invalid upsert snapshot rejects", func(t *testing.T) {
		testInvalidUpsertSnapshotRejects(t, factory, schema)
	})

	t.Run("schema invalid traverse output rejects", func(t *testing.T) {
		testSchemaInvalidTraverseOutputRejects(t, factory, schema)
	})

	t.Run("schema invalid upsert snapshot rejects", func(t *testing.T) {
		testSchemaInvalidUpsertSnapshotRejects(t, factory, schema)
	})

	t.Run("undeclared graph filter rejects", func(t *testing.T) {
		testUndeclaredGraphFilterRejects(t, factory, base, schema)
	})

	t.Run("wrong graph filter kind rejects", func(t *testing.T) {
		testWrongGraphFilterKindRejects(t, factory, base, schema)
	})
}

func buildGraphSchema(t *testing.T) graph.Schema {
	t.Helper()

	nodeBuilder := filter.NewSchema()
	if _, err := nodeBuilder.String("tenant"); err != nil {
		t.Fatalf("nodeBuilder.String(tenant): %v", err)
	}
	nodeSchema, err := nodeBuilder.Build()
	if err != nil {
		t.Fatalf("nodeBuilder.Build(): %v", err)
	}

	edgeBuilder := filter.NewSchema()
	if _, fieldErr := edgeBuilder.String("kind"); fieldErr != nil {
		t.Fatalf("edgeBuilder.String(kind): %v", fieldErr)
	}
	edgeSchema, err := edgeBuilder.Build()
	if err != nil {
		t.Fatalf("edgeBuilder.Build(): %v", err)
	}

	schema, err := graph.NewSchema(nodeSchema, edgeSchema)
	if err != nil {
		t.Fatalf("graph.NewSchema(): %v", err)
	}

	return schema
}

func testDirectionAndDepth(t *testing.T, factory GraphStoreFactory, base graph.Snapshot, schema graph.Schema) {
	t.Helper()

	store := factory(t, base, schema)
	out, err := store.Traverse(context.Background(), graph.TraversalRequest{
		Seeds:      []string{"n1"},
		Direction:  graph.DirectionOutbound,
		Depth:      1,
		NodeFilter: nil,
		EdgeFilter: nil,
		Page:       nil,
	})
	if err != nil {
		t.Fatalf("Traverse(outbound): %v", err)
	}

	if got := idsOfNodes(out.Nodes); !equalStrings(got, []string{"n1", "n2"}) {
		t.Fatalf("outbound nodes = %v, want [n1 n2]", got)
	}
	if got := idsOfEdges(out.Edges); !equalStrings(got, []string{"e12"}) {
		t.Fatalf("outbound edges = %v, want [e12]", got)
	}

	in, err := store.Traverse(context.Background(), graph.TraversalRequest{
		Seeds:      []string{"n1"},
		Direction:  graph.DirectionInbound,
		Depth:      1,
		NodeFilter: nil,
		EdgeFilter: nil,
		Page:       nil,
	})
	if err != nil {
		t.Fatalf("Traverse(inbound): %v", err)
	}

	if got := idsOfNodes(in.Nodes); !equalStrings(got, []string{"n1", "n3"}) {
		t.Fatalf("inbound nodes = %v, want [n1 n3]", got)
	}
	if got := idsOfEdges(in.Edges); !equalStrings(got, []string{"e31"}) {
		t.Fatalf("inbound edges = %v, want [e31]", got)
	}
}

func testNodeAndEdgeFilters(
	t *testing.T,
	factory GraphStoreFactory,
	base graph.Snapshot,
	schema graph.Schema,
) {
	t.Helper()

	store := factory(t, base, schema)
	nodeTenant, err := store.Schema().NodeAttributes.StringField("tenant")
	if err != nil {
		t.Fatalf("Schema().NodeAttributes.StringField(tenant): %v", err)
	}
	nodeFilter, err := filter.Normalize(filter.Equal(nodeTenant, "acme"))
	if err != nil {
		t.Fatalf("Normalize(nodeFilter): %v", err)
	}
	edgeKind, err := store.Schema().EdgeAttributes.StringField("kind")
	if err != nil {
		t.Fatalf("Schema().EdgeAttributes.StringField(kind): %v", err)
	}
	edgeFilter, err := filter.Normalize(filter.Equal(edgeKind, "keep"))
	if err != nil {
		t.Fatalf("Normalize(edgeFilter): %v", err)
	}

	out, err := store.Traverse(context.Background(), graph.TraversalRequest{
		Seeds:      []string{"n1"},
		Direction:  graph.DirectionUndirected,
		Depth:      2,
		NodeFilter: nodeFilter,
		EdgeFilter: edgeFilter,
		Page:       nil,
	})
	if err != nil {
		t.Fatalf("Traverse(filtered): %v", err)
	}

	if got := idsOfNodes(out.Nodes); !equalStrings(got, []string{"n1", "n2"}) {
		t.Fatalf("filtered nodes = %v, want [n1 n2]", got)
	}
	if got := idsOfEdges(out.Edges); !equalStrings(got, []string{"e12"}) {
		t.Fatalf("filtered edges = %v, want [e12]", got)
	}
}

func testPageTrimsNodesAndEdges(
	t *testing.T,
	factory GraphStoreFactory,
	base graph.Snapshot,
	schema graph.Schema,
) {
	t.Helper()

	store := factory(t, base, schema)
	page, err := ragy.NewPage(1, 1)
	if err != nil {
		t.Fatalf("NewPage(): %v", err)
	}

	out, err := store.Traverse(context.Background(), graph.TraversalRequest{
		Seeds:      []string{"n1"},
		Direction:  graph.DirectionUndirected,
		Depth:      2,
		NodeFilter: nil,
		EdgeFilter: nil,
		Page:       page,
	})
	if err != nil {
		t.Fatalf("Traverse(paged): %v", err)
	}

	if got := idsOfNodes(out.Nodes); !equalStrings(got, []string{"n2"}) {
		t.Fatalf("paged nodes = %v, want [n2]", got)
	}
	if len(out.Edges) != 0 {
		t.Fatalf("paged edges = %v, want none", idsOfEdges(out.Edges))
	}
}

func testInvalidTraverseOutputRejects(t *testing.T, factory GraphStoreFactory, schema graph.Schema) {
	t.Helper()

	store := factory(t, graph.Snapshot{
		Nodes: []graph.Node{{
			ID:         "n1",
			Labels:     []string{"Doc"},
			Content:    "",
			Attributes: nil,
		}},
		Edges: []graph.Edge{{
			ID:         "e1",
			SourceID:   "n1",
			TargetID:   "missing",
			Type:       "LINKS",
			Attributes: nil,
		}},
	}, schema)

	_, err := store.Traverse(context.Background(), graph.TraversalRequest{
		Seeds:      []string{"n1"},
		Direction:  graph.DirectionOutbound,
		Depth:      1,
		NodeFilter: nil,
		EdgeFilter: nil,
		Page:       nil,
	})
	if err == nil {
		t.Fatal("Traverse() error = nil, want invalid snapshot error")
	}
}

func testInvalidUpsertSnapshotRejects(t *testing.T, factory GraphStoreFactory, schema graph.Schema) {
	t.Helper()

	store := factory(t, graph.Snapshot{
		Nodes: nil,
		Edges: nil,
	}, schema)
	err := store.Upsert(context.Background(), graph.Snapshot{
		Nodes: []graph.Node{{
			ID:         "n1",
			Labels:     []string{"Doc"},
			Content:    "",
			Attributes: nil,
		}},
		Edges: []graph.Edge{{
			ID:         "e1",
			SourceID:   "n1",
			TargetID:   "missing",
			Type:       "LINKS",
			Attributes: nil,
		}},
	})
	if err == nil {
		t.Fatal("Upsert() error = nil, want invalid snapshot error")
	}
}

func testSchemaInvalidTraverseOutputRejects(t *testing.T, factory GraphStoreFactory, schema graph.Schema) {
	t.Helper()

	store := factory(t, graph.Snapshot{
		Nodes: []graph.Node{{
			ID:         "n1",
			Labels:     []string{"Doc"},
			Content:    "",
			Attributes: ragy.Attributes{"tenant": 1},
		}},
		Edges: nil,
	}, schema)

	_, err := store.Traverse(context.Background(), graph.TraversalRequest{
		Seeds:      []string{"n1"},
		Direction:  graph.DirectionOutbound,
		Depth:      1,
		NodeFilter: nil,
		EdgeFilter: nil,
		Page:       nil,
	})
	if err == nil {
		t.Fatal("Traverse() error = nil, want schema-invalid snapshot error")
	}
}

func testSchemaInvalidUpsertSnapshotRejects(t *testing.T, factory GraphStoreFactory, schema graph.Schema) {
	t.Helper()

	store := factory(t, graph.Snapshot{Nodes: nil, Edges: nil}, schema)

	err := store.Upsert(context.Background(), graph.Snapshot{
		Nodes: []graph.Node{
			{ID: "n1", Labels: []string{"Doc"}, Content: "", Attributes: nil},
			{ID: "n2", Labels: []string{"Doc"}, Content: "", Attributes: nil},
		},
		Edges: []graph.Edge{{
			ID:         "e1",
			SourceID:   "n1",
			TargetID:   "n2",
			Type:       "LINKS",
			Attributes: ragy.Attributes{"kind": true},
		}},
	})
	if err == nil {
		t.Fatal("Upsert() error = nil, want schema-invalid snapshot error")
	}
}

func testUndeclaredGraphFilterRejects(
	t *testing.T,
	factory GraphStoreFactory,
	base graph.Snapshot,
	schema graph.Schema,
) {
	t.Helper()

	store := factory(t, base, schema)
	_, err := store.Schema().NodeAttributes.StringField("other")
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Schema().NodeAttributes.StringField(other) error = %v, want invalid argument", err)
	}
}

func testWrongGraphFilterKindRejects(
	t *testing.T,
	factory GraphStoreFactory,
	base graph.Snapshot,
	schema graph.Schema,
) {
	t.Helper()

	store := factory(t, base, schema)
	_, err := store.Schema().NodeAttributes.IntField("tenant")
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Schema().NodeAttributes.IntField(tenant) error = %v, want invalid argument", err)
	}
}

func idsOfNodes(nodes []graph.Node) []string {
	out := make([]string, len(nodes))
	for i, node := range nodes {
		out[i] = node.ID
	}
	return out
}

func idsOfEdges(edges []graph.Edge) []string {
	out := make([]string, len(edges))
	for i, edge := range edges {
		out[i] = edge.ID
	}
	return out
}

func equalStrings(left, right []string) bool {
	return slices.Equal(left, right)
}
