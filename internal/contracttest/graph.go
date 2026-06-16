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

type GraphStoreFactory func(t *testing.T, snapshot graph.Snapshot[StructMeta], schema graph.Schema) graph.Store[StructMeta]

// RunGraphStoreSuite checks common graph.Store traversal semantics.
func RunGraphStoreSuite(t *testing.T, factory GraphStoreFactory) {
	t.Helper()

	base := graph.Snapshot[StructMeta]{
		Nodes: []graph.Node[StructMeta]{
			{ID: "n1", Labels: []string{"Doc"}, Content: "", Meta: StructMeta{Tenant: tenantAcme}},
			{ID: "n2", Labels: []string{"Doc"}, Content: "", Meta: StructMeta{Tenant: tenantAcme}},
			{ID: "n3", Labels: []string{"Doc"}, Content: "", Meta: StructMeta{Tenant: "globex"}},
		},
		Edges: []graph.Edge[StructMeta]{
			{ID: "e12", SourceID: "n1", TargetID: "n2", Type: "LINKS", Meta: StructMeta{Kind: "keep"}},
			{ID: "e23", SourceID: "n2", TargetID: "n3", Type: "LINKS", Meta: StructMeta{Kind: "drop"}},
			{ID: "e31", SourceID: "n3", TargetID: "n1", Type: "LINKS", Meta: StructMeta{Kind: "keep"}},
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

func testDirectionAndDepth(
	t *testing.T,
	factory GraphStoreFactory,
	base graph.Snapshot[StructMeta],
	schema graph.Schema,
) {
	t.Helper()

	store := factory(t, base, schema)
	out, err := store.Traverse(context.Background(), graph.TraversalRequest{
		Seeds:      []string{"n1"},
		Direction:  graph.DirectionOutbound,
		Depth:      1,
		NodeFilter: emptyFilter(t, schema.NodeAttributes),
		EdgeFilter: emptyFilter(t, schema.EdgeAttributes),
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
		NodeFilter: emptyFilter(t, schema.NodeAttributes),
		EdgeFilter: emptyFilter(t, schema.EdgeAttributes),
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
	base graph.Snapshot[StructMeta],
	schema graph.Schema,
) {
	t.Helper()

	store := factory(t, base, schema)
	nodeTenant, err := store.Schema().NodeAttributes.StringField("tenant")
	if err != nil {
		t.Fatalf("Schema().NodeAttributes.StringField(tenant): %v", err)
	}
	nodeBuilder, err := filter.NewBuilder(store.Schema().NodeAttributes)
	if err != nil {
		t.Fatalf("NewBuilder(node): %v", err)
	}
	nodeFilter, err := filter.Eq(nodeBuilder, nodeTenant, tenantAcme).Build()
	if err != nil {
		t.Fatalf("Build(nodeFilter): %v", err)
	}
	edgeKind, err := store.Schema().EdgeAttributes.StringField("kind")
	if err != nil {
		t.Fatalf("Schema().EdgeAttributes.StringField(kind): %v", err)
	}
	edgeBuilder, err := filter.NewBuilder(store.Schema().EdgeAttributes)
	if err != nil {
		t.Fatalf("NewBuilder(edge): %v", err)
	}
	edgeFilter, err := filter.Eq(edgeBuilder, edgeKind, "keep").Build()
	if err != nil {
		t.Fatalf("Build(edgeFilter): %v", err)
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
	base graph.Snapshot[StructMeta],
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
		NodeFilter: emptyFilter(t, schema.NodeAttributes),
		EdgeFilter: emptyFilter(t, schema.EdgeAttributes),
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

	store := factory(t, graph.Snapshot[StructMeta]{
		Nodes: []graph.Node[StructMeta]{{
			ID:      "n1",
			Labels:  []string{"Doc"},
			Content: "",
		}},
		Edges: []graph.Edge[StructMeta]{{
			ID:       "e1",
			SourceID: "n1",
			TargetID: "missing",
			Type:     "LINKS",
		}},
	}, schema)

	_, err := store.Traverse(context.Background(), graph.TraversalRequest{
		Seeds:      []string{"n1"},
		Direction:  graph.DirectionOutbound,
		Depth:      1,
		NodeFilter: emptyFilter(t, schema.NodeAttributes),
		EdgeFilter: emptyFilter(t, schema.EdgeAttributes),
		Page:       nil,
	})
	if !errors.Is(err, ragy.ErrInvalidGraph) {
		t.Fatalf("Traverse() error = %v, want invalid graph", err)
	}
}

func testInvalidUpsertSnapshotRejects(t *testing.T, factory GraphStoreFactory, schema graph.Schema) {
	t.Helper()

	store := factory(t, graph.Snapshot[StructMeta]{
		Nodes: nil,
		Edges: nil,
	}, schema)
	err := store.Upsert(context.Background(), graph.Snapshot[StructMeta]{
		Nodes: []graph.Node[StructMeta]{{
			ID:      "n1",
			Labels:  []string{"Doc"},
			Content: "",
		}},
		Edges: []graph.Edge[StructMeta]{{
			ID:       "e1",
			SourceID: "n1",
			TargetID: "missing",
			Type:     "LINKS",
		}},
	})
	if !errors.Is(err, ragy.ErrInvalidGraph) {
		t.Fatalf("Upsert() error = %v, want invalid graph", err)
	}
}

func testSchemaInvalidTraverseOutputRejects(t *testing.T, factory GraphStoreFactory, schema graph.Schema) {
	t.Helper()

	store := factory(t, graph.Snapshot[StructMeta]{
		Nodes: []graph.Node[StructMeta]{{
			ID:      "n1",
			Labels:  []string{"bad-label"},
			Content: "",
			Meta:    StructMeta{Tenant: tenantAcme},
		}},
		Edges: nil,
	}, schema)

	_, err := store.Traverse(context.Background(), graph.TraversalRequest{
		Seeds:      []string{"n1"},
		Direction:  graph.DirectionOutbound,
		Depth:      1,
		NodeFilter: emptyFilter(t, schema.NodeAttributes),
		EdgeFilter: emptyFilter(t, schema.EdgeAttributes),
		Page:       nil,
	})
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Traverse() error = %v, want invalid argument", err)
	}
}

func testSchemaInvalidUpsertSnapshotRejects(t *testing.T, factory GraphStoreFactory, schema graph.Schema) {
	t.Helper()

	store := factory(t, graph.Snapshot[StructMeta]{Nodes: nil, Edges: nil}, schema)

	err := store.Upsert(context.Background(), graph.Snapshot[StructMeta]{
		Nodes: []graph.Node[StructMeta]{
			{ID: "n1", Labels: []string{"Doc"}, Content: ""},
			{ID: "n2", Labels: []string{"Doc"}, Content: ""},
		},
		Edges: []graph.Edge[StructMeta]{{
			ID:       "e1",
			SourceID: "n1",
			TargetID: "n2",
			Type:     "bad-type",
			Meta:     StructMeta{Kind: "keep"},
		}},
	})
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Upsert() error = %v, want invalid argument", err)
	}
}

func testUndeclaredGraphFilterRejects(
	t *testing.T,
	factory GraphStoreFactory,
	base graph.Snapshot[StructMeta],
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
	base graph.Snapshot[StructMeta],
	schema graph.Schema,
) {
	t.Helper()

	store := factory(t, base, schema)
	_, err := store.Schema().NodeAttributes.IntField("tenant")
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Schema().NodeAttributes.IntField(tenant) error = %v, want invalid argument", err)
	}
}

func emptyFilter(t *testing.T, schema filter.Schema) filter.Condition {
	t.Helper()

	builder, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatalf("NewBuilder(): %v", err)
	}
	cond, err := builder.Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}
	return cond
}

func idsOfNodes(nodes []graph.Node[StructMeta]) []string {
	out := make([]string, len(nodes))
	for i, node := range nodes {
		out[i] = node.ID
	}
	return out
}

func idsOfEdges(edges []graph.Edge[StructMeta]) []string {
	out := make([]string, len(edges))
	for i, edge := range edges {
		out[i] = edge.ID
	}
	return out
}

func equalStrings(left, right []string) bool {
	return slices.Equal(left, right)
}
