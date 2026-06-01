package graph

import (
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
)

type testMeta map[string]any

func TestSnapshotValidateRejectsDuplicateNodeIDs(t *testing.T) {
	err := Snapshot[testMeta]{
		Nodes: []Node[testMeta]{
			{ID: "n1", Labels: []string{"Doc"}},
			{ID: "n1", Labels: []string{"Doc"}},
		},
	}.Validate()
	if !errors.Is(err, ragy.ErrInvalidGraph) {
		t.Fatalf("Validate() error = %v, want invalid graph", err)
	}
}

func TestSnapshotValidateRejectsDuplicateEdgeIDs(t *testing.T) {
	err := Snapshot[testMeta]{
		Nodes: []Node[testMeta]{
			{ID: "n1", Labels: []string{"Doc"}},
			{ID: "n2", Labels: []string{"Doc"}},
		},
		Edges: []Edge[testMeta]{
			{ID: "e1", SourceID: "n1", TargetID: "n2", Type: "LINKS"},
			{ID: "e1", SourceID: "n2", TargetID: "n1", Type: "LINKS"},
		},
	}.Validate()
	if !errors.Is(err, ragy.ErrInvalidGraph) {
		t.Fatalf("Validate() error = %v, want invalid graph", err)
	}
}

func TestSnapshotValidateRejectsDanglingEdges(t *testing.T) {
	err := Snapshot[testMeta]{
		Nodes: []Node[testMeta]{
			{ID: "n1", Labels: []string{"Doc"}},
		},
		Edges: []Edge[testMeta]{{
			ID:       "e1",
			SourceID: "n1",
			TargetID: "missing",
			Type:     "LINKS",
		}},
	}.Validate()
	if !errors.Is(err, ragy.ErrInvalidGraph) {
		t.Fatalf("Validate() error = %v, want invalid graph", err)
	}
}

func TestSchemaValidateTraversalRejectsUndeclaredAndWrongKind(t *testing.T) {
	nodeBuilder := filter.NewSchema()
	tenant, err := nodeBuilder.String("tenant")
	if err != nil {
		t.Fatalf("nodeBuilder.String(tenant): %v", err)
	}
	nodeSchema, err := nodeBuilder.Build()
	if err != nil {
		t.Fatalf("nodeBuilder.Build(): %v", err)
	}

	edgeSchema, err := filter.NewSchema().Build()
	if err != nil {
		t.Fatalf("edgeBuilder.Build(): %v", err)
	}

	schema, err := NewSchema(nodeSchema, edgeSchema)
	if err != nil {
		t.Fatalf("NewSchema(): %v", err)
	}

	undeclaredSchemaBuilder := filter.NewSchema()
	other, err := undeclaredSchemaBuilder.String("other")
	if err != nil {
		t.Fatalf("undeclaredSchemaBuilder.String(other): %v", err)
	}
	undeclaredSchema, err := undeclaredSchemaBuilder.Build()
	if err != nil {
		t.Fatalf("undeclaredSchemaBuilder.Build(): %v", err)
	}
	undeclaredFilterBuilder, err := filter.NewBuilder(undeclaredSchema)
	if err != nil {
		t.Fatalf("NewBuilder(undeclared): %v", err)
	}
	undeclaredFilter, err := filter.Eq(undeclaredFilterBuilder, other, "acme").Build()
	if err != nil {
		t.Fatalf("Build(undeclaredFilter): %v", err)
	}

	err = schema.ValidateTraversal(TraversalRequest{
		Seeds:      []string{"n1"},
		Direction:  DirectionOutbound,
		Depth:      1,
		NodeFilter: undeclaredFilter,
	})
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("ValidateTraversal(undeclared) error = %v, want invalid argument", err)
	}

	wrongKindSchemaBuilder := filter.NewSchema()
	wrongKindField, err := wrongKindSchemaBuilder.Int("tenant")
	if err != nil {
		t.Fatalf("wrongKindSchemaBuilder.Int(tenant): %v", err)
	}
	wrongKindSchema, err := wrongKindSchemaBuilder.Build()
	if err != nil {
		t.Fatalf("wrongKindSchemaBuilder.Build(): %v", err)
	}
	wrongKindFilterBuilder, err := filter.NewBuilder(wrongKindSchema)
	if err != nil {
		t.Fatalf("NewBuilder(wrongKind): %v", err)
	}
	wrongKindFilter, err := filter.Eq(wrongKindFilterBuilder, wrongKindField, int64(7)).Build()
	if err != nil {
		t.Fatalf("Build(wrongKindFilter): %v", err)
	}

	err = schema.ValidateTraversal(TraversalRequest{
		Seeds:      []string{"n1"},
		Direction:  DirectionOutbound,
		Depth:      1,
		NodeFilter: wrongKindFilter,
	})
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("ValidateTraversal(wrong kind) error = %v, want invalid argument", err)
	}

	validFilterBuilder, err := filter.NewBuilder(nodeSchema)
	if err != nil {
		t.Fatalf("NewBuilder(node): %v", err)
	}
	validFilter, err := filter.Eq(validFilterBuilder, tenant, "acme").Build()
	if err != nil {
		t.Fatalf("Build(validFilter): %v", err)
	}

	if err := schema.ValidateTraversal(TraversalRequest{
		Seeds:      []string{"n1"},
		Direction:  DirectionOutbound,
		Depth:      1,
		NodeFilter: validFilter,
	}); err != nil {
		t.Fatalf("ValidateTraversal(valid) error = %v", err)
	}
}

func TestSchemaNormalizeSnapshotRejectsWrongAttributeKinds(t *testing.T) {
	nodeBuilder := filter.NewSchema()
	nodeTenant, err := nodeBuilder.String("tenant")
	if err != nil {
		t.Fatalf("nodeBuilder.String(tenant): %v", err)
	}
	_ = nodeTenant
	nodeSchema, err := nodeBuilder.Build()
	if err != nil {
		t.Fatalf("nodeBuilder.Build(): %v", err)
	}

	edgeBuilder := filter.NewSchema()
	edgeWeight, err := edgeBuilder.Int("weight")
	if err != nil {
		t.Fatalf("edgeBuilder.Int(weight): %v", err)
	}
	_ = edgeWeight
	edgeSchema, err := edgeBuilder.Build()
	if err != nil {
		t.Fatalf("edgeBuilder.Build(): %v", err)
	}

	schema, err := NewSchema(nodeSchema, edgeSchema)
	if err != nil {
		t.Fatalf("NewSchema(): %v", err)
	}

	_, err = NormalizeSnapshot(schema, Snapshot[testMeta]{
		Nodes: []Node[testMeta]{{
			ID:     "n1",
			Labels: []string{"Doc"},
			Meta:   testMeta{"tenant": 1},
		}},
	})
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("NormalizeSnapshot(node wrong kind) error = %v, want invalid argument", err)
	}

	_, err = NormalizeSnapshot(schema, Snapshot[testMeta]{
		Nodes: []Node[testMeta]{
			{ID: "n1", Labels: []string{"Doc"}},
			{ID: "n2", Labels: []string{"Doc"}},
		},
		Edges: []Edge[testMeta]{{
			ID:       "e1",
			SourceID: "n1",
			TargetID: "n2",
			Type:     "LINKS",
			Meta:     testMeta{"weight": "heavy"},
		}},
	})
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("NormalizeSnapshot(edge wrong kind) error = %v, want invalid argument", err)
	}
}
