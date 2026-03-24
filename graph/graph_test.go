package graph

import (
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
)

func TestSnapshotValidateRejectsDuplicateNodeIDs(t *testing.T) {
	err := Snapshot{
		Nodes: []Node{
			{ID: "n1", Labels: []string{"Doc"}},
			{ID: "n1", Labels: []string{"Doc"}},
		},
	}.Validate()
	if !errors.Is(err, ragy.ErrInvalidGraph) {
		t.Fatalf("Validate() error = %v, want invalid graph", err)
	}
}

func TestSnapshotValidateRejectsDuplicateEdgeIDs(t *testing.T) {
	err := Snapshot{
		Nodes: []Node{
			{ID: "n1", Labels: []string{"Doc"}},
			{ID: "n2", Labels: []string{"Doc"}},
		},
		Edges: []Edge{
			{ID: "e1", SourceID: "n1", TargetID: "n2", Type: "LINKS"},
			{ID: "e1", SourceID: "n2", TargetID: "n1", Type: "LINKS"},
		},
	}.Validate()
	if !errors.Is(err, ragy.ErrInvalidGraph) {
		t.Fatalf("Validate() error = %v, want invalid graph", err)
	}
}

func TestSnapshotValidateRejectsDanglingEdges(t *testing.T) {
	err := Snapshot{
		Nodes: []Node{
			{ID: "n1", Labels: []string{"Doc"}},
		},
		Edges: []Edge{{
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

	undeclaredBuilder := filter.NewSchema()
	other, err := undeclaredBuilder.String("other")
	if err != nil {
		t.Fatalf("undeclaredBuilder.String(other): %v", err)
	}
	undeclaredFilter, err := filter.Normalize(filter.Equal(other, "acme"))
	if err != nil {
		t.Fatalf("Normalize(undeclaredFilter): %v", err)
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

	wrongKindBuilder := filter.NewSchema()
	wrongKindField, err := wrongKindBuilder.Int("tenant")
	if err != nil {
		t.Fatalf("wrongKindBuilder.Int(tenant): %v", err)
	}
	wrongKindFilter, err := filter.Normalize(filter.Equal(wrongKindField, int64(7)))
	if err != nil {
		t.Fatalf("Normalize(wrongKindFilter): %v", err)
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

	validFilter, err := filter.Normalize(filter.Equal(tenant, "acme"))
	if err != nil {
		t.Fatalf("Normalize(validFilter): %v", err)
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

	_, err = schema.NormalizeSnapshot(Snapshot{
		Nodes: []Node{{
			ID:         "n1",
			Labels:     []string{"Doc"},
			Attributes: ragy.Attributes{"tenant": 1},
		}},
	})
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("NormalizeSnapshot(node wrong kind) error = %v, want invalid argument", err)
	}

	_, err = schema.NormalizeSnapshot(Snapshot{
		Nodes: []Node{
			{ID: "n1", Labels: []string{"Doc"}},
			{ID: "n2", Labels: []string{"Doc"}},
		},
		Edges: []Edge{{
			ID:         "e1",
			SourceID:   "n1",
			TargetID:   "n2",
			Type:       "LINKS",
			Attributes: ragy.Attributes{"weight": "heavy"},
		}},
	})
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("NormalizeSnapshot(edge wrong kind) error = %v, want invalid argument", err)
	}
}
