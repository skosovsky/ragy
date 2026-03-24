// Package graph provides graph traversal and storage contracts.
package graph

import (
	"context"
	"fmt"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/internal/ident"
)

// Direction controls edge traversal semantics.
type Direction string

const (
	DirectionOutbound   Direction = "outbound"
	DirectionInbound    Direction = "inbound"
	DirectionUndirected Direction = "undirected"
)

// Node is the canonical graph node.
type Node struct {
	ID         string
	Labels     []string
	Content    string
	Attributes ragy.Attributes
}

// Validate checks node invariants.
func (n Node) Validate() error {
	if n.ID == "" {
		return fmt.Errorf("%w: graph node id", ragy.ErrMissingID)
	}

	if len(n.Labels) == 0 {
		return fmt.Errorf("%w: graph node labels", ragy.ErrInvalidGraph)
	}
	for _, label := range n.Labels {
		if err := filter.ValidateSQLIdentifier(label); err != nil {
			return err
		}
	}
	for key := range n.Attributes {
		if !ident.IsField(key) {
			return fmt.Errorf("%w: invalid identifier %q", ragy.ErrInvalidArgument, key)
		}
	}

	return nil
}

// Edge is the canonical graph edge.
type Edge struct {
	ID         string
	SourceID   string
	TargetID   string
	Type       string
	Attributes ragy.Attributes
}

// Validate checks edge invariants.
func (e Edge) Validate() error {
	switch {
	case e.ID == "":
		return fmt.Errorf("%w: graph edge id", ragy.ErrMissingID)
	case e.SourceID == "":
		return fmt.Errorf("%w: graph edge source id", ragy.ErrInvalidGraph)
	case e.TargetID == "":
		return fmt.Errorf("%w: graph edge target id", ragy.ErrInvalidGraph)
	case e.Type == "":
		return fmt.Errorf("%w: graph edge type", ragy.ErrInvalidGraph)
	default:
		if err := filter.ValidateSQLIdentifier(e.Type); err != nil {
			return err
		}
		for key := range e.Attributes {
			if !ident.IsField(key) {
				return fmt.Errorf("%w: invalid identifier %q", ragy.ErrInvalidArgument, key)
			}
		}
		return nil
	}
}

// Snapshot is a graph payload.
type Snapshot struct {
	Nodes []Node
	Edges []Edge
}

// Validate checks snapshot invariants.
func (s Snapshot) Validate() error {
	nodeIDs := make(map[string]struct{}, len(s.Nodes))
	for _, node := range s.Nodes {
		if err := node.Validate(); err != nil {
			return err
		}
		if _, exists := nodeIDs[node.ID]; exists {
			return fmt.Errorf("%w: duplicate graph node id %q", ragy.ErrInvalidGraph, node.ID)
		}
		nodeIDs[node.ID] = struct{}{}
	}

	edgeIDs := make(map[string]struct{}, len(s.Edges))
	for _, edge := range s.Edges {
		if err := edge.Validate(); err != nil {
			return err
		}
		if _, exists := edgeIDs[edge.ID]; exists {
			return fmt.Errorf("%w: duplicate graph edge id %q", ragy.ErrInvalidGraph, edge.ID)
		}
		edgeIDs[edge.ID] = struct{}{}
		if _, ok := nodeIDs[edge.SourceID]; !ok {
			return fmt.Errorf("%w: graph edge source %q missing node", ragy.ErrInvalidGraph, edge.SourceID)
		}
		if _, ok := nodeIDs[edge.TargetID]; !ok {
			return fmt.Errorf("%w: graph edge target %q missing node", ragy.ErrInvalidGraph, edge.TargetID)
		}
	}

	return nil
}

// Schema defines the allowed node and edge attribute fields for traversal and payloads.
type Schema struct {
	NodeAttributes filter.Schema
	EdgeAttributes filter.Schema
}

// EmptySchema constructs a finalized graph schema with no declared attributes.
func EmptySchema() Schema {
	return Schema{
		NodeAttributes: filter.EmptySchema(),
		EdgeAttributes: filter.EmptySchema(),
	}
}

// NewSchema constructs a graph schema from finalized node and edge attribute schemas.
func NewSchema(nodeAttrs, edgeAttrs filter.Schema) (Schema, error) {
	if !nodeAttrs.IsFinalized() {
		return Schema{}, fmt.Errorf("%w: graph node schema", ragy.ErrInvalidArgument)
	}
	if !edgeAttrs.IsFinalized() {
		return Schema{}, fmt.Errorf("%w: graph edge schema", ragy.ErrInvalidArgument)
	}

	return Schema{
		NodeAttributes: nodeAttrs,
		EdgeAttributes: edgeAttrs,
	}, nil
}

// Validate checks graph schema invariants.
func (s Schema) Validate() error {
	if !s.NodeAttributes.IsFinalized() {
		return fmt.Errorf("%w: graph node schema", ragy.ErrInvalidArgument)
	}
	if !s.EdgeAttributes.IsFinalized() {
		return fmt.Errorf("%w: graph edge schema", ragy.ErrInvalidArgument)
	}

	return nil
}

// ValidateTraversal validates traversal filters against the graph schema.
func (s Schema) ValidateTraversal(req TraversalRequest) error {
	if err := s.Validate(); err != nil {
		return err
	}
	if err := req.Validate(); err != nil {
		return err
	}

	if err := s.NodeAttributes.ValidateSchemaIR(req.NodeFilter); err != nil {
		return err
	}

	return s.EdgeAttributes.ValidateSchemaIR(req.EdgeFilter)
}

// NormalizeSnapshot validates and normalizes a graph payload against the schema.
func (s Schema) NormalizeSnapshot(snapshot Snapshot) (Snapshot, error) {
	if err := s.Validate(); err != nil {
		return Snapshot{}, err
	}
	if err := snapshot.Validate(); err != nil {
		return Snapshot{}, err
	}

	out := Snapshot{
		Nodes: make([]Node, len(snapshot.Nodes)),
		Edges: make([]Edge, len(snapshot.Edges)),
	}

	for i, node := range snapshot.Nodes {
		attrs, err := s.NodeAttributes.NormalizeAttributes(node.Attributes)
		if err != nil {
			return Snapshot{}, err
		}

		out.Nodes[i] = Node{
			ID:         node.ID,
			Labels:     append([]string(nil), node.Labels...),
			Content:    node.Content,
			Attributes: ragy.CloneAttributes(attrs),
		}
	}

	for i, edge := range snapshot.Edges {
		attrs, err := s.EdgeAttributes.NormalizeAttributes(edge.Attributes)
		if err != nil {
			return Snapshot{}, err
		}

		out.Edges[i] = Edge{
			ID:         edge.ID,
			SourceID:   edge.SourceID,
			TargetID:   edge.TargetID,
			Type:       edge.Type,
			Attributes: ragy.CloneAttributes(attrs),
		}
	}

	return out, nil
}

// TraversalRequest is an explicit graph traversal contract.
//
// NodeFilter applies to nodes in the returned snapshot. EdgeFilter applies to
// edges considered traversable and returned.
type TraversalRequest struct {
	Seeds      []string
	Direction  Direction
	Depth      int
	NodeFilter filter.IR
	EdgeFilter filter.IR
	Page       *ragy.Page
}

// Validate checks traversal invariants.
func (r TraversalRequest) Validate() error {
	if len(r.Seeds) == 0 {
		return fmt.Errorf("%w: graph seeds", ragy.ErrInvalidGraph)
	}

	if r.Depth <= 0 {
		return fmt.Errorf("%w: graph depth must be > 0", ragy.ErrInvalidGraph)
	}

	switch r.Direction {
	case DirectionOutbound, DirectionInbound, DirectionUndirected:
	default:
		return fmt.Errorf("%w: graph direction %q", ragy.ErrInvalidGraph, r.Direction)
	}

	return r.Page.Validate()
}

// Store provides graph traversal and writes.
type Store interface {
	Traverse(ctx context.Context, req TraversalRequest) (Snapshot, error)
	Upsert(ctx context.Context, snapshot Snapshot) error
	Schema() Schema
}
