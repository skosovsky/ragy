// Package graph provides graph traversal and storage contracts.
package graph

import (
	"context"
	"fmt"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/internal/metaattrs"
)

// Direction controls edge traversal semantics.
type Direction string

const (
	DirectionOutbound   Direction = "outbound"
	DirectionInbound    Direction = "inbound"
	DirectionUndirected Direction = "undirected"
)

// Node is the canonical graph node.
type Node[TMeta any] struct {
	ID      string
	Labels  []string
	Content string
	Meta    TMeta
}

// Validate checks node invariants.
func (n Node[TMeta]) Validate() error {
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
	if err := validateMetaKeys(n.Meta); err != nil {
		return err
	}

	return nil
}

// Edge is the canonical graph edge.
type Edge[TMeta any] struct {
	ID       string
	SourceID string
	TargetID string
	Type     string
	Meta     TMeta
}

// Validate checks edge invariants.
func (e Edge[TMeta]) Validate() error {
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
		if err := validateMetaKeys(e.Meta); err != nil {
			return err
		}
		return nil
	}
}

// Snapshot is a graph payload.
type Snapshot[TMeta any] struct {
	Nodes []Node[TMeta]
	Edges []Edge[TMeta]
}

// Validate checks snapshot invariants.
func (s Snapshot[TMeta]) Validate() error {
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

func validateMetaKeys[TMeta any](meta TMeta) error {
	if _, ok := metaattrs.FromValue(meta); ok {
		return fmt.Errorf("%w: map metadata is not supported in public API", ragy.ErrInvalidArgument)
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

	if err := s.NodeAttributes.ValidateSchemaIR(req.NodeFilter.IR()); err != nil {
		return err
	}

	return s.EdgeAttributes.ValidateSchemaIR(req.EdgeFilter.IR())
}

// NormalizeSnapshot validates and normalizes a graph payload against the schema.
func NormalizeSnapshot[TMeta any](s Schema, snapshot Snapshot[TMeta]) (Snapshot[TMeta], error) {
	if err := s.Validate(); err != nil {
		return Snapshot[TMeta]{}, err
	}
	if err := snapshot.Validate(); err != nil {
		return Snapshot[TMeta]{}, err
	}

	out := Snapshot[TMeta]{
		Nodes: make([]Node[TMeta], len(snapshot.Nodes)),
		Edges: make([]Edge[TMeta], len(snapshot.Edges)),
	}

	for i, node := range snapshot.Nodes {
		meta, err := NormalizeMeta(s.NodeAttributes, node.Meta)
		if err != nil {
			return Snapshot[TMeta]{}, err
		}

		out.Nodes[i] = Node[TMeta]{
			ID:      node.ID,
			Labels:  append([]string(nil), node.Labels...),
			Content: node.Content,
			Meta:    meta,
		}
	}

	for i, edge := range snapshot.Edges {
		meta, err := NormalizeMeta(s.EdgeAttributes, edge.Meta)
		if err != nil {
			return Snapshot[TMeta]{}, err
		}

		out.Edges[i] = Edge[TMeta]{
			ID:       edge.ID,
			SourceID: edge.SourceID,
			TargetID: edge.TargetID,
			Type:     edge.Type,
			Meta:     meta,
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
	NodeFilter filter.Condition
	EdgeFilter filter.Condition
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
type Store[TMeta any] interface {
	Traverse(ctx context.Context, req TraversalRequest) (Snapshot[TMeta], error)
	Upsert(ctx context.Context, snapshot Snapshot[TMeta]) error
	Schema() Schema
}
