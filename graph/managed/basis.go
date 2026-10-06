package managed

import (
	"context"
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/graph"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/source"
)

// SetHostBasis owns an immutable explicitly identified host foundation. The same
// ID cannot be reused for changed facts or re-registered after release, for the
// lifetime of this in-process adapter. It does not create managed source refs or
// participate in source cleanup; readers select the exact basis ID explicitly.
func (a *Adapter[TMeta]) SetHostBasis(ctx context.Context, id string, snapshot graph.Snapshot[TMeta]) error {
	if a == nil || id == "" || !utf8.ValidString(id) || len(snapshot.Nodes)+len(snapshot.Edges) > a.config.MaxRecords {
		return ragy.ErrInvalidArgument
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := snapshot.Validate(); err != nil {
		return err
	}
	var noManifest lifecycle.Manifest
	var noReference source.Reference
	basis := version[TMeta]{manifest: noManifest, target: "", hostBasis: id, nodes: nil, edges: nil}
	for _, node := range snapshot.Nodes {
		if err := ctx.Err(); err != nil {
			return err
		}
		stored, err := a.captureNode(Node[TMeta]{Reference: noReference, Value: node})
		if err != nil {
			return err
		}
		basis.nodes = append(basis.nodes, stored)
	}
	for _, edge := range snapshot.Edges {
		if err := ctx.Err(); err != nil {
			return err
		}
		stored, err := a.captureEdge(Edge[TMeta]{Reference: noReference, Value: edge})
		if err != nil {
			return err
		}
		basis.edges = append(basis.edges, stored)
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return err
	}
	if _, retired := a.retiredBases[id]; retired {
		return lifecycle.ErrConflict
	}
	if previous, exists := a.bases[id]; exists {
		if !sameVersion(previous, basis) {
			return lifecycle.ErrConflict
		}
		return nil
	}
	a.bases[id] = basis
	return nil
}

// ReleaseHostBasis is an explicit host retention operation, separate from managed
// source cleanup. Captured readers of a released basis fail unavailable.
func (a *Adapter[TMeta]) ReleaseHostBasis(ctx context.Context, id string) error {
	if a == nil || id == "" {
		return ragy.ErrInvalidArgument
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if _, exists := a.bases[id]; exists {
		if a.retiredBases == nil {
			a.retiredBases = make(map[string]struct{})
		}
		a.retiredBases[id] = struct{}{}
		delete(a.bases, id)
	}
	return ctx.Err()
}
