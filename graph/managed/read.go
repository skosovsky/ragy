package managed

import (
	"bytes"
	"context"
	"slices"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/graph"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/source"
)

type Request struct {
	Read      access.Binding
	HostBasis string
	Traversal graph.TraversalRequest
	MaxNodes  int
	MaxEdges  int
}
type Support struct {
	Kind       string
	ID         string
	References []source.Reference
	HostBases  []string
}
type Conflict struct {
	Kind       string
	ID         string
	References []source.Reference
	HostBases  []string
}
type Result[TMeta any] struct {
	Snapshot  graph.Snapshot[TMeta]
	Supports  []Support
	Conflicts []Conflict
}
type ReadCapabilities struct {
	OutputScope       bool
	TraversalScope    bool
	PinnedPublication bool
}

func (*Adapter[TMeta]) ReadCapabilities() ReadCapabilities {
	return ReadCapabilities{OutputScope: true, TraversalScope: true, PinnedPublication: true}
}
func (a *Adapter[TMeta]) Schema() graph.Schema { return a.config.Schema }

// AdmitTraversal validates the complete traversal/scope/publication profile before
// any target I/O. Recipes can preflight every hop before reserving or dispatching
// a graph call; admission does not load facts or grant a fresh authorization scope.
func (a *Adapter[TMeta]) AdmitTraversal(ctx context.Context, request Request) error {
	_, err := a.prepare(ctx, request)
	if err != nil {
		return access.NonSkippable(err)
	}
	return request.Read.Check(ctx)
}
func emptyResult[TMeta any]() Result[TMeta] {
	return Result[TMeta]{Snapshot: graph.Snapshot[TMeta]{Nodes: nil, Edges: nil}, Supports: nil, Conflicts: nil}
}

// Traverse applies mandatory and query node/edge admission before expansion.
// This profile uses NodeFilter for traversal as well as output, and rejects paging.
func (a *Adapter[TMeta]) Traverse(ctx context.Context, request Request) (Result[TMeta], error) {
	empty := emptyResult[TMeta]()
	result, err := a.traverse(ctx, request)
	if err != nil {
		return empty, access.NonSkippable(err)
	}
	if err = request.Read.Check(ctx); err != nil {
		return empty, err
	}
	return result, nil
}

// FindByIDs uses the same admitted snapshot without traversing edges. Forbidden
// and nonexistent IDs both produce no node payload or diagnostic identifier.
func (a *Adapter[TMeta]) FindByIDs(ctx context.Context, request Request) (Result[TMeta], error) {
	empty := emptyResult[TMeta]()
	prepared, err := a.prepare(ctx, request)
	if err != nil {
		return empty, access.NonSkippable(err)
	}
	if len(prepared.Traversal.Seeds) > prepared.MaxNodes {
		return empty, access.NonSkippable(ragy.ErrInvalidArgument)
	}
	versions, err := a.selected(ctx, prepared.Read, prepared.HostBasis)
	if err != nil {
		return empty, access.NonSkippable(err)
	}
	view, err := admitted(ctx, versions, prepared)
	if err != nil {
		return empty, access.NonSkippable(err)
	}
	ids := []string{}
	seen := map[string]bool{}
	for _, id := range prepared.Traversal.Seeds {
		if _, exists := view.nodes[id]; exists && !seen[id] {
			ids = append(ids, id)
			seen[id] = true
		}
	}
	slices.Sort(ids)
	result, err := a.deliver(ctx, prepared.Read, view, ids, nil)
	if err != nil {
		return empty, access.NonSkippable(err)
	}
	if err = prepared.Read.Check(ctx); err != nil {
		return empty, err
	}
	return result, nil
}
func (a *Adapter[TMeta]) traverse(ctx context.Context, request Request) (Result[TMeta], error) {
	prepared, err := a.prepare(ctx, request)
	if err != nil {
		return emptyResult[TMeta](), err
	}
	versions, err := a.selected(ctx, prepared.Read, prepared.HostBasis)
	if err != nil {
		return emptyResult[TMeta](), err
	}
	view, err := admitted(ctx, versions, prepared)
	if err != nil {
		return emptyResult[TMeta](), err
	}
	nodes, edges, err := walk(ctx, prepared, view)
	if err != nil {
		return emptyResult[TMeta](), err
	}
	return a.deliver(ctx, prepared.Read, view, nodes, edges)
}
func (a *Adapter[TMeta]) prepare(ctx context.Context, request Request) (Request, error) {
	if a == nil || request.MaxNodes <= 0 || request.MaxEdges <= 0 {
		return request, ragy.ErrInvalidArgument
	}
	request.Traversal.Seeds = slices.Clone(request.Traversal.Seeds)
	if err := request.Traversal.Validate(); err != nil {
		return request, err
	}
	if request.Traversal.Page != nil {
		return request, ragy.ErrUnsupported
	}
	caps := access.Capabilities{ScopeProfile: true, PinnedPublication: true, RequirePinnedPublication: true}
	if err := request.Read.Check(ctx); err != nil {
		return Request{}, err
	}
	if err := request.Read.Publication().AdmitTarget(a.config.Target); err != nil {
		return Request{}, err
	}
	nodes, err := request.Read.Prepare(ctx, a.config.Schema.NodeAttributes, request.Traversal.NodeFilter, caps)
	if err != nil {
		return request, err
	}
	edges, err := request.Read.Prepare(ctx, a.config.Schema.EdgeAttributes, request.Traversal.EdgeFilter, caps)
	if err != nil {
		return request, err
	}
	request.Traversal.NodeFilter, request.Traversal.EdgeFilter = nodes, edges
	return request, nil
}

func (a *Adapter[TMeta]) selected(
	ctx context.Context,
	read access.Binding,
	hostBasis string,
) ([]version[TMeta], error) {
	if err := read.Check(ctx); err != nil {
		return nil, err
	}
	snapshot, err := a.config.Store.Load(ctx, a.config.Namespace)
	if err != nil {
		return nil, err
	}
	if snapshot.Namespace != a.config.Namespace || snapshot.Validate() != nil {
		return nil, ragy.ErrProtocol
	}
	a.mu.RLock()
	defer a.mu.RUnlock()
	var out []version[TMeta]
	for _, target := range read.Publication().Targets() {
		if target.Target != a.config.Target {
			continue
		}
		if target.Namespace != a.config.Namespace {
			return nil, ragy.ErrUnavailable
		}
		candidate, findErr := selectedVersion(a.versions, snapshot, target)
		if findErr != nil {
			return nil, findErr
		}
		out = append(out, candidate)
	}
	if hostBasis != "" {
		basis, exists := a.bases[hostBasis]
		if !exists {
			return nil, ragy.ErrUnavailable
		}
		out = append(out, basis)
	}
	return out, nil
}

type view[TMeta any] struct {
	nodes        map[string]storedNode[TMeta]
	edges        map[string]storedEdge[TMeta]
	nodeSupports map[string][]source.Reference
	edgeSupports map[string][]source.Reference
	nodeBases    map[string][]string
	edgeBases    map[string][]string
	conflicts    []Conflict
}

func admitted[TMeta any](ctx context.Context, versions []version[TMeta], request Request) (view[TMeta], error) {
	out := view[TMeta]{
		nodes:        make(map[string]storedNode[TMeta]),
		edges:        make(map[string]storedEdge[TMeta]),
		nodeSupports: make(map[string][]source.Reference),
		edgeSupports: make(map[string][]source.Reference),
		nodeBases:    make(map[string][]string), edgeBases: make(map[string][]string),
		conflicts: nil,
	}
	badNodes, badEdges := map[string]bool{}, map[string]bool{}
	for _, version := range versions {
		if err := admitFacts(
			ctx,
			request.Read,
			request.Traversal.NodeFilter,
			supportOrigin{manifest: version.manifest, basis: version.hostBasis},
			version.nodes,
			out.nodes,
			out.nodeSupports,
			out.nodeBases,
			badNodes,
		); err != nil {
			return out, err
		}
		if err := admitFacts(
			ctx,
			request.Read,
			request.Traversal.EdgeFilter,
			supportOrigin{manifest: version.manifest, basis: version.hostBasis},
			version.edges,
			out.edges,
			out.edgeSupports,
			out.edgeBases,
			badEdges,
		); err != nil {
			return out, err
		}
	}

	for id := range badNodes {
		out.conflicts = append(
			out.conflicts,
			Conflict{
				Kind:       "node",
				ID:         id,
				References: unique(out.nodeSupports[id]),
				HostBases:  slices.Clone(out.nodeBases[id]),
			},
		)
		delete(out.nodes, id)
	}
	for id := range badEdges {
		out.conflicts = append(
			out.conflicts,
			Conflict{
				Kind:       "edge",
				ID:         id,
				References: unique(out.edgeSupports[id]),
				HostBases:  slices.Clone(out.edgeBases[id]),
			},
		)
		delete(out.edges, id)
	}
	return out, nil
}
func match(condition filter.Condition, attrs filter.RawAttributes) (bool, error) {
	return filter.MatchCondition(
		condition,
		func(field string) (any, bool) { value, exists := attrs[field]; return value, exists },
	)
}
func supports(manifest lifecycle.Manifest, ref source.Reference) []source.Reference {
	for _, target := range manifest.Targets {
		for _, artifact := range target.Artifacts {
			if artifact.Reference == ref {
				return artifact.Supports
			}
		}
	}
	return nil
}

func walk[TMeta any](ctx context.Context, request Request, view view[TMeta]) ([]string, []string, error) {
	visited := map[string]bool{}
	edgeIDs := map[string]bool{}
	frontier := []string{}
	for _, seed := range request.Traversal.Seeds {
		if _, exists := view.nodes[seed]; exists && !visited[seed] {
			visited[seed] = true
			frontier = append(frontier, seed)
		}
	}
	if len(visited) > request.MaxNodes {
		return nil, nil, ragy.ErrInvalidArgument
	}
	for depth := 0; depth < request.Traversal.Depth && len(frontier) > 0; depth++ {
		next := []string{}
		for _, id := range frontier {
			additional, err := expand(ctx, request, view, id, visited, edgeIDs)
			if err != nil {
				return nil, nil, err
			}
			next = append(next, additional...)
		}

		frontier = next
	}
	nodes, edges := []string{}, []string{}
	for id := range visited {
		nodes = append(nodes, id)
	}
	for id := range edgeIDs {
		edges = append(edges, id)
	}
	slices.Sort(nodes)
	slices.Sort(edges)
	return nodes, edges, nil
}
func neighbor[TMeta any](edge graph.Edge[TMeta], id string, direction graph.Direction) string {
	if edge.SourceID == id && (direction == graph.DirectionOutbound || direction == graph.DirectionUndirected) {
		return edge.TargetID
	}
	if edge.TargetID == id && (direction == graph.DirectionInbound || direction == graph.DirectionUndirected) {
		return edge.SourceID
	}
	return ""
}

func (a *Adapter[TMeta]) deliver(
	ctx context.Context,
	read access.Binding,
	view view[TMeta],
	nodes, edges []string,
) (Result[TMeta], error) {
	out := emptyResult[TMeta]()
	for _, id := range nodes {
		if err := read.Check(ctx); err != nil {
			return out, err
		}
		node := view.nodes[id].record.Value
		meta, err := a.config.CloneMeta(node.Meta)
		if err != nil {
			return out, err
		}
		if err = read.Check(ctx); err != nil {
			return out, err
		}
		node.Meta, node.Labels = meta, slices.Clone(node.Labels)
		out.Snapshot.Nodes = append(out.Snapshot.Nodes, node)
		out.Supports = append(
			out.Supports,
			Support{
				Kind:       "node",
				ID:         id,
				References: unique(view.nodeSupports[id]),
				HostBases:  slices.Clone(view.nodeBases[id]),
			},
		)
	}
	for _, id := range edges {
		if err := read.Check(ctx); err != nil {
			return out, err
		}
		edge := view.edges[id].record.Value
		meta, err := a.config.CloneMeta(edge.Meta)
		if err != nil {
			return out, err
		}
		if err = read.Check(ctx); err != nil {
			return out, err
		}
		edge.Meta = meta
		out.Snapshot.Edges = append(out.Snapshot.Edges, edge)
		out.Supports = append(
			out.Supports,
			Support{
				Kind:       "edge",
				ID:         id,
				References: unique(view.edgeSupports[id]),
				HostBases:  slices.Clone(view.edgeBases[id]),
			},
		)
	}
	out.Conflicts = view.conflicts
	return out, nil
}
func unique(refs []source.Reference) []source.Reference {
	seen := map[source.Reference]bool{}
	var out []source.Reference
	for _, ref := range refs {
		if !seen[ref] {
			seen[ref] = true
			out = append(out, ref)
		}
	}
	return out
}

func admitFacts[T fact](
	ctx context.Context,
	read access.Binding,
	condition filter.Condition,
	origin supportOrigin,
	records []T,
	output map[string]T,
	refs map[string][]source.Reference,
	bases map[string][]string,
	bad map[string]bool,
) error {
	for _, record := range records {
		if err := read.Check(ctx); err != nil {
			return err
		}
		allowed, err := match(condition, record.factAttrs())
		if err != nil {
			return err
		}
		if !allowed {
			continue
		}
		id := record.factID()
		if origin.basis != "" {
			bases[id] = append(bases[id], origin.basis)
		}
		refs[id] = append(refs[id], supports(origin.manifest, record.factRef())...)
		if previous, exists := output[id]; exists &&
			!bytes.Equal(previous.factFingerprint(), record.factFingerprint()) {
			bad[id] = true
		} else {
			output[id] = record
		}
	}
	return nil
}

func expand[TMeta any](
	ctx context.Context,
	request Request,
	view view[TMeta],
	id string,
	visited, edgeIDs map[string]bool,
) ([]string, error) {
	next := []string{}
	for edgeID, edge := range view.edges {
		if err := request.Read.Check(ctx); err != nil {
			return nil, err
		}
		other := neighbor(edge.record.Value, id, request.Traversal.Direction)
		if other == "" {
			continue
		}
		if _, exists := view.nodes[other]; !exists {
			continue
		}
		edgeIDs[edgeID] = true
		if len(edgeIDs) > request.MaxEdges {
			return nil, ragy.ErrInvalidArgument
		}
		if visited[other] {
			continue
		}
		visited[other] = true
		if len(visited) > request.MaxNodes {
			return nil, ragy.ErrInvalidArgument
		}
		next = append(next, other)
	}
	return next, nil
}

func confirmed(snapshot lifecycle.Snapshot, captured lifecycle.Manifest, target string) bool {
	for _, manifest := range snapshot.Manifests {
		if manifest.ID != captured.ID || manifest.Identity != captured.Identity ||
			manifest.Payload != captured.Payload ||
			manifest.PublishedAt.IsZero() ||
			manifest.Tombstone {
			continue
		}
		for _, item := range manifest.Targets {
			if item.Name == target && item.State == lifecycle.TargetReady &&
				lifecycle.SameTargetInventory(manifest, captured, target) {
				return true
			}
		}
	}
	return false
}

func selectedVersion[TMeta any](
	versions map[string]version[TMeta],
	snapshot lifecycle.Snapshot,
	target access.TargetRevision,
) (version[TMeta], error) {
	var selected version[TMeta]
	for _, candidate := range versions {
		id := candidate.manifest.Identity
		if id.Namespace != target.Namespace || id.Source != target.Source || id.Revision != target.Revision ||
			id.Transformation != target.Transformation ||
			id.Access != target.AccessFingerprint {
			continue
		}
		if !confirmed(snapshot, candidate.manifest, target.Target) || selected.manifest.ID != "" {
			return version[TMeta]{}, ragy.ErrUnavailable
		}
		selected = candidate
	}
	if selected.manifest.ID == "" {
		return version[TMeta]{}, ragy.ErrUnavailable
	}
	return selected, nil
}

type supportOrigin struct {
	manifest lifecycle.Manifest
	basis    string
}
