// Package managed implements a source-supported in-process graph target.
// Missing retained inventory after process restart is explicitly unavailable.
package managed

import (
	"bytes"
	"context"
	"encoding/json"
	"slices"
	"sync"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/graph"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

type Node[TMeta any] struct {
	Reference source.Reference
	Value     graph.Node[TMeta]
}
type Edge[TMeta any] struct {
	Reference source.Reference
	Value     graph.Edge[TMeta]
}
type Payload[TMeta any] struct {
	Nodes []Node[TMeta]
	Edges []Edge[TMeta]
}
type Config[TMeta any] struct {
	Namespace  string
	Target     string
	Store      lifecycle.Store
	Schema     graph.Schema
	NodeCodec  retrieval.MetadataCodec[TMeta]
	EdgeCodec  retrieval.MetadataCodec[TMeta]
	CloneMeta  func(TMeta) (TMeta, error)
	MaxRecords int
}
type storedNode[TMeta any] struct {
	record      Node[TMeta]
	attributes  filter.RawAttributes
	fingerprint []byte
}
type storedEdge[TMeta any] struct {
	record      Edge[TMeta]
	attributes  filter.RawAttributes
	fingerprint []byte
}
type version[TMeta any] struct {
	manifest  lifecycle.Manifest
	hostBasis string
	nodes     []storedNode[TMeta]
	edges     []storedEdge[TMeta]
}
type Adapter[TMeta any] struct {
	mu       sync.RWMutex
	config   Config[TMeta]
	versions map[string]version[TMeta]
	bases    map[string]version[TMeta]
}

func New[TMeta any](config Config[TMeta]) (*Adapter[TMeta], error) {
	if config.Namespace == "" || config.Target == "" || config.Store == nil || config.CloneMeta == nil ||
		config.MaxRecords <= 0 {
		return nil, ragy.ErrInvalidArgument
	}
	if err := config.Schema.Validate(); err != nil {
		return nil, err
	}
	if config.NodeCodec == nil {
		config.NodeCodec = retrieval.NewJSONCodec[TMeta](config.Schema.NodeAttributes)
	}
	if config.EdgeCodec == nil {
		config.EdgeCodec = retrieval.NewJSONCodec[TMeta](config.Schema.EdgeAttributes)
	}
	return &Adapter[TMeta]{
		mu:       sync.RWMutex{},
		config:   config,
		versions: make(map[string]version[TMeta]),
		bases:    make(map[string]version[TMeta]),
	}, nil
}

func (a *Adapter[TMeta]) Stage(
	ctx context.Context,
	request lifecycle.StageRequest,
	input Payload[TMeta],
) (lifecycle.StageResult, error) {
	request.Manifest = request.Manifest.Clone()
	captured, err := a.capture(ctx, request, input)
	if err != nil {
		return lifecycle.StageResult{}, err
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if err = a.checkStage(ctx, request); err != nil {
		return lifecycle.StageResult{}, err
	}
	if previous, exists := a.versions[request.Manifest.ID]; exists {
		if !sameVersion(previous, captured) {
			return lifecycle.StageResult{}, lifecycle.ErrConflict
		}
	} else {
		a.versions[request.Manifest.ID] = captured
	}
	if err = ctx.Err(); err != nil {
		return lifecycle.StageResult{}, err
	}
	return lifecycle.StageResult{State: lifecycle.TargetReady, Revision: request.Manifest.Identity.Revision}, nil
}
func (a *Adapter[TMeta]) Inspect(ctx context.Context, request lifecycle.StageRequest) (lifecycle.StageResult, error) {
	if a == nil || request.Target != a.config.Target || request.Manifest.Identity.Namespace != a.config.Namespace {
		return lifecycle.StageResult{}, ragy.ErrInvalidArgument
	}
	if err := ctx.Err(); err != nil {
		return lifecycle.StageResult{}, err
	}
	if request.Manifest.Tombstone {
		return lifecycle.StageResult{}, ragy.ErrInvalidArgument
	}
	if err := request.Manifest.Validate(); err != nil {
		return lifecycle.StageResult{}, err
	}
	if inventory(request.Manifest, request.Target) == nil {
		return lifecycle.StageResult{}, ragy.ErrProtocol
	}
	a.mu.RLock()
	defer a.mu.RUnlock()
	stored, exists := a.versions[request.Manifest.ID]
	if !exists {
		return lifecycle.StageResult{State: lifecycle.TargetPending, Revision: ""}, nil
	}
	if stored.manifest.Identity != request.Manifest.Identity || stored.manifest.Payload != request.Manifest.Payload ||
		!lifecycle.SameTargetInventory(stored.manifest, request.Manifest, request.Target) {
		return lifecycle.StageResult{}, ragy.ErrProtocol
	}
	if err := ctx.Err(); err != nil {
		return lifecycle.StageResult{}, err
	}
	return lifecycle.StageResult{State: lifecycle.TargetReady, Revision: stored.manifest.Identity.Revision}, nil
}

func (a *Adapter[TMeta]) capture(
	ctx context.Context,
	request lifecycle.StageRequest,
	input Payload[TMeta],
) (version[TMeta], error) {
	if a == nil || request.Target != a.config.Target || request.Manifest.Identity.Namespace != a.config.Namespace ||
		request.Manifest.Tombstone ||
		len(input.Nodes)+len(input.Edges) > a.config.MaxRecords {
		return version[TMeta]{}, ragy.ErrInvalidArgument
	}
	if err := request.Manifest.Validate(); err != nil {
		return version[TMeta]{}, err
	}
	refs := inventory(request.Manifest, request.Target)
	if refs == nil || len(refs) != len(input.Nodes)+len(input.Edges) {
		return version[TMeta]{}, ragy.ErrProtocol
	}
	if err := validateSupports(request); err != nil {
		return version[TMeta]{}, err
	}
	out := version[TMeta]{manifest: request.Manifest, hostBasis: "", nodes: nil, edges: nil}
	snapshot := graph.Snapshot[TMeta]{Nodes: nil, Edges: nil}
	for _, node := range input.Nodes {
		if err := ctx.Err(); err != nil {
			return version[TMeta]{}, err
		}
		if err := consumeReference(refs, node.Reference, node.Value.ID, "graph-node"); err != nil {
			return version[TMeta]{}, err
		}
		stored, err := a.captureNode(node)
		if err != nil {
			return version[TMeta]{}, err
		}
		out.nodes = append(out.nodes, stored)
		snapshot.Nodes = append(snapshot.Nodes, stored.record.Value)
	}
	var err error
	out.edges, snapshot.Edges, err = a.captureEdges(ctx, refs, input.Edges)
	if err != nil {
		return version[TMeta]{}, err
	}

	if err = snapshot.Validate(); err != nil {
		return version[TMeta]{}, err
	}
	// Own the manifest inventory independently of the caller's slices.
	data, err := json.Marshal(request.Manifest)
	if err != nil {
		return version[TMeta]{}, err
	}
	if err = json.Unmarshal(data, &out.manifest); err != nil {
		return version[TMeta]{}, err
	}
	return out, nil
}
func inventory(manifest lifecycle.Manifest, target string) map[source.Reference]struct{} {
	for _, item := range manifest.Targets {
		if item.Name != target {
			continue
		}
		refs := make(map[source.Reference]struct{}, len(item.Artifacts))
		for _, artifact := range item.Artifacts {
			refs[artifact.Reference] = struct{}{}
		}
		return refs
	}
	return nil
}
func consumeReference(refs map[source.Reference]struct{}, ref source.Reference, id, representation string) error {
	if ref.Artifact != id || ref.Representation != representation {
		return ragy.ErrProtocol
	}
	if _, exists := refs[ref]; !exists {
		return ragy.ErrProtocol
	}
	delete(refs, ref)
	return nil
}

func captureAttributes[TMeta any](
	codec retrieval.MetadataCodec[TMeta],
	schema filter.Schema,
	meta TMeta,
) (filter.RawAttributes, error) {
	attrs, err := codec.Encode(meta)
	if err != nil {
		return nil, err
	}
	data, err := json.Marshal(attrs)
	if err != nil {
		return nil, err
	}
	var owned filter.RawAttributes
	if err = json.Unmarshal(data, &owned); err != nil {
		return nil, err
	}
	return schema.NormalizeAttributes(owned)
}
func (a *Adapter[TMeta]) captureNode(node Node[TMeta]) (storedNode[TMeta], error) {
	if err := node.Value.Validate(); err != nil {
		return storedNode[TMeta]{}, err
	}
	node.Value.Labels = slices.Clone(node.Value.Labels)
	slices.Sort(node.Value.Labels)
	node.Value.Labels = slices.Compact(node.Value.Labels)
	attrs, err := captureAttributes(a.config.NodeCodec, a.config.Schema.NodeAttributes, node.Value.Meta)
	if err != nil {
		return storedNode[TMeta]{}, err
	}
	node.Value.Meta, err = a.config.CloneMeta(node.Value.Meta)
	if err != nil {
		return storedNode[TMeta]{}, err
	}
	data, err := json.Marshal(struct {
		ID         string               `json:"id"`
		Labels     []string             `json:"labels"`
		Content    string               `json:"content"`
		Attributes filter.RawAttributes `json:"attributes"`
	}{ID: node.Value.ID, Labels: node.Value.Labels, Content: node.Value.Content, Attributes: attrs})
	return storedNode[TMeta]{record: node, attributes: attrs, fingerprint: data}, err
}
func (a *Adapter[TMeta]) captureEdge(edge Edge[TMeta]) (storedEdge[TMeta], error) {
	if err := edge.Value.Validate(); err != nil {
		return storedEdge[TMeta]{}, err
	}
	attrs, err := captureAttributes(a.config.EdgeCodec, a.config.Schema.EdgeAttributes, edge.Value.Meta)
	if err != nil {
		return storedEdge[TMeta]{}, err
	}
	edge.Value.Meta, err = a.config.CloneMeta(edge.Value.Meta)
	if err != nil {
		return storedEdge[TMeta]{}, err
	}
	data, err := json.Marshal(struct {
		ID         string               `json:"id"`
		Source     string               `json:"source"`
		Target     string               `json:"target"`
		Type       string               `json:"type"`
		Attributes filter.RawAttributes `json:"attributes"`
	}{ID: edge.Value.ID, Source: edge.Value.SourceID, Target: edge.Value.TargetID, Type: edge.Value.Type, Attributes: attrs})
	return storedEdge[TMeta]{record: edge, attributes: attrs, fingerprint: data}, err
}
func sameVersion[TMeta any](a, b version[TMeta]) bool {
	if len(a.nodes) != len(b.nodes) || len(a.edges) != len(b.edges) {
		return false
	}
	for i := range a.nodes {
		if a.nodes[i].record.Reference != b.nodes[i].record.Reference ||
			!bytes.Equal(a.nodes[i].fingerprint, b.nodes[i].fingerprint) {
			return false
		}
	}
	for i := range a.edges {
		if a.edges[i].record.Reference != b.edges[i].record.Reference ||
			!bytes.Equal(a.edges[i].fingerprint, b.edges[i].fingerprint) {
			return false
		}
	}
	return true
}
func (a *Adapter[TMeta]) checkStage(ctx context.Context, request lifecycle.StageRequest) error {
	snapshot, err := a.config.Store.Load(ctx, a.config.Namespace)
	if err != nil {
		return err
	}
	if snapshot.Namespace != a.config.Namespace || snapshot.Validate() != nil {
		return ragy.ErrProtocol
	}
	found := false
	for _, manifest := range snapshot.Manifests {
		if manifest.ID != request.Manifest.ID || manifest.Identity != request.Manifest.Identity ||
			manifest.Payload != request.Manifest.Payload ||
			manifest.ExpectedPublication != request.Manifest.ExpectedPublication {
			continue
		}
		for _, target := range manifest.Targets {
			if target.Name == request.Target && target.State == lifecycle.TargetUnknown &&
				lifecycle.SameTargetInventory(manifest, request.Manifest, request.Target) {
				found = true
			}
		}
	}
	if !found {
		return ragy.ErrProtocol
	}
	active := ""
	for _, publication := range snapshot.Publications {
		if publication.Source == request.Manifest.Identity.Source {
			active = publication.Manifest
		}
	}
	if active != request.Manifest.ExpectedPublication {
		return lifecycle.ErrConflict
	}
	return ctx.Err()
}
func validateSupports(request lifecycle.StageRequest) error {
	for _, target := range request.Manifest.Targets {
		if target.Name != request.Target {
			continue
		}
		for _, artifact := range target.Artifacts {
			for _, support := range artifact.Supports {
				id := request.Manifest.Identity
				if support.Namespace != id.Namespace || support.Source != id.Source ||
					support.Revision != id.Revision ||
					support.AccessFingerprint != id.Access {
					return ragy.ErrUnsupported
				}
			}
		}
	}
	return nil
}

func (a *Adapter[TMeta]) captureEdges(
	ctx context.Context,
	refs map[source.Reference]struct{},
	edges []Edge[TMeta],
) ([]storedEdge[TMeta], []graph.Edge[TMeta], error) {
	storedEdges := []storedEdge[TMeta]{}
	graphEdges := []graph.Edge[TMeta]{}
	for _, edge := range edges {
		if err := ctx.Err(); err != nil {
			return nil, nil, err
		}
		if err := consumeReference(refs, edge.Reference, edge.Value.ID, "graph-edge"); err != nil {
			return nil, nil, err
		}
		stored, err := a.captureEdge(edge)
		if err != nil {
			return nil, nil, err
		}
		storedEdges = append(storedEdges, stored)
		graphEdges = append(graphEdges, stored.record.Value)
	}
	return storedEdges, graphEdges, nil
}

type fact interface {
	factID() string
	factAttrs() filter.RawAttributes
	factFingerprint() []byte
	factRef() source.Reference
}

func (n storedNode[TMeta]) factID() string                  { return n.record.Value.ID }
func (n storedNode[TMeta]) factAttrs() filter.RawAttributes { return n.attributes }
func (n storedNode[TMeta]) factFingerprint() []byte         { return n.fingerprint }
func (n storedNode[TMeta]) factRef() source.Reference       { return n.record.Reference }
func (e storedEdge[TMeta]) factID() string                  { return e.record.Value.ID }
func (e storedEdge[TMeta]) factAttrs() filter.RawAttributes { return e.attributes }
func (e storedEdge[TMeta]) factFingerprint() []byte         { return e.fingerprint }
func (e storedEdge[TMeta]) factRef() source.Reference       { return e.record.Reference }
