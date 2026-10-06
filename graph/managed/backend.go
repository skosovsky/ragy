package managed

import (
	"context"
	"errors"
	"math"
	"slices"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/graph"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

var ErrConflictingFacts = errors.New("graph conflicting facts require an explicit projection policy")

type FactKind string

const (
	NodeFact FactKind = "node"
	EdgeFact FactKind = "edge"
)

// FactIdentity names an admitted graph fact independently of display/document ID.
type FactIdentity struct {
	Kind FactKind
	ID   string
}

// Projection declares the complete facts used by a host document projection.
// Their admitted original source supports are attached by the backend.
type Projection[TMeta any] struct {
	Document retrieval.Document[TMeta]
	Facts    []FactIdentity
}

type BackendConfig[TMeta any] struct {
	Adapter   *Adapter[TMeta]
	HostBasis string
	MaxNodes  int
	MaxEdges  int
	Project   func(Result[TMeta]) ([]Projection[TMeta], error)
}

// Backend projects admitted graph evidence into ResultSet. Default node projection
// is score-absent; no synthetic similarity is assigned to graph reachability.
type Backend[TMeta any] struct{ config BackendConfig[TMeta] }

func NewBackend[TMeta any](config BackendConfig[TMeta]) (*Backend[TMeta], error) {
	if config.Adapter == nil || config.MaxNodes <= 0 || config.MaxEdges <= 0 ||
		config.MaxNodes > math.MaxInt-config.MaxEdges {
		return nil, ragy.ErrInvalidArgument
	}
	if config.Project == nil {
		config.Project = projectNodes[TMeta]
	}
	return &Backend[TMeta]{config: config}, nil
}
func (b *Backend[TMeta]) Schema() filter.Schema {
	if b == nil || b.config.Adapter == nil {
		return filter.Schema{}
	}
	return b.config.Adapter.Schema().NodeAttributes
}
func (*Backend[TMeta]) ReadCapabilities() access.Capabilities {
	return access.Capabilities{ScopeProfile: true, PinnedPublication: true, RequirePinnedPublication: true}
}
func (b *Backend[TMeta]) AdmitRead(ctx context.Context, req retrieval.Query[struct{}]) (retrieval.ReadCoverage, error) {
	request, err := b.request(ctx, req)
	if err != nil {
		return retrieval.UnobservedReadCoverage(), err
	}
	if _, err = b.config.Adapter.prepare(ctx, request); err != nil {
		return retrieval.UnobservedReadCoverage(), err
	}
	return retrieval.CompleteReadCoverage(), nil
}

func (b *Backend[TMeta]) Retrieve(
	ctx context.Context,
	req retrieval.Query[struct{}],
) (retrieval.ResultSet[TMeta], error) {
	empty := retrieval.NewResultSet[TMeta](nil, nil)
	result, err := b.retrieve(ctx, retrieval.CopyRequestOptions(req))
	if err != nil {
		return retrieval.DeliverRead(ctx, req.Read, empty, access.NonSkippable(err), nil)
	}
	return retrieval.DeliverRead(ctx, req.Read, result, nil, nil)
}
func (b *Backend[TMeta]) request(ctx context.Context, req retrieval.Query[struct{}]) (Request, error) {
	if b == nil || b.config.Adapter == nil || req.Options.Graph == nil {
		return Request{}, ragy.ErrInvalidArgument
	}
	if err := req.Read.Check(ctx); err != nil {
		return Request{}, err
	}
	if err := req.Options.Validate(); err != nil {
		return Request{}, err
	}
	if len(req.Options.Vector) != 0 || req.Options.Threshold != nil {
		return Request{}, ragy.ErrUnsupported
	}
	conditions := []filter.Condition{req.Options.Filters, req.Options.Graph.NodeFilter}
	if req.Plan != nil {
		conditions = append(conditions, req.Plan.Filters)
	}
	nodes, err := filter.Intersect(b.Schema(), conditions...)
	if err != nil {
		return Request{}, access.UnsupportedCapability(err)
	}
	options := req.Options.Graph
	return Request{
		Read:      req.Read,
		HostBasis: b.config.HostBasis,
		Traversal: graph.TraversalRequest{
			Seeds:      slices.Clone(options.Seeds),
			Direction:  options.Direction,
			Depth:      options.Depth,
			NodeFilter: nodes,
			EdgeFilter: options.EdgeFilter,
			Page:       options.Page,
		},
		MaxNodes: b.config.MaxNodes,
		MaxEdges: b.config.MaxEdges,
	}, nil
}

func (b *Backend[TMeta]) retrieve(
	ctx context.Context,
	req retrieval.Query[struct{}],
) (retrieval.ResultSet[TMeta], error) {
	request, err := b.request(ctx, req)
	if err != nil {
		return nil, err
	}
	mandatory, err := req.Read.Prepare(ctx, b.Schema(), filter.Condition{}, b.ReadCapabilities())
	if err != nil {
		return nil, err
	}
	result, err := b.config.Adapter.Traverse(ctx, request)
	if err != nil {
		return nil, err
	}
	if err = req.Read.Check(ctx); err != nil {
		return nil, err
	}
	inventory, err := captureFactSupports(result)
	if err != nil {
		return nil, err
	}
	docs, err := b.config.Project(result)
	if err != nil {
		return nil, err
	}
	if err = req.Read.Check(ctx); err != nil {
		return nil, err
	}
	if len(docs) > b.config.MaxNodes+b.config.MaxEdges {
		return nil, ragy.ErrProtocol
	}
	owned := make([]retrieval.Document[TMeta], len(docs))
	for i, projection := range docs {
		if err = req.Read.Check(ctx); err != nil {
			return nil, err
		}
		owned[i], err = bindProjection(projection, inventory)
		if err != nil {
			return nil, err
		}
		owned[i], err = b.snapshotDocument(ctx, req.Read, mandatory, owned[i])
		if err != nil {
			return nil, err
		}
	}
	// Preserve projector order for rank-only graph evidence.
	limit := req.Options.TopK
	if limit == 0 {
		limit = req.Options.FetchLimit
	}
	if len(owned) > limit {
		owned = owned[:limit]
	}
	for i := range owned {
		owned[i].Rank = i + 1
	}
	return retrieval.NewResultSet(owned, nil), nil
}
func projectNodes[TMeta any](result Result[TMeta]) ([]Projection[TMeta], error) {
	if len(result.Conflicts) != 0 {
		return nil, ErrConflictingFacts
	}
	docs := make([]Projection[TMeta], 0, len(result.Snapshot.Nodes))
	for _, node := range result.Snapshot.Nodes {
		docs = append(
			docs,
			Projection[TMeta]{
				Document: retrieval.Document[TMeta]{ID: node.ID, Content: node.Content, Meta: node.Meta},
				Facts:    []FactIdentity{{Kind: NodeFact, ID: node.ID}},
			},
		)
	}
	return docs, nil
}

func captureFactSupports[TMeta any](result Result[TMeta]) (map[FactIdentity][]source.Reference, error) {
	facts := make(map[FactIdentity][]source.Reference)
	for _, node := range result.Snapshot.Nodes {
		facts[FactIdentity{Kind: NodeFact, ID: node.ID}] = nil
	}
	for _, edge := range result.Snapshot.Edges {
		facts[FactIdentity{Kind: EdgeFact, ID: edge.ID}] = nil
	}
	seen := make(map[FactIdentity]bool)
	for _, support := range result.Supports {
		fact := FactIdentity{Kind: FactKind(support.Kind), ID: support.ID}
		if _, exists := facts[fact]; !exists || seen[fact] {
			return nil, ragy.ErrProtocol
		}
		for _, ref := range support.References {
			if ref.Validate() != nil {
				return nil, ragy.ErrProtocol
			}
		}
		if len(support.References) == 0 && len(support.HostBases) == 0 {
			return nil, ragy.ErrProtocol
		}
		seen[fact] = true
		facts[fact] = slices.Clone(support.References)
	}
	if len(seen) != len(facts) {
		return nil, ragy.ErrProtocol
	}
	return facts, nil
}

func bindProjection[TMeta any](
	projection Projection[TMeta],
	inventory map[FactIdentity][]source.Reference,
) (retrieval.Document[TMeta], error) {
	doc := projection.Document
	if len(projection.Facts) == 0 || retrieval.ValidateDocument(doc) != nil {
		return retrieval.Document[TMeta]{}, ragy.ErrProtocol
	}
	var refs []source.Reference
	for _, fact := range projection.Facts {
		supports, exists := inventory[fact]
		if !exists {
			return retrieval.Document[TMeta]{}, access.NonSkippable(ragy.ErrUnavailable)
		}
		refs = append(refs, supports...)
	}
	refs = unique(refs)
	for _, location := range doc.SourceLocations() {
		if !slices.Contains(refs, location.Reference) {
			return retrieval.Document[TMeta]{}, access.NonSkippable(ragy.ErrUnavailable)
		}
	}
	doc.SourceSupports = doc.SourceLocations()
	for _, ref := range refs {
		var location source.Locator
		location.Reference, location.Kind = ref, source.DocumentLocation
		if !slices.Contains(doc.SourceSupports, location) {
			doc.SourceSupports = append(doc.SourceSupports, location)
		}
	}
	return doc, nil
}

func (b *Backend[TMeta]) snapshotDocument(
	ctx context.Context,
	read access.Binding,
	mandatory filter.Condition,
	doc retrieval.Document[TMeta],
) (retrieval.Document[TMeta], error) {
	if read.IsScoped() {
		allowed, err := retrieval.MatchDocument(b.config.Adapter.config.NodeCodec, doc, mandatory)
		if err != nil {
			return retrieval.Document[TMeta]{}, err
		}
		if err = read.Check(ctx); err != nil {
			return retrieval.Document[TMeta]{}, err
		}
		if !allowed {
			return retrieval.Document[TMeta]{}, access.NonSkippable(ragy.ErrUnavailable)
		}
	}
	var err error
	doc.Meta, err = b.config.Adapter.config.CloneMeta(doc.Meta)
	if err != nil {
		return retrieval.Document[TMeta]{}, err
	}
	doc.ScoreHistory = slices.Clone(doc.ScoreHistory)
	doc.SourceSupports = slices.Clone(doc.SourceSupports)
	if err = read.Check(ctx); err != nil {
		return retrieval.Document[TMeta]{}, err
	}
	if err = retrieval.ValidateDocument(doc); err != nil {
		return retrieval.Document[TMeta]{}, err
	}
	return doc, nil
}

// AdmitPublication rejects an excluded graph branch before projection or traversal.
func (b *Backend[TMeta]) AdmitPublication(publication access.Publication) error {
	if b == nil || b.config.Adapter == nil {
		return access.Protect(ragy.ErrInvalidArgument)
	}
	return publication.AdmitTarget(b.config.Adapter.config.Target)
}
