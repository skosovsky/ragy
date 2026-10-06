// Package materialization projects resolved, source-supported assertions into an
// owned managed graph payload and a planned lifecycle manifest. It performs no writes.
package materialization

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"slices"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/graph"
	"github.com/skosovsky/ragy/graph/managed"
	"github.com/skosovsky/ragy/graphingest/resolution"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/source"
)

var (
	ErrConflictingAssertions  = errors.New("source revision has conflicting graph assertions")
	ErrUnresolvedAssertions   = errors.New("source revision has unresolved graph assertions")
	ErrMissingEndpointSupport = errors.New("source revision does not support relation endpoints")
)

type NodeValue[TMeta any] struct {
	Labels  []string
	Content string
	Meta    TMeta
}
type EdgeValue[TMeta any] struct {
	Type string
	Meta TMeta
}
type Config[TKind, TRel comparable, TAttr, TMeta any] struct {
	OntologyIdentity string
	PolicyIdentity   string
	Schema           graph.Schema
	MaxFacts         int
	MaxSupports      int
	CloneAttributes  func(TAttr) (TAttr, error)
	CloneMeta        func(TMeta) (TMeta, error)
	Node             func(resolution.Decision, TKind, TAttr) (NodeValue[TMeta], error)
	Edge             func(TRel, TAttr) (EdgeValue[TMeta], error)
	AdmitSupport     func(context.Context, access.Binding, source.Locator) error
}
type Request struct {
	Identity            lifecycle.Identity
	Target              string
	ManifestID          string
	Key                 string
	PayloadFingerprint  string
	ExpectedPublication string
}
type Result[TMeta any] struct {
	Manifest lifecycle.Manifest
	Payload  managed.Payload[TMeta]
}
type Materializer[TKind, TRel comparable, TAttr, TMeta any] struct {
	config Config[TKind, TRel, TAttr, TMeta]
}

func New[TKind, TRel comparable, TAttr, TMeta any](
	config Config[TKind, TRel, TAttr, TMeta],
) (*Materializer[TKind, TRel, TAttr, TMeta], error) {
	if config.OntologyIdentity == "" || config.PolicyIdentity == "" ||
		config.Schema.Validate() != nil ||
		config.MaxFacts <= 0 ||
		config.MaxSupports <= 0 ||
		config.CloneAttributes == nil ||
		config.CloneMeta == nil ||
		config.Node == nil ||
		config.Edge == nil ||
		config.AdmitSupport == nil {
		return nil, ragy.ErrInvalidArgument
	}
	return &Materializer[TKind, TRel, TAttr, TMeta]{config: config}, nil
}

// Build never guesses missing evidence or chooses a conflict winner. Host support
// admission governs ingestion of the requested source revision, which need not yet
// be published. Read freshness still gates every projection and final delivery.
func (m *Materializer[TKind, TRel, TAttr, TMeta]) Build(
	ctx context.Context,
	read access.Binding,
	request Request,
	input resolution.Result[TKind, TRel, TAttr],
) (Result[TMeta], error) {
	if m == nil || request.Identity.Validate() != nil || request.Target == "" ||
		request.ManifestID == "" ||
		request.Key == "" ||
		request.PayloadFingerprint == "" {
		return Result[TMeta]{}, ragy.ErrInvalidArgument
	}
	if err := read.Check(ctx); err != nil {
		return Result[TMeta]{}, err
	}
	if input.OntologyIdentity != m.config.OntologyIdentity ||
		input.PolicyIdentity != m.config.PolicyIdentity {
		return Result[TMeta]{}, ragy.ErrProtocol
	}
	if !boundedInput(input, m.config.MaxFacts, m.config.MaxSupports) {
		return Result[TMeta]{}, ragy.ErrInvalidArgument
	}
	for _, unresolved := range input.Unresolved {
		if len(selected(request.Identity, unresolved.Supports)) > 0 {
			return Result[TMeta]{}, ErrUnresolvedAssertions
		}
	}
	nodes, edges, err := m.selectFacts(ctx, read, request.Identity, input)
	if err != nil {
		return Result[TMeta]{}, err
	}
	out := Result[TMeta]{
		Manifest: manifest(request, input.OntologyIdentity, input.PolicyIdentity),
		Payload:  managed.Payload[TMeta]{Nodes: nil, Edges: nil},
	}
	for _, node := range nodes {
		if err = m.node(ctx, read, node, &out); err != nil {
			return Result[TMeta]{}, err
		}
	}
	for _, edge := range edges {
		if err = m.edge(ctx, read, edge, &out); err != nil {
			return Result[TMeta]{}, err
		}
	}
	if err = validateResult(m.config.Schema, out); err != nil {
		return Result[TMeta]{}, err
	}
	if err = read.Check(ctx); err != nil {
		return Result[TMeta]{}, err
	}
	return out, nil
}

type selectedEntity[TKind comparable, TAttr any] struct {
	group   resolution.EntityGroup[TKind, TAttr]
	variant resolution.Variant[TKind, TAttr]
}
type selectedRelation[TRel comparable, TAttr any] struct {
	group   resolution.RelationGroup[TRel, TAttr]
	variant resolution.Variant[TRel, TAttr]
}

func selected(identity lifecycle.Identity, input []source.Locator) []source.Locator {
	var out []source.Locator
	for _, location := range input {
		ref := location.Reference
		if ref.Namespace == identity.Namespace && ref.Source == identity.Source &&
			ref.Revision == identity.Revision &&
			ref.AccessFingerprint == identity.Access {
			out = append(out, location)
		}
	}
	return out
}

func selectVariant[TKind comparable, TAttr any](
	identity lifecycle.Identity,
	variants []resolution.Variant[TKind, TAttr],
) (resolution.Variant[TKind, TAttr], bool, error) {
	var out resolution.Variant[TKind, TAttr]
	found := false
	for _, variant := range variants {
		supports := selected(identity, variant.Supports)
		if len(supports) == 0 {
			continue
		}
		if found {
			return out, false, ErrConflictingAssertions
		}
		out, found = variant, true
		out.Supports = supports
	}
	return out, found, nil
}

func (m *Materializer[TKind, TRel, TAttr, TMeta]) selectFacts(
	ctx context.Context,
	read access.Binding,
	identity lifecycle.Identity,
	input resolution.Result[TKind, TRel, TAttr],
) ([]selectedEntity[TKind, TAttr], []selectedRelation[TRel, TAttr], error) {
	var nodes []selectedEntity[TKind, TAttr]
	ids := make(map[string]bool)
	var supports []source.Locator
	for _, group := range input.Entities {
		variant, found, err := selectVariant(identity, group.Variants)
		if err != nil {
			return nil, nil, err
		}
		if !found {
			continue
		}
		if group.ID == "" || ids[group.ID] || group.Identity.State != resolution.Resolved {
			return nil, nil, ragy.ErrProtocol
		}
		ids[group.ID] = true
		nodes = append(nodes, selectedEntity[TKind, TAttr]{group: group, variant: variant})
		supports = append(supports, variant.Supports...)
	}
	var edgeSupports []source.Locator
	edges, edgeSupports, err := selectEdges(identity, input.Relations, ids)
	if err != nil {
		return nil, nil, err
	}
	supports = append(supports, edgeSupports...)
	if len(supports) > m.config.MaxSupports {
		return nil, nil, ragy.ErrInvalidArgument
	}
	for _, support := range supports {
		if err := support.Validate(); err != nil {
			return nil, nil, err
		}
	}
	for _, support := range supports {
		if err := read.Check(ctx); err != nil {
			return nil, nil, err
		}
		if err := m.config.AdmitSupport(ctx, read, support); err != nil {
			return nil, nil, access.NonSkippable(err)
		}
		if err := read.Check(ctx); err != nil {
			return nil, nil, err
		}
	}
	return nodes, edges, nil
}
func manifest(request Request, ontology, policy string) lifecycle.Manifest {
	parts, _ := json.Marshal([]string{request.Identity.Transformation, ontology, policy})
	digest := sha256.Sum256(parts)
	identity := request.Identity
	identity.Transformation = "graph-resolution:" + hex.EncodeToString(digest[:])
	return lifecycle.Manifest{
		Retired:             false,
		ArtifactFences:      nil,
		ID:                  request.ManifestID,
		Identity:            identity,
		Key:                 request.Key,
		Payload:             request.PayloadFingerprint,
		ExpectedPublication: request.ExpectedPublication,
		State:               lifecycle.Planned,
		Tombstone:           false,
		Partial:             false,
		Checkpoint:          "",
		PublishedAt:         time.Time{},
		Targets: []lifecycle.Target{
			{
				Name:      request.Target,
				Required:  true,
				State:     lifecycle.TargetPending,
				Revision:  "",
				Artifacts: nil,
			},
		},
	}
}

func artifact(
	identity lifecycle.Identity,
	id, representation string,
	supports []source.Locator,
) lifecycle.Artifact {
	ref := source.Reference{
		Namespace:         identity.Namespace,
		Source:            identity.Source,
		Revision:          identity.Revision,
		Transformation:    identity.Transformation,
		AccessFingerprint: identity.Access,
		Artifact:          id,
		Representation:    representation,
	}
	var refs []source.Reference
	for _, support := range supports {
		if !slices.Contains(refs, support.Reference) {
			refs = append(refs, support.Reference)
		}
	}
	return lifecycle.Artifact{Reference: ref, Supports: refs}
}

func (m *Materializer[TKind, TRel, TAttr, TMeta]) node(
	ctx context.Context,
	read access.Binding,
	selected selectedEntity[TKind, TAttr],
	out *Result[TMeta],
) error {
	if err := read.Check(ctx); err != nil {
		return err
	}
	attrs, err := m.config.CloneAttributes(selected.variant.Attributes)
	if err != nil {
		return err
	}
	if err = read.Check(ctx); err != nil {
		return err
	}
	value, err := m.config.Node(selected.group.Identity, selected.variant.Kind, attrs)
	if err != nil {
		return err
	}
	if err = read.Check(ctx); err != nil {
		return err
	}
	meta, err := m.config.CloneMeta(value.Meta)
	if err != nil {
		return err
	}
	if err = read.Check(ctx); err != nil {
		return err
	}
	entry := artifact(
		out.Manifest.Identity,
		selected.group.ID,
		"graph-node",
		selected.variant.Supports,
	)
	out.Manifest.Targets[0].Artifacts = append(out.Manifest.Targets[0].Artifacts, entry)
	out.Payload.Nodes = append(
		out.Payload.Nodes,
		managed.Node[TMeta]{
			Reference: entry.Reference,
			Value: graph.Node[TMeta]{
				ID:      selected.group.ID,
				Labels:  slices.Clone(value.Labels),
				Content: value.Content,
				Meta:    meta,
			},
		},
	)
	return nil
}

func (m *Materializer[TKind, TRel, TAttr, TMeta]) edge(
	ctx context.Context,
	read access.Binding,
	selected selectedRelation[TRel, TAttr],
	out *Result[TMeta],
) error {
	if err := read.Check(ctx); err != nil {
		return err
	}
	attrs, err := m.config.CloneAttributes(selected.variant.Attributes)
	if err != nil {
		return err
	}
	if err = read.Check(ctx); err != nil {
		return err
	}
	value, err := m.config.Edge(selected.variant.Kind, attrs)
	if err != nil {
		return err
	}
	if err = read.Check(ctx); err != nil {
		return err
	}
	meta, err := m.config.CloneMeta(value.Meta)
	if err != nil {
		return err
	}
	if err = read.Check(ctx); err != nil {
		return err
	}
	entry := artifact(
		out.Manifest.Identity,
		selected.group.ID,
		"graph-edge",
		selected.variant.Supports,
	)
	out.Manifest.Targets[0].Artifacts = append(out.Manifest.Targets[0].Artifacts, entry)
	out.Payload.Edges = append(
		out.Payload.Edges,
		managed.Edge[TMeta]{
			Reference: entry.Reference,
			Value: graph.Edge[TMeta]{
				ID:       selected.group.ID,
				SourceID: selected.group.From,
				TargetID: selected.group.To,
				Type:     value.Type,
				Meta:     meta,
			},
		},
	)
	return nil
}

func boundedInput[TKind, TRel comparable, TAttr any](
	input resolution.Result[TKind, TRel, TAttr],
	factsLimit, supportLimit int,
) bool {
	for _, entity := range input.Entities {
		if !countVariants(entity.Variants, &factsLimit, &supportLimit) {
			return false
		}
	}
	for _, edge := range input.Relations {
		if !countVariants(edge.Variants, &factsLimit, &supportLimit) {
			return false
		}
	}
	if len(input.Unresolved) > factsLimit {
		return false
	}
	for _, unresolved := range input.Unresolved {
		if len(unresolved.Supports) > supportLimit {
			return false
		}
		supportLimit -= len(unresolved.Supports)
	}
	return true
}

func countVariants[TKind comparable, TAttr any](
	variants []resolution.Variant[TKind, TAttr],
	facts, supports *int,
) bool {
	if len(variants) == 0 || len(variants) > *facts {
		return false
	}
	*facts -= len(variants)
	for _, variant := range variants {
		if len(variant.Supports) == 0 || len(variant.Supports) > *supports {
			return false
		}
		*supports -= len(variant.Supports)
	}
	return true
}

func validateResult[TMeta any](schema graph.Schema, out Result[TMeta]) error {
	snapshot := graph.Snapshot[TMeta]{Nodes: nil, Edges: nil}
	for _, node := range out.Payload.Nodes {
		snapshot.Nodes = append(snapshot.Nodes, node.Value)
	}
	for _, edge := range out.Payload.Edges {
		snapshot.Edges = append(snapshot.Edges, edge.Value)
	}
	if _, err := graph.NormalizeSnapshot(schema, snapshot); err != nil {
		return err
	}
	if err := out.Manifest.Validate(); err != nil {
		return err
	}
	return nil
}

func selectEdges[TRel comparable, TAttr any](
	identity lifecycle.Identity,
	groups []resolution.RelationGroup[TRel, TAttr],
	ids map[string]bool,
) ([]selectedRelation[TRel, TAttr], []source.Locator, error) {
	var edges []selectedRelation[TRel, TAttr]
	var supports []source.Locator
	edgeIDs := make(map[string]bool)
	for _, group := range groups {
		variant, found, err := selectVariant(identity, group.Variants)
		if err != nil {
			return nil, nil, err
		}
		if !found {
			continue
		}
		if group.ID == "" || edgeIDs[group.ID] {
			return nil, nil, ragy.ErrProtocol
		}
		if !ids[group.From] || !ids[group.To] {
			return nil, nil, ErrMissingEndpointSupport
		}
		edgeIDs[group.ID] = true
		edges = append(edges, selectedRelation[TRel, TAttr]{group: group, variant: variant})
		supports = append(supports, variant.Supports...)
	}
	return edges, supports, nil
}
