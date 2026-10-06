//go:build darwin || linux

package main

import (
	"context"
	"slices"
	"sync/atomic"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/graph"
	"github.com/skosovsky/ragy/graph/managed"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/recipe/graphsummary"
	"github.com/skosovsky/ragy/source"
)

type summarySourceHost struct {
	store         lifecycle.Store
	fixture       fixture
	metadataCalls atomic.Uint64
	payloadCalls  atomic.Uint64
}

func (h *summarySourceHost) liveRows(ctx context.Context, req source.LookupRequest) ([]sourceRow, error) {
	snapshot, err := h.store.Load(ctx, "n")
	if err != nil {
		return nil, err
	}
	var rows []sourceRow
	for _, ref := range req.References {
		active := originalIsActive(snapshot, ref)
		if !active {
			return nil, ragy.ErrUnavailable
		}
		found := false
		for _, row := range h.fixture.Sources {
			if originalReference(row.ID) == ref {
				rows = append(rows, row)
				found = true
				break
			}
		}
		if !found {
			return nil, ragy.ErrUnavailable
		}
	}
	return rows, nil
}

func originalIsActive(snapshot lifecycle.Snapshot, ref source.Reference) bool {
	for _, pub := range snapshot.Publications {
		if pub.Source != ref.Source {
			continue
		}
		for _, manifest := range snapshot.Manifests {
			if manifest.ID == pub.Manifest && !manifest.Tombstone && manifest.Identity.Revision == ref.Revision &&
				manifest.Identity.Access == ref.AccessFingerprint && manifest.Identity.Namespace == ref.Namespace &&
				manifest.Identity.Source == ref.Source {
				return true
			}
		}
	}
	return false
}

func (h *summarySourceHost) Describe(
	ctx context.Context,
	req source.LookupRequest,
) ([]source.Descriptor[baselineMetadata], error) {
	h.metadataCalls.Add(1)
	rows, err := h.liveRows(ctx, req)
	if err != nil {
		return nil, err
	}
	var descriptors []source.Descriptor[baselineMetadata]
	for _, row := range rows {
		descriptors = append(
			descriptors,
			source.Descriptor[baselineMetadata]{
				Reference: originalReference(row.ID),
				Access:    baselineMetadata{Tenant: "a", SourceID: row.ID},
			},
		)
	}
	return descriptors, nil
}
func (h *summarySourceHost) Load(ctx context.Context, req source.LookupRequest) ([]source.Materialized[string], error) {
	h.payloadCalls.Add(1)
	rows, err := h.liveRows(ctx, req)
	if err != nil {
		return nil, err
	}
	var values []source.Materialized[string]
	for _, row := range rows {
		values = append(values, source.Materialized[string]{Reference: originalReference(row.ID), Payload: row.Text})
	}
	if err = req.Read.Check(ctx); err != nil {
		return nil, err
	}
	return values, nil
}

type summarySources struct {
	host                  *summarySourceHost
	reader                *source.Reader[baselineMetadata, string]
	schema                filter.Schema
	membership            map[string][]string
	memberSupports        map[string][]source.Reference
	binding               string
	preparationGraphCalls uint64
	fixture               fixture
}

func (c graphCorpus) summarySources(
	ctx context.Context,
	read access.Binding,
	f fixture,
	schema filter.Schema,
) (summarySources, error) {
	host := &summarySourceHost{store: c.store, fixture: f}
	reader, err := newSummaryReader(host, f, schema)
	if err != nil {
		return summarySources{}, err
	}
	fingerprint, err := read.Fingerprint()
	if err != nil {
		return summarySources{}, err
	}
	out := summarySources{
		host:       host,
		reader:     reader,
		schema:     schema,
		membership: make(map[string][]string), memberSupports: make(map[string][]source.Reference),
		binding: fingerprint,
		fixture: f,
	}
	for _, declared := range f.Communities {
		ids, result, lookupErr := c.communityMembership(ctx, read, declared)
		out.preparationGraphCalls++
		if lookupErr != nil {
			return summarySources{}, lookupErr
		}
		out.membership[declared.ID] = ids
		for _, support := range result.Supports {
			if support.Kind == "node" {
				out.memberSupports[support.ID] = slices.Clone(support.References)
			}
		}
	}
	return out, nil
}

func newSummaryReader(
	host *summarySourceHost,
	f fixture,
	schema filter.Schema,
) (*source.Reader[baselineMetadata, string], error) {
	return source.NewReader(
		source.ReadConfig[baselineMetadata, string]{Target: lexicalTarget, Schema: schema, Catalog: host, Loader: host,
			Attributes: summaryAttributes,
			ValidatePayload: func(ref source.Reference, text string) error {
				for _, row := range f.Sources {
					if ref == originalReference(row.ID) && text == row.Text {
						return nil
					}
				}
				return ragy.ErrProtocol
			},
			ClonePayload: func(text string) (string, error) { return text, nil }},
	)
}
func summaryAttributes(m baselineMetadata) (filter.RawAttributes, error) {
	return filter.RawAttributes{"tenant": m.Tenant, "source_key": m.SourceID}, nil
}
func (s summarySources) freshSources() (summarySources, error) {
	s.host = &summarySourceHost{store: s.host.store, fixture: s.fixture}
	var err error
	s.reader, err = newSummaryReader(s.host, s.fixture, s.schema)
	return s, err
}
func (c graphCorpus) communityMembership(
	ctx context.Context,
	read access.Binding,
	declared community,
) ([]string, managed.Result[graphMetadata], error) {
	ids, err := c.communityIDs(declared.Members)
	if err != nil {
		return nil, managed.Result[graphMetadata]{}, err
	}
	result, err := c.adapter.FindByIDs(
		ctx,
		managed.Request{
			Read:      read,
			Traversal: graph.TraversalRequest{Seeds: ids, Direction: graph.DirectionUndirected, Depth: 1},
			MaxNodes:  localNodeCap,
			MaxEdges:  localEdgeCap,
		},
	)
	if err != nil {
		return nil, result, err
	}
	if !exactMembershipNodes(ids, result) {
		return nil, result, ragy.ErrUnavailable
	}
	return ids, result, nil
}
func exactMembershipNodes(ids []string, result managed.Result[graphMetadata]) bool {
	if len(result.Snapshot.Nodes) != len(ids) || len(result.Conflicts) != 0 {
		return false
	}
	seen := make(map[string]bool)
	for _, node := range result.Snapshot.Nodes {
		if !slices.Contains(ids, node.ID) || seen[node.ID] {
			return false
		}
		seen[node.ID] = true
	}
	return true
}
func (c graphCorpus) communityIDs(keys []string) ([]string, error) {
	var ids []string
	for _, key := range keys {
		found := false
		for _, entity := range c.resolved.Entities {
			if entity.Identity.Namespace+"/"+entity.Identity.Key == key {
				ids = append(ids, entity.ID)
				found = true
				break
			}
		}
		if !found {
			return nil, ragy.ErrUnavailable
		}
	}
	return ids, nil
}
func (s summarySources) admitMembership(ctx context.Context, read access.Binding, id string, members []string) error {
	if err := read.Check(ctx); err != nil {
		return err
	}
	fingerprint, err := read.Fingerprint()
	if err != nil {
		return err
	}
	expected, exists := s.membership[id]
	if fingerprint != s.binding || !exists || !slices.Equal(expected, members) {
		return ragy.ErrUnavailable
	}
	return nil
}
func (s summarySources) admitSource(ctx context.Context, read access.Binding, loc source.Locator) error {
	if err := originalAdmission(s.fixture)(ctx, read, loc); err != nil {
		return err
	}
	values, err := s.reader.Lookup(ctx, source.LookupRequest{Read: read, References: []source.Reference{loc.Reference}})
	if err != nil {
		return err
	}
	if len(values) != 1 {
		return ragy.ErrUnavailable
	}
	mapping, err := source.OriginalText(loc, values[0].Payload)
	if err != nil {
		return err
	}

	if mapping.Text() != values[0].Payload {
		return ragy.ErrProtocol
	}
	return nil
}

func (s summarySources) communities(
	ctx context.Context,
	read access.Binding,
	global bool,
) ([]graphsummary.Community[baselineMetadata], error) {
	declared := s.fixture.Communities
	if !global {
		declared = declared[:1]
	}
	var result []graphsummary.Community[baselineMetadata]
	for _, item := range declared {
		community := graphsummary.Community[baselineMetadata]{ID: item.ID, Members: slices.Clone(s.membership[item.ID])}
		if err := s.admitMembership(ctx, read, item.ID, community.Members); err != nil {
			return nil, err
		}
		for _, id := range item.Sources {
			snippet, err := s.communitySnippet(ctx, read, item.ID, id)
			if err != nil {
				return nil, err
			}
			community.Snippets = append(community.Snippets, snippet)
		}
		result = append(result, community)
	}
	return result, nil
}

func (s summarySources) communitySnippet(
	ctx context.Context,
	read access.Binding,
	communityID, id string,
) (graphsummary.Snippet[baselineMetadata], error) {
	var empty graphsummary.Snippet[baselineMetadata]
	rows, err := s.reader.Lookup(
		ctx,
		source.LookupRequest{Read: read, References: []source.Reference{originalReference(id)}},
	)
	if err != nil {
		return empty, err
	}
	if len(rows) != 1 {
		return empty, ragy.ErrUnavailable
	}
	var rowValue sourceRow
	for _, row := range s.fixture.Sources {
		if row.ID == id {
			rowValue = row
		}
	}
	mapping, err := mappedSource(rowValue)
	if err != nil || rows[0].Payload != mapping.Text() {
		return empty, ragy.ErrProtocol
	}
	return graphsummary.Snippet[baselineMetadata]{
		Mapping: mapping,
		Access:  baselineMetadata{Tenant: "a", SourceID: id},
		Members: s.sourceMembers(communityID, id),
	}, nil
}
func (s summarySources) sourceMembers(communityID, id string) []string {
	var result []string
	for _, member := range s.membership[communityID] {
		if slices.Contains(s.memberSupports[member], originalReference(id)) {
			result = append(result, member)
		}
	}
	return result
}
