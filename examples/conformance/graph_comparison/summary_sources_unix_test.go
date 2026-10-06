//go:build darwin || linux

package main

import (
	"slices"
	"testing"
	"time"

	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/graph"
	"github.com/skosovsky/ragy/graph/managed"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/source"
)

func TestSummarySourcePreparationUsesActualGraphSupportsAndScopedPayloads(t *testing.T) {
	// Arrange: actual reference publication; no expected gold supports are used by the producer.
	f, graph, baseline, read := publishedLocalFixture(t)
	prepared, err := graph.summarySources(t.Context(), read, f, baseline.lexical.Schema())
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	communities, err := prepared.communities(t.Context(), read, true)
	// Assert: C1/C2, real membership support and original texts, with preparation I/O separate.
	if err != nil || prepared.preparationGraphCalls != 2 || len(communities) != 2 ||
		len(
			communities[0].Snippets,
		) != 2 || len(communities[1].Snippets) != 1 || prepared.host.payloadCalls.Load() != 3 {
		t.Fatal(communities, err)
	}
	first, second := communities[0].Snippets[0], communities[0].Snippets[1]
	if len(first.Members) != 3 || len(second.Members) != 2 ||
		first.Mapping.Supports()[0].Reference != originalReference("s1") ||
		second.Mapping.Supports()[0].Reference != originalReference("s2") ||
		first.Mapping.Text() != f.Sources[0].Text {
		t.Fatal(first, second)
	}
	for _, member := range second.Members {
		if !slices.Contains(communities[0].Members, member) {
			t.Fatal("foreign member", member)
		}
	}
}
func TestSummarySourceLocatorAndMembershipPoisoningRejectBeforePayload(t *testing.T) {
	// Arrange.
	f, graph, baseline, read := publishedLocalFixture(t)
	prepared, err := graph.summarySources(t.Context(), read, f, baseline.lexical.Schema())
	if err != nil {
		t.Fatal(err)
	}
	mapping, err := mappedSource(f.Sources[0])
	if err != nil {
		t.Fatal(err)
	}
	loc := mapping.Supports()[0]
	loc.Span.Start++
	// Act.
	locErr := prepared.admitSource(t.Context(), read, loc)
	memberErr := prepared.admitMembership(t.Context(), read, "C1", []string{"unrelated-canonical-node"})
	// Assert.
	if locErr == nil || memberErr == nil || prepared.host.payloadCalls.Load() != 0 ||
		prepared.host.metadataCalls.Load() != 0 {
		t.Fatal(locErr, memberErr)
	}
}
func TestSummaryLiveSourceAdmissionRejectsActualTombstone(t *testing.T) {
	// Arrange: payload remains in the fixed host corpus after source retirement.
	f, graph, baseline, read := publishedLocalFixture(t)
	prepared, err := graph.summarySources(t.Context(), read, f, baseline.lexical.Schema())
	if err != nil {
		t.Fatal(err)
	}
	mapping, err := mappedSource(f.Sources[0])
	if err != nil {
		t.Fatal(err)
	}
	loc := mapping.Supports()[0]
	if err = prepared.admitSource(t.Context(), read, loc); err != nil {
		t.Fatal(err)
	}
	before := prepared.host.payloadCalls.Load()
	retireSummaryGraphSource(t, graph, "s1")
	// Act.
	err = prepared.admitSource(t.Context(), read, loc)
	// Assert: current tombstone denies before another original payload load.
	if !access.IsProtectionFailure(err) || prepared.host.payloadCalls.Load() != before {
		t.Fatal(err)
	}
}
func retireSummaryGraphSource(t *testing.T, graph graphCorpus, src string) {
	t.Helper()
	snapshot, err := graph.store.Load(t.Context(), "n")
	if err != nil {
		t.Fatal(err)
	}
	var identity lifecycle.Identity
	for _, manifest := range snapshot.Manifests {
		if manifest.ID == src {
			identity = manifest.Identity
		}
	}
	identity.Revision = "deleted"
	executor, err := lifecycle.NewExecutor(lifecycle.ExecutorConfig[managed.Payload[graphMetadata]]{
		Store: graph.store,
		Targets: []lifecycle.Registration[managed.Payload[graphMetadata]]{
			{Name: graphTarget, Port: graph.adapter},
		},
		ClonePayload:    cloneGraphPayload,
		ValidatePayload: func(lifecycle.Manifest, managed.Payload[graphMetadata]) error { return nil },
		Now:             time.Now,
	})
	if err != nil {
		t.Fatal(err)
	}
	tombstone := lifecycle.Manifest{
		ID:                  "delete-" + src,
		Key:                 "delete-" + src,
		Payload:             "delete-" + src,
		ExpectedPublication: src,
		Identity:            identity,
		Tombstone:           true,
		State:               lifecycle.Planned,
	}
	if _, err = executor.Prepare(t.Context(), tombstone); err != nil {
		t.Fatal(err)
	}
	if _, err = executor.Publish(t.Context(), "n", tombstone.ID); err != nil {
		t.Fatal(err)
	}
}

var _ source.Catalog[baselineMetadata] = (*summarySourceHost)(nil)

func TestSummaryMembershipValidationRequiresExactNodes(t *testing.T) {
	// Arrange: equal counts alone cannot certify actual host membership.
	ids := []string{"a", "b"}
	for _, actual := range [][]string{{"a", "foreign"}, {"a", "a"}, {"a"}} {
		var nodes []graph.Node[graphMetadata]
		for _, id := range actual {
			nodes = append(nodes, graph.Node[graphMetadata]{ID: id})
		}
		// Act/Assert.
		if exactMembershipNodes(
			ids,
			managed.Result[graphMetadata]{Snapshot: graph.Snapshot[graphMetadata]{Nodes: nodes}},
		) {
			t.Fatal("incorrect membership", actual)
		}
	}
	if !exactMembershipNodes(
		ids,
		managed.Result[graphMetadata]{
			Snapshot: graph.Snapshot[graphMetadata]{Nodes: []graph.Node[graphMetadata]{{ID: "b"}, {ID: "a"}}},
		},
	) {
		t.Fatal("actual exact membership rejected")
	}
}
