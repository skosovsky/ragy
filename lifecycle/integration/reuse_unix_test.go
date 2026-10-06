//go:build darwin || linux

package integration_test

import (
	"testing"

	"github.com/skosovsky/ragy/graph"
	graphmanaged "github.com/skosovsky/ragy/graph/managed"
	"github.com/skosovsky/ragy/lexical"
	lexicalmanaged "github.com/skosovsky/ragy/lexical/managed"
	"github.com/skosovsky/ragy/lifecycle"
)

func TestActualJointReuseRequiresBackendCompletenessAndACLIdentity(t *testing.T) {
	for _, target := range []string{"lexical", "tensor", "graph"} {
		t.Run(target, func(t *testing.T) { jointReuseCase(t, target) })
	}
}
func jointReuseCase(t *testing.T, target string) {
	t.Helper()
	// Arrange: actual joint targets and durable publication, with no model calls.
	f := newFixture(t, target)
	input := sourceBatch("policy", "r1", []string{"p1", "p2"})
	manifest := plan("policy-r1", "", target, input)
	f.ingest(t, manifest, input)
	before, err := f.store.Load(t.Context(), "fixture-a")
	if err != nil {
		t.Fatal(err)
	}
	stages := f.secondary.calls
	// Act: exact identity/profile must inspect actual persistent/volatile inventories.
	decision, err := f.executor.CheckReuse(t.Context(), manifest.Identity, []string{"dense", target})
	// Assert: no write or stage while confirming physical readiness.
	if err != nil || !decision.CanSkip() || decision.Publication != manifest.ID {
		t.Fatal("actual ready target not confirmed", decision, err)
	}
	aclOnly := manifest.Identity
	aclOnly.Access = "changed-acl"
	changed, err := f.executor.CheckReuse(t.Context(), aclOnly, []string{"dense", target})
	if err != nil || changed.CanSkip() || changed.Reason != lifecycle.ReuseChanged {
		t.Fatal("ACL-only change skipped", changed, err)
	}
	after, err := f.store.Load(t.Context(), "fixture-a")
	if err != nil || after.Generation != before.Generation || f.secondary.calls != stages {
		t.Fatal("reuse inspection mutated targets", err)
	}
	// Act: same content/revision with changed access identity is explicitly staged/published.
	updated, cloneErr := cloneBatch(input)
	if cloneErr != nil {
		t.Fatal(cloneErr)
	}
	setBatchAccess(&updated, aclOnly.Access)
	aclPlan := plan("policy-acl", manifest.ID, target, updated)
	aclPlan.Identity.Content = manifest.Identity.Content
	f.ingest(t, aclPlan, updated)
	aclDecision, checkErr := f.executor.CheckReuse(t.Context(), aclPlan.Identity, []string{"dense", target})
	// Assert: fresh complete targets can be reused only under the new access identity.
	if checkErr != nil || !aclDecision.CanSkip() || aclDecision.Publication != aclPlan.ID {
		t.Fatal("ACL-only update not materialized", aclDecision, checkErr)
	}
	manifest = aclPlan
	if target == "tensor" {
		return
	}
	// Arrange: volatile backend restarts while its durable manifest still says ready.
	if target == "lexical" {
		f.secondary.lexical, err = lexicalmanaged.New(lexicalmanaged.Config[meta]{
			Namespace: "fixture-a", Target: target, Store: f.store, Schema: f.schema,
			BM25: lexical.Config[meta]{SearchFields: []string{"content"}}, CloneMeta: cloneMeta,
		})
	} else {
		f.secondary.graph, err = graphmanaged.New(graphmanaged.Config[meta]{
			Namespace: "fixture-a",
			Target:    target,
			Store:     f.store,
			Schema: graph.Schema{
				NodeAttributes: f.schema,
				EdgeAttributes: f.schema,
			},
			CloneMeta:  cloneMeta,
			MaxRecords: 100,
		})
	}
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	missing, err := f.executor.CheckReuse(t.Context(), manifest.Identity, []string{"dense", target})
	// Assert: matching ledger fingerprints cannot stand in for lost physical data.
	if err != nil || missing.CanSkip() || missing.Reason != lifecycle.ReuseIncomplete {
		t.Fatal("lost target skipped", missing, err)
	}
}

func setBatchAccess(input *batch, access string) {
	for i := range input.Dense {
		input.Dense[i].Reference.AccessFingerprint = access
	}
	for i := range input.Tensor {
		input.Tensor[i].Reference.AccessFingerprint = access
	}
	for i := range input.Lexical {
		input.Lexical[i].Reference.AccessFingerprint = access
	}
	for i := range input.Graph.Nodes {
		input.Graph.Nodes[i].Reference.AccessFingerprint = access
	}
}
