//go:build darwin || linux

package integration_test

import (
	"context"
	"errors"
	"slices"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/dense"
	densefs "github.com/skosovsky/ragy/dense/persistent"
	"github.com/skosovsky/ragy/evidence"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/graph"
	graphmanaged "github.com/skosovsky/ragy/graph/managed"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
	"github.com/skosovsky/ragy/tensor"
	tensorquery "github.com/skosovsky/ragy/tensor/query"
)

type countedPartialBackend struct {
	next  retrieval.Backend[struct{}, meta]
	calls int
}

func (b *countedPartialBackend) Retrieve(
	ctx context.Context, req retrieval.Query[struct{}],
) (retrieval.ResultSet[meta], error) {
	b.calls++
	return b.next.Retrieve(ctx, req)
}
func (b *countedPartialBackend) Schema() filter.Schema {
	return b.next.(retrieval.ReadCapabilityProvider).Schema()
}
func (b *countedPartialBackend) ReadCapabilities() access.Capabilities {
	return b.next.(retrieval.ReadCapabilityProvider).ReadCapabilities()
}
func (b *countedPartialBackend) AdmitPublication(pub access.Publication) error {
	return b.next.(retrieval.PublicationAdmission).AdmitPublication(pub)
}

func secondaryBackend(t *testing.T, f *fixture, input batch) retrieval.Backend[struct{}, meta] {
	t.Helper()
	switch f.target {
	case "tensor":
		return retrieval.ProjectedBackend[struct{}, retrieval.NoRequestMeta, tensorquery.Intent, retrieval.NoRequestMeta, meta]{
			Next: f.secondary.tensor,
			Project: func(req retrieval.Query[struct{}]) retrieval.Query[tensorquery.Intent] {
				return retrieval.Query[tensorquery.Intent]{
					Read:    req.Read,
					Options: req.Options,
					Intent: tensorquery.Intent{
						Embedding:       tensor.Embedding{Space: tensorSpace(), Tokens: tensor.Tensor{{1, 0}}},
						Candidates:      []source.Reference{input.Tensor[0].Reference},
						CandidateBudget: 100,
					},
				}
			},
		}
	case "graph":
		backend, err := graphmanaged.NewBackend(
			graphmanaged.BackendConfig[meta]{Adapter: f.secondary.graph, MaxNodes: 100, MaxEdges: 100},
		)
		if err != nil {
			t.Fatal(err)
		}
		return backend
	default:
		return f.secondary.lexical
	}
}

func assertPartialCaptureFanout(t *testing.T, f *fixture, input batch) {
	t.Helper()
	// Arrange: one active source has a missing target; faq still has its ready revision.
	pub, err := lifecycle.CapturePartialPublication(t.Context(), f.store, "fixture-a", []string{"dense", f.target})
	if err != nil || !pub.IsPartial() || !slices.Equal(pub.ExcludedTargets(), []string{f.target}) {
		t.Fatal("missing partial capture", err)
	}
	for _, target := range pub.Targets() {
		if target.Target != "dense" {
			t.Fatal("incomplete branch retained faq-only inventory")
		}
	}
	read := f.bind(t, pub)
	secondary := &countedPartialBackend{next: secondaryBackend(t, f, input)}
	denseBackend := retrieval.ProjectedBackend[struct{}, retrieval.NoRequestMeta, densefs.Intent, retrieval.NoRequestMeta, meta]{
		Next: f.dense,
		Project: func(req retrieval.Query[struct{}]) retrieval.Query[densefs.Intent] {
			return retrieval.Query[densefs.Intent]{Read: req.Read, Options: req.Options,
				Intent: densefs.Intent{Embedding: dense.Embedding{Space: denseSpace(), Vector: []float32{1, 0}}}}
		},
	}
	root := retrieval.RequestExecutionAggregateNode[struct{}, retrieval.NoRequestMeta, meta, struct{}]{
		Nodes: []retrieval.RequestExecutionNode[struct{}, retrieval.NoRequestMeta, meta, struct{}]{
			retrieval.BackendNode[struct{}, meta, struct{}]{Backend: denseBackend},
			retrieval.PartialReadNode[struct{}, meta, struct{}]{
				Name:  f.target,
				Child: retrieval.BackendNode[struct{}, meta, struct{}]{Backend: secondary},
			},
		}, Concurrency: 2,
	}
	request := retrieval.Query[struct{}]{Read: read, Text: "needle", Options: retrieval.RetrieveOptions{TopK: 10}}
	// Act: parallel composition preserves the one captured pin and skips missing target before I/O.
	result, err := root.Execute(t.Context(), request, struct{}{})
	// Assert: no phantom complete-empty secondary and no r1 substitution for policy.
	if err != nil || secondary.calls != 0 || !result.Coverage.IsPartial() {
		t.Fatal("partial fan-out failed", err)
	}
	assertRevision(t, result.ResultSet.Documents(), "r2", 1)
	if err = secondary.AdmitPublication(pub); !errors.Is(err, ragy.ErrUnsupported) {
		t.Fatal("excluded target admitted", err)
	}
	var hits []evidence.Hit[meta]
	for _, doc := range result.ResultSet.Documents() {
		var refs []source.Reference
		for _, location := range doc.SourceLocations() {
			refs = append(refs, location.Reference)
		}
		hits = append(hits, evidence.Hit[meta]{Document: doc, Sources: refs})
	}
	// Act: export receives an erroneously complete producer outcome.
	record, err := evidence.Capture(t.Context(), read, evidence.Input[meta]{
		Schema:          f.schema,
		Codec:           retrieval.NewJSONCodec[meta](f.schema),
		SourceAdmission: func(context.Context, access.Binding, source.Reference) error { return nil },
		RetrievalID:     "partial-query",
		RecipeRevision:  "baseline",
		Outcome:         evidence.Complete,
		Reason:          evidence.NoReason,
		Coverage:        retrieval.CompleteReadCoverage(),
		Stages: []evidence.Stage[meta]{{Name: "dense", Status: evidence.StageObserved, Scores: evidence.Observed,
			Sources: evidence.Observed, Judgments: evidence.Unavailable, Hits: hits}},
	}, evidence.Policy{AllowIdentifier: func(evidence.IdentifierKind, string) bool { return true }, AllowNumbers: true})
	// Assert: immutable binding coverage prevents promotion even in an empty/redacted record.
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := record.Snapshot()
	if err != nil || len(snapshot.Stages) != 1 || len(snapshot.Stages[0].Hits) != 2 ||
		snapshot.Outcome != evidence.Partial ||
		snapshot.Reason != evidence.PartialTargets ||
		!snapshot.Coverage.IsPartial() ||
		!slices.Equal(snapshot.Coverage.SkippedBranches(), []string{f.target}) {
		t.Fatal("export promoted incomplete publication", snapshot, err)
	}
}

func TestPartialCaptureKeepsAvailableSecondaryTargetsReadable(t *testing.T) {
	for _, target := range []string{"lexical", "tensor", "graph"} {
		t.Run(target, func(t *testing.T) { availableSecondaryCase(t, target) })
	}
}

func availableSecondaryCase(t *testing.T, target string) {
	t.Helper()
	// Arrange: only secondary r2 ready; dense r2 must remain excluded.
	f := newFixture(t, target)
	old := sourceBatch("policy", "r1", []string{"p1", "p2"})
	faq := sourceBatch("faq", "r1", []string{"f1"})
	f.ingest(t, plan("policy-r1", "", target, old), old)
	f.ingest(t, plan("faq-r1", "", target, faq), faq)
	newer := sourceBatch("policy", "r2", []string{"p3"})
	manifest := plan("policy-r2", "policy-r1", target, newer)
	manifest.Partial = true
	if _, err := f.executor.Prepare(t.Context(), manifest); err != nil {
		t.Fatal(err)
	}
	if _, err := f.executor.Stage(t.Context(), "fixture-a", manifest.ID, target, newer); err != nil {
		t.Fatal(err)
	}
	if _, err := f.executor.Publish(t.Context(), "fixture-a", manifest.ID); err != nil {
		t.Fatal(err)
	}
	pub, err := lifecycle.CapturePartialPublication(
		t.Context(),
		f.store,
		"fixture-a",
		[]string{"dense", target},
	)
	if err != nil || !slices.Equal(pub.ExcludedTargets(), []string{"dense"}) {
		t.Fatal("dense exclusion lost", err)
	}
	read := f.bind(t, pub)
	request := retrieval.Query[struct{}]{
		Read:    read,
		Text:    "needle",
		Options: retrieval.RetrieveOptions{TopK: 10},
	}
	if target == "graph" {
		request.Options.Graph = &retrieval.GraphOptions{
			Seeds:     []string{"p3"},
			Direction: graph.DirectionOutbound,
			Depth:     1,
		}
	}
	// Act: actual remaining lexical/tensor/graph target uses the shared partial pin.
	result, err := secondaryBackend(t, f, newer).Retrieve(t.Context(), request)
	// Assert: no blanket refusal and no implicit old policy revision.
	if err != nil || result.Len() == 0 {
		t.Fatal("available partial target refused", err)
	}
	assertPartialPolicyRevision(t, result.Documents())
}

func assertPartialPolicyRevision(t *testing.T, documents []retrieval.Document[meta]) {
	t.Helper()
	policyCount := 0
	for _, doc := range documents {
		if doc.Meta.Source == "policy" {
			policyCount++
			if doc.Meta.Revision != "r2" {
				t.Fatal("old policy substituted", doc.Meta)
			}
		}
	}
	if policyCount != 1 {
		t.Fatal("partial policy inventory lost", policyCount)
	}
}
