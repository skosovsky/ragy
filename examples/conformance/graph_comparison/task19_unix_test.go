//go:build darwin || linux

package main

import (
	"context"
	"slices"
	"testing"

	"github.com/skosovsky/ragy/examples/conformance/internal/task19"
)

func TestTask19GraphAdmitsVersionedScopedOriginals(t *testing.T) {
	// Arrange: one public seed connects to an allowed private original, an
	// inaccessible original and a superseded revision. Only the first is usable.
	c := task19.Corpus{SchemaVersion: "1.0.0", DatasetID: "graph-test", Documents: []task19.Document{
		{
			ID:       "seed",
			SourceID: "seed-source",
			Revision: "v1",
			Scope:    "public",
			Current:  true,
			Text:     "sapphire lookup",
			Relations: []task19.Relation{
				{TargetID: "neighbor", Type: "requires"},
				{TargetID: "foreign", Type: "requires"},
				{TargetID: "old", Type: "requires"},
			},
		},
		{ID: "neighbor", SourceID: "policy", Revision: "v2", Scope: "team-a", Current: true, Text: "eligible policy"},
		{ID: "foreign", SourceID: "secret", Revision: "v1", Scope: "team-b", Current: true, Text: "foreign policy"},
		{ID: "old", SourceID: "policy", Revision: "v1", Scope: "team-a", Current: false, Text: "superseded policy"},
	}}
	q := task19.Query{ID: "q", Scope: "team-a", Text: "sapphire"}
	// Act.
	row, err := runTask19Graph(context.Background(), c, q, "graph-expansion", 1)
	// Assert.
	if err != nil {
		t.Fatal(err)
	}
	if len(row.Retrieved) < 2 || row.Retrieved[0] != "neighbor" || !slices.Contains(row.Retrieved, "seed") {
		t.Fatalf("missing actual admitted neighbor: %+v", row)
	}
	if slices.Contains(row.Retrieved, "foreign") || slices.Contains(row.Retrieved, "old") {
		t.Fatalf("scope/publication leak: %+v", row)
	}
	if row.RetrievalCalls != 2 || row.ModelCalls != 0 || row.Publication == "" {
		t.Fatalf("missing dispatch/publication evidence: %+v", row)
	}
	task19.Audit(c, q, &row)
	if row.CitationViolations+row.ScopeViolations+row.StaleViolations != 0 {
		t.Fatalf("invalid original citations: %+v", row)
	}
	for _, loc := range row.Sources {
		if loc.Reference.Source == "policy" && loc.Reference.Revision != "v2" {
			t.Fatalf("wrong original revision: %+v", loc)
		}
	}
}
