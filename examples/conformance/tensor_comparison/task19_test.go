//go:build darwin || linux

package main

import (
	"context"
	"testing"

	"github.com/skosovsky/ragy/examples/conformance/internal/task19"
)

func TestTask19TensorManagedScopeAndOriginalSupports(t *testing.T) {
	// Arrange: a private exact match must never enter the public candidate set.
	c := task19.Corpus{DatasetID: "tensor-scope-test", Documents: []task19.Document{
		{
			ID:       "public",
			SourceID: "public-source",
			Revision: "r2",
			Scope:    "public",
			Current:  true,
			Text:     "refund window fourteen days",
		},
		{
			ID:       "private",
			SourceID: "private-source",
			Revision: "r1",
			Scope:    "staff",
			Current:  true,
			Text:     "refund window fourteen days",
		},
		{
			ID:       "stale",
			SourceID: "public-source",
			Revision: "r1",
			Scope:    "public",
			Current:  false,
			Text:     "refund window ninety days",
		},
	}}
	q := task19.Query{
		ID:         "q",
		Text:       "refund window",
		Scope:      "customer",
		Answerable: true,
		Qrels:      []task19.Qrel{{DocumentID: "public", Grade: 3}},
	}
	// Act: run the real persistent dense candidate and tensor MaxSim queries.
	row, err := task19Run(context.Background(), c, q, "tensor-candidate-maxsim", 1)
	// Assert: citations retain the original public revision; private/stale docs stay out.
	if err != nil {
		t.Fatal(err)
	}
	task19.Audit(c, q, &row)
	if len(row.Retrieved) != 1 || row.Retrieved[0] != "public" || len(row.Delivered) != 1 || len(row.Sources) != 1 {
		t.Fatalf("unexpected row: %+v", row)
	}
	if row.Sources[0] != task19.Locator(c, c.Documents[0]) ||
		row.ScopeViolations+row.StaleViolations+row.CitationViolations != 0 {
		t.Fatalf("lost source/scope: %+v", row)
	}
	if row.RetrievalCalls != 2 || row.ModelCalls != 0 || !row.UsageKnown || row.ContextUnits > task19.ContextBytes {
		t.Fatalf("bad accounting: %+v", row)
	}
}
