package main

import (
	"context"
	"testing"

	"example.com/ragyconsumer/internal/task19"
)

func TestTask19RecipesExecuteRealBoundedRetrieval(t *testing.T) {
	// Arrange: one admitted original and a matching foreign source.
	c := task19.Corpus{
		SchemaVersion: "1.0.0",
		DatasetID:     "smoke",
		Documents: []task19.Document{
			{
				ID:       "gold",
				SourceID: "s",
				Revision: "r2",
				Scope:    "public",
				Current:  true,
				Text:     "Atlas rollback preserves checkpoint offsets.",
			},
			{
				ID:       "foreign",
				SourceID: "private",
				Revision: "r1",
				Scope:    "red",
				Current:  true,
				Text:     "Atlas rollback preserves checkpoint offsets.",
			},
		},
	}
	q := task19.Query{ID: "q", Text: "Atlas rollback checkpoint offsets", Scope: "public"}
	for _, strategy := range []string{rewriteProfile, multiProfile, decompositionProfile} {
		// Act.
		row, err := task19Run(context.Background(), c, q, strategy, 1)
		// Assert: actual recipe dispatches use planner, assessor and admitted source only.
		if err != nil {
			t.Fatalf("%s: %v", strategy, err)
		}
		if row.ModelCalls != 2 || row.RetrievalCalls == 0 || len(row.Delivered) != 1 || row.Delivered[0] != "gold" ||
			row.LocalInputUnits == 0 {
			t.Fatalf("%s: %+v", strategy, row)
		}
	}
}
