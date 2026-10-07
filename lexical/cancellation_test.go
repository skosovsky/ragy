package lexical

import (
	"context"
	"errors"
	"testing"

	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
)

type checkpointContext struct {
	context.Context

	remaining int
}

func (c *checkpointContext) Err() error {
	c.remaining--
	if c.remaining <= 0 {
		return context.Canceled
	}
	return nil
}

func TestBM25ScoringAndRankingCancelWithoutPayload(t *testing.T) {
	// Arrange: enough documents to cancel inside accumulation or construction.
	index, err := NewBM25Index(filter.EmptySchema(), Config[struct{}]{SearchFields: []string{"content"}}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err = index.Index(
		[]retrieval.Document[struct{}]{
			{ID: "a", Content: "needle"},
			{ID: "b", Content: "needle"},
			{ID: "c", Content: "needle"},
		},
	); err != nil {
		t.Fatal(err)
	}
	snapshot := index.snapshotLocked([]string{"needle"})
	t.Run("scoring", func(t *testing.T) {
		// Act.
		scores, err := index.scoreQuery(
			&checkpointContext{Context: context.Background(), remaining: 4},
			snapshot,
			[]string{"needle"},
		)
		// Assert.
		if !errors.Is(err, context.Canceled) || scores != nil {
			t.Fatal("canceled scoring payload", scores, err)
		}
	})
	t.Run("ranking", func(t *testing.T) {
		// Act.
		docs, err := index.rankScoredDocs(
			&checkpointContext{Context: context.Background(), remaining: 6},
			snapshot,
			map[string]float64{"a": 1, "b": 1, "c": 1},
			3,
		)
		// Assert.
		if !errors.Is(err, context.Canceled) || docs != nil {
			t.Fatal("canceled ranking payload", docs, err)
		}
	})
}
