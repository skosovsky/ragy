package retrieval

import (
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/graph"
)

func TestRetrieveOptionsValidateFetchLimitNegative(t *testing.T) {
	t.Parallel()

	err := (RetrieveOptions{FetchLimit: -1}).Validate()
	if err == nil {
		t.Fatal("Validate() error = nil, want error")
	}
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Validate() error = %v, want invalid argument", err)
	}
}

func TestRetrieveOptionsValidateFetchLimitLessThanTopK(t *testing.T) {
	t.Parallel()

	err := (RetrieveOptions{FetchLimit: 5, TopK: 10}).Validate()
	if err == nil {
		t.Fatal("Validate() error = nil, want error")
	}
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Validate() error = %v, want invalid argument", err)
	}
}

func TestRetrieveOptionsValidateFetchLimitZeroWithTopK(t *testing.T) {
	t.Parallel()

	if err := (RetrieveOptions{FetchLimit: 0, TopK: 10}).Validate(); err != nil {
		t.Fatalf("Validate() error = %v, want nil", err)
	}
}

func TestRetrieveOptionsBackendFetchLimitFallsBackToTopK(t *testing.T) {
	t.Parallel()

	limit := (RetrieveOptions{FetchLimit: 0, TopK: 10}).BackendFetchLimit()
	if limit != 10 {
		t.Fatalf("BackendFetchLimit() = %d, want 10", limit)
	}
}

func TestRetrieveOptionsBackendFetchLimitUsesExplicitValue(t *testing.T) {
	t.Parallel()

	limit := (RetrieveOptions{FetchLimit: 50, TopK: 10}).BackendFetchLimit()
	if limit != 50 {
		t.Fatalf("BackendFetchLimit() = %d, want 50", limit)
	}
}

func TestRetrieveOptionsValidateMinSimilarityOutOfRange(t *testing.T) {
	t.Parallel()

	for _, ms := range []float64{-0.1, 1.1} {
		err := (RetrieveOptions{TopK: 1, MinSimilarity: ms}).Validate()
		if err == nil {
			t.Fatalf("Validate(min_similarity=%v) error = nil, want error", ms)
		}
		if !errors.Is(err, ragy.ErrInvalidArgument) {
			t.Fatalf("Validate(min_similarity=%v) error = %v, want invalid argument", ms, err)
		}
	}
}

func TestRetrieveOptionsValidateTopKNegative(t *testing.T) {
	t.Parallel()

	err := (RetrieveOptions{TopK: -1}).Validate()
	if err == nil {
		t.Fatal("Validate() error = nil, want error")
	}
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Validate() error = %v, want invalid argument", err)
	}
}

func TestRetrieveOptionsValidateRejectsZeroTopKAndFetchLimit(t *testing.T) {
	t.Parallel()

	err := (RetrieveOptions{}).Validate()
	if err == nil {
		t.Fatal("Validate() error = nil, want error")
	}
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Validate() error = %v, want invalid argument", err)
	}
}

func TestRetrieveOptionsValidateAllowsZeroTopKWithFetchLimit(t *testing.T) {
	t.Parallel()

	if err := (RetrieveOptions{FetchLimit: 10, TopK: 0}).Validate(); err != nil {
		t.Fatalf("Validate() error = %v, want nil", err)
	}
}

func TestGraphOptionsValidateRejectsInvalidConfig(t *testing.T) {
	t.Parallel()

	if err := (RetrieveOptions{TopK: 1}).Validate(); err != nil {
		t.Fatalf("Validate(empty options with top_k) error = %v, want nil", err)
	}

	err := (RetrieveOptions{TopK: 1, Graph: &GraphOptions{}}).Validate()
	if !errors.Is(err, ragy.ErrInvalidGraph) {
		t.Fatalf("Validate(empty seeds) error = %v, want invalid graph", err)
	}

	err = (RetrieveOptions{TopK: 1, Graph: &GraphOptions{
		Seeds:     []string{"n1"},
		Direction: graph.DirectionOutbound,
		Depth:     0,
	}}).Validate()
	if !errors.Is(err, ragy.ErrInvalidGraph) {
		t.Fatalf("Validate(depth=0) error = %v, want invalid graph", err)
	}

	err = (RetrieveOptions{TopK: 1, Graph: &GraphOptions{
		Seeds:     []string{"n1"},
		Direction: graph.Direction("invalid"),
		Depth:     1,
	}}).Validate()
	if !errors.Is(err, ragy.ErrInvalidGraph) {
		t.Fatalf("Validate(bad direction) error = %v, want invalid graph", err)
	}
}
