package retrieval

import (
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
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
