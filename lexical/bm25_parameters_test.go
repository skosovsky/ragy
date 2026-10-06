package lexical

import (
	"context"
	"errors"
	"math"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
)

func TestBM25RejectsInvalidNumericConfiguration(t *testing.T) {
	// Arrange: invalid parameters must be rejected before indexing.
	for _, values := range [][2]float64{{math.NaN(), 0}, {math.Inf(1), 0}, {math.Inf(-1), 0}, {-1, 0}, {0, math.NaN()}, {0, math.Inf(1)}, {0, -1}, {0, 1.01}} {
		// Act.
		index, err := NewBM25Index(
			filter.EmptySchema(),
			Config[struct{}]{SearchFields: []string{"content"}, K1: values[0], B: values[1]},
			nil,
			nil,
		)
		// Assert.
		if index != nil || !errors.Is(err, ragy.ErrInvalidArgument) {
			t.Fatalf("invalid parameters accepted: %v, %v", values, err)
		}
	}
}

func TestBM25SuppressesNonfiniteComputedScore(t *testing.T) {
	// Arrange: finite parameters can still overflow native arithmetic.
	index, err := NewBM25Index(
		filter.EmptySchema(),
		Config[struct{}]{SearchFields: []string{"content"}, K1: math.MaxFloat64, B: 1},
		nil,
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if err = index.Index([]retrieval.Document[struct{}]{{ID: "d", Content: "hello hello"}}); err != nil {
		t.Fatal(err)
	}
	// Act.
	result, err := index.Retrieve(
		context.Background(),
		retrieval.Query[struct{}]{
			Read:    retrieval.UnrestrictedRead(),
			Text:    "hello",
			Options: retrieval.RetrieveOptions{TopK: 1},
		},
	)
	// Assert: invalid scores never reach the result boundary.
	if !errors.Is(err, ragy.ErrProtocol) || result.Len() != 0 {
		t.Fatal("non-finite computed score delivered", err)
	}
}
