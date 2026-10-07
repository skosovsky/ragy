package lexical

import (
	"context"
	"errors"
	"fmt"
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
			Config[struct{}]{
				SearchFields: []string{"content"},
				Parameters:   &BM25Parameters{K1: values[0], B: values[1]},
			},
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
		Config[struct{}]{SearchFields: []string{"content"}, Parameters: &BM25Parameters{K1: math.MaxFloat64, B: 1}},
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

func TestBM25ExplicitZeroParametersAndOwnedConfiguration(t *testing.T) {
	for _, parameters := range []BM25Parameters{{K1: 0, B: 0}, {K1: 1.2, B: 0}, {K1: 0, B: 1}} {
		t.Run(fmt.Sprintf("K1=%g/B=%g", parameters.K1, parameters.B), func(t *testing.T) {
			// Arrange: frequencies and lengths differ; each zero has observable meaning.
			owned := parameters
			index, err := NewBM25Index(
				filter.EmptySchema(),
				Config[struct{}]{SearchFields: []string{"content"}, Parameters: &owned},
				nil,
				nil,
			)
			if err != nil {
				t.Fatal(err)
			}
			if err = index.Index(
				[]retrieval.Document[struct{}]{
					{ID: "a", Content: "needle"},
					{ID: "b", Content: "needle needle filler filler"},
				},
			); err != nil {
				t.Fatal(err)
			}
			// Act: caller mutation must not change the scorer after construction.
			owned.K1, owned.B = 99, 99
			result, err := retrieveBM25(t.Context(), index, "needle", retrieval.RetrieveOptions{TopK: 2})
			// Assert: compare actual native scores against the independent BM25 formula.
			if err != nil || result.Len() != 2 {
				t.Fatal(result, err)
			}
			assertBM25ParameterScores(t, result.Documents(), parameters)
		})
	}
}

func TestBM25AbsentParametersSelectDefaults(t *testing.T) {
	// Arrange.
	index, err := NewBM25Index(filter.EmptySchema(), Config[struct{}]{SearchFields: []string{"content"}}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	parameters := *index.config.Parameters
	// Assert.
	if parameters != (BM25Parameters{K1: 1.2, B: 0.75}) {
		t.Fatal("defaults", parameters)
	}
}

func assertBM25ParameterScores(t *testing.T, documents []retrieval.Document[struct{}], parameters BM25Parameters) {
	t.Helper()
	idf := math.Log(1 + 0.5/2.5)
	for _, doc := range documents {
		tf, length := 1.0, 1.0
		if doc.ID == "b" {
			tf, length = 2, 4
		}
		want := idf * tf * (parameters.K1 + 1) / (tf + parameters.K1*(1-parameters.B+parameters.B*length/2.5))
		if math.Abs(doc.Score-want) > 1e-12 ||
			doc.ScoreSemantics != retrieval.ScoreSemantics(
				fmt.Sprintf("lexical.bm25:k1=%g,b=%g", parameters.K1, parameters.B),
			) {
			t.Fatalf("doc=%+v want=%v", doc, want)
		}
	}
}
