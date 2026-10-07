package lexical

import (
	"context"
	"math"
	"strconv"
	"testing"

	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
)

func TestBM25QuerySnapshotOwnsOnlyCandidatesAndKeepsGlobalStatistics(t *testing.T) {
	// Arrange.
	index, err := NewBM25Index[struct{}](
		filter.EmptySchema(),
		Config[struct{}]{SearchFields: []string{"content"}},
		nil,
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	docs := []retrieval.Document[struct{}]{{ID: "rare", Content: "needle needle short"}}
	for i := range 100 {
		docs = append(docs, retrieval.Document[struct{}]{ID: strconv.Itoa(i), Content: "common ordinary words"})
	}
	if err = index.Index(docs); err != nil {
		t.Fatal(err)
	}
	// Act.
	index.mu.RLock()
	snapshot := index.snapshotLocked([]string{"needle", "needle"})
	index.mu.RUnlock()
	beforeScores, err := index.scoreQuery(context.Background(), snapshot, []string{"needle"})
	if err != nil {
		t.Fatal(err)
	}
	before := beforeScores["rare"]
	if err = index.Upsert(
		retrieval.Document[struct{}]{ID: "rare", Content: "common replacement much longer words"},
	); err != nil {
		t.Fatal(err)
	}
	// Assert: immutable old view, globally correct IDF, no unrelated corpus copied.
	expected := math.Log(1+(101.0-1+0.5)/(1+0.5)) * 2 * (1.2 + 1) / (2 + 1.2)
	if len(snapshot.docs) != 1 || len(snapshot.docLengths) != 1 || len(snapshot.postings) != 1 ||
		snapshot.docCount != 101 ||
		snapshot.avgLength != 3 ||
		math.Abs(before-expected) > 1e-12 {
		t.Fatalf("snapshot=%+v score=%v expected=%v", snapshot, before, expected)
	}
	afterScores, err := index.scoreQuery(context.Background(), snapshot, []string{"needle"})
	if err != nil {
		t.Fatal(err)
	}
	if got := afterScores["rare"]; got != before {
		t.Fatalf("reader changed after upsert: %v vs %v", got, before)
	}
	if index.totalLength != 305 || index.avgLength != 305.0/101 {
		t.Fatalf("replacement aggregate total=%d avg=%v", index.totalLength, index.avgLength)
	}
}

func TestBM25OwnsSearchConfiguration(t *testing.T) {
	// Arrange.
	fields := []string{"content"}
	variants := []string{"needle"}
	synonyms := SynonymMap{"alias": variants}
	index, err := NewBM25Index[struct{}](filter.EmptySchema(), Config[struct{}]{SearchFields: fields}, nil, synonyms)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	fields[0] = "unknown"
	variants[0] = "wrong"
	delete(synonyms, "alias")
	err = index.Upsert(retrieval.Document[struct{}]{ID: "one", Content: "needle"})
	result, retrieveErr := retrieveBM25(t.Context(), index, "alias", retrieval.RetrieveOptions{TopK: 1})
	// Assert.
	if err != nil || retrieveErr != nil || result.Len() != 1 {
		t.Fatalf("upsert=%v retrieve=%v docs=%v", err, retrieveErr, result.Documents())
	}
}
