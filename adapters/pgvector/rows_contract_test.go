package pgvector

import (
	"errors"
	"testing"

	"github.com/skosovsky/ragy/retrieval"
)

type closingRows struct {
	*fakeRows

	closeErr error
	closes   int
}

func (r *closingRows) Close() error { r.closes++; return r.closeErr }

func TestRowsCloseCleanupAndErrOutcomeAreDistinct(t *testing.T) {
	for _, operation := range []string{"query", "find"} {
		for _, failed := range []bool{false, true} {
			checkRowsCloseOutcome(t, operation, failed)
		}
	}
}

func checkRowsCloseOutcome(t *testing.T, operation string, failed bool) {
	t.Helper()
	// Arrange: host cleanup diagnostic is separate from query outcome.
	outcome := errors.New("host iteration failed")
	rows := &closingRows{
		fakeRows: &fakeRows{rows: []fakeRow{{id: "d", content: "text", attrsJSON: []byte(`{}`), relevance: 1}}},
		closeErr: errors.New("cleanup diagnostic"),
	}
	if failed {
		rows.rowsErr = outcome
	}
	db := &fakeDB{queryRows: rows}
	store := newStoreEmptySchema(t, db)
	// Act.
	var err error
	var count int
	if operation == "query" {
		result, queryErr := retrieveStore(
			t.Context(),
			store,
			"",
			retrieval.RetrieveOptions{Space: fixtureSpace(), Vector: []float32{1}, TopK: 1},
		)
		count, err = result.Len(), queryErr
	} else {
		docs, findErr := store.FindByIDs(t.Context(), []string{"d"})
		count, err = len(docs), findErr
	}
	// Assert: Close always runs; ordinary iteration errors retain their prefix.
	if count != 1 || rows.closes != 1 {
		t.Fatal("prefix/cleanup", operation, failed, count, rows.closes)
	}
	if failed {
		if !errors.Is(err, outcome) {
			t.Fatal("lost query outcome", operation, err)
		}
	} else if err != nil {
		t.Fatal("cleanup became query outcome", operation, err)
	}
}
