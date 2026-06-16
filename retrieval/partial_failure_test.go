package retrieval

import (
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
)

func TestPreserveResultOnError(t *testing.T) {
	t.Parallel()

	resolver := DocumentIDResolver[struct{}]{}
	docs := []Document[struct{}]{{ID: "a", Content: "hit", Score: 1}}

	t.Run("nil error returns rs", func(t *testing.T) {
		t.Parallel()
		rs := NewResultSet(docs, resolver)
		out, err := PreserveResultOnError(rs, nil, resolver)
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		if out.Len() != 1 {
			t.Fatalf("Len() = %d, want 1", out.Len())
		}
	})

	t.Run("plain error preserves non-empty rs", func(t *testing.T) {
		t.Parallel()
		rs := NewResultSet(docs, resolver)
		out, err := PreserveResultOnError(rs, ragy.ErrUnavailable, resolver)
		if !errors.Is(err, ragy.ErrUnavailable) {
			t.Fatalf("err = %v, want unavailable", err)
		}
		if out.Len() != 1 || out.Documents()[0].ID != "a" {
			t.Fatalf("Documents() = %#v, want preserved docs", out.Documents())
		}
	})

	t.Run("partial failure error preserves result", func(t *testing.T) {
		t.Parallel()
		rs := NewResultSet(docs, resolver)
		partial := &PartialFailureError[struct{}]{Errors: []error{ragy.ErrProtocol}, Result: rs}
		out, err := PreserveResultOnError(NewResultSet[struct{}](nil, resolver), partial, resolver)
		var got *PartialFailureError[struct{}]
		if !errors.As(err, &got) {
			t.Fatalf("err = %v, want PartialFailureError", err)
		}
		if out.Len() != 1 {
			t.Fatalf("Len() = %d, want partial result", out.Len())
		}
	})

	t.Run("empty rs on error", func(t *testing.T) {
		t.Parallel()
		out, err := PreserveResultOnError(NewResultSet[struct{}](nil, resolver), ragy.ErrUnavailable, resolver)
		if !errors.Is(err, ragy.ErrUnavailable) {
			t.Fatalf("err = %v, want unavailable", err)
		}
		if out == nil || !out.IsEmpty() {
			t.Fatalf("out = %#v, want non-nil empty", out)
		}
	})
}

func TestSyncPartialFailureResult(t *testing.T) {
	t.Parallel()

	resolver := DocumentIDResolver[struct{}]{}
	docs := []Document[struct{}]{{ID: "a", Content: "hit", Score: 1}}
	rs := NewResultSet(docs, resolver)

	t.Run("plain error unchanged", func(t *testing.T) {
		t.Parallel()
		err := syncPartialFailureResult(ragy.ErrUnavailable, rs)
		if !errors.Is(err, ragy.ErrUnavailable) {
			t.Fatalf("err = %v, want unavailable", err)
		}
	})

	t.Run("partial updates result", func(t *testing.T) {
		t.Parallel()
		partial := &PartialFailureError[struct{}]{
			Errors: []error{ragy.ErrProtocol},
			Result: NewResultSet([]Document[struct{}]{
				{ID: "old", Content: "stale", Score: 0.1},
			}, resolver),
		}
		updated := syncPartialFailureResult(partial, rs)
		got, ok := AsPartialFailure[struct{}](updated)
		if !ok {
			t.Fatalf("err = %v, want partial failure", updated)
		}
		if got.Result.Len() != 1 || got.Result.Documents()[0].ID != "a" {
			t.Fatalf("partial.Result = %#v, want synced docs", got.Result.Documents())
		}
	})

	t.Run("nil partial error unchanged", func(t *testing.T) {
		t.Parallel()
		if syncPartialFailureResult(nil, rs) != nil {
			t.Fatal("syncPartialFailureResult(nil) = err, want nil")
		}
	})

	t.Run("empty final result", func(t *testing.T) {
		t.Parallel()
		partial := &PartialFailureError[struct{}]{
			Errors: []error{ragy.ErrProtocol},
			Result: rs,
		}
		empty := NewResultSet[struct{}](nil, resolver)
		updated := syncPartialFailureResult(partial, empty)
		got, ok := AsPartialFailure[struct{}](updated)
		if !ok {
			t.Fatalf("err = %v, want partial failure", updated)
		}
		if !got.Result.IsEmpty() {
			t.Fatalf("partial.Result = %#v, want empty", got.Result.Documents())
		}
	})
}
