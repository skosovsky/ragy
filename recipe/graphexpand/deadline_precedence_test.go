package graphexpand_test

import (
	"context"
	"errors"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/lifecycle"
)

type failingTraversalStore struct {
	lifecycle.Store

	f     *fixture
	cause error
	calls int
}

func (s *failingTraversalStore) Load(ctx context.Context, namespace string) (lifecycle.Snapshot, error) {
	if s.f.quotes == 0 {
		return s.Store.Load(ctx, namespace)
	}
	s.calls++
	s.f.now = s.f.now.Add(6 * time.Second)
	return lifecycle.Snapshot{}, s.cause
}

func TestTraversalCauseSurvivesExpansionExpiry(t *testing.T) {
	for _, cause := range []error{ragy.ErrProtocol, errors.New("traversal failed"), access.NonSkippable(ragy.ErrUnavailable), errors.Join(access.NonSkippable(ragy.ErrUnavailable), context.DeadlineExceeded)} {
		// Arrange: lawful admission, one failing traversal and simultaneous local expiry.
		f := newFixture(t)
		adapter := &failingTraversalStore{Store: f.store.Store, f: f, cause: cause, calls: 0}
		f.store.Store = adapter
		// Act.
		result, ledger, err := run(context.Background(), t, f)
		// Assert: the adapter's protection wrapper may normalize the joined object,
		// but every public cause class must survive the recipe boundary.
		expected := cause
		if access.IsProtectionFailure(cause) {
			expected = ragy.ErrUnavailable
		}
		if !errors.Is(err, expected) || !errors.Is(err, context.DeadlineExceeded) ||
			len(result.Evidence.Snapshot.Nodes) != 0 ||
			adapter.calls != 1 ||
			ledger.Snapshot().Outstanding != 0 ||
			ledger.Snapshot().Occupied.RetrievalCalls != 1 {
			t.Fatal("expansion cause lost", err)
		}
		if access.IsProtectionFailure(cause) && !access.IsProtectionFailure(err) {
			t.Fatal("protection lost", err)
		}
	}
}
