//go:build darwin || linux

package lifecycle_test

import (
	"context"
	"errors"
	"slices"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/lifecycle/filestore"
)

func pinStore(t *testing.T) (*filestore.Store, string) {
	t.Helper()
	root := t.TempDir()
	store, err := filestore.New(root, 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	manifest := manifestFixture()
	_, err = store.CompareSwap(t.Context(), 0, lifecycle.Snapshot{
		Schema: lifecycle.SchemaIdentity, Namespace: "n", Manifests: []lifecycle.Manifest{manifest},
		Publications: []lifecycle.Publication{{Source: "policy", Manifest: manifest.ID}},
	})
	if err != nil {
		t.Fatal(err)
	}
	return store, root
}

func advancePinHistory(t *testing.T, store lifecycle.Store) lifecycle.Snapshot {
	t.Helper()
	snapshot, err := store.Load(t.Context(), "n")
	if err != nil {
		t.Fatal(err)
	}
	owner := manifestFixture()
	owner.ID, owner.Key, owner.ExpectedPublication = "deleted", "delete-key", "pub1"
	owner.Tombstone, owner.Targets, owner.State = true, nil, lifecycle.Complete
	owner.PublishedAt = time.Unix(200, 0).UTC()
	snapshot.Manifests = append(snapshot.Manifests, owner)
	snapshot.Publications = []lifecycle.Publication{{Source: "policy", Manifest: owner.ID}}
	snapshot.Cleanups = []lifecycle.CleanupJob{{
		Owner: owner.ID, StartedAt: owner.PublishedAt, Deadline: owner.PublishedAt.Add(time.Minute), Complete: true,
		Items: []lifecycle.RetiredTarget{
			{
				Manifest: "pub1",
				Target:   "dense",
				State:    lifecycle.CleanupDone,
				Attempts: 1,
				NextAt:   owner.PublishedAt,
			},
			{
				Manifest: "pub1",
				Target:   "tensor",
				State:    lifecycle.CleanupDone,
				Attempts: 1,
				NextAt:   owner.PublishedAt,
			},
		},
	}}
	snapshot, err = store.CompareSwap(t.Context(), snapshot.Generation, snapshot)
	if err != nil {
		t.Fatal(err)
	}
	return snapshot
}

func TestPublicationPinRestartHistoricalReplayReleaseAndRetirement(t *testing.T) {
	// Arrange: acquire the exact active publication before advancing visibility.
	store, root := pinStore(t)
	first, err := lifecycle.AcquirePublicationPin(
		t.Context(),
		store,
		"n",
		"reader",
		[]string{"tensor", "dense"},
	)
	if err != nil {
		t.Fatal(err)
	}
	advanced := advancePinHistory(t, store)
	restarted, err := filestore.New(root, 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	// Act: replay the live handle after restart and after cleanup confirmation.
	replay, replayErr := lifecycle.AcquirePublicationPin(
		t.Context(),
		restarted,
		"n",
		"reader",
		[]string{"dense", "tensor"},
	)
	_, protectedErr := restarted.Maintain(
		t.Context(),
		advanced.Generation,
		lifecycle.RetirementRequest{Namespace: "n", Manifests: []string{"pub1"}},
	)
	// Assert: replay remains historical; registration blocks metadata retirement.
	if replayErr != nil || replay.Reference() != first.Reference() ||
		!slices.Equal(replay.Targets(), first.Targets()) ||
		!errors.Is(protectedErr, lifecycle.ErrProtected) {
		t.Fatal("historical registration lost", replayErr, protectedErr)
	}
	changed := replay.Targets()
	changed[0].Revision = "caller-change"
	if first.Targets()[0].Revision != "r1" {
		t.Fatal("returned publication aliases inventory")
	}
	if err = lifecycle.ReleasePublicationPin(t.Context(), restarted, "n", "reader"); err != nil {
		t.Fatal(err)
	}
	released, err := restarted.Load(t.Context(), "n")
	if err != nil {
		t.Fatal(err)
	}
	if err = lifecycle.ReleasePublicationPin(t.Context(), restarted, "n", "reader"); err != nil {
		t.Fatal(err)
	}
	unchanged, err := restarted.Load(t.Context(), "n")
	if err != nil || unchanged.Generation != released.Generation {
		t.Fatal("idempotent release mutated generation", err)
	}
	retired, err := restarted.Maintain(
		t.Context(),
		released.Generation,
		lifecycle.RetirementRequest{Namespace: "n", Manifests: []string{"pub1"}},
	)
	if err != nil || !retired.Manifests[0].Retired ||
		len(retired.Manifests[0].Targets[0].Artifacts) != 0 {
		t.Fatal("released metadata cannot retire", err)
	}
	restarted, err = filestore.New(root, 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	_, err = lifecycle.AcquirePublicationPin(
		t.Context(),
		restarted,
		"n",
		"reader",
		[]string{"dense", "tensor"},
	)
	if !errors.Is(err, lifecycle.ErrRetired) {
		t.Fatal("released ID reused after retirement/restart", err)
	}
}

func TestPublicationPinCompleteEmptyReservesRequestedProfile(t *testing.T) {
	// Arrange: empty namespace must still record the requested branches.
	store, err := filestore.New(t.TempDir(), 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	first, err := lifecycle.AcquirePublicationPin(
		t.Context(),
		store,
		"n",
		"empty-reader",
		[]string{"dense"},
	)
	_, conflict := lifecycle.AcquirePublicationPin(
		t.Context(),
		store,
		"n",
		"empty-reader",
		[]string{"tensor"},
	)
	// Assert.
	if err != nil || first.IsCurrent() || len(first.Targets()) != 0 ||
		!errors.Is(conflict, lifecycle.ErrIdempotencyConflict) {
		t.Fatal("empty registration lost request profile", err, conflict)
	}
}

type interleavingPinStore struct {
	lifecycle.Store

	beforeCAS func()
	committed bool
}

func (s *interleavingPinStore) CompareSwap(
	ctx context.Context,
	expected uint64,
	next lifecycle.Snapshot,
) (lifecycle.Snapshot, error) {
	if s.beforeCAS != nil {
		run := s.beforeCAS
		s.beforeCAS = nil
		run()
	}
	result, err := s.Store.CompareSwap(ctx, expected, next)
	if err == nil && s.committed {
		return lifecycle.Snapshot{}, context.DeadlineExceeded
	}
	return result, err
}

func TestPublicationPinCASRejectsConcurrentPublicationRetirement(t *testing.T) {
	// Arrange: retirement completes between acquisition Load and registration CAS.
	store, _ := pinStore(t)
	wrapper := &interleavingPinStore{Store: store, beforeCAS: func() {
		advanced := advancePinHistory(t, store)
		if _, err := store.Maintain(
			t.Context(),
			advanced.Generation,
			lifecycle.RetirementRequest{Namespace: "n", Manifests: []string{"pub1"}},
		); err != nil {
			t.Fatal(err)
		}
	}}
	// Act.
	_, err := lifecycle.AcquirePublicationPin(
		t.Context(),
		wrapper,
		"n",
		"reader",
		[]string{"dense"},
	)
	loaded, loadErr := store.Load(t.Context(), "n")
	// Assert: stale acquisition cannot resurrect compacted payload or register a pin.
	if !errors.Is(err, lifecycle.ErrConflict) || loadErr != nil || len(loaded.Pins) != 0 ||
		!loaded.Manifests[0].Retired {
		t.Fatal("stale pin restored retired metadata", err, loadErr)
	}
}

func TestPublicationPinUncertainCommitReplaysDurableHandle(t *testing.T) {
	// Arrange: store commits the pin but loses its acknowledgement.
	store, _ := pinStore(t)
	wrapper := &interleavingPinStore{Store: store, committed: true}
	// Act.
	_, uncertain := lifecycle.AcquirePublicationPin(
		t.Context(),
		wrapper,
		"n",
		"reader",
		[]string{"dense"},
	)
	registered, err := store.Load(t.Context(), "n")
	if err != nil {
		t.Fatal(err)
	}
	replay, err := lifecycle.AcquirePublicationPin(
		t.Context(),
		wrapper,
		"n",
		"reader",
		[]string{"dense"},
	)
	loaded, loadErr := store.Load(t.Context(), "n")
	// Assert: uncertain outcomes retain the cause and replay never writes again.
	if !errors.Is(uncertain, lifecycle.ErrOutcomeUnknown) || !errors.Is(uncertain, context.DeadlineExceeded) ||
		err != nil ||
		loadErr != nil ||
		len(loaded.Pins) != 1 ||
		loaded.Generation != registered.Generation ||
		replay.Reference() != registered.Pins[0].Publication {
		t.Fatal("uncertain registration cannot reconcile", uncertain, err, loadErr)
	}
}

func TestPublicationPinInputAndUnknownPublication(t *testing.T) {
	// Arrange.
	store, _ := pinStore(t)
	canceled, cancel := context.WithCancel(t.Context())
	cancel()
	// Act/Assert: invalid identities/profiles and cancellation precede mutation.
	for _, input := range []struct {
		namespace, id string
		targets       []string
	}{
		{"", "reader", []string{"dense"}}, {"n", "\xff", []string{"dense"}},
		{"n", "reader", nil}, {"n", "reader", []string{"dense", "dense"}}, {"n", "reader", []string{"\xff"}},
	} {
		if _, err := lifecycle.AcquirePublicationPin(
			t.Context(),
			store,
			input.namespace,
			input.id,
			input.targets,
		); !errors.Is(
			err,
			ragy.ErrInvalidArgument,
		) {
			t.Fatal("invalid pin accepted", err)
		}
	}
	if _, err := lifecycle.AcquirePublicationPin(canceled, store, "n", "reader", []string{"dense"}); !errors.Is(
		err,
		context.Canceled,
	) {
		t.Fatal("canceled acquisition", err)
	}
	if err := lifecycle.ReleasePublicationPin(t.Context(), store, "n", "missing"); !errors.Is(
		err,
		ragy.ErrUnavailable,
	) {
		t.Fatal("missing release", err)
	}
	var absent *filestore.Store
	if _, err := lifecycle.AcquirePublicationPin(t.Context(), absent, "n", "reader", []string{"dense"}); !errors.Is(
		err,
		ragy.ErrInvalidArgument,
	) {
		t.Fatal("typed nil store accepted", err)
	}
	if err := lifecycle.ReleasePublicationPin(canceled, store, "n", "reader"); !errors.Is(
		err,
		context.Canceled,
	) {
		t.Fatal("canceled release", err)
	}
	if _, err := lifecycle.AcquirePublicationPin(t.Context(), store, "n", "reader", []string{"missing"}); !errors.Is(
		err,
		ragy.ErrUnsupported,
	) {
		t.Fatal("unavailable target registered", err)
	}
	loaded, err := store.Load(t.Context(), "n")
	if err != nil {
		t.Fatal(err)
	}
	loaded.Manifests[0].State, loaded.Manifests[0].Checkpoint = lifecycle.Unknown, lifecycle.Published
	if _, err = store.CompareSwap(t.Context(), loaded.Generation, loaded); err != nil {
		t.Fatal(err)
	}
	if _, err = lifecycle.AcquirePublicationPin(t.Context(), store, "n", "reader", []string{"dense"}); !errors.Is(
		err,
		ragy.ErrUnsupported,
	) {
		t.Fatal("unknown publication registered", err)
	}
}
