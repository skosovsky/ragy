//go:build darwin || linux

package lifecycle_test

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/lifecycle/filestore"
)

func cleanedHistorySnapshot() lifecycle.Snapshot {
	old := manifestFixture()
	owner := manifestFixture()
	owner.ID, owner.Key, owner.ExpectedPublication = "deleted", "delete-key", old.ID
	owner.Targets = nil
	owner.Tombstone = true
	owner.State = lifecycle.Complete
	job := lifecycle.CleanupJob{
		Owner:     owner.ID,
		StartedAt: owner.PublishedAt,
		Deadline:  owner.PublishedAt.Add(time.Hour),
		Complete:  true,
	}
	for _, target := range old.Targets {
		job.Items = append(
			job.Items,
			lifecycle.RetiredTarget{
				Manifest: old.ID,
				Target:   target.Name,
				State:    lifecycle.CleanupDone,
				Attempts: 1,
				NextAt:   owner.PublishedAt,
			},
		)
	}
	return lifecycle.Snapshot{
		Schema:       lifecycle.SchemaIdentity,
		Namespace:    "n",
		Manifests:    []lifecycle.Manifest{old, owner},
		Publications: []lifecycle.Publication{{Source: "policy", Manifest: owner.ID}},
		Cleanups:     []lifecycle.CleanupJob{job},
	}
}

func TestCompactionRetainsIdentityAncestryFencesAndOwnsOutput(t *testing.T) {
	// Arrange: target cleanup has already confirmed both exact historical inventories.
	before := cleanedHistorySnapshot()
	beforeBytes, _ := json.Marshal(before)
	// Act.
	after, err := lifecycle.CompactHistory(before, []string{"pub1"})
	afterBytes, _ := json.Marshal(after)
	// Assert: compaction invalidates the old handle but preserves deletion/checkpoint proofs.
	if err != nil || after.Validate() != nil || !after.Manifests[0].Retired ||
		len(after.Manifests[0].ArtifactFences) != 1 ||
		len(after.Manifests[0].Targets[0].Artifacts) != 0 ||
		len(afterBytes) >= len(beforeBytes) {
		t.Fatal("unsafe or ineffective compaction", err, len(beforeBytes), len(afterBytes))
	}
	if after.Manifests[0].ID != before.Manifests[0].ID ||
		after.Manifests[0].Key != before.Manifests[0].Key ||
		!reflect.DeepEqual(after.Cleanups, before.Cleanups) ||
		after.Manifests[1].ExpectedPublication != "pub1" {
		t.Fatal("identity or cleanup fences lost")
	}
	after.Manifests[0].Targets[0].Name = "mutated"
	after.Cleanups[0].Items[0].Target = "mutated"
	if before.Manifests[0].Targets[0].Name != "dense" ||
		before.Cleanups[0].Items[0].Target != "dense" {
		t.Fatal("compaction aliases host input")
	}
}

func TestRetirementRejectsProtectedStatesAndMalformedSelection(t *testing.T) {
	cases := map[string]struct {
		mutate func(*lifecycle.Snapshot)
		id     string
		want   error
	}{
		"current publication": {func(*lifecycle.Snapshot) {}, "deleted", lifecycle.ErrProtected},
		"no cleanup proof": {
			func(s *lifecycle.Snapshot) { s.Cleanups = nil; s.Manifests[1].State = lifecycle.Published },
			"pub1",
			lifecycle.ErrProtected,
		},
		"unknown stage": {func(s *lifecycle.Snapshot) {
			s.Manifests[0].State = lifecycle.Unknown
			s.Manifests[0].Checkpoint = lifecycle.Published
		}, "pub1", lifecycle.ErrProtected},
		"unknown target": {func(s *lifecycle.Snapshot) {
			s.Manifests[0].Partial = true
			s.Manifests[0].Targets[0].State = lifecycle.TargetUnknown
			s.Manifests[0].Targets[0].Revision = ""
		}, "pub1", lifecycle.ErrProtected},
		"unfinished cleanup": {func(s *lifecycle.Snapshot) {
			s.Cleanups[0].Complete = false
			s.Cleanups[0].Items[1].State = lifecycle.CleanupUnknown
			s.Manifests[1].State = lifecycle.CleanupPending
		}, "pub1", lifecycle.ErrProtected},
		"active plan ancestor": {func(s *lifecycle.Snapshot) {
			plan := plannedManifest()
			plan.ID, plan.Key, plan.ExpectedPublication = "active", "active-key", "deleted"
			s.Manifests = append(s.Manifests, plan)
		}, "pub1", lifecycle.ErrProtected},
		"live pin": {func(s *lifecycle.Snapshot) {
			s.Pins = []lifecycle.PublicationPin{
				{
					ID:               "pin",
					Publication:      "captured",
					RequestedTargets: []string{"dense"},
					Targets: []access.TargetRevision{
						{
							Target:            "dense",
							Namespace:         "n",
							Source:            "policy",
							Revision:          "r1",
							Transformation:    "chunk",
							AccessFingerprint: "acl",
						},
					},
				},
			}
		}, "pub1", lifecycle.ErrProtected},
		"absent": {func(*lifecycle.Snapshot) {}, "missing", ragy.ErrUnavailable},
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			// Arrange.
			snapshot := cleanedHistorySnapshot()
			c.mutate(&snapshot)
			if err := snapshot.Validate(); err != nil {
				t.Fatal("invalid test fixture", err)
			}
			encoded, _ := json.Marshal(snapshot)
			// Act.
			_, err := lifecycle.CompactHistory(snapshot, []string{c.id})
			retained, _ := json.Marshal(snapshot)
			// Assert.
			if !errors.Is(err, c.want) || string(retained) != string(encoded) {
				t.Fatal("protection or input ownership failed", err)
			}
		})
	}
	// Arrange, Act, Assert: an exact host selection must be nonempty and unique.
	for _, ids := range [][]string{nil, {""}, {"pub1", "pub1"}} {
		if _, err := lifecycle.CompactHistory(cleanedHistorySnapshot(), ids); !errors.Is(
			err,
			ragy.ErrInvalidArgument,
		) {
			t.Fatal("invalid selection accepted", err)
		}
	}
}

func TestRetiredHandlesAndArtifactIdentitiesCannotBeReused(t *testing.T) {
	// Arrange: real durable compaction with an executor whose old plan can be replayed.
	ctx := context.Background()
	store, err := filestore.New(t.TempDir(), 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := store.CompareSwap(ctx, 0, cleanedHistorySnapshot())
	if err != nil {
		t.Fatal(err)
	}
	retired, err := store.Maintain(
		ctx,
		snapshot.Generation,
		lifecycle.RetirementRequest{Namespace: "n", Manifests: []string{"pub1"}},
	)
	if err != nil {
		t.Fatal(err)
	}
	host := &stageHost{}
	executor, err := lifecycle.NewExecutor(
		lifecycle.ExecutorConfig[string]{
			Store: store,
			Targets: []lifecycle.Registration[string]{
				{Name: "dense", Port: host},
				{Name: "tensor", Port: host},
			},
			ClonePayload:    func(p string) (string, error) { return p, nil },
			ValidatePayload: func(lifecycle.Manifest, string) error { return nil },
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	old := manifestFixture()
	old.State = lifecycle.Planned
	old.PublishedAt = time.Time{}
	for i := range old.Targets {
		old.Targets[i].State = lifecycle.TargetPending
		old.Targets[i].Revision = ""
	}
	// Act, Assert: every operational use of the old handle explicitly fails.
	if _, err = executor.Prepare(ctx, old); !errors.Is(err, lifecycle.ErrRetired) {
		t.Fatal("retired prepare replay", err)
	}
	if _, err = executor.Publish(ctx, "n", "pub1"); !errors.Is(err, lifecycle.ErrRetired) {
		t.Fatal("retired publication replay", err)
	}
	if _, err = executor.Stage(ctx, "n", "pub1", "dense", "payload"); !errors.Is(
		err,
		lifecycle.ErrRetired,
	) {
		t.Fatal("retired stage replay", err)
	}
	if _, err = executor.Reconcile(ctx, "n", "pub1", "dense"); !errors.Is(
		err,
		lifecycle.ErrRetired,
	) {
		t.Fatal("retired reconcile replay", err)
	}
	old.ID, old.Key, old.ExpectedPublication = "new-handle", "new-key", "deleted"
	if _, err = executor.Prepare(ctx, old); !errors.Is(err, lifecycle.ErrConflict) {
		t.Fatal("exact artifact identity reused after compaction", err)
	}
	// Arrange, Act, Assert: direct storage writes cannot discard or change reservations.
	for _, mutate := range []func(*lifecycle.Snapshot){
		func(s *lifecycle.Snapshot) { s.Manifests = s.Manifests[1:] },
		func(s *lifecycle.Snapshot) { s.Manifests[0].Key = "other-key" },
		func(s *lifecycle.Snapshot) { s.Manifests[0].ArtifactFences = nil },
	} {
		candidate, loadErr := store.Load(ctx, "n")
		if loadErr != nil {
			t.Fatal(loadErr)
		}
		mutate(&candidate)
		if _, err = store.CompareSwap(ctx, retired.Generation, candidate); err == nil {
			t.Fatal("direct CAS dropped retirement reservation")
		}
	}
}

func TestNewCleanupFollowsRetiredAncestryWithoutRedispatch(t *testing.T) {
	// Arrange: old inventory already compacted, then a newer tombstone publishes.
	ctx := context.Background()
	store, err := filestore.New(t.TempDir(), 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := store.CompareSwap(ctx, 0, cleanedHistorySnapshot())
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err = store.Maintain(
		ctx,
		snapshot.Generation,
		lifecycle.RetirementRequest{Namespace: "n", Manifests: []string{"pub1"}},
	)
	if err != nil {
		t.Fatal(err)
	}
	owner := snapshot.Manifests[1]
	owner.ID, owner.Key, owner.ExpectedPublication = "next", "next-key", "deleted"
	owner.State = lifecycle.Published
	owner.PublishedAt = owner.PublishedAt.Add(time.Minute)
	snapshot.Manifests = append(snapshot.Manifests, owner)
	snapshot.Publications[0].Manifest = owner.ID
	if _, err = store.CompareSwap(ctx, snapshot.Generation, snapshot); err != nil {
		t.Fatal(err)
	}
	now := owner.PublishedAt
	host := &cleanupHost{}
	cleaner := newCleaner(t, store, &now, host)
	// Act.
	job, err := cleaner.Begin(ctx, "n", owner.ID)
	// Assert: retained ancestry validates; already cleaned retired inventory is omitted.
	if err != nil || !job.Complete || len(job.Items) != 0 || host.calls != 0 ||
		host.inspected != 0 {
		t.Fatal("retired inventory reentered destructive cleanup", err, job)
	}
	current, err := store.Load(ctx, "n")
	if err != nil || current.Validate() != nil || len(current.Cleanups) != 2 ||
		!current.Manifests[0].Retired {
		t.Fatal("new cleanup lost retained proof", err)
	}
}

func TestDirectCASCannotForgetOrRebindUnretiredHistory(t *testing.T) {
	for name, mutate := range map[string]func(*lifecycle.Snapshot){
		"drop cleaned history": func(s *lifecycle.Snapshot) {
			s.Manifests = s.Manifests[1:]
			s.Cleanups = nil
			s.Manifests[0].ExpectedPublication = ""
		},
		"rebind source":             func(s *lifecycle.Snapshot) { s.Manifests[0].Identity.Content = "different-data" },
		"reuse key":                 func(s *lifecycle.Snapshot) { s.Manifests[0].Key = "replacement-key" },
		"drop exact artifact fence": func(s *lifecycle.Snapshot) { s.Manifests[0].Targets[0].Artifacts = nil },
	} {
		t.Run(name, func(t *testing.T) {
			// Arrange: historical cleanup completed, but the host has not selected retirement.
			ctx := context.Background()
			store, err := filestore.New(t.TempDir(), 8<<20)
			if err != nil {
				t.Fatal(err)
			}
			snapshot, err := store.CompareSwap(ctx, 0, cleanedHistorySnapshot())
			if err != nil {
				t.Fatal(err)
			}
			mutate(&snapshot)
			// Act.
			_, err = store.CompareSwap(ctx, snapshot.Generation, snapshot)
			retained, loadErr := store.Load(ctx, "n")
			// Assert: a raw CAS cannot bypass retirement/reference reservations.
			if err == nil || loadErr != nil || retained.Generation != 1 || len(retained.Manifests) != 2 ||
				retained.Manifests[0].Key != "key" ||
				len(retained.Manifests[0].Targets[0].Artifacts) != 1 {
				t.Fatal("unretired history forgotten/rebound", err, loadErr)
			}
		})
	}
}

func TestDirectCASPreservesCleanupAndBootstrapReceipts(t *testing.T) {
	for name, mutate := range map[string]func(*lifecycle.Snapshot){
		"drop cleanup":        func(s *lifecycle.Snapshot) { s.Cleanups = nil },
		"rebind cleanup item": func(s *lifecycle.Snapshot) { s.Cleanups[0].Items[0].Attempts = 0 },
		"drop watermark":      func(s *lifecycle.Snapshot) { s.Inventories = nil },
		"rebind watermark":    func(s *lifecycle.Snapshot) { s.Inventories[0].Fingerprint = "different" },
	} {
		t.Run(name, func(t *testing.T) {
			// Arrange.
			ctx := context.Background()
			store, err := filestore.New(t.TempDir(), 8<<20)
			if err != nil {
				t.Fatal(err)
			}
			input := cleanedHistorySnapshot()
			input.Inventories = []lifecycle.InventoryReceipt{
				{
					Kind:        lifecycle.DeltaInventory,
					Watermark:   "receipt",
					Fingerprint: "input",
					Coverage:    lifecycle.FullInventory,
					Targets:     []string{"dense"},
					Imported:    []string{"pub1"},
				},
			}
			snapshot, err := store.CompareSwap(ctx, 0, input)
			if err != nil {
				t.Fatal(err)
			}
			mutate(&snapshot)
			// Act.
			_, err = store.CompareSwap(ctx, snapshot.Generation, snapshot)
			retained, loadErr := store.Load(ctx, "n")
			// Assert.
			if err == nil || loadErr != nil || retained.Generation != 1 || len(retained.Cleanups) != 1 ||
				len(retained.Inventories) != 1 ||
				retained.Inventories[0].Fingerprint != "input" {
				t.Fatal("cleanup/bootstrap reservation lost", err, loadErr)
			}
		})
	}
}

func TestDirectCASCannotReuseRetiredArtifactFenceUnderNewHandle(t *testing.T) {
	// Arrange: actual retirement cleared inventory payload but kept its digest fences.
	ctx := context.Background()
	store, err := filestore.New(t.TempDir(), 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := store.CompareSwap(ctx, 0, cleanedHistorySnapshot())
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err = store.Maintain(
		ctx,
		snapshot.Generation,
		lifecycle.RetirementRequest{Namespace: "n", Manifests: []string{"pub1"}},
	)
	if err != nil {
		t.Fatal(err)
	}
	candidate := manifestFixture()
	candidate.ID, candidate.Key, candidate.ExpectedPublication = "different-handle", "different-key", "deleted"
	candidate.State = lifecycle.Planned
	candidate.PublishedAt = time.Time{}
	for i := range candidate.Targets {
		candidate.Targets[i].State = lifecycle.TargetPending
		candidate.Targets[i].Revision = ""
	}
	snapshot.Manifests = append(snapshot.Manifests, candidate)
	// Act.
	_, err = store.CompareSwap(ctx, snapshot.Generation, snapshot)
	// Assert: raw storage mutation also obeys the exact reserved target/reference identity.
	if !errors.Is(err, lifecycle.ErrConflict) {
		t.Fatal("direct CAS reused compacted artifact identity", err)
	}
}
