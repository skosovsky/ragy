//go:build darwin || linux

package final_test

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/lexical"
	lexicalmanaged "github.com/skosovsky/ragy/lexical/managed"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/lifecycle/filestore"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

func lifecyclePlan(id, revision string) lifecycle.Manifest {
	ref := source.Reference{Namespace: "consumer", Source: "policy", Revision: revision,
		Transformation: "chunk", AccessFingerprint: "acl", Artifact: "p1", Representation: "utf8"}
	return lifecycle.Manifest{
		ID: id, Key: id, Payload: id + "-payload", State: lifecycle.Planned,
		Identity: lifecycle.Identity{Namespace: ref.Namespace, Source: ref.Source, Revision: revision,
			Content: id + "-content", Transformation: ref.Transformation, Access: ref.AccessFingerprint},
		Targets: []lifecycle.Target{{Name: "lexical", Required: true, State: lifecycle.TargetPending,
			Artifacts: []lifecycle.Artifact{{Reference: ref, Supports: []source.Reference{ref}}}}},
	}
}

func lifecycleHistory() lifecycle.Snapshot {
	old, current := lifecyclePlan("old", "r1"), lifecyclePlan("current", "r2")
	for _, m := range []*lifecycle.Manifest{&old, &current} {
		m.State = lifecycle.Complete
		m.PublishedAt = time.Unix(100, 0).UTC()
		m.Targets[0].State, m.Targets[0].Revision = lifecycle.TargetReady, m.Identity.Revision
	}
	current.ExpectedPublication = old.ID
	current.PublishedAt = current.PublishedAt.Add(time.Second)
	return lifecycle.Snapshot{Schema: lifecycle.SchemaIdentity, Namespace: "consumer",
		Manifests:    []lifecycle.Manifest{old, current},
		Publications: []lifecycle.Publication{{Source: "policy", Manifest: current.ID}},
		Cleanups: []lifecycle.CleanupJob{{Owner: current.ID, StartedAt: current.PublishedAt,
			Deadline: current.PublishedAt.Add(time.Minute), Complete: true,
			Items: []lifecycle.RetiredTarget{{Manifest: old.ID, Target: "lexical", State: lifecycle.CleanupDone,
				Attempts: 1, NextAt: current.PublishedAt}}}},
	}
}

func lifecycleStore(t *testing.T, root string) *filestore.Store {
	t.Helper()
	store, err := filestore.New(root, 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	return store
}

func lifecycleLoad(t *testing.T, store lifecycle.Store) lifecycle.Snapshot {
	t.Helper()
	snapshot, err := store.Load(t.Context(), "consumer")
	if err != nil {
		t.Fatal(err)
	}
	return snapshot
}

func TestLifecycleExternalMaintenanceFencesAndOwnership(t *testing.T) {
	// Arrange: real durable history, with confirmed exact-inventory cleanup.
	root := t.TempDir()
	store := lifecycleStore(t, root)
	input := lifecycleHistory()
	committed, err := store.CompareSwap(t.Context(), 0, input)
	if err != nil {
		t.Fatal(err)
	}
	input.Manifests[0].Targets[0].Artifacts[0].Supports[0].Artifact = "caller-input"
	committed.Manifests[0].Targets[0].Artifacts[0].Supports[0].Artifact = "caller-output"
	before := lifecycleLoad(t, store)
	if before.Manifests[0].Targets[0].Artifacts[0].Supports[0].Artifact != "p1" {
		t.Fatal("CAS inventory aliases consumer input or output")
	}
	refBytes, err := json.Marshal(before.Manifests[0].Targets[0].Artifacts[0].Reference)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(refBytes)

	// Act: maintenance commits through the same generation fence as raw CAS.
	retired, err := store.Maintain(t.Context(), before.Generation,
		lifecycle.RetirementRequest{Namespace: "consumer", Manifests: []string{"old"}})
	if err != nil {
		t.Fatal(err)
	}
	if !retired.Manifests[0].Retired || len(retired.Manifests[0].Targets[0].Artifacts) != 0 ||
		len(retired.Manifests[0].ArtifactFences) != 1 ||
		retired.Manifests[0].ArtifactFences[0].Digest != hex.EncodeToString(digest[:]) {
		t.Fatal("maintenance did not reserve exact original inventory")
	}
	retired.Manifests[0].ArtifactFences[0].Digest = strings.Repeat("0", 64)
	restarted := lifecycleStore(t, root)
	baseline := lifecycleLoad(t, restarted)

	// Assert: raw writes cannot bypass retirement; each failure preserves all state.
	if baseline.Generation != before.Generation+1 ||
		baseline.Manifests[0].ArtifactFences[0].Digest != hex.EncodeToString(digest[:]) {
		t.Fatal("retirement was lost or returned nested fences alias durable state")
	}
	for _, attack := range []struct {
		name   string
		change func(*lifecycle.Snapshot)
	}{
		{"drop retired handle", func(s *lifecycle.Snapshot) {
			s.Manifests = s.Manifests[1:]
			s.Manifests[0].ExpectedPublication = ""
			s.Cleanups = nil
		}},
		{"rebind retired payload", func(s *lifecycle.Snapshot) { s.Manifests[0].Payload = "rebound" }},
		{"drop artifact fence", func(s *lifecycle.Snapshot) { s.Manifests[0].ArtifactFences = nil }},
	} {
		t.Run(attack.name, func(t *testing.T) {
			next := lifecycleLoad(t, restarted)
			attack.change(&next)
			_, casErr := restarted.CompareSwap(t.Context(), baseline.Generation, next)
			if !errors.Is(casErr, lifecycle.ErrRetired) || !reflect.DeepEqual(baseline, lifecycleLoad(t, restarted)) {
				t.Fatal("raw CAS bypassed reserved history", casErr)
			}
		})
	}
	if _, err = restarted.CompareSwap(t.Context(), before.Generation, before); !errors.Is(err, lifecycle.ErrConflict) {
		t.Fatal("stale pre-maintenance CAS resurrected inventory", err)
	}
	if _, err = restarted.Maintain(
		t.Context(),
		before.Generation,
		lifecycle.RetirementRequest{
			Namespace: "consumer",
			Manifests: []string{"old"},
		},
	); !errors.Is(err, lifecycle.ErrConflict) {
		t.Fatal("stale maintenance crossed generation fence", err)
	}
	if _, err = restarted.Maintain(
		t.Context(),
		baseline.Generation,
		lifecycle.RetirementRequest{
			Namespace: "consumer",
			Manifests: []string{"current"},
		},
	); !errors.Is(err, lifecycle.ErrProtected) {
		t.Fatal("active publication retired", err)
	}
	assertLifecycleArtifactReservation(t, restarted, baseline)
	other, err := restarted.Load(t.Context(), "other-consumer")
	if err != nil || other.Generation != 0 || len(other.Manifests) != 0 {
		t.Fatal("namespace inventory leaked", err)
	}
}

func assertLifecycleArtifactReservation(t *testing.T, store lifecycle.Store, before lifecycle.Snapshot) {
	t.Helper()
	// Arrange: a new handle attempts to reuse the compacted exact target/reference.
	next := lifecycleLoad(t, store)
	rebound := lifecyclePlan("rebound", "r1")
	rebound.ExpectedPublication = "current"
	next.Manifests = append(next.Manifests, rebound)
	if err := next.Validate(); err != nil {
		t.Fatal("invalid reference reservation attack", err)
	}
	// Act.
	_, err := store.CompareSwap(t.Context(), before.Generation, next)
	// Assert: the compacted digest still fences new inventory reservations.
	if !errors.Is(err, lifecycle.ErrConflict) || !reflect.DeepEqual(before, lifecycleLoad(t, store)) {
		t.Fatal("compacted artifact reference rebound under a new handle", err)
	}
}

func TestLifecycleExternalBatchAtomicityAndCapacity(t *testing.T) {
	for _, existing := range []bool{false, true} {
		t.Run(map[bool]string{false: "empty", true: "existing"}[existing], func(t *testing.T) {
			// Arrange: distinct payloads attempt to reserve one exact target/reference.
			store := lifecycleStore(t, t.TempDir())
			if existing {
				prior := lifecyclePlan("prior", "r0")
				if _, err := store.CompareSwap(t.Context(), 0, lifecycle.Snapshot{Schema: lifecycle.SchemaIdentity,
					Namespace: "consumer", Manifests: []lifecycle.Manifest{prior}}); err != nil {
					t.Fatal(err)
				}
			}
			before := lifecycleLoad(t, store)
			next := lifecycleLoad(t, store)
			next.Manifests = append(next.Manifests, lifecyclePlan("first", "r1"), lifecyclePlan("second", "r1"))
			if err := next.Validate(); err != nil {
				t.Fatal("individually invalid plans", err)
			}
			// Act.
			_, err := store.CompareSwap(t.Context(), before.Generation, next)
			// Assert.
			if !errors.Is(err, lifecycle.ErrConflict) || !reflect.DeepEqual(before, lifecycleLoad(t, store)) {
				t.Fatal("conflicting batch partially persisted", err)
			}
		})
	}
}

func TestLifecycleExternalCapacity(t *testing.T) {
	// Arrange: capacity is exact committed wire bytes, including incremented generation.
	root := t.TempDir()
	input := lifecycleHistory()
	input.Generation = 1
	wire, err := json.Marshal(input)
	if err != nil {
		t.Fatal(err)
	}
	bounded, err := filestore.New(root, int64(len(wire)))
	if err != nil {
		t.Fatal(err)
	}
	input.Generation = 0
	before, err := bounded.CompareSwap(t.Context(), 0, input)
	if err != nil {
		t.Fatal(err)
	}
	next := lifecycleLoad(t, bounded)
	extra := lifecyclePlan("large", "r3")
	extra.Payload = strings.Repeat("x", len(wire))
	extra.ExpectedPublication = "current"
	next.Manifests = append(next.Manifests, extra)
	// Act.
	_, capacityErr := bounded.CompareSwap(t.Context(), before.Generation, next)
	// Assert.
	if !errors.Is(capacityErr, lifecycle.ErrCapacity) || !reflect.DeepEqual(before, lifecycleLoad(t, bounded)) {
		t.Fatal("oversized write replaced committed state", capacityErr)
	}
	tooSmall, err := filestore.New(root, int64(len(wire)-1))
	if err != nil {
		t.Fatal(err)
	}
	if _, err = tooSmall.Maintain(
		t.Context(),
		before.Generation,
		lifecycle.RetirementRequest{
			Namespace: "consumer",
			Manifests: []string{"old"},
		},
	); !errors.Is(err, lifecycle.ErrCapacity) {
		t.Fatal("maintenance read exceeded explicit byte bound", err)
	}
}

// Host metadata and payload have no dependency on a library-owned domain model.
type lifecycleConsumerMeta struct {
	Label string `json:"label"`
}
type lifecycleConsumerPayload = []lexicalmanaged.Record[lifecycleConsumerMeta]

func lifecycleComposition(
	t *testing.T,
) (string, *filestore.Store, *lexicalmanaged.Adapter[lifecycleConsumerMeta], *lifecycle.Executor[lifecycleConsumerPayload]) {
	t.Helper()
	// Arrange: compose actual filestore, managed BM25, Executor and Cleaner.
	root := t.TempDir()
	store := lifecycleStore(t, root)
	schema, err := filter.NewSchema().Build()
	if err != nil {
		t.Fatal(err)
	}
	adapter, err := lexicalmanaged.New(lexicalmanaged.Config[lifecycleConsumerMeta]{
		Namespace:          "consumer",
		Target:             "lexical",
		Store:              store,
		Schema:             schema,
		BM25:               lexical.Config[lifecycleConsumerMeta]{SearchFields: []string{"content"}},
		CloneMeta:          func(m lifecycleConsumerMeta) (lifecycleConsumerMeta, error) { return m, nil },
		MaxCachedSnapshots: 4,
	})
	if err != nil {
		t.Fatal(err)
	}
	now := time.Unix(100, 0).UTC()
	executor, err := lifecycle.NewExecutor(lifecycle.ExecutorConfig[lifecycleConsumerPayload]{
		Store:        store,
		Now:          func() time.Time { return now },
		Targets:      []lifecycle.Registration[lifecycleConsumerPayload]{{Name: "lexical", Port: adapter}},
		ClonePayload: func(p lifecycleConsumerPayload) (lifecycleConsumerPayload, error) { return slices.Clone(p), nil },
		ValidatePayload: func(m lifecycle.Manifest, p lifecycleConsumerPayload) error {
			if len(p) != 1 || p[0].Reference != m.Targets[0].Artifacts[0].Reference ||
				p[0].Document.Content != m.Payload {
				return ragy.ErrInvalidArgument
			}
			return nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	return root, store, adapter, executor
}

func lifecyclePublish(t *testing.T, executor *lifecycle.Executor[lifecycleConsumerPayload], m lifecycle.Manifest) {
	t.Helper()
	if _, prepareErr := executor.Prepare(t.Context(), m); prepareErr != nil {
		t.Fatal(prepareErr)
	}
	payload := lifecycleConsumerPayload{
		{
			Reference: m.Targets[0].Artifacts[0].Reference,
			Document: retrieval.Document[lifecycleConsumerMeta]{
				ID:      "p1",
				Content: m.Payload,
				Meta:    lifecycleConsumerMeta{Label: m.ID},
			},
		},
	}
	if _, stageErr := executor.Stage(t.Context(), "consumer", m.ID, "lexical", payload); stageErr != nil {
		t.Fatal(stageErr)
	}
	if _, publishErr := executor.Publish(t.Context(), "consumer", m.ID); publishErr != nil {
		t.Fatal(publishErr)
	}
}

func TestLifecycleExternalRegisteredPinActualCleanupAndExplicitRetirement(t *testing.T) {
	// Arrange: actual backend composition with an explicitly registered reader.
	root, store, adapter, executor := lifecycleComposition(t)
	now := time.Unix(101, 0).UTC()
	old := lifecyclePlan("old", "r1")
	lifecyclePublish(t, executor, old)
	pin, err := lifecycle.AcquirePublicationPin(
		t.Context(),
		store,
		"consumer",
		"registered-reader",
		[]string{"lexical"},
	)
	if err != nil {
		t.Fatal(err)
	}
	current := lifecyclePlan("current", "r2")
	current.ExpectedPublication = old.ID
	lifecyclePublish(t, executor, current)
	cleaner, err := lifecycle.NewCleaner(lifecycle.CleanerConfig{Store: store, Now: func() time.Time { return now },
		Targets: []lifecycle.CleanupRegistration{{Name: "lexical", Port: adapter}},
		Policy:  lifecycle.CleanupPolicy{Deadline: time.Minute, Backoff: []time.Duration{time.Second}}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err = cleaner.Begin(t.Context(), "consumer", current.ID); err != nil {
		t.Fatal(err)
	}
	// Act: real cleanup completes; a registered pin still protects historical metadata.
	job, err := cleaner.Attempt(t.Context(), "consumer", current.ID, old.ID, "lexical", false)
	if err != nil || !job.Complete || len(job.Items) != 1 || job.Items[0].State != lifecycle.CleanupDone {
		t.Fatal("actual cleanup did not complete", job, err)
	}
	restarted := lifecycleStore(t, root)
	before := lifecycleLoad(t, restarted)
	replay, err := lifecycle.AcquirePublicationPin(
		t.Context(),
		restarted,
		"consumer",
		"registered-reader",
		[]string{"lexical"},
	)
	if err != nil || replay.Reference() != pin.Reference() || !slices.Equal(replay.Targets(), pin.Targets()) {
		t.Fatal("restart substituted current publication for registered historical pin", err)
	}
	_, protectedErr := restarted.Maintain(t.Context(), before.Generation,
		lifecycle.RetirementRequest{Namespace: "consumer", Manifests: []string{old.ID}})
	// Assert: cleanup is not metadata retirement or automatic pin release.
	if !errors.Is(protectedErr, lifecycle.ErrProtected) || !reflect.DeepEqual(before, lifecycleLoad(t, restarted)) {
		t.Fatal("live registered pin lost metadata protection", protectedErr)
	}
	changedTargets := replay.Targets()
	changedTargets[0].Revision = "caller-change"
	if pin.Targets()[0].Revision != "r1" {
		t.Fatal("returned pin targets alias inventory")
	}
	assertLifecycleExplicitReleaseAndRetirement(t, root, restarted, old)
}

func assertLifecycleExplicitReleaseAndRetirement(
	t *testing.T,
	root string,
	restarted *filestore.Store,
	old lifecycle.Manifest,
) {
	t.Helper()
	if err := lifecycle.ReleasePublicationPin(t.Context(), restarted, "consumer", "registered-reader"); err != nil {
		t.Fatal(err)
	}
	released := lifecycleLoad(t, restarted)
	if released.Manifests[0].Retired {
		t.Fatal("release automatically retired metadata")
	}
	if err := lifecycle.ReleasePublicationPin(t.Context(), restarted, "consumer", "registered-reader"); err != nil {
		t.Fatal(err)
	}
	if lifecycleLoad(t, restarted).Generation != released.Generation {
		t.Fatal("release replay mutated generation")
	}
	retired, err := restarted.Maintain(t.Context(), released.Generation,
		lifecycle.RetirementRequest{Namespace: "consumer", Manifests: []string{old.ID}})
	if err != nil || !retired.Manifests[0].Retired || retired.Manifests[1].Retired ||
		len(retired.Manifests[0].Targets[0].Artifacts) != 0 {
		t.Fatal("explicit old retirement failed or retired current publication", err)
	}
	retired.Pins = nil
	if _, err = restarted.CompareSwap(t.Context(), retired.Generation, retired); !errors.Is(err, lifecycle.ErrRetired) {
		t.Fatal("raw CAS dropped released pin reservation", err)
	}
	finalStore := lifecycleStore(t, root)
	if _, err = lifecycle.AcquirePublicationPin(
		t.Context(),
		finalStore,
		"consumer",
		"registered-reader",
		[]string{"lexical"},
	); !errors.Is(
		err,
		lifecycle.ErrRetired,
	) {
		t.Fatal("released pin ID reused after retirement and restart", err)
	}
}
