//go:build darwin || linux

package filestore_test

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/lifecycle/filestore"
	"github.com/skosovsky/ragy/source"
)

func TestSnapshotBudgetRejectsOversizedWriteWithoutReplacingPublication(t *testing.T) {
	// Arrange.
	input := fixture()
	input.Generation = 1
	data, err := json.Marshal(input)
	if err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	store, err := filestore.New(root, int64(len(data)))
	if err != nil {
		t.Fatal(err)
	}
	committed, err := store.CompareSwap(context.Background(), 0, fixture())
	if err != nil {
		t.Fatal(err)
	}
	additional := fixture().Manifests[0]
	additional.ID, additional.Key = "publication-2", "request-2"
	additional.Identity.Source = "other-policy"
	additional.Payload = strings.Repeat("x", len(data))
	additional.Targets[0].Artifacts[0].Reference.Source = "other-policy"
	additional.Targets[0].Artifacts[0].Supports[0].Source = "other-policy"
	committed.Manifests = append(committed.Manifests, additional)
	// Act.
	_, err = store.CompareSwap(context.Background(), 1, committed)
	loaded, loadErr := store.Load(context.Background(), "n")
	// Assert.
	if !errors.Is(err, lifecycle.ErrCapacity) || loadErr != nil || loaded.Generation != 1 ||
		loaded.Manifests[0].Payload != "payload-1" {
		t.Fatal("oversized write replaced known publication", err, loadErr)
	}
	smaller, err := filestore.New(root, int64(len(data)-1))
	if err != nil {
		t.Fatal(err)
	}
	if _, err = smaller.Load(context.Background(), "n"); !errors.Is(err, lifecycle.ErrCapacity) {
		t.Fatal("oversized persisted state was accepted", err)
	}
}

func TestSnapshotBudgetInvalidBeforeFilesystemIO(t *testing.T) {
	// Arrange.
	root := filepath.Join(t.TempDir(), "must-not-exist")
	// Act and Assert.
	for _, limit := range []int64{0, -1, int64(^uint64(0) >> 1)} {
		if _, err := filestore.New(root, limit); !errors.Is(err, ragy.ErrInvalidArgument) {
			t.Fatal("invalid budget accepted", limit, err)
		}
	}
	if _, err := os.Stat(root); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("invalid budget performed filesystem IO", err)
	}
}

func fixture() lifecycle.Snapshot {
	reference := source.Reference{
		Namespace:         "n",
		Source:            "policy",
		Revision:          "r1",
		Transformation:    "chunk",
		AccessFingerprint: "acl",
		Artifact:          "p1",
		Representation:    "text",
	}
	return lifecycle.Snapshot{
		Schema: lifecycle.SchemaIdentity, Namespace: "n",
		Manifests: []lifecycle.Manifest{
			{
				ID: "publication-1",
				Identity: lifecycle.Identity{
					Namespace:      "n",
					Source:         "policy",
					Revision:       "r1",
					Content:        "content",
					Transformation: "chunk",
					Access:         "acl",
				},
				Key:         "request-1",
				Payload:     "payload-1",
				State:       lifecycle.Published,
				PublishedAt: time.Unix(100, 0).UTC(),
				Targets: []lifecycle.Target{
					{
						Name:     "lexical",
						Required: true,
						State:    lifecycle.TargetReady,
						Revision: "r1",
						Artifacts: []lifecycle.Artifact{
							{Reference: reference, Supports: []source.Reference{reference}},
						},
					},
				},
			},
		},
		Publications: []lifecycle.Publication{{Source: "policy", Manifest: "publication-1"}},
	}
}

func TestDurableCASRestartOwnershipAndStaleWriter(t *testing.T) {
	// Arrange.
	root := t.TempDir()
	store, err := filestore.New(root, 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	input := fixture()
	// Act.
	committed, err := store.CompareSwap(ctx, 0, input)
	if err != nil {
		t.Fatal(err)
	}
	input.Manifests[0].Targets[0].Artifacts[0].Supports[0].Artifact = "mutated-input"
	committed.Manifests[0].Targets[0].Artifacts[0].Supports[0].Artifact = "mutated-output"
	restarted, err := filestore.New(root, 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	loaded, err := restarted.Load(ctx, "n")
	// Assert.
	if err != nil || loaded.Generation != 1 ||
		loaded.Manifests[0].Targets[0].Artifacts[0].Supports[0].Artifact != "p1" {
		t.Fatal("durable snapshot was lost or aliased")
	}
	if _, err = store.CompareSwap(ctx, 0, fixture()); !errors.Is(err, lifecycle.ErrConflict) {
		t.Fatal("stale writer overwrote publication")
	}
	loaded.Manifests[0].State = lifecycle.CleanupPending
	updated, err := store.CompareSwap(ctx, 1, loaded)
	if err != nil || updated.Generation != 2 {
		t.Fatal("checkpoint CAS failed", err)
	}
}

func TestConcurrentIndependentStoresOnlyOneGenerationWins(t *testing.T) {
	// Arrange: independent store objects contend on the same durable namespace.
	root := t.TempDir()
	const writers = 12
	var wait sync.WaitGroup
	outcomes := make(chan error, writers)
	// Act.
	for range writers {
		wait.Go(func() {
			store, err := filestore.New(root, 8<<20)
			if err == nil {
				_, err = store.CompareSwap(context.Background(), 0, fixture())
			}
			outcomes <- err
		})
	}
	wait.Wait()
	close(outcomes)
	// Assert.
	winners := 0
	for err := range outcomes {
		if err == nil {
			winners++
		} else if !errors.Is(err, lifecycle.ErrConflict) {
			t.Fatal(err)
		}
	}
	if winners != 1 {
		t.Fatalf("got %d winners", winners)
	}
}

func TestCorruptionAndCanceledCASNeverEraseKnownState(t *testing.T) {
	// Arrange.
	root := t.TempDir()
	store, err := filestore.New(root, 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = store.CompareSwap(context.Background(), 0, fixture()); err != nil {
		t.Fatal(err)
	}
	paths, err := filepath.Glob(filepath.Join(root, "*.json"))
	if err != nil || len(paths) != 1 {
		t.Fatal("missing snapshot", err)
	}
	corrupt := []byte(`{"schema":"unknown"}`)
	if err = os.WriteFile(paths[0], corrupt, 0o600); err != nil {
		t.Fatal(err)
	}
	// Act/Assert.
	if _, err = store.Load(context.Background(), "n"); !errors.Is(err, ragy.ErrProtocol) {
		t.Fatal("corruption treated as absence")
	}
	if _, err = store.CompareSwap(context.Background(), 0, fixture()); !errors.Is(err, ragy.ErrProtocol) {
		t.Fatal("corrupt inventory replaced")
	}
	retained, err := os.ReadFile(paths[0])
	if err != nil || string(retained) != string(corrupt) {
		t.Fatal("corrupt inventory erased")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err = store.CompareSwap(ctx, 0, fixture()); !errors.Is(err, context.Canceled) {
		t.Fatal("canceled CAS performed IO")
	}
}

func TestDurableSnapshotAcrossProcessRestart(t *testing.T) {
	if root := os.Getenv("RAGY_LIFECYCLE_CHILD_ROOT"); root != "" {
		store, err := filestore.New(root, 8<<20)
		if err != nil {
			t.Fatal(err)
		}
		snapshot, err := store.Load(context.Background(), "n")
		if err != nil || snapshot.Generation != 1 {
			t.Fatal("child cannot load committed state", err)
		}
		snapshot.Manifests[0].State = lifecycle.CleanupPending
		if _, err = store.CompareSwap(context.Background(), 1, snapshot); err != nil {
			t.Fatal(err)
		}
		return
	}
	// Arrange: a source publication is committed before a fresh process starts.
	root := t.TempDir()
	store, err := filestore.New(root, 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = store.CompareSwap(context.Background(), 0, fixture()); err != nil {
		t.Fatal(err)
	}
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	// Act: another process loads and advances only the cleanup checkpoint.
	command := exec.CommandContext(t.Context(), executable, "-test.run=^TestDurableSnapshotAcrossProcessRestart$")
	command.Env = []string{"RAGY_LIFECYCLE_CHILD_ROOT=" + root}
	if output, runErr := command.CombinedOutput(); runErr != nil {
		t.Fatalf("child: %v %s", runErr, output)
	}
	snapshot, err := store.Load(context.Background(), "n")
	// Assert: both publication and updated checkpoint survive process termination.
	if err != nil || snapshot.Generation != 2 || snapshot.Manifests[0].State != lifecycle.CleanupPending ||
		snapshot.Publications[0].Manifest != "publication-1" {
		t.Fatal("process restart lost checkpoint/publication", err)
	}
}
