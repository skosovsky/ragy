//go:build darwin || linux

package filestore_test

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/lifecycle/filestore"
)

func historyFixture() lifecycle.Snapshot {
	snapshot := fixture()
	prior := snapshot.Manifests[0]
	snapshot.Manifests[0] = prior
	data, _ := json.Marshal(prior)
	var active lifecycle.Manifest
	_ = json.Unmarshal(data, &active)
	active.ID, active.Key, active.Payload = "publication-2", "request-2", "payload-2"
	active.Identity.Revision, active.Targets[0].Revision = "r2", "r2"
	active.Targets[0].Artifacts[0].Reference.Revision = "r2"
	for i := range active.Targets[0].Artifacts[0].Supports {
		active.Targets[0].Artifacts[0].Supports[i].Revision = "r2"
	}
	active.ExpectedPublication, active.State = prior.ID, lifecycle.Complete
	active.PublishedAt = prior.PublishedAt.Add(time.Second)
	snapshot.Manifests = append(snapshot.Manifests, active)
	snapshot.Publications[0].Manifest = active.ID
	snapshot.Cleanups = []lifecycle.CleanupJob{
		{
			Owner:     active.ID,
			StartedAt: active.PublishedAt,
			Deadline:  active.PublishedAt.Add(time.Hour),
			Complete:  true,
			Items: []lifecycle.RetiredTarget{
				{
					Manifest: prior.ID,
					Target:   "lexical",
					State:    lifecycle.CleanupDone,
					Attempts: 1,
					NextAt:   active.PublishedAt,
				},
			},
		},
	}
	return snapshot
}

func TestMaintenanceRestartReservationsAndOwnedResult(t *testing.T) {
	// Arrange.
	root := t.TempDir()
	store, err := filestore.New(root, 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	initial, err := store.CompareSwap(t.Context(), 0, historyFixture())
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	retired, err := store.Maintain(
		t.Context(),
		initial.Generation,
		lifecycle.RetirementRequest{Namespace: "n", Manifests: []string{"publication-1"}},
	)
	if err != nil {
		t.Fatal(err)
	}
	retired.Manifests[0].ArtifactFences[0].Digest = strings.Repeat("0", 64)
	restarted, err := filestore.New(root, 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	loaded, err := restarted.Load(t.Context(), "n")
	// Assert.
	if err != nil || loaded.Generation != 2 || !loaded.Manifests[0].Retired ||
		len(loaded.Manifests[0].Targets[0].Artifacts) != 0 ||
		loaded.Manifests[0].ArtifactFences[0].Digest == retired.Manifests[0].ArtifactFences[0].Digest {
		t.Fatal("retirement lost or aliased", err)
	}
	loaded.Manifests[0].Payload = "rebound"
	if _, err = restarted.CompareSwap(t.Context(), 2, loaded); !errors.Is(err, lifecycle.ErrRetired) {
		t.Fatal("CAS rebound retired handle", err)
	}
	if _, err = restarted.Maintain(
		t.Context(),
		1,
		lifecycle.RetirementRequest{Namespace: "n", Manifests: []string{"publication-1"}},
	); !errors.Is(
		err,
		lifecycle.ErrConflict,
	) {
		t.Fatal("stale retirement", err)
	}
	if _, err = restarted.Maintain(
		t.Context(),
		2,
		lifecycle.RetirementRequest{Namespace: "n", Manifests: []string{"publication-2"}},
	); !errors.Is(
		err,
		lifecycle.ErrProtected,
	) {
		t.Fatal("active publication retired", err)
	}
}

func TestConcurrentMaintenanceAndCASShareGeneration(t *testing.T) {
	// Arrange: independent store objects receive one synchronized start signal.
	root := t.TempDir()
	store, _ := filestore.New(root, 8<<20)
	initial, err := store.CompareSwap(t.Context(), 0, historyFixture())
	if err != nil {
		t.Fatal(err)
	}
	other, _ := filestore.New(root, 8<<20)
	start := make(chan struct{})
	outcomes := make(chan error, 2)
	var ready sync.WaitGroup
	ready.Add(2)
	// Act.
	go func() {
		ready.Done()
		<-start
		_, e := store.Maintain(
			t.Context(),
			1,
			lifecycle.RetirementRequest{Namespace: "n", Manifests: []string{"publication-1"}},
		)
		outcomes <- e
	}()
	go func() { ready.Done(); <-start; _, e := other.CompareSwap(t.Context(), 1, initial); outcomes <- e }()
	ready.Wait()
	close(start)
	first, second := <-outcomes, <-outcomes
	loaded, loadErr := store.Load(t.Context(), "n")
	// Assert.
	if (first == nil) == (second == nil) || (first != nil && !errors.Is(first, lifecycle.ErrConflict)) ||
		(second != nil && !errors.Is(second, lifecycle.ErrConflict)) ||
		loadErr != nil ||
		loaded.Generation != 2 {
		t.Fatal("generation race", first, second, loadErr, loaded.Generation)
	}
}

func TestOldSchemaMaintenanceNeverOverwrites(t *testing.T) {
	// Arrange.
	root := t.TempDir()
	store, _ := filestore.New(root, 8<<20)
	if _, err := store.CompareSwap(t.Context(), 0, historyFixture()); err != nil {
		t.Fatal(err)
	}
	paths, _ := filepath.Glob(filepath.Join(root, "*.json"))
	data, err := os.ReadFile(paths[0])
	if err != nil {
		t.Fatal(err)
	}
	data = []byte(strings.Replace(string(data), lifecycle.SchemaIdentity, "ragy.lifecycle/v1", 1))
	if err = os.WriteFile(paths[0], data, 0600); err != nil {
		t.Fatal(err)
	}
	// Act.
	_, loadErr := store.Load(t.Context(), "n")
	_, maintenanceErr := store.Maintain(
		t.Context(),
		1,
		lifecycle.RetirementRequest{Namespace: "n", Manifests: []string{"publication-1"}},
	)
	_, casErr := store.CompareSwap(
		t.Context(),
		1,
		func() lifecycle.Snapshot { s := historyFixture(); s.Generation = 1; return s }(),
	)
	after, err := os.ReadFile(paths[0])
	// Assert.
	if !errors.Is(loadErr, ragy.ErrUnsupported) || !errors.Is(maintenanceErr, ragy.ErrUnsupported) ||
		!errors.Is(casErr, ragy.ErrUnsupported) ||
		err != nil ||
		string(after) != string(data) {
		t.Fatal("old schema overwritten", loadErr, maintenanceErr, casErr, err)
	}
}

func TestCanceledMaintenanceKeepsState(t *testing.T) {
	// Arrange.
	store, _ := filestore.New(t.TempDir(), 8<<20)
	before, err := store.CompareSwap(t.Context(), 0, historyFixture())
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	// Act.
	_, err = store.Maintain(ctx, 1, lifecycle.RetirementRequest{Namespace: "n", Manifests: []string{"publication-1"}})
	after, loadErr := store.Load(t.Context(), "n")
	// Assert.
	if !errors.Is(err, context.Canceled) || loadErr != nil || !reflect.DeepEqual(before, after) {
		t.Fatal("canceled maintenance changed state", err, loadErr)
	}
}

// The only Err call made while a temporary exists is persist's pre-rename
// checkpoint, after both file fsync and close. Block that actual Maintain call.
type crashCommitBarrier struct {
	context.Context

	root         string
	ready, block *os.File
}

func (c *crashCommitBarrier) Err() error {
	temporary, _ := filepath.Glob(filepath.Join(c.root, ".lifecycle-*"))
	if len(temporary) > 0 {
		_, _ = c.ready.Write([]byte{1})
		_ = c.ready.Close()
		var signal [1]byte
		_, _ = c.block.Read(signal[:])
		return context.Canceled
	}
	return nil
}

func TestCrashReleasesLockAndRestartMaintainsCommittedHistory(t *testing.T) {
	if root := os.Getenv("RAGY_MAINTENANCE_CRASH_ROOT"); root != "" {
		runMaintenanceCrashChild(t, root)
		return
	}
	// Arrange: child signals only after acquiring the actual flock and syncing temp.
	root := t.TempDir()
	store, _ := filestore.New(root, 8<<20)
	before, err := store.CompareSwap(t.Context(), 0, historyFixture())
	if err != nil {
		t.Fatal(err)
	}
	readyRead, readyWrite, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer readyRead.Close()
	defer readyWrite.Close()
	blockRead, blockWrite, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer blockRead.Close()
	defer blockWrite.Close()
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	child := exec.CommandContext(
		t.Context(),
		executable,
		"-test.run=^TestCrashReleasesLockAndRestartMaintainsCommittedHistory$",
	)
	child.Env = append(os.Environ(), "RAGY_MAINTENANCE_CRASH_ROOT="+root)
	child.ExtraFiles = []*os.File{readyWrite, blockRead}
	if err = child.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() { _ = child.Process.Kill(); _ = child.Wait() }()
	_ = readyWrite.Close()
	_ = blockRead.Close()
	if err = readyRead.SetReadDeadline(time.Now().Add(10 * time.Second)); err != nil {
		t.Fatal(err)
	}
	var signal [1]byte
	if _, err = readyRead.Read(signal[:]); err != nil {
		t.Fatal(err)
	}
	// Act.
	_, busyErr := store.Maintain(
		t.Context(),
		1,
		lifecycle.RetirementRequest{Namespace: "n", Manifests: []string{"publication-1"}},
	)
	if err = child.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	_ = child.Wait()
	restarted, _ := filestore.New(root, 8<<20)
	after, loadErr := restarted.Load(t.Context(), "n")
	retired, maintainErr := restarted.Maintain(
		t.Context(),
		1,
		lifecycle.RetirementRequest{Namespace: "n", Manifests: []string{"publication-1"}},
	)
	// Assert.
	if !errors.Is(busyErr, lifecycle.ErrConflict) || loadErr != nil || !reflect.DeepEqual(before, after) ||
		maintainErr != nil ||
		retired.Generation != 2 ||
		!retired.Manifests[0].Retired {
		t.Fatal("crash recovery", busyErr, loadErr, maintainErr)
	}
}

func TestCASPreservesReleasedPinReservation(t *testing.T) {
	// Arrange.
	root := t.TempDir()
	store, _ := filestore.New(root, 8<<20)
	snapshot := historyFixture()
	snapshot.Pins = []lifecycle.PublicationPin{
		{
			ID:               "pin-1",
			Publication:      "read-1",
			RequestedTargets: []string{"lexical"},
			Released:         true,
			Targets: []access.TargetRevision{
				{
					Target:            "lexical",
					Namespace:         "n",
					Source:            "policy",
					Revision:          "r1",
					Transformation:    "chunk",
					AccessFingerprint: "acl",
				},
			},
		},
	}
	committed, err := store.CompareSwap(t.Context(), 0, snapshot)
	if err != nil {
		t.Fatal(err)
	}
	retired, err := store.Maintain(
		t.Context(),
		1,
		lifecycle.RetirementRequest{Namespace: "n", Manifests: []string{"publication-1"}},
	)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	retired.Pins = nil
	_, err = store.CompareSwap(t.Context(), 2, retired)
	after, loadErr := store.Load(t.Context(), "n")
	// Assert.
	if !errors.Is(err, lifecycle.ErrRetired) || loadErr != nil || len(after.Pins) != 1 ||
		after.Pins[0].ID != committed.Pins[0].ID ||
		!after.Pins[0].Released {
		t.Fatal("released pin reservation dropped", err, loadErr)
	}
}

func TestMaintenanceAndCASRejectOversizedExistingSnapshot(t *testing.T) {
	// Arrange.
	root := t.TempDir()
	store, _ := filestore.New(root, 8<<20)
	committed, err := store.CompareSwap(t.Context(), 0, historyFixture())
	if err != nil {
		t.Fatal(err)
	}
	paths, _ := filepath.Glob(filepath.Join(root, "*.json"))
	before, err := os.ReadFile(paths[0])
	if err != nil {
		t.Fatal(err)
	}
	bounded, err := filestore.New(root, int64(len(before)-1))
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	_, maintainErr := bounded.Maintain(
		t.Context(),
		1,
		lifecycle.RetirementRequest{Namespace: "n", Manifests: []string{"publication-1"}},
	)
	_, casErr := bounded.CompareSwap(t.Context(), 1, committed)
	after, readErr := os.ReadFile(paths[0])
	// Assert.
	if !errors.Is(maintainErr, lifecycle.ErrCapacity) || !errors.Is(casErr, lifecycle.ErrCapacity) || readErr != nil ||
		string(before) != string(after) {
		t.Fatal("oversized existing state replaced", maintainErr, casErr, readErr)
	}
}

func runMaintenanceCrashChild(t *testing.T, root string) {
	t.Helper()
	store, err := filestore.New(root, 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	ctx := &crashCommitBarrier{
		Context: context.Background(),
		root:    root,
		ready:   os.NewFile(3, "ready"),
		block:   os.NewFile(4, "barrier"),
	}
	_, err = store.Maintain(
		ctx,
		1,
		lifecycle.RetirementRequest{Namespace: "n", Manifests: []string{"publication-1"}},
	)
	t.Fatal("barrier unexpectedly released", err)
}
