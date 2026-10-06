//go:build darwin || linux

package persistent_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/lifecycle/filestore"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/tensor/persistent"
)

const physicalCrashExit = 86

type physicalStopContext struct {
	context.Context

	root, phase, id string
}

func (c physicalStopContext) Err() error {
	sum := sha256.Sum256([]byte(c.id))
	id := hex.EncodeToString(sum[:])
	installed := filepath.Join(c.root, id)
	staging := filepath.Join(c.root, ".stage-"+id)
	retired := filepath.Join(c.root, ".retired-"+id)
	stop := false
	switch c.phase {
	case "payload":
		files, _ := filepath.Glob(filepath.Join(staging, "*.json"))
		stop = len(files) > 0
	case "catalog":
		_, err := os.Stat(filepath.Join(staging, "catalog.json"))
		stop = err == nil
	case "installed":
		_, err := os.Stat(filepath.Join(installed, "catalog.json"))
		stop = err == nil
	case "retired":
		_, err := os.Stat(retired)
		stop = err == nil
	case "removed":
		_, a := os.Stat(installed)
		_, b := os.Stat(retired)
		stop = errors.Is(a, os.ErrNotExist) && errors.Is(b, os.ErrNotExist)
	}
	if stop {
		os.Exit(physicalCrashExit)
	}
	return c.Context.Err()
}

func TestActualPhysicalCrashChild(t *testing.T) {
	root := os.Getenv("RAGY_tensor_PHASE_ROOT")
	if root == "" {
		t.Skip("subprocess helper")
	}
	config := newConfig(t)
	store, err := filestore.New(os.Getenv("RAGY_tensor_PHASE_STORE"), 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	config.Root, config.Store = root, store
	adapter, err := persistent.New(config)
	if err != nil {
		t.Fatal(err)
	}
	directories, err := filepath.Glob(filepath.Join(root, "*"))
	if err != nil || len(directories) != 1 {
		t.Fatal("target root missing", err)
	}
	phase := os.Getenv("RAGY_tensor_PHASE")
	id := "operation"
	if phase == "retired" || phase == "removed" {
		id = "old-operation"
	}
	ctx := physicalStopContext{Context: t.Context(), root: directories[0], phase: phase, id: id}
	if id == "operation" {
		_, err = newExecutor(t, config, adapter).Stage(ctx, "n", id, "tensor", records())
	} else {
		_, err = phaseCleaner(
			t,
			config,
			adapter,
			time.Now,
		).Attempt(ctx, "n", "deleted", id, "tensor", false)
	}
	t.Fatal("physical phase not reached", phase, err)
}

type phaseCleanupPort struct {
	adapter            *persistent.Adapter[metadata]
	calls, inspections int
}

func (p *phaseCleanupPort) Cleanup(
	ctx context.Context,
	request lifecycle.CleanupRequest,
) (lifecycle.CleanupState, error) {
	p.calls++
	return p.adapter.Cleanup(ctx, request)
}

func (p *phaseCleanupPort) InspectCleanup(
	ctx context.Context,
	request lifecycle.CleanupRequest,
) (lifecycle.CleanupState, error) {
	p.inspections++
	return p.adapter.InspectCleanup(ctx, request)
}

func phaseCleaner(
	t *testing.T,
	config persistent.Config[metadata],
	port lifecycle.CleanupPort,
	now func() time.Time,
) *lifecycle.Cleaner {
	t.Helper()
	cleaner, err := lifecycle.NewCleaner(
		lifecycle.CleanerConfig{
			Store:   config.Store,
			Now:     now,
			Targets: []lifecycle.CleanupRegistration{{Name: "tensor", Port: port}},
			Policy:  lifecycle.CleanupPolicy{Deadline: time.Minute, Backoff: []time.Duration{time.Second}},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	return cleaner
}
func TestActualPersistentPhysicalPhaseCrashRecovery(t *testing.T) {
	for _, phase := range []string{"payload", "catalog", "installed", "retired", "removed"} {
		t.Run(phase, func(t *testing.T) { physicalCrashCase(t, phase) })
	}
}
func physicalCrashCase(t *testing.T, phase string) {
	t.Helper()
	// Arrange: actual readable r0, durable replacement/cleanup ownership and child process.
	config := newConfig(t)
	storeRoot := filepath.Join(t.TempDir(), "manifests")
	store, err := filestore.New(storeRoot, 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	config.Store = store
	adapter, err := persistent.New(config)
	if err != nil {
		t.Fatal(err)
	}
	old := records()
	for i := range old {
		old[i].Reference.Revision = "r0"
	}
	original := plan(old)
	original.ID, original.Key, original.Identity.Revision = "old-operation", "old-request", "r0"
	executor := newExecutor(t, config, adapter)
	if _, err = executor.Prepare(t.Context(), original); err != nil {
		t.Fatal(err)
	}
	if _, err = executor.Stage(t.Context(), "n", original.ID, "tensor", old); err != nil {
		t.Fatal(err)
	}
	if _, err = executor.Publish(t.Context(), "n", original.ID); err != nil {
		t.Fatal(err)
	}
	captured := pin(t, config)
	cleanup := phase == "retired" || phase == "removed"
	preparePhysicalOperation(t, config, adapter, cleanup)
	binary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	// Act: os.Exit bypasses Go defers while real files are in the selected physical state.
	child := exec.CommandContext(t.Context(), binary, "-test.run=^TestActualPhysicalCrashChild$")
	child.Env = append(
		os.Environ(),
		"RAGY_tensor_PHASE_ROOT="+config.Root,
		"RAGY_tensor_PHASE_STORE="+storeRoot,
		"RAGY_tensor_PHASE="+phase,
		"TMPDIR="+t.TempDir(),
	)
	output, err := child.CombinedOutput()
	var exit *exec.ExitError
	if !errors.As(err, &exit) || exit.ExitCode() != physicalCrashExit {
		t.Fatalf("phase did not crash: %s %v %s", phase, err, output)
	}
	freshStore, err := filestore.New(storeRoot, 8<<20)
	if err != nil {
		t.Fatal(err)
	}
	config.Store = freshStore
	restarted, err := persistent.New(config)
	if err != nil {
		t.Fatal(err)
	}
	if cleanup {
		recoverPhysicalCleanup(t, config, restarted, phase, captured, old)
		return
	}
	recoverPhysicalStage(t, config, restarted, phase, old)
}

func preparePhysicalOperation(
	t *testing.T,
	config persistent.Config[metadata],
	adapter *persistent.Adapter[metadata],
	cleanup bool,
) {
	t.Helper()
	next := plan(records())
	next.ExpectedPublication = "old-operation"
	if cleanup {
		next.ID, next.Key, next.Payload = "deleted", "delete-request", "delete"
		next.Tombstone, next.Targets = true, nil
	}
	executor := newExecutor(t, config, adapter)
	if _, err := executor.Prepare(t.Context(), next); err != nil {
		t.Fatal(err)
	}
	if !cleanup {
		return
	}
	if _, err := executor.Publish(t.Context(), "n", next.ID); err != nil {
		t.Fatal(err)
	}
	if _, err := phaseCleaner(t, config, adapter, time.Now).Begin(t.Context(), "n", next.ID); err != nil {
		t.Fatal(err)
	}
}

func recoverPhysicalStage(
	t *testing.T,
	config persistent.Config[metadata],
	adapter *persistent.Adapter[metadata],
	phase string,
	old []persistent.Record[metadata],
) {
	t.Helper()
	// Assert: no staging residue becomes default published visibility after process loss.
	result, err := adapter.Query(t.Context(), query(pin(t, config), old))
	if err != nil || result.Documents.Len() != len(old) {
		t.Fatal("old publication lost", err)
	}
	assertPhysicalSourceRevision(t, result.Documents.Documents(), "r0")
	executor := newExecutor(t, config, adapter)
	manifest, err := executor.Reconcile(t.Context(), "n", "operation", "tensor")
	if err != nil {
		t.Fatal(err)
	}
	want := lifecycle.TargetPending
	if phase == "installed" {
		want = lifecycle.TargetReady
	}
	if manifest.Targets[0].State != want {
		t.Fatal("physical phase inferred ready", phase, manifest.Targets[0])
	}
	if _, err = executor.Stage(t.Context(), "n", "operation", "tensor", records()); err != nil {
		t.Fatal(err)
	}
	if _, err = executor.Publish(t.Context(), "n", "operation"); err != nil {
		t.Fatal(err)
	}
	result, err = adapter.Query(t.Context(), query(pin(t, config), records()))
	if err != nil || result.Documents.Len() != len(records()) {
		t.Fatal("physical stage recovery failed", err)
	}
	assertPhysicalSourceRevision(t, result.Documents.Documents(), "r1")
}

func recoverPhysicalCleanup(
	t *testing.T,
	config persistent.Config[metadata],
	adapter *persistent.Adapter[metadata],
	phase string,
	captured access.Binding,
	old []persistent.Record[metadata],
) {
	t.Helper()
	// Assert: tombstone closes new reads and unavailable old snapshot never falls back.
	result, err := adapter.Query(t.Context(), query(pin(t, config), old))
	if err != nil || result.Documents.Len() != 0 {
		t.Fatal("tombstone barrier lost", err)
	}
	_, err = adapter.Query(t.Context(), query(captured, old))
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatal("retired snapshot substituted", err)
	}
	port := &phaseCleanupPort{adapter: adapter}
	now := time.Now()
	cleaner := phaseCleaner(t, config, port, func() time.Time { return now })
	job, err := cleaner.Reconcile(t.Context(), "n", "deleted", "old-operation", "tensor")
	if err != nil || port.calls != 0 || port.inspections != 1 {
		t.Fatal("physical cleanup reconciliation", err, port)
	}
	if phase == "retired" {
		if job.Items[0].State != lifecycle.CleanupWaiting {
			t.Fatal("trash inferred deleted", job)
		}
		now = job.Items[0].NextAt
	} else if !job.Complete {
		t.Fatal("removed files not confirmed", job)
	}
	job, err = cleaner.Attempt(t.Context(), "n", "deleted", "old-operation", "tensor", false)
	wantCalls := 0
	if phase == "retired" {
		wantCalls = 1
	}
	if err != nil || !job.Complete || port.calls != wantCalls || port.inspections != 1 {
		t.Fatal("cleanup recovery repeated confirmed removal", err, port)
	}
}

func assertPhysicalSourceRevision(t *testing.T, documents []retrieval.Document[metadata], revision string) {
	t.Helper()
	for _, document := range documents {
		locations := document.SourceLocations()
		if len(locations) == 0 {
			t.Fatal("source revision evidence missing")
		}
		for _, location := range locations {
			if location.Reference.Revision != revision {
				t.Fatal("publication replaced by another physical revision", location.Reference, revision)
			}
		}
	}
}
