package lifecycle_test

import (
	"encoding/json"
	"errors"
	"os"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/source"
)

func manifestFixture() lifecycle.Manifest {
	reference := source.Reference{
		Namespace: "n", Source: "policy", Revision: "r1", Transformation: "chunk",
		AccessFingerprint: "acl", Artifact: "p1", Representation: "text",
	}
	return lifecycle.Manifest{
		ID: "pub1", Identity: lifecycle.Identity{
			Namespace: "n", Source: "policy", Revision: "r1", Content: "c", Transformation: "chunk", Access: "acl",
		}, Key: "key", Payload: "payload", State: lifecycle.Published, PublishedAt: time.Unix(100, 0).UTC(),
		Targets: []lifecycle.Target{
			{Name: "dense", Required: true, State: lifecycle.TargetReady, Revision: "r1",
				Artifacts: []lifecycle.Artifact{{Reference: reference, Supports: []source.Reference{reference}}}},
			{Name: "tensor", Required: true, State: lifecycle.TargetReady, Revision: "r1"},
		},
	}
}

func TestManifestReadinessPartialAndConfirmedPublication(t *testing.T) {
	// Arrange: dense r2 can exist physically while a required tensor is not ready.
	manifest := manifestFixture()
	manifest.Targets[1].State = lifecycle.TargetFailed
	manifest.Targets[1].Revision = ""
	// Act/Assert: default publication rejects that inventory; staging keeps it.
	if !errors.Is(manifest.Validate(), ragy.ErrInvalidArgument) {
		t.Fatal("mixed required publication accepted")
	}
	manifest.State = lifecycle.Staging
	manifest.PublishedAt = time.Time{}
	if err := manifest.Validate(); err != nil {
		t.Fatal(err)
	}
	manifest.State = lifecycle.Published
	manifest.PublishedAt = time.Unix(100, 0).UTC()
	manifest.Partial = true
	if err := manifest.Validate(); err != nil {
		t.Fatal("explicit partial rejected", err)
	}
	manifest.State = lifecycle.Unknown
	manifest.Checkpoint = lifecycle.Published
	if err := manifest.Validate(); err != nil {
		t.Fatal("unknown lost confirmed publication", err)
	}
	snapshot := lifecycle.Snapshot{
		Schema:     lifecycle.SchemaIdentity,
		Namespace:  "n",
		Generation: 1,
		Manifests: []lifecycle.Manifest{
			manifest,
		},
		Publications: []lifecycle.Publication{{Source: "policy", Manifest: "pub1"}},
	}
	if err := snapshot.Validate(); err != nil {
		t.Fatal(err)
	}
	snapshot.Manifests[0].Checkpoint = lifecycle.Staging
	if !errors.Is(snapshot.Validate(), ragy.ErrInvalidArgument) {
		t.Fatal("unconfirmed publication accepted")
	}
}

func TestManifestInvalidIdentitySupportsAndSchema(t *testing.T) {
	mutations := map[string]func(*lifecycle.Manifest){
		"missing checkpoint":        func(m *lifecycle.Manifest) { m.State = lifecycle.Unknown },
		"wrong revision":            func(m *lifecycle.Manifest) { m.Targets[0].Revision = "r2" },
		"wrong access":              func(m *lifecycle.Manifest) { m.Targets[0].Artifacts[0].Reference.AccessFingerprint = "other" },
		"foreign namespace support": func(m *lifecycle.Manifest) { m.Targets[0].Artifacts[0].Supports[0].Namespace = "foreign" },
		"missing support":           func(m *lifecycle.Manifest) { m.Targets[0].Artifacts[0].Supports = nil },
		"duplicate support": func(m *lifecycle.Manifest) {
			a := &m.Targets[0].Artifacts[0]
			a.Supports = append(a.Supports, a.Supports[0])
		},
		"duplicate target":   func(m *lifecycle.Manifest) { m.Targets = append(m.Targets, m.Targets[0]) },
		"duplicate artifact": func(m *lifecycle.Manifest) { a := &m.Targets[0]; a.Artifacts = append(a.Artifacts, a.Artifacts[0]) },
		"unknown state":      func(m *lifecycle.Manifest) { m.State = "surprise" },
		"empty fingerprint":  func(m *lifecycle.Manifest) { m.Identity.Content = "" },
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			// Arrange.
			manifest := manifestFixture()
			mutate(&manifest)
			// Act/Assert.
			if !errors.Is(manifest.Validate(), ragy.ErrInvalidArgument) {
				t.Fatal("invalid manifest accepted")
			}
		})
	}
	snapshot := lifecycle.Snapshot{
		Schema:     lifecycle.SchemaIdentity,
		Namespace:  "n",
		Generation: 1,
		Manifests: []lifecycle.Manifest{
			manifestFixture(),
		},
		Publications: []lifecycle.Publication{{Source: "policy", Manifest: "pub1"}},
	}
	data, err := json.Marshal(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	var restored lifecycle.Snapshot
	if err = json.Unmarshal(data, &restored); err != nil {
		t.Fatal(err)
	}
	if err = restored.Validate(); err != nil {
		t.Fatal(err)
	}
	restored.Schema = "unknown"
	if !errors.Is(restored.Validate(), ragy.ErrInvalidArgument) {
		t.Fatal("unknown schema accepted")
	}
}

func TestTombstonePublicationDoesNotDependOnPhysicalCleanup(t *testing.T) {
	// Arrange: cleanup has not reached the unavailable backend.
	manifest := manifestFixture()
	manifest.Tombstone = true
	manifest.State = lifecycle.CleanupPending
	for i := range manifest.Targets {
		manifest.Targets[i].State = lifecycle.TargetPending
	}
	snapshot := lifecycle.Snapshot{
		Schema:     lifecycle.SchemaIdentity,
		Namespace:  "n",
		Generation: 1,
		Manifests: []lifecycle.Manifest{
			manifest,
		},
		Publications: []lifecycle.Publication{{Source: "policy", Manifest: "pub1"}},
	}
	// Act/Assert: the acknowledged barrier is durable even with cleanup pending.
	if err := snapshot.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestLifecycleWireFixtureMatchesSemanticContract(t *testing.T) {
	// Arrange: the independently validated schema fixture is also decoded by Go.
	data, err := os.ReadFile("testdata/lifecycle_snapshot.json")
	if err != nil {
		t.Fatal(err)
	}
	var snapshot lifecycle.Snapshot
	// Act.
	if err = json.Unmarshal(data, &snapshot); err != nil {
		t.Fatal(err)
	}
	// Assert: cross-field namespace/source/publication/support identities are checked.
	if err = snapshot.Validate(); err != nil {
		t.Fatal(err)
	}
	if snapshot.Generation != 1 || snapshot.Publications[0].Manifest != "publication-1" {
		t.Fatal("wire fixture drifted from the durable profile")
	}
}

func TestCleanupWireFixtureMatchesSemanticContract(t *testing.T) {
	// Arrange.
	data, err := os.ReadFile("testdata/lifecycle_cleanup.json")
	if err != nil {
		t.Fatal(err)
	}
	var snapshot lifecycle.Snapshot
	// Act/Assert: retained inventory, read barrier, checkpoint and cleanup uncertainty.
	if err = json.Unmarshal(data, &snapshot); err != nil {
		t.Fatal(err)
	}
	if err = snapshot.Validate(); err != nil {
		t.Fatal(err)
	}
	snapshot.Cleanups[0].Items[0].Attempts = 0
	if !errors.Is(snapshot.Validate(), ragy.ErrInvalidArgument) {
		t.Fatal("unknown cleanup without dispatch accepted")
	}
	snapshot.Cleanups[0].Items = nil
	snapshot.Cleanups[0].Complete = true
	snapshot.Manifests[1].State = lifecycle.Complete
	if !errors.Is(snapshot.Validate(), ragy.ErrInvalidArgument) {
		t.Fatal("incomplete cleanup inventory declared complete")
	}
}
