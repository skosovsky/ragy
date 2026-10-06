//go:build darwin || linux

package history_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/graphingest/resolution"
	"github.com/skosovsky/ragy/graphingest/resolution/history"
	"github.com/skosovsky/ragy/source"
)

type attributes struct {
	Owners []string `json:"owners"`
}

const maxBytes = 64 << 10
const maxSupports = 20

func admitted(context.Context, access.Binding, source.Locator) error { return nil }

func location(sourceID, revision string) source.Locator {
	return source.Locator{Kind: source.DocumentLocation, Reference: source.Reference{
		Namespace:         "prod",
		Source:            sourceID,
		Revision:          revision,
		Transformation:    "original",
		AccessFingerprint: "access",
		Artifact:          "chunk1",
		Representation:    "text",
	}}
}

func extraction(revision, owner string) resolution.Extraction[string, string, attributes] {
	return resolution.Extraction[string, string, attributes]{Entities: []resolution.Entity[string, attributes]{
		{
			ID:         "billing",
			Name:       "Billing",
			Namespace:  "production",
			Kind:       "Service",
			Attributes: attributes{Owners: []string{owner}},
			Supports:   []source.Locator{location("s1", revision)},
		},
		{
			ID:         "pay",
			Name:       "Pay",
			Namespace:  "production",
			Kind:       "Service",
			Attributes: attributes{Owners: []string{"TeamB"}},
			Supports:   []source.Locator{location("s2", "r1")},
		},
		{
			ID:         "unknown",
			Name:       "Billing",
			Namespace:  "",
			Kind:       "Service",
			Attributes: attributes{Owners: nil},
			Supports:   []source.Locator{location("s3", "r1")},
		},
	}, Relations: []resolution.Relation[string, attributes]{
		{
			ID:         "link",
			From:       "billing",
			To:         "pay",
			Kind:       "depends_on",
			Attributes: attributes{Owners: nil},
			Supports:   []source.Locator{location("s1", revision)},
		},
		{
			ID:         "ambiguous-link",
			From:       "unknown",
			To:         "pay",
			Kind:       "depends_on",
			Attributes: attributes{Owners: nil},
			Supports:   []source.Locator{location("s3", "r1")},
		},
	}}
}

func record(t *testing.T, revision, owner, policy, parent string) history.Record[string, string, attributes] {
	t.Helper()
	resolver, err := resolution.New(resolution.Config[string, string, attributes]{
		OntologyIdentity: "host-ontology",
		PolicyIdentity:   policy,
		MaxEntities:      10,
		MaxRelations:     10,
		MaxSupports:      maxSupports,
		ValidateEntity:   func(string, attributes) error { return nil },
		ValidateRelation: func(string, string, string, attributes) error { return nil },
		Identity: func(entity resolution.Entity[string, attributes]) (resolution.Decision, error) {
			if entity.Namespace == "" {
				return resolution.Decision{State: resolution.Ambiguous, Namespace: "", Key: "", Name: ""}, nil
			}
			return resolution.Decision{
				State:     resolution.Resolved,
				Namespace: entity.Namespace,
				Key:       "billing",
				Name:      "Billing",
			}, nil
		},
		RelationKey:     func(relation resolution.Relation[string, attributes]) (string, error) { return relation.Kind, nil },
		CloneAttributes: func(a attributes) (attributes, error) { a.Owners = slices.Clone(a.Owners); return a, nil },
		Equivalent:      func(a, b attributes) bool { return slices.Equal(a.Owners, b.Owners) },
		AdmitSupport:    admitted,
	})
	if err != nil {
		t.Fatal(err)
	}
	input := extraction(revision, owner)
	result, err := resolver.Resolve(context.Background(), access.Unrestricted(), input)
	if err != nil {
		t.Fatal(err)
	}
	return history.Record[string, string, attributes]{
		Metadata: history.Metadata{Run: revision, ExtractionFingerprint: "extract-config", Parent: parent},
		Input:    input,
		Result:   result,
	}
}

func capture(
	t *testing.T,
	record history.Record[string, string, attributes],
) history.Snapshot[string, string, attributes] {
	t.Helper()
	snapshot, err := history.Capture(
		context.Background(),
		access.Unrestricted(),
		record,
		admitted,
		maxBytes,
		maxSupports,
	)
	if err != nil {
		t.Fatal(err)
	}
	return snapshot
}

func store(t *testing.T, root string, admit history.Admission) *history.FileStore[string, string, attributes] {
	t.Helper()
	result, err := history.NewFileStore[string, string, attributes](root, maxBytes, maxSupports, admit)
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func TestHistoryRetainsRecomputationAndConflictDecisionsAcrossRestart(t *testing.T) {
	// Arrange: alias merge, ambiguity and source conflict are produced by real resolver.
	root := t.TempDir()
	archive := store(t, root, admitted)
	firstRecord := record(t, "r1", "TeamA", "policy-a", "")
	first := capture(t, firstRecord)
	firstRecord.Input.Entities[0].Attributes.Owners[0] = "mutated-input"
	firstRecord.Result.EntityDecisions[0].Supports[0] = location("foreign", "r9")
	second := capture(t, record(t, "r2", "TeamC", "policy-b", first.Reference().ID))
	// Act: immutable writes, then a fresh storage instance reads both generations.
	for _, snapshot := range []history.Snapshot[string, string, attributes]{first, second} {
		if err := archive.Append(context.Background(), access.Unrestricted(), snapshot); err != nil {
			t.Fatal(err)
		}
	}
	restarted := store(t, root, admitted)
	oldSnapshot, err := restarted.Read(context.Background(), access.Unrestricted(), first.Reference())
	if err != nil {
		t.Fatal(err)
	}
	newSnapshot, err := restarted.Read(context.Background(), access.Unrestricted(), second.Reference())
	if err != nil {
		t.Fatal(err)
	}
	oldRecord, err := oldSnapshot.Record()
	if err != nil {
		t.Fatal(err)
	}
	newRecord, err := newSnapshot.Record()
	if err != nil {
		t.Fatal(err)
	}
	// Assert: explicit alias/ambiguity traces, conflicting owners and both source refs survive.
	if oldRecord.Result.PolicyIdentity != "policy-a" || newRecord.Result.PolicyIdentity != "policy-b" ||
		newRecord.Metadata.Parent != first.Reference().ID || oldRecord.Input.Entities[0].Attributes.Owners[0] != "TeamA" ||
		newRecord.Input.Entities[0].Attributes.Owners[0] != "TeamC" || len(oldRecord.Result.Entities[0].Variants) != 2 {
		t.Fatal(oldRecord, newRecord)
	}
	assertTraces(t, oldRecord.Result)
	if oldRecord.Result.EntityDecisions[0].CanonicalID != newRecord.Result.EntityDecisions[0].CanonicalID ||
		oldRecord.Result.EntityDecisions[0].Supports[0].Reference.Revision != "r1" || newRecord.Result.EntityDecisions[0].Supports[0].Reference.Revision != "r2" {
		t.Fatal(oldRecord, newRecord)
	}
	oldRecord.Result.Entities[0].Variants[0].Attributes.Owners[0] = "mutated-read"
	again, err := oldSnapshot.Record()
	if err != nil {
		t.Fatal(err)
	}
	if again.Result.Entities[0].Variants[0].Attributes.Owners[0] != "TeamA" {
		t.Fatal("snapshot aliases returned record")
	}
}

func assertTraces(t *testing.T, result resolution.Result[string, string, attributes]) {
	t.Helper()
	if len(result.EntityDecisions) != 3 || len(result.RelationDecisions) != 2 {
		t.Fatal(result)
	}
	entities := result.EntityDecisions
	if entities[0].CanonicalID != entities[1].CanonicalID || entities[2].Identity.State != resolution.Ambiguous ||
		entities[2].CanonicalID != "" {
		t.Fatal(entities)
	}
	relations := result.RelationDecisions
	if relations[0].Key != "depends_on" || relations[1].State != resolution.Ambiguous || relations[1].Key != "" {
		t.Fatal(relations)
	}
	for _, variant := range result.Entities[0].Variants {
		if len(variant.Supports) != 1 {
			t.Fatal(variant)
		}
	}
}

func TestConcurrentIdempotentAppendAndCorruption(t *testing.T) {
	// Arrange.
	root := t.TempDir()
	archive := store(t, root, admitted)
	snapshot := capture(t, record(t, "r1", "TeamA", "policy-a", ""))
	var group sync.WaitGroup
	errorsOut := make(chan error, 8)
	// Act.
	for range 8 {
		group.Go(func() { errorsOut <- archive.Append(context.Background(), access.Unrestricted(), snapshot) })
	}
	group.Wait()
	close(errorsOut)
	// Assert.
	for err := range errorsOut {
		if err != nil {
			t.Fatal(err)
		}
	}
	files, err := os.ReadDir(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(files) != 1 {
		t.Fatal("idempotent append created extra files", files)
	}
	if err = os.WriteFile(filepath.Join(root, files[0].Name()), []byte(`{"corrupt":true}`), 0o600); err != nil {
		t.Fatal(err)
	}
	_, err = archive.Read(context.Background(), access.Unrestricted(), snapshot.Reference())
	if !errors.Is(err, ragy.ErrProtocol) {
		t.Fatal(err)
	}
}

func TestAdmissionPrecedesStorageAndForgedSupportsCannotAddressPayload(t *testing.T) {
	// Arrange.
	root := filepath.Join(t.TempDir(), "never-created")
	snapshot := capture(t, record(t, "r1", "TeamA", "policy-a", ""))
	denied := store(t, root, func(context.Context, access.Binding, source.Locator) error { return ragy.ErrUnavailable })
	// Act.
	err := denied.Append(context.Background(), access.Unrestricted(), snapshot)
	_, readErr := denied.Read(context.Background(), access.Unrestricted(), snapshot.Reference())
	// Assert.
	if !access.IsProtectionFailure(err) || !access.IsProtectionFailure(readErr) {
		t.Fatal(err, readErr)
	}
	if _, statErr := os.Stat(root); !errors.Is(statErr, os.ErrNotExist) {
		t.Fatal("denied append touched directory", statErr)
	}
	allowed := store(t, root, admitted)
	if err = allowed.Append(context.Background(), access.Unrestricted(), snapshot); err != nil {
		t.Fatal(err)
	}
	forged := snapshot.Reference()
	forged.Supports[0] = location("permitted-forgery", "r1")
	_, err = allowed.Read(context.Background(), access.Unrestricted(), forged)
	if !errors.Is(err, os.ErrNotExist) {
		t.Fatal("forged support inventory addressed original payload", err)
	}
}

func TestCaptureRejectsMissingTraceForeignSupportAndBoundsBeforeAdmission(t *testing.T) {
	for _, scenario := range []string{"missing-trace", "foreign-support", "bound", "canceled"} {
		t.Run(scenario, func(t *testing.T) {
			// Arrange.
			input := record(t, "r1", "TeamA", "policy-a", "")
			limit := maxSupports
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			switch scenario {
			case "missing-trace":
				input.Result.EntityDecisions = nil
			case "foreign-support":
				input.Result.Entities[0].Variants[0].Supports[0] = location("foreign", "r1")
			case "bound":
				limit = 1
			case "canceled":
				cancel()
			}
			calls := 0
			// Act.
			snapshot, err := history.Capture(
				ctx,
				access.Unrestricted(),
				input,
				func(context.Context, access.Binding, source.Locator) error { calls++; return nil },
				maxBytes,
				limit,
			)
			// Assert.
			if err == nil || snapshot.Reference().ID != "" || calls != 0 {
				t.Fatal(snapshot.Reference(), err, calls)
			}
		})
	}
}
