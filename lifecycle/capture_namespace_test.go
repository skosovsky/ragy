package lifecycle_test

import (
	"context"
	"errors"
	"reflect"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/lifecycle"
)

type captureResponseStore struct {
	response      lifecycle.Snapshot
	loads, writes int
	requested     string
}

func (s *captureResponseStore) Load(_ context.Context, namespace string) (lifecycle.Snapshot, error) {
	s.loads++
	s.requested = namespace
	return s.response, nil
}

func (s *captureResponseStore) CompareSwap(context.Context, uint64, lifecycle.Snapshot) (lifecycle.Snapshot, error) {
	s.writes++
	return lifecycle.Snapshot{}, errors.New("capture must not mutate store")
}

func captureResponse(namespace string, populated bool) lifecycle.Snapshot {
	snapshot := lifecycle.Snapshot{Schema: lifecycle.SchemaIdentity, Namespace: namespace}
	if !populated {
		return snapshot
	}
	manifest := manifestFixture()
	manifest.Identity.Namespace = namespace
	for i := range manifest.Targets {
		for j := range manifest.Targets[i].Artifacts {
			artifact := &manifest.Targets[i].Artifacts[j]
			artifact.Reference.Namespace = namespace
			for k := range artifact.Supports {
				artifact.Supports[k].Namespace = namespace
			}
		}
	}
	snapshot.Generation = 1
	snapshot.Manifests = []lifecycle.Manifest{manifest}
	snapshot.Publications = []lifecycle.Publication{{Source: manifest.Identity.Source, Manifest: manifest.ID}}
	return snapshot
}

func TestCaptureAdmitsRequestedNamespaceBeforeDerivation(t *testing.T) {
	captures := map[string]func(context.Context, lifecycle.Store, string, []string) (access.Publication, error){
		"strict": lifecycle.CapturePublication, "partial": lifecycle.CapturePartialPublication,
	}
	for mode, capture := range captures {
		for _, populated := range []bool{false, true} {
			for _, responseKind := range []string{"matching", "wrong_namespace", "wrong_schema", "malformed"} {
				t.Run(
					mode+"/"+responseKind+map[bool]string{false: "/empty", true: "/nonempty"}[populated],
					func(t *testing.T) {
						checkCaptureResponse(t, capture, populated, responseKind)
					},
				)
			}
		}
	}
}

func TestCaptureInvalidCallerArgumentsDoNotLoad(t *testing.T) {
	for _, capture := range []func(context.Context, lifecycle.Store, string, []string) (access.Publication, error){
		lifecycle.CapturePublication, lifecycle.CapturePartialPublication,
	} {
		for _, args := range []struct {
			namespace string
			targets   []string
		}{
			{namespace: "", targets: []string{"dense"}}, {namespace: "n"},
			{namespace: "n", targets: []string{"dense", "dense"}}, {namespace: "n", targets: []string{""}},
		} {
			// Arrange.
			store := &captureResponseStore{response: captureResponse("other", true)}
			// Act.
			publication, err := capture(t.Context(), store, args.namespace, args.targets)
			// Assert.
			if !errors.Is(err, ragy.ErrInvalidArgument) || store.loads != 0 || store.writes != 0 ||
				!reflect.DeepEqual(publication, access.Publication{}) {
				t.Fatal(publication, err, store)
			}
		}
	}
}

func checkCaptureResponse(
	t *testing.T,
	capture func(context.Context, lifecycle.Store, string, []string) (access.Publication, error),
	populated bool,
	responseKind string,
) {
	t.Helper()
	// Arrange: a pluggable Store returns a protocol response, never a caller plan.
	response := captureResponse("n", populated)
	switch responseKind {
	case "matching":
	case "wrong_namespace":
		response = captureResponse("other", populated)
	case "wrong_schema":
		response.Schema = "unsupported"
	case "malformed":
		if populated {
			response.Publications[0].Manifest = "missing"
		} else {
			response.Namespace = ""
		}
	}
	if responseKind == "matching" || responseKind == "wrong_namespace" {
		if err := response.Validate(); err != nil {
			t.Fatal("fixture not structurally valid", err)
		}
	}
	store := &captureResponseStore{response: response}
	targets := []string{"tensor", "dense"}
	// Act.
	publication, err := capture(t.Context(), store, "n", targets)
	// Assert: mismatch/malformed response is zero+protocol, capture never writes.
	if store.loads != 1 || store.writes != 0 || store.requested != "n" {
		t.Fatal(store)
	}
	if responseKind != "matching" {
		if !errors.Is(err, ragy.ErrProtocol) ||
			!reflect.DeepEqual(publication, access.Publication{}) {
			t.Fatal("invalid response admitted", publication, err)
		}
		return
	}
	checkCaptureOwnership(t, publication, err, populated, store, targets)
}

func checkCaptureOwnership(
	t *testing.T,
	publication access.Publication,
	err error,
	populated bool,
	store *captureResponseStore,
	targets []string,
) {
	t.Helper()
	want := 0
	if populated {
		want = 2
	}
	if err != nil || publication.IsCurrent() || publication.IsPartial() ||
		len(publication.Targets()) != want {
		t.Fatal("matching capture changed", publication, err)
	}
	// Returned inventory owns its containers, independent of caller/store slices.
	targets[0] = "caller mutation"
	owned := publication.Targets()
	if len(owned) > 0 {
		owned[0].Namespace = "reader mutation"
		store.response.Manifests[0].Identity.Namespace = "store mutation"
		if publication.Targets()[0].Namespace != "n" {
			t.Fatal("capture aliases inventory")
		}
	}
}
