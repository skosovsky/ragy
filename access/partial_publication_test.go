package access_test

import (
	"errors"
	"slices"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
)

func TestPartialPublicationOwnsExclusionsAndPartitionsBindings(t *testing.T) {
	// Arrange: exact same available inventory, distinct excluded branches.
	targets := []access.TargetRevision{
		{Target: "dense", Namespace: "n", Source: "s", Revision: "r2", Transformation: "t", AccessFingerprint: "a"},
	}
	excluded := []string{"tensor"}
	pub, err := access.PinPartialPublication("p", targets, excluded)
	if err != nil {
		t.Fatal(err)
	}
	other, err := access.PinPartialPublication("p", targets, []string{"graph"})
	if err != nil {
		t.Fatal(err)
	}
	// Act: mutate caller input and accessor copy.
	excluded[0] = "changed"
	targets[0].Revision = "changed"
	copyNames := pub.ExcludedTargets()
	copyNames[0] = "changed"
	firstRead, _ := access.UnrestrictedAt(pub)
	secondRead, _ := access.UnrestrictedAt(other)
	first, err := firstRead.Fingerprint()
	if err != nil {
		t.Fatal(err)
	}
	second, err := secondRead.Fingerprint()
	// Assert: no mutable alias and no cache identity collision.
	if err != nil || first == second || !slices.Equal(pub.ExcludedTargets(), []string{"tensor"}) ||
		pub.Targets()[0].Revision != "r2" {
		t.Fatal("partial publication ownership/identity failed", err)
	}
	if err = pub.AdmitTarget("dense"); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"tensor", "unrequested", ""} {
		if err = pub.AdmitTarget(name); !errors.Is(err, ragy.ErrUnsupported) {
			t.Fatal("unavailable target admitted", name, err)
		}
	}
}

func TestPartialPublicationRejectsAmbiguousOrUnsafeLabels(t *testing.T) {
	// Arrange/Act/Assert: static labels are not raw source diagnostics.
	for _, excluded := range [][]string{nil, {"a", "a"}, {"source/private"}, {""}} {
		if _, err := access.PinPartialPublication("p", nil, excluded); !errors.Is(err, ragy.ErrInvalidArgument) {
			t.Fatal("invalid exclusion accepted", excluded, err)
		}
	}
	_, err := access.PinPartialPublication(
		"p",
		[]access.TargetRevision{
			{Target: "dense", Namespace: "n", Source: "s", Revision: "r2", Transformation: "t", AccessFingerprint: "a"},
		},
		[]string{"dense"},
	)
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatal("available/excluded overlap accepted", err)
	}
}
