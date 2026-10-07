//go:build darwin || linux

package persistent

import (
	"errors"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/lifecycle"
)

func TestSelectedManifestExplicitlyRejectsRetired(t *testing.T) {
	// Arrange: matching identity and ready target; published timestamp survives
	// retirement. Selection must explicitly exclude this retained skeleton.
	pinned := access.TargetRevision{
		Target:            "target",
		Namespace:         "n",
		Source:            "s",
		Revision:          "r",
		Transformation:    "transform",
		AccessFingerprint: "access",
	}
	manifest := lifecycle.Manifest{
		ID: "m",
		Identity: lifecycle.Identity{
			Namespace:      "n",
			Source:         "s",
			Revision:       "r",
			Transformation: "transform",
			Access:         "access",
		},
		PublishedAt: time.Unix(1, 0),
		Targets:     []lifecycle.Target{{Name: "target", State: lifecycle.TargetReady}},
	}
	for _, retired := range []bool{false, true} {
		manifest.Retired = retired
		// Act.
		selected, err := selectedManifest(lifecycle.Snapshot{Manifests: []lifecycle.Manifest{manifest}}, pinned)
		// Assert.
		if retired {
			if !errors.Is(err, ragy.ErrUnavailable) || selected.ID != "" {
				t.Fatal("retired manifest selected", selected, err)
			}
		} else if err != nil || selected.ID != "m" {
			t.Fatal("live manifest rejected", selected, err)
		}
	}
	// Act: a retired duplicate must not make the live candidate ambiguous.
	live := manifest
	live.ID, live.Retired = "live", false
	selected, err := selectedManifest(lifecycle.Snapshot{Manifests: []lifecycle.Manifest{manifest, live}}, pinned)
	// Assert.
	if err != nil || selected.ID != "live" {
		t.Fatal("retired duplicate affects selection", selected, err)
	}
}
