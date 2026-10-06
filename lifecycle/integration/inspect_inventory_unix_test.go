//go:build darwin || linux

package integration_test

import (
	"slices"
	"testing"

	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/source"
)

func TestActualVolatileInspectRejectsChangedInventory(t *testing.T) {
	for _, target := range []string{"lexical", "graph"} {
		t.Run(target, func(t *testing.T) {
			// Arrange: actual installed target data, not a readiness stub.
			f := newFixture(t, target)
			input := sourceBatch("policy", "r1", []string{"p1", "p2"})
			manifest := plan("policy-r1", "", target, input)
			f.ingest(t, manifest, input)
			reordered := manifest.Clone()
			slices.Reverse(reordered.Targets[1].Artifacts)
			positive, inspectErr := f.secondary.Inspect(
				t.Context(),
				lifecycle.StageRequest{Manifest: reordered, Target: target},
			)
			if inspectErr != nil || positive.State != lifecycle.TargetReady {
				t.Fatal("order changed inventory identity", inspectErr)
			}
			for _, change := range []string{"missing", "replacement", "target-missing", "duplicate", "payload", "supports", "content"} {
				t.Run(change, func(t *testing.T) {
					request := manifest
					request.Targets = slices.Clone(manifest.Targets)
					request.Targets[1].Artifacts = slices.Clone(manifest.Targets[1].Artifacts)
					switch change {
					case "missing":
						request.Targets[1].Artifacts = request.Targets[1].Artifacts[1:]
					case "replacement":
						request.Targets[1].Artifacts[0].Reference.Artifact = "unwritten"
					case "target-missing":
						request.Targets[1].Name = "other"
					case "duplicate":
						request.Targets[1].Artifacts[1] = request.Targets[1].Artifacts[0]
					case "payload":
						request.Payload = "different"
					case "content":
						request.Identity.Content = "different-content"
					case "supports":
						support := request.Targets[1].Artifacts[0].Supports[0]
						support.Artifact = "different-original"
						request.Targets[1].Artifacts[0].Supports = []source.Reference{support}
					}
					// Act.
					result, err := f.secondary.Inspect(
						t.Context(),
						lifecycle.StageRequest{Manifest: request, Target: target},
					)
					// Assert: existing version identity cannot attest another inventory/payload.
					if err == nil || result.State == lifecycle.TargetReady {
						t.Fatal("changed inventory attested ready", result, err)
					}
				})
			}
		})
	}
}
