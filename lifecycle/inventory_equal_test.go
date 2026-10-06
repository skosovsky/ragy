package lifecycle_test

import (
	"slices"
	"testing"

	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/source"
)

func TestManifestCloneOwnsNestedSupportsAndInventoryComparisonIsSetBased(t *testing.T) {
	// Arrange: multiple original supports for one managed artifact.
	original := manifestFixture()
	first := original.Targets[0].Artifacts[0].Supports[0]
	second := first
	second.Artifact = "another-original"
	original.Targets[0].Artifacts[0].Supports = []source.Reference{first, second}
	// Act.
	owned := original.Clone()
	slices.Reverse(owned.Targets[0].Artifacts[0].Supports)
	// Assert: order-independent exact supports and no alias to input manifest.
	if !lifecycle.SameTargetInventory(original, owned, "dense") {
		t.Fatal("support order changed inventory")
	}
	owned.Targets[0].Artifacts[0].Supports[0].Artifact = "changed"
	if original.Targets[0].Artifacts[0].Supports[1].Artifact != second.Artifact ||
		lifecycle.SameTargetInventory(original, owned, "dense") {
		t.Fatal("owned support mutation changed original or matched inventory")
	}
	if lifecycle.SameTargetInventory(original, original, "absent") {
		t.Fatal("missing target matched empty inventory")
	}
	duplicate := original.Clone()
	duplicate.Targets[0].Artifacts[0].Supports[1] = first
	if lifecycle.SameTargetInventory(original, duplicate, "dense") {
		t.Fatal("duplicate support matched set")
	}
}
