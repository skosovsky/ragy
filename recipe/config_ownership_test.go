package recipe_test

import (
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"

	"github.com/skosovsky/ragy/recipe"
	"github.com/skosovsky/ragy/retrieval"
)

func TestRecipeSnapshotsArtifactOptions(t *testing.T) {
	// Arrange.
	f := newFixture(t, recipe.SingleRewrite)
	f.results["original"] = []retrieval.Document[meta]{document("d1")}
	options := &retrieval.ArtifactRenderOptions[meta]{
		Resource:              retrieval.RuneResource(4096),
		CloneMeta:             f.config.CloneMeta,
		UntrustedDataBoundary: "original boundary",
		Diagnostics:           []retrieval.PlannerDiagnostic{{Key: "owner", Value: "original"}},
	}
	f.config.Artifact = options
	r, err := recipe.New(f.config)
	if err != nil {
		t.Fatal(err)
	}
	options.UntrustedDataBoundary = "mutated boundary"
	options.Diagnostics[0].Value = "mutated"
	options.CloneMeta = nil
	options.Resource.Limit = 0
	// Act.
	result, err := r.RunOwn(t.Context(), recordedRequest(f))
	// Assert.
	if err != nil || result.Artifact == nil || result.Artifact.UntrustedDataBoundary != "original boundary" {
		t.Fatal(result, err)
	}
	found := false
	for _, diagnostic := range result.Artifact.Diagnostics {
		if diagnostic.Key == "owner" {
			found = true
			if diagnostic.Value != "original" {
				t.Fatal("diagnostics borrowed", diagnostic)
			}
		}
	}
	if !found {
		t.Fatal("configuration diagnostics lost")
	}
}

func TestRecipeRejectsTypedNilOptionalEncoder(t *testing.T) {
	// Arrange.
	f := newFixture(t, recipe.SingleRewrite)
	var encoder *boundedEncoder
	f.config.QueryEncoder = encoder
	// Act.
	r, err := recipe.New(f.config)
	// Assert.
	if r != nil || !errors.Is(err, ragy.ErrInvalidArgument) || f.modelCalls != 0 || len(f.retrieved) != 0 {
		t.Fatal(r, err)
	}
}
