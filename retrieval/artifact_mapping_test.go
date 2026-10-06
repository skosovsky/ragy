package retrieval_test

import (
	"context"
	"errors"
	"maps"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

func artifactLocation(id string) source.Locator {
	return source.Locator{
		Reference: source.Reference{
			Namespace:         "docs",
			Source:            id,
			Revision:          "r1",
			Transformation:    "parser",
			AccessFingerprint: "acl",
			Artifact:          id,
			Representation:    "page-text",
		},
		Kind: source.TextLocation,
		Span: source.ByteSpan{Start: 6, End: 10},
	}
}

func TestArtifactDedupRetainsBothSourcesAndExactProjection(t *testing.T) {
	// Arrange.
	docs := []retrieval.Document[struct{}]{
		{ID: "first", Content: "Alpha beta. Gamma."},
		{ID: "second", Content: "Alpha beta. Gamma."},
	}
	rs := retrieval.NewResultSet(docs, retrieval.DocumentIDResolver[struct{}]{})
	options := retrieval.ArtifactRenderOptions[struct{}]{Resource: retrieval.RuneResource(1000),
		CloneMeta: func(meta struct{}) (struct{}, error) { return meta, nil },
		Mapping: func(doc retrieval.Document[struct{}]) (source.MappedText, error) {
			return source.OriginalText(artifactLocation(doc.ID), doc.Content)
		},
		DedupKey: func(retrieval.Document[struct{}]) string { return "same" },
	}
	// Act.
	artifact, err := (retrieval.DefaultArtifactRenderer[struct{}]{}).Render(
		context.Background(),
		retrieval.UnrestrictedRead(),
		rs,
		options,
	)
	// Assert.
	if err != nil {
		t.Fatal(err)
	}
	if len(artifact.Snippets) != 1 || artifact.Snippets[0].Content != "beta" {
		t.Fatalf("wrong artifact %+v", artifact)
	}
	snippet := artifact.Snippets[0]
	fragment := snippet.Mapping.Fragments()[0]
	if fragment.Location.Span != (source.ByteSpan{Start: 6, End: 10}) || fragment.Precision != source.ExactPrecision {
		t.Fatal("lost exact mapping")
	}
	supports := map[string]bool{}
	for _, support := range snippet.Supports {
		supports[support.Reference.Source] = true
	}
	if !supports["first"] || !supports["second"] {
		t.Fatal("dedup dropped original source support")
	}
}

func TestArtifactPreservesWhitespaceWithUTF8SourceCoordinates(t *testing.T) {
	// Arrange.
	doc := retrieval.Document[struct{}]{ID: "letters", Content: "  АБВ  "}
	rs := retrieval.NewResultSet([]retrieval.Document[struct{}]{doc}, retrieval.DocumentIDResolver[struct{}]{})
	options := retrieval.ArtifactRenderOptions[struct{}]{
		Resource:  retrieval.RuneResource(1000),
		CloneMeta: func(meta struct{}) (struct{}, error) { return meta, nil },
		Mapping: func(doc retrieval.Document[struct{}]) (source.MappedText, error) {
			location := artifactLocation(doc.ID)
			location.Span = source.ByteSpan{Start: 0, End: len(doc.Content)}
			return source.OriginalText(location, doc.Content)
		},
	}
	// Act.
	artifact, err := (retrieval.DefaultArtifactRenderer[struct{}]{}).Render(
		context.Background(),
		retrieval.UnrestrictedRead(),
		rs,
		options,
	)
	// Assert.
	if err != nil {
		t.Fatal(err)
	}
	snippet := artifact.Snippets[0]
	if snippet.Content != "  АБВ  " ||
		snippet.Mapping.Fragments()[0].Location.Span != (source.ByteSpan{Start: 0, End: len(doc.Content)}) {
		t.Fatalf("whole Unicode source coordinates changed %+v", snippet)
	}
}

func TestArtifactScopeGateStopsCallbacksAndDelivery(t *testing.T) {
	for _, stage := range []string{"before", "mapping", "clone", "provenance", "format", "measure"} {
		t.Run(stage, func(t *testing.T) {
			// Arrange.
			revoked := false
			fixture := newScopeFixture(t, access.AuthorityFunc(func(context.Context, access.Snapshot) error {
				if revoked {
					return ragy.ErrUnavailable
				}
				return nil
			}))
			rs, err := fixture.index.Retrieve(
				context.Background(),
				retrieval.Query[struct{}]{
					Read:    fixture.binding,
					Text:    "policy",
					Options: retrieval.RetrieveOptions{TopK: 10},
				},
			)
			if err != nil {
				t.Fatal(err)
			}
			callbacks := 0
			mark := func(current string) { callbacks++; revoked = revoked || stage == current }
			options := retrieval.ArtifactRenderOptions[accessMeta]{
				Resource: retrieval.RuneResource(1000),
				CloneMeta: func(meta accessMeta) (accessMeta, error) {
					mark("clone")
					return meta, nil
				},
				Mapping: func(doc retrieval.Document[accessMeta]) (source.MappedText, error) {
					mark("mapping")
					location := artifactLocation(doc.ID)
					location.Span = source.ByteSpan{Start: 0, End: len(doc.Content)}
					return source.OriginalText(location, doc.Content)
				},
				Provenance: func(doc retrieval.Document[accessMeta]) retrieval.Provenance {
					mark("provenance")
					return retrieval.Provenance{SourceID: doc.ID}
				},
				FormatSnippet: func(snippet retrieval.ContextSnippet[accessMeta]) (retrieval.FormattedSnippet, error) {
					mark("format")
					return retrieval.FormattedSnippet{
						Text:        snippet.Content,
						ContentSpan: source.ByteSpan{Start: 0, End: len(snippet.Content)},
					}, nil
				},
			}
			options.Resource.Measure = func(_ context.Context, text string) (int64, error) { mark("measure"); return int64(len(text)), nil }
			if stage == "before" {
				revoked = true
			}
			// Act.
			artifact, err := (retrieval.DefaultArtifactRenderer[accessMeta]{}).Render(
				context.Background(),
				fixture.binding,
				rs,
				options,
			)
			// Assert.
			if !errors.Is(err, ragy.ErrUnavailable) || len(artifact.Snippets) != 0 || artifact.RenderedText != "" {
				t.Fatalf("revoked artifact leaked: %+v error=%v", artifact, err)
			}
			if stage == "before" && callbacks != 0 {
				t.Fatal("consumer ran after denial")
			}
		})
	}
}

type mutableArtifactMeta struct{ Labels map[string]string }

func TestArtifactFormatterCannotMutateSnapshotOrSourceMetadata(t *testing.T) {
	// Arrange.
	doc := retrieval.Document[mutableArtifactMeta]{
		ID:      "doc",
		Content: "Alpha beta. Gamma.",
		Meta:    mutableArtifactMeta{Labels: map[string]string{"title": "original"}},
	}
	rs := retrieval.NewResultSet(
		[]retrieval.Document[mutableArtifactMeta]{doc},
		retrieval.DocumentIDResolver[mutableArtifactMeta]{},
	)
	options := retrieval.ArtifactRenderOptions[mutableArtifactMeta]{
		Resource: retrieval.RuneResource(1000),
		CloneMeta: func(meta mutableArtifactMeta) (mutableArtifactMeta, error) {
			return mutableArtifactMeta{Labels: maps.Clone(meta.Labels)}, nil
		},
		Mapping: func(doc retrieval.Document[mutableArtifactMeta]) (source.MappedText, error) {
			return source.OriginalText(artifactLocation(doc.ID), doc.Content)
		},
		FormatSnippet: func(snippet retrieval.ContextSnippet[mutableArtifactMeta]) (retrieval.FormattedSnippet, error) {
			snippet.Meta.Labels["title"] = "changed"
			snippet.Supports[0].Reference.Source = "other"
			return retrieval.FormattedSnippet{
				Text:        snippet.Content,
				ContentSpan: source.ByteSpan{Start: 0, End: len(snippet.Content)},
			}, nil
		},
	}
	// Act.
	artifact, err := (retrieval.DefaultArtifactRenderer[mutableArtifactMeta]{}).Render(
		context.Background(),
		retrieval.UnrestrictedRead(),
		rs,
		options,
	)
	// Assert.
	if err != nil {
		t.Fatal(err)
	}
	if artifact.Snippets[0].Meta.Labels["title"] != "original" ||
		artifact.Snippets[0].Supports[0].Reference.Source != "doc" ||
		doc.Meta.Labels["title"] != "original" {
		t.Fatal("formatter mutated artifact/source snapshot")
	}
}

func TestArtifactRejectsMappingWithStringOnlyRewrite(t *testing.T) {
	rs := retrieval.NewResultSet(
		[]retrieval.Document[struct{}]{{ID: "doc", Content: "Alpha beta. Gamma."}},
		retrieval.DocumentIDResolver[struct{}]{},
	)
	options := retrieval.ArtifactRenderOptions[struct{}]{
		Resource:  retrieval.RuneResource(1000),
		CloneMeta: func(meta struct{}) (struct{}, error) { return meta, nil },
		Mapping: func(doc retrieval.Document[struct{}]) (source.MappedText, error) {
			return source.OriginalText(artifactLocation(doc.ID), doc.Content)
		},
		Snippet: func(retrieval.Document[struct{}]) string { return "rewritten" },
	}
	artifact, err := (retrieval.DefaultArtifactRenderer[struct{}]{}).Render(
		context.Background(),
		retrieval.UnrestrictedRead(),
		rs,
		options,
	)
	if !errors.Is(err, ragy.ErrInvalidArgument) || len(artifact.Snippets) != 0 {
		t.Fatal("rewrite inherited false exact mapping")
	}
}
