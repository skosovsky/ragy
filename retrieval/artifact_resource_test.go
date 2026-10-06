package retrieval

import (
	"context"
	"errors"
	"strings"
	"testing"
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/source"
)

func resourceFixture() ResultSet[struct{}] {
	return NewResultSet(
		[]Document[struct{}]{{ID: "a", Content: "АБВ"}, {ID: "b", Content: "beta"}, {ID: "c", Content: "gamma"}},
		nil,
	)
}

func TestArtifactResourceMeasuresExactWholeNonMonotonicOutput(t *testing.T) {
	// Arrange: the host resource is intentionally non-additive and non-monotonic.
	var measured []string
	resource := RuneResource(3)
	resource.Unit = "host-unit"
	resource.Profile = "contextual/v1"
	resource.Measure = func(_ context.Context, text string) (int64, error) {
		measured = append(measured, text)
		switch {
		case strings.Contains(text, "beta"):
			return 8, nil
		case strings.Contains(text, "gamma"):
			return 2, nil
		default:
			return 3, nil
		}
	}
	formatCalls := 0
	opts := ArtifactRenderOptions[struct{}]{
		Resource:              resource,
		UntrustedDataBoundary: "界",
		CloneMeta:             cloneArtifactValue[struct{}],
		FormatSnippet: func(s ContextSnippet[struct{}]) (FormattedSnippet, error) {
			formatCalls++
			prefix := s.DocumentID + ":"
			return FormattedSnippet{
				Text:        prefix + s.Content,
				ContentSpan: source.ByteSpan{Start: len(prefix), End: len(prefix) + len(s.Content)},
			}, nil
		},
	}
	// Act.
	got, err := (DefaultArtifactRenderer[struct{}]{}).Render(
		context.Background(),
		UnrestrictedRead(),
		resourceFixture(),
		opts,
	)
	// Assert: beta is rejected whole; gamma is measured against the exact retained prefix.
	if err != nil || got.RenderedText != "界\n\na:АБВ\n\nc:gamma" || got.Resource.Used != 2 || len(measured) != 4 ||
		formatCalls != 3 ||
		got.Resource.Packing != ArtifactPackingResourceLimited {
		t.Fatal(got, measured, formatCalls, err)
	}
	if got.RenderedText != measured[len(measured)-1] {
		t.Fatal("returned text was reformatted after measurement")
	}
	for _, s := range got.Snippets {
		if got.RenderedText[s.RenderedSpan.Start:s.RenderedSpan.End] != s.Content || !s.FullDocument ||
			s.DeliveryUncertain {
			t.Fatal(s)
		}
	}
}

func TestArtifactErrorsSuppressPreviouslyPackedContent(t *testing.T) {
	sentinel := errors.New("secret beta source text")
	for _, stage := range []string{"envelope", "measurement", "format", "span", "bytes", "cancel"} {
		t.Run(stage, func(t *testing.T) {
			// Arrange.
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			opts := artifactErrorOptions(stage, cancel, sentinel)
			// Act.
			got, err := (DefaultArtifactRenderer[struct{}]{}).Render(ctx, UnrestrictedRead(), resourceFixture(), opts)
			// Assert.
			if err == nil || got.RenderedText != "" || len(got.Snippets) != 0 {
				t.Fatal("failed artifact leaked", got, err)
			}
			if strings.Contains(err.Error(), "secret") {
				t.Fatal("source leaked through error string", err)
			}
			var failure *ArtifactError
			if stage != "cancel" && !errors.As(err, &failure) {
				t.Fatal("missing typed failure", err)
			}
			if stage == "span" && !errors.Is(err, ragy.ErrProtocol) {
				t.Fatal(err)
			}
			if stage == "cancel" && !errors.Is(err, context.Canceled) {
				t.Fatal(err)
			}
		})
	}
}

func artifactErrorOptions(stage string, cancel context.CancelFunc, sentinel error) ArtifactRenderOptions[struct{}] {
	calls := 0
	r := RuneResource(1000)
	r.Measure = func(_ context.Context, text string) (int64, error) {
		calls++
		if calls == 3 {
			switch stage {
			case "measurement":
				return 0, sentinel
			case "cancel":
				cancel()
			}
		}
		return int64(utf8.RuneCountInString(text)), nil
	}
	if stage == "envelope" {
		r.Limit = 1
	}
	if stage == "bytes" {
		r.MaxOutputBytes = 40
	}
	opts := ArtifactRenderOptions[struct{}]{Resource: r, CloneMeta: cloneArtifactValue[struct{}]}
	if stage == "format" || stage == "span" {
		opts.FormatSnippet = func(s ContextSnippet[struct{}]) (FormattedSnippet, error) {
			if s.DocumentID == "b" {
				if stage == "format" {
					return FormattedSnippet{}, sentinel
				}
				return FormattedSnippet{Text: "wrong", ContentSpan: source.ByteSpan{Start: 0, End: 5}}, nil
			}
			return FormattedSnippet{
				Text:        s.Content,
				ContentSpan: source.ByteSpan{Start: 0, End: len(s.Content)},
			}, nil
		}
	}
	return opts
}

func TestArtifactFiniteLimitsStopCallbacks(t *testing.T) {
	// Arrange.
	r := RuneResource(1000)
	r.MaxMeasurements = 2
	calls := 0
	opts := ArtifactRenderOptions[struct{}]{
		Resource:  r,
		CloneMeta: cloneArtifactValue[struct{}],
		FormatSnippet: func(s ContextSnippet[struct{}]) (FormattedSnippet, error) {
			calls++
			return FormattedSnippet{Text: s.Content, ContentSpan: source.ByteSpan{Start: 0, End: len(s.Content)}}, nil
		},
	}
	// Act.
	got, err := (DefaultArtifactRenderer[struct{}]{}).Render(
		context.Background(),
		UnrestrictedRead(),
		resourceFixture(),
		opts,
	)
	// Assert.
	if err != nil || calls != 1 || got.Resource.Measurements != 2 || len(got.Snippets) != 1 ||
		got.Resource.Packing != ArtifactPackingMeasurementLimited {
		t.Fatal(got, calls, err)
	}
	// Arrange: input limit must reject before arbitrary projection/format callbacks.
	opts.Resource.MaxCandidates = 2
	calls = 0
	// Act.
	got, err = (DefaultArtifactRenderer[struct{}]{}).Render(
		context.Background(),
		UnrestrictedRead(),
		resourceFixture(),
		opts,
	)
	// Assert.
	if !errors.Is(err, ErrArtifactLimit) || calls != 0 || len(got.Snippets) != 0 {
		t.Fatal(got, calls, err)
	}
}

func TestArtifactCountsEnvelopeLabelsSeparatorsAndWholeUnicode(t *testing.T) {
	// Arrange.
	opts := ArtifactRenderOptions[struct{}]{
		Resource:              RuneResource(11),
		UntrustedDataBoundary: "界",
		CloneMeta:             cloneArtifactValue[struct{}],
	}
	rs := NewResultSet([]Document[struct{}]{{ID: "a", Content: "АБВ"}}, nil)
	// Act.
	got, err := (DefaultArtifactRenderer[struct{}]{}).Render(context.Background(), UnrestrictedRead(), rs, opts)
	// Assert: 1 boundary + 2 separator + 6 label/prefix + 3 content = 12.
	if err != nil || len(got.Snippets) != 0 || got.RenderedText != "界" || got.Resource.Used != 1 ||
		got.Resource.Packing != ArtifactPackingResourceLimited {
		t.Fatal(got, err)
	}
	opts.Resource = RuneResource(12)
	got, err = (DefaultArtifactRenderer[struct{}]{}).Render(context.Background(), UnrestrictedRead(), rs, opts)
	if err != nil || got.RenderedText != "界\n\n[1] a\nАБВ" || got.Resource.Used != 12 || len(got.Snippets) != 1 {
		t.Fatal(got, err)
	}
}

type artifactCountSet struct {
	ResultSet[struct{}]

	count int
	calls *int
}

func (r artifactCountSet) Len() int { return r.count }
func (r artifactCountSet) Documents() []Document[struct{}] {
	*r.calls++
	return r.ResultSet.Documents()
}

func TestArtifactRejectsUnboundedAndInconsistentResultBeforeProjection(t *testing.T) {
	for _, count := range []int{2, 1000000} {
		t.Run(string(rune(count)), func(t *testing.T) {
			// Arrange.
			documentsCalls, projectionCalls := 0, 0
			rs := artifactCountSet{ResultSet: resourceFixture(), count: count, calls: &documentsCalls}
			resource := RuneResource(1000)
			resource.MaxCandidates = 3
			opts := ArtifactRenderOptions[struct{}]{
				Resource:  resource,
				CloneMeta: cloneArtifactValue[struct{}],
				Snippet:   func(d Document[struct{}]) string { projectionCalls++; return d.Content },
			}
			// Act.
			got, err := (DefaultArtifactRenderer[struct{}]{}).Render(context.Background(), UnrestrictedRead(), rs, opts)
			// Assert.
			if err == nil || projectionCalls != 0 || len(got.Snippets) != 0 || got.RenderedText != "" {
				t.Fatal(got, err)
			}
			if count > resource.MaxCandidates && documentsCalls != 0 {
				t.Fatal("copied oversized result")
			}
			if count == 2 && !errors.Is(err, ragy.ErrProtocol) {
				t.Fatal(err)
			}
		})
	}
}

func TestArtifactContributorsUseInputPositionInsteadOfDocumentID(t *testing.T) {
	// Arrange: equal IDs do not identify equal host documents.
	rs := NewResultSet([]Document[struct{}]{{ID: "same", Content: "alpha"}, {ID: "same", Content: "different"}}, nil)
	r := RuneResource(17)
	opts := ArtifactRenderOptions[struct{}]{
		Resource:              r,
		UntrustedDataBoundary: "X",
		CloneMeta:             cloneArtifactValue[struct{}],
	}
	// Act.
	got, err := (DefaultArtifactRenderer[struct{}]{}).Render(context.Background(), UnrestrictedRead(), rs, opts)
	// Assert: only alpha fits. Equal IDs cannot invent delivery of the second input.
	if err != nil || len(got.Snippets) != 1 || got.Snippets[0].Content != "alpha" ||
		len(got.Snippets[0].Contributors) != 1 ||
		got.Snippets[0].Contributors[0].InputIndex != 0 ||
		!got.Snippets[0].Contributors[0].FullDocument {
		t.Fatal(got, err)
	}
}

func TestArtifactDedupDifferentContentDoesNotMergeContributorsOrSupports(t *testing.T) {
	// Arrange.
	docs := []Document[struct{}]{{ID: "first", Content: "alpha"}, {ID: "second", Content: "beta"}}
	for i := range docs {
		docs[i].SourceSupports = []source.Locator{artifactResourceLocator(docs[i].ID, len(docs[i].Content))}
	}
	opts := ArtifactRenderOptions[struct{}]{
		Resource:  RuneResource(1000),
		CloneMeta: cloneArtifactValue[struct{}],
		DedupKey:  func(Document[struct{}]) string { return "same" },
	}
	// Act.
	got, err := (DefaultArtifactRenderer[struct{}]{}).Render(
		context.Background(),
		UnrestrictedRead(),
		NewResultSet(docs, nil),
		opts,
	)
	// Assert.
	if err != nil || len(got.Snippets) != 1 {
		t.Fatal(got, err)
	}
	s := got.Snippets[0]
	if len(s.Contributors) != 1 || s.Contributors[0].InputIndex != 0 || len(s.Supports) != 1 ||
		s.Supports[0].Reference.Source != "first" {
		t.Fatal(s)
	}
}

func artifactResourceLocator(id string, size int) source.Locator {
	return source.Locator{
		Reference: source.Reference{
			Namespace:         "docs",
			Source:            id,
			Revision:          "r1",
			Transformation:    "original",
			AccessFingerprint: "acl",
			Artifact:          id,
			Representation:    "text",
		},
		Kind: source.TextLocation,
		Span: source.ByteSpan{Start: 0, End: size},
	}
}

func TestArtifactDedupEqualContentPreservesPerInputDeliveryAndOwnedContributors(t *testing.T) {
	// Arrange: equal projected payloads retain both distinct input positions.
	docs := []Document[struct{}]{{ID: "first", Content: "alpha"}, {ID: "second", Content: "alpha"}}
	for i := range docs {
		m, err := source.OriginalText(artifactResourceLocator(docs[i].ID, len(docs[i].Content)), docs[i].Content)
		if err != nil {
			t.Fatal(err)
		}
		docs[i].SourceMapping = m
	}
	derived, derivedErr := source.DerivedText(
		docs[1].Content,
		[]source.Locator{artifactResourceLocator(docs[1].ID, len(docs[1].Content))},
	)
	if derivedErr != nil {
		t.Fatal(derivedErr)
	}
	docs[1].SourceMapping = derived
	opts := ArtifactRenderOptions[struct{}]{
		Resource:  RuneResource(1000),
		CloneMeta: cloneArtifactValue[struct{}],
		DedupKey:  func(Document[struct{}]) string { return "same" },
		FormatSnippet: func(s ContextSnippet[struct{}]) (FormattedSnippet, error) {
			s.Contributors[0].InputIndex = 999
			return FormattedSnippet{Text: s.Content, ContentSpan: source.ByteSpan{Start: 0, End: len(s.Content)}}, nil
		},
	}
	// Act.
	got, err := (DefaultArtifactRenderer[struct{}]{}).Render(
		context.Background(),
		UnrestrictedRead(),
		NewResultSet(docs, nil),
		opts,
	)
	// Assert: formatter owns its contributor slice; original source precision stays local.
	if err != nil || len(got.Snippets) != 1 {
		t.Fatal(got, err)
	}
	s := got.Snippets[0]
	if len(s.Contributors) != 2 || s.Contributors[0].InputIndex != 0 || s.Contributors[1].InputIndex != 1 ||
		!s.Contributors[0].FullDocument ||
		s.Contributors[1].FullDocument || !s.Contributors[1].DeliveryUncertain ||
		len(s.Supports) != 2 ||
		s.Mapping.Fragments()[0].Location.Reference.Source != "first" ||
		s.Mapping.Fragments()[0].Precision != source.ExactPrecision {
		t.Fatal(s)
	}
}
