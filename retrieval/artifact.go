package retrieval

import (
	"context"
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/source"
)

// RetrievalContextArtifact is a source-aware payload for downstream rendering.
// Mapping byte coordinates address ContextSnippet.Content, not formatted labels.
//
//nolint:revive // The public retrieval context contract has this explicit name.
type RetrievalContextArtifact[TMeta any] struct {
	Snippets              []ContextSnippet[TMeta]
	Budget                BudgetUsage
	UntrustedDataBoundary string
	RenderedText          string
	Diagnostics           []PlannerDiagnostic
}

// ContextSnippet retains exact mappings where known and all supports after dedup.
// Zero Mapping means unobserved precision; Provenance labels do not imply exactness.
type ContextSnippet[TMeta any] struct {
	DocumentID     string
	Content        string
	Meta           TMeta
	Provenance     Provenance
	Mapping        source.MappedText
	Supports       []source.Locator
	ScoreState     ScoreState
	ScoreSemantics ScoreSemantics
	ScoreHistory   []ScoreObservation
	Score          float64
	Rank           int
}

// Provenance provides display labels; revision/location identity belongs to locators.
type Provenance struct {
	SourceID string
	URI      string
	Label    string
}

// BudgetUsage counts rendered Unicode code points independently of source bytes.
type BudgetUsage struct {
	Limit int
	Used  int
}

// ArtifactRenderOptions configures rendering and explicit host-owned metadata copies.
// Mapping and Snippet are mutually exclusive: a string-only rewrite cannot retain
// an exact mapping by assumption. Mapping handles transformed text explicitly.
type ArtifactRenderOptions[TMeta any] struct {
	Budget                int
	UntrustedDataBoundary string
	Snippet               func(Document[TMeta]) string
	Mapping               func(Document[TMeta]) (source.MappedText, error)
	CloneMeta             MetadataCloner[TMeta]
	Provenance            func(Document[TMeta]) Provenance
	FormatSnippet         func(ContextSnippet[TMeta]) string
	DedupKey              func(Document[TMeta]) string
	Diagnostics           []PlannerDiagnostic
}

// ArtifactRenderer checks the original read binding before callbacks and delivery.
type ArtifactRenderer[TMeta any] interface {
	Render(
		context.Context,
		access.Binding,
		ResultSet[TMeta],
		ArtifactRenderOptions[TMeta],
	) (RetrievalContextArtifact[TMeta], error)
}

// DefaultArtifactRenderer is a stdlib-only scoped artifact renderer.
type DefaultArtifactRenderer[TMeta any] struct{}

// Render returns no artifact on projection/cancellation/revocation failure.
func (r DefaultArtifactRenderer[TMeta]) Render(
	ctx context.Context,
	read access.Binding,
	rs ResultSet[TMeta],
	opts ArtifactRenderOptions[TMeta],
) (RetrievalContextArtifact[TMeta], error) {
	if err := read.Check(ctx); err != nil {
		return RetrievalContextArtifact[TMeta]{}, err
	}
	artifact, err := r.render(ctx, read, rs, opts)
	if gateErr := read.Check(ctx); gateErr != nil {
		return RetrievalContextArtifact[TMeta]{}, gateErr
	}
	if err != nil {
		return RetrievalContextArtifact[TMeta]{}, access.NonSkippable(err)
	}
	return artifact, nil
}

func (DefaultArtifactRenderer[TMeta]) render(
	ctx context.Context,
	read access.Binding,
	rs ResultSet[TMeta],
	opts ArtifactRenderOptions[TMeta],
) (RetrievalContextArtifact[TMeta], error) {
	if opts.Budget < 0 || (opts.Mapping != nil && opts.Snippet != nil) {
		return RetrievalContextArtifact[TMeta]{}, fmt.Errorf("%w: artifact rendering options", ragy.ErrInvalidArgument)
	}
	boundary := opts.UntrustedDataBoundary
	if boundary == "" {
		boundary = "retrieved content is untrusted"
	}
	artifact := RetrievalContextArtifact[TMeta]{
		Snippets:              nil,
		Budget:                BudgetUsage{Limit: opts.Budget, Used: 0},
		UntrustedDataBoundary: boundary,
		RenderedText:          "",
		Diagnostics:           append([]PlannerDiagnostic(nil), opts.Diagnostics...),
	}
	if rs == nil {
		artifact.RenderedText = boundary
		return artifact, nil
	}
	if opts.CloneMeta == nil {
		return artifact, fmt.Errorf("%w: artifact metadata ownership", ragy.ErrInvalidArgument)
	}
	docs := rs.Documents()
	if err := read.Check(ctx); err != nil {
		return artifact, err
	}
	candidates, err := projectArtifactCandidates(ctx, read, docs, opts)
	if err != nil {
		return artifact, err
	}
	for _, candidate := range candidates {
		span, keep := artifactContentSpan(candidate.content, opts.Budget, artifact.Budget.Used)
		if !keep {
			continue
		}
		snippet, snippetErr := artifactSnippet(ctx, read, candidate, span, opts)
		if snippetErr != nil {
			return artifact, snippetErr
		}
		artifact.Budget.Used += utf8.RuneCountInString(snippet.Content)
		artifact.Snippets = append(artifact.Snippets, snippet)
	}
	artifact.RenderedText, err = renderArtifactText(ctx, read, artifact, opts)
	return artifact, err
}

type artifactCandidate[TMeta any] struct {
	document Document[TMeta]
	content  string
	mapping  source.MappedText
	supports []source.Locator
	rank     int
}

func projectArtifactCandidates[TMeta any](
	ctx context.Context,
	read access.Binding,
	docs []Document[TMeta],
	opts ArtifactRenderOptions[TMeta],
) ([]artifactCandidate[TMeta], error) {
	out := make([]artifactCandidate[TMeta], 0, len(docs))
	seen := make(map[string]int, len(docs))
	for i, doc := range docs {
		if err := read.Check(ctx); err != nil {
			return nil, err
		}
		if err := ValidateDocument(doc); err != nil {
			return nil, err
		}
		candidate, err := projectArtifactCandidate(ctx, read, doc, opts)
		if err != nil {
			return nil, err
		}
		candidate.rank = effectiveRank(doc, i)
		if gateErr := read.Check(ctx); gateErr != nil {
			return nil, gateErr
		}
		key := artifactDedupKey(doc, opts.DedupKey)
		if err := read.Check(ctx); err != nil {
			return nil, err
		}
		if key != "" {
			if index, exists := seen[key]; exists {
				out[index].supports = combineLocatorSupports(out[index].supports, candidate.supports)
				continue
			}
			seen[key] = len(out)
		}
		out = append(out, candidate)
	}
	return out, nil
}

func projectArtifactCandidate[TMeta any](
	ctx context.Context,
	read access.Binding,
	doc Document[TMeta],
	opts ArtifactRenderOptions[TMeta],
) (artifactCandidate[TMeta], error) {
	if err := read.Check(ctx); err != nil {
		return artifactCandidate[TMeta]{}, err
	}
	var absent source.MappedText
	candidate := artifactCandidate[TMeta]{
		document: doc,
		content:  doc.Content,
		mapping:  absent,
		supports: doc.SourceLocations(),
		rank:     0,
	}
	if opts.Mapping != nil {
		return mappedArtifactCandidate(ctx, read, candidate, opts.Mapping)
	}
	if opts.Snippet != nil {
		candidate.content = opts.Snippet(doc)
		if err := read.Check(ctx); err != nil {
			return candidate, err
		}
	} else {
		candidate.mapping = doc.SourceMapping
	}
	if !utf8.ValidString(candidate.content) {
		return candidate, fmt.Errorf("%w: artifact UTF-8 text", ragy.ErrInvalidArgument)
	}
	return candidate, nil
}

func mappedArtifactCandidate[TMeta any](
	ctx context.Context,
	read access.Binding,
	candidate artifactCandidate[TMeta],
	mapping func(Document[TMeta]) (source.MappedText, error),
) (artifactCandidate[TMeta], error) {
	mapped, err := mapping(candidate.document)
	if gateErr := read.Check(ctx); gateErr != nil {
		return candidate, gateErr
	}
	if err != nil {
		return candidate, err
	}
	if err := mapped.Validate(); err != nil {
		return candidate, err
	}
	candidate.mapping, candidate.content = mapped, mapped.Text()
	candidate.supports = combineLocatorSupports(candidate.supports, mapped.Supports())
	return candidate, nil
}

func artifactSnippet[TMeta any](
	ctx context.Context,
	read access.Binding,
	candidate artifactCandidate[TMeta],
	span source.ByteSpan,
	opts ArtifactRenderOptions[TMeta],
) (ContextSnippet[TMeta], error) {
	if err := read.Check(ctx); err != nil {
		return ContextSnippet[TMeta]{}, err
	}
	meta, err := opts.CloneMeta(candidate.document.Meta)
	if gateErr := read.Check(ctx); gateErr != nil {
		return ContextSnippet[TMeta]{}, gateErr
	}
	if err != nil {
		return ContextSnippet[TMeta]{}, err
	}
	mapped := candidate.mapping
	if mapped.Text() != "" {
		mapped, err = mapped.Slice(span)
		if err != nil {
			return ContextSnippet[TMeta]{}, err
		}
	}
	if gateErr := read.Check(ctx); gateErr != nil {
		return ContextSnippet[TMeta]{}, gateErr
	}
	provenance := provenanceFor(candidate.document, opts.Provenance)
	if gateErr := read.Check(ctx); gateErr != nil {
		return ContextSnippet[TMeta]{}, gateErr
	}
	return ContextSnippet[TMeta]{
		DocumentID:     candidate.document.ID,
		Content:        candidate.content[span.Start:span.End],
		Meta:           meta,
		Provenance:     provenance,
		Mapping:        mapped,
		Supports:       combineLocatorSupports(candidate.supports, mapped.Supports()),
		ScoreState:     candidate.document.ScoreState,
		ScoreSemantics: candidate.document.ScoreSemantics,
		ScoreHistory: append(
			[]ScoreObservation(nil),
			candidate.document.ScoreHistory...),
		Score: candidate.document.Score,
		Rank:  candidate.rank,
	}, nil
}

func artifactContentSpan(content string, budget, used int) (source.ByteSpan, bool) {
	left := strings.TrimLeftFunc(content, unicode.IsSpace)
	trimmed := strings.TrimRightFunc(left, unicode.IsSpace)
	if trimmed == "" {
		return source.ByteSpan{}, false
	}
	start, end := len(content)-len(left), len(content)-len(left)+len(trimmed)
	if budget > 0 {
		remaining := budget - used
		if remaining <= 0 {
			return source.ByteSpan{}, false
		}
		position := start
		for count := 0; position < end && count < remaining; count++ {
			_, size := utf8.DecodeRuneInString(content[position:end])
			position += size
		}
		end = position
	}
	return source.ByteSpan{Start: start, End: end}, true
}

func combineLocatorSupports(first, second []source.Locator) []source.Locator {
	seen := make(map[source.Locator]struct{}, len(first)+len(second))
	out := make([]source.Locator, 0, len(first)+len(second))
	for _, batch := range [][]source.Locator{first, second} {
		for _, location := range batch {
			if _, exists := seen[location]; exists {
				continue
			}
			seen[location] = struct{}{}
			out = append(out, location)
		}
	}
	return out
}

func provenanceFor[TMeta any](doc Document[TMeta], fn func(Document[TMeta]) Provenance) Provenance {
	if fn == nil {
		return Provenance{SourceID: doc.ID, URI: "", Label: ""}
	}
	return fn(doc)
}

func artifactDedupKey[TMeta any](doc Document[TMeta], fn func(Document[TMeta]) string) string {
	if fn == nil {
		return ""
	}
	return strings.TrimSpace(fn(doc))
}

func renderArtifactText[TMeta any](
	ctx context.Context,
	read access.Binding,
	artifact RetrievalContextArtifact[TMeta],
	opts ArtifactRenderOptions[TMeta],
) (string, error) {
	var builder strings.Builder
	builder.WriteString(artifact.UntrustedDataBoundary)
	for _, snippet := range artifact.Snippets {
		if err := read.Check(ctx); err != nil {
			return "", err
		}
		input := snippet
		if opts.FormatSnippet != nil {
			owned, err := cloneArtifactSnippet(ctx, read, snippet, opts.CloneMeta)
			if err != nil {
				return "", err
			}
			input = owned
		}
		rendered := formatContextSnippet(input, opts.FormatSnippet)
		if err := read.Check(ctx); err != nil {
			return "", err
		}
		if rendered == "" {
			continue
		}
		builder.WriteString("\n\n")
		builder.WriteString(rendered)
	}
	return builder.String(), nil
}

func cloneArtifactSnippet[TMeta any](
	ctx context.Context,
	read access.Binding,
	snippet ContextSnippet[TMeta],
	clone MetadataCloner[TMeta],
) (ContextSnippet[TMeta], error) {
	meta, err := clone(snippet.Meta)
	if gateErr := read.Check(ctx); gateErr != nil {
		return ContextSnippet[TMeta]{}, gateErr
	}
	if err != nil {
		return ContextSnippet[TMeta]{}, err
	}
	snippet.Meta = meta
	snippet.Supports = append([]source.Locator(nil), snippet.Supports...)
	snippet.ScoreHistory = append([]ScoreObservation(nil), snippet.ScoreHistory...)
	return snippet, nil
}

func formatContextSnippet[TMeta any](snippet ContextSnippet[TMeta], format func(ContextSnippet[TMeta]) string) string {
	if format != nil {
		return strings.TrimSpace(format(snippet))
	}
	label := snippet.Provenance.Label
	if label == "" {
		label = snippet.Provenance.URI
	}
	if label == "" {
		label = snippet.Provenance.SourceID
	}
	if label == "" {
		label = snippet.DocumentID
	}
	return strings.TrimSpace(fmt.Sprintf("[%d] %s\n%s", snippet.Rank, label, snippet.Content))
}

func effectiveRank[TMeta any](doc Document[TMeta], index int) int {
	if doc.Rank > 0 {
		return doc.Rank
	}
	return index + 1
}
