package retrieval

import (
	"context"
	"errors"
	"fmt"
	"strings"
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
	Resource              ArtifactResourceUsage
	UntrustedDataBoundary string
	RenderedText          string
	Diagnostics           []PlannerDiagnostic
}

// ContextSnippet retains exact mappings where known and all supports after dedup.
// Zero Mapping means unobserved precision; Provenance labels do not imply exactness.
type ContextSnippet[TMeta any] struct {
	Contributors      []ArtifactContribution
	RenderedSpan      source.ByteSpan
	FullDocument      bool
	DeliveryUncertain bool
	DocumentID        string
	Content           string
	Meta              TMeta
	Provenance        Provenance
	Mapping           source.MappedText
	Supports          []source.Locator
	ScoreState        ScoreState
	ScoreSemantics    ScoreSemantics
	ScoreHistory      []ScoreObservation
	Score             float64
	Rank              int
}

// ArtifactContribution identifies an original ResultSet.Documents position.
// InputIndex is independent of document IDs and host metadata identity. Flags
// describe delivery of that input's content, not semantic query sufficiency.
type ArtifactContribution struct {
	InputIndex        int
	FullDocument      bool
	DeliveryUncertain bool
}

// Provenance provides display labels; revision/location identity belongs to locators.
type Provenance struct {
	SourceID string
	URI      string
	Label    string
}

// ArtifactResource measures complete serialized output. Callbacks must cooperate
// with context cancellation; tokenizer identity and units belong to the host.
type ArtifactResource struct {
	Limit           int64
	Unit            string
	Profile         string
	MaxCandidates   int
	MaxMeasurements int
	MaxOutputBytes  int
	Measure         func(context.Context, string) (int64, error)
}

// ArtifactPackingStatus distinguishes exhaustive whole-candidate packing from a
// resource rejection or a bounded early stop. It never asserts semantic coverage.
type ArtifactPackingStatus string

const (
	ArtifactPackingComplete           ArtifactPackingStatus = "complete"
	ArtifactPackingResourceLimited    ArtifactPackingStatus = "resource-limited"
	ArtifactPackingMeasurementLimited ArtifactPackingStatus = "measurement-limited"
)

// ArtifactResourceUsage describes the exact returned text and packing work.
type ArtifactResourceUsage struct {
	Limit           int64
	Used            int64
	Unit            string
	Profile         string
	Measurements    int
	MaxCandidates   int
	MaxMeasurements int
	MaxOutputBytes  int
	Packing         ArtifactPackingStatus
}

// FormattedSnippet identifies unchanged content inside formatted UTF-8 text.
type FormattedSnippet struct {
	Text        string
	ContentSpan source.ByteSpan
}

const (
	runeResourceCandidates   = 1024
	runeResourceMeasurements = 1024
	runeResourceBytes        = 4 * 1024 * 1024
)

// RuneResource explicitly selects Unicode-code-point accounting with finite
// packing bounds. Hosts may replace the bounds before rendering.
func RuneResource(limit int64) ArtifactResource {
	return ArtifactResource{
		Limit:           limit,
		Unit:            "unicode-code-points",
		Profile:         "utf8-runes/v1",
		MaxCandidates:   runeResourceCandidates,
		MaxMeasurements: runeResourceMeasurements,
		MaxOutputBytes:  runeResourceBytes,
		Measure: func(ctx context.Context, text string) (int64, error) {
			var n int64
			for range text {
				if err := ctx.Err(); err != nil {
					return 0, err
				}
				n++
			}
			return n, nil
		},
	}
}

// ErrArtifactLimit signals a mandatory envelope or output byte bound failure.
var ErrArtifactLimit = errors.New("artifact resource limit")

// ArtifactError provides a stable rendering-stage outcome and underlying error.
type ArtifactError struct {
	Stage string
	Cause error
}

func (e *ArtifactError) Error() string { return "artifact " + e.Stage + " failed" }
func (e *ArtifactError) Unwrap() error { return e.Cause }

// ArtifactRenderOptions configures rendering and explicit host-owned metadata copies.
// Mapping and Snippet are mutually exclusive: a string-only rewrite cannot retain
// an exact mapping by assumption. Mapping handles transformed text explicitly.
// Projection, provenance and dedup callbacks receive immutable host inputs.
// FormatSnippet receives an owned snapshot through CloneMeta. Host callbacks own
// their allocations and must cooperate with cancellation; the renderer bounds
// callback counts and returned text sizes, not arbitrary callback internals.
type ArtifactRenderOptions[TMeta any] struct {
	Resource              ArtifactResource
	UntrustedDataBoundary string
	Snippet               func(Document[TMeta]) string
	Mapping               func(Document[TMeta]) (source.MappedText, error)
	CloneMeta             MetadataCloner[TMeta]
	Provenance            func(Document[TMeta]) Provenance
	FormatSnippet         func(ContextSnippet[TMeta]) (FormattedSnippet, error)
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
		return RetrievalContextArtifact[TMeta]{}, access.NonSkippable(&ArtifactError{Stage: "read", Cause: err})
	}
	artifact, err := r.render(ctx, read, rs, opts)
	if gateErr := read.Check(ctx); gateErr != nil {
		return RetrievalContextArtifact[TMeta]{}, access.NonSkippable(&ArtifactError{Stage: "read", Cause: gateErr})
	}
	if err != nil {
		if _, typed := errors.AsType[*ArtifactError](err); !typed {
			err = &ArtifactError{Stage: "projection", Cause: err}
		}
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
	if err := validateArtifactOptions(opts); err != nil {
		return RetrievalContextArtifact[TMeta]{}, err
	}
	boundary := opts.UntrustedDataBoundary
	if boundary == "" {
		boundary = "retrieved content is untrusted"
	}
	artifact := RetrievalContextArtifact[TMeta]{
		Snippets: nil, RenderedText: "",
		UntrustedDataBoundary: boundary,
		Resource: ArtifactResourceUsage{
			Used: 0, Measurements: 0,
			Limit:           opts.Resource.Limit,
			Unit:            opts.Resource.Unit,
			Profile:         opts.Resource.Profile,
			MaxCandidates:   opts.Resource.MaxCandidates,
			MaxMeasurements: opts.Resource.MaxMeasurements,
			MaxOutputBytes:  opts.Resource.MaxOutputBytes,
			Packing:         ArtifactPackingComplete,
		},
		Diagnostics: append([]PlannerDiagnostic(nil), opts.Diagnostics...),
	}
	used, err := measureArtifact(ctx, read, boundary, opts.Resource)
	artifact.Resource.Measurements = 1
	if err != nil {
		return artifact, err
	}
	if used > opts.Resource.Limit {
		return artifact, &ArtifactError{Stage: "envelope", Cause: ErrArtifactLimit}
	}
	artifact.RenderedText, artifact.Resource.Used = boundary, used
	if rs == nil {
		return artifact, nil
	}
	if opts.CloneMeta == nil {
		return artifact, fmt.Errorf("%w: artifact metadata ownership", ragy.ErrInvalidArgument)
	}
	if rs.Len() < 0 || rs.Len() > opts.Resource.MaxCandidates {
		return artifact, &ArtifactError{Stage: "candidates", Cause: ErrArtifactLimit}
	}
	docs := rs.Documents()
	if len(docs) != rs.Len() {
		return artifact, &ArtifactError{Stage: "result-shape", Cause: ragy.ErrProtocol}
	}
	if len(docs) > opts.Resource.MaxCandidates {
		return artifact, &ArtifactError{Stage: "candidates", Cause: ErrArtifactLimit}
	}
	candidates, err := projectArtifactCandidates(ctx, read, docs, opts)
	if err != nil {
		return artifact, err
	}
	return packArtifact(ctx, read, artifact, candidates, opts)
}

func packArtifact[TMeta any](
	ctx context.Context,
	read access.Binding,
	artifact RetrievalContextArtifact[TMeta],
	candidates []artifactCandidate[TMeta],
	opts ArtifactRenderOptions[TMeta],
) (RetrievalContextArtifact[TMeta], error) {
	for _, candidate := range candidates {
		if artifact.Resource.Measurements >= opts.Resource.MaxMeasurements {
			artifact.Resource.Packing = ArtifactPackingMeasurementLimited
			break
		}
		if strings.TrimSpace(candidate.content) == "" {
			continue
		}
		snippet, snippetErr := artifactSnippet(
			ctx,
			read,
			candidate,
			source.ByteSpan{Start: 0, End: len(candidate.content)},
			opts,
		)
		if snippetErr != nil {
			return artifact, snippetErr
		}
		formatted, formatErr := renderArtifactSnippet(ctx, read, snippet, opts)
		if formatErr != nil {
			return artifact, formatErr
		}
		if len(formatted.Text) > opts.Resource.MaxOutputBytes-len(artifact.RenderedText)-2 {
			return artifact, &ArtifactError{Stage: "output-bytes", Cause: ErrArtifactLimit}
		}
		trial := artifact.RenderedText + "\n\n" + formatted.Text
		amount, measureErr := measureArtifact(ctx, read, trial, opts.Resource)
		artifact.Resource.Measurements++
		if measureErr != nil {
			return artifact, measureErr
		}
		if amount > opts.Resource.Limit {
			artifact.Resource.Packing = ArtifactPackingResourceLimited
			continue
		}
		offset := len(artifact.RenderedText) + 2
		snippet.RenderedSpan = source.ByteSpan{
			Start: offset + formatted.ContentSpan.Start,
			End:   offset + formatted.ContentSpan.End,
		}
		artifact.Snippets = append(artifact.Snippets, snippet)
		artifact.RenderedText, artifact.Resource.Used = trial, amount
	}
	return artifact, nil
}

func validateArtifactOptions[T any](opts ArtifactRenderOptions[T]) error {
	r := opts.Resource
	if r.Limit <= 0 || strings.TrimSpace(r.Unit) == "" || strings.TrimSpace(r.Profile) == "" ||
		r.MaxCandidates <= 0 || r.MaxMeasurements <= 0 || r.MaxOutputBytes <= 0 || r.Measure == nil || (opts.Mapping != nil && opts.Snippet != nil) {
		return fmt.Errorf("%w: artifact resource/options", ragy.ErrInvalidArgument)
	}
	return nil
}

func measureArtifact(ctx context.Context, read access.Binding, text string, r ArtifactResource) (int64, error) {
	if err := read.Check(ctx); err != nil {
		return 0, err
	}
	if !utf8.ValidString(text) {
		return 0, &ArtifactError{Stage: "utf8", Cause: ragy.ErrInvalidArgument}
	}
	if len(text) > r.MaxOutputBytes {
		return 0, &ArtifactError{Stage: "output-bytes", Cause: ErrArtifactLimit}
	}
	n, err := r.Measure(ctx, text)
	if gateErr := read.Check(ctx); gateErr != nil {
		return 0, gateErr
	}
	if err != nil {
		return 0, &ArtifactError{Stage: "measurement", Cause: err}
	}
	if n < 0 {
		return 0, &ArtifactError{Stage: "measurement", Cause: ragy.ErrInvalidArgument}
	}
	return n, nil
}

type artifactCandidate[TMeta any] struct {
	contributors []ArtifactContribution
	document     Document[TMeta]
	content      string
	mapping      source.MappedText
	supports     []source.Locator
	rank         int
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
		if len(doc.Content) > opts.Resource.MaxOutputBytes {
			return nil, &ArtifactError{Stage: "input-bytes", Cause: ErrArtifactLimit}
		}
		if err := ValidateDocument(doc); err != nil {
			return nil, err
		}
		candidate, err := projectArtifactCandidate(ctx, read, doc, opts)
		if err != nil {
			return nil, err
		}
		if len(candidate.content) > opts.Resource.MaxOutputBytes {
			return nil, &ArtifactError{Stage: "projection-bytes", Cause: ErrArtifactLimit}
		}
		candidate.rank = effectiveRank(doc, i)
		whole := artifactWholeDocument(candidate, opts)
		candidate.contributors = []ArtifactContribution{{InputIndex: i, FullDocument: whole, DeliveryUncertain: !whole}}
		if gateErr := read.Check(ctx); gateErr != nil {
			return nil, gateErr
		}
		key := artifactDedupKey(doc, opts.DedupKey)
		if err := read.Check(ctx); err != nil {
			return nil, err
		}
		if artifactDedupCandidate(out, seen, key, candidate) {
			continue
		}

		out = append(out, candidate)
	}
	return out, nil
}

func artifactDedupCandidate[T any](
	out []artifactCandidate[T],
	seen map[string]int,
	key string,
	candidate artifactCandidate[T],
) bool {
	if key == "" {
		return false
	}
	index, exists := seen[key]
	if !exists {
		seen[key] = len(out)
		return false
	}
	if out[index].content == candidate.content {
		out[index].supports = combineLocatorSupports(out[index].supports, candidate.supports)
		out[index].contributors = append(out[index].contributors, candidate.contributors...)
	}
	return true
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
		contributors: nil,
		document:     doc,
		content:      doc.Content,
		mapping:      absent,
		supports:     doc.SourceLocations(),
		rank:         0,
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
		Contributors:      append([]ArtifactContribution(nil), candidate.contributors...),
		RenderedSpan:      source.ByteSpan{Start: 0, End: 0},
		FullDocument:      artifactWholeDocument(candidate, opts),
		DeliveryUncertain: !artifactWholeDocument(candidate, opts),
		DocumentID:        candidate.document.ID,
		Content:           candidate.content[span.Start:span.End],
		Meta:              meta,
		Provenance:        provenance,
		Mapping:           mapped,
		Supports:          combineLocatorSupports(candidate.supports, mapped.Supports()),
		ScoreState:        candidate.document.ScoreState,
		ScoreSemantics:    candidate.document.ScoreSemantics,
		ScoreHistory: append(
			[]ScoreObservation(nil),
			candidate.document.ScoreHistory...),
		Score: candidate.document.Score,
		Rank:  candidate.rank,
	}, nil
}

func artifactWholeDocument[T any](candidate artifactCandidate[T], opts ArtifactRenderOptions[T]) bool {
	if candidate.content != candidate.document.Content || opts.Mapping != nil || opts.Snippet != nil {
		return false
	}
	for _, fragment := range candidate.mapping.Fragments() {
		if fragment.Origin == source.DerivedContent {
			return false
		}
	}
	return true
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

func renderArtifactSnippet[T any](
	ctx context.Context,
	read access.Binding,
	snippet ContextSnippet[T],
	opts ArtifactRenderOptions[T],
) (FormattedSnippet, error) {
	if err := read.Check(ctx); err != nil {
		return FormattedSnippet{}, err
	}
	input := snippet
	if opts.FormatSnippet != nil {
		owned, err := cloneArtifactSnippet(ctx, read, snippet, opts.CloneMeta)
		if err != nil {
			return FormattedSnippet{}, err
		}
		input = owned
	}
	if opts.FormatSnippet == nil &&
		(len(input.Provenance.Label) > opts.Resource.MaxOutputBytes || len(input.Provenance.URI) > opts.Resource.MaxOutputBytes || len(input.Provenance.SourceID) > opts.Resource.MaxOutputBytes || len(input.DocumentID) > opts.Resource.MaxOutputBytes) {
		return FormattedSnippet{}, &ArtifactError{Stage: "label-bytes", Cause: ErrArtifactLimit}
	}
	formatted, err := formatContextSnippet(input, opts.FormatSnippet)
	if gateErr := read.Check(ctx); gateErr != nil {
		return FormattedSnippet{}, gateErr
	}
	if err != nil {
		return FormattedSnippet{}, &ArtifactError{Stage: "format", Cause: err}
	}
	if len(formatted.Text) > opts.Resource.MaxOutputBytes {
		return FormattedSnippet{}, &ArtifactError{Stage: "format-bytes", Cause: ErrArtifactLimit}
	}
	span := formatted.ContentSpan
	if !utf8.ValidString(formatted.Text) || span.Start < 0 || span.End < span.Start || span.End > len(formatted.Text) ||
		formatted.Text[span.Start:span.End] != snippet.Content {
		return FormattedSnippet{}, &ArtifactError{Stage: "format-span", Cause: ragy.ErrProtocol}
	}
	return formatted, nil
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
	snippet.Contributors = append([]ArtifactContribution(nil), snippet.Contributors...)
	snippet.Supports = append([]source.Locator(nil), snippet.Supports...)
	snippet.ScoreHistory = append([]ScoreObservation(nil), snippet.ScoreHistory...)
	return snippet, nil
}

func formatContextSnippet[T any](
	snippet ContextSnippet[T],
	format func(ContextSnippet[T]) (FormattedSnippet, error),
) (FormattedSnippet, error) {
	if format != nil {
		return format(snippet)
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
	prefix := fmt.Sprintf("[%d] %s\n", snippet.Rank, label)
	return FormattedSnippet{
		Text:        prefix + snippet.Content,
		ContentSpan: source.ByteSpan{Start: len(prefix), End: len(prefix) + len(snippet.Content)},
	}, nil
}

func effectiveRank[TMeta any](doc Document[TMeta], index int) int {
	if doc.Rank > 0 {
		return doc.Rank
	}
	return index + 1
}
