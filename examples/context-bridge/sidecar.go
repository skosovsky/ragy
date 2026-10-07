package bridge

import (
	"bytes"
	"context"
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"errors"

	"io"
	"math"
	"slices"
	"unicode/utf8"

	"github.com/santhosh-tekuri/jsonschema/v6"

	"github.com/skosovsky/contexty"
	"github.com/skosovsky/memy"

	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

// SidecarSchema identifies the only accepted evidence wire contract.
//
//go:embed sidecar.schema.json
var wireSchema []byte

const SidecarSchema = "host.retrieval-context/1"
const maxJSONDepth = 64
const preservedPrecision = "preserved"
const unavailablePrecision = "unavailable"
const maxWireBytes = 16 * 1024 * 1024

// SourceIdentity is canonical source provenance, independent of canonical record revision.
type SourceIdentity struct {
	ID       string `json:"id"`
	Revision string `json:"revision"`
}

// Input retains original index score evidence and explicit canonical mapping.
type Input[U any] struct {
	Extractor      string                       `json:"extractor"`
	Losses         []string                     `json:"losses"`
	Uncertainties  []string                     `json:"uncertainties"`
	Sources        []SourceIdentity             `json:"sources"`
	Reference      Reference                    `json:"reference"`
	ArtifactID     string                       `json:"artifact_id"`
	Rank           int                          `json:"rank"`
	Score          float64                      `json:"score"`
	ScoreState     retrieval.ScoreState         `json:"score_state"`
	ScoreSemantics retrieval.ScoreSemantics     `json:"score_semantics"`
	ScoreHistory   []retrieval.ScoreObservation `json:"score_history"`
	Uncertainty    *U                           `json:"uncertainty"`
}

// Snippet excludes private metadata but keeps every renderer evidence association.
type Snippet struct {
	DocumentID        string                           `json:"document_id"`
	Content           string                           `json:"content"`
	Span              *source.ByteSpan                 `json:"span"`
	Mapping           source.MappedText                `json:"mapping"`
	Supports          []source.Locator                 `json:"supports"`
	Contributors      []retrieval.ArtifactContribution `json:"contributors"`
	FullDocument      bool                             `json:"full_document"`
	DeliveryUncertain bool                             `json:"delivery_uncertain"`
	Provenance        retrieval.Provenance             `json:"provenance"`
	Score             float64                          `json:"score"`
	ScoreState        retrieval.ScoreState             `json:"score_state"`
	ScoreSemantics    retrieval.ScoreSemantics         `json:"score_semantics"`
	ScoreHistory      []retrieval.ScoreObservation     `json:"score_history"`
	Rank              int                              `json:"rank"`
}

// Sidecar binds private evidence to one final decoded text, not JSON packing coordinates.
type Sidecar[U any] struct {
	Digest               string                          `json:"digest"`
	ProjectionCount      int                             `json:"projection_count"`
	RenderedSnippetCount int                             `json:"rendered_snippet_count"`
	UncertaintyType      string                          `json:"uncertainty_type"`
	Schema               string                          `json:"schema"`
	Scope                memy.Scope                      `json:"scope"`
	Epoch                memy.Version                    `json:"epoch"`
	Text                 string                          `json:"text"`
	RankPolicy           string                          `json:"rank_policy"`
	References           []Reference                     `json:"references"`
	Inputs               []Input[U]                      `json:"inputs"`
	Snippets             []Snippet                       `json:"snippets"`
	Coverage             []memy.Coverage                 `json:"coverage"`
	Progress             memy.RecallProgress             `json:"progress"`
	Partial              bool                            `json:"partial"`
	Omissions            int                             `json:"omissions"`
	Truncated            bool                            `json:"truncated"`
	Precision            string                          `json:"precision"`
	Resource             retrieval.ArtifactResourceUsage `json:"resource"`
	Diagnostics          []retrieval.PlannerDiagnostic   `json:"diagnostics"`
	Boundary             string                          `json:"boundary"`
}

func newSidecar[M, U, P, R any](
	a retrieval.RetrievalContextArtifact[M],
	refs []Reference,
	uncertainties []*U,
	scope memy.Scope,
	epoch memy.Version,
	batch Batch[M],
	progress memy.RecallProgress,
	items []mapped[M],
	records []memy.Ranked[P, R],
) Sidecar[U] {
	side := Sidecar[U]{
		ProjectionCount: len(refs), RenderedSnippetCount: len(a.Snippets),
		Schema:      SidecarSchema,
		Scope:       scope,
		Epoch:       epoch,
		Text:        a.RenderedText,
		RankPolicy:  RankPolicy,
		References:  refs,
		Coverage:    slices.Clone(batch.Coverage),
		Progress:    progress,
		Partial:     batch.Partial,
		Omissions:   batch.Omissions,
		Precision:   preservedPrecision,
		Resource:    a.Resource,
		Diagnostics: a.Diagnostics,
		Boundary:    a.UntrustedDataBoundary,
	}
	for i, ref := range refs {
		for _, item := range items {
			if item.ref == ref {
				d := item.doc
				side.Inputs = append(
					side.Inputs,
					Input[U]{
						Extractor:      records[i].Record.Provenance.Extractor,
						Losses:         slices.Clone(records[i].Record.Provenance.Losses),
						Uncertainties:  slices.Clone(records[i].Record.Provenance.Uncertainties),
						Sources:        canonicalSources(records[i].Record),
						Reference:      ref,
						ArtifactID:     d.ID,
						Rank:           d.Rank,
						Score:          d.Score,
						ScoreState:     d.ScoreState,
						ScoreSemantics: d.ScoreSemantics,
						ScoreHistory:   d.ScoreHistory,
						Uncertainty:    uncertainties[i],
					},
				)
				break
			}
		}
	}
	for _, s := range a.Snippets {
		span := s.RenderedSpan
		side.Snippets = append(side.Snippets, Snippet{DocumentID: s.DocumentID, Content: s.Content, Span: &span,
			Mapping: s.Mapping, Supports: s.Supports, Contributors: s.Contributors, FullDocument: s.FullDocument,
			DeliveryUncertain: s.DeliveryUncertain, Provenance: s.Provenance, Score: s.Score, ScoreState: s.ScoreState,
			ScoreSemantics: s.ScoreSemantics, ScoreHistory: s.ScoreHistory, Rank: s.Rank})
	}
	return side
}

func transform[M, U any](ctx context.Context, s Sidecar[U], o Options[M]) (Sidecar[U], error) {
	if err := ctx.Err(); err != nil {
		return Sidecar[U]{}, err
	}
	rewritten := o.Rewrite != nil
	if rewritten {
		text, err := o.Rewrite(ctx, s.Text)
		if err != nil {
			return Sidecar[U]{}, err
		}
		s.Text = text
	}
	s.Text = Wrap(s.Text, o.Prefix, o.Suffix)
	if s.UncertaintyType == "" || !utf8.ValidString(s.Text) {
		return Sidecar[U]{}, memy.ErrInvalid
	}
	for i := range s.Snippets {
		if s.Snippets[i].Span != nil {
			s.Snippets[i].Span.Start += len(o.Prefix)
			s.Snippets[i].Span.End += len(o.Prefix)
		}
	}
	if o.TruncateRunes > 0 && utf8.RuneCountInString(s.Text) > o.TruncateRunes {
		end := 0
		for range o.TruncateRunes {
			_, size := utf8.DecodeRuneInString(s.Text[end:])
			end += size
		}
		s.Text = s.Text[:end]
		s.Truncated = true
	}
	if rewritten || s.Truncated {
		s.Precision = unavailablePrecision
		for i := range s.Snippets {
			s.Snippets[i].Span = nil
			s.Snippets[i].DeliveryUncertain = true
			s.Snippets[i].FullDocument = false
			for j := range s.Snippets[i].Contributors {
				s.Snippets[i].Contributors[j].DeliveryUncertain = true
				s.Snippets[i].Contributors[j].FullDocument = false
			}
		}
	}
	return s, ctx.Err()
}

func validateSidecar[U any](s Sidecar[U]) error {
	if s.Schema != SidecarSchema {
		return memy.ErrUnsupported
	}
	if err := s.Scope.Validate(); err != nil {
		return err
	}
	if s.UncertaintyType == "" || !utf8.ValidString(s.Text) || s.RankPolicy != RankPolicy || s.Omissions < 0 ||
		s.Epoch > memy.MaxVersion ||
		(s.Precision != preservedPrecision && s.Precision != unavailablePrecision) ||
		len(
			s.References,
		) != len(
			s.Inputs,
		) || len(s.Inputs) != s.ProjectionCount || len(s.Snippets) != s.RenderedSnippetCount {
		return memy.ErrSchema
	}
	if err := validateInventory(s); err != nil {
		return err
	}
	seen := map[string]bool{}
	for i, ref := range s.References {
		if ref.Scope != s.Scope || ref != s.Inputs[i].Reference || seen[ref.RecordID] {
			return memy.ErrSchema
		}
		if err := (memy.RevisionRef{RecordID: ref.RecordID, Revision: ref.Revision}).Validate(); err != nil {
			return memy.ErrSchema
		}
		seen[ref.RecordID] = true
		input := s.Inputs[i]
		if err := retrieval.ValidateDocument(retrieval.Document[struct{}]{
			ID:             input.ArtifactID,
			Rank:           input.Rank,
			Score:          input.Score,
			ScoreState:     input.ScoreState,
			ScoreSemantics: input.ScoreSemantics,
			ScoreHistory:   input.ScoreHistory,
		}); err != nil {
			return memy.ErrSchema
		}
	}
	return validateSnippets(s, seen)
}

func validateSnippets[U any](s Sidecar[U], seen map[string]bool) error {
	if s.Truncated && s.Precision != unavailablePrecision {
		return memy.ErrSchema
	}
	contributors := map[int]bool{}
	for _, snippet := range s.Snippets {
		if err := validateSnippet(s, snippet, seen); err != nil {
			return err
		}
		index := snippet.Contributors[0].InputIndex
		if contributors[index] {
			return memy.ErrSchema
		}
		contributors[index] = true
	}
	return nil
}

type extension[U any] struct{ raw json.RawMessage }

func (extension[U]) ExtensionType() string { return SidecarSchema }
func (e extension[U]) CloneExtension() contexty.Extension {
	return extension[U]{raw: slices.Clone(e.raw)}
}
func (e extension[U]) MarshalJSON() ([]byte, error) { return slices.Clone(e.raw), nil }

// Registry creates a fresh registry containing the host sidecar decoder.
func Registry[U any](uncertaintyType string) *contexty.ExtensionRegistry {
	reg := contexty.NewExtensionRegistry()
	schema, compileErr := compileWireSchema()
	reg.Register(SidecarSchema, func(data []byte) (contexty.Extension, error) {
		if compileErr != nil {
			return nil, memy.ErrSchema
		}
		if wireErr := validateWireSchema(schema, data); wireErr != nil {
			return nil, wireErr
		}
		s, err := decodeSidecar[U](data)
		if err != nil {
			return nil, err
		}
		if s.UncertaintyType != uncertaintyType || uncertaintyType == "" {
			return nil, memy.ErrUnsupported
		}
		return extension[U]{raw: slices.Clone(data)}, nil
	})
	return reg
}

func decodeSidecar[U any](data []byte) (Sidecar[U], error) {
	var out Sidecar[U]
	if err := requiredSidecarFields(data); err != nil {
		return out, err
	}
	if err := strictJSON(data, &out); err != nil {
		return out, memy.ErrSchema
	}
	if err := validateSidecar(out); err != nil {
		return Sidecar[U]{}, err
	}
	digest, digestErr := sidecarDigest(out)
	if digestErr != nil || digest != out.Digest {
		return Sidecar[U]{}, memy.ErrSchema
	}
	return out, nil
}

// Decode requires exactly one sidecar and verifies it against the actual decoded message text.
func Decode[U any](ctx context.Context, data []byte, registry *contexty.ExtensionRegistry) (Published[U], error) {
	if err := ctx.Err(); err != nil {
		return Published[U]{}, failure("decode", err)
	}
	if len(data) > maxWireBytes || rejectDuplicateJSON(data) != nil {
		return Published[U]{}, failure("decode", memy.ErrSchema)
	}
	msg, err := contexty.UnmarshalMessageJSON(data, contexty.MessageCodec{Extensions: registry})
	if err != nil {
		return Published[U]{}, failure("codec", memy.ErrSchema)
	}
	if len(msg.Extensions) != 1 || len(msg.Parts) != 1 || msg.ID == "" ||
		(msg.Role != contexty.RoleUser && msg.Role != contexty.RoleTool) {
		return Published[U]{}, failure("decode", memy.ErrSchema)
	}
	ext, ok := msg.Extensions[0].(extension[U])
	if !ok {
		return Published[U]{}, failure("decode", memy.ErrSchema)
	}
	s, err := decodeSidecar[U](ext.raw)
	if err != nil {
		return Published[U]{}, failure("sidecar", err)
	}
	part, ok := msg.Parts[0].(contexty.TextPart)
	if !ok || part.Text != s.Text {
		return Published[U]{}, failure("text-binding", memy.ErrSchema)
	}
	public, err := PublicJSON(msg.Role, s.Text)
	if err != nil {
		return Published[U]{}, failure("projection", memy.ErrSchema)
	}
	if err := ctx.Err(); err != nil {
		return Published[U]{}, failure("decode", err)
	}
	return Published[U]{Message: msg, Durable: slices.Clone(data), Public: public, Evidence: s}, nil
}

func encode[M, U any](ctx context.Context, o Options[M], s Sidecar[U]) (Published[U], error) {
	if err := ctx.Err(); err != nil {
		return Published[U]{}, err
	}
	digest, digestErr := sidecarDigest(s)
	if digestErr != nil {
		return Published[U]{}, errors.Join(memy.ErrSchema, digestErr)
	}
	s.Digest = digest
	if err := validateSidecar(s); err != nil {
		return Published[U]{}, err
	}
	raw, err := json.Marshal(s)
	if err != nil {
		return Published[U]{}, errors.Join(memy.ErrSchema, err)
	}
	msg := contexty.Message{
		ID:         o.MessageID,
		Role:       o.Role,
		Parts:      []contexty.ContentPart{contexty.TextPart{Text: s.Text}},
		Extensions: []contexty.Extension{extension[U]{raw: raw}},
	}
	durable, err := contexty.MarshalMessageJSON(msg, contexty.MessageCodec{Extensions: Registry[U](o.UncertaintyType)})
	if err != nil {
		return Published[U]{}, memy.ErrSchema
	}
	public, err := PublicJSON(o.Role, s.Text)
	if err != nil {
		return Published[U]{}, memy.ErrSchema
	}
	n, err := o.Limits.MeasureTokens(ctx, string(public))
	if err != nil {
		return Published[U]{}, err
	}
	if n < 0 {
		return Published[U]{}, memy.ErrInvalid
	}
	if len(s.Text) > o.Limits.Bytes || utf8.RuneCountInString(s.Text) > o.Limits.Runes ||
		len(durable) > o.Limits.JSONBytes || n > o.Limits.Tokens {
		return Published[U]{}, memy.ErrBudget
	}
	return Decode[U](ctx, durable, Registry[U](o.UncertaintyType))
}

func strictJSON(data []byte, out any) error {
	if len(data) > maxWireBytes {
		return memy.ErrBudget
	}
	if err := rejectDuplicateJSON(data); err != nil {
		return err
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(out); err != nil {
		return err
	}
	if err := dec.Decode(new(any)); err != io.EOF {
		return memy.ErrSchema
	}
	return nil
}

func rejectDuplicateJSON(data []byte) error {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	if err := jsonValue(dec, 0); err != nil {
		return err
	}
	if _, err := dec.Token(); err != io.EOF {
		return memy.ErrSchema
	}
	return nil
}
func jsonValue(d *json.Decoder, depth int) error {
	if depth > maxJSONDepth {
		return memy.ErrSchema
	}
	token, tokenErr := d.Token()
	if tokenErr != nil {
		return tokenErr
	}
	delim, ok := token.(json.Delim)
	if !ok {
		return validateJSONScalar(token)
	}
	seen := map[string]bool{}
	for d.More() {
		if delim == '{' {
			if keyErr := jsonKey(d, seen); keyErr != nil {
				return keyErr
			}
		}
		if valueErr := jsonValue(d, depth+1); valueErr != nil {
			return valueErr
		}
	}
	end, endErr := d.Token()
	if endErr != nil {
		return endErr
	}
	if (delim == '{' && end != json.Delim('}')) || (delim == '[' && end != json.Delim(']')) {
		return memy.ErrSchema
	}
	return nil
}
func validateJSONScalar(token any) error {
	if number, ok := token.(json.Number); ok {
		value, numberErr := number.Float64()
		if numberErr != nil || math.IsInf(value, 0) {
			return memy.ErrSchema
		}
	}
	return nil
}
func jsonKey(d *json.Decoder, seen map[string]bool) error {
	key, keyErr := d.Token()
	if keyErr != nil {
		return keyErr
	}
	name, ok := key.(string)
	if !ok || seen[name] {
		return memy.ErrSchema
	}
	seen[name] = true
	return nil
}

func requiredSidecarFields(data []byte) error {
	if len(data) > maxWireBytes {
		return memy.ErrBudget
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(data, &fields); err != nil {
		return memy.ErrSchema
	}
	for _, name := range []string{"uncertainty_type", "schema", "scope", "epoch", "text", "rank_policy", "references", "inputs", "snippets", "coverage", "progress", "partial", "omissions", "truncated", "precision", "resource", "diagnostics", "boundary"} {
		if _, ok := fields[name]; !ok {
			return memy.ErrSchema
		}
	}
	return nil
}

func validateSnippet[U any](s Sidecar[U], snippet Snippet, seen map[string]bool) error {
	if !seen[snippet.DocumentID] || len(snippet.Contributors) == 0 {
		return memy.ErrSchema
	}
	d := retrieval.Document[struct{}]{ID: snippet.DocumentID, Content: snippet.Content, Score: snippet.Score,
		ScoreState: snippet.ScoreState, ScoreSemantics: snippet.ScoreSemantics, ScoreHistory: snippet.ScoreHistory,
		Rank: snippet.Rank, SourceMapping: snippet.Mapping, SourceSupports: snippet.Supports}
	if err := retrieval.ValidateDocument(d); err != nil {
		return memy.ErrSchema
	}
	if snippet.Span != nil {
		if s.Precision != preservedPrecision || snippet.Span.ValidateText(s.Text) != nil ||
			s.Text[snippet.Span.Start:snippet.Span.End] != snippet.Content {
			return memy.ErrSchema
		}
	} else if s.Precision != unavailablePrecision {
		return memy.ErrSchema
	}
	if len(snippet.Contributors) != 1 {
		return memy.ErrSchema
	}
	c := snippet.Contributors[0]
	if c.InputIndex < 0 || c.InputIndex >= len(s.Inputs) {
		return memy.ErrSchema
	}
	input := s.Inputs[c.InputIndex]
	// The renderer uses the input position for unranked documents; retain native zero in Input.
	renderedRank := input.Rank
	if renderedRank == 0 {
		renderedRank = c.InputIndex + 1
	}
	if s.Precision == unavailablePrecision &&
		(snippet.FullDocument || !snippet.DeliveryUncertain || c.FullDocument || !c.DeliveryUncertain) {
		return memy.ErrSchema
	}
	if input.Reference.RecordID != snippet.DocumentID || renderedRank != snippet.Rank || input.Score != snippet.Score ||
		input.ScoreState != snippet.ScoreState ||
		input.ScoreSemantics != snippet.ScoreSemantics ||
		!slices.Equal(input.ScoreHistory, snippet.ScoreHistory) {
		return memy.ErrSchema
	}
	return validateSnippetSources(s.Scope, snippet, input.Sources)
}

func compileWireSchema() (*jsonschema.Schema, error) {
	value, err := jsonschema.UnmarshalJSON(bytes.NewReader(wireSchema))
	if err != nil {
		return nil, err
	}
	compiler := jsonschema.NewCompiler()
	if err := compiler.AddResource("urn:host:retrieval-context:1", value); err != nil {
		return nil, err
	}
	return compiler.Compile("urn:host:retrieval-context:1")
}
func validateWireSchema(schema *jsonschema.Schema, data []byte) error {
	if len(data) > maxWireBytes || !utf8.Valid(data) {
		return memy.ErrSchema
	}
	value, err := jsonschema.UnmarshalJSON(bytes.NewReader(data))
	if err != nil {
		return memy.ErrSchema
	}
	if err := schema.Validate(value); err != nil {
		return memy.ErrSchema
	}
	return nil
}

func canonicalSources[P, R any](r memy.Record[P, R]) []SourceIdentity {
	sources := make([]SourceIdentity, 0, len(r.Provenance.Sources))
	for _, s := range r.Provenance.Sources {
		sources = append(sources, SourceIdentity{ID: s.ID, Revision: s.Revision})
	}
	return sources
}
func validateSnippetSources(scope memy.Scope, snippet Snippet, sources []SourceIdentity) error {
	for _, loc := range append(snippet.Mapping.Supports(), snippet.Supports...) {
		if loc.Reference.Namespace != scope.Key() ||
			!slices.Contains(sources, SourceIdentity{ID: loc.Reference.Source, Revision: loc.Reference.Revision}) {
			return memy.ErrSchema
		}
	}
	return nil
}
func validateInventory[U any](s Sidecar[U]) error {
	p := s.Progress
	if p.CanonicalFiltered != 0 || p.RankingOmitted != 0 || p.CanonicalChecked != len(s.Inputs) ||
		p.ReturnedCandidates != len(s.Inputs) {
		return memy.ErrSchema
	}
	for _, input := range s.Inputs {
		if len(input.Sources) == 0 || input.Extractor == "" {
			return memy.ErrSchema
		}
	}
	return nil
}
func sidecarDigest[U any](s Sidecar[U]) (string, error) {
	s.Digest = ""
	data, err := json.Marshal(s)
	if err != nil {
		return "", err
	}
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:]), nil
}
