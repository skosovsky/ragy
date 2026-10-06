package evidence

import (
	"context"
	"errors"
	"math"
	"slices"
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

var ErrPrivacy = errors.New("required evidence field denied by export policy")

type capture struct {
	ctx       context.Context
	read      access.Binding
	policy    Policy
	err       error
	mandatory filter.Condition
}

func Capture[TMeta any](ctx context.Context, read access.Binding, input Input[TMeta], policy Policy) (Record, error) {
	if err := read.Check(ctx); err != nil {
		return Record{}, err
	}
	mandatory := filter.Condition{}
	if read.IsScoped() {
		if nilInterface(input.Codec) || input.SourceAdmission == nil {
			return Record{}, access.UnsupportedCapability(ragy.ErrUnsupported)
		}
		var err error
		mandatory, err = read.Prepare(
			ctx,
			input.Schema,
			filter.Condition{},
			access.Capabilities{ScopeProfile: true, PinnedPublication: true, RequirePinnedPublication: false},
		)
		if err != nil {
			return Record{}, err
		}
	}
	input.Coverage = retrieval.BindPublicationCoverage(read, input.Coverage)
	outcome, reason := bindPublicationOutcome(read, input.Outcome, input.Reason)
	c := capture{ctx: ctx, read: read, policy: policy, err: nil, mandatory: mandatory}
	snapshot := Snapshot{
		Schema:      SchemaIdentity,
		RetrievalID: c.identifier(RetrievalIdentifier, input.RetrievalID),
		Scope: c.identifier(
			ScopeIdentifier,
			read.Snapshot().Identity,
		),
		Publication: c.identifier(PublicationIdentifier, read.Publication().Reference()),
		Recipe: c.identifier(
			RecipeIdentifier,
			input.RecipeRevision,
		),
		Query:       hidden(Omitted),
		Outcome:     outcome,
		Reason:      reason,
		Coverage:    input.Coverage,
		Stages:      make([]WireStage, 0, len(input.Stages)),
		Diagnostics: slices.Clone(input.Diagnostics),
	}
	for i := range snapshot.Diagnostics {
		if snapshot.Diagnostics[i].Number.State == Observed && !policy.AllowNumbers {
			snapshot.Diagnostics[i].Number = Number{State: Omitted, Value: nil}
		}
	}
	if policy.AllowQuery {
		snapshot.Query = textValue(input.Query)
	}
	for i, stage := range input.Stages {
		captured, err := captureStage(&c, input, i, stage)
		if err != nil {
			return Record{}, err
		}
		snapshot.Stages = append(snapshot.Stages, captured)
	}
	if c.err != nil {
		return Record{}, c.err
	}
	if err := checkRequired(snapshot, input.Required); err != nil {
		return Record{}, err
	}
	if err := read.Check(ctx); err != nil {
		return Record{}, err
	}
	return fromSnapshot(snapshot)
}

func hidden(state CaptureState) Text { return Text{State: state, Value: nil} }
func textValue(value string) Text {
	if value == "" {
		return hidden(Unavailable)
	}
	return Text{State: Observed, Value: &value}
}
func (c *capture) identifier(kind IdentifierKind, value string) Text {
	if value == "" {
		return hidden(Unavailable)
	}
	if c.err != nil {
		return hidden(Omitted)
	}
	if c.err = c.read.Check(c.ctx); c.err != nil {
		return hidden(Omitted)
	}
	allowed := c.policy.AllowIdentifier != nil && c.policy.AllowIdentifier(kind, value)
	if c.err = c.read.Check(c.ctx); c.err != nil {
		return hidden(Omitted)
	}
	if !allowed {
		return hidden(Omitted)
	}
	return textValue(value)
}
func (c *capture) number(value float64) Number {
	if !c.policy.AllowNumbers {
		return Number{State: Omitted, Value: nil}
	}
	return Number{State: Observed, Value: &value}
}

func captureStage[TMeta any](c *capture, input Input[TMeta], index int, stage Stage[TMeta]) (WireStage, error) {
	out := WireStage{
		Index:     index,
		Name:      c.identifier(StageIdentifier, stage.Name),
		Status:    stage.Status,
		HitsState: Unavailable,
		Hits:      nil,
	}
	if stage.Name == "" || !utf8.ValidString(stage.Name) {
		return WireStage{}, ragy.ErrInvalidArgument
	}
	if stage.Status != StageObserved {
		if len(stage.Hits) != 0 {
			return WireStage{}, ragy.ErrProtocol
		}
		return out, nil
	}
	if input.Outcome == CompleteEmpty && len(stage.Hits) != 0 {
		return WireStage{}, ragy.ErrProtocol
	}
	if err := requiredCapabilities(input.Required, stage.Scores, stage.Sources, stage.Judgments); err != nil {
		return WireStage{}, err
	}
	for _, state := range []CaptureState{stage.Scores, stage.Sources, stage.Judgments} {
		if state != Observed && state != Unavailable && state != Unsupported {
			return WireStage{}, ragy.ErrInvalidArgument
		}
	}
	out.HitsState = Omitted
	if c.policy.AllowIdentifier != nil {
		out.HitsState = Observed
	}
	hits, err := captureHits(c, input, stage)
	if err != nil {
		return WireStage{}, err
	}
	out.Hits = hits
	return out, nil
}

func captureHits[TMeta any](c *capture, input Input[TMeta], stage Stage[TMeta]) ([]WireHit, error) {
	var hits []WireHit
	for _, hit := range stage.Hits {
		if err := validateHit(c, input, stage, hit); err != nil {
			return nil, err
		}
		id := c.identifier(DocumentIdentifier, hit.Document.ID)
		if id.State != Observed {
			for _, field := range input.Required {
				if field == SnippetField || field == SourceField || field == ScoreField || field == JudgmentField {
					return nil, ErrPrivacy
				}
			}
			continue
		}
		captured, err := captureHit(c, input, stage, hit, id)
		if err != nil {
			return nil, err
		}
		hits = append(hits, captured)
	}
	return hits, nil
}

func requiredCapabilities(required []Field, scores, sources, judgments CaptureState) error {
	for _, field := range required {
		capability := Observed
		switch field {
		case ScoreField:
			capability = scores
		case SourceField:
			capability = sources
		case JudgmentField:
			capability = judgments
		case SnippetField, QueryField, ScopeField, PublicationField:
		default:
			return ragy.ErrUnsupported
		}
		if capability == Unsupported {
			return ragy.ErrUnsupported
		}
		if capability == Unavailable {
			return ragy.ErrUnavailable
		}
	}
	return nil
}

func validateHit[TMeta any](c *capture, input Input[TMeta], stage Stage[TMeta], hit Hit[TMeta]) error {
	if err := c.read.Check(c.ctx); err != nil {
		return err
	}
	if err := retrieval.ValidateDocument(hit.Document); err != nil {
		return err
	}
	if err := admitHitSources(c, input, stage, hit); err != nil {
		return err
	}
	if err := validateContributions(hit); err != nil {
		return err
	}
	if err := validateLocations(hit); err != nil {
		return err
	}
	if hit.Judgment != nil {
		label := hit.Judgment
		if stage.Judgments != Observed || label.Query != input.RetrievalID || label.Rubric == "" ||
			math.IsNaN(label.Grade) ||
			math.IsInf(label.Grade, 0) ||
			!slices.Contains(hit.Sources, label.Source) {
			return ragy.ErrInvalidArgument
		}
	}
	return nil
}

func belongs(publication access.Publication, ref source.Reference) bool {
	if publication.IsCurrent() {
		return true
	}
	for _, target := range publication.Targets() {
		if target.Namespace == ref.Namespace && target.Source == ref.Source && target.Revision == ref.Revision &&
			target.AccessFingerprint == ref.AccessFingerprint {
			return true
		}
	}
	return false
}

func captureHit[TMeta any](
	c *capture,
	input Input[TMeta],
	stage Stage[TMeta],
	hit Hit[TMeta],
	id Text,
) (WireHit, error) {
	out := WireHit{
		ID:                 id,
		Rank:               c.number(float64(hit.Document.Rank)),
		Score:              captureScore(c, stage.Scores, hit.Document),
		Snippet:            hidden(Omitted),
		SourcesState:       stage.Sources,
		Sources:            nil,
		Judgment:           emptyLabel(),
		LocationsState:     Unavailable,
		Locations:          nil,
		ContributionsState: Unavailable, Contributions: nil,
	}
	if hit.Document.Rank == 0 {
		out.Rank = Number{State: Unavailable, Value: nil}
	}
	if stage.Sources == Observed && len(hit.Sources) == 0 {
		out.SourcesState = Unavailable
	}
	for _, ref := range hit.Sources {
		out.Sources = append(out.Sources, captureSource(c, ref))
	}
	out.LocationsState, out.Locations = captureLocations(c, hitLocations(hit))
	out.ContributionsState, out.Contributions = captureContributions(c, hit.Contributions)
	if c.policy.AllowSnippet != nil && c.err == nil {
		allowed := c.policy.AllowSnippet(hit.Document.ID)
		c.err = c.read.Check(c.ctx)
		if allowed {
			out.Snippet = textValue(hit.Document.Content)
		}
	}
	if hit.Judgment != nil {
		out.Judgment = captureLabel(c, *hit.Judgment)
	}
	if err := requiredHit(input.Required, out); err != nil {
		return WireHit{}, err
	}
	return out, nil
}

func captureScore[TMeta any](c *capture, capability CaptureState, doc retrieval.Document[TMeta]) Score {
	out := Score{State: string(capability), Value: nil, Semantics: hidden(capability)}
	if capability != Observed {
		return out
	}
	out.State, out.Semantics = ScoreAbsent, hidden(Unavailable)
	if !doc.ScoreState.IsScored() {
		return out
	}
	out.State = ScoreNative
	if doc.ScoreState == retrieval.ScoreNormalized {
		out.State = ScoreNormalized
	}
	out.Semantics = c.identifier(ScoreIdentifier, string(doc.ScoreSemantics))
	if c.policy.AllowNumbers {
		value := doc.Score
		out.Value = &value
	} else {
		out.State = string(Omitted)
	}
	return out
}

func emptySource(state CaptureState) Source {
	return Source{
		Namespace:      hidden(state),
		ID:             hidden(state),
		Revision:       hidden(state),
		Transformation: hidden(state),
		Artifact:       hidden(state),
		Representation: hidden(state),
	}
}
func captureSource(c *capture, ref source.Reference) Source {
	return Source{
		Namespace: c.identifier(NamespaceIdentifier, ref.Namespace),
		ID:        c.identifier(SourceIdentifier, ref.Source),
		Revision:  c.identifier(RevisionIdentifier, ref.Revision),
		Transformation: c.identifier(
			TransformationIdentifier,
			ref.Transformation,
		),
		Artifact:       c.identifier(ArtifactIdentifier, ref.Artifact),
		Representation: c.identifier(RepresentationIdentifier, ref.Representation),
	}
}
func emptyLabel() Label {
	return Label{
		State:  Ungradable,
		Query:  hidden(Unavailable),
		Source: emptySource(Unavailable),
		Grade:  Number{State: Unavailable, Value: nil},
		Rubric: hidden(Unavailable),
	}
}
func captureLabel(c *capture, label Judgment) Label {
	return Label{
		State:  string(Observed),
		Query:  c.identifier(RetrievalIdentifier, label.Query),
		Source: captureSource(c, label.Source),
		Grade:  c.number(label.Grade),
		Rubric: c.identifier(RubricIdentifier, label.Rubric),
	}
}

func checkRequired(snapshot Snapshot, required []Field) error {
	for _, field := range required {
		if err := requiredField(snapshot, field); err != nil {
			return err
		}
	}
	return nil
}
func requiredField(snapshot Snapshot, field Field) error {
	switch field {
	case QueryField:
		return requireText(snapshot.Query)
	case ScopeField:
		return requireText(snapshot.Scope)
	case PublicationField:
		return requireText(snapshot.Publication)
	case SnippetField, SourceField, ScoreField, JudgmentField:
		return requiredStages(snapshot.Stages)
	default:
		return ragy.ErrUnsupported
	}
}
func requiredStages(stages []WireStage) error {
	if len(stages) == 0 {
		return ragy.ErrUnavailable
	}
	for _, stage := range stages {
		if stage.Status == StageUnsupported {
			return ragy.ErrUnsupported
		}
		if stage.Status == MissingObservation || stage.Status == StageUnavailable {
			return ragy.ErrUnavailable
		}
		if stage.Status == StageObserved && stage.HitsState == Omitted {
			return ErrPrivacy
		}
	}
	return nil
}
func requireText(text Text) error {
	switch text.State {
	case Observed:
		return nil
	case Omitted:
		return ErrPrivacy
	case Unsupported:
		return ragy.ErrUnsupported
	case Unavailable:
		return ragy.ErrUnavailable
	default:
		return ragy.ErrProtocol
	}
}
func requiredHit(required []Field, hit WireHit) error {
	for _, field := range required {
		if err := requiredHitField(field, hit); err != nil {
			return err
		}
	}
	return nil
}
func requiredHitField(field Field, hit WireHit) error {
	switch field {
	case SnippetField:
		return requireText(hit.Snippet)
	case SourceField:
		if hit.SourcesState != Observed {
			return requireText(Text{State: hit.SourcesState, Value: nil})
		}
		for _, ref := range hit.Sources {
			if err := requireSource(ref); err != nil {
				return err
			}
		}
		return nil
	case ScoreField:
		return requireScore(hit.Score)
	case JudgmentField:
		return requireLabel(hit.Judgment)
	case QueryField, ScopeField, PublicationField:
		return nil
	default:
		return ragy.ErrUnsupported
	}
}
func requireSource(ref Source) error {
	for _, value := range []Text{ref.Namespace, ref.ID, ref.Revision, ref.Transformation, ref.Artifact, ref.Representation} {
		if err := requireText(value); err != nil {
			return err
		}
	}
	return nil
}
func requireScore(score Score) error {
	if score.State == string(Unsupported) {
		return ragy.ErrUnsupported
	}
	if score.State == string(Unavailable) || score.State == ScoreAbsent {
		return ragy.ErrUnavailable
	}
	if score.Value == nil {
		return ErrPrivacy
	}
	return requireText(score.Semantics)
}
func requireLabel(label Label) error {
	if label.State != string(Observed) {
		return ragy.ErrUnavailable
	}
	if err := requireText(label.Rubric); err != nil {
		return err
	}
	if err := requireText(label.Query); err != nil {
		return err
	}
	if err := requireSource(label.Source); err != nil {
		return err
	}
	if label.Grade.State != Observed {
		return ErrPrivacy
	}
	return nil
}

func admitHitSources[TMeta any](c *capture, input Input[TMeta], stage Stage[TMeta], hit Hit[TMeta]) error {
	if c.read.IsScoped() {
		allowed, err := retrieval.MatchDocument(input.Codec, hit.Document, c.mandatory)
		if gateErr := c.read.Check(c.ctx); gateErr != nil {
			return gateErr
		}
		if err != nil {
			return access.NonSkippable(err)
		}
		if !allowed {
			return access.NonSkippable(ragy.ErrUnavailable)
		}
	}
	if stage.Sources != Observed && len(hit.Sources) > 0 {
		return ragy.ErrProtocol
	}
	for _, ref := range hit.Sources {
		if err := admitSource(c, input.SourceAdmission, ref); err != nil {
			return err
		}
	}

	return nil
}

func admitSource(
	c *capture,
	admission func(context.Context, access.Binding, source.Reference) error,
	ref source.Reference,
) error {
	if err := ref.Validate(); err != nil {
		return err
	}
	if !belongs(c.read.Publication(), ref) {
		return access.NonSkippable(ragy.ErrUnavailable)
	}
	if !c.read.IsScoped() {
		return nil
	}
	if err := admission(c.ctx, c.read, ref); err != nil {
		return access.NonSkippable(err)
	}
	return c.read.Check(c.ctx)
}

func bindPublicationOutcome(read access.Binding, outcome Outcome, reason Reason) (Outcome, Reason) {
	if read.Publication().IsPartial() && (outcome == Complete || outcome == CompleteEmpty) {
		return Partial, PartialTargets
	}
	return outcome, reason
}
