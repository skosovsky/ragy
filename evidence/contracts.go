// Package evidence snapshots explicit retrieval observations under an export
// policy. It does not infer unobserved stages, own telemetry or evaluate truth.
package evidence

import (
	"context"

	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

const SchemaIdentity = "ragy.retrieval-evidence/v2"

const (
	ScoreAbsent     = "absent"
	ScoreNative     = "native"
	ScoreNormalized = "normalized"
	Ungradable      = "ungradable"
)

type DiagnosticKind string

const (
	ModelCalls     DiagnosticKind = "model_calls"
	RetrievalCalls DiagnosticKind = "retrieval_calls"
	InputTokens    DiagnosticKind = "input_tokens"
	OutputTokens   DiagnosticKind = "output_tokens"
	CostUnits      DiagnosticKind = "cost_units"
	LatencyMillis  DiagnosticKind = "latency_ms"
)

type Diagnostic struct {
	Kind   DiagnosticKind `json:"kind"`
	Number Number         `json:"number"`
}

type CaptureState string

const (
	Observed    CaptureState = "observed"
	Omitted     CaptureState = "omitted"
	Unavailable CaptureState = "unavailable"
	Unsupported CaptureState = "unsupported"
)

type Status string

const (
	StageObserved      Status = "observed"
	NotRun             Status = "not_run"
	StageUnsupported   Status = "unsupported"
	MissingObservation Status = "missing_observation"
	StageUnavailable   Status = "unavailable"
)

type Outcome string

const (
	Complete      Outcome = "complete"
	CompleteEmpty Outcome = "complete_empty"
	Partial       Outcome = "partial"
	Insufficient  Outcome = "insufficient"
	Failed        Outcome = "failed"
)

type Reason string

const (
	NoReason        Reason = "none"
	Budget          Reason = "budget"
	Deadline        Reason = "deadline"
	MissingEvidence Reason = "missing_evidence"
	PartialTargets  Reason = "partial_targets"
	TargetFailure   Reason = "target_failure"
)

type IdentifierKind string

const (
	ModelIdentifier          IdentifierKind = "model_revision"
	PromptIdentifier         IdentifierKind = "prompt_revision"
	ConfigIdentifier         IdentifierKind = "config_revision"
	RetrievalIdentifier      IdentifierKind = "retrieval"
	ScopeIdentifier          IdentifierKind = "scope"
	PublicationIdentifier    IdentifierKind = "publication"
	RecipeIdentifier         IdentifierKind = "recipe"
	StageIdentifier          IdentifierKind = "stage"
	DocumentIdentifier       IdentifierKind = "document"
	SourceIdentifier         IdentifierKind = "source"
	RevisionIdentifier       IdentifierKind = "revision"
	NamespaceIdentifier      IdentifierKind = "namespace"
	TransformationIdentifier IdentifierKind = "transformation"
	ArtifactIdentifier       IdentifierKind = "artifact"
	RepresentationIdentifier IdentifierKind = "representation"
	ScoreIdentifier          IdentifierKind = "score_semantics"
	RubricIdentifier         IdentifierKind = "rubric"
)

type Field string

const (
	QueryField       Field = "query"
	SnippetField     Field = "snippet"
	SourceField      Field = "source_association"
	ScoreField       Field = "score"
	JudgmentField    Field = "judgment"
	ScopeField       Field = "scope"
	PublicationField Field = "publication"
)

// Policy denies identifiers and raw text by default. It never accepts arbitrary
// metadata/auth/error/diagnostic strings for serialization. Callbacks are pure
// policy decisions and must not mutate inputs or call a model.
type Policy struct {
	// AllowDecisions permits bounded ordinal associations and execution decisions.
	AllowDecisions  bool
	AllowIdentifier func(IdentifierKind, string) bool
	AllowQuery      bool
	AllowSnippet    func(string) bool
	AllowNumbers    bool
	// AllowLocation explicitly permits geometry and location labels after source
	// admission. All source identifiers and numbers must also be exportable.
	AllowLocation func(source.Locator) bool
	// AllowContribution permits the query/document/rank association explicitly.
	AllowContribution func(Contribution) bool
}

type Judgment struct {
	Query  string
	Source source.Reference
	Grade  float64
	Rubric string
}

type Hit[TMeta any] struct {
	Document      retrieval.Document[TMeta]
	Sources       []source.Reference
	Judgment      *Judgment
	Locations     []source.Locator
	Contributions []Contribution
}

// Contribution uses an executed query ordinal, never raw query or domain metadata.
// The observation producer attests the ordinal; decoding does not authenticate it.
type Contribution struct {
	QueryIndex int
	DocumentID string
	Rank       int
	Locations  []source.Locator
}

type Stage[TMeta any] struct {
	Name      string
	Status    Status
	Scores    CaptureState
	Sources   CaptureState
	Judgments CaptureState
	Hits      []Hit[TMeta]
}

type Input[TMeta any] struct {
	Schema          filter.Schema
	Codec           retrieval.MetadataCodec[TMeta]
	SourceAdmission func(context.Context, access.Binding, source.Reference) error
	RetrievalID     string
	RecipeRevision  string
	Query           string
	Outcome         Outcome
	Reason          Reason
	Coverage        retrieval.ReadCoverage
	Required        []Field
	Stages          []Stage[TMeta]
	Diagnostics     []Diagnostic
	Decision        *DecisionInput
}

// Text is an explicit observed/omitted/unavailable/unsupported value.
type Text struct {
	State CaptureState `json:"state"`
	Value *string      `json:"value"`
}
type Number struct {
	State CaptureState `json:"state"`
	Value *float64     `json:"value"`
}

type Score struct {
	State     string   `json:"state"`
	Value     *float64 `json:"value"`
	Semantics Text     `json:"semantics"`
}
type Source struct {
	Namespace      Text `json:"namespace"`
	ID             Text `json:"id"`
	Revision       Text `json:"revision"`
	Transformation Text `json:"transformation"`
	Artifact       Text `json:"artifact"`
	Representation Text `json:"representation"`
}
type Label struct {
	State  string `json:"state"`
	Query  Text   `json:"query"`
	Source Source `json:"source"`
	Grade  Number `json:"grade"`
	Rubric Text   `json:"rubric"`
}
type WireHit struct {
	ID                 Text               `json:"id"`
	Rank               Number             `json:"rank"`
	Score              Score              `json:"score"`
	Snippet            Text               `json:"snippet"`
	SourcesState       CaptureState       `json:"sources_state"`
	Sources            []Source           `json:"sources"`
	Judgment           Label              `json:"judgment"`
	LocationsState     CaptureState       `json:"locations_state"`
	Locations          []Location         `json:"locations"`
	ContributionsState CaptureState       `json:"contributions_state"`
	Contributions      []WireContribution `json:"contributions"`
}

type WireContribution struct {
	QueryIndex     int          `json:"query_index"`
	DocumentID     Text         `json:"document_id"`
	Rank           int          `json:"rank"`
	LocationsState CaptureState `json:"locations_state"`
	Locations      []Location   `json:"locations"`
}

// Location carries canonical original coordinates without auth fingerprints.
// Decode validates shape only; source authorization/retention stays with the host.
type Location struct {
	Source Source              `json:"source"`
	Kind   source.LocationKind `json:"kind"`
	Span   source.ByteSpan     `json:"span"`
	Page   source.PageGeometry `json:"page"`
	Region source.Rectangle    `json:"region"`
	Cell   source.TableCell    `json:"cell"`
}
type WireStage struct {
	Index     int          `json:"index"`
	Name      Text         `json:"name"`
	Status    Status       `json:"status"`
	HitsState CaptureState `json:"hits_state"`
	Hits      []WireHit    `json:"hits"`
}
type Snapshot struct {
	Schema      string                 `json:"schema"`
	RetrievalID Text                   `json:"retrieval_id"`
	Scope       Text                   `json:"scope"`
	Publication Text                   `json:"publication"`
	Recipe      Text                   `json:"recipe"`
	Query       Text                   `json:"query"`
	Outcome     Outcome                `json:"outcome"`
	Reason      Reason                 `json:"reason"`
	Coverage    retrieval.ReadCoverage `json:"coverage"`
	Stages      []WireStage            `json:"stages"`
	Diagnostics []Diagnostic           `json:"diagnostics"`
	Decision    Decision               `json:"decision"`
}

type Sink interface {
	Write(context.Context, Record) error
}
