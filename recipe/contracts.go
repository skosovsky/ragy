// Package recipe implements optional bounded retrieval strategies. Host ports
// supply planning, assessment, pricing and metadata ownership; recipes do not own
// conversation state, authorization, scheduling or a model engine.
package recipe

import (
	"context"
	"time"

	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

type Strategy string

const (
	SingleRewrite Strategy = "single-rewrite"
	MultiQuery    Strategy = "multi-query"
	Decomposition Strategy = "decomposition"
)

type Outcome string

const (
	Complete     Outcome = "complete"
	Partial      Outcome = "partial"
	Insufficient Outcome = "insufficient"
	Failure      Outcome = "failed"
)

type StopReason string

const (
	Assessed         StopReason = "assessed"
	BudgetExhausted  StopReason = "budget-exhausted"
	DeadlineReached  StopReason = "deadline"
	PriceUnavailable StopReason = "price-unavailable"
	StageFailure     StopReason = "stage-failure"
)

// FusionObservation distinguishes an unstarted fusion from a dispatched fusion
// whose output could not be retained. Failed attempts never expose selected hits.
type FusionObservation string

const (
	FusionNotRun   FusionObservation = "not-run"
	FusionMissing  FusionObservation = "missing-observation"
	FusionObserved FusionObservation = "observed"
)

type Usage struct {
	Value budget.Usage
	Known bool
}

// Planning contains text variants/subquestions only. Nested plans and autonomous
// actions cannot be represented. The original intent, filters and binding remain.
type Planning struct {
	Queries []string
	Usage   Usage
}

type QueryEvidence[TMeta any] struct {
	Index     int
	Text      string
	Keys      []string
	Documents []retrieval.Document[TMeta]
	Supports  [][]source.Locator
}

type AssessmentInput[TIntent, TRequestMeta, TMeta any] struct {
	Original retrieval.Request[TIntent, TRequestMeta]
	Queries  []QueryEvidence[TMeta]
}

// Assessment selects executed query indices. Sufficiency is a host strategy
// signal, not a truth guarantee. Rewrite can retain original evidence by selecting
// index 0; decomposition reports coverage for each planned subquestion separately.
type Assessment struct {
	Selected   []int
	Sufficient bool
	Usage      Usage
}

type Operation string

const (
	Retrieve Operation = "retrieve"
	Plan     Operation = "plan"
	Assess   Operation = "assess"
)

// Quote executes before reservation/dispatch. It must not call a model itself.
// Pricing units and maximum token reservations are supplied by the host.
type Quote struct {
	Usage     budget.Usage
	CostKnown bool
}

// ModelLimits are the actual token maxima reserved for this model dispatch.
// Ports must enforce these limits before/during dispatch, including invisible
// completion tokens. They must not substitute separate static configuration.
type ModelLimits struct {
	InputTokens  uint64
	OutputTokens uint64
}

// Config ports must be bounded and concurrency-safe. Concurrent Run calls share
// this configuration but own independent attempt ledgers and captured metadata.
// Callers must keep input data stable while clone ports capture it.
type Config[TIntent, TRequestMeta, TMeta any] struct {
	Strategy         Strategy
	Revision         string
	Backend          retrieval.RequestBackend[TIntent, TRequestMeta, TMeta]
	Admission        func(context.Context, retrieval.Request[TIntent, TRequestMeta]) (retrieval.ReadCoverage, error)
	Planner          func(context.Context, retrieval.Request[TIntent, TRequestMeta], ModelLimits) (Planning, error)
	Assessor         func(context.Context, AssessmentInput[TIntent, TRequestMeta, TMeta], ModelLimits) (Assessment, error)
	Pricing          func(context.Context, Operation) (Quote, error)
	CloneIntent      func(TIntent) (TIntent, error)
	CloneRequestMeta func(TRequestMeta) (TRequestMeta, error)
	CloneMeta        func(TMeta) (TMeta, error)
	Supports         func(context.Context, access.Binding, retrieval.Document[TMeta]) ([]source.Locator, error)
	Identity         retrieval.IdentityResolver[TMeta]
	Limits           budget.Limits
	RequireKnownCost bool
	Duration         time.Duration
	Now              func() time.Time
	MaxQueries       int
	MaxDocuments     int
	FusionK          int
}

// Contribution retains source and query provenance for every dedup contributor.
// Result owns its slices and BYOT metadata; consumers must not mutate a result
// concurrently with use or export. Exported immutable records are a separate layer.
type Contribution struct {
	QueryIndex int
	DocumentID string
	Rank       int
	Supports   []source.Locator
}

type SelectedEvidence[TMeta any] struct {
	Document     retrieval.Document[TMeta]
	Contributors []Contribution
}

type Subquestion struct {
	Index       int
	Retrieved   bool
	Selected    bool
	HasEvidence bool
}

type Stage struct {
	Operation Operation
	Usage     Usage
}

type Result[TMeta any] struct {
	Outcome        Outcome
	Stop           StopReason
	Strategy       Strategy
	RecipeRevision string
	Publication    string
	Queries        []QueryEvidence[TMeta]
	Selected       []SelectedEvidence[TMeta]
	Coverage       []Subquestion
	Admission      retrieval.ReadCoverage
	Stages         []Stage
	Budget         budget.Snapshot
	Fusion         FusionObservation
}
