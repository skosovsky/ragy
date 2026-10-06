package retrieval

import (
	"context"
	"time"

	"github.com/skosovsky/ragy/access"

	"github.com/skosovsky/ragy/filter"
)

// NoRequestMeta is the default request metadata type for callers that do not
// need a separate request metadata payload.
type NoRequestMeta struct{}

// Query is the no-request-metadata retrieval envelope.
type Query[TIntent any] = Request[TIntent, NoRequestMeta]

// Request carries retrieval text, host intent, request metadata, tuning options,
// and an optional planned query produced by QueryPlanner.
// Intent, Meta and Plan.Intent (including referenced maps/slices/pointers) are
// host-owned immutable inputs during execution. Passing Request by value is not a
// deep copy; parallel nodes must not mutate those inputs.
type Request[TIntent, TRequestMeta any] struct {
	Read    access.Binding
	Text    string
	Intent  TIntent
	Meta    TRequestMeta
	Options RetrieveOptions
	Plan    *PlannedQuery[TIntent]
}

// WithPlan returns a shallow copy with plan attached.
func (r Request[TIntent, TRequestMeta]) WithPlan(plan PlannedQuery[TIntent]) Request[TIntent, TRequestMeta] {
	r.Plan = &plan
	return r
}

// EffectiveText returns the planned text when present, otherwise the raw text.
func (r Request[TIntent, TRequestMeta]) EffectiveText() string {
	if r.Plan == nil {
		return r.Text
	}
	if r.Plan.ExpandedText != "" {
		return r.Plan.ExpandedText
	}
	if r.Plan.Text != "" {
		return r.Plan.Text
	}
	return r.Text
}

// PlannedQuery is the explicit output of query planning.
type PlannedQuery[TIntent any] struct {
	Text         string
	ExpandedText string
	Intent       TIntent
	Filters      filter.Condition
	Ranges       []RangeConstraint
	Diagnostics  []PlannerDiagnostic
	CacheKey     string
}

// ProjectPlannedQuery copies a planned query across intent types while preserving
// backend-relevant query text, filters, ranges, diagnostics, and cache identity.
func ProjectPlannedQuery[TFromIntent, TToIntent any](
	plan *PlannedQuery[TFromIntent],
	intent TToIntent,
) *PlannedQuery[TToIntent] {
	if plan == nil {
		return nil
	}
	return &PlannedQuery[TToIntent]{
		Text:         plan.Text,
		ExpandedText: plan.ExpandedText,
		Intent:       intent,
		Filters:      plan.Filters,
		Ranges:       append([]RangeConstraint(nil), plan.Ranges...),
		Diagnostics:  append([]PlannerDiagnostic(nil), plan.Diagnostics...),
		CacheKey:     plan.CacheKey,
	}
}

// RangeConstraint describes a universal planned range without binding ragy to
// an application domain type.
type RangeConstraint struct {
	Field string      `json:"field"`
	Start *RangeBound `json:"start"`
	End   *RangeBound `json:"end"`
}

// RangeBound is a typed range endpoint.
type RangeBound struct {
	Text      string     `json:"text"`
	Number    *float64   `json:"number"`
	Time      *time.Time `json:"time"`
	Inclusive bool       `json:"inclusive"`
}

// PlannerDiagnostic captures planner decisions for tests and observability.
type PlannerDiagnostic struct {
	Key   string
	Value string
}

// QueryPlanner turns a raw request into an explicit planned query.
type QueryPlanner[TIntent, TRequestMeta any] interface {
	Plan(ctx context.Context, req Request[TIntent, TRequestMeta]) (PlannedQuery[TIntent], error)
}

// QueryPlannerFunc adapts a function into QueryPlanner.
type QueryPlannerFunc[TIntent, TRequestMeta any] func(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
) (PlannedQuery[TIntent], error)

// Plan implements QueryPlanner.
func (f QueryPlannerFunc[TIntent, TRequestMeta]) Plan(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
) (PlannedQuery[TIntent], error) {
	return f(ctx, req)
}

// StaticPlanner returns the same planned query for every request.
type StaticPlanner[TIntent, TRequestMeta any] struct {
	Planned PlannedQuery[TIntent]
}

// Plan implements QueryPlanner.
func (p StaticPlanner[TIntent, TRequestMeta]) Plan(
	context.Context,
	Request[TIntent, TRequestMeta],
) (PlannedQuery[TIntent], error) {
	return p.Planned, nil
}

// CopyRequestOptions snapshots core mutable query options and planning slices.
// Host intent/metadata require an explicit BYOT snapshot policy. The caller must
// not mutate inputs concurrently with this capture; returned core fields are owned.
func CopyRequestOptions[TIntent, TRequestMeta any](req Request[TIntent, TRequestMeta]) Request[TIntent, TRequestMeta] {
	req.Options.Vector = append([]float32(nil), req.Options.Vector...)
	if req.Options.Threshold != nil {
		threshold := *req.Options.Threshold
		req.Options.Threshold = &threshold
	}
	if req.Options.Graph != nil {
		graphOptions := *req.Options.Graph
		graphOptions.Seeds = append([]string(nil), graphOptions.Seeds...)
		if graphOptions.Page != nil {
			page := *graphOptions.Page
			graphOptions.Page = &page
		}
		req.Options.Graph = &graphOptions
	}
	if req.Plan != nil {
		plan := *req.Plan
		plan.Diagnostics = append([]PlannerDiagnostic(nil), plan.Diagnostics...)
		plan.Ranges = append([]RangeConstraint(nil), plan.Ranges...)
		for i := range plan.Ranges {
			plan.Ranges[i].Start = copyRangeBound(plan.Ranges[i].Start)
			plan.Ranges[i].End = copyRangeBound(plan.Ranges[i].End)
		}
		req.Plan = &plan
	}
	return req
}
func copyRangeBound(bound *RangeBound) *RangeBound {
	if bound == nil {
		return nil
	}
	copied := *bound
	if bound.Number != nil {
		number := *bound.Number
		copied.Number = &number
	}
	if bound.Time != nil {
		instant := *bound.Time
		copied.Time = &instant
	}
	return &copied
}
