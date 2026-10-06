// Package graphsummary implements optional bounded community and global summaries.
// Membership, source access, model transport, schemas and prices belong to hosts.
package graphsummary

import (
	"context"
	"time"

	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/recipe"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/source"
)

type Snippet[TAccess any] struct {
	Mapping source.MappedText
	Access  TAccess
	Members []string
}
type Community[TAccess any] struct {
	ID       string
	Members  []string
	Snippets []Snippet[TAccess]
}
type Stage string

const (
	Map    Stage = "community"
	Reduce Stage = "global"
)

type ModelSnippet struct {
	Index int    `json:"index"`
	Text  string `json:"text"`
}
type ModelInput struct {
	Stage           Stage          `json:"stage"`
	Question        string         `json:"question"`
	Snippets        []ModelSnippet `json:"snippets"`
	MaxInputTokens  uint64         `json:"max_input_tokens"`
	MaxOutputTokens uint64         `json:"max_output_tokens"`
}
type ModelOutput struct {
	Text     string `json:"text"`
	Selected []int  `json:"selected"`
}
type Usage struct {
	Value budget.Usage
	Known bool
}
type Model func(context.Context, ModelInput) (ModelOutput, Usage, error)

// Config host ports must be bounded/concurrency-safe. Membership and metadata/token/price
// callbacks perform no hidden model calls. Model dispatches once without retries.
type Config[TAccess any] struct {
	Schema           filter.Schema
	MaxCommunities   int
	MaxModelCalls    uint64
	MaxMembers       int
	MaxSnippets      int // Per community; MaxSupports and source text MaxInputBytes also bound the aggregate.
	MaxSupports      int
	MaxInputBytes    int // Aggregate admitted source bytes, and complete JSON ModelInput per dispatch.
	MaxSummaryBytes  int
	Duration         time.Duration
	Now              func() time.Time
	CloneAccess      func(TAccess) (TAccess, error)
	Attributes       func(TAccess) (filter.RawAttributes, error)
	Membership       func(context.Context, access.Binding, string, []string) error
	AdmitSource      func(context.Context, access.Binding, source.Locator) error
	Quote            func(context.Context, Stage) (budget.Reservation, error)
	CountInputTokens func(ModelInput) (uint64, error)
	Model            Model
}
type Request[TAccess any] struct {
	Read        access.Binding
	Question    string
	Communities []Community[TAccess]
}
type StopReason string

const (
	Summarized       StopReason = "summarized"
	BudgetExhausted  StopReason = "budget-exhausted"
	PriceUnavailable StopReason = "price-unavailable"
	MissingCoverage  StopReason = "missing-coverage"
)

type Result struct {
	Communities []Summary
	Global      *Summary
	Outcome     recipe.Outcome
	Stop        StopReason
	ModelCalls  uint64
}

// Summary is an immutable derived artifact. Text is accessible only through a
// fresh Resolve, which verifies the original binding and every retained support.
type Summary struct {
	text        string
	communities []string
	supports    []source.Locator
	binding     string
	schema      filter.Schema
	covered     bool
}
