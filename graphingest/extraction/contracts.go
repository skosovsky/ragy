// Package extraction adapts an injected bounded model client to typed, source-bound
// graph mentions. Models cannot supply canonical identities, permissions or revisions.
package extraction

import (
	"context"
	"time"

	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/graphingest/resolution"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/source"
)

type Snippet[TAccess any] struct {
	Namespace string
	Mapping   source.MappedText
	Access    TAccess
}
type ModelSnippet struct {
	Index int    `json:"index"`
	Text  string `json:"text"`
}
type ModelInput struct {
	OntologyIdentity string         `json:"ontology"`
	Configuration    string         `json:"configuration"`
	Snippets         []ModelSnippet `json:"snippets"`
	MaxInputTokens   uint64         `json:"max_input_tokens"`
	MaxOutputTokens  uint64         `json:"max_output_tokens"`
}
type Entity[TKind comparable, TAttr any] struct {
	ID         string `json:"id"`
	Name       string `json:"name"`
	Kind       TKind  `json:"kind"`
	Attributes TAttr  `json:"attributes"`
	Snippets   []int  `json:"snippets"`
}
type Relation[TRel comparable, TAttr any] struct {
	ID         string `json:"id"`
	From       string `json:"from"`
	To         string `json:"to"`
	Kind       TRel   `json:"kind"`
	Attributes TAttr  `json:"attributes"`
	Snippets   []int  `json:"snippets"`
}
type ModelOutput[TKind, TRel comparable, TAttr any] struct {
	Entities  []Entity[TKind, TAttr]  `json:"entities"`
	Relations []Relation[TRel, TAttr] `json:"relations"`
}
type Usage struct {
	Value budget.Usage
	Known bool
}
type Result[TKind, TRel comparable, TAttr any] struct {
	Extraction resolution.Extraction[TKind, TRel, TAttr]
	Usage      Usage
}

// Model executes exactly one bounded call, with no retries, enforcing the reserved
// input/output token limits. It reports actual usage on failures when known.
type Model[TKind, TRel comparable, TAttr any] func(context.Context, ModelInput) (ModelOutput[TKind, TRel, TAttr], Usage, error)
type Config[TAccess any, TKind, TRel comparable, TAttr any] struct {
	OntologyIdentity string
	Configuration    string
	Schema           filter.Schema
	MaxSnippets      int
	// MaxInputBytes caps the sum of source snippet text bytes, not the provider envelope.
	MaxInputBytes int
	MaxEntities   int
	MaxRelations  int
	MaxSupports   int
	// Duration bounds local work; its clock is independent from the supplied ledger clock.
	Duration time.Duration
	// Now must be stable and concurrency-safe. Equality with the local deadline expires.
	Now              func() time.Time
	CloneAccess      func(TAccess) (TAccess, error)
	Attributes       func(TAccess) (filter.RawAttributes, error)
	AdmitSnippet     func(context.Context, access.Binding, Snippet[TAccess]) error
	CloneAttributes  func(TAttr) (TAttr, error)
	ValidateEntity   func(TKind, TAttr) error
	ValidateRelation func(TRel, TKind, TKind, TAttr) error
	Quote            func(context.Context) (budget.Reservation, error)
	// CountInputTokens must count the actual complete provider request, including
	// ontology/configuration, instructions, framing and reserved token limits.
	CountInputTokens func(ModelInput) (uint64, error)
	Model            Model[TKind, TRel, TAttr]
}
