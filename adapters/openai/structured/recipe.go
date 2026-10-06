package structured

import (
	"context"
	"strings"
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/recipe"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/retrieval"
)

// PlannerInput exposes retrieval text and strategy, never domain metadata or scope.
type PlannerInput struct {
	Text       string          `json:"text"`
	Strategy   recipe.Strategy `json:"strategy"`
	MaxQueries int             `json:"max_queries"`
}
type PlannerOutput struct {
	Queries []string `json:"queries"`
}

type Planner[TIntent, TRequestMeta any] struct {
	client     *Client[PlannerOutput]
	strategy   recipe.Strategy
	maxQueries int
	maxBytes   int
	price      func(Usage) (uint64, error)
}

func NewPlanner[TIntent, TRequestMeta any](
	cfg Config,
	strategy recipe.Strategy,
	maxQueryBytes int,
	price func(Usage) (uint64, error),
) (*Planner[TIntent, TRequestMeta], error) {
	maximum := 0
	switch strategy {
	case recipe.SingleRewrite:
		maximum = 1
	case recipe.MultiQuery:
		maximum = 2
	case recipe.Decomposition:
		maximum = 3
	}
	if maximum == 0 || maxQueryBytes <= 0 || price == nil {
		return nil, ragy.ErrInvalidArgument
	}
	client, err := New[PlannerOutput](cfg)
	if err != nil {
		return nil, err
	}
	return &Planner[TIntent, TRequestMeta]{
		client:     client,
		strategy:   strategy,
		maxQueries: maximum,
		maxBytes:   maxQueryBytes,
		price:      price,
	}, nil
}

func validRecipeText(text string, maximum int) bool {
	return len(text) <= maximum && utf8.ValidString(text) && strings.TrimSpace(text) != ""
}

func recipeUsage(tokens Usage, price func(Usage) (uint64, error)) recipe.Usage {
	usage := recipe.Usage{
		Value: budget.Usage{InputTokens: tokens.InputTokens, OutputTokens: tokens.OutputTokens, Cost: 0},
		Known: false,
	}
	if tokens.Known {
		cost, err := price(tokens)
		if err == nil {
			usage.Value.Cost, usage.Known = cost, true
		}
	}
	return usage
}

func (p *Planner[TIntent, TRequestMeta]) Plan(
	ctx context.Context,
	req retrieval.Request[TIntent, TRequestMeta],
	limits recipe.ModelLimits,
) (recipe.Planning, error) {
	if p == nil || ctx == nil {
		return recipe.Planning{}, ragy.ErrInvalidArgument
	}
	if err := req.Read.Check(ctx); err != nil {
		return recipe.Planning{}, err
	}
	text := req.EffectiveText()
	if !validRecipeText(text, p.maxBytes) {
		return recipe.Planning{}, ragy.ErrInvalidArgument
	}
	output, tokens, err := p.client.call(
		ctx,
		PlannerInput{Text: text, Strategy: p.strategy, MaxQueries: p.maxQueries},
		Limits{InputTokens: limits.InputTokens, OutputTokens: limits.OutputTokens},
		req.Read.Check,
	)
	usage := recipeUsage(tokens, p.price)
	if gateErr := req.Read.Check(ctx); gateErr != nil {
		return recipe.Planning{Queries: nil, Usage: usage}, gateErr
	}
	if err != nil {
		return recipe.Planning{Queries: nil, Usage: usage}, err
	}
	if len(output.Queries) > p.maxQueries {
		return recipe.Planning{Queries: nil, Usage: usage}, ragy.ErrProtocol
	}
	seen := make(map[string]bool, len(output.Queries))
	for _, query := range output.Queries {
		if !validRecipeText(query, p.maxBytes) || seen[query] {
			return recipe.Planning{Queries: nil, Usage: usage}, ragy.ErrProtocol
		}
		seen[query] = true
	}
	return recipe.Planning{Queries: output.Queries, Usage: usage}, nil
}

// AssessmentQuery exposes only query ordinals/text and already admitted snippets.
// Source refs, IDs, metadata and authorization stay with the core.
type AssessmentQuery struct {
	Index     int      `json:"index"`
	Text      string   `json:"text"`
	Documents []string `json:"documents"`
}
type AssessorInput struct {
	Text    string            `json:"text"`
	Queries []AssessmentQuery `json:"queries"`
}
type AssessorOutput struct {
	Selected   []int `json:"selected"`
	Sufficient bool  `json:"sufficient"`
}

type Assessor[TIntent, TRequestMeta, TMeta any] struct {
	client       *Client[AssessorOutput]
	maxQueries   int
	maxDocuments int
	maxBytes     int
	price        func(Usage) (uint64, error)
}

// NewAssessor bounds aggregate query/snippet bytes; Config additionally bounds the
// complete request with instructions/schema. The host supplies schema and price.
func NewAssessor[TIntent, TRequestMeta, TMeta any](
	cfg Config,
	maxQueries, maxDocuments, maxBytes int,
	price func(Usage) (uint64, error),
) (*Assessor[TIntent, TRequestMeta, TMeta], error) {
	if maxQueries <= 0 || maxQueries > 3 || maxDocuments <= 0 || maxBytes <= 0 || price == nil {
		return nil, ragy.ErrInvalidArgument
	}
	client, err := New[AssessorOutput](cfg)
	if err != nil {
		return nil, err
	}
	return &Assessor[TIntent, TRequestMeta, TMeta]{
		client:       client,
		maxQueries:   maxQueries,
		maxDocuments: maxDocuments,
		maxBytes:     maxBytes,
		price:        price,
	}, nil
}

func (a *Assessor[TIntent, TRequestMeta, TMeta]) project(
	input recipe.AssessmentInput[TIntent, TRequestMeta, TMeta],
) (AssessorInput, map[int]bool, error) {
	text := input.Original.EffectiveText()
	if !validRecipeText(text, a.maxBytes) || len(input.Queries) == 0 || len(input.Queries) > a.maxQueries {
		return AssessorInput{}, nil, ragy.ErrInvalidArgument
	}
	wire := AssessorInput{Text: text, Queries: make([]AssessmentQuery, 0, len(input.Queries))}
	remaining, documents := a.maxBytes-len(text), 0
	indices := make(map[int]bool, len(input.Queries))
	for _, query := range input.Queries {
		if query.Index < 0 || indices[query.Index] || !validRecipeText(query.Text, remaining) ||
			len(query.Documents) > a.maxDocuments-documents {
			return AssessorInput{}, nil, ragy.ErrInvalidArgument
		}
		remaining -= len(query.Text)
		indices[query.Index] = true
		projected := AssessmentQuery{
			Index:     query.Index,
			Text:      query.Text,
			Documents: make([]string, 0, len(query.Documents)),
		}
		for _, doc := range query.Documents {
			if !utf8.ValidString(doc.Content) || len(doc.Content) > remaining {
				return AssessorInput{}, nil, ragy.ErrInvalidArgument
			}
			remaining -= len(doc.Content)
			documents++
			projected.Documents = append(projected.Documents, doc.Content)
		}
		wire.Queries = append(wire.Queries, projected)
	}
	return wire, indices, nil
}

func (a *Assessor[TIntent, TRequestMeta, TMeta]) Assess(
	ctx context.Context,
	input recipe.AssessmentInput[TIntent, TRequestMeta, TMeta],
	limits recipe.ModelLimits,
) (recipe.Assessment, error) {
	if a == nil || ctx == nil {
		return recipe.Assessment{}, ragy.ErrInvalidArgument
	}
	if err := input.Original.Read.Check(ctx); err != nil {
		return recipe.Assessment{}, err
	}
	wire, indices, err := a.project(input)
	if err != nil {
		return recipe.Assessment{}, err
	}
	output, tokens, err := a.client.call(
		ctx,
		wire,
		Limits{InputTokens: limits.InputTokens, OutputTokens: limits.OutputTokens},
		input.Original.Read.Check,
	)
	usage := recipeUsage(tokens, a.price)
	if gateErr := input.Original.Read.Check(ctx); gateErr != nil {
		return recipe.Assessment{Selected: nil, Sufficient: false, Usage: usage}, gateErr
	}
	if err != nil {
		return recipe.Assessment{Selected: nil, Sufficient: false, Usage: usage}, err
	}
	selected := make(map[int]bool, len(output.Selected))
	for _, index := range output.Selected {
		if !indices[index] || selected[index] {
			return recipe.Assessment{Selected: nil, Sufficient: false, Usage: usage}, ragy.ErrProtocol
		}
		selected[index] = true
	}
	return recipe.Assessment{Selected: output.Selected, Sufficient: output.Sufficient, Usage: usage}, nil
}
