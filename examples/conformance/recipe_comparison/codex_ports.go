package main

import (
	"context"
	"encoding/json"
	"slices"

	"github.com/skosovsky/ragy/examples/conformance/internal/codexcall"

	"github.com/skosovsky/ragy/adapters/openai/structured"
	"github.com/skosovsky/ragy/recipe"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/retrieval"
)

// CLI ports preserve actual executor receipts separately from the library ledger.
// Their profile must declare unknown provider price and advisory token bounds.
type codexPorts struct {
	config   codexcall.Config
	strategy recipe.Strategy
	receipts []codexcall.Result
}

func (p *codexPorts) call(
	ctx context.Context,
	instructions, schema string,
	input any,
) (codexcall.Result, recipe.Usage, error) {
	result, err := codexcall.Call(ctx, p.config, instructions, json.RawMessage(schema), input)
	p.receipts = append(p.receipts, result)
	usage := recipe.Usage{}
	if result.Usage != nil {
		usage = recipe.Usage{
			Known: true,
			Value: budget.Usage{InputTokens: result.Usage.Input, OutputTokens: result.Usage.Output},
		}
	}
	return result, usage, err
}

func (p *codexPorts) Plan(
	ctx context.Context,
	req retrieval.Request[struct{}, retrieval.NoRequestMeta],
	_ recipe.ModelLimits,
) (recipe.Planning, error) {
	if err := req.Read.Check(ctx); err != nil {
		return recipe.Planning{}, err
	}
	limit := maximumAssessmentQueries
	if p.strategy == recipe.SingleRewrite {
		limit = 1
	}
	if p.strategy == recipe.MultiQuery {
		limit = normalModelCalls
	}
	input := structured.PlannerInput{Text: req.EffectiveText(), Strategy: p.strategy, MaxQueries: limit}
	result, usage, err := p.call(ctx, plannerInstructions, plannerSchema, input)
	if gateErr := req.Read.Check(ctx); gateErr != nil {
		return recipe.Planning{Usage: usage}, gateErr
	}
	if err != nil {
		return recipe.Planning{Usage: usage}, err
	}
	var output structured.PlannerOutput
	if err = codexcall.Decode(
		result.Output,
		&output,
	); err != nil || output.Queries == nil || len(output.Queries) == 0 ||
		len(output.Queries) > limit {
		return recipe.Planning{Usage: usage}, errInvalid
	}
	for _, text := range output.Queries {
		if text == "" || len(text) > maximumQueryBytes {
			return recipe.Planning{Usage: usage}, errInvalid
		}
	}
	return recipe.Planning{Queries: slices.Clone(output.Queries), Usage: usage}, nil
}

func (p *codexPorts) Assess(
	ctx context.Context,
	input recipe.AssessmentInput[struct{}, retrieval.NoRequestMeta, comparisonMetadata],
	_ recipe.ModelLimits,
) (recipe.Assessment, error) {
	if err := input.Original.Read.Check(ctx); err != nil {
		return recipe.Assessment{}, err
	}
	wire, indices, err := projectCodexAssessment(input)
	if err != nil {
		return recipe.Assessment{}, err
	}
	result, usage, err := p.call(ctx, assessorInstructions, assessorSchema, wire)
	if gateErr := input.Original.Read.Check(ctx); gateErr != nil {
		return recipe.Assessment{Usage: usage}, gateErr
	}
	if err != nil {
		return recipe.Assessment{Usage: usage}, err
	}
	var output struct {
		Selected   []int `json:"selected"`
		Sufficient *bool `json:"sufficient"`
	}
	if err = codexcall.Decode(
		result.Output,
		&output,
	); err != nil || output.Selected == nil ||
		output.Sufficient == nil {
		return recipe.Assessment{Usage: usage}, errInvalid
	}
	seen := make(map[int]bool)
	for _, index := range output.Selected {
		if !indices[index] || seen[index] {
			return recipe.Assessment{Usage: usage}, errInvalid
		}
		seen[index] = true
	}
	return recipe.Assessment{Selected: slices.Clone(output.Selected), Sufficient: *output.Sufficient, Usage: usage}, nil
}

func projectCodexAssessment(
	input recipe.AssessmentInput[struct{}, retrieval.NoRequestMeta, comparisonMetadata],
) (structured.AssessorInput, map[int]bool, error) {
	wire := structured.AssessorInput{Text: input.Original.EffectiveText()}
	remaining, documents := maximumAssessmentBytes-len(wire.Text), 0
	indices := make(map[int]bool)
	if len(input.Queries) == 0 || len(input.Queries) > maximumAssessmentQueries || remaining < 0 {
		return wire, nil, errInvalid
	}
	for _, query := range input.Queries {
		if query.Index < 0 || indices[query.Index] || len(query.Text) > remaining {
			return wire, nil, errInvalid
		}
		remaining -= len(query.Text)
		indices[query.Index] = true
		projected := structured.AssessmentQuery{Index: query.Index, Text: query.Text, Documents: []string{}}
		for _, doc := range query.Documents {
			if len(doc.Content) > remaining || documents >= maximumAssessmentDocuments {
				return wire, nil, errInvalid
			}
			remaining -= len(doc.Content)
			documents++
			projected.Documents = append(projected.Documents, doc.Content)
		}
		wire.Queries = append(wire.Queries, projected)
	}
	return wire, indices, nil
}
