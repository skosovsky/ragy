package graphsummary

import (
	"context"
	"errors"
	"slices"
	"unicode/utf8"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/recipe"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/source"
)

type Recipe[TAccess any] struct{ config Config[TAccess] }

func New[TAccess any](cfg Config[TAccess]) (*Recipe[TAccess], error) {
	if !validLimits(cfg) || !validPorts(cfg) {
		return nil, ragy.ErrInvalidArgument
	}
	return &Recipe[TAccess]{config: cfg}, nil
}

// Community runs one map summary. Global performs one map per admitted community
// and one non-recursive reduction, with at most three model calls in total.
func (r *Recipe[TAccess]) Community(
	ctx context.Context,
	request Request[TAccess],
	ledger *budget.Ledger,
) (Result, error) {
	if len(request.Communities) != 1 {
		return Result{}, ragy.ErrInvalidArgument
	}
	return r.run(ctx, request, ledger, false)
}
func (r *Recipe[TAccess]) Global(ctx context.Context, request Request[TAccess], ledger *budget.Ledger) (Result, error) {
	if len(request.Communities) != maxCommunities {
		return Result{}, ragy.ErrInvalidArgument
	}
	return r.run(ctx, request, ledger, true)
}

func (r *Recipe[TAccess]) run(
	ctx context.Context,
	request Request[TAccess],
	ledger *budget.Ledger,
	global bool,
) (Result, error) {
	if r == nil || ledger == nil {
		return Result{}, ragy.ErrInvalidArgument
	}
	child, cancel := context.WithTimeout(ctx, r.config.Duration)
	defer cancel()
	deadline := r.config.Now().Add(r.config.Duration)
	gate := func() error {
		if err := request.Read.Check(child); err != nil {
			return err
		}
		if !r.config.Now().Before(deadline) {
			return context.DeadlineExceeded
		}
		return nil
	}
	if err := gate(); err != nil {
		return Result{}, err
	}
	communities, binding, err := r.admit(child, request, gate)
	if err != nil {
		return Result{}, err
	}
	result := Result{Communities: nil, Global: nil, Outcome: recipe.Complete, Stop: Summarized, ModelCalls: 0}
	for _, community := range communities {
		if err = r.refresh(child, request.Read, communities, gate); err != nil {
			return Result{}, err
		}
		output, callErr := r.call(
			child,
			ledger,
			ModelInput{
				Stage:           Map,
				Question:        request.Question,
				Snippets:        mapSnippets(community),
				MaxInputTokens:  0,
				MaxOutputTokens: 0,
			},
			&result.ModelCalls,
			gate,
			func() error { return r.refresh(child, request.Read, communities, gate) },
		)
		if callErr != nil {
			return r.stopped(child, request.Read, result, callErr, gate)
		}
		if err = r.refresh(child, request.Read, communities, gate); err != nil {
			return Result{}, err
		}
		supports, coverage := selectedSupports(community, output.Selected)
		summary := Summary{
			text:        output.Text,
			communities: []string{community.ID},
			supports:    supports,
			binding:     binding,
			schema:      r.config.Schema,
			covered:     coverage,
		}
		result.Communities = append(result.Communities, summary)
		if !coverage {
			result.Outcome = recipe.Insufficient
			result.Stop = MissingCoverage
			return r.deliver(child, request.Read, result, gate)
		}
	}
	if global {
		return r.reduce(child, request, communities, result, ledger, gate)
	}

	return r.deliver(child, request.Read, result, gate)
}

func (r *Recipe[TAccess]) reduce(
	ctx context.Context,
	request Request[TAccess],
	communities []Community[TAccess],
	result Result,
	ledger *budget.Ledger,
	gate func() error,
) (Result, error) {
	var err error
	if err = r.refresh(ctx, request.Read, communities, gate); err != nil {
		return Result{}, err
	}
	output, err := r.call(
		ctx,
		ledger,
		reduceInput(request.Question, result.Communities),
		&result.ModelCalls,
		gate,
		func() error { return r.refresh(ctx, request.Read, communities, gate) },
	)
	if err != nil {
		return r.stopped(ctx, request.Read, result, err, gate)
	}
	if err = r.refresh(ctx, request.Read, communities, gate); err != nil {
		return Result{}, err
	}
	if len(output.Selected) != len(result.Communities) {
		return Result{}, ragy.ErrProtocol
	}
	var supports []source.Locator
	var ids []string
	for _, index := range output.Selected {
		summary := result.Communities[index]
		supports = union(supports, summary.supports)
		ids = append(ids, summary.communities...)
	}
	result.Global = &Summary{
		text:        output.Text,
		communities: ids,
		supports:    supports,
		binding:     result.Communities[0].binding,
		schema:      r.config.Schema,
		covered:     true,
	}
	return r.deliver(ctx, request.Read, result, gate)
}

func (r *Recipe[TAccess]) call(
	ctx context.Context,
	ledger *budget.Ledger,
	input ModelInput,
	calls *uint64,
	gate func() error,
	ready func() error,
) (ModelOutput, error) {
	var empty ModelOutput
	if err := gate(); err != nil {
		return empty, err
	}
	quote, err := r.config.Quote(ctx, input.Stage)
	if err != nil {
		return empty, err
	}
	if err = gate(); err != nil {
		return empty, err
	}
	if quote.Kind != budget.Model || quote.Usage.InputTokens == 0 || quote.Usage.OutputTokens == 0 {
		return empty, ragy.ErrInvalidArgument
	}
	input.MaxInputTokens, input.MaxOutputTokens = quote.Usage.InputTokens, quote.Usage.OutputTokens
	counter := input
	counter.Snippets = slices.Clone(input.Snippets)
	tokens, err := r.config.CountInputTokens(counter)
	if err != nil {
		return empty, err
	}
	if err = gate(); err != nil {
		return empty, err
	}
	if tokens == 0 || tokens > quote.Usage.InputTokens {
		return empty, budget.ErrExhausted
	}
	if err = ready(); err != nil {
		return empty, err
	}
	lease, err := ledger.Reserve(ctx, quote)
	if err != nil {
		return empty, err
	}
	if err = gate(); err != nil {
		_ = lease.Settle(budget.Usage{InputTokens: 0, OutputTokens: 0, Cost: 0}, false)
		return empty, err
	}
	if err = ready(); err != nil {
		_ = lease.Settle(budget.Usage{InputTokens: 0, OutputTokens: 0, Cost: 0}, false)
		return empty, err
	}
	input.Snippets = slices.Clone(input.Snippets)
	*calls++
	output, usage, callErr := r.config.Model(ctx, input)
	settleErr := lease.Settle(usage.Value, usage.Known && quote.CostKnown)
	if usage.Known &&
		(usage.Value.InputTokens > quote.Usage.InputTokens || usage.Value.OutputTokens > quote.Usage.OutputTokens) {
		settleErr = errors.Join(settleErr, budget.ErrUsageExceeded)
	}
	if err = gate(); err != nil {
		return empty, err
	}
	if err = errors.Join(callErr, settleErr); err != nil {
		return empty, err
	}
	if err = r.validateOutput(output, len(input.Snippets)); err != nil {
		return empty, err
	}
	output.Selected = slices.Clone(output.Selected)
	return output, nil
}

func (r *Recipe[TAccess]) validateOutput(output ModelOutput, count int) error {
	if output.Text == "" || !utf8.ValidString(output.Text) || len(output.Text) > r.config.MaxSummaryBytes ||
		len(output.Selected) == 0 ||
		len(output.Selected) > count {
		return ragy.ErrProtocol
	}
	seen := make(map[int]bool)
	for _, index := range output.Selected {
		if index < 0 || index >= count || seen[index] {
			return ragy.ErrProtocol
		}
		seen[index] = true
	}
	return nil
}

func reduceInput(question string, summaries []Summary) ModelInput {
	input := ModelInput{
		Stage:           Reduce,
		Question:        question,
		Snippets:        make([]ModelSnippet, len(summaries)),
		MaxInputTokens:  0,
		MaxOutputTokens: 0,
	}
	for i, summary := range summaries {
		input.Snippets[i] = ModelSnippet{Index: i, Text: summary.text}
	}
	return input
}

func (r *Recipe[TAccess]) stopped(
	ctx context.Context,
	read access.Binding,
	result Result,
	err error,
	gate func() error,
) (Result, error) {
	if check := read.Check(ctx); check != nil {
		return Result{}, check
	}
	stop := BudgetExhausted
	if errors.Is(err, budget.ErrUnknownPrice) {
		stop = PriceUnavailable
	} else if !errors.Is(err, budget.ErrExhausted) {
		return Result{}, err
	}
	result.Outcome = recipe.Insufficient
	if len(result.Communities) > 0 {
		result.Outcome = recipe.Partial
	}
	result.Stop = stop
	return r.deliver(ctx, read, result, gate)
}

func (r *Recipe[TAccess]) deliver(
	ctx context.Context,
	read access.Binding,
	result Result,
	gate func() error,
) (Result, error) {
	admit := func(ctx context.Context, read access.Binding, location source.Locator) error {
		if err := gate(); err != nil {
			return err
		}
		if err := r.config.AdmitSource(ctx, read, location); err != nil {
			return err
		}
		return gate()
	}
	for _, summary := range result.Communities {
		if err := gate(); err != nil {
			return Result{}, err
		}
		if _, err := summary.Resolve(ctx, read, admit); err != nil {
			return Result{}, err
		}
	}
	if result.Global != nil {
		if err := gate(); err != nil {
			return Result{}, err
		}
		if _, err := result.Global.Resolve(ctx, read, admit); err != nil {
			return Result{}, err
		}
	}
	if err := gate(); err != nil {
		return Result{}, err
	}
	return result, nil
}

func validLimits[TAccess any](cfg Config[TAccess]) bool {
	return cfg.Schema.IsFinalized() && cfg.MaxMembers > 0 && cfg.MaxSnippets > 0 &&
		cfg.MaxSnippets <= maxCommunitySnippets &&
		cfg.MaxSupports > 0 &&
		cfg.MaxInputBytes > 0 &&
		cfg.MaxSummaryBytes > 0 &&
		cfg.MaxSummaryBytes <= cfg.MaxInputBytes/maxCommunities &&
		cfg.Duration > 0
}
func validPorts[TAccess any](cfg Config[TAccess]) bool {
	return cfg.Now != nil && cfg.CloneAccess != nil && cfg.Attributes != nil && cfg.Membership != nil &&
		cfg.AdmitSource != nil &&
		cfg.Quote != nil &&
		cfg.CountInputTokens != nil &&
		cfg.Model != nil
}
