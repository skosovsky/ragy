// Package recording connects bounded recipes to optional immutable evidence sinks.
package recording

import (
	"context"
	"reflect"
	"slices"
	"strconv"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/evidence"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/recipe"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

type Config[TIntent, TRequestMeta, TMeta any] struct {
	Recipe          *recipe.Recipe[TIntent, TRequestMeta, TMeta]
	Mode            evidence.Mode
	Sink            evidence.Sink
	RetrievalID     string
	Schema          filter.Schema
	Codec           retrieval.MetadataCodec[TMeta]
	SourceAdmission func(context.Context, access.Binding, source.Reference) error
	CloneMeta       func(TMeta) (TMeta, error)
	Policy          evidence.Policy
	Required        []evidence.Field
}

// Run executes one recipe attempt; sink failure cannot restart retrieval or model
// calls. Parent context governs both execution and subsequent evidence export.
func Run[TIntent, TRequestMeta, TMeta any](
	ctx context.Context,
	request retrieval.Request[TIntent, TRequestMeta],
	config Config[TIntent, TRequestMeta, TMeta],
) (evidence.Execution[recipe.Result[TMeta]], error) {
	if config.Recipe == nil {
		return evidence.Execution[recipe.Result[TMeta]]{}, ragy.ErrInvalidArgument
	}
	if config.Mode != evidence.Disabled {
		if config.CloneMeta == nil || config.RetrievalID == "" {
			return evidence.Execution[recipe.Result[TMeta]]{}, ragy.ErrInvalidArgument
		}
		if err := admission(ctx, request.Read, config); err != nil {
			return evidence.Execution[recipe.Result[TMeta]]{}, err
		}
	}
	config.Required = slices.Clone(config.Required)
	return evidence.Run(ctx, request.Read, evidence.RecordingConfig[recipe.Result[TMeta]]{
		Mode: config.Mode,
		Sink: config.Sink,
		Execute: func(callCtx context.Context) (recipe.Result[TMeta], error) {
			if config.Mode == evidence.Disabled {
				return config.Recipe.Run(callCtx, request)
			}
			return config.Recipe.RunObserved(callCtx, request)
		},
		CloneResult: func(result recipe.Result[TMeta]) (recipe.Result[TMeta], error) {
			return recipe.SnapshotResult(ctx, request.Read, result, config.CloneMeta)
		},
		Capture: func(callCtx context.Context, read access.Binding, result recipe.Result[TMeta], executionErr error) (evidence.Record, error) {
			input, err := observations(request.Text, read, result, executionErr, config)
			if err != nil {
				return evidence.Record{}, err
			}
			return evidence.Capture(callCtx, read, input, config.Policy)
		},
	})
}

func admission[TIntent, TRequestMeta, TMeta any](
	ctx context.Context,
	read access.Binding,
	config Config[TIntent, TRequestMeta, TMeta],
) error {
	if err := read.Check(ctx); err != nil {
		return err
	}
	if read.IsScoped() {
		if nilCodec(config.Codec) || config.SourceAdmission == nil {
			return access.UnsupportedCapability(ragy.ErrUnsupported)
		}
		if _, err := read.Prepare(
			ctx,
			config.Schema,
			filter.Condition{},
			access.Capabilities{ScopeProfile: true, PinnedPublication: true, RequirePinnedPublication: false},
		); err != nil {
			return err
		}
	}
	for _, field := range config.Required {
		switch field {
		case evidence.QueryField:
			if !config.Policy.AllowQuery {
				return evidence.ErrPrivacy
			}
		case evidence.SnippetField:
			if config.Policy.AllowSnippet == nil {
				return evidence.ErrPrivacy
			}
		case evidence.JudgmentField:
			return ragy.ErrUnsupported
		case evidence.ScopeField, evidence.PublicationField, evidence.SourceField, evidence.ScoreField:
		default:
			return ragy.ErrUnsupported
		}
	}
	return nil
}

func nilCodec(value any) bool {
	if value == nil {
		return true
	}
	v := reflect.ValueOf(value)
	switch v.Kind() {
	case reflect.Pointer, reflect.Interface, reflect.Func, reflect.Map, reflect.Slice, reflect.Chan:
		return v.IsNil()
	case reflect.Invalid, reflect.Bool, reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr,
		reflect.Float32, reflect.Float64, reflect.Complex64, reflect.Complex128,
		reflect.Array, reflect.String, reflect.Struct, reflect.UnsafePointer:
		return false
	default:
		return false
	}
}

func observations[TIntent, TRequestMeta, TMeta any](
	query string,
	read access.Binding,
	result recipe.Result[TMeta],
	executionErr error,
	config Config[TIntent, TRequestMeta, TMeta],
) (evidence.Input[TMeta], error) {
	input := evidence.Input[TMeta]{Schema: config.Schema, Codec: config.Codec, SourceAdmission: config.SourceAdmission,
		RetrievalID: config.RetrievalID, RecipeRevision: result.RecipeRevision, Query: query,
		Outcome: evidence.Failed, Reason: evidence.TargetFailure, Coverage: result.Admission,
		Required: config.Required, Stages: nil, Diagnostics: nil}
	if executionErr != nil && result.Publication == "" {
		// Pre-attempt failure has no journal; no earlier hits/counts are invented.
		input.Stages = []evidence.Stage[TMeta]{unobservedStage[TMeta]("recipe", evidence.MissingObservation)}
		return input, nil
	}
	if result.Publication != read.Publication().Reference() {
		return evidence.Input[TMeta]{}, access.NonSkippable(ragy.ErrUnavailable)
	}
	var err error
	if executionErr == nil {
		input.Outcome, input.Reason, err = outcome(result)
		if err != nil {
			return evidence.Input[TMeta]{}, err
		}
	}
	input.Stages, err = stages(result)
	if err != nil {
		return evidence.Input[TMeta]{}, err
	}
	input.Diagnostics = diagnostics(result)
	return input, nil
}

func outcome[TMeta any](result recipe.Result[TMeta]) (evidence.Outcome, evidence.Reason, error) {
	value := evidence.Insufficient
	switch result.Outcome {
	case recipe.Complete:
		value = evidence.Complete
	case recipe.Partial:
		value = evidence.Partial
	case recipe.Insufficient:
	case recipe.Failure:
		return "", "", ragy.ErrProtocol
	default:
		return "", "", ragy.ErrProtocol
	}
	reason := evidence.NoReason
	switch result.Stop {
	case recipe.Assessed:
		if value == evidence.Partial {
			reason = evidence.PartialTargets
		}
		if value == evidence.Insufficient {
			reason = evidence.MissingEvidence
		}
	case recipe.BudgetExhausted, recipe.PriceUnavailable:
		reason = evidence.Budget
	case recipe.DeadlineReached:
		reason = evidence.Deadline
	case recipe.StageFailure:
		return "", "", ragy.ErrProtocol
	default:
		return "", "", ragy.ErrProtocol
	}
	return value, reason, nil
}

func unobservedStage[TMeta any](name string, status evidence.Status) evidence.Stage[TMeta] {
	return evidence.Stage[TMeta]{
		Name:      name,
		Status:    status,
		Scores:    evidence.Unavailable,
		Sources:   evidence.Unavailable,
		Judgments: evidence.Unavailable,
		Hits:      nil,
	}
}

func stages[TMeta any](result recipe.Result[TMeta]) ([]evidence.Stage[TMeta], error) {
	if err := validateQueryContributions(result); err != nil {
		return nil, err
	}
	switch result.Fusion {
	case recipe.FusionNotRun, recipe.FusionMissing, recipe.FusionObserved:
	default:
		return nil, ragy.ErrProtocol
	}
	out := make([]evidence.Stage[TMeta], 0, len(result.Stages)+1)
	nextQuery := 0
	retrievalOrdinal := 0
	planned, assessed := false, false
	for _, stage := range result.Stages {
		switch stage.Operation {
		case recipe.Plan:
			planned = true
			out = append(out, observedStage[TMeta]("plan"))
		case recipe.Assess:
			assessed = true
			out = append(out, observedStage[TMeta]("assess"))
		case recipe.Retrieve:
			name := "retrieve/" + strconv.Itoa(retrievalOrdinal)
			retrievalOrdinal++
			if nextQuery >= len(result.Queries) {
				out = append(out, unobservedStage[TMeta](name, evidence.MissingObservation))
				continue
			}
			query := result.Queries[nextQuery]
			if query.Index != nextQuery || len(query.Documents) != len(query.Supports) {
				return nil, ragy.ErrProtocol
			}
			captured := observedStage[TMeta](name)
			for i, doc := range query.Documents {
				captured.Hits = append(
					captured.Hits,
					evidence.Hit[TMeta]{
						Document:  doc,
						Sources:   references(query.Supports[i]),
						Judgment:  nil,
						Locations: slices.Clone(query.Supports[i]),
						Contributions: []evidence.Contribution{
							{
								QueryIndex: query.Index,
								DocumentID: doc.ID,
								Rank:       i + 1,
								Locations:  slices.Clone(query.Supports[i]),
							},
						},
					},
				)
			}
			out = append(out, captured)
			nextQuery++
		default:
			return nil, ragy.ErrProtocol
		}
	}
	if nextQuery != len(result.Queries) {
		return nil, ragy.ErrProtocol
	}
	if !planned {
		out = append(out, unobservedStage[TMeta]("plan", evidence.NotRun))
	}
	if !assessed {
		out = append(out, unobservedStage[TMeta]("assess", evidence.NotRun))
	}
	return append(out, fusionStage(result)), nil
}

func fusionStage[TMeta any](result recipe.Result[TMeta]) evidence.Stage[TMeta] {
	fusion := observedStage[TMeta]("fusion")
	if result.Fusion != recipe.FusionObserved {
		status := evidence.NotRun
		if result.Fusion == recipe.FusionMissing {
			status = evidence.MissingObservation
		}
		return unobservedStage[TMeta]("fusion", status)
	}
	for _, selected := range result.Selected {
		var refs []source.Reference
		var locations []source.Locator
		var contributions []evidence.Contribution
		for _, contributor := range selected.Contributors {
			refs = append(refs, references(contributor.Supports)...)
			locations = append(locations, contributor.Supports...)
			contributions = append(
				contributions,
				evidence.Contribution{
					QueryIndex: contributor.QueryIndex,
					DocumentID: contributor.DocumentID,
					Rank:       contributor.Rank,
					Locations:  slices.Clone(contributor.Supports),
				},
			)
		}
		fusion.Hits = append(
			fusion.Hits,
			evidence.Hit[TMeta]{
				Document:  selected.Document,
				Sources:   unique(refs),
				Judgment:  nil,
				Locations: locations, Contributions: contributions,
			},
		)
	}
	return fusion
}

func observedStage[TMeta any](name string) evidence.Stage[TMeta] {
	// Model events have an observed empty document set; no scores are fabricated.
	return evidence.Stage[TMeta]{
		Name:      name,
		Status:    evidence.StageObserved,
		Scores:    evidence.Observed,
		Sources:   evidence.Observed,
		Judgments: evidence.Unavailable,
		Hits:      nil,
	}
}
func references(locations []source.Locator) []source.Reference {
	out := make([]source.Reference, len(locations))
	for i, location := range locations {
		out[i] = location.Reference
	}
	return unique(out)
}
func unique(input []source.Reference) []source.Reference {
	seen := make(map[source.Reference]bool)
	out := make([]source.Reference, 0, len(input))
	for _, ref := range input {
		if !seen[ref] {
			seen[ref] = true
			out = append(out, ref)
		}
	}
	return out
}
func diagnostics[TMeta any](result recipe.Result[TMeta]) []evidence.Diagnostic {
	modelCalls, retrievalCalls := 0, 0
	for _, stage := range result.Stages {
		if stage.Operation == recipe.Retrieve {
			retrievalCalls++
		} else {
			modelCalls++
		}
	}
	known := result.Budget.UnknownUsage == 0
	return []evidence.Diagnostic{
		{Kind: evidence.ModelCalls, Number: number(float64(modelCalls), true)},
		{Kind: evidence.RetrievalCalls, Number: number(float64(retrievalCalls), true)},
		{Kind: evidence.InputTokens, Number: integerNumber(result.Budget.Actual.InputTokens, known)},
		{Kind: evidence.OutputTokens, Number: integerNumber(result.Budget.Actual.OutputTokens, known)},
		{
			Kind:   evidence.CostUnits,
			Number: integerNumber(result.Budget.Actual.Cost, known && !result.Budget.UnknownCost),
		},
	}
}
func integerNumber(value uint64, known bool) evidence.Number {
	// The wire number must not silently round exact host accounting units.
	return number(float64(value), known && value <= 1<<53)
}
func number(value float64, known bool) evidence.Number {
	if !known {
		return evidence.Number{State: evidence.Unavailable, Value: nil}
	}
	return evidence.Number{State: evidence.Observed, Value: &value}
}
