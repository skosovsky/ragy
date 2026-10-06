package retrieval

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
)

type configNilBackend struct{}

func (*configNilBackend) Retrieve(context.Context, Query[struct{}]) (ResultSet[struct{}], error) {
	panic("typed-nil backend dispatched")
}

type configProcessor struct{ calls int }

func (p *configProcessor) Process(
	_ context.Context,
	_ access.Binding,
	rs ResultSet[struct{}],
) (ResultSet[struct{}], error) {
	p.calls++
	return rs, nil
}

type configResolver struct{}

func (configResolver) Resolve(d Document[struct{}]) Identity {
	return Identity{DocumentID: d.ID, MergeKey: "group"}
}

type capabilityResultSet struct {
	ResultSet[struct{}]

	resolver IdentityResolver[struct{}]
}

func (r capabilityResultSet) IdentityResolver() IdentityResolver[struct{}] { return r.resolver }

func TestCompositionTypedNilBackendRejected(t *testing.T) {
	// Arrange.
	var backend *configNilBackend
	node := BackendNode[struct{}, struct{}, NoExecutionMeta]{Backend: backend}
	req := Query[struct{}]{Read: UnrestrictedRead(), Options: RetrieveOptions{TopK: 1}}
	// Act.
	result, err := node.Execute(t.Context(), req, NoExecutionMeta{})
	_, buildErr := NewExecutionPipelineBuilder[struct{}, struct{}, NoExecutionMeta]().WithRoot(node).
		Build()
	// Assert.
	if !errors.Is(err, ragy.ErrInvalidArgument) || !result.IsEmpty() ||
		!errors.Is(buildErr, ragy.ErrInvalidArgument) {
		t.Fatal(result, err, buildErr)
	}
}

func TestNilChainProcessorsRejectBeforeNextCallback(t *testing.T) {
	var typedNil *configProcessor
	for _, invalid := range []PostProcessor[struct{}]{nil, typedNil} {
		// Arrange.
		next := &configProcessor{}
		chain := NewPostProcessorChain(invalid, next)
		input := NewResultSet(
			[]Document[struct{}]{{ID: "owned", Content: "retained", ScoreState: ScoreAbsent}},
			nil,
		)
		// Act.
		result, err := chain.Process(
			t.Context(),
			UnrestrictedRead(),
			RetrieveOptions{TopK: 1},
			input,
		)
		// Assert: ordinary configuration failure can preserve admitted input.
		if !errors.Is(err, ragy.ErrInvalidArgument) || result.Len() != 1 || next.calls != 0 {
			t.Fatal(result, err, next.calls)
		}
	}
}

func TestCustomResultSetResolverCapabilitySurvivesNormalization(t *testing.T) {
	// Arrange.
	input := capabilityResultSet{
		ResultSet: NewResultSet(
			[]Document[struct{}]{
				{ID: "left", Content: "same", ScoreState: ScoreAbsent, Rank: 1},
				{ID: "right", Content: "same", ScoreState: ScoreAbsent, Rank: 2},
			},
			nil,
		),
		resolver: configResolver{},
	}
	// Act.
	normalized, err := NormalizeRankOnlyResultSet[struct{}](input, LinearRankNormalizer{})
	if err != nil {
		t.Fatal(err)
	}
	deduplicated, dedupErr := normalized.Dedup()
	// Assert.
	if dedupErr != nil || deduplicated.Len() != 1 ||
		ResolverFor(normalized).Resolve(normalized.Documents()[0]).MergeKey != "group" {
		t.Fatal(deduplicated, dedupErr)
	}
	if input.Len() != 2 {
		t.Fatal("normalization mutated input")
	}
}

func TestAggregateNilChildRejectsBeforeDispatch(t *testing.T) {
	// Arrange.
	spy := &configNilBackend{}
	node := AggregateNode[struct{}, struct{}, NoExecutionMeta]{
		Nodes: []ExecutionNode[struct{}, struct{}, NoExecutionMeta]{
			BackendNode[struct{}, struct{}, NoExecutionMeta]{Backend: spy},
			nil,
		},
	}
	// Act.
	result, err := node.Execute(
		t.Context(),
		Query[struct{}]{Read: UnrestrictedRead(), Options: RetrieveOptions{TopK: 1}},
		NoExecutionMeta{},
	)
	// Assert: spy would panic if dispatched, so all child config is admitted first.
	if !errors.Is(err, ragy.ErrInvalidArgument) || !result.IsEmpty() {
		t.Fatal(result, err)
	}
}

type thresholdStageProcessor struct {
	seen []string
}

func (p *thresholdStageProcessor) Process(
	_ context.Context,
	_ access.Binding,
	input ResultSet[struct{}],
) (ResultSet[struct{}], error) {
	docs := input.Documents()
	for i := range docs {
		p.seen = append(p.seen, docs[i].ID)
		docs[i].Score = 0.1
	}
	return NewResultSet(docs, ResolverFor(input)), nil
}

func TestThresholdFiltersChainInputAndPipelineOutput(t *testing.T) {
	// Arrange: backend ignores threshold; processor lowers the surviving score.
	docs := []Document[struct{}]{
		{
			ID:             "above",
			Content:        "a",
			Score:          0.9,
			ScoreState:     ScorePresent,
			ScoreSemantics: "fixture",
		},
		{
			ID:             "below",
			Content:        "b",
			Score:          0.2,
			ScoreState:     ScorePresent,
			ScoreSemantics: "fixture",
		},
	}
	opts := RetrieveOptions{
		TopK:      2,
		Threshold: &ScoreThreshold{Value: 0.5, State: ScorePresent, Semantics: "fixture"},
	}
	chainProcessor := &thresholdStageProcessor{}
	pipelineProcessor := &thresholdStageProcessor{}
	pipeline, buildErr := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().
		WithRoot(stubNode[struct{}]{docs: docs}).WithPostProcessors(pipelineProcessor).Build()
	if buildErr != nil {
		t.Fatal(buildErr)
	}
	// Act.
	chainOut, chainErr := NewPostProcessorChain[struct{}](
		chainProcessor,
	).Process(t.Context(), UnrestrictedRead(), opts, NewResultSet(docs, nil))
	result, pipelineErr := pipeline.Execute(
		t.Context(),
		Query[stubIntent]{Read: UnrestrictedRead(), Options: opts},
	)
	// Assert: chain prefilter leaves lowered output; pipeline applies terminal threshold.
	if chainErr != nil || pipelineErr != nil || chainOut.Len() != 1 || !result.IsEmpty() ||
		len(
			chainProcessor.seen,
		) != 1 || chainProcessor.seen[0] != "above" || len(pipelineProcessor.seen) != 1 || pipelineProcessor.seen[0] != "above" {
		t.Fatal(
			chainOut,
			result,
			chainErr,
			pipelineErr,
			chainProcessor.seen,
			pipelineProcessor.seen,
		)
	}
	if docs[0].Score != 0.9 {
		t.Fatal("input mutated")
	}
}

func TestTypedNilPlannerBinderAndResolverPolicy(t *testing.T) {
	// Arrange.
	var planner QueryPlannerFunc[stubIntent, NoRequestMeta]
	var binder RequestPlanBinderFunc[stubIntent, NoRequestMeta, NoExecutionMeta]
	var resolver *configResolver
	root := stubNode[struct{}]{docs: []Document[struct{}]{{ID: "a", Content: "a"}}}
	// Act.
	_, planErr := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().WithRoot(root).
		WithPlanner(planner).
		Build()
	_, bindErr := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().WithRoot(root).
		WithPlanBinder(binder).
		Build()
	pipeline, resolverErr := newResultPipelineBuilderNoMeta[stubIntent, struct{}]().WithRoot(root).
		WithResolver(resolver).
		Build()
	// Assert.
	if !errors.Is(planErr, ragy.ErrInvalidArgument) ||
		!errors.Is(bindErr, ragy.ErrInvalidArgument) ||
		resolverErr != nil {
		t.Fatal(planErr, bindErr, resolverErr)
	}
	out, err := pipeline.Execute(
		t.Context(),
		Query[stubIntent]{Read: UnrestrictedRead(), Options: RetrieveOptions{TopK: 1}},
	)
	if err != nil || ResolverFor(out.ResultSet).Resolve(out.Documents()[0]).MergeKey != "a" {
		t.Fatal(out, err)
	}
}
