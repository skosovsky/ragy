package retrieval

import (
	"context"
	"errors"
	"testing"
)

type executionMetaFixture struct {
	Bound       bool
	SideOutputs []string
}

func TestExecutionPipelinePlanBinderUpdatesRequestBeforeBackend(t *testing.T) {
	t.Parallel()

	spy := &requestMetaSpyBackend[intentWithMode, requestMetaFixture, struct{}]{
		docs: []Document[struct{}]{
			{ID: "first", Content: "ok", Score: 1},
			{ID: "second", Content: "tail", Score: 0.5},
		},
	}
	pipeline, err := NewRequestExecutionPipelineBuilder[intentWithMode, requestMetaFixture, struct{}, executionMetaFixture]().
		WithPlanner(QueryPlannerFunc[intentWithMode, requestMetaFixture](
			func(_ context.Context, req Request[intentWithMode, requestMetaFixture]) (PlannedQuery[intentWithMode], error) {
				return PlannedQuery[intentWithMode]{
					Text:        req.Text,
					Intent:      req.Intent,
					CacheKey:    "trace-from-plan",
					Diagnostics: []PlannerDiagnostic{{Key: "planner", Value: "hit"}},
				}, nil
			},
		)).
		WithPlanBinder(RequestPlanBinderFunc[intentWithMode, requestMetaFixture, executionMetaFixture](
			func(
				_ context.Context,
				req Request[intentWithMode, requestMetaFixture],
				plan *PlannedQuery[intentWithMode],
				exec executionMetaFixture,
			) (BoundRequest[intentWithMode, requestMetaFixture, executionMetaFixture], error) {
				req.Meta.TraceID = plan.CacheKey
				req.Options.TopK = 1
				exec.Bound = true
				return BoundRequest[intentWithMode, requestMetaFixture, executionMetaFixture]{
					Request:     req,
					Executed:    exec,
					Diagnostics: []ExecutionDiagnostic{{Stage: "binder", Key: "trace", Value: plan.CacheKey}},
				}, nil
			},
		)).
		WithRoot(RequestBackendNode[intentWithMode, requestMetaFixture, struct{}, executionMetaFixture]{Backend: spy}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	result, err := pipeline.Execute(context.Background(), Request[intentWithMode, requestMetaFixture]{
		Text:   "raw",
		Intent: intentWithMode{Mode: "run"},
		Meta:   requestMetaFixture{TraceID: "incoming"},
	})
	if err != nil {
		t.Fatalf("Execute(): %v", err)
	}
	if spy.lastRequest.Meta.TraceID != "trace-from-plan" {
		t.Fatalf("backend meta = %#v, want trace-from-plan", spy.lastRequest.Meta)
	}
	if spy.lastRequest.Options.TopK != 1 {
		t.Fatalf("backend TopK = %d, want binder override 1", spy.lastRequest.Options.TopK)
	}
	if result.ResultSet.Len() != 1 || result.ResultSet.Documents()[0].ID != "first" {
		t.Fatalf("result docs = %#v, want terminal TopK applied", result.ResultSet.Documents())
	}
	if !result.Executed.Bound {
		t.Fatalf("Executed = %#v, want binder update", result.Executed)
	}
	if len(result.Diagnostics) != 2 {
		t.Fatalf("Diagnostics = %#v, want planner and binder diagnostics", result.Diagnostics)
	}
}

func TestPipelineExecuteReturnsEnvelopeAndRunsPlanBinder(t *testing.T) {
	t.Parallel()

	spy := &querySpyBackend[intentWithMode, struct{}]{
		orchestratorStubBackend: orchestratorStubBackend[intentWithMode, struct{}]{
			docs: []Document[struct{}]{{ID: "hit", Content: "ok", Score: 1}},
		},
	}
	pipeline, err := NewExecutionPipelineBuilder[intentWithMode, struct{}, NoExecutionMeta]().
		WithPlanner(StaticPlanner[intentWithMode, NoRequestMeta]{
			Planned: PlannedQuery[intentWithMode]{
				ExpandedText: "planned",
				Diagnostics:  []PlannerDiagnostic{{Key: "planner", Value: "done"}},
			},
		}).
		WithPlanBinder(RequestPlanBinderFunc[intentWithMode, NoRequestMeta, NoExecutionMeta](
			func(
				_ context.Context,
				req Query[intentWithMode],
				_ *PlannedQuery[intentWithMode],
				exec NoExecutionMeta,
			) (BoundRequest[intentWithMode, NoRequestMeta, NoExecutionMeta], error) {
				req.Options.TopK = 1
				return BoundRequest[intentWithMode, NoRequestMeta, NoExecutionMeta]{
					Request:     req,
					Executed:    exec,
					Diagnostics: []ExecutionDiagnostic{{Stage: "binder", Key: "top_k", Value: "1"}},
				}, nil
			},
		)).
		WithRoot(BackendNode[intentWithMode, struct{}, NoExecutionMeta]{Backend: spy}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	result, err := pipeline.Execute(context.Background(), Query[intentWithMode]{
		Text: "raw",
	})
	if err != nil {
		t.Fatalf("Execute(): %v", err)
	}
	if result.ResultSet.Len() != 1 || spy.lastRequest.Options.TopK != 1 {
		t.Fatalf("result/backend = %#v / %#v, want binder-bound request", result.ResultSet.Documents(), spy.lastRequest)
	}
	if len(result.Diagnostics) != 2 || len(result.BranchTrace) == 0 {
		t.Fatalf(
			"envelope diagnostics/trace = %#v / %#v, want populated envelope",
			result.Diagnostics,
			result.BranchTrace,
		)
	}
}

func TestExecutionPipelineBinderRunsForPreplannedRequest(t *testing.T) {
	t.Parallel()

	spy := &querySpyBackend[intentWithMode, struct{}]{
		orchestratorStubBackend: orchestratorStubBackend[intentWithMode, struct{}]{
			docs: []Document[struct{}]{{ID: "hit", Content: "ok", Score: 1}},
		},
	}
	plannerCalls := 0
	binderCalls := 0
	pipeline, err := NewExecutionPipelineBuilder[intentWithMode, struct{}, executionMetaFixture]().
		WithPlanner(QueryPlannerFunc[intentWithMode, NoRequestMeta](
			func(context.Context, Query[intentWithMode]) (PlannedQuery[intentWithMode], error) {
				plannerCalls++
				return PlannedQuery[intentWithMode]{Text: "unexpected"}, nil
			},
		)).
		WithPlanBinder(RequestPlanBinderFunc[intentWithMode, NoRequestMeta, executionMetaFixture](
			func(
				_ context.Context,
				req Query[intentWithMode],
				plan *PlannedQuery[intentWithMode],
				exec executionMetaFixture,
			) (BoundRequest[intentWithMode, NoRequestMeta, executionMetaFixture], error) {
				binderCalls++
				req.Options.TopK = 1
				exec.Bound = plan != nil && plan.CacheKey == "cached"
				return BoundRequest[intentWithMode, NoRequestMeta, executionMetaFixture]{
					Request:  req,
					Executed: exec,
				}, nil
			},
		)).
		WithRoot(BackendNode[intentWithMode, struct{}, executionMetaFixture]{Backend: spy}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	result, err := pipeline.Execute(context.Background(), Query[intentWithMode]{
		Text: "raw",
		Plan: &PlannedQuery[intentWithMode]{ExpandedText: "cached expanded", CacheKey: "cached"},
	})
	if err != nil {
		t.Fatalf("Execute(): %v", err)
	}
	if plannerCalls != 0 {
		t.Fatalf("planner calls = %d, want 0 for preplanned request", plannerCalls)
	}
	if binderCalls != 1 {
		t.Fatalf("binder calls = %d, want 1", binderCalls)
	}
	if !result.Executed.Bound {
		t.Fatalf("Executed = %#v, want binder to see cached plan", result.Executed)
	}
	if spy.lastRequest.Options.TopK != 1 || spy.lastRequest.EffectiveText() != "cached expanded" {
		t.Fatalf("backend request = %#v, want binder options plus cached plan", spy.lastRequest)
	}
}

func TestExecutionPipelineSeedAndBinderCanBindMissingOptions(t *testing.T) {
	t.Parallel()

	spy := &querySpyBackend[intentWithMode, struct{}]{
		orchestratorStubBackend: orchestratorStubBackend[intentWithMode, struct{}]{
			docs: []Document[struct{}]{{ID: "hit", Content: "ok", Score: 1}},
		},
	}
	pipeline, err := NewExecutionPipelineBuilder[intentWithMode, struct{}, executionMetaFixture]().
		WithExecutionSeed(func(req Query[intentWithMode]) executionMetaFixture {
			return executionMetaFixture{SideOutputs: []string{"seed:" + req.Text}}
		}).
		WithPlanBinder(RequestPlanBinderFunc[intentWithMode, NoRequestMeta, executionMetaFixture](
			func(
				_ context.Context,
				req Query[intentWithMode],
				_ *PlannedQuery[intentWithMode],
				exec executionMetaFixture,
			) (BoundRequest[intentWithMode, NoRequestMeta, executionMetaFixture], error) {
				req.Options.TopK = 1
				exec.Bound = true
				return BoundRequest[intentWithMode, NoRequestMeta, executionMetaFixture]{
					Request:  req,
					Executed: exec,
				}, nil
			},
		)).
		WithRoot(BackendNode[intentWithMode, struct{}, executionMetaFixture]{Backend: spy}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	result, err := pipeline.Execute(context.Background(), Query[intentWithMode]{Text: "raw"})
	if err != nil {
		t.Fatalf("Execute(): %v", err)
	}
	if spy.lastRequest.Options.TopK != 1 {
		t.Fatalf("backend TopK = %d, want binder-bound TopK", spy.lastRequest.Options.TopK)
	}
	if !result.Executed.Bound || len(result.Executed.SideOutputs) != 1 || result.Executed.SideOutputs[0] != "seed:raw" {
		t.Fatalf("Executed = %#v, want seed and binder metadata", result.Executed)
	}
	if result.Len() != 1 {
		t.Fatalf("Len() = %d, want backend hit", result.Len())
	}
}

func TestPipelineExecutePreservesPlannerDiagnosticsOnError(t *testing.T) {
	t.Parallel()

	plannerErr := errors.New("planner failed")
	pipeline, err := NewExecutionPipelineBuilder[intentWithMode, struct{}, NoExecutionMeta]().
		WithPlanner(QueryPlannerFunc[intentWithMode, NoRequestMeta](
			func(context.Context, Query[intentWithMode]) (PlannedQuery[intentWithMode], error) {
				return PlannedQuery[intentWithMode]{
					Diagnostics: []PlannerDiagnostic{{Key: "planner", Value: "failed"}},
				}, plannerErr
			},
		)).
		WithRoot(BackendNode[intentWithMode, struct{}, NoExecutionMeta]{
			Backend: orchestratorStubBackend[intentWithMode, struct{}]{docs: nil},
		}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	result, err := pipeline.Execute(context.Background(), Query[intentWithMode]{
		Text:    "raw",
		Options: RetrieveOptions{TopK: 1},
	})
	if !errors.Is(err, plannerErr) {
		t.Fatalf("Execute() error = %v, want planner error", err)
	}
	if len(result.Diagnostics) != 1 || result.Diagnostics[0].Value != "failed" {
		t.Fatalf("Diagnostics = %#v, want planner diagnostics on error", result.Diagnostics)
	}
	if result.ResultSet == nil || !result.ResultSet.IsEmpty() {
		t.Fatalf("ResultSet = %#v, want non-nil empty result set", result.ResultSet)
	}
	if !hasPipelineErrorTrace(result.BranchTrace, "planner") {
		t.Fatalf("BranchTrace = %#v, want planner error trace", result.BranchTrace)
	}
}

func TestExecutionPipelinePreservesPlannerDiagnosticsOnError(t *testing.T) {
	t.Parallel()

	plannerErr := errors.New("planner failed")
	pipeline, err := NewExecutionPipelineBuilder[intentWithMode, struct{}, executionMetaFixture]().
		WithPlanner(QueryPlannerFunc[intentWithMode, NoRequestMeta](
			func(context.Context, Query[intentWithMode]) (PlannedQuery[intentWithMode], error) {
				return PlannedQuery[intentWithMode]{
					Diagnostics: []PlannerDiagnostic{{Key: "planner", Value: "failed"}},
				}, plannerErr
			},
		)).
		WithRoot(BackendNode[intentWithMode, struct{}, executionMetaFixture]{
			Backend: orchestratorStubBackend[intentWithMode, struct{}]{docs: nil},
		}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	result, err := pipeline.Execute(context.Background(), Query[intentWithMode]{
		Text:    "raw",
		Options: RetrieveOptions{TopK: 1},
	})
	if !errors.Is(err, plannerErr) {
		t.Fatalf("Execute() error = %v, want planner error", err)
	}
	if len(result.Diagnostics) != 1 || result.Diagnostics[0].Value != "failed" {
		t.Fatalf("Diagnostics = %#v, want planner diagnostics on error", result.Diagnostics)
	}
	if result.ResultSet == nil || !result.ResultSet.IsEmpty() {
		t.Fatalf("ResultSet = %#v, want non-nil empty result set", result.ResultSet)
	}
	if !hasPipelineErrorTrace(result.BranchTrace, "planner") {
		t.Fatalf("BranchTrace = %#v, want planner error trace", result.BranchTrace)
	}
}

func TestExecutionPipelinePreservesBinderTraceOnError(t *testing.T) {
	t.Parallel()

	binderErr := errors.New("binder failed")
	pipeline, err := NewExecutionPipelineBuilder[intentWithMode, struct{}, executionMetaFixture]().
		WithPlanBinder(RequestPlanBinderFunc[intentWithMode, NoRequestMeta, executionMetaFixture](
			func(
				context.Context,
				Query[intentWithMode],
				*PlannedQuery[intentWithMode],
				executionMetaFixture,
			) (BoundRequest[intentWithMode, NoRequestMeta, executionMetaFixture], error) {
				return BoundRequest[intentWithMode, NoRequestMeta, executionMetaFixture]{
					Diagnostics: []ExecutionDiagnostic{{Stage: "binder", Key: "status", Value: "failed"}},
				}, binderErr
			},
		)).
		WithRoot(BackendNode[intentWithMode, struct{}, executionMetaFixture]{
			Backend: orchestratorStubBackend[intentWithMode, struct{}]{docs: nil},
		}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	result, err := pipeline.Execute(context.Background(), Query[intentWithMode]{
		Text:    "raw",
		Options: RetrieveOptions{TopK: 1},
	})
	if !errors.Is(err, binderErr) {
		t.Fatalf("Execute() error = %v, want binder error", err)
	}
	if len(result.Diagnostics) != 1 || result.Diagnostics[0].Stage != "binder" {
		t.Fatalf("Diagnostics = %#v, want binder diagnostics on error", result.Diagnostics)
	}
	if !hasPipelineErrorTrace(result.BranchTrace, "binder") {
		t.Fatalf("BranchTrace = %#v, want binder error trace", result.BranchTrace)
	}
}

func TestExecutionPipelinePreservesOptionsTraceOnError(t *testing.T) {
	t.Parallel()

	pipeline, err := NewExecutionPipelineBuilder[intentWithMode, struct{}, NoExecutionMeta]().
		WithRoot(BackendNode[intentWithMode, struct{}, NoExecutionMeta]{
			Backend: orchestratorStubBackend[intentWithMode, struct{}]{docs: nil},
		}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	result, err := pipeline.Execute(context.Background(), Query[intentWithMode]{
		Text:    "raw",
		Options: RetrieveOptions{TopK: -1},
	})
	if err == nil {
		t.Fatal("Execute() error = nil, want invalid options")
	}
	if !hasPipelineErrorTrace(result.BranchTrace, "options") {
		t.Fatalf("BranchTrace = %#v, want options error trace", result.BranchTrace)
	}
}

func TestExecutionFallbackRecordsSelectedAndSkippedTrace(t *testing.T) {
	t.Parallel()

	empty := BackendNode[intentWithMode, struct{}, NoExecutionMeta]{
		Backend: orchestratorStubBackend[intentWithMode, struct{}]{docs: nil},
	}
	hit := BackendNode[intentWithMode, struct{}, NoExecutionMeta]{
		Backend: orchestratorStubBackend[intentWithMode, struct{}]{
			docs: []Document[struct{}]{{ID: "hit", Content: "ok", Score: 1}},
		},
	}

	selected, err := NewExecutionPipelineBuilder[intentWithMode, struct{}, NoExecutionMeta]().
		WithRoot(FallbackNode[intentWithMode, struct{}, NoExecutionMeta]{
			Primary:   empty,
			Secondary: hit,
			Name:      "fallback",
		}).
		Build()
	if err != nil {
		t.Fatalf("Build(selected): %v", err)
	}
	selectedResult, err := selected.Execute(context.Background(), Query[intentWithMode]{
		Text:    "q",
		Options: RetrieveOptions{TopK: 1},
	})
	if err != nil {
		t.Fatalf("Execute(selected): %v", err)
	}
	if !hasBranch(selectedResult.BranchTrace, BranchKindFallback, "", BranchStateSelected) {
		t.Fatalf("BranchTrace(selected) = %#v, want selected fallback edge", selectedResult.BranchTrace)
	}

	skipped, err := NewExecutionPipelineBuilder[intentWithMode, struct{}, NoExecutionMeta]().
		WithRoot(FallbackNode[intentWithMode, struct{}, NoExecutionMeta]{
			Primary:   hit,
			Secondary: empty,
			Name:      "fallback",
		}).
		Build()
	if err != nil {
		t.Fatalf("Build(skipped): %v", err)
	}
	skippedResult, err := skipped.Execute(context.Background(), Query[intentWithMode]{
		Text:    "q",
		Options: RetrieveOptions{TopK: 1},
	})
	if err != nil {
		t.Fatalf("Execute(skipped): %v", err)
	}
	if !hasBranch(skippedResult.BranchTrace, BranchKindFallback, "", BranchStateSkipped) {
		t.Fatalf("BranchTrace(skipped) = %#v, want skipped fallback edge", skippedResult.BranchTrace)
	}
}

func TestExecutionRescueRecordsSelectedAndSkippedTrace(t *testing.T) {
	t.Parallel()

	failing := BackendNode[intentWithMode, struct{}, NoExecutionMeta]{
		Backend: orchestratorFailingBackend[intentWithMode, struct{}]{},
	}
	hit := BackendNode[intentWithMode, struct{}, NoExecutionMeta]{
		Backend: orchestratorStubBackend[intentWithMode, struct{}]{
			docs: []Document[struct{}]{{ID: "hit", Content: "ok", Score: 1}},
		},
	}

	selected, err := NewExecutionPipelineBuilder[intentWithMode, struct{}, NoExecutionMeta]().
		WithRoot(RescueNode[intentWithMode, struct{}, NoExecutionMeta]{
			Primary:   failing,
			Secondary: hit,
			Name:      "rescue",
		}).
		Build()
	if err != nil {
		t.Fatalf("Build(selected): %v", err)
	}
	selectedResult, err := selected.Execute(context.Background(), Query[intentWithMode]{
		Text:    "q",
		Options: RetrieveOptions{TopK: 1},
	})
	if err != nil {
		t.Fatalf("Execute(selected): %v", err)
	}
	if !hasBranch(selectedResult.BranchTrace, BranchKindRescue, "", BranchStateSelected) {
		t.Fatalf("BranchTrace(selected) = %#v, want selected rescue edge", selectedResult.BranchTrace)
	}

	skipped, err := NewExecutionPipelineBuilder[intentWithMode, struct{}, NoExecutionMeta]().
		WithRoot(RescueNode[intentWithMode, struct{}, NoExecutionMeta]{
			Primary:   hit,
			Secondary: failing,
			Name:      "rescue",
		}).
		Build()
	if err != nil {
		t.Fatalf("Build(skipped): %v", err)
	}
	skippedResult, err := skipped.Execute(context.Background(), Query[intentWithMode]{
		Text:    "q",
		Options: RetrieveOptions{TopK: 1},
	})
	if err != nil {
		t.Fatalf("Execute(skipped): %v", err)
	}
	if !hasBranch(skippedResult.BranchTrace, BranchKindRescue, "", BranchStateSkipped) {
		t.Fatalf("BranchTrace(skipped) = %#v, want skipped rescue edge", skippedResult.BranchTrace)
	}
}

func TestExecutionConditionalRecordsSelectedAndSkippedTrace(t *testing.T) {
	t.Parallel()

	child := BackendNode[intentWithMode, struct{}, NoExecutionMeta]{
		Backend: orchestratorStubBackend[intentWithMode, struct{}]{
			docs: []Document[struct{}]{{ID: "hit", Content: "ok", Score: 1}},
		},
	}
	pipeline, err := NewExecutionPipelineBuilder[intentWithMode, struct{}, NoExecutionMeta]().
		WithRoot(ConditionalNode[intentWithMode, struct{}, NoExecutionMeta]{
			Predicate: func(req Query[intentWithMode]) bool {
				return req.Intent.Mode == "run"
			},
			Child: child,
			Name:  "conditional",
		}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	selected, err := pipeline.Execute(context.Background(), Query[intentWithMode]{
		Text:    "q",
		Intent:  intentWithMode{Mode: "run"},
		Options: RetrieveOptions{TopK: 1},
	})
	if err != nil {
		t.Fatalf("Execute(selected): %v", err)
	}
	if !hasBranch(selected.BranchTrace, BranchKindNode, "", BranchStateSelected) {
		t.Fatalf("BranchTrace(selected) = %#v, want selected conditional node", selected.BranchTrace)
	}

	skipped, err := pipeline.Execute(context.Background(), Query[intentWithMode]{
		Text:    "q",
		Intent:  intentWithMode{Mode: "skip"},
		Options: RetrieveOptions{TopK: 1},
	})
	if err != nil {
		t.Fatalf("Execute(skipped): %v", err)
	}
	if !hasBranch(skipped.BranchTrace, BranchKindNode, "", BranchStateSkipped) {
		t.Fatalf("BranchTrace(skipped) = %#v, want skipped conditional node", skipped.BranchTrace)
	}
}

type sideOutputBackend struct{}

func (sideOutputBackend) Retrieve(
	_ context.Context,
	_ Query[intentWithMode],
	exec executionMetaFixture,
) (RetrievalResult[struct{}, executionMetaFixture], error) {
	exec.SideOutputs = append(exec.SideOutputs, "web-hit")
	return RetrievalResult[struct{}, executionMetaFixture]{
		ResultSet: NewResultSet(
			[]Document[struct{}]{{ID: "hit", Content: "ok", Score: 1}},
			DocumentIDResolver[struct{}]{},
		),
		Executed: exec,
		Diagnostics: []ExecutionDiagnostic{{
			Stage: "backend",
			Key:   "side-output",
			Value: "emitted",
		}},
	}, nil
}

type namedSideOutputBackend struct {
	id string
}

func (b namedSideOutputBackend) Retrieve(
	_ context.Context,
	_ Query[intentWithMode],
	exec executionMetaFixture,
) (RetrievalResult[struct{}, executionMetaFixture], error) {
	exec.SideOutputs = append(exec.SideOutputs, b.id)
	return RetrievalResult[struct{}, executionMetaFixture]{
		ResultSet: NewResultSet(
			[]Document[struct{}]{{ID: b.id, Content: b.id, Score: 1}},
			DocumentIDResolver[struct{}]{},
		),
		Executed: exec,
		Diagnostics: []ExecutionDiagnostic{{
			Stage: "backend",
			Key:   "id",
			Value: b.id,
		}},
		BranchTrace: nil,
	}, nil
}

type omittedExecutionBackend struct{}

func (omittedExecutionBackend) Retrieve(
	_ context.Context,
	_ Query[intentWithMode],
	_ executionMetaFixture,
) (RetrievalResult[struct{}, executionMetaFixture], error) {
	return RetrievalResult[struct{}, executionMetaFixture]{
		ResultSet: NewResultSet(
			[]Document[struct{}]{{ID: "hit", Content: "ok", Score: 1}},
			DocumentIDResolver[struct{}]{},
		),
	}, nil
}

func TestExecutionBackendReturnsTypedSideOutputs(t *testing.T) {
	t.Parallel()

	pipeline, err := NewExecutionPipelineBuilder[intentWithMode, struct{}, executionMetaFixture]().
		WithRoot(RequestExecutionRetrieverNode[intentWithMode, NoRequestMeta, struct{}, executionMetaFixture]{
			Backend: sideOutputBackend{},
		}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	result, err := pipeline.Execute(context.Background(), Query[intentWithMode]{
		Text:    "q",
		Options: RetrieveOptions{TopK: 1},
	})
	if err != nil {
		t.Fatalf("Execute(): %v", err)
	}
	if result.ResultSet.Len() != 1 {
		t.Fatalf("Len() = %d, want 1", result.ResultSet.Len())
	}
	if len(result.Executed.SideOutputs) != 1 || result.Executed.SideOutputs[0] != "web-hit" {
		t.Fatalf("SideOutputs = %#v, want typed backend output", result.Executed.SideOutputs)
	}
}

func TestExecutionRetrieverPreservesIncomingExecutionWhenBackendOmitsIt(t *testing.T) {
	t.Parallel()

	pipeline, err := NewExecutionPipelineBuilder[intentWithMode, struct{}, executionMetaFixture]().
		WithExecutionSeed(func(Query[intentWithMode]) executionMetaFixture {
			return executionMetaFixture{Bound: true, SideOutputs: []string{"seed"}}
		}).
		WithRoot(RequestExecutionRetrieverNode[intentWithMode, NoRequestMeta, struct{}, executionMetaFixture]{
			Backend: omittedExecutionBackend{},
		}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	result, err := pipeline.Execute(context.Background(), Query[intentWithMode]{
		Text:    "q",
		Options: RetrieveOptions{TopK: 1},
	})
	if err != nil {
		t.Fatalf("Execute(): %v", err)
	}
	if !result.Executed.Bound || len(result.Executed.SideOutputs) != 1 || result.Executed.SideOutputs[0] != "seed" {
		t.Fatalf("Executed = %#v, want incoming execution metadata preserved", result.Executed)
	}
}

func TestExecutionAggregatePreservesTraceDiagnosticsAndSideOutputs(t *testing.T) {
	t.Parallel()

	pipeline, err := NewExecutionPipelineBuilder[intentWithMode, struct{}, executionMetaFixture]().
		WithAggregate(
			[]ExecutionNode[intentWithMode, struct{}, executionMetaFixture]{
				RequestExecutionRetrieverNode[intentWithMode, NoRequestMeta, struct{}, executionMetaFixture]{
					Backend:  namedSideOutputBackend{id: "left"},
					Resolver: nil,
					Name:     "left",
				},
				RequestExecutionRetrieverNode[intentWithMode, NoRequestMeta, struct{}, executionMetaFixture]{
					Backend:  namedSideOutputBackend{id: "right"},
					Resolver: nil,
					Name:     "right",
				},
			},
			2,
			nil,
			func(current executionMetaFixture, child executionMetaFixture) executionMetaFixture {
				current.SideOutputs = append(current.SideOutputs, child.SideOutputs...)
				return current
			},
		).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	result, err := pipeline.Execute(context.Background(), Query[intentWithMode]{
		Text:    "q",
		Options: RetrieveOptions{TopK: 10},
	})
	if err != nil {
		t.Fatalf("Execute(): %v", err)
	}
	if result.Len() != 2 {
		t.Fatalf("Len() = %d, want merged aggregate docs", result.Len())
	}
	if got := result.Executed.SideOutputs; len(got) != 2 || got[0] != "left" || got[1] != "right" {
		t.Fatalf("SideOutputs = %#v, want both child outputs", got)
	}
	if len(result.Diagnostics) != 2 {
		t.Fatalf("Diagnostics = %#v, want child diagnostics", result.Diagnostics)
	}
	if !hasBranch(result.BranchTrace, BranchKindNode, "", BranchStateSelected) {
		t.Fatalf("BranchTrace = %#v, want aggregate branch trace", result.BranchTrace)
	}
}

func hasPipelineErrorTrace(trace []BranchStep, node string) bool {
	for _, step := range trace {
		if step.Kind == BranchKindNode && step.Node == node && step.State == BranchStateErrored {
			return true
		}
	}
	return false
}
