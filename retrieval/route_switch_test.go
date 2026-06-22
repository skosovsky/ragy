package retrieval

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
)

type routeExecMeta struct {
	Route      routeName
	SignalSeen bool
}

func routeCaseNode(
	node resultNode[routeIntent, NoRequestMeta, struct{}],
) RequestExecutionNode[routeIntent, NoRequestMeta, struct{}, routeExecMeta] {
	return requestNodeExecutionAdapter[routeIntent, NoRequestMeta, struct{}, routeExecMeta]{
		Node:     node,
		Resolver: nil,
		Name:     "",
	}
}

func TestRouteSwitchDispatchesOnceAndFallsBackOnEmpty(t *testing.T) {
	t.Parallel()

	routeCalls := 0
	switchNode, err := NewRouteSwitchBuilder[routeIntent, routeName, routeSignal, struct{}, routeExecMeta](
		RoutePlannerFunc[routeIntent, routeName, routeSignal](
			func(_ context.Context, req Query[routeIntent]) (RouteDecision[routeName, routeSignal], error) {
				routeCalls++
				if req.Plan == nil || req.EffectiveText() != "planned" {
					t.Fatalf("route planner request plan = %#v, want planned text", req.Plan)
				}
				return RouteDecision[routeName, routeSignal]{
					Route:       "catalog",
					Signal:      routeSignal{Confidence: 0.8},
					Diagnostics: []PlannerDiagnostic{{Key: "route", Value: "catalog"}},
				}, nil
			},
		),
	).
		Name("search-route").
		RecordDecision(func(exec routeExecMeta, decision RouteDecision[routeName, routeSignal]) routeExecMeta {
			exec.Route = decision.Route
			exec.SignalSeen = decision.Signal.Confidence > 0.5
			return exec
		}).
		Case("catalog", routeCaseNode(routeStubNode[struct{}]{docs: nil})).
		Case("knowledge", routeCaseNode(routeStubNode[struct{}]{
			docs: []Document[struct{}]{{ID: "kb", Content: "hit", Score: 1}},
		})).
		FallbackOnEmpty("catalog", "knowledge").
		Build()
	if err != nil {
		t.Fatalf("Build route switch: %v", err)
	}

	pipeline, err := NewExecutionPipelineBuilder[routeIntent, struct{}, routeExecMeta]().
		WithPlanner(StaticPlanner[routeIntent, NoRequestMeta]{
			Planned: PlannedQuery[routeIntent]{
				ExpandedText: "planned",
				Diagnostics:  []PlannerDiagnostic{{Key: "planner", Value: "expanded"}},
			},
		}).
		WithRoot(switchNode).
		Build()
	if err != nil {
		t.Fatalf("Build pipeline: %v", err)
	}

	result, err := pipeline.Execute(context.Background(), Query[routeIntent]{
		Text:    "raw",
		Options: RetrieveOptions{TopK: 5},
	})
	if err != nil {
		t.Fatalf("Execute(): %v", err)
	}
	if routeCalls != 1 {
		t.Fatalf("route calls = %d, want 1", routeCalls)
	}
	if result.ResultSet.Len() != 1 || result.ResultSet.Documents()[0].ID != "kb" {
		t.Fatalf("Documents() = %#v, want fallback knowledge doc", result.ResultSet.Documents())
	}
	if result.Executed.Route != "catalog" || !result.Executed.SignalSeen {
		t.Fatalf("Executed = %#v, want typed route decision", result.Executed)
	}
	if !hasBranch(result.BranchTrace, BranchKindFallback, "knowledge", BranchStateSelected) {
		t.Fatalf("BranchTrace = %#v, want selected fallback to knowledge", result.BranchTrace)
	}
	if len(result.Diagnostics) != 2 {
		t.Fatalf("Diagnostics = %#v, want planner and route diagnostics", result.Diagnostics)
	}
}

func TestRouteSwitchConditionalFallbackCanSkip(t *testing.T) {
	t.Parallel()

	switchNode, err := NewRouteSwitchBuilder[routeIntent, routeName, routeSignal, struct{}, routeExecMeta](
		RoutePlannerFunc[routeIntent, routeName, routeSignal](
			func(context.Context, Query[routeIntent]) (RouteDecision[routeName, routeSignal], error) {
				return RouteDecision[routeName, routeSignal]{Route: "primary"}, nil
			},
		),
	).
		Case("primary", routeCaseNode(routeStubNode[struct{}]{docs: nil})).
		Case("fallback", routeCaseNode(routeStubNode[struct{}]{
			docs: []Document[struct{}]{{ID: "fb", Content: "hit", Score: 1}},
		})).
		ConditionalFallback("primary", "fallback",
			func(ctx RequestRouteExecutionContext[routeIntent, NoRequestMeta, routeName, routeSignal, struct{}, routeExecMeta]) bool {
				return ctx.Request.Intent.Mode == "allow"
			},
		).
		Build()
	if err != nil {
		t.Fatalf("Build route switch: %v", err)
	}
	pipeline, err := NewExecutionPipelineBuilder[routeIntent, struct{}, routeExecMeta]().
		WithRoot(switchNode).
		Build()
	if err != nil {
		t.Fatalf("Build pipeline: %v", err)
	}

	skipped, err := pipeline.Execute(context.Background(), Query[routeIntent]{
		Text:    "q",
		Intent:  routeIntent{Mode: "deny"},
		Options: RetrieveOptions{TopK: 5},
	})
	if err != nil {
		t.Fatalf("Execute(skip): %v", err)
	}
	if !skipped.ResultSet.IsEmpty() {
		t.Fatalf("Documents(skip) = %#v, want empty", skipped.ResultSet.Documents())
	}
	if !hasBranch(skipped.BranchTrace, BranchKindFallback, "fallback", BranchStateSkipped) {
		t.Fatalf("BranchTrace(skip) = %#v, want skipped fallback edge", skipped.BranchTrace)
	}

	allowed, err := pipeline.Execute(context.Background(), Query[routeIntent]{
		Text:    "q",
		Intent:  routeIntent{Mode: "allow"},
		Options: RetrieveOptions{TopK: 5},
	})
	if err != nil {
		t.Fatalf("Execute(allow): %v", err)
	}
	if allowed.ResultSet.Len() != 1 || allowed.ResultSet.Documents()[0].ID != "fb" {
		t.Fatalf("Documents(allow) = %#v, want fallback doc", allowed.ResultSet.Documents())
	}
}

func TestRouteSwitchRecordsNoDefaultSkip(t *testing.T) {
	t.Parallel()

	switchNode, err := NewRouteSwitchBuilder[routeIntent, routeName, routeSignal, struct{}, routeExecMeta](
		RoutePlannerFunc[routeIntent, routeName, routeSignal](
			func(context.Context, Query[routeIntent]) (RouteDecision[routeName, routeSignal], error) {
				return RouteDecision[routeName, routeSignal]{Route: "missing"}, nil
			},
		),
	).
		Case("known", routeCaseNode(routeStubNode[struct{}]{
			docs: []Document[struct{}]{{ID: "hit", Content: "ok", Score: 1}},
		})).
		Build()
	if err != nil {
		t.Fatalf("Build route switch: %v", err)
	}
	pipeline, err := NewExecutionPipelineBuilder[routeIntent, struct{}, routeExecMeta]().
		WithRoot(switchNode).
		Build()
	if err != nil {
		t.Fatalf("Build pipeline: %v", err)
	}

	result, err := pipeline.Execute(context.Background(), Query[routeIntent]{
		Text:    "q",
		Options: RetrieveOptions{TopK: 5},
	})
	if err != nil {
		t.Fatalf("Execute(): %v", err)
	}
	if !result.IsEmpty() {
		t.Fatalf("Documents() = %#v, want empty", result.Documents())
	}
	if !hasBranch(result.BranchTrace, BranchKindCase, "missing", BranchStateSkipped) {
		t.Fatalf("BranchTrace = %#v, want skipped missing route", result.BranchTrace)
	}
}

func TestRouteSwitchRescuesBranchError(t *testing.T) {
	t.Parallel()

	switchNode, err := NewRouteSwitchBuilder[routeIntent, routeName, routeSignal, struct{}, routeExecMeta](
		RoutePlannerFunc[routeIntent, routeName, routeSignal](
			func(context.Context, Query[routeIntent]) (RouteDecision[routeName, routeSignal], error) {
				return RouteDecision[routeName, routeSignal]{Route: "primary"}, nil
			},
		),
	).
		Case("primary", routeCaseNode(errorNode[routeIntent, struct{}]{err: ragy.ErrUnavailable})).
		Case("rescue", routeCaseNode(routeStubNode[struct{}]{
			docs: []Document[struct{}]{{ID: "rescued", Content: "ok", Score: 1}},
		})).
		RescueOnError("primary", "rescue",
			func(ctx RequestRouteExecutionContext[routeIntent, NoRequestMeta, routeName, routeSignal, struct{}, routeExecMeta]) bool {
				return errors.Is(ctx.Err, ragy.ErrUnavailable) && ctx.Result.IsEmpty()
			},
		).
		Build()
	if err != nil {
		t.Fatalf("Build route switch: %v", err)
	}
	pipeline, err := NewExecutionPipelineBuilder[routeIntent, struct{}, routeExecMeta]().
		WithRoot(switchNode).
		Build()
	if err != nil {
		t.Fatalf("Build pipeline: %v", err)
	}

	result, err := pipeline.Execute(context.Background(), Query[routeIntent]{
		Text:    "q",
		Options: RetrieveOptions{TopK: 5},
	})
	if err != nil {
		t.Fatalf("Execute(): %v", err)
	}
	if result.ResultSet.Len() != 1 || result.ResultSet.Documents()[0].ID != "rescued" {
		t.Fatalf("Documents() = %#v, want rescue doc", result.ResultSet.Documents())
	}
	if !hasBranch(result.BranchTrace, BranchKindRescue, "rescue", BranchStateSelected) {
		t.Fatalf("BranchTrace = %#v, want selected rescue branch", result.BranchTrace)
	}
}

func TestRouteSwitchRecordsSkippedRescuePredicateAndCaseError(t *testing.T) {
	t.Parallel()

	switchNode, err := NewRouteSwitchBuilder[routeIntent, routeName, routeSignal, struct{}, routeExecMeta](
		RoutePlannerFunc[routeIntent, routeName, routeSignal](
			func(context.Context, Query[routeIntent]) (RouteDecision[routeName, routeSignal], error) {
				return RouteDecision[routeName, routeSignal]{Route: "primary"}, nil
			},
		),
	).
		Case("primary", routeCaseNode(errorNode[routeIntent, struct{}]{err: ragy.ErrUnavailable})).
		Case("rescue", routeCaseNode(routeStubNode[struct{}]{
			docs: []Document[struct{}]{{ID: "rescued", Content: "ok", Score: 1}},
		})).
		RescueOnError("primary", "rescue",
			func(RequestRouteExecutionContext[routeIntent, NoRequestMeta, routeName, routeSignal, struct{}, routeExecMeta]) bool {
				return false
			},
		).
		Build()
	if err != nil {
		t.Fatalf("Build route switch: %v", err)
	}
	pipeline, err := NewExecutionPipelineBuilder[routeIntent, struct{}, routeExecMeta]().
		WithRoot(switchNode).
		Build()
	if err != nil {
		t.Fatalf("Build pipeline: %v", err)
	}

	result, err := pipeline.Execute(context.Background(), Query[routeIntent]{
		Text:    "q",
		Options: RetrieveOptions{TopK: 5},
	})
	if !errors.Is(err, ragy.ErrUnavailable) {
		t.Fatalf("Execute() error = %v, want unavailable", err)
	}
	if !hasBranch(result.BranchTrace, BranchKindRescue, "rescue", BranchStateSkipped) {
		t.Fatalf("BranchTrace = %#v, want skipped rescue edge", result.BranchTrace)
	}
	if !hasErroredCase(result.BranchTrace, "primary") {
		t.Fatalf("BranchTrace = %#v, want primary case error text", result.BranchTrace)
	}
}

func TestRouteSwitchRejectsDuplicateCases(t *testing.T) {
	t.Parallel()

	_, err := NewRouteSwitchBuilder[routeIntent, routeName, routeSignal, struct{}, routeExecMeta](
		RoutePlannerFunc[routeIntent, routeName, routeSignal](
			func(context.Context, Query[routeIntent]) (RouteDecision[routeName, routeSignal], error) {
				return RouteDecision[routeName, routeSignal]{Route: "primary"}, nil
			},
		),
	).
		Case("primary", routeCaseNode(routeStubNode[struct{}]{docs: nil})).
		ExecutionCase("primary", routeCaseNode(routeStubNode[struct{}]{docs: nil})).
		Build()
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Build() error = %v, want invalid argument for duplicate route", err)
	}
}

func TestRouteSwitchRejectsFallbackCycle(t *testing.T) {
	t.Parallel()

	_, err := NewRouteSwitchBuilder[routeIntent, routeName, routeSignal, struct{}, routeExecMeta](
		RoutePlannerFunc[routeIntent, routeName, routeSignal](
			func(context.Context, Query[routeIntent]) (RouteDecision[routeName, routeSignal], error) {
				return RouteDecision[routeName, routeSignal]{Route: "a"}, nil
			},
		),
	).
		Case("a", routeCaseNode(routeStubNode[struct{}]{docs: nil})).
		Case("b", routeCaseNode(routeStubNode[struct{}]{docs: nil})).
		FallbackOnEmpty("a", "b").
		FallbackOnEmpty("b", "a").
		Build()
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Build() error = %v, want invalid argument for fallback cycle", err)
	}
}

func TestRouteSwitchRejectsRescueCycle(t *testing.T) {
	t.Parallel()

	_, err := NewRouteSwitchBuilder[routeIntent, routeName, routeSignal, struct{}, routeExecMeta](
		RoutePlannerFunc[routeIntent, routeName, routeSignal](
			func(context.Context, Query[routeIntent]) (RouteDecision[routeName, routeSignal], error) {
				return RouteDecision[routeName, routeSignal]{Route: "a"}, nil
			},
		),
	).
		Case("a", routeCaseNode(routeStubNode[struct{}]{docs: nil})).
		Case("b", routeCaseNode(routeStubNode[struct{}]{docs: nil})).
		RescueOnError("a", "b", nil).
		RescueOnError("b", "a", nil).
		Build()
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Build() error = %v, want invalid argument for rescue cycle", err)
	}
}

func TestRouteSwitchPreservesDiagnosticsWhenPlannerErrors(t *testing.T) {
	t.Parallel()

	routeErr := errors.New("route planner failed")
	switchNode, err := NewRouteSwitchBuilder[routeIntent, routeName, routeSignal, struct{}, routeExecMeta](
		RoutePlannerFunc[routeIntent, routeName, routeSignal](
			func(context.Context, Query[routeIntent]) (RouteDecision[routeName, routeSignal], error) {
				return RouteDecision[routeName, routeSignal]{
					Route:       "primary",
					Diagnostics: []PlannerDiagnostic{{Key: "route", Value: "primary"}},
				}, routeErr
			},
		),
	).
		RecordDecision(func(exec routeExecMeta, decision RouteDecision[routeName, routeSignal]) routeExecMeta {
			exec.Route = decision.Route
			return exec
		}).
		Case("primary", routeCaseNode(routeStubNode[struct{}]{docs: nil})).
		Build()
	if err != nil {
		t.Fatalf("Build route switch: %v", err)
	}
	pipeline, err := NewExecutionPipelineBuilder[routeIntent, struct{}, routeExecMeta]().
		WithRoot(switchNode).
		Build()
	if err != nil {
		t.Fatalf("Build pipeline: %v", err)
	}

	result, err := pipeline.Execute(context.Background(), Query[routeIntent]{
		Text:    "q",
		Options: RetrieveOptions{TopK: 5},
	})
	if !errors.Is(err, routeErr) {
		t.Fatalf("Execute() error = %v, want route planner error", err)
	}
	if result.Executed.Route != "primary" {
		t.Fatalf("Executed = %#v, want route decision recorded", result.Executed)
	}
	if len(result.Diagnostics) != 1 || result.Diagnostics[0].Stage != "route" {
		t.Fatalf("Diagnostics = %#v, want route diagnostics on error", result.Diagnostics)
	}
	if !hasBranch(result.BranchTrace, BranchKindRoute, "primary", BranchStateErrored) {
		t.Fatalf("BranchTrace = %#v, want errored route step", result.BranchTrace)
	}
}

func hasBranch(trace []BranchStep, kind, route, state string) bool {
	for _, step := range trace {
		if step.Kind == kind && step.Route == route && step.State == state {
			return true
		}
	}
	return false
}

func hasErroredCase(trace []BranchStep, route string) bool {
	for _, step := range trace {
		if step.Kind == BranchKindCase && step.Route == route && step.State == BranchStateErrored && step.Error != "" {
			return true
		}
	}
	return false
}
