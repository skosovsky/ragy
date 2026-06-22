package retrieval

import (
	"context"
	"testing"
)

type routeIntent struct {
	Mode string
}

type routeName string

type routeSignal struct {
	Confidence float64
}

type routeStubNode[TMeta any] struct {
	docs []Document[TMeta]
}

func (n routeStubNode[TMeta]) Retrieve(_ context.Context, _ Query[routeIntent]) (ResultSet[TMeta], error) {
	return NewResultSet(n.docs, DocumentIDResolver[TMeta]{}), nil
}

func TestRoutePlannerFuncReturnsTypedDecision(t *testing.T) {
	t.Parallel()

	planner := RoutePlannerFunc[routeIntent, routeName, routeSignal](
		func(_ context.Context, query Query[routeIntent]) (RouteDecision[routeName, routeSignal], error) {
			return RouteDecision[routeName, routeSignal]{
				Route:       routeName(query.Intent.Mode),
				Signal:      routeSignal{Confidence: 0.9},
				Diagnostics: []PlannerDiagnostic{{Key: "mode", Value: query.Intent.Mode}},
			}, nil
		},
	)

	decision, err := planner.PlanRoute(context.Background(), Query[routeIntent]{
		Intent: routeIntent{Mode: "run"},
	})
	if err != nil {
		t.Fatalf("PlanRoute(): %v", err)
	}
	if decision.Route != "run" || decision.Signal.Confidence != 0.9 {
		t.Fatalf("decision = %#v, want typed route decision", decision)
	}
	if len(decision.Diagnostics) != 1 || decision.Diagnostics[0].Value != "run" {
		t.Fatalf("diagnostics = %#v, want mode diagnostic", decision.Diagnostics)
	}
}

func TestPipelineBuilderInjectsResolverThroughRouteSwitch(t *testing.T) {
	t.Parallel()

	resolver := mergeKeyResolver[struct{}]{key: func(doc Document[struct{}]) string { return doc.Content }}
	routeSwitch, err := NewRouteSwitchBuilder[routeIntent, routeName, routeSignal, struct{}, routeExecMeta](
		RoutePlannerFunc[routeIntent, routeName, routeSignal](
			func(context.Context, Query[routeIntent]) (RouteDecision[routeName, routeSignal], error) {
				return RouteDecision[routeName, routeSignal]{Route: "run"}, nil
			},
		),
	).
		Case("run", routeCaseNode(resultAggregateNodeNoMeta[routeIntent, struct{}]{
			Nodes: []resultNodeNoMeta[routeIntent, struct{}]{
				routeStubNode[struct{}]{docs: []Document[struct{}]{{ID: "a", Content: "same", Score: 0.9}}},
				routeStubNode[struct{}]{docs: []Document[struct{}]{{ID: "b", Content: "same", Score: 0.1}}},
			},
		})).
		Build()
	if err != nil {
		t.Fatalf("Build route switch: %v", err)
	}
	pipeline, err := NewExecutionPipelineBuilder[routeIntent, struct{}, routeExecMeta]().
		WithRoot(routeSwitch).
		WithResolver(resolver).
		Build()
	if err != nil {
		t.Fatalf("Build pipeline: %v", err)
	}

	result, err := pipeline.Execute(context.Background(), Query[routeIntent]{
		Text:    "q",
		Options: RetrieveOptions{TopK: 10},
	})
	if err != nil {
		t.Fatalf("Execute(): %v", err)
	}
	if result.Len() != 1 {
		t.Fatalf("Len() = %d, want 1 document after route child resolver injection", result.Len())
	}
}
