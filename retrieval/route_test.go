package retrieval

import (
	"context"
	"errors"
	"testing"

	ragy "github.com/skosovsky/ragy"
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

func TestRouteNodeRunsChildForTypedRouteDecision(t *testing.T) {
	t.Parallel()

	node := RouteNode[routeIntent, routeName, routeSignal, struct{}]{
		Planner: RoutePlannerFunc[routeIntent, routeName, routeSignal](
			func(_ context.Context, query Query[routeIntent]) (RouteDecision[routeName, routeSignal], error) {
				return RouteDecision[routeName, routeSignal]{
					Route:  routeName(query.Intent.Mode),
					Signal: routeSignal{Confidence: 0.9},
				}, nil
			},
		),
		Predicate: func(_ Query[routeIntent], decision RouteDecision[routeName, routeSignal]) bool {
			return decision.Route == "run" && decision.Signal.Confidence > 0.5
		},
		Child: routeStubNode[struct{}]{docs: []Document[struct{}]{{ID: "hit", Content: "ok", Score: 1}}},
	}

	rs, err := node.Retrieve(context.Background(), Query[routeIntent]{
		Text:    "q",
		Intent:  routeIntent{Mode: "run"},
		Options: RetrieveOptions{TopK: 1},
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Len() != 1 || rs.Documents()[0].ID != "hit" {
		t.Fatalf("Documents() = %#v, want route hit", rs.Documents())
	}
}

func TestRouteNodeSkipsChildForPredicateFalse(t *testing.T) {
	t.Parallel()

	node := RouteNode[routeIntent, routeName, routeSignal, struct{}]{
		Planner: RoutePlannerFunc[routeIntent, routeName, routeSignal](
			func(context.Context, Query[routeIntent]) (RouteDecision[routeName, routeSignal], error) {
				return RouteDecision[routeName, routeSignal]{Route: "skip"}, nil
			},
		),
		Predicate: func(Query[routeIntent], RouteDecision[routeName, routeSignal]) bool {
			return false
		},
		Child: routeStubNode[struct{}]{docs: []Document[struct{}]{{ID: "hit", Content: "ok", Score: 1}}},
	}

	rs, err := node.Retrieve(context.Background(), Query[routeIntent]{
		Text:    "q",
		Options: RetrieveOptions{TopK: 1},
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if !rs.IsEmpty() {
		t.Fatalf("Documents() = %#v, want route skip", rs.Documents())
	}
}

func TestPipelineBuilderValidatesRouteNode(t *testing.T) {
	t.Parallel()

	_, err := NewPipelineBuilder[routeIntent, struct{}]().
		WithRoot(RouteNode[routeIntent, routeName, routeSignal, struct{}]{
			Child: routeStubNode[struct{}]{},
		}).
		Build()
	if !errors.Is(err, ragy.ErrInvalidArgument) {
		t.Fatalf("Build() error = %v, want invalid argument", err)
	}
}

func TestPipelineBuilderInjectsResolverThroughRouteNode(t *testing.T) {
	t.Parallel()

	resolver := mergeKeyResolver[struct{}]{key: func(doc Document[struct{}]) string { return doc.Content }}
	pipeline, err := NewPipelineBuilder[routeIntent, struct{}]().
		WithRoot(RouteNode[routeIntent, routeName, routeSignal, struct{}]{
			Planner: RoutePlannerFunc[routeIntent, routeName, routeSignal](
				func(context.Context, Query[routeIntent]) (RouteDecision[routeName, routeSignal], error) {
					return RouteDecision[routeName, routeSignal]{Route: "run"}, nil
				},
			),
			Child: AggregateNode[routeIntent, struct{}]{
				Nodes: []Node[routeIntent, struct{}]{
					routeStubNode[struct{}]{docs: []Document[struct{}]{{ID: "a", Content: "same", Score: 0.9}}},
					routeStubNode[struct{}]{docs: []Document[struct{}]{{ID: "b", Content: "same", Score: 0.1}}},
				},
			},
		}).
		WithResolver(resolver).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Retrieve(context.Background(), Query[routeIntent]{
		Text:    "q",
		Options: RetrieveOptions{TopK: 10},
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Len() != 1 {
		t.Fatalf("Len() = %d, want 1 document after route child resolver injection", rs.Len())
	}
}

func TestPipelinePlannerAttachesPlanBeforeRoutePlannerAndPredicate(t *testing.T) {
	t.Parallel()

	var plannerSawPlan bool
	var predicateSawPlan bool
	pipeline, err := NewPipelineBuilder[routeIntent, struct{}]().
		WithPlanner(StaticPlanner[routeIntent, NoRequestMeta]{
			Planned: PlannedQuery[routeIntent]{ExpandedText: "planned"},
		}).
		WithRoot(RouteNode[routeIntent, routeName, routeSignal, struct{}]{
			Planner: RoutePlannerFunc[routeIntent, routeName, routeSignal](
				func(_ context.Context, req Query[routeIntent]) (RouteDecision[routeName, routeSignal], error) {
					plannerSawPlan = req.Plan != nil && req.EffectiveText() == "planned"
					return RouteDecision[routeName, routeSignal]{Route: "run"}, nil
				},
			),
			Predicate: func(req Query[routeIntent], _ RouteDecision[routeName, routeSignal]) bool {
				predicateSawPlan = req.Plan != nil && req.EffectiveText() == "planned"
				return true
			},
			Child: routeStubNode[struct{}]{docs: []Document[struct{}]{{ID: "hit", Content: "ok", Score: 1}}},
		}).
		Build()
	if err != nil {
		t.Fatalf("Build(): %v", err)
	}

	rs, err := pipeline.Retrieve(context.Background(), Query[routeIntent]{
		Text:    "raw",
		Options: RetrieveOptions{TopK: 1},
	})
	if err != nil {
		t.Fatalf("Retrieve(): %v", err)
	}
	if rs.Len() != 1 {
		t.Fatalf("Len() = %d, want routed hit", rs.Len())
	}
	if !plannerSawPlan || !predicateSawPlan {
		t.Fatalf("route saw plan = planner:%v predicate:%v, want both true", plannerSawPlan, predicateSawPlan)
	}
}
