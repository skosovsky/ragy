package retrieval

import (
	"context"
	"testing"

	"github.com/skosovsky/ragy/observation"
)

func TestRouteSwitchObservationTracksActualFallbackAndSkippedEdge(t *testing.T) {
	t.Parallel()
	for _, allow := range []bool{false, true} {
		t.Run(map[bool]string{false: "skipped", true: "selected"}[allow], func(t *testing.T) {
			t.Parallel()
			// Arrange.
			var events []observation.Event
			session, err := observation.New(
				observation.Config{
					MaxEvents: 100,
					Observer: observation.ObserverFunc(
						func(_ context.Context, e observation.Event) error { events = append(events, e); return nil },
					),
				},
			)
			if err != nil {
				t.Fatal(err)
			}
			node, err := NewRouteSwitchBuilder[routeIntent, routeName, routeSignal, struct{}, routeExecMeta](
				RoutePlannerFunc[routeIntent, routeName, routeSignal](
					func(context.Context, Query[routeIntent]) (RouteDecision[routeName, routeSignal], error) {
						return RouteDecision[routeName, routeSignal]{Route: "private-primary"}, nil
					},
				),
			).Case("private-primary", routeCaseNode(routeStubNode[struct{}]{})).
				Case("private-second", routeCaseNode(routeStubNode[struct{}]{docs: []Document[struct{}]{{ID: "private-id", Content: "private-content", ScoreState: ScorePresent, ScoreSemantics: "fixture-similarity", Score: 1}}})).
				ConditionalFallback("private-primary", "private-second", func(RequestRouteExecutionContext[routeIntent, NoRequestMeta, routeName, routeSignal, struct{}, routeExecMeta]) bool {
					return allow
				}).
				Build()
			if err != nil {
				t.Fatal(err)
			}
			// Act.
			result, err := node.Execute(
				observation.WithSession(context.Background(), session),
				Query[routeIntent]{Read: UnrestrictedRead()},
				routeExecMeta{},
			)
			// Assert.
			if err != nil {
				t.Fatal(err)
			}
			if result.ResultSet.Len() != map[bool]int{false: 0, true: 1}[allow] {
				t.Fatalf("result len=%d", result.ResultSet.Len())
			}
			assertObservedRouteEvents(t, events, allow)
		})
	}
}

func assertObservedRouteEvents(t *testing.T, events []observation.Event, allow bool) {
	t.Helper()
	children := make(map[uint64]bool)
	sawPlan, sawRoute, sawDecision := false, false, false
	for _, event := range events {
		if event.Kind != observation.KindEnd {
			continue
		}
		//nolint:exhaustive // Only stages emitted by route switch are asserted here.
		switch event.Stage {
		case observation.StagePlan:
			sawPlan = true
		case observation.StageRoute:
			sawRoute = true
		case observation.StageRetrieval:
			if event.Branch.Known {
				children[event.Branch.Value] = true
			}
		case observation.StageFallback:
			sawDecision = true
			want := observation.OutcomeSkipped
			if allow {
				want = observation.OutcomeSuccess
			}
			if event.Completion.Outcome != want || !event.Branch.Known || event.Branch.Value != 2 {
				t.Fatalf("fallback event=%#v", event)
			}
		}
	}
	if !sawPlan || !sawRoute || !sawDecision || !children[1] || children[2] != allow {
		t.Fatalf("events=%#v", events)
	}
}

func TestRouteSwitchObservationDefaultBranchOrdinal(t *testing.T) {
	t.Parallel()
	// Arrange.
	var events []observation.Event
	session, err := observation.New(
		observation.Config{
			MaxEvents: 100,
			Observer: observation.ObserverFunc(
				func(_ context.Context, event observation.Event) error { events = append(events, event); return nil },
			),
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	node, err := NewRouteSwitchBuilder[routeIntent, routeName, routeSignal, struct{}, routeExecMeta](
		RoutePlannerFunc[routeIntent, routeName, routeSignal](
			func(context.Context, Query[routeIntent]) (RouteDecision[routeName, routeSignal], error) {
				return RouteDecision[routeName, routeSignal]{Route: "unknown-private-route"}, nil
			},
		),
	).Case("first-private-route", routeCaseNode(routeStubNode[struct{}]{})).
		Default(routeCaseNode(routeStubNode[struct{}]{})).
		Build()
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	_, err = node.Execute(
		observation.WithSession(context.Background(), session),
		Query[routeIntent]{Read: UnrestrictedRead()},
		routeExecMeta{},
	)
	// Assert.
	if err != nil {
		t.Fatal(err)
	}
	childObserved := false
	for _, event := range events {
		if event.Stage == observation.StageRetrieval && event.Kind == observation.KindStart {
			childObserved = true
			if !event.Branch.Known || event.Branch.Value != 2 {
				t.Fatalf("default branch=%#v", event.Branch)
			}
		}
	}
	if !childObserved {
		t.Fatal("default child event missing")
	}
}
