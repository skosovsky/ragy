package retrieval

import (
	"context"
)

// RouteDecision is the typed output of route planning.
type RouteDecision[TRoute, TSignal any] struct {
	Route       TRoute
	Signal      TSignal
	Diagnostics []PlannerDiagnostic
}

// RequestRoutePlanner derives routing signals from the retrieval request.
type RequestRoutePlanner[TIntent, TRequestMeta, TRoute, TSignal any] interface {
	PlanRoute(ctx context.Context, req Request[TIntent, TRequestMeta]) (RouteDecision[TRoute, TSignal], error)
}

// RoutePlanner is the no-request-metadata route planner.
type RoutePlanner[TIntent, TRoute, TSignal any] = RequestRoutePlanner[TIntent, NoRequestMeta, TRoute, TSignal]

// RequestRoutePlannerFunc adapts a function into RequestRoutePlanner.
type RequestRoutePlannerFunc[TIntent, TRequestMeta, TRoute, TSignal any] func(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
) (RouteDecision[TRoute, TSignal], error)

// RoutePlannerFunc adapts a function into RoutePlanner.
type RoutePlannerFunc[TIntent, TRoute, TSignal any] = RequestRoutePlannerFunc[TIntent, NoRequestMeta, TRoute, TSignal]

// PlanRoute implements RoutePlanner.
func (f RequestRoutePlannerFunc[TIntent, TRequestMeta, TRoute, TSignal]) PlanRoute(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
) (RouteDecision[TRoute, TSignal], error) {
	return f(ctx, req)
}
