package retrieval

import (
	"context"
	"fmt"

	ragy "github.com/skosovsky/ragy"
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

// RequestRoutePredicate decides whether a child branch should run for a route decision.
type RequestRoutePredicate[TIntent, TRequestMeta, TRoute, TSignal any] func(
	Request[TIntent, TRequestMeta],
	RouteDecision[TRoute, TSignal],
) bool

// RoutePredicate is the no-request-metadata route predicate.
type RoutePredicate[TIntent, TRoute, TSignal any] = RequestRoutePredicate[TIntent, NoRequestMeta, TRoute, TSignal]

// RequestRouteNode gates a child node behind an explicit typed route decision.
type RequestRouteNode[TIntent, TRequestMeta, TRoute, TSignal, TMeta any] struct {
	Planner   RequestRoutePlanner[TIntent, TRequestMeta, TRoute, TSignal]
	Predicate RequestRoutePredicate[TIntent, TRequestMeta, TRoute, TSignal]
	Child     RequestNode[TIntent, TRequestMeta, TMeta]
	Resolver  IdentityResolver[TMeta]
}

// RouteNode is the no-request-metadata route node.
type RouteNode[TIntent, TRoute, TSignal, TMeta any] = RequestRouteNode[TIntent, NoRequestMeta, TRoute, TSignal, TMeta]

// Retrieve implements Node.
func (n RequestRouteNode[TIntent, TRequestMeta, TRoute, TSignal, TMeta]) Retrieve(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
) (ResultSet[TMeta], error) {
	resolver := n.Resolver
	if resolver == nil {
		resolver = DocumentIDResolver[TMeta]{}
	}
	if n.Planner == nil {
		return NewResultSet[TMeta](nil, resolver), fmt.Errorf("%w: route planner", ragy.ErrInvalidArgument)
	}
	if n.Child == nil {
		return NewResultSet[TMeta](nil, resolver), fmt.Errorf("%w: route child node", ragy.ErrInvalidArgument)
	}
	decision, err := n.Planner.PlanRoute(ctx, req)
	if err != nil {
		return NewResultSet[TMeta](nil, resolver), err
	}
	if n.Predicate != nil && !n.Predicate(req, decision) {
		return NewResultSet[TMeta](nil, resolver), nil
	}
	rs, err := n.Child.Retrieve(ctx, req)
	if err != nil {
		return preserveResultOnError(rs, err, resolver)
	}
	if rs == nil {
		return NewResultSet[TMeta](nil, resolver), nil
	}
	return RewrapResultSet(rs, resolver), nil
}

func (n RequestRouteNode[TIntent, TRequestMeta, TRoute, TSignal, TMeta]) validateNode() error {
	if n.Planner == nil {
		return fmt.Errorf("%w: route planner", ragy.ErrInvalidArgument)
	}
	if n.Child == nil {
		return fmt.Errorf("%w: route child node", ragy.ErrInvalidArgument)
	}
	return validateNodeTree[TIntent, TRequestMeta, TMeta](n.Child)
}

//nolint:unused // injectNodeResolver discovers route nodes through this internal hook.
func (n RequestRouteNode[TIntent, TRequestMeta, TRoute, TSignal, TMeta]) withResolver(
	resolver IdentityResolver[TMeta],
) (RequestNode[TIntent, TRequestMeta, TMeta], error) {
	n.Resolver = resolver
	child, err := injectNodeResolver[TIntent, TRequestMeta, TMeta](n.Child, resolver)
	if err != nil {
		return nil, err
	}
	n.Child = child
	return n, nil
}
