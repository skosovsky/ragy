package retrieval

import (
	"context"
	"fmt"
	"slices"

	ragy "github.com/skosovsky/ragy"
)

// RouteDecisionRecorder projects a typed route decision into caller-owned
// execution metadata.
type RouteDecisionRecorder[TExecMeta, TRoute, TSignal any] func(
	TExecMeta,
	RouteDecision[TRoute, TSignal],
) TExecMeta

// RequestRouteExecutionContext is passed to route-aware fallback predicates.
type RequestRouteExecutionContext[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta any] struct {
	Request  Request[TIntent, TRequestMeta]
	Decision RouteDecision[TRoute, TSignal]
	Route    TRoute
	Result   ResultSet[TMeta]
	Err      error
	Executed TExecMeta
}

// RequestRouteFallbackPredicate decides whether a route fallback edge should run.
type RequestRouteFallbackPredicate[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta any] func(
	RequestRouteExecutionContext[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta],
) bool

// RequestRouteSwitchCase binds one route key to one retrieval branch.
type RequestRouteSwitchCase[TIntent, TRequestMeta, TRoute comparable, TSignal, TMeta, TExecMeta any] struct {
	Route TRoute
	Node  RequestExecutionNode[TIntent, TRequestMeta, TMeta, TExecMeta]
	Name  string
}

// RequestRouteFallbackEdge describes route-aware fallback or rescue transition.
type RequestRouteFallbackEdge[TIntent, TRequestMeta, TRoute comparable, TSignal, TMeta, TExecMeta any] struct {
	From      TRoute
	To        TRoute
	OnEmpty   bool
	OnError   bool
	Predicate RequestRouteFallbackPredicate[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]
}

func (e RequestRouteFallbackEdge[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]) allows(
	ctx RequestRouteExecutionContext[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta],
) bool {
	return e.Predicate == nil || e.Predicate(ctx)
}

// RequestRouteSwitchNode dispatches one planned route decision into typed
// retrieval branches and records branch trace in RetrievalResult.
type RequestRouteSwitchNode[TIntent, TRequestMeta, TRoute comparable, TSignal, TMeta, TExecMeta any] struct {
	Planner        RequestRoutePlanner[TIntent, TRequestMeta, TRoute, TSignal]
	Cases          []RequestRouteSwitchCase[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]
	Default        RequestExecutionNode[TIntent, TRequestMeta, TMeta, TExecMeta]
	DefaultName    string
	Fallbacks      []RequestRouteFallbackEdge[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]
	Rescues        []RequestRouteFallbackEdge[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]
	RecordDecision RouteDecisionRecorder[TExecMeta, TRoute, TSignal]
	Resolver       IdentityResolver[TMeta]
	Name           string
}

// RouteSwitchNode is the no-request-metadata route switch shape.
type RouteSwitchNode[TIntent, TRoute comparable, TSignal, TMeta, TExecMeta any] = RequestRouteSwitchNode[
	TIntent,
	NoRequestMeta,
	TRoute,
	TSignal,
	TMeta,
	TExecMeta,
]

// NewRequestRouteSwitchBuilder starts a typed route switch builder.
func NewRequestRouteSwitchBuilder[TIntent, TRequestMeta, TRoute comparable, TSignal, TMeta, TExecMeta any](
	planner RequestRoutePlanner[TIntent, TRequestMeta, TRoute, TSignal],
) *RequestRouteSwitchBuilder[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta] {
	return &RequestRouteSwitchBuilder[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]{
		node: RequestRouteSwitchNode[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]{
			Planner:        planner,
			Cases:          nil,
			Default:        nil,
			DefaultName:    "",
			Fallbacks:      nil,
			Rescues:        nil,
			RecordDecision: nil,
			Resolver:       nil,
			Name:           "",
		},
	}
}

// NewRouteSwitchBuilder starts a no-request-metadata typed route switch builder.
func NewRouteSwitchBuilder[TIntent, TRoute comparable, TSignal, TMeta, TExecMeta any](
	planner RoutePlanner[TIntent, TRoute, TSignal],
) *RequestRouteSwitchBuilder[TIntent, NoRequestMeta, TRoute, TSignal, TMeta, TExecMeta] {
	return NewRequestRouteSwitchBuilder[TIntent, NoRequestMeta, TRoute, TSignal, TMeta, TExecMeta](planner)
}

// RequestRouteSwitchBuilder builds RequestRouteSwitchNode.
type RequestRouteSwitchBuilder[TIntent, TRequestMeta, TRoute comparable, TSignal, TMeta, TExecMeta any] struct {
	node RequestRouteSwitchNode[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]
}

// Name sets a trace label for the route switch.
func (b *RequestRouteSwitchBuilder[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]) Name(
	name string,
) *RequestRouteSwitchBuilder[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta] {
	b.node.Name = name
	return b
}

// RecordDecision stores the typed route decision in caller-owned execution metadata.
func (b *RequestRouteSwitchBuilder[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]) RecordDecision(
	record RouteDecisionRecorder[TExecMeta, TRoute, TSignal],
) *RequestRouteSwitchBuilder[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta] {
	b.node.RecordDecision = record
	return b
}

// Case adds an execution-aware branch for route.
func (b *RequestRouteSwitchBuilder[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]) Case(
	route TRoute,
	node RequestExecutionNode[TIntent, TRequestMeta, TMeta, TExecMeta],
) *RequestRouteSwitchBuilder[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta] {
	b.node.Cases = append(
		b.node.Cases,
		RequestRouteSwitchCase[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]{
			Route: route,
			Node:  node,
			Name:  fmt.Sprint(route),
		},
	)
	return b
}

// ExecutionCase is an alias for Case.
func (b *RequestRouteSwitchBuilder[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]) ExecutionCase(
	route TRoute,
	node RequestExecutionNode[TIntent, TRequestMeta, TMeta, TExecMeta],
) *RequestRouteSwitchBuilder[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta] {
	return b.Case(route, node)
}

// Default sets an execution-aware default branch when no case matches the route.
func (b *RequestRouteSwitchBuilder[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]) Default(
	node RequestExecutionNode[TIntent, TRequestMeta, TMeta, TExecMeta],
) *RequestRouteSwitchBuilder[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta] {
	b.node.Default = node
	b.node.DefaultName = "default"
	return b
}

// ExecutionDefault is an alias for Default.
func (b *RequestRouteSwitchBuilder[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]) ExecutionDefault(
	node RequestExecutionNode[TIntent, TRequestMeta, TMeta, TExecMeta],
) *RequestRouteSwitchBuilder[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta] {
	return b.Default(node)
}

// FallbackOnEmpty runs route to when route from returns an empty successful result.
func (b *RequestRouteSwitchBuilder[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]) FallbackOnEmpty(
	from TRoute,
	to TRoute,
) *RequestRouteSwitchBuilder[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta] {
	b.node.Fallbacks = append(
		b.node.Fallbacks,
		RequestRouteFallbackEdge[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]{
			From:      from,
			To:        to,
			OnEmpty:   true,
			OnError:   false,
			Predicate: nil,
		},
	)
	return b
}

// ConditionalFallback runs route to when route from returns empty and predicate passes.
func (b *RequestRouteSwitchBuilder[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]) ConditionalFallback(
	from TRoute,
	to TRoute,
	predicate RequestRouteFallbackPredicate[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta],
) *RequestRouteSwitchBuilder[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta] {
	b.node.Fallbacks = append(
		b.node.Fallbacks,
		RequestRouteFallbackEdge[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]{
			From:      from,
			To:        to,
			OnEmpty:   true,
			OnError:   false,
			Predicate: predicate,
		},
	)
	return b
}

// RescueOnError runs route to when route from returns an error without partial success.
func (b *RequestRouteSwitchBuilder[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]) RescueOnError(
	from TRoute,
	to TRoute,
	predicate RequestRouteFallbackPredicate[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta],
) *RequestRouteSwitchBuilder[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta] {
	b.node.Rescues = append(
		b.node.Rescues,
		RequestRouteFallbackEdge[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]{
			From:      from,
			To:        to,
			OnEmpty:   false,
			OnError:   true,
			Predicate: predicate,
		},
	)
	return b
}

// Build returns the route switch node.
func (b *RequestRouteSwitchBuilder[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]) Build() (
	RequestRouteSwitchNode[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta],
	error,
) {
	if err := b.node.validateExecutionNode(); err != nil {
		return b.node, err
	}
	return b.node, nil
}

// Execute implements RequestExecutionNode.
func (n RequestRouteSwitchNode[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]) Execute(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
	exec TExecMeta,
) (RetrievalResult[TMeta, TExecMeta], error) {
	result, decision, err := n.startRouteSwitch(ctx, req, exec)
	if err != nil {
		return result, err
	}
	currentRoute := decision.Route
	currentCase, ok := n.caseFor(currentRoute)
	if !ok {
		var hasDefault bool
		currentCase, result, hasDefault = n.defaultCase(currentRoute, result)
		if !hasDefault {
			return result, nil
		}
	}
	return n.executeRouteChain(ctx, req, decision, currentRoute, currentCase, result)
}

func (n RequestRouteSwitchNode[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]) startRouteSwitch(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
	exec TExecMeta,
) (RetrievalResult[TMeta, TExecMeta], RouteDecision[TRoute, TSignal], error) {
	resolver := n.Resolver
	if resolver == nil {
		resolver = DocumentIDResolver[TMeta]{}
	}
	if n.Planner == nil {
		return emptyRetrievalResult(resolver, exec),
			RouteDecision[TRoute, TSignal]{
				Route:       *new(TRoute),
				Signal:      *new(TSignal),
				Diagnostics: nil,
			},
			fmt.Errorf("%w: route switch planner", ragy.ErrInvalidArgument)
	}
	decision, err := n.Planner.PlanRoute(ctx, req)
	if n.RecordDecision != nil {
		exec = n.RecordDecision(exec, decision)
	}
	result := emptyRetrievalResult(resolver, exec)
	result.Diagnostics = append(result.Diagnostics, routeDiagnostics(decision.Diagnostics)...)
	state := BranchStateSelected
	errText := ""
	if err != nil {
		state = BranchStateErrored
		errText = err.Error()
	}
	result.BranchTrace = append(result.BranchTrace, BranchStep{
		Node:  n.traceName(),
		Kind:  BranchKindRoute,
		Route: fmt.Sprint(decision.Route),
		State: state,
		Error: errText,
	})
	if err != nil {
		return result, decision, err
	}
	return result, decision, nil
}

func (n RequestRouteSwitchNode[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]) defaultCase(
	currentRoute TRoute,
	result RetrievalResult[TMeta, TExecMeta],
) (RequestRouteSwitchCase[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta], RetrievalResult[TMeta, TExecMeta], bool) {
	if n.Default == nil {
		result.BranchTrace = append(result.BranchTrace, BranchStep{
			Node:  n.traceName(),
			Kind:  BranchKindCase,
			Route: fmt.Sprint(currentRoute),
			State: BranchStateSkipped,
			Error: "",
		})
		return emptyRouteSwitchCase[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta](), result, false
	}
	return RequestRouteSwitchCase[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]{
		Route: currentRoute,
		Node:  n.Default,
		Name:  n.DefaultName,
	}, result, true
}

func (n RequestRouteSwitchNode[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]) executeRouteChain(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
	decision RouteDecision[TRoute, TSignal],
	currentRoute TRoute,
	currentCase RequestRouteSwitchCase[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta],
	result RetrievalResult[TMeta, TExecMeta],
) (RetrievalResult[TMeta, TExecMeta], error) {
	var err error
	result, err = n.runCase(ctx, req, decision, currentCase, result)
	if err != nil {
		return n.handleRouteError(ctx, req, decision, currentRoute, result, err)
	}
	return n.runFallbacks(ctx, req, decision, currentRoute, result)
}

func (n RequestRouteSwitchNode[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]) runFallbacks(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
	decision RouteDecision[TRoute, TSignal],
	currentRoute TRoute,
	result RetrievalResult[TMeta, TExecMeta],
) (RetrievalResult[TMeta, TExecMeta], error) {
	for steps := 0; result.ResultSet.IsEmpty() && steps <= len(n.Fallbacks); steps++ {
		next, nextRoute, nextResult, ok, err := n.selectFallback(req, decision, currentRoute, result)
		if err != nil || !ok {
			return nextResult, err
		}
		currentRoute = nextRoute
		result = nextResult
		var runErr error
		result, runErr = n.runCase(ctx, req, decision, next, result)
		if runErr != nil {
			return n.handleRouteError(ctx, req, decision, currentRoute, result, runErr)
		}
	}
	if result.ResultSet.IsEmpty() && len(n.Fallbacks) > 0 {
		result.BranchTrace = append(result.BranchTrace, BranchStep{
			Node:  n.traceName(),
			Kind:  BranchKindFallback,
			Route: fmt.Sprint(currentRoute),
			State: BranchStateSkipped,
			Error: "fallback limit exhausted",
		})
	}
	return result, nil
}

func (n RequestRouteSwitchNode[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]) selectFallback(
	req Request[TIntent, TRequestMeta],
	decision RouteDecision[TRoute, TSignal],
	currentRoute TRoute,
	result RetrievalResult[TMeta, TExecMeta],
) (
	RequestRouteSwitchCase[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta],
	TRoute,
	RetrievalResult[TMeta, TExecMeta],
	bool,
	error,
) {
	matched := false
	for _, edge := range n.Fallbacks {
		if edge.From != currentRoute || !edge.OnEmpty {
			continue
		}
		matched = true
		if !edge.allows(routeExecutionContext(req, decision, currentRoute, result, nil)) {
			result.BranchTrace = append(result.BranchTrace, skippedEdgeTrace(BranchKindFallback, edge.To))
			continue
		}
		nextCase, ok := n.caseFor(edge.To)
		if !ok {
			return emptyRouteSwitchCase[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta](),
				currentRoute,
				result,
				false,
				fmt.Errorf("%w: route switch fallback target %q", ragy.ErrInvalidArgument, fmt.Sprint(edge.To))
		}
		result.BranchTrace = append(result.BranchTrace, selectedEdgeTrace(BranchKindFallback, nextCase.Name, edge.To))
		return nextCase, edge.To, result, true, nil
	}
	if !matched {
		result.BranchTrace = append(result.BranchTrace, BranchStep{
			Node:  n.traceName(),
			Kind:  BranchKindFallback,
			Route: fmt.Sprint(currentRoute),
			State: BranchStateSkipped,
			Error: "",
		})
	}
	return emptyRouteSwitchCase[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta](),
		currentRoute,
		result,
		false,
		nil
}

func (n RequestRouteSwitchNode[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]) handleRouteError(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
	decision RouteDecision[TRoute, TSignal],
	currentRoute TRoute,
	result RetrievalResult[TMeta, TExecMeta],
	err error,
) (RetrievalResult[TMeta, TExecMeta], error) {
	resolver := n.Resolver
	if resolver == nil {
		resolver = DocumentIDResolver[TMeta]{}
	}
	if partialSuccessRS(result.ResultSet, err) {
		result.ResultSet, _ = preserveResultOnError(result.ResultSet, err, resolver)
		return result, err
	}
	rescued, rescueErr := n.tryRescue(ctx, req, decision, currentRoute, result, err)
	if rescueErr != nil || rescued.BranchTrace != nil {
		return rescued, rescueErr
	}
	return result, err
}

func (n RequestRouteSwitchNode[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]) runCase(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
	_ RouteDecision[TRoute, TSignal],
	c RequestRouteSwitchCase[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta],
	result RetrievalResult[TMeta, TExecMeta],
) (RetrievalResult[TMeta, TExecMeta], error) {
	resolver := n.Resolver
	if resolver == nil {
		resolver = DocumentIDResolver[TMeta]{}
	}
	child, err := c.Node.Execute(ctx, req, result.Executed)
	child.ResultSet = ensureResultSet(child.ResultSet, resolver)
	caseStep := BranchStep{
		Node:  c.Name,
		Kind:  BranchKindCase,
		Route: fmt.Sprint(c.Route),
		State: resultState(child.ResultSet, err),
		Error: "",
	}
	if err != nil {
		caseStep.Error = err.Error()
	}
	child.BranchTrace = append([]BranchStep{caseStep}, child.BranchTrace...)
	child.Diagnostics = append(result.Diagnostics, child.Diagnostics...)
	child.BranchTrace = append(result.BranchTrace, child.BranchTrace...)
	return child, err
}

func (n RequestRouteSwitchNode[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]) tryRescue(
	ctx context.Context,
	req Request[TIntent, TRequestMeta],
	decision RouteDecision[TRoute, TSignal],
	current TRoute,
	result RetrievalResult[TMeta, TExecMeta],
	primaryErr error,
) (RetrievalResult[TMeta, TExecMeta], error) {
	nextCase, nextResult, ok, selectErr := n.selectRescue(req, decision, current, result, primaryErr)
	if selectErr != nil {
		return nextResult, selectErr
	}
	if !ok {
		return nextResult, primaryErr
	}
	rescued, rescueErr := n.runCase(ctx, req, decision, nextCase, nextResult)
	if rescueErr != nil {
		return rescued, fmt.Errorf("%w; route rescue: %w", primaryErr, rescueErr)
	}
	if rescued.ResultSet.IsEmpty() {
		return rescued, fmt.Errorf("%w: route rescue empty", primaryErr)
	}
	return rescued, nil
}

func (n RequestRouteSwitchNode[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]) selectRescue(
	req Request[TIntent, TRequestMeta],
	decision RouteDecision[TRoute, TSignal],
	current TRoute,
	result RetrievalResult[TMeta, TExecMeta],
	err error,
) (
	RequestRouteSwitchCase[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta],
	RetrievalResult[TMeta, TExecMeta],
	bool,
	error,
) {
	matched := false
	for _, edge := range n.Rescues {
		if edge.From != current || !edge.OnError {
			continue
		}
		matched = true
		if !edge.allows(routeExecutionContext(req, decision, current, result, err)) {
			result.BranchTrace = append(result.BranchTrace, skippedEdgeTrace(BranchKindRescue, edge.To))
			continue
		}
		nextCase, ok := n.caseFor(edge.To)
		if !ok {
			return emptyRouteSwitchCase[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta](),
				result,
				false,
				fmt.Errorf("%w: route switch rescue target %q", ragy.ErrInvalidArgument, fmt.Sprint(edge.To))
		}
		result.BranchTrace = append(result.BranchTrace, selectedEdgeTrace(BranchKindRescue, nextCase.Name, edge.To))
		return nextCase, result, true, nil
	}
	if !matched {
		result.BranchTrace = append(result.BranchTrace, BranchStep{
			Node:  n.traceName(),
			Kind:  BranchKindRescue,
			Route: fmt.Sprint(current),
			State: BranchStateSkipped,
			Error: "",
		})
	}
	return emptyRouteSwitchCase[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta](), result, false, nil
}

func (n RequestRouteSwitchNode[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]) caseFor(
	route TRoute,
) (RequestRouteSwitchCase[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta], bool) {
	for _, c := range n.Cases {
		if c.Route == route {
			return c, true
		}
	}
	return emptyRouteSwitchCase[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta](), false
}

func emptyRouteSwitchCase[TIntent, TRequestMeta, TRoute comparable, TSignal, TMeta, TExecMeta any]() RequestRouteSwitchCase[
	TIntent,
	TRequestMeta,
	TRoute,
	TSignal,
	TMeta,
	TExecMeta,
] {
	return RequestRouteSwitchCase[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]{
		Route: *new(TRoute),
		Node:  nil,
		Name:  "",
	}
}

func routeExecutionContext[TIntent, TRequestMeta, TRoute comparable, TSignal, TMeta, TExecMeta any](
	req Request[TIntent, TRequestMeta],
	decision RouteDecision[TRoute, TSignal],
	current TRoute,
	result RetrievalResult[TMeta, TExecMeta],
	err error,
) RequestRouteExecutionContext[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta] {
	return RequestRouteExecutionContext[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]{
		Request:  req,
		Decision: decision,
		Route:    current,
		Result:   result.ResultSet,
		Err:      err,
		Executed: result.Executed,
	}
}

func selectedEdgeTrace[TRoute any](kind, name string, route TRoute) BranchStep {
	return BranchStep{
		Node:  name,
		Kind:  kind,
		Route: fmt.Sprint(route),
		State: BranchStateSelected,
		Error: "",
	}
}

func skippedEdgeTrace[TRoute any](kind string, route TRoute) BranchStep {
	return BranchStep{
		Node:  "",
		Kind:  kind,
		Route: fmt.Sprint(route),
		State: BranchStateSkipped,
		Error: "",
	}
}

func (n RequestRouteSwitchNode[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]) traceName() string {
	if n.Name != "" {
		return n.Name
	}
	return "route-switch"
}

func (n RequestRouteSwitchNode[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]) validateExecutionNode() error {
	if n.Planner == nil {
		return fmt.Errorf("%w: route switch planner", ragy.ErrInvalidArgument)
	}
	routes := make(map[TRoute]struct{}, len(n.Cases))
	for i, c := range n.Cases {
		if c.Node == nil {
			return fmt.Errorf("%w: route switch case at index %d", ragy.ErrInvalidArgument, i)
		}
		if _, exists := routes[c.Route]; exists {
			return fmt.Errorf("%w: duplicate route switch case %q", ragy.ErrInvalidArgument, fmt.Sprint(c.Route))
		}
		if err := validateExecutionNodeTree[TIntent, TRequestMeta, TMeta, TExecMeta](c.Node); err != nil {
			return err
		}
		routes[c.Route] = struct{}{}
	}
	if n.Default != nil {
		if err := validateExecutionNodeTree[TIntent, TRequestMeta, TMeta, TExecMeta](n.Default); err != nil {
			return err
		}
	}
	if err := validateRouteEdges("fallback", n.Fallbacks, routes); err != nil {
		return err
	}
	if err := validateRouteEdges("rescue", n.Rescues, routes); err != nil {
		return err
	}
	return nil
}

func validateRouteEdges[TIntent, TRequestMeta, TRoute comparable, TSignal, TMeta, TExecMeta any](
	name string,
	edges []RequestRouteFallbackEdge[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta],
	routes map[TRoute]struct{},
) error {
	graph := make(map[TRoute][]TRoute, len(edges))
	for _, edge := range edges {
		if _, ok := routes[edge.From]; !ok {
			return fmt.Errorf("%w: route switch %s source %q", ragy.ErrInvalidArgument, name, fmt.Sprint(edge.From))
		}
		if _, ok := routes[edge.To]; !ok {
			return fmt.Errorf("%w: route switch %s target %q", ragy.ErrInvalidArgument, name, fmt.Sprint(edge.To))
		}
		graph[edge.From] = append(graph[edge.From], edge.To)
	}
	if hasRouteCycle(graph) {
		return fmt.Errorf("%w: route switch %s cycle", ragy.ErrInvalidArgument, name)
	}
	return nil
}

func hasRouteCycle[TRoute comparable](graph map[TRoute][]TRoute) bool {
	visiting := make(map[TRoute]bool, len(graph))
	visited := make(map[TRoute]bool, len(graph))
	var visit func(TRoute) bool
	visit = func(route TRoute) bool {
		if visiting[route] {
			return true
		}
		if visited[route] {
			return false
		}
		visiting[route] = true
		if slices.ContainsFunc(graph[route], visit) {
			return true
		}
		visiting[route] = false
		visited[route] = true
		return false
	}
	for route := range graph {
		if visit(route) {
			return true
		}
	}
	return false
}

//nolint:unused // injectExecutionNodeResolver discovers route switches through this internal hook.
func (n RequestRouteSwitchNode[TIntent, TRequestMeta, TRoute, TSignal, TMeta, TExecMeta]) withExecutionResolver(
	resolver IdentityResolver[TMeta],
) (RequestExecutionNode[TIntent, TRequestMeta, TMeta, TExecMeta], error) {
	n.Resolver = resolver
	for i, c := range n.Cases {
		rebound, err := injectExecutionNodeResolver[TIntent, TRequestMeta, TMeta, TExecMeta](c.Node, resolver)
		if err != nil {
			return nil, err
		}
		n.Cases[i].Node = rebound
	}
	if n.Default != nil {
		rebound, err := injectExecutionNodeResolver[TIntent, TRequestMeta, TMeta, TExecMeta](n.Default, resolver)
		if err != nil {
			return nil, err
		}
		n.Default = rebound
	}
	return n, nil
}
