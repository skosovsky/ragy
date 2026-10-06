package retrieval_test

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/retrieval"
)

type admittedScopeBackend struct {
	fixture scopeFixture
	calls   atomic.Int64
}

func (b *admittedScopeBackend) Schema() filter.Schema { return b.fixture.schema }
func (*admittedScopeBackend) ReadCapabilities() access.Capabilities {
	return access.Capabilities{RequirePinnedPublication: false, ScopeProfile: true}
}

func (b *admittedScopeBackend) Retrieve(
	ctx context.Context,
	req retrieval.Query[struct{}],
) (retrieval.ResultSet[accessMeta], error) {
	b.calls.Add(1)
	return b.fixture.index.Retrieve(ctx, req)
}

type opaqueScopeNode struct{ calls int }

func (n *opaqueScopeNode) Execute(
	context.Context,
	retrieval.Query[struct{}],
	retrieval.NoExecutionMeta,
) (retrieval.RetrievalResult[accessMeta, retrieval.NoExecutionMeta], error) {
	n.calls++
	return retrieval.RetrievalResult[accessMeta, retrieval.NoExecutionMeta]{}, nil
}

func TestScopedCompositionPreflightRejectsBeforeAnyDispatch(t *testing.T) {
	// Arrange: supported siblings must not start before an unsupported branch is found.
	fixture := newScopeFixture(t, allowAuthority())
	allowed := &admittedScopeBackend{fixture: fixture}
	denied := &incompatibleReadBackend{}
	supported := retrieval.BackendNode[struct{}, accessMeta, retrieval.NoExecutionMeta]{Backend: allowed}
	unsupported := retrieval.BackendNode[struct{}, accessMeta, retrieval.NoExecutionMeta]{Backend: denied}
	callbacks := 0
	route := retrieval.RouteSwitchNode[struct{}, string, struct{}, accessMeta, retrieval.NoExecutionMeta]{
		Planner: retrieval.RoutePlannerFunc[struct{}, string, struct{}](
			func(context.Context, retrieval.Query[struct{}]) (retrieval.RouteDecision[string, struct{}], error) {
				callbacks++
				return retrieval.RouteDecision[string, struct{}]{Route: "supported"}, nil
			},
		),
		Cases: []retrieval.RequestRouteSwitchCase[struct{}, retrieval.NoRequestMeta, string, struct{}, accessMeta, retrieval.NoExecutionMeta]{
			{Route: "supported", Node: supported},
			{Route: "unsupported", Node: unsupported},
		},
	}
	cases := []struct {
		name string
		root retrieval.ExecutionNode[struct{}, accessMeta, retrieval.NoExecutionMeta]
	}{
		{
			name: "parallel aggregate",
			root: retrieval.AggregateNode[struct{}, accessMeta, retrieval.NoExecutionMeta]{
				Nodes: []retrieval.ExecutionNode[struct{}, accessMeta, retrieval.NoExecutionMeta]{
					supported,
					unsupported,
				},
				Concurrency: 2,
			},
		},
		{
			name: "fallback",
			root: retrieval.FallbackNode[struct{}, accessMeta, retrieval.NoExecutionMeta]{
				Primary:   supported,
				Secondary: unsupported,
			},
		},
		{
			name: "rescue",
			root: retrieval.RescueNode[struct{}, accessMeta, retrieval.NoExecutionMeta]{
				Primary:   supported,
				Secondary: unsupported,
			},
		},
		{
			name: "conditional",
			root: retrieval.ConditionalNode[struct{}, accessMeta, retrieval.NoExecutionMeta]{
				Predicate: func(retrieval.Query[struct{}]) bool { callbacks++; return false },
				Child:     unsupported,
			},
		},
		{name: "route cases", root: route},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Act: exercise direct public composition, independently of pipeline wrapping.
			result, err := tc.root.Execute(
				context.Background(),
				retrieval.Query[struct{}]{
					Read:    fixture.binding,
					Text:    "policy",
					Options: retrieval.RetrieveOptions{TopK: 10},
				},
				retrieval.NoExecutionMeta{},
			)
			// Assert: neither payload I/O nor route/predicate callback has run.
			if !errors.Is(err, ragy.ErrUnsupported) || !access.IsProtectionFailure(err) || !result.IsEmpty() ||
				allowed.calls.Load() != 0 ||
				denied.calls != 0 ||
				callbacks != 0 {
				t.Fatalf(
					"unsupported branch dispatched: %v, allowed=%d, denied=%d, callbacks=%d",
					err,
					allowed.calls.Load(),
					denied.calls,
					callbacks,
				)
			}
		})
	}
}

func TestScopedPipelineRejectsOpaqueCustomNodeBeforePlanner(t *testing.T) {
	// Arrange: custom execution nodes need a declared admission contract.
	fixture := newScopeFixture(t, allowAuthority())
	node := &opaqueScopeNode{}
	plannerCalls := 0
	pipeline, err := retrieval.NewExecutionPipelineBuilder[struct{}, accessMeta, retrieval.NoExecutionMeta]().WithRoot(node).
		WithPlanner(
			retrieval.QueryPlannerFunc[struct{}, retrieval.NoRequestMeta](
				func(context.Context, retrieval.Query[struct{}]) (retrieval.PlannedQuery[struct{}], error) {
					plannerCalls++
					return retrieval.PlannedQuery[struct{}]{}, nil
				},
			),
		).
		Build()
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	result, err := pipeline.Execute(
		context.Background(),
		retrieval.Query[struct{}]{Read: fixture.binding, Text: "policy", Options: retrieval.RetrieveOptions{TopK: 10}},
	)
	// Assert.
	if !errors.Is(err, ragy.ErrUnsupported) || !result.IsEmpty() || node.calls != 0 || plannerCalls != 0 {
		t.Fatalf("opaque node admitted: %v, node=%d, planner=%d", err, node.calls, plannerCalls)
	}
}

func TestScopedNestedCompositionExecutesOnlyAllowedPayload(t *testing.T) {
	// Arrange: every nested branch supports scope; restrictions still apply in leaves.
	fixture := newScopeFixture(t, allowAuthority())
	a := &admittedScopeBackend{fixture: fixture}
	b := &admittedScopeBackend{fixture: fixture}
	c := &admittedScopeBackend{fixture: fixture}
	leaf := func(backend *admittedScopeBackend) retrieval.ExecutionNode[struct{}, accessMeta, retrieval.NoExecutionMeta] {
		return retrieval.BackendNode[struct{}, accessMeta, retrieval.NoExecutionMeta]{Backend: backend}
	}
	root := retrieval.AggregateNode[struct{}, accessMeta, retrieval.NoExecutionMeta]{
		Nodes: []retrieval.ExecutionNode[struct{}, accessMeta, retrieval.NoExecutionMeta]{
			leaf(
				a,
			),
			retrieval.FallbackNode[struct{}, accessMeta, retrieval.NoExecutionMeta]{
				Primary:   leaf(b),
				Secondary: leaf(c),
			},
		}, Concurrency: 2,
	}
	// Act.
	result, err := root.Execute(
		context.Background(),
		retrieval.Query[struct{}]{Read: fixture.binding, Text: "policy", Options: retrieval.RetrieveOptions{TopK: 10}},
		retrieval.NoExecutionMeta{},
	)
	// Assert: fallback is admitted, but remains unexecuted when primary is nonempty.
	if err != nil || result.Len() != 1 || result.Documents()[0].ID != "a-public" || a.calls.Load() != 1 ||
		b.calls.Load() != 1 ||
		c.calls.Load() != 0 {
		t.Fatalf(
			"nested scope changed: %v, %v, calls=%d/%d/%d",
			result.Documents(),
			err,
			a.calls.Load(),
			b.calls.Load(),
			c.calls.Load(),
		)
	}
}

func TestRouteRevocationBlocksRescueAndDecisionConsumers(t *testing.T) {
	for _, phase := range []string{"planner", "target"} {
		t.Run(phase, func(t *testing.T) {
			// Arrange: host epoch changes inside one route stage.
			revoked := false
			fixture := newScopeFixture(t, access.AuthorityFunc(func(context.Context, access.Snapshot) error {
				if revoked {
					return ragy.ErrUnavailable
				}
				return nil
			}))
			primary := &revokingBackend{fixture: fixture, revoked: &revoked}
			secondary := &admittedScopeBackend{fixture: fixture}
			decisionCalls, predicateCalls := 0, 0
			root := retrieval.RouteSwitchNode[struct{}, string, struct{}, accessMeta, retrieval.NoExecutionMeta]{
				Planner: retrieval.RoutePlannerFunc[struct{}, string, struct{}](
					func(context.Context, retrieval.Query[struct{}]) (retrieval.RouteDecision[string, struct{}], error) {
						if phase == "planner" {
							revoked = true
						}
						return retrieval.RouteDecision[string, struct{}]{Route: "primary"}, nil
					},
				),
				RecordDecision: func(exec retrieval.NoExecutionMeta, _ retrieval.RouteDecision[string, struct{}]) retrieval.NoExecutionMeta {
					decisionCalls++
					return exec
				},
				Cases: []retrieval.RequestRouteSwitchCase[struct{}, retrieval.NoRequestMeta, string, struct{}, accessMeta, retrieval.NoExecutionMeta]{
					{
						Route: "primary",
						Node:  retrieval.BackendNode[struct{}, accessMeta, retrieval.NoExecutionMeta]{Backend: primary},
					},
					{
						Route: "secondary",
						Node: retrieval.BackendNode[struct{}, accessMeta, retrieval.NoExecutionMeta]{
							Backend: secondary,
						},
					},
				},
				Rescues: []retrieval.RequestRouteFallbackEdge[struct{}, retrieval.NoRequestMeta, string, struct{}, accessMeta, retrieval.NoExecutionMeta]{
					{
						From:    "primary",
						To:      "secondary",
						OnError: true,
						Predicate: func(retrieval.RequestRouteExecutionContext[struct{}, retrieval.NoRequestMeta, string, struct{}, accessMeta, retrieval.NoExecutionMeta]) bool {
							predicateCalls++
							return true
						},
					},
				},
			}
			// Act.
			result, err := root.Execute(
				context.Background(),
				retrieval.Query[struct{}]{
					Read:    fixture.binding,
					Text:    "policy",
					Options: retrieval.RetrieveOptions{TopK: 10},
				},
				retrieval.NoExecutionMeta{},
			)
			// Assert: no rescue callback or secondary I/O after revocation.
			wantPrimary, wantDecision := 1, 1
			if phase == "planner" {
				wantPrimary, wantDecision = 0, 0
			}
			if !access.IsProtectionFailure(err) || !result.IsEmpty() || primary.calls != wantPrimary ||
				decisionCalls != wantDecision ||
				predicateCalls != 0 ||
				secondary.calls.Load() != 0 {
				t.Fatalf(
					"revoked route consumed: %v, primary=%d, decisions=%d, predicates=%d, secondary=%d",
					err,
					primary.calls,
					decisionCalls,
					predicateCalls,
					secondary.calls.Load(),
				)
			}
		})
	}
}

func TestScopedPlanUnsupportedBeforeCompositionDispatch(t *testing.T) {
	// Arrange: the immutable plan contains a condition outside this target schema.
	fixture := newScopeFixture(t, allowAuthority())
	backend := &admittedScopeBackend{fixture: fixture}
	fields := filter.NewSchema()
	foreign, err := fields.String("foreign_plan_field")
	if err != nil {
		t.Fatal(err)
	}
	schema, err := fields.Build()
	if err != nil {
		t.Fatal(err)
	}
	builder, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatal(err)
	}
	condition, err := filter.Eq(builder, foreign, "value").Build()
	if err != nil {
		t.Fatal(err)
	}
	node := retrieval.BackendNode[struct{}, accessMeta, retrieval.NoExecutionMeta]{Backend: backend}
	root := retrieval.AggregateNode[struct{}, accessMeta, retrieval.NoExecutionMeta]{
		Nodes:       []retrieval.ExecutionNode[struct{}, accessMeta, retrieval.NoExecutionMeta]{node, node},
		Concurrency: 2,
	}
	request := retrieval.Query[struct{}]{
		Read: fixture.binding,
		Text: "policy",
		Plan: &retrieval.PlannedQuery[struct{}]{Filters: condition},
	}
	// Act: preflight must reject both leaves before a backend method starts.
	result, err := root.Execute(t.Context(), request, retrieval.NoExecutionMeta{})
	// Assert.
	if !errors.Is(err, ragy.ErrUnsupported) || !access.IsProtectionFailure(err) || !result.IsEmpty() ||
		backend.calls.Load() != 0 {
		t.Fatal("unsupported plan dispatched", err, backend.calls.Load())
	}
}
