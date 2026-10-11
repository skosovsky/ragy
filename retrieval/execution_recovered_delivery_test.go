package retrieval_test

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/retrieval"
)

type recoveredNode = retrieval.RequestExecutionNode[struct{}, struct{}, accessMeta, string]

func recoveredRescue(t *testing.T, cause error) recoveredNode {
	t.Helper()
	schema := newScopeFixture(t, allowAuthority()).schema
	return retrieval.RequestRescueNode[struct{}, struct{}, accessMeta, string]{
		Primary: callbackNode(
			"execution",
			schema,
			func() (retrieval.ResultSet[accessMeta], error) { return nil, cause },
		),
		Secondary: callbackNode(
			"execution",
			schema,
			func() (retrieval.ResultSet[accessMeta], error) { return callbackDocuments(), nil },
		),
	}
}

type recoveredRewrapProcessor struct{}

func (recoveredRewrapProcessor) Process(
	_ context.Context,
	_ access.Binding,
	rs retrieval.ResultSet[accessMeta],
) (retrieval.ResultSet[accessMeta], error) {
	return retrieval.RewrapResultSet(rs, nil), nil
}

func TestRecoveredDeliveryComposition(t *testing.T) {
	// Arrange: two recovered children and an unexecuted branch, post-chain and terminal top-k.
	first, second, skipped, denial := errors.New(
		"first recovered",
	), errors.New(
		"second recovered",
	), errors.New(
		"not executed",
	), errors.New(
		"denial",
	)
	root := retrieval.RequestExecutionAggregateNode[struct{}, struct{}, accessMeta, string]{Nodes: []recoveredNode{
		recoveredRescue(t, first),
		recoveredRescue(t, second),
		retrieval.RequestConditionalNode[struct{}, struct{}, accessMeta, string]{
			Predicate: func(retrieval.Request[struct{}, struct{}]) bool { return false },
			Child:     recoveredRescue(t, skipped),
		},
	}, Concurrency: 2}
	pipeline, err := retrieval.NewRequestExecutionPipelineBuilder[struct{}, struct{}, accessMeta, string]().WithRoot(root).
		WithPostProcessors(recoveredRewrapProcessor{}).
		Build()
	if err != nil {
		t.Fatal(err)
	}
	// Act: preserve envelope through host projection and rewrap, then revoke at final delivery.
	request := callbackRequest(access.Unrestricted())
	request.Options.TopK = 1
	result, err := pipeline.Execute(t.Context(), request)
	if err != nil || result.IsEmpty() {
		t.Fatalf("successful recovery: %v", err)
	}
	result.ResultSet = retrieval.RewrapResultSet(result.ResultSet, nil)
	read := newScopeFixture(
		t,
		access.AuthorityFunc(func(context.Context, access.Snapshot) error { return denial }),
	).binding
	result, err = retrieval.DeliverExecutionRead(t.Context(), read, result, err, nil)
	// Assert: identities in child order, no skipped cause, no payload, no duplicate recovered history.
	assertCallbackProtection(t, result, err, denial)
	if !errors.Is(err, first) || !errors.Is(err, second) || errors.Is(err, skipped) {
		t.Fatalf("composition identities: %v", err)
	}
	if strings.Index(err.Error(), first.Error()) > strings.Index(err.Error(), second.Error()) {
		t.Fatalf("child order: %v", err)
	}
	for range 5 {
		result, err = retrieval.DeliverExecutionRead(t.Context(), read, result, err, nil)
	}
	if strings.Count(err.Error(), first.Error()) != 1 || strings.Count(err.Error(), second.Error()) != 1 {
		t.Fatalf("history duplicated: %v", err)
	}
}

func TestRecoveredDeliveryNestedAndCancellation(t *testing.T) {
	for _, deadline := range []bool{false, true} {
		t.Run(map[bool]string{false: "cancel", true: "deadline"}[deadline], func(t *testing.T) {
			// Arrange: the secondary itself recovers, both causes must survive a late context failure.
			outer, inner := errors.New("outer rescue"), errors.New("inner rescue")
			schema := newScopeFixture(t, allowAuthority()).schema
			root := retrieval.RequestRescueNode[struct{}, struct{}, accessMeta, string]{
				Primary: callbackNode(
					"backend",
					schema,
					func() (retrieval.ResultSet[accessMeta], error) { return nil, outer },
				),
				Secondary: recoveredRescue(t, inner),
			}
			result, err := root.Execute(t.Context(), callbackRequest(access.Unrestricted()), "side output")
			if err != nil || result.IsEmpty() {
				t.Fatalf("nested recovery: %v", err)
			}
			ctx, cancel := context.WithCancel(t.Context())
			cancel()
			want := context.Canceled
			if deadline {
				ctx, cancel = context.WithDeadline(t.Context(), time.Now().Add(-time.Second))
				defer cancel()
				want = context.DeadlineExceeded
			}
			// Act.
			result, err = retrieval.DeliverExecutionRead(ctx, access.Unrestricted(), result, err, nil)
			// Assert.
			assertCallbackProtection(t, result, err, want)
			if !errors.Is(err, outer) || !errors.Is(err, inner) {
				t.Fatalf("nested causes: %v", err)
			}
		})
	}
}

func TestRecoveredDeliveryConcurrentIsolation(t *testing.T) {
	// Arrange independent nested executions using the same parent context.
	const count = 16
	causes := make([]error, count)
	nodes := make([]recoveredNode, count)
	for i := range causes {
		causes[i] = errors.New("independent rescue")
		nodes[i] = recoveredRescue(t, causes[i])
	}
	errs := make([]error, count)
	var workers sync.WaitGroup
	// Act.
	for i := range nodes {
		workers.Go(func() {
			result, err := nodes[i].Execute(t.Context(), callbackRequest(access.Unrestricted()), "")
			ctx, cancel := context.WithCancel(t.Context())
			cancel()
			_, errs[i] = retrieval.DeliverExecutionRead(ctx, access.Unrestricted(), result, err, nil)
		})
	}
	workers.Wait()
	// Assert.
	for i, err := range errs {
		for j, cause := range causes {
			if errors.Is(err, cause) != (i == j) {
				t.Fatalf("execution %d contains cause %d: %v", i, j, err)
			}
		}
	}
}

func TestRecoveredNativeBackendLateGate(t *testing.T) {
	// Arrange a native backend returning an existing recovered envelope.
	cause, denial := errors.New("native recovered"), errors.New("native revoked")
	var revoked bool
	schema := newScopeFixture(t, allowAuthority()).schema
	result, err := recoveredRescue(t, cause).Execute(t.Context(), callbackRequest(access.Unrestricted()), "side")
	if err != nil {
		t.Fatal(err)
	}
	read := newScopeFixture(t, access.AuthorityFunc(func(context.Context, access.Snapshot) error {
		if revoked {
			return denial
		}
		return nil
	})).binding
	node := retrieval.RequestExecutionRetrieverNode[struct{}, struct{}, accessMeta, string]{
		Backend: recoveredEnvelopeBackend{
			callbackRequestBackend: callbackRequestBackend{schema: schema},
			run:                    func() (retrieval.RetrievalResult[accessMeta, string], error) { revoked = true; return result, nil },
		},
	}
	// Act.
	result, err = node.Execute(t.Context(), callbackRequest(read), "")
	// Assert.
	assertCallbackProtection(t, result, err, denial)
	if !errors.Is(err, cause) {
		t.Fatalf("native envelope lost: %v", err)
	}
}

type recoveredEnvelopeBackend struct {
	callbackRequestBackend

	run func() (retrieval.RetrievalResult[accessMeta, string], error)
}

func (b recoveredEnvelopeBackend) Retrieve(
	context.Context,
	retrieval.Request[struct{}, struct{}],
	string,
) (retrieval.RetrievalResult[accessMeta, string], error) {
	return b.run()
}

func TestExecutionDeliveryExistingProtectionAndNilSet(t *testing.T) {
	// Arrange an existing partial error wrapped in protection, with authority allowing access.
	cause := errors.New("partial cause")
	original := &retrieval.PartialFailureError[accessMeta]{Errors: []error{cause}, Result: callbackDocuments()}
	result := retrieval.RetrievalResult[accessMeta, string]{Executed: "side"}
	// Act.
	success, err := retrieval.DeliverExecutionRead(t.Context(), access.Unrestricted(), result, nil, nil)
	if err != nil || success.ResultSet == nil || success.Executed != "side" {
		t.Fatalf("nil normalization: %+v %v", success, err)
	}
	result, err = retrieval.DeliverExecutionRead(
		t.Context(),
		access.Unrestricted(),
		result,
		access.Protect(original),
		nil,
	)
	// Assert.
	assertCallbackProtection(t, result, err, cause)
	view, ok := retrieval.AsPartialFailure[accessMeta](err)
	if !ok || !view.Result.IsEmpty() || original.Result.IsEmpty() {
		t.Fatal("diagnostic view leaked or caller error mutated")
	}
}

type recoveredNodeFunc func(context.Context) (retrieval.RetrievalResult[accessMeta, string], error)

func (f recoveredNodeFunc) Execute(
	ctx context.Context,
	_ retrieval.Request[struct{}, struct{}],
	_ string,
) (retrieval.RetrievalResult[accessMeta, string], error) {
	return f(ctx)
}

func TestRecoveredAggregateCancellationAfterCompletedChild(t *testing.T) {
	// Arrange: a recovered child completes before another child cancels execution.
	cause := errors.New("completed child recovery")
	recovered, err := recoveredRescue(t, cause).Execute(t.Context(), callbackRequest(access.Unrestricted()), "side")
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	root := retrieval.RequestExecutionAggregateNode[struct{}, struct{}, accessMeta, string]{
		Concurrency: 1,
		Nodes: []recoveredNode{
			recoveredNodeFunc(
				func(context.Context) (retrieval.RetrievalResult[accessMeta, string], error) { return recovered, nil },
			),
			recoveredNodeFunc(func(context.Context) (retrieval.RetrievalResult[accessMeta, string], error) {
				cancel()
				return retrieval.RetrievalResult[accessMeta, string]{}, context.Canceled
			}),
		},
	}
	// Act.
	result, err := root.Execute(ctx, callbackRequest(access.Unrestricted()), "")
	// Assert.
	assertCallbackProtection(t, result, err, context.Canceled)
	if !errors.Is(err, cause) {
		t.Fatalf("completed child diagnostic lost: %v", err)
	}
}

//nolint:gocognit // Keep the rescue policy matrix and AAA checks together.
func TestRecoveredRescuePolicy(t *testing.T) {
	for _, policy := range []string{"failed-secondary", "empty-secondary", "partial-primary", "protected-primary", "no-secondary"} {
		t.Run(policy, func(t *testing.T) {
			// Arrange.
			primaryCause, secondaryCause := errors.New("primary policy"), errors.New("secondary policy")
			schema := newScopeFixture(t, allowAuthority()).schema
			var primaryErr error = primaryCause
			primaryDocs := retrieval.NewResultSet[accessMeta](nil, nil)
			switch policy {
			case "partial-primary":
				primaryDocs = callbackDocuments()
				primaryErr = &retrieval.PartialFailureError[accessMeta]{
					Errors: []error{primaryCause},
					Result: primaryDocs,
				}
			case "protected-primary":
				primaryErr = access.Protect(primaryCause)
			}
			calls := 0
			root := retrieval.RequestRescueNode[struct{}, struct{}, accessMeta, string]{
				Primary: callbackNode(
					"backend",
					schema,
					func() (retrieval.ResultSet[accessMeta], error) { return primaryDocs, primaryErr },
				),
				Secondary: callbackNode("backend", schema, func() (retrieval.ResultSet[accessMeta], error) {
					calls++
					if policy == "failed-secondary" {
						return nil, secondaryCause
					}
					return retrieval.NewResultSet[accessMeta](nil, nil), nil
				}),
			}
			if policy == "no-secondary" {
				root.Secondary = nil
			}
			// Act.
			result, err := root.Execute(t.Context(), callbackRequest(access.Unrestricted()), "")
			// Assert: rescue has not converted these failures into recovered success.
			if !errors.Is(err, primaryCause) {
				t.Fatalf("primary policy lost: %v", err)
			}
			if errors.Is(err, secondaryCause) != (policy == "failed-secondary") {
				t.Fatalf("secondary policy: %v", err)
			}
			wantCalls := 0
			if policy == "failed-secondary" || policy == "empty-secondary" {
				wantCalls = 1
			}
			if calls != wantCalls {
				t.Fatalf("secondary calls=%d want=%d", calls, wantCalls)
			}
			if (result.Len() > 0) != (policy == "partial-primary") {
				t.Fatalf("partial/protection payload: %+v", result)
			}
		})
	}
}

func TestRecoveredRouteDeliverySweep(t *testing.T) {
	// Arrange: route-native rescue and an unselected recovered branch.
	cause, denial, skipped := errors.New("route recovered"), errors.New("route denied"), errors.New("route skipped")
	run := func(revokeAt int) (int, bool, retrieval.RetrievalResult[accessMeta, string], error) {
		checks, called := 0, false
		fixture := newScopeFixture(t, access.AuthorityFunc(func(context.Context, access.Snapshot) error {
			checks++
			if revokeAt > 0 && checks >= revokeAt {
				return denial
			}
			return nil
		}))
		planner := retrieval.RequestRoutePlannerFunc[struct{}, struct{}, string, struct{}](
			func(context.Context, retrieval.Request[struct{}, struct{}]) (retrieval.RouteDecision[string, struct{}], error) {
				return retrieval.RouteDecision[string, struct{}]{Route: "primary"}, nil
			},
		)
		root, err := retrieval.NewRequestRouteSwitchBuilder[struct{}, struct{}, string, struct{}, accessMeta, string](
			planner,
		).
			Case("primary", callbackNode("execution", fixture.schema, func() (retrieval.ResultSet[accessMeta], error) { called = true; return nil, cause })).
			Case("secondary", callbackNode("execution", fixture.schema, func() (retrieval.ResultSet[accessMeta], error) { return callbackDocuments(), nil })).
			Case("skipped", recoveredRescue(t, skipped)).
			RescueOnError("primary", "secondary", nil).
			Build()
		if err != nil {
			t.Fatal(err)
		}
		result, err := root.Execute(t.Context(), callbackRequest(fixture.binding), "side")
		result, err = retrieval.DeliverExecutionRead(t.Context(), fixture.binding, result, err, nil)
		return checks, called, result, err
	}
	total, called, result, err := run(0)
	if err != nil || !called || result.IsEmpty() {
		t.Fatalf("route success: %v", err)
	}
	// Act and Assert including the final host gate; skipped errors never appear.
	for gate := 1; gate <= total; gate++ {
		_, called, result, err = run(gate)
		assertCallbackProtection(t, result, err, denial)
		if errors.Is(err, cause) != called || errors.Is(err, skipped) {
			t.Fatalf("route gate %d: called=%v %v", gate, called, err)
		}
	}
}

type recoveredTypedError struct{ label string }

func (e *recoveredTypedError) Error() string { return e.label }

func TestRecoveredDeliveryTypedIdentityAndSerialization(t *testing.T) {
	// Arrange a caller-owned typed error actually absorbed by rescue.
	cause := &recoveredTypedError{label: "private recovered diagnostic"}
	result, err := recoveredRescue(t, cause).Execute(t.Context(), callbackRequest(access.Unrestricted()), "")
	if err != nil {
		t.Fatal(err)
	}
	// Act: exclude the existing textual branch trace; serialize the private carrier.
	result.BranchTrace = nil
	serialized, err := json.Marshal(result)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	_, err = retrieval.DeliverExecutionRead(ctx, access.Unrestricted(), result, nil, nil)
	view, ok := errors.AsType[*recoveredTypedError](err)
	// Assert the original identity survives, but was not serialized in the envelope.
	if !ok || view != cause || !errors.Is(err, context.Canceled) {
		t.Fatalf("typed recovered identity: %v", err)
	}
	if strings.Contains(string(serialized), cause.Error()) {
		t.Fatalf("private cause serialized: %s", serialized)
	}
}
