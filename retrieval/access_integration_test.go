package retrieval_test

import (
	"context"
	"errors"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/lexical"
	"github.com/skosovsky/ragy/retrieval"
)

type accessMeta struct {
	Tenant     string `json:"tenant"`
	Visibility string `json:"visibility"`
}

type scopeFixture struct {
	binding  access.Binding
	index    *lexical.BM25Index[accessMeta]
	schema   filter.Schema
	conflict filter.Condition
}

func newScopeFixture(t *testing.T, authority access.Authority) scopeFixture {
	t.Helper()
	builder := filter.NewSchema()
	tenant, err := builder.String("tenant")
	if err != nil {
		t.Fatal(err)
	}
	visibility, err := builder.String("visibility")
	if err != nil {
		t.Fatal(err)
	}
	schema, err := builder.Build()
	if err != nil {
		t.Fatal(err)
	}
	predicates, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatal(err)
	}
	mandatory, err := filter.In(filter.Eq(predicates, tenant, "a"), visibility, "public").Build()
	if err != nil {
		t.Fatal(err)
	}
	query, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatal(err)
	}
	conflict, err := filter.Eq(query, tenant, "b").Build()
	if err != nil {
		t.Fatal(err)
	}
	now := time.Unix(100, 0)
	binding, err := access.Scoped(
		access.ScopedConfig{
			Snapshot: access.Snapshot{
				Identity:    "host-policy-7",
				PolicyEpoch: 7,
				IssuedAt:    now,
				ExpiresAt:   now.Add(30 * time.Second),
			},
			Schema:      schema,
			Mandatory:   mandatory,
			Publication: access.CurrentPublication(),
			Authority:   authority,
			Now:         func() time.Time { return now },
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	index, err := lexical.NewBM25Index[accessMeta](
		schema,
		lexical.Config[accessMeta]{SearchFields: []string{"content"}},
		nil,
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	err = index.Index([]retrieval.Document[accessMeta]{
		{ID: "a-public", Content: "policy", Meta: accessMeta{Tenant: "a", Visibility: "public"}},
		{ID: "a-private", Content: "policy", Meta: accessMeta{Tenant: "a", Visibility: "private"}},
		{ID: "b-public", Content: "policy", Meta: accessMeta{Tenant: "b", Visibility: "public"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	return scopeFixture{binding: binding, index: index, schema: schema, conflict: conflict}
}
func allowAuthority() access.Authority {
	return access.AuthorityFunc(func(context.Context, access.Snapshot) error { return nil })
}

func TestLexicalMandatoryScope(t *testing.T) {
	// Arrange.
	fixture := newScopeFixture(t, allowAuthority())
	for _, tc := range []struct {
		name  string
		query filter.Condition
		want  int
	}{{name: "no optional predicate", want: 1}, {name: "contradictory planner", query: fixture.conflict, want: 0}} {
		t.Run(tc.name, func(t *testing.T) {
			// Act.
			rs, err := fixture.index.Retrieve(
				context.Background(),
				retrieval.Query[struct{}]{
					Read:    fixture.binding,
					Text:    "policy",
					Options: retrieval.RetrieveOptions{TopK: 10, Filters: tc.query},
				},
			)
			// Assert.
			if err != nil || rs.Len() != tc.want {
				t.Fatalf("scope result = %v, %v", rs.Documents(), err)
			}
			if rs.Len() > 0 && rs.Documents()[0].ID != "a-public" {
				t.Fatalf("private/foreign payload escaped: %+v", rs.Documents())
			}
		})
	}
}

func TestPlannerBinderAndProjectorCannotReplaceScope(t *testing.T) {
	// Arrange: both binder and projector manufacture a fresh unrestricted request.
	fixture := newScopeFixture(t, allowAuthority())
	projected := retrieval.ProjectedBackend[struct{}, retrieval.NoRequestMeta, struct{}, retrieval.NoRequestMeta, accessMeta]{
		Next: fixture.index,
		Project: func(retrieval.Query[struct{}]) retrieval.Query[struct{}] {
			return retrieval.Query[struct{}]{
				Read:    retrieval.UnrestrictedRead(),
				Text:    "policy",
				Options: retrieval.RetrieveOptions{TopK: 10},
			}
		},
	}
	planner := retrieval.QueryPlannerFunc[struct{}, retrieval.NoRequestMeta](
		func(context.Context, retrieval.Query[struct{}]) (retrieval.PlannedQuery[struct{}], error) {
			return retrieval.PlannedQuery[struct{}]{Text: "policy", Filters: fixture.conflict}, nil
		},
	)
	binder := retrieval.RequestPlanBinderFunc[struct{}, retrieval.NoRequestMeta, retrieval.NoExecutionMeta](
		func(context.Context, retrieval.Query[struct{}], *retrieval.PlannedQuery[struct{}], retrieval.NoExecutionMeta) (retrieval.BoundRequest[struct{}, retrieval.NoRequestMeta, retrieval.NoExecutionMeta], error) {
			return retrieval.BoundRequest[struct{}, retrieval.NoRequestMeta, retrieval.NoExecutionMeta]{
				Request: retrieval.Query[struct{}]{
					Read:    retrieval.UnrestrictedRead(),
					Text:    "policy",
					Options: retrieval.RetrieveOptions{TopK: 10},
				},
			}, nil
		},
	)
	pipeline, err := retrieval.NewExecutionPipelineBuilder[struct{}, accessMeta, retrieval.NoExecutionMeta]().WithPlanner(planner).
		WithPlanBinder(binder).
		WithRoot(retrieval.BackendNode[struct{}, accessMeta, retrieval.NoExecutionMeta]{Backend: projected}).
		Build()
	if err != nil {
		t.Fatal(err)
	}
	// Act: erasing query filters does not erase mandatory restrictions.
	result, err := pipeline.Execute(
		context.Background(),
		retrieval.Query[struct{}]{Read: fixture.binding, Text: "policy", Options: retrieval.RetrieveOptions{TopK: 10}},
	)
	// Assert.
	if err != nil || result.ResultSet.Len() != 1 || result.ResultSet.Documents()[0].ID != "a-public" {
		t.Fatalf("binding lost through projection: %v, %v", result.ResultSet.Documents(), err)
	}
}

type revokingBackend struct {
	fixture scopeFixture
	revoked *bool
	calls   int
}

func (b *revokingBackend) Schema() filter.Schema { return b.fixture.schema }
func (*revokingBackend) ReadCapabilities() access.Capabilities {
	return access.Capabilities{RequirePinnedPublication: false, ScopeProfile: true}
}

func (b *revokingBackend) Retrieve(
	ctx context.Context,
	req retrieval.Query[struct{}],
) (retrieval.ResultSet[accessMeta], error) {
	b.calls++
	result, err := b.fixture.index.Retrieve(ctx, req)
	*b.revoked = true // The host policy changes while the outer target call is in flight.
	return result, err
}
func TestRevocationDuringIOBlocksDeliveryAndRescue(t *testing.T) {
	// Arrange.
	revoked := false
	fixture := newScopeFixture(t, access.AuthorityFunc(func(context.Context, access.Snapshot) error {
		if revoked {
			return ragy.ErrUnavailable
		}
		return nil
	}))
	backend := &revokingBackend{fixture: fixture, revoked: &revoked}
	secondary := &revokingBackend{fixture: fixture, revoked: &revoked}
	root := retrieval.RescueNode[struct{}, accessMeta, retrieval.NoExecutionMeta]{
		Primary:   retrieval.BackendNode[struct{}, accessMeta, retrieval.NoExecutionMeta]{Backend: backend},
		Secondary: retrieval.BackendNode[struct{}, accessMeta, retrieval.NoExecutionMeta]{Backend: secondary},
	}
	pipeline, err := retrieval.NewExecutionPipelineBuilder[struct{}, accessMeta, retrieval.NoExecutionMeta]().WithRoot(root).
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
	if !access.IsProtectionFailure(err) || !errors.Is(err, ragy.ErrUnavailable) || !result.ResultSet.IsEmpty() ||
		backend.calls != 1 ||
		secondary.calls != 0 {
		t.Fatalf(
			"revoked result rescued/exposed: %+v, err=%v, calls=%d/%d",
			result,
			err,
			backend.calls,
			secondary.calls,
		)
	}
}

func TestMissingReadRejectedBeforePlanner(t *testing.T) {
	// Arrange.
	fixture := newScopeFixture(t, allowAuthority())
	calls := 0
	planner := retrieval.QueryPlannerFunc[struct{}, retrieval.NoRequestMeta](
		func(context.Context, retrieval.Query[struct{}]) (retrieval.PlannedQuery[struct{}], error) {
			calls++
			return retrieval.PlannedQuery[struct{}]{}, nil
		},
	)
	pipeline, err := retrieval.NewExecutionPipelineBuilder[struct{}, accessMeta, retrieval.NoExecutionMeta]().WithPlanner(planner).
		WithRoot(retrieval.BackendNode[struct{}, accessMeta, retrieval.NoExecutionMeta]{Backend: fixture.index}).
		Build()
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	result, err := pipeline.Execute(
		context.Background(),
		retrieval.Query[struct{}]{Read: access.Binding{}, Text: "policy", Options: retrieval.RetrieveOptions{TopK: 10}},
	)
	// Assert.
	if !access.IsProtectionFailure(err) || calls != 0 || !result.ResultSet.IsEmpty() {
		t.Fatalf("unbound request dispatched: %v, calls=%d", err, calls)
	}
}

type incompatibleReadBackend struct{ calls int }

func (b *incompatibleReadBackend) Retrieve(
	context.Context,
	retrieval.Query[struct{}],
) (retrieval.ResultSet[accessMeta], error) {
	b.calls++
	return retrieval.NewResultSet[accessMeta](nil, nil), nil
}

func TestScopedUnknownBackendRejectedBeforeDispatch(t *testing.T) {
	// Arrange: this adapter never declares pre-payload enforcement guarantees.
	fixture := newScopeFixture(t, allowAuthority())
	backend := &incompatibleReadBackend{}
	pipeline, err := retrieval.NewExecutionPipelineBuilder[struct{}, accessMeta, retrieval.NoExecutionMeta]().WithRoot(retrieval.BackendNode[struct{}, accessMeta, retrieval.NoExecutionMeta]{Backend: backend}).
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
	if !access.IsProtectionFailure(err) || !errors.Is(err, ragy.ErrUnsupported) || backend.calls != 0 ||
		!result.ResultSet.IsEmpty() {
		t.Fatalf("unknown scoped adapter dispatched: err=%v, calls=%d", err, backend.calls)
	}
}

type protectedProcessor struct {
	run func(context.Context, access.Binding, retrieval.ResultSet[accessMeta]) (retrieval.ResultSet[accessMeta], error)
}

func (p protectedProcessor) Process(
	ctx context.Context,
	read access.Binding,
	rs retrieval.ResultSet[accessMeta],
) (retrieval.ResultSet[accessMeta], error) {
	return p.run(ctx, read, rs)
}

func TestRevocationBetweenProcessorsStopsNextConsumer(t *testing.T) {
	for _, failure := range []bool{false, true} {
		t.Run(map[bool]string{false: "success-return", true: "partial-error-return"}[failure], func(t *testing.T) {
			// Arrange.
			revoked := false
			firstCalls := 0
			nextCalls := 0
			fixture := newScopeFixture(t, access.AuthorityFunc(func(context.Context, access.Snapshot) error {
				if revoked {
					return ragy.ErrUnavailable
				}
				return nil
			}))
			rs, err := fixture.index.Retrieve(
				context.Background(),
				retrieval.Query[struct{}]{
					Read:    fixture.binding,
					Text:    "policy",
					Options: retrieval.RetrieveOptions{TopK: 10},
				},
			)
			if err != nil {
				t.Fatal(err)
			}
			first := protectedProcessor{
				run: func(_ context.Context, read access.Binding, rs retrieval.ResultSet[accessMeta]) (retrieval.ResultSet[accessMeta], error) {
					firstCalls++
					if read.Snapshot() != fixture.binding.Snapshot() {
						t.Fatal("processor binding replaced")
					}
					revoked = true
					if failure {
						return rs, ragy.ErrProtocol
					}
					return rs, nil
				},
			}
			next := protectedProcessor{
				run: func(_ context.Context, _ access.Binding, rs retrieval.ResultSet[accessMeta]) (retrieval.ResultSet[accessMeta], error) {
					nextCalls++
					return rs, nil
				},
			}
			chain := retrieval.NewPostProcessorChain[accessMeta](first, next)
			// Act.
			out, err := chain.Process(context.Background(), fixture.binding, retrieval.RetrieveOptions{TopK: 10}, rs)
			// Assert.
			if !access.IsProtectionFailure(err) || !out.IsEmpty() || firstCalls != 1 || nextCalls != 0 {
				t.Fatalf("revoked payload reached next stage: err=%v, calls=%d/%d", err, firstCalls, nextCalls)
			}
		})
	}
}

func TestProcessorReceivesDeadlineAndCancellationStopsChain(t *testing.T) {
	// Arrange.
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	deadline, _ := ctx.Deadline()
	nextCalls := 0
	rs := retrieval.NewResultSet([]retrieval.Document[accessMeta]{{ID: "a-public", Content: "policy"}}, nil)
	first := protectedProcessor{
		run: func(received context.Context, _ access.Binding, rs retrieval.ResultSet[accessMeta]) (retrieval.ResultSet[accessMeta], error) {
			got, ok := received.Deadline()
			if !ok || !got.Equal(deadline) {
				t.Fatal("processor deadline lost")
			}
			cancel()
			return rs, nil
		},
	}
	next := protectedProcessor{
		run: func(_ context.Context, _ access.Binding, rs retrieval.ResultSet[accessMeta]) (retrieval.ResultSet[accessMeta], error) {
			nextCalls++
			return rs, nil
		},
	}
	chain := retrieval.NewPostProcessorChain[accessMeta](first, next)
	// Act.
	out, err := chain.Process(ctx, retrieval.UnrestrictedRead(), retrieval.RetrieveOptions{TopK: 10}, rs)
	// Assert.
	if !errors.Is(err, context.Canceled) || !out.IsEmpty() || nextCalls != 0 {
		t.Fatalf("cancelled chain continued: %v, calls=%d", err, nextCalls)
	}
}
