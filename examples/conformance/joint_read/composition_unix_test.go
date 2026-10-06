//go:build darwin || linux

package joint_test

import (
	"context"
	"errors"
	"io"
	"os"
	"sync"
	"sync/atomic"
	"testing"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/dense"
	densefs "github.com/skosovsky/ragy/dense/persistent"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/graph"
	graphmanaged "github.com/skosovsky/ragy/graph/managed"
	"github.com/skosovsky/ragy/lifecycle"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
	"github.com/skosovsky/ragy/tensor"
	tensorquery "github.com/skosovsky/ragy/tensor/query"
)

type readProbe struct {
	mu           sync.Mutex
	revoked      bool
	revokeDuring bool
	ids          []string
}

func (p *readProbe) authorize(context.Context, access.Snapshot) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.revoked {
		return ragy.ErrUnavailable
	}
	return nil
}
func (p *readProbe) ReadPayload(ctx context.Context, r lifecycle.PayloadRead) ([]byte, error) {
	file, err := os.Open(r.Path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = file.Close() }()
	data, err := io.ReadAll(io.LimitReader(file, r.MaxBytes+1))
	if err != nil {
		return nil, err
	}
	p.mu.Lock()
	p.ids = append(p.ids, r.Reference.Artifact)
	if p.revokeDuring {
		p.revoked = true
	}
	p.mu.Unlock()
	if int64(len(data)) > r.MaxBytes {
		return nil, ragy.ErrProtocol
	}
	return data, ctx.Err()
}

type mixedBackend struct {
	f               *fixture
	input           batch
	target          string
	calls           atomic.Int64
	extraFilter     filter.Condition
	ordinaryFailure bool
}

func (b *mixedBackend) Schema() filter.Schema { return b.f.schema }
func (b *mixedBackend) ReadCapabilities() access.Capabilities {
	return access.Capabilities{ScopeProfile: true, PinnedPublication: true, RequirePinnedPublication: true}
}
func (b *mixedBackend) Retrieve(ctx context.Context, q retrieval.Query[struct{}]) (retrieval.ResultSet[meta], error) {
	b.calls.Add(1)
	if b.ordinaryFailure {
		return retrieval.NewResultSet[meta](nil, nil), errors.New("injected ordinary target failure")
	}
	if !filter.IsEmpty(b.extraFilter.IR()) {
		effective, err := filter.Intersect(b.f.schema, q.Options.Filters, b.extraFilter)
		if err != nil {
			return nil, err
		}
		q.Options.Filters = effective
	}

	switch b.target {
	case "dense":
		return b.f.dense.Retrieve(
			ctx,
			retrieval.Query[densefs.Intent]{
				Read: q.Read,
				Text: q.Text,
				Plan: retrieval.ProjectPlannedQuery(
					q.Plan,
					densefs.Intent{Embedding: dense.Embedding{Space: denseSpace(), Vector: []float32{1, 0}}},
				),
				Intent:  densefs.Intent{Embedding: dense.Embedding{Space: denseSpace(), Vector: []float32{1, 0}}},
				Options: q.Options,
			},
		)
	case "lexical":
		return b.f.secondary.lexical.Retrieve(ctx, q)
	case "tensor":
		var refs []source.Reference
		for _, r := range b.input.Tensor {
			refs = append(refs, r.Reference)
		}
		intent := tensorquery.Intent{
			Embedding:       tensor.Embedding{Space: tensorSpace(), Tokens: tensor.Tensor{{1, 0}}},
			Candidates:      refs,
			CandidateBudget: 100,
		}
		return b.f.secondary.tensor.Retrieve(
			ctx,
			retrieval.Query[tensorquery.Intent]{
				Read:    q.Read,
				Text:    q.Text,
				Plan:    retrieval.ProjectPlannedQuery(q.Plan, intent),
				Intent:  intent,
				Options: q.Options,
			},
		)
	default:
		backend, err := graphmanaged.NewBackend(
			graphmanaged.BackendConfig[meta]{Adapter: b.f.secondary.graph, MaxNodes: 50, MaxEdges: 100},
		)
		if err != nil {
			return nil, err
		}
		q.Options.Graph = &retrieval.GraphOptions{
			Seeds:     []string{"allowed", "private", "foreign"},
			Direction: graph.DirectionOutbound,
			Depth:     2,
		}
		return backend.Retrieve(ctx, q)
	}
}
func jointCorpus() batch {
	input := sourceBatch("policy", "r1", []string{"allowed", "private", "foreign"})
	for i := range input.Dense {
		m := input.Dense[i].Value.Meta
		if i == 1 {
			m.Visibility = "private"
		}
		if i == 2 {
			m.Tenant = "b"
		}
		input.Dense[i].Value.Meta = m
		input.Tensor[i].Value.Meta = m
		input.Lexical[i].Document.Meta = m
		input.Graph.Nodes[i].Value.Meta = m
	}
	return input
}

type mixedNode = retrieval.ExecutionNode[struct{}, meta, retrieval.NoExecutionMeta]

func mixedNodes(
	t *testing.T,
	target string,
) (*fixture, access.Binding, *mixedBackend, *mixedBackend, mixedNode) {
	t.Helper()
	f := newFixture(t, target)
	input := jointCorpus()
	f.ingest(t, plan("joint", "", target, input), input)
	read := f.pin(t)
	d := &mixedBackend{f: f, input: input, target: "dense"}
	other := &mixedBackend{f: f, input: input, target: target}
	denseNode := retrieval.BackendNode[struct{}, meta, retrieval.NoExecutionMeta]{Backend: d, Name: "dense"}
	secondNode := retrieval.BackendNode[struct{}, meta, retrieval.NoExecutionMeta]{Backend: other, Name: target}
	merger, err := retrieval.NewReciprocalRankFusion[meta](60, nil)
	if err != nil {
		t.Fatal(err)
	}
	root := retrieval.RequestExecutionAggregateNode[struct{}, retrieval.NoRequestMeta, meta, retrieval.NoExecutionMeta]{
		Nodes:       []mixedNode{denseNode, secondNode},
		Concurrency: 1,
		Merger:      merger,
		Name:        "joint",
	}
	return f, read, d, other, root
}
func TestActualExternalJointCompositionScope(t *testing.T) {
	for _, target := range []string{"lexical", "tensor", "graph"} {
		for _, path := range []string{"aggregate", "fallback", "rescue", "route"} {
			t.Run(target+"/"+path, func(t *testing.T) {
				// Arrange: common durable publication, mixed BYOT targets and forbidden stored records.
				_, read, _, _, root := mixedNodes(t, target)
				node := wrappedComposition(t, path, root)
				request := retrieval.Query[struct{}]{
					Read:    read,
					Text:    "needle",
					Options: retrieval.RetrieveOptions{TopK: 10},
				}
				// Act.
				result, err := node.Execute(t.Context(), request, retrieval.NoExecutionMeta{})
				// Assert: every returned record is admitted before source projection/fusion.
				if err != nil || result.IsEmpty() {
					t.Fatal("joint composition failed", err)
				}
				for _, doc := range result.Documents() {
					if doc.Meta.Tenant != "a" || doc.Meta.Visibility != "public" || doc.Meta.Artifact != "allowed" {
						t.Fatal("forbidden joint payload", doc.ID)
					}
				}
			})
		}
	}
}
func wrappedComposition(t *testing.T, path string, root mixedNode) mixedNode {
	t.Helper()
	switch path {
	case "fallback":
		return retrieval.FallbackNode[struct{}, meta, retrieval.NoExecutionMeta]{
			Primary:   root,
			Secondary: root,
			Name:      "fallback",
		}
	case "rescue":
		return retrieval.RescueNode[struct{}, meta, retrieval.NoExecutionMeta]{
			Primary:   root,
			Secondary: root,
			Name:      "rescue",
		}
	case "route":
		planner := retrieval.RoutePlannerFunc[struct{}, string, struct{}](
			func(context.Context, retrieval.Query[struct{}]) (retrieval.RouteDecision[string, struct{}], error) {
				return retrieval.RouteDecision[string, struct{}]{Route: "joint"}, nil
			},
		)
		node, err := retrieval.NewRouteSwitchBuilder[struct{}, string, struct{}, meta, retrieval.NoExecutionMeta](
			planner,
		).Case("joint", root).
			Default(root).
			Build()
		if err != nil {
			t.Fatal(err)
		}
		return node
	default:
		return root
	}
}
func TestJointRevocationDuringActualPayloadIO(t *testing.T) {
	for _, target := range []string{"lexical", "tensor", "graph"} {
		t.Run(target, func(t *testing.T) {
			// Arrange: revoke only after actual dense payload bytes have been read.
			f, read, _, second, root := mixedNodes(t, target)
			f.probe.revokeDuring = true
			request := retrieval.Query[struct{}]{
				Read:    read,
				Text:    "needle",
				Options: retrieval.RetrieveOptions{TopK: 10},
			}
			// Act.
			result, err := root.Execute(t.Context(), request, retrieval.NoExecutionMeta{})
			// Assert: the revoked payload is suppressed; no secondary downstream dispatch.
			if !errors.Is(err, ragy.ErrUnavailable) || !access.IsProtectionFailure(err) || !result.IsEmpty() ||
				second.calls.Load() != 0 {
				t.Fatal("joint revocation bypassed", err)
			}
			f.probe.mu.Lock()
			defer f.probe.mu.Unlock()
			if len(f.probe.ids) != 1 || f.probe.ids[0] != "allowed" {
				t.Fatal("forbidden payload I/O", f.probe.ids)
			}
		})
	}
}

type unsupportedBackend struct {
	schema filter.Schema
	calls  atomic.Int64
}

func (b *unsupportedBackend) Schema() filter.Schema { return b.schema }
func (*unsupportedBackend) ReadCapabilities() access.Capabilities {
	return access.Capabilities{ScopeProfile: false, PinnedPublication: false}
}
func (b *unsupportedBackend) Retrieve(context.Context, retrieval.Query[struct{}]) (retrieval.ResultSet[meta], error) {
	b.calls.Add(1)
	return nil, ragy.ErrProtocol
}

func TestExternalJointPlannerAndCapabilityNegotiation(t *testing.T) {
	for _, target := range []string{"lexical", "tensor", "graph"} {
		for _, profile := range []string{"foreign-plan", "empty-plan", "unsupported-plan", "strict-unsupported", "explicit-partial"} {
			t.Run(target+"/"+profile, func(t *testing.T) {
				testJointNegotiation(t, target, profile)
			})
		}
	}
}
func testJointNegotiation(t *testing.T, target, profile string) {
	t.Helper()
	// Arrange: actual joint targets; the extra adapter explicitly lacks scoped capability.
	f, read, denseBackend, second, root := mixedNodes(t, target)
	rejected := &unsupportedBackend{schema: f.schema}
	plannerCalls := 0
	condition := jointPlanCondition(t, f.schema, profile)
	if profile == "strict-unsupported" || profile == "explicit-partial" {
		missing := mixedNode(
			retrieval.BackendNode[struct{}, meta, retrieval.NoExecutionMeta]{Backend: rejected, Name: "unsupported"},
		)
		if profile == "explicit-partial" {
			missing = retrieval.PartialReadNode[struct{}, meta, retrieval.NoExecutionMeta]{
				Child: missing,
				Name:  "unsupported",
			}
		}
		merger, err := retrieval.NewReciprocalRankFusion[meta](60, nil)
		if err != nil {
			t.Fatal(err)
		}
		root = retrieval.RequestExecutionAggregateNode[struct{}, retrieval.NoRequestMeta, meta, retrieval.NoExecutionMeta]{
			Nodes:       []mixedNode{root, missing},
			Concurrency: 1,
			Merger:      merger,
			Name:        "mixed-negotiation",
		}
	}
	root = wrappedComposition(
		t,
		"route",
		retrieval.RescueNode[struct{}, meta, retrieval.NoExecutionMeta]{Primary: root, Secondary: root, Name: "rescue"},
	)
	planner := retrieval.QueryPlannerFunc[struct{}, retrieval.NoRequestMeta](
		func(context.Context, retrieval.Query[struct{}]) (retrieval.PlannedQuery[struct{}], error) {
			plannerCalls++
			return retrieval.PlannedQuery[struct{}]{Text: "needle", Filters: condition}, nil
		},
	)
	pipeline, err := retrieval.NewExecutionPipelineBuilder[struct{}, meta, retrieval.NoExecutionMeta]().WithRoot(root).
		WithPlanner(planner).
		Build()
	if err != nil {
		t.Fatal(err)
	}
	// Act: nested route/rescue/aggregate preflights the complete reachable tree.
	result, err := pipeline.Execute(
		t.Context(),
		retrieval.Query[struct{}]{Read: read, Text: "needle", Options: retrieval.RetrieveOptions{TopK: 10}},
	)
	// Assert: no scope escape or disguised complete result from partial negotiation.
	assertJointNegotiation(t, profile, f, denseBackend, second, plannerCalls, result, err)
	if rejected.calls.Load() != 0 {
		t.Fatal("unsupported branch dispatched")
	}
}

func assertJointNegotiation(
	t *testing.T,
	profile string,
	f *fixture,
	denseBackend, second *mixedBackend,
	plannerCalls int,
	result retrieval.RetrievalResult[meta, retrieval.NoExecutionMeta],
	err error,
) {
	t.Helper()
	switch profile {
	case "foreign-plan":
		if err != nil || !result.IsEmpty() {
			t.Fatal("foreign planner expanded mandatory scope", err)
		}
		f.probe.mu.Lock()
		defer f.probe.mu.Unlock()
		if len(f.probe.ids) != 0 {
			t.Fatal("foreign plan loaded dense payload", f.probe.ids)
		}
	case "unsupported-plan", "strict-unsupported":
		assertJointUnsupported(t, profile, denseBackend, second, plannerCalls, result, err)
	default:
		assertJointAdmitted(t, profile, result, err)
	}
}

func assertJointUnsupported(
	t *testing.T,
	profile string,
	denseBackend, second *mixedBackend,
	plannerCalls int,
	result retrieval.RetrievalResult[meta, retrieval.NoExecutionMeta],
	err error,
) {
	t.Helper()
	if !errors.Is(err, ragy.ErrUnsupported) || !result.IsEmpty() {
		t.Fatal("unsupported joint profile delivered payload", err)
	}
	if denseBackend.calls.Load() != 0 || second.calls.Load() != 0 {
		t.Fatal("unsupported profile reached targets")
	}
	if profile == "strict-unsupported" && plannerCalls != 0 {
		t.Fatal("unsupported branch reached planning")
	}
}

func assertJointAdmitted(
	t *testing.T,
	profile string,
	result retrieval.RetrievalResult[meta, retrieval.NoExecutionMeta],
	err error,
) {
	t.Helper()
	if err != nil || result.IsEmpty() {
		t.Fatal("admitted joint profile failed", err)
	}
	for _, doc := range result.Documents() {
		if doc.Meta.Tenant != "a" || doc.Meta.Visibility != "public" {
			t.Fatal("joint profile returned forbidden metadata")
		}
	}
	if profile == "explicit-partial" && result.Coverage.State() != retrieval.CoveragePartial {
		t.Fatal("partial negotiation disguised as complete", result.Coverage)
	}
}

func jointPlanCondition(t *testing.T, schema filter.Schema, profile string) filter.Condition {
	t.Helper()
	if profile != "foreign-plan" && profile != "unsupported-plan" {
		return filter.Condition{}
	}
	if profile == "unsupported-plan" {
		fields := filter.NewSchema()
		field, err := fields.String("undeclared")
		if err != nil {
			t.Fatal(err)
		}
		other, err := fields.Build()
		if err != nil {
			t.Fatal(err)
		}
		builder, err := filter.NewBuilder(other)
		if err != nil {
			t.Fatal(err)
		}
		condition, err := filter.Eq(builder, field, "x").Build()
		if err != nil {
			t.Fatal(err)
		}
		return condition
	}
	field, err := schema.StringField("tenant")
	if err != nil {
		t.Fatal(err)
	}
	builder, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatal(err)
	}
	condition, err := filter.Eq(builder, field, "b").Build()
	if err != nil {
		t.Fatal(err)
	}
	return condition
}

func TestActualJointFallbackAndRescueSecondaryDispatch(t *testing.T) {
	for _, target := range []string{"lexical", "tensor", "graph"} {
		for _, path := range []string{"fallback", "rescue"} {
			t.Run(target+"/"+path, func(t *testing.T) { testJointSecondaryDispatch(t, target, path) })
		}
	}
}
func testJointSecondaryDispatch(t *testing.T, target, path string) {
	t.Helper()
	// Arrange: empty admitted primary or ordinary failed primary activates real secondary.
	f, read, d, other, _ := mixedNodes(t, target)
	if path == "fallback" {
		d.extraFilter = jointPlanCondition(t, f.schema, "foreign-plan")
	} else {
		d.ordinaryFailure = true
	}
	primary := retrieval.BackendNode[struct{}, meta, retrieval.NoExecutionMeta]{Backend: d, Name: "dense"}
	secondary := retrieval.BackendNode[struct{}, meta, retrieval.NoExecutionMeta]{
		Backend: other,
		Name:    target,
	}
	var root mixedNode = retrieval.FallbackNode[struct{}, meta, retrieval.NoExecutionMeta]{Primary: primary, Secondary: secondary, Name: "fallback"}
	if path == "rescue" {
		root = retrieval.RescueNode[struct{}, meta, retrieval.NoExecutionMeta]{
			Primary:   primary,
			Secondary: secondary,
			Name:      "rescue",
		}
	}
	// Act.
	result, err := root.Execute(
		t.Context(),
		retrieval.Query[struct{}]{Read: read, Text: "needle", Options: retrieval.RetrieveOptions{TopK: 10}},
		retrieval.NoExecutionMeta{},
	)
	// Assert: actual secondary is called once with the original mandatory publication binding.
	if err != nil || result.IsEmpty() || d.calls.Load() != 1 || other.calls.Load() != 1 {
		t.Fatal("joint secondary dispatch failed", err)
	}
	for _, doc := range result.Documents() {
		if doc.Meta.Tenant != "a" || doc.Meta.Visibility != "public" {
			t.Fatal("secondary escaped scope")
		}
	}
}
