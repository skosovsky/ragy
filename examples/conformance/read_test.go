package consumer_test

import (
	"context"
	"errors"
	"slices"
	"sync"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/contracttest"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/lexical"
	"github.com/skosovsky/ragy/retrieval"
)

const (
	publicID  = "a-public"
	privateID = "a-private"
	foreignID = "b-public"
)

type searchIntent struct{ Collection string }
type requestMeta struct{ Correlation string }
type sourceMeta struct {
	Organization string `json:"organization"`
	Access       string `json:"access"`
}

type payloadPort struct {
	mu           sync.Mutex
	clock        time.Time
	epoch        int64
	calls        int
	ids          []string
	revokeDuring bool
	deadline     time.Time
	hasDeadline  bool
}

func (p *payloadPort) now() time.Time { p.mu.Lock(); defer p.mu.Unlock(); return p.clock }
func (p *payloadPort) validate(_ context.Context, s access.Snapshot) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if s.PolicyEpoch != p.epoch {
		return ragy.ErrUnavailable
	}
	return nil
}
func (p *payloadPort) revoke() { p.mu.Lock(); defer p.mu.Unlock(); p.epoch = 8 }
func (p *payloadPort) expire() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.clock = p.clock.Add(31 * time.Second)
}
func (p *payloadPort) revokeOnRead() { p.mu.Lock(); defer p.mu.Unlock(); p.revokeDuring = true }
func (p *payloadPort) ioCount() int  { p.mu.Lock(); defer p.mu.Unlock(); return p.calls }
func (p *payloadPort) payloadIDs() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return slices.Clone(p.ids)
}
func (p *payloadPort) observedDeadline() (time.Time, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.deadline, p.hasDeadline
}

func (p *payloadPort) load(
	ctx context.Context,
	docs []retrieval.Document[sourceMeta],
) ([]retrieval.Document[sourceMeta], error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.calls++
	p.deadline, p.hasDeadline = ctx.Deadline()
	for _, doc := range docs {
		p.ids = append(p.ids, doc.ID)
	}
	if p.revokeDuring {
		p.epoch = 8
	}
	return slices.Clone(docs), nil
}

// adapter is a host-owned scoped lexical adapter. Only payloadPort materializes
// payloads for downstream delivery; the candidate index selects references first.
type adapter struct {
	index          *lexical.BM25Index[sourceMeta]
	schema         filter.Schema
	payload        *payloadPort
	postfilterOnly bool
}

func (a *adapter) Schema() filter.Schema { return a.schema }
func (*adapter) ReadCapabilities() access.Capabilities {
	return access.Capabilities{RequirePinnedPublication: false, ScopeProfile: true}
}

func (a *adapter) Retrieve(
	ctx context.Context,
	request retrieval.Request[searchIntent, requestMeta],
) (retrieval.ResultSet[sourceMeta], error) {
	empty := retrieval.NewResultSet[sourceMeta](nil, nil)
	if request.Intent.Collection != "contracts" || request.Meta.Correlation != "qa" {
		return empty, ragy.ErrInvalidArgument
	}
	prepared, err := retrieval.PrepareRead(ctx, request, a)
	if err != nil {
		return empty, err
	}
	candidateRequest := retrieval.Query[struct{}]{Read: prepared.Read, Text: prepared.Text, Options: prepared.Options}
	if a.postfilterOnly {
		candidateRequest.Read = retrieval.UnrestrictedRead()
		candidateRequest.Options.Filters = filter.Condition{}
	}
	candidates, err := a.index.Retrieve(ctx, candidateRequest)
	if err != nil {
		return empty, err
	}
	if gateErr := prepared.Read.Check(ctx); gateErr != nil {
		return empty, gateErr
	}
	docs, err := a.payload.load(ctx, candidates.Documents())
	if err != nil {
		return retrieval.DeliverRead(ctx, prepared.Read, empty, err, nil)
	}
	if a.postfilterOnly {
		filtered := make([]retrieval.Document[sourceMeta], 0, len(docs))
		for _, doc := range docs {
			allowed, matchErr := filter.MatchCondition(prepared.Options.Filters, func(field string) (any, bool) {
				switch field {
				case "organization":
					return doc.Meta.Organization, true
				case "access":
					return doc.Meta.Access, true
				default:
					return nil, false
				}
			})
			if matchErr != nil {
				return empty, matchErr
			}
			if allowed {
				filtered = append(filtered, doc)
			}
		}
		docs = filtered
	}
	return retrieval.DeliverRead(ctx, prepared.Read, retrieval.NewResultSet(docs, nil), nil, nil)
}

func newFixture(t *testing.T) contracttest.ScopedReadFixture[searchIntent, requestMeta, sourceMeta] {
	t.Helper()
	fields := filter.NewSchema()
	org, err := fields.String("organization")
	if err != nil {
		t.Fatal(err)
	}
	visibility, err := fields.String("access")
	if err != nil {
		t.Fatal(err)
	}
	schema, err := fields.Build()
	if err != nil {
		t.Fatal(err)
	}
	predicates, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatal(err)
	}
	mandatory, err := filter.In(filter.Eq(predicates, org, "a"), visibility, "public").Build()
	if err != nil {
		t.Fatal(err)
	}
	query, err := filter.NewBuilder(schema)
	if err != nil {
		t.Fatal(err)
	}
	conflict, err := filter.Eq(query, org, "b").Build()
	if err != nil {
		t.Fatal(err)
	}
	unsupported := foreignPredicate(t)
	payload := &payloadPort{clock: time.Unix(100, 0), epoch: 7}
	binding, err := access.Scoped(access.ScopedConfig{
		Snapshot: access.Snapshot{
			Identity:    "host-policy-7",
			PolicyEpoch: 7,
			IssuedAt:    payload.now(),
			ExpiresAt:   payload.now().Add(30 * time.Second),
		},
		Mandatory:   mandatory,
		Schema:      schema,
		Publication: access.CurrentPublication(),
		Authority:   access.AuthorityFunc(payload.validate),
		Now:         payload.now,
	})
	if err != nil {
		t.Fatal(err)
	}
	index, err := lexical.NewBM25Index[sourceMeta](
		schema,
		lexical.Config[sourceMeta]{SearchFields: []string{"content"}},
		nil,
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if indexErr := index.Index([]retrieval.Document[sourceMeta]{
		{ID: publicID, Content: "policy", Meta: sourceMeta{Organization: "a", Access: "public"}},
		{ID: privateID, Content: "policy", Meta: sourceMeta{Organization: "a", Access: "private"}},
		{ID: foreignID, Content: "policy", Meta: sourceMeta{Organization: "b", Access: "public"}},
	}); indexErr != nil {
		t.Fatal(indexErr)
	}
	backend := &adapter{index: index, schema: schema, payload: payload}
	return contracttest.ScopedReadFixture[searchIntent, requestMeta, sourceMeta]{
		Backend: backend,
		Request: retrieval.Request[searchIntent, requestMeta]{
			Read:    binding,
			Text:    "policy",
			Intent:  searchIntent{Collection: "contracts"},
			Meta:    requestMeta{Correlation: "qa"},
			Options: retrieval.RetrieveOptions{TopK: 10},
		},
		Conflict:            conflict,
		Unsupported:         unsupported,
		ExpectedIDs:         []string{publicID},
		ForbiddenPayloadIDs: []string{privateID, foreignID},
		IOCount:             payload.ioCount,
		PayloadIDs:          payload.payloadIDs,
		Revoke:              payload.revoke,
		Expire:              payload.expire,
		RevokeDuringIO:      payload.revokeOnRead,
		ObservedDeadline:    payload.observedDeadline,
	}
}
func foreignPredicate(t *testing.T) filter.Condition {
	t.Helper()
	fields := filter.NewSchema()
	field, err := fields.String("undeclared")
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
	condition, err := filter.Eq(builder, field, "value").Build()
	if err != nil {
		t.Fatal(err)
	}
	return condition
}

func TestExternalScopedAdapterConformance(t *testing.T) {
	contracttest.RunScopedReadSuite(t, newFixture)
}

func TestExternalSuiteRejectsPostfilterOnlyAdapter(t *testing.T) {
	// Arrange: this adapter declares scope but loads forbidden payload before filtering.
	fixture := newFixture(t)
	backend, ok := fixture.Backend.(*adapter)
	if !ok {
		t.Fatal("fixture adapter type")
	}
	backend.postfilterOnly = true
	// Act.
	violations := contracttest.CheckScopedReadCase(t.Context(), fixture, contracttest.ReadAllowed)
	// Assert: final allowed IDs alone cannot certify the adapter.
	if !slices.ContainsFunc(
		violations,
		func(v contracttest.ReadViolation) bool { return v.Code == "forbidden-payload-materialized" },
	) {
		t.Fatalf("postfilter adapter certified: %+v", violations)
	}
}

type undeclaredAdapter struct {
	calls int
	next  *adapter
}

func (a *undeclaredAdapter) Retrieve(
	ctx context.Context,
	req retrieval.Request[searchIntent, requestMeta],
) (retrieval.ResultSet[sourceMeta], error) {
	a.calls++
	return a.next.Retrieve(ctx, req)
}
func TestExternalSuiteRejectsUndeclaredAdapterBeforeIO(t *testing.T) {
	// Arrange: no capability declaration; no target call is authorized by admission.
	fixture := newFixture(t)
	backend, ok := fixture.Backend.(*adapter)
	if !ok {
		t.Fatal("fixture adapter type")
	}
	undeclared := &undeclaredAdapter{next: backend}
	fixture.Backend = undeclared
	// Act.
	violations := contracttest.CheckScopedReadCase(t.Context(), fixture, contracttest.ReadAllowed)
	// Assert.
	if len(violations) != 1 || violations[0].Code != "unsupported-admission" || undeclared.calls != 0 ||
		fixture.IOCount() != 0 {
		t.Fatalf("undeclared target dispatched: %+v, calls=%d", violations, undeclared.calls)
	}
}

func TestExternalPartialAdmissionPreservesScope(t *testing.T) {
	// Arrange: explicit partial composition around an undeclared optional target.
	fixture := newFixture(t)
	backend, ok := fixture.Backend.(*adapter)
	if !ok {
		t.Fatal("fixture adapter type")
	}
	undeclared := &undeclaredAdapter{next: backend}
	optional := retrieval.RequestPartialReadNode[searchIntent, requestMeta, sourceMeta, retrieval.NoExecutionMeta]{
		Child: retrieval.RequestBackendNode[searchIntent, requestMeta, sourceMeta, retrieval.NoExecutionMeta]{
			Backend: undeclared,
		},
		Name: "optional-target",
	}
	root := retrieval.RequestAggregateNode[searchIntent, requestMeta, sourceMeta, retrieval.NoExecutionMeta]{
		Nodes: []retrieval.RequestExecutionNode[searchIntent, requestMeta, sourceMeta, retrieval.NoExecutionMeta]{
			retrieval.RequestBackendNode[searchIntent, requestMeta, sourceMeta, retrieval.NoExecutionMeta]{
				Backend: backend,
			},
			optional,
		}, Concurrency: 2,
	}
	// Act.
	result, err := root.Execute(t.Context(), fixture.Request, retrieval.NoExecutionMeta{})
	// Assert.
	if err != nil || result.Len() != 1 || result.Documents()[0].ID != publicID || !result.Coverage.IsPartial() ||
		undeclared.calls != 0 {
		t.Fatalf("external partial scope failed: %v, %v", result.Documents(), err)
	}
	if !slices.Equal(result.Coverage.SkippedBranches(), []string{"optional-target"}) {
		t.Fatal("external partial coverage lost")
	}
	if errors.Is(err, ragy.ErrUnsupported) {
		t.Fatal("optional mismatch escaped as a runtime failure")
	}
}

type prematureIOAdapter struct{ next *adapter }

func (a prematureIOAdapter) Schema() filter.Schema { return a.next.schema }
func (prematureIOAdapter) ReadCapabilities() access.Capabilities {
	return access.Capabilities{RequirePinnedPublication: false, ScopeProfile: true}
}

func (a prematureIOAdapter) Retrieve(
	ctx context.Context,
	req retrieval.Request[searchIntent, requestMeta],
) (retrieval.ResultSet[sourceMeta], error) {
	if _, err := a.next.payload.load(ctx, nil); err != nil {
		return retrieval.NewResultSet[sourceMeta](nil, nil), err
	}
	return a.next.Retrieve(ctx, req)
}
func TestExternalSuiteDetectsMissingLeafGateDespiteEmptyDeniedOutput(t *testing.T) {
	// Arrange: output denial alone does not excuse payload I/O before leaf validation.
	fixture := newFixture(t)
	backend, ok := fixture.Backend.(*adapter)
	if !ok {
		t.Fatal("fixture adapter type")
	}
	fixture.Backend = prematureIOAdapter{next: backend}
	// Act.
	violations := contracttest.CheckScopedReadCase(t.Context(), fixture, contracttest.ReadMissingBinding)
	// Assert.
	if !slices.ContainsFunc(
		violations,
		func(v contracttest.ReadViolation) bool { return v.Code == "denied-payload-io" },
	) {
		t.Fatalf("missing leaf gate certified: %+v", violations)
	}
}
func TestExternalSuitePreservesEarlierParentDeadline(t *testing.T) {
	// Arrange: the suite must never lengthen an existing host deadline.
	fixture := newFixture(t)
	ctx, cancel := context.WithDeadline(t.Context(), time.Now().Add(time.Second))
	defer cancel()
	// Act.
	violations := contracttest.CheckScopedReadCase(ctx, fixture, contracttest.ReadDeadline)
	// Assert.
	if len(violations) > 0 {
		t.Fatalf("parent deadline lost: %+v", violations)
	}
}
