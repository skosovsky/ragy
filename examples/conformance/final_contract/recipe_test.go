package final_test

import (
	"context"
	"errors"
	"slices"
	"testing"
	"time"

	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/lexical"
	"github.com/skosovsky/ragy/recipe"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

// These are consumer-owned types, intentionally distinct from backend query types.
type recipeIntent struct{ Tasks []string }
type recipeRequestMeta struct{ Trace []string }
type recipeDocMeta struct {
	Tenant string   `json:"tenant"`
	Labels []string `json:"-"`
}
type recipeRequest = retrieval.Request[recipeIntent, recipeRequestMeta]
type recipeFixture struct {
	config                      recipe.Config[recipeIntent, recipeRequestMeta, recipeDocMeta]
	request                     recipeRequest
	planned                     []string
	selected                    []int
	retrieved                   []string
	plannerCalls, assessorCalls int
	cancelInPlanner             context.CancelFunc
}

func recipeCloneDoc(m recipeDocMeta) (recipeDocMeta, error) {
	m.Labels = slices.Clone(m.Labels)
	return m, nil
}
func recipeObservedUsage() recipe.Usage {
	return recipe.Usage{Known: true, Value: budget.Usage{InputTokens: 8, OutputTokens: 2, Cost: 3}}
}
func recipeLimits() budget.Limits {
	return budget.Limits{
		RetrievalCalls: 6,
		ModelCalls:     4,
		Usage:          budget.Usage{InputTokens: 256, OutputTokens: 64, Cost: 100},
	}
}
func recipeLedger(t *testing.T, limits budget.Limits) *budget.Ledger {
	t.Helper()
	ledger, err := budget.New(
		budget.Config{Limits: limits, Deadline: time.Now().Add(time.Minute), Now: time.Now, RequireKnownCost: true},
	)
	if err != nil {
		t.Fatal(err)
	}
	return ledger
}
func recipeReadBinding(t *testing.T) (filter.Schema, access.Binding) {
	t.Helper()
	fields := filter.NewSchema()
	tenant, err := fields.String("tenant")
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
	mandatory, err := filter.Eq(builder, tenant, "allowed").Build()
	if err != nil {
		t.Fatal(err)
	}
	pub, err := access.PinPublication(
		"recipe-pub",
		[]access.TargetRevision{
			{
				Target:            "lexical",
				Namespace:         "recipe",
				Source:            "manual",
				Revision:          "r1",
				Transformation:    "original",
				AccessFingerprint: "acl",
			},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	read, err := access.Scoped(access.ScopedConfig{
		Snapshot: access.Snapshot{
			Identity:    "recipe-scope",
			PolicyEpoch: 1,
			IssuedAt:    now,
			ExpiresAt:   now.Add(time.Minute),
		},
		Mandatory:   mandatory,
		Schema:      schema,
		Publication: pub,
		Now:         time.Now,
		Authority:   access.AuthorityFunc(func(ctx context.Context, _ access.Snapshot) error { return ctx.Err() }),
	})
	if err != nil {
		t.Fatal(err)
	}
	return schema, read
}

func recipeBackend(
	t *testing.T,
	f *recipeFixture,
	schema filter.Schema,
	read access.Binding,
) retrieval.ProjectedBackend[recipeIntent, recipeRequestMeta, struct{}, retrieval.NoRequestMeta, recipeDocMeta] {
	t.Helper()
	documents := []retrieval.Document[recipeDocMeta]{
		{
			ID:      "refund",
			Content: "refund payment ten days",
			Meta:    recipeDocMeta{Tenant: "allowed", Labels: []string{"owned"}},
		},
		{
			ID:      "password",
			Content: "reset recover password",
			Meta:    recipeDocMeta{Tenant: "allowed", Labels: []string{"owned"}},
		},
		{
			ID:      "card",
			Content: "card payment supported",
			Meta:    recipeDocMeta{Tenant: "allowed", Labels: []string{"owned"}},
		},
		{
			ID:      "private",
			Content: "refund reset recover card private payload",
			Meta:    recipeDocMeta{Tenant: "denied", Labels: []string{"private"}},
		},
	}
	index, err := lexical.NewBM25Snapshot(
		context.Background(),
		schema,
		lexical.Config[recipeDocMeta]{SearchFields: []string{"content"}},
		read,
		documents,
		recipeCloneDoc,
	)
	if err != nil {
		t.Fatal(err)
	}
	project := func(req recipeRequest) retrieval.Query[struct{}] {
		return retrieval.Query[struct{}]{
			Read:    req.Read,
			Text:    req.Text,
			Options: req.Options,
			Plan:    retrieval.ProjectPlannedQuery(req.Plan, struct{}{}),
		}
	}
	backend := retrieval.ProjectedBackend[recipeIntent, recipeRequestMeta, struct{}, retrieval.NoRequestMeta, recipeDocMeta]{
		Next:             index,
		AdmissionProject: project,
		Project: func(req recipeRequest) retrieval.Query[struct{}] {
			if !slices.Equal(req.Intent.Tasks, []string{"answer"}) ||
				!slices.Equal(req.Meta.Trace, []string{"consumer"}) {
				t.Fatal("BYOT request changed during projection")
			}
			f.retrieved = append(f.retrieved, req.EffectiveText())
			return project(req)
		},
	}
	return backend
}

func recipeAssertAssessorScope(t *testing.T, queries []recipe.QueryEvidence[recipeDocMeta]) {
	t.Helper()
	for _, query := range queries {
		for _, doc := range query.Documents {
			if doc.ID == "private" || doc.Meta.Tenant != "allowed" {
				t.Fatal("private evidence reached host assessor")
			}
		}
	}
}

func recipeNewFixture(t *testing.T, strategy recipe.Strategy) *recipeFixture {
	t.Helper()
	f := &recipeFixture{}
	schema, read := recipeReadBinding(t)
	backend := recipeBackend(t, f, schema, read)
	maxQueries := 2
	if strategy == recipe.SingleRewrite {
		maxQueries = 1
	}
	if strategy == recipe.Decomposition {
		maxQueries = 3
	}
	f.config = recipe.Config[recipeIntent, recipeRequestMeta, recipeDocMeta]{
		Strategy:         strategy,
		Revision:         "consumer-bounded-v1",
		BackendModelFree: true,
		Backend:          backend,
		Identity:         retrieval.DocumentIDResolver[recipeDocMeta]{},
		Admission: func(ctx context.Context, req recipeRequest) (retrieval.ReadCoverage, error) {
			_, err := retrieval.PrepareRead(ctx, req, backend)
			return retrieval.CompleteReadCoverage(), err
		},
		Planner: func(ctx context.Context, req recipeRequest, limits recipe.ModelLimits) (recipe.Planning, error) {
			f.plannerCalls++
			if limits != (recipe.ModelLimits{InputTokens: 32, OutputTokens: 8}) {
				t.Fatal("planner did not receive reserved maxima", limits)
			}
			if !slices.Equal(req.Intent.Tasks, []string{"answer"}) ||
				!slices.Equal(req.Meta.Trace, []string{"consumer"}) {
				t.Fatal("planner lost BYOT input")
			}
			if f.cancelInPlanner != nil {
				f.cancelInPlanner()
				return recipe.Planning{Usage: recipeObservedUsage()}, ctx.Err()
			}
			return recipe.Planning{Queries: slices.Clone(f.planned), Usage: recipeObservedUsage()}, nil
		},
		Assessor: func(_ context.Context, input recipe.AssessmentInput[recipeIntent, recipeRequestMeta, recipeDocMeta], limits recipe.ModelLimits) (recipe.Assessment, error) {
			f.assessorCalls++
			if limits != (recipe.ModelLimits{InputTokens: 32, OutputTokens: 8}) {
				t.Fatal("assessor did not receive reserved maxima", limits)
			}
			recipeAssertAssessorScope(t, input.Queries)
			return recipe.Assessment{
				Selected:   slices.Clone(f.selected),
				Sufficient: true,
				Usage:      recipeObservedUsage(),
			}, nil
		},
		Pricing: func(_ context.Context, op recipe.Operation) (recipe.Quote, error) {
			if op == recipe.Retrieve {
				return recipe.Quote{CostKnown: true}, nil
			}
			return recipe.Quote{CostKnown: true, Usage: budget.Usage{InputTokens: 32, OutputTokens: 8, Cost: 5}}, nil
		},
		CloneIntent:      func(i recipeIntent) (recipeIntent, error) { i.Tasks = slices.Clone(i.Tasks); return i, nil },
		CloneRequestMeta: func(m recipeRequestMeta) (recipeRequestMeta, error) { m.Trace = slices.Clone(m.Trace); return m, nil },
		CloneMeta:        recipeCloneDoc,
		Supports: func(_ context.Context, _ access.Binding, doc retrieval.Document[recipeDocMeta]) ([]source.Locator, error) {
			return []source.Locator{
				{
					Kind: source.DocumentLocation,
					Reference: source.Reference{
						Namespace:         "recipe",
						Source:            "manual",
						Revision:          "r1",
						Transformation:    "original",
						AccessFingerprint: "acl",
						Artifact:          doc.ID,
						Representation:    "text",
					},
				},
			}, nil
		},
		Limits:           recipeLimits(),
		RequireKnownCost: true,
		Duration:         5 * time.Second,
		Now:              time.Now,
		MaxQueries:       maxQueries,
		MaxDocuments:     10,
		FusionK:          60,
	}
	f.request = recipeRequest{
		Read:    read,
		Text:    "refund",
		Intent:  recipeIntent{Tasks: []string{"answer"}},
		Meta:    recipeRequestMeta{Trace: []string{"consumer"}},
		Options: retrieval.RetrieveOptions{TopK: 3},
	}
	return f
}

func (f *recipeFixture) run(
	ctx context.Context,
	t *testing.T,
	ledger *budget.Ledger,
) (recipe.Result[recipeDocMeta], error) {
	t.Helper()
	r, err := recipe.New(f.config)
	if err != nil {
		t.Fatal(err)
	}
	return r.Run(ctx, f.request, ledger)
}

func recipeAssertEvidence(t *testing.T, result recipe.Result[recipeDocMeta], wantedIDs []string) {
	t.Helper()
	ids := make([]string, 0, len(result.Selected))
	for _, selected := range result.Selected {
		ids = append(ids, selected.Document.ID)
		if selected.Document.Meta.Tenant != "allowed" || len(selected.Contributors) == 0 {
			t.Fatal("scope or provenance lost", selected)
		}
		for _, contributor := range selected.Contributors {
			if len(contributor.Supports) != 1 || contributor.Supports[0].Reference.Artifact != selected.Document.ID {
				t.Fatal("source support lost", contributor)
			}
		}
	}
	slices.Sort(ids)
	want := slices.Clone(wantedIDs)
	slices.Sort(want)
	if !slices.Equal(ids, want) {
		t.Fatal(ids, want)
	}
}

func recipeAssertStrategy(t *testing.T, strategy recipe.Strategy, result recipe.Result[recipeDocMeta]) {
	t.Helper()
	if strategy == recipe.MultiQuery && len(result.Selected[0].Contributors) != 2 {
		t.Fatal("BM25 duplicate lost query contributors")
	}
	if strategy == recipe.Decomposition {
		if len(result.Coverage) != 2 {
			t.Fatal(result.Coverage)
		}
		for _, coverage := range result.Coverage {
			if !coverage.Retrieved || !coverage.SelectedEvidence || !coverage.DeliveredEvidence {
				t.Fatal(coverage)
			}
		}
	}
}

func TestRecipeActualScopedBM25Strategies(t *testing.T) {
	cases := []struct {
		strategy recipe.Strategy
		queries  []string
		selected []int
		calls    []string
		ids      []string
	}{
		{recipe.SingleRewrite, []string{"refund"}, []int{1}, []string{"refund", "refund"}, []string{"refund"}},
		{
			recipe.MultiQuery,
			[]string{"reset", "recover"},
			[]int{1, 2},
			[]string{"refund", "reset", "recover"},
			[]string{"password"},
		},
		{
			recipe.Decomposition,
			[]string{"refund", "card"},
			[]int{0, 1},
			[]string{"refund", "card"},
			[]string{"refund", "card"},
		},
	}
	for _, tc := range cases {
		t.Run(string(tc.strategy), func(t *testing.T) {
			// Arrange: real BM25 snapshot with a forbidden document and host-scripted model ports.
			f := recipeNewFixture(t, tc.strategy)
			f.planned, f.selected = tc.queries, tc.selected
			ledger := recipeLedger(t, recipeLimits())
			// Act.
			result, err := f.run(context.Background(), t, ledger)
			// Assert: actual corpus evidence, dispatched texts, provenance and aggregate usage.
			if err != nil || result.Outcome != recipe.Complete || result.Stop != recipe.Assessed {
				t.Fatal(result, err)
			}
			if !slices.Equal(f.retrieved, tc.calls) || f.plannerCalls != 1 || f.assessorCalls != 1 {
				t.Fatal(f.retrieved, f.plannerCalls, f.assessorCalls)
			}
			recipeAssertEvidence(t, result, tc.ids)
			recipeAssertStrategy(t, tc.strategy, result)
			snapshot := ledger.Snapshot()
			if snapshot.Occupied.RetrievalCalls != uint64(len(tc.calls)) || snapshot.Occupied.ModelCalls != 2 ||
				snapshot.Actual != (budget.Usage{InputTokens: 16, OutputTokens: 4, Cost: 6}) ||
				snapshot.Outstanding != 0 ||
				result.Budget != snapshot ||
				result.Publication != "recipe-pub" {
				t.Fatal(snapshot, result.Budget)
			}
		})
	}
}

func recipeSpendModelCall(t *testing.T, ledger *budget.Ledger) {
	t.Helper()
	lease, err := ledger.Reserve(context.Background(), budget.Reservation{Kind: budget.Model, CostKnown: true})
	if err != nil {
		t.Fatal(err)
	}
	if settleErr := lease.Settle(budget.Usage{}, true); settleErr != nil {
		t.Fatal(settleErr)
	}
}

func TestRecipeSharedLedgerRefusesBeforeHostModelDispatch(t *testing.T) {
	for _, stage := range []string{"planner", "assessor"} {
		t.Run(stage, func(t *testing.T) {
			// Arrange: another branch has already spent one model call from the same ledger.
			f := recipeNewFixture(t, recipe.SingleRewrite)
			f.planned, f.selected = []string{"refund"}, []int{1}
			limits := recipeLimits()
			limits.ModelCalls = 1
			if stage == "assessor" {
				limits.ModelCalls = 2
			}
			ledger := recipeLedger(t, limits)
			recipeSpendModelCall(t, ledger)
			// Act.
			result, err := f.run(context.Background(), t, ledger)
			// Assert: refusal precedes dispatch of the exhausted host model port.
			wantPlanner, wantRetrieval := 0, 1
			if stage == "assessor" {
				wantPlanner, wantRetrieval = 1, 2
			}
			if err != nil || result.Stop != recipe.BudgetExhausted || result.Outcome != recipe.Partial ||
				result.Sufficiency != nil ||
				f.plannerCalls != wantPlanner ||
				f.assessorCalls != 0 ||
				len(f.retrieved) != wantRetrieval {
				t.Fatal(result, err, f.plannerCalls, f.assessorCalls, f.retrieved)
			}
			if ledger.Snapshot().Occupied.ModelCalls != limits.ModelCalls || ledger.Snapshot().Outstanding != 0 {
				t.Fatal(ledger.Snapshot())
			}
		})
	}
}

func TestRecipeModelTokenAndCostRefusalPrecedesPlanner(t *testing.T) {
	for _, dimension := range []string{"input", "output", "cost"} {
		t.Run(dimension, func(t *testing.T) {
			// Arrange: enough calls, but insufficient reservation capacity in one dimension.
			f := recipeNewFixture(t, recipe.SingleRewrite)
			limits := recipeLimits()
			switch dimension {
			case "input":
				limits.Usage.InputTokens = 31
			case "output":
				limits.Usage.OutputTokens = 7
			case "cost":
				limits.Usage.Cost = 4
			}
			ledger := recipeLedger(t, limits)
			// Act.
			result, err := f.run(context.Background(), t, ledger)
			// Assert.
			if err != nil || result.Stop != recipe.BudgetExhausted || f.plannerCalls != 0 || f.assessorCalls != 0 ||
				!slices.Equal(f.retrieved, []string{"refund"}) ||
				ledger.Snapshot().Occupied.ModelCalls != 0 {
				t.Fatal(result, err, f.plannerCalls, f.assessorCalls)
			}
		})
	}
}

func TestRecipeCanceledAttemptSuppressesPayloadAndFurtherDispatch(t *testing.T) {
	for _, stage := range []string{"before-run", "planner"} {
		t.Run(stage, func(t *testing.T) {
			// Arrange.
			f := recipeNewFixture(t, recipe.SingleRewrite)
			f.planned, f.selected = []string{"refund"}, []int{1}
			ledger := recipeLedger(t, recipeLimits())
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if stage == "before-run" {
				cancel()
			} else {
				f.cancelInPlanner = cancel
			}
			// Act.
			result, err := f.run(ctx, t, ledger)
			// Assert: canceled outputs never become deliverable evidence or assessment.
			wantPlanner, wantRetrieval := 0, 0
			if stage == "planner" {
				wantPlanner, wantRetrieval = 1, 1
			}
			if !errors.Is(err, context.Canceled) || len(result.Selected) != 0 || len(result.Queries) != 0 ||
				len(result.Stages) != 0 ||
				f.plannerCalls != wantPlanner ||
				f.assessorCalls != 0 ||
				len(f.retrieved) != wantRetrieval {
				t.Fatal(result, err, f.plannerCalls, f.assessorCalls, f.retrieved)
			}
			if ledger.Snapshot().Occupied.ModelCalls != uint64(wantPlanner) || ledger.Snapshot().Outstanding != 0 {
				t.Fatal("cancellation refunded a dispatched call", ledger.Snapshot())
			}
		})
	}
}

func TestRecipeTwoAttemptsShareActualUsageAndCallCaps(t *testing.T) {
	// Arrange: two bounded recipes compose with a caller-owned ledger.
	first := recipeNewFixture(t, recipe.SingleRewrite)
	second := recipeNewFixture(t, recipe.SingleRewrite)
	first.planned, first.selected = []string{"refund"}, []int{1}
	second.planned, second.selected = []string{"refund"}, []int{1}
	limits := recipeLimits()
	limits.ModelCalls = 3
	ledger := recipeLedger(t, limits)
	// Act.
	complete, firstErr := first.run(context.Background(), t, ledger)
	partial, secondErr := second.run(context.Background(), t, ledger)
	// Assert: the second attempt cannot reset spent capacity or dispatch its assessor.
	if firstErr != nil || secondErr != nil || complete.Outcome != recipe.Complete ||
		partial.Outcome != recipe.Partial ||
		partial.Stop != recipe.BudgetExhausted ||
		partial.Sufficiency != nil {
		t.Fatal(complete, partial, firstErr, secondErr)
	}
	if first.plannerCalls != 1 || first.assessorCalls != 1 || second.plannerCalls != 1 || second.assessorCalls != 0 ||
		len(first.retrieved) != 2 ||
		len(second.retrieved) != 2 {
		t.Fatal("shared ledger allowed unexpected dispatch")
	}
	snapshot := ledger.Snapshot()
	if snapshot.Occupied.ModelCalls != 3 || snapshot.Occupied.RetrievalCalls != 4 ||
		snapshot.Actual != (budget.Usage{InputTokens: 24, OutputTokens: 6, Cost: 9}) ||
		snapshot.Outstanding != 0 ||
		partial.Budget != snapshot {
		t.Fatal("attempts did not share usage accounting", snapshot, partial.Budget)
	}
}
