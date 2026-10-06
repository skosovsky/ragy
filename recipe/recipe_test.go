package recipe_test

import (
	"context"
	"errors"
	"slices"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/filter"
	"github.com/skosovsky/ragy/recipe"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

type meta struct {
	Tenant string   `json:"tenant"`
	Tags   []string `json:"-"`
}
type request = retrieval.Request[[]string, []string]
type fixture struct {
	now          time.Time
	epoch        int
	results      map[string][]retrieval.Document[meta]
	retrieved    []string
	planned      []string
	selected     []int
	sufficient   bool
	modelCalls   int
	planHook     func(request)
	retrieveHook func(context.Context)
	assessHook   func(recipe.AssessmentInput[[]string, []string, meta])
	config       recipe.Config[[]string, []string, meta]
	read         access.Binding
}

type backend struct {
	f      *fixture
	schema filter.Schema
}

func (b backend) Schema() filter.Schema { return b.schema }
func (backend) ReadCapabilities() access.Capabilities {
	return access.Capabilities{ScopeProfile: true, PinnedPublication: true, RequirePinnedPublication: true}
}
func (b backend) Retrieve(ctx context.Context, req request) (retrieval.ResultSet[meta], error) {
	b.f.retrieved = append(b.f.retrieved, req.EffectiveText())
	if b.f.retrieveHook != nil {
		b.f.retrieveHook(ctx)
	}
	prepared, err := retrieval.PrepareRead(ctx, req, b)
	if err != nil {
		return retrieval.NewResultSet[meta](nil, nil), err
	}
	codec := retrieval.NewJSONCodec[meta](b.schema)
	var docs []retrieval.Document[meta]
	for _, doc := range b.f.results[req.EffectiveText()] {
		allowed, matchErr := retrieval.MatchDocument(codec, doc, prepared.Options.Filters)
		if matchErr != nil {
			return nil, matchErr
		}
		if allowed {
			docs = append(docs, doc)
		}
	}
	return retrieval.NewResultSet(docs, nil), nil
}

func document(id string) retrieval.Document[meta] {
	return retrieval.Document[meta]{ID: id, Content: id + " text", Meta: meta{Tenant: "a", Tags: []string{"owned"}}}
}
func location(id string) source.Locator {
	return source.Locator{Kind: source.DocumentLocation, Reference: source.Reference{
		Namespace:         "n",
		Source:            "corpus",
		Revision:          "r1",
		Transformation:    "original",
		AccessFingerprint: "acl",
		Artifact:          id,
		Representation:    "text",
	}}
}

func newFixture(t *testing.T, strategy recipe.Strategy) *fixture {
	t.Helper()
	f := &fixture{now: time.Now(), epoch: 7, results: make(map[string][]retrieval.Document[meta]), sufficient: true}
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
	mandatory, err := filter.Eq(builder, tenant, "a").Build()
	if err != nil {
		t.Fatal(err)
	}
	publication, err := access.PinPublication(
		"pub1",
		[]access.TargetRevision{
			{
				Target:            "lexical",
				Namespace:         "n",
				Source:            "corpus",
				Revision:          "r1",
				Transformation:    "chunk",
				AccessFingerprint: "acl",
			},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	f.read, err = access.Scoped(access.ScopedConfig{
		Snapshot: access.Snapshot{
			Identity:    "scope",
			PolicyEpoch: 7,
			IssuedAt:    f.now,
			ExpiresAt:   f.now.Add(time.Minute),
		},
		Mandatory:   mandatory,
		Schema:      schema,
		Publication: publication,
		Now:         func() time.Time { return f.now },
		Authority: access.AuthorityFunc(func(context.Context, access.Snapshot) error {
			if f.epoch != 7 {
				return ragy.ErrUnavailable
			}
			return nil
		}),
	})
	if err != nil {
		t.Fatal(err)
	}
	b := backend{f: f, schema: schema}
	maxQueries := 2
	if strategy == recipe.SingleRewrite {
		maxQueries = 1
	}
	if strategy == recipe.Decomposition {
		maxQueries = 3
	}
	f.config = recipe.Config[[]string, []string, meta]{
		Strategy: strategy,
		Revision: "bounded-test",
		Backend:  b,
		Identity: retrieval.DocumentIDResolver[meta]{},
		Admission: func(ctx context.Context, req request) (retrieval.ReadCoverage, error) {
			_, admissionErr := retrieval.PrepareRead(ctx, req, b)
			return retrieval.CompleteReadCoverage(), admissionErr
		},
		Planner: func(ctx context.Context, req request, _ recipe.ModelLimits) (recipe.Planning, error) {
			f.modelCalls++
			if err := req.Read.Check(ctx); err != nil {
				return recipe.Planning{}, err
			}
			if f.planHook != nil {
				f.planHook(req)
			}
			return recipe.Planning{Queries: f.planned, Usage: observed()}, nil
		},
		Assessor: func(ctx context.Context, input recipe.AssessmentInput[[]string, []string, meta], _ recipe.ModelLimits) (recipe.Assessment, error) {
			f.modelCalls++
			if err := input.Original.Read.Check(ctx); err != nil {
				return recipe.Assessment{}, err
			}
			if f.assessHook != nil {
				f.assessHook(input)
			}
			return recipe.Assessment{Selected: f.selected, Sufficient: f.sufficient, Usage: observed()}, nil
		},
		Pricing: func(_ context.Context, operation recipe.Operation) (recipe.Quote, error) {
			if operation == recipe.Retrieve {
				return recipe.Quote{CostKnown: true}, nil
			}
			return recipe.Quote{
				Usage:     budget.Usage{InputTokens: 1024, OutputTokens: 256, Cost: 30},
				CostKnown: true,
			}, nil
		},
		CloneIntent:      func(value []string) ([]string, error) { return slices.Clone(value), nil },
		CloneRequestMeta: func(value []string) ([]string, error) { return slices.Clone(value), nil },
		CloneMeta:        func(value meta) (meta, error) { value.Tags = slices.Clone(value.Tags); return value, nil },
		Supports: func(_ context.Context, _ access.Binding, doc retrieval.Document[meta]) ([]source.Locator, error) {
			return []source.Locator{location(doc.ID)}, nil
		},
		Limits: budget.Limits{
			RetrievalCalls: 3,
			ModelCalls:     2,
			Usage:          budget.Usage{InputTokens: 2048, OutputTokens: 512, Cost: 100},
		},
		RequireKnownCost: true,
		Duration:         5 * time.Second,
		Now:              func() time.Time { return f.now },
		MaxQueries:       maxQueries,
		MaxDocuments:     10,
		FusionK:          60,
	}
	return f
}

func observed() recipe.Usage {
	return recipe.Usage{Known: true, Value: budget.Usage{InputTokens: 100, OutputTokens: 20, Cost: 30}}
}
func (f *fixture) run(ctx context.Context, t *testing.T) (recipe.Result[meta], error) {
	t.Helper()
	r, err := recipe.New(f.config)
	if err != nil {
		t.Fatal(err)
	}
	return r.Run(
		ctx,
		request{
			Read:    f.read,
			Text:    "original",
			Intent:  []string{"intent"},
			Meta:    []string{"request-meta"},
			Options: retrieval.RetrieveOptions{TopK: 3},
		},
	)
}

func TestReferenceRecipeCases(t *testing.T) {
	// Arrange.
	cases := []struct {
		name       string
		strategy   recipe.Strategy
		queries    []string
		docs       map[string][]retrieval.Document[meta]
		selected   []int
		sufficient bool
		outcome    recipe.Outcome
		ids        []string
	}{
		{
			name:       "helpful rewrite",
			strategy:   recipe.SingleRewrite,
			queries:    []string{"refund"},
			docs:       map[string][]retrieval.Document[meta]{"refund": {document("d1")}},
			selected:   []int{1},
			sufficient: true,
			outcome:    recipe.Complete,
			ids:        []string{"d1"},
		},
		{
			name:       "harmful rewrite retains original",
			strategy:   recipe.SingleRewrite,
			queries:    []string{"device"},
			docs:       map[string][]retrieval.Document[meta]{"original": {document("d1")}, "device": {document("d3")}},
			selected:   []int{0},
			sufficient: true,
			outcome:    recipe.Complete,
			ids:        []string{"d1"},
		},
		{
			name:       "no answer",
			strategy:   recipe.SingleRewrite,
			queries:    []string{"receipt"},
			docs:       nil,
			selected:   nil,
			sufficient: false,
			outcome:    recipe.Insufficient,
			ids:        nil,
		},
		{
			name:       "multi query dedup",
			strategy:   recipe.MultiQuery,
			queries:    []string{"reset", "recover"},
			docs:       map[string][]retrieval.Document[meta]{"reset": {document("d2")}, "recover": {document("d2")}},
			selected:   []int{1, 2},
			sufficient: true,
			outcome:    recipe.Complete,
			ids:        []string{"d2"},
		},
		{
			name:       "decomposition",
			strategy:   recipe.Decomposition,
			queries:    []string{"refund", "card"},
			docs:       map[string][]retrieval.Document[meta]{"refund": {document("d1")}, "card": {document("d4")}},
			selected:   []int{0, 1},
			sufficient: true,
			outcome:    recipe.Complete,
			ids:        []string{"d1", "d4"},
		},
		{
			name:       "partial decomposition",
			strategy:   recipe.Decomposition,
			queries:    []string{"refund", "card"},
			docs:       map[string][]retrieval.Document[meta]{"refund": {document("d1")}},
			selected:   []int{0},
			sufficient: true,
			outcome:    recipe.Partial,
			ids:        []string{"d1"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := newFixture(t, tc.strategy)
			f.planned, f.results, f.selected, f.sufficient = tc.queries, tc.docs, tc.selected, tc.sufficient
			// Act.
			result, err := f.run(context.Background(), t)
			// Assert.
			if err != nil || result.Outcome != tc.outcome || result.Stop != recipe.Assessed {
				t.Fatal(result.Outcome, err)
			}
			var ids []string
			for _, evidence := range result.Selected {
				ids = append(ids, evidence.Document.ID)
			}
			if !slices.Equal(ids, tc.ids) || f.modelCalls != 2 || len(f.retrieved) > 3 || result.Publication != "pub1" {
				t.Fatal(ids, f.modelCalls, f.retrieved)
			}
			if tc.strategy == recipe.MultiQuery && len(result.Selected[0].Contributors) != 2 {
				t.Fatal("query provenance lost during dedup")
			}
			if tc.strategy == recipe.Decomposition && len(result.Coverage) != 2 {
				t.Fatal("subquestion coverage lost")
			}
		})
	}
}

func TestRecipeBudgetStopBeforeModelDispatch(t *testing.T) {
	// Arrange.
	f := newFixture(t, recipe.SingleRewrite)
	f.results["original"] = []retrieval.Document[meta]{document("d1")}
	f.config.Limits.ModelCalls = 0
	// Act.
	result, err := f.run(context.Background(), t)
	// Assert.
	if err != nil || result.Stop != recipe.BudgetExhausted || result.Outcome != recipe.Partial || f.modelCalls != 0 ||
		len(result.Selected) != 1 {
		t.Fatal(result, err)
	}
}

func TestRecipeRevocationAndCancellationSuppressEverySideOutput(t *testing.T) {
	for _, stage := range []string{"planner", "assessor", "retrieval"} {
		t.Run(stage, func(t *testing.T) {
			// Arrange.
			f := newFixture(t, recipe.SingleRewrite)
			f.planned, f.selected = []string{"refund"}, []int{1}
			f.results["refund"] = []retrieval.Document[meta]{document("d1")}
			if stage == "planner" {
				f.planHook = func(request) { f.epoch = 8 }
			}
			if stage == "assessor" {
				f.assessHook = func(recipe.AssessmentInput[[]string, []string, meta]) { f.epoch = 8 }
			}
			if stage == "retrieval" {
				f.retrieveHook = func(context.Context) { f.epoch = 8 }
			}
			// Act.
			result, err := f.run(context.Background(), t)
			// Assert.
			if !errors.Is(err, ragy.ErrUnavailable) || len(result.Queries) != 0 || len(result.Selected) != 0 ||
				len(result.Stages) != 0 {
				t.Fatal("revoked side outputs escaped", err)
			}
		})
	}
	f := newFixture(t, recipe.MultiQuery)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	result, err := f.run(ctx, t)
	if !errors.Is(err, context.Canceled) || len(result.Queries) != 0 || f.modelCalls != 0 || len(f.retrieved) != 0 {
		t.Fatal("canceled attempt dispatched", err)
	}
}
