package recipe_test

import (
	"context"
	"errors"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/dense"
	"github.com/skosovsky/ragy/recipe"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/retrieval"
)

type deadlineBackend struct {
	next  retrieval.RequestBackend[[]string, []string, meta]
	calls int
	fail  func() error
}

func (b *deadlineBackend) Retrieve(ctx context.Context, req request) (retrieval.ResultSet[meta], error) {
	b.calls++
	if b.calls == 2 {
		return retrieval.NewResultSet([]retrieval.Document[meta]{document("late")}, nil), b.fail()
	}
	return b.next.Retrieve(ctx, req)
}

type deadlineEncoder struct {
	boundedEncoder

	f     *fixture
	fail  func() error
	usage recipe.Usage
}

func (e *deadlineEncoder) Encode(
	ctx context.Context,
	input dense.Request,
	limits recipe.ModelLimits,
) (dense.Result, recipe.Usage, error) {
	e.f.modelCalls++
	if len(e.inputs) == 1 {
		e.inputs = append(e.inputs, input.Inputs[0])
		return dense.Result{}, e.usage, e.fail()
	}
	result, _, err := e.boundedEncoder.Encode(ctx, input, limits)
	return result, observed(), err
}

func configureDeadlineFailure(f *fixture, stage recipe.Operation, cause error, usage recipe.Usage) {
	fail := func() error { f.now = f.now.Add(6 * time.Second); return cause }
	switch stage {
	case recipe.Plan:
		f.config.Planner = func(context.Context, request, recipe.ModelLimits) (recipe.Planning, error) {
			f.modelCalls++
			return recipe.Planning{Usage: usage}, fail()
		}
	case recipe.Assess:
		f.config.Assessor = func(context.Context, recipe.AssessmentInput[[]string, []string, meta], recipe.ModelLimits) (recipe.Assessment, error) {
			f.modelCalls++
			return recipe.Assessment{Usage: usage}, fail()
		}
	case recipe.Retrieve:
		f.config.Backend = &deadlineBackend{next: f.config.Backend, calls: 0, fail: fail}
	case recipe.Encode:
		f.config.QueryEncoder = &deadlineEncoder{boundedEncoder: boundedEncoder{}, f: f, fail: fail, usage: usage}
	}
}

func TestCallbackFailureSurvivesLocalDeadline(t *testing.T) {
	ordinary := errors.New("callback unavailable")
	cases := []struct {
		name  string
		cause error
	}{
		{"ordinary", ordinary},
		{"protocol", ragy.ErrProtocol},
		{"protection", access.NonSkippable(ragy.ErrUnavailable)},
		{"joined-protection", errors.Join(access.NonSkippable(ragy.ErrUnavailable), context.DeadlineExceeded)},
		{"joined-ordinary", errors.Join(ordinary, context.DeadlineExceeded)},
		{"joined-budget", errors.Join(ordinary, budget.ErrExhausted)},
	}
	for _, stage := range []recipe.Operation{recipe.Plan, recipe.Retrieve, recipe.Assess, recipe.Encode} {
		for _, tc := range cases {
			for _, mode := range []string{"run", "own", "observed", "own-observed"} {
				t.Run(string(stage)+"/"+tc.name+"/"+mode, func(t *testing.T) {
					assertDeadlineFailure(t, stage, tc.cause, mode)
				})
			}
		}
	}
}

func TestDeadlineUsageAccounting(t *testing.T) {
	for _, unknown := range []bool{false, true} {
		// Arrange: planner expires with known overrun or unknown usage.
		f := newFixture(t, recipe.SingleRewrite)
		f.results["original"] = []retrieval.Document[meta]{document("d0")}
		usage := observed()
		usage.Known = !unknown
		if !unknown {
			usage.Value.InputTokens = 1025
		}
		configureDeadlineFailure(f, recipe.Plan, nil, usage)
		r, err := recipe.New(f.config)
		if err != nil {
			t.Fatal(err)
		}
		ledger, err := budget.New(
			budget.Config{
				Limits:           f.config.Limits,
				Deadline:         f.now.Add(f.config.Duration),
				Now:              f.config.Now,
				RequireKnownCost: true,
			},
		)
		if err != nil {
			t.Fatal(err)
		}
		// Act.
		result, err := r.RunObserved(context.Background(), recordedRequest(f), ledger)
		// Assert: expiry never refunds or loses an overrun.
		snapshot := ledger.Snapshot()
		if snapshot.Outstanding != 0 || snapshot.Occupied.ModelCalls != 1 || f.modelCalls != 1 {
			t.Fatal("lease not settled once", snapshot)
		}
		if unknown {
			if err != nil || result.Stop != recipe.DeadlineReached || len(result.Selected) != 1 ||
				snapshot.UnknownUsage != 1 ||
				snapshot.Occupied.Usage.InputTokens != 1024 {
				t.Fatal("unknown reservation lost", snapshot, err)
			}
		} else if !errors.Is(err, budget.ErrUsageExceeded) || !errors.Is(err, context.DeadlineExceeded) || len(result.Selected) != 0 || result.Outcome != recipe.Failure || snapshot.UnknownUsage != 1 || snapshot.Occupied.Usage.InputTokens != 1024 || result.Stages[len(result.Stages)-1].Usage.Value.InputTokens != 1025 {
			t.Fatal("overrun hidden", snapshot, err)
		}
	}
}

func assertDeadlineFailure(t *testing.T, stage recipe.Operation, cause error, mode string) {
	t.Helper()
	// Arrange: original evidence is captured before the failing callback.
	f := newFixture(t, recipe.SingleRewrite)
	f.planned = []string{"rewrite"}
	f.results["original"] = []retrieval.Document[meta]{document("d0")}
	f.results["rewrite"] = []retrieval.Document[meta]{document("d1")}
	f.config.Limits.ModelCalls = 4
	f.config.Limits.Usage = budget.Usage{InputTokens: 4096, OutputTokens: 1024, Cost: 200}
	configureDeadlineFailure(f, stage, cause, observed())
	r, err := recipe.New(f.config)
	if err != nil {
		t.Fatal(err)
	}
	ledger, err := budget.New(
		budget.Config{
			Limits:           f.config.Limits,
			Deadline:         f.now.Add(f.config.Duration),
			Now:              f.config.Now,
			RequireKnownCost: true,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	req := recordedRequest(f)
	// Act.
	var result recipe.Result[meta]
	switch mode {
	case "run":
		result, err = r.Run(context.Background(), req, ledger)
	case "own":
		result, err = r.RunOwn(context.Background(), req)
	case "observed":
		result, err = r.RunObserved(context.Background(), req, ledger)
	case "own-observed":
		result, err = r.RunOwnObserved(context.Background(), req)
	}
	// Assert: every cause survives; only observed ordinary failures retain a journal.
	if !errors.Is(err, cause) || !errors.Is(err, context.DeadlineExceeded) || len(result.Selected) != 0 {
		t.Fatalf("callback demoted or cause lost: result=%v err=%v", result.Outcome, err)
	}
	protected := access.IsProtectionFailure(cause)
	journal := (mode == "observed" || mode == "own-observed") && !protected
	if journal {
		if result.Outcome != recipe.Failure || result.Stop != recipe.StageFailure || len(result.Queries) == 0 ||
			result.Budget.Outstanding != 0 {
			t.Fatal("missing failed journal", result, err)
		}
	} else if len(result.Queries) != 0 || len(result.Stages) != 0 || len(result.Encoding) != 0 || result.Artifact != nil || result.Publication != "" {
		t.Fatal("suppressed result retained payload", err)
	}
	assertDeadlineDispatch(t, f, stage, mode, ledger)
}

func TestParentCancellationRetainsCallbackCause(t *testing.T) {
	// Arrange: cancellation races a protocol failure after prior evidence.
	f := newFixture(t, recipe.SingleRewrite)
	f.results["original"] = []retrieval.Document[meta]{document("d0")}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	f.config.Planner = func(context.Context, request, recipe.ModelLimits) (recipe.Planning, error) {
		f.modelCalls++
		cancel()
		return recipe.Planning{Usage: observed()}, ragy.ErrProtocol
	}
	r, err := recipe.New(f.config)
	if err != nil {
		t.Fatal(err)
	}
	// Act.
	result, err := r.RunOwnObserved(ctx, recordedRequest(f))
	// Assert.
	if !errors.Is(err, context.Canceled) || !errors.Is(err, ragy.ErrProtocol) || !access.IsProtectionFailure(err) ||
		len(result.Queries) != 0 ||
		len(result.Stages) != 0 ||
		f.modelCalls != 1 {
		t.Fatal("parent cancellation lost cause or retained journal", err)
	}
}

func TestMalformedModelOutputIsNotHiddenByDeadline(t *testing.T) {
	for _, stage := range []recipe.Operation{recipe.Plan, recipe.Assess, recipe.Encode} {
		t.Run(string(stage), func(t *testing.T) {
			// Arrange: malformed success arrives after expiry.
			f := newFixture(t, recipe.SingleRewrite)
			f.planned = []string{"rewrite"}
			f.results["original"] = []retrieval.Document[meta]{document("d0")}
			f.results["rewrite"] = []retrieval.Document[meta]{document("d1")}
			f.config.Limits.ModelCalls = 4
			f.config.Limits.Usage = budget.Usage{InputTokens: 4096, OutputTokens: 1024, Cost: 200}
			configureDeadlineFailure(f, stage, nil, observed())
			if stage == recipe.Plan {
				f.config.Planner = func(context.Context, request, recipe.ModelLimits) (recipe.Planning, error) {
					f.modelCalls++
					f.now = f.now.Add(6 * time.Second)
					return recipe.Planning{Queries: []string{""}, Usage: observed()}, nil
				}
			}
			if stage == recipe.Assess {
				f.config.Assessor = func(context.Context, recipe.AssessmentInput[[]string, []string, meta], recipe.ModelLimits) (recipe.Assessment, error) {
					f.modelCalls++
					f.now = f.now.Add(6 * time.Second)
					return recipe.Assessment{Selected: []int{99}, Usage: observed()}, nil
				}
			}
			// Act.
			result, err := f.run(context.Background(), t)
			// Assert.
			if !errors.Is(err, ragy.ErrProtocol) || !errors.Is(err, context.DeadlineExceeded) ||
				len(result.Selected) != 0 {
				t.Fatal("malformed output hidden by deadline", err)
			}
		})
	}
}

func TestArtifactCallbackProtectionSurvivesLocalDeadline(t *testing.T) {
	for _, cause := range []error{access.NonSkippable(context.DeadlineExceeded), errors.Join(access.NonSkippable(ragy.ErrUnavailable), context.DeadlineExceeded), errors.Join(ragy.ErrProtocol, context.DeadlineExceeded)} {
		// Arrange: renderer measurement fails independently while its timer expires.
		f := newFixture(t, recipe.Decomposition)
		f.planned = []string{"one"}
		f.selected = []int{0}
		f.results["one"] = []retrieval.Document[meta]{document("d1")}
		resource := retrieval.RuneResource(100)
		resource.Measure = func(context.Context, string) (int64, error) {
			f.now = f.now.Add(6 * time.Second)
			return 0, cause
		}
		f.config.Artifact = &retrieval.ArtifactRenderOptions[meta]{Resource: resource, CloneMeta: f.config.CloneMeta}
		// Act.
		result, err := f.run(context.Background(), t)
		// Assert: explicit host protection never masquerades as renderer-local expiry.
		if !errors.Is(err, cause) || !access.IsProtectionFailure(err) || len(result.Queries) != 0 ||
			result.Artifact != nil {
			t.Fatal("artifact callback failure rescued", err)
		}
	}
}

func assertDeadlineDispatch(t *testing.T, f *fixture, stage recipe.Operation, mode string, ledger *budget.Ledger) {
	t.Helper()
	models, retrieves := uint64(1), uint64(1)
	if stage == recipe.Assess {
		models, retrieves = 2, 2
	}
	if stage == recipe.Retrieve {
		retrieves = 2
	}
	if stage == recipe.Encode {
		models = 3
	}
	if uint64(f.modelCalls) != models {
		t.Fatal("unexpected model retry", f.modelCalls)
	}
	if mode == "run" || mode == "observed" {
		snapshot := ledger.Snapshot()
		if snapshot.Outstanding != 0 || snapshot.Occupied.ModelCalls != models ||
			snapshot.Occupied.RetrievalCalls != retrieves {
			t.Fatal("dispatch/settlement not exact", snapshot)
		}
	}
}
