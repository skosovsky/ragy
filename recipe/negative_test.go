package recipe_test

import (
	"context"
	"errors"
	"testing"
	"time"

	ragy "github.com/skosovsky/ragy"
	"github.com/skosovsky/ragy/access"
	"github.com/skosovsky/ragy/recipe"
	"github.com/skosovsky/ragy/recipe/budget"
	"github.com/skosovsky/ragy/retrieval"
	"github.com/skosovsky/ragy/source"
)

func TestRecipeSnapshotOwnershipAcrossCallbacksAndOutputs(t *testing.T) {
	// Arrange.
	f := newFixture(t, recipe.SingleRewrite)
	f.planned, f.selected = []string{"refund"}, []int{1}
	f.results["refund"] = []retrieval.Document[meta]{document("d1")}
	f.planHook = func(req request) {
		req.Intent[0] = "changed intent"
		req.Meta[0] = "changed request"
	}
	f.assessHook = func(input recipe.AssessmentInput[[]string, []string, meta]) {
		if input.Original.Intent[0] != "intent" || input.Original.Meta[0] != "request-meta" ||
			input.Original.Read.Publication().Reference() != "pub1" {
			t.Fatal("planner mutated captured original")
		}
		input.Queries[1].Documents[0].Meta.Tags[0] = "changed evidence"
		input.Queries[1].Supports[0][0].Reference.Revision = "changed revision"
		input.Queries[1].Keys[0] = "changed identity"
	}
	// Act.
	result, err := f.run(context.Background(), t)
	// Assert.
	if err != nil || result.Selected[0].Document.Meta.Tags[0] != "owned" ||
		result.Queries[1].Documents[0].Meta.Tags[0] != "owned" ||
		result.Selected[0].Contributors[0].Supports[0].Reference.Revision != "r1" {
		t.Fatal("callback alias escaped", err)
	}
	result.Selected[0].Document.Meta.Tags[0] = "changed selection"
	if result.Queries[1].Documents[0].Meta.Tags[0] != "owned" || f.results["refund"][0].Meta.Tags[0] != "owned" {
		t.Fatal("output aliases another stage or backend")
	}
}

func TestRecipeDeadlineStopsWithoutPostDeadlineHostCallbacks(t *testing.T) {
	// Arrange: injected budget clock expires during the planning call.
	f := newFixture(t, recipe.SingleRewrite)
	f.results["original"] = []retrieval.Document[meta]{document("d1")}
	f.planned = []string{"refund"}
	callbackAfterDeadline := false
	clone := f.config.CloneMeta
	f.config.CloneMeta = func(value meta) (meta, error) {
		if len(f.retrieved) > 0 && !f.now.Before(f.read.Snapshot().IssuedAt.Add(5*time.Second)) {
			callbackAfterDeadline = true
		}
		return clone(value)
	}
	f.planHook = func(request) { f.now = f.now.Add(5 * time.Second) }
	// Act.
	result, err := f.run(context.Background(), t)
	// Assert.
	if err != nil || result.Stop != recipe.DeadlineReached || result.Outcome != recipe.Partial || f.modelCalls != 1 ||
		len(f.retrieved) != 1 ||
		callbackAfterDeadline {
		t.Fatal("deadline dispatched or projected more work", err)
	}
}

func TestRecipeUnknownPriceAndUnsupportedBeforeDispatch(t *testing.T) {
	// Arrange.
	f := newFixture(t, recipe.SingleRewrite)
	f.config.Pricing = func(_ context.Context, operation recipe.Operation) (recipe.Quote, error) {
		if operation == recipe.Retrieve {
			return recipe.Quote{CostKnown: true}, nil
		}
		return recipe.Quote{Usage: budget.Usage{InputTokens: 1024, OutputTokens: 256}, CostKnown: false}, nil
	}
	// Act.
	result, err := f.run(context.Background(), t)
	// Assert.
	if err != nil || result.Stop != recipe.PriceUnavailable || f.modelCalls != 0 || len(f.retrieved) != 1 {
		t.Fatal("unknown required price dispatched model", err)
	}
	f = newFixture(t, recipe.MultiQuery)
	f.config.Admission = func(context.Context, request) (retrieval.ReadCoverage, error) {
		return retrieval.UnobservedReadCoverage(), access.UnsupportedCapability(ragy.ErrUnsupported)
	}
	result, err = f.run(context.Background(), t)
	if !errors.Is(err, ragy.ErrUnsupported) || len(result.Queries) != 0 || f.modelCalls != 0 || len(f.retrieved) != 0 {
		t.Fatal("unsupported recipe admitted IO", err)
	}
}

func TestModelPortsReceiveReservedLimits(t *testing.T) {
	// Arrange.
	f := newFixture(t, recipe.SingleRewrite)
	f.planned, f.selected = []string{"refund"}, []int{1}
	f.results["refund"] = []retrieval.Document[meta]{document("d1")}
	var received []recipe.ModelLimits
	planner, assessor := f.config.Planner, f.config.Assessor
	f.config.Pricing = func(_ context.Context, operation recipe.Operation) (recipe.Quote, error) {
		if operation == recipe.Retrieve {
			return recipe.Quote{CostKnown: true}, nil
		}
		input, output := uint64(700), uint64(120)
		if operation == recipe.Assess {
			input, output = 900, 180
		}
		return recipe.Quote{
			Usage:     budget.Usage{InputTokens: input, OutputTokens: output, Cost: 30},
			CostKnown: true,
		}, nil
	}
	f.config.Planner = func(ctx context.Context, req request, limits recipe.ModelLimits) (recipe.Planning, error) {
		received = append(received, limits)
		return planner(ctx, req, limits)
	}
	f.config.Assessor = func(ctx context.Context, input recipe.AssessmentInput[[]string, []string, meta], limits recipe.ModelLimits) (recipe.Assessment, error) {
		received = append(received, limits)
		return assessor(ctx, input, limits)
	}
	// Act.
	result, err := f.run(context.Background(), t)
	// Assert.
	if err != nil || result.Outcome != recipe.Complete || len(received) != 2 ||
		received[0] != (recipe.ModelLimits{InputTokens: 700, OutputTokens: 120}) ||
		received[1] != (recipe.ModelLimits{InputTokens: 900, OutputTokens: 180}) {
		t.Fatal("model dispatch did not receive its reservation", received, err)
	}
}

func TestAdvisoryPriceDoesNotHideTokenOverrun(t *testing.T) {
	for _, operation := range []recipe.Operation{recipe.Plan, recipe.Assess} {
		t.Run(string(operation), func(t *testing.T) {
			// Arrange.
			f := newFixture(t, recipe.SingleRewrite)
			f.planned, f.selected = []string{"refund"}, []int{1}
			f.results["refund"] = []retrieval.Document[meta]{document("d1")}
			f.config.RequireKnownCost = false
			pricing := f.config.Pricing
			f.config.Pricing = func(ctx context.Context, op recipe.Operation) (recipe.Quote, error) {
				quote, err := pricing(ctx, op)
				quote.CostKnown = op == recipe.Retrieve
				if !quote.CostKnown {
					quote.Usage.Cost = 0
				}
				return quote, err
			}
			if operation == recipe.Plan {
				f.config.Planner = func(context.Context, request, recipe.ModelLimits) (recipe.Planning, error) {
					return recipe.Planning{
						Usage: recipe.Usage{Known: true, Value: budget.Usage{InputTokens: 1025}},
					}, nil
				}
			} else {
				f.config.Assessor = func(context.Context, recipe.AssessmentInput[[]string, []string, meta], recipe.ModelLimits) (recipe.Assessment, error) {
					return recipe.Assessment{
						Usage: recipe.Usage{Known: true, Value: budget.Usage{OutputTokens: 257}},
					}, nil
				}
			}
			// Act.
			result, err := f.run(context.Background(), t)
			// Assert.
			if !errors.Is(err, budget.ErrUsageExceeded) || len(result.Selected) != 0 || len(result.Queries) != 0 {
				t.Fatal("unknown cost concealed observed token overrun", err)
			}
		})
	}
}

func TestZeroModelReservationRejectsBeforeDispatch(t *testing.T) {
	// Arrange.
	f := newFixture(t, recipe.SingleRewrite)
	f.config.Pricing = func(context.Context, recipe.Operation) (recipe.Quote, error) {
		return recipe.Quote{CostKnown: true}, nil
	}
	// Act.
	result, err := f.run(context.Background(), t)
	// Assert.
	if !errors.Is(err, ragy.ErrInvalidArgument) || f.modelCalls != 0 || len(result.Selected) != 0 {
		t.Fatal("unbounded model dispatch admitted", err)
	}
}

func TestRecipeInvalidPlanningAssessmentAndUsageFailWithoutPayloads(t *testing.T) {
	for _, failure := range []string{"overflow", "duplicate", "selection", "usage", "foreign-support", "oversized-result"} {
		t.Run(failure, func(t *testing.T) {
			// Arrange.
			f := newFixture(t, recipe.SingleRewrite)
			f.planned, f.selected = []string{"refund"}, []int{1}
			f.results["refund"] = []retrieval.Document[meta]{document("d1")}
			switch failure {
			case "overflow":
				f.planned = []string{"a", "b"}
			case "duplicate":
				f.config.Strategy = recipe.MultiQuery
				f.config.MaxQueries = 2
				f.planned = []string{"a", "a"}
			case "selection":
				f.selected = []int{100}
			case "usage":
				f.config.Planner = func(context.Context, request, recipe.ModelLimits) (recipe.Planning, error) {
					f.modelCalls++
					return recipe.Planning{
						Queries: f.planned,
						Usage:   recipe.Usage{Known: true, Value: budget.Usage{InputTokens: 1025}},
					}, nil
				}
			case "foreign-support":
				f.config.Supports = func(context.Context, access.Binding, retrieval.Document[meta]) ([]source.Locator, error) {
					loc := location("d1")
					loc.Reference.Source = "private"
					return []source.Locator{loc}, nil
				}
			case "oversized-result":
				f.config.MaxDocuments = 1
				f.results["refund"] = append(f.results["refund"], document("d4"))
			}
			// Act.
			result, err := f.run(context.Background(), t)
			// Assert.
			if err == nil || len(result.Queries) != 0 || len(result.Selected) != 0 || len(result.Stages) != 0 {
				t.Fatal("invalid recipe emitted successful payload", err)
			}
		})
	}
}

func TestParentDeadlineAndUnsafePrecomputedVectorProfile(t *testing.T) {
	// Arrange.
	f := newFixture(t, recipe.SingleRewrite)
	f.planned, f.selected = []string{"refund"}, []int{1}
	f.results["refund"] = []retrieval.Document[meta]{document("d1")}
	parent, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	parentDeadline, _ := parent.Deadline()
	f.config.Planner = func(ctx context.Context, _ request, _ recipe.ModelLimits) (recipe.Planning, error) {
		deadline, present := ctx.Deadline()
		if !present || deadline.After(parentDeadline) {
			t.Fatal("earlier parent deadline was extended")
		}
		return recipe.Planning{Queries: f.planned, Usage: observed()}, nil
	}
	// Act.
	result, err := f.run(parent, t)
	// Assert.
	if err != nil || result.Outcome != recipe.Complete {
		t.Fatal(err)
	}
	r, err := recipe.New(f.config)
	if err != nil {
		t.Fatal(err)
	}
	f.retrieved = nil
	result, err = r.Run(
		context.Background(),
		request{Read: f.read, Text: "original", Options: retrieval.RetrieveOptions{TopK: 3, Vector: []float32{1, 0}}},
	)
	if !errors.Is(err, ragy.ErrUnsupported) || len(result.Queries) != 0 || len(f.retrieved) != 0 {
		t.Fatal("text rewrite silently reused original vector", err)
	}
}

func TestAdvisoryUnknownPricingRetainsReservationsAndDiagnostic(t *testing.T) {
	// Arrange.
	f := newFixture(t, recipe.SingleRewrite)
	f.planned, f.selected = []string{"refund"}, []int{1}
	f.results["refund"] = []retrieval.Document[meta]{document("d1")}
	f.config.RequireKnownCost = false
	f.config.Pricing = func(_ context.Context, operation recipe.Operation) (recipe.Quote, error) {
		if operation == recipe.Retrieve {
			return recipe.Quote{CostKnown: true}, nil
		}
		return recipe.Quote{Usage: budget.Usage{InputTokens: 1024, OutputTokens: 256}, CostKnown: false}, nil
	}
	// Act.
	result, err := f.run(context.Background(), t)
	// Assert: unknown pricing never becomes observed free calls or refunded usage.
	if err != nil || result.Outcome != recipe.Complete || !result.Budget.UnknownCost ||
		result.Budget.UnknownUsage != 2 ||
		result.Budget.Occupied.Usage.InputTokens != 2048 ||
		result.Budget.Actual.Cost != 0 {
		t.Fatal(result.Budget, err)
	}
	for _, stage := range result.Stages {
		if stage.Operation != recipe.Retrieve && stage.Usage.Known {
			t.Fatal("unknown-price stage claims known actual cost")
		}
	}
}

func TestRecipeDocumentSourcesOwnedAcrossAssessorAndFusion(t *testing.T) {
	// Arrange: the source port confirms every document support.
	f := newFixture(t, recipe.SingleRewrite)
	doc := document("d1")
	doc.SourceSupports = []source.Locator{location("d1")}
	f.results["refund"] = []retrieval.Document[meta]{doc}
	f.planned, f.selected = []string{"refund"}, []int{1}
	f.assessHook = func(input recipe.AssessmentInput[[]string, []string, meta]) {
		input.Queries[1].Documents[0].SourceSupports[0].Reference.Source = "private"
	}
	// Act.
	result, err := f.run(context.Background(), t)
	// Assert: native result, fusion and backend storage remain independent.
	if err != nil || len(result.Selected) != 1 {
		t.Fatal(err)
	}
	if result.Queries[1].Documents[0].SourceSupports[0].Reference.Source != "corpus" ||
		result.Selected[0].Document.SourceSupports[0].Reference.Source != "corpus" {
		t.Fatal("assessor mutated provenance")
	}
	result.Selected[0].Document.SourceSupports[0].Reference.Source = "changed"
	if result.Queries[1].Documents[0].SourceSupports[0].Reference.Source != "corpus" ||
		f.results["refund"][0].SourceSupports[0].Reference.Source != "corpus" {
		t.Fatal("fusion provenance aliases query/backend")
	}
}

func TestRecipeSourcePortMustConfirmAllDocumentSupports(t *testing.T) {
	// Arrange: a structurally valid support is omitted by the host admission port.
	f := newFixture(t, recipe.SingleRewrite)
	doc := document("d1")
	doc.SourceSupports = []source.Locator{location("another-artifact")}
	f.results["original"] = []retrieval.Document[meta]{doc}
	// Act.
	result, err := f.run(context.Background(), t)
	// Assert.
	if !access.IsProtectionFailure(err) || len(result.Selected) != 0 || len(result.Queries) != 0 || f.modelCalls != 0 {
		t.Fatal(result, err)
	}
}
